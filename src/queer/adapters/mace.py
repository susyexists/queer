"""Optional direct MACE tensor adapter; Queer owns structure/neighbor handling.

Torch and checkpoint model classes are external prediction producers. This
module never constructs ASE atoms or delegates any scientific algorithm to an
ASE calculator. Importing it does not import Torch or MACE.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path

import numpy as np

from ..structures import Evaluation


class MACEForceProvider:
    """Evaluate a trained MACE module on a native single-structure tensor graph.

    ``checkpoint`` must be an explicitly selected full trained-module .model/.pt
    checkpoint. Alternatively inject ``model`` (useful for an already-loaded
    model or a producer integration). Foundation/catalog names must be resolved
    to checkpoints externally. Model class imports are optional dependencies.
    """

    @classmethod
    def from_checkpoint(cls, path, **options):
        """Load an explicitly resolved checkpoint through the optional adapter."""
        return cls(checkpoint=path, **options)

    def __init__(
        self,
        checkpoint=None,
        *,
        model=None,
        device="cpu",
        dtype=None,
        head=None,
        energy_units_to_eV=1.0,
        length_units_to_A=1.0,
        compute_stress=False,
    ):
        if (checkpoint is None) == (model is None):
            raise ValueError("provide exactly one checkpoint path or loaded MACE model")
        try:
            torch = importlib.import_module("torch")
        except ImportError as error:
            raise ImportError(
                "MACE force evaluation requires the optional torch/MACE model environment"
            ) from error
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.device = torch.device(device)
        checkpoint_sha256 = None
        if model is None:
            path = Path(checkpoint).expanduser().resolve()
            checksum = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    checksum.update(chunk)
            checkpoint_sha256 = checksum.hexdigest()
            try:
                model = torch.load(path, map_location=self.device, weights_only=False)
            except ModuleNotFoundError as error:
                raise ImportError(
                    "checkpoint model classes are missing; install its matching MACE producer version"
                ) from error
        if not callable(model) or not hasattr(model, "atomic_numbers") or not hasattr(model, "r_max"):
            raise ValueError("expected a trained MACE module with atomic_numbers and r_max metadata")
        self.model = model.to(self.device) if hasattr(model, "to") else model
        if dtype is not None and dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if dtype is None:
            parameters = iter(model.parameters()) if hasattr(model, "parameters") else iter(())
            first = next(parameters, None)
            self.dtype = first.dtype if first is not None else torch.float64
        else:
            self.dtype = getattr(torch, dtype)
            if hasattr(self.model, "to"):
                self.model = self.model.to(dtype=self.dtype)
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("supported MACE producers must use float32 or float64")
        if hasattr(self.model, "eval"):
            self.model.eval()
        if hasattr(self.model, "parameters"):
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        self.atomic_numbers = [
            int(number) for number in torch.as_tensor(model.atomic_numbers).reshape(-1).cpu()
        ]
        if len(set(self.atomic_numbers)) != len(self.atomic_numbers):
            raise ValueError("MACE producer has duplicate atomic numbers")
        self.cutoff_model = float(torch.as_tensor(model.r_max).reshape(-1)[0].cpu())
        self.energy_units_to_eV, self.length_units_to_A = float(energy_units_to_eV), float(length_units_to_A)
        if (
            not np.isfinite([self.cutoff_model, self.energy_units_to_eV, self.length_units_to_A]).all()
            or min(self.cutoff_model, self.energy_units_to_eV, self.length_units_to_A) <= 0
        ):
            raise ValueError("cutoff and unit conversion factors must be finite and positive")
        self.compute_stress = bool(compute_stress)
        heads = list(getattr(model, "heads", []))
        if head is None:
            if len(heads) > 1:
                raise ValueError("multi-head MACE checkpoints require an explicit head")
            self.head_index = 0
        else:
            if head not in heads:
                raise ValueError(f"unknown MACE head {head!r}; supported: {heads}")
            self.head_index = heads.index(head)
        self.head = head
        model_state_sha256 = None
        if checkpoint_sha256 is None and hasattr(model, "state_dict"):
            checksum = hashlib.sha256()
            for key, value in sorted(model.state_dict().items()):
                checksum.update(key.encode())
                if hasattr(value, "detach"):
                    array = value.detach().cpu().numpy()
                    checksum.update(str(array.dtype).encode())
                    checksum.update(str(array.shape).encode())
                    checksum.update(array.tobytes())
                else:
                    checksum.update(repr(value).encode())
            model_state_sha256 = checksum.hexdigest()
        self.provenance = {
            "producer": "MACE",
            "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "checkpoint_sha256": checkpoint_sha256,
            "model_state_sha256": model_state_sha256,
            "torch_version": str(torch.__version__),
            "device": str(self.device),
            "dtype": str(self.dtype),
            "head": self.head,
            "energy_units_to_eV": self.energy_units_to_eV,
            "length_units_to_A": self.length_units_to_A,
            "compute_stress": self.compute_stress,
        }
        self.fingerprint = hashlib.sha256(json.dumps(self.provenance, sort_keys=True).encode()).hexdigest()

    def tensor_graph(self, structure):
        """Construct graph tensors with native periodic-neighbor enumeration."""
        from ..geometry.structures import neighbor_list

        torch = self.torch
        unknown = set(map(int, structure.atomic_numbers)) - set(self.atomic_numbers)
        if unknown:
            raise ValueError(f"MACE checkpoint does not support atomic numbers {sorted(unknown)}")
        if structure.natoms < 1:
            raise ValueError("MACE evaluation requires at least one atom")
        i, j, offsets, _, _ = neighbor_list(structure, self.cutoff_model * self.length_units_to_A)
        lookup = {number: index for index, number in enumerate(self.atomic_numbers)}
        indices = torch.tensor(
            [lookup[int(number)] for number in structure.atomic_numbers], device=self.device
        )
        node_attrs = torch.nn.functional.one_hot(indices, num_classes=len(lookup)).to(self.dtype)

        def floating(value):
            return torch.tensor(np.asarray(value), dtype=self.dtype, device=self.device)

        return {
            "positions": floating(structure.positions / self.length_units_to_A).requires_grad_(True),
            "cell": floating(structure.cell / self.length_units_to_A),
            "node_attrs": node_attrs,
            "edge_index": torch.tensor(np.asarray([i, j]), dtype=torch.long, device=self.device),
            "unit_shifts": floating(offsets),
            "shifts": floating(offsets @ structure.cell / self.length_units_to_A),
            "batch": torch.zeros(structure.natoms, dtype=torch.long, device=self.device),
            "ptr": torch.tensor([0, structure.natoms], dtype=torch.long, device=self.device),
            "head": torch.tensor([self.head_index], dtype=torch.long, device=self.device),
        }

    def evaluate(self, structure):
        torch = self.torch
        # Forces require position gradients even when the caller disables grads.
        with torch.enable_grad():
            graph = self.tensor_graph(structure)
            forward = self.model.forward if hasattr(self.model, "forward") else self.model
            signature = inspect.signature(forward)
            options = {"training": False, "compute_force": True, "compute_stress": self.compute_stress}
            if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
                options = {key: value for key, value in options.items() if key in signature.parameters}
            output = self.model(graph, **options)
        if not isinstance(output, dict) or output.get("energy") is None or output.get("forces") is None:
            raise ValueError("MACE producer must return energy and forces tensors")

        def numpy(value):
            return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

        energies = numpy(output["energy"]).reshape(-1)
        if len(energies) != 1:
            raise ValueError("MACE producer returned multiple energies for one structure")
        forces = numpy(output["forces"]) * self.energy_units_to_eV / self.length_units_to_A
        stress = None
        if self.compute_stress:
            if output.get("stress") is None:
                raise ValueError("MACE producer did not return the requested stress")
            stress = numpy(output["stress"]) * self.energy_units_to_eV / self.length_units_to_A**3
            if stress.size == 6:
                xx, yy, zz, yz, xz, xy = stress.reshape(-1)
                stress = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            else:
                stress = stress.reshape(3, 3)
        value = Evaluation(float(energies[0] * self.energy_units_to_eV), forces, stress)
        if value.forces_eV_A.shape != (structure.natoms, 3):
            raise ValueError("MACE producer force shape differs from the input structure")
        return value
