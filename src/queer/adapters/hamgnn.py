"""Optional external-model and legacy PyG bridges; native numerics stay in Queer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io.hamiltonian import HamiltonianGraph
from ..io.openmx import BOHR_ANGSTROM, HARTREE_EV


def to_pyg(graph: HamiltonianGraph):
    """Convert declared scientific units to the existing HamGNN graph contract."""
    blocks = (graph.hamiltonian, graph.overlap) + (() if graph.h0 is None else (graph.h0,))
    if any(np.iscomplexobj(block) and np.any(block.imag) for block in blocks):
        raise ValueError("Legacy non-SOC HamGNN graphs cannot represent complex blocks")
    if graph.nao_max not in (14, 19, 26):
        raise ValueError("Legacy non-SOC HamGNN orbital width must be 14, 19, or 26")
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as error:
        raise ImportError("HamGNN serialization requires queer[hamgnn]") from error
    n = graph.natoms
    H = graph.hamiltonian.real.reshape(-1, graph.nao_max**2) / HARTREE_EV
    S = graph.overlap.real.reshape(-1, graph.nao_max**2)
    H0 = (graph.h0 if graph.h0 is not None else graph.hamiltonian).real.reshape(H.shape) / HARTREE_EV
    values = {
        "z": graph.atomic_numbers,
        "cell": graph.cell[None] / BOHR_ANGSTROM,
        "pos": graph.positions / BOHR_ANGSTROM,
        "node_counts": np.array([n]),
        "edge_index": graph.edge_index,
        "inv_edge_idx": graph.inverse_edges,
        "cell_shift": graph.cell_shift,
        "nbr_shift": graph.cell_shift @ graph.cell / BOHR_ANGSTROM,
        "hamiltonian": H,
        "overlap": S,
        "Hon": H[:n],
        "Hoff": H[n:],
        "Hon0": H0[:n],
        "Hoff0": H0[n:],
        "Son": S[:n],
        "Soff": S[n:],
        "doping_charge": np.array([graph.metadata.get("doping_charge", 0.0)], dtype=float),
    }
    return Data(
        **{
            key: torch.as_tensor(
                np.ascontiguousarray(value), dtype=torch.int64 if value.dtype.kind in "iu" else torch.float32
            )
            for key, value in values.items()
        }
    )


def write_legacy_graphs(path, graphs):
    """Explicit opt-in writer for the external pickled graph_data.npz format."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as stream:
        np.savez(stream, graph={key: to_pyg(value) for key, value in graphs.items()})
    return target


def read_legacy_graphs(path, *, basis_layout, trusted=False):
    """Read a trusted PyG pickle with an explicit checkpoint orbital layout."""
    if not trusted:
        raise ValueError("Legacy graph_data.npz contains pickle; trusted=True is required")
    with np.load(path, allow_pickle=True) as archive:
        graphs = archive["graph"].item()
    result = {}

    def array(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    for key, value in graphs.items():

        def get(name):
            return array(value[name] if isinstance(value, dict) else getattr(value, name))

        z, edges = get("z"), get("edge_index")
        H, S = get("hamiltonian"), get("overlap")
        if H.ndim != 2 or H.shape != S.shape:
            raise ValueError("legacy Hamiltonian and overlap must be equally shaped padded rows")
        width = int(round(np.sqrt(H.shape[1])))
        if width not in (14, 19, 26) or width**2 != H.shape[1]:
            raise ValueError("Legacy non-SOC HamGNN orbital width must be 14, 19, or 26")
        if any(np.iscomplexobj(block) and np.any(block.imag) for block in (H, S)):
            raise ValueError("Legacy non-SOC HamGNN graphs cannot represent complex blocks")
        slots: np.ndarray = np.full((len(z), width), -1, int)
        counts = []
        for i, element in enumerate(z):
            indices = (
                basis_layout[int(element)]
                if int(element) in basis_layout
                else basis_layout[str(int(element))]
            )
            counts.append(len(indices))
            slots[i, : len(indices)] = indices
        h0 = np.concatenate((get("Hon0"), get("Hoff0"))).reshape(-1, width, width) * HARTREE_EV
        if np.iscomplexobj(h0) and np.any(h0.imag):
            raise ValueError("Legacy non-SOC HamGNN graphs cannot represent complex blocks")
        doping = (
            value.get("doping_charge", [0.0])
            if isinstance(value, dict)
            else getattr(value, "doping_charge", [0.0])
        )
        doping = array(doping).reshape(-1)
        if doping.shape != (1,) or not np.isfinite(doping).all():
            raise ValueError("legacy doping_charge must contain one finite value per graph")
        result[str(key)] = HamiltonianGraph(
            z,
            get("cell").reshape(3, 3) * BOHR_ANGSTROM,
            get("pos") * BOHR_ANGSTROM,
            np.array(counts),
            slots,
            edges,
            get("cell_shift"),
            get("inv_edge_idx"),
            H.reshape(-1, width, width) * HARTREE_EV,
            S.reshape(-1, width, width),
            h0,
            {
                "has_scf_reference": False,
                "source": str(path),
                "reference_provenance": "not established by legacy archive",
                "doping_charge": float(doping[0]),
            },
        )
    return result


class HamGNNAdapter:
    """Execute an external Hamiltonian predictor; this is not a force provider."""

    def __init__(self, model, *, device="cpu", prediction_unit="Ha", fingerprint=None):
        if prediction_unit not in {"Ha", "eV"}:
            raise ValueError("prediction_unit must be Ha or eV")
        self.model, self.device, self.prediction_unit = model, device, prediction_unit
        self.fingerprint = fingerprint
        if fingerprint is not None and (not isinstance(fingerprint, str) or not fingerprint):
            raise ValueError("model fingerprint must be a nonempty string when supplied")
        if not callable(model):
            raise TypeError("HamGNN prediction model must be callable")
        self.provenance = {
            "producer": "HamGNN",
            "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "model_fingerprint": fingerprint,
            "device": str(device),
            "prediction_unit": prediction_unit,
        }
        if hasattr(model, "to"):
            model.to(device)
        if hasattr(model, "eval"):
            model.eval()

    @classmethod
    def from_checkpoint(cls, path, *, loader, device="cpu", prediction_unit="Ha"):
        from ..io.artifacts import digest_file

        checkpoint = Path(path).expanduser().resolve()
        instance = cls(
            loader(checkpoint),
            device=device,
            prediction_unit=prediction_unit,
            fingerprint=digest_file(checkpoint),
        )
        instance.provenance["checkpoint_sha256"] = instance.fingerprint
        return instance

    def predict(self, graph: HamiltonianGraph):
        import torch

        data = to_pyg(graph).to(self.device)
        # Single-structure batch fields preserve the producer's usual contract.
        data.batch = torch.zeros(graph.natoms, dtype=torch.int64, device=self.device)
        data.ptr = torch.tensor([0, graph.natoms], dtype=torch.int64, device=self.device)
        with torch.no_grad():
            output = self.model(data)
        value = output["hamiltonian"] if isinstance(output, dict) else output
        values = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        prediction = graph.with_predictions(values, unit=self.prediction_unit)
        prediction.metadata["prediction_provenance"] = dict(self.provenance)
        return prediction
