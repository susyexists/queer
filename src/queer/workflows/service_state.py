"""Force-provider resolution and durable atomistic restart adapters."""

import hashlib
import json

import numpy as np

from ..io.artifacts import canonical_json
from ..io.results import read_arrays


class HarmonicForceProvider:
    """Analytical fixture provider, E=k/2 sum(|r-r0|²), eV/Å units."""

    def __init__(self, structure, spring_constant_eV_A2=1.0, reference_positions=None):
        self.reference = np.asarray(
            structure.positions if reference_positions is None else reference_positions, float
        )
        self.spring = float(spring_constant_eV_A2)
        if self.reference.shape != structure.positions.shape or not np.isfinite(self.reference).all():
            raise ValueError("Harmonic reference positions must match structure atoms")
        if not np.isfinite(self.spring) or self.spring <= 0:
            raise ValueError("Harmonic spring constant must be positive eV/Å²")
        self.fingerprint = hashlib.sha256(
            canonical_json({"kind": "harmonic", "k": self.spring, "r0": self.reference.tolist()}).encode()
        ).hexdigest()

    def evaluate(self, structure):
        from ..structures import Evaluation

        if structure.positions.shape != self.reference.shape:
            raise ValueError("Harmonic provider reference must match the evaluated supercell")
        displacement = structure.positions - self.reference
        return Evaluation(0.5 * self.spring * float(np.sum(displacement**2)), -self.spring * displacement)


def resolve_provider(config, structure, provider=None):
    if provider is not None:
        return provider
    options = dict(config.parameters.get("provider", {}))
    kind = options.pop("kind", "mace" if "checkpoint" in config.inputs else "")
    if kind == "harmonic":
        return HarmonicForceProvider(structure, **options)
    if kind == "mace":
        if "checkpoint" not in config.inputs:
            raise ValueError("MACE workflow requires a resolved inputs.checkpoint file")
        from ..adapters.mace import MACEForceProvider

        return MACEForceProvider(checkpoint=config.material_dir / config.inputs["checkpoint"], **options)
    raise ValueError("Force workflows need a force_provider or a configured provider/checkpoint")


def _commit(store, result, **extra):
    from ..io.records import record_units

    arrays = {**result.arrays, **extra.pop("arrays", {})}
    metadata = {**result.artifact_metadata(), **extra}
    metadata["units"] = {**record_units(arrays), **result.units}
    store.commit(
        "result",
        arrays,
        metadata=metadata,
    )


def _record_provider(store, provider):
    store.manifest["scientific_model"].update(
        {
            "provider_fingerprint": getattr(provider, "fingerprint", None),
            "provider_provenance": getattr(provider, "provenance", {"class": type(provider).__qualname__}),
        }
    )
    store.save()


def _checkpoint(store, state):
    # Each state is an immutable case; a killed next update cannot invalidate it.
    current = state.get("current", {})
    step = int(current.get("step", 0))
    case = f"state_{step:012d}"
    if not store.completed(case):
        store.commit(
            case,
            {"state_json": np.asarray(canonical_json(state))},
            metadata={"kind": "restart_state", "step": step},
        )


def _latest_state(store):
    cases = [
        name
        for name, record in store.manifest["cases"].items()
        if record["metadata"].get("kind") == "restart_state"
    ]
    if not cases:
        return None
    name = max(cases)
    store.completed(name)
    record = read_arrays(store.root / store.manifest["cases"][name]["path"])
    return json.loads(str(record["state_json"]))
