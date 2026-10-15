"""Declared scientific reductions of immutable sampled and additive cases."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..io.results import RunStoreCheckpoint


@dataclass(frozen=True)
class ReductionSpec:
    arrays: Mapping[str, str]
    finalizer: str | None = None
    invariant_metadata: tuple[str, ...] = ()

    @property
    def convergence_fields(self):
        return tuple(key for key, operation in self.arrays.items() if operation in {"sum", "derived"})


def _spec(*, summed=(), sampled=(), invariant=(), derived=(), finalizer=None, metadata=()):
    return ReductionSpec(
        {
            **dict.fromkeys(summed, "sum"),
            **dict.fromkeys(sampled, "concat"),
            **dict.fromkeys(invariant, "invariant"),
            **dict.fromkeys(derived, "derived"),
        },
        finalizer,
        tuple(metadata),
    )


REDUCTIONS = {
    "dos": _spec(summed=("dos",), invariant=("omega",)),
    "pdos": _spec(summed=("pdos", "dos"), invariant=("omega", "groups")),
    "jdos": _spec(summed=("jdos",), invariant=("omega",)),
    "fat_band": _spec(
        sampled=("kpoints", "energies", "weights", "manifold_sizes"), invariant=("groups", "band_channels")
    ),
    "spin_texture": _spec(
        sampled=("kpoints", "energies", "spin", "manifold_sizes"), invariant=("band_channels",)
    ),
    "berry_curvature": _spec(sampled=("kpoints", "curvature")),
    "quantum_metric": _spec(sampled=("kpoints", "metric", "trace")),
    "band_pair_geometry": _spec(
        sampled=(
            "kpoints",
            "energies",
            "gap",
            "energy_reference",
            "precision_floor",
            "pair_metric",
            "total_metric",
            "curvature",
            "velocity_cv",
            "connection_cv",
            "valid",
            "pair_valid",
            "gap_valid",
            "band_valid",
        ),
        metadata=("bands",),
    ),
    "fermi_surface": _spec(sampled=("kpoints", "energies", "mask", "weights")),
    "optical_conductivity": _spec(summed=("conductivity", "dielectric"), invariant=("omega",)),
    "shg": _spec(summed=("shg",), invariant=("omega",)),
    "shift_current": _spec(summed=("shift_current",), invariant=("omega",)),
    "cpge": _spec(summed=("cpge", "helicity_rate", "steady_cpge"), invariant=("omega",)),
    "pockels": _spec(
        summed=("chi2", "dielectric"), invariant=("omega",), derived=("pockels",), finalizer="pockels"
    ),
    "second_order_static": _spec(summed=("second_order_static",), metadata=("occupied_rank_by_channel",)),
    "orbital_magnetization": _spec(
        summed=("orbital_magnetization", "local", "itinerant"), invariant=("chemical_potentials",)
    ),
    "boltz_transport": _spec(
        summed=(
            "transport_moment_0",
            "transport_moment_1",
            "transport_moment_2",
            "electron_density",
            "hole_density",
            "carrier_dos",
            "electron_count",
            "net_carrier_density",
            "inverse_mass_numerator",
            "inverse_mass_normalization",
        ),
        sampled=(
            "relaxation_time",
            "energies_absolute",
            "velocities",
            "intraband_velocity_products",
            "resolved_band_mask",
            "inverse_mass",
            "scattering_dos",
        ),
        invariant=("temperatures", "chemical_potentials"),
        derived=(
            "conductivity",
            "seebeck",
            "thermal_conductivity",
            "mobility",
            "mobility_valid_mask",
            "effective_mass",
            "effective_mass_valid_mask",
        ),
        finalizer="transport",
    ),
}
for _name in ("ahc", "anc", "berry_curvature_dipole", "shc", "drude_weight"):
    REDUCTIONS[_name] = _spec(summed=(_name,), invariant=("chemical_potentials",))

SAMPLED = frozenset(
    ("fat_band", "spin_texture", "berry_curvature", "quantum_metric", "band_pair_geometry", "fermi_surface")
)
ADDITIVE = frozenset(REDUCTIONS) - SAMPLED


def calculation_policy(name, parameters):
    if name in SAMPLED:
        return "sampled"
    if (
        name in ADDITIVE
        and parameters.get("engine") != "fhs"
        and not (name == "boltz_transport" and parameters.get("transport_method", "CRTA").upper() != "CRTA")
    ):
        return "additive"
    return "global"


def reduce_cases(store, cases, name, spec=None):
    """Reduce in scientific case order; nonlinear products are finalized once."""
    spec = spec or REDUCTIONS[name]
    reader = RunStoreCheckpoint(store)
    result: dict[str, np.ndarray] = {}
    concatenated: dict[str, list[np.ndarray]] = {}
    first_metadata = None
    first_keys = None
    reference_arrays = {}
    records = []
    for case in cases:
        saved = reader.load(case)
        if saved is None:
            raise ValueError(f"Missing scientific partial: {case}")
        arrays, metadata = saved
        if set(arrays) - set(spec.arrays):
            raise ValueError(f"Undeclared {name} reduction arrays: {sorted(set(arrays) - set(spec.arrays))}")
        if first_metadata is None:
            first_metadata, first_keys = metadata, set(arrays)
            reference_arrays = arrays
        elif (
            set(arrays) != first_keys
            or {key: tuple(value) for key, value in metadata.get("dims", {}).items()}
            != {key: tuple(value) for key, value in first_metadata.get("dims", {}).items()}
            or (metadata.get("units", {}) != first_metadata.get("units", {}))
        ):
            raise ValueError(f"Incompatible {name} partial schemas")
        for key in spec.invariant_metadata:
            if metadata.get(key) != first_metadata.get(key):
                raise ValueError(f"{name} requires the same {key} across all kpoints")
        weight = metadata.get("quadrature_weight")
        if not isinstance(weight, (float, int)) or not np.isfinite(weight) or weight < 0:
            raise ValueError("Scientific partial weights must be finite and nonnegative")
        records.append(metadata)
        for key, value in arrays.items():
            operation = spec.arrays[key]
            if operation not in {"concat", "sum", "invariant", "derived"}:
                raise ValueError(f"Unknown reduction operation for {key}: {operation}")
            reference = reference_arrays[key]
            if value.dtype != reference.dtype:
                raise ValueError(f"Incompatible {name} partial dtype: {key}")
            if operation == "concat":
                dimensions = metadata.get("dims", {}).get(key, ())
                axis = next((i for i, dim in enumerate(dimensions) if dim in {"k", "kpoint"}), None)
                if axis is None:
                    raise ValueError(f"Sampled array {key} lacks a point dimension")
                if value.ndim != reference.ndim or any(
                    size != reference.shape[i] for i, size in enumerate(value.shape) if i != axis
                ):
                    raise ValueError(f"Incompatible {name} sampled shape: {key}")
                concatenated.setdefault(key, []).append(value)
            elif operation == "sum":
                if value.shape != reference.shape:
                    raise ValueError(f"Incompatible {name} additive shape: {key}")
                result[key] = result.get(key, 0) + value * metadata["quadrature_weight"]
            elif operation == "invariant":
                if key in result and (
                    result[key].dtype != value.dtype or not np.array_equal(result[key], value)
                ):
                    raise ValueError(f"Incompatible {name} partial coordinate: {key}")
                result.setdefault(key, value)
    if first_metadata is None:
        raise ValueError("No scientific partials to reduce")
    total_weight = sum(record["quadrature_weight"] for record in records)
    if not np.isclose(total_weight, 1.0, rtol=1e-12, atol=1e-12):
        raise ValueError("Scientific partial weights must sum to one")
    axes = first_metadata.get("dims", {})
    for key, values in concatenated.items():
        axis = next((i for i, dim in enumerate(axes[key]) if dim in {"k", "kpoint"}), None)
        if axis is None:
            raise ValueError(f"Sampled array {key} lacks a point dimension")
        result[key] = np.concatenate(values, axis=axis)
    if spec.finalizer == "pockels":
        from ..observables.nonlinear import pockels_from_tensors

        result["pockels"] = pockels_from_tensors(result["chi2"], result["dielectric"])
    elif spec.finalizer == "transport":
        from ..observables.transport import finalize_transport

        result = finalize_transport(result)
    metadata = {
        **first_metadata,
        "observable": name,
        "reduction": "deterministic scientific tiles",
        "partial_cases": cases,
        "quadrature_weight": 1.0,
        "sample_count": sum(record["sample_count"] for record in records),
        "kpoint_count": sum(record["sample_count"] for record in records),
        "integration_weights": {
            "definition": "normalized nonnegative mesh weights",
            "storage": "global normalized weights in each partial case's metadata",
        },
    }
    return result, metadata
