"""Declarative localized-orbital workflows and durable numerical case plans."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib import import_module

import numpy as np

from ..capabilities import CapabilityError


@dataclass(frozen=True)
class WorkflowSpec:
    module: str
    function: str
    capabilities: tuple[str, ...] = ()

    def calculator(self):
        return getattr(import_module(self.module, "queer"), self.function)

    def parameter_names(self):
        names = set(inspect.signature(self.calculator()).parameters) - {"model", "execution"}
        return (names - {"radial_data", "pao_dir"}) | {"ef", "grid", "kpoints", "integration"}


REGISTRY: dict[str, WorkflowSpec] = {}
for _module, _names in {
    ".observables.electronic": (
        "fat_band",
        "dos",
        "pdos",
        "jdos",
        "fermi_energy",
        "fermi_surface",
        "find_nodes",
        "spin_texture",
        "reduce_basis",
    ),
    ".observables.surface": ("surface_state",),
    ".observables.unfolding": ("bandunfolding", "bandunfolding_spin_texture"),
    ".observables.geometry": (
        "berry_curvature",
        "quantum_metric",
        "band_pair_geometry",
        "chern_number",
        "chirality",
        "wilson_loop",
        "polarization",
    ),
    ".observables.integrated": (
        "ahc",
        "anc",
        "berry_curvature_dipole",
        "orbital_magnetization",
        "shc",
        "drude_weight",
    ),
    ".observables.transport": ("boltz_transport",),
    ".observables.optical": ("optical_conductivity",),
    ".observables.nonlinear": ("shift_current", "cpge", "shg", "pockels", "second_order_static"),
}.items():
    for _name in _names:
        _capabilities: tuple[str, ...] = ()
        if _module.endswith(("geometry", "integrated", "optical", "nonlinear", "transport")):
            _capabilities = ("position",)
        if _name in {"bandunfolding", "bandunfolding_spin_texture"}:
            _capabilities = ("orbital_metadata",)
        REGISTRY[_name] = WorkflowSpec(_module, _name, _capabilities)

NAMES = tuple(REGISTRY)
# These calculations are linear BZ sums. Their unnormalized partials can be
# committed and reduced deterministically, independent of execution batching.
_INTEGRALS = {
    "dos",
    "pdos",
    "jdos",
    "ahc",
    "anc",
    "berry_curvature_dipole",
    "orbital_magnetization",
    "shc",
    "drude_weight",
    "optical_conductivity",
    "shift_current",
    "cpge",
    "shg",
    "pockels",
    "second_order_static",
    "boltz_transport",
}
_SAMPLED = {
    "fat_band",
    "spin_texture",
    "berry_curvature",
    "quantum_metric",
    "band_pair_geometry",
    "fermi_surface",
}
_TILE_SIZE = 128


def validate_parameters(name, parameters):
    spec = REGISTRY[name]
    unexpected = set(parameters) - spec.parameter_names()
    if unexpected:
        raise ValueError(f"Unknown {name} parameters: {', '.join(sorted(unexpected))}")
    signature = inspect.signature(spec.calculator())
    for key, value in signature.parameters.items():
        if key in {"model", "execution"} or value.default is not inspect.Parameter.empty:
            continue
        if key == "kpoints" and "grid" in parameters:
            continue
        if key not in parameters:
            raise ValueError(f"Missing {name} parameter: {key}")
    if "grid" in parameters:
        from ..observables._tb import kmesh

        kmesh(parameters["grid"])
    if "integration" in parameters:
        options = parameters["integration"]
        if name not in _INTEGRALS | {"chern_number"} or not isinstance(options, dict):
            raise ValueError("integration controls are supported for BZ integrals only")
        from ..observables.integration import validate_integration

        validate_integration(options)
        if options.get("method") in {"threshold", "cubature"} and name not in {
            "ahc",
            "chern_number",
            "shift_current",
        }:
            raise ValueError("local integration is supported for ahc, chern_number, and shift_current")


def _arguments(config, model):
    signature = inspect.signature(REGISTRY[config.workflow].calculator())
    params = {key: value for key, value in config.parameters.items() if key not in {"ef", "integration"}}
    if "kpoints" in signature.parameters and "kpoints" not in params and params.get("engine") != "fhs":
        from ..observables._tb import kmesh

        default_grid = signature.parameters.get("grid")
        params["kpoints"] = kmesh(params.get("grid", default_grid.default if default_grid else (8, 8, 8)))
    if "grid" not in signature.parameters:
        params.pop("grid", None)
    if "pao_dir" in signature.parameters and "pao_dir" in config.inputs:
        params["pao_dir"] = str(config.material_dir / config.inputs["pao_dir"])
    if "radial_data" in signature.parameters and "pao_files" in config.inputs:
        params["radial_data"] = {
            key: str(config.material_dir / value) for key, value in config.inputs["pao_files"].items()
        }
    params["execution"] = config.execution
    return params


def _save_result(store, case, result, **metadata):
    store.commit(case, result.arrays, metadata={**result.artifact_metadata(), **metadata})


def _reduce(store, cases, name, sampled=False):
    records = [store.manifest["cases"][case] for case in cases]
    metadata = records[0]["metadata"]
    result: dict[str, np.ndarray] = {}
    concatenated: dict[str, list[np.ndarray]] = {}
    axes = metadata.get("dims", {})
    for record in records:
        with np.load(store.root / record["path"], allow_pickle=False) as data:
            for key in data.files:
                value = data[key]
                dimensions = axes.get(key, [])
                if (sampled or name == "boltz_transport") and any(
                    axis in {"k", "kpoint"} for axis in dimensions
                ):
                    concatenated.setdefault(key, []).append(value)
                elif name == "boltz_transport" and key in {
                    "electron_density",
                    "hole_density",
                    "carrier_dos",
                    "electron_count",
                    "net_carrier_density",
                    "inverse_mass_normalization",
                }:
                    result[key] = result.get(key, 0) + value * record["metadata"]["quadrature_weight"]
                elif (
                    not sampled
                    and key in axes
                    and value.dtype.kind in "fci"
                    and any(
                        axis
                        in {"cartesian", "cartesian2", "cartesian_a", "cartesian_b", "cartesian_c", "energy"}
                        for axis in dimensions
                    )
                    and key not in {"omega", "frequencies", "frequencies_eV"}
                ):
                    weighted = value * record["metadata"]["quadrature_weight"]
                    result[key] = result.get(key, 0) + weighted
                elif key not in result:
                    result[key] = value
    for key, values in concatenated.items():
        axis = next(i for i, value in enumerate(axes[key]) if value in {"k", "kpoint"})
        result[key] = np.concatenate(values, axis=axis)
    if name == "pockels":
        from ..observables.nonlinear import pockels_from_tensors

        result["pockels"] = pockels_from_tensors(result["chi2"], result["dielectric"])
    elif name == "boltz_transport":
        from ..observables.transport import finalize_transport

        result = finalize_transport(result)
    metadata = {
        **metadata,
        "observable": name,
        "reduction": "deterministic scientific tiles",
        "partial_cases": cases,
        "quadrature_weight": 1.0,
        "sample_count": sum(record["metadata"]["sample_count"] for record in records),
        "kpoint_count": sum(record["metadata"]["sample_count"] for record in records),
        "integration_weights": {
            "definition": "normalized nonnegative mesh weights",
            "storage": "global normalized weights in each partial case's metadata",
        },
    }
    return result, metadata


def calculate_registered(config, model, store):
    """Run or resume immutable scientific tiles, then commit the final result."""
    from ..observables._tb import kmesh, normalized_weights

    name = config.workflow
    for case in store.manifest["cases"]:
        store.completed(case)
    if store.completed("result"):
        return
    spec = REGISTRY[name]
    for capability in spec.capabilities:
        if getattr(model, "data", None) is not None and capability not in getattr(model, "capabilities", ()):
            raise CapabilityError(f"{name} requires model capability {capability}")
    function, params = spec.calculator(), _arguments(config, model)
    signature = inspect.signature(function)
    integration = config.parameters.get("integration", {})
    if integration.get("method") in {"threshold", "cubature"} or (name == "chern_number" and integration):
        from ..observables.integration import quadrature_checkpoint

        params["integration"] = integration
        if "kpoints" not in config.parameters:
            params.pop("kpoints", None)
        with quadrature_checkpoint(store):
            result = function(model, **params)
        _save_result(store, "result", result)
        if "integration" in result.metadata:
            store.manifest["integration"] = result.metadata["integration"]
            store.save()
        return
    linear = (
        name in _INTEGRALS
        and config.parameters.get("engine") != "fhs"
        and not (
            name == "boltz_transport" and config.parameters.get("transport_method", "CRTA").upper() != "CRTA"
        )
    )
    sampled = name in _SAMPLED
    integration = config.parameters.get("integration", {})
    adaptive = integration.get("method") == "adaptive"
    if not (linear or sampled or adaptive):
        _save_result(store, "result", function(model, **params))
        return
    points = np.asarray(
        params["kpoints"] if params.get("kpoints") is not None else kmesh(params.get("grid", (8, 8, 8))),
        dtype=float,
    )
    weights = normalized_weights(points, params.get("weights"))
    if adaptive and ("kpoints" in config.parameters or "weights" in config.parameters):
        raise ValueError("Adaptive integration requires a grid without explicit kpoints or weights")
    params.pop("grid", None)
    maximum_levels = integration.get("max_levels", 4) if adaptive else 1
    default_grid = signature.parameters.get("grid")
    base_grid = np.asarray(
        config.parameters.get("grid", default_grid.default if default_grid else (8, 8, 8)), int
    )
    active = base_grid > 1
    if not active.any():
        active[:] = True
        if name == "ahc" and params.get("engine") == "fhs" and params.get("dimension", 3) == 2:
            plane = params.get("plane", (0, 1))
            active[list(set(range(3)) - set(plane))] = False
    previous = None
    for level in range(maximum_levels):
        grid = base_grid * np.where(active, 2**level, 1)
        if level:
            points = kmesh(grid)
            weights = normalized_weights(points)
        if adaptive and len(points) > integration.get("max_points", 1000000):
            raise RuntimeError("Adaptive integration exhausted max_points before convergence")
        cases = []
        # FHS plaquettes and EMPC scattering depend on a complete mesh. Each
        # refinement therefore owns one durable global case, rather than tiles.
        if not (linear or sampled):
            case = f"level_{level:02d}_global"
            cases.append(case)
            if not store.completed(case):
                global_params = dict(params)
                if "grid" in signature.parameters:
                    global_params["grid"] = grid
                if config.parameters.get("engine") == "fhs":
                    global_params.pop("kpoints", None)
                    global_params.pop("weights", None)
                else:
                    global_params["kpoints"] = points
                    if "weights" in signature.parameters:
                        global_params["weights"] = weights
                _save_result(
                    store,
                    case,
                    function(model, **global_params),
                    quadrature_weight=1.0,
                    sample_count=len(points),
                    refinement_level=level,
                    integration_weights=weights.tolist(),
                )
            record = store.manifest["cases"][case]
            with np.load(store.root / record["path"], allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            metadata = {**record["metadata"], "partial_cases": cases}
        for start in range(0, len(points) if linear or sampled else 0, _TILE_SIZE):
            case = f"level_{level:02d}_tile_{start // _TILE_SIZE:08d}"
            cases.append(case)
            if store.completed(case):
                continue
            end = start + _TILE_SIZE
            weight_sum = float(weights[start:end].sum())
            if not sampled and weight_sum == 0:
                cases.pop()
                continue
            tile_params = {**params, "kpoints": points[start:end]}
            if "weights" in signature.parameters:
                tile_params["weights"] = weights[start:end]
            result = function(model, **tile_params)
            if name == "second_order_static" and cases[:-1]:
                ranks = result.metadata.get("occupied_rank_by_channel")
                first_ranks = store.manifest["cases"][cases[0]]["metadata"].get("occupied_rank_by_channel")
                if ranks is not None and ranks != first_ranks:
                    raise ValueError(
                        "second_order_static requires the same insulating occupied rank across all kpoints"
                    )
            _save_result(
                store,
                case,
                result,
                quadrature_weight=weight_sum,
                sample_count=len(points[start:end]),
                refinement_level=level,
                integration_weights=weights[start:end].tolist(),
                kpoint_range=[start, min(end, len(points))],
            )
        if linear or sampled:
            arrays, metadata = _reduce(store, cases, name, sampled)
            metadata["evaluated_kpoint_count"] = metadata["kpoint_count"]
            metadata["kpoint_count"] = len(points)
        if maximum_levels == 1:
            if adaptive:
                raise RuntimeError("Adaptive integration requires at least two levels to estimate error")
            store.commit("result", arrays, metadata=metadata)
            return
        current = {
            key: value
            for key, value in arrays.items()
            if value.dtype.kind in "fc"
            and key not in {"omega", "frequencies", "frequencies_eV", "chemical_potentials", "temperatures"}
            and not any(axis in {"k", "kpoint"} for axis in metadata.get("dims", {}).get(key, []))
        }
        if previous is not None:
            diagnostics = {}
            for key, value in current.items():
                error = float(np.max(np.abs(value - previous[key]), initial=0))
                scale = float(np.max(np.abs(value), initial=0))
                threshold = (
                    integration.get("absolute_tolerance", 1e-6)
                    + integration.get("relative_tolerance", 1e-4) * scale
                )
                diagnostics[key] = {"error_estimate": error, "threshold": threshold}
            converged = all(
                record["error_estimate"] <= record["threshold"] for record in diagnostics.values()
            )
            store.manifest["integration"] = {
                "method": "nested_grid_refinement",
                "level": level,
                "diagnostics": diagnostics,
                "points": len(points),
                "converged": converged,
                "base_grid": base_grid.tolist(),
                "active_axes": active.tolist(),
            }
            store.save()
            if converged:
                store.commit(
                    "result", arrays, metadata={**metadata, "integration": store.manifest["integration"]}
                )
                return
        previous = current
        reduction_case = f"level_{level:02d}_reduction"
        if not store.completed(reduction_case):
            store.commit(reduction_case, arrays, metadata=metadata)
    raise RuntimeError("Adaptive integration exhausted max_levels before convergence")
