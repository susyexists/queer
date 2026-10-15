"""Finite-field execution and compatibility entrypoints for staged workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..io.artifacts import RunStore
from ..io.results import read_arrays, restore_result
from ..observables.result import ObservableResult
from .finite_field_maps import _analytical_member as _analytical_member
from .finite_field_maps import _crossing_map as _crossing_map
from .finite_field_maps import _harmonic_map as _harmonic_map
from .finite_field_maps import aggregate_member_runs as aggregate_member_runs
from .finite_field_maps import split_harmonic_map as split_harmonic_map
from .finite_field_recipes import _figure_orders as _figure_orders
from .finite_field_recipes import _paper_figure as _paper_figure
from .finite_field_sources import _model_files as _model_files
from .finite_field_sources import _path as _path
from .finite_field_sources import finite_field_input_files as finite_field_input_files
from .finite_field_sources import load_finite_field_source as load_finite_field_source
from .finite_field_specs import FINITE_FIELD_INPUTS as FINITE_FIELD_INPUTS
from .finite_field_specs import FINITE_FIELD_NAMES as FINITE_FIELD_NAMES
from .finite_field_specs import _numbers as _numbers
from .finite_field_specs import _options as _options
from .finite_field_specs import validate_finite_field as validate_finite_field

if TYPE_CHECKING:
    from ..config import WorkflowConfig


def _kwargs(config, source=None):
    kwargs = {
        "options": _options(config.parameters),
        "execution": config.execution,
        **{
            key: config.parameters[key]
            for key in (
                "plane",
                "band_pair",
                "occupied_bands",
                "layer_projector",
            )
            if key in config.parameters
        },
    }
    if "family_manifest" in config.inputs and getattr(source, "native_band_geometry", False):
        # The family loader already resolved global defaults and member
        # overrides. Keep those selections while allowing a supplied projector.
        for key in ("plane", "band_pair", "occupied_bands"):
            kwargs.pop(key, None)
    return kwargs


def _read(store, case):
    store.completed(case)
    record = store.manifest["cases"][case]
    return read_arrays(store.root / record["path"]), record["metadata"]


_restore_result = restore_result


def _commit_result(store, case, result):
    if not isinstance(result, ObservableResult):
        raise TypeError("Finite-field calculations must return ObservableResult")
    if case == "result":
        preset = result.metadata.get("options", {}).get("label")
        if result.metadata.get("observable") == "harmonic_map" and preset in {"fig5c", "ext7a"}:
            requested = (3, 7, 11, 15, 19, 23, 27, 31) if preset == "fig5c" else (3, 11, 19, 27)
            selected = [order for order in requested if order in result["orders"]]
            if selected:
                result = ObservableResult(
                    result.arrays,
                    result.dims,
                    result.units,
                    {**result.metadata, "figure": preset, "render_orders": selected},
                )
        scientific = store.manifest.setdefault("scientific_model", {})
        if "options" in result.metadata:
            scientific["resolved_options"] = result.metadata["options"]
            scientific["preset"] = result.metadata["options"].get(
                "label", result.metadata.get("figure", scientific.get("preset"))
            )
        if "figure" in result.metadata:
            scientific["recipe"] = result.metadata["figure"]
            if "options" not in result.metadata:
                scientific["preset"] = result.metadata["figure"]
        if "interpretation" in result.metadata:
            scientific["interpretation"] = result.metadata["interpretation"]
    store.commit(case, result.arrays, metadata=result.artifact_metadata())


def _prepare(config, source, store, namespace="stages"):
    from ..finite_field import PreparedHallResponse, decode_line, encode_line
    from ..finite_field.transport.hall import TransverseLayout

    def load_layout():
        case = f"{namespace}/layout"
        if not store.completed(case):
            return None
        arrays, metadata = _read(store, case)
        if not {"t_max", "mirrored", "flags"} <= set(metadata):
            raise ValueError("Resume rejected: incomplete transverse layout checkpoint")
        return TransverseLayout(
            arrays["offsets"],
            arrays["weights"],
            arrays["breaks"],
            tuple(metadata["t_max"]),
            bool(metadata["mirrored"]),
            set(metadata["flags"]),
        )

    def load_line(index, offset):
        case = f"{namespace}/lines/{index:06d}"
        if not store.completed(case):
            return None
        arrays, metadata = _read(store, case)
        if metadata["offset_Ainv"] != float(offset):
            raise ValueError("Resume rejected: transverse layout changed")
        return decode_line(arrays, metadata["line"])

    def save_line(index, offset, solution):
        arrays, metadata = encode_line(solution)
        store.commit(
            f"{namespace}/lines/{index:06d}",
            arrays,
            metadata={"offset_Ainv": float(offset), "line": metadata, "stage": "lines"},
        )

    def save_layout(layout):
        case = f"{namespace}/layout"
        arrays = {"offsets": layout.t_nodes, "weights": layout.t_weights, "breaks": layout.breaks}
        if store.completed(case):
            saved, _ = _read(store, case)
            if any(not np.array_equal(saved[key], value) for key, value in arrays.items()):
                raise ValueError("Resume rejected: transverse layout changed")
        else:
            store.commit(
                case,
                arrays,
                metadata={
                    "stage": "layout",
                    "t_max": list(layout.t_max),
                    "mirrored": bool(layout.mirrored),
                    "flags": sorted(layout.flags),
                },
            )

    prepared = PreparedHallResponse(
        source,
        cache_dir=config.parameters.get("cache_dir"),
        acceleration=config.parameters.get("acceleration"),
        **_kwargs(config, source),
    ).prepare(
        load_line=load_line,
        save_line=save_line,
        save_layout=save_layout,
        load_layout=load_layout,
    )
    case = f"{namespace}/tables"
    if not store.completed(case):
        store.commit(
            case,
            {"line_count": np.asarray(len(prepared.lines))},
            metadata={
                "stage": "tables",
                "diagnostics": prepared.describe(),
                "representation": "encoded per-line tables",
            },
        )
    return prepared


def calculate_finite_field(config: WorkflowConfig, source, store: RunStore):
    """Dispatch staged finite-field calculations under the existing RunStore owner."""
    if store.completed("result"):
        for case in store.manifest["cases"]:
            store.completed(case)
        return
    from .. import finite_field as engine

    name, params = config.workflow, config.parameters
    kwargs = _kwargs(config)
    if name == "paper_figure":
        _commit_result(store, "result", _paper_figure(config, store))
    elif name == "geometry_grid":
        from ..finite_field.geometry.gridded import GriddedGeometry
        from ..finite_field.geometry.provider import ModelGeometry
        from ..finite_field.native import NativePlaneModel

        if hasattr(source, "n_orb"):
            model = source
        else:
            model = NativePlaneModel(
                source,
                plane=params.get("plane"),
                band_pair=params.get("band_pair"),
                occupied_bands=params.get("occupied_bands"),
            )
        provider = ModelGeometry(model, patch=params.get("patch", "auto"))
        grid = GriddedGeometry.sample(
            provider, **{key: params[key] for key in ("r", "phi", "kx", "ky") if key in params}
        )
        arrays = dict(grid._raw)
        from ..finite_field._storage import json_safe

        arrays["__meta__"] = np.asarray(json.dumps(json_safe(grid.meta), allow_nan=False))
        store.commit(
            "result", arrays, metadata={"observable": name, "grid": grid.meta, "layout": grid.layout}
        )
    elif name == "family_report":
        from ..capabilities import CapabilityError

        try:
            critical = source.critical_D_report(
                **{key: params[key] for key in ("min_weight", "jump_tol", "mix_tol") if key in params},
                **({"ref": params["reference"]} if "reference" in params else {}),
            )
        except (CapabilityError, ValueError) as error:
            if not isinstance(error, CapabilityError) and any(member.has_model for member in source.members):
                raise
            critical = {"available": False, "reason": str(error)}
        report = {"consistency": source.consistency(), "critical": critical}
        store.commit(
            "result",
            {"D": np.asarray(source.D)},
            metadata={
                "observable": name,
                "dims": {"D": ("member",)},
                "units": {"D": "D"},
                **report,
            },
        )
    elif name in {"landau_zener_hall", "hall_harmonics"}:
        stage = "stages/fields" if name == "landau_zener_hall" else "stages/Fourier"
        if store.completed(stage):
            arrays, metadata = _read(store, stage)
            result = _restore_result(arrays, metadata)
            _commit_result(store, "result", result)
            return
        prepared = _prepare(config, source, store)
        if name == "landau_zener_hall":
            result = prepared.hall(params["fields"])
            _commit_result(store, "stages/fields", result)
        else:
            result = prepared.harmonics(params["amplitudes"], n_max=params.get("n_max"))
            _commit_result(store, "stages/Fourier", result)
        _commit_result(store, "result", result)
    elif name == "harmonic_map":
        result = _harmonic_map(config, source, store)
        _commit_result(store, "result", result)
    elif name == "crossing_scan":
        result = (
            _crossing_map(config, source, store)
            if "family_manifest" in config.inputs or "model_values" in params
            else engine.crossing_scan(source, params.get("offsets", [0.0]), **kwargs)
        )
        _commit_result(store, "result", result)
    elif name == "critical_hall":
        _commit_result(store, "result", engine.critical_hall(source, params["fields"], **kwargs))
    elif name == "houston_line":
        _commit_result(
            store,
            "result",
            engine.houston_line(
                source,
                params.get("offset", 0.0),
                params["fields"],
                window=params.get("window", (-0.03, 0.03)),
                **kwargs,
            ),
        )
    elif name == "lindblad_line":
        _commit_result(
            store,
            "result",
            engine.lindblad_line(
                source,
                params.get("offset", 0.0),
                params["fields"],
                window=params.get("window", (-0.03, 0.03)),
                **{key: params[key] for key in ("dphi", "basis") if key in params},
                **kwargs,
            ),
        )
    elif name == "berry_flux":
        _commit_result(
            store,
            "result",
            engine.berry_flux(
                source,
                **{
                    key: params[key]
                    for key in (
                        "plane",
                        "band_pair",
                        "occupied_bands",
                        "window",
                        "n",
                        "graded_scale",
                        "tolerance",
                    )
                    if key in params
                },
            ),
        )
    elif name == "fit_hall":
        fields, values = _fit_data(config)
        _commit_result(
            store,
            "result",
            engine.fit_hall(
                fields,
                values,
                window=tuple(params["window"]),
                f_bounds=params.get("f_bounds"),
                oscillations=params.get("oscillations", True),
            ),
        )


def _fit_data(config):
    params = config.parameters
    if "response_artifact" not in config.inputs:
        return params["fields"], params["values"]
    path = Path(config.inputs["response_artifact"])
    if path.is_dir():
        from ..io.artifacts import load_manifest

        manifest = load_manifest(path)
        path = path / manifest["cases"]["result"]["path"]
    with np.load(path, allow_pickle=False) as archive:
        fields = archive["amplitudes"] if "amplitudes" in archive else archive["fields"]
        key = params.get("value_key", "b" if "b" in archive else "current")
        values = archive[key]
        if "member" in params:
            values = values[int(params["member"])]
        if "harmonic" in params:
            values = values[..., int(params["harmonic"])]
        if np.asarray(values).ndim != 1:
            raise ValueError("fit_hall requires a one-dimensional series; select member/harmonic")
        return fields.copy(), values.copy()
