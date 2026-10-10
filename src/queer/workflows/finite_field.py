"""Finite-field workflow specifications and owner-written numerical checkpoints.

These workflows have preparation and propagation stages, rather than the
single point-reducer contract used by equilibrium observables.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import numpy as np

from ..io.artifacts import RunStore
from ..observables.result import ObservableResult

if TYPE_CHECKING:
    from ..config import WorkflowConfig

FINITE_FIELD_NAMES = (
    "geometry_grid",
    "crossing_scan",
    "landau_zener_hall",
    "hall_harmonics",
    "harmonic_map",
    "family_report",
    "critical_hall",
    "fit_hall",
    "houston_line",
    "lindblad_line",
    "berry_flux",
    "paper_figure",
)
FINITE_FIELD_INPUTS = {
    "geometry_grid",
    "family_manifest",
    "response_artifact",
    "tb",
    "centres",
    "wout",
}
_COMMON = {
    "model",
    "preset",
    "options",
    "plane",
    "band_pair",
    "occupied_bands",
    "layer_projector",
    "conventions",
    "relaxation",
    "crossings",
    "phase",
    "quadrature",
    "fourier",
    "drive",
    "ef",
    "interpolation",
    "use_symmetry_shortcuts",
    "cache_dir",
    "lattice",
    "acceleration",
}
_PARAMETERS = {
    "geometry_grid": {"r", "phi", "kx", "ky", "patch"},
    "crossing_scan": {"offsets", "model_values", "model_parameter"},
    "landau_zener_hall": {"fields"},
    "hall_harmonics": {"amplitudes", "n_max"},
    "harmonic_map": {"amplitudes", "n_max", "normalize_by_Dc", "model_values", "model_parameter"},
    "family_report": {"reference", "min_weight", "jump_tol", "mix_tol"},
    "critical_hall": {"fields"},
    "fit_hall": {"fields", "values", "window", "f_bounds", "value_key", "harmonic", "member", "oscillations"},
    "houston_line": {"offset", "fields", "window"},
    "lindblad_line": {"offset", "fields", "window", "dphi", "basis"},
    "berry_flux": {"window", "n", "graded_scale", "tolerance"},
    "paper_figure": {"figure", "recipe"},
}


def _numbers(value, name, *, positive=False, count=None):
    values = np.asarray(value)
    if values.ndim != 1 or not len(values) or values.dtype.kind not in "fiu":
        raise ValueError(f"parameters.{name} must be a nonempty numerical list")
    if not np.isfinite(values).all() or (positive and np.any(values <= 0)):
        raise ValueError(f"parameters.{name} must contain finite {'positive ' if positive else ''}values")
    if count is not None and len(values) != count:
        raise ValueError(f"parameters.{name} must have {count} values")


def validate_finite_field(name, inputs, params):
    allowed = _PARAMETERS[name] if name == "paper_figure" else _COMMON | _PARAMETERS[name]
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} parameters: {', '.join(sorted(unknown))}")
    sources = [key for key in ("geometry_grid", "family_manifest", "model") if key in inputs]
    if name == "fit_hall":
        if sources:
            raise ValueError("fit_hall consumes response data, not a band model")
        if "response_artifact" not in inputs and not {"fields", "values"} <= set(params):
            raise ValueError("fit_hall needs inputs.response_artifact or parameters.fields and values")
        if "window" not in params:
            raise ValueError("fit_hall requires an explicit parameters.window")
    elif len(sources) != 1:
        raise ValueError("Specify exactly one inputs.model, geometry_grid, or family_manifest")
    if "model" in inputs:
        if inputs["model"] not in {"fourband", "wannier", "openmx"}:
            raise ValueError("Finite-field inputs.model must be fourband, wannier, or openmx")
        if inputs["model"] != "fourband" and "material_dir" not in inputs:
            raise ValueError("File-backed models require inputs.material_dir")
    if name == "family_report" and "family_manifest" not in inputs:
        raise ValueError("family_report requires inputs.family_manifest")
    if name == "paper_figure":
        if inputs.get("model") != "fourband":
            raise ValueError("paper_figure uses the analytical fourband reference")
        if not isinstance(params.get("figure"), str) or not params["figure"]:
            raise ValueError("paper_figure requires a named parameters.figure")
        if "recipe" in params and not isinstance(params["recipe"], Mapping):
            raise ValueError("parameters.recipe must be a table")
        from ..finite_field.figures import FIGURES

        if params["figure"] not in FIGURES:
            raise ValueError(f"Unknown paper figure {params['figure']!r}; choose {', '.join(FIGURES)}")
    if name in {"berry_flux", "houston_line", "lindblad_line"} and "geometry_grid" in inputs:
        raise ValueError(f"{name} requires wavefunctions from a band model")
    if name not in {"family_report", "harmonic_map", "crossing_scan"} and "family_manifest" in inputs:
        raise ValueError(f"{name} requires one model or geometry grid, not a family")
    if name == "harmonic_map" and "family_manifest" not in inputs and "model_values" not in params:
        raise ValueError("harmonic_map needs a family manifest or parameters.model_values")
    if "model_values" in params:
        if "family_manifest" in inputs:
            raise ValueError("Choose a family manifest or model_values for an ensemble")
        if inputs.get("model") != "fourband":
            raise ValueError("model_values scans require inputs.model='fourband'")
    for key in ("fields", "amplitudes", "offsets", "model_values", "r", "phi", "kx", "ky", "values"):
        if key in params:
            _numbers(params[key], key, positive=key in {"amplitudes", "r"})
    for key in ("window", "f_bounds"):
        if key in params:
            if name == "berry_flux" and key == "window":
                values = np.atleast_1d(params[key])
                _numbers(values, key, positive=True)
                if len(values) not in {1, 2}:
                    raise ValueError("parameters.window needs one or two positive half-widths")
                continue
            _numbers(params[key], key, count=2)
            if params[key][1] <= params[key][0]:
                raise ValueError(f"parameters.{key} must have increasing bounds")
    required = {
        "landau_zener_hall": {"fields"},
        "hall_harmonics": {"amplitudes"},
        "harmonic_map": {"amplitudes"},
        "critical_hall": {"fields"},
        "houston_line": {"fields"},
        "lindblad_line": {"fields"},
    }.get(name, set())
    if missing := required - set(params):
        raise ValueError(f"Missing {name} parameters: {', '.join(sorted(missing))}")
    if "n_max" in params and (
        isinstance(params["n_max"], bool) or not isinstance(params["n_max"], int) or params["n_max"] < 1
    ):
        raise ValueError("parameters.n_max must be a positive integer")
    if "acceleration" in params and not isinstance(params["acceleration"], bool):
        raise ValueError("parameters.acceleration must be a boolean")
    if "n" in params and (
        isinstance(params["n"], bool) or not isinstance(params["n"], int) or params["n"] < 3
    ):
        raise ValueError("parameters.n must be an integer of at least three")
    for key in ("dphi", "graded_scale", "tolerance"):
        if key in params and (
            isinstance(params[key], bool)
            or not isinstance(params[key], (int, float))
            or not np.isfinite(params[key])
            or params[key] <= 0
        ):
            raise ValueError(f"parameters.{key} must be finite and positive")
    if "offset" in params and (
        isinstance(params["offset"], bool)
        or not isinstance(params["offset"], (int, float))
        or not np.isfinite(params["offset"])
    ):
        raise ValueError("parameters.offset must be finite")
    if name in {"houston_line", "lindblad_line"} and np.any(np.asarray(params["fields"]) == 0):
        raise ValueError(f"{name} requires nonzero fields")
    if "basis" in params and params["basis"] not in {"adiabatic", "superadiabatic"}:
        raise ValueError("parameters.basis must be adiabatic or superadiabatic")
    if "band_pair" in params:
        values = params["band_pair"]
        if (
            not isinstance(values, list)
            or len(values) != 2
            or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in values)
            or values[0] == values[1]
        ):
            raise ValueError("parameters.band_pair must contain two distinct nonnegative band indices")
    if "occupied_bands" in params and (
        isinstance(params["occupied_bands"], bool)
        or not isinstance(params["occupied_bands"], int)
        or params["occupied_bands"] < 1
    ):
        raise ValueError("parameters.occupied_bands must be a positive integer")
    if name == "geometry_grid":
        polar = {"r", "phi"} <= set(params)
        cartesian = {"kx", "ky"} <= set(params)
        if (
            polar == cartesian
            or (polar and set(params) & {"kx", "ky"})
            or (cartesian and set(params) & {"r", "phi"})
        ):
            raise ValueError("geometry_grid requires r/phi or kx/ky coordinates")
    for key in (
        "model",
        "plane",
        "conventions",
        "relaxation",
        "crossings",
        "phase",
        "quadrature",
        "fourier",
        "drive",
    ):
        if key in params and not isinstance(params[key], Mapping):
            raise ValueError(f"parameters.{key} must be a table")
    for key in ("hr", "tb", "centres", "wout", "poscar", "wsvec", "scfout", "openmx_input"):
        if key in inputs and (not isinstance(inputs[key], str) or not inputs[key]):
            raise ValueError(f"inputs.{key} must be a nonempty filename")
    from ..finite_field.config import resolve_options

    resolve_options(_options(params))
    if "plane" in params:
        from ..finite_field.native import FieldPlane

        FieldPlane(**params["plane"])


def _path(directory, value):
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else directory / candidate).resolve()


def finite_field_input_files(config: WorkflowConfig) -> list[Path]:
    """Hash every consumed file, including the leaves of a family manifest."""
    inputs, params = config.inputs, config.parameters
    if "family_manifest" in inputs:
        family_path = Path(inputs["family_manifest"])
        data = json.loads(family_path.read_text())
        members = data if isinstance(data, list) else data.get("members", data.get("models", []))
        files = [family_path]
        for member in members:
            leaves = member.get("files", {})
            kind = member.get("model", "openmx" if "scfout" in leaves else "wannier")
            resolved = {key: _path(family_path.parent, value) for key, value in leaves.items() if value}
            files.extend(resolved.values())
            model_files = {key: str(path) for key, path in resolved.items() if key != "grid"}
            if model_files:
                directory = Path(next(iter(model_files.values()))).parent
                family_params = {
                    **params,
                    "interpolation": "ws_distance" if "wsvec" in model_files else "standard",
                }
                files.extend(
                    _model_files({"material_dir": directory, "model": kind, **model_files}, family_params)
                )
        return sorted(set(files))
    for key in ("geometry_grid", "response_artifact"):
        if key in inputs:
            source = Path(inputs[key])
            if source.is_dir():
                from ..io.artifacts import load_manifest

                manifest = load_manifest(source)
                return [
                    source / "manifest.json",
                    *[source / item["path"] for item in manifest["cases"].values()],
                ]
            return [source]
    return _model_files(inputs, params)


def _model_files(inputs, params):
    if inputs.get("model") == "fourband":
        return []
    directory = Path(inputs["material_dir"])
    if inputs.get("model") == "openmx":
        files = (
            [_path(directory, inputs["scfout"])] if "scfout" in inputs else list(directory.glob("*.scfout"))
        )
        if "scfout" not in inputs and len(files) != 1:
            raise ValueError("Provide inputs.scfout when there is not exactly one SCFOUT file")
        files.extend(_path(directory, inputs[key]) for key in ("openmx_input", "poscar") if key in inputs)
        return files
    from ..models.wannier import WannierModel

    model = WannierModel.from_directory(
        directory,
        hr=inputs.get("hr", "wannier90_hr.dat"),
        poscar=inputs.get("poscar"),
        ef=params.get("ef", 0.0),
        interpolation=params.get("interpolation", "standard"),
        wsvec=inputs.get("wsvec"),
        **{key: inputs[key] for key in ("tb", "centres", "wout") if key in inputs},
    )
    return [Path(path) for path in model.source_files.values()]


def load_finite_field_source(config: WorkflowConfig):
    inputs, params = config.inputs, config.parameters
    if config.workflow == "fit_hall":
        return None
    if "geometry_grid" in inputs:
        from ..finite_field.geometry.gridded import GriddedGeometry

        path = Path(inputs["geometry_grid"])
        if path.is_dir():
            from ..io.artifacts import load_manifest

            manifest = load_manifest(path)
            path = path / manifest["cases"]["result"]["path"]
        from ..finite_field.config import resolve_options

        return GriddedGeometry.from_npz(path, angle=resolve_options(_options(params)).drive.angle)
    if "family_manifest" in inputs:
        from ..finite_field.family import ModelFamily

        return ModelFamily.from_manifest(
            inputs["family_manifest"],
            n_occ=params.get("occupied_bands"),
            plane=params.get("plane"),
            band_pair=params.get("band_pair"),
        )
    if inputs["model"] == "fourband":
        from ..finite_field.models.fourband import FourBandModel, FourBandParams

        values = dict(params.get("model", {}))
        ratio = values.pop("x", None)
        model_options = {
            key: values.pop(key)
            for key in (
                "warping",
                "warping_matrix",
                "d3_layer",
                "include_identity",
            )
            if key in values
        }
        parameters = FourBandParams(**values)
        if ratio is not None:
            parameters = parameters.at(float(ratio))
        return FourBandModel(parameters, **model_options)
    if inputs["model"] == "openmx":
        from ..models.openmx import OpenMXModel

        return OpenMXModel.from_directory(
            config.material_dir,
            scfout=inputs.get("scfout"),
            ef=params.get("ef"),
            openmx_input=inputs.get("openmx_input"),
            execution=config.execution,
        )
    from ..models.wannier import WannierModel

    model = WannierModel.from_directory(
        config.material_dir,
        hr=inputs.get("hr", "wannier90_hr.dat"),
        poscar=inputs.get("poscar"),
        ef=params.get("ef", 0.0),
        interpolation=params.get("interpolation", "standard"),
        wsvec=inputs.get("wsvec"),
        **{key: inputs[key] for key in ("tb", "centres", "wout") if key in inputs},
    )
    if model.g_vec is None and "lattice" in params:
        from ..geometry.lattice import reciprocal_lattice

        model.lattice = np.asarray(params["lattice"], float)
        model.g_vec = reciprocal_lattice(model.lattice)
    return model


def _options(params):
    options = params.get("options", {})
    if isinstance(options, str):
        options = {"preset": options}
    elif not isinstance(options, Mapping):
        raise ValueError("parameters.options must be a table or preset name")
    return {
        "preset": params.get("preset", "physical"),
        **options,
        **{
            key: params[key]
            for key in (
                "conventions",
                "relaxation",
                "crossings",
                "phase",
                "quadrature",
                "fourier",
                "drive",
                "use_symmetry_shortcuts",
            )
            if key in params
        },
    }


def _kwargs(config):
    return {
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


def _read(store, case):
    record = store.manifest["cases"][case]
    with np.load(store.root / record["path"], allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}, record["metadata"]


def _restore_result(arrays, metadata):
    return ObservableResult(
        arrays,
        metadata.get("dims", {}),
        metadata.get("units", {}),
        {key: value for key, value in metadata.items() if key not in {"dims", "units"}},
    )


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
        **_kwargs(config),
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


def _paper_figure(config, store):
    """Keep transport recipes under the same staged owner as ordinary scans."""
    from dataclasses import replace

    from ..finite_field.figures import compute_figure

    figure = config.parameters["figure"]
    recipe = dict(config.parameters.get("recipe", {}))
    if figure not in {"fig5c", "ext7a", "ext9b"}:
        return compute_figure(figure, **recipe)
    if figure in {"fig5c", "ext7a"}:
        unknown = set(recipe) - {"xs", "amplitudes", "orders", "model_parameters", "options"}
        if unknown:
            raise ValueError(f"Unknown {figure} recipe options: {sorted(unknown)}")
        xs = recipe.get("xs", np.linspace(1.0, 2.0, 101) if figure == "fig5c" else np.linspace(0.0, 2.0, 201))
        amplitudes = recipe.get(
            "amplitudes", np.linspace(2e2, 3.5e4, 141) if figure == "fig5c" else np.linspace(2.5e2, 5e4, 201)
        )
        orders = np.asarray(
            recipe.get("orders", (3, 7, 11, 15, 19, 23, 27, 31) if figure == "fig5c" else (3, 11, 19, 27))
        )
        _numbers(xs, "recipe.xs")
        _numbers(amplitudes, "recipe.amplitudes", positive=True)
        _figure_orders(orders)
        member = replace(
            config,
            workflow="harmonic_map",
            parameters={
                "model_values": list(xs),
                "amplitudes": list(amplitudes),
                "n_max": int(max(orders)),
                "model": recipe.get("model_parameters", {}),
                "options": {"preset": figure, **recipe.get("options", {})},
            },
        )
        result = _harmonic_map(member, None, store)
        arrays = {**result.arrays, "orders": orders}
        for key in ("a", "b", "c"):
            arrays[key] = arrays[key][..., orders]
        return ObservableResult(
            arrays,
            result.dims,
            result.units,
            {
                **result.metadata,
                "observable": "paper_figure",
                "figure": figure,
                "recipe_version": 1,
                "interpretation": "Explicit paper reproduction conventions",
            },
        )
    unknown = set(recipe) - {"scaled_amplitudes", "orders", "model_parameters", "options"}
    if unknown:
        raise ValueError(f"Unknown ext9b recipe options: {sorted(unknown)}")
    scaled = np.asarray(recipe.get("scaled_amplitudes", np.geomspace(1e-5, 1e3, 33)), float)
    orders = np.asarray(recipe.get("orders", tuple(range(3, 22, 2))))
    _numbers(scaled, "recipe.scaled_amplitudes", positive=True)
    _figure_orders(orders)
    member = replace(
        config,
        workflow="hall_harmonics",
        parameters={
            "model": {**recipe.get("model_parameters", {}), "x": 1.0},
            "options": {
                "preset": figure,
                "drive.E_max": float(max(orders) * max(scaled)),
                "drive.E_min": float(min(1e-5, min(scaled))),
                **recipe.get("options", {}),
            },
        },
    )
    prepared = None
    results: list[ObservableResult] = []
    for order in orders:
        case = f"orders/{int(order):06d}/Fourier"
        if store.completed(case):
            arrays, metadata = _read(store, case)
            result = _restore_result(arrays, metadata)
        else:
            if prepared is None:
                prepared = _prepare(member, load_finite_field_source(member), store)
            result = prepared.harmonics(scaled * order, int(order))
            _commit_result(store, case, result)
        results.append(result)
    return ObservableResult(
        {
            "scaled_amplitudes": scaled,
            "orders": orders,
            "scaled_sine": np.stack([n * result["b"][:, n] for n, result in zip(orders, results)]),
            "converged": np.stack([result["converged"] for result in results]),
        },
        {
            "scaled_amplitudes": ("amplitude",),
            "orders": ("harmonic",),
            "scaled_sine": ("harmonic", "amplitude"),
            "converged": ("harmonic", "amplitude"),
        },
        {"scaled_amplitudes": "V/m", "scaled_sine": "A/m", "orders": "1", "converged": "1"},
        {
            **results[0].metadata,
            "observable": "paper_figure",
            "figure": figure,
            "recipe_version": 1,
            "interpretation": "Explicit paper reproduction conventions",
            "members": [result.metadata for result in results],
        },
    )


def _figure_orders(orders):
    if orders.ndim != 1 or not orders.size or orders.dtype.kind not in "iu" or np.any(orders < 1):
        raise ValueError("parameters.recipe.orders must contain positive integers")


def _harmonic_map(config, source, store):
    params = config.parameters
    if "family_manifest" in config.inputs:
        from ..finite_field.api import resolve_options

        options = resolve_options(_options(params))
        coordinates, labels = np.asarray(source.D), source.labels
        count = len(source.members)
    else:
        if config.inputs.get("model") != "fourband":
            raise ValueError("model_values scans currently require an analytical fourband model")
        coordinates = np.asarray(params["model_values"], float)
        parameter = params.get("model_parameter", "x")
        labels = [str(value) for value in coordinates]
        count = len(coordinates)
    results = []
    for index in range(count):
        case = f"members/{index:06d}/Fourier"
        if store.completed(case):
            arrays, metadata = _read(store, case)
            result = _restore_result(arrays, metadata)
        else:
            item = (
                source.source(index, angle=options.drive.angle)
                if "family_manifest" in config.inputs
                else _analytical_member(config, parameter, coordinates[index])
            )
            model = (
                item()
                if callable(item) and not hasattr(item, "hamiltonian") and not hasattr(item, "geometry")
                else item
            )
            prepared = _prepare(config, model, store, namespace=f"members/{index:06d}")
            result = prepared.harmonics(params["amplitudes"], n_max=params.get("n_max"))
            _commit_result(store, case, result)
        results.append(result)
    first = results[0]
    arrays = {
        key: np.stack([result[key] for result in results])
        if key not in {"amplitudes", "harmonics", "orders"}
        else value
        for key, value in first.arrays.items()
    }
    arrays["parameter"] = coordinates
    arrays["labels"] = np.asarray(labels)
    dims = {
        key: (
            ("member", *first.dims.get(key, tuple(f"axis{i}" for i in range(value.ndim))))
            if arrays[key].ndim == value.ndim + 1
            else first.dims.get(key, ())
        )
        for key, value in first.arrays.items()
    }
    dims.update(parameter=("member",), labels=("member",))
    units = {**first.units, "parameter": "1", "labels": "1"}
    normalization = params.get("normalize_by_Dc", False)
    critical = None
    if normalization is not False:
        if "family_manifest" not in config.inputs and normalization is True:
            raise ValueError("normalize_by_Dc=true requires a family; supply a numerical critical field")
        critical = source.critical_D() if normalization is True else float(normalization)
        if critical is None or not np.isfinite(critical) or critical == 0:
            raise ValueError("A unique finite nonzero critical field is required for D/Dc normalization")
        arrays["parameter"] = coordinates / critical
    return ObservableResult(
        arrays,
        dims,
        units,
        {
            **first.metadata,
            "observable": "harmonic_map",
            "labels": list(labels),
            "normalization": normalization,
            "critical_field": critical,
            "critical_field_source": "family_tracking"
            if normalization is True
            else "explicit"
            if normalization is not False
            else None,
            "member_count": len(results),
            "members": [result.metadata for result in results],
        },
    )


def _analytical_member(config, parameter, value):
    from dataclasses import replace

    model = dict(config.parameters.get("model", {}))
    if parameter != "x":
        model.pop("x", None)
    model[parameter] = float(value)
    member = replace(config, parameters={**config.parameters, "model": model})
    return load_finite_field_source(member)


def _crossing_map(config, source, store):
    from ..finite_field import crossing_scan, resolve_options

    params = config.parameters
    family = "family_manifest" in config.inputs
    coordinates = np.asarray(source.D if family else params["model_values"], float)
    labels = source.labels if family else [str(value) for value in coordinates]
    angle = resolve_options(_options(params)).drive.angle
    rows = []
    for index, value in enumerate(coordinates):
        case = f"members/{index:06d}/crossings"
        if store.completed(case):
            arrays, metadata = _read(store, case)
            row = _restore_result(arrays, metadata)
        else:
            item = (
                source.source(index, angle=angle)
                if family
                else _analytical_member(config, params.get("model_parameter", "x"), value)
            )
            row = crossing_scan(item, params.get("offsets", [0.0]), **_kwargs(config))
            _commit_result(store, case, row)
        rows.append(row)
    arrays = {key: np.concatenate([row[key] for row in rows]) for key in rows[0].arrays}
    line_base = np.cumsum([0, *[len(row["offsets"]) for row in rows]])
    for key in ("line_index", "segment_line_index"):
        arrays[key] = np.concatenate([row[key] + line_base[index] for index, row in enumerate(rows)])
    arrays["parameter"] = coordinates
    arrays["labels"] = np.asarray(labels)
    arrays["member_index"] = np.concatenate(
        [np.full(len(row["position"]), index, int) for index, row in enumerate(rows)]
    )
    arrays["line_member_index"] = np.concatenate(
        [np.full(len(row["offsets"]), index, int) for index, row in enumerate(rows)]
    )
    arrays["segment_member_index"] = np.concatenate(
        [np.full(len(row["segment_start"]), index, int) for index, row in enumerate(rows)]
    )
    dims = {**rows[0].dims, "parameter": ("member",), "labels": ("member",), "member_index": ("crossing",)}
    units = {**rows[0].units, "parameter": "D" if family else "1", "labels": "1", "member_index": "index"}
    dims.update(line_member_index=("line",), segment_member_index=("segment",))
    units.update(line_member_index="index", segment_member_index="index")
    return ObservableResult(
        arrays, dims, units, {"observable": "crossing_map", "members": [row.metadata for row in rows]}
    )


def split_harmonic_map(config):
    """Resolve isolated child workflows without retaining loaded family models."""
    from ..config import config_from_mapping

    if config.workflow != "harmonic_map":
        raise ValueError("--split-members requires a harmonic_map configuration")
    params = dict(config.parameters)
    params.pop("model_values", None)
    parameter = params.pop("model_parameter", "x")
    normalization = params.pop("normalize_by_Dc", False)
    family = load_finite_field_source(config) if "family_manifest" in config.inputs else None
    coordinates = np.asarray(family.D if family is not None else config.parameters["model_values"], float)
    labels = family.labels if family is not None else [str(value) for value in coordinates]
    critical = (
        family.critical_D()
        if normalization is True and family is not None
        else float(normalization)
        if normalization is not False
        else None
    )
    if normalization is True and family is None:
        raise ValueError("normalize_by_Dc=true requires validated family tracking")
    if critical is not None and (not np.isfinite(critical) or critical == 0):
        raise ValueError("Critical-field normalization requires a finite nonzero value")
    children = []
    for index, value in enumerate(coordinates):
        child = config.as_dict()
        child["workflow"] = "hall_harmonics"
        child_params = dict(params)
        if family is not None:
            member = family.members[index]
            if "grid" in member.files:
                child_inputs = {"geometry_grid": member.files["grid"]}
            else:
                loader = member.model_loader or member.loader
                child_inputs = {
                    "model": "openmx" if "scfout" in member.files else "wannier",
                    "material_dir": str(Path(next(iter(member.files.values()))).parent),
                    **member.files,
                }
                child_params["plane"] = {
                    key: getattr(loader.plane, key) for key in ("origin", "direction", "transverse")
                }
                if loader.band_pair is not None:
                    child_params["band_pair"] = list(loader.band_pair)
                elif loader.n_occ is not None:
                    child_params["occupied_bands"] = loader.n_occ
                if getattr(loader, "lattice", None) is not None:
                    child_params["lattice"] = np.asarray(loader.lattice).tolist()
        else:
            child_inputs = dict(config.inputs)
            model = dict(params.get("model", {}))
            if parameter != "x":
                model.pop("x", None)
            model[parameter] = float(value)
            child_params["model"] = model
        child["inputs"], child["parameters"] = child_inputs, child_params
        children.append(config_from_mapping(child))
    return children, {
        "config": config.as_dict(),
        "parameters": coordinates.tolist(),
        "labels": list(labels),
        "normalization": normalization,
        "critical_field": critical,
    }


def aggregate_member_runs(dispatch_path, *, run_dir=None):
    """Aggregate completed isolated members after checking inputs, schemas, and checksums."""
    from ..config import config_from_mapping
    from ..io.artifacts import canonical_json, digest_file, load_manifest

    path = Path(dispatch_path).expanduser().resolve()
    dispatch = json.loads(path.read_text())
    if "aggregation" not in dispatch:
        raise ValueError("Dispatch has no split-member harmonic-map aggregation contract")
    specification = dispatch["aggregation"]
    config = config_from_mapping(specification["config"])
    coordinates, labels = specification["parameters"], specification["labels"]
    if len(dispatch["cases"]) != len(coordinates) or len(labels) != len(coordinates):
        raise ValueError("Member dispatch count does not match the parent map")
    files = [path, *finite_field_input_files(config)]
    results: list[ObservableResult] = []
    for index, case in enumerate(dispatch["cases"]):
        if digest_file(Path(case["config"])) != case["config_sha256"]:
            raise ValueError("Submitted member configuration changed")
        for item in case.get("inputs", []):
            if digest_file(Path(item["path"])) != item["sha256"]:
                raise ValueError("Submitted scientific input changed")
        root = Path(dispatch["run_root"]) / f"task-{index:04d}"
        manifest = load_manifest(root)
        expected = case["resolved_config"]
        actual = manifest["config"]
        if (
            actual["workflow"] != "hall_harmonics"
            or canonical_json(actual["parameters"]) != canonical_json(expected["parameters"])
            or canonical_json(actual["inputs"]) != canonical_json(expected["inputs"])
        ):
            raise ValueError("Member run scientific configuration differs from dispatch")
        record = manifest["cases"]["result"]
        product = root / record["path"]
        files.extend([root / "manifest.json", product])
        with np.load(product, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        metadata = record["metadata"]
        result = _restore_result(arrays, metadata)
        if results and (
            set(result.arrays) != set(results[0].arrays)
            or result.dims != results[0].dims
            or result.units != results[0].units
            or metadata.get("source_sha256") != results[0].metadata.get("source_sha256")
            or any(not np.array_equal(result[key], results[0][key]) for key in ("amplitudes", "orders"))
        ):
            raise ValueError("Member harmonic schemas, grids, units, or engine revisions differ")
        results.append(result)
    first = results[0]
    arrays = {
        key: value if key in {"amplitudes", "orders"} else np.stack([row[key] for row in results])
        for key, value in first.arrays.items()
    }
    critical = specification.get("critical_field")
    arrays["parameter"] = (
        np.asarray(coordinates, float) / critical if critical is not None else np.asarray(coordinates, float)
    )
    arrays["labels"] = np.asarray(labels)
    dims = {
        key: first.dims[key] if key in {"amplitudes", "orders"} else ("member", *first.dims[key])
        for key in first.arrays
    }
    dims.update(parameter=("member",), labels=("member",))
    result = ObservableResult(
        arrays,
        dims,
        {**first.units, "parameter": "1", "labels": "1"},
        {
            **first.metadata,
            "observable": "harmonic_map",
            "aggregation": "isolated checked member runs",
            "source_sha256": first.metadata.get("source_sha256"),
            "members": [row.metadata for row in results],
            "critical_field": critical,
            "critical_field_source": "family_tracking"
            if specification["normalization"] is True
            else "explicit"
            if critical is not None
            else None,
            "normalization": specification["normalization"],
        },
    )
    target = Path(run_dir).expanduser().resolve() if run_dir else Path(dispatch["run_root"]) / "map"
    store = RunStore(target, config.as_dict(), files, resume=(target / "manifest.json").exists())
    try:
        if not store.completed("result"):
            _commit_result(store, "result", result)
        store.finish()
    except BaseException as error:
        store.finish(status="failed", error=error)
        raise
    finally:
        store.release()
    return target
