"""Finite-field configuration schemas, independent of workflow execution."""

from __future__ import annotations

from typing import Mapping

import numpy as np

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


def _options(params):
    options = params.get("options", {})
    if isinstance(options, str):
        options = {"preset": options}
    elif not isinstance(options, Mapping):
        raise ValueError("parameters.options must be a table or preset name")
    return {
        "preset": params.get("preset", "paper"),
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
