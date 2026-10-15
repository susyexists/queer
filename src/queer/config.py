"""Validated configuration shared by installed commands and batch workflows."""

from __future__ import annotations

import math
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .execution import ExecutionOptions
from .workflows.catalog import NAMES as OBSERVABLE_WORKFLOWS
from .workflows.catalog import WORKFLOWS
from .workflows.finite_field_specs import FINITE_FIELD_INPUTS, FINITE_FIELD_NAMES, validate_finite_field
from .workflows.service_specs import SERVICE_INPUTS, SERVICE_NAMES, validate_service

_PARAMETERS = {
    "bands": {"ef", "interpolation", "position", "kpath", "n_points", "kpoints", "bands"},
    "susceptibility": {
        "ef",
        "interpolation",
        "position",
        "q_path",
        "k_mesh",
        "band_indices",
        "temperature",
        "broadening",
    },
    "spectral": {
        "ef",
        "interpolation",
        "position",
        "kpoints",
        "omega",
        "eta",
        "photon_energy",
        "v0_values",
        "align_zero",
    },
    "momentum": {
        "ef",
        "interpolation",
        "position",
        "photon_energy",
        "fermi_energy",
        "binding_min",
        "binding_max",
        "binding_step",
        "v0",
        "v0_values",
        "n_points",
        "factor",
        "align",
        "energy_shift",
        "sigma",
    },
    "epw": {"ef", "nk", "nq", "nph", "q_indices", "temperature", "broadening"},
}
_REQUIRED = {
    "bands": {"ef", "interpolation"},
    "susceptibility": {"ef", "interpolation", "q_path", "k_mesh", "temperature", "broadening"},
    "spectral": {"ef", "interpolation", "kpoints", "omega", "eta"},
    "momentum": {
        "ef",
        "interpolation",
        "photon_energy",
        "fermi_energy",
        "binding_min",
        "binding_max",
        "binding_step",
        "n_points",
        "sigma",
    },
    "epw": {"q_indices", "temperature", "broadening"},
}
# Sidecar files and modes a Wannier model directory may name explicitly.
WANNIER_INPUTS = frozenset({"win", "tb", "centres", "wout"})
INTERPOLATION_MODES = ("standard", "ws_distance", "ndegen_applied", "auto")
POSITION_MODES = ("auto", "full", "diagonal")


def _finite(value: Any, key: str, *, positive: bool = False, nonnegative: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"parameters.{key} must be a finite number")
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(f"parameters.{key} must be {'positive' if positive else 'nonnegative'}")


def _points(value: Any, key: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"parameters.{key} must contain at least one three-dimensional point")
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(f"parameters.{key} must have shape (N, 3)")
        for coordinate in row:
            _finite(coordinate, key)


@dataclass(frozen=True)
class WorkflowConfig:
    workflow: str
    inputs: Mapping[str, Any]
    parameters: Mapping[str, Any]
    execution: ExecutionOptions = field(default_factory=ExecutionOptions)
    results_dir: Path = field(default_factory=lambda: Path.cwd() / "results")

    @property
    def material_dir(self) -> Path:
        return Path(self.inputs["material_dir"])

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        parameters = dict(self.parameters)
        if self.workflow in FINITE_FIELD_NAMES:
            from .finite_field._storage import json_safe

            parameters = json_safe(parameters)

        return {
            "workflow": self.workflow,
            "inputs": {
                key: str(value) if isinstance(value, Path) else value for key, value in self.inputs.items()
            },
            "parameters": parameters,
            "execution": asdict(self.execution),
            "output": {"results_dir": str(self.results_dir)},
        }


def config_from_mapping(
    data: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
    workflow: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> WorkflowConfig:
    """Resolve CLI overrides, configuration, environment roots, then cwd defaults."""
    base = Path(base_dir or Path.cwd()).resolve()
    unknown = set(data) - {"workflow", "inputs", "parameters", "execution", "output"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {', '.join(sorted(unknown))}")
    configured_workflow = data.get("workflow")
    if workflow and configured_workflow and workflow != configured_workflow:
        raise ValueError(
            f"Command {workflow!r} does not match configuration workflow {configured_workflow!r}"
        )
    name = workflow or configured_workflow
    if name not in WORKFLOWS:
        raise ValueError(f"workflow must be one of {', '.join(WORKFLOWS)}")
    sections = {}
    for section in ("inputs", "parameters", "execution", "output"):
        value = data.get(section, {})
        if not isinstance(value, Mapping):
            raise ValueError(f"{section} must be a table")
        sections[section] = dict(value)
    inputs, params, execution, output = (sections[s] for s in ("inputs", "parameters", "execution", "output"))
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key == "material_dir":
            inputs.pop("material", None)
            inputs[key] = Path(value).expanduser().resolve()
        elif key == "results_dir":
            output[key] = Path(value).expanduser().resolve()
        elif key == "v0":
            params.pop("v0_values", None)
            params["v0"] = value
        elif key in {"workers", "native_threads", "batch_size", "memory_mb"}:
            execution[key] = value
        else:
            raise ValueError(f"Unknown configuration override: {key}")
    unknown_inputs = set(inputs) - (
        {
            "material",
            "material_dir",
            "hr",
            "poscar",
            "wsvec",
            "model",
            "scfout",
            "openmx_input",
            "pao_dir",
            "pao_files",
        }
        | (SERVICE_INPUTS if name in SERVICE_NAMES else {"export_metadata"})
        | (FINITE_FIELD_INPUTS if name in FINITE_FIELD_NAMES else set())
        | (WANNIER_INPUTS if name in OBSERVABLE_WORKFLOWS or name in _PARAMETERS.keys() - {"epw"} else set())
    )
    if unknown_inputs or set(output) - {"results_dir"}:
        raise ValueError(
            f"Unknown input/output options: {sorted(unknown_inputs | (set(output) - {'results_dir'}))}"
        )
    if "material" in inputs and "material_dir" in inputs:
        raise ValueError("Specify inputs.material or inputs.material_dir, not both")
    if "material" in inputs:
        material = inputs.pop("material")
        if (
            not isinstance(material, str)
            or not material
            or Path(material).name != material
            or material in {".", ".."}
        ):
            raise ValueError("inputs.material must be a single material directory name")
        data_root = Path(os.environ.get("QUEER_DATA_DIR", str(Path.cwd() / "data"))).expanduser().resolve()
        inputs["material_dir"] = data_root / "materials" / material
    if name in FINITE_FIELD_NAMES:
        for key in ("material_dir", "geometry_grid", "family_manifest", "response_artifact"):
            if key in inputs:
                if not isinstance(inputs[key], (str, Path)) or not str(inputs[key]):
                    raise ValueError(f"inputs.{key} must be a nonempty path")
                source = Path(inputs[key]).expanduser()
                inputs[key] = (source if source.is_absolute() else base / source).resolve()
        if "cache_dir" in params:
            if not isinstance(params["cache_dir"], str) or not params["cache_dir"]:
                raise ValueError("parameters.cache_dir must be a nonempty directory path")
            cache = Path(params["cache_dir"]).expanduser()
            params["cache_dir"] = str((cache if cache.is_absolute() else base / cache).resolve())
        validate_finite_field(name, inputs, params)
        return _resolved_config(name, inputs, params, execution, output, base)
    if not inputs.get("material_dir"):
        raise ValueError("inputs.material_dir or inputs.material is required")
    if not isinstance(inputs["material_dir"], (str, Path)):
        raise ValueError("inputs.material_dir must be a directory path")
    directory = Path(inputs["material_dir"]).expanduser()
    inputs["material_dir"] = (directory if directory.is_absolute() else base / directory).resolve()
    if name in SERVICE_NAMES:
        validate_service(name, inputs, params)
        return _resolved_config(name, inputs, params, execution, output, base)
    model_kind = inputs.get("model", "wannier")
    if model_kind not in {"wannier", "openmx"}:
        raise ValueError("inputs.model must be wannier or openmx")
    if name == "epw" and model_kind == "openmx":
        raise ValueError("EPW workflows require EPW inputs, not OpenMX")
    for key in ("hr", "poscar", "wsvec", "scfout", "openmx_input", "pao_dir", *sorted(WANNIER_INPUTS)):
        if key in inputs and (not isinstance(inputs[key], str) or not inputs[key]):
            raise ValueError(f"inputs.{key} must be a nonempty filename")
    if model_kind == "openmx" and WANNIER_INPUTS & set(inputs):
        names = ", ".join(sorted(WANNIER_INPUTS & set(inputs)))
        raise ValueError(f"OpenMX models do not read Wannier inputs: {names}")
    if "pao_files" in inputs and (
        not isinstance(inputs["pao_files"], dict)
        or not inputs["pao_files"]
        or any(
            not isinstance(key, str) or not isinstance(value, str) or not value
            for key, value in inputs["pao_files"].items()
        )
    ):
        raise ValueError("inputs.pao_files must map PAO keys to nonempty filenames")
    if name in OBSERVABLE_WORKFLOWS:
        from .workflows.catalog import validate_parameters

        validate_parameters(name, params)
    else:
        allowed = _PARAMETERS[name] | ({"scissors", "n_occ", "spin"} if name == "bands" else set())
        unknown_params = set(params) - allowed
        if unknown_params:
            raise ValueError(f"Unknown {name} parameters: {', '.join(sorted(unknown_params))}")
        required = _REQUIRED[name] - ({"ef", "interpolation"} if model_kind == "openmx" else set())
        missing = required - set(params)
        if missing:
            raise ValueError(f"Missing {name} parameters: {', '.join(sorted(missing))}")
    for key in ("ef", "photon_energy", "fermi_energy", "binding_min", "binding_max", "v0", "energy_shift"):
        if key in params:
            _finite(params[key], key)
    for key in ("eta", "broadening", "binding_step", "sigma", "factor"):
        if key in params:
            _finite(params[key], key, positive=True)
    if "temperature" in params:
        _finite(params["temperature"], "temperature", nonnegative=True)
    if "temperature_kelvin" in params:
        values = params["temperature_kelvin"]
        if name == "boltz_transport" and isinstance(values, list):
            if not values:
                raise ValueError("parameters.temperature_kelvin must be a nonempty scan")
            for value in values:
                _finite(value, "temperature_kelvin", positive=True)
        else:
            _finite(values, "temperature_kelvin", positive=name == "boltz_transport", nonnegative=True)
    for key in ("n_points", "nk", "nq", "nph"):
        if key in params and (
            isinstance(params[key], bool) or not isinstance(params[key], int) or params[key] < 1
        ):
            raise ValueError(f"parameters.{key} must be a positive integer")
    for key in ("kpoints", "k_mesh", "q_path"):
        if key in params:
            _points(params[key], key)
    for key in ("omega", "v0_values"):
        if key in params:
            if not isinstance(params[key], list) or not params[key]:
                raise ValueError(f"parameters.{key} must be a nonempty list")
            for item in params[key]:
                _finite(item, key)
    for key in ("bands", "band_indices", "q_indices"):
        if key in params:
            values = params[key]
            if (
                not isinstance(values, list)
                or not values
                or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in values)
            ):
                raise ValueError(f"parameters.{key} must be a nonempty list of nonnegative integer indices")
    if "interpolation" in params and params["interpolation"] not in INTERPOLATION_MODES:
        raise ValueError("parameters.interpolation must be standard, ws_distance, ndegen_applied or auto")
    if model_kind == "openmx" and "interpolation" in params:
        raise ValueError("OpenMX matrices do not use Wannier interpolation")
    if "position" in params and params["position"] not in POSITION_MODES:
        raise ValueError("parameters.position must be auto, full or diagonal")
    if model_kind == "openmx" and "position" in params:
        raise ValueError("OpenMX matrices do not use Wannier position modes")
    if name == "bands":
        if "spin" in params and (model_kind != "openmx" or params["spin"] not in {"up", "down"}):
            raise ValueError("parameters.spin must be up/down for a collinear OpenMX model")
        if ("kpoints" in params) == ("kpath" in params):
            raise ValueError("bands requires exactly one of parameters.kpoints or parameters.kpath")
        if "kpath" in params and (
            not isinstance(params["kpath"], str)
            or not params["kpath"]
            or ("poscar" not in inputs and model_kind != "openmx")
        ):
            raise ValueError("Named bands kpath requires a nonempty route and inputs.poscar")
        if "kpath" in params and "n_points" not in params:
            raise ValueError("Named bands kpath requires parameters.n_points")
    if name == "spectral" and set(params) & {"photon_energy", "v0_values", "align_zero"}:
        if not {"photon_energy", "v0_values"}.issubset(params) or (
            "poscar" not in inputs and model_kind != "openmx"
        ):
            raise ValueError("Curved ARPES spectral runs require photon_energy, v0_values and inputs.poscar")
        if "align_zero" in params and not isinstance(params["align_zero"], bool):
            raise ValueError("parameters.align_zero must be a boolean")
    if name == "momentum":
        if ("v0" in params) == ("v0_values" in params):
            raise ValueError("momentum requires exactly one of parameters.v0 or parameters.v0_values")
        if params["binding_max"] <= params["binding_min"]:
            raise ValueError("binding_max must be greater than binding_min for a half-open interval")
        if "poscar" not in inputs and model_kind != "openmx":
            raise ValueError("momentum requires inputs.poscar")
        if "align" in params:
            _points(params["align"], "align")
            if len(params["align"]) != 2 or any(not any(row) for row in params["align"]):
                raise ValueError("parameters.align must contain two nonzero vectors")
    return _resolved_config(name, inputs, params, execution, output, base)


def _resolved_config(name, inputs, params, execution, output, base):
    try:
        resources = ExecutionOptions(**execution)
    except TypeError as exc:
        raise ValueError(f"Invalid execution options: {exc}") from exc
    results_value = output.get(
        "results_dir", os.environ.get("QUEER_RESULTS_DIR", str(Path.cwd() / "results"))
    )
    if not isinstance(results_value, (str, Path)) or not str(results_value):
        raise ValueError("output.results_dir must be a directory path")
    results = Path(results_value).expanduser()
    output_base = base if "results_dir" in output else Path.cwd()
    results = (results if results.is_absolute() else output_base / results).resolve()
    return WorkflowConfig(name, inputs, params, resources, results)


def load_config(
    path: str | Path,
    *,
    workflow: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    base_dir: Path | None = None,
) -> WorkflowConfig:
    source = Path(path).expanduser().resolve()
    with source.open("rb") as stream:
        return config_from_mapping(
            tomllib.load(stream), base_dir=base_dir or source.parent, workflow=workflow, overrides=overrides
        )
