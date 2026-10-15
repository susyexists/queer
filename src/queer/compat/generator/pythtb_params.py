"""Parameters of the Generator's PythTB steps, resolved for QUEER's Wannier runners.

The Generator writes one parameter file per step: its catalog defaults, overlaid
by the preset, the step's own fields and the context its submitter resolved
from the source WANNIER job. :func:`translate_pythtb` reads such a file for the
QUEER counterpart of the step. It keeps the shared model fields, the step's own
fields and the submitter context with the driver's defaults; holds catalog
fields to the Generator's kinds and bounds; and refuses every other key with
``queer_bad_params``, so a misspelt or foreign setting fails before any
calculation. A ``None`` value keeps the driver default ("auto").

QUEER step names are ``QUEER_BANDS``, ``QUEER_TOPOLOGY``, ``QUEER_BERRY`` and
``QUEER_SLAB``. The ``PYTHTB_*`` names of the counterparts are accepted as
aliases, so a saved PythTB parameter file replays unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .results import ERR_BAD_PARAMS, RunnerError

STEP_BANDS = "QUEER_BANDS"
STEP_TOPOLOGY = "QUEER_TOPOLOGY"
STEP_BERRY = "QUEER_BERRY"
STEP_SLAB = "QUEER_SLAB"
STEP_TYPES: tuple[str, ...] = (STEP_BANDS, STEP_TOPOLOGY, STEP_BERRY, STEP_SLAB)
COUNTERPARTS: Mapping[str, str] = MappingProxyType(
    {step: step.replace("QUEER_", "PYTHTB_", 1) for step in STEP_TYPES}
)
ALIASES: Mapping[str, str] = MappingProxyType({value: key for key, value in COUNTERPARTS.items()})

# Every parameter the runners read, with the Generator driver's default.
DEFAULT_PARAMS: Mapping[str, Any] = MappingProxyType(
    {
        # Context the submitter resolves from the source WANNIER job.
        "fermi_energy": None,
        "fermi_source": None,
        "ws_hint": None,
        "spinors": None,
        "structure": None,
        "kpath": None,
        "source_wannier_job_id": None,
        "reference_band_job_id": None,
        "wannier_omega_I_A2": None,
        "time_budget_s": None,
        # Model build and validation.
        "ws_correction": "auto",
        "min_hopping_norm": None,
        "max_distance": None,
        "periodic_dirs": None,
        "validation_warn_meV": 5.0,
        "validation_fail_meV": 100.0,
        "fermi_tolerance_eV": 0.02,
        "occupied_bands": None,
        "isolation_threshold_meV": 5.0,
        # Execution.
        "workers": None,
        "chunk_size": None,
        # Plots.
        "energy_window_eV": (-3.0, 3.0),
        # Bands.
        "kpath_points": 400,
        "dos_mesh": None,
        "dos_sigma_eV": 0.03,
        "dos_step_eV": 0.005,
        "pdos": True,
        "fermi_surface": True,
        "fermi_surface_mesh": None,
        # Topology.
        "planes": None,
        "loop_points": 81,
        "transverse_points": 61,
        "max_refinements": 2,
        "z2": "auto",
        "kramers_tol_meV": 1.0,
        "z2_pos_tol": 0.01,
        # Berry curvature.
        "berry_plane": None,
        "berry_mesh": 150,
        "occupation": "auto",
        "quantum_metric": True,
        "path_curvature": True,
        # Slab.
        "slab_axis": None,
        "slab_cells": None,
        "slab_kpoints": 200,
        "slab_window_eV": 0.5,
        "edge_cells": 2,
    }
)

# Keys the submitter resolves from the source WANNIER job; never wizard fields.
CONTEXT_PARAM_KEYS: tuple[str, ...] = (
    "fermi_source",
    "ws_hint",
    "spinors",
    "structure",
    "kpath",
    "source_wannier_job_id",
    "reference_band_job_id",
    "wannier_omega_I_A2",
    "time_budget_s",
)
SHARED_PARAM_KEYS: tuple[str, ...] = (
    "fermi_energy",
    "ws_correction",
    "min_hopping_norm",
    "max_distance",
    "periodic_dirs",
    "validation_warn_meV",
    "validation_fail_meV",
    "workers",
    "chunk_size",
    "energy_window_eV",
)
_OCCUPATION_KEYS = ("occupied_bands", "fermi_tolerance_eV", "isolation_threshold_meV")
STEP_PARAM_KEYS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        STEP_BANDS: (
            "kpath_points",
            "dos_mesh",
            "dos_sigma_eV",
            "dos_step_eV",
            "pdos",
            "fermi_tolerance_eV",
            "fermi_surface",
            "fermi_surface_mesh",
        ),
        STEP_TOPOLOGY: (
            *_OCCUPATION_KEYS,
            "planes",
            "loop_points",
            "transverse_points",
            "max_refinements",
            "z2",
            "kramers_tol_meV",
            "z2_pos_tol",
        ),
        STEP_BERRY: (
            *_OCCUPATION_KEYS,
            "berry_plane",
            "berry_mesh",
            "occupation",
            "quantum_metric",
            "path_curvature",
            "kpath_points",
        ),
        STEP_SLAB: ("slab_axis", "slab_cells", "slab_kpoints", "slab_window_eV", "edge_cells"),
    }
)

# The Generator catalog's kinds and bounds. ``ws_correction`` also accepts
# ``file``, which the driver honours from a hand-written parameter file: the
# wsvec file's own images when one is staged.
MODEL_FIELDS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "fermi_energy": {"kind": "number", "unit": "eV"},
        "ws_correction": {"kind": "select", "options": ("auto", "recompute", "off", "file")},
        "min_hopping_norm": {"kind": "number", "min": 0, "unit": "eV"},
        "max_distance": {"kind": "number", "min_exclusive": 0, "unit": "Angstrom"},
        "validation_warn_meV": {"kind": "number", "min": 0, "unit": "meV"},
        "validation_fail_meV": {"kind": "number", "min": 0, "unit": "meV"},
        "energy_window_eV": {"kind": "float_vector", "shape": 2, "ordered": True, "unit": "eV"},
        "workers": {"kind": "integer", "min": 1},
    }
)
_OCCUPATION_FIELDS: dict[str, Mapping[str, Any]] = {
    "occupied_bands": {"kind": "integer", "min": 1},
    "fermi_tolerance_eV": {"kind": "number", "min": 0, "unit": "eV"},
    "isolation_threshold_meV": {"kind": "number", "min": 0, "unit": "meV"},
}
STEP_FIELDS: Mapping[str, Mapping[str, Mapping[str, Any]]] = MappingProxyType(
    {
        STEP_BANDS: {
            "kpath_points": {"kind": "integer", "min": 2},
            "dos_mesh": {"kind": "integer", "min": 2},
            "dos_sigma_eV": {"kind": "number", "min_exclusive": 0, "unit": "eV"},
            "dos_step_eV": {"kind": "number", "min_exclusive": 0, "unit": "eV"},
            "pdos": {"kind": "boolean"},
            "fermi_tolerance_eV": {"kind": "number", "min": 0, "unit": "eV"},
            "fermi_surface": {"kind": "boolean"},
            "fermi_surface_mesh": {"kind": "integer", "min": 2},
        },
        STEP_TOPOLOGY: {
            **_OCCUPATION_FIELDS,
            "loop_points": {"kind": "integer", "min": 3},
            "transverse_points": {"kind": "integer", "min": 2},
            "max_refinements": {"kind": "integer", "min": 0},
            "z2": {"kind": "select", "options": ("auto", "on", "off")},
            "kramers_tol_meV": {"kind": "number", "min": 0, "unit": "meV"},
            "z2_pos_tol": {"kind": "number", "min": 0},
        },
        STEP_BERRY: {
            **_OCCUPATION_FIELDS,
            "berry_mesh": {"kind": "integer", "min": 2},
            "occupation": {"kind": "select", "options": ("auto", "bands", "fermi")},
            "quantum_metric": {"kind": "boolean"},
            "path_curvature": {"kind": "boolean"},
            "kpath_points": {"kind": "integer", "min": 2},
        },
        STEP_SLAB: {
            "slab_axis": {"kind": "select", "options": (None, 0, 1, 2)},
            "slab_cells": {"kind": "integer", "min": 2},
            "slab_kpoints": {"kind": "integer", "min": 2},
            "slab_window_eV": {"kind": "number", "min_exclusive": 0, "unit": "eV"},
            "edge_cells": {"kind": "integer", "min": 1},
        },
    }
)


@dataclass(frozen=True)
class WannierStepParameters:
    """Resolved settings of one QUEER Wannier step.

    ``values`` holds every driver default overlaid with the validated non-null
    settings of the file, including the submitter context; ``step`` is the QUEER
    step and ``counterpart`` the Generator PythTB step it reproduces.
    """

    step: str
    counterpart: str
    values: Mapping[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        """The setting, or ``default`` when it is None."""
        value = self.values.get(key)
        return default if value is None else value

    def driver_value(self, key: str, default: Any) -> Any:
        """The setting as the driver reads ``params.get(key) or default``: zero and empty fall back too."""
        return self.values.get(key) or default

    def __getitem__(self, key: str) -> Any:
        return self.values[key]


def canonical_step(name: Any) -> str:
    """The QUEER step for a QUEER or PythTB step name; ``queer_bad_params`` for any other."""
    token = str(name or "").strip().upper()
    step = ALIASES.get(token, token)
    if step not in STEP_TYPES:
        allowed = ", ".join(STEP_TYPES)
        raise RunnerError(
            ERR_BAD_PARAMS, f"unknown step_type {name!r}: this build runs {allowed} (or their PYTHTB_ names)"
        )
    return step


def step_keys(step: str) -> tuple[str, ...]:
    """The parameter keys a step accepts: shared, its own and the submitter context."""
    return SHARED_PARAM_KEYS + STEP_PARAM_KEYS[canonical_step(step)] + CONTEXT_PARAM_KEYS


def _number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("a boolean is not a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("not a finite number")
    return number


def _coerce(field: Mapping[str, Any], value: Any) -> Any:
    """``value`` as its catalog field's kind, within its bounds; ValueError says why not."""
    kind = field.get("kind")
    if kind == "boolean":
        if isinstance(value, bool):
            return value
        token = str(value).strip().lower()
        if token in ("true", "1", "yes", "on"):
            return True
        if token in ("false", "0", "no", "off"):
            return False
        raise ValueError("expected true or false")
    if kind == "select":
        token = str(value).strip().lower()
        for option in field.get("options") or ():
            if option is not None and str(option).lower() == token:
                return option
        allowed = [str(option) for option in field.get("options") or () if option is not None]
        raise ValueError(f"expected one of {', '.join(allowed)}")
    if kind == "float_vector":
        shape = int(field.get("shape") or 2)
        parts = value.replace(",", " ").split() if isinstance(value, str) else value
        if not isinstance(parts, (list, tuple)) or len(parts) != shape:
            raise ValueError(f"expected {shape} numbers")
        vector = [_number(part) for part in parts]
        if field.get("ordered") and any(b <= a for a, b in zip(vector, vector[1:])):
            raise ValueError("expected increasing numbers")
        if "min" in field and any(x < field["min"] for x in vector):
            raise ValueError(f"expected every value at least {field['min']}")
        if "max" in field and any(x > field["max"] for x in vector):
            raise ValueError(f"expected every value at most {field['max']}")
        return vector
    number = _number(value)
    if kind == "integer":
        if number != int(number):
            raise ValueError("expected a whole number")
        number = int(number)
    if "min" in field and number < field["min"]:
        raise ValueError(f"expected at least {field['min']}")
    if "min_exclusive" in field and number <= field["min_exclusive"]:
        raise ValueError(f"expected more than {field['min_exclusive']}")
    if "max" in field and number > field["max"]:
        raise ValueError(f"expected at most {field['max']}")
    return number


def _axis(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in (0, 1, 2):
        raise ValueError("expected a lattice axis 0, 1 or 2")
    return int(value)


def _plane(entry: Any) -> dict[str, Any]:
    if isinstance(entry, Mapping):
        unknown = set(entry) - {"fixed_axis", "value"}
        if unknown:
            raise ValueError(f"unknown plane keys {sorted(unknown)}")
        axis, value = entry.get("fixed_axis", 2), entry.get("value", 0.0)
    elif isinstance(entry, (list, tuple)) and len(entry) == 2:
        axis, value = entry
    else:
        raise ValueError("expected {fixed_axis, value} or [fixed_axis, value]")
    return {"fixed_axis": _axis(axis), "value": _number(value)}


def _segments(kpath: Any) -> Any:
    if not isinstance(kpath, Mapping):
        raise ValueError("expected an object with 'segments'")
    segments = kpath.get("segments")
    if segments is None:
        return dict(kpath)
    if not isinstance(segments, (list, tuple)):
        raise ValueError("segments must be a list")
    checked = []
    for row in segments:
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise ValueError("each segment is [label, k_start, label, k_end]")
        start, end = row[1], row[3]
        for point in (start, end):
            if not isinstance(point, (list, tuple)) or len(point) != 3:
                raise ValueError("segment k-points need three fractional coordinates")
        checked.append([str(row[0]), [_number(x) for x in start], str(row[2]), [_number(x) for x in end]])
    return {**dict(kpath), "segments": checked}


def _special(key: str, value: Any) -> Any:
    """API-only and context values: shapes the runners rely on."""
    if key in ("fermi_source", "source_wannier_job_id", "reference_band_job_id"):
        if not isinstance(value, str):
            raise ValueError("expected a string")
        return value
    if key in ("ws_hint", "spinors"):
        return _coerce({"kind": "boolean"}, value)
    if key in ("wannier_omega_I_A2", "time_budget_s"):
        return _number(value)
    if key == "structure":
        if not isinstance(value, Mapping):
            raise ValueError("expected an object with species and Cartesian positions")
        return dict(value)
    if key == "kpath":
        return _segments(value)
    if key == "periodic_dirs":
        if not isinstance(value, (list, tuple)):
            raise ValueError("expected a list of lattice axes")
        if not value:
            # The driver reads an empty list as unset: periodicity from the table.
            return None
        axes = sorted({_axis(axis) for axis in value})
        if len(axes) != len(value):
            raise ValueError("axes must be distinct")
        return axes
    if key == "chunk_size":
        return _coerce({"kind": "integer", "min": 1}, value)
    if key == "planes":
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("expected a nonempty list of planes")
        return [_plane(entry) for entry in value]
    if key == "berry_plane":
        if not isinstance(value, Mapping):
            raise ValueError("expected {fixed_axis, value}")
        return _plane(value)
    raise ValueError("unsupported parameter")


def translate_pythtb(params: Mapping[str, Any], step: str | None = None) -> WannierStepParameters:
    """Resolve a Generator PythTB parameter file for the QUEER counterpart step.

    ``step`` (a QUEER or PythTB step name) and the file's ``step_type`` must name
    the same step when both are given. The result holds every driver default
    overlaid with the file's validated non-null values. Raises
    :class:`~queer.compat.generator.results.RunnerError` with code
    ``queer_bad_params`` naming every unknown key and every value outside the
    Generator catalog's kind and bounds; parity references and other bookkeeping
    are not runner parameters.
    """
    if not isinstance(params, Mapping):
        raise RunnerError(ERR_BAD_PARAMS, "the parameter file must hold a JSON object")
    declared = params.get("step_type")
    if step is None and declared is None:
        raise RunnerError(ERR_BAD_PARAMS, "no step: pass --step or set step_type in the parameter file")
    resolved = canonical_step(step if step is not None else declared)
    if step is not None and declared is not None and canonical_step(declared) != resolved:
        raise RunnerError(
            ERR_BAD_PARAMS,
            f"step_type {declared!r} in the parameter file differs from the requested {step!r}",
        )
    allowed = set(step_keys(resolved)) | {"step_type"}
    unknown = sorted(str(key) for key in params if key not in allowed)
    problems: list[str] = []
    if unknown:
        problems.append(f"unknown parameters for {resolved}: {', '.join(unknown)}")
    fields = {**MODEL_FIELDS, **STEP_FIELDS[resolved]}
    values: dict[str, Any] = {
        key: (list(value) if isinstance(value, tuple) else value) for key, value in DEFAULT_PARAMS.items()
    }
    for key, value in params.items():
        if key not in allowed or key == "step_type" or value is None:
            continue
        try:
            coerced = _coerce(fields[key], value) if key in fields else _special(key, value)
        except (TypeError, ValueError, OverflowError) as error:
            problems.append(f"{key} = {value!r}: {error}")
            continue
        if coerced is not None:
            values[key] = coerced
    if problems:
        raise RunnerError(ERR_BAD_PARAMS, "Invalid QUEER parameters: " + "; ".join(problems))
    values["step_type"] = resolved
    return WannierStepParameters(resolved, COUNTERPARTS[resolved], MappingProxyType(values))
