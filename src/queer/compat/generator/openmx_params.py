"""The parameter file of QUEER's OpenMX (PyATB counterpart) steps: ``queer_params.json``.

The Generator writes, beside the staged scfout, one JSON object::

    {
      "step_type": "QUEER_AHC" | "QUEER_CHERN_NUMBER",
      "counterpart_step_type": "PYATB_AHC" | "PYATB_CHERN_NUMBER",
      "input_parameters": {...},   # PyATB INPUT_PARAMETERS without package and scfout_route
      "parameters": {...},         # the PyATB function block (AHC or CHERN_NUMBER)
      "workers": 8,                # optional; null or absent: every CPU of the allocation
      "scfout": "scf.scfout",      # optional; the staged scfout, relative to the workspace
      "self_convergence": false    # optional; see below
    }

``input_parameters`` and ``parameters`` hold exactly the values the Generator
writes into PyATB's ``Input`` for the counterpart step, so both runs read the same
settings; ``package`` is always ``OPENMX`` and the scfout is the staged file, so
neither ``package`` nor ``scfout_route`` is accepted. Unknown keys and values
PyATB would not accept are refused with ``queer_bad_params`` before any
calculation. :func:`translate_openmx` validates a file and
:meth:`OpenMXStepParameters.recipe` resolves it through
:func:`~queer.compat.generator.translate_pyatb` once the scfout's Fermi energy
and spin mode are known.

Execution-only settings never change a result: ``workers``,
``input_parameters.max_kpoint_num`` (a cap on the k-points of one task) and
``input_parameters.sparse_format`` (PyATB's matrix storage; recorded only).
``self_convergence: true`` asks a run whose integration refines (a Grid
threshold below 1e30) to evaluate the same integration on the doubled grid as
well, and to record ``|sigma(N) - sigma(2N)|``: the error estimate a parity
comparison of such runs needs. It costs 4 (two active axes) to 8 times the run.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any

from .pyatb_conventions import HALL_CONVENTIONS, pyatb_input_fermi_energy, resolve_hall_convention
from .results import ERR_BAD_PARAMS, RunnerError

STEP_AHC = "QUEER_AHC"
STEP_CHERN = "QUEER_CHERN_NUMBER"
OPENMX_STEPS: tuple[str, ...] = (STEP_AHC, STEP_CHERN)
COUNTERPARTS: Mapping[str, str] = MappingProxyType({STEP_AHC: "PYATB_AHC", STEP_CHERN: "PYATB_CHERN_NUMBER"})
ALIASES: Mapping[str, str] = MappingProxyType({value: key for key, value in COUNTERPARTS.items()})
FUNCTION_BLOCKS: Mapping[str, str] = MappingProxyType({STEP_AHC: "AHC", STEP_CHERN: "CHERN_NUMBER"})
FAMILIES: Mapping[str, str] = MappingProxyType({STEP_AHC: "AHC", STEP_CHERN: "CHERN_NUMBER"})
PARAMS_FILENAME = "queer_params.json"
DEFAULT_SCFOUT = "scf.scfout"
# Every refusal of an integration too large for the job's memory says this, so a
# parity comparison can tell QUEER's own limit from a failed calculation.
BUDGET_REFUSAL = "exceeds QUEER's integration budget"

TOP_LEVEL_KEYS: tuple[str, ...] = (
    "step_type",
    "counterpart_step_type",
    "input_parameters",
    "parameters",
    "workers",
    "scfout",
    "self_convergence",
)
INPUT_PARAMETER_KEYS: tuple[str, ...] = (
    "nspin",
    "fermi_energy",
    "fermi_energy_unit",
    "max_kpoint_num",
    "sparse_format",
    "hall_convention",
)
EXECUTION_ONLY_KEYS = frozenset({"max_kpoint_num", "sparse_format"})
FIXED_INPUT_KEYS = frozenset({"package", "scfout_route"})

# PyATB's defaults of the keys a comparison resolves (its documented INPUT defaults).
GRID_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {"integrate_grid": (4, 4, 4), "adaptive_grid": (4, 4, 4), "adaptive_grid_threshold": 50.0}
)
ADAPTIVE_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {"relative_error": 1e-6, "absolute_error": 0.1, "initial_grid": (1, 1, 1)}
)
FHS_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "fhs_grid": (100, 100, 1),
        "fhs_energy_window": (-0.5, 0.5),
        "fhs_energy_points": 101,
        "fhs_dimension": 0,
    }
)
AHC_DEFAULTS: Mapping[str, Any] = MappingProxyType({"engine": "kubo", "method": 0, "integrate_mode": "Grid"})
CHERN_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "method": 0,
        "occ_band": -1,
        "k_start": (0.0, 0.0, 0.0),
        "k_vect1": (1.0, 0.0, 0.0),
        "k_vect2": (0.0, 1.0, 0.0),
        "integrate_mode": "Grid",
    }
)
FUNCTION_KEYS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        STEP_AHC: frozenset(
            {*AHC_DEFAULTS, *GRID_DEFAULTS, *ADAPTIVE_DEFAULTS, *FHS_DEFAULTS},
        ),
        STEP_CHERN: frozenset({*CHERN_DEFAULTS, *GRID_DEFAULTS, *ADAPTIVE_DEFAULTS}),
    }
)


@dataclass(frozen=True)
class OpenMXStepParameters:
    """A validated ``queer_params.json`` of one OpenMX step.

    ``step`` is the QUEER step and ``counterpart`` the PyATB step it reproduces;
    ``input_parameters`` and ``parameters`` are the file's PyATB settings as given.
    """

    step: str
    counterpart: str
    input_parameters: Mapping[str, Any]
    parameters: Mapping[str, Any]
    workers: int | None
    scfout: str
    self_convergence: bool

    @property
    def function(self) -> str:
        """The PyATB function block: ``AHC`` or ``CHERN_NUMBER``."""
        return FUNCTION_BLOCKS[self.step]

    def get(self, key: str, default: Any = None) -> Any:
        """A ``parameters`` value, or ``default`` when it is absent or None."""
        value = self.parameters.get(key)
        return default if value is None else value

    def recipe(self, *, fermi_energy: float | None = None, spin_mode: str | None = None) -> dict[str, Any]:
        """:func:`~queer.compat.generator.translate_pyatb` of the settings; ``queer_bad_params`` on refusal.

        ``fermi_energy`` is the scfout's E_F (QUEER eV) that ``Auto`` takes and
        ``spin_mode`` the scfout's, checked against an explicit ``nspin``.
        """
        from ._legacy import translate_pyatb

        try:
            return translate_pyatb(
                self.counterpart,
                dict(self.parameters),
                input_parameters={**self.input_parameters, "package": "OPENMX"},
                fermi_energy=fermi_energy,
                spin_mode=spin_mode,
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise RunnerError(ERR_BAD_PARAMS, f"Invalid {self.counterpart} settings: {error}") from error

    def effective(self) -> dict[str, Any]:
        """The physics settings PyATB resolves from these values; see :func:`effective_parameters`."""
        return effective_parameters(self.step, self.input_parameters, self.parameters)


def is_openmx_step(name: Any) -> bool:
    """Whether ``name`` is a QUEER or PyATB OpenMX step (``QUEER_AHC``, ``PYATB_CHERN_NUMBER``, ...)."""
    token = str(name or "").strip().upper()
    return token in OPENMX_STEPS or token in ALIASES


def canonical_openmx_step(name: Any) -> str:
    """The QUEER step of a QUEER or PyATB OpenMX step name; ``queer_bad_params`` for any other."""
    token = str(name or "").strip().upper()
    step = ALIASES.get(token, token)
    if step not in OPENMX_STEPS:
        allowed = ", ".join(OPENMX_STEPS)
        raise RunnerError(ERR_BAD_PARAMS, f"unknown step_type {name!r}: the OpenMX steps are {allowed}")
    return step


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _integer(value: Any, name: str, minimum: int | None = None) -> int:
    number = _number(value, name)
    if number != int(number):
        raise ValueError(f"{name} must be a whole number")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(number)


def _vector(value: Any, name: str, size: int, kind: str = "float", minimum: int | None = None) -> list[Any]:
    parts = value.replace(",", " ").split() if isinstance(value, str) else value
    if not isinstance(parts, (list, tuple)) or len(parts) != size:
        raise ValueError(f"{name} must hold {size} numbers")
    if kind == "int":
        return [_integer(part, name, minimum) for part in parts]
    return [_number(part, name) for part in parts]


def _select(value: Any, name: str, options: tuple[str, ...], *, case: bool = False) -> str:
    token = str(value).strip()
    for option in options:
        if (token == option) if case else (token.lower() == option.lower()):
            return option
    raise ValueError(f"{name} must be one of {', '.join(options)}")


def _input_value(key: str, value: Any, unit: Any) -> Any:
    """An INPUT_PARAMETERS value in the type PyATB reads it as; ValueError says why not."""
    if key == "nspin":
        number = _integer(value, key)
        if number not in (1, 4):
            raise ValueError("nspin must be 1 or 4 for an OpenMX scfout")
        return number
    if key == "fermi_energy":
        if pyatb_input_fermi_energy(value, unit or "eV") is None:
            return "Auto"
        return _number(value, key)
    if key == "fermi_energy_unit":
        return _select(value, key, ("eV", "Ry"), case=True)
    if key == "max_kpoint_num":
        return _integer(value, key, 1)
    if key == "sparse_format":
        number = _integer(value, key)
        if number not in (0, 1):
            raise ValueError("sparse_format must be 0 or 1")
        return number
    if key == "hall_convention":
        return _select(value, key, HALL_CONVENTIONS)
    raise ValueError("unsupported key")


def _function_value(key: str, value: Any) -> Any:
    """A function-block value in the type PyATB reads it as; ValueError says why not."""
    if key == "engine":
        return _select(value, key, ("kubo", "pyatb", "fhs"))
    if key == "method":
        number = _integer(value, key)
        if number not in (0, 1):
            raise ValueError("method must be 0 (direct) or 1 (Kubo)")
        return number
    if key == "integrate_mode":
        return _select(value, key, ("Grid", "Adaptive"), case=True)
    if key in {"integrate_grid", "adaptive_grid", "initial_grid", "fhs_grid"}:
        return _vector(value, key, 3, "int", 1)
    if key in {"adaptive_grid_threshold", "relative_error", "absolute_error"}:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError(f"{key} must be a nonnegative number")
        threshold = float(value)
        if math.isnan(threshold) or threshold < 0:
            raise ValueError(f"{key} must be a nonnegative number")
        if math.isinf(threshold) and key != "adaptive_grid_threshold":
            raise ValueError(f"{key} must be finite")
        return threshold
    if key == "fhs_energy_window":
        low, high = _vector(value, key, 2)
        if high < low:
            raise ValueError("fhs_energy_window must be increasing")
        return [low, high]
    if key == "fhs_energy_points":
        return _integer(value, key, 1)
    if key == "fhs_dimension":
        number = _integer(value, key)
        if number not in (0, 1):
            raise ValueError("fhs_dimension must be 0 (xy) or 1 (all components)")
        return number
    if key == "occ_band":
        return _integer(value, key, -1)
    if key in {"k_start", "k_vect1", "k_vect2"}:
        return _vector(value, key, 3)
    raise ValueError("unsupported key")


def _normalized(values: Mapping[str, Any], convert: Any, prefix: str) -> tuple[dict[str, Any], list[str]]:
    normalized, problems = {}, []
    for key, value in values.items():
        if value is None:
            continue
        try:
            normalized[key] = convert(key, value)
        except (TypeError, ValueError, OverflowError) as error:
            problems.append(f"{prefix}.{key} = {value!r}: {error}")
    return normalized, problems


def translate_openmx(params: Mapping[str, Any], step: str | None = None) -> OpenMXStepParameters:
    """Validate a ``queer_params.json`` object of an OpenMX step.

    ``step`` (a QUEER or PyATB name) and the file's ``step_type`` must name the
    same step when both are given; ``counterpart_step_type`` must be that step's
    PyATB counterpart. Raises
    :class:`~queer.compat.generator.results.RunnerError` ``queer_bad_params``
    naming every unknown key and every value PyATB would refuse.
    """
    if not isinstance(params, Mapping):
        raise RunnerError(ERR_BAD_PARAMS, "the parameter file must hold a JSON object")
    declared = params.get("step_type")
    if step is None and declared is None:
        raise RunnerError(ERR_BAD_PARAMS, "no step: pass --step or set step_type in the parameter file")
    resolved = canonical_openmx_step(step if step is not None else declared)
    if step is not None and declared is not None and canonical_openmx_step(declared) != resolved:
        raise RunnerError(
            ERR_BAD_PARAMS,
            f"step_type {declared!r} in the parameter file differs from the requested {step!r}",
        )
    problems: list[str] = []
    unknown = sorted(str(key) for key in params if key not in TOP_LEVEL_KEYS)
    if unknown:
        problems.append(f"unknown keys for {resolved}: {', '.join(unknown)}")
    counterpart = params.get("counterpart_step_type")
    if counterpart is not None and str(counterpart).strip().upper() != COUNTERPARTS[resolved]:
        problems.append(f"counterpart_step_type {counterpart!r} is not {COUNTERPARTS[resolved]}")
    inputs = params.get("input_parameters") or {}
    function = params.get("parameters") or {}
    if not isinstance(inputs, Mapping):
        problems.append("input_parameters must be an object")
        inputs = {}
    if not isinstance(function, Mapping):
        problems.append("parameters must be an object")
        function = {}
    fixed = sorted(FIXED_INPUT_KEYS & set(inputs))
    if fixed:
        problems.append(
            f"input_parameters.{', '.join(fixed)} are fixed by the runner (package OPENMX, the staged scfout)"
        )
    foreign = sorted(set(inputs) - set(INPUT_PARAMETER_KEYS) - FIXED_INPUT_KEYS)
    if foreign:
        problems.append(f"unknown input_parameters: {', '.join(foreign)}")
    foreign = sorted(set(function) - FUNCTION_KEYS[resolved])
    if foreign:
        problems.append(f"unknown {FUNCTION_BLOCKS[resolved]} parameters: {', '.join(foreign)}")
    unit = inputs.get("fermi_energy_unit")
    inputs, found = _normalized(
        {k: v for k, v in inputs.items() if k in INPUT_PARAMETER_KEYS},
        lambda key, value: _input_value(key, value, unit),
        "input_parameters",
    )
    problems += found
    function, found = _normalized(
        {k: v for k, v in function.items() if k in FUNCTION_KEYS[resolved]}, _function_value, "parameters"
    )
    problems += found
    workers = params.get("workers")
    if workers is not None:
        try:
            workers = _integer(workers, "workers", 1)
        except (TypeError, ValueError) as error:
            problems.append(f"workers = {params.get('workers')!r}: {error}")
            workers = None
    scfout = params.get("scfout") or DEFAULT_SCFOUT
    path = PurePosixPath(str(scfout))
    if not isinstance(scfout, str) or path.is_absolute() or ".." in path.parts or not path.name:
        problems.append(f"scfout = {scfout!r}: a file name relative to the workspace")
    self_convergence = params.get("self_convergence")
    if self_convergence is not None and not isinstance(self_convergence, bool):
        problems.append(f"self_convergence = {self_convergence!r}: true or false")
    if problems:
        raise RunnerError(ERR_BAD_PARAMS, "Invalid QUEER parameters: " + "; ".join(problems))
    parameters = OpenMXStepParameters(
        resolved,
        COUNTERPARTS[resolved],
        MappingProxyType(inputs),
        MappingProxyType(function),
        workers,
        str(scfout),
        bool(self_convergence),
    )
    # The translation's own checks (combinations, Auto with FHS) on a placeholder E_F.
    parameters.recipe(fermi_energy=0.0)
    return parameters


def _resolved(key: str, value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_resolved(key, item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return float(stripped) if any(c in stripped for c in ".eE") else int(stripped)
        except ValueError:
            return stripped
    return value


def _comparable(key: str, value: Any, convert: Any) -> Any:
    """``value`` as :func:`translate_openmx` normalizes it (text vectors, numbers as text); else as given."""
    try:
        return convert(key, value)
    except (TypeError, ValueError, OverflowError):
        return _resolved(key, value)


def effective_parameters(
    step: str,
    input_parameters: Mapping[str, Any],
    parameters: Mapping[str, Any],
    *,
    missing_convention: str = "auto",
) -> dict[str, Any]:
    """The physics settings PyATB resolves for a step: defaults applied, inapplicable keys dropped.

    Values are normalized as :func:`translate_openmx` reads them (``"100 100 1"``
    is ``[100, 100, 1]``, ``"1e30"`` is ``1e30``). The function block keeps the keys
    of the selected engine and integration mode; a Chern grid's third axis is 1,
    as PyATB sets it. ``input`` holds ``nspin`` and the resolved
    ``hall_convention`` (omitted for the FHS engine, which ignores it); a missing
    convention is ``missing_convention``: ``auto`` for an ``Input`` file, ``pyatb``
    for PyATB's resolved echo, which names the key once PyATB has one.
    Execution-only settings and the Fermi energy (compared by its own rule) are
    left out.
    """
    resolved = canonical_openmx_step(step)
    defaults = dict(AHC_DEFAULTS if resolved == STEP_AHC else CHERN_DEFAULTS)
    block = {
        key: _comparable(key, value, _function_value)
        for key, value in {**defaults, **parameters}.items()
        if value is not None
    }
    engine = "kubo"
    if resolved == STEP_AHC:
        engine = "fhs" if str(block.get("engine", "kubo")).strip().lower() == "fhs" else "kubo"
        block["engine"] = engine
    kept: dict[str, Any] = {}
    if engine == "fhs":
        for key, default in FHS_DEFAULTS.items():
            kept[key] = _resolved(key, block.get(key, default))
    else:
        mode = str(block.get("integrate_mode") or "Grid")
        kept["integrate_mode"] = mode
        kept["method"] = int(block.get("method", 0))
        group = GRID_DEFAULTS if mode == "Grid" else ADAPTIVE_DEFAULTS
        for key, default in group.items():
            kept[key] = _resolved(key, block.get(key, default))
        if resolved == STEP_CHERN:
            for key in ("occ_band", "k_start", "k_vect1", "k_vect2"):
                kept[key] = _resolved(key, block.get(key, CHERN_DEFAULTS[key]))
            for key in ("integrate_grid", "adaptive_grid", "initial_grid"):
                if isinstance(kept.get(key), list) and len(kept[key]) == 3:
                    kept[key] = [*kept[key][:2], 1]
    if resolved == STEP_AHC:
        kept["engine"] = engine
    for key in ("integrate_grid", "adaptive_grid", "initial_grid", "fhs_grid"):
        if key in kept:
            kept[key] = [int(value) for value in kept[key]]
    for key in ("adaptive_grid_threshold", "relative_error", "absolute_error"):
        if key in kept:
            kept[key] = float(kept[key])
    inputs: dict[str, Any] = {}
    if input_parameters.get("nspin") is not None:
        inputs["nspin"] = int(
            _comparable("nspin", input_parameters["nspin"], lambda k, v: _input_value(k, v, None))
        )
    if engine != "fhs":
        convention = input_parameters.get("hall_convention")
        inputs["hall_convention"] = resolve_hall_convention(
            missing_convention if convention is None else convention
        )
    return {"function": kept, "input": inputs}


def params_from_pyatb_input(
    blocks: Mapping[str, Mapping[str, Any]],
    step: str | None = None,
    *,
    workers: int | None = None,
    scfout: str = DEFAULT_SCFOUT,
    self_convergence: bool | None = None,
    echo: bool = False,
) -> dict[str, Any]:
    """The ``queer_params.json`` of the PyATB ``Input`` blocks (:func:`~.pyatb_files.parse_input`).

    ``step`` defaults to the step of the single function block present. The
    INPUT_PARAMETERS keep every key but ``package`` and ``scfout_route``, and
    ``hall_convention`` is always written: an absent key is PyATB's default
    ``auto`` in an ``Input`` file and, with ``echo`` (the blocks of PyATB's
    resolved ``Out/input.json``), ``pyatb``, the native sign of a PyATB build
    older than the key.
    """
    if step is None:
        present = [name for name, block in FUNCTION_BLOCKS.items() if block in blocks]
        if len(present) != 1:
            raise ValueError("the Input names no single AHC or CHERN_NUMBER block; pass the step")
        step = present[0]
    resolved = canonical_openmx_step(step)
    inputs = {
        key: value
        for key, value in (blocks.get("INPUT_PARAMETERS") or {}).items()
        if key not in FIXED_INPUT_KEYS
    }
    if inputs.get("hall_convention") is None:
        inputs["hall_convention"] = "pyatb" if echo else "auto"
    params: dict[str, Any] = {
        "step_type": resolved,
        "counterpart_step_type": COUNTERPARTS[resolved],
        "input_parameters": inputs,
        "parameters": dict(blocks.get(FUNCTION_BLOCKS[resolved]) or {}),
        "scfout": scfout,
    }
    if workers is not None:
        params["workers"] = int(workers)
    if self_convergence is not None:
        params["self_convergence"] = bool(self_convergence)
    return params
