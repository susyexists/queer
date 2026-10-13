"""Run configuration: every convention of Eqs. 7-13 is an explicit, recorded setting.

`PAPER` is the default, matching HHH's printed reference equations. `PHYSICAL`
explicitly includes corrections validated against the time-dependent
Schroedinger equation. `PHYSICAL_EQ8` explicitly selects the
metric approximation when complex continuation and wavefunction phases are
unavailable;
`PAPER_FIGURES` adds the reference figures' 1/(4 pi) scale.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np


def _check(name, value, allowed):
    if value not in allowed:
        raise ValueError(f"{name}={value!r} not in {allowed}")


def _positive(name, value, *, zero=False, infinity=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a number")
    if (
        math.isnan(float(value))
        or (not infinity and not math.isfinite(float(value)))
        or (value < 0 if zero else value <= 0)
    ):
        raise ValueError(
            f"{name} must be {'nonnegative' if zero else 'positive'} {'or infinity' if infinity else 'and finite'}"
        )


def _integer(name, value, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _boolean(name, value):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")


def _coerce(obj) -> None:
    """Store numpy scalars as Python scalars and ints of float fields as floats.

    Keeps to_json()/digest() serializable for settings taken from numpy arrays (np.int64, np.float32, ...)
    and makes the digest independent of the spelling (5e4 and 50000 give the same cache keys).
    """
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if isinstance(v, np.generic):
            v = v.item()
        if f.type in ("float", "float | None") and isinstance(v, int) and not isinstance(v, bool):
            v = float(v)
        object.__setattr__(obj, f.name, v)


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


_NONFINITE = ("inf", "+inf", "-inf", "nan", "infinity", "+infinity", "-infinity")


def _section_from_dict(cls, values: dict):
    """Section dataclass from a dict; float fields also accept "inf", "-inf", "nan" (io.cache writes these)."""
    types = {f.name: f.type for f in dataclasses.fields(cls)}
    kw = {}
    for k, v in values.items():
        if (
            isinstance(v, str)
            and str(types.get(k, "")).startswith("float")
            and v.strip().lower() in _NONFINITE
        ):
            v = float(v)
        kw[k] = v
    return cls(**kw)


@dataclass(frozen=True)
class Conventions:
    berry: str = "c"  # Omega in Eqs. 10-11: "c" | "c_minus_v" | "v"
    metric: str = "cv"  # metric in Eq. 8: "cv" (band-resolved) | "v_total" | "c_total"
    ec_method: str = "eq8"  # "eq8" (as printed) | "ddp" (complex branch-point exponent)
    prefactor: str = "e2_h"  # "e2_h" (as printed) | "e2_2pi_h" (Kubo normalization)
    figure_scale: float = 1.0  # 1/(4 pi) reproduces the paper's plotted J^{N w}
    charge_sign: int = 1  # +1: paper's hbar dk/dt = +eE; -1: physical electrons
    vlc_reference: str = "computed"

    def __post_init__(self):
        _coerce(self)
        _check("berry", self.berry, ("c", "c_minus_v", "v"))
        _check("metric", self.metric, ("cv", "v_total", "c_total"))
        _check("ec_method", self.ec_method, ("eq8", "ddp"))
        _check("prefactor", self.prefactor, ("e2_h", "e2_2pi_h"))
        _check("charge_sign", self.charge_sign, (1, -1))
        if self.vlc_reference != "computed":
            raise ValueError(
                "vlc_reference must be 'computed'; use FourBandParams.paper_vlc_matched() "
                "(M=-1.0215e-3 eV) to reproduce the paper's critical field"
            )
        _positive("figure_scale", self.figure_scale)


@dataclass(frozen=True)
class Relaxation:
    T1: float = 2e-12  # s; math.inf allowed
    T2: float = 2e-12  # s; pure dephasing time ("paper") or total coherence time ("bloch"); math.inf allowed
    coherence: str = "paper"  # "paper": rho_cv decays at 1/(2 T1) + 1/T2; "bloch": at 1/T2 (needs T2 <= 2 T1)

    def __post_init__(self):
        _coerce(self)
        _check("coherence", self.coherence, ("paper", "bloch"))
        _positive("T1", self.T1, infinity=True)
        _positive("T2", self.T2, infinity=True)
        if not (self.T1 > 0 and self.T2 > 0):
            raise ValueError("T1 and T2 must be positive (use math.inf for no relaxation)")
        if self.coherence == "bloch" and self.T2 > 2.0 * self.T1:
            raise ValueError(
                f"coherence='bloch' needs T2 <= 2 T1 (got T1 = {self.T1:g} s, T2 = {self.T2:g} s): "
                "T2 is then the total coherence time, and the population after two crossings turns negative"
            )


@dataclass(frozen=True)
class CrossingPolicy:
    definition: str = "gap_minima"  # "gap_minima" | "diabatic" | "branch_point"
    mode: str = "per_line"  # "per_line" | "global" (k0 and Lambda of the t = 0 line for |t| < k_ring)
    merge: str = "none"  # "none" | "lz_width" (merge crossings closer than merge_factor * k_LZ)
    merge_factor: float = 1.0
    window: float = 0.03  # half-width of the k_s window, 1/A
    scan_points: int = 2049
    prominence_rel: float = 1e-12  # double wells with lower barriers count as one minimum (rounding ~1e-15)
    touching_floor: float = 1e-12  # eV; smaller gaps are touching points (E_c = 0, P = 1)
    max_crossings: int = 8
    use_radial: bool = True  # isotropic geometry: line minima from the radial gap profile (exact)

    def __post_init__(self):
        _coerce(self)
        _check("definition", self.definition, ("gap_minima", "diabatic", "branch_point"))
        _check("mode", self.mode, ("per_line", "global"))
        _check("merge", self.merge, ("none", "lz_width"))
        for name in ("window", "merge_factor", "touching_floor"):
            _positive(name, getattr(self, name))
        _positive("prominence_rel", self.prominence_rel, zero=True)
        _integer("scan_points", self.scan_points, 5)
        _integer("max_crossings", self.max_crossings)
        _boolean("use_radial", self.use_radial)


@dataclass(frozen=True)
class PhaseModel:
    geometric: bool = False  # add the gauge-invariant shift-vector phase gamma_R between crossings
    stokes: bool = False  # add the Stokes phase at each crossing
    twisted_lz: bool = False  # shift-vector correction of P_LZ (nonreciprocal in k_t and E)
    phase_nodes: int = 201
    phase_max_nodes: int = 3201
    phase_tolerance: float = 1e-5

    def __post_init__(self):
        _coerce(self)
        for name in ("geometric", "stokes", "twisted_lz"):
            _boolean(name, getattr(self, name))
        _integer("phase_nodes", self.phase_nodes, 3)
        _integer("phase_max_nodes", self.phase_max_nodes, self.phase_nodes)
        _positive("phase_tolerance", self.phase_tolerance)


@dataclass(frozen=True)
class Quadrature:
    du: float = 0.3  # step of the single-exponential (logistic) map; ~1/du nodes per e-fold
    dmin: float = 1e-12  # smallest node distance to a feature point, 1/A
    t_max: float | None = None  # transverse extent; None: where E_c(t) > p_cut * E_max
    p_cut: float = 40.0
    lam_per_decade: int = 8  # density of the Laplace lambda-tables
    rtol: float = 1e-5  # unused (accuracy is set by du and dmin); kept so saved configurations reload
    k_patch: float = (
        1e-5  # below this |k| the near-Gamma precision patch is used, 1/A (FP64 is accurate above)
    )

    def __post_init__(self):
        _coerce(self)
        for name in ("du", "dmin", "p_cut", "rtol", "k_patch"):
            _positive(name, getattr(self, name))
        if self.t_max is not None:
            _positive("t_max", self.t_max)
        _integer("lam_per_decade", self.lam_per_decade)


@dataclass(frozen=True)
class FourierCfg:
    n_theta: int = 384  # Gauss-Legendre nodes in tau, theta = (pi/2) tau^2, per quarter wave
    n_max: int = 41
    n_cap: int = 16384
    adaptive: bool = True
    tol: float = 1e-8

    def __post_init__(self):
        _coerce(self)
        _integer("n_max", self.n_max, 0)
        _integer("n_theta", self.n_theta)
        _integer("n_cap", self.n_cap)
        _positive("tol", self.tol)
        _boolean("adaptive", self.adaptive)


@dataclass(frozen=True)
class DriveCfg:
    angle: float = 0.0  # field direction in the model frame, rad (Wannier: measured from Gamma-K)
    E_max: float = 5e4  # largest |E| (V/m) the geometry stage must support (finite, > E_min)
    E_min: float = 1.0  # smallest |E| (V/m) resolved without extrapolation (sets the node clustering; > 0)

    def __post_init__(self):
        _coerce(self)
        if not (math.isfinite(self.E_min) and self.E_min > 0):
            raise ValueError(f"drive.E_min must be finite and > 0 V/m, got {self.E_min!r}")
        if not (math.isfinite(self.E_max) and self.E_max > self.E_min):
            raise ValueError(
                f"drive.E_max must be finite and > drive.E_min = {self.E_min:g} V/m, got {self.E_max!r}"
            )
        if not math.isfinite(self.angle):
            raise ValueError(f"drive.angle must be finite (rad), got {self.angle!r}")


_SECTIONS = {
    "conventions": Conventions,
    "relaxation": Relaxation,
    "crossings": CrossingPolicy,
    "phase": PhaseModel,
    "quadrature": Quadrature,
    "fourier": FourierCfg,
    "drive": DriveCfg,
}


@dataclass(frozen=True)
class RunConfig:
    conventions: Conventions = field(default_factory=Conventions)
    relaxation: Relaxation = field(default_factory=Relaxation)
    crossings: CrossingPolicy = field(default_factory=CrossingPolicy)
    phase: PhaseModel = field(default_factory=PhaseModel)
    quadrature: Quadrature = field(default_factory=Quadrature)
    fourier: FourierCfg = field(default_factory=FourierCfg)
    drive: DriveCfg = field(default_factory=DriveCfg)
    use_symmetry_shortcuts: bool = False  # shortcuts are applied only if certified as well
    label: str = "paper"

    def __post_init__(self):
        _coerce(self)
        _boolean("use_symmetry_shortcuts", self.use_symmetry_shortcuts)
        if not isinstance(self.label, str):
            raise ValueError("label must be a string")
        for name, cls in _SECTIONS.items():
            if not isinstance(getattr(self, name), cls):
                raise ValueError(f"{name} must be a {cls.__name__} settings object")

    def with_(self, **changes) -> "RunConfig":
        """Return a copy with changes; keys are top-level names or dotted "section.key"."""
        sections = {}
        top = {}
        for key, value in changes.items():
            if "." in key:
                section, sub = key.split(".", 1)
                if section not in _SECTIONS:
                    raise KeyError(key)
                part = getattr(self, section)
                if not hasattr(part, sub):
                    raise KeyError(key)
                sections.setdefault(section, {})[sub] = value
            else:
                if key not in {f.name for f in dataclasses.fields(self)}:
                    raise KeyError(key)
                top[key] = value
        for section, updates in sections.items():
            part = top.get(section, getattr(self, section))
            top[section] = dataclasses.replace(part, **updates)
        return dataclasses.replace(self, **top)

    def to_dict(self) -> dict:
        from ._storage import json_safe

        return json_safe(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        """Inverse of to_dict; also reads configurations stored in NPZ metadata (non-finite floats as strings)."""
        kwargs = {}
        for name, value in d.items():
            if name in _SECTIONS:
                kwargs[name] = _section_from_dict(_SECTIONS[name], value)
            elif name in ("use_symmetry_shortcuts", "label"):
                kwargs[name] = value
            else:
                raise KeyError(name)
        return cls(**kwargs)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=_json_default)

    @classmethod
    def from_json(cls, s: str) -> "RunConfig":
        return cls.from_dict(json.loads(s))

    def digest(self) -> str:
        """SHA-256 of the canonical JSON form (labels excluded)."""
        d = self.to_dict()
        d.pop("label", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=_json_default).encode()).hexdigest()


PAPER = RunConfig(label="paper")
PAPER_FIGURES = PAPER.with_(**{"conventions.figure_scale": 1.0 / (4.0 * math.pi)}, label="paper_figures")
PHYSICAL = RunConfig(
    conventions=Conventions(berry="c_minus_v", ec_method="ddp", prefactor="e2_2pi_h", charge_sign=-1),
    crossings=CrossingPolicy(definition="branch_point"),
    phase=PhaseModel(geometric=True, stokes=True, twisted_lz=True),
    label="physical",
)
PHYSICAL_EQ8 = PHYSICAL.with_(
    **{
        "conventions.ec_method": "eq8",
        "crossings.definition": "gap_minima",
        "phase.geometric": False,
        "phase.stokes": False,
        "phase.twisted_lz": False,
    },
    label="physical_eq8",
)

_NO_RELAXATION = {"relaxation.T1": math.inf, "relaxation.T2": math.inf}

FIGURE_PRESETS = {
    "fig4b": PAPER.with_(label="fig4b"),  # P_LZ on the k_y = 0 line at VL/VLc = 0.9
    "fig4d": PAPER_FIGURES.with_(**_NO_RELAXATION, **{"crossings.definition": "diabatic"}, label="fig4d"),
    # Fig. 5c / Ext. Data Fig. 7a: the pixel test (analysis/ext7a.py) favours diabatic crossings
    "fig5c": PAPER_FIGURES.with_(
        **{"crossings.definition": "diabatic", "relaxation.T2": math.inf}, label="fig5c"
    ),
    "ext7a": PAPER_FIGURES.with_(
        **{"crossings.definition": "diabatic", "relaxation.T2": math.inf}, label="ext7a"
    ),
    "ext7e_left": PAPER.with_(label="ext7e_left"),  # E_c at the gap minimum
    "ext7e_right": PAPER.with_(
        **{"crossings.definition": "diabatic"}, label="ext7e_right"
    ),  # Lambda between +-k_d
    "ext9b": PAPER_FIGURES.with_(**_NO_RELAXATION, label="ext9b"),
    "ext9f": PAPER.with_(**{"crossings.definition": "diabatic"}, label="ext9f"),  # Eq. 8 at k_d
}


def preset(name: str) -> RunConfig:
    """Explicit physical, metric-approximation, or reference-paper profile."""
    named = {
        "paper": PAPER,
        "paper_figures": PAPER_FIGURES,
        "physical": PHYSICAL,
        "physical_eq8": PHYSICAL_EQ8,
        **FIGURE_PRESETS,
    }
    try:
        return named[name]
    except KeyError:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(named)}") from None


@dataclass(frozen=True)
class FiniteFieldOptions(RunConfig):
    """Native options with HHH's PAPER defaults; physical profiles are explicit.

    Infinite relaxation serializes as "inf".
    """


def resolve_options(options=None, *, preset_name=None):
    """Resolve a preset plus explicit section overrides, rejecting unknown settings."""
    if options is None:
        return preset(preset_name or "paper")
    if isinstance(options, RunConfig):
        return options
    if isinstance(options, str):
        return preset(options)
    if not isinstance(options, dict):
        raise TypeError("options must be FiniteFieldOptions, a preset string, or a settings mapping")
    supplied = dict(options)
    selected = supplied.pop("preset", preset_name or "paper")
    base = preset(selected).to_dict()
    for key, value in supplied.items():
        if key in _SECTIONS:
            if not isinstance(value, dict):
                raise ValueError(f"options.{key} must be a mapping")
            base[key] = {**base[key], **value}
        elif "." in key:
            section, name = key.split(".", 1)
            if section not in _SECTIONS or name not in base[section]:
                raise ValueError(f"Unknown finite-field setting: {key}")
            base[section][name] = value
        elif key in ("label", "use_symmetry_shortcuts"):
            base[key] = value
        else:
            raise ValueError(f"Unknown finite-field option section: {key}")
    return FiniteFieldOptions.from_dict(base)
