"""Parity of a Generator Wannier step between a reference run and a candidate run.

:func:`compare` reads two run directories of the same step and returns a report
(a JSON-able dict); :func:`render_markdown` renders it. A run directory holds the
results document (QUEER's ``queer_results.json`` or the Generator PythTB driver's
``pythtb_results.json``), optionally the parameter file (``queer_params.json`` or
``pythtb_params.json``) and the ``Out/*.json`` data files of the step. Model files
are never read: a run's inputs are known by the SHA-256 digests it recorded.

**Preconditions.** Any failure makes the verdict ``not_comparable`` with its
reasons, and no quantity is compared:

- one readable results document per directory, both of the requested step family
  (``PYTHTB_X`` and ``QUEER_X`` are the same family X);
- the same source WANNIER job when both runs name one;
- the same inputs: equal ``provenance.inputs`` SHA-256 for every file both runs
  recorded. Without recorded inputs a model fingerprint must match instead and the
  report sets ``inputs_verified_by_fingerprint``: ``num_wann``, the lattice to
  1e-9 Å, ``n_R``, ``n_R_after_ws`` and ``n_R_model``, ``folded_axes``,
  ``periodic_dirs``, the Wigner-Seitz mode and its scores to 1e-6 meV and the
  ``_band.dat`` deviations to 1e-6 meV;
- the same effective step parameters when both parameter files are present (the
  step's keys with the driver defaults; ``workers``, ``chunk_size`` and
  ``time_budget_s`` are ignored and the structure's atom order does not count);
- the same Fermi energy and the same derived decisions: Wigner-Seitz mode,
  metal or insulator, occupied band count, isolation, occupation mode,
  applicability, the planes and their transverse sampling, and the slab kind,
  cells and finite axis. A threshold decision that flipped is reported here, not
  as a spurious failure.

**Reference semantics.** :data:`REFERENCE_SEMANTICS`, keyed by the producer's
``provenance.core_sha256``, adapts the comparison to the revision that wrote the
reference. The first driver revision (9f43886) took DOS(E_F) from the first bin
centred at or above E_F, wrote its Fermi-surface contours without a plane frame,
used another localization estimate and recorded the DOS mesh on all three axes:
its DOS(E_F) is compared with the candidate's value recomputed under that rule,
its Fermi surface by the crossing bands only, its localization not at all, and its
mesh at the candidate's mesh axes. A fermi-mode BERRY reference without
``berry.conventions`` predates the sign fix: its maps, path curvature and integrals
are compared after a factor -1. A difference removed by a recorded convention is
``agree_after_convention``.

**Tolerances** (data-driven, the same for every material). With δ_W the
Hermiticity Weyl bound the candidate's model block records (0 when absent),
eigenvalues agree to τ_E = max(1e-8 eV, δ_W) + 1e-13 max|E|; gaps, band edges and
Kramers splittings to 2 τ_E; the DOS, the PDOS and DOS(E_F) to 1e-9 of their
largest value (eigenvalues crossing a histogram bin edge are allowed only when
δ_W > 0); Fermi-surface contours to a symmetric Hausdorff distance of
max(1e-9 |b|, 1e-6) in their stored coordinates; Chern numbers, Z2 indices,
convergence flags and transverse samplings exactly, Wannier charge centres to 1e-9
after an optimal circular matching per loop; bands-mode Berry maps to 1e-8 of
their own largest magnitude, the Kubo Chern number to 1e-8 and the quantum weight
to 1e-6 relative. In fermi mode the points where the maps disagree form the
flagged set F (a level within τ_E of E_F, or an occupied-empty pair at the 1e-6 eV
skip threshold, decides such a point), and the plane integral must agree to
1e-8 + sum_F (|Ω_Q| + |Ω_P|) / (n^2 2 pi); F must stay a minority of the points
carrying curvature and each document's integral must equal its own map. The path
curvature agrees pointwise to 1e-8 of its largest magnitude. Slab levels follow
the band tolerance per k-point (a level within τ_E of the window edge may be
missing on one side), edge weights summed over clusters of levels within 1e-5 eV
(clusters at the window edge excluded) agree to 1e-9, and the edge-crossing counts
exactly unless a per-state weight lies within 1e-6 of 1/2. Float leaves of the
model and validation blocks agree to 1e-6 relative, other leaves exactly; the
warning sets agree by code.

**Classes and verdicts.** A quantity is ``agree``, ``agree_after_convention``,
``differs_documented``, ``differs_unexplained`` or ``not_applicable``. The verdict
is ``not_comparable`` when a precondition fails, ``fail`` when any quantity
``differs_unexplained``, ``pass_with_documented_differences`` when any
``differs_documented`` and ``pass`` otherwise. Two failed runs with equivalent
codes (``pythtb_X`` and ``queer_X``) pass.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .berry import plane_file_stem
from .estimators import dos_at_fermi
from .pythtb_params import DEFAULT_PARAMS, step_keys, translate_pythtb
from .results import (
    OUT_DIR,
    WARN_BAND_DAT_DEVIATION,
    WARN_BAND_DAT_MISSING,
    WARN_CHERN_NOT_INTEGER,
    WARN_DOS_OUTSIDE_GRID,
    WARN_FERMI_OUTSIDE_FROZEN,
    WARN_METRIC_SKIPPED,
    WARN_PATH_CURVATURE_SKIPPED,
    WARN_PLOT_SKIPPED,
    WARN_WS_ASSUMED_DEFAULT,
    WARN_WS_NO_MP_GRID,
    WARN_Z2_FORCED,
    RunnerError,
    to_jsonable,
)

REPORT_SCHEMA = "queer-generator-parity/1"
RESULT_FILENAMES = {"queer": "queer_results.json", "pythtb": "pythtb_results.json"}
PARAMS_FILENAMES = {"queer": "queer_params.json", "pythtb": "pythtb_params.json"}
FAMILIES: tuple[str, ...] = ("BANDS", "TOPOLOGY", "BERRY", "SLAB")
ENGINE_PREFIXES = {"QUEER_": "queer", "PYTHTB_": "pythtb"}

AGREE = "agree"
AFTER_CONVENTION = "agree_after_convention"
DOCUMENTED = "differs_documented"
UNEXPLAINED = "differs_unexplained"
NOT_APPLICABLE = "not_applicable"
CLASSES: tuple[str, ...] = (AGREE, AFTER_CONVENTION, DOCUMENTED, UNEXPLAINED, NOT_APPLICABLE)

PASS = "pass"
PASS_DOCUMENTED = "pass_with_documented_differences"
FAIL = "fail"
NOT_COMPARABLE = "not_comparable"
VERDICTS: tuple[str, ...] = (PASS, PASS_DOCUMENTED, FAIL, NOT_COMPARABLE)
PASSING_VERDICTS: tuple[str, ...] = (PASS, PASS_DOCUMENTED)

# Tolerances.
EIGENVALUE_FLOOR_EV = 1e-8
EIGENVALUE_ROUNDING = 1e-13
EDGE_FACTOR = 2.0
DOS_RELATIVE = 1e-9
FERMI_SURFACE_RELATIVE = 1e-9
FERMI_SURFACE_FLOOR = 1e-6
WCC_TOL = 1e-9
BERRY_MAP_RELATIVE = 1e-8
CHERN_KUBO_TOL = 1e-8
QUANTUM_WEIGHT_RELATIVE = 1e-6
FERMI_INTEGRAL_TOL = 1e-8
SLAB_CLUSTER_EV = 1e-5
SLAB_WEIGHT_TOL = 1e-9
SLAB_HALF_WEIGHT_MARGIN = 1e-6
SLAB_EDGE_THRESHOLD = 0.5
BLOCK_RELATIVE = 1e-6
FINGERPRINT_LATTICE_A = 1e-9
FINGERPRINT_MEV = 1e-6
FERMI_ENERGY_RELATIVE = 1e-12
PARAMETER_RELATIVE = 1e-12
COORDINATE_TOL = 1e-12
# Points of a flagged set listed in a report.
FLAGGED_LISTED = 50
# Points per block of the contour distance evaluation.
DISTANCE_BLOCK = 1 << 20

# What reading a document of an unexpected shape raises.
MALFORMED = (KeyError, IndexError, TypeError, ValueError, AttributeError)
# Parameters that change how a run executes, never its result.
EXECUTION_PARAMS = frozenset({"workers", "chunk_size", "time_budget_s"})
# Model keys that name the producer or are recorded by one producer only.
PRODUCER_MODEL_KEYS = frozenset(
    {"approximation", "pythtb_version", "queer_version", "hermiticity_weyl_bound"}
)
# Species lists whose order follows the structure's atom order only.
ORDER_FREE_LEAVES = frozenset({"model.species", "dos.species"})
FINGERPRINT_EXACT_KEYS = ("num_wann", "n_R", "n_R_after_ws", "n_R_model", "folded_axes", "periodic_dirs")
BAND_DAT_DEVIATIONS = ("max_meV", "rms_meV", "in_window_max_meV", "in_window_rms_meV")

OCCUPATION_DERIVED = ("n_occ", "metal", "isolated")
OCCUPATION_ENERGIES_EV = ("gap_eV", "direct_gap_eV", "vbm_eV", "cbm_eV")
OCCUPATION_ENERGIES_MEV = ("isolation_meV",)
OCCUPATION_KPOINTS = {"vbm_k": "vbm_eV", "cbm_k": "cbm_eV", "direct_gap_k": "direct_gap_eV"}

# Messages of both producers, by stable warning code. Plot warnings depend on the
# plotting environment and are left out of the comparison.
WARNING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (code, re.compile(pattern))
    for code, pattern in (
        (WARN_PLOT_SKIPPED, r"plot skipped: |\.png skipped: "),
        (WARN_WS_NO_MP_GRID, r"^The \.win has neither mp_grid nor a kpoints block"),
        (WARN_WS_ASSUMED_DEFAULT, r"^No wannier90_band\.dat to check the Wigner-Seitz correction"),
        (WARN_BAND_DAT_DEVIATION, r"^(PythTB|Model) bands differ from wannier90_band\.dat"),
        (WARN_BAND_DAT_MISSING, r"^wannier90_band\.dat was not staged"),
        (WARN_FERMI_OUTSIDE_FROZEN, r"^E_F lies outside the frozen window"),
        (WARN_DOS_OUTSIDE_GRID, r"mesh eigenvalues fell outside the DOS grid$"),
        (WARN_Z2_FORCED, r"^Z2 forced on without Kramers degeneracy"),
        (WARN_CHERN_NOT_INTEGER, r"^the Kubo Chern number .* from an integer"),
        (WARN_METRIC_SKIPPED, r"^quantum metric skipped"),
        (WARN_PATH_CURVATURE_SKIPPED, r"^path curvature skipped: "),
    )
)
ENVIRONMENT_WARNINGS = frozenset({WARN_PLOT_SKIPPED})


@dataclass(frozen=True)
class Semantics:
    """How a producer revision computed and recorded the quantities that changed since."""

    name: str
    description: str
    dos_at_fermi_rule: str = "linear_interpolation"
    fermi_surface_contours: bool = True
    localization: bool = True
    dos_mesh_on_every_axis: bool = False

    def rules(self) -> dict[str, Any]:
        return {
            "dos_at_fermi_rule": self.dos_at_fermi_rule,
            "fermi_surface_contours_compared": self.fermi_surface_contours,
            "localization_compared": self.localization,
            "dos_mesh_on_every_axis": self.dos_mesh_on_every_axis,
        }


CURRENT_SEMANTICS = Semantics(
    "current",
    "QUEER and the vendored Generator driver: DOS(E_F) interpolated linearly, Fermi-surface contours in "
    "the plane frame, DOS mesh per periodic axis",
)
DRIVER_9F43886 = Semantics(
    "driver-9f43886",
    "the first Generator PythTB driver revision: DOS(E_F) from the first bin centred at or above E_F, "
    "Fermi-surface contours without a plane frame, another localization estimate, the DOS mesh on all "
    "three axes, no recorded inputs",
    dos_at_fermi_rule="first_nonnegative_bin",
    fermi_surface_contours=False,
    localization=False,
    dos_mesh_on_every_axis=True,
)
REFERENCE_SEMANTICS: Mapping[str, Semantics] = MappingProxyType(
    {"f6c9b7af239b530253cb7eeb9b209847e7d47baaf0237cb835a326c17dd5ec32": DRIVER_9F43886}
)


def step_family(name: Any) -> str:
    """The step family (``BANDS``, ``TOPOLOGY``, ``BERRY``, ``SLAB``) of a QUEER, PythTB or bare step name."""
    token = str(name or "").strip().upper()
    for prefix in ENGINE_PREFIXES:
        if token.startswith(prefix):
            token = token[len(prefix) :]
            break
    if token not in FAMILIES:
        raise ValueError(f"unknown step {name!r}: compare one of {', '.join(FAMILIES)} (QUEER_ or PYTHTB_)")
    return token


def semantics_of(document: Mapping[str, Any]) -> tuple[Semantics, str | None]:
    """The registered semantics of the revision that wrote ``document``, and its producer digest."""
    provenance = document.get("provenance") or {}
    for key in ("core_sha256", "qgeom_sha256"):
        digest = provenance.get(key)
        if isinstance(digest, str) and digest in REFERENCE_SEMANTICS:
            return REFERENCE_SEMANTICS[digest], digest
    digest = provenance.get("core_sha256") or provenance.get("queer_record_sha256")
    return CURRENT_SEMANTICS, digest if isinstance(digest, str) else None


def eigenvalue_tolerance(weyl_bound: float, magnitude: float) -> float:
    """τ_E = max(1e-8 eV, δ_W) + 1e-13 max|E|."""
    return max(EIGENVALUE_FLOOR_EV, float(weyl_bound)) + EIGENVALUE_ROUNDING * float(magnitude)


def warning_code(message: str) -> str:
    """The stable code of a warning message of either producer; unknown messages keep their text."""
    for code, pattern in WARNING_PATTERNS:
        if pattern.search(message):
            return code
    return f"message:{message}"


def equivalent_failure_codes(reference: Any, candidate: Any) -> bool:
    """Whether two failure codes name the same failure (``pythtb_oom`` and ``queer_oom``)."""

    def stem(code: Any) -> str | None:
        if not isinstance(code, str):
            return None
        for prefix in ("pythtb_", "queer_"):
            if code.startswith(prefix):
                return code[len(prefix) :]
        return code

    return stem(reference) is not None and stem(reference) == stem(candidate)


# ---------------------------------------------------------------------------
# Run directories
# ---------------------------------------------------------------------------


@dataclass
class RunDirectory:
    """The results document, parameter file and data files of one run."""

    role: str
    directory: Path
    document: dict[str, Any] = field(default_factory=dict)
    results_file: str | None = None
    params: dict[str, Any] | None = None
    params_file: str | None = None
    problems: list[str] = field(default_factory=list)
    _out: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, role: str, directory: str | Path) -> RunDirectory:
        run = cls(role, Path(directory))
        if not run.directory.is_dir():
            run.problems.append(f"{role} directory {run.directory} does not exist")
            return run
        found = [
            (engine, name) for engine, name in RESULT_FILENAMES.items() if (run.directory / name).is_file()
        ]
        if not found:
            names = " or ".join(RESULT_FILENAMES.values())
            run.problems.append(f"{role} directory {run.directory} holds no {names}")
            return run
        if len(found) > 1:
            run.problems.append(
                f"{role} directory {run.directory} holds both {' and '.join(n for _, n in found)}"
            )
            return run
        engine, name = found[0]
        run.results_file = name
        try:
            document = json.loads((run.directory / name).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            run.problems.append(f"cannot read {role} {name}: {error}")
            return run
        if not isinstance(document, dict):
            run.problems.append(f"{role} {name} does not hold a JSON object")
            return run
        run.document = document
        preferred = [PARAMS_FILENAMES[engine]] + [p for e, p in PARAMS_FILENAMES.items() if e != engine]
        for params_name in preferred:
            path = run.directory / params_name
            if not path.is_file():
                continue
            try:
                params = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                run.problems.append(f"cannot read {role} {params_name}: {error}")
                break
            if isinstance(params, dict):
                run.params, run.params_file = params, params_name
            else:
                run.problems.append(f"{role} {params_name} does not hold a JSON object")
            break
        return run

    @property
    def engine(self) -> str:
        engine = self.document.get("engine")
        if isinstance(engine, str):
            return engine
        return "queer" if self.results_file == RESULT_FILENAMES["queer"] else "pythtb"

    @property
    def ok(self) -> bool:
        return self.document.get("status") == "ok"

    def block(self, name: str) -> dict[str, Any]:
        value = self.document.get(name)
        return value if isinstance(value, dict) else {}

    def out(self, name: str) -> Any:
        """The JSON data file ``Out/<name>`` of the run, or None when it is missing or unreadable."""
        if name in self._out:
            return self._out[name]
        value = None
        if re.fullmatch(r"[A-Za-z0-9_.-]+\.json", name):
            try:
                value = json.loads((self.directory / OUT_DIR / name).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                value = None
        self._out[name] = value
        return value

    def summary(self) -> dict[str, Any]:
        provenance = self.document.get("provenance") or {}
        return {
            "directory": str(self.directory),
            "results_file": self.results_file,
            "params_file": self.params_file,
            "engine": self.engine if self.document else None,
            "step_type": self.document.get("step_type"),
            "status": self.document.get("status"),
            "core_sha256": provenance.get("core_sha256"),
            "queer_version": provenance.get("queer_version"),
            "queer_record_sha256": provenance.get("queer_record_sha256"),
        }


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def _array(value: Any) -> np.ndarray | None:
    """``value`` as a float array (null as NaN), or None for anything that is not numeric."""
    if value is None or isinstance(value, (bool, str)):
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return array


def _max_difference(reference: Any, candidate: Any) -> float | None:
    """Largest elementwise |candidate - reference|; None when shapes or null positions differ."""
    a, b = _array(reference), _array(candidate)
    if a is None or b is None or a.shape != b.shape:
        return None
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if not np.array_equal(nan_a, nan_b):
        return None
    if a.size == 0 or nan_a.all():
        return 0.0
    return float(np.max(np.abs(a[~nan_a] - b[~nan_b])))


def _max_magnitude(*values: Any) -> float:
    largest = 0.0
    for value in values:
        array = _array(value)
        if array is not None and array.size and not np.isnan(array).all():
            largest = max(largest, float(np.nanmax(np.abs(array))))
    return largest


def _has_float(value: Any) -> bool:
    if isinstance(value, float):
        return True
    if isinstance(value, list):
        return any(_has_float(item) for item in value)
    return False


def _relative_difference(reference: Any, candidate: Any) -> float | None:
    """Largest |candidate - reference| relative to |reference| (absolute where the reference is 0)."""
    a, b = _array(reference), _array(candidate)
    if a is None or b is None or a.shape != b.shape:
        return None
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if not np.array_equal(nan_a, nan_b):
        return None
    keep = ~nan_a
    if not keep.any():
        return 0.0
    scale = np.where(np.abs(a[keep]) > 0, np.abs(a[keep]), 1.0)
    return float(np.max(np.abs(b[keep] - a[keep]) / scale))


def _leaves(value: Any, prefix: str) -> dict[str, Any]:
    """Every leaf of a block by dotted path; a list of objects is followed per element."""
    if isinstance(value, Mapping):
        if not value:
            return {prefix: {}}
        leaves: dict[str, Any] = {}
        for key, item in value.items():
            leaves.update(_leaves(item, f"{prefix}.{key}"))
        return leaves
    if isinstance(value, list) and value and all(isinstance(item, Mapping) for item in value):
        leaves = {}
        for index, item in enumerate(value):
            leaves.update(_leaves(item, f"{prefix}[{index}]"))
        return leaves
    return {prefix: value}


def _circular_difference(a: np.ndarray, b: np.ndarray, period: float) -> np.ndarray:
    difference = np.mod(np.asarray(a, float) - np.asarray(b, float) + period / 2.0, period) - period / 2.0
    return np.abs(difference)


def circular_matching_distance(reference: Sequence[float], candidate: Sequence[float]) -> float:
    """Largest distance on the unit circle under the best cyclic matching of two sets of centres."""
    a = np.sort(np.mod(np.asarray(reference, float), 1.0))
    b = np.sort(np.mod(np.asarray(candidate, float), 1.0))
    if a.shape != b.shape:
        return math.inf
    if a.size == 0:
        return 0.0
    return float(min(_circular_difference(a, np.roll(b, shift), 1.0).max() for shift in range(len(b))))


def _polylines(segments: Any) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (N, 2) and straight pieces (M, 2, 2) of a list of contour polylines."""
    vertices: list[np.ndarray] = []
    pieces: list[np.ndarray] = []
    for segment in segments or []:
        points = np.asarray(segment, dtype=float).reshape(-1, 2)
        if not len(points):
            continue
        vertices.append(points)
        if len(points) == 1:
            pieces.append(np.stack([points, points], axis=1))
        else:
            pieces.append(np.stack([points[:-1], points[1:]], axis=1))
    if not vertices:
        return np.zeros((0, 2)), np.zeros((0, 2, 2))
    return np.concatenate(vertices), np.concatenate(pieces)


def _directed_distance(points: np.ndarray, pieces: np.ndarray) -> float:
    """Largest distance from a point of ``points`` to the nearest straight piece."""
    if not len(points):
        return 0.0
    if not len(pieces):
        return math.inf
    start = pieces[:, 0]
    along = pieces[:, 1] - pieces[:, 0]
    length2 = np.einsum("md,md->m", along, along)
    safe = np.where(length2 > 0, length2, 1.0)
    block = max(1, DISTANCE_BLOCK // len(pieces))
    worst = 0.0
    for first in range(0, len(points), block):
        chunk = points[first : first + block][:, None, :]
        offset = chunk - start[None]
        t = np.clip(np.einsum("nmd,md->nm", offset, along) / safe, 0.0, 1.0)
        t = np.where(length2[None] > 0, t, 0.0)
        nearest = np.sqrt(((offset - t[..., None] * along[None]) ** 2).sum(-1)).min(axis=1)
        worst = max(worst, float(nearest.max()))
    return worst


def contour_hausdorff(reference: Any, candidate: Any) -> float:
    """Symmetric Hausdorff distance between two sets of polylines (vertex to nearest piece)."""
    points_a, pieces_a = _polylines(reference)
    points_b, pieces_b = _polylines(candidate)
    if not len(points_a) and not len(points_b):
        return 0.0
    return max(_directed_distance(points_a, pieces_b), _directed_distance(points_b, pieces_a))


def dos_flip_budget(n_bands: int, k_points: int, step: float, tolerance: float) -> float:
    """L1 budget (states) of a DOS whose eigenvalues may cross bin edges by up to ``tolerance``.

    Eigenvalues within ``tolerance`` of one of the bin edges, ``step`` apart, are
    expected for ``n_bands * k_points * 2 tolerance / step`` of the mesh levels; each
    crossing moves 1/k_points state between neighbouring bins, which the
    normalized smoothing kernel cannot enlarge in L1. One crossing is always allowed.
    """
    expected = n_bands * k_points * min(1.0, 2.0 * tolerance / step)
    return 2.0 * (expected + 1.0) / max(k_points, 1)


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


class _Comparison:
    def __init__(self, family: str, reference: RunDirectory, candidate: RunDirectory):
        self.family = family
        self.reference = reference
        self.candidate = candidate
        self.preconditions: list[dict[str, Any]] = []
        self.quantities: list[dict[str, Any]] = []
        self.flagged: dict[str, Any] = {}
        self.fingerprint = False
        self.semantics, self.producer = semantics_of(reference.document)
        self.candidate_semantics, _ = semantics_of(candidate.document)
        # Conversions apply only between different semantics.
        self.rules = self.semantics if self.semantics != self.candidate_semantics else CURRENT_SEMANTICS
        self.berry_sign = 1.0
        self.weyl = 0.0
        self.eigenvalue_tol = EIGENVALUE_FLOOR_EV

    # ----------------------------------------------------------- records
    def precondition(self, name: str, ok: bool, detail: str) -> bool:
        self.preconditions.append({"name": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    def quantity(
        self,
        name: str,
        kind: str,
        metric: float | None = None,
        tolerance: float | None = None,
        detail: str = "",
    ) -> None:
        self.quantities.append(
            {"name": name, "class": kind, "metric": metric, "tolerance": tolerance, "detail": detail}
        )

    def converted(self, kind: str) -> str:
        """``agree`` becomes ``agree_after_convention`` when a convention was applied."""
        return AFTER_CONVENTION if kind == AGREE else kind

    def bounded(
        self,
        name: str,
        metric: float | None,
        tolerance: float,
        detail: str = "",
        *,
        convention: bool = False,
    ) -> bool:
        """A quantity that agrees when ``metric <= tolerance``; a missing metric is a shape mismatch."""
        if metric is None:
            self.quantity(name, UNEXPLAINED, None, tolerance, detail or "shapes or null entries differ")
            return False
        kind = AGREE if metric <= tolerance else UNEXPLAINED
        if convention:
            kind = self.converted(kind)
        self.quantity(name, kind, metric, tolerance, detail)
        return kind != UNEXPLAINED

    def exact(
        self, name: str, reference: Mapping[str, Any], candidate: Mapping[str, Any], keys: Sequence[str]
    ) -> bool:
        """Every key equal; floats to 1e-12 relative."""
        differing = []
        for key in keys:
            a, b = reference.get(key), candidate.get(key)
            if a == b:
                continue
            if _has_float(a) or _has_float(b):
                relative = _relative_difference(a, b)
                if relative is not None and relative <= PARAMETER_RELATIVE:
                    continue
            differing.append(f"{key}: {a!r} vs {b!r}")
        kind = AGREE if not differing else UNEXPLAINED
        detail = "; ".join(differing) if differing else f"{len(keys)} values equal"
        self.quantity(name, kind, float(len(differing)), 0.0, _shorten(detail))
        return not differing

    # ----------------------------------------------------- preconditions
    def check_documents(self) -> bool:
        problems = self.reference.problems + self.candidate.problems
        return self.precondition(
            "documents", not problems, "; ".join(problems) or "both results documents read"
        )

    def check_family(self) -> bool:
        steps: dict[str, str | None] = {}
        for run in (self.reference, self.candidate):
            step = run.document.get("step_type")
            try:
                steps[run.role] = step_family(step)
            except ValueError:
                steps[run.role] = None
        ok = all(family == self.family for family in steps.values())
        detail = (
            f"reference {self.reference.document.get('step_type')}, candidate "
            f"{self.candidate.document.get('step_type')}: the {self.family} step"
        )
        if not ok:
            detail += " is not both runs' step"
        return self.precondition("step", ok, detail)

    def check_source_wannier_job(self) -> bool:
        names = []
        for run in (self.reference, self.candidate):
            name = run.block("model").get("source_wannier_job_id")
            if name is None and run.params is not None:
                name = run.params.get("source_wannier_job_id")
            names.append(name)
        if None in names:
            return self.precondition("source_wannier_job", True, "not named by both runs")
        ok = names[0] == names[1]
        return self.precondition("source_wannier_job", ok, f"{names[0]} vs {names[1]}")

    def check_inputs(self) -> bool:
        inputs = []
        for run in (self.reference, self.candidate):
            recorded = (run.document.get("provenance") or {}).get("inputs")
            inputs.append(recorded if isinstance(recorded, dict) and recorded else None)
        reference, candidate = inputs
        if reference is not None and candidate is not None:
            common = sorted(set(reference) & set(candidate))
            if common:
                differing = []
                for name in common:
                    a = (reference.get(name) or {}).get("sha256")
                    b = (candidate.get(name) or {}).get("sha256")
                    if not a or a != b:
                        differing.append(name)
                one_sided = sorted(set(reference) ^ set(candidate))
                detail = (
                    f"SHA-256 differs for {', '.join(differing)}"
                    if differing
                    else f"SHA-256 equal for {len(common)} files ({', '.join(common)})"
                )
                if one_sided:
                    detail += f"; recorded by one run only: {', '.join(one_sided)}"
                return self.precondition("inputs", not differing, detail)
        mismatches = self.fingerprint_mismatches()
        self.fingerprint = not mismatches
        if reference is None:
            why = "the reference recorded no inputs"
        elif candidate is None:
            why = "the candidate recorded no inputs"
        else:
            why = "the runs recorded no common input"
        if mismatches:
            detail = f"{why}; the model fingerprint differs: " + "; ".join(mismatches)
        else:
            detail = f"{why}; the model fingerprint matches"
        return self.precondition("inputs", not mismatches, _shorten(detail))

    def fingerprint_mismatches(self) -> list[str]:
        """Where the two models' fingerprints differ (empty when they match)."""
        model_a, model_b = self.reference.block("model"), self.candidate.block("model")
        mismatches = [
            f"{key}: {model_a.get(key)!r} vs {model_b.get(key)!r}"
            for key in FINGERPRINT_EXACT_KEYS
            if model_a.get(key) is None or model_a.get(key) != model_b.get(key)
        ]
        lattice = _max_difference(model_a.get("lattice"), model_b.get("lattice"))
        if lattice is None or lattice > FINGERPRINT_LATTICE_A:
            mismatches.append(f"lattice differs by {lattice} Å")
        ws_a = model_a.get("ws_correction") or {}
        ws_b = model_b.get("ws_correction") or {}
        if ws_a.get("mode") is None or ws_a.get("mode") != ws_b.get("mode"):
            mismatches.append(f"Wigner-Seitz mode {ws_a.get('mode')!r} vs {ws_b.get('mode')!r}")
        scores_a, scores_b = (
            _leaves(ws_a.get("scores") or {}, "scores"),
            _leaves(ws_b.get("scores") or {}, "scores"),
        )
        if set(scores_a) != set(scores_b):
            mismatches.append("Wigner-Seitz scores cover different modes")
        else:
            for key, value in scores_a.items():
                difference = _max_difference(value, scores_b[key])
                if difference is None or difference > FINGERPRINT_MEV:
                    mismatches.append(f"Wigner-Seitz {key} differs by {difference} meV")
        band_a = self.reference.block("validation").get("band_dat") or {}
        band_b = self.candidate.block("validation").get("band_dat") or {}
        if bool(band_a.get("available")) != bool(band_b.get("available")):
            mismatches.append("_band.dat validated by one run only")
        elif band_a.get("available"):
            for key in BAND_DAT_DEVIATIONS:
                difference = _max_difference(band_a.get(key), band_b.get(key))
                if difference is None or difference > FINGERPRINT_MEV:
                    mismatches.append(f"_band.dat {key} differs by {difference} meV")
        return mismatches

    def effective_params(self, run: RunDirectory) -> tuple[dict[str, Any], str | None]:
        """The step's settings with the driver defaults, and why the file is not a valid QUEER file."""
        assert run.params is not None
        step = f"QUEER_{self.family}"
        try:
            return dict(translate_pythtb(run.params, step).values), None
        except RunnerError as error:
            values = dict(DEFAULT_PARAMS)
            values.update({key: value for key, value in run.params.items() if value is not None})
            return values, str(error)

    def check_parameters(self) -> bool:
        reference, candidate = self.reference, self.candidate
        if reference.params is None or candidate.params is None:
            digests = [
                (run.document.get("provenance") or {}).get("params_sha256") for run in (reference, candidate)
            ]
            if digests[0] and digests[0] == digests[1]:
                return self.precondition("parameters", True, "identical parameter files (params_sha256)")
            missing = [run.role for run in (reference, candidate) if run.params is None]
            return self.precondition(
                "parameters", True, f"not compared: no parameter file for {' and '.join(missing)}"
            )
        values_a, problem_a = self.effective_params(reference)
        values_b, problem_b = self.effective_params(candidate)
        differing = []
        for key in step_keys(f"QUEER_{self.family}"):
            if key in EXECUTION_PARAMS:
                continue
            a, b = values_a.get(key), values_b.get(key)
            if key == "structure":
                a, b = _atom_set(a), _atom_set(b)
            if a == b:
                continue
            if (_has_float(a) or _has_float(b) or isinstance(a, float) or isinstance(b, float)) and (
                (relative := _relative_difference(a, b)) is not None and relative <= PARAMETER_RELATIVE
            ):
                continue
            differing.append(key)
        notes = [
            f"{run.role}: {problem}"
            for run, problem in ((reference, problem_a), (candidate, problem_b))
            if problem
        ]
        detail = (
            f"differing: {', '.join(differing)}" if differing else "the step's effective parameters are equal"
        )
        if notes:
            detail += "; " + "; ".join(notes)
        return self.precondition("parameters", not differing, _shorten(detail))

    def check_fermi_energy(self) -> bool:
        a = self.reference.block("model").get("fermi_energy")
        b = self.candidate.block("model").get("fermi_energy")
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return self.precondition("fermi_energy", False, f"E_F {a!r} vs {b!r}")
        difference = abs(float(a) - float(b))
        ok = difference <= FERMI_ENERGY_RELATIVE * max(1.0, abs(float(a)))
        return self.precondition(
            "fermi_energy", ok, f"E_F {a!r} vs {b!r} eV (difference {difference:.3g} eV)"
        )

    def derived_decisions(self, run: RunDirectory) -> dict[str, Any]:
        decisions: dict[str, Any] = {"ws_mode": (run.block("model").get("ws_correction") or {}).get("mode")}
        if self.family == "BANDS":
            electronic = run.block("electronic")
            decisions.update(n_occ=electronic.get("n_occ"), metal=electronic.get("metal"))
        elif self.family == "TOPOLOGY":
            topology = run.block("topology")
            occupation = topology.get("occupation") or {}
            decisions.update({key: occupation.get(key) for key in OCCUPATION_DERIVED})
            decisions["applicable"] = topology.get("applicable")
            decisions["z2_computed"] = (topology.get("time_reversal") or {}).get("z2_computed")
            decisions["planes"] = {
                _plane_key(plane): plane.get("transverse_points") for plane in topology.get("planes") or []
            }
        elif self.family == "BERRY":
            berry = run.block("berry")
            occupation = berry.get("occupation_info") or {}
            decisions.update({key: occupation.get(key) for key in OCCUPATION_DERIVED})
            decisions["occupation_mode"] = berry.get("occupation")
        elif self.family == "SLAB":
            slab = run.block("slab")
            decisions.update({key: slab.get(key) for key in ("kind", "cells", "finite_axis")})
        return decisions

    def check_derived_decisions(self) -> bool:
        a, b = self.derived_decisions(self.reference), self.derived_decisions(self.candidate)
        differing = [f"{key}: {a.get(key)!r} vs {b.get(key)!r}" for key in a if a.get(key) != b.get(key)]
        detail = (
            "; ".join(differing)
            if differing
            else ", ".join(f"{key}={_brief(value)}" for key, value in a.items())
        )
        return self.precondition("derived_decisions", not differing, _shorten(detail))

    def check_candidate_semantics(self) -> bool:
        if self.candidate_semantics.name in (CURRENT_SEMANTICS.name, self.semantics.name):
            return True
        return self.precondition(
            "candidate_semantics",
            False,
            f"the candidate was written by {self.candidate_semantics.name}; "
            "compare against current semantics",
        )

    # ------------------------------------------------------- tolerances
    def set_tolerances(self) -> None:
        for run in (self.candidate, self.reference):
            bound = run.block("model").get("hermiticity_weyl_bound")
            if isinstance(bound, (int, float)) and not isinstance(bound, bool):
                self.weyl = float(bound)
                break
        magnitude = _max_magnitude(
            self.reference.block("bands").get("eigenvalues_eV"),
            self.candidate.block("bands").get("eigenvalues_eV"),
            self.reference.block("model").get("fermi_energy"),
        )
        self.eigenvalue_tol = eigenvalue_tolerance(self.weyl, magnitude)
        berry = self.reference.block("berry")
        if (
            self.family == "BERRY"
            and berry.get("occupation") == "fermi"
            and "conventions" not in berry
            and "conventions" in self.candidate.block("berry")
        ):
            self.berry_sign = -1.0

    # ------------------------------------------------------- common blocks
    def compare_block(self, name: str, skip: frozenset[str] = frozenset()) -> None:
        """Float leaves to 1e-6 relative (the worst reported), other leaves exactly."""
        reference = {key: value for key, value in self.reference.block(name).items() if key not in skip}
        candidate = {key: value for key, value in self.candidate.block(name).items() if key not in skip}
        if not reference and not candidate:
            self.quantity(name, NOT_APPLICABLE, detail="absent from both documents")
            return
        leaves_a, leaves_b = _leaves(reference, name), _leaves(candidate, name)
        worst, worst_path = 0.0, None
        unequal = []
        for path in sorted(set(leaves_a) & set(leaves_b)):
            a, b = leaves_a[path], leaves_b[path]
            if path in ORDER_FREE_LEAVES and isinstance(a, list) and isinstance(b, list):
                a, b = sorted(map(str, a)), sorted(map(str, b))
            if _has_float(a) or _has_float(b):
                relative = _relative_difference(a, b)
                if relative is None:
                    unequal.append(path)
                elif relative > worst:
                    worst, worst_path = relative, path
            elif a != b:
                unequal.append(path)
        one_sided = sorted(set(leaves_a) ^ set(leaves_b))
        kind = AGREE if worst <= BLOCK_RELATIVE and not unequal else UNEXPLAINED
        detail = f"worst float leaf {worst_path}" if worst_path else "every float leaf equal"
        if unequal:
            detail += f"; unequal: {', '.join(unequal)}"
        if one_sided:
            detail += f"; recorded by one document only: {', '.join(one_sided)}"
        if skip:
            detail += f"; not compared: {', '.join(sorted(skip))}"
        self.quantity(name, kind, worst, BLOCK_RELATIVE, _shorten(detail))

    def compare_warnings(self) -> None:
        def codes(run: RunDirectory) -> tuple[list[str], list[str]]:
            messages = [str(message) for message in run.document.get("warnings") or []]
            recorded = run.document.get("warning_codes")
            if isinstance(recorded, list) and len(recorded) == len(messages):
                return [str(code) for code in recorded], messages
            return [warning_code(message) for message in messages], messages

        codes_a, messages_a = codes(self.reference)
        codes_b, messages_b = codes(self.candidate)
        kept_a = {code for code in codes_a if code not in ENVIRONMENT_WARNINGS}
        kept_b = {code for code in codes_b if code not in ENVIRONMENT_WARNINGS}
        texts_a = {m for c, m in zip(codes_a, messages_a) if c not in ENVIRONMENT_WARNINGS}
        texts_b = {m for c, m in zip(codes_b, messages_b) if c not in ENVIRONMENT_WARNINGS}
        ignored = sum(code in ENVIRONMENT_WARNINGS for code in codes_a + codes_b)
        note = f"; {ignored} plot warnings not compared" if ignored else ""
        coded = all(
            isinstance(run.document.get("warning_codes"), list) for run in (self.reference, self.candidate)
        )
        if kept_a != kept_b:
            detail = f"reference {sorted(kept_a)} vs candidate {sorted(kept_b)}"
            self.quantity("warnings", UNEXPLAINED, None, None, _shorten(detail + note))
        elif coded or texts_a == texts_b:
            self.quantity("warnings", AGREE, None, None, f"codes {sorted(kept_a)}{note}")
        else:
            detail = f"same codes {sorted(kept_a)}, reworded messages (known renames){note}"
            self.quantity("warnings", DOCUMENTED, None, None, detail)

    def compare_outputs(self) -> None:
        def data(run: RunDirectory) -> set[str]:
            return {
                str(entry.get("path"))
                for entry in run.document.get("outputs") or []
                if isinstance(entry, dict) and entry.get("kind") == "data"
            }

        a, b = data(self.reference), data(self.candidate)
        kind = AGREE if a == b else UNEXPLAINED
        detail = f"data files {sorted(a)}" if a == b else f"reference {sorted(a)} vs candidate {sorted(b)}"
        self.quantity("outputs", kind, None, None, detail)

    def compare_common(self) -> None:
        self.compare_block("model", PRODUCER_MODEL_KEYS)
        self.compare_block(
            "validation", frozenset() if self.rules.localization else frozenset({"localization"})
        )
        self.compare_warnings()
        self.compare_outputs()

    # --------------------------------------------- occupation and electronic
    def compare_occupation(
        self, name: str, reference: Mapping[str, Any], candidate: Mapping[str, Any]
    ) -> None:
        """An ``electronic``-like block: counts and flags exactly, energies to 2 τ_E, edge k-point ties."""
        if not reference and not candidate:
            self.quantity(name, NOT_APPLICABLE, detail="absent from both documents")
            return
        handled = set(OCCUPATION_DERIVED) | set(OCCUPATION_ENERGIES_EV) | set(OCCUPATION_ENERGIES_MEV)
        handled |= set(OCCUPATION_KPOINTS) | {"dos_at_fermi_per_eV"}
        keys = sorted((set(reference) | set(candidate)) - handled)
        self.exact(f"{name}.counts", reference, candidate, keys)
        edge_tol = EDGE_FACTOR * self.eigenvalue_tol
        worst = 0.0
        unequal = []
        for key, scale in [(key, 1.0) for key in OCCUPATION_ENERGIES_EV] + [
            (key, 1000.0) for key in OCCUPATION_ENERGIES_MEV
        ]:
            a, b = reference.get(key), candidate.get(key)
            if a is None and b is None:
                continue
            difference = _max_difference(a, b)
            if difference is None:
                unequal.append(key)
            else:
                worst = max(worst, difference / scale)
        if unequal:
            self.quantity(
                f"{name}.edges", UNEXPLAINED, worst, edge_tol, f"recorded by one run only: {unequal}"
            )
        else:
            self.bounded(f"{name}.edges", worst, edge_tol, "gap, band edges and isolation (eV)")
        moved = []
        for key, energy in OCCUPATION_KPOINTS.items():
            a, b = reference.get(key), candidate.get(key)
            difference = _max_difference(a, b) if a is not None or b is not None else 0.0
            if difference is None or difference > COORDINATE_TOL:
                moved.append((key, energy))
        if not moved:
            self.quantity(f"{name}.edge_kpoints", AGREE, 0.0, COORDINATE_TOL, "band-edge k-points equal")
            return
        tied = True
        for _, energy in moved:
            difference = _max_difference(reference.get(energy), candidate.get(energy))
            tied = tied and difference is not None and difference <= edge_tol
        detail = ", ".join(f"{key} {reference.get(key)} vs {candidate.get(key)}" for key, _ in moved)
        if tied:
            self.quantity(
                f"{name}.edge_kpoints",
                DOCUMENTED,
                None,
                COORDINATE_TOL,
                _shorten(f"an extremum tied within 2 τ_E at another k-point: {detail}"),
            )
        else:
            self.quantity(f"{name}.edge_kpoints", UNEXPLAINED, None, COORDINATE_TOL, _shorten(detail))

    # ---------------------------------------------------------------- BANDS
    def compare_bands(self) -> None:
        ref_bands, cand_bands = self.reference.block("bands"), self.candidate.block("bands")
        self.bounded(
            "bands.eigenvalues_eV",
            _max_difference(ref_bands.get("eigenvalues_eV"), cand_bands.get("eigenvalues_eV")),
            self.eigenvalue_tol,
            f"τ_E = max(1e-8, δ_W={self.weyl:.3g}) + 1e-13 max|E|",
        )
        self.bounded(
            "bands.kpoints_frac",
            _max_difference(ref_bands.get("kpoints_frac"), cand_bands.get("kpoints_frac")),
            COORDINATE_TOL,
        )
        self.exact(
            "bands.path",
            ref_bands,
            cand_bands,
            ["tick_indices", "tick_labels", "break_indices", "source", "fermi_energy"],
        )
        electronic_a, electronic_b = self.reference.block("electronic"), self.candidate.block("electronic")
        self.compare_occupation("electronic", electronic_a, electronic_b)
        dos_a, dos_b = self.reference.out("dos.json"), self.candidate.out("dos.json")
        self.compare_dos_at_fermi(electronic_a, electronic_b, dos_a, dos_b)
        self.compare_dos_block()
        self.compare_dos_file(dos_a, dos_b)
        self.compare_fermi_surface()

    def compare_dos_at_fermi(self, electronic_a, electronic_b, dos_a, dos_b) -> None:
        name = "electronic.dos_at_fermi_per_eV"
        reported_a, reported_b = (
            electronic_a.get("dos_at_fermi_per_eV"),
            electronic_b.get("dos_at_fermi_per_eV"),
        )
        if reported_a is None and reported_b is None:
            self.quantity(name, NOT_APPLICABLE, detail="not reported")
            return
        scale = _max_magnitude((dos_a or {}).get("dos_per_eV"), (dos_b or {}).get("dos_per_eV"))
        tolerance = DOS_RELATIVE * scale
        rule = self.rules.dos_at_fermi_rule
        if rule == CURRENT_SEMANTICS.dos_at_fermi_rule:
            self.bounded(name, _max_difference(reported_a, reported_b), tolerance, "the reported values")
            return
        if not isinstance(dos_b, dict):
            self.quantity(name, UNEXPLAINED, None, tolerance, "the candidate wrote no Out/dos.json")
            return
        recomputed = dos_at_fermi(dos_b.get("energy_eV"), dos_b.get("dos_per_eV"), rule)
        self.bounded(
            name,
            _max_difference(reported_a, recomputed),
            tolerance,
            f"the candidate's DOS(E_F) recomputed with the reference rule {rule} ({recomputed!r}; "
            f"reported {reported_b!r})",
            convention=True,
        )

    def compare_dos_block(self) -> None:
        dos_a, dos_b = self.reference.block("dos"), self.candidate.block("dos")
        if not dos_a and not dos_b:
            self.quantity("dos", NOT_APPLICABLE, detail="absent from both documents")
            return
        # The species follow the structure's atom order: compared as a set.
        self.exact(
            "dos.block",
            {**dos_a, "species": sorted(map(str, dos_a.get("species") or []))},
            {**dos_b, "species": sorted(map(str, dos_b.get("species") or []))},
            ["k_points", "sigma_eV", "species", "file"],
        )
        mesh_a, mesh_b = dos_a.get("mesh"), dos_b.get("mesh")
        axes = dos_b.get("mesh_axes")
        if mesh_a == mesh_b and dos_a.get("mesh_axes") == axes:
            self.quantity("dos.mesh", AGREE, detail=f"{mesh_b} along axes {axes}")
        elif (
            self.rules.dos_mesh_on_every_axis
            and isinstance(mesh_a, list)
            and len(mesh_a) == 3
            and "mesh_axes" not in dos_a
            and isinstance(axes, list)
            and [mesh_a[int(axis)] for axis in axes] == mesh_b
        ):
            self.quantity(
                "dos.mesh",
                AFTER_CONVENTION,
                detail=f"reference {mesh_a} on every axis = {mesh_b} along {axes}",
            )
        else:
            self.quantity(
                "dos.mesh", UNEXPLAINED, detail=f"reference {mesh_a} vs candidate {mesh_b} ({axes})"
            )
        integral_a, integral_b = dos_a.get("integral"), dos_b.get("integral")
        self.bounded(
            "dos.integral",
            _max_difference(integral_a, integral_b),
            DOS_RELATIVE * max(1.0, _max_magnitude(integral_a)),
        )

    def compare_dos_file(self, dos_a: Any, dos_b: Any) -> None:
        if not isinstance(dos_a, dict) or not isinstance(dos_b, dict):
            present = [
                role
                for role, value in (("reference", dos_a), ("candidate", dos_b))
                if isinstance(value, dict)
            ]
            if not present and not self.reference.block("dos"):
                self.quantity("dos.file", NOT_APPLICABLE, detail="no Out/dos.json")
            else:
                self.quantity(
                    "dos.file", UNEXPLAINED, detail=f"Out/dos.json read for {present or 'neither'} only"
                )
            return
        grid_a, grid_b = _array(dos_a.get("energy_eV")), _array(dos_b.get("energy_eV"))
        if grid_a is None or grid_b is None or grid_a.shape != grid_b.shape:
            self.quantity("dos.energy_grid", UNEXPLAINED, detail="the energy grids differ in length")
            return
        self.bounded(
            "dos.energy_grid", _max_difference(grid_a, grid_b), self.eigenvalue_tol, "bin centres (eV)"
        )
        step = float(np.median(np.diff(grid_a))) if len(grid_a) > 1 else 1.0
        n_bands = int(self.reference.block("model").get("num_wann") or 0)
        k_points = int(self.reference.block("dos").get("k_points") or 0)
        budget = dos_flip_budget(n_bands, k_points, step, self.eigenvalue_tol) if self.weyl > 0 else 0.0
        self.compare_density("dos.dos_per_eV", dos_a.get("dos_per_eV"), dos_b.get("dos_per_eV"), step, budget)
        pdos_a, pdos_b = dos_a.get("pdos_per_eV") or {}, dos_b.get("pdos_per_eV") or {}
        if not pdos_a and not pdos_b:
            self.quantity("dos.pdos_per_eV", NOT_APPLICABLE, detail="no PDOS")
            return
        if set(pdos_a) != set(pdos_b):
            self.quantity("dos.pdos_per_eV", UNEXPLAINED, detail=f"species {list(pdos_a)} vs {list(pdos_b)}")
            return
        for species in sorted(pdos_a):
            self.compare_density(f"dos.pdos_per_eV.{species}", pdos_a[species], pdos_b[species], step, budget)

    def compare_density(self, name: str, reference: Any, candidate: Any, step: float, budget: float) -> None:
        """A DOS row to 1e-9 of its largest value; with δ_W > 0, bin-edge crossings within an L1 budget."""
        tolerance = DOS_RELATIVE * _max_magnitude(reference)
        difference = _max_difference(reference, candidate)
        if difference is None or difference <= tolerance or budget <= 0:
            self.bounded(name, difference, tolerance)
            return
        a, b = np.asarray(reference, float), np.asarray(candidate, float)
        moved = float(np.abs(b - a).sum() * step)
        conserved = abs(float((b - a).sum() * step)) <= DOS_RELATIVE * max(1.0, float(np.abs(a).sum() * step))
        if conserved and moved <= budget:
            detail = f"bin-edge crossings: L1 {moved:.3g} states within the budget {budget:.3g} (δ_W > 0)"
            self.quantity(name, DOCUMENTED, difference, tolerance, detail)
        else:
            detail = f"L1 {moved:.3g} states (budget {budget:.3g}), states conserved: {conserved}"
            self.quantity(name, UNEXPLAINED, difference, tolerance, detail)

    def compare_fermi_surface(self) -> None:
        surface_a, surface_b = self.reference.block("fermi_surface"), self.candidate.block("fermi_surface")
        if not surface_a and not surface_b:
            self.quantity(
                "fermi_surface", NOT_APPLICABLE, detail="no Fermi surface (insulator or switched off)"
            )
            return
        self.exact("fermi_surface.plane", surface_a, surface_b, ["fixed_axis", "value"])
        self.exact("fermi_surface.bands_crossing", surface_a, surface_b, ["bands_crossing"])
        if not self.rules.fermi_surface_contours:
            self.quantity(
                "fermi_surface.contours",
                NOT_APPLICABLE,
                detail=f"the reference ({self.semantics.name}) wrote its contours without a plane frame",
            )
            return
        file_a, file_b = self.reference.out("fermi_surface.json"), self.candidate.out("fermi_surface.json")
        if not isinstance(file_a, dict) or not isinstance(file_b, dict):
            self.quantity("fermi_surface.contours", UNEXPLAINED, detail="Out/fermi_surface.json missing")
            return
        self.exact(
            "fermi_surface.frame", file_a, file_b, ["mesh", "k_frac_range", "bands", "axes_cart", "units"]
        )
        lattice = _array(self.reference.block("model").get("lattice"))
        fixed = int(surface_a.get("fixed_axis", 2))
        reciprocal = (
            np.linalg.inv(lattice).T if lattice is not None and lattice.shape == (3, 3) else np.eye(3)
        )
        in_plane = [axis for axis in range(3) if axis != fixed]
        length = float(max(np.linalg.norm(reciprocal[axis]) for axis in in_plane))
        tolerance = max(FERMI_SURFACE_RELATIVE * length, FERMI_SURFACE_FLOOR)
        contours_a, contours_b = file_a.get("contours_cart") or {}, file_b.get("contours_cart") or {}
        if set(contours_a) != set(contours_b):
            empty = [
                role for role, value in (("reference", contours_a), ("candidate", contours_b)) if not value
            ]
            kind = DOCUMENTED if len(empty) == 1 else UNEXPLAINED
            detail = f"bands {sorted(contours_a)} vs {sorted(contours_b)}"
            if kind == DOCUMENTED:
                detail += f"; the {empty[0]} drew no contours (no Matplotlib)"
            self.quantity("fermi_surface.contours", kind, None, tolerance, detail)
            return
        worst = max(
            (contour_hausdorff(contours_a[band], contours_b[band]) for band in contours_a), default=0.0
        )
        self.bounded(
            "fermi_surface.contours",
            worst,
            tolerance,
            f"symmetric Hausdorff distance over {len(contours_a)} bands (1/Å without 2 pi)",
        )

    # ------------------------------------------------------------- TOPOLOGY
    def compare_topology(self) -> None:
        topology_a, topology_b = self.reference.block("topology"), self.candidate.block("topology")
        self.compare_occupation(
            "topology.occupation", topology_a.get("occupation") or {}, topology_b.get("occupation") or {}
        )
        if not topology_a.get("applicable"):
            self.quantity(
                "topology.planes", NOT_APPLICABLE, detail=str(topology_a.get("reason") or "not applicable")
            )
            return
        reversal_a, reversal_b = topology_a.get("time_reversal") or {}, topology_b.get("time_reversal") or {}
        self.exact(
            "topology.time_reversal",
            reversal_a,
            reversal_b,
            ["spinors", "kramers_degenerate_at_trims", "z2_computed"],
        )
        split_a, split_b = reversal_a.get("max_kramers_split_meV"), reversal_b.get("max_kramers_split_meV")
        self.bounded(
            "topology.time_reversal.max_kramers_split_meV",
            0.0 if split_a is None and split_b is None else _max_difference(split_a, split_b),
            EDGE_FACTOR * self.eigenvalue_tol * 1000.0,
        )
        occupied = int((topology_a.get("occupation") or {}).get("n_occ") or 1)
        planes_b = {_plane_key(plane): plane for plane in topology_b.get("planes") or []}
        for plane_a in topology_a.get("planes") or []:
            key = _plane_key(plane_a)
            self.compare_plane(f"topology.plane[k{key[0] + 1}={key[1]:g}]", plane_a, planes_b[key], occupied)
        indices_a, indices_b = topology_a.get("z2_indices"), topology_b.get("z2_indices")
        if indices_a is None and indices_b is None:
            self.quantity("topology.z2_indices", NOT_APPLICABLE, detail="Z2 not computed")
        else:
            kind = AGREE if indices_a == indices_b else UNEXPLAINED
            self.quantity("topology.z2_indices", kind, detail=f"{indices_a} vs {indices_b}")

    def compare_plane(
        self, name: str, plane_a: Mapping[str, Any], plane_b: Mapping[str, Any], occupied: int
    ) -> None:
        keys = ["loop_axis", "transverse_axis", "loop_points", "chern", "converged", "file"]
        keys += [key for key in ("z2", "z2_ambiguous", "z2_crossings") if key in plane_a or key in plane_b]
        self.exact(f"{name}.invariants", plane_a, plane_b, keys)
        points = int(plane_a.get("transverse_points") or 1)
        phase_tol = 2.0 * math.pi * occupied * WCC_TOL
        self.bounded(
            f"{name}.chern_raw",
            _max_difference(plane_a.get("chern_raw"), plane_b.get("chern_raw")),
            2.0 * points * occupied * WCC_TOL,
            "a sum of transverse_points phase steps, each within 2 n_occ 2 pi 1e-9",
        )
        self.bounded(
            f"{name}.max_phase_step",
            _max_difference(plane_a.get("max_phase_step"), plane_b.get("max_phase_step")),
            2.0 * phase_tol,
        )
        stem = Path(str(plane_a.get("file") or "")).name
        flow_a, flow_b = self.reference.out(stem), self.candidate.out(stem)
        if not isinstance(flow_a, dict) or not isinstance(flow_b, dict):
            self.quantity(f"{name}.wcc", UNEXPLAINED, detail=f"Out/{stem} missing")
            return
        self.bounded(f"{name}.t", _max_difference(flow_a.get("t"), flow_b.get("t")), COORDINATE_TOL)
        wcc_a, wcc_b = flow_a.get("wcc") or [], flow_b.get("wcc") or []
        if len(wcc_a) != len(wcc_b):
            self.quantity(f"{name}.wcc", UNEXPLAINED, detail=f"{len(wcc_a)} vs {len(wcc_b)} loops")
            return
        worst = max((circular_matching_distance(a, b) for a, b in zip(wcc_a, wcc_b)), default=0.0)
        self.bounded(f"{name}.wcc", worst, WCC_TOL, "optimal circular matching per loop")
        phases_a, phases_b = _array(flow_a.get("berry_phase")), _array(flow_b.get("berry_phase"))
        if phases_a is None or phases_b is None or phases_a.shape != phases_b.shape:
            self.quantity(f"{name}.berry_phase", UNEXPLAINED, detail="the phase lists differ")
        else:
            difference = float(_circular_difference(phases_a, phases_b, 2.0 * math.pi).max(initial=0.0))
            self.bounded(f"{name}.berry_phase", difference, phase_tol, "modulo 2 pi")
        mids_a, mids_b = flow_a.get("gap_midpoints"), flow_b.get("gap_midpoints")
        if mids_a is None and mids_b is None:
            return
        if mids_a is None or mids_b is None or len(mids_a) != len(mids_b):
            self.quantity(f"{name}.gap_midpoints", UNEXPLAINED, detail="recorded by one run only")
            return
        difference = float(
            _circular_difference(np.asarray(mids_a, float), np.asarray(mids_b, float), 1.0).max(initial=0.0)
        )
        if difference <= WCC_TOL:
            self.quantity(f"{name}.gap_midpoints", AGREE, difference, WCC_TOL)
        elif plane_a.get("z2") == plane_b.get("z2") and plane_a.get("z2_crossings") == plane_b.get(
            "z2_crossings"
        ):
            self.quantity(
                f"{name}.gap_midpoints",
                DOCUMENTED,
                difference,
                WCC_TOL,
                "the largest WCC gap is tied: its midpoint moved, Z2 and the crossings agree",
            )
        else:
            self.quantity(f"{name}.gap_midpoints", UNEXPLAINED, difference, WCC_TOL)

    # ---------------------------------------------------------------- BERRY
    def compare_berry(self) -> None:
        berry_a, berry_b = self.reference.block("berry"), self.candidate.block("berry")
        sign = self.berry_sign
        convention = sign < 0
        self.exact(
            "berry.plane", berry_a, berry_b, ["fixed_axis", "value", "plane_axes", "mesh", "path_file"]
        )
        self.compare_occupation(
            "berry.occupation_info",
            berry_a.get("occupation_info") or {},
            berry_b.get("occupation_info") or {},
        )
        conventions_a, conventions_b = berry_a.get("conventions"), berry_b.get("conventions")
        if conventions_a == conventions_b:
            self.quantity("berry.conventions", AGREE, detail="the same sign conventions")
        elif conventions_a is None:
            detail = "the reference predates the recorded conventions"
            detail += ": its fermi-mode signs are taken as the pre-fix ones (x -1)" if convention else ""
            self.quantity("berry.conventions", AFTER_CONVENTION, detail=detail)
        else:
            self.quantity(
                "berry.conventions", UNEXPLAINED, detail=_shorten(f"{conventions_a} vs {conventions_b}")
            )
        mesh = int(berry_a.get("mesh") or 0)
        stem = plane_file_stem("berry", int(berry_a.get("fixed_axis", 2)), float(berry_a.get("value", 0.0)))
        map_a, map_b = self.reference.out(f"{stem}.json"), self.candidate.out(f"{stem}.json")
        if not isinstance(map_a, dict) or not isinstance(map_b, dict):
            self.quantity("berry.maps", UNEXPLAINED, detail=f"Out/{stem}.json missing")
            return
        self.bounded(
            "berry.k_frac", _max_difference(map_a.get("k_frac"), map_b.get("k_frac")), COORDINATE_TOL
        )
        reduced_a = _array(map_a.get("omega_reduced"))
        reduced_b = _array(map_b.get("omega_reduced"))
        if (
            reduced_a is None
            or reduced_b is None
            or reduced_a.shape != reduced_b.shape
            or reduced_a.size != mesh * mesh
        ):
            self.quantity("berry.omega_reduced", UNEXPLAINED, detail="the maps differ in size")
            return
        self.check_integral_consistency(berry_a, reduced_a, mesh, "reference")
        self.check_integral_consistency(berry_b, reduced_b, mesh, "candidate")
        reduced_a = sign * reduced_a
        if berry_a.get("occupation") == "bands":
            self.compare_bands_mode(berry_a, berry_b, map_a, map_b)
        else:
            self.compare_fermi_mode(berry_a, berry_b, map_a, map_b, reduced_a, reduced_b, mesh, convention)
        self.compare_path(berry_a, berry_b, sign, convention)

    def check_integral_consistency(
        self, berry: Mapping[str, Any], reduced: np.ndarray, mesh: int, role: str
    ) -> None:
        """A document's plane integral is its own map's sum / (n^2 2 pi)."""
        key = "chern_kubo" if berry.get("occupation") == "bands" else "fermi_sea_integral_e2_h"
        recorded = berry.get(key)
        if not isinstance(recorded, (int, float)):
            self.quantity(f"berry.{role}_integral_consistency", UNEXPLAINED, detail=f"no {key}")
            return
        integral = float(reduced.sum() / (mesh * mesh * 2.0 * math.pi))
        rounding = 1e-12 * float(np.abs(reduced).sum() / (mesh * mesh * 2.0 * math.pi))
        self.bounded(
            f"berry.{role}_integral_consistency",
            abs(float(recorded) - integral),
            FERMI_INTEGRAL_TOL + rounding,
            f"{key} against its own map",
        )

    def compare_bands_mode(self, berry_a, berry_b, map_a, map_b) -> None:
        for key in ("omega_reduced", "omega_cart_A2", "metric_trace_A2"):
            a, b = map_a.get(key), map_b.get(key)
            if a is None and b is None:
                continue
            self.bounded(
                f"berry.{key}", _max_difference(a, b), BERRY_MAP_RELATIVE * _max_magnitude(a), "1e-8 max|X|"
            )
        for key in ("chern_kubo", "chern_kubo_integer_distance"):
            self.bounded(f"berry.{key}", _max_difference(berry_a.get(key), berry_b.get(key)), CHERN_KUBO_TOL)
        weight_a, weight_b = berry_a.get("quantum_weight"), berry_b.get("quantum_weight")
        if weight_a is None and weight_b is None:
            self.quantity("berry.quantum_weight", NOT_APPLICABLE, detail="quantum metric off")
        else:
            self.bounded(
                "berry.quantum_weight", _relative_difference(weight_a, weight_b), QUANTUM_WEIGHT_RELATIVE
            )
            self.exact("berry.quantum_weight_bound_ok", berry_a, berry_b, ["quantum_weight_bound_ok"])
        maximum_a, maximum_b = berry_a.get("omega_cart_max_A2"), berry_b.get("omega_cart_max_A2")
        self.bounded(
            "berry.omega_cart_max_A2",
            _max_difference(maximum_a, maximum_b),
            BERRY_MAP_RELATIVE * _max_magnitude(maximum_a),
        )

    def compare_fermi_mode(
        self, berry_a, berry_b, map_a, map_b, reduced_a, reduced_b, mesh, convention
    ) -> None:
        sign = self.berry_sign
        tolerance = BERRY_MAP_RELATIVE * _max_magnitude(reduced_a)
        difference = np.abs(reduced_b - reduced_a)
        flagged = difference > tolerance
        support = np.maximum(np.abs(reduced_a), np.abs(reduced_b)) > tolerance
        share = float((reduced_b - reduced_a)[flagged].sum() / (mesh * mesh * 2.0 * math.pi))
        allowance = float(
            (np.abs(reduced_a) + np.abs(reduced_b))[flagged].sum() / (mesh * mesh * 2.0 * math.pi)
        )
        minority = int(flagged.sum()) * 2 <= max(int(support.sum()), 1)
        self.flagged["berry.omega_reduced"] = {
            "size": int(flagged.sum()),
            "points": int(flagged.size),
            "points_with_curvature": int(support.sum()),
            "indices": [int(index) for index in np.flatnonzero(flagged)[:FLAGGED_LISTED]],
            "integral_share_e2_h": share,
            "integral_allowance_e2_h": allowance,
        }
        for key in ("omega_reduced", "omega_cart_A2"):
            a = sign * np.asarray(map_a.get(key), float)
            b = np.asarray(map_b.get(key), float)
            if a.shape != b.shape:
                self.quantity(f"berry.{key}", UNEXPLAINED, detail="the maps differ in size")
                continue
            tol = BERRY_MAP_RELATIVE * _max_magnitude(a)
            outside = float(np.abs(b - a)[~flagged].max(initial=0.0))
            worst = float(np.abs(b - a).max(initial=0.0))
            if not flagged.any() and worst <= tol:
                self.quantity(
                    f"berry.{key}", self.converted(AGREE) if convention else AGREE, worst, tol, "1e-8 max|X|"
                )
            elif minority and outside <= tol:
                detail = (
                    f"{int(flagged.sum())} flagged points (near-degenerate at E_F or the 1e-6 eV skip) of "
                    f"{flagged.size}; agree elsewhere"
                )
                self.quantity(f"berry.{key}", DOCUMENTED, worst, tol, detail)
            else:
                detail = (
                    f"the maps disagree at {int(flagged.sum())} of {int(support.sum())} points carrying "
                    "curvature: more than a near-degeneracy can explain"
                )
                if np.abs(b + a).max(initial=0.0) <= tol and worst > tol:
                    detail += "; the candidate is the negative of the reference (a sign convention)"
                self.quantity(f"berry.{key}", UNEXPLAINED, worst, tol, detail)
        bound = FERMI_INTEGRAL_TOL + allowance
        for key in ("fermi_sea_integral_e2_h", "sigma_xy_e2_h"):
            a, b = berry_a.get(key), berry_b.get(key)
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                self.quantity(f"berry.{key}", UNEXPLAINED, detail="not recorded by both runs")
                continue
            delta = abs(float(b) - sign * float(a))
            if delta <= FERMI_INTEGRAL_TOL:
                self.quantity(
                    f"berry.{key}", AFTER_CONVENTION if convention else AGREE, delta, FERMI_INTEGRAL_TOL
                )
            elif delta <= bound and minority:
                self.quantity(
                    f"berry.{key}",
                    DOCUMENTED,
                    delta,
                    bound,
                    f"within 1e-8 + sum_F(|Ω_Q| + |Ω_P|)/(n² 2π) over {int(flagged.sum())} flagged points",
                )
            else:
                self.quantity(f"berry.{key}", UNEXPLAINED, delta, bound)
        maximum_a, maximum_b = berry_a.get("omega_cart_max_A2"), berry_b.get("omega_cart_max_A2")
        tol = BERRY_MAP_RELATIVE * _max_magnitude(maximum_a)
        delta_max = _max_difference(maximum_a, maximum_b)
        if delta_max is not None and delta_max > tol and minority and flagged.any():
            self.quantity(
                "berry.omega_cart_max_A2", DOCUMENTED, delta_max, tol, "the maximum lies in the flagged set"
            )
        else:
            self.bounded("berry.omega_cart_max_A2", delta_max, tol)

    def compare_path(self, berry_a, berry_b, sign: float, convention: bool) -> None:
        if not berry_a.get("path_file") and not berry_b.get("path_file"):
            self.quantity("berry.path", NOT_APPLICABLE, detail="path curvature off or skipped")
            return
        path_a, path_b = self.reference.out("berry_path.json"), self.candidate.out("berry_path.json")
        if not isinstance(path_a, dict) or not isinstance(path_b, dict):
            self.quantity("berry.path", UNEXPLAINED, detail="Out/berry_path.json missing")
            return
        self.exact("berry.path.ticks", path_a, path_b, ["tick_indices", "tick_labels"])
        self.bounded(
            "berry.path.k_distance",
            _max_difference(path_a.get("k_distance"), path_b.get("k_distance")),
            COORDINATE_TOL * max(1.0, _max_magnitude(path_a.get("k_distance"))),
        )
        along_a = _array(path_a.get("omega_cart_A2"))
        along_b = _array(path_b.get("omega_cart_A2"))
        if along_a is None or along_b is None:
            self.quantity("berry.path.omega_cart_A2", UNEXPLAINED, detail="no path curvature")
            return
        along_a = sign * along_a
        self.bounded(
            "berry.path.omega_cart_A2",
            _max_difference(along_a, along_b),
            BERRY_MAP_RELATIVE * _max_magnitude(along_a),
            "pointwise, 1e-8 max|Ω|",
            convention=convention,
        )

    # ----------------------------------------------------------------- SLAB
    def compare_slab(self) -> None:
        slab_a, slab_b = self.reference.block("slab"), self.candidate.block("slab")
        self.exact("slab.block", slab_a, slab_b, ["periodic_axes", "states", "window_eV", "edge_cells"])
        kind = str(slab_a.get("kind") or "")
        if kind not in ("ribbon", "slab"):
            self.quantity("slab.levels", UNEXPLAINED, detail=f"unknown slab kind {kind!r}")
            return
        file_a, file_b = self.reference.out(f"{kind}_bands.json"), self.candidate.out(f"{kind}_bands.json")
        if not isinstance(file_a, dict) or not isinstance(file_b, dict):
            self.quantity("slab.levels", UNEXPLAINED, detail=f"Out/{kind}_bands.json missing")
            return
        self.bounded(
            "slab.kpoints_frac",
            _max_difference(file_a.get("kpoints_frac"), file_b.get("kpoints_frac")),
            COORDINATE_TOL,
        )
        self.bounded(
            "slab.k_distance",
            _max_difference(file_a.get("k_distance"), file_b.get("k_distance")),
            COORDINATE_TOL * max(1.0, _max_magnitude(file_a.get("k_distance"))),
        )
        self.exact("slab.ticks", file_a, file_b, ["tick_indices", "tick_labels"])
        window = float(slab_a.get("window_eV") or 0.5)
        levels_a = [np.asarray(row, float) for row in file_a.get("energies_eV") or []]
        levels_b = [np.asarray(row, float) for row in file_b.get("energies_eV") or []]
        magnitude = max(
            _max_magnitude(*levels_a, *levels_b),
            abs(float(self.reference.block("model").get("fermi_energy") or 0)),
        )
        tolerance = eigenvalue_tolerance(self.weyl, magnitude)
        weights = {
            side: (
                [np.asarray(row, float) for row in file_a.get(f"{side}_weight") or []],
                [np.asarray(row, float) for row in file_b.get(f"{side}_weight") or []],
            )
            for side in ("left", "right")
        }
        # Every weight row has its k-point's level count, left (reference, candidate) then right.
        counts = [len(row) for row in levels_a + levels_b]
        weight_counts = [len(row) for pair in weights.values() for rows in pair for row in rows]
        if len(levels_a) != len(levels_b) or weight_counts != counts * 2:
            self.quantity(
                "slab.levels", UNEXPLAINED, detail="the k-point lists or weight rows differ in length"
            )
            return
        worst_level, worst_weight, trimmed, unmatched = 0.0, 0.0, 0, []
        near_half = False
        for side in ("left", "right"):
            for rows in weights[side]:
                for row in rows:
                    if np.any(np.abs(row - SLAB_EDGE_THRESHOLD) <= SLAB_HALF_WEIGHT_MARGIN):
                        near_half = True
        for index, (energies_a, energies_b) in enumerate(zip(levels_a, levels_b)):
            order_a, order_b = np.argsort(energies_a), np.argsort(energies_b)
            e_a, e_b = energies_a[order_a], energies_b[order_b]
            sides_a = [weights[side][0][index][order_a] for side in ("left", "right")]
            sides_b = [weights[side][1][index][order_b] for side in ("left", "right")]
            if len(e_a) != len(e_b):
                keep_a = np.abs(np.abs(e_a) - window) > tolerance
                keep_b = np.abs(np.abs(e_b) - window) > tolerance
                if keep_a.sum() != keep_b.sum():
                    unmatched.append(index)
                    continue
                trimmed += 1
                e_a, e_b = e_a[keep_a], e_b[keep_b]
                sides_a = [values[keep_a] for values in sides_a]
                sides_b = [values[keep_b] for values in sides_b]
            if len(e_a):
                worst_level = max(worst_level, float(np.abs(e_a - e_b).max()))
            for start, stop in _clusters(e_a, SLAB_CLUSTER_EV):
                if np.any(np.abs(np.abs(e_a[start:stop]) - window) < SLAB_CLUSTER_EV):
                    continue
                for values_a, values_b in zip(sides_a, sides_b):
                    worst_weight = max(
                        worst_weight, abs(float(values_a[start:stop].sum() - values_b[start:stop].sum()))
                    )
        if unmatched:
            self.quantity(
                "slab.levels",
                UNEXPLAINED,
                worst_level,
                tolerance,
                f"level counts differ away from the window edge at k-points {unmatched[:FLAGGED_LISTED]}",
            )
        else:
            note = f"; {trimmed} k-points with a level at the window edge on one side" if trimmed else ""
            self.bounded("slab.levels", worst_level, tolerance, f"ragged levels per k-point{note}")
        self.bounded(
            "slab.edge_weights",
            worst_weight,
            SLAB_WEIGHT_TOL,
            "left and right weights summed over clusters of levels within 1e-5 eV",
        )
        crossings_a, crossings_b = (
            slab_a.get("edge_crossings_at_fermi"),
            slab_b.get("edge_crossings_at_fermi"),
        )
        if crossings_a == crossings_b:
            self.quantity("slab.edge_crossings_at_fermi", AGREE, detail=_brief(crossings_a))
        elif near_half:
            self.quantity(
                "slab.edge_crossings_at_fermi",
                DOCUMENTED,
                detail=f"{crossings_a} vs {crossings_b}: a per-state edge weight lies within 1e-6 of 1/2",
            )
        else:
            self.quantity(
                "slab.edge_crossings_at_fermi", UNEXPLAINED, detail=f"{crossings_a} vs {crossings_b}"
            )

    # --------------------------------------------------------------- driver
    def run(self) -> None:
        if not self.check_documents():
            return
        if not self.check_family():
            return
        ok_a, ok_b = self.reference.ok, self.candidate.ok
        if not ok_a or not ok_b:
            self.compare_failures(ok_a, ok_b)
            return
        checks: list[Callable[[], bool]] = [
            self.check_candidate_semantics,
            self.check_source_wannier_job,
            self.check_inputs,
            self.check_parameters,
            self.check_fermi_energy,
            self.check_derived_decisions,
        ]
        results = [self.guarded(check) for check in checks]
        if not all(results):
            return
        self.set_tolerances()
        step = {
            "BANDS": self.compare_bands,
            "TOPOLOGY": self.compare_topology,
            "BERRY": self.compare_berry,
            "SLAB": self.compare_slab,
        }[self.family]
        for stage in (step, self.compare_common):
            try:
                stage()
            except MALFORMED as error:
                # A document of an unexpected shape is a difference, reported as such.
                detail = f"the documents could not be compared: {type(error).__name__}: {error}"
                self.quantity(stage.__name__, UNEXPLAINED, detail=_shorten(detail))

    def guarded(self, check: Callable[[], bool]) -> bool:
        """Run a precondition ``check_<name>``; a document of an unexpected shape fails it."""
        try:
            return check()
        except MALFORMED as error:
            detail = f"the documents could not be checked: {type(error).__name__}: {error}"
            return self.precondition(check.__name__.removeprefix("check_"), False, _shorten(detail))

    def compare_failures(self, ok_a: bool, ok_b: bool) -> None:
        error_a = (self.reference.document.get("error") or {}).get("code")
        error_b = (self.candidate.document.get("error") or {}).get("code")
        status = f"reference {self.reference.document.get('status')} ({error_a}), candidate "
        status += f"{self.candidate.document.get('status')} ({error_b})"
        if not ok_a and not ok_b and equivalent_failure_codes(error_a, error_b):
            self.quantity("status", AGREE, detail=f"both failed with equivalent codes: {status}")
        else:
            self.quantity("status", UNEXPLAINED, detail=status)

    def verdict(self) -> str:
        if any(not item["ok"] for item in self.preconditions):
            return NOT_COMPARABLE
        classes = {item["class"] for item in self.quantities}
        if UNEXPLAINED in classes:
            return FAIL
        if DOCUMENTED in classes:
            return PASS_DOCUMENTED
        return PASS

    def report(self) -> dict[str, Any]:
        verdict = self.verdict()
        worst = [
            item
            for item in self.quantities
            if item["metric"] and item["tolerance"] and item["class"] != NOT_APPLICABLE
        ]
        worst.sort(key=lambda item: item["metric"] / item["tolerance"], reverse=True)
        counts = {kind: sum(item["class"] == kind for item in self.quantities) for kind in CLASSES}
        return to_jsonable(
            {
                "schema": REPORT_SCHEMA,
                "step": self.family,
                "verdict": verdict,
                "reference": self.reference.summary(),
                "candidate": self.candidate.summary(),
                "semantics": {
                    "profile": self.semantics.name,
                    "key": self.producer,
                    "description": self.semantics.description,
                    "rules": self.semantics.rules(),
                    "conversions_applied": self.rules != CURRENT_SEMANTICS,
                    "berry_sign_applied_to_reference": self.berry_sign < 0,
                },
                "inputs_verified_by_fingerprint": self.fingerprint,
                "tolerance_basis": {
                    "weyl_bound_eV": self.weyl,
                    "eigenvalue_eV": self.eigenvalue_tol,
                },
                "preconditions": self.preconditions,
                "quantities": self.quantities,
                "flagged_sets": self.flagged,
                "summary": {
                    "classes": counts,
                    "failed_preconditions": [item["name"] for item in self.preconditions if not item["ok"]],
                    "unexplained": [item["name"] for item in self.quantities if item["class"] == UNEXPLAINED],
                    "worst_relative_to_tolerance": [
                        {"name": item["name"], "ratio": item["metric"] / item["tolerance"]}
                        for item in worst[:5]
                    ],
                },
            }
        )


def _plane_key(plane: Mapping[str, Any]) -> tuple[int, float]:
    return int(plane.get("fixed_axis", 2)), round(float(plane.get("value", 0.0)), 9)


def _atom_set(structure: Any) -> Any:
    """A structure's atoms as an order-free list of (species, rounded Cartesian position)."""
    if not isinstance(structure, Mapping):
        return structure
    species = list(structure.get("species") or [])
    positions = structure.get("cart") or []
    atoms = sorted(
        (str(name), tuple(round(float(value), 8) for value in position))
        for name, position in zip(species, positions)
    )
    rest = {key: value for key, value in structure.items() if key not in ("species", "cart")}
    return [atoms, sorted(rest.items()) if rest else []]


def _clusters(energies: np.ndarray, gap: float) -> list[tuple[int, int]]:
    """Index ranges of sorted levels whose neighbours are closer than ``gap``."""
    if not len(energies):
        return []
    breaks = np.flatnonzero(np.diff(energies) >= gap) + 1
    edges = [0, *breaks.tolist(), len(energies)]
    return list(zip(edges[:-1], edges[1:]))


def _brief(value: Any) -> str:
    return _shorten(json.dumps(to_jsonable(value), sort_keys=True), 120)


def _shorten(text: str, limit: int = 400) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def compare(step_type: str, reference_dir: str | Path, candidate_dir: str | Path) -> dict[str, Any]:
    """Compare a candidate run of a Generator Wannier step with a reference run of the same step.

    ``step_type`` names the step family (``QUEER_BANDS``, ``PYTHTB_BANDS`` or
    ``BANDS``; likewise TOPOLOGY, BERRY and SLAB); an unknown step raises
    ``ValueError``. Returns the report: ``verdict``, ``preconditions``,
    ``semantics``, ``quantities`` (``name``, ``class``, ``metric``,
    ``tolerance``, ``detail``), ``flagged_sets`` and
    ``inputs_verified_by_fingerprint``.
    """
    family = step_family(step_type)
    comparison = _Comparison(
        family, RunDirectory.load("reference", reference_dir), RunDirectory.load("candidate", candidate_dir)
    )
    comparison.run()
    return comparison.report()


def _number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: Mapping[str, Any]) -> str:
    """A Markdown page of a :func:`compare` report."""
    reference, candidate = report.get("reference") or {}, report.get("candidate") or {}
    semantics = report.get("semantics") or {}
    lines = [
        f"# Parity of {report.get('step')}: {report.get('verdict')}",
        "",
        "| | Reference | Candidate |",
        "|---|---|---|",
    ]
    for label, key in (
        ("Directory", "directory"),
        ("Results", "results_file"),
        ("Parameters", "params_file"),
        ("Engine", "engine"),
        ("Step", "step_type"),
        ("Status", "status"),
        ("Driver core SHA-256", "core_sha256"),
        ("QUEER version", "queer_version"),
    ):
        lines.append(f"| {label} | {_cell(reference.get(key) or '')} | {_cell(candidate.get(key) or '')} |")
    basis = report.get("tolerance_basis") or {}
    lines += [
        "",
        f"- **Verdict:** `{report.get('verdict')}`",
        f"- **Reference semantics:** `{semantics.get('profile')}`: {semantics.get('description')}",
        "- **Inputs verified by fingerprint:** "
        + ("yes" if report.get("inputs_verified_by_fingerprint") else "no"),
        f"- **Eigenvalue tolerance:** {_number(basis.get('eigenvalue_eV'))} eV "
        f"(Weyl bound {_number(basis.get('weyl_bound_eV'))} eV)",
    ]
    if semantics.get("berry_sign_applied_to_reference"):
        lines.append("- **Convention:** the reference's fermi-mode curvature signs were multiplied by -1")
    lines += ["", "## Preconditions", "", "| Check | OK | Detail |", "|---|---|---|"]
    for item in report.get("preconditions") or []:
        lines.append(f"| {item['name']} | {'yes' if item['ok'] else '**no**'} | {_cell(item['detail'])} |")
    lines += ["", "## Quantities", ""]
    quantities = report.get("quantities") or []
    if quantities:
        lines += ["| Quantity | Class | Metric | Tolerance | Detail |", "|---|---|---|---|---|"]
        for item in quantities:
            kind = item["class"] if item["class"] != UNEXPLAINED else f"**{item['class']}**"
            lines.append(
                f"| {item['name']} | {kind} | {_number(item['metric'])} | {_number(item['tolerance'])} | "
                f"{_cell(item['detail'])} |"
            )
    else:
        lines.append("No quantity was compared.")
    flagged = report.get("flagged_sets") or {}
    if flagged:
        lines += ["", "## Flagged sets", ""]
        for name, entry in flagged.items():
            lines.append(
                f"- `{name}`: {entry.get('size')} of {entry.get('points')} points "
                f"({entry.get('points_with_curvature')} carry curvature); integral share "
                f"{_number(entry.get('integral_share_e2_h'))} e²/h, allowance "
                f"{_number(entry.get('integral_allowance_e2_h'))} e²/h"
            )
    summary = report.get("summary") or {}
    worst = summary.get("worst_relative_to_tolerance") or []
    if worst:
        lines += ["", "## Closest to tolerance", ""]
        lines += [f"- `{entry['name']}`: {_number(entry['ratio'])} of its tolerance" for entry in worst]
    return "\n".join(lines) + "\n"


def passing(report: Mapping[str, Any]) -> bool:
    """Whether a report's verdict passes (``pass`` or ``pass_with_documented_differences``)."""
    return report.get("verdict") in PASSING_VERDICTS
