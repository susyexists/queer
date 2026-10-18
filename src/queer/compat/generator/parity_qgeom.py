"""Parity of a QUEER_QGEOM run against a PYTHTB_QGEOM reference (or another QUEER run).

:class:`QgeomComparison` extends the Wannier-family comparison of
:mod:`~queer.compat.generator.parity` (same documents, inputs by SHA-256,
parameters, Fermi energy and derived decisions, which here are the position level,
the band window, the occupied count and its source and whether the occupied
manifold's gap is open) with the quantum-geometry rules below. No error estimate
or magnitude a run reports enters a tolerance: every tolerance is the rounding of
the documents' own print format, a fixed relative tolerance on a quantity's values
or a bound derived from it, and a scale taken from compared values can only grow
when those values differ, which fails them.

**Producer.** A PythTB reference must come from the vendored kernel
(``provenance.qgeom_sha256`` 4fb51a38…) and a core whose quantum-geometry code is
the vendored one (:data:`QGEOM_CORES`, 56fda912… to 0efcc4b3…); any other revision
is ``not_comparable``.

**Replay.** The adaptive mesh is the integration's sampling, so it must be the
same. A reference whose integration stopped on its vertex budget, at the depth
limit or not at all is deterministic: the candidate must not have stopped on its
own time budget. A reference that stopped on its time budget is reproduced only by
a candidate run with ``qgeom_replay_vertices`` equal to the reference's vertex
count; without that count, or without that replay, the verdict is
``not_comparable`` with the reason (never an error-bar comparison). The vertex
sets are equal when both runs recorded ``Out/qgeom_vertices.json`` with equal
SHA-256 of the sorted integer vertex keys (exact coordinates), or, against a
reference without the record, when the recorded mesh (vertex count, vertices per
rule, levels, finest cell, seeds, controlled mu, the capped gap rule and the last
12 history entries) is equal. A record must be its own run's mesh (its count that
run's vertex count, its keys hashing to its digest, a QUEER run's digest its
``integration.vertex_set``'s), and a QUEER candidate without one fails against a
reference that has one. The stop reason and the convergence flags must be equal.
Exact symmetric ties are broken by rounding, which another BLAS kernel breaks
another way: where both runs record bands C3-symmetric to the eigenvalue tolerance
(``symmetry.c3_band_floor_meV`` on a hexagonal lattice), different meshes make the
verdict ``not_comparable`` (the reference's adaptive mesh depends on rounding at
exact symmetric ties: replay on the same BLAS kernel or compare on a fixed uniform
grid), never ``fail``; without that record a mesh difference fails.

**Quantities.** With δ(T) = 1e-8 max(max A3, max |M|) over both runs at
temperature T: the occupied manifold's C and C_in to 1e-8 max(1, |C|), K_in, K_diag
and K_upper_loose to 1e-8 of their largest element, K_in - |C_in| to the sum, the
PSD ratio to 1e-8; the headline A3, A3_err to δ(T) (2 δ); theta0 to asin(δ/A3)/3 on
3 theta0 modulo 360 degrees where A3 > δ, and not judged where A3 <= δ (there M's
tolerance turns its trigonal part to any direction: the entry is recorded as
undetermined, and a row is ``not_comparable`` only if nothing else in it is
judged); the three chi amplitudes, whose trigonal part is exactly
-(c_out + 2 c_in) CHI_PER_M times M's, to |c_out + 2 c_in| CHI_PER_M sqrt(2) δ plus
the rounding of building and projecting chi; sigma_xy to 1e-8 max(1, |sigma_xy|),
the density to the count tolerance (1e-8 of the band count, at least one ulp of it
per vertex) per cell. The per-T curves of ``Out/qgeom_response.json`` and the path
data of ``Out/qgeom_path.json`` are written with 5 significant digits: each value
agrees to half a unit in that digit of each run plus the quantity's tolerance. The
dipole sources, the symmetry residuals, the position checks, the grid, window,
electron count and verdict agree exactly or to the bounds derived from these.

**mu_CNP** is judged by the electron count: each run's count (its density curve
interpolated linearly in mu, the rule that placed its mu_CNP) must be n_occ at its
own mu_CNP and at the other run's, within the count tolerance plus the print
rounding. Two values in one gap, where the count is n_occ over a range of mu and
rounding picks the interpolated value, are equal by this rule. Each run's CNP row
must sit at the grid mu nearest its mu_CNP, and mu - mu_CNP must differ between the
runs by exactly the mu_CNP difference. A headline point the two runs place at
different grid mu (two mu_CNP that agree, or a near-tie of the A3 maximum) is
documented, and its values are checked against the other run's curves at its own
mu. A run without a mu_CNP fails unless its count misses n_occ by no more than the
count tolerance, where the other run's mu_CNP is rounding: ``not_comparable``.

**Producer gaps.** The driver never reads the ``.win``'s ``kpoints`` block, so its
``.eig`` check is always skipped where QUEER runs it: ``dft_check`` is documented
(``not_applicable`` when QUEER could not read the ``.eig`` either). The driver
checks its diagonal level against PythTB and QUEER against its own band geometry:
``parity`` is not compared, but the candidate's own check must pass. The warnings of
those checks are left out of the warning comparison, and the
QUEER-only replay record out of the output comparison.
"""

from __future__ import annotations

import functools
import hashlib
import math
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import numpy as np

from .parity import (
    AGREE,
    DOCUMENTED,
    NOT_APPLICABLE,
    UNEXPLAINED,
    RunDirectory,
    _array,
    _Comparison,
    _shorten,
)
from .qgeom_step import FULL_RESPONSE_FILE, PROBE_TOLERANCE, VERTICES_FILE, compact
from .results import WARN_DFT_EIG_UNREADABLE, WARN_DFT_FROZEN_MISMATCH, WARN_QGEOM_PROBE

# The vendored kernel and the driver cores whose quantum-geometry code is the vendored one.
QGEOM_KERNEL_SHA256 = "4fb51a38691931f2b687d731619a21ae2caa094b1dd4681d1d97073041f23ee3"
QGEOM_CORES = frozenset(
    {
        "56fda912ba570e2293b3a9b4bf20d11038c64efa5d2f192c8ac03c615034e215",
        "93167386d4b15b24ff6ba84e656a4d4580164a3f02e52f6d2541f99b0027eb13",
        "dbdf34d11e655e7bca1d49e4e632aa74644a49dd9917cc50d1e7f3059ce7bef5",
        "d75fb8fc3c08b87d776023f804fe0e6d122d231cd655ef61ba9fecb86544d49b",
        "f4a73779a891de356cc53e55d8c077f0fdc6befce931a6ed0bf9782973f5551b",
        "0efcc4b3079266a8d35644d3d06f648857e31d86b12be173c6f03298a5d505c0",
    }
)
RELATIVE = 1e-8
PRINT_DIGITS = 5
# Recorded mesh fields that must be equal when a run has no vertex record.
MESH_FIELDS = (
    "base_mesh",
    "seeds",
    "mu_subset",
    "vertices",
    "vertices_by_rule",
    "gap_rule_capped",
    "max_level",
)
HISTORY_EXACT = ("phase", "vertices", "leaves", "target")
CHI_SETS = {"wpd_1_m05": (1.0, -0.5), "lka_2_m05": (2.0, -0.5), "qk_05_m1": (0.5, -1.0)}
CHI_PER_M = 1.602176634e-19**2 / (6.62607015e-34 / (2.0 * math.pi)) * 1e-10 * 1e12
# Building chi from M and projecting it: a few ulp of the largest M element per run (generously 64 eps).
CHI_ROUNDING = 64.0 * float(np.finfo(float).eps)
# The triangle rule's count: one ulp of the band count per vertex, at least the plan's 1e-8.
COUNT_ROUNDING = float(np.finfo(float).eps)
RESIDUAL_TOL = 1e-12
FULL_RESPONSE_NAME = FULL_RESPONSE_FILE.split("/", 1)[1]
SYMMETRIC_TIES = (
    "the reference's adaptive mesh depends on rounding at exact symmetric ties: replay on the same BLAS "
    "kernel or compare on a fixed uniform grid"
)


def print_precision(values: Any) -> np.ndarray:
    """Half a unit in the fifth significant digit of each value (0 for 0 or null): the rounding of
    the response and path files."""
    a = np.abs(np.asarray(values, float))
    out = np.zeros(a.shape)
    finite = np.isfinite(a) & (a > 0)
    out[finite] = 0.5 * 10.0 ** (np.floor(np.log10(a[finite])) - (PRINT_DIGITS - 1))
    return out


def wrap_degrees(x: Any) -> np.ndarray:
    """Angles in degrees mapped into [-180, 180)."""
    return np.mod(np.asarray(x, float) + 180.0, 360.0) - 180.0


def theta_tolerance(delta: float, A3: np.ndarray) -> np.ndarray:
    """The largest change of theta0 (degrees) a move of (alpha, beta) by ``delta`` allows at amplitude A3.

    Where A3 <= delta the move can turn (alpha, beta) to any direction: theta0 is not
    determined there, and the tolerance is NaN (the entry is not judged).
    """
    a = np.asarray(A3, float)
    determined = a > delta
    ratio = np.where(determined, delta / np.where(determined, a, 1.0), np.nan)
    return np.degrees(np.arcsin(ratio)) / 3.0


def vertex_digest(keys: Any) -> str | None:
    """SHA-256 of the sorted integer vertex keys as QUEER writes it, or None for keys that are not integers."""
    try:
        array = np.asarray(keys)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or (array.size and not np.issubdtype(array.dtype, np.integer)):
        return None
    return hashlib.sha256(np.sort(array.astype("<i8")).tobytes()).hexdigest()


def _finite(value: Any) -> float | None:
    """A finite JSON number as a float, else None."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _number(value: Any) -> bool:
    return _finite(value) is not None


class CountCurve(NamedTuple):
    """A run's electron count per cell on its response grid (eV), its print rounding and the file's stride."""

    mu: np.ndarray
    electrons: np.ndarray
    rounding: np.ndarray
    stride: int
    n_occ: int

    def at(self, mu: float) -> tuple[float, float]:
        """The linear interpolation at ``mu`` (eV) and the print rounding of that value."""
        if len(self.mu) == 1:
            return float(self.electrons[0]), float(self.rounding[0])
        j = int(np.clip(np.searchsorted(self.mu, mu) - 1, 0, len(self.mu) - 2))
        return float(np.interp(mu, self.mu, self.electrons)), float(
            max(self.rounding[j], self.rounding[j + 1])
        )


def _max_abs(*values: Any) -> float:
    largest = 0.0
    for value in values:
        array = _array(value)
        if array is not None and array.size and np.isfinite(array).any():
            largest = max(largest, float(np.nanmax(np.abs(array[np.isfinite(array)]))))
    return largest


class QgeomComparison(_Comparison):
    """The Wannier-family comparison with the quantum-geometry replay, stop rule and tolerances."""

    def __init__(self, family: str, reference: RunDirectory, candidate: RunDirectory):
        super().__init__(family, reference, candidate)
        self.producer_warning_codes = frozenset(
            {WARN_QGEOM_PROBE, WARN_DFT_FROZEN_MISMATCH, WARN_DFT_EIG_UNREADABLE}
        )
        self.queer_only_outputs = frozenset({VERTICES_FILE, FULL_RESPONSE_FILE})
        self.ties: dict[str, str] = {}
        # Per temperature: the candidate's mu_CNP minus the reference's (eV).
        self.cnp_shift: dict[str, float] = {}
        self._mesh: tuple[list[str], list[str], str] | None = None
        # Both runs wrote unrounded curves (each the source of its 5-digit file): no print rounding.
        self.full_curves = False

    # ------------------------------------------------------- preconditions
    def step_preconditions(self) -> list[Callable[[], bool]]:
        return [self.check_qgeom_producer, self.check_stop_rule, self.check_mesh_determined]

    def check_qgeom_producer(self) -> bool:
        problems = []
        for run in (self.reference, self.candidate):
            if run.engine == "queer":
                continue
            provenance = run.document.get("provenance") or {}
            kernel, core = provenance.get("qgeom_sha256"), provenance.get("core_sha256")
            if kernel != QGEOM_KERNEL_SHA256:
                problems.append(f"{run.role} kernel qgeom_sha256 {str(kernel)[:12]}… is not the vendored one")
            elif core not in QGEOM_CORES:
                problems.append(f"{run.role} core {str(core)[:12]}… has another quantum-geometry driver")
        detail = "; ".join(problems) or "both runs' quantum-geometry code is QUEER's or the vendored driver's"
        return self.precondition("qgeom_producer", not problems, detail)

    def check_stop_rule(self) -> bool:
        integ_a = self.reference.block("qgeom").get("integration") or {}
        integ_b = self.candidate.block("qgeom").get("integration") or {}
        stop_a, stop_b = integ_a.get("stop"), integ_b.get("stop")
        replay_b = (integ_b.get("replay") or {}).get("stop_vertices")
        replay_a = (integ_a.get("replay") or {}).get("stop_vertices")
        vertices_a = integ_a.get("vertices")
        if stop_a == "time" and replay_a is None:
            if not isinstance(vertices_a, int):
                return self.precondition(
                    "stop_rule",
                    False,
                    "the reference stopped on its time budget and records no vertex count: its sampling "
                    "cannot be replayed",
                )
            if replay_b != vertices_a:
                return self.precondition(
                    "stop_rule",
                    False,
                    f"the reference stopped on its time budget at {vertices_a} vertices: rerun the candidate "
                    f"with qgeom_replay_vertices = {vertices_a} (it ran with {replay_b})",
                )
            return self.precondition(
                "stop_rule", True, f"the candidate replays the reference's time stop at {vertices_a} vertices"
            )
        if stop_b == "time" and replay_b is None:
            return self.precondition(
                "stop_rule",
                False,
                "the candidate stopped on its own time budget: rerun it with qgeom_time_stop = false",
            )
        if stop_a == "time" and replay_a is not None and replay_b != replay_a:
            return self.precondition(
                "stop_rule",
                False,
                f"the reference is a replay stopped at {replay_a} vertices; the candidate replayed {replay_b}",
            )
        return self.precondition(
            "stop_rule",
            True,
            f"deterministic: the reference stopped with {stop_a!r}, the candidate with {stop_b!r}",
        )

    # ------------------------------------------------------------ helpers
    def response(self, run: RunDirectory) -> dict[str, Any]:
        """The run's response file; its curves unrounded where both runs wrote them."""
        value = run.out("qgeom_response.json")
        if not isinstance(value, dict):
            return {}
        if self.full_curves:
            full = run.out(FULL_RESPONSE_NAME)
            return {**value, "mu_eV": full["mu_eV"], "per_T": full["per_T"]}
        return value

    @property
    def printed(self) -> bool:
        """Whether the response curves carry the 5-digit print rounding."""
        return not self.full_curves

    def printing(self, values: Any) -> np.ndarray:
        """The print rounding of response values (none for unrounded curves)."""
        return print_precision(values) if self.printed else np.zeros(np.shape(np.asarray(values, float)))

    def main_level(self, run: RunDirectory) -> str:
        return "full" if (run.block("qgeom").get("position") or {}).get("mode") == "full" else "diag"

    def level_curve(self, run: RunDirectory, T: str, level: str, key: str) -> np.ndarray | None:
        entry = (self.response(run).get("per_T") or {}).get(T) or {}
        lv = (entry.get("levels") or {}).get(level) or {}
        return _array(lv.get(key))

    def delta(self, T: str, level: str = "main") -> float:
        """δ(T) = 1e-8 max(A3, |M|) over both runs' curves (and headline rows) at temperature T."""
        values: list[Any] = []
        for run in (self.reference, self.candidate):
            name = self.main_level(run) if level == "main" else level
            values += [self.level_curve(run, T, name, "A3"), self.level_curve(run, T, name, "M")]
            if level == "main":
                rows = ((run.block("qgeom").get("headline") or {}).get(T)) or {}
                values += [row.get("A3_A_per_eV") for row in rows.values() if isinstance(row, dict)]
        return RELATIVE * _max_abs(*values)

    def count_tolerance(self) -> float:
        """1e-8 of the band count, or one ulp of it per vertex where the triangle rule sums more terms."""
        bands = (self.reference.block("qgeom").get("window") or {}).get("bands") or [0, 1]
        vertices = [
            (run.block("qgeom").get("integration") or {}).get("vertices")
            for run in (self.reference, self.candidate)
        ]
        terms = max([v for v in vertices if isinstance(v, int) and not isinstance(v, bool)] or [0])
        return max(RELATIVE, terms * COUNT_ROUNDING) * max(1.0, float(bands[-1]))

    def area(self) -> float:
        lattice = np.asarray(self.reference.block("model").get("lattice"), float).reshape(3, 3)
        return float(np.linalg.norm(np.cross(lattice[0], lattice[1])))

    def values(self, name: str, a: Any, b: Any, tolerance: float, detail: str = "") -> bool:
        """Numbers or arrays equal to ``tolerance`` (absolute), null where both are null."""
        A, B = _array(a), _array(b)
        if A is None or B is None or A.shape != B.shape:
            if a is None and b is None:
                self.quantity(name, AGREE, 0.0, tolerance, detail or "null in both runs")
                return True
            self.quantity(
                name,
                UNEXPLAINED,
                None,
                tolerance,
                detail or f"{_shorten(repr(a), 80)} vs {_shorten(repr(b), 80)}",
            )
            return False
        nan_a, nan_b = np.isnan(A), np.isnan(B)
        if not np.array_equal(nan_a, nan_b):
            self.quantity(name, UNEXPLAINED, None, tolerance, "null entries differ")
            return False
        metric = float(np.max(np.abs(A - B)[~nan_a])) if (~nan_a).any() else 0.0
        return self.bounded(name, metric, tolerance, detail)

    def curve(
        self,
        name: str,
        a: Any,
        b: Any,
        derived: Any,
        *,
        circular: bool = False,
        detail: str = "",
        undetermined: str = "",
        printed: bool = True,
    ) -> bool:
        """5-digit curves: |a - b| <= print(a) + print(b) + derived at every entry (metric: the worst ratio).

        An entry whose derived tolerance is NaN is not determined by the documents
        (``undetermined`` says why): it is not judged, and a curve without any determined
        entry is recorded as undetermined.
        """
        A, B = _array(a), _array(b)
        if a is None and b is None:
            self.quantity(name, AGREE, 0.0, 1.0, "null in both runs")
            return True
        if A is None or B is None or A.shape != B.shape:
            self.quantity(name, UNEXPLAINED, None, 1.0, f"shapes differ: {np.shape(a)} vs {np.shape(b)}")
            return False
        nan_a, nan_b = np.isnan(A), np.isnan(B)
        if not np.array_equal(nan_a, nan_b):
            self.quantity(name, UNEXPLAINED, None, 1.0, "null entries differ")
            return False
        bound = np.broadcast_to(np.asarray(derived, float), A.shape)
        skipped = ~nan_a & np.isnan(bound)
        keep = ~nan_a & ~skipped
        if skipped.any() and not keep.any():
            self.not_determined(
                name, f"{undetermined or 'no derived tolerance'} at all {int(skipped.sum())} points"
            )
            return True
        if circular:
            difference = np.abs(wrap_degrees(3.0 * (A - B))) / 3.0
        else:
            difference = np.abs(A - B)
        rounding = print_precision(A) + print_precision(B) if printed else np.zeros(A.shape)
        tolerance = rounding + np.nan_to_num(bound, nan=0.0)
        ratio = np.where(
            tolerance > 0,
            difference / np.where(tolerance > 0, tolerance, 1.0),
            np.where(difference > 0, np.inf, 0.0),
        )
        worst = float(np.max(ratio[keep])) if keep.any() else 0.0
        detail = detail or "worst |a - b| over print precision plus tolerance"
        if skipped.any():
            detail += (
                f"; {int(skipped.sum())} of {int((~nan_a).sum())} points not judged "
                f"({undetermined or 'no derived tolerance'})"
            )
        return self.bounded(name, worst, 1.0, detail)

    def shifted_curve(self, name: str, a: Any, b: Any, shift: float) -> bool:
        """Response curves that differ by ``shift``: |a - b - shift| <= print(a) + print(b) + 1e-12."""
        A, B = _array(a), _array(b)
        if a is None and b is None:
            self.quantity(name, AGREE, 0.0, 1.0, "null in both runs")
            return True
        if A is None or B is None or A.shape != B.shape or not np.array_equal(np.isnan(A), np.isnan(B)):
            self.quantity(
                name, UNEXPLAINED, None, 1.0, f"shapes or nulls differ: {np.shape(a)} vs {np.shape(b)}"
            )
            return False
        keep = ~np.isnan(A)
        tolerance = self.printing(A) + self.printing(B) + 1e-12
        worst = float(np.max(np.abs(A - B - shift)[keep] / tolerance[keep])) if keep.any() else 0.0
        return self.bounded(
            name,
            worst,
            1.0,
            f"worst |a - b - {shift:.6g} eV| over print precision ({shift:.6g} eV: the mu_CNP difference)",
        )

    def not_determined(self, name: str, reason: str) -> None:
        """A quantity the documents do not determine: recorded, not judged."""
        self.quantity(name, NOT_APPLICABLE, None, None, _shorten(f"undetermined: {reason}"))
        self.flagged.setdefault("qgeom_undetermined", {})[name] = reason

    # ------------------------------------------------------------ the step
    def compare_qgeom(self) -> None:
        qa, qb = self.reference.block("qgeom"), self.candidate.block("qgeom")
        self.compare_full_curves()
        self.compare_mesh(qa, qb)
        self.compare_occupied(qa.get("occupied") or {}, qb.get("occupied") or {})
        self.compare_electron_count(qa.get("electron_count") or {}, qb.get("electron_count") or {})
        self.compare_headline(qa.get("headline") or {}, qb.get("headline") or {})
        self.compare_response()
        self.compare_qgeom_path()
        self.compare_dipole_sources(qa.get("dipole_sources"), qb.get("dipole_sources"))
        self.compare_symmetry(qa.get("symmetry") or {}, qb.get("symmetry") or {}, qa, qb)
        self.compare_position(qa.get("position") or {}, qb.get("position") or {})
        self.compare_settings(qa, qb)
        self.compare_dft_check(qa.get("dft_check"), qb.get("dft_check"))
        self.compare_probe(qa.get("parity"), qb.get("parity"))

    def compare_full_curves(self) -> None:
        """QUEER's unrounded curves: each must round to its own 5-digit file; both present, they are compared."""
        states: dict[str, bool] = {}
        for run in (self.reference, self.candidate):
            full = run.out(FULL_RESPONSE_NAME)
            if full is None and not (run.directory / FULL_RESPONSE_FILE).is_file():
                continue
            five = run.out("qgeom_response.json")
            states[run.role] = (
                isinstance(full, dict)
                and isinstance(five, dict)
                and compact(full.get("mu_eV")) == five.get("mu_eV")
                and compact(full.get("per_T")) == five.get("per_T")
            )
        if not states:
            return
        name = "qgeom.response.full_precision"
        broken = [role for role, ok in states.items() if not ok]
        if broken:
            detail = (
                " and ".join(broken) + f"'s {FULL_RESPONSE_FILE} does not round to its qgeom_response.json"
            )
            self.quantity(name, UNEXPLAINED, float(len(broken)), 0.0, detail)
        elif len(states) == 2:
            self.full_curves = True
            self.quantity(
                name, AGREE, detail="both runs wrote unrounded curves: compared without print rounding"
            )
        else:
            only = next(iter(states))
            detail = f"only the {only} wrote unrounded curves (they round to its file): compared at 5 digits"
            self.quantity(name, AGREE, detail=detail)

    # --------------------------------------------------------------- mesh
    def compare_mesh(self, qa: Mapping[str, Any], qb: Mapping[str, Any]) -> None:
        ia, ib = qa.get("integration") or {}, qb.get("integration") or {}
        flags = [
            f"{key}: {ia.get(key)!r} vs {ib.get(key)!r}"
            for key in ("stop", "converged", "confirmed")
            if ia.get(key) != ib.get(key)
        ]
        self.quantity(
            "qgeom.stop",
            UNEXPLAINED if flags else AGREE,
            float(len(flags)),
            0.0,
            "; ".join(flags)
            or f"stop {ia.get('stop')!r}, converged {ia.get('converged')}, confirmed {ia.get('confirmed')}",
        )
        inconsistent, differing, same = self.mesh_comparison()
        problems = inconsistent or differing
        self.quantity(
            "qgeom.mesh",
            UNEXPLAINED if problems else AGREE,
            float(len(problems)),
            0.0,
            _shorten("; ".join(problems) or same),
        )

    def mesh_comparison(self) -> tuple[list[str], list[str], str]:
        """The vertex records that are not their run's mesh, the mesh differences, and the agreement detail.

        Every record must be its run's mesh: its count that run's vertex count, its keys
        (when written) hashing to its digest, and a QUEER run's digest its
        ``integration.vertex_set``'s.
        """
        if self._mesh is not None:
            return self._mesh
        ia = self.reference.block("qgeom").get("integration") or {}
        ib = self.candidate.block("qgeom").get("integration") or {}
        record_a, record_b = (
            self.reference.out("qgeom_vertices.json"),
            self.candidate.out("qgeom_vertices.json"),
        )
        inconsistent: list[str] = []
        for run, record, integration in ((self.reference, record_a, ia), (self.candidate, record_b, ib)):
            if not isinstance(record, dict):
                continue
            if record.get("count") != integration.get("vertices"):
                inconsistent.append(
                    f"the {run.role}'s vertex record ({record.get('count')} vertices) is not its mesh "
                    f"({integration.get('vertices')} vertices)"
                )
            keys = record.get("keys")
            if keys is not None:
                digest = vertex_digest(keys)
                if digest is None or not isinstance(keys, list) or len(keys) != record.get("count"):
                    inconsistent.append(
                        f"the {run.role}'s vertex keys are not {record.get('count')} integers"
                    )
                elif digest != record.get("sha256"):
                    inconsistent.append(
                        f"the {run.role}'s vertex keys hash to {digest[:12]}…, not to its record's digest "
                        f"{str(record.get('sha256'))[:12]}…"
                    )
            own = integration.get("vertex_set")
            if isinstance(own, dict) and own.get("sha256") != record.get("sha256"):
                inconsistent.append(f"the {run.role}'s vertex record differs from its integration.vertex_set")
        if self.candidate.engine == "queer" and isinstance(record_a, dict) and not isinstance(record_b, dict):
            inconsistent.append("the candidate wrote no Out/qgeom_vertices.json")
        differing: list[str] = []
        if isinstance(record_a, dict) and isinstance(record_b, dict):
            differing = [
                f"{key}: {record_a.get(key)!r} vs {record_b.get(key)!r}"
                for key in ("count", "base", "unit", "sha256")
                if record_a.get(key) != record_b.get(key)
            ]
            keys_a, keys_b = record_a.get("keys"), record_b.get("keys")
            if differing and isinstance(keys_a, list) and isinstance(keys_b, list):
                only_a, only_b = set(keys_a) - set(keys_b), set(keys_b) - set(keys_a)
                differing.append(
                    f"{len(only_a)} vertices only in the reference, {len(only_b)} only in the candidate"
                )
            same = (
                f"identical vertex sets: {record_a.get('count')} vertices, SHA-256 "
                f"{str(record_a.get('sha256'))[:12]}…"
            )
        else:
            differing = [
                f"{key}: {ia.get(key)!r} vs {ib.get(key)!r}"
                for key in MESH_FIELDS
                if ia.get(key) != ib.get(key)
            ]
            finest_a, finest_b = ia.get("finest_cell_invA"), ib.get("finest_cell_invA")
            if (
                not isinstance(finest_a, (int, float))
                or not isinstance(finest_b, (int, float))
                or abs(finest_a - finest_b) > 1e-12 * max(abs(finest_a), 1e-300)
            ):
                differing.append(f"finest_cell_invA: {finest_a!r} vs {finest_b!r}")
            history_a, history_b = ia.get("history") or [], ib.get("history") or []
            if len(history_a) != len(history_b):
                differing.append(f"history of {len(history_a)} vs {len(history_b)} entries")
            else:
                for index, (x, y) in enumerate(zip(history_a, history_b)):
                    if any(x.get(key) != y.get(key) for key in HISTORY_EXACT):
                        differing.append(f"history[{index}]: {x} vs {y}")
                        break
            same = (
                f"recorded meshes equal ({ia.get('vertices')} vertices, by rule {ia.get('vertices_by_rule')}); "
                "the reference records no vertex keys, so the sets are compared by their record"
            )
        self._mesh = (inconsistent, differing, same)
        return self._mesh

    def symmetric_ties(self) -> str | None:
        """Why rounding decides the adaptive mesh, from the symmetry record both runs carry, or None.

        Both runs record how far their bands are from C3-symmetric at 512 random
        k-points (``symmetry.c3_band_floor_meV``). Where both floors lie within the
        eigenvalue tolerance, symmetry-related k-points tie exactly: the strict local
        extrema of the occupation pass (the seeds), a band velocity that vanishes by
        symmetry in the edge rule and the error estimates of symmetry-related cells in the
        refinement order are then decided by rounding, which another BLAS kernel breaks
        another way.
        """
        self.set_tolerances()
        tau = self.eigenvalue_tol * 1e3
        floors = []
        for run in (self.reference, self.candidate):
            symmetry = run.block("qgeom").get("symmetry") or {}
            floor = symmetry.get("c3_band_floor_meV")
            if symmetry.get("lattice_hexagonal") is not True or not isinstance(floor, (int, float)):
                return None
            if isinstance(floor, bool) or not math.isfinite(floor) or floor > tau:
                return None
            floors.append(float(floor))
        return (
            f"both runs' bands are C3-symmetric to {max(floors):.3g} meV, within the eigenvalue tolerance "
            f"{tau:.3g} meV, so symmetry-related k-points tie exactly in the occupation pass's extrema and the "
            "refinement decisions"
        )

    def check_mesh_determined(self) -> bool:
        """Different vertex sets of an exactly symmetric model are rounding at its ties: not comparable."""
        inconsistent, differing, same = self.mesh_comparison()
        if inconsistent:
            return self.precondition(
                "mesh_determined", True, "a vertex record is not its run's mesh: compared (qgeom.mesh)"
            )
        if not differing:
            return self.precondition("mesh_determined", True, same)
        ties = self.symmetric_ties()
        if ties is None:
            return self.precondition(
                "mesh_determined",
                True,
                "the meshes differ and neither run records an exact symmetry that rounding could break: "
                "compared (qgeom.mesh)",
            )
        detail = f"{SYMMETRIC_TIES} ({ties}; the meshes differ: {'; '.join(differing)})"
        return self.precondition("mesh_determined", False, _shorten(detail, 1200))

    # ---------------------------------------------------- occupied manifold
    def compare_occupied(self, a: Mapping[str, Any], b: Mapping[str, Any]) -> None:
        exact = [key for key in ("applicable", "n_occ", "verdict", "reason") if a.get(key) != b.get(key)]
        self.quantity(
            "qgeom.occupied.verdict",
            UNEXPLAINED if exact else AGREE,
            float(len(exact)),
            0.0,
            "; ".join(f"{key}: {a.get(key)!r} vs {b.get(key)!r}" for key in exact)
            or f"verdict {a.get('verdict')!r}",
        )
        if not a.get("applicable") and not b.get("applicable"):
            self.quantity(
                "qgeom.occupied", NOT_APPLICABLE, detail="the occupied manifold's gap closes in both runs"
            )
            return
        tol_c = RELATIVE * max(1.0, _max_abs(a.get("chern"), b.get("chern")))
        tol_cin = RELATIVE * max(1.0, _max_abs(a.get("chern_in"), b.get("chern_in")))
        tol_k = RELATIVE * _max_abs(a.get("K_in"), b.get("K_in"))
        self.values("qgeom.occupied.chern", a.get("chern"), b.get("chern"), tol_c)
        self.values(
            "qgeom.occupied.chern_integer_distance",
            a.get("chern_integer_distance"),
            b.get("chern_integer_distance"),
            tol_c,
        )
        self.values("qgeom.occupied.chern_in", a.get("chern_in"), b.get("chern_in"), tol_cin)
        self.values("qgeom.occupied.K_in", a.get("K_in"), b.get("K_in"), tol_k)
        self.values(
            "qgeom.occupied.K_in_minus_abs_C",
            a.get("K_in_minus_abs_C"),
            b.get("K_in_minus_abs_C"),
            2 * tol_k + tol_cin,
        )
        self.values(
            "qgeom.occupied.psd_in_min_rel", a.get("psd_in_min_rel"), b.get("psd_in_min_rel"), RELATIVE
        )
        for key in ("K_diag", "K_upper_loose"):
            if key in a or key in b:
                tolerance = 2 * RELATIVE * _max_abs(a.get(key), b.get(key)) + 2 * tol_k
                self.values(f"qgeom.occupied.{key}", a.get(key), b.get(key), tolerance)

    # ------------------------------------------------------- electron count
    def compare_electron_count(self, a: Mapping[str, Any], b: Mapping[str, Any]) -> None:
        tau = 2.0 * self.eigenvalue_tol * 1e3  # meV
        exact = [key for key in ("n_occ", "source") if a.get(key) != b.get(key)]
        self.quantity(
            "qgeom.electron_count.n_occ",
            UNEXPLAINED if exact else AGREE,
            float(len(exact)),
            0.0,
            "; ".join(f"{key}: {a.get(key)!r} vs {b.get(key)!r}" for key in exact)
            or f"n_occ {a.get('n_occ')}",
        )
        self.values(
            "qgeom.electron_count.count_at_EF",
            a.get("count_at_EF"),
            b.get("count_at_EF"),
            1e-12 * max(1.0, _max_abs(a.get("count_at_EF"))),
        )
        for key in ("gap_at_gamma_meV", "min_direct_gap_meV", "global_gap_meV"):
            self.values(f"qgeom.electron_count.{key}", a.get(key), b.get(key), tau)
        if a.get("state_at_cnp") == b.get("state_at_cnp"):
            self.quantity("qgeom.electron_count.state_at_cnp", AGREE, detail=str(a.get("state_at_cnp")))
        else:
            gap = a.get("global_gap_meV")
            minimum = float((self.reference.params or {}).get("insulator_min_gap_meV") or 0.1)
            near = isinstance(gap, (int, float)) and min(abs(gap - minimum), abs(gap + minimum)) <= tau
            self.quantity(
                "qgeom.electron_count.state_at_cnp",
                DOCUMENTED if near else UNEXPLAINED,
                detail=f"{a.get('state_at_cnp')!r} vs {b.get('state_at_cnp')!r} (global gap {gap} meV)",
            )
        cnp_a, cnp_b = a.get("mu_cnp_meV") or {}, b.get("mu_cnp_meV") or {}
        if set(cnp_a) != set(cnp_b):
            self.quantity(
                "qgeom.electron_count.mu_cnp_meV",
                UNEXPLAINED,
                detail=f"temperatures {sorted(cnp_a)} vs {sorted(cnp_b)}",
            )
            return
        for T in sorted(cnp_a, key=lambda key: float(key[:-1])):
            self.compare_mu_cnp(T, cnp_a[T], cnp_b[T])

    def full_grid(self, run: RunDirectory) -> np.ndarray | None:
        """The run's mu grid (eV) at full precision, from its grid block (the driver's uniform grid)."""
        grid = run.block("qgeom").get("grid") or {}
        lo, hi = _finite(grid.get("mu_min_meV")), _finite(grid.get("mu_max_meV"))
        step = grid.get("mu_step_meV")
        if lo is None or hi is None:
            return None
        if step is None:
            return np.array([lo * 1e-3])
        width = _finite(step)
        if width is None or not width > 0:
            return None
        n = int(round((hi - lo) / width)) + 1
        return (lo + width * np.arange(max(n, 1))) * 1e-3

    def count_curve(self, run: RunDirectory, T: str) -> CountCurve | None:
        """The run's electron count per cell on its response grid, or None when it has no such curve.

        The count is n_occ plus the 5-digit density times the cell area; the mu grid is
        the grid block's at the response file's stride, checked against the file's
        5-digit mu.
        """
        response = self.response(run)
        mu_file = _array(response.get("mu_eV"))
        density = _array(((response.get("per_T") or {}).get(T) or {}).get("density_cm2"))
        n_occ = (run.block("qgeom").get("electron_count") or {}).get("n_occ")
        full = self.full_grid(run)
        stride = (run.block("qgeom").get("grid") or {}).get("downsampled") or 1
        if (
            mu_file is None
            or density is None
            or full is None
            or not isinstance(n_occ, int)
            or not isinstance(stride, int)
            or stride < 1
            or mu_file.ndim != 1
            or density.shape != mu_file.shape
            or not len(mu_file)
            or np.isnan(density).any()
        ):
            return None
        mu = full[::stride]
        if mu.shape != mu_file.shape or np.any(np.abs(mu - mu_file) > print_precision(mu_file) + 1e-12):
            return None
        per_cell = self.area() * 1e-16
        return CountCurve(mu, n_occ + density * per_cell, self.printing(density) * per_cell, stride, n_occ)

    def compare_mu_cnp(self, T: str, a: Any, b: Any) -> None:
        """mu_CNP judged by the electron count: each run's count at the other run's mu_CNP must be n_occ.

        Each run's count curve (its own response file) is interpolated linearly in mu, the
        rule that placed its mu_CNP, so its count at its own mu_CNP is n_occ up to the
        print rounding; at the other run's mu_CNP it must be n_occ within the count
        tolerance (1e-8 of the band count, which bounds the triangle rule's rounding over
        the run's vertices) plus that rounding. Two values in one gap, where the count is
        n_occ over a range of mu and its rounding picks the interpolated mu_CNP, are equal
        by this rule, and the detail says so. On a decimated file the count between grid
        points is the file's interpolation, so the rule compares each run's count at the
        two values with each other. Each run's CNP row must sit at the grid mu nearest its
        mu_CNP. A wrong mu_CNP fails; a difference no count curve can judge makes the
        comparison not comparable.
        """
        name = f"qgeom.electron_count.mu_cnp_meV.{T}"
        tolerance = self.count_tolerance()
        allowed = f"count tolerance {tolerance:.3g} plus print rounding"
        curves = {run.role: self.count_curve(run, T) for run in (self.reference, self.candidate)}
        problems = []
        for run, value in ((self.reference, a), (self.candidate, b)):
            curve = curves[run.role]
            row = (((run.block("qgeom").get("headline") or {}).get(T)) or {}).get("CNP")
            if not _number(value):
                if isinstance(row, dict):
                    problems.append(f"the {run.role} has a CNP row without a mu_CNP")
                continue
            full = self.full_grid(run)
            if isinstance(row, dict) and full is not None and len(full):
                nearest = float(full[int(np.argmin(np.abs(full - float(value) * 1e-3)))]) * 1e3
                if abs(float(row.get("mu_meV", math.nan)) - nearest) > 1e-9:
                    problems.append(
                        f"the {run.role}'s CNP row sits at {row.get('mu_meV')} meV, not at the grid mu "
                        f"nearest its mu_CNP ({nearest:g} meV)"
                    )
            if curve is not None and curve.stride == 1:
                count, rounding = curve.at(float(value) * 1e-3)
                if abs(count - curve.n_occ) > tolerance + rounding:
                    problems.append(
                        f"the {run.role}'s own count at its mu_CNP {value} meV is {count:.12g}, "
                        f"not n_occ {curve.n_occ} ({allowed})"
                    )
        if not (_number(a) and _number(b)):
            if a is None and b is None and not problems:
                self.quantity(name, AGREE, detail="the count reaches n_occ in neither run")
            elif (a is None and _number(b) or b is None and _number(a)) and not problems:
                self.cnp_edge(
                    name, T, a, b, curves[self.reference.role if a is None else self.candidate.role]
                )
            else:
                self.quantity(name, UNEXPLAINED, detail=_shorten("; ".join(problems + [f"{a!r} vs {b!r}"])))
            return
        self.cnp_shift[T] = (float(b) - float(a)) * 1e-3
        if a == b:
            self.quantity(
                name,
                UNEXPLAINED if problems else AGREE,
                float(len(problems)),
                0.0,
                _shorten("; ".join(problems) or f"{a} meV in both runs"),
            )
            return
        curve_a, curve_b = curves[self.reference.role], curves[self.candidate.role]
        if curve_a is None or curve_b is None:
            if curve_a is None and curve_b is None:
                self.precondition(
                    "mu_cnp_determined",
                    False,
                    f"{T}: mu_CNP {a} vs {b} meV and neither run has a count curve to judge the difference",
                )
            else:
                self.quantity(name, UNEXPLAINED, detail=f"{a} vs {b} meV; one run has no count curve")
            return
        mu_a, mu_b = float(a) * 1e-3, float(b) * 1e-3
        worst, reasons, same_gap = 0.0, [], True
        for role, curve, own, other in (
            (self.reference.role, curve_a, mu_a, mu_b),
            (self.candidate.role, curve_b, mu_b, mu_a),
        ):
            count, rounding = curve.at(other)
            if curve.stride == 1:
                deviation = abs(count - curve.n_occ)
            else:
                base, base_rounding = curve.at(own)
                deviation, rounding = abs(count - base), rounding + base_rounding
            worst = max(worst, deviation / (tolerance + rounding))
            reasons.append(f"the {role}'s count at {other * 1e3:.9g} meV deviates by {deviation:.3g}")
            between = (curve.mu >= min(mu_a, mu_b)) & (curve.mu <= max(mu_a, mu_b))
            off = np.abs(curve.electrons[between] - curve.n_occ) > tolerance + curve.rounding[between]
            same_gap = same_gap and bool(between.any()) and not bool(off.any())
        if worst > 1.0:
            problems.append(f"{a} vs {b} meV: " + "; ".join(reasons) + f" ({allowed})")
        if problems:
            self.quantity(name, UNEXPLAINED, worst, 1.0, _shorten("; ".join(problems)))
            return
        # A headline point the difference snaps to another grid mu is checked against the other run's curve.
        self.ties[f"CNP.{T}"] = "mu_CNP"
        if same_gap:
            detail = (
                f"{a} vs {b} meV: both lie in one gap of the count (n_occ within the {allowed} at every grid "
                "mu between them in both runs), where rounding picks the interpolated mu_CNP: equal"
            )
        else:
            detail = f"{a} vs {b} meV: each run's count at the other's mu_CNP is n_occ ({'; '.join(reasons)}; {allowed})"
        self.quantity(name, AGREE, worst, 1.0, _shorten(detail))

    def cnp_edge(self, name: str, T: str, a: Any, b: Any, curve: CountCurve | None) -> None:
        """One run without a mu_CNP: its count must never reach n_occ on the grid.

        Its count is monotonic in mu, so it misses n_occ at an edge of the mu range. Where
        that miss is within the count tolerance plus print rounding, whether mu_CNP lies on
        the grid is rounding: not comparable. A count that crosses n_occ inside the range,
        or misses it by more, fails.
        """
        missing, value = (self.reference, b) if a is None else (self.candidate, a)
        if curve is None:
            self.quantity(
                name, UNEXPLAINED, detail=f"{a!r} vs {b!r} meV; the {missing.role} has no count curve"
            )
            return
        tolerance = self.count_tolerance()
        low, high = int(np.argmin(curve.electrons)), int(np.argmax(curve.electrons))
        bottom = float(curve.electrons[low]) - float(curve.rounding[low])
        top = float(curve.electrons[high]) + float(curve.rounding[high])
        if bottom < curve.n_occ < top:
            self.quantity(
                name,
                UNEXPLAINED,
                detail=f"{a!r} vs {b!r} meV: the {missing.role}'s count reaches n_occ {curve.n_occ} on the grid",
            )
            return
        miss = min(abs(curve.n_occ - bottom), abs(curve.n_occ - top))
        if miss <= tolerance:
            self.precondition(
                "mu_cnp_determined",
                False,
                f"{T}: the {missing.role}'s count misses n_occ {curve.n_occ} by {miss:.3g} at the edge of the mu "
                f"range, within the count tolerance {tolerance:.3g}, where the other run places mu_CNP at "
                f"{value} meV: whether mu_CNP lies on the grid is decided by rounding",
            )
            return
        self.quantity(
            name,
            UNEXPLAINED,
            detail=f"{a!r} vs {b!r} meV: the {missing.role}'s count misses n_occ {curve.n_occ} by {miss:.3g}",
        )

    # ------------------------------------------------------------ headline
    def chi_tolerance(self, delta: float, scale_M: float, convention: str) -> float:
        """chi's amplitude to the bound of M's: its trigonal part is exactly -(c_out + 2 c_in) CHI_PER_M times
        M's, so it moves by |c_out + 2 c_in| CHI_PER_M sqrt(2) δ, plus the rounding of building and projecting
        chi (CHI_ROUNDING of the M scale, weighted by |c_out| + 2 |c_in|)."""
        c_out, c_in = CHI_SETS.get(convention, (2.0, 1.0))
        exact = abs(c_out + 2.0 * c_in) * math.sqrt(2.0) * delta
        rounding = CHI_ROUNDING * (abs(c_out) + 2.0 * abs(c_in)) * max(scale_M, delta)
        return CHI_PER_M * (exact + rounding)

    def curve_value(self, run: RunDirectory, T: str, key: str, mu_meV: float) -> float | None:
        """A run's 5-digit curve value at ``mu_meV`` (the nearest grid point), for a documented tie."""
        response = self.response(run)
        mu = _array(response.get("mu_eV"))
        if mu is None or not len(mu):
            return None
        j = int(np.argmin(np.abs(mu - mu_meV * 1e-3)))
        if key == "density_cm2":
            values = _array(((response.get("per_T") or {}).get(T) or {}).get("density_cm2"))
        else:
            values = self.level_curve(run, T, self.main_level(run), key)
        if values is None or j >= len(values):
            return None
        return float(values[j])

    def a3_near_tie(self, T: str, mu_a: float, mu_b: float, delta: float) -> bool:
        """Whether each run's A3 at the other's maximum lies within tolerance of its own maximum."""
        for run in (self.reference, self.candidate):
            top = self.curve_value(run, T, "A3", mu_a)
            other = self.curve_value(run, T, "A3", mu_b)
            if top is None or other is None:
                return False
            if abs(top - other) > float(self.printing(top) + self.printing(other)) + 2.0 * delta:
                return False
        return True

    def compare_headline(self, a: Mapping[str, Any], b: Mapping[str, Any]) -> None:
        if set(a) != set(b):
            self.quantity("qgeom.headline", UNEXPLAINED, detail=f"temperatures {sorted(a)} vs {sorted(b)}")
            return
        area = self.area()
        tol_density = self.count_tolerance() / (area * 1e-16)
        for T in sorted(a, key=lambda key: float(key[:-1])):
            rows_a, rows_b = a[T] or {}, b[T] or {}
            delta = self.delta(T)
            scale_M = delta / RELATIVE
            if set(rows_a) != set(rows_b):
                self.quantity(
                    f"qgeom.headline.{T}", UNEXPLAINED, detail=f"points {sorted(rows_a)} vs {sorted(rows_b)}"
                )
                continue
            for point in sorted(rows_a):
                self.compare_row(
                    f"qgeom.headline.{T}.{point}",
                    T,
                    point,
                    rows_a[point],
                    rows_b[point],
                    delta,
                    scale_M,
                    tol_density,
                )

    def compare_row(
        self,
        name: str,
        T: str,
        point: str,
        a: Mapping[str, Any],
        b: Mapping[str, Any],
        delta: float,
        scale_M: float,
        tol_density: float,
    ) -> None:
        mu_a, mu_b = float(a.get("mu_meV", math.nan)), float(b.get("mu_meV", math.nan))
        sigma_tol = RELATIVE * max(1.0, _max_abs(a.get("sigma_xy_e2_h"), b.get("sigma_xy_e2_h")))
        if abs(mu_a - mu_b) > 1e-9:
            documented = False
            if point == "CNP" and self.ties.get(f"CNP.{T}"):
                documented, why = (
                    True,
                    (
                        "each run's count is n_occ at both mu_CNP (qgeom.electron_count.mu_cnp_meV), and the "
                        "row sits at the grid mu nearest its own"
                    ),
                )
            elif point == "max_A3" and self.a3_near_tie(T, mu_a, mu_b, delta):
                documented, why = True, "A3 at the two maxima is within tolerance in both runs"
            if not documented:
                self.quantity(f"{name}.mu_meV", UNEXPLAINED, abs(mu_a - mu_b), 1e-9, f"{mu_a} vs {mu_b} meV")
                return
            self.quantity(
                f"{name}.mu_meV", DOCUMENTED, abs(mu_a - mu_b), None, f"{mu_a} vs {mu_b} meV: {why}"
            )
            # Each run's row against the other run's curve at the row's own mu.
            checks = [
                ("A3_A_per_eV", "A3", delta),
                ("sigma_xy_e2_h", "sigma_xy_e2_h", sigma_tol),
                ("density_cm2", "density_cm2", tol_density),
            ]
            for key, curve_key, tolerance in checks:
                for row, other, mu in ((a, self.candidate, mu_a), (b, self.reference, mu_b)):
                    value = self.curve_value(other, T, curve_key, mu)
                    self.curve(
                        f"{name}.{key}@{mu:g}meV.{other.role}_curve",
                        row.get(key),
                        value,
                        tolerance,
                        printed=self.printed,
                    )
            return
        self.quantity(f"{name}.mu_meV", AGREE, 0.0, 1e-9, f"{mu_a} meV")
        A3_a, A3_b = float(a.get("A3_A_per_eV") or 0.0), float(b.get("A3_A_per_eV") or 0.0)
        self.values(f"{name}.A3_A_per_eV", a.get("A3_A_per_eV"), b.get("A3_A_per_eV"), delta)
        self.values(f"{name}.A3_err", a.get("A3_err"), b.get("A3_err"), 2.0 * delta)
        theta_a, theta_b = a.get("theta0_deg"), b.get("theta0_deg")
        if isinstance(theta_a, (int, float)) and isinstance(theta_b, (int, float)):
            difference = abs(float(wrap_degrees(3.0 * (theta_a - theta_b)))) / 3.0
            tolerance = float(theta_tolerance(delta, np.array([min(A3_a, A3_b)]))[0])
            if math.isnan(tolerance):
                self.not_determined(
                    f"{name}.theta0_deg",
                    f"A3 {min(A3_a, A3_b):.3g} is within the M tolerance δ = {delta:.3g}, so the direction of "
                    "the trigonal part is not determined",
                )
                judged = [
                    key
                    for key in ("A3_A_per_eV", "A3_err", "sigma_xy_e2_h", "density_cm2")
                    if _array(a.get(key)) is not None and _array(b.get(key)) is not None
                ]
                if not judged and not (a.get("chi_mA_nm_per_V2") and b.get("chi_mA_nm_per_V2")):
                    self.precondition(
                        "determined", False, f"{name}: theta0 is undetermined and the row holds nothing else"
                    )
            else:
                self.bounded(
                    f"{name}.theta0_deg", difference, tolerance + 1e-9, "3 theta0 modulo 360 degrees"
                )
        else:
            self.values(f"{name}.theta0_deg", theta_a, theta_b, 0.0)
        chi_a, chi_b = a.get("chi_mA_nm_per_V2") or {}, b.get("chi_mA_nm_per_V2") or {}
        err_a, err_b = a.get("chi_err") or {}, b.get("chi_err") or {}
        if set(chi_a) != set(chi_b) or set(err_a) != set(err_b):
            self.quantity(
                f"{name}.chi", UNEXPLAINED, detail=f"conventions {sorted(chi_a)} vs {sorted(chi_b)}"
            )
        else:
            for convention in sorted(chi_a):
                tolerance = self.chi_tolerance(delta, scale_M, convention)
                self.values(f"{name}.chi.{convention}", chi_a[convention], chi_b[convention], tolerance)
                self.values(
                    f"{name}.chi_err.{convention}",
                    err_a.get(convention),
                    err_b.get(convention),
                    2.0 * tolerance,
                )
        self.values(f"{name}.sigma_xy_e2_h", a.get("sigma_xy_e2_h"), b.get("sigma_xy_e2_h"), sigma_tol)
        self.values(f"{name}.density_cm2", a.get("density_cm2"), b.get("density_cm2"), tol_density)
        if "full_over_diag_A3" in a or "full_over_diag_A3" in b:
            ratio_a, ratio_b = a.get("full_over_diag_A3"), b.get("full_over_diag_A3")
            if (
                isinstance(ratio_a, (int, float))
                and isinstance(ratio_b, (int, float))
                and A3_a > 0
                and ratio_a != 0
            ):
                delta_diag = self.delta(T, "diag")
                A3_diag = A3_a / float(ratio_a)
                tolerance = abs(float(ratio_a)) * (delta / A3_a + delta_diag / max(A3_diag, 1e-300))
                self.values(f"{name}.full_over_diag_A3", ratio_a, ratio_b, tolerance)
            else:
                self.values(f"{name}.full_over_diag_A3", ratio_a, ratio_b, 0.0)

    # ------------------------------------------------------- response curves
    def compare_response(self) -> None:
        ra, rb = self.response(self.reference), self.response(self.candidate)
        curve = functools.partial(self.curve, printed=self.printed)
        if not ra and not rb:
            self.quantity("qgeom.response", NOT_APPLICABLE, detail="no Out/qgeom_response.json in either run")
            return
        curve("qgeom.response.mu_eV", ra.get("mu_eV"), rb.get("mu_eV"), 1e-12)
        exact = [
            key for key in ("temperatures_K", "M_components", "conventions") if ra.get(key) != rb.get(key)
        ]
        units_a = {k: v for k, v in (ra.get("units") or {}).items() if isinstance(v, str)}
        units_b = {k: v for k, v in (rb.get("units") or {}).items() if isinstance(v, str)}
        if units_a != units_b:
            exact.append("units")
        self.quantity(
            "qgeom.response.settings",
            UNEXPLAINED if exact else AGREE,
            detail="; ".join(exact) or "temperatures, components, units and conventions equal",
        )
        per_a, per_b = ra.get("per_T") or {}, rb.get("per_T") or {}
        if set(per_a) != set(per_b):
            self.quantity(
                "qgeom.response.per_T", UNEXPLAINED, detail=f"temperatures {sorted(per_a)} vs {sorted(per_b)}"
            )
            return
        area = self.area()
        tol_density = self.count_tolerance() / (area * 1e-16)
        for T in sorted(per_a, key=lambda key: float(key[:-1])):
            ea, eb = per_a[T] or {}, per_b[T] or {}
            prefix = f"qgeom.response.{T}"
            delta = self.delta(T)
            curve(
                f"{prefix}.dos",
                ea.get("dos"),
                eb.get("dos"),
                RELATIVE * _max_abs(ea.get("dos"), eb.get("dos")),
            )
            curve(f"{prefix}.density_cm2", ea.get("density_cm2"), eb.get("density_cm2"), tol_density)
            curve(
                f"{prefix}.drude_eV",
                ea.get("drude_eV"),
                eb.get("drude_eV"),
                RELATIVE * _max_abs(ea.get("drude_eV"), eb.get("drude_eV")),
            )
            # The curve is mu - mu_CNP: the two differ by exactly the mu_CNP difference judged above.
            self.shifted_curve(
                f"{prefix}.mu_minus_cnp_eV",
                ea.get("mu_minus_cnp_eV"),
                eb.get("mu_minus_cnp_eV"),
                self.cnp_shift.get(T, 0.0),
            )
            err_a, err_b = ea.get("err") or {}, eb.get("err") or {}
            sigma_scale = RELATIVE * max(
                1.0,
                _max_abs(
                    self.level_curve(self.reference, T, self.main_level(self.reference), "sigma_xy_e2_h")
                ),
            )
            curve(f"{prefix}.err.A3", err_a.get("A3"), err_b.get("A3"), 2.0 * delta)
            curve(f"{prefix}.err.M", err_a.get("M"), err_b.get("M"), 2.0 * delta)
            curve(f"{prefix}.err.sigma_xy", err_a.get("sigma_xy"), err_b.get("sigma_xy"), 2.0 * sigma_scale)
            levels_a, levels_b = ea.get("levels") or {}, eb.get("levels") or {}
            if set(levels_a) != set(levels_b):
                self.quantity(
                    f"{prefix}.levels", UNEXPLAINED, detail=f"levels {sorted(levels_a)} vs {sorted(levels_b)}"
                )
                continue
            for level in sorted(levels_a):
                la, lb = levels_a[level] or {}, levels_b[level] or {}
                d = self.delta(T, "main" if level == self.main_level(self.reference) else level)
                name = f"{prefix}.{level}"
                curve(f"{name}.M", la.get("M"), lb.get("M"), d)
                curve(f"{name}.A3", la.get("A3"), lb.get("A3"), d)
                amplitude_a, amplitude_b = _array(la.get("A3")), _array(lb.get("A3"))
                if (
                    amplitude_a is not None
                    and amplitude_b is not None
                    and amplitude_a.shape == amplitude_b.shape
                ):
                    smaller = np.nan_to_num(np.minimum(np.abs(amplitude_a), np.abs(amplitude_b)))
                    # The 5-digit amplitudes may be larger than the values the angles came from.
                    lowest = np.maximum(smaller - 2.0 * self.printing(smaller), 0.0)
                    theta_tol: Any = theta_tolerance(d, lowest)
                else:
                    theta_tol = np.nan
                curve(
                    f"{name}.theta0_deg",
                    la.get("theta0_deg"),
                    lb.get("theta0_deg"),
                    theta_tol,
                    circular=True,
                    undetermined=f"A3 within the M tolerance δ = {d:.3g}: the trigonal direction is not determined",
                )
                sigma_tol = RELATIVE * max(1.0, _max_abs(la.get("sigma_xy_e2_h"), lb.get("sigma_xy_e2_h")))
                curve(f"{name}.sigma_xy_e2_h", la.get("sigma_xy_e2_h"), lb.get("sigma_xy_e2_h"), sigma_tol)
                curve(
                    f"{name}.BCD",
                    la.get("BCD"),
                    lb.get("BCD"),
                    RELATIVE * _max_abs(la.get("BCD"), lb.get("BCD")),
                )

    # ------------------------------------------------------------ the path
    def compare_qgeom_path(self) -> None:
        pa, pb = self.reference.out("qgeom_path.json"), self.candidate.out("qgeom_path.json")
        if pa is None and pb is None:
            self.quantity("qgeom.path", NOT_APPLICABLE, detail="no Out/qgeom_path.json in either run")
            return
        if not isinstance(pa, dict) or not isinstance(pb, dict):
            self.quantity("qgeom.path", UNEXPLAINED, detail="Out/qgeom_path.json written by one run only")
            return
        exact = [
            key for key in ("tick_indices", "tick_labels", "k_distance_note") if pa.get(key) != pb.get(key)
        ]
        self.quantity(
            "qgeom.path.ticks",
            UNEXPLAINED if exact else AGREE,
            detail=", ".join(exact) or "ticks and labels equal",
        )
        self.curve("qgeom.path.k_distance_invA", pa.get("k_distance_invA"), pb.get("k_distance_invA"), 1e-12)
        levels_a, levels_b = pa.get("levels") or {}, pb.get("levels") or {}
        if set(levels_a) != set(levels_b):
            self.quantity(
                "qgeom.path.levels", UNEXPLAINED, detail=f"levels {sorted(levels_a)} vs {sorted(levels_b)}"
            )
            return
        for level in sorted(levels_a):
            la, lb = levels_a[level] or {}, levels_b[level] or {}
            name = f"qgeom.path.{level}"
            if la.get("bands") != lb.get("bands"):
                self.quantity(f"{name}.bands", UNEXPLAINED, detail=f"{la.get('bands')} vs {lb.get('bands')}")
                continue
            self.curve(f"{name}.E_eV", la.get("E_eV"), lb.get("E_eV"), self.eigenvalue_tol)
            for key in ("v_eVA", "trace_g_A2", "omega_A2", "G_A2_per_eV"):
                self.curve(
                    f"{name}.{key}", la.get(key), lb.get(key), RELATIVE * _max_abs(la.get(key), lb.get(key))
                )
            sea_a, sea_b = la.get("fermi_sea_omega_A2") or {}, lb.get("fermi_sea_omega_A2") or {}
            if set(sea_a) != set(sea_b):
                self.quantity(
                    f"{name}.fermi_sea_omega_A2", UNEXPLAINED, detail=f"mu {sorted(sea_a)} vs {sorted(sea_b)}"
                )
                continue
            for mu in sorted(sea_a):
                tolerance = RELATIVE * _max_abs(sea_a[mu], sea_b[mu], la.get("omega_A2"))
                self.curve(f"{name}.fermi_sea_omega_A2.{mu}", sea_a[mu], sea_b[mu], tolerance)

    # ------------------------------------------------------ dipole sources
    def compare_dipole_sources(self, a: Any, b: Any) -> None:
        if a is None and b is None:
            self.quantity("qgeom.dipole_sources", NOT_APPLICABLE, detail="no dipole sources in either run")
            return
        if not isinstance(a, dict) or not isinstance(b, dict):
            self.quantity(
                "qgeom.dipole_sources", UNEXPLAINED, detail="dipole sources recorded by one run only"
            )
            return
        exact = [key for key in ("level", "T_K", "gap_bins_meV") if a.get(key) != b.get(key)]
        self.quantity(
            "qgeom.dipole_sources.settings",
            UNEXPLAINED if exact else AGREE,
            detail=", ".join(exact) or "level, temperature and bins equal",
        )
        self.values(
            "qgeom.dipole_sources.frozen_window_meV",
            a.get("frozen_window_meV"),
            b.get("frozen_window_meV"),
            1e-9,
        )
        points_a, points_b = a.get("points") or {}, b.get("points") or {}
        if set(points_a) != set(points_b):
            self.quantity(
                "qgeom.dipole_sources.points", UNEXPLAINED, detail=f"{sorted(points_a)} vs {sorted(points_b)}"
            )
            return
        delta = self.delta("0K")
        for point in sorted(points_a):
            ra, rb = points_a[point], points_b[point]
            name = f"qgeom.dipole_sources.{point}"
            if ra is None or rb is None:
                self.quantity(
                    name,
                    AGREE if ra is rb else UNEXPLAINED,
                    detail=f"{_shorten(repr(ra), 80)} vs {_shorten(repr(rb), 80)}",
                )
                continue
            if abs(float(ra.get("mu_meV", 0.0)) - float(rb.get("mu_meV", 0.0))) > 1e-9:
                tied = (point == "CNP" and self.ties.get("CNP.0K")) or (
                    point == "max_A3"
                    and self.a3_near_tie("0K", float(ra["mu_meV"]), float(rb["mu_meV"]), delta)
                )
                self.quantity(
                    name,
                    DOCUMENTED if tied else UNEXPLAINED,
                    detail=f"mu {ra.get('mu_meV')} vs {rb.get('mu_meV')} meV",
                )
                continue
            self.exact(f"{name}.record", ra, rb, ["mu_meV", "triangles", "reason"])
            A3 = min(float(ra.get("A3_A_per_eV") or 0.0), float(rb.get("A3_A_per_eV") or 0.0))
            self.values(f"{name}.A3_A_per_eV", ra.get("A3_A_per_eV"), rb.get("A3_A_per_eV"), delta)
            share_tol = min(2.0, 4.0 * delta / A3) if A3 > 0 else 2.0
            shares_a, shares_b = ra.get("shares") or {}, rb.get("shares") or {}
            if set(shares_a) != set(shares_b):
                self.quantity(
                    f"{name}.shares", UNEXPLAINED, detail=f"bins {sorted(shares_a)} vs {sorted(shares_b)}"
                )
            elif shares_a:
                keys = sorted(shares_a)
                self.values(
                    f"{name}.shares", [shares_a[k] for k in keys], [shares_b[k] for k in keys], share_tol
                )
            if "outside_frozen" in ra or "outside_frozen" in rb:
                self.values(
                    f"{name}.outside_frozen", ra.get("outside_frozen"), rb.get("outside_frozen"), share_tol
                )

    # ------------------------------------------------------------ the rest
    def compare_symmetry(self, a: Mapping[str, Any], b: Mapping[str, Any], qa, qb) -> None:
        if a.get("lattice_hexagonal") != b.get("lattice_hexagonal"):
            self.quantity(
                "qgeom.symmetry.lattice_hexagonal",
                UNEXPLAINED,
                detail=f"{a.get('lattice_hexagonal')} vs {b.get('lattice_hexagonal')}",
            )
        tau = 2.0 * self.eigenvalue_tol * 1e3
        self.values(
            "qgeom.symmetry.c3_band_floor_meV", a.get("c3_band_floor_meV"), b.get("c3_band_floor_meV"), tau
        )
        delta0 = self.delta("0K")
        rows_a = ((qa.get("headline") or {}).get("0K") or {}).get("E_F") or {}
        rows_b = ((qb.get("headline") or {}).get("0K") or {}).get("E_F") or {}
        A3_ef = min(float(rows_a.get("A3_A_per_eV") or 0.0), float(rows_b.get("A3_A_per_eV") or 0.0))
        bound = min(2.0, 2.0 * delta0 / A3_ef) if A3_ef > 0 else 2.0
        for key in ("M_c3_residual", "sigma_dT_residual"):
            self.values(f"qgeom.symmetry.{key}", a.get(key), b.get(key), bound + RESIDUAL_TOL)
        self.values(
            "qgeom.symmetry.bcd_residual",
            a.get("bcd_residual"),
            b.get("bcd_residual"),
            1e-6 * _max_abs(a.get("bcd_residual"), b.get("bcd_residual")) + 1e-9,
        )
        if "K_c3_residual" in a or "K_c3_residual" in b:
            occupied_a = qa.get("occupied") or {}
            K = _max_abs(occupied_a.get("K_in"))
            bound_k = min(2.0, 4.0 * RELATIVE) if K > 0 else 2.0
            self.values(
                "qgeom.symmetry.K_c3_residual",
                a.get("K_c3_residual"),
                b.get("K_c3_residual"),
                bound_k + RESIDUAL_TOL,
            )

    def compare_position(self, a: Mapping[str, Any], b: Mapping[str, Any]) -> None:
        exact = [key for key in ("mode", "source") if a.get(key) != b.get(key)]
        tb_a, tb_b = a.get("tb") or {}, b.get("tb") or {}
        exact += [
            f"tb.{key}" for key in ("bytes", "expected_bytes", "nrpts") if tb_a.get(key) != tb_b.get(key)
        ]
        self.quantity(
            "qgeom.position",
            UNEXPLAINED if exact else AGREE,
            detail=", ".join(exact) or f"mode {a.get('mode')!r}",
        )
        checks_a, checks_b = a.get("checks") or {}, b.get("checks") or {}
        if not checks_a and not checks_b:
            return
        if set(checks_a) != set(checks_b):
            self.quantity(
                "qgeom.position.checks", UNEXPLAINED, detail=f"{sorted(checks_a)} vs {sorted(checks_b)}"
            )
            return
        tolerances = {
            "h_vs_hr_max_meV": 2.0 * self.eigenvalue_tol * 1e3,
            "centres_inplane_max_A": 1e-9,
        }
        for key in sorted(checks_a):
            tolerance = tolerances.get(key, 1e-9 * _max_abs(checks_a[key], checks_b[key]) + 1e-15)
            self.values(f"qgeom.position.checks.{key}", checks_a[key], checks_b[key], tolerance)

    def compare_settings(self, qa: Mapping[str, Any], qb: Mapping[str, Any]) -> None:
        for key in ("approximation_level", "verdict", "verdict_reasons", "conventions", "files"):
            if qa.get(key) == qb.get(key):
                self.quantity(f"qgeom.{key}", AGREE, detail=_shorten(repr(qa.get(key)), 120))
            else:
                self.quantity(
                    f"qgeom.{key}", UNEXPLAINED, detail=_shorten(f"{qa.get(key)!r} vs {qb.get(key)!r}")
                )
        window_a, window_b = qa.get("window") or {}, qb.get("window") or {}
        self.exact(
            "qgeom.window", window_a, window_b, ["bands", "always_occupied", "degeneracy_tol_used_meV"]
        )
        pairs_a, pairs_b = window_a.get("pt_pairs") or {}, window_b.get("pt_pairs") or {}
        if pairs_a.get("detected") != pairs_b.get("detected"):
            self.quantity(
                "qgeom.window.pt_pairs",
                UNEXPLAINED,
                detail=f"{pairs_a.get('detected')} vs {pairs_b.get('detected')}",
            )
        else:
            # A median of level splittings: an eigenvalue difference.
            split_tol = 2.0 * self.eigenvalue_tol * 1e3
            self.values(
                "qgeom.window.pt_pairs",
                pairs_a.get("median_split_meV"),
                pairs_b.get("median_split_meV"),
                split_tol,
            )
        grid_a, grid_b = qa.get("grid") or {}, qb.get("grid") or {}
        self.exact(
            "qgeom.grid",
            grid_a,
            grid_b,
            ["mu_min_meV", "mu_max_meV", "mu_step_meV", "temperatures_K", "downsampled"],
        )
        ia, ib = qa.get("integration") or {}, qb.get("integration") or {}
        self.exact("qgeom.integration", ia, ib, ["tolerance", "base_mesh"])
        rel_a, rel_b = ia.get("smeared_check_rel"), ib.get("smeared_check_rel")
        self.values(
            "qgeom.integration.smeared_check_rel", rel_a, rel_b, 1e-6 * _max_abs(rel_a, rel_b) + 1e-12
        )

    def compare_dft_check(self, a: Any, b: Any) -> None:
        if a == b:
            self.quantity(
                "qgeom.dft_check", AGREE if a is not None else NOT_APPLICABLE, detail=_shorten(repr(a), 120)
            )
            return
        skipped = isinstance(a, dict) and "skipped" in a and self.reference.engine == "pythtb"
        if skipped and isinstance(b, dict) and "skipped" in b:
            self.quantity(
                "qgeom.dft_check",
                NOT_APPLICABLE,
                detail=_shorten(f"neither run checked the .eig: {a['skipped']}; {b['skipped']}"),
            )
            return
        if skipped and isinstance(b, dict) and "skipped" not in b:
            self.quantity(
                "qgeom.dft_check",
                DOCUMENTED,
                detail=_shorten(
                    "producer gap: the driver's .win parser stores no kpoints block, so its .eig check is "
                    f"always skipped ({a['skipped']}); QUEER's: {b}"
                ),
            )
            return
        if isinstance(a, dict) and isinstance(b, dict) and "skipped" not in a and "skipped" not in b:
            same = a.get("k_points") == b.get("k_points") and a.get("states") == b.get("states")
            same = same and a.get("counts_match") == b.get("counts_match")
            tolerance = 2.0 * self.eigenvalue_tol * 1e3
            if same:
                self.values("qgeom.dft_check", a.get("max_meV"), b.get("max_meV"), tolerance)
                return
        self.quantity("qgeom.dft_check", UNEXPLAINED, detail=_shorten(f"{a!r} vs {b!r}"))

    def compare_probe(self, a: Any, b: Any) -> None:
        own = b if isinstance(b, dict) else {}
        if self.candidate.engine == "queer":
            rel = own.get("max_rel")
            if own.get("skipped"):
                self.quantity("qgeom.parity.candidate", NOT_APPLICABLE, detail=str(own["skipped"]))
            else:
                self.bounded(
                    "qgeom.parity.candidate",
                    float(rel) if isinstance(rel, (int, float)) else None,
                    PROBE_TOLERANCE,
                    "the candidate's diagonal level against its own band geometry (an internal consistency check)",
                )
        detail = (
            f"each producer checks its diagonal level against its own library (reference: {_shorten(repr(a), 120)}); "
            "not compared"
        )
        self.quantity("qgeom.parity", NOT_APPLICABLE, detail=detail)
