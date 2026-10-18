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
12 history entries) is equal. A record must be its own run's mesh, and a QUEER
candidate without one fails against a reference that has one. The stop reason and
the convergence flags must be equal.

**Quantities.** With δ(T) = 1e-8 max(max A3, max |M|) over both runs at
temperature T: the occupied manifold's C and C_in to 1e-8 max(1, |C|), K_in, K_diag
and K_upper_loose to 1e-8 of their largest element, K_in - |C_in| to the sum, the
PSD ratio to 1e-8; the headline A3, A3_err to δ(T) (2 δ), theta0 to
asin(min(1, δ/A3))/3 on 3 theta0 modulo 360 degrees, the three chi amplitudes to
the bound δ propagates through their projection on M's trigonal direction,
sigma_xy to 1e-8 max(1, |sigma_xy|), the density to 1e-8 of the band count; a
headline point whose mu differs (mu_CNP inside a gap without states, where its
interpolation is undetermined, or a near-tie of the A3 maximum) is documented and
its values are checked against the other run's curves at its own mu. The per-T
curves of ``Out/qgeom_response.json`` and the path data of ``Out/qgeom_path.json``
are written with 5 significant digits: each value agrees to half a unit in that
digit of each run plus the quantity's 1e-8 tolerance. The dipole sources, the
symmetry residuals, the position checks, the grid, window, electron count and
verdict agree exactly or to the bounds derived from these. mu_CNP agrees to the
count tolerance over the smallest density of states between the two values; where
that bound exceeds the mu range (inside a gap without states, where the
interpolation of mu_CNP is undetermined) a difference is documented. Any difference
may snap a headline point to another grid mu, which is then documented.

**Producer gaps.** The driver never reads the ``.win``'s ``kpoints`` block, so its
``.eig`` check is always skipped where QUEER runs it: ``dft_check`` is documented.
The driver checks its diagonal level against PythTB and QUEER against its own band
geometry: ``parity`` is not compared, but the candidate's own check must pass. The
warnings of those two checks are left out of the warning comparison, and the
QUEER-only replay record out of the output comparison.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

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
from .qgeom_step import PROBE_TOLERANCE, VERTICES_FILE
from .results import WARN_DFT_FROZEN_MISMATCH, WARN_QGEOM_PROBE

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
RESIDUAL_TOL = 1e-12


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
    """The largest change of theta0 (degrees) a move of (alpha, beta) by ``delta`` allows at amplitude A3."""
    a = np.asarray(A3, float)
    ratio = np.where(a > 0, delta / np.where(a > 0, a, 1.0), 1.0)
    return np.degrees(np.arcsin(np.minimum(1.0, ratio))) / 3.0


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
        self.producer_warning_codes = frozenset({WARN_QGEOM_PROBE, WARN_DFT_FROZEN_MISMATCH})
        self.queer_only_outputs = frozenset({VERTICES_FILE})
        self.ties: dict[str, str] = {}
        # Per temperature: how far the two mu_CNP lie apart (eV), and where that is undetermined.
        self.cnp_shift: dict[str, float] = {}
        self.cnp_undetermined: set[str] = set()

    # ------------------------------------------------------- preconditions
    def step_preconditions(self) -> list[Callable[[], bool]]:
        return [self.check_qgeom_producer, self.check_stop_rule]

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
        value = run.out("qgeom_response.json")
        return value if isinstance(value, dict) else {}

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
        bands = (self.reference.block("qgeom").get("window") or {}).get("bands") or [0, 1]
        return RELATIVE * max(1.0, float(bands[-1]))

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
        self, name: str, a: Any, b: Any, derived: Any, *, circular: bool = False, detail: str = ""
    ) -> bool:
        """5-digit curves: |a - b| <= print(a) + print(b) + derived at every entry (metric: the worst ratio)."""
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
        keep = ~nan_a
        if circular:
            difference = np.abs(wrap_degrees(3.0 * (A - B))) / 3.0
        else:
            difference = np.abs(A - B)
        tolerance = (
            print_precision(A) + print_precision(B) + np.broadcast_to(np.asarray(derived, float), A.shape)
        )
        ratio = np.where(
            tolerance > 0,
            difference / np.where(tolerance > 0, tolerance, 1.0),
            np.where(difference > 0, np.inf, 0.0),
        )
        worst = float(np.max(ratio[keep])) if keep.any() else 0.0
        return self.bounded(name, worst, 1.0, detail or "worst |a - b| over print precision plus tolerance")

    # ------------------------------------------------------------ the step
    def compare_qgeom(self) -> None:
        qa, qb = self.reference.block("qgeom"), self.candidate.block("qgeom")
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
        record_a, record_b = (
            self.reference.out("qgeom_vertices.json"),
            self.candidate.out("qgeom_vertices.json"),
        )
        # Each record must be its own run's mesh; QUEER always writes one.
        inconsistent = [
            f"the {run.role}'s vertex record ({record.get('count')} vertices) is not its mesh "
            f"({integration.get('vertices')} vertices)"
            for run, record, integration in ((self.reference, record_a, ia), (self.candidate, record_b, ib))
            if isinstance(record, dict) and record.get("count") != integration.get("vertices")
        ]
        own = ib.get("vertex_set")
        if (
            isinstance(record_b, dict)
            and isinstance(own, dict)
            and own.get("sha256") != record_b.get("sha256")
        ):
            inconsistent.append("the candidate's vertex record differs from its integration.vertex_set")
        if self.candidate.engine == "queer" and isinstance(record_a, dict) and not isinstance(record_b, dict):
            inconsistent.append("the candidate wrote no Out/qgeom_vertices.json")
        if inconsistent:
            self.quantity(
                "qgeom.mesh", UNEXPLAINED, float(len(inconsistent)), 0.0, _shorten("; ".join(inconsistent))
            )
            return
        if isinstance(record_a, dict) and isinstance(record_b, dict):
            fields = ("count", "base", "unit", "sha256")
            differing = [
                f"{key}: {record_a.get(key)!r} vs {record_b.get(key)!r}"
                for key in fields
                if record_a.get(key) != record_b.get(key)
            ]
            keys_a, keys_b = record_a.get("keys"), record_b.get("keys")
            if differing and isinstance(keys_a, list) and isinstance(keys_b, list):
                only_a, only_b = set(keys_a) - set(keys_b), set(keys_b) - set(keys_a)
                differing.append(
                    f"{len(only_a)} vertices only in the reference, {len(only_b)} only in the candidate"
                )
            self.quantity(
                "qgeom.mesh",
                UNEXPLAINED if differing else AGREE,
                float(len(differing)),
                0.0,
                _shorten("; ".join(differing))
                or f"identical vertex sets: {record_a.get('count')} vertices, SHA-256 {str(record_a.get('sha256'))[:12]}…",
            )
            return
        differing = [
            f"{key}: {ia.get(key)!r} vs {ib.get(key)!r}" for key in MESH_FIELDS if ia.get(key) != ib.get(key)
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
        detail = "; ".join(differing) or (
            f"recorded meshes equal ({ia.get('vertices')} vertices, by rule {ia.get('vertices_by_rule')}); "
            "the reference records no vertex keys, so the sets are compared by their record"
        )
        self.quantity(
            "qgeom.mesh", UNEXPLAINED if differing else AGREE, float(len(differing)), 0.0, _shorten(detail)
        )

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

    def dos_curve(self, run: RunDirectory, T: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        response = self.response(run)
        mu = _array(response.get("mu_eV"))
        entry = (response.get("per_T") or {}).get(T) or {}
        return mu, _array(entry.get("dos"))

    def cnp_tolerance(self, T: str, low: float, high: float) -> float:
        """The count tolerance over the smallest DOS (both runs) on the grid between ``low`` and ``high`` (eV)."""
        smallest = math.inf
        for run in (self.reference, self.candidate):
            mu, dos = self.dos_curve(run, T)
            if mu is None or dos is None or mu.shape != dos.shape or not len(mu):
                return math.inf
            lo = max(int(np.searchsorted(mu, low)) - 1, 0)
            hi = min(int(np.searchsorted(mu, high)) + 1, len(mu))
            smallest = min(smallest, float(np.nanmin(np.abs(dos[lo:hi]))) if hi > lo else 0.0)
        if smallest <= 0.0:
            return math.inf
        return self.count_tolerance() / smallest * 1e3  # meV

    def compare_mu_cnp(self, T: str, a: Any, b: Any) -> None:
        name = f"qgeom.electron_count.mu_cnp_meV.{T}"
        if a is None or b is None:
            self.quantity(name, AGREE if a is b else UNEXPLAINED, detail=f"{a!r} vs {b!r}")
            return
        difference = abs(float(a) - float(b))
        self.cnp_shift[T] = difference * 1e-3
        if difference == 0.0:
            self.quantity(name, AGREE, 0.0, 0.0, f"{a} meV")
            return
        # Any difference can snap the headline point to another grid mu (a value at a midpoint).
        self.ties[f"CNP.{T}"] = "mu_CNP"
        tolerance = self.cnp_tolerance(T, min(a, b) * 1e-3, max(a, b) * 1e-3)
        mu = _array(self.response(self.reference).get("mu_eV"))
        span = float(np.ptp(mu)) * 1e3 if mu is not None and len(mu) else 0.0
        if difference <= 1e-9:
            self.quantity(name, AGREE, difference, 1e-9, f"{a} vs {b} meV")
        elif math.isinf(tolerance) or tolerance > span:
            self.cnp_undetermined.add(T)
            detail = (
                f"{a} vs {b} meV: the count is flat between them (no states: the interpolation of mu_CNP is "
                "undetermined there; the count tolerance over the DOS exceeds the mu range)"
            )
            self.quantity(name, DOCUMENTED, difference, None, detail)
        elif difference <= tolerance:
            detail = (
                f"{a} vs {b} meV within the count tolerance over the DOS between them ({tolerance:.3g} meV)"
            )
            self.quantity(name, AGREE, difference, tolerance, detail)
        else:
            self.quantity(name, UNEXPLAINED, difference, tolerance, f"{a} vs {b} meV")

    # ------------------------------------------------------------ headline
    def chi_tolerance(self, delta: float, scale_M: float, A3: float, convention: str) -> float:
        c_out, c_in = CHI_SETS.get(convention, (2.0, 1.0))
        weight = (abs(c_out) + 2.0 * abs(c_in)) * math.sqrt(2.0)
        trivial = 2.0 * CHI_PER_M * weight * max(scale_M, delta)
        if A3 <= 2.0 * math.sqrt(2.0) * delta:
            return trivial
        derived = CHI_PER_M * weight * (delta + 2.0 * math.sqrt(2.0) * max(scale_M, delta) * delta / A3)
        return min(trivial, derived)

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
            if abs(top - other) > float(print_precision(top) + print_precision(other)) + 2.0 * delta:
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
                documented, why = True, "mu_CNP is undetermined between the two"
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
                    self.curve(f"{name}.{key}@{mu:g}meV.{other.role}_curve", row.get(key), value, tolerance)
            return
        self.quantity(f"{name}.mu_meV", AGREE, 0.0, 1e-9, f"{mu_a} meV")
        A3_a, A3_b = float(a.get("A3_A_per_eV") or 0.0), float(b.get("A3_A_per_eV") or 0.0)
        self.values(f"{name}.A3_A_per_eV", a.get("A3_A_per_eV"), b.get("A3_A_per_eV"), delta)
        self.values(f"{name}.A3_err", a.get("A3_err"), b.get("A3_err"), 2.0 * delta)
        theta_a, theta_b = a.get("theta0_deg"), b.get("theta0_deg")
        if isinstance(theta_a, (int, float)) and isinstance(theta_b, (int, float)):
            difference = abs(float(wrap_degrees(3.0 * (theta_a - theta_b)))) / 3.0
            tolerance = float(theta_tolerance(delta, np.array([min(A3_a, A3_b)]))[0]) + 1e-9
            self.bounded(f"{name}.theta0_deg", difference, tolerance, "3 theta0 modulo 360 degrees")
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
                tolerance = self.chi_tolerance(delta, scale_M, min(A3_a, A3_b), convention)
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
        if not ra and not rb:
            self.quantity("qgeom.response", NOT_APPLICABLE, detail="no Out/qgeom_response.json in either run")
            return
        self.curve("qgeom.response.mu_eV", ra.get("mu_eV"), rb.get("mu_eV"), 1e-12)
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
            self.curve(
                f"{prefix}.dos",
                ea.get("dos"),
                eb.get("dos"),
                RELATIVE * _max_abs(ea.get("dos"), eb.get("dos")),
            )
            self.curve(f"{prefix}.density_cm2", ea.get("density_cm2"), eb.get("density_cm2"), tol_density)
            self.curve(
                f"{prefix}.drude_eV",
                ea.get("drude_eV"),
                eb.get("drude_eV"),
                RELATIVE * _max_abs(ea.get("drude_eV"), eb.get("drude_eV")),
            )
            undetermined = T in self.cnp_undetermined
            if (
                undetermined
                and ea.get("mu_minus_cnp_eV") is not None
                and eb.get("mu_minus_cnp_eV") is not None
            ):
                shift = _max_abs(
                    np.asarray(ea["mu_minus_cnp_eV"], float) - np.asarray(eb["mu_minus_cnp_eV"], float)
                )
                self.quantity(
                    f"{prefix}.mu_minus_cnp_eV",
                    DOCUMENTED,
                    shift,
                    None,
                    "shifted by the documented mu_CNP difference",
                )
            else:
                # The curve is mu - mu_CNP: it moves with mu_CNP, by no more than the two lie apart.
                self.curve(
                    f"{prefix}.mu_minus_cnp_eV",
                    ea.get("mu_minus_cnp_eV"),
                    eb.get("mu_minus_cnp_eV"),
                    self.cnp_shift.get(T, 0.0) + 1e-12,
                )
            err_a, err_b = ea.get("err") or {}, eb.get("err") or {}
            sigma_scale = RELATIVE * max(
                1.0,
                _max_abs(
                    self.level_curve(self.reference, T, self.main_level(self.reference), "sigma_xy_e2_h")
                ),
            )
            self.curve(f"{prefix}.err.A3", err_a.get("A3"), err_b.get("A3"), 2.0 * delta)
            self.curve(f"{prefix}.err.M", err_a.get("M"), err_b.get("M"), 2.0 * delta)
            self.curve(
                f"{prefix}.err.sigma_xy", err_a.get("sigma_xy"), err_b.get("sigma_xy"), 2.0 * sigma_scale
            )
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
                self.curve(f"{name}.M", la.get("M"), lb.get("M"), d)
                self.curve(f"{name}.A3", la.get("A3"), lb.get("A3"), d)
                amplitude_a, amplitude_b = _array(la.get("A3")), _array(lb.get("A3"))
                if (
                    amplitude_a is not None
                    and amplitude_b is not None
                    and amplitude_a.shape == amplitude_b.shape
                ):
                    smaller = np.nan_to_num(np.minimum(np.abs(amplitude_a), np.abs(amplitude_b)))
                    # The 5-digit amplitudes may be larger than the values the angles came from.
                    lowest = np.maximum(smaller - 2.0 * print_precision(smaller), 0.0)
                    theta_tol: Any = theta_tolerance(d, lowest)
                else:
                    theta_tol = 60.0
                self.curve(
                    f"{name}.theta0_deg", la.get("theta0_deg"), lb.get("theta0_deg"), theta_tol, circular=True
                )
                sigma_tol = RELATIVE * max(1.0, _max_abs(la.get("sigma_xy_e2_h"), lb.get("sigma_xy_e2_h")))
                self.curve(
                    f"{name}.sigma_xy_e2_h", la.get("sigma_xy_e2_h"), lb.get("sigma_xy_e2_h"), sigma_tol
                )
                self.curve(
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
