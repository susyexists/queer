"""The shift-current parity rule: a ``QUEER_SHIFT_CURRENT`` run against PyATB's ``SHIFT_CURRENT`` run.

QUEER's reported values are its physical estimators, and they are what is judged.
Every number that lets a difference from PyATB pass is derived here from PyATB's
own table, from an exact identity between the candidate's records and its value,
or from an independent recomputation of the scfout: no magnitude the candidate
supplies (a weight, a bound, an allowance) widens a tolerance.
:func:`compare_shift` judges a candidate directory against a PyATB reference
directory (``Input``, optional ``inputs.sha256``, ``Out/input.json``,
``Out/running.log`` and ``Out/Shift_Current/shift_current.dat``) in this order:

1. **Preconditions** (``not_comparable`` with the reason): the same scfout
   (SHA-256), the same settings as PyATB resolves them (``occ_band``, ``omega``,
   ``domega``, ``grid``, ``smearing_method``, ``eta`` unless there is no smearing,
   ``method``, ``n_occ``/``m_unocc`` as PyATB reads them, ``scissors``, ``nspin``),
   the same photon energies (the candidate's list equals PyATB's ``start + i *
   domega`` to 1e-9 eV, and PyATB's printed column equals it to its five
   decimals), the Gaussian ``eta -> eta / sqrt(2)`` and the bin width ``domega``,
   PyATB's spin factor (2 for ``nspin 1``, 1 for ``nspin 4``) and the lattice's
   handedness as the candidate's file sign.
2. **The candidate's records** (a failure is ``differs_unexplained`` and fails the
   comparison whatever PyATB printed; no cause explains a contradiction):

   - ``candidate_record``: the runner always writes ``diagnostics.regularisation``
     and, with ``smearing_method`` 0 or 1, ``diagnostics.pyatb_scheme`` with its
     ``untruncated_uA_V2`` (Gaussian), ``window_dropped_uA_V2`` (Gaussian without a
     band pair) or ``moved_transitions`` (bins), and with a flagged k-point its
     ``flagged_records``; a missing one is a contradiction, never an unmeasured cause;
   - ``regularisation_record``: its k-points are PyATB's grid; a k-point is flagged
     exactly when a pair spacing lies within PyATB's 1e-4 eV; without a flagged
     k-point there is no shift and the value without them is the value; with them,
     the recorded transitions of every flagged k-point, broadened on the compared
     kernel, sum to the value less the value without them, and their physical less
     their cut transitions to the shift;
   - ``emulation_consistency``: with PyATB's Gaussian, the candidate's untruncated
     table (PyATB's scale and constants) moved to QUEER's (:func:`move_to_queer_scale`,
     from its own curve) meets its value less its measured cut within that move's
     uncertainty, and its record of the transitions PyATB's window drops vanishes
     beyond 6 eta of the window's ends (no Gaussian reaches farther); with PyATB's
     bins, its emulation equals ``r`` times its histogram less the measured cut plus
     the recorded moves of the transitions PyATB's bins place in other rows (and of
     the cut's at the flagged k-points), each recorded row checked from its energy;
   - ``regularisation_recomputed`` (the caller's scfout, the candidate's by SHA-256,
     and a flagged k-point): every flagged k-point recomputed twice, QUEER's
     estimator through its own entry point (:func:`~queer.observables.nonlinear.shift_current_point`)
     and an independent transcription of PyATB's method-1 integrand
     (:mod:`~queer.compat.generator.pyatb_shift_integrand`); the recorded physical
     transitions must meet both, the recorded cut QUEER's, within tolerances derived
     from the recomputation (rounding over the k-point's absolute transition weight,
     times the conditioning ``1 + max|E| / min spacing`` for the transcription);
   - ``candidate_file``: its ``shift_current.dat`` holds what its document says.

3. **Symmetry gate** (a failure is ``differs_unexplained``): the components whose
   largest magnitude exceeds 1e-4 of the table's are the same in both runs, and
   each one's integral over omega has the same sign where both integrals exceed
   1e-4 of the component's integrated magnitude (:func:`symmetry_gate`), against
   QUEER's value or QUEER's value moved to PyATB's scheme (step 5). It is not judged
   (``not_applicable``) where PyATB's Gaussian is truncated above 1e-6 of its peak or an
   unrecomputed method-1 cut could change a sign.
4. **Clean control**: scissors 0, Gaussian smearing, ``method 1``, PyATB's
   truncated Gaussian tail ``exp(-(int(5 eta/domega) domega / eta)^2) <= 1e-6``, the
   photon energies reaching 5 eta beyond the extreme transitions, and no k-point
   where a band of a pair PyATB selects lies within 1e-3 eV of a level it is not
   exactly degenerate with (PyATB's ``get_D_degenerate`` regularizes ``|dE| <= 1e-4``
   eV; exact degeneracies, Kramers pairs among them, are skipped by both codes and do
   not count). Such a comparison is a clean control: every component must agree
   directly (step 5), else it is ``differs_unexplained``. Every ``differs_documented``
   needs a passed clean-control report of this build (``clean_control``: the same
   ``queer_version``, ``queer_record_sha256``, ``build_sha`` and
   ``shift_current.build.queer_source_sha256`` of the candidate, the SHA-256 of every
   ``.py`` file of the queer package that ran it, and this comparator's
   ``comparator_sha256``, the same digest of the package it runs from); without one,
   a comparison that does not agree directly is ``not_comparable`` ("no passed clean
   control for this build"), unless the candidate contradicts itself (step 2), which
   fails. A failed report of this build makes every component ``differs_unexplained``;
   a report of another build is ``not_applicable``.
5. **Classes**, per component, with ``tol = 1e-4 max|sigma_P| + PyATB's print
   rounding``: ``agree`` when PyATB's table meets QUEER's value
   (``agree_after_convention`` for the legacy single channel, a left-handed lattice or
   the histogram's ``x domega`` and half-bin transform); ``agree_after_convention``
   too when, with PyATB's Gaussian, its table moved to QUEER's Hartree constant and SI
   constants (:func:`move_to_queer_scale`: PyATB's own curve, its derivatives by
   finite differences, no candidate number) meets QUEER's value on every row within
   ``tol`` and the move's uncertainty, on the rows where that uncertainty is at most
   half of ``tol`` (elsewhere directly). Otherwise ``differs_documented``, reporting
   the physical gap, when PyATB's scheme reproduces it exactly (the plan's bounded
   causes, documented on the user's approval of 2026-10-17 without an unbounded cause
   or self-convergence): PyATB's table (moved to QUEER's scale, with a Gaussian) meets
   QUEER's value less the transitions PyATB's window drops (the candidate's record,
   tied to its untruncated table by step 2 and confined to the rows within 6 eta of
   the window's ends, which must leave a tenth of the component's size on the other
   rows), with PyATB's bins moved as recorded (bins), and with PyATB's method-1 cut at
   the flagged k-points recomputed from the scfout (step 2): QUEER's value there
   replaced by the transcription's cut value. Without the scfout (or the records) the
   cut is never documented on trust: PyATB's cut can change any value at the flagged
   k-points, so every component that does not agree directly is ``not_comparable``
   ("regularisation", with the reason), unless the candidate contradicts itself. A
   Gaussian truncated above 1e-6 of its peak, or one whose move to QUEER's scale is
   unresolved on the rows the difference needs while the gap lies within the reach of
   PyATB's Hartree scale (:func:`energy_scale_reach`), cannot be reproduced from QUEER's
   value: ``not_comparable`` ("pyatb_scheme_resolution"); a component PyATB's table
   contradicts beyond that reach fails all the same.
   Or through an unbounded cause (a nonzero ``scissors``, ``method 0``'s 0.04 eV
   regularization of every denominator, ``smearing_method 2``) with step 3 and
   **self-convergence**: ``|sigma_Q(N) - sigma_Q(2N)|_inf <= 0.1 max|sigma_Q(2N)|`` and
   ``max|sigma_Q|`` within a factor 10 of ``max|sigma_P|`` either way (else
   ``not_comparable``). Otherwise ``differs_unexplained``. A method-1 run needs no
   self-convergence estimate: its cut is recomputed, not converged away.
6. **Literature anchor** (:func:`literature_anchor`, run on its own): a GaAs
   sigma^xyz peak within 2 eta and 20 % of the published height at the paper's
   scissors and broadening.

What PyATB's window drops is QUEER's own computation: no PyATB run with that window
checks it, so a fault confined to those transitions is reported (the largest dropped
value, and the rows it reaches) and never verified.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ...physics.constants import E_CHARGE, HBAR
from .parity import (
    AFTER_CONVENTION,
    AGREE,
    DOCUMENTED,
    MALFORMED,
    NOT_APPLICABLE,
    PASS,
    UNEXPLAINED,
)
from .parity_pyatb import PyatbRun, QueerRun, _PyatbComparison
from .pyatb_conventions import (
    HARTREE_EV,
    PYATB_DEGENERATE_EV,
    PYATB_HARTREE_EV,
    PYATB_SHIFT_REGULARIZATION_EV,
    pyatb_shift_constant_ratio,
    pyatb_shift_spin_factor,
)
from .pyatb_files import SHIFT_COMPONENTS, SHIFT_CURRENT_FILE, parse_shift_current
from .results import package_source_sha256, sha256_file, to_jsonable

SYMMETRY_THRESHOLD = 1e-4
SIGN_SIGNIFICANCE = 1e-4
AGREEMENT_RELATIVE = 1e-4
PRINT_RELATIVE = 5.01e-7
OMEGA_TOLERANCE_EV = 1e-9
OMEGA_PRINT_EV = 5.01e-6
CLEAN_PAIR_GAP_EV = 1e-3
CLEAN_MARGIN_ETA = 5.0
CLEAN_TRUNCATION_TAIL = 1e-6
# Rounding of two summation orders over the same transitions, relative to the tables' size.
CONSISTENCY_FLOOR = 1e-9
# PyATB's table moved to QUEER's energy scale: finite differences on this many photon energies, used when the step
# is at most SCALE_MAX_STEP_ETA eta; the move's uncertainty is SCALE_MARGIN_FACTOR times the change from the next
# coarser stencil plus SCALE_MOVE_SHARE (2 domega / eta)^6 of the move itself (the measured error of the move on
# random transitions grows as that power of the step); a row's move counts where that is at most SCALE_RESOLVED of tol.
SCALE_STENCIL = 7
SCALE_CHECK_STENCIL = 5
SCALE_MARGIN_FACTOR = 2.0
SCALE_MOVE_SHARE = 0.05
SCALE_MAX_STEP_ETA = 0.4
SCALE_RESOLVED = 0.5
# Where the move is unresolved, PyATB's Hartree scale moves its table by at most (1 - s) (|P| + this many times its
# local magnitude (|w| / eta + 1)) (energy_scale_reach): a larger gap is no question of PyATB's Hartree scale.
SCALE_REACH_FACTOR = 3.0
# A Gaussian of PyATB's eta reaches this many eta: beyond it a dropped transition adds at most e^-36 of its peak.
WINDOW_REACH_ETA = 6.0
# The rows PyATB's dropped transitions cannot reach must hold at least this share of a component's size.
INFORMATIVE_SHARE = 0.1
# Recomputing a flagged k-point: QUEER's estimator at its own entry point differs from the record by summation
# order (this times the k-point's absolute transition weight at the kernel's peak); the transcription of PyATB's
# integrand by rounding amplified by the conditioning 1 + max|E| / (smallest spacing).
RECOMPUTE_ROUNDING = 1e-12
RECOMPUTE_CONDITIONING = 1e-13
# The candidate's sums over the flagged k-points' transitions round to this share of their absolute weight at the
# kernel's peak (near-degenerate transitions carry large weights of both signs that cancel in the spectrum).
RECORD_ROUNDING = 1e-13
SELF_CONVERGENCE_RELATIVE = 0.1
MAGNITUDE_FACTOR = 10.0
ANCHOR_PEAK_ETA = 2.0
ANCHOR_HEIGHT_RELATIVE = 0.2
UNBOUNDED_CAUSES = ("scissors", "regularisation", "adaptive_smearing")
# The candidate's checks of itself: a failure is a contradiction no cause explains.
SELF_CHECKS = (
    "candidate_record",
    "regularisation_record",
    "emulation_consistency",
    "regularisation_recomputed",
    "candidate_file",
    "compare_candidate_file",
    "compare_shift",
)
DIRECT = (AGREE, AFTER_CONVENTION)
NO_CLEAN_CONTROL = "no passed clean control for this build"
SQRT_PI = math.sqrt(math.pi)
ENERGY_SCALE = PYATB_HARTREE_EV / HARTREE_EV
BUILD_KEYS = ("queer_version", "queer_record_sha256", "build_sha", "queer_source_sha256", "comparator_sha256")


def comparator_sha256() -> str | None:
    """SHA-256 of every ``.py`` file of the queer package this comparator runs from: a clean-control report
    vouches for the rules that judged it and everything they import."""
    return package_source_sha256()


def table(columns: Mapping[str, Any] | None) -> np.ndarray | None:
    """``{component: column}`` as a ``(len(omega), 18)`` array in PyATB's order; None for None."""
    if columns is None:
        return None
    return np.column_stack([np.asarray(columns[name], float) for name in SHIFT_COMPONENTS])


def gaussian(offset: Any, eta: float) -> np.ndarray:
    """PyATB's Gaussian ``exp(-(x / eta)^2) / (eta sqrt(pi))``."""
    return np.exp(-((np.asarray(offset, float) / eta) ** 2)) / (eta * SQRT_PI)


def _stencil(offsets: np.ndarray, order: int) -> np.ndarray:
    """Finite-difference weights of the ``order``-th derivative on integer ``offsets`` (unit step)."""
    matrix = np.vander(offsets.astype(float), len(offsets), increasing=True).T
    target = np.zeros(len(offsets))
    target[order] = math.factorial(order)
    return np.linalg.solve(matrix, target)


def finite_difference(values: Any, step: float, order: int, points: int) -> np.ndarray:
    """The ``order``-th derivative of rows of ``values`` on a uniform grid of ``step``: centred stencils of
    ``points`` rows inside, the same width shifted onto the grid at its ends."""
    values = np.asarray(values, float)
    count = len(values)
    if count < points:
        raise ValueError(f"{count} rows are too few for a {points}-point stencil")
    half = points // 2
    out = np.empty_like(values)
    weights = _stencil(np.arange(-half, half + 1), order)
    interior = np.zeros_like(values[half : count - half])
    for index, weight in enumerate(weights):
        interior += weight * values[index : count - points + 1 + index]
    out[half : count - half] = interior
    for row in (*range(half), *range(count - half, count)):
        low = min(max(row - half, 0), count - points)
        out[row] = _stencil(np.arange(low, low + points) - row, order) @ values[low : low + points]
    return out / step**order


def move_to_queer_scale(
    values: Any, *, omega: Any, eta: float, domega: float
) -> tuple[np.ndarray, np.ndarray]:
    """A table on PyATB's Hartree scale and constants moved to QUEER's, and the uncertainty of the move.

    PyATB places a transition ``E`` at ``s E`` (``s = 27.211324570273 / HARTREE_EV``) and
    multiplies by ``r``, so its Gaussian table is ``P(w) = (r / s) X_{eta/s}(w / s)`` with
    ``X_eta`` QUEER's spectrum at width ``eta``. Hence ``r X_eta(w) = s [P(s w) - c P''(s
    w) + c^2 / 2 P''''(s w)]`` with ``c = eta^2 (1 - s^2) / 4`` (the heat flow that turns
    width ``eta / s`` into ``eta``), and ``P(s w) = P(w - a)``, ``a = (1 - s) w``, by its
    Taylor series. The derivatives are finite differences of the table itself
    (:data:`SCALE_STENCIL` points); the uncertainty is :data:`SCALE_MARGIN_FACTOR` times
    the change from the :data:`SCALE_CHECK_STENCIL`-point stencils plus
    :data:`SCALE_MOVE_SHARE` ``(2 domega / eta)^6`` of the move. It is an a posteriori
    estimate, not a bound: on random transitions with near-cancelling pairs, moves of up
    to 2 % of the peak and steps up to :data:`SCALE_MAX_STEP_ETA` eta, the move's error on
    the rows where this estimate is below half the comparison's tolerance stayed below
    that tolerance plus the estimate. Only the photon energies, ``eta`` and the constants
    enter beside the table.
    """
    values = np.asarray(values, float)
    shift = (1 - ENERGY_SCALE) * np.asarray(omega, float)[:, None]
    heat = eta**2 * (1 - ENERGY_SCALE**2) / 4
    ratio = pyatb_shift_constant_ratio()

    def moved(points: int) -> np.ndarray:
        first, second, third, fourth = (
            finite_difference(values, domega, order, points) for order in (1, 2, 3, 4)
        )
        body = (
            values
            - shift * first
            + shift**2 / 2 * second
            - shift**3 / 6 * third
            + shift**4 / 24 * fourth
            - heat * (second - shift * third + shift**2 / 2 * fourth)
            + heat**2 / 2 * fourth
        )
        return ENERGY_SCALE * body / ratio

    fine = moved(SCALE_STENCIL)
    share = SCALE_MOVE_SHARE * (2 * domega / eta) ** 6
    return fine, SCALE_MARGIN_FACTOR * np.abs(fine - moved(SCALE_CHECK_STENCIL)) + share * np.abs(
        fine - values / ratio
    )


def energy_scale_reach(values: Any, *, omega: Any, eta: float, domega: float) -> np.ndarray:
    """How far PyATB's Hartree scale can move its table where the move is unresolved, from the table alone.

    The move is ``(1 - s) (P + w P' + eta^2 / 2 P'')`` to first order. A Gaussian sum of
    width eta changes on that scale: its slope stays within ``SCALE_REACH_FACTOR M / eta``
    and its curvature within ``2 SCALE_REACH_FACTOR M / eta^2``, ``M`` the table's largest
    magnitude within 2 eta of the row, unless transitions cancel to far below their own
    weight. So the reach is ``(1 - s) (|P| + SCALE_REACH_FACTOR M (|w| / eta + 1))`` plus
    ``|1 - r| |P|``: a gap beyond it is no question of PyATB's Hartree scale.
    """
    values = np.asarray(values, float)
    magnitude = np.abs(values)
    span = max(1, math.ceil(2 * eta / domega))
    local = magnitude.copy()
    for offset in range(1, span + 1):
        local[offset:] = np.maximum(local[offset:], magnitude[:-offset])
        local[:-offset] = np.maximum(local[:-offset], magnitude[offset:])
    omega = np.abs(np.asarray(omega, float))[:, None]
    size = (1 - ENERGY_SCALE) * (magnitude + SCALE_REACH_FACTOR * local * (omega / eta + 1))
    return size + abs(1 - pyatb_shift_constant_ratio()) * magnitude


def broaden(energies: Any, weights: Any, omega: Any, eta: float) -> np.ndarray:
    """``sum_t w_t G(w_i - E_t)`` with PyATB's Gaussian (QUEER's kernel at its standard deviation eta/sqrt(2))."""
    energies = np.asarray(energies, float)
    weights = np.asarray(weights, float).reshape(len(energies), len(SHIFT_COMPONENTS))
    return gaussian(np.asarray(omega, float)[:, None] - energies[None, :], eta) @ weights


def queer_rows(energies: Any, start: float, domega: float, count: int) -> np.ndarray:
    """QUEER's histogram row of each transition: ``E`` in ``(w_i, w_i + domega]``, or -1."""
    position = (np.asarray(energies, float) - start) / domega
    rows = np.ceil(position).astype(int) - 1
    return np.where((rows >= 0) & (rows < count), rows, -1)


def pyatb_rows(energies: Any, start: float, domega: float, count: int, pair_mode: bool) -> np.ndarray:
    """PyATB's bin of each transition (QUEER's eV): ``int((s E - start) / domega)`` in its window, or -1."""
    scaled = np.asarray(energies, float) * ENERGY_SCALE
    rows = np.trunc((scaled - start) / domega).astype(int)
    inside = (
        np.ones(len(scaled), bool) if pair_mode else (scaled >= start) & (scaled < start + count * domega)
    )
    return np.where(inside & (rows >= 0) & (rows < count), rows, -1)


def place(rows: np.ndarray, weights: Any, count: int) -> np.ndarray:
    """The ``(count, 18)`` table of weights added to their rows (-1: none)."""
    weights = np.asarray(weights, float).reshape(len(rows), len(SHIFT_COMPONENTS))
    out = np.zeros((count, len(SHIFT_COMPONENTS)))
    keep = rows >= 0
    np.add.at(out, rows[keep], weights[keep])
    return out


def symmetry_gate(
    reference: np.ndarray, candidate: np.ndarray, domega: float, tolerance: np.ndarray | None = None
) -> tuple[bool, dict[str, Any]]:
    """Gate 3: the same significant components and the same sign of each one's integral over omega.

    A component is significant in a run when its largest magnitude exceeds
    :data:`SYMMETRY_THRESHOLD` of the run's; its integral's sign counts when the
    integral exceeds :data:`SIGN_SIGNIFICANCE` of its integrated magnitude in both
    runs. ``tolerance`` (per component) is the agreement allowed at one photon
    energy: a component significant in one run only, or whose integrals' signs
    differ, fails the gate only when the runs differ by more than it (its largest
    magnitudes by more than ``tolerance``, its integrals by more than ``tolerance``
    times the photon-energy range), so values that agree are never split by a
    threshold between them.
    """
    width = len(SHIFT_COMPONENTS)
    allowed = np.zeros(width) if tolerance is None else np.asarray(tolerance, float)
    facts: dict[str, Any] = {"threshold": SYMMETRY_THRESHOLD, "sign_significance": SIGN_SIGNIFICANCE}
    sets, peaks = [], []
    for values in (reference, candidate):
        largest = float(np.max(np.abs(values))) if values.size else 0.0
        column = np.max(np.abs(values), axis=0) if values.size else np.zeros(width)
        peaks.append(column)
        sets.append(
            {name for name, value in zip(SHIFT_COMPONENTS, column) if value > SYMMETRY_THRESHOLD * largest}
        )
    facts["reference_components"] = sorted(sets[0], key=SHIFT_COMPONENTS.index)
    facts["candidate_components"] = sorted(sets[1], key=SHIFT_COMPONENTS.index)
    span = len(reference) * domega
    integrals = [values.sum(axis=0) * domega for values in (reference, candidate)]
    magnitudes = [np.abs(values).sum(axis=0) * domega for values in (reference, candidate)]
    unmatched, flipped = [], []
    for index, name in enumerate(SHIFT_COMPONENTS):
        if (name in sets[0]) != (name in sets[1]) and abs(peaks[0][index] - peaks[1][index]) > allowed[index]:
            unmatched.append(name)
        if name not in sets[0] & sets[1]:
            continue
        significant = all(
            abs(integral[index]) > SIGN_SIGNIFICANCE * magnitude[index]
            for integral, magnitude in zip(integrals, magnitudes)
        )
        apart = abs(integrals[0][index] - integrals[1][index]) > allowed[index] * span
        if significant and apart and np.sign(integrals[0][index]) != np.sign(integrals[1][index]):
            flipped.append(name)
    facts["components_differ"] = unmatched
    facts["integral_sign_differs"] = flipped
    facts["integrals_reference"] = dict(zip(SHIFT_COMPONENTS, integrals[0].tolist()))
    facts["integrals_candidate"] = dict(zip(SHIFT_COMPONENTS, integrals[1].tolist()))
    return not unmatched and not flipped, facts


def load_report(value: Any) -> Mapping[str, Any] | None:
    """A parity report given as a mapping or as the path of its JSON file."""
    if value is None or isinstance(value, Mapping):
        return value
    return json.loads(Path(value).read_text(encoding="utf-8"))


def _largest(values: Any) -> float:
    return float(np.max(np.abs(np.asarray(values, float)), initial=0.0))


def _transitions(part: Mapping[str, Any] | None) -> tuple[np.ndarray, np.ndarray] | None:
    """A recorded ``{energy_eV, weight_uA_V2_eV}`` as arrays, or None."""
    if part is None:
        return None
    energies = np.asarray(part["energy_eV"], float).reshape(-1)
    weights = np.asarray(part["weight_uA_V2_eV"], float).reshape(len(energies), len(SHIFT_COMPONENTS))
    if not np.all(np.isfinite(energies)) or not np.all(np.isfinite(weights)):
        raise ValueError("a recorded transition is not finite")
    return energies, weights


@dataclass
class FlaggedCut:
    """PyATB's method-1 cut at the flagged k-points, as the comparison could establish it."""

    verified: bool
    reason: str
    # QUEER's value at the flagged k-points less PyATB's cut value there (compared kernel), when verified.
    change: np.ndarray | None = None
    details: dict[str, Any] = field(default_factory=dict)


class _ShiftComparison(_PyatbComparison):
    def __init__(
        self,
        reference: PyatbRun,
        candidate: QueerRun,
        scfout: str | Path | None = None,
        clean_control: Any = None,
    ):
        super().__init__("SHIFT_CURRENT", reference, candidate, scfout)
        self.tier = "shift"
        self.clean_control_report = clean_control
        self.rules: dict[str, Any] = {}
        self.basis: dict[str, Any] = {}
        self.spin_factor = 1
        self.clean = False
        self.release = "not_supplied"
        self.direct_everywhere = True

    # ----------------------------------------------------- preconditions
    @property
    def block(self) -> dict[str, Any]:
        return self.candidate.block("shift_current")

    @property
    def effective(self) -> dict[str, Any]:
        return self.notes.get("parameters", {}).get("function", {})

    @property
    def pair_mode(self) -> bool:
        return int(self.effective.get("n_occ", -1)) >= 1

    @property
    def smearing(self) -> int:
        return int(self.effective["smearing_method"])

    @property
    def eta(self) -> float:
        return float(self.effective.get("eta") or 0.0)

    @property
    def domega(self) -> float:
        return float(self.effective["domega"])

    @property
    def kpoints(self) -> int:
        return int(np.prod([int(value) for value in self.effective["grid"]]))

    def reference_omega(self) -> list[float]:
        start, end = (float(value) for value in self.effective["omega"])
        step = float(self.effective["domega"])
        return [start + step * index for index in range(int((end - start) / step) + 1)]

    def window(self) -> tuple[float, float]:
        """PyATB's window ``[start, start + count domega)`` on its scale."""
        omega = self.reference_omega()
        return omega[0], omega[0] + len(omega) * self.domega

    def truncation_tail(self) -> float:
        """``exp(-(int(5 eta / domega) domega / eta)^2)``: the largest share of a transition's peak PyATB's truncated
        Gaussian drops."""
        if self.smearing != 1 or self.eta <= 0:
            return 1.0
        return math.exp(-(((int(5 * self.eta / self.domega) * self.domega) / self.eta) ** 2))

    def check_omega(self) -> bool:
        expected = np.asarray(self.reference_omega())
        candidate = np.asarray(self.block.get("omega_eV") or [], float)
        printed = np.asarray((self.reference.product or {}).get("omega_eV") or [], float)
        problems = []
        if candidate.shape != expected.shape:
            problems.append(
                f"the candidate has {len(candidate)} photon energies, PyATB's list {len(expected)}"
            )
        elif np.max(np.abs(candidate - expected), initial=0.0) > OMEGA_TOLERANCE_EV:
            problems.append(
                f"the candidate's photon energies differ from PyATB's by up to "
                f"{np.max(np.abs(candidate - expected)):.3g} eV (allowed {OMEGA_TOLERANCE_EV:g})"
            )
        if printed.shape != expected.shape:
            problems.append(f"PyATB printed {len(printed)} rows for its {len(expected)} photon energies")
        elif np.max(np.abs(printed - expected), initial=0.0) > OMEGA_PRINT_EV:
            problems.append("PyATB's printed photon energies are not its start + i * domega")
        detail = "; ".join(problems) or (
            f"{len(expected)} photon energies {expected[0]:.5f}..{expected[-1]:.5f} eV, step "
            f"{self.effective['domega']} eV, in both runs"
        )
        return self.precondition("omega", not problems, detail)

    def check_spin(self) -> bool:
        nspin = int(self.notes.get("parameters", {}).get("input", {}).get("nspin") or 0)
        model = self.candidate.block("model")
        nspin = nspin or (4 if model.get("spin_mode") == "soc" else 1)
        expected = pyatb_shift_spin_factor(nspin)
        recorded = (self.block.get("convention") or {}).get("pyatb_spin_factor")
        self.spin_factor = expected
        self.legacy = model.get("spin_mode") == "collinear"
        ok = recorded == expected and (
            not self.legacy or isinstance(self.block.get("legacy_single_channel"), dict)
        )
        detail = (
            f"PyATB's spin factor {expected} (nspin {nspin}); the candidate's {recorded}, spin mode "
            f"{model.get('spin_mode')}"
        )
        if self.legacy:
            detail += ": compared with legacy_single_channel (twice the spin-up channel)"
        return self.precondition("spin_factor", ok, detail)

    def check_smearing(self) -> bool:
        smearing = self.block.get("smearing") or {}
        method = int(self.effective["smearing_method"])
        if method == 1:
            expected = float(self.effective["eta"]) / math.sqrt(2)
            recorded = smearing.get("gaussian_sigma_eV")
            ok = recorded is not None and abs(float(recorded) - expected) <= 1e-12 * expected
            detail = f"Gaussian eta {self.effective['eta']} eV -> standard deviation {expected:.12g} eV; candidate {recorded}"
        elif method == 0:
            expected = float(self.effective["domega"])
            recorded = smearing.get("bin_width_eV")
            ok = recorded is not None and abs(float(recorded) - expected) <= 1e-12 * expected
            detail = f"no smearing: histogram bins of domega {expected} eV; candidate {recorded}"
        else:
            ok = smearing.get("smearing_method") == 2
            detail = "adaptive smearing in both runs (QUEER's own width)"
        ok = ok and smearing.get("smearing_method") == method
        return self.precondition("smearing", ok, detail)

    def check_sign(self) -> bool:
        handedness = self.candidate_handedness()
        recorded = (self.block.get("convention") or {}).get("export_sign")
        self.sign = handedness
        detail = f"lattice handedness {handedness:+d}; the candidate's file sign {recorded}"
        if handedness < 0:
            detail += " (PyATB's signed cell volume flips every component it prints)"
        return self.precondition("sign", recorded == handedness, detail)

    # ------------------------------------------------------------ tables
    def compared_values(self) -> tuple[np.ndarray, str]:
        """QUEER's value in PyATB's conventions (export sign, spin factor, legacy channel, bins) and its name."""
        variant = self.block["legacy_single_channel"] if self.legacy else self.block
        name = "legacy_single_channel." if self.legacy else ""
        binned = self.smearing == 0
        values = table(variant["binned_uA_V2" if binned else "sigma_uA_V2"])
        name += "binned_uA_V2" if binned else "sigma_uA_V2"
        assert values is not None
        return self.sign * values, name

    def kernel(self, energies: Any, weights: Any, *, pyatb: bool = False) -> np.ndarray:
        """Transitions (QUEER's eV, weights in microA/V^2 eV) on the compared kernel: PyATB's Gaussian at QUEER's
        energies, or the bins (QUEER's histogram rows, or PyATB's with ``pyatb``)."""
        omega = np.asarray(self.reference_omega())
        energies = np.asarray(energies, float)
        if self.smearing == 1:
            return broaden(energies, weights, omega, self.eta)
        if pyatb:
            rows = pyatb_rows(energies, float(omega[0]), self.domega, len(omega), self.pair_mode)
        else:
            rows = queer_rows(energies, float(omega[0]), self.domega, len(omega))
        return place(rows, weights, len(omega))

    def kernel_peak(self) -> float:
        """The compared kernel's largest value per unit transition weight: the Gaussian's peak or one bin."""
        return 1.0 / (self.eta * SQRT_PI) if self.smearing == 1 else 1.0

    # ----------------------------------------------------- the records
    def check_candidate_record(self, diagnostics: Mapping[str, Any]) -> bool:
        """``candidate_record``: what the runner always writes is there; a missing record is a contradiction."""
        problems = []
        binned = self.smearing == 0
        regularisation = diagnostics.get("regularisation")
        if not isinstance(regularisation, Mapping):
            problems.append("no diagnostics.regularisation")
        else:
            for key in (
                ("binned_shift_uA_V2", "binned_unflagged_uA_V2")
                if binned
                else (
                    "shift_uA_V2",
                    "sigma_unflagged_uA_V2",
                )
            ):
                if not isinstance(regularisation.get(key), Mapping):
                    problems.append(f"no regularisation.{key}")
            flagged = int(regularisation.get("flagged_kpoints") or 0)
            if (
                int(self.effective["method"]) == 1
                and flagged
                and not isinstance(regularisation.get("flagged_records"), Mapping)
            ):
                problems.append(f"no regularisation.flagged_records for its {flagged} flagged k-points")
        if self.smearing in (0, 1):
            scheme = diagnostics.get("pyatb_scheme")
            if not isinstance(scheme, Mapping) or not isinstance(scheme.get("sigma_uA_V2"), Mapping):
                problems.append("no diagnostics.pyatb_scheme.sigma_uA_V2")
            else:
                if self.smearing == 1 and not isinstance(scheme.get("transitions_uA_V2"), Mapping):
                    problems.append("no pyatb_scheme.transitions_uA_V2")
                if (
                    self.smearing == 1
                    and not self.pair_mode
                    and not isinstance(scheme.get("window_dropped_uA_V2"), Mapping)
                ):
                    problems.append("no pyatb_scheme.window_dropped_uA_V2")
                if binned and not isinstance(scheme.get("moved_transitions"), Mapping):
                    problems.append("no pyatb_scheme.moved_transitions")
        self.quantity(
            "candidate_record",
            AGREE if not problems else UNEXPLAINED,
            detail="the records the runner always writes for this smearing and method are present"
            + ("" if not problems else ": they are not, " + "; ".join(problems)),
        )
        return not problems

    def record_rounding(self, regularisation: Mapping[str, Any]) -> float:
        """:data:`RECORD_ROUNDING` times the flagged k-points' absolute transition weight at the kernel's peak: the
        rounding of the candidate's own sums over them (a floor of its self-checks, never an allowance against
        PyATB)."""
        total = 0.0
        for item in self.flagged_items(regularisation) or []:
            for part in (item.get("physical"), item.get("cut")):
                if part is not None:
                    total += float(np.abs(np.asarray(part["weight_uA_V2_eV"], float)).sum())
        return RECORD_ROUNDING * total * self.kernel_peak()

    def flagged_items(self, regularisation: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
        """The recorded flagged k-points, or None when the run kept none (its record's limit)."""
        records = regularisation.get("flagged_records") or {}
        if not records.get("recorded"):
            return None
        return list(records.get("kpoints") or [])

    def check_regularisation_record(
        self, diagnostics: Mapping[str, Any], values: np.ndarray, shift: np.ndarray, unflagged: np.ndarray
    ) -> None:
        """``regularisation_record``: consistent with the grid, the pair spacings, the candidate's value and the
        recorded transitions of its flagged k-points."""
        regularisation = diagnostics["regularisation"]
        kpoints = self.kpoints
        flagged = int(regularisation.get("flagged_kpoints") or 0)
        emulated = int(regularisation.get("emulated_kpoints") or 0)
        missing = int(regularisation.get("not_emulated_kpoints") or 0)
        problems = []
        if int(regularisation.get("kpoints") or 0) != kpoints:
            problems.append(f"it counts {regularisation.get('kpoints')} k-points on PyATB's {kpoints}")
        if not 0 <= flagged <= kpoints or min(emulated, missing) < 0 or emulated + missing != flagged:
            problems.append(
                f"{flagged} flagged k-points are not {emulated} emulated plus {missing} unmeasured"
            )
        listed = regularisation.get("listed") or []
        if int(self.effective["method"]) == 1:
            gap = diagnostics.get("min_pair_gap_eV")
            near = gap is not None and float(gap) * ENERGY_SCALE <= PYATB_DEGENERATE_EV
            if near != (flagged > 0):
                problems.append(
                    f"it flags {flagged} k-points while its smallest pair spacing is {gap} eV (PyATB's threshold "
                    f"{PYATB_DEGENERATE_EV} eV on its scale)"
                )
            if len(listed) > emulated or any(
                float(item.get("pair_gap_eV", math.inf)) * ENERGY_SCALE > PYATB_DEGENERATE_EV
                for item in listed
            ):
                problems.append("it lists k-points that are not flagged")
        elif flagged:
            problems.append("method 0 flags no k-point: its regularization covers every denominator")
        for other in (shift, unflagged):
            if other.shape != values.shape:
                problems.append("its tables have another shape than the value")
        if not problems:
            floor = CONSISTENCY_FLOOR * max(_largest(values), _largest(unflagged), _largest(shift))
            if flagged == 0:
                if _largest(shift) > floor:
                    problems.append("it records a shift without a flagged k-point")
                if _largest(values - unflagged) > floor:
                    problems.append(
                        f"its value without the flagged k-points differs from its value by "
                        f"{_largest(values - unflagged):.4g} microA/V^2 although no k-point is flagged"
                    )
            else:
                items = self.flagged_items(regularisation)
                if items is not None:
                    problems += self.check_flagged_items(items, values, shift, unflagged, flagged, missing)
        self.quantity(
            "regularisation_record",
            AGREE if not problems else UNEXPLAINED,
            detail="diagnostics.regularisation is consistent with the grid, the pair spacings, the candidate's "
            "value and its flagged k-points' transitions"
            + ("" if not problems else ": it is not, " + "; ".join(problems)),
        )

    def check_flagged_items(
        self,
        items: Sequence[Mapping[str, Any]],
        values: np.ndarray,
        shift: np.ndarray,
        unflagged: np.ndarray,
        flagged: int,
        missing: int,
    ) -> list[str]:
        """The flagged k-points' records against the value: on the grid, distinct, flagged, summing to it."""
        problems = []
        if len(items) != flagged:
            problems.append(f"it records {len(items)} of its {flagged} flagged k-points")
        grid = np.asarray([int(value) for value in self.effective["grid"]], float)
        keys = set()
        physical = np.zeros_like(values)
        change = np.zeros_like(values)
        failed = 0
        for item in items:
            k = np.asarray(item["k"], float)
            scaled = k * grid
            if (
                k.shape != (3,)
                or np.any(np.abs(scaled - np.round(scaled)) > 1e-9)
                or np.any(k < 0)
                or np.any(k >= 1)
            ):
                problems.append(f"its flagged k-point {item['k']} is not on PyATB's grid")
                continue
            keys.add(tuple(np.round(scaled).astype(int)))
            if float(item.get("pair_gap_eV", math.inf)) * ENERGY_SCALE > PYATB_DEGENERATE_EV:
                problems.append(f"its flagged k-point {item['k']} has a pair spacing above PyATB's threshold")
            part = _transitions(item["physical"])
            assert part is not None
            spectrum = self.kernel(*part)
            physical += spectrum
            cut = _transitions(item.get("cut"))
            if cut is None:
                failed += 1
            else:
                change += spectrum - self.kernel(*cut)
        if len(keys) != len(items):
            problems.append("it records a flagged k-point twice")
        if failed != missing:
            problems.append(f"{failed} records lack the cut, {missing} k-points are unmeasured")
        floor = CONSISTENCY_FLOOR * max(
            _largest(values), _largest(unflagged), _largest(physical), _largest(shift)
        ) + self.record_rounding(self.candidate.block("shift_current")["diagnostics"]["regularisation"])
        if _largest(values - unflagged - physical) > floor:
            problems.append(
                f"its flagged k-points' transitions add {_largest(physical):.4g} where its value less its value "
                f"without them is {_largest(values - unflagged):.4g} microA/V^2 (differing by "
                f"{_largest(values - unflagged - physical):.4g})"
            )
        if _largest(shift - change) > floor:
            problems.append(
                f"its shift differs from its flagged k-points' physical less cut transitions by "
                f"{_largest(shift - change):.4g} microA/V^2"
            )
        return problems

    def check_emulation(self, diagnostics: Mapping[str, Any], values: np.ndarray, shift: np.ndarray) -> None:
        """``emulation_consistency``: the candidate's PyATB-side tables follow from its value exactly."""
        scheme = diagnostics["pyatb_scheme"]
        omega = np.asarray(self.reference_omega())
        problems: list[str] = []
        facts: dict[str, Any] = {}
        evaluated = True
        if self.smearing == 1:
            # The emulation's own sum of its transitions under QUEER's Gaussian is the value less the measured cut.
            listed = table(scheme["transitions_uA_V2"])
            assert listed is not None
            target = values - shift
            if listed.shape != values.shape:
                problems.append("its transitions table has another shape than the value")
            else:
                floor = CONSISTENCY_FLOOR * max(_largest(listed), _largest(target)) + self.record_rounding(
                    diagnostics["regularisation"]
                )
                facts["max_gap_uA_V2"] = _largest(listed - target)
                if _largest(listed - target) > floor:
                    row, column = np.unravel_index(int(np.argmax(np.abs(listed - target))), listed.shape)
                    problems.append(
                        f"its emulation's transitions under QUEER's Gaussian differ from its value less its cut by "
                        f"{abs(listed[row, column] - target[row, column]):.4g} microA/V^2 at "
                        f"sigma_{SHIFT_COMPONENTS[column]}, omega index {row}: they are not the same transitions"
                    )
            if not self.pair_mode:
                dropped = table(scheme["window_dropped_uA_V2"])
                assert dropped is not None
                start, end = self.window()
                far = np.minimum(omega - start, end - omega) >= WINDOW_REACH_ETA * self.eta
                floor = CONSISTENCY_FLOOR * _largest(values)
                if dropped.shape != values.shape:
                    problems.append("its window record has another shape than the value")
                elif _largest(dropped[far]) > floor:
                    problems.append(
                        f"its record of the transitions PyATB's window drops reaches {_largest(dropped[far]):.4g} "
                        f"microA/V^2 more than {WINDOW_REACH_ETA:g} eta inside the window, where no Gaussian of a "
                        "dropped transition reaches"
                    )
        else:
            emulated = table(scheme["sigma_uA_V2"])
            assert emulated is not None
            moves, alternative, issues = self.recorded_moves(scheme["moved_transitions"])
            problems += issues
            items = self.flagged_items(diagnostics["regularisation"])
            flagged = int(diagnostics["regularisation"].get("flagged_kpoints") or 0)
            if moves is None or (flagged and items is None):
                evaluated = False
            elif not issues:
                extra = np.zeros_like(values)
                for item in items or []:
                    part = _transitions(item.get("cut") or item["physical"])
                    assert part is not None
                    extra += self.kernel(*part, pyatb=True) - self.kernel(*part)
                expected = pyatb_shift_constant_ratio() * (values - shift + moves + extra)
                floor = CONSISTENCY_FLOOR * max(
                    _largest(emulated), _largest(expected)
                ) + self.record_rounding(diagnostics["regularisation"])
                facts["max_gap_uA_V2"] = _largest(emulated - expected)
                if _largest(emulated - expected) > floor:
                    problems.append(
                        f"its emulation of PyATB's bins differs from r times its histogram less its cut plus its "
                        f"recorded moves by {_largest(emulated - expected):.4g} microA/V^2"
                    )
        kind = UNEXPLAINED if problems else (AGREE if evaluated else NOT_APPLICABLE)
        detail = (
            "the candidate's PyATB-side tables follow from its value: "
            + (
                "its emulation's transitions under QUEER's Gaussian sum to its value less its cut, and its window "
                "record stays within 6 eta of the window's ends"
                if self.smearing == 1
                else "its emulation of PyATB's bins is r times its histogram less its cut plus its recorded moves"
            )
            + ("" if evaluated else " (not evaluated: no record to check by)")
            + ("" if not problems else ": they do not, " + "; ".join(problems))
        )
        self.quantity("emulation_consistency", kind, facts.get("max_gap_uA_V2"), None, detail)

    def recorded_moves(
        self, record: Mapping[str, Any]
    ) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]:
        """The bins' recorded moves (PyATB's rows less QUEER's), the same with the ambiguous transitions in their
        other PyATB row, and any row the comparator's own placement contradicts."""
        if not record.get("recorded"):
            return None, None, []
        omega = np.asarray(self.reference_omega())
        start, count = float(omega[0]), len(omega)
        energies = np.asarray(record["energy_eV"], float).reshape(-1)
        weights = np.asarray(record["weight_uA_V2_eV"], float).reshape(len(energies), len(SHIFT_COMPONENTS))
        own = np.asarray(record["queer_row"], int).reshape(-1)
        theirs = np.asarray(record["pyatb_row"], int).reshape(-1)
        ambiguous = np.asarray(record["ambiguous"], bool).reshape(-1)
        problems = []
        if not len(energies) == len(own) == len(theirs) == len(ambiguous):
            return None, None, ["its moved transitions' lists differ in length"]
        if not np.all(np.isfinite(energies)) or not np.all(np.isfinite(weights)):
            return None, None, ["a moved transition is not finite"]
        derived_own = queer_rows(energies, start, self.domega, count)
        derived_theirs = pyatb_rows(energies, start, self.domega, count, self.pair_mode)
        wrong = ~ambiguous & ((derived_own != own) | (derived_theirs != theirs))
        if wrong.any():
            problems.append(
                f"{int(np.count_nonzero(wrong))} recorded moves contradict the rows their energies give "
                f"(first at {float(energies[np.argmax(wrong)]):.9g} eV)"
            )
        if np.any(~ambiguous & (own == theirs)):
            problems.append("it records transitions that neither code moves")
        moves = place(theirs, weights, count) - place(own, weights, count)
        # An ambiguous transition sits within 1e-9 eV of an edge: PyATB's rounding may put it on the other side.
        lower = pyatb_rows(energies - 2e-9 / ENERGY_SCALE, start, self.domega, count, self.pair_mode)
        upper = pyatb_rows(energies + 2e-9 / ENERGY_SCALE, start, self.domega, count, self.pair_mode)
        other = np.where(lower != theirs, lower, upper)
        alternative_rows = np.where(ambiguous, other, theirs)
        alternative = place(alternative_rows, weights, count) - place(own, weights, count)
        return moves, alternative, problems

    def recompute_flagged(self, diagnostics: Mapping[str, Any], values: np.ndarray) -> FlaggedCut:
        """PyATB's method-1 cut at the recorded flagged k-points, recomputed from the caller's scfout."""
        regularisation = diagnostics["regularisation"]
        flagged = int(regularisation.get("flagged_kpoints") or 0)
        if flagged == 0:
            return FlaggedCut(True, "no flagged k-point", np.zeros_like(values))
        items = self.flagged_items(regularisation)
        if items is None:
            reason = (regularisation.get("flagged_records") or {}).get("reason") or "no record"
            return FlaggedCut(False, f"the run kept none of its {flagged} flagged k-points ({reason})")
        if self.scfout is None:
            return FlaggedCut(
                False,
                f"no scfout was supplied to recompute its {flagged} flagged k-points (compare with scfout=, or "
                "--scfout)",
            )
        digest = sha256_file(self.scfout)
        model_block = self.candidate.block("model")
        recorded = model_block.get("scfout_sha256")
        if digest is None or digest != recorded:
            return FlaggedCut(
                False,
                f"the supplied scfout {self.scfout} (SHA-256 {str(digest)[:16]}...) is not the candidate's "
                f"({str(recorded)[:16]}...)",
            )
        if float(self.effective["scissors"]) != 0.0:
            return FlaggedCut(
                False, "a nonzero scissors changes PyATB's denominators: its cut is not recomputed"
            )
        try:
            checks = self.recompute_items(items)
        except (OSError, ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            self.quantity(
                "regularisation_recomputed",
                UNEXPLAINED,
                detail=f"the flagged k-points could not be recomputed from the scfout: {type(error).__name__}: "
                f"{error}",
            )
            return FlaggedCut(False, "the recomputation failed")
        problems = checks.pop("problems")
        self.quantity(
            "regularisation_recomputed",
            AGREE if not problems else UNEXPLAINED,
            checks["worst_ratio"],
            1.0,
            detail=f"{len(items)} flagged k-points recomputed from the scfout: the recorded physical transitions meet "
            "QUEER's estimator called at each k-point and an independent transcription of PyATB's method-1 "
            "integrand, the recorded cut QUEER's estimator with its degeneracy tolerance at 1e-4 eV, within "
            "tolerances derived from the recomputation (worst ratio to its tolerance given as the metric)"
            + ("" if not problems else ": they do not, " + "; ".join(problems)),
        )
        if problems:
            return FlaggedCut(
                False, "the recorded flagged k-points contradict their recomputation", details=checks
            )
        return FlaggedCut(
            True,
            f"recomputed from the scfout at its {len(items)} flagged k-points",
            checks.pop("change"),
            details=checks,
        )

    def recompute_items(self, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Each recorded flagged k-point against QUEER's estimator at that k-point and PyATB's integrand there."""
        import dataclasses

        from ...models.evaluation import spin_degeneracy, volume
        from ...models.openmx import OpenMXModel
        from ...observables.nonlinear import shift_current_point, shift_current_settings
        from .openmx_steps import single_channel_model
        from .pyatb_shift_integrand import queer_weights, shift_integrand

        assert self.scfout is not None
        model = OpenMXModel.from_scfout(self.scfout)
        view = model if model.spin_mode == "soc" else single_channel_model(model)
        effective = self.effective
        occupied = int(effective["occ_band"])
        pairs = [(int(effective["n_occ"]) - 1, int(effective["m_unocc"]) - 1)] if self.pair_mode else None
        omega = np.asarray(self.reference_omega())
        conversion = -math.pi * E_CHARGE**2 / HBAR * 1e6 * spin_degeneracy(view) / volume(view)
        factor = self.sign * self.spin_factor * conversion / self.kpoints
        tolerance_q = float(view.degeneracy_tolerance)
        settings = shift_current_settings(
            view,
            omega,
            float(view.fermi_energy),
            eta=self.eta / math.sqrt(2) if self.eta > 0 else 0.01,
            smearing="gaussian",
            occupied_bands=occupied,
            band_pairs=None if pairs is None else [list(pairs[0])],
            method="analytic",
            scissors_occupied_bands=occupied,
        )
        cut_settings = dataclasses.replace(settings, degeneracy_tol=PYATB_DEGENERATE_EV / ENERGY_SCALE)
        peak = self.kernel_peak()

        def own(point: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
            source, target = np.nonzero(point["factor"])
            weights = point["factor"][source, target] * point["integrand"][..., source, target]
            index = [
                ("xyz".index(name[0]), "xyz".index(name[1]), "xyz".index(name[2]))
                for name in SHIFT_COMPONENTS
            ]
            columns = np.stack([weights[a, b, c] for a, b, c in index], axis=-1)
            return point["gap"][source, target], factor * columns.reshape(len(source), len(SHIFT_COMPONENTS))

        problems: list[str] = []
        change = np.zeros((len(omega), len(SHIFT_COMPONENTS)))
        per_k = []
        worst = 0.0
        start, end = self.window()
        for item in items:
            k = np.asarray(item["k"], float)
            physical = _transitions(item["physical"])
            cut = _transitions(item.get("cut"))
            assert physical is not None
            recorded = self.kernel(*physical)
            direct = own(shift_current_point(view, k, settings))
            reference = self.kernel(*direct)
            if pairs is None:
                port = shift_integrand(
                    view,
                    k,
                    None,
                    energy_scale=ENERGY_SCALE,
                    threshold_eV=ENERGY_SCALE * tolerance_q,
                    regularization_eV=None,
                    occupied=occupied,
                )
            else:
                port = shift_integrand(
                    view,
                    k,
                    pairs,
                    energy_scale=ENERGY_SCALE,
                    threshold_eV=ENERGY_SCALE * tolerance_q,
                    regularization_eV=None,
                    occupied=occupied,
                )
            port_cut = shift_integrand(
                view,
                k,
                pairs,
                energy_scale=ENERGY_SCALE,
                threshold_eV=PYATB_DEGENERATE_EV,
                regularization_eV=PYATB_SHIFT_REGULARIZATION_EV,
                occupied=occupied,
                window=None if pairs is not None else (start, end),
            )
            port_weights = factor * queer_weights(port["integrand"])
            transcribed = self.kernel(port["energies_eV"], port_weights)
            weight = np.abs(port_weights).sum(axis=0) * peak
            levels = np.asarray(port["levels_eV"], float)
            spacing = np.abs(levels[:, None] - levels[None, :])
            distinct = spacing[spacing > tolerance_q]
            smallest = float(distinct.min()) if distinct.size else math.inf
            condition = 1.0 + _largest(levels) / smallest
            rounding = RECOMPUTE_ROUNDING * weight + CONSISTENCY_FLOOR * _largest(reference)
            conditioned = RECOMPUTE_CONDITIONING * condition * weight + rounding
            ratios = {
                "queer": float(np.max(np.abs(recorded - reference) / rounding[None, :], initial=0.0)),
                "transcription": float(
                    np.max(np.abs(recorded - transcribed) / conditioned[None, :], initial=0.0)
                ),
            }
            if cut is not None:
                direct_cut = own(shift_current_point(view, k, cut_settings))
                ratios["queer_cut"] = float(
                    np.max(
                        np.abs(self.kernel(*cut) - self.kernel(*direct_cut)) / rounding[None, :], initial=0.0
                    )
                )
            selected = sorted({band for pair in port_cut["pairs"] for band in pair})
            near = [
                abs(levels[band] - levels[other]) * ENERGY_SCALE
                for band in selected
                for other in range(len(levels))
                if other != band and tolerance_q < abs(levels[band] - levels[other])
            ]
            gap = min(near, default=math.inf)
            if gap > PYATB_DEGENERATE_EV:
                problems.append(
                    f"k-point {item['k']}: no band of a pair PyATB selects lies within {PYATB_DEGENERATE_EV} eV of "
                    f"another level (smallest {gap:.3g} eV on PyATB's scale): it is not flagged"
                )
            for name, ratio in ratios.items():
                if ratio > 1:
                    problems.append(
                        f"k-point {item['k']}: the record differs from the {name} recomputation by "
                        f"{ratio:.3g} times its tolerance"
                    )
            worst = max(worst, *ratios.values())
            cut_value = self.kernel(
                port_cut["energies_eV"], factor * queer_weights(port_cut["integrand"]), pyatb=True
            )
            change += recorded - cut_value
            per_k.append(
                {
                    "k": item["k"],
                    "pair_gap_pyatb_eV": gap,
                    "conditioning": condition,
                    "ratios": ratios,
                    "max_change_uA_V2": _largest(recorded - cut_value),
                }
            )
        return {"problems": problems, "change": change, "worst_ratio": worst, "kpoints": per_k}

    # ------------------------------------------------------------ causes
    def causes(self, cut: FlaggedCut, flagged: int) -> dict[str, dict[str, Any]]:
        effective = self.effective
        method = int(effective["method"])
        smearing = self.smearing
        if method == 0:
            reg = {
                "active": True,
                "bounded": False,
                "detail": "method 0: PyATB regularizes every energy denominator of its eigenvector derivatives at "
                f"every k-point, d / (d^2 + {PYATB_SHIFT_REGULARIZATION_EV}^2) "
                "(get_partial_eigenvectors_1k_SumOverStates)",
            }
        else:
            reg = {
                "active": flagged > 0,
                "bounded": True,
                "verified": cut.verified,
                "flagged_kpoints": flagged,
                "detail": f"method 1: at {flagged} k-points a band of a selected pair lies within "
                f"{PYATB_DEGENERATE_EV} eV of a level it is not exactly degenerate with, where PyATB's "
                "get_D_degenerate regularizes the denominators: "
                + cut.reason
                + (
                    "; PyATB's cut value there comes from an independent transcription of its integrand"
                    if cut.verified and flagged
                    else ""
                ),
            }
        return {
            "scissors": {
                "active": float(effective["scissors"]) != 0.0,
                "bounded": False,
                "detail": "PyATB adds the scissors to its eigenvalues before its derivative kernels; QUEER "
                "shifts the transition energies only",
            },
            "regularisation": reg,
            "adaptive_smearing": {
                "active": smearing == 2,
                "bounded": False,
                "detail": "PyATB's adaptive width |grad(E_n - E_m)| dk eta (dk = max|b_i|/100, clipped to "
                "[1e-4, 1] eV) differs from QUEER's",
            },
            "pyatb_scheme": {
                "active": smearing in (0, 1),
                "bounded": True,
                "detail": "PyATB's Hartree scale (its table moved to QUEER's scale from its own curve), its window "
                "(QUEER's record of the transitions it drops), its integer bins (the recorded moves) and its "
                "Gaussian truncation (at most 1e-6 of a transition's peak, else not reproducible)",
            },
        }

    def clean_conditions(self, diagnostics: Mapping[str, Any]) -> dict[str, bool]:
        transitions = diagnostics.get("transitions") or {}
        regularisation = diagnostics.get("regularisation")
        omega = self.reference_omega()
        low, high = transitions.get("min_pyatb_eV"), transitions.get("max_pyatb_eV")
        eta = self.eta
        margin = False
        if low is not None and high is not None and eta > 0:
            margin = min(low - omega[0], omega[-1] - high) >= CLEAN_MARGIN_ETA * eta
        measured = isinstance(regularisation, Mapping)
        flagged = (
            int(regularisation.get("flagged_kpoints") or 0) if isinstance(regularisation, Mapping) else 1
        )
        gap = diagnostics.get("min_pair_gap_eV")
        return {
            "scissors_zero": float(self.effective["scissors"]) == 0.0,
            "gaussian": self.smearing == 1,
            "method_1": int(self.effective["method"]) == 1,
            "truncation_tail_below_1e-6": self.smearing == 1
            and self.truncation_tail() <= CLEAN_TRUNCATION_TAIL,
            "window_margin_5_eta": bool(margin),
            "no_pair_within_1e-4_eV": measured and flagged == 0,
            "pair_gap_above_1e-3_eV": measured and (gap is None or float(gap) > CLEAN_PAIR_GAP_EV),
        }

    def build_identity(self) -> dict[str, Any]:
        """The build this candidate and this comparator are: what a clean-control report must match."""
        provenance = self.candidate.document.get("provenance") or {}
        build = self.block.get("build") or {}
        return {
            "queer_version": provenance.get("queer_version"),
            "queer_record_sha256": provenance.get("queer_record_sha256"),
            "build_sha": provenance.get("build_sha"),
            "queer_source_sha256": build.get("queer_source_sha256"),
            "comparator_sha256": comparator_sha256(),
        }

    def release_clean_control(self) -> str:
        """``passed``, ``failed``, ``other_build``, ``not_a_clean_control`` or ``not_supplied``."""
        report = load_report(self.clean_control_report)
        if report is None:
            return "not_supplied"
        rules = (report.get("semantics") or {}).get("rules") or {}
        eligible = ((rules.get("clean_control") or {}).get("eligible")) is True
        if report.get("step") != "SHIFT_CURRENT" or not eligible:
            return "not_a_clean_control"
        theirs = rules.get("build") or {}
        mine = self.build_identity()
        if (
            mine["queer_source_sha256"] is None
            or mine["comparator_sha256"] is None
            or any(theirs.get(key) != mine[key] for key in BUILD_KEYS)
        ):
            return "other_build"
        compared = [
            item for item in report.get("quantities") or [] if str(item.get("name")).startswith("sigma_")
        ]
        direct = bool(compared) and all(item.get("class") in DIRECT for item in compared)
        return "passed" if report.get("verdict") == PASS and direct else "failed"

    def self_convergence(self, values: np.ndarray, reference: np.ndarray) -> tuple[bool, str]:
        convergence = (self.block.get("diagnostics") or {}).get("self_convergence")
        if not isinstance(convergence, Mapping):
            return (
                False,
                "the candidate has no self-convergence estimate (run it with self_convergence: true)",
            )
        compared = convergence.get("legacy_single_channel") if self.legacy else convergence.get("physical")
        if not isinstance(compared, Mapping):
            return False, "the candidate's self-convergence estimate lacks the compared quantity"
        error, largest = float(compared["e_inf_uA_V2"]), float(compared["max_doubled_uA_V2"])
        magnitude, scale = _largest(values), _largest(reference)
        converged = error <= SELF_CONVERGENCE_RELATIVE * largest
        bounded = scale / MAGNITUDE_FACTOR <= magnitude <= MAGNITUDE_FACTOR * scale
        detail = (
            f"|sigma_Q(N) - sigma_Q(2N)|_inf {error:.4g} vs {SELF_CONVERGENCE_RELATIVE} max|sigma_Q(2N)| "
            f"{SELF_CONVERGENCE_RELATIVE * largest:.4g} (grids {convergence.get('grid')} and "
            f"{convergence.get('grid_doubled')}); max|sigma_Q| {magnitude:.4g} within a factor "
            f"{MAGNITUDE_FACTOR:g} of max|sigma_P| {scale:.4g}: {bounded}"
        )
        return converged and bounded, detail

    # ----------------------------------------------------------- compare
    def scheme_route(
        self,
        diagnostics: Mapping[str, Any],
        values: np.ndarray,
        reference: np.ndarray,
        cut: FlaggedCut,
        moved: tuple[np.ndarray | None, np.ndarray],
        tolerance: np.ndarray,
    ) -> dict[str, Any]:
        """QUEER's value moved to PyATB's scheme, per component: ``reproduced``, ``unresolved``, ``not_reproduced``
        or ``unverifiable`` (a flagged k-point not recomputed), with what entered."""
        width = len(SHIFT_COMPONENTS)
        omega = np.asarray(self.reference_omega())
        regularisation = diagnostics.get("regularisation") or {}
        flagged = int(regularisation.get("flagged_kpoints") or 0)
        verified = cut.verified
        route: dict[str, Any] = {
            "status": ["not_reproduced"] * width,
            "verified_cut": verified,
            "flagged": flagged,
            "reason": None,
        }
        scheme = diagnostics.get("pyatb_scheme") or {}
        if not verified or cut.change is None:
            # Unrecomputed, PyATB's cut is unknown: nothing is reproduced from a claim.
            route["status"] = ["unverifiable"] * width
            route["reason"] = cut.reason
            return route
        removed = cut.change
        if self.smearing == 1:
            if self.truncation_tail() > CLEAN_TRUNCATION_TAIL:
                route["status"] = ["unresolved"] * width
                route["reason"] = (
                    f"PyATB's Gaussian truncated at int(5 eta/domega) = {int(5 * self.eta / self.domega)} bins drops "
                    f"up to {self.truncation_tail():.3g} of a transition's peak, which QUEER's value cannot reproduce "
                    "(refine domega against eta)"
                )
                return route
            moved_x, margin = moved
            if moved_x is None:
                # Not movable: a row counts only where PyATB's table meets the target directly.
                moved_x = np.zeros_like(values)
                margin = np.full_like(values, np.inf)
            dropped = np.zeros_like(values) if self.pair_mode else table(scheme.get("window_dropped_uA_V2"))
            assert dropped is not None
            target = values - dropped - removed
            route["target"] = target
            route["dropped_max"] = np.max(np.abs(dropped), axis=0)
            route["removed_max"] = np.max(np.abs(removed), axis=0)
            resolved = margin <= SCALE_RESOLVED * tolerance[None, :]
            close = np.abs(reference - target) <= tolerance[None, :]
            moved_ok = resolved & (np.abs(moved_x - target) <= tolerance[None, :] + margin)
            failing = ~close & ~moved_ok
            # A row the move cannot resolve is undecided only within the reach of PyATB's Hartree scale.
            reach = energy_scale_reach(reference, omega=omega, eta=self.eta, domega=self.domega)
            beyond_reach = np.abs(reference - target) > tolerance[None, :] + reach
            beyond = np.ones(len(omega), bool)
            if not self.pair_mode:
                start, end = self.window()
                beyond = np.minimum(omega - start, end - omega) >= WINDOW_REACH_ETA * self.eta
            floor = CONSISTENCY_FLOOR * _largest(values)
            magnitude = np.abs(reference)
            total = magnitude.sum(axis=0)
            share = np.where(
                total > 0, (magnitude * beyond[:, None]).sum(axis=0) / np.where(total > 0, total, 1.0), 1.0
            )
            reaches = np.max(np.abs(dropped), axis=0) > floor
            route["informative_share"] = share
            route["beyond_rows"] = int(np.count_nonzero(beyond))
            status = []
            for column in range(width):
                rows = failing[:, column]
                if np.any(rows & (resolved[:, column] | beyond_reach[:, column])):
                    status.append("not_reproduced")
                elif rows.any() or (reaches[column] and share[column] < INFORMATIVE_SHARE):
                    status.append("unresolved")
                else:
                    status.append("reproduced")
            route["status"] = status
            if "unresolved" in status:
                route["reason"] = (
                    "PyATB's table cannot be moved to QUEER's energy scale within half the tolerance on rows the "
                    f"difference needs (it moves on {SCALE_STENCIL} or more photon energies at most "
                    f"{SCALE_MAX_STEP_ETA:g} eta apart; here {len(omega)} at {self.domega / self.eta:.3g} eta), or "
                    f"PyATB's window leaves less than {INFORMATIVE_SHARE:g} of a component's size on the rows its "
                    f"dropped transitions do not reach (more than {WINDOW_REACH_ETA:g} eta from its ends): refine "
                    "domega against eta or widen the window"
                )
            return route
        moves, alternative = moved
        if moves is None:
            route["status"] = ["unresolved"] * width
            route["reason"] = (
                "the run kept no record of the transitions PyATB's bins move "
                f"({(scheme.get('moved_transitions') or {}).get('reason')})"
            )
            return route
        ratio = pyatb_shift_constant_ratio()
        best = None
        for option in (moves, alternative):
            target = ratio * (values - removed + option)
            fits = np.max(np.abs(reference - target) - tolerance[None, :], axis=0) <= 0
            if best is None or fits.sum() > best[1].sum():
                best = (target, fits)
        assert best is not None
        route["target"] = best[0] / ratio
        route["moved_max"] = np.max(np.abs(moves), axis=0)
        route["removed_max"] = np.max(np.abs(removed), axis=0)
        route["status"] = ["reproduced" if fit else "not_reproduced" for fit in best[1]]
        return route

    def compare_shift(self) -> None:
        product = self.reference.product or {}
        reference = np.asarray(product["sigma"], float)
        values, compared = self.compared_values()
        diagnostics = self.block.get("diagnostics") or {}
        if reference.shape != values.shape:
            raise ValueError(f"tables of shape {reference.shape} and {values.shape} cannot be compared")
        omega = np.asarray(self.reference_omega())
        domega = self.domega
        scale = _largest(reference)
        base = AGREEMENT_RELATIVE * scale
        tolerance = base + PRINT_RELATIVE * np.max(np.abs(reference), axis=0, initial=0.0)
        binned = self.smearing == 0
        method = int(self.effective["method"])
        records = self.check_candidate_record(diagnostics)
        regularisation = diagnostics.get("regularisation") if records else {}
        assert isinstance(regularisation, Mapping)
        flagged = int(regularisation.get("flagged_kpoints") or 0)
        shift = np.zeros_like(values)
        if records:
            recorded_shift = table(regularisation["binned_shift_uA_V2" if binned else "shift_uA_V2"])
            unflagged = table(regularisation["binned_unflagged_uA_V2" if binned else "sigma_unflagged_uA_V2"])
            assert recorded_shift is not None and unflagged is not None
            shift = recorded_shift
            self.check_regularisation_record(diagnostics, values, shift, unflagged)
        conditions = self.clean_conditions(diagnostics)
        self.clean = clean = all(conditions.values())
        moved_x: np.ndarray | None = None
        margin = np.full_like(values, np.inf)
        if self.smearing == 1 and len(omega) >= SCALE_STENCIL and domega <= SCALE_MAX_STEP_ETA * self.eta:
            moved_x, margin = move_to_queer_scale(reference, omega=omega, eta=self.eta, domega=domega)
        resolved = margin <= SCALE_RESOLVED * tolerance[None, :]
        moves: tuple[np.ndarray | None, np.ndarray] = (moved_x, margin)
        if records and self.smearing in (0, 1):
            self.check_emulation(diagnostics, values, shift)
            if binned:
                recorded = self.recorded_moves(diagnostics["pyatb_scheme"]["moved_transitions"])
                moves = (recorded[0], recorded[1] if recorded[1] is not None else np.zeros_like(values))
        cut = FlaggedCut(True, "no flagged k-point", np.zeros_like(values))
        if flagged and records and method == 1:
            cut = self.recompute_flagged(diagnostics, values)
        elif flagged:
            cut = FlaggedCut(False, "the flagged k-points cannot be recomputed")
        contradiction = sorted(
            {
                item["name"]
                for item in self.quantities
                if item["name"] in SELF_CHECKS and item["class"] == UNEXPLAINED
            }
        )
        causes = self.causes(cut, flagged)
        unbounded = [
            name for name in UNBOUNDED_CAUSES if causes[name]["active"] and not causes[name]["bounded"]
        ]
        route: dict[str, Any] | None = None
        if records and not contradiction and not unbounded and self.smearing in (0, 1):
            route = self.scheme_route(diagnostics, values, reference, cut, moves, tolerance)
        hartree = np.zeros(len(SHIFT_COMPONENTS))
        if moved_x is not None:
            hartree = np.max(np.abs(reference - moved_x), axis=0)
        elif self.smearing == 1 and len(omega) >= 3:
            hartree = np.max(energy_scale_reach(reference, omega=omega, eta=self.eta, domega=domega), axis=0)
        widest = np.max(np.where(resolved, margin, 0.0), axis=0)
        allowed = tolerance + hartree + widest
        # PyATB's truncation above 1e-6 of a peak distorts its table beyond what its integral's sign can tell.
        readable = self.smearing != 1 or self.truncation_tail() <= CLEAN_TRUNCATION_TAIL
        # PyATB's method-1 cut can change any value at the flagged k-points, signs included: unrecomputed, it leaves
        # every difference from PyATB undecided (never documented on trust, never failed on a guess).
        undecided = bool(flagged) and method == 1 and not cut.verified and not unbounded and not contradiction
        symmetric, facts = symmetry_gate(reference, values, domega, allowed)
        facts["against"] = compared
        if not symmetric and route is not None and "target" in route:
            view = route["target"] * (pyatb_shift_constant_ratio() if binned else 1.0)
            passes, other_facts = symmetry_gate(reference, view, domega, allowed)
            if passes:
                symmetric, facts = True, {**other_facts, "against": f"{compared} moved to PyATB's scheme"}
        judged = readable and (symmetric or not undecided)
        self.quantity(
            "symmetry",
            (AGREE if symmetric else UNEXPLAINED) if judged else NOT_APPLICABLE,
            detail=(
                f"significant components {facts['reference_components']} (PyATB) and {facts['candidate_components']} "
                f"(QUEER's {facts['against']}), differing for {facts['components_differ']}; integral signs differ "
                f"for {facts['integral_sign_differs']}"
            )
            + (
                f": not judged, PyATB's Gaussian truncated at {int(5 * self.eta / domega)} bins distorts its table"
                if not readable
                else (
                    f": not judged, PyATB's method-1 cut at the {flagged} flagged k-points is not recomputed"
                    if not judged
                    else ""
                )
            ),
        )
        if not judged:
            symmetric = True
        self.release = release = self.release_clean_control()
        converted = self.legacy or self.sign < 0 or binned
        converged, convergence = None, ""
        self.direct_everywhere = True
        unresolved: list[str] = []
        unverified: list[str] = []
        # A component PyATB's table contradicts outright: what cannot be judged elsewhere does not hide it.
        definite: list[str] = []
        for index, name in enumerate(SHIFT_COMPONENTS):
            gap = np.abs(reference[:, index] - values[:, index])
            direct = float(np.max(gap, initial=0.0))
            limit = float(tolerance[index])
            label = f"sigma_{name}"
            if release == "failed":
                self.quantity(
                    label, UNEXPLAINED, direct, limit, "the clean control of this QUEER build failed"
                )
                self.direct_everywhere = False
                continue
            if direct <= limit:
                self.quantity(
                    label,
                    AFTER_CONVENTION if converted else AGREE,
                    direct,
                    limit,
                    f"PyATB's table meets QUEER's {compared}"
                    + (" in PyATB's conventions" if converted else ""),
                )
                continue
            if moved_x is not None:
                rows = (gap <= limit) | (
                    resolved[:, index]
                    & (np.abs(moved_x[:, index] - values[:, index]) <= limit + margin[:, index])
                )
                if rows.all():
                    self.quantity(
                        label,
                        AFTER_CONVENTION,
                        direct,
                        limit + float(hartree[index]) + float(widest[index]),
                        f"PyATB's table, moved to QUEER's Hartree constant and constants from its own curve "
                        f"(move_to_queer_scale: up to {float(hartree[index]):.4g} microA/V^2, uncertain by "
                        f"{float(widest[index]):.3g}), meets QUEER's {compared}"
                        + (" in PyATB's conventions" if converted else "")
                        + f" to {float(np.max(np.abs(moved_x[:, index] - values[:, index]))):.4g}",
                    )
                    continue
            self.direct_everywhere = False
            if clean:
                self.quantity(
                    label,
                    UNEXPLAINED,
                    direct,
                    limit,
                    "a clean control must agree with QUEER's physical value directly: to 1e-4 max|sigma_P|, PyATB's "
                    "table moved to QUEER's energy scale",
                )
                continue
            if contradiction:
                self.quantity(
                    label,
                    UNEXPLAINED,
                    direct,
                    limit,
                    f"the candidate contradicts itself ({', '.join(contradiction)}): no cause can explain a difference",
                )
                continue
            if undecided:
                unverified.append(name)
                self.quantity(
                    label,
                    UNEXPLAINED,
                    direct,
                    limit,
                    f"not judged: PyATB's method-1 cut at the {flagged} flagged k-points is not recomputed ({cut.reason})",
                )
                continue
            if not readable:
                unresolved.append(name)
                self.quantity(
                    label,
                    UNEXPLAINED,
                    direct,
                    limit,
                    f"PyATB's Gaussian truncated at {int(5 * self.eta / domega)} bins drops up to "
                    f"{self.truncation_tail():.3g} of a transition's peak: this run cannot judge QUEER's value",
                )
                continue
            status = route["status"][index] if route is not None else None
            if (
                route is not None
                and status == "reproduced"
                and symmetric
                and (flagged == 0 or route["verified_cut"])
            ):
                pieces = []
                if self.smearing == 1:
                    pieces.append(
                        f"its Hartree scale moved from its own curve (up to {float(hartree[index]):.4g})"
                    )
                    if not self.pair_mode and float(route["dropped_max"][index]) > 0:
                        pieces.append(
                            f"the transitions its window drops ({float(route['dropped_max'][index]):.4g} at most, "
                            f"QUEER's record, within {WINDOW_REACH_ETA:g} eta of its ends)"
                        )
                else:
                    pieces.append(
                        f"the transitions its integer bins place in other rows ({float(route['moved_max'][index]):.4g} "
                        "at most, recorded and checked from their energies)"
                    )
                if flagged:
                    pieces.append(
                        f"its method-1 cut at the {flagged} flagged k-points, recomputed from the scfout "
                        f"({float(route['removed_max'][index]):.4g} at most)"
                    )
                self.quantity(
                    label,
                    DOCUMENTED,
                    direct,
                    limit
                    + float(hartree[index])
                    + float(widest[index])
                    + float(np.asarray(route.get("dropped_max", route.get("moved_max")))[index])
                    + float(route["removed_max"][index]),
                    f"QUEER's value differs by {direct:.4g} microA/V^2; PyATB's scheme reproduces it: PyATB's table "
                    "meets QUEER's value moved to its scheme ("
                    + "; ".join(pieces)
                    + ") within the tolerance",
                )
                continue
            if status in ("reproduced", "unverifiable") and symmetric:
                assert route is not None
                unverified.append(name)
                self.quantity(
                    label,
                    UNEXPLAINED,
                    direct,
                    limit,
                    f"PyATB's scheme with the candidate's own measured method-1 cut would reproduce it, but the cut is "
                    f"unverified: {cut.reason}",
                )
                continue
            if status == "unresolved" and symmetric:
                assert route is not None
                unresolved.append(name)
                self.quantity(
                    label,
                    UNEXPLAINED,
                    direct,
                    limit,
                    f"PyATB's scheme cannot be reproduced from QUEER's value here: {route['reason']}",
                )
                continue
            if unbounded and symmetric:
                if converged is None:
                    converged, convergence = self.self_convergence(values, reference)
                if converged:
                    self.quantity(
                        label,
                        DOCUMENTED,
                        direct,
                        limit,
                        f"differs by {direct:.4g} microA/V^2 with unbounded causes {', '.join(unbounded)}",
                    )
                    continue
            reason = "no cause explains it"
            if not symmetric:
                reason = "the symmetry gate failed"
            elif unbounded and converged is False:
                reason = "the candidate is unconverged"
            elif status == "not_reproduced":
                reason = "PyATB's scheme applied to QUEER's value does not reproduce PyATB's table"
            if not (unbounded and converged is False):
                definite.append(name)
            self.quantity(label, UNEXPLAINED, direct, limit, reason)
        if definite:
            # What the run cannot judge is reported per component; the verdict is the failure.
            unresolved, unverified = [], []
        if unresolved:
            why = (
                route["reason"]
                if route is not None and route["reason"]
                else f"PyATB's Gaussian truncated at {int(5 * self.eta / domega)} bins drops up to "
                f"{self.truncation_tail():.3g} of a transition's peak (refine domega against eta)"
            )
            self.precondition("pyatb_scheme_resolution", False, f"{why}: {unresolved}")
        if unverified:
            self.precondition(
                "regularisation",
                False,
                f"PyATB's method-1 cut at the {flagged} flagged k-points is documented only when recomputed: "
                f"{cut.reason} ({unverified})",
            )
        if converged is False:
            self.precondition("self_convergence", False, f"queer_unconverged: {convergence}")
        elif converged:
            self.precondition("self_convergence", True, convergence)
        if clean:
            self.quantity(
                "clean_control",
                AGREE if self.direct_everywhere and symmetric else UNEXPLAINED,
                detail="a clean-control configuration: every component must agree with QUEER's physical value "
                "directly, to 1e-4 max|sigma_P|, PyATB's table moved to QUEER's energy scale",
            )
        if release != "not_supplied":
            kind = {"passed": AGREE, "failed": UNEXPLAINED}.get(release, NOT_APPLICABLE)
            self.quantity(
                "clean_control_release", kind, detail=f"the supplied clean-control report: {release}"
            )
        self.rules = {
            "export_sign": self.sign,
            "spin_factor": self.spin_factor,
            "legacy_single_channel": self.legacy,
            "compared": compared,
            "symmetry": facts,
            "clean_control": {"eligible": clean, "conditions": conditions, "release": release},
            "build": self.build_identity(),
            "causes": causes,
            "unbounded_causes": unbounded,
            "self_convergence": convergence or None,
            "flagged_kpoints": {
                "count": flagged,
                "verified": cut.verified,
                "reason": cut.reason,
                **cut.details,
            },
        }
        energy_scale: dict[str, Any] = {
            "rule": "PyATB's Gaussian table moved to QUEER's Hartree constant and SI constants from its own curve: "
            "r X(w) = s [P(w - a) - c P''(w - a) + c^2/2 P''''], a = (1 - s) w, c = eta^2 (1 - s^2)/4, its derivatives "
            f"by {SCALE_STENCIL}-point finite differences; uncertainty {SCALE_MARGIN_FACTOR:g} x the change from "
            f"{SCALE_CHECK_STENCIL} points, used on rows where it is at most {SCALE_RESOLVED:g} of tol",
            "s": ENERGY_SCALE,
            "constant_ratio": pyatb_shift_constant_ratio(),
            "applied": moved_x is not None,
        }
        if moved_x is not None:
            energy_scale.update(
                max_move_uA_V2=dict(zip(SHIFT_COMPONENTS, hartree.tolist())),
                max_uncertainty_uA_V2=dict(zip(SHIFT_COMPONENTS, widest.tolist())),
                unresolved_rows=int(np.count_nonzero(~resolved.all(axis=1))),
            )
        route_facts: dict[str, Any] | None = None
        if route is not None:
            route_facts = {
                "status": dict(zip(SHIFT_COMPONENTS, route["status"])),
                "reason": route["reason"],
                "verified_cut": route["verified_cut"],
            }
            for key in ("informative_share", "dropped_max", "moved_max", "removed_max"):
                if key in route:
                    route_facts[key] = dict(zip(SHIFT_COMPONENTS, np.asarray(route[key]).tolist()))
            if "beyond_rows" in route:
                route_facts["rows_beyond_reach"] = route["beyond_rows"]
        self.basis.update(
            {
                "relative": AGREEMENT_RELATIVE,
                "scale_uA_V2": scale,
                "base_uA_V2": base,
                "print_relative": PRINT_RELATIVE,
                "energy_scale": energy_scale,
                "truncation_tail": self.truncation_tail() if self.smearing == 1 else None,
                "window_reach_eta": WINDOW_REACH_ETA,
                "scheme": route_facts,
                "emulated": self.smearing in (0, 1) and records,
            }
        )

    def compare_candidate_file(self) -> None:
        path = self.candidate.directory / SHIFT_CURRENT_FILE
        try:
            written = parse_shift_current(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            self.quantity("candidate_file", UNEXPLAINED, detail=f"{SHIFT_CURRENT_FILE}: {error}")
            return
        name = "binned_uA_V2" if int(self.effective["smearing_method"]) == 0 else "sigma_uA_V2"
        expected = table(self.block[name])
        assert expected is not None
        expected = self.sign * expected
        values = np.asarray(written["sigma"], float)
        omega = np.asarray(written["omega_eV"], float)
        if values.shape != expected.shape or omega.shape != (len(expected),):
            self.quantity("candidate_file", UNEXPLAINED, detail=f"{SHIFT_CURRENT_FILE} has another shape")
            return
        scale = _largest(expected)
        excess = np.abs(values - expected) - PRINT_RELATIVE * np.abs(expected) - 1e-12 * scale
        omega_error = _largest(omega - np.asarray(self.block["omega_eV"], float))
        ok = float(np.max(excess, initial=0.0)) <= 0 and omega_error <= OMEGA_PRINT_EV
        self.quantity(
            "candidate_file",
            AGREE if ok else UNEXPLAINED,
            detail=f"{SHIFT_CURRENT_FILE} holds export_sign times {name}" + ("" if ok else ": it does not"),
        )

    def gate_clean_control(self) -> None:
        """Step 4: without a passed clean control of this build only a direct agreement is judged."""
        if self.direct_everywhere or self.clean or self.release in ("passed", "failed"):
            return
        defects = {
            item["name"]
            for item in self.quantities
            if item["name"] in SELF_CHECKS and item["class"] == UNEXPLAINED
        }
        if defects:
            # The candidate contradicts itself: that fails whatever PyATB printed.
            return
        self.precondition(
            "clean_control",
            False,
            f"{NO_CLEAN_CONTROL}: the supplied report is {self.release.replace('_', ' ')}; a difference from PyATB "
            "is judged only after this build's clean control has agreed directly",
        )

    def check_fermi_energy(self) -> bool:
        return self.precondition("fermi_energy", True, "occupation by band count (occ_band): E_F is not used")

    def run(self) -> None:
        if not self.check_documents() or not self.check_family():
            return
        for check in (self.check_inputs, self.check_parameters):
            if not self.guarded(check):
                return
        if not self.reference.ok or not self.candidate.ok:
            self.compare_failures()
            return
        for check in (
            self.check_fermi_energy,
            self.check_omega,
            self.check_spin,
            self.check_smearing,
            self.check_sign,
        ):
            if not self.guarded(check):
                return
        # The candidate's own file first: a contradiction there fails before any cause is weighed.
        for step in (self.compare_candidate_file, self.compare_shift):
            try:
                step()
            except MALFORMED as error:
                detail = f"the documents could not be compared: {type(error).__name__}: {error}"
                self.quantity(step.__name__, UNEXPLAINED, detail=detail)
                self.direct_everywhere = False
        self.gate_clean_control()

    def report(self) -> dict[str, Any]:
        report = super().report()
        report["semantics"] = to_jsonable(
            {
                "profile": "pyatb-openmx-shift-current",
                "key": None,
                "description": (
                    "PyATB's SHIFT_CURRENT on an OpenMX scfout: Hartree 27.211324570273 eV, its SI constants and "
                    "signed cell volume, its Gaussian exp(-(x/eta)^2)/(eta sqrt(pi)) truncated at int(5 eta/domega) "
                    "bins, integer bins without smearing, spin factor 2 for nspin 1 (the spin-up channel only); "
                    "QUEER's physical value is the estimator compared; PyATB's scheme is reproduced from it by derived "
                    "or recomputed steps only"
                ),
                "rules": self.rules,
                "conversions_applied": self.sign < 0 or self.legacy,
                "berry_sign_applied_to_reference": False,
            }
        )
        report["tolerance_basis"] = to_jsonable({"tier": "shift", **self.basis})
        return report


def compare_shift(
    reference: PyatbRun, candidate: QueerRun, *, scfout: str | Path | None = None, clean_control: Any = None
) -> dict[str, Any]:
    """The report of a shift-current comparison (:mod:`~queer.compat.generator.parity`'s schema). ``scfout`` (the
    candidate's, by SHA-256) lets the comparison recompute the k-points PyATB's method 1 regularizes."""
    comparison = _ShiftComparison(reference, candidate, scfout, clean_control)
    comparison.run()
    return comparison.report()


# ---------------------------------------------------------------------------
# Literature anchor
# ---------------------------------------------------------------------------

ANCHOR_REQUIRED = ("component", "peak_eV", "height_uA_V2", "scissors_eV", "eta_eV")


def literature_anchor(candidate: str | Path | Mapping[str, Any], anchor: Mapping[str, Any]) -> dict[str, Any]:
    """Step 7 of the rule: QUEER's spectrum against a published peak of one component.

    ``candidate`` is a ``QUEER_SHIFT_CURRENT`` run directory or its results
    document; ``anchor`` holds ``component`` (``xyz``), the published ``peak_eV`` and
    signed ``height_uA_V2``, the paper's ``scissors_eV``, ``eta_eV`` (PyATB's eta)
    and ``smearing`` (``gaussian``), and optionally ``window_eV`` (the photon
    energies searched) and ``convention_factor`` (+1 or -1, the paper's sign
    convention against QUEER's). ``status`` is ``placeholder`` while a required value
    is missing, ``not_comparable`` when the run's scissors or broadening differ from
    the paper's, else ``pass`` when QUEER's extremum of the published sign lies within
    2 eta of the published photon energy and within 20 % of its height, else ``fail``.
    """
    missing = [key for key in ANCHOR_REQUIRED if anchor.get(key) is None]
    if missing:
        return {"status": "placeholder", "missing": missing, "source": anchor.get("source")}
    if isinstance(candidate, Mapping):
        document = candidate
    else:
        document = json.loads((Path(candidate) / "queer_results.json").read_text(encoding="utf-8"))
    block = document["shift_current"]
    smearing = block["smearing"]
    eta = float(anchor["eta_eV"])
    scissors = float(block["occupation"]["scissors_eV"])
    problems = []
    if abs(scissors - float(anchor["scissors_eV"])) > 1e-9:
        problems.append(f"scissors {scissors} eV, the paper's {anchor['scissors_eV']} eV")
    if smearing.get("name") != str(anchor.get("smearing") or "gaussian"):
        problems.append(
            f"smearing {smearing.get('name')}, the paper's {anchor.get('smearing') or 'gaussian'}"
        )
    elif abs(float(smearing.get("eta_pyatb_eV") or 0.0) - eta) > 1e-9:
        problems.append(f"eta {smearing.get('eta_pyatb_eV')} eV, the paper's {eta} eV")
    if problems:
        return {"status": "not_comparable", "problems": problems, "source": anchor.get("source")}
    omega = np.asarray(block["omega_eV"], float)
    values = float(anchor.get("convention_factor", 1)) * np.asarray(
        block["sigma_uA_V2"][anchor["component"]], float
    )
    window = anchor.get("window_eV")
    mask = np.ones(len(omega), bool) if window is None else (omega >= window[0]) & (omega <= window[1])
    if not mask.any():
        return {"status": "not_comparable", "problems": ["no photon energy in the anchor's window"]}
    height = float(anchor["height_uA_V2"])
    index = np.flatnonzero(mask)[np.argmax(values[mask] if height > 0 else -values[mask])]
    peak, value = float(omega[index]), float(values[index])
    position_ok = abs(peak - float(anchor["peak_eV"])) <= ANCHOR_PEAK_ETA * eta
    height_ok = abs(value - height) <= ANCHOR_HEIGHT_RELATIVE * abs(height)
    return {
        "status": "pass" if position_ok and height_ok else "fail",
        "component": anchor["component"],
        "peak_eV": peak,
        "height_uA_V2": value,
        "published_peak_eV": float(anchor["peak_eV"]),
        "published_height_uA_V2": height,
        "peak_tolerance_eV": ANCHOR_PEAK_ETA * eta,
        "height_tolerance_relative": ANCHOR_HEIGHT_RELATIVE,
        "position_ok": position_ok,
        "height_ok": height_ok,
        "source": anchor.get("source"),
    }
