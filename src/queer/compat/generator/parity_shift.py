"""The shift-current parity rule: a ``QUEER_SHIFT_CURRENT`` run against PyATB's ``SHIFT_CURRENT`` run.

QUEER's reported values are its physical estimators, and they are what is judged.
Every number that lets a difference from PyATB pass is derived here from PyATB's
own table, from an exact identity between the candidate's records and its value,
or from an independent recomputation of the scfout: no magnitude the candidate
supplies (a weight, a bound, an allowance) widens a tolerance unless the scfout has
verified it. :func:`compare_shift` judges a candidate directory against a PyATB
reference directory (``Input``, optional ``inputs.sha256``, ``Out/input.json``,
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

   - ``candidate_record``: the runner always writes ``diagnostics.regularisation``,
     ``diagnostics.rounding`` (each k-point's rounding scale), with ``method 1``
     ``diagnostics.exact_degeneracies``, for a collinear scfout
     ``diagnostics.spin_channels``, and with ``smearing_method`` 0 or 1
     ``diagnostics.pyatb_scheme`` with its ``transitions_uA_V2`` (Gaussian),
     ``window_dropped_uA_V2`` and ``window_transitions`` (Gaussian without a band
     pair) or ``moved_transitions`` (bins), and with a flagged k-point its
     ``flagged_records``; a missing one is a contradiction, never an unmeasured cause;
   - ``regularisation_record``: its k-points are PyATB's grid; a k-point is flagged
     exactly when a pair spacing lies within PyATB's 1e-4 eV; without a flagged
     k-point there is no shift and the value without them is the value; with them,
     the recorded transitions of every flagged k-point, broadened on the compared
     kernel, sum to the value less the value without them, and their physical less
     their cut transitions to the shift;
   - ``emulation_consistency``: with PyATB's Gaussian, the emulation's transitions
     under QUEER's Gaussian are the value less its measured cut, and its window table
     is the Gaussian of its window record's dropped transitions (each one outside
     PyATB's window on its scale and within 6 eta of a photon energy, or one of those
     it keeps within 1e-9 eV of its ends; no Gaussian reaches farther); with PyATB's
     bins, its emulation equals ``r`` times its histogram less the measured cut plus
     the recorded moves (and those of the cut's at the flagged k-points), each row
     checked from its transition's energy and each transition recorded at an edge
     lying at one;
   - ``channel_sum`` (a collinear scfout): QUEER's physical value is its spin-up plus
     spin-down channel and ``legacy_single_channel`` PyATB's spin factor times the
     spin-up one;
   - ``candidate_file``: its ``shift_current.dat`` holds what its document says.

   These floors are a small share of the tables plus the rounding floor
   (:data:`RECORD_ROUNDING` times the kernel's peak times the sum of the k-points'
   rounding scales, each ``(1 + max|E| / s) max_c sum_t |f| |r^b r^c_;a|``: the size
   of the terms that cancel, so a response that vanishes by symmetry is no
   contradiction). With the caller's scfout (the candidate's by SHA-256) the
   comparator recomputes what the records rest on with its own Bloch sums and
   eigensolver, the transcription of PyATB's integrand
   (:mod:`~queer.compat.generator.pyatb_shift_integrand`), whose rounding at a
   k-point is the difference of the same evaluation in another orbital basis:
   ``regularisation_recomputed`` (every flagged k-point against QUEER's estimator
   called there and the transcription's covariant physical integrand, the cut
   against QUEER's with its tolerance at 1e-4 eV), ``rounding_recomputed`` (the
   largest recorded rounding scales against the scfout's, so no recorded scale
   exceeds a recomputed one: the rounding floor is verified), ``degeneracies_recomputed``
   (QUEER's degeneracy tolerance against the scfout's ``max(1e-9 eV, 10 x`` its
   Hamiltonian's adjoint residual), the recorded exact degeneracies and every
   k-point's pair spacings against the comparator's own levels),
   ``window_recomputed`` and ``moves_recomputed`` (every k-point of those records:
   their transitions must be the scfout's the record's rule names there) and
   ``spin_down_recomputed`` (a collinear scfout's spin-down channel, which PyATB never
   reads, on the whole grid).

3. **Symmetry gate** (a failure is ``differs_unexplained``): the components whose
   largest magnitude exceeds 1e-4 of the table's and the rounding floor are the same
   in both runs, and each one's integral over omega has the same sign where both
   integrals exceed 1e-4 of the component's integrated magnitude (:func:`symmetry_gate`),
   against QUEER's value or QUEER's value moved to PyATB's scheme (step 5). It is
   not judged (``not_applicable``) where PyATB's Gaussian is truncated above 1e-6 of
   its peak, where an unrecomputed method-1 cut could change a sign, or where every
   component of both runs lies within the rounding floor (no sign to compare; on an
   unrecomputed floor that also suspends every documented difference).
4. **Clean control**: scissors 0, Gaussian smearing, ``method 1``, PyATB's
   truncated Gaussian tail ``exp(-(int(5 eta/domega) domega / eta)^2) <= 1e-6``, a
   step of at most 0.4 eta (PyATB's table can then be moved to QUEER's energy scale
   from its own curve), the photon energies reaching 5 eta beyond the extreme
   transitions, no k-point where a band of a pair PyATB selects lies within 1e-3 eV of
   a level it is not exactly degenerate with (PyATB's ``get_D_degenerate`` regularizes
   ``|dE| <= 1e-4`` eV; exact degeneracies, Kramers pairs among them, are skipped by
   both codes and do not count, once ``degeneracies_recomputed`` has confirmed them
   and PyATB's own treatment of them is no cause, step 5), a collinear scfout's
   spin-down channel recomputed, and a table above its rounding floor (some
   component above ``floor / 1e-4``: a vanishing response vouches for no build).
   Such a comparison is a clean control: every component must agree directly
   (step 5), else it is ``differs_unexplained``, or ``not_comparable`` on rows PyATB's
   table cannot be moved within half the tolerance where the gap lies within the
   reach of its Hartree scale. Every ``differs_documented`` needs a passed
   clean-control report of this build (``clean_control``: the same
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
   half of ``tol`` (elsewhere directly); ``agree`` within ``tol`` plus the rounding
   floor of the k-points PyATB does not regularize once ``rounding_recomputed`` has
   verified it (without the scfout such a component is ``not_comparable``,
   "rounding"). Otherwise ``differs_documented``, reporting the physical gap, when
   PyATB's scheme reproduces it exactly (the plan's bounded causes, documented on the
   user's approval of 2026-10-17 without an unbounded cause or self-convergence):
   PyATB's table (moved to QUEER's scale, with a Gaussian) meets QUEER's value less the
   Gaussian of the transitions its window drops (the per-transition record), with
   PyATB's bins moved as recorded (bins), and with PyATB's method-1 cut at the flagged
   k-points recomputed from the scfout (step 2): QUEER's value there replaced by the
   transcription's cut value. A transition within 1e-9 eV of PyATB's window ends or of
   a bin edge may fall on either side in PyATB's arithmetic: every combination of
   their placements is tried (up to :data:`EDGE_ENUMERATION` of them; beyond, a
   component another placement might reproduce is ``not_comparable``). What a record
   documents rests on its recomputation from the scfout: without it (or the records)
   PyATB's cut, window or bins are never documented on trust, and every component
   that does not agree directly is ``not_comparable`` ("regularisation", with the
   reason), unless the candidate contradicts itself. Where an exact degeneracy takes
   part in a nonorthogonal basis or with position moments, PyATB's own value there
   (its derivative zero inside the degenerate levels, the block's velocity
   eigenvalues) depends on its eigensolver's basis inside them: estimated over
   :data:`EXACT_GAUGE_SAMPLES` bases, beyond :data:`EXACT_GAUGE_SHARE` of ``tol`` that
   is a cause no recomputation fixes (``exact_degeneracy``): no clean control, and a
   component that does not agree directly is ``not_comparable``. A Gaussian truncated
   above 1e-6 of its peak, or one whose move to QUEER's scale is unresolved on the
   rows the difference needs while the gap lies within the reach of PyATB's Hartree
   scale (:func:`energy_scale_reach`), cannot be reproduced from QUEER's value:
   ``not_comparable`` ("pyatb_scheme_resolution"); a component PyATB's table
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

For a collinear scfout PyATB computes twice the spin-up channel, which is what its
table is compared with; QUEER's physical value adds the spin-down channel, which
``channel_sum`` and ``spin_down_recomputed`` check.
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
# A transition this close (eV, PyATB's scale) to one of PyATB's bin or window edges may fall on either side.
EDGE_EV = 1e-9
# The transitions at an edge whose two placements are tried in every combination (per bin cluster or window end).
EDGE_ENUMERATION = 12
# The rounding floor: this times the kernel's peak times the sum of the k-points' rounding scales (the
# candidate's record, each (1 + max|E| / s) max_c sum_t |f| |r^b r^c_;a|_sym). Measured on the nonorthogonal SOC
# calibration scfout, whose shift current vanishes by inversion: QUEER and the transcription differ by at most
# 0.31 machine epsilon times that, so the floor holds about 1500 times what they differ by.
RECORD_ROUNDING = 1e-13
# The rounding scales verified from the scfout: the k-points with the largest recorded scales, so that no other
# k-point's scale exceeds a verified one.
ROUNDING_VERIFIED_KPOINTS = 32
# Recomputing a k-point: the transcription is evaluated twice, in the model's orbital basis and in a fixed random
# orthogonal change of it; the two differ by rounding alone, so a record must meet the transcription within this
# many times their difference (beside the conditioning bound RECOMPUTE_CONDITIONING (1 + max|E| / s) times the
# k-point's absolute weight at the kernel's peak, and RECOMPUTE_ROUNDING for QUEER's estimator called directly).
RECOMPUTE_SPREAD = 100.0
RECOMPUTE_BASIS_SEED = 20261017
RECOMPUTE_ROUNDING = 1e-12
RECOMPUTE_CONDITIONING = 1e-13
# The k-points one check recomputes at most, and the grids whose levels (N_k nbnd^3) or spin-down channel are
# recomputed everywhere.
RECOMPUTE_KPOINT_LIMIT = 4096
GRID_LEVELS_LIMIT = 2e10
DOWN_RECOMPUTE_LIMIT = 5e9
# PyATB's own value at an exact degeneracy depends on its eigensolver's basis inside it: estimated over this many
# bases (the transcription's and seeded random unitaries inside each degenerate level), doubled and summed over the
# k-points, it is no cause while below this share of the tolerance.
EXACT_GAUGE_SAMPLES = 4
EXACT_GAUGE_SHARE = 0.1
# A spacing this close to a threshold (eV) may fall on either side of it in two eigensolvers.
BORDERLINE_EV = 1e-12
BORDERLINE_RELATIVE = 1e-6
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
    "channel_sum",
    "regularisation_recomputed",
    "rounding_recomputed",
    "degeneracies_recomputed",
    "window_recomputed",
    "moves_recomputed",
    "spin_down_recomputed",
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
    reference: np.ndarray,
    candidate: np.ndarray,
    domega: float,
    tolerance: np.ndarray | None = None,
    floor: np.ndarray | float | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Gate 3: the same significant components and the same sign of each one's integral over omega.

    A component is significant in a run when its largest magnitude exceeds
    :data:`SYMMETRY_THRESHOLD` of the run's and ``floor`` (per component: the rounding
    of the values, below which a value is no sign of anything); its integral's sign
    counts when the integral exceeds :data:`SIGN_SIGNIFICANCE` of its integrated
    magnitude in both runs. ``tolerance`` (per component) is the agreement allowed at
    one photon energy: a component significant in one run only, or whose integrals'
    signs differ, fails the gate only when the runs differ by more than it (its largest
    magnitudes by more than ``tolerance``, its integrals by more than ``tolerance``
    times the photon-energy range), so values that agree are never split by a
    threshold between them. ``facts["noise"]``: no component exceeds ``floor`` in
    either run.
    """
    width = len(SHIFT_COMPONENTS)
    allowed = np.zeros(width) if tolerance is None else np.asarray(tolerance, float)
    floors = np.zeros(width) if floor is None else np.broadcast_to(np.asarray(floor, float), (width,))
    facts: dict[str, Any] = {"threshold": SYMMETRY_THRESHOLD, "sign_significance": SIGN_SIGNIFICANCE}
    sets, peaks = [], []
    for values in (reference, candidate):
        largest = float(np.max(np.abs(values))) if values.size else 0.0
        column = np.max(np.abs(values), axis=0) if values.size else np.zeros(width)
        peaks.append(column)
        sets.append(
            {
                name
                for name, value, low in zip(SHIFT_COMPONENTS, column, floors)
                if value > max(SYMMETRY_THRESHOLD * largest, low)
            }
        )
    facts["reference_components"] = sorted(sets[0], key=SHIFT_COMPONENTS.index)
    facts["candidate_components"] = sorted(sets[1], key=SHIFT_COMPONENTS.index)
    facts["noise"] = bool(floor is not None) and all(bool(np.all(column <= floors)) for column in peaks)
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


def grid_point(index: int, grid: Sequence[int]) -> np.ndarray:
    """The k-point (fractional) at ``index`` of PyATB's Gamma-origin grid in C order: ``(i/N1, j/N2, l/N3)`` at
    ``(i N2 + j) N3 + l``."""
    sizes = [int(value) for value in grid]
    if not 0 <= int(index) < int(np.prod(sizes)):
        raise ValueError(f"k-point index {index} lies outside the {sizes} grid")
    first, rest = divmod(int(index), sizes[1] * sizes[2])
    second, third = divmod(rest, sizes[2])
    return np.array([first / sizes[0], second / sizes[1], third / sizes[2]])


def is_trim(k: np.ndarray) -> bool:
    """A time-reversal-invariant k-point (every coordinate 0 or 1/2)."""
    return bool(np.all(np.isclose(np.asarray(k, float) * 2, np.round(np.asarray(k, float) * 2), atol=1e-12)))


@dataclass(frozen=True)
class ChannelView:
    """One channel of a model's blocks (or the whole spinor model) for the transcription."""

    nbnd: int
    lattice: np.ndarray
    hamiltonian_blocks: Mapping[Any, Any]
    overlap_blocks: Mapping[Any, Any]
    position_blocks: tuple[Mapping[Any, Any], ...]


def channel_view(model: Any, channel: int | None) -> ChannelView:
    """The blocks the transcription reads: every orbital of a spinor model (``channel`` None), or the rows and
    columns of one spin channel of a model without spin-orbit coupling, taken from the model's blocks here."""
    size = int(model.nbnd)
    if channel is None:
        rows = slice(0, size)
    else:
        half = size // 2 if model.spin_mode == "collinear" else size
        rows = slice(channel * half, (channel + 1) * half)

    def part(blocks: Mapping[Any, Any]) -> dict[Any, Any]:
        return {key: block[rows, rows] for key, block in blocks.items()}

    position = tuple(part(axis) for axis in getattr(model, "position_blocks", ()) or ())
    first = next(iter(part(model.hamiltonian_blocks).values()))
    return ChannelView(
        int(first.shape[0]),
        np.asarray(model.lattice, float),
        part(model.hamiltonian_blocks),
        part(model.overlap_blocks),
        position,
    )


def exactness_bound(model: Any) -> float:
    """``max(1e-9 eV, 10 x)``, ``x`` the largest ``|H(R) - H(-R)^dagger|`` (rows' absolute sums) of the scfout's own
    records: the spacing below which two levels are one degenerate level of the data, measured here."""
    data = getattr(model, "data", None)
    blocks = getattr(data, "hamiltonian", None) if data is not None else None
    if not blocks:
        return 1e-9
    largest = 0.0
    for key, block in blocks.items():
        other = blocks.get(tuple(-value for value in key))
        dense = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
        partner = (
            np.zeros_like(dense)
            if other is None
            else (other.toarray() if hasattr(other, "toarray") else np.asarray(other)).conj().T
        )
        largest = max(largest, float(np.max(np.abs(dense - partner).sum(axis=1), initial=0.0)))
    return max(1e-9, 10.0 * largest)


def orthogonal_basis(size: int) -> np.ndarray:
    """A fixed random orthogonal matrix (:data:`RECOMPUTE_BASIS_SEED`): another orbital basis for the
    transcription."""
    matrix = np.random.default_rng(RECOMPUTE_BASIS_SEED + size).standard_normal((size, size))
    basis, _ = np.linalg.qr(matrix)
    return basis


def best_assignment(
    count: int, score: Any, limit: int = EDGE_ENUMERATION
) -> tuple[tuple[bool, ...] | None, Any]:
    """The choice of ``count`` independent two-way placements that ``score`` (a tuple of flags to a comparable
    value) rates highest, trying every combination; None beyond ``limit`` choices."""
    if count > limit:
        return None, None
    best: tuple[tuple[bool, ...], Any] | None = None
    for code in range(2**count):
        flags = tuple(bool(code >> bit & 1) for bit in range(count))
        value = score(flags)
        if best is None or value > best[1]:
            best = (flags, value)
    assert best is not None
    return best


@dataclass
class FlaggedCut:
    """PyATB's method-1 cut at the flagged k-points, as the comparison could establish it."""

    verified: bool
    reason: str
    # QUEER's value at the flagged k-points less PyATB's cut value there (compared kernel), when verified.
    change: np.ndarray | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoundingFloor:
    """The rounding floor of the candidate's tables, from its k-points' recorded rounding scales."""

    recorded: bool = False
    scales: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Over every k-point (the candidate's own sums), and over the k-points PyATB's method 1 does not regularize
    # (an agreement with PyATB; the flagged k-points are recomputed from the scfout instead, and their conditioning
    # near a degeneracy would make this floor meaningless); None when those k-points are not known.
    floor: float = 0.0
    agreement: float | None = None
    flagged: frozenset[int] = frozenset()
    verified: bool = False
    reason: str = "the run kept no rounding record"


@dataclass
class WindowTerms:
    """PyATB's window from the candidate's per-transition record: the dropped transitions' Gaussian at QUEER's
    energies, and the transitions at its ends either side of which PyATB's rounding may fall."""

    fixed: np.ndarray
    edges: list[tuple[np.ndarray, bool]]
    records: dict[int, tuple[np.ndarray, np.ndarray]]


@dataclass
class BinTerms:
    """PyATB's bins from the candidate's per-transition record: the moves of the transitions placed in another
    row, and the transitions at an edge with the two rows PyATB's rounding may give them."""

    fixed: np.ndarray
    options: list[tuple[np.ndarray, np.ndarray]]
    records: dict[int, tuple[np.ndarray, np.ndarray]]


def compare_groups(
    recorded: tuple[np.ndarray, np.ndarray],
    recomputed: tuple[np.ndarray, np.ndarray],
    optional: np.ndarray,
    tolerance: float,
    join: float = EDGE_EV,
) -> tuple[float, str | None]:
    """Recorded transitions against recomputed ones of the same k-point, summed over groups of energies within
    ``join`` of each other (a degenerate level's transitions are one); a recomputed transition marked ``optional``
    (a record may hold it or not) counts only where the record has a transition of its group. Returns the worst
    difference over ``tolerance`` and the first mismatch."""
    energies = np.concatenate([recorded[0], recomputed[0]])
    if not len(energies):
        return 0.0, None
    source = np.concatenate([np.zeros(len(recorded[0]), int), np.ones(len(recomputed[0]), int)])
    weights = np.concatenate([recorded[1], recomputed[1]])
    spare = np.concatenate([np.zeros(len(recorded[0]), bool), np.asarray(optional, bool)])
    order = np.argsort(energies, kind="stable")
    labels = np.zeros(len(order), int)
    sorted_energy = energies[order]
    labels[1:] = np.cumsum(np.diff(sorted_energy) > join)
    worst, first = 0.0, None
    for label in np.unique(labels):
        members = order[labels == label]
        mine = members[source[members] == 0]
        theirs = members[source[members] == 1]
        if not len(mine):
            theirs = theirs[~spare[theirs]]
        if not len(mine) and not len(theirs):
            continue
        gap = _largest(weights[mine].sum(axis=0) - weights[theirs].sum(axis=0))
        ratio = gap / tolerance if tolerance > 0 else (math.inf if gap > 0 else 0.0)
        if ratio > worst:
            worst = ratio
            if ratio > 1 and first is None:
                first = (
                    f"{len(mine)} recorded and {len(theirs)} recomputed transitions at "
                    f"{float(energies[members].mean()):.9g} eV differ by {gap:.4g} microA/V^2 eV"
                )
    return worst, first


class _Recompute:
    """The comparator's own evaluation of k-points from the scfout: the transcription of PyATB's integrand
    (:mod:`~queer.compat.generator.pyatb_shift_integrand`) on its own Bloch sums and eigensolver, cached."""

    def __init__(self, comparison: _ShiftComparison):
        from ...models.openmx import OpenMXModel

        assert comparison.scfout is not None
        self.comparison = comparison
        self.model = OpenMXModel.from_scfout(comparison.scfout)
        spin = self.model.spin_mode
        self.views = {0: channel_view(self.model, None if spin == "soc" else 0)}
        if spin == "collinear":
            self.views[1] = channel_view(self.model, 1)
        self.exact_eV = exactness_bound(self.model)
        effective = comparison.effective
        self.occupied = int(effective["occ_band"])
        self.pairs = (
            [(int(effective["n_occ"]) - 1, int(effective["m_unocc"]) - 1)] if comparison.pair_mode else None
        )
        self.scissors = float(effective["scissors"])
        self.grid = [int(value) for value in effective["grid"]]
        volume = abs(float(np.linalg.det(np.asarray(self.model.lattice, float))))
        self.conversion = -math.pi * E_CHARGE**2 / HBAR * 1e6 / volume / comparison.kpoints
        self.cache: dict[Any, Any] = {}

    def factor(self, channel: int) -> float:
        """The recorded weights' units per transcribed weight: PyATB's conventions for the channel PyATB reads, the
        physical normalization for the spin-down channel."""
        comparison = self.comparison
        if channel == 0:
            return comparison.sign * comparison.spin_factor * self.conversion
        return self.conversion

    def point(self, index: int) -> np.ndarray:
        return grid_point(index, self.grid)

    def levels(self, k: np.ndarray, channel: int = 0) -> np.ndarray:
        from .pyatb_shift_integrand import levels

        key = ("levels", channel, tuple(np.round(np.asarray(k, float), 14)))
        if key not in self.cache:
            self.cache[key] = levels(self.views[channel], k)
        return self.cache[key]

    def _integrand(self, k: np.ndarray, channel: int, *, basis: bool) -> dict[str, Any]:
        from .pyatb_shift_integrand import queer_weights, shift_integrand

        view = self.views[channel]
        result = shift_integrand(
            view,
            k,
            self.pairs,
            energy_scale=ENERGY_SCALE,
            threshold_eV=ENERGY_SCALE * self.exact_eV,
            regularization_eV=None,
            occupied=self.occupied,
            degenerate="covariant",
            basis=orthogonal_basis(view.nbnd) if basis else None,
        )
        energies = np.asarray(result["energies_eV"], float) + self.scissors
        return {"energies": energies, "weights": queer_weights(result["integrand"]), "result": result}

    def physical(self, k: np.ndarray, channel: int = 0, *, spread: bool = False) -> dict[str, Any]:
        """QUEER's physical transitions at ``k`` by the transcription (unconverted weights, scissored energies),
        the conditioning, the magnitude bound and, with ``spread``, the differences of the same evaluation in
        another orbital basis (of the compared kernel's spectrum and of the transitions grouped by energy)."""
        point = np.asarray(k, float)
        key = ("physical", channel, tuple(np.round(point, 14)))
        entry = self.cache.get(key)
        if entry is None:
            first = self._integrand(point, channel, basis=False)
            result = first["result"]
            levels = np.asarray(result["levels_eV"], float)
            spacing = np.abs(levels[:, None] - levels[None, :])
            distinct = spacing[spacing > self.exact_eV * (1 - 1e-3)]
            condition = 1.0 + _largest(levels) / float(distinct.min()) if distinct.size else 1.0
            entry = {
                "energies": first["energies"],
                "weights": first["weights"],
                "condition": condition,
                "bound": 0.5 * np.asarray(result["magnitude_bound"], float),
                "levels": levels,
            }
            self.cache[key] = entry
        if spread and "spread" not in entry:
            second = self._integrand(point, channel, basis=True)
            comparison = self.comparison
            spectrum = comparison.kernel(entry["energies"], entry["weights"]) - comparison.kernel(
                second["energies"], second["weights"]
            )
            entry["spread"] = _largest(spectrum)
            worst, _ = compare_groups(
                (entry["energies"], entry["weights"]),
                (second["energies"], second["weights"]),
                np.zeros(len(second["energies"]), bool),
                1.0,
            )
            entry["weight_spread"] = worst
        return entry

    def tolerance(self, k: np.ndarray, channel: int = 0, *, spectrum: bool = True) -> float:
        """What a record of QUEER's transitions at ``k`` may differ from the transcription by (recorded units):
        :data:`RECOMPUTE_SPREAD` times the two bases' difference, or the conditioning bound, whichever is larger
        (of the compared kernel's spectrum, or of the transitions grouped by energy)."""
        entry = self.physical(k, channel, spread=True)
        scale = abs(self.factor(channel))
        weight = float(np.max(np.abs(entry["weights"]).sum(axis=0), initial=0.0))
        peak = self.comparison.kernel_peak() if spectrum else 1.0
        spread = entry["spread"] if spectrum else entry["weight_spread"]
        conditioned = RECOMPUTE_CONDITIONING * entry["condition"] * weight * peak
        return scale * (max(RECOMPUTE_SPREAD * spread, conditioned) + 1e-300)

    def cut(self, k: np.ndarray) -> dict[str, Any]:
        """PyATB's own method-1 integrand at ``k``: its 1e-4 eV threshold, 0.04 eV regularization and window."""
        from .pyatb_shift_integrand import shift_integrand

        start, end = self.comparison.window()
        return shift_integrand(
            self.views[0],
            k,
            self.pairs,
            energy_scale=ENERGY_SCALE,
            threshold_eV=PYATB_DEGENERATE_EV,
            regularization_eV=PYATB_SHIFT_REGULARIZATION_EV,
            occupied=self.occupied,
            window=None if self.pairs is not None else (start, end),
        )

    def pyatb_exact(self, k: np.ndarray, seed: int | None = None) -> dict[str, Any]:
        """PyATB's own treatment of an exact degeneracy at ``k`` (its cluster's derivative zero, its velocities the
        cluster block's eigenvalues), on the transitions QUEER selects; with ``seed`` in another basis inside each
        degenerate level (a seeded random unitary), as another eigensolver may return it."""
        from .pyatb_shift_integrand import clusters, queer_weights, shift_integrand

        gauge = None
        if seed is not None:
            generator = np.random.default_rng(RECOMPUTE_BASIS_SEED + 7919 * seed)
            threshold = self.exact_eV

            def gauge(vectors: np.ndarray, raw: np.ndarray) -> np.ndarray:
                out = vectors.copy()
                labels = clusters(raw, threshold)
                for label in np.unique(labels):
                    members = np.flatnonzero(labels == label)
                    if len(members) > 1:
                        size = len(members)
                        unitary, _ = np.linalg.qr(
                            generator.standard_normal((size, size))
                            + 1j * generator.standard_normal((size, size))
                        )
                        out[:, members] = out[:, members] @ unitary
                return out

        result = shift_integrand(
            self.views[0],
            k,
            self.pairs,
            energy_scale=ENERGY_SCALE,
            threshold_eV=ENERGY_SCALE * self.exact_eV,
            regularization_eV=None,
            occupied=self.occupied,
            degenerate="pyatb",
            gauge=gauge,
        )
        return {
            "energies": np.asarray(result["energies_eV"], float) + self.scissors,
            "weights": queer_weights(result["integrand"]),
        }

    def classify(self, k: np.ndarray) -> dict[str, Any]:
        """PyATB's method-1 view of ``k`` from the comparator's own levels: the smallest spacing between a band of
        a pair PyATB selects and a level it is not exactly degenerate with (scissored, QUEER's eV), whether an
        exact degeneracy takes part, and whether a spacing lies at a threshold."""
        comparison = self.comparison
        raw = self.levels(k)
        energy = raw.copy()
        energy[self.occupied :] += self.scissors
        selected = np.zeros(len(energy), bool)
        if self.pairs is not None:
            for pair in self.pairs:
                selected[list(pair)] = True
        elif 0 < self.occupied < len(energy):
            start, end = comparison.window()
            transitions = (energy[None, self.occupied :] - energy[: self.occupied, None]) * ENERGY_SCALE
            inside = (transitions >= start) & (transitions < end)
            selected[: self.occupied] = inside.any(axis=1)
            selected[self.occupied :] = inside.any(axis=0)
        exact_t = self.exact_eV
        other = selected[None, :] & ~np.eye(len(energy), dtype=bool)
        raw_spacing = np.abs(raw[:, None] - raw[None, :])
        exact = other & (raw_spacing <= exact_t)
        spacing = np.abs(energy[:, None] - energy[None, :])[other & ~exact]
        gap = float(spacing.min()) if spacing.size else math.inf
        threshold = PYATB_DEGENERATE_EV / ENERGY_SCALE
        near_exact = other & (np.abs(raw_spacing - exact_t) <= BORDERLINE_EV + BORDERLINE_RELATIVE * exact_t)
        near_threshold = (
            math.isfinite(gap) and abs(gap - threshold) <= BORDERLINE_EV + BORDERLINE_RELATIVE * threshold
        )
        return {
            "gap": gap,
            "exact": bool(exact.any()),
            "flagged": bool(gap <= threshold),
            "borderline": bool(near_exact.any() or near_threshold),
        }


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
        self.floor = RoundingFloor()
        self.recompute: _Recompute | None = None
        self.recompute_reason = "no scfout was supplied (compare with scfout=, or --scfout)"
        self.window_terms: WindowTerms | None = None
        self.bin_terms: BinTerms | None = None
        # What the recomputation from the scfout established ("verified", "unverified" or "none") and why.
        self.states: dict[str, str] = {}
        self.reasons: dict[str, str] = {}
        self.exact_differs: list[Any] = []
        # The self-checks whose agreement rests on the rounding floor (above the tables' own rounding).
        self.pending: set[str] = set()

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
            detail += (
                ": compared with legacy_single_channel (twice the spin-up channel, what PyATB computes); the "
                "physical value's spin-down channel is recomputed from the scfout (spin_down_recomputed)"
            )
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
        if not isinstance(diagnostics.get("rounding"), Mapping):
            problems.append("no diagnostics.rounding")
        if int(self.effective["method"]) == 1 and not isinstance(
            diagnostics.get("exact_degeneracies"), Mapping
        ):
            problems.append("no diagnostics.exact_degeneracies")
        if self.legacy:
            channels = diagnostics.get("spin_channels")
            if not isinstance(channels, Mapping) or not isinstance(channels.get("down_uA_V2"), Mapping):
                problems.append("no diagnostics.spin_channels of the collinear scfout's two channels")
        if self.smearing in (0, 1):
            scheme = diagnostics.get("pyatb_scheme")
            if not isinstance(scheme, Mapping) or not isinstance(scheme.get("sigma_uA_V2"), Mapping):
                problems.append("no diagnostics.pyatb_scheme.sigma_uA_V2")
            else:
                if self.smearing == 1 and not isinstance(scheme.get("transitions_uA_V2"), Mapping):
                    problems.append("no pyatb_scheme.transitions_uA_V2")
                if self.smearing == 1 and not self.pair_mode:
                    if not isinstance(scheme.get("window_dropped_uA_V2"), Mapping):
                        problems.append("no pyatb_scheme.window_dropped_uA_V2")
                    if not isinstance(scheme.get("window_transitions"), Mapping):
                        problems.append("no pyatb_scheme.window_transitions")
                if binned and not isinstance(scheme.get("moved_transitions"), Mapping):
                    problems.append("no pyatb_scheme.moved_transitions")
        self.quantity(
            "candidate_record",
            AGREE if not problems else UNEXPLAINED,
            detail="the records the runner always writes for this smearing and method are present"
            + ("" if not problems else ": they are not, " + "; ".join(problems)),
        )
        return not problems

    def rounding_floor(self, diagnostics: Mapping[str, Any]) -> RoundingFloor:
        """The candidate's rounding floor: :data:`RECORD_ROUNDING` times the kernel's peak times the sum of its
        k-points' recorded rounding scales (uniform over photon energies and components); unverified until the
        largest of them are recomputed from the scfout."""
        record = diagnostics.get("rounding") or {}
        if not record.get("recorded"):
            return RoundingFloor(reason="the run kept no record of its k-points' rounding scales")
        scales = np.asarray(record.get("kpoint_scale_uA_V2_eV") or [], float).reshape(-1)
        if len(scales) != self.kpoints or not np.all(np.isfinite(scales)) or np.any(scales < 0):
            return RoundingFloor(reason="its rounding record does not hold one finite scale per k-point")
        floor = RECORD_ROUNDING * self.kernel_peak() * float(scales.sum())
        regularisation = diagnostics.get("regularisation") or {}
        flagged = int(regularisation.get("flagged_kpoints") or 0)
        known = frozenset(self.flagged_indices(regularisation))
        agreement = None
        if not flagged or (self.flagged_items(regularisation) is not None and len(known) == flagged):
            keep = np.ones(len(scales), bool)
            keep[list(known)] = False
            agreement = RECORD_ROUNDING * self.kernel_peak() * float(scales[keep].sum())
        return RoundingFloor(
            True,
            scales,
            floor,
            agreement,
            known,
            False,
            "the rounding scales are not recomputed from the scfout",
        )

    def record_rounding(self, regularisation: Mapping[str, Any]) -> float:
        """:data:`RECORD_ROUNDING` times the flagged k-points' absolute transition weight at the kernel's peak: the
        rounding of the candidate's own sums over their records (a floor of its self-checks, never an allowance
        against PyATB)."""
        total = 0.0
        for item in self.flagged_items(regularisation) or []:
            for part in (item.get("physical"), item.get("cut")):
                if part is not None:
                    total += float(np.abs(np.asarray(part["weight_uA_V2_eV"], float)).sum())
        return RECORD_ROUNDING * total * self.kernel_peak()

    def exceeds(self, name: str, gap: float, base: float) -> bool:
        """A self-check's gap against its floor: beyond ``base`` plus the rounding floor a contradiction; between the
        two it rests on the rounding floor (``pending``), which only its recomputation from the scfout settles: until
        then no difference from PyATB is judged on such records."""
        if gap > base + self.floor.floor:
            return True
        if gap > base:
            self.pending.add(name)
        return False

    def flagged_items(self, regularisation: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
        """The recorded flagged k-points, or None when the run kept none (its record's limit)."""
        records = regularisation.get("flagged_records") or {}
        if not records.get("recorded"):
            return None
        return list(records.get("kpoints") or [])

    def flagged_indices(self, regularisation: Mapping[str, Any]) -> set[int]:
        """The grid indices of the recorded flagged k-points."""
        grid = np.asarray([int(value) for value in self.effective["grid"]])
        out = set()
        for item in self.flagged_items(regularisation) or []:
            scaled = np.round(np.asarray(item["k"], float) * grid).astype(int) % grid
            out.add(int((scaled[0] * grid[1] + scaled[1]) * grid[2] + scaled[2]))
        return out

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
            floor = CONSISTENCY_FLOOR * max(
                _largest(values), _largest(unflagged), _largest(shift)
            ) + self.record_rounding(regularisation)
            if flagged == 0:
                if self.exceeds("regularisation_record", _largest(shift), floor):
                    problems.append("it records a shift without a flagged k-point")
                if self.exceeds("regularisation_record", _largest(values - unflagged), floor):
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
        if self.exceeds("regularisation_record", _largest(values - unflagged - physical), floor):
            problems.append(
                f"its flagged k-points' transitions add {_largest(physical):.4g} where its value less its value "
                f"without them is {_largest(values - unflagged):.4g} microA/V^2 (differing by "
                f"{_largest(values - unflagged - physical):.4g})"
            )
        if self.exceeds("regularisation_record", _largest(shift - change), floor):
            problems.append(
                f"its shift differs from its flagged k-points' physical less cut transitions by "
                f"{_largest(shift - change):.4g} microA/V^2"
            )
        return problems

    def check_channels(self, diagnostics: Mapping[str, Any]) -> None:
        """``channel_sum`` (a collinear scfout): QUEER's physical value is its two channels' sum, and the legacy
        channel PyATB's spin factor times the spin-up one."""
        channels = diagnostics["spin_channels"]
        binned = self.smearing == 0
        up = table(channels["binned_up_uA_V2" if binned else "up_uA_V2"])
        down = table(channels["binned_down_uA_V2" if binned else "down_uA_V2"])
        physical = table(self.block["binned_uA_V2" if binned else "sigma_uA_V2"])
        legacy = table(self.block["legacy_single_channel"]["binned_uA_V2" if binned else "sigma_uA_V2"])
        problems = []
        if up is None or down is None or physical is None or legacy is None:
            problems.append("a channel's table is missing")
        elif not up.shape == down.shape == physical.shape == legacy.shape:
            problems.append("its channels' tables have another shape than the value")
        else:
            floor = CONSISTENCY_FLOOR * max(_largest(up), _largest(down), _largest(physical))
            if self.exceeds("channel_sum", _largest(physical - up - down), floor):
                problems.append(
                    f"its physical value differs from its spin-up plus spin-down channels by "
                    f"{_largest(physical - up - down):.4g} microA/V^2"
                )
            if self.exceeds("channel_sum", _largest(legacy - self.spin_factor * up), floor):
                problems.append(
                    f"its legacy channel differs from {self.spin_factor} x its spin-up channel by "
                    f"{_largest(legacy - self.spin_factor * up):.4g} microA/V^2"
                )
        self.quantity(
            "channel_sum",
            AGREE if not problems else UNEXPLAINED,
            detail="QUEER's physical value is its spin-up plus spin-down channel and legacy_single_channel PyATB's "
            "spin factor times the spin-up one"
            + ("" if not problems else ": they are not, " + "; ".join(problems)),
        )

    def check_emulation(self, diagnostics: Mapping[str, Any], values: np.ndarray, shift: np.ndarray) -> None:
        """``emulation_consistency``: the candidate's PyATB-side tables follow from its value and its per-transition
        records exactly."""
        scheme = diagnostics["pyatb_scheme"]
        problems: list[str] = []
        facts: dict[str, Any] = {}
        evaluated = True
        floor0 = self.record_rounding(diagnostics["regularisation"])
        if self.smearing == 1:
            # The emulation's own sum of its transitions under QUEER's Gaussian is the value less the measured cut.
            listed = table(scheme["transitions_uA_V2"])
            assert listed is not None
            target = values - shift
            if listed.shape != values.shape:
                problems.append("its transitions table has another shape than the value")
            else:
                floor = CONSISTENCY_FLOOR * max(_largest(listed), _largest(target)) + floor0
                facts["max_gap_uA_V2"] = _largest(listed - target)
                if self.exceeds("emulation_consistency", _largest(listed - target), floor):
                    row, column = np.unravel_index(int(np.argmax(np.abs(listed - target))), listed.shape)
                    problems.append(
                        f"its emulation's transitions under QUEER's Gaussian differ from its value less its cut by "
                        f"{abs(listed[row, column] - target[row, column]):.4g} microA/V^2 at "
                        f"sigma_{SHIFT_COMPONENTS[column]}, omega index {row}: they are not the same transitions"
                    )
            if not self.pair_mode:
                terms, issues = self.window_record(
                    scheme["window_transitions"], diagnostics["regularisation"]
                )
                problems += issues
                dropped = table(scheme["window_dropped_uA_V2"])
                assert dropped is not None
                if dropped.shape != values.shape:
                    problems.append("its window table has another shape than the value")
                elif terms is not None and not issues:
                    self.window_terms = terms
                    # The window table sums every dropped transition; the record those within reach (the rest add
                    # below the rounding floor) and the dropped ones at its ends.
                    recorded = terms.fixed + sum(
                        (contribution for contribution, kept in terms.edges if not kept),
                        np.zeros_like(values),
                    )
                    floor = CONSISTENCY_FLOOR * max(_largest(dropped), _largest(recorded)) + floor0
                    if self.exceeds("emulation_consistency", _largest(dropped - recorded), floor):
                        problems.append(
                            f"its window table differs from the Gaussian of the transitions its window record "
                            f"drops by {_largest(dropped - recorded):.4g} microA/V^2"
                        )
        else:
            emulated = table(scheme["sigma_uA_V2"])
            assert emulated is not None
            bin_terms, issues = self.recorded_moves(
                scheme["moved_transitions"], diagnostics["regularisation"]
            )
            problems += issues
            items = self.flagged_items(diagnostics["regularisation"])
            flagged = int(diagnostics["regularisation"].get("flagged_kpoints") or 0)
            if bin_terms is None or (flagged and items is None):
                evaluated = False
            elif not issues:
                self.bin_terms = bin_terms
                extra = np.zeros_like(values)
                for item in items or []:
                    part = _transitions(item.get("cut") or item["physical"])
                    assert part is not None
                    extra += self.kernel(*part, pyatb=True) - self.kernel(*part)
                # The runner's own placement of the transitions at an edge is its record's pyatb_row.
                moves = bin_terms.fixed + sum(
                    (option[1] for option in bin_terms.options), np.zeros_like(values)
                )
                expected = pyatb_shift_constant_ratio() * (values - shift + moves + extra)
                floor = CONSISTENCY_FLOOR * max(_largest(emulated), _largest(expected)) + floor0
                facts["max_gap_uA_V2"] = _largest(emulated - expected)
                if self.exceeds("emulation_consistency", _largest(emulated - expected), floor):
                    problems.append(
                        f"its emulation of PyATB's bins differs from r times its histogram less its cut plus its "
                        f"recorded moves by {_largest(emulated - expected):.4g} microA/V^2"
                    )
        kind = UNEXPLAINED if problems else (AGREE if evaluated else NOT_APPLICABLE)
        detail = (
            "the candidate's PyATB-side tables follow from its value: "
            + (
                "its emulation's transitions under QUEER's Gaussian sum to its value less its cut, and its window "
                "table is the Gaussian of its window record's dropped transitions, each outside PyATB's window and "
                "within reach of a photon energy, or at the window's ends"
                if self.smearing == 1
                else "its emulation of PyATB's bins is r times its histogram less its cut plus its recorded moves, "
                "each row checked from its transition's energy and each transition at an edge lying at one"
            )
            + ("" if evaluated else " (not evaluated: no record to check by)")
            + ("" if not problems else ": they do not, " + "; ".join(problems))
        )
        self.quantity("emulation_consistency", kind, facts.get("max_gap_uA_V2"), None, detail)

    def window_record(
        self, record: Mapping[str, Any], regularisation: Mapping[str, Any]
    ) -> tuple[WindowTerms | None, list[str]]:
        """PyATB's window from the candidate's record of the transitions it drops or keeps at its ends: each
        checked from its energy (outside the window on PyATB's scale and within reach of a photon energy, or at an
        end), at a k-point of the grid PyATB does not regularize."""
        if not record.get("recorded"):
            return None, []
        omega = np.asarray(self.reference_omega())
        energies = np.asarray(record["energy_eV"], float).reshape(-1)
        weights = np.asarray(record["weight_uA_V2_eV"], float).reshape(len(energies), len(SHIFT_COMPONENTS))
        indices = np.asarray(record["k_index"], int).reshape(-1)
        kept = np.asarray(record["kept"], bool).reshape(-1)
        if not len(energies) == len(indices) == len(kept):
            return None, ["its window record's lists differ in length"]
        if not np.all(np.isfinite(energies)) or not np.all(np.isfinite(weights)):
            return None, ["a window transition is not finite"]
        problems = []
        if np.any((indices < 0) | (indices >= self.kpoints)):
            problems.append("its window record names k-points outside the grid")
        flagged = self.flagged_indices(regularisation)
        if flagged & set(indices.tolist()):
            problems.append("its window record holds a k-point PyATB's method 1 regularizes")
        start, end = self.window()
        scaled = energies * ENERGY_SCALE
        inside = (scaled >= start) & (scaled < end)
        edge = (np.abs(scaled - start) < 2 * EDGE_EV) | (np.abs(scaled - end) < 2 * EDGE_EV)
        reach = WINDOW_REACH_ETA * self.eta + 2 * EDGE_EV
        near = (energies >= omega[0] - reach) & (energies <= omega[-1] + reach)
        wrong = (kept & ~edge) | (~kept & inside & ~edge) | (~kept & ~near)
        if wrong.any():
            problems.append(
                f"{int(np.count_nonzero(wrong))} window transitions lie where the record's rule puts none (first "
                f"at {float(energies[np.argmax(wrong)]):.9g} eV)"
            )
        if problems:
            return None, problems
        fixed_rows = ~kept & ~edge
        fixed = self.kernel(energies[fixed_rows], weights[fixed_rows])
        edges = [
            (self.kernel(energies[[index]], weights[[index]]), bool(kept[index]))
            for index in np.flatnonzero(edge)
        ]
        grouped: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for index in np.unique(indices):
            members = indices == index
            grouped[int(index)] = (energies[members], weights[members])
        return WindowTerms(fixed, edges, grouped), []

    def recorded_moves(
        self, record: Mapping[str, Any], regularisation: Mapping[str, Any]
    ) -> tuple[BinTerms | None, list[str]]:
        """The bins' recorded moves (PyATB's rows less QUEER's) of the transitions not at an edge, and for each one
        at an edge its two possible PyATB rows (the recorded one first), with any record the comparator's own
        placement contradicts."""
        if not record.get("recorded"):
            return None, []
        omega = np.asarray(self.reference_omega())
        start, count = float(omega[0]), len(omega)
        energies = np.asarray(record["energy_eV"], float).reshape(-1)
        weights = np.asarray(record["weight_uA_V2_eV"], float).reshape(len(energies), len(SHIFT_COMPONENTS))
        own = np.asarray(record["queer_row"], int).reshape(-1)
        theirs = np.asarray(record["pyatb_row"], int).reshape(-1)
        ambiguous = np.asarray(record["ambiguous"], bool).reshape(-1)
        indices = np.asarray(record.get("k_index") or [], int).reshape(-1)
        problems = []
        if not len(energies) == len(own) == len(theirs) == len(ambiguous) == len(indices):
            return None, ["its moved transitions' lists differ in length"]
        if not np.all(np.isfinite(energies)) or not np.all(np.isfinite(weights)):
            return None, ["a moved transition is not finite"]
        if np.any((indices < 0) | (indices >= self.kpoints)):
            problems.append("its moved transitions name k-points outside the grid")
        if self.flagged_indices(regularisation) & set(indices.tolist()):
            problems.append("its moved transitions hold a k-point PyATB's method 1 regularizes")
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
        # A transition at an edge lies within 1e-9 eV of one of QUEER's or PyATB's edges (or PyATB's window ends):
        # each code's rounding may put it on either side.
        position = (energies - start) / self.domega
        scaled_position = (energies * ENERGY_SCALE - start) / self.domega
        at_edge = (np.abs(position - np.round(position)) * self.domega < 2 * EDGE_EV) | (
            np.abs(scaled_position - np.round(scaled_position)) * self.domega < 2 * EDGE_EV
        )
        lower_own = queer_rows(energies - 2 * EDGE_EV, start, self.domega, count)
        upper_own = queer_rows(energies + 2 * EDGE_EV, start, self.domega, count)
        lower = pyatb_rows(energies - 2 * EDGE_EV / ENERGY_SCALE, start, self.domega, count, self.pair_mode)
        upper = pyatb_rows(energies + 2 * EDGE_EV / ENERGY_SCALE, start, self.domega, count, self.pair_mode)
        stray = ambiguous & (
            ~at_edge | ((own != lower_own) & (own != upper_own)) | ((theirs != lower) & (theirs != upper))
        )
        if stray.any():
            problems.append(
                f"{int(np.count_nonzero(stray))} transitions recorded at an edge lie at none, or in a row neither "
                f"side of it gives (first at {float(energies[np.argmax(stray)]):.9g} eV)"
            )
        if problems:
            return None, problems
        fixed_rows = ~ambiguous
        fixed = place(theirs[fixed_rows], weights[fixed_rows], count) - place(
            own[fixed_rows], weights[fixed_rows], count
        )
        options = []
        for index in np.flatnonzero(ambiguous):
            base = -place(own[[index]], weights[[index]], count)
            other = lower[index] if lower[index] != theirs[index] else upper[index]
            options.append(
                (
                    base + place(np.asarray([other]), weights[[index]], count),
                    base + place(theirs[[index]], weights[[index]], count),
                )
            )
        grouped: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for index in np.unique(indices):
            members = indices == index
            grouped[int(index)] = (energies[members], weights[members])
        return BinTerms(fixed, options, grouped), []

    # ------------------------------------------------- the recomputation
    def recomputation(self) -> _Recompute | None:
        """The comparator's recomputation from the caller's scfout, when it is the candidate's (by SHA-256)."""
        if self.recompute is not None:
            return self.recompute
        if self.scfout is None:
            self.recompute_reason = "no scfout was supplied (compare with scfout=, or --scfout)"
            return None
        digest = sha256_file(self.scfout)
        recorded = self.candidate.block("model").get("scfout_sha256")
        if digest is None or digest != recorded:
            self.recompute_reason = (
                f"the supplied scfout {self.scfout} (SHA-256 {str(digest)[:16]}...) is not the candidate's "
                f"({str(recorded)[:16]}...)"
            )
            return None
        try:
            self.recompute = _Recompute(self)
        except (OSError, ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            self.recompute_reason = f"the scfout could not be read: {type(error).__name__}: {error}"
            return None
        return self.recompute

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
        recompute = self.recomputation()
        if recompute is None:
            reason = self.recompute_reason
            if reason.startswith("no scfout"):
                reason = (
                    f"no scfout was supplied to recompute its {flagged} flagged k-points (compare with scfout=, or "
                    "--scfout)"
                )
            return FlaggedCut(False, reason)
        if float(self.effective["scissors"]) != 0.0:
            return FlaggedCut(
                False, "a nonzero scissors changes PyATB's denominators: its cut is not recomputed"
            )
        try:
            checks = self.recompute_items(items, recompute)
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
            "tolerances derived from the recomputation (its rounding, measured as the difference of the same "
            "evaluation in another orbital basis; worst ratio to its tolerance given as the metric)"
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

    def recompute_items(self, items: Sequence[Mapping[str, Any]], recompute: _Recompute) -> dict[str, Any]:
        """Each recorded flagged k-point against QUEER's estimator at that k-point and PyATB's integrand there."""
        import dataclasses

        from ...observables.nonlinear import shift_current_point, shift_current_settings
        from .openmx_steps import single_channel_model
        from .pyatb_shift_integrand import queer_weights

        model = recompute.model
        view = model if model.spin_mode == "soc" else single_channel_model(model)
        effective = self.effective
        occupied = int(effective["occ_band"])
        pairs = recompute.pairs
        omega = np.asarray(self.reference_omega())
        factor = recompute.factor(0)
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
        for item in items:
            k = np.asarray(item["k"], float)
            physical = _transitions(item["physical"])
            cut = _transitions(item.get("cut"))
            assert physical is not None
            recorded = self.kernel(*physical)
            direct = own(shift_current_point(view, k, settings))
            reference = self.kernel(*direct)
            entry = recompute.physical(k, spread=True)
            transcribed = self.kernel(entry["energies"], factor * entry["weights"])
            # QUEER's estimator called at its own entry point repeats the record's sums in another order; the
            # transcription differs by rounding, measured from its two orbital bases.
            tolerance = recompute.tolerance(k)
            rounding = RECOMPUTE_ROUNDING * abs(factor) * peak * float(
                np.max(np.abs(entry["weights"]).sum(axis=0), initial=0.0)
            ) + CONSISTENCY_FLOOR * _largest(reference)
            rounding = max(rounding, tolerance / RECOMPUTE_SPREAD)
            ratios = {
                "queer": _largest(recorded - reference) / rounding,
                "transcription": _largest(recorded - transcribed)
                / (tolerance + CONSISTENCY_FLOOR * _largest(reference)),
            }
            if cut is not None:
                direct_cut = own(shift_current_point(view, k, cut_settings))
                ratios["queer_cut"] = _largest(self.kernel(*cut) - self.kernel(*direct_cut)) / rounding
            port_cut = recompute.cut(k)
            levels = np.asarray(port_cut["levels_eV"], float)
            selected = sorted({band for pair in port_cut["pairs"] for band in pair})
            raw = recompute.levels(k)
            near = [
                abs(levels[band] - levels[other]) * ENERGY_SCALE
                for band in selected
                for other in range(len(levels))
                if other != band and recompute.exact_eV < abs(raw[band] - raw[other])
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
                    "conditioning": entry["condition"],
                    "ratios": ratios,
                    "max_change_uA_V2": _largest(recorded - cut_value),
                }
            )
        return {"problems": problems, "change": change, "worst_ratio": worst, "kpoints": per_k}

    def verify_rounding(self) -> None:
        """``rounding_recomputed``: the largest recorded rounding scales are no larger than the scfout gives (each
        k-point's conditioning times its transitions' magnitude bound, by the transcription), so no recorded scale
        exceeds a recomputed one; the floor is then verified."""
        floor = self.floor
        if not floor.recorded:
            return
        recompute = self.recomputation()
        if recompute is None:
            floor.reason = f"the rounding scales are not recomputed: {self.recompute_reason}"
            return
        # The largest scales outside the flagged k-points (recomputed on their own): no other exceeds the smallest
        # of them.
        candidates = np.asarray(
            [index for index in range(len(floor.scales)) if index not in floor.flagged], int
        )
        order = candidates[np.argsort(-floor.scales[candidates], kind="stable")][:ROUNDING_VERIFIED_KPOINTS]
        scale = abs(recompute.factor(0))
        worst = 0.0
        problems: list[str] = []
        for index in order:
            k = recompute.point(int(index))
            entry = recompute.physical(k)
            allowed = (1 + 1e-6) * entry["condition"] * float(np.max(entry["bound"], initial=0.0)) * scale
            ratio = (
                float(floor.scales[index]) / allowed
                if allowed > 0
                else (math.inf if floor.scales[index] > 0 else 0.0)
            )
            worst = max(worst, ratio)
            if ratio > 1 and len(problems) < 3:
                problems.append(
                    f"k-point {k.tolist()}: its recorded rounding scale {float(floor.scales[index]):.4g} exceeds the "
                    f"scfout's {allowed:.4g}"
                )
        floor.verified = not problems
        floor.reason = (
            f"the {len(order)} largest rounding scales recomputed from the scfout"
            if not problems
            else "the recorded rounding scales exceed their recomputation"
        )
        self.quantity(
            "rounding_recomputed",
            AGREE if not problems else UNEXPLAINED,
            worst,
            1.0,
            detail=f"the {len(order)} largest recorded rounding scales against the scfout's (the transcription's "
            "conditioning times its transitions' magnitudes, Frobenius norms over degenerate levels): none may "
            "exceed it, so no recorded scale exceeds a recomputed one and the rounding floor "
            f"{floor.floor:.4g} microA/V^2 rests on the scfout"
            + ("" if not problems else ": " + "; ".join(problems)),
        )

    def verify_degeneracies(self, diagnostics: Mapping[str, Any], tolerance: np.ndarray) -> None:
        """``degeneracies_recomputed``: QUEER's degeneracy tolerance and its exact and near degeneracies against the
        comparator's own levels; and at each exact degeneracy, whether PyATB's own treatment of it (its cluster's
        derivative zero, its velocities the block's eigenvalues) meets the physical integrand there."""
        count = int(diagnostics.get("exact_degenerate_kpoints") or 0)
        block = diagnostics.get("exact_degeneracies") or {}
        self.states["exact"] = "none" if count == 0 else "unverified"
        recompute = self.recomputation()
        if recompute is None:
            if count:
                self.reasons["exact"] = (
                    f"{count} k-points with an exact degeneracy, whose classification rests on QUEER's degeneracy "
                    f"tolerance until it is recomputed: {self.recompute_reason}"
                )
            return
        problems: list[str] = []
        exact_t = recompute.exact_eV
        model = self.candidate.block("model")
        for label, value in (
            ("its model's degeneracy tolerance", model.get("degeneracy_tolerance_eV")),
            ("its exact-degeneracy record's tolerance", block.get("degeneracy_tolerance_eV")),
        ):
            if value is None or abs(float(value) - exact_t) > 1e-6 * exact_t:
                problems.append(
                    f"{label} {value} eV is not the scfout's max(1e-9 eV, 10 x its Hamiltonian's adjoint residual) "
                    f"= {exact_t:.6g} eV"
                )
        items = list(block.get("kpoints") or []) if block.get("recorded") else []
        if count and not block.get("recorded"):
            self.reasons["exact"] = "the run kept no record of its k-points with an exact degeneracy"
        elif len(items) != count:
            problems.append(f"it records {len(items)} of its {count} k-points with an exact degeneracy")
        recorded_exact = set()
        for item in items:
            index = int(item["k_index"])
            k = recompute.point(index)
            if _largest(np.asarray(item.get("k") or k, float) - k) > 1e-12:
                problems.append(f"its exact degeneracy at index {index} names another k-point")
                continue
            recorded_exact.add(index)
            levels = recompute.levels(k)
            for i, j, _ in item.get("pairs") or []:
                spacing = abs(float(levels[int(j)] - levels[int(i)]))
                if spacing > exact_t * (1 + 1e-3) + BORDERLINE_EV:
                    problems.append(
                        f"k-point {k.tolist()}: levels {int(i)} and {int(j)} are {spacing:.3g} eV apart, not one "
                        f"degenerate level (the scfout's {exact_t:.3g} eV)"
                    )
        regularisation = diagnostics.get("regularisation") or {}
        flagged_k = self.flagged_indices(regularisation)
        nbnd = int(recompute.views[0].nbnd)
        full = self.kpoints * nbnd**3 <= GRID_LEVELS_LIMIT
        grid_indices = (
            range(self.kpoints)
            if full
            else sorted(
                recorded_exact
                | flagged_k
                | {index for index in range(self.kpoints) if is_trim(recompute.point(index))}
            )
        )
        found_exact, found_flagged, borderline, smallest = set(), set(), 0, math.inf
        for index in grid_indices:
            facts = recompute.classify(recompute.point(int(index)))
            if facts["borderline"]:
                borderline += 1
                continue
            if facts["exact"]:
                found_exact.add(int(index))
            if facts["flagged"]:
                found_flagged.add(int(index))
            smallest = min(smallest, facts["gap"])
        flagged = int(regularisation.get("flagged_kpoints") or 0)
        if full:
            if abs(len(found_exact) - count) > borderline:
                problems.append(
                    f"the scfout has {len(found_exact)} k-points with an exact degeneracy in PyATB's pairs, the "
                    f"candidate {count}"
                )
            if int(self.effective["method"]) == 1 and abs(len(found_flagged) - flagged) > borderline:
                problems.append(
                    f"the scfout has {len(found_flagged)} k-points PyATB's method 1 regularizes, the candidate flags "
                    f"{flagged}"
                )
            recorded_gap = diagnostics.get("min_pair_gap_eV")
            if not borderline and math.isfinite(smallest) and recorded_gap is not None:
                if abs(float(recorded_gap) - smallest) > 1e-9 + 1e-6 * smallest:
                    problems.append(
                        f"the scfout's smallest pair spacing is {smallest:.9g} eV, the candidate's {recorded_gap}"
                    )
        else:
            missed = (found_exact - recorded_exact) | (recorded_exact - found_exact)
            if missed and items:
                problems.append(f"{len(missed)} k-points' exact degeneracies differ from the scfout's")
        if problems:
            self.states["exact"] = "contradicted"
        elif count and block.get("recorded"):
            self.states["exact"] = "verified"
            # PyATB's own value at an exact degeneracy depends on its eigensolver's basis inside the degenerate
            # levels (in a nonorthogonal basis, or with position moments): its range, estimated over
            # EXACT_GAUGE_SAMPLES bases and doubled, summed over those k-points, must stay below
            # EXACT_GAUGE_SHARE of the tolerance.
            factor = recompute.factor(0)
            total = np.zeros(len(SHIFT_COMPONENTS))
            details = []
            for index in sorted(recorded_exact):
                k = recompute.point(index)
                entry = recompute.physical(k)
                physical = self.kernel(entry["energies"], factor * entry["weights"])
                deviation = np.zeros(len(SHIFT_COMPONENTS))
                for seed in (None, *range(1, EXACT_GAUGE_SAMPLES)):
                    literal = recompute.pyatb_exact(k, seed)
                    spectrum = self.kernel(literal["energies"], factor * literal["weights"])
                    deviation = np.maximum(deviation, np.max(np.abs(spectrum - physical), axis=0))
                ambiguity = 2 * deviation + recompute.tolerance(k)
                total += ambiguity
                details.append({"k": k.tolist(), "range_uA_V2": float(np.max(ambiguity))})
            allowed = EXACT_GAUGE_SHARE * np.asarray(tolerance, float)
            if np.any(total > allowed):
                self.exact_differs = details
                column = int(np.argmax(total / allowed))
                self.reasons["exact"] = (
                    f"at {len(details)} k-points with an exact degeneracy PyATB's own treatment of it (its cluster's "
                    "derivative zero, its velocities the block's eigenvalues) depends on its eigensolver's basis "
                    f"inside the degenerate levels: its value there may move sigma_{SHIFT_COMPONENTS[column]} by "
                    f"{float(total[column]):.4g} microA/V^2, beyond {EXACT_GAUGE_SHARE:g} of the tolerance, which no "
                    "recomputation fixes"
                )
        where = "every k-point" if full else "the recorded, flagged and time-reversal-invariant k-points"
        self.quantity(
            "degeneracies_recomputed",
            AGREE if not problems else UNEXPLAINED,
            detail=f"QUEER's degeneracy tolerance, its {count} k-points with an exact degeneracy and its pair "
            f"spacings against the scfout's own levels ({where}; {borderline} at a threshold not judged)"
            + (f"; PyATB's own treatment differs at {len(self.exact_differs)}" if self.exact_differs else "")
            + ("" if not problems else ": " + "; ".join(problems)),
        )

    def verify_window(self) -> None:
        """``window_recomputed``: every k-point of the window record recomputed from the scfout; its transitions
        must be the scfout's that the record's rule names there."""
        terms = self.window_terms
        if terms is None:
            return
        self.states["window"] = "unverified"
        recompute = self.recomputation()
        if recompute is None:
            self.reasons["window"] = f"its window record is not recomputed: {self.recompute_reason}"
            return
        if len(terms.records) > RECOMPUTE_KPOINT_LIMIT:
            self.reasons["window"] = (
                f"its window record spans {len(terms.records)} k-points, more than the {RECOMPUTE_KPOINT_LIMIT} "
                "recomputed"
            )
            return
        omega = np.asarray(self.reference_omega())
        start, end = self.window()
        reach = WINDOW_REACH_ETA * self.eta
        factor = recompute.factor(0)
        worst = 0.0
        problems: list[str] = []
        for index, recorded in sorted(terms.records.items()):
            k = recompute.point(index)
            entry = recompute.physical(k, spread=True)
            energies, weights = entry["energies"], factor * entry["weights"]
            scaled = energies * ENERGY_SCALE
            inside = (scaled >= start) & (scaled < end)
            distance = np.minimum(np.abs(scaled - start), np.abs(scaled - end))
            low, high = omega[0] - reach, omega[-1] + reach
            near = (energies >= low) & (energies <= high)
            rule = (~inside & near) | (inside & (distance < EDGE_EV))
            # Where the two codes' energies (1e-14 eV apart) may fall either side of the record's rule: a record
            # may hold such a transition or not.
            optional = ((distance >= EDGE_EV / 2) & (distance < 2 * EDGE_EV)) | (
                np.minimum(np.abs(energies - low), np.abs(energies - high)) < 2 * EDGE_EV
            )
            select = rule | optional
            ratio, first = compare_groups(
                recorded,
                (energies[select], weights[select]),
                optional[select],
                recompute.tolerance(k, spectrum=False),
            )
            worst = max(worst, ratio)
            if first is not None and len(problems) < 3:
                problems.append(f"k-point {k.tolist()}: {first}")
        self.states["window"] = "verified" if not problems else "contradicted"
        self.quantity(
            "window_recomputed",
            AGREE if not problems else UNEXPLAINED,
            worst,
            1.0,
            detail=f"the window record's {len(terms.records)} k-points recomputed from the scfout: the transitions "
            "PyATB's window drops within reach of a photon energy and those it keeps at its ends are the scfout's"
            + ("" if not problems else ": they are not, " + "; ".join(problems)),
        )

    def verify_moves(self) -> None:
        """``moves_recomputed``: every k-point of the bins' record recomputed from the scfout; at each the recorded
        transitions must be the scfout's that PyATB's bins place in another row or that lie at an edge."""
        terms = self.bin_terms
        if terms is None:
            return
        self.states["moves"] = "unverified"
        recompute = self.recomputation()
        if recompute is None:
            self.reasons["moves"] = (
                f"its record of the moved transitions is not recomputed: {self.recompute_reason}"
            )
            return
        if len(terms.records) > RECOMPUTE_KPOINT_LIMIT:
            self.reasons["moves"] = (
                f"its moved transitions span {len(terms.records)} k-points, more than the {RECOMPUTE_KPOINT_LIMIT} "
                "recomputed"
            )
            return
        omega = np.asarray(self.reference_omega())
        start, count = float(omega[0]), len(omega)
        factor = recompute.factor(0)
        worst = 0.0
        problems: list[str] = []
        for index, recorded in sorted(terms.records.items()):
            k = recompute.point(index)
            entry = recompute.physical(k, spread=True)
            energies, weights = entry["energies"], factor * entry["weights"]
            moved = queer_rows(energies, start, self.domega, count) != pyatb_rows(
                energies, start, self.domega, count, self.pair_mode
            )
            position = (energies - start) / self.domega
            scaled_position = (energies * ENERGY_SCALE - start) / self.domega
            distance = (
                np.minimum(
                    np.abs(position - np.round(position)), np.abs(scaled_position - np.round(scaled_position))
                )
                * self.domega
            )
            optional = (distance >= EDGE_EV / 2) & (distance < 2 * EDGE_EV)
            select = moved | (distance < EDGE_EV) | optional
            ratio, first = compare_groups(
                recorded,
                (energies[select], weights[select]),
                optional[select] & ~moved[select],
                recompute.tolerance(k, spectrum=False),
            )
            worst = max(worst, ratio)
            if first is not None and len(problems) < 3:
                problems.append(f"k-point {k.tolist()}: {first}")
        self.states["moves"] = "verified" if not problems else "contradicted"
        self.quantity(
            "moves_recomputed",
            AGREE if not problems else UNEXPLAINED,
            worst,
            1.0,
            detail=f"the moved transitions' {len(terms.records)} k-points recomputed from the scfout: the transitions "
            "PyATB's bins place in another row than QUEER's histogram, and those at an edge, are the scfout's"
            + ("" if not problems else ": they are not, " + "; ".join(problems)),
        )

    def verify_down(self, diagnostics: Mapping[str, Any]) -> None:
        """``spin_down_recomputed`` (a collinear scfout): the physical value's spin-down channel, which PyATB never
        reads, recomputed from the scfout on the whole grid by the transcription."""
        self.states["down"] = "unverified"
        recompute = self.recomputation()
        if recompute is None:
            self.reasons["down"] = f"its spin-down channel is not recomputed: {self.recompute_reason}"
            return
        view = recompute.views.get(1)
        if view is None:
            self.reasons["down"] = "the scfout has no spin-down channel"
            return
        if self.kpoints * int(view.nbnd) ** 3 > DOWN_RECOMPUTE_LIMIT:
            self.reasons["down"] = (
                f"its spin-down channel is not recomputed: {self.kpoints} k-points of {view.nbnd} levels exceed "
                f"the recomputation's limit ({DOWN_RECOMPUTE_LIMIT:g} N_k nbnd^3)"
            )
            return
        channels = diagnostics["spin_channels"]
        recorded = table(channels["binned_down_uA_V2" if self.smearing == 0 else "down_uA_V2"])
        assert recorded is not None
        factor = recompute.factor(1)
        total = np.zeros_like(recorded)
        scales = 0.0
        for index in range(self.kpoints):
            k = recompute.point(index)
            entry = recompute.physical(k, channel=1)
            total += self.kernel(entry["energies"], factor * entry["weights"])
            scales += entry["condition"] * float(np.max(entry["bound"], initial=0.0)) * abs(factor)
        floor = RECORD_ROUNDING * self.kernel_peak() * scales + CONSISTENCY_FLOOR * _largest(total)
        gap = _largest(recorded - total)
        ok = gap <= floor
        self.states["down"] = "verified" if ok else "contradicted"
        self.quantity(
            "spin_down_recomputed",
            AGREE if ok else UNEXPLAINED,
            gap,
            floor,
            detail="the physical value's spin-down channel (which PyATB never reads) recomputed from the scfout on "
            f"every k-point by the transcription: it meets the candidate's to {gap:.4g} microA/V^2 (its rounding "
            f"floor {floor:.4g})" + ("" if ok else ": it does not"),
        )

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
            "exact_degeneracy": {
                "active": bool(self.exact_differs),
                "bounded": False,
                "detail": self.reasons.get("exact")
                or "PyATB's own treatment of every exact degeneracy meets the physical integrand there, or none "
                "takes part",
            },
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
                "(QUEER's per-transition record of what it drops, recomputed from the scfout), its integer bins "
                "(the recorded moves, recomputed from the scfout) and its Gaussian truncation (at most 1e-6 of a "
                "transition's peak, else not reproducible)",
            },
        }

    def clean_conditions(self, diagnostics: Mapping[str, Any], reference: np.ndarray) -> dict[str, bool]:
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
        exact = int(diagnostics.get("exact_degenerate_kpoints") or 0)
        columns = np.max(np.abs(reference), axis=0, initial=0.0)
        bar = self.floor.floor if self.floor.agreement is None else self.floor.agreement
        nontrivial = self.floor.recorded and bool(np.any(columns > bar / AGREEMENT_RELATIVE))
        return {
            "scissors_zero": float(self.effective["scissors"]) == 0.0,
            "gaussian": self.smearing == 1,
            "method_1": int(self.effective["method"]) == 1,
            "truncation_tail_below_1e-6": self.smearing == 1
            and self.truncation_tail() <= CLEAN_TRUNCATION_TAIL,
            "step_at_most_0.4_eta": self.smearing == 1 and self.domega <= SCALE_MAX_STEP_ETA * eta,
            "window_margin_5_eta": bool(margin),
            "no_pair_within_1e-4_eV": measured and flagged == 0,
            "pair_gap_above_1e-3_eV": measured and (gap is None or float(gap) > CLEAN_PAIR_GAP_EV),
            "exact_degeneracies_verified": exact == 0
            or (self.states.get("exact") == "verified" and not self.exact_differs),
            "spin_down_recomputed": not self.legacy or self.states.get("down") == "verified",
            "nontrivial_table": nontrivial,
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
        or ``unverifiable`` (a flagged k-point not recomputed, or a per-transition record the scheme rests on not
        recomputed from the scfout), with what entered."""
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
        if not verified or cut.change is None:
            # Unrecomputed, PyATB's cut is unknown: nothing is reproduced from a claim.
            route["status"] = ["unverifiable"] * width
            route["reason"] = cut.reason
            return route
        if self.exact_differs:
            route["status"] = ["unverifiable"] * width
            route["reason"] = self.reasons["exact"]
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
            terms = self.window_terms
            if not self.pair_mode and terms is None:
                route["status"] = ["unresolved"] * width
                route["reason"] = (
                    "the run kept no record of the transitions PyATB's window drops "
                    f"({((diagnostics.get('pyatb_scheme') or {}).get('window_transitions') or {}).get('reason')})"
                )
                return route
            fixed = np.zeros_like(values) if terms is None else terms.fixed
            edges = [] if terms is None else terms.edges
            resolved = margin <= SCALE_RESOLVED * tolerance[None, :]
            reach = energy_scale_reach(reference, omega=omega, eta=self.eta, domega=self.domega)

            def evaluate(flags: tuple[bool, ...]) -> tuple[list[str], np.ndarray, np.ndarray]:
                # flags[t]: PyATB drops the transition at an edge.
                dropped = fixed + sum(
                    (contribution for (contribution, _), drop in zip(edges, flags) if drop),
                    np.zeros_like(values),
                )
                target = values - dropped - removed
                close = np.abs(reference - target) <= tolerance[None, :]
                moved_ok = resolved & (np.abs(moved_x - target) <= tolerance[None, :] + margin)
                failing = ~close & ~moved_ok
                beyond_reach = np.abs(reference - target) > tolerance[None, :] + reach
                status = []
                for column in range(width):
                    rows = failing[:, column]
                    if np.any(rows & (resolved[:, column] | beyond_reach[:, column])):
                        status.append("not_reproduced")
                    elif rows.any():
                        status.append("unresolved")
                    else:
                        status.append("reproduced")
                return status, target, dropped

            def score(flags: tuple[bool, ...]) -> tuple[int, int]:
                status = evaluate(flags)[0]
                return status.count("reproduced"), -status.count("not_reproduced")

            choice, _ = best_assignment(len(edges), score)
            if choice is None:
                # Too many to try: the runner's placement stands where it reproduces PyATB's table; elsewhere
                # another placement might.
                choice = tuple(not kept for _, kept in edges)
                status, target, dropped = evaluate(choice)
                spread = sum((np.abs(contribution) for contribution, _ in edges), np.zeros_like(values))
                status = [
                    "unresolved"
                    if state == "not_reproduced" and _largest(spread[:, column]) > tolerance[column]
                    else state
                    for column, state in enumerate(status)
                ]
                route["reason"] = (
                    f"{len(edges)} transitions lie within {EDGE_EV:g} eV of PyATB's window ends, more than the "
                    f"{EDGE_ENUMERATION} whose placements are tried in every combination"
                )
            else:
                status, target, dropped = evaluate(choice)
            route["edge_transitions"] = len(edges)
            route["edge_dropped"] = int(sum(choice))
            if not self.pair_mode and self.states.get("window") != "verified":
                status = [
                    "unverifiable"
                    if state == "reproduced" and _largest(dropped[:, column]) > tolerance[column]
                    else state
                    for column, state in enumerate(status)
                ]
                if "unverifiable" in status:
                    route["reason"] = self.reasons.get("window") or "its window record is not recomputed"
            route["target"] = target
            route["dropped_max"] = np.max(np.abs(dropped), axis=0)
            route["removed_max"] = np.max(np.abs(removed), axis=0)
            route["status"] = status
            if "unresolved" in status and route["reason"] is None:
                route["reason"] = (
                    "PyATB's table cannot be moved to QUEER's energy scale within half the tolerance on rows the "
                    f"difference needs (it moves on {SCALE_STENCIL} or more photon energies at most "
                    f"{SCALE_MAX_STEP_ETA:g} eta apart; here {len(omega)} at {self.domega / self.eta:.3g} eta): refine "
                    "domega against eta"
                )
            return route
        terms_b = self.bin_terms
        if terms_b is None:
            route["status"] = ["unresolved"] * width
            route["reason"] = (
                "the run kept no record of the transitions PyATB's bins move "
                f"({((diagnostics.get('pyatb_scheme') or {}).get('moved_transitions') or {}).get('reason')})"
            )
            return route
        ratio = pyatb_shift_constant_ratio()

        def bins(flags: tuple[bool, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            # flags[t]: the transition at an edge keeps its recorded PyATB row.
            moves = terms_b.fixed + sum(
                (option[1] if keep else option[0] for option, keep in zip(terms_b.options, flags)),
                np.zeros_like(values),
            )
            target = ratio * (values - removed + moves)
            fits = np.max(np.abs(reference - target) - tolerance[None, :], axis=0) <= 0
            return target, fits, moves

        choice, _ = best_assignment(len(terms_b.options), lambda flags: int(bins(flags)[1].sum()))
        if choice is None:
            choice = tuple(True for _ in terms_b.options)
            target, fits, moves = bins(choice)
            spread = sum((np.abs(option[0] - option[1]) for option in terms_b.options), np.zeros_like(values))
            status = [
                "reproduced"
                if fit
                else ("unresolved" if _largest(spread[:, column]) > tolerance[column] else "not_reproduced")
                for column, fit in enumerate(fits)
            ]
            route["reason"] = (
                f"{len(terms_b.options)} transitions lie within {EDGE_EV:g} eV of a bin edge, more than the "
                f"{EDGE_ENUMERATION} whose placements are tried in every combination"
            )
        else:
            target, fits, moves = bins(choice)
            status = ["reproduced" if fit else "not_reproduced" for fit in fits]
        if self.states.get("moves") != "verified":
            status = [
                "unverifiable"
                if state == "reproduced" and _largest(moves[:, column]) > tolerance[column]
                else state
                for column, state in enumerate(status)
            ]
            if "unverifiable" in status:
                route["reason"] = (
                    self.reasons.get("moves") or "its record of the moved transitions is not recomputed"
                )
        route["target"] = target / ratio
        route["moved_max"] = np.max(np.abs(moves), axis=0)
        route["removed_max"] = np.max(np.abs(removed), axis=0)
        route["edge_transitions"] = len(terms_b.options)
        route["status"] = status
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
        if records:
            self.floor = self.rounding_floor(diagnostics)
        shift = np.zeros_like(values)
        if records:
            recorded_shift = table(regularisation["binned_shift_uA_V2" if binned else "shift_uA_V2"])
            unflagged = table(regularisation["binned_unflagged_uA_V2" if binned else "sigma_unflagged_uA_V2"])
            assert recorded_shift is not None and unflagged is not None
            shift = recorded_shift
            self.check_regularisation_record(diagnostics, values, shift, unflagged)
            if self.legacy:
                self.check_channels(diagnostics)
        moved_x: np.ndarray | None = None
        margin = np.full_like(values, np.inf)
        if self.smearing == 1 and len(omega) >= SCALE_STENCIL and domega <= SCALE_MAX_STEP_ETA * self.eta:
            moved_x, margin = move_to_queer_scale(reference, omega=omega, eta=self.eta, domega=domega)
        resolved = margin <= SCALE_RESOLVED * tolerance[None, :]
        moves: tuple[np.ndarray | None, np.ndarray] = (moved_x, margin)
        if records and self.smearing in (0, 1):
            self.check_emulation(diagnostics, values, shift)
        cut = FlaggedCut(True, "no flagged k-point", np.zeros_like(values))
        if flagged and records and method == 1:
            cut = self.recompute_flagged(diagnostics, values)
        elif flagged:
            cut = FlaggedCut(False, "the flagged k-points cannot be recomputed")
        if records:
            # What the scfout can establish of the candidate's records: each check is a contradiction when it fails.
            self.guard_recompute("rounding_recomputed", self.verify_rounding)
            if method == 1:
                self.guard_recompute(
                    "degeneracies_recomputed", lambda: self.verify_degeneracies(diagnostics, tolerance)
                )
            if self.smearing == 1 and not self.pair_mode:
                self.guard_recompute("window_recomputed", self.verify_window)
            if binned:
                self.guard_recompute("moves_recomputed", self.verify_moves)
            if self.legacy:
                self.guard_recompute("spin_down_recomputed", lambda: self.verify_down(diagnostics))
        contradiction = sorted(
            {
                item["name"]
                for item in self.quantities
                if item["name"] in SELF_CHECKS and item["class"] == UNEXPLAINED
            }
        )
        conditions = self.clean_conditions(diagnostics, reference)
        self.clean = clean = all(conditions.values())
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
        floor = self.floor.agreement or 0.0
        floor_verified = self.floor.verified and floor > 0
        allowed = tolerance + hartree + widest + (floor if floor_verified else 0.0)
        # PyATB's truncation above 1e-6 of a peak distorts its table beyond what its integral's sign can tell.
        readable = self.smearing != 1 or self.truncation_tail() <= CLEAN_TRUNCATION_TAIL
        # PyATB's method-1 cut can change any value at the flagged k-points, signs included: unrecomputed, it leaves
        # every difference from PyATB undecided (never documented on trust, never failed on a guess); so does its
        # own treatment of an exact degeneracy where that depends on its eigensolver.
        undecided = (
            ((bool(flagged) and method == 1 and not cut.verified) or bool(self.exact_differs))
            and not unbounded
            and not contradiction
        )
        # The floor sets which components are signs of anything only once the scfout has verified it.
        gate_floor = floor if floor_verified else None
        symmetric, facts = symmetry_gate(reference, values, domega, allowed, gate_floor)
        facts["against"] = compared
        if not symmetric and route is not None and "target" in route:
            view = route["target"] * (pyatb_shift_constant_ratio() if binned else 1.0)
            passes, other_facts = symmetry_gate(reference, view, domega, allowed, gate_floor)
            if passes:
                symmetric, facts = True, {**other_facts, "against": f"{compared} moved to PyATB's scheme"}
        # Every component within the rounding floor in both runs: no sign or component to compare. Resting on an
        # unrecomputed floor, that only suspends the gate (and every documented difference with it).
        everywhere_below = floor > 0 and bool(
            np.all(np.max(np.abs(reference), axis=0, initial=0.0) <= floor)
            and np.all(np.max(np.abs(values), axis=0, initial=0.0) <= floor)
        )
        noise = bool(facts.get("noise")) if floor_verified else everywhere_below
        noise_unverified = noise and not floor_verified
        judged = readable and not noise and (symmetric or not undecided)
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
                    ": not judged, every component lies within the rounding floor "
                    f"{floor:.4g} microA/V^2 in both runs (a response that vanishes, no sign to compare)"
                    + ("" if floor_verified else f"; the floor is not recomputed ({self.floor.reason})")
                    if noise
                    else (
                        f": not judged, PyATB's method-1 cut at the {flagged} flagged k-points is not recomputed"
                        if not judged
                        else ""
                    )
                )
            ),
        )
        if not judged:
            symmetric = not noise_unverified
        self.release = release = self.release_clean_control()
        floor_pending = bool(self.pending) and not self.floor.verified
        converted = self.legacy or self.sign < 0 or binned
        converged, convergence = None, ""
        self.direct_everywhere = True
        unresolved: list[str] = []
        unverified: list[str] = []
        rounding_pending: list[str] = []
        # A component PyATB's table contradicts outright: what cannot be judged elsewhere does not hide it.
        definite: list[str] = []
        reach_table = (
            energy_scale_reach(reference, omega=omega, eta=self.eta, domega=domega)
            if self.smearing == 1 and len(omega) >= 3
            else np.zeros_like(reference)
        )
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
            moved_rows = None
            if moved_x is not None:
                moved_rows = resolved[:, index] & (
                    np.abs(moved_x[:, index] - values[:, index]) <= limit + margin[:, index]
                )
                if np.all((gap <= limit) | moved_rows):
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
            # Within the rounding floor (a response that vanishes, or nearly): an agreement only on a floor the
            # scfout has verified.
            within = np.all(
                (gap <= limit + floor)
                | (
                    np.zeros(len(gap), bool)
                    if moved_x is None
                    else resolved[:, index]
                    & (np.abs(moved_x[:, index] - values[:, index]) <= limit + floor + margin[:, index])
                )
            )
            if floor > 0 and within and floor_verified:
                self.quantity(
                    label,
                    AFTER_CONVENTION if converted else AGREE,
                    direct,
                    limit + floor,
                    f"PyATB's table meets QUEER's {compared} within the rounding floor {floor:.4g} microA/V^2 "
                    f"(its k-points' rounding scales, recomputed from the scfout)"
                    + (" in PyATB's conventions" if converted else ""),
                )
                continue
            self.direct_everywhere = False
            if contradiction:
                self.quantity(
                    label,
                    UNEXPLAINED,
                    direct,
                    limit,
                    f"the candidate contradicts itself ({', '.join(contradiction)}): no cause can explain a difference",
                )
                continue
            if floor_pending:
                rounding_pending.append(name)
                self.quantity(
                    label,
                    UNEXPLAINED,
                    direct,
                    limit,
                    f"not judged: the candidate's records ({', '.join(sorted(self.pending))}) meet its value only "
                    f"within its rounding floor, which is not recomputed ({self.floor.reason})",
                )
                continue
            if floor > 0 and within:
                rounding_pending.append(name)
                self.quantity(
                    label,
                    UNEXPLAINED,
                    direct,
                    limit + floor,
                    f"not judged: QUEER's value meets PyATB's within the rounding floor {floor:.4g} microA/V^2 only, "
                    f"which rests on the candidate's rounding scales until recomputed ({self.floor.reason})",
                )
                continue
            if clean:
                failing = ~(gap <= limit)
                if moved_rows is not None:
                    failing &= ~moved_rows
                definite_rows = failing & (
                    (resolved[:, index] if moved_x is not None else np.ones(len(gap), bool))
                    | (gap > limit + reach_table[:, index])
                )
                if definite_rows.any() or moved_x is None:
                    self.quantity(
                        label,
                        UNEXPLAINED,
                        direct,
                        limit,
                        "a clean control must agree with QUEER's physical value directly: to 1e-4 max|sigma_P|, "
                        "PyATB's table moved to QUEER's energy scale",
                    )
                    continue
                unresolved.append(name)
                self.quantity(
                    label,
                    UNEXPLAINED,
                    direct,
                    limit,
                    "a clean control whose difference from PyATB lies on rows where PyATB's table cannot be moved to "
                    "QUEER's energy scale within half the tolerance, and within the reach of PyATB's Hartree scale",
                )
                continue
            if undecided:
                unverified.append(name)
                self.quantity(
                    label,
                    UNEXPLAINED,
                    direct,
                    limit,
                    "not judged: "
                    + (
                        f"PyATB's method-1 cut at the {flagged} flagged k-points is not recomputed ({cut.reason})"
                        if flagged and not cut.verified
                        else self.reasons.get("exact", "")
                    ),
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
            if noise_unverified:
                rounding_pending.append(name)
                self.quantity(
                    label,
                    UNEXPLAINED,
                    direct,
                    limit,
                    "not judged: every component lies within the candidate's rounding floor, which is not recomputed "
                    f"({self.floor.reason})",
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
                            "QUEER's per-transition record, recomputed from the scfout)"
                        )
                else:
                    pieces.append(
                        f"the transitions its integer bins place in other rows ({float(route['moved_max'][index]):.4g} "
                        "at most, recorded, checked from their energies and recomputed from the scfout)"
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
                    f"PyATB's scheme from the candidate's own records would reproduce it, but they are not recomputed: "
                    f"{route['reason'] or cut.reason}",
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
            unresolved, unverified, rounding_pending = [], [], []
        if unresolved:
            why = (
                route["reason"]
                if route is not None and route["reason"]
                else (
                    "PyATB's table cannot be moved to QUEER's energy scale on the rows the difference needs (refine "
                    "domega against eta)"
                    if readable
                    else f"PyATB's Gaussian truncated at {int(5 * self.eta / domega)} bins drops up to "
                    f"{self.truncation_tail():.3g} of a transition's peak (refine domega against eta)"
                )
            )
            self.precondition("pyatb_scheme_resolution", False, f"{why}: {unresolved}")
        if unverified:
            self.precondition(
                "regularisation",
                False,
                (
                    f"PyATB's method-1 cut at the {flagged} flagged k-points is documented only when recomputed: "
                    f"{cut.reason} ({unverified})"
                    if flagged and not cut.verified
                    else f"PyATB's scheme is documented only from records recomputed from the scfout: "
                    f"{(route or {}).get('reason') or self.reasons.get('exact') or cut.reason} ({unverified})"
                ),
            )
        if rounding_pending:
            self.precondition(
                "rounding",
                False,
                (
                    f"the candidate's records ({', '.join(sorted(self.pending))}) meet its value only within its "
                    f"rounding floor {self.floor.floor:.4g} microA/V^2"
                    if floor_pending
                    else f"QUEER's value meets PyATB's only within its rounding floor {floor:.4g} microA/V^2"
                )
                + f", which rests on its own rounding scales until they are recomputed from the scfout: "
                f"{self.floor.reason} ({rounding_pending})",
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
            "rounding_floor": {
                "floor_uA_V2": floor,
                "recorded": self.floor.recorded,
                "verified": self.floor.verified,
                "reason": self.floor.reason,
                "relative": RECORD_ROUNDING,
            },
            "recomputed": {
                "scfout": self.recompute is not None,
                "reason": None if self.recompute is not None else self.recompute_reason,
                "states": dict(self.states),
                "reasons": dict(self.reasons),
                "exact_degeneracy_differs": self.exact_differs,
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
            for key in ("dropped_max", "moved_max", "removed_max"):
                if key in route:
                    route_facts[key] = dict(zip(SHIFT_COMPONENTS, np.asarray(route[key]).tolist()))
            for key in ("edge_transitions", "edge_dropped"):
                if key in route:
                    route_facts[key] = route[key]
        self.basis.update(
            {
                "relative": AGREEMENT_RELATIVE,
                "scale_uA_V2": scale,
                "base_uA_V2": base,
                "print_relative": PRINT_RELATIVE,
                "rounding_floor_uA_V2": floor,
                "energy_scale": energy_scale,
                "truncation_tail": self.truncation_tail() if self.smearing == 1 else None,
                "window_reach_eta": WINDOW_REACH_ETA,
                "scheme": route_facts,
                "emulated": self.smearing in (0, 1) and records,
            }
        )

    def guard_recompute(self, name: str, check: Any) -> None:
        """A record that cannot be recomputed from the scfout the candidate names contradicts it."""
        try:
            check()
        except (OSError, ValueError, ArithmeticError, np.linalg.LinAlgError, KeyError, IndexError) as error:
            self.quantity(
                name,
                UNEXPLAINED,
                detail=f"a record could not be recomputed from the scfout: {type(error).__name__}: {error}",
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
