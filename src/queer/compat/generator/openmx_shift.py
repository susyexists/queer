"""QUEER's OpenMX shift current: the counterpart of PyATB's ``SHIFT_CURRENT`` block.

``QUEER_SHIFT_CURRENT`` reads the staged scfout and the ``queer_params.json`` of
:mod:`~queer.compat.generator.openmx_params` and evaluates QUEER's resonant shift
conductivity ``sigma^abc(0; w, -w)`` (microA/V^2; :func:`~queer.observables.nonlinear.shift_current`)
on PyATB's Gamma-origin grid and photon energies ``start + i * domega``,
``i < int((end - start) / domega) + 1``. The result is written to
``queer_results.json`` (``shift_current``) and to PyATB's
``Out/Shift_Current/shift_current.dat``.

**Settings,** as :func:`~queer.compat.generator.translate_pyatb` maps them:
``occ_band`` occupies the lowest levels (and starts the scissors), ``smearing_method
1`` is a Gaussian of standard deviation ``eta / sqrt(2)`` (PyATB's ``exp(-(x /
eta)^2) / (eta sqrt(pi))``), ``0`` QUEER's histogram of bin width ``domega`` and ``2``
QUEER's adaptive Gaussian, ``n_occ``/``m_unocc`` one band pair (as PyATB reads
them), ``scissors`` a rigid upward shift of the transition energies. QUEER's
geometry is its analytic derivative whatever PyATB's ``method`` asks.

**Spin.** PyATB reads only the spin-up Hamiltonian of a scfout without spin-orbit
coupling (``nspin 1``) and multiplies its shift current by 2; with spin-orbit
coupling (``nspin 4``) it multiplies by 1. QUEER's physical value is the spinor
model's, twice the single channel of a non-spin-polarized SCF (the same as
PyATB's) and, for a collinear SCF, the sum of its two channels, each with
``occ_band`` occupied levels; ``legacy_single_channel`` then holds PyATB's value,
twice the spin-up channel.

**Sign and file.** PyATB's prefactor carries the signed cell volume, so it prints
``lattice_handedness`` times the conductivity. ``Out/Shift_Current/shift_current.dat``
holds ``convention.export_sign`` times QUEER's physical value; without smearing it
holds QUEER's histogram in PyATB's normalization instead (``binned_uA_V2``: QUEER's
histogram evaluated half a bin higher, times ``domega``, which is the weight of the
transitions in ``(omega_i, omega_i + domega]``).

**Diagnostics** (comparison aids, never the result), evaluated on the model PyATB
reads: the bands of the pairs PyATB selects at each k-point and the smallest
spacing between such a band and any other level that QUEER does not treat as
degenerate with it (exact degeneracies, Kramers pairs among them, are skipped by
both codes); ``regularisation``, PyATB's ``method 1`` cut at the k-points where such
a spacing is at most 1e-4 eV (its sum-over-states derivative there drops the
pair's ``1 / dE``), measured as QUEER's spectrum minus QUEER's spectrum with that
cut (QUEER's degeneracy tolerance raised to 1e-4 eV at those k-points only), with
QUEER's spectrum without those k-points and ``flagged_records``: every flagged
k-point (fractional coordinates) with its transitions (energy and 18 weights),
QUEER's physical ones and those with the cut, from which a comparison given the
scfout recomputes those k-points independently; the transitions inside PyATB's
window, PyATB's Gaussian truncation, and ``pyatb_scheme``: what PyATB prints from
the same transitions (its Hartree scale, its window rule, its integer bins and
``5 eta / domega`` truncation, its constants, spin factor and signed volume, and
its cut at the flagged k-points), the same transitions under QUEER's Gaussian
(``transitions_uA_V2``: the value less the measured cut, summed by the emulation's
own code), QUEER's value of the transitions PyATB's window drops at the other
k-points (``window_dropped_uA_V2``,
with a Gaussian and no band pair) with each such transition whose Gaussian reaches
a photon energy within :data:`WINDOW_REACH_ETA` eta and each one the window keeps
within :data:`EDGE_EV` of its ends (``window_transitions``) and, without smearing,
every transition there that PyATB's integer bins place in another row than QUEER's
histogram (``moved_transitions``), so that QUEER's value moves to PyATB's scheme
exactly. ``rounding`` holds each k-point's rounding scale (:func:`rounding_scale`:
its conditioning times its transitions' product moduli, the size of what cancels),
``exact_degeneracies`` the k-points where an exact degeneracy takes part with its
spacings, and for a collinear scfout ``spin_channels`` the two channels the physical
value sums. Every per-k-point record names its k-point by its index in the grid's C
order (``kpoint_order``), so that a comparison given the scfout recomputes it.
The per-k-point records are bounded: at most :data:`FLAGGED_RECORD_LIMIT` flagged
k-points, :data:`MOVED_RECORD_LIMIT` moved transitions, :data:`WINDOW_RECORD_LIMIT`
window transitions, :data:`EXACT_RECORD_LIMIT` k-points with an exact degeneracy and
:data:`ROUNDING_RECORD_LIMIT` rounding scales, else that record is not kept
(``recorded: false``). ``self_convergence: true`` adds the physical
value (and the legacy channel) on the doubled grid. ``build.queer_source_sha256``
hashes every ``.py`` file of the installed ``queer`` package
(:func:`~queer.compat.generator.results.package_source_sha256`).
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...models.evaluation import spin_degeneracy, volume
from ...observables._sampling import kmesh
from ...observables.nonlinear import (
    ShiftCurrentSettings,
    shift_current_point,
    shift_current_settings,
    shift_current_spectrum,
)
from ...physics.constants import E_CHARGE, HBAR
from .memory import chunks, pmap
from .openmx_params import STEP_SHIFT
from .pyatb_conventions import (
    HARTREE_EV,
    PYATB_DEGENERATE_EV,
    PYATB_HARTREE_EV,
    pyatb_shift_constant_ratio,
    pyatb_shift_spin_factor,
)
from .pyatb_files import SHIFT_COMPONENTS, SHIFT_CURRENT_FILE, shift_current_text, write_text
from .results import ERR_BAD_PARAMS, ERR_VALIDATION, RunnerError, package_source_sha256

# Complex N x N matrices one shift-current k-point keeps alive, and per photon energy
# of a batch of at most FREQUENCY_BATCH (QUEER's own estimate in shift_current).
SHIFT_MATRICES = 200
SHIFT_FREQUENCY_MATRICES = 2
FREQUENCY_BATCH = 64
# A transition this close (eV, PyATB's scale) to a bin or window edge may fall on
# either side in the two codes: its weight is allowed for.
EDGE_EV = 1e-9
# The clean-control conditions of the parity rule.
CLEAN_PAIR_GAP_EV = 1e-3
CLEAN_MARGIN_ETA = 5.0
CLEAN_TRUNCATION_TAIL = 1e-6
# Flagged k-points listed in the results (largest shift first).
FLAGGED_LISTED = 10
# The per-k-point records a comparison recomputes: at most this many flagged k-points and numbers (energies
# and weights) in all, this many transitions PyATB's bins move, this many transitions PyATB's window drops within
# reach of it (or keeps at its edges), this many k-points with an exact degeneracy and this many k-points' rounding
# scales; beyond any of them that record is not kept.
FLAGGED_RECORD_LIMIT = 256
FLAGGED_RECORD_NUMBERS = 2_000_000
MOVED_RECORD_LIMIT = 100_000
WINDOW_RECORD_LIMIT = 100_000
EXACT_RECORD_LIMIT = 4096
ROUNDING_RECORD_LIMIT = 1_000_000
# A Gaussian of PyATB's eta reaches this many eta: a transition PyATB's window drops farther from every photon
# energy adds at most e^-36 of its peak there.
WINDOW_REACH_ETA = 6.0
# The self-convergence rule: |sigma(N) - sigma(2N)|_inf <= this * max|sigma(2N)|.
SELF_CONVERGENCE_RELATIVE = 0.1
SQRT_PI = math.sqrt(math.pi)
_AXES = {"x": 0, "y": 1, "z": 2}
COMPONENT_INDEX = tuple(tuple(_AXES[axis] for axis in name) for name in SHIFT_COMPONENTS)
_A, _B, _C = (np.array([index[position] for index in COMPONENT_INDEX]) for position in range(3))

WARN_SHIFT_COLLINEAR = "shift_collinear_channels"
WARN_SHIFT_WINDOW = "shift_transitions_outside_window"
WARN_SHIFT_THRESHOLD = "shift_pair_gap_at_pyatb_threshold"
WARN_SHIFT_UNCONVERGED = "shift_current_unconverged"
EMULATION_ROLE = (
    "comparison diagnostic: PyATB's numerical scheme applied to QUEER's transitions, never QUEER's result"
)


@dataclass(frozen=True)
class PyatbShiftScheme:
    """PyATB's broadening of the transitions: its photon energies, width and rules (its eV)."""

    start: float
    domega: float
    count: int
    eta: float
    smearing: int
    pair_mode: bool
    energy_scale: float = PYATB_HARTREE_EV / HARTREE_EV

    @property
    def interval(self) -> int:
        """Bins on either side of a transition's bin that PyATB's Gaussian reaches."""
        return int(5 * self.eta / self.domega)

    @property
    def end(self) -> float:
        """The upper end of PyATB's window, ``start + count * domega`` (excluded)."""
        return self.start + self.count * self.domega

    def inside(self, energies: np.ndarray) -> np.ndarray:
        """Which transitions (QUEER's eV) PyATB keeps: those in ``[start, end)`` on its scale, every one of a pair."""
        scaled = np.asarray(energies, float) * self.energy_scale
        if self.pair_mode:
            return np.ones(len(scaled), bool)
        return (scaled >= self.start) & (scaled < self.end)

    def gaussian(self, energies: np.ndarray) -> np.ndarray:
        """``exp(-((w_i - E) / eta)^2) / (eta sqrt(pi))`` at every photon energy, ``(count, len(energies))``."""
        grid = self.start + np.arange(self.count) * self.domega
        offset = grid[:, None] - np.asarray(energies, float)[None, :]
        return np.exp(-((offset / self.eta) ** 2)) / (self.eta * SQRT_PI)


def component_table(tensor: np.ndarray) -> np.ndarray:
    """The 18 independent components ``(len(omega), 18)`` of ``(len(omega), 3, 3, 3)`` tensors."""
    return np.asarray(tensor)[:, _A, _B, _C]


def components(table: np.ndarray) -> dict[str, list[float]]:
    """A ``(len(omega), 18)`` table as ``{component: column}`` in PyATB's order."""
    values = np.asarray(table, float)
    return {name: values[:, index].tolist() for index, name in enumerate(SHIFT_COMPONENTS)}


def pyatb_scheme_spectrum(
    scheme: PyatbShiftScheme, energies: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray | None]:
    """PyATB's broadening of transitions (energies in QUEER's eV, weights ``(T, 18)``).

    Returns PyATB's spectrum ``(count, 18)`` and, for its Gaussian, the same
    transitions under the untruncated Gaussian at every photon energy with no
    window rule. Energies move to PyATB's Hartree scale; without a band pair only
    the transitions in ``[start, end)`` count; a transition's bin is ``int((E -
    start) / domega)`` (C's truncation); without smearing it adds its weight to that
    bin, with PyATB's Gaussian ``exp(-((E - w_i) / eta)^2) / eta / sqrt(pi)`` to the
    bins within ``int(5 eta / domega)`` of it.
    """
    spectrum = np.zeros((scheme.count, len(SHIFT_COMPONENTS)))
    if not len(energies):
        return spectrum, (np.zeros_like(spectrum) if scheme.smearing == 1 else None)
    energy = np.asarray(energies, float) * scheme.energy_scale
    weight = np.asarray(weights, float)
    inside = np.ones(len(energy), bool)
    if not scheme.pair_mode:
        inside = (energy >= scheme.start) & (energy < scheme.end)
    index = np.trunc((energy - scheme.start) / scheme.domega).astype(int)
    if scheme.smearing == 0:
        keep = inside & (index >= 0) & (index < scheme.count)
        np.add.at(spectrum, index[keep], weight[keep])
        return spectrum, None
    offsets = np.arange(-scheme.interval, scheme.interval + 1)
    rows = index[:, None] + offsets[None, :]
    keep = inside[:, None] & (rows >= 0) & (rows < scheme.count)
    omega = scheme.start + rows * scheme.domega
    kernel = np.exp(-(((energy[:, None] - omega) / scheme.eta) ** 2)) / scheme.eta / SQRT_PI
    transition, _ = np.nonzero(keep)
    np.add.at(spectrum, rows[keep], weight[transition] * kernel[keep][:, None])
    grid = scheme.start + np.arange(scheme.count) * scheme.domega
    full = np.exp(-(((energy[:, None] - grid[None, :]) / scheme.eta) ** 2)) / scheme.eta / SQRT_PI
    return spectrum, full.T @ weight


def edge_transitions(scheme: PyatbShiftScheme, energies: np.ndarray) -> int:
    """How many transitions (QUEER's eV) lie within :data:`EDGE_EV` of one of PyATB's bin or window edges on its
    scale, where its rounding may place them on either side."""
    if not len(energies):
        return 0
    energy = np.asarray(energies, float) * scheme.energy_scale
    position = (energy - scheme.start) / scheme.domega
    fraction = position - np.floor(position)
    near = np.minimum(fraction, 1 - fraction) * scheme.domega < EDGE_EV
    if not scheme.pair_mode:
        near |= (np.abs(energy - scheme.start) < EDGE_EV) | (np.abs(energy - scheme.end) < EDGE_EV)
    return int(np.count_nonzero(near))


def pair_gaps(
    point: Mapping[str, Any], scheme: PyatbShiftScheme, occupied: int, band_pairs: Any
) -> tuple[float, list[list[float]]]:
    """The smallest spacing (eV) PyATB's ``method 1`` may regularize at one k-point, and the exact degeneracies
    that take part (``[i, j, spacing]``, ``i < j``, unscissored levels).

    PyATB's sum-over-states derivative (``get_D_degenerate``) divides by ``E_l -
    E_j`` for every level ``l`` and every band ``j`` of the pairs it selects (the
    window's occupied-to-empty pairs, or the named pair), and replaces ``1 / dE`` by
    ``dE / (dE^2 + 0.04^2)`` when ``|dE| <= 1e-4`` eV. A spacing within QUEER's
    degeneracy tolerance is an exact degeneracy: PyATB's term is then zero and QUEER
    sums the degenerate block covariantly, and the two agree (a Kramers pair at a
    time-reversal-invariant k-point). The spacing returned is the smallest of the
    others (scissored levels, QUEER's eV), ``inf`` when there is none.
    """
    energy = np.asarray(point["energies"], float)
    raw = np.asarray(point["raw"]["energies"], float)
    tolerance = float(point["raw"]["degeneracy_tol"])
    selected = np.zeros(len(energy), bool)
    if band_pairs:
        for pair in band_pairs:
            selected[list(pair)] = True
    elif 0 < occupied < len(energy):
        transitions = (energy[None, occupied:] - energy[:occupied, None]) * scheme.energy_scale
        inside = (transitions >= scheme.start) & (transitions < scheme.end)
        selected[:occupied] = inside.any(axis=1)
        selected[occupied:] = inside.any(axis=0)
    if not selected.any():
        return math.inf, []
    other = selected[None, :] & ~np.eye(len(energy), dtype=bool)
    exact = other & (np.abs(raw[:, None] - raw[None, :]) <= tolerance)
    spacing = np.abs(energy[:, None] - energy[None, :])[other & ~exact]
    pairs = sorted({(int(min(i, j)), int(max(i, j))) for i, j in zip(*np.nonzero(exact))})
    found = [[i, j, float(abs(raw[j] - raw[i]))] for i, j in pairs]
    return (float(spacing.min()) if spacing.size else math.inf), found


def bin_rows(
    scheme: PyatbShiftScheme, binned_omega: np.ndarray, bin_width: float, energies: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The row of each transition in QUEER's histogram half a bin higher and in PyATB's integer bins (-1: none),
    and whether a rounding at an edge could move it.

    QUEER's row is the one whose ``binned_omega`` lies within half a ``bin_width`` of
    the transition (the histogram of :func:`~queer.observables.nonlinear.shift_current_spectrum`,
    ``(w_i, w_i + domega]``); PyATB's is ``int((E - start) / domega)`` on its Hartree
    scale with its window rule (:func:`pyatb_scheme_spectrum`, ``[w_i, w_i + domega)``).
    A transition within :data:`EDGE_EV` of an edge of either is ambiguous.
    """
    count = scheme.count
    energy = np.asarray(energies, float)
    centre = np.floor((energy - scheme.start) / scheme.domega).astype(int)
    hits = []
    for offset in (-1, 0, 1):
        row = centre + offset
        valid = (row >= 0) & (row < count)
        difference = np.asarray(binned_omega, float)[np.clip(row, 0, count - 1)] - energy
        hits.append(np.where(valid & (difference >= -bin_width / 2) & (difference < bin_width / 2), row, -1))
    rows = np.stack(hits)
    own = np.max(rows, axis=0)
    scaled = energy * scheme.energy_scale
    pyatb = np.trunc((scaled - scheme.start) / scheme.domega).astype(int)
    pyatb = np.where(scheme.inside(energy) & (pyatb >= 0) & (pyatb < count), pyatb, -1)
    position = (energy - scheme.start) / scheme.domega
    near_own = np.abs(position - np.round(position)) * scheme.domega < EDGE_EV
    scaled_position = (scaled - scheme.start) / scheme.domega
    near_pyatb = np.abs(scaled_position - np.round(scaled_position)) * scheme.domega < EDGE_EV
    ambiguous = (np.count_nonzero(rows >= 0, axis=0) > 1) | near_own | near_pyatb
    return own, pyatb, ambiguous


@dataclass(frozen=True)
class ShiftPlan:
    """What one pass evaluates at each k-point.

    ``binned`` is QUEER's histogram half a bin higher (PyATB's bins without
    smearing); with a ``scheme`` the pass also keeps the diagnostics of the model
    PyATB reads, and with ``cut`` (``method 1``) QUEER's settings with PyATB's 1e-4
    eV pair cut, used at the k-points :func:`pair_gaps` flags; with ``rounding`` the
    pass records each k-point's rounding scale (:func:`rounding_scale`).
    """

    settings: ShiftCurrentSettings
    frequency_batch: int
    binned: ShiftCurrentSettings | None = None
    scheme: PyatbShiftScheme | None = None
    occupied: int = 0
    cut: ShiftCurrentSettings | None = None
    rounding: bool = False


_SUMMED = (
    "physical",
    "binned",
    "pyatb",
    "untruncated",
    "transitions_table",
    "dropped",
    "shift",
    "binned_shift",
    "flagged_physical",
    "flagged_binned",
)
_COUNTED = (
    "points",
    "edge_count",
    "flagged",
    "cut_failed",
    "exact",
    "transitions",
    "inside",
    "below",
    "above",
    "record_numbers",
)


def _transitions(point: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """The energies (QUEER's eV) and the ``(T, 18)`` weights ``(f_n - f_m) Im[r^b r^c_;a]_sym`` of a k-point's
    transitions (unconverted)."""
    source, target = np.nonzero(point["factor"])
    energies = point["gap"][source, target]
    weights = (point["factor"][source, target] * point["integrand"][..., source, target])[_A, _B, _C].T
    return np.asarray(energies, float), np.asarray(weights, float).reshape(
        len(energies), len(SHIFT_COMPONENTS)
    )


def rounding_scale(point: Mapping[str, Any]) -> float:
    """A k-point's rounding scale (unconverted): ``(1 + max|E| / s) max_c sum_t |f_t| |r^b r^c_;a|_sym``, ``s`` the
    smallest spacing of its levels above the degeneracy tolerance. Its transitions' weights round to a small
    multiple of the machine epsilon times this, in this code or another, however much they cancel across the
    k-points (a response that vanishes by symmetry); the moduli replace each product (``magnitude`` of
    :func:`~queer.observables.nonlinear.shift_current_point`) and the conditioning ``1 + max|E| / s`` the
    eigenvectors' rounding."""
    source, target = np.nonzero(point["factor"])
    if not len(source):
        return 0.0
    magnitudes = np.abs(point["factor"][source, target]) * point["magnitude"][..., source, target]
    largest = float(np.max(magnitudes[_A, _B, _C].sum(axis=-1)))
    levels = np.asarray(point["raw"]["energies"], float)
    spacing = np.abs(levels[:, None] - levels[None, :])
    distinct = spacing[spacing > float(point["raw"]["degeneracy_tol"])]
    condition = 1.0 + float(np.max(np.abs(levels))) / float(distinct.min()) if distinct.size else 1.0
    return condition * largest


def shift_rows(model: Any, plan: ShiftPlan, chunk: Any) -> dict[str, Any]:
    """The k-point sums of a chunk: QUEER's spectrum and, with a scheme, its transitions' diagnostics.

    ``chunk`` is ``(index of its first k-point on the grid, k-points)`` (or the k-points
    alone, from index 0). ``physical`` sums :func:`~queer.observables.nonlinear.shift_current_spectrum`
    (unconverted, as :func:`~queer.observables.nonlinear.shift_current` sums it) and
    ``binned`` the same estimator half a bin higher; with ``rounding`` ``scales`` lists each
    k-point's :func:`rounding_scale`. With a ``scheme``: at a k-point :func:`pair_gaps`
    flags, ``shift`` (and ``binned_shift``) sums QUEER's spectrum minus its spectrum under
    PyATB's cut, the cut's transitions stand in for QUEER's in what follows,
    ``flagged_physical`` (and ``flagged_binned``) sums QUEER's spectrum there and
    ``records`` keeps the k-point with both sets of transitions; ``exact_records`` keeps
    each k-point's exact degeneracies; ``pyatb`` and ``untruncated`` sum
    :func:`pyatb_scheme_spectrum`, ``transitions_table`` QUEER's Gaussian of the same
    transitions (its eV); at the other k-points ``dropped`` sums QUEER's Gaussian of the
    transitions PyATB's window drops (no band pair), ``window`` keeps those within
    :data:`WINDOW_REACH_ETA` eta of a photon energy and those it keeps within
    :data:`EDGE_EV` of its ends, and ``moved`` keeps the transitions PyATB's bins place
    in another row than QUEER's histogram (:func:`bin_rows`); with the level and
    transition facts.
    """
    offset, points = chunk if isinstance(chunk, tuple) else (0, chunk)
    settings, scheme = plan.settings, plan.scheme
    count = len(settings.omega)
    width = len(SHIFT_COMPONENTS)
    batch = plan.frequency_batch
    result: dict[str, Any] = {"physical": np.zeros((count, 3, 3, 3)), "points": len(points), "scales": []}
    if plan.binned is not None:
        result["binned"] = np.zeros((count, 3, 3, 3))
    emulated = scheme is not None and scheme.smearing in (0, 1)
    windowed = scheme is not None and scheme.smearing == 1 and not scheme.pair_mode
    if scheme is not None:
        result.update(
            pyatb=np.zeros((count, width)) if emulated else None,
            untruncated=np.zeros((count, width)) if emulated and scheme.smearing == 1 else None,
            transitions_table=np.zeros((count, width)) if scheme.smearing == 1 else None,
            dropped=np.zeros((count, width)) if windowed else None,
            edge_count=0,
            pair_gap=math.inf,
            pair_gap_k=None,
            exact=0,
            exact_records=[],
            flagged=0,
            cut_failed=0,
            flagged_points=[],
            records=[],
            record_numbers=0,
            moved=[],
            moved_overflow=False,
            window=[],
            window_overflow=False,
            shift=np.zeros((count, 3, 3, 3)),
            binned_shift=np.zeros((count, 3, 3, 3)) if plan.binned is not None else None,
            flagged_physical=np.zeros((count, 3, 3, 3)),
            flagged_binned=np.zeros((count, 3, 3, 3)) if plan.binned is not None else None,
            transitions=0,
            inside=0,
            below=0,
            above=0,
            low=math.inf,
            high=-math.inf,
        )
    for position, k in enumerate(points):
        index = offset + position
        point = shift_current_point(model, k, settings)
        spectrum = shift_current_spectrum(point, settings, batch)
        result["physical"] += spectrum
        if plan.rounding:
            result["scales"].append(rounding_scale(point))
        binned = None
        if plan.binned is not None:
            binned = shift_current_spectrum(point, plan.binned, batch)
            result["binned"] += binned
        if scheme is None:
            continue
        used = point
        gap, exact = pair_gaps(point, scheme, plan.occupied, settings.band_pairs)
        if exact:
            result["exact"] += 1
            result["exact_records"].append({"k_index": index, "pairs": exact})
        if gap < result["pair_gap"]:
            result["pair_gap"], result["pair_gap_k"] = gap, np.asarray(k, float).tolist()
        flagged = plan.cut is not None and gap * scheme.energy_scale <= PYATB_DEGENERATE_EV
        if flagged:
            result["flagged"] += 1
            result["flagged_physical"] += spectrum
            if binned is not None:
                result["flagged_binned"] += binned
            record: dict[str, Any] = {
                "k": np.asarray(k, float).tolist(),
                "pair_gap_eV": float(gap),
                "physical": _transitions(point),
                "cut": None,
            }
            try:
                cut = shift_current_point(model, k, plan.cut)
            except ValueError:
                result["cut_failed"] += 1
            else:
                difference = spectrum - shift_current_spectrum(cut, settings, batch)
                result["shift"] += difference
                if binned is not None:
                    result["binned_shift"] += binned - shift_current_spectrum(cut, plan.binned, batch)
                result["flagged_points"].append(
                    (float(np.max(np.abs(difference))), np.asarray(k, float).tolist(), gap)
                )
                record["cut"] = _transitions(cut)
                used = cut
            numbers = sum(
                len(part[0]) * (1 + width) for part in (record["physical"], record["cut"]) if part is not None
            )
            result["record_numbers"] += numbers
            if (
                result["flagged"] <= FLAGGED_RECORD_LIMIT
                and result["record_numbers"] <= FLAGGED_RECORD_NUMBERS
            ):
                result["records"].append(record)
        energies, weights = _transitions(used)
        if emulated:
            scheme_spectrum, untruncated = pyatb_scheme_spectrum(scheme, energies, weights)
            result["pyatb"] += scheme_spectrum
            if untruncated is not None:
                result["untruncated"] += untruncated
            result["edge_count"] += edge_transitions(scheme, energies)
        if result["transitions_table"] is not None and len(energies):
            kernel = scheme.gaussian(energies)
            result["transitions_table"] += kernel @ weights
            if not flagged and result["dropped"] is not None:
                outside = ~scheme.inside(energies)
                if outside.any():
                    result["dropped"] += kernel[:, outside] @ weights[outside]
                kept = window_records(scheme, settings.omega, energies)
                if kept.size and not result["window_overflow"]:
                    for item in kept:
                        result["window"].append(
                            (
                                index,
                                float(energies[item]),
                                weights[item].copy(),
                                bool(scheme.inside(energies[[item]])[0]),
                            )
                        )
                    if len(result["window"]) > WINDOW_RECORD_LIMIT:
                        result["window"], result["window_overflow"] = [], True
        if not flagged and plan.binned is not None and len(energies):
            assert plan.binned.bin_width is not None
            own, pyatb, ambiguous = bin_rows(
                scheme, plan.binned.omega, float(plan.binned.bin_width), energies
            )
            moved = (own != pyatb) | ambiguous
            if moved.any() and not result["moved_overflow"]:
                for item in np.flatnonzero(moved):
                    result["moved"].append(
                        (
                            float(energies[item]),
                            weights[item].copy(),
                            int(own[item]),
                            int(pyatb[item]),
                            bool(ambiguous[item]),
                            index,
                        )
                    )
                if len(result["moved"]) > MOVED_RECORD_LIMIT:
                    result["moved"], result["moved_overflow"] = [], True
        if len(energies):
            scaled = energies * scheme.energy_scale
            result["transitions"] += len(energies)
            result["below"] += int(np.count_nonzero(scaled < scheme.start))
            result["above"] += int(np.count_nonzero(scaled >= scheme.end))
            result["inside"] += int(np.count_nonzero((scaled >= scheme.start) & (scaled < scheme.end)))
            result["low"] = min(result["low"], float(energies.min()))
            result["high"] = max(result["high"], float(energies.max()))
    if scheme is not None:
        result["flagged_points"] = sorted(result["flagged_points"], key=lambda item: -item[0])[
            :FLAGGED_LISTED
        ]
    return result


def window_records(scheme: PyatbShiftScheme, omega: np.ndarray, energies: np.ndarray) -> np.ndarray:
    """The transitions (QUEER's eV) of a k-point a record of PyATB's window holds: those it drops whose Gaussian
    reaches a photon energy within :data:`WINDOW_REACH_ETA` eta, and those it keeps within :data:`EDGE_EV` of its
    ends on its scale, where its rounding may drop them."""
    energy = np.asarray(energies, float)
    if not len(energy):
        return np.zeros(0, int)
    grid = np.asarray(omega, float)
    reach = WINDOW_REACH_ETA * scheme.eta
    near = (energy >= grid[0] - reach) & (energy <= grid[-1] + reach)
    scaled = energy * scheme.energy_scale
    edge = (np.abs(scaled - scheme.start) < EDGE_EV) | (np.abs(scaled - scheme.end) < EDGE_EV)
    inside = scheme.inside(energy)
    return np.flatnonzero((~inside & near) | (inside & edge))


def _reduce(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = dict(parts[0])
    for part in parts[1:]:
        for key, value in part.items():
            if value is None:
                continue
            if key in _SUMMED:
                total[key] = total[key] + value
            elif key in _COUNTED:
                total[key] += value
            elif key == "pair_gap" and value < total["pair_gap"]:
                total["pair_gap"], total["pair_gap_k"] = value, part["pair_gap_k"]
            elif key == "low":
                total["low"] = min(total["low"], value)
            elif key in ("high", "energy_max"):
                total[key] = max(total[key], value)
            elif key == "flagged_points":
                merged = total["flagged_points"] + value
                total["flagged_points"] = sorted(merged, key=lambda item: -item[0])[:FLAGGED_LISTED]
            elif key in ("records", "moved", "window", "scales", "exact_records"):
                total[key] = total[key] + value
            elif key in ("moved_overflow", "window_overflow"):
                total[key] = total[key] or value
    if "moved" in total and len(total["moved"]) > MOVED_RECORD_LIMIT:
        total["moved"], total["moved_overflow"] = [], True
    if "window" in total and len(total["window"]) > WINDOW_RECORD_LIMIT:
        total["window"], total["window_overflow"] = [], True
    return total


def conversion(model: Any) -> float:
    """QUEER's shift-current prefactor of a model (microA/V^2 per unit k-mean of the spectrum)."""
    return -np.pi * E_CHARGE**2 / HBAR * 1e6 * spin_degeneracy(model) / volume(model)


@dataclass
class ShiftEvaluation:
    """One pass of a model over a grid: its physical tensor and, when asked, its diagnostics (converted)."""

    physical: np.ndarray
    binned: np.ndarray | None
    sums: dict[str, Any]
    factor: float


class ShiftStep:
    """The stages of one ``QUEER_SHIFT_CURRENT`` run on an :class:`~.openmx_steps.OpenMXStepContext`."""

    def __init__(self, context: Any):
        self.context = context
        self.native = context.recipe["parameters"]
        self.meta = context.recipe["export_metadata"]
        self.block = context.params.effective()["function"]
        self.model = context.model
        self.view = context.pyatb_model
        self.omega = np.asarray(self.native["omega"], float)
        self.grid = [int(value) for value in self.native["grid"]]
        self.smearing = int(self.block["smearing_method"])
        self.occupied = int(self.block["occ_band"])
        self.pair = (int(self.block["n_occ"]), int(self.block["m_unocc"]))
        self.pair_mode = self.pair[0] >= 1
        self.spin_factor = pyatb_shift_spin_factor(4 if self.model.spin_mode == "soc" else 1)
        self.sign = int(context.handedness)
        self.ratio = pyatb_shift_constant_ratio()
        eta = float(self.block.get("eta", 0.01) if self.smearing else 0.01)
        self.scheme = PyatbShiftScheme(
            start=float(self.block["omega"][0]),
            domega=float(self.block["domega"]),
            count=len(self.omega),
            eta=eta,
            smearing=self.smearing,
            pair_mode=self.pair_mode,
        )

    @functools.cached_property
    def down_model(self) -> Any:
        """The spin-down channel of a collinear scfout, occupied once per level."""
        from .openmx_steps import single_channel_model

        return single_channel_model(self.model, 1)

    # ------------------------------------------------------------ checks
    def check_bands(self) -> None:
        bands = int(self.view.nbnd)
        reads = "spinor bands" if self.model.spin_mode == "soc" else "bands of the spin-up channel"
        if self.occupied >= bands:
            raise RunnerError(
                ERR_BAD_PARAMS,
                f"occ_band {self.occupied} must be below the {bands} {reads} PyATB reads from the scfout",
            )
        if self.pair_mode and self.pair[1] > bands:
            raise RunnerError(
                ERR_BAD_PARAMS,
                f"m_unocc {self.pair[1]} exceeds the {bands} {reads} PyATB reads from the scfout",
            )

    def settings(self, model: Any, grid: Sequence[int]) -> ShiftCurrentSettings:
        native = self.native
        try:
            return shift_current_settings(
                model,
                self.omega,
                self.context.mu,
                eta=float(native["eta"]),
                grid=tuple(grid),
                scissors=float(native["scissors"]),
                smearing=native["smearing"],
                occupied_bands=self.occupied,
                band_pairs=native.get("band_pairs"),
                method="analytic",
                scissors_occupied_bands=self.occupied,
                bin_width=native.get("bin_width"),
            )
        except ValueError as error:
            raise RunnerError(ERR_BAD_PARAMS, f"Invalid PYATB_SHIFT_CURRENT settings: {error}") from error

    # ------------------------------------------------------------ passes
    def evaluate(
        self,
        stage: str,
        model: Any,
        grid: Sequence[int],
        *,
        diagnostics: bool,
        binned: bool,
        rounding: bool = False,
    ) -> ShiftEvaluation:
        """One pass of ``model`` over ``grid`` on the planned processes (``rounding``: each k-point's rounding
        scale too)."""
        points = kmesh(grid)
        settings = self.settings(model, grid)
        frequency_batch = min(len(self.omega), FREQUENCY_BATCH)
        per_k = SHIFT_MATRICES + SHIFT_FREQUENCY_MATRICES * frequency_batch
        options = self.context.options(stage, model, len(points), per_k=per_k)
        domega = self.scheme.domega
        plan = ShiftPlan(
            settings,
            frequency_batch,
            binned=dataclasses.replace(settings, omega=self.omega + 0.5 * domega, bin_width=domega)
            if binned
            else None,
            scheme=self.scheme if diagnostics else None,
            occupied=self.occupied,
            cut=dataclasses.replace(settings, degeneracy_tol=PYATB_DEGENERATE_EV / self.scheme.energy_scale)
            if diagnostics and int(self.block["method"]) == 1
            else None,
            rounding=rounding or diagnostics,
        )
        step = max(1, int(options.batch_size))
        indexed = [(start, part) for start, part in zip(range(0, len(points), step), chunks(points, step))]
        try:
            parts = pmap(functools.partial(shift_rows, model, plan), indexed, options)
        except ValueError as error:
            raise RunnerError(
                ERR_VALIDATION, f"QUEER cannot evaluate this shift current on the {stage} grid: {error}"
            ) from error
        sums = _reduce(parts)
        factor = conversion(model) / len(points)
        binned_table = domega * factor * component_table(sums["binned"]) if binned else None
        return ShiftEvaluation(factor * sums["physical"], binned_table, sums, factor)

    def run(self) -> None:
        context, record = self.context, self.context.record
        self.check_bands()
        binned = self.smearing == 0
        collinear = self.model.spin_mode == "collinear"
        main = self.evaluate("shift current", self.view, self.grid, diagnostics=True, binned=binned)
        down = None
        if collinear:
            down = self.evaluate(
                "shift current (spin-down channel)",
                self.down_model,
                self.grid,
                diagnostics=False,
                binned=binned,
                rounding=True,
            )
        physical, binned_physical = self.combine(main, down)
        legacy = None
        if collinear:
            legacy = self.legacy(main)
        doubled = None
        if context.params.self_convergence:
            doubled = self.self_convergence(component_table(physical), legacy)
        exported = binned_physical if binned else component_table(physical)
        assert exported is not None
        record.results["shift_current"] = {
            "omega_eV": self.omega.tolist(),
            "omega_rule": "PyATB's photon energies start + i * domega, i < int((end - start) / domega) + 1 (eV)",
            "components": list(SHIFT_COMPONENTS),
            "components_computed": list(SHIFT_COMPONENTS),
            "sigma_uA_V2": components(component_table(physical)),
            "units": "microA/V^2",
            "binned_uA_V2": components(binned_physical) if binned_physical is not None else None,
            "smearing": self.smearing_block(),
            "method": {
                "pyatb_method": int(self.block["method"]),
                "queer_method": "analytic",
                "note": self.meta.get("method_translation"),
            },
            "occupation": self.occupation_block(),
            "grid": self.grid,
            "kpoints": int(np.prod(self.grid)),
            "convention": self.convention_block(),
            "legacy_single_channel": legacy,
            "diagnostics": self.diagnostics(main, doubled, down),
            "build": {
                "queer_source_sha256": package_source_sha256(),
                "rule": "SHA-256 over every .py file of the installed queer package (relative path and bytes, "
                "sorted): with provenance.queer_version, queer_record_sha256 and build_sha it names the build a "
                "clean-control comparison vouches for",
            },
        }
        record.results["integration"] = {
            "mode": "Grid",
            "grid": self.grid,
            "nodes": "Gamma-origin k = n_i / N_i, upper end excluded (PyATB's mp grid)",
            "points": int(np.prod(self.grid)),
            "self_convergence": None
            if doubled is None
            else {key: doubled[key] for key in ("grid", "grid_doubled")},
            "max_kpoint_num": context.params.input_parameters.get("max_kpoint_num"),
            "sparse_format": context.params.input_parameters.get("sparse_format"),
            "execution_only": ["workers", "max_kpoint_num", "sparse_format"],
        }
        self.warn(main, legacy, doubled)
        write_text(
            record.workspace / SHIFT_CURRENT_FILE, shift_current_text(self.omega, self.sign * exported)
        )
        record.add_output(SHIFT_CURRENT_FILE, "Shift current (PyATB layout)", "data")
        peak = float(np.max(np.abs(physical))) if physical.size else 0.0
        context.log(
            f"shift current: {len(self.omega)} photon energies, max |sigma| {peak:.6g} microA/V^2 (physical; "
            f"file sign {self.sign:+d})"
        )

    # ------------------------------------------------------------ results
    def combine(
        self, main: ShiftEvaluation, down: ShiftEvaluation | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """QUEER's physical tensor and its PyATB-normalized histogram from the passes."""
        mode = self.model.spin_mode
        if mode == "soc":
            return main.physical, main.binned
        if mode == "nonmagnetic":
            degeneracy = spin_degeneracy(self.model)
            return degeneracy * main.physical, None if main.binned is None else degeneracy * main.binned
        assert down is not None
        binned = None
        if main.binned is not None and down.binned is not None:
            binned = main.binned + down.binned
        return main.physical + down.physical, binned

    def legacy(self, main: ShiftEvaluation) -> dict[str, Any]:
        reason = "PyATB reads the spin-up Hamiltonian of a collinear scfout only and doubles it (nspin 1)"
        self.context.record.warn(
            WARN_SHIFT_COLLINEAR,
            f"{reason}: legacy_single_channel holds that value beside the physical sum of both channels, each "
            f"with occ_band {self.occupied} occupied levels",
        )
        factor = self.spin_factor
        return {
            "channel": 0,
            "spin_factor": factor,
            "reason": reason,
            "sigma_uA_V2": components(factor * component_table(main.physical)),
            "binned_uA_V2": components(factor * main.binned) if main.binned is not None else None,
        }

    def smearing_block(self) -> dict[str, Any]:
        names = {0: "none", 1: "gaussian", 2: "adaptive"}
        eta = self.block.get("eta")
        return {
            "smearing_method": self.smearing,
            "name": names[self.smearing],
            "eta_pyatb_eV": eta,
            "gaussian_sigma_eV": float(self.native["eta"]) if self.smearing == 1 else None,
            "adaptive": {"scale": 1.0, "min_eV": 1e-4, "max_eV": 1.0} if self.smearing == 2 else None,
            "bin_width_eV": float(self.native["bin_width"]) if self.smearing == 0 else None,
            "domega_eV": float(self.block["domega"]),
            "rule": {
                0: "QUEER's histogram of bin width domega centred on each photon energy, per eV",
                1: "Gaussian of standard deviation eta/sqrt(2): exp(-(x/eta)^2)/(eta sqrt(pi)), untruncated",
                2: "QUEER's adaptive Gaussian (RMS transition change across a reciprocal cell / sqrt(12), "
                "clipped to [1e-4, 1] eV); PyATB's adaptive width is defined differently",
            }[self.smearing],
        }

    def occupation_block(self) -> dict[str, Any]:
        pair = self.native.get("band_pairs")
        return {
            "occ_band": self.occupied,
            "n_occ": self.pair[0],
            "m_unocc": self.pair[1],
            "band_pair": pair[0] if pair else None,
            "band_pair_rule": "one pair (zero-based) when n_occ >= 1, m_unocc >= 1 and m_unocc > n_occ, as PyATB "
            "reads them; else every occupied-empty pair",
            "scissors_eV": float(self.native["scissors"]),
            "scissors_from_band": self.occupied,
            "rule": "the lowest occ_band levels of each k-point are occupied (per spin channel without "
            "spin-orbit coupling); the scissors raises the levels from occ_band on",
        }

    def convention_block(self) -> dict[str, Any]:
        smearing_0 = self.smearing == 0
        return {
            "lattice_handedness": self.sign,
            "export_sign": self.sign,
            "pyatb_spin_factor": self.spin_factor,
            "pyatb_nspin": 4 if self.model.spin_mode == "soc" else 1,
            "pyatb_constant_ratio": self.ratio,
            "physical": "sigma^abc(0; w, -w) = -(pi e^2/hbar) (1/V) sum_k sum_nm f_nm Im[r^b_mn r^c_nm;a]_sym "
            "delta(w - w_mn), carrier charge -e",
            "file": (
                f"{SHIFT_CURRENT_FILE} holds export_sign times binned_uA_V2 (PyATB's normalization without smearing)"
                if smearing_0
                else f"{SHIFT_CURRENT_FILE} holds export_sign times sigma_uA_V2"
            ),
        }

    def scheme_tables(self, main: ShiftEvaluation) -> tuple[np.ndarray | None, np.ndarray | None]:
        """PyATB's printed table from QUEER's transitions and its untruncated counterpart (or None)."""
        if self.smearing == 2:
            return None, None
        scale = self.sign * self.spin_factor * self.ratio * main.factor
        pyatb = scale * main.sums["pyatb"]
        untruncated = main.sums.get("untruncated")
        return pyatb, None if untruncated is None else scale * untruncated

    @property
    def record_scale(self) -> float:
        """QUEER's conversion per k-point in PyATB's conventions (export sign, spin factor; QUEER's constants)."""
        return float(self.sign * self.spin_factor)

    def transition_record(
        self, part: tuple[np.ndarray, np.ndarray] | None, factor: float
    ) -> dict[str, Any] | None:
        """``{energy_eV, weight_uA_V2_eV}`` of a k-point's transitions: weights in microA/V^2 eV per transition."""
        if part is None:
            return None
        energies, weights = part
        return {
            "energy_eV": np.asarray(energies, float).tolist(),
            "weight_uA_V2_eV": (self.record_scale * factor * np.asarray(weights, float)).tolist(),
        }

    def rounding_block(self, evaluation: ShiftEvaluation, scale: float) -> dict[str, Any]:
        """Each k-point's rounding scale (:func:`rounding_scale`) times ``|scale|`` (microA/V^2 eV), in the grid's
        order (``kpoint_order``)."""
        scales = evaluation.sums.get("scales") or []
        recorded = len(scales) == evaluation.sums["points"] and len(scales) <= ROUNDING_RECORD_LIMIT
        return {
            "recorded": recorded,
            "limit": ROUNDING_RECORD_LIMIT,
            "kpoint_scale_uA_V2_eV": (abs(scale) * np.asarray(scales, float)).tolist() if recorded else [],
            "rule": "per k-point (1 + max|E| / s) max over components of sum over its transitions of |f| "
            "|r^b r^c_;a|_sym (s the smallest level spacing above the degeneracy tolerance), in microA/V^2 eV like "
            "the transitions' weights: the transitions' weights round to a small multiple of the machine epsilon "
            "times it, however much they cancel across the k-points",
        }

    def exact_block(self, main: ShiftEvaluation) -> dict[str, Any]:
        """The k-points where an exact degeneracy takes part in PyATB's pairs, with the spacings QUEER treats as
        exact."""
        sums = main.sums
        records = sums.get("exact_records") or []
        recorded = len(records) == sums["exact"] and len(records) <= EXACT_RECORD_LIMIT
        points = kmesh(self.grid)
        return {
            "recorded": recorded,
            "limit": EXACT_RECORD_LIMIT,
            "degeneracy_tolerance_eV": float(self.view.degeneracy_tolerance),
            "kpoints": [
                {"k": points[item["k_index"]].tolist(), "k_index": item["k_index"], "pairs": item["pairs"]}
                for item in records
            ]
            if recorded
            else [],
            "rule": "every k-point where a band of a pair PyATB selects lies within QUEER's degeneracy tolerance "
            "of another level (unscissored levels, zero-based [i, j, spacing eV]): QUEER sums such a block "
            "covariantly and PyATB's method 1 skips it",
        }

    def window_block(self, main: ShiftEvaluation) -> dict[str, Any] | None:
        """With PyATB's Gaussian and no band pair: the transitions at the k-points PyATB does not regularize that
        its window drops within reach of a photon energy, and those it keeps at its ends."""
        if self.smearing != 1 or self.pair_mode:
            return None
        sums = main.sums
        rule = (
            "transitions at the k-points PyATB does not regularize that PyATB's window [start, start + count "
            f"domega) on its Hartree scale drops while QUEER's Gaussian of them reaches a photon energy within "
            f"{WINDOW_REACH_ETA:g} eta (kept false), and those it keeps within {EDGE_EV:g} eV of its ends (kept "
            "true); k_index in kpoint_order, weights in microA/V^2 eV (PyATB's conventions, QUEER's constants)"
        )
        if sums.get("window_overflow"):
            return {
                "recorded": False,
                "limit": WINDOW_RECORD_LIMIT,
                "reason": f"more than {WINDOW_RECORD_LIMIT} transitions",
                "rule": rule,
            }
        items = sorted(sums.get("window") or [], key=lambda item: (item[0], item[1]))
        return {
            "recorded": True,
            "limit": WINDOW_RECORD_LIMIT,
            "reach_eta": WINDOW_REACH_ETA,
            "edge_eV": EDGE_EV,
            "k_index": [item[0] for item in items],
            "energy_eV": [item[1] for item in items],
            "weight_uA_V2_eV": [(self.record_scale * main.factor * item[2]).tolist() for item in items],
            "kept": [item[3] for item in items],
            "rule": rule,
        }

    def channel_block(self, main: ShiftEvaluation, down: ShiftEvaluation | None) -> dict[str, Any] | None:
        """A collinear scfout's two channels (physical normalization), which sum to the physical value."""
        if down is None:
            return None
        return {
            "up_uA_V2": components(component_table(main.physical)),
            "down_uA_V2": components(component_table(down.physical)),
            "binned_up_uA_V2": None if main.binned is None else components(main.binned),
            "binned_down_uA_V2": None if down.binned is None else components(down.binned),
            "down_rounding": self.rounding_block(down, down.factor),
            "rule": "sigma_uA_V2 = up + down (binned_uA_V2 = binned_up + binned_down); legacy_single_channel = "
            "pyatb_spin_factor x up; each channel occupied once per level with occ_band levels",
        }

    def moved_block(self, main: ShiftEvaluation) -> dict[str, Any] | None:
        """Without smearing: the transitions at the k-points PyATB does not regularize that its integer bins place
        in another row than QUEER's histogram, or that an edge's rounding could move."""
        if self.smearing != 0:
            return None
        sums = main.sums
        rule = (
            "transitions at the k-points PyATB does not regularize whose row in QUEER's histogram half a bin higher "
            "((w_i, w_i + domega]) differs from PyATB's int((E - start)/domega) on its Hartree scale with its window "
            "rule ([w_i, w_i + domega)), or that lie within 1e-9 eV of an edge of either; -1: no row; k_index in "
            "kpoint_order; weights in microA/V^2 per row (PyATB's conventions, QUEER's constants)"
        )
        if sums.get("moved_overflow"):
            return {
                "recorded": False,
                "limit": MOVED_RECORD_LIMIT,
                "reason": f"more than {MOVED_RECORD_LIMIT} transitions",
                "rule": rule,
            }
        moved = sorted(sums.get("moved") or [], key=lambda item: (item[5], item[0], item[2], item[3]))
        return {
            "recorded": True,
            "limit": MOVED_RECORD_LIMIT,
            "k_index": [item[5] for item in moved],
            "energy_eV": [item[0] for item in moved],
            "weight_uA_V2_eV": [(self.record_scale * main.factor * item[1]).tolist() for item in moved],
            "queer_row": [item[2] for item in moved],
            "pyatb_row": [item[3] for item in moved],
            "ambiguous": [item[4] for item in moved],
            "rule": rule,
        }

    def diagnostics(
        self, main: ShiftEvaluation, doubled: dict[str, Any] | None, down: ShiftEvaluation | None = None
    ) -> dict[str, Any]:
        sums, scheme = main.sums, self.scheme
        pyatb, untruncated = self.scheme_tables(main)
        low, high = float(sums["low"]), float(sums["high"])
        found = math.isfinite(low)
        last = float(self.omega[-1])
        low_p = high_p = None
        margins = None
        if found:
            low_p, high_p = low * scheme.energy_scale, high * scheme.energy_scale
            margins = {"low_eV": low_p - scheme.start, "high_eV": last - high_p}
        eta = float(self.block.get("eta", 0.0) or 0.0) if self.smearing else None
        margin_eta = None
        if margins is not None and eta:
            margin_eta = min(margins["low_eV"], margins["high_eV"]) / eta
        transitions = {
            "count": sums["transitions"],
            "inside_window": sums["inside"],
            "below_window": sums["below"],
            "above_window": sums["above"],
            "min_eV": low if found else None,
            "max_eV": high if found else None,
            "min_pyatb_eV": low_p,
            "max_pyatb_eV": high_p,
            "window_pyatb_eV": [scheme.start, scheme.end],
            "last_omega_eV": last,
            "margin_low_eV": None if margins is None else margins["low_eV"],
            "margin_high_eV": None if margins is None else margins["high_eV"],
            "margin_eta": margin_eta,
            "rule": "PyATB keeps the transitions in [start, start + count*domega) of its scale (every one of a "
            "named band pair); the margins run from the extreme transitions to the first and last photon energy",
        }
        tail = math.exp(-(((scheme.interval * scheme.domega) / scheme.eta) ** 2))
        truncation = None
        if self.smearing == 1 and pyatb is not None and untruncated is not None:
            effect = np.max(np.abs(untruncated - pyatb), axis=0)
            cut = scheme.interval * scheme.domega
            truncation = {
                "interval_bins": scheme.interval,
                "cut_eV": cut,
                "tail_relative": tail,
                "effect_uA_V2": dict(zip(SHIFT_COMPONENTS, effect.tolist())),
                "max_effect_uA_V2": float(effect.max()),
                "rule": "PyATB sums its Gaussian over the bins within int(5 eta/domega) of a transition's bin and "
                "keeps only the transitions inside its window: effect = max over omega of |untruncated - PyATB|",
            }
        regularisation = self.regularisation_block(main)
        scheme_block: dict[str, Any] = {"role": EMULATION_ROLE}
        if pyatb is None:
            scheme_block.update(
                sigma_uA_V2=None, reason="PyATB's adaptive widths (smearing_method 2) are not emulated"
            )
        else:
            dropped = sums.get("dropped")
            listed = sums.get("transitions_table")
            scheme_block.update(
                sigma_uA_V2=components(pyatb),
                scheme="PyATB's Hartree scale for the transition energies, its window rule, int() bins, "
                + (
                    "bin weights without 1/domega"
                    if self.smearing == 0
                    else "Gaussian truncated at int(5 eta/domega) bins"
                )
                + ", its SI constants, spin factor and signed volume, and its method-1 cut at the flagged "
                "k-points, on QUEER's transitions of the model PyATB reads",
                energy_scale=scheme.energy_scale,
                constant_ratio=self.ratio,
                spin_factor=self.spin_factor,
                export_sign=self.sign,
                transitions_uA_V2=None
                if listed is None
                else components(self.record_scale * main.factor * listed),
                transitions_rule="the transitions this emulation sums (QUEER's cut at the flagged k-points) under "
                "QUEER's Gaussian, eV and constants, in PyATB's export sign and spin factor: the value less the "
                "measured cut, summed by the emulation's own code",
                window_dropped_uA_V2=None
                if dropped is None
                else components(self.record_scale * main.factor * dropped),
                window_dropped_rule="QUEER's spectrum (its eV, kernel and constants; PyATB's export sign and spin "
                "factor) of the transitions PyATB's window drops, at the k-points it does not regularize",
                moved_transitions=self.moved_block(main),
                window_transitions=self.window_block(main),
                edge_transitions=sums["edge_count"],
                edge_eV=EDGE_EV,
            )
        gap = sums["pair_gap"]
        clean = {
            "scissors_zero": float(self.native["scissors"]) == 0.0,
            "gaussian": self.smearing == 1,
            "method_1": int(self.block["method"]) == 1,
            "truncation_tail_below_1e-6": self.smearing == 1 and tail <= CLEAN_TRUNCATION_TAIL,
            "window_margin_5_eta": bool(margin_eta is not None and margin_eta >= CLEAN_MARGIN_ETA),
            "no_pair_within_1e-4_eV": sums["flagged"] == 0,
            "pair_gap_above_1e-3_eV": bool(gap > CLEAN_PAIR_GAP_EV),
        }
        return {
            "model": (
                "the scfout's spinor model"
                if self.model.spin_mode == "soc"
                else "the spin-up channel occupied once, the model PyATB reads"
            ),
            "min_pair_gap_eV": gap if math.isfinite(gap) else None,
            "min_pair_gap_k": sums["pair_gap_k"],
            "pair_gap_rule": "the smallest spacing (scissored levels, as PyATB's derivatives see them) between a "
            "band of a pair PyATB selects at a k-point and any other level QUEER does not treat as degenerate "
            "with it; null when there is none. Exact degeneracies (within QUEER's degeneracy tolerance, Kramers "
            "pairs among them) are skipped by both codes",
            "exact_degenerate_kpoints": sums["exact"],
            "exact_degeneracies": self.exact_block(main),
            "kpoints_at_pyatb_threshold": sums["flagged"],
            "pyatb_degenerate_threshold_eV": PYATB_DEGENERATE_EV,
            "kpoint_order": "the grid's k-points (i/N1, j/N2, l/N3) at index (i N2 + j) N3 + l",
            "rounding": self.rounding_block(main, self.record_scale * main.factor),
            "spin_channels": self.channel_block(main, down),
            "regularisation": regularisation,
            "transitions": transitions,
            "gaussian_truncation": truncation,
            "pyatb_scheme": scheme_block,
            "clean_control": {"eligible": all(clean.values()), "conditions": clean},
            "self_convergence": doubled,
        }

    def regularisation_block(self, main: ShiftEvaluation) -> dict[str, Any]:
        """PyATB's ``method 1`` cut at the flagged k-points, measured on QUEER's spectrum (PyATB's conventions)."""
        sums = main.sums
        method = int(self.block["method"])
        scale = self.sign * self.spin_factor * main.factor
        shift = scale * component_table(sums["shift"])
        unflagged = scale * component_table(sums["physical"] - sums["flagged_physical"])
        binned = binned_unflagged = None
        if sums.get("binned_shift") is not None:
            binned = self.scheme.domega * scale * component_table(sums["binned_shift"])
            binned_unflagged = (
                self.scheme.domega * scale * component_table(sums["binned"] - sums["flagged_binned"])
            )
        records = sorted(sums.get("records") or [], key=lambda item: item["k"])
        recorded = (
            method == 1
            and sums["flagged"] <= FLAGGED_RECORD_LIMIT
            and sums.get("record_numbers", 0) <= FLAGGED_RECORD_NUMBERS
            and len(records) == sums["flagged"]
        )
        reason = None
        if not recorded and method == 1:
            reason = (
                f"{sums['flagged']} flagged k-points ({sums.get('record_numbers', 0)} numbers) exceed the record's "
                f"limit of {FLAGGED_RECORD_LIMIT} k-points and {FLAGGED_RECORD_NUMBERS} numbers"
            )
        flagged_records = {
            "recorded": recorded,
            "limit_kpoints": FLAGGED_RECORD_LIMIT,
            "limit_numbers": FLAGGED_RECORD_NUMBERS,
            "reason": reason,
            "kpoints": [
                {
                    "k": item["k"],
                    "pair_gap_eV": item["pair_gap_eV"],
                    "physical": self.transition_record(item["physical"], main.factor),
                    "cut": self.transition_record(item["cut"], main.factor),
                }
                for item in records
            ]
            if recorded
            else [],
            "rule": "every flagged k-point (fractional, PyATB's grid) with QUEER's transitions there (energy, QUEER's "
            "eV; the 18 weights (f_n - f_m) Im[r^b r^c_;a]_sym times QUEER's conversion over the k-points, in PyATB's "
            "export sign and spin factor, microA/V^2 eV) and the same with its degeneracy tolerance at 1e-4 eV (cut; "
            "null where that failed); their Gaussian or bins sum to the value less sigma_unflagged, and to the shift",
        }
        return {
            "pyatb_method": method,
            "threshold_eV": PYATB_DEGENERATE_EV,
            "flagged_kpoints": sums["flagged"],
            "kpoints": sums["points"],
            "emulated_kpoints": sums["flagged"] - sums["cut_failed"],
            "not_emulated_kpoints": sums["cut_failed"],
            "shift_uA_V2": components(shift),
            "binned_shift_uA_V2": None if binned is None else components(binned),
            "sigma_unflagged_uA_V2": components(unflagged),
            "binned_unflagged_uA_V2": None if binned_unflagged is None else components(binned_unflagged),
            "flagged_records": flagged_records,
            "listed": [
                {"k": k, "pair_gap_eV": gap, "max_shift_uA_V2": abs(scale) * largest}
                for largest, k, gap in sums["flagged_points"]
            ],
            "rule": (
                "method 0 regularizes every energy denominator of its eigenvector derivatives with 0.04 eV at "
                "every k-point (get_partial_eigenvectors_1k_SumOverStates): no flagged set"
                if method == 0
                else "method 1 (get_D_degenerate) replaces 1/dE by dE/(dE^2 + 0.04^2) for |dE| <= 1e-4 eV between "
                "a band of a selected pair and any other level: at the k-points with such a spacing above QUEER's "
                "degeneracy tolerance, shift = QUEER's spectrum - QUEER's spectrum with its degeneracy tolerance "
                "at 1e-4 eV, in PyATB's conventions (export sign, spin factor), QUEER's kernel and constants; "
                "sigma_unflagged is QUEER's spectrum over the other k-points in the same conventions and "
                "flagged_records holds every flagged k-point's transitions"
            ),
        }

    def self_convergence(self, physical: np.ndarray, legacy: Mapping[str, Any] | None) -> dict[str, Any]:
        """The physical value (and the legacy channel) on the doubled grid, with ``|sigma(N) - sigma(2N)|``."""
        grid = [value * 2 if value > 1 else value for value in self.grid]
        if all(value == 1 for value in self.grid):
            grid = [2, 2, 2]
        collinear = self.model.spin_mode == "collinear"
        main = self.evaluate("shift current (doubled grid)", self.view, grid, diagnostics=False, binned=False)
        down = None
        if collinear:
            down = self.evaluate(
                "shift current (doubled grid, spin-down channel)",
                self.down_model,
                grid,
                diagnostics=False,
                binned=False,
            )
        doubled, _ = self.combine(main, down)
        block = {"grid": self.grid, "grid_doubled": grid, "relative_rule": SELF_CONVERGENCE_RELATIVE}
        block["physical"] = _convergence(physical, component_table(doubled))
        block["legacy_single_channel"] = None
        if legacy is not None:
            single = np.column_stack([legacy["sigma_uA_V2"][name] for name in SHIFT_COMPONENTS])
            block["legacy_single_channel"] = _convergence(
                single, self.spin_factor * component_table(main.physical)
            )
        return block

    def warn(self, main: ShiftEvaluation, legacy: Any, doubled: Mapping[str, Any] | None) -> None:
        record, sums = self.context.record, main.sums
        if not self.pair_mode and (sums["below"] or sums["above"]):
            record.warn(
                WARN_SHIFT_WINDOW,
                f"{sums['below'] + sums['above']} of {sums['transitions']} sampled transitions lie outside PyATB's "
                "window: PyATB drops them (and their Gaussian tails), QUEER's physical value keeps them",
            )
        if sums["flagged"]:
            record.warn(
                WARN_SHIFT_THRESHOLD,
                f"{sums['flagged']} k-points hold a band of a selected pair within {PYATB_DEGENERATE_EV} eV of "
                "another level that is not exactly degenerate with it: PyATB's method 1 regularizes those energy "
                "denominators (diagnostics.regularisation measures the effect)",
            )
        if doubled is not None:
            compared = doubled["legacy_single_channel"] if legacy is not None else doubled["physical"]
            if not compared["converged"]:
                record.warn(
                    WARN_SHIFT_UNCONVERGED,
                    f"|sigma(N) - sigma(2N)| = {compared['e_inf_uA_V2']:.4g} microA/V^2 exceeds "
                    f"{SELF_CONVERGENCE_RELATIVE} max|sigma(2N)| = {compared['max_doubled_uA_V2']:.4g}: refine the grid",
                )


def _convergence(coarse: np.ndarray, fine: np.ndarray) -> dict[str, Any]:
    difference = float(np.max(np.abs(coarse - fine))) if coarse.size else 0.0
    largest = float(np.max(np.abs(fine))) if fine.size else 0.0
    return {
        "sigma_doubled_uA_V2": components(fine),
        "e_inf_uA_V2": difference,
        "max_doubled_uA_V2": largest,
        "ratio": difference / largest if largest > 0 else (0.0 if difference == 0 else math.inf),
        "converged": difference <= SELF_CONVERGENCE_RELATIVE * largest,
    }


def run_shift_current(context: Any) -> None:
    """Evaluate the shift current of the step, record it and write ``Out/Shift_Current/shift_current.dat``."""
    if context.params.step != STEP_SHIFT:
        raise RunnerError(ERR_BAD_PARAMS, f"{context.params.step} is not {STEP_SHIFT}")
    ShiftStep(context).run()
