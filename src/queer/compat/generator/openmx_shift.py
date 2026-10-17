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
holds QUEER's histogram in PyATB's normalization instead (``binned_uA_V2``: the
weight of the transitions in ``[omega_i, omega_i + domega)``, which is ``domega``
times QUEER's histogram half a bin higher).

**Diagnostics** (comparison aids, never the result), evaluated on the model PyATB
reads with QUEER's own transitions: the smallest spacing of two levels at any
sampled k (scissored, as PyATB's derivatives see them) and the k-points within
PyATB's 1e-4 eV threshold, the transitions inside PyATB's window, PyATB's Gaussian
truncation, and ``pyatb_scheme``: what PyATB prints from QUEER's transitions (its
Hartree scale, its window rule, its integer bins and ``5 eta / domega`` truncation,
its constants, spin factor and signed volume), with the allowance of the
transitions within 1e-9 eV of a bin or window edge. ``self_convergence: true``
adds the physical value (and the legacy channel) on the doubled grid.
"""

from __future__ import annotations

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
from .results import ERR_BAD_PARAMS, ERR_VALIDATION, RunnerError

# Complex N x N matrices one shift-current k-point keeps alive, and per photon energy
# of a batch of at most FREQUENCY_BATCH (QUEER's own estimate in shift_current).
SHIFT_MATRICES = 180
SHIFT_FREQUENCY_MATRICES = 2
FREQUENCY_BATCH = 64
# A transition this close (eV, PyATB's scale) to a bin or window edge may fall on
# either side in the two codes: its weight is allowed for.
EDGE_EV = 1e-9
# The clean-control conditions of the parity rule.
CLEAN_PAIR_GAP_EV = 1e-3
CLEAN_MARGIN_ETA = 5.0
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

    def edge_bound(self) -> float:
        """The largest change at one photon energy when one transition moves across an edge (per unit weight)."""
        if self.smearing == 0:
            return 1.0
        return 1.0 / (self.eta * SQRT_PI)

    def bin_edge_bound(self) -> float:
        """The same for a move across a bin edge, which shifts PyATB's truncated window by one bin."""
        if self.smearing == 0:
            return 1.0
        reach = max(0.0, (self.interval - 1) * self.domega)
        return math.exp(-((reach / self.eta) ** 2)) / (self.eta * SQRT_PI)


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


def edge_allowance(
    scheme: PyatbShiftScheme, energies: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, int]:
    """The weight of the transitions within :data:`EDGE_EV` of a bin or window edge, times the
    largest change their move makes at one photon energy; and their count."""
    if not len(energies):
        return np.zeros(len(SHIFT_COMPONENTS)), 0
    energy = np.asarray(energies, float) * scheme.energy_scale
    magnitude = np.abs(np.asarray(weights, float))
    position = (energy - scheme.start) / scheme.domega
    fraction = position - np.floor(position)
    near_bin = np.minimum(fraction, 1 - fraction) * scheme.domega < EDGE_EV
    near_window = np.zeros(len(energy), bool)
    if not scheme.pair_mode:
        near_window = (np.abs(energy - scheme.start) < EDGE_EV) | (np.abs(energy - scheme.end) < EDGE_EV)
    bound = np.where(near_window, scheme.edge_bound(), np.where(near_bin, scheme.bin_edge_bound(), 0.0))
    return bound @ magnitude, int(np.count_nonzero(near_bin | near_window))


def floor_bins(
    start: float, domega: float, count: int, energies: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Transition weights summed in ``[start + i domega, start + (i + 1) domega)`` (QUEER's eV)."""
    table = np.zeros((count, len(SHIFT_COMPONENTS)))
    if len(energies):
        index = np.floor((np.asarray(energies, float) - start) / domega).astype(int)
        keep = (index >= 0) & (index < count)
        np.add.at(table, index[keep], np.asarray(weights, float)[keep])
    return table


def shift_rows(
    model: Any,
    settings: ShiftCurrentSettings,
    frequency_batch: int,
    scheme: PyatbShiftScheme | None,
    binned: bool,
    points: np.ndarray,
) -> dict[str, Any]:
    """The k-point sums of a chunk: QUEER's spectrum and, when asked, its transitions' diagnostics.

    ``physical`` sums :func:`~queer.observables.nonlinear.shift_current_spectrum`
    (unconverted, as :func:`~queer.observables.nonlinear.shift_current` sums it);
    ``binned`` sums :func:`floor_bins`; with a ``scheme``, ``pyatb`` and
    ``untruncated`` sum :func:`pyatb_scheme_spectrum`, ``edge`` sums
    :func:`edge_allowance` and the level and transition facts are kept.
    """
    count = len(settings.omega)
    width = len(SHIFT_COMPONENTS)
    result: dict[str, Any] = {"physical": np.zeros((count, 3, 3, 3)), "points": len(points)}
    if binned:
        result["binned"] = np.zeros((count, width))
    emulated = scheme is not None and scheme.smearing in (0, 1)
    if scheme is not None:
        result.update(
            pyatb=np.zeros((count, width)) if emulated else None,
            untruncated=np.zeros((count, width)) if emulated and scheme.smearing == 1 else None,
            edge=np.zeros(width),
            edge_count=0,
            pair_gap=math.inf,
            pair_gap_k=None,
            at_threshold=0,
            transitions=0,
            inside=0,
            below=0,
            above=0,
            low=math.inf,
            high=-math.inf,
        )
    start = float(settings.omega[0])
    step = scheme.domega if scheme is not None else (settings.bin_width or 0.0)
    for k in points:
        point = shift_current_point(model, k, settings)
        result["physical"] += shift_current_spectrum(point, settings, frequency_batch)
        if not binned and scheme is None:
            continue
        source, target = np.nonzero(point["factor"])
        energies = point["gap"][source, target]
        weights = (point["factor"][source, target] * point["integrand"][..., source, target])[_A, _B, _C].T
        if binned:
            result["binned"] += floor_bins(start, step, count, energies, weights)
        if scheme is None:
            continue
        if emulated:
            spectrum, untruncated = pyatb_scheme_spectrum(scheme, energies, weights)
            result["pyatb"] += spectrum
            if untruncated is not None:
                result["untruncated"] += untruncated
            allowance, edges = edge_allowance(scheme, energies, weights)
            result["edge"] += allowance
            result["edge_count"] += edges
        levels = np.sort(point["energies"])
        spacing = float(np.min(np.diff(levels))) if len(levels) > 1 else math.inf
        if spacing < result["pair_gap"]:
            result["pair_gap"], result["pair_gap_k"] = spacing, np.asarray(k, float).tolist()
        result["at_threshold"] += int(spacing <= PYATB_DEGENERATE_EV)
        if len(energies):
            scaled = energies * scheme.energy_scale
            result["transitions"] += len(energies)
            result["below"] += int(np.count_nonzero(scaled < scheme.start))
            result["above"] += int(np.count_nonzero(scaled >= scheme.end))
            result["inside"] += int(np.count_nonzero((scaled >= scheme.start) & (scaled < scheme.end)))
            result["low"] = min(result["low"], float(energies.min()))
            result["high"] = max(result["high"], float(energies.max()))
    return result


def _reduce(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = dict(parts[0])
    for part in parts[1:]:
        for key, value in part.items():
            if value is None:
                continue
            if key in ("physical", "binned", "pyatb", "untruncated", "edge"):
                total[key] = total[key] + value
            elif key in ("points", "edge_count", "at_threshold", "transitions", "inside", "below", "above"):
                total[key] += value
            elif key == "pair_gap" and value < total["pair_gap"]:
                total["pair_gap"], total["pair_gap_k"] = value, part["pair_gap_k"]
            elif key == "low":
                total["low"] = min(total["low"], value)
            elif key == "high":
                total["high"] = max(total["high"], value)
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
        self, stage: str, model: Any, grid: Sequence[int], *, diagnostics: bool, binned: bool
    ) -> ShiftEvaluation:
        """One pass of ``model`` over ``grid`` on the planned processes."""
        points = kmesh(grid)
        settings = self.settings(model, grid)
        frequency_batch = min(len(self.omega), FREQUENCY_BATCH)
        per_k = SHIFT_MATRICES + SHIFT_FREQUENCY_MATRICES * frequency_batch
        options = self.context.options(stage, model, len(points), per_k=per_k)
        function = functools.partial(
            shift_rows, model, settings, frequency_batch, self.scheme if diagnostics else None, binned
        )
        try:
            parts = pmap(function, chunks(points, options.batch_size), options)
        except ValueError as error:
            raise RunnerError(
                ERR_VALIDATION, f"QUEER cannot evaluate this shift current on the {stage} grid: {error}"
            ) from error
        sums = _reduce(parts)
        factor = conversion(model) / len(points)
        binned_table = factor * sums["binned"] if binned else None
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
            "diagnostics": self.diagnostics(main, doubled),
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

    def diagnostics(self, main: ShiftEvaluation, doubled: dict[str, Any] | None) -> dict[str, Any]:
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
        truncation = None
        if self.smearing == 1 and pyatb is not None and untruncated is not None:
            effect = np.max(np.abs(untruncated - pyatb), axis=0)
            cut = scheme.interval * scheme.domega
            truncation = {
                "interval_bins": scheme.interval,
                "cut_eV": cut,
                "tail_relative": math.exp(-((cut / scheme.eta) ** 2)),
                "effect_uA_V2": dict(zip(SHIFT_COMPONENTS, effect.tolist())),
                "max_effect_uA_V2": float(effect.max()),
                "rule": "PyATB sums its Gaussian over the bins within int(5 eta/domega) of a transition's bin and "
                "keeps only the transitions inside its window: effect = max over omega of |untruncated - PyATB|",
            }
        scheme_block: dict[str, Any] = {"role": EMULATION_ROLE}
        if pyatb is None:
            scheme_block.update(
                sigma_uA_V2=None, reason="PyATB's adaptive widths (smearing_method 2) are not emulated"
            )
        else:
            compared = component_table(main.physical) * self.spin_factor * self.sign
            scale_effect = None
            if untruncated is not None:
                scale_effect = dict(
                    zip(SHIFT_COMPONENTS, np.max(np.abs(untruncated - compared), axis=0).tolist())
                )
            scheme_block.update(
                sigma_uA_V2=components(pyatb),
                scheme="PyATB's Hartree scale for the transition energies, its window rule, int() bins, "
                + (
                    "bin weights without 1/domega"
                    if self.smearing == 0
                    else "Gaussian truncated at int(5 eta/domega) bins"
                )
                + ", its SI constants, spin factor and signed volume, on QUEER's transitions of the model PyATB reads",
                energy_scale=scheme.energy_scale,
                constant_ratio=self.ratio,
                spin_factor=self.spin_factor,
                export_sign=self.sign,
                energy_scale_effect_uA_V2=scale_effect,
                edge_allowance_uA_V2=dict(
                    zip(
                        SHIFT_COMPONENTS,
                        (
                            abs(self.sign * self.spin_factor * self.ratio * main.factor) * sums["edge"]
                        ).tolist(),
                    )
                ),
                edge_transitions=sums["edge_count"],
                edge_eV=EDGE_EV,
            )
        gap = sums["pair_gap"]
        clean = {
            "scissors_zero": float(self.native["scissors"]) == 0.0,
            "gaussian": self.smearing == 1,
            "method_1": int(self.block["method"]) == 1,
            "pair_gap_above_1e-3_eV": bool(math.isfinite(gap) and gap > CLEAN_PAIR_GAP_EV),
            "window_margin_5_eta": bool(margin_eta is not None and margin_eta >= CLEAN_MARGIN_ETA),
        }
        return {
            "model": (
                "the scfout's spinor model"
                if self.model.spin_mode == "soc"
                else "the spin-up channel occupied once, the model PyATB reads"
            ),
            "min_pair_gap_eV": gap if math.isfinite(gap) else None,
            "min_pair_gap_k": sums["pair_gap_k"],
            "pair_gap_rule": "the smallest spacing of two levels (scissored, as PyATB's derivatives see them) at "
            "any sampled k",
            "kpoints_at_pyatb_threshold": sums["at_threshold"],
            "pyatb_degenerate_threshold_eV": PYATB_DEGENERATE_EV,
            "transitions": transitions,
            "gaussian_truncation": truncation,
            "pyatb_scheme": scheme_block,
            "clean_control": {"eligible": all(clean.values()), "conditions": clean},
            "self_convergence": doubled,
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
        if sums["at_threshold"]:
            record.warn(
                WARN_SHIFT_THRESHOLD,
                f"{sums['at_threshold']} k-points hold two levels within {PYATB_DEGENERATE_EV} eV: PyATB "
                "regularizes their energy denominators",
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
