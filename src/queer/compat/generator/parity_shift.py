"""The shift-current parity rule: a ``QUEER_SHIFT_CURRENT`` run against PyATB's ``SHIFT_CURRENT`` run.

QUEER's reported values are its physical estimators. The candidate's emulation of
PyATB's numerical scheme (``diagnostics.pyatb_scheme``) and its measurement of
PyATB's ``method 1`` cut (``diagnostics.regularisation``) are comparison
diagnostics: they may explain a difference, never stand in for agreement.
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
2. **Consistency** (``emulation_consistency``; a failure is ``differs_unexplained``
   and fails the comparison whatever PyATB printed): the emulation and QUEER's
   value come from the same transitions, so ``|emulation - r (sigma_Q - shift)|``
   must stay within :func:`consistency_bound`, derived here from PyATB's constants
   and the candidate's absolute transition weights: PyATB's Hartree scale moves a
   transition by ``|s - 1| E``, its Gaussian drops the bins beyond ``int(5 eta /
   domega)``, its window drops the transitions outside it, and its integer bins
   place the transitions ``bin_moves`` counts elsewhere. ``r`` is PyATB's constant
   ratio, ``sigma_Q`` QUEER's value in PyATB's conventions (export sign, spin
   factor, legacy channel, bins) and ``shift`` the measured cut. The weights must
   bound the candidate's own table (``|sigma| <= sum |w| max G``).
3. **Symmetry gate** (a failure is ``differs_unexplained``): the components whose
   largest magnitude exceeds 1e-4 of the table's are the same in both runs, and
   each one's integral over omega has the same sign where both integrals exceed
   1e-4 of the component's integrated magnitude (:func:`symmetry_gate`), against
   QUEER's value, QUEER's value with PyATB's cut, or the emulation when it is
   consistent (step 2).
4. **Clean control**: scissors 0, Gaussian smearing, ``method 1``, PyATB's
   truncated Gaussian tail ``exp(-(int(5 eta/domega) domega / eta)^2) <= 1e-6``, the
   photon energies reaching 5 eta beyond the extreme transitions with no transition
   outside PyATB's window, and no k-point where a band of a pair PyATB selects lies
   within 1e-3 eV of a level it is not exactly degenerate with (PyATB's
   ``get_D_degenerate`` regularizes ``|dE| <= 1e-4`` eV; exact degeneracies, Kramers
   pairs among them, are skipped by both codes and do not count). Such a comparison
   is a clean control: every component must agree directly (step 5), else it is
   ``differs_unexplained``. Every ``differs_documented`` needs a passed clean-control
   report of this build (``clean_control``: the same ``queer_version``,
   ``queer_record_sha256``, ``build_sha`` and ``shift_current.build.shift_source_sha256``
   of the candidate, and this comparator's ``comparator_sha256``); without one, a
   comparison that does not agree directly is ``not_comparable`` ("no passed clean
   control for this build"), unless the candidate contradicts itself (step 2 or its
   own file), which fails. A failed report of this build makes every component
   ``differs_unexplained``; a report of another build is ``not_applicable``.
5. **Classes**, per component, with ``tol = 1e-4 max|sigma_P| + PyATB's print
   rounding + the edge allowance``: ``agree`` when PyATB's table meets QUEER's value
   (``agree_after_convention`` for the legacy single channel, a left-handed lattice
   or the histogram's ``x domega`` and half-bin transform). Otherwise
   ``differs_documented``, reporting the measured physical gap, when PyATB meets
   QUEER's value minus the measured shift of PyATB's cut at the flagged k-points
   (every one emulated), or meets the emulation while step 2 holds (PyATB's scheme,
   bounded; when that bound stays below 1 % of the component's largest value only on
   rows holding less than a tenth of its integrated size, the emulation cannot check
   QUEER's value and the comparison is ``not_comparable``), or an unbounded cause is
   active (a nonzero ``scissors``, ``method 0``'s 0.04 eV regularization of every
   denominator, ``smearing_method 2``, a flagged k-point the cut could not be
   measured at) with step 3 and **self-convergence**:
   ``|sigma_Q(N) - sigma_Q(2N)|_inf <= 0.1 max|sigma_Q(2N)|`` and ``max|sigma_Q|``
   within a factor 10 of ``max|sigma_P|`` either way (else ``not_comparable``).
   Otherwise ``differs_unexplained``.
6. **Literature anchor** (:func:`literature_anchor`, run on its own): a GaAs
   sigma^xyz peak within 2 eta and 20 % of the published height at the paper's
   scissors and broadening.

The candidate's own ``shift_current.dat`` must hold what its document says.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

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
from .results import source_sha256, to_jsonable

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
# The bound checks QUEER's value only where it is at most this share of the component's largest
# value, and it must do so on the rows holding at least INFORMATIVE_SHARE of its integrated size
# (a wrong sign, factor, column or row shows on any row it checks).
INFORMATIVE_BOUND = 1e-2
INFORMATIVE_SHARE = 0.1
SELF_CONVERGENCE_RELATIVE = 0.1
MAGNITUDE_FACTOR = 10.0
ANCHOR_PEAK_ETA = 2.0
ANCHOR_HEIGHT_RELATIVE = 0.2
UNBOUNDED_CAUSES = ("scissors", "regularisation", "adaptive_smearing")
DIRECT = (AGREE, AFTER_CONVENTION)
NO_CLEAN_CONTROL = "no passed clean control for this build"
SQRT_PI = math.sqrt(math.pi)
# The comparator's own source files (relative to the queer package): a clean-control report
# vouches for the rules that judged it.
COMPARATOR_SOURCES = (
    "compat/generator/parity.py",
    "compat/generator/parity_pyatb.py",
    "compat/generator/parity_shift.py",
)
BUILD_KEYS = ("queer_version", "queer_record_sha256", "build_sha", "shift_source_sha256", "comparator_sha256")


def comparator_sha256() -> str | None:
    """SHA-256 of this comparator's source files (:data:`COMPARATOR_SOURCES`)."""
    return source_sha256(COMPARATOR_SOURCES)


def table(columns: Mapping[str, Any] | None) -> np.ndarray | None:
    """``{component: column}`` as a ``(len(omega), 18)`` array in PyATB's order; None for None."""
    if columns is None:
        return None
    return np.column_stack([np.asarray(columns[name], float) for name in SHIFT_COMPONENTS])


def _vector(values: Mapping[str, Any]) -> np.ndarray:
    return np.array([float(values[name]) for name in SHIFT_COMPONENTS])


def gaussian(offset: Any, eta: float) -> np.ndarray:
    """PyATB's Gaussian ``exp(-(x / eta)^2) / (eta sqrt(pi))``."""
    return np.exp(-((np.asarray(offset, float) / eta) ** 2)) / (eta * SQRT_PI)


def _nonnegative(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, float)
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"transition_bounds.{name} must be finite and nonnegative")
    return array


def consistency_bound(
    bounds: Mapping[str, Any],
    *,
    smearing: int,
    eta: float,
    domega: float,
    omega: Any,
    pair_mode: bool,
) -> np.ndarray:
    """The largest ``|PyATB's scheme - r sigma_Q|`` the candidate's transitions allow, ``(len(omega), 18)``.

    Both tables sum the same transitions ``(E, w)``. Without smearing they differ
    only by the transitions the two bin rules place apart: ``r bin_moves``. With
    PyATB's Gaussian ``G`` they differ, at photon energy ``omega_i``, by

    - PyATB's Hartree scale ``s``: ``|G(omega - sE) - G(omega - E)| <= |s - 1| |E|
      (|G'(omega - E)| + |s - 1| |E| max|G''|)``, so ``|s - 1| E_max abs_slope + (|s -
      1| E_max)^2 (2 / (eta^3 sqrt(pi))) abs_weight``;
    - its truncation: a row it drops lies more than ``int(5 eta / domega) domega``
      from the transition, so ``G(int(5 eta / domega) domega) abs_weight``;
    - its window (unless a band pair is named): a transition below it lies more than
      ``omega_i - start`` below ``omega_i`` and one at or above ``end = start + count
      domega`` more than ``end - omega_i`` above, so ``G(omega_i - start) abs_below +
      G(end - omega_i) abs_above``;

    all times ``r``, PyATB's constant ratio. ``bounds`` is the candidate's
    ``diagnostics.transition_bounds``; a negative or non-finite entry raises
    ``ValueError``.
    """
    ratio = pyatb_shift_constant_ratio()
    omega = np.asarray(omega, float)
    if smearing == 0:
        return ratio * _nonnegative("bin_moves_uA_V2", table(bounds["bin_moves_uA_V2"]))
    if smearing != 1:
        raise ValueError("only PyATB's Gaussian and its bins are emulated")
    weight = _nonnegative("abs_weight_uA_V2_eV", _vector(bounds["abs_weight_uA_V2_eV"]))
    slope = _nonnegative("abs_slope_uA_V2_per_eV", table(bounds["abs_slope_uA_V2_per_eV"]))
    energy = float(_nonnegative("max_abs_energy_eV", bounds["max_abs_energy_eV"]))
    move = abs(PYATB_HARTREE_EV / HARTREE_EV - 1) * energy
    curvature = 2 / (eta**3 * SQRT_PI)
    total = move * slope + (move**2 * curvature + gaussian(int(5 * eta / domega) * domega, eta)) * weight
    if not pair_mode:
        below = _nonnegative(
            "abs_weight_below_window_uA_V2_eV", _vector(bounds["abs_weight_below_window_uA_V2_eV"])
        )
        above = _nonnegative(
            "abs_weight_above_window_uA_V2_eV", _vector(bounds["abs_weight_above_window_uA_V2_eV"])
        )
        start = float(omega[0])
        end = start + len(omega) * domega
        total = total + np.outer(gaussian(np.maximum(omega - start, 0.0), eta), below)
        total = total + np.outer(gaussian(np.maximum(end - omega, 0.0), eta), above)
    return ratio * total


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


def _largest(values: np.ndarray) -> float:
    return float(np.max(np.abs(values), initial=0.0))


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
        self.informative = np.ones(len(SHIFT_COMPONENTS), bool)

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

    def reference_omega(self) -> list[float]:
        start, end = (float(value) for value in self.effective["omega"])
        step = float(self.effective["domega"])
        return [start + step * index for index in range(int((end - start) / step) + 1)]

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
    def compared_tables(self) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, str]:
        """QUEER's value in PyATB's conventions, its emulation of PyATB's scheme, the measured shift of
        PyATB's cut (all in PyATB's conventions) and the compared name."""
        variant = self.block["legacy_single_channel"] if self.legacy else self.block
        name = "legacy_single_channel." if self.legacy else ""
        binned = int(self.effective["smearing_method"]) == 0
        values = table(variant["binned_uA_V2" if binned else "sigma_uA_V2"])
        name += "binned_uA_V2" if binned else "sigma_uA_V2"
        assert values is not None
        diagnostics = self.block.get("diagnostics") or {}
        scheme = diagnostics.get("pyatb_scheme") or {}
        regularisation = diagnostics.get("regularisation") or {}
        shift = table(regularisation.get("binned_shift_uA_V2" if binned else "shift_uA_V2"))
        return self.sign * values, table(scheme.get("sigma_uA_V2")), shift, name

    def causes(self, diagnostics: Mapping[str, Any], emulated: bool) -> dict[str, dict[str, Any]]:
        effective = self.effective
        method = int(effective["method"])
        smearing = int(effective["smearing_method"])
        regularisation = diagnostics.get("regularisation")
        if method == 0:
            reg = {
                "active": True,
                "bounded": False,
                "detail": "method 0: PyATB regularizes every energy denominator of its eigenvector derivatives at "
                f"every k-point, d / (d^2 + {PYATB_SHIFT_REGULARIZATION_EV}^2) "
                "(get_partial_eigenvectors_1k_SumOverStates)",
            }
        elif not isinstance(regularisation, Mapping):
            reg = {
                "active": True,
                "bounded": False,
                "detail": "the candidate records no diagnostics.regularisation: PyATB's method-1 cut is unmeasured",
            }
        else:
            flagged = int(regularisation.get("flagged_kpoints") or 0)
            missing = int(regularisation.get("not_emulated_kpoints") or 0)
            reg = {
                "active": flagged > 0,
                "bounded": missing == 0,
                "flagged_kpoints": flagged,
                "not_emulated_kpoints": missing,
                "detail": f"method 1: at {flagged} of {regularisation.get('kpoints')} k-points a band of a selected "
                f"pair lies within {PYATB_DEGENERATE_EV} eV of a level it is not exactly degenerate with; PyATB's "
                "get_D_degenerate regularizes those denominators, measured as QUEER's spectrum minus QUEER's "
                f"spectrum with that cut (diagnostics.regularisation; {missing} k-points could not be measured)",
            }
        truncation = diagnostics.get("gaussian_truncation") or {}
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
                "active": emulated,
                "bounded": True,
                "measured_truncation_uA_V2": truncation.get("max_effect_uA_V2"),
                "detail": "PyATB's Hartree scale, its Gaussian truncated at int(5 eta/domega) bins, its window "
                "rule, its integer bins without smearing and its constants: the emulation reproduces them and "
                "step 2 bounds how far they move QUEER's value",
            },
        }

    def clean_conditions(self, diagnostics: Mapping[str, Any]) -> dict[str, bool]:
        effective = self.effective
        transitions = diagnostics.get("transitions") or {}
        bounds = diagnostics.get("transition_bounds") or {}
        regularisation = diagnostics.get("regularisation")
        omega = self.reference_omega()
        domega = float(effective["domega"])
        smearing = int(effective["smearing_method"])
        low, high = transitions.get("min_pyatb_eV"), transitions.get("max_pyatb_eV")
        eta = float(effective.get("eta") or 0.0)
        margin = False
        if low is not None and high is not None and eta > 0:
            margin = min(low - omega[0], omega[-1] - high) >= CLEAN_MARGIN_ETA * eta
        outside = [
            bounds.get(key)
            for key in ("abs_weight_below_window_uA_V2_eV", "abs_weight_above_window_uA_V2_eV")
        ]
        empty = all(isinstance(value, Mapping) and not any(value.values()) for value in outside)
        tail = math.exp(-(((int(5 * eta / domega) * domega) / eta) ** 2)) if eta > 0 else 1.0
        measured = isinstance(regularisation, Mapping)
        flagged = (
            int(regularisation.get("flagged_kpoints") or 0) if isinstance(regularisation, Mapping) else 1
        )
        gap = diagnostics.get("min_pair_gap_eV")
        return {
            "scissors_zero": float(effective["scissors"]) == 0.0,
            "gaussian": smearing == 1,
            "method_1": int(effective["method"]) == 1,
            "truncation_tail_below_1e-6": smearing == 1 and tail <= CLEAN_TRUNCATION_TAIL,
            "window_margin_5_eta": bool(margin and empty),
            "no_pair_within_1e-4_eV": measured and flagged == 0,
            "pair_gap_above_1e-3_eV": measured and (gap is None or float(gap) > CLEAN_PAIR_GAP_EV),
        }

    def edge_allowance(self, diagnostics: Mapping[str, Any], clean: bool) -> np.ndarray:
        """The candidate's allowance for transitions on PyATB's bin or window edges, bounded by its weights.

        A transition moves at most its weight times PyATB's largest kernel value, so
        the allowance is capped at ``abs_weight / (eta sqrt(pi))`` (``abs_weight``
        without smearing). A clean control has no transition near the window's edges
        and a bin edge moves its truncated Gaussian by less than its tail: none.
        """
        if clean:
            return np.zeros(len(SHIFT_COMPONENTS))
        scheme = diagnostics.get("pyatb_scheme") or {}
        edge = np.array(
            [float((scheme.get("edge_allowance_uA_V2") or {}).get(name, 0.0)) for name in SHIFT_COMPONENTS]
        )
        bounds = diagnostics.get("transition_bounds") or {}
        weight = bounds.get("abs_weight_uA_V2_eV")
        if not isinstance(weight, Mapping):
            return np.zeros(len(SHIFT_COMPONENTS))
        cap = _nonnegative("abs_weight_uA_V2_eV", _vector(weight))
        eta = float(self.effective.get("eta") or 0.0)
        if int(self.effective["smearing_method"]) == 1 and eta > 0:
            cap = cap / (eta * SQRT_PI)
        return np.clip(edge, 0.0, cap)

    def build_identity(self) -> dict[str, Any]:
        """The build this candidate and this comparator are: what a clean-control report must match."""
        provenance = self.candidate.document.get("provenance") or {}
        build = self.block.get("build") or {}
        return {
            "queer_version": provenance.get("queer_version"),
            "queer_record_sha256": provenance.get("queer_record_sha256"),
            "build_sha": provenance.get("build_sha"),
            "shift_source_sha256": build.get("shift_source_sha256"),
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
            mine["shift_source_sha256"] is None
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
    def check_consistency(
        self, diagnostics: Mapping[str, Any], regularised: np.ndarray, emulated: np.ndarray
    ) -> np.ndarray:
        """Step 2, per component: the emulation within :func:`consistency_bound` of ``r`` times QUEER's value."""
        effective = self.effective
        smearing = int(effective["smearing_method"])
        eta = float(effective.get("eta") or 0.0)
        bounds = diagnostics.get("transition_bounds")
        if not isinstance(bounds, Mapping):
            self.quantity(
                "emulation_consistency",
                UNEXPLAINED,
                detail="the candidate holds an emulation of PyATB's scheme but no transition_bounds to check it by",
            )
            return np.zeros(len(SHIFT_COMPONENTS), bool)
        bound = consistency_bound(
            bounds,
            smearing=smearing,
            eta=eta,
            domega=float(effective["domega"]),
            omega=self.reference_omega(),
            pair_mode=self.pair_mode,
        )
        ratio = pyatb_shift_constant_ratio()
        target = ratio * regularised
        floor = CONSISTENCY_FLOOR * max(_largest(emulated), _largest(target))
        gap = np.abs(emulated - target)
        excess = gap - bound - floor
        weight = _nonnegative("abs_weight_uA_V2_eV", _vector(bounds["abs_weight_uA_V2_eV"]))
        peak = weight / (eta * SQRT_PI) if smearing == 1 else weight
        plausible = np.all(np.abs(regularised) <= peak * (1 + CONSISTENCY_FLOOR) + floor, axis=0)
        consistent = (np.max(excess, axis=0, initial=-np.inf) <= 0) & plausible
        worst = int(np.argmax(np.max(excess, axis=0)))
        row = int(np.argmax(excess[:, worst]))
        name = SHIFT_COMPONENTS[worst]
        detail = (
            f"|emulation - r sigma_Q| {gap[row, worst]:.4g} vs the derived bound {bound[row, worst] + floor:.4g} "
            f"microA/V^2 at sigma_{name}, omega index {row} (r = {ratio:.12g})"
        )
        if not plausible.all():
            implausible = [n for n, ok in zip(SHIFT_COMPONENTS, plausible) if not ok]
            detail += f"; the absolute transition weights do not bound QUEER's own table for {implausible}"
        if not consistent.all():
            detail += (
                "; the candidate's physical value and its emulation of PyATB's scheme are not computed from the same "
                f"transitions for {[n for n, ok in zip(SHIFT_COMPONENTS, consistent) if not ok]}"
            )
        self.quantity(
            "emulation_consistency",
            AGREE if consistent.all() else UNEXPLAINED,
            float(gap[row, worst]),
            float(bound[row, worst] + floor),
            detail,
        )
        magnitude = np.abs(regularised)
        checked = bound <= INFORMATIVE_BOUND * np.max(magnitude, axis=0, initial=0.0)
        total = magnitude.sum(axis=0)
        share = np.where(total > 0, (magnitude * checked).sum(axis=0) / np.where(total > 0, total, 1.0), 1.0)
        self.informative = share >= INFORMATIVE_SHARE
        self.basis["consistency"] = {
            "informative_share": dict(zip(SHIFT_COMPONENTS, share.tolist())),
            "informative_rule": f"the bound is at most {INFORMATIVE_BOUND:g} of the component's largest |sigma_Q| "
            f"on rows holding at least {INFORMATIVE_SHARE:g} of its integrated |sigma_Q|",
            "constant_ratio": ratio,
            "energy_scale": PYATB_HARTREE_EV / HARTREE_EV,
            "floor_relative": CONSISTENCY_FLOOR,
            "max_bound_uA_V2": dict(zip(SHIFT_COMPONENTS, np.max(bound, axis=0).tolist())),
            "max_gap_uA_V2": dict(zip(SHIFT_COMPONENTS, np.max(gap, axis=0).tolist())),
        }
        return consistent

    def compare_shift(self) -> None:
        product = self.reference.product or {}
        reference = np.asarray(product["sigma"], float)
        values, emulated, shift, compared = self.compared_tables()
        diagnostics = self.block.get("diagnostics") or {}
        for other in (emulated, shift):
            if other is not None and other.shape != values.shape:
                raise ValueError(f"tables of shape {other.shape} and {values.shape} cannot be compared")
        if reference.shape != values.shape:
            raise ValueError(f"tables of shape {reference.shape} and {values.shape} cannot be compared")
        domega = float(self.effective["domega"])
        scale = _largest(reference)
        scheme = diagnostics.get("pyatb_scheme") or {}
        conditions = self.clean_conditions(diagnostics)
        self.clean = clean = all(conditions.values())
        edge = self.edge_allowance(diagnostics, clean)
        base = AGREEMENT_RELATIVE * scale
        tolerance = base + PRINT_RELATIVE * np.max(np.abs(reference), axis=0, initial=0.0) + edge
        causes = self.causes(diagnostics, emulated is not None)
        flagged = bool(causes["regularisation"].get("flagged_kpoints"))
        regularised = values - shift if flagged and shift is not None else values
        consistent = None
        if emulated is not None:
            consistent = self.check_consistency(diagnostics, regularised, emulated)
        symmetric, facts = symmetry_gate(reference, values, domega, tolerance)
        facts["against"] = compared
        views = []
        if regularised is not values:
            views.append((regularised, f"{compared} - diagnostics.regularisation shift"))
        if consistent is not None and consistent.all() and emulated is not None:
            views.append((emulated, "diagnostics.pyatb_scheme.sigma_uA_V2"))
        for view, label in views:
            if symmetric:
                break
            passes, other_facts = symmetry_gate(reference, view, domega, tolerance)
            if passes:
                symmetric, facts = True, {**other_facts, "against": label}
        self.quantity(
            "symmetry",
            AGREE if symmetric else UNEXPLAINED,
            detail=(
                f"significant components {facts['reference_components']} (PyATB) and {facts['candidate_components']} "
                f"(QUEER's {facts['against']}), differing for {facts['components_differ']}; integral signs differ "
                f"for {facts['integral_sign_differs']}"
            ),
        )
        self.release = release = self.release_clean_control()
        unbounded = [
            name for name in UNBOUNDED_CAUSES if causes[name]["active"] and not causes[name]["bounded"]
        ]
        convention = self.legacy or self.sign < 0 or int(self.effective["smearing_method"]) == 0
        converged, convergence = None, ""
        bound_max = self.basis.get("consistency", {}).get("max_bound_uA_V2") or {}
        self.direct_everywhere = True
        coarse: list[str] = []
        for index, name in enumerate(SHIFT_COMPONENTS):
            direct = _largest(reference[:, index] - values[:, index])
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
                    AFTER_CONVENTION if convention else AGREE,
                    direct,
                    limit,
                    f"PyATB's table meets QUEER's {compared}"
                    + (" in PyATB's conventions" if convention else ""),
                )
                continue
            self.direct_everywhere = False
            if clean:
                self.quantity(
                    label,
                    UNEXPLAINED,
                    direct,
                    limit,
                    "a clean control must agree with QUEER's physical value directly to 1e-4 max|sigma_P|",
                )
                continue
            if regularised is not values and causes["regularisation"]["bounded"] and symmetric:
                cut = _largest(reference[:, index] - regularised[:, index])
                if cut <= limit:
                    moved = _largest(shift[:, index]) if shift is not None else 0.0
                    self.quantity(
                        label,
                        DOCUMENTED,
                        direct,
                        limit + moved,
                        f"QUEER's value differs by {direct:.4g} microA/V^2; PyATB meets it with PyATB's method-1 cut "
                        f"at the {causes['regularisation']['flagged_kpoints']} flagged k-points (measured shift up to "
                        f"{moved:.4g}) to {cut:.4g}",
                    )
                    continue
            if emulated is not None and consistent is not None and consistent[index] and symmetric:
                through = _largest(reference[:, index] - emulated[:, index])
                if through <= limit and not self.informative[index]:
                    coarse.append(name)
                    self.quantity(
                        label,
                        UNEXPLAINED,
                        direct,
                        limit,
                        f"PyATB meets QUEER's emulation of its scheme to {through:.4g}, but that scheme moves its "
                        "table too far from QUEER's value for the emulation to check it",
                    )
                    continue
                if through <= limit:
                    allowance = float(bound_max.get(name, 0.0))
                    self.quantity(
                        label,
                        DOCUMENTED,
                        direct,
                        limit + allowance + _largest(regularised[:, index] - values[:, index]),
                        f"QUEER's value differs by {direct:.4g} microA/V^2; PyATB meets QUEER's emulation of its "
                        f"scheme (Hartree scale, truncation, window, integer bins, constants"
                        + (", method-1 cut" if regularised is not values else "")
                        + f") to {through:.4g}, which lies within the derived bound {allowance:.4g} of QUEER's value",
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
            elif consistent is not None and not consistent[index]:
                reason = "QUEER's emulation of PyATB's scheme is inconsistent with its physical value"
            self.quantity(label, UNEXPLAINED, direct, limit, reason)
        if coarse:
            self.precondition(
                "pyatb_scheme_resolution",
                False,
                f"PyATB's scheme (int(5 eta/domega) = {int(5 * float(self.effective.get('eta') or 0) / domega)} "
                f"bins, its window) moves its table from QUEER's value by more than {INFORMATIVE_BOUND:g} of the "
                f"component's largest value on rows holding more than {1 - INFORMATIVE_SHARE:g} of its "
                f"integrated size for {coarse}: this PyATB run cannot check QUEER's value (refine domega against "
                "eta or widen the window)",
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
                "directly to 1e-4 max|sigma_P|",
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
        }
        self.basis.update(
            {
                "relative": AGREEMENT_RELATIVE,
                "scale_uA_V2": scale,
                "base_uA_V2": base,
                "print_relative": PRINT_RELATIVE,
                "edge_allowance_uA_V2": dict(zip(SHIFT_COMPONENTS, edge.tolist())),
                "edge_transitions": scheme.get("edge_transitions"),
                "emulated": emulated is not None,
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
            if item["name"] in ("emulation_consistency", "candidate_file") and item["class"] == UNEXPLAINED
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
        for step in (self.compare_shift, self.compare_candidate_file):
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
                    "QUEER's physical value is the estimator compared, its emulation of PyATB a diagnostic"
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
    """The report of a shift-current comparison (:mod:`~queer.compat.generator.parity`'s schema)."""
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
