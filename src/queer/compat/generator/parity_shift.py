"""The shift-current parity rule: a ``QUEER_SHIFT_CURRENT`` run against PyATB's ``SHIFT_CURRENT`` run.

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
2. **Symmetry gate** (a failure is ``differs_unexplained``): the components whose
   largest magnitude exceeds 1e-4 of the table's are the same in both runs, and
   each one's integral over omega has the same sign where both integrals exceed
   1e-4 of the component's integrated magnitude (:func:`symmetry_gate`), against
   QUEER's value in PyATB's conventions or against its emulation of PyATB's
   scheme, which a coarse step can reshape.
3. **Clean control**: scissors 0, Gaussian smearing, ``method 1``, every sampled
   spacing of two levels above 1e-3 eV and the photon energies reaching at least
   5 eta beyond the extreme transitions on both sides (the candidate's
   diagnostics). Such a comparison must agree (step 6) or every component is
   ``differs_unexplained``. The report of this release's clean-control comparison
   (``clean_control``) makes every component ``differs_unexplained`` when it failed.
4. **Causes** from the data: a nonzero ``scissors`` (PyATB adds it to its
   eigenvalues before its derivative kernels; QUEER shifts transition energies
   only), regularization (``method 0`` regularizes every energy denominator of its
   eigenvector derivatives with 0.04 eV; ``method 1`` the pairs within 1e-4 eV),
   ``smearing_method 2`` (PyATB's adaptive width differs from QUEER's) are
   unbounded; no smearing (the ``x domega`` and half-bin transform) and PyATB's
   Gaussian truncation and window rule are bounded and measured by the
   candidate's emulation.
5. **Self-convergence**, needed by a documented difference:
   ``|sigma_Q(N) - sigma_Q(2N)|_inf <= 0.1 max|sigma_Q(2N)|`` and ``max|sigma_Q| <=
   10 max|sigma_P|``; otherwise the candidate is unconverged and the comparison
   ``not_comparable``.
6. **Classes**, per component, with ``tol = 1e-4 max|sigma_P| + PyATB's print
   rounding + the edge allowance``: ``agree`` when PyATB's table meets QUEER's
   value (``agree_after_convention`` for the legacy single channel, a left-handed
   lattice or the histogram's ``x domega`` and half-bin transform); otherwise
   ``agree_after_convention`` when it meets QUEER's emulation of PyATB's scheme
   (its Hartree scale, window rule, integer bins or truncated Gaussian,
   constants); otherwise ``differs_documented`` only when gates 2, 3 and 5 pass and
   an unbounded cause is active, else ``differs_unexplained``.
7. **Literature anchor** (:func:`literature_anchor`, run on its own): a GaAs
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
from .pyatb_conventions import PYATB_DEGENERATE_EV, PYATB_SHIFT_REGULARIZATION_EV, pyatb_shift_spin_factor
from .pyatb_files import SHIFT_COMPONENTS, SHIFT_CURRENT_FILE, parse_shift_current
from .results import to_jsonable

SYMMETRY_THRESHOLD = 1e-4
SIGN_SIGNIFICANCE = 1e-4
AGREEMENT_RELATIVE = 1e-4
PRINT_RELATIVE = 5.01e-7
OMEGA_TOLERANCE_EV = 1e-9
OMEGA_PRINT_EV = 5.01e-6
CLEAN_PAIR_GAP_EV = 1e-3
CLEAN_MARGIN_ETA = 5.0
SELF_CONVERGENCE_RELATIVE = 0.1
MAGNITUDE_FACTOR = 10.0
ANCHOR_PEAK_ETA = 2.0
ANCHOR_HEIGHT_RELATIVE = 0.2
UNBOUNDED_CAUSES = ("scissors", "regularisation", "adaptive_smearing")


def table(columns: Mapping[str, Any] | None) -> np.ndarray | None:
    """``{component: column}`` as a ``(len(omega), 18)`` array in PyATB's order; None for None."""
    if columns is None:
        return None
    return np.column_stack([np.asarray(columns[name], float) for name in SHIFT_COMPONENTS])


def symmetry_gate(
    reference: np.ndarray, candidate: np.ndarray, domega: float, tolerance: np.ndarray | None = None
) -> tuple[bool, dict[str, Any]]:
    """Gate 2: the same significant components and the same sign of each one's integral over omega.

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

    # ----------------------------------------------------- preconditions
    @property
    def block(self) -> dict[str, Any]:
        return self.candidate.block("shift_current")

    @property
    def effective(self) -> dict[str, Any]:
        return self.notes.get("parameters", {}).get("function", {})

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
    def compared_tables(self) -> tuple[np.ndarray, np.ndarray | None, str]:
        """QUEER's table in PyATB's conventions, its emulation of PyATB's scheme and the compared name."""
        variant = self.block["legacy_single_channel"] if self.legacy else self.block
        name = "legacy_single_channel." if self.legacy else ""
        if int(self.effective["smearing_method"]) == 0:
            values = table(variant["binned_uA_V2"])
            name += "binned_uA_V2"
        else:
            values = table(variant["sigma_uA_V2"])
            name += "sigma_uA_V2"
        assert values is not None
        scheme = (self.block.get("diagnostics") or {}).get("pyatb_scheme") or {}
        return self.sign * values, table(scheme.get("sigma_uA_V2")), name

    def causes(self, diagnostics: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        effective = self.effective
        method = int(effective["method"])
        smearing = int(effective["smearing_method"])
        gap = diagnostics.get("min_pair_gap_eV")
        close = int(diagnostics.get("kpoints_at_pyatb_threshold") or 0)
        regularised = method == 0 or (method == 1 and gap is not None and gap <= PYATB_DEGENERATE_EV)
        regularisation = (
            f"method 0: PyATB regularizes every energy denominator of its eigenvector derivatives, d / (d^2 + "
            f"{PYATB_SHIFT_REGULARIZATION_EV}^2)"
            if method == 0
            else f"method 1: {close} sampled k-points hold two levels within {PYATB_DEGENERATE_EV} eV "
            f"(smallest spacing {gap} eV), whose denominators PyATB regularizes"
        )
        truncation = diagnostics.get("gaussian_truncation") or {}
        return {
            "scissors": {
                "active": float(effective["scissors"]) != 0.0,
                "bounded": False,
                "detail": "PyATB adds the scissors to its eigenvalues before its derivative kernels; QUEER "
                "shifts the transition energies only",
            },
            "regularisation": {"active": regularised, "bounded": False, "detail": regularisation},
            "adaptive_smearing": {
                "active": smearing == 2,
                "bounded": False,
                "detail": "PyATB's adaptive width |grad(E_n - E_m)| dk eta (dk = max|b_i|/100, clipped to "
                "[1e-4, 1] eV) differs from QUEER's",
            },
            "no_smearing": {
                "active": smearing == 0,
                "bounded": True,
                "detail": "PyATB adds each transition's weight to its integer bin: domega times QUEER's "
                "histogram half a bin higher (binned_uA_V2)",
            },
            "gaussian_truncation": {
                "active": smearing == 1,
                "bounded": True,
                "measured_uA_V2": truncation.get("max_effect_uA_V2"),
                "detail": "PyATB's Gaussian reaches int(5 eta/domega) bins and only the transitions inside its "
                "window count; the candidate's emulation measures the effect",
            },
        }

    def clean_conditions(self, diagnostics: Mapping[str, Any]) -> dict[str, bool]:
        effective = self.effective
        transitions = diagnostics.get("transitions") or {}
        omega = self.reference_omega()
        low, high = transitions.get("min_pyatb_eV"), transitions.get("max_pyatb_eV")
        eta = float(effective.get("eta") or 0.0)
        margin = False
        if low is not None and high is not None and eta > 0:
            margin = min(low - omega[0], omega[-1] - high) >= CLEAN_MARGIN_ETA * eta
        gap = diagnostics.get("min_pair_gap_eV")
        return {
            "scissors_zero": float(effective["scissors"]) == 0.0,
            "gaussian": int(effective["smearing_method"]) == 1,
            "method_1": int(effective["method"]) == 1,
            "pair_gap_above_1e-3_eV": gap is not None and gap > CLEAN_PAIR_GAP_EV,
            "window_margin_5_eta": bool(margin),
        }

    def release_clean_control(self) -> str:
        """``passed``, ``failed``, ``other_release`` or ``not_supplied`` for the supplied clean-control report."""
        report = load_report(self.clean_control_report)
        if report is None:
            return "not_supplied"
        rules = (report.get("semantics") or {}).get("rules") or {}
        eligible = ((rules.get("clean_control") or {}).get("eligible")) is True
        version = (report.get("candidate") or {}).get("queer_version")
        if report.get("step") != "SHIFT_CURRENT" or not eligible:
            return "not_a_clean_control"
        if version != (self.candidate.document.get("provenance") or {}).get("queer_version"):
            return "other_release"
        return "passed" if report.get("verdict") == PASS else "failed"

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
        magnitude = float(np.max(np.abs(values))) if values.size else 0.0
        scale = float(np.max(np.abs(reference))) if reference.size else 0.0
        converged = error <= SELF_CONVERGENCE_RELATIVE * largest
        bounded = magnitude <= MAGNITUDE_FACTOR * scale
        detail = (
            f"|sigma_Q(N) - sigma_Q(2N)|_inf {error:.4g} vs {SELF_CONVERGENCE_RELATIVE} max|sigma_Q(2N)| "
            f"{SELF_CONVERGENCE_RELATIVE * largest:.4g} (grids {convergence.get('grid')} and "
            f"{convergence.get('grid_doubled')}); max|sigma_Q| {magnitude:.4g} vs {MAGNITUDE_FACTOR:g} max|sigma_P| "
            f"{MAGNITUDE_FACTOR * scale:.4g}"
        )
        return converged and bounded, detail

    # ----------------------------------------------------------- compare
    def compare_shift(self) -> None:
        product = self.reference.product or {}
        reference = np.asarray(product["sigma"], float)
        values, emulated, compared = self.compared_tables()
        diagnostics = self.block.get("diagnostics") or {}
        if reference.shape != values.shape or (emulated is not None and emulated.shape != values.shape):
            raise ValueError(f"tables of shape {reference.shape} and {values.shape} cannot be compared")
        domega = float(self.effective["domega"])
        scale = float(np.max(np.abs(reference))) if reference.size else 0.0
        scheme = diagnostics.get("pyatb_scheme") or {}
        edge = np.array(
            [float((scheme.get("edge_allowance_uA_V2") or {}).get(name, 0.0)) for name in SHIFT_COMPONENTS]
        )
        base = AGREEMENT_RELATIVE * scale
        tolerance = base + PRINT_RELATIVE * np.max(np.abs(reference), axis=0, initial=0.0) + edge
        symmetric, facts = symmetry_gate(reference, values, domega, tolerance)
        facts["against"] = compared
        if not symmetric and emulated is not None:
            # PyATB's scheme can reshape a spectrum (a truncation on a coarse step): its view of the same
            # transitions is the fair one when it is what PyATB printed.
            passes, scheme_facts = symmetry_gate(reference, emulated, domega, tolerance)
            if passes:
                symmetric, facts = True, {**scheme_facts, "against": "diagnostics.pyatb_scheme.sigma_uA_V2"}
        self.quantity(
            "symmetry",
            AGREE if symmetric else UNEXPLAINED,
            detail=(
                f"significant components {facts['reference_components']} (PyATB) and {facts['candidate_components']} "
                f"(QUEER's {facts['against']}), differing for {facts['components_differ']}; integral signs differ "
                f"for {facts['integral_sign_differs']}"
            ),
        )
        conditions = self.clean_conditions(diagnostics)
        clean = all(conditions.values())
        release = self.release_clean_control()
        causes = self.causes(diagnostics)
        unbounded = [name for name in UNBOUNDED_CAUSES if causes[name]["active"]]
        convention = self.legacy or self.sign < 0 or int(self.effective["smearing_method"]) == 0
        converged, convergence = None, ""
        failed_release = release == "failed"
        agreed_all = True
        for index, name in enumerate(SHIFT_COMPONENTS):
            direct = float(np.max(np.abs(reference[:, index] - values[:, index]), initial=0.0))
            through = None
            if emulated is not None:
                through = float(np.max(np.abs(reference[:, index] - emulated[:, index]), initial=0.0))
            limit = float(tolerance[index])
            label = f"sigma_{name}"
            if failed_release:
                self.quantity(
                    label, UNEXPLAINED, direct, limit, "the clean control of this QUEER release failed"
                )
                agreed_all = False
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
            if emulated is not None and through is not None and through <= limit:
                gap = float(np.max(np.abs(values[:, index] - emulated[:, index]), initial=0.0))
                self.quantity(
                    label,
                    AFTER_CONVENTION,
                    through,
                    limit,
                    f"PyATB's table meets QUEER's emulation of PyATB's scheme (Hartree scale, window rule, "
                    f"integer bins or truncated Gaussian, constants), which lies {gap:.4g} microA/V^2 from QUEER's "
                    f"{compared}",
                )
                continue
            agreed_all = False
            if clean:
                self.quantity(
                    label,
                    UNEXPLAINED,
                    direct,
                    limit,
                    "a clean-control configuration must agree to 1e-4 max|sigma_P|"
                    + ("" if through is None else f" (QUEER's emulation of PyATB's scheme: {through:.4g})"),
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
            self.quantity(
                label,
                UNEXPLAINED,
                direct,
                limit,
                "no unbounded cause explains it"
                if not unbounded
                else ("the symmetry gate failed" if not symmetric else "the candidate is unconverged"),
            )
        if converged is False:
            self.precondition("self_convergence", False, f"queer_unconverged: {convergence}")
        elif converged:
            self.precondition("self_convergence", True, convergence)
        if clean:
            self.quantity(
                "clean_control",
                AGREE if agreed_all and symmetric else UNEXPLAINED,
                detail="a clean-control configuration: every component must agree to 1e-4 max|sigma_P|",
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
            "causes": causes,
            "unbounded_causes": unbounded,
            "self_convergence": convergence or None,
        }
        self.basis = {
            "relative": AGREEMENT_RELATIVE,
            "scale_uA_V2": scale,
            "base_uA_V2": base,
            "print_relative": PRINT_RELATIVE,
            "edge_allowance_uA_V2": dict(zip(SHIFT_COMPONENTS, edge.tolist())),
            "edge_transitions": scheme.get("edge_transitions"),
            "emulated": emulated is not None,
        }

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
        scale = float(np.max(np.abs(expected))) if expected.size else 0.0
        excess = np.abs(values - expected) - PRINT_RELATIVE * np.abs(expected) - 1e-12 * scale
        omega_error = float(np.max(np.abs(omega - np.asarray(self.block["omega_eV"], float)), initial=0.0))
        ok = float(np.max(excess, initial=0.0)) <= 0 and omega_error <= OMEGA_PRINT_EV
        self.quantity(
            "candidate_file",
            AGREE if ok else UNEXPLAINED,
            detail=f"{SHIFT_CURRENT_FILE} holds export_sign times {name}" + ("" if ok else ": it does not"),
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

    def report(self) -> dict[str, Any]:
        report = super().report()
        report["semantics"] = to_jsonable(
            {
                "profile": "pyatb-openmx-shift-current",
                "key": None,
                "description": (
                    "PyATB's SHIFT_CURRENT on an OpenMX scfout: Hartree 27.211324570273 eV, its SI constants and "
                    "signed cell volume, its Gaussian exp(-(x/eta)^2)/(eta sqrt(pi)) truncated at int(5 eta/domega) "
                    "bins, integer bins without smearing, spin factor 2 for nspin 1 (the spin-up channel only)"
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
