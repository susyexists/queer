"""The shift-current parity rule: a ``QUEER_SHIFT_CURRENT`` run against PyATB's ``SHIFT_CURRENT`` run, judged by a
full independent recomputation of both from the scfout.

QUEER's reported values are its physical estimator, and they are what is judged. The comparison reads no record
of the candidate's but its tables: every number it compares them with, and every tolerance, comes from its own
evaluation of the scfout (:mod:`~queer.compat.generator.pyatb_shift_integrand`, on its own parse of the scfout,
:mod:`~queer.compat.generator.shift_scfout`, so that no defect of QUEER's reader is shared), which computes on the
run's own k grid and photon energies QUEER's physical spectrum and the table PyATB 60fc6b2a prints, each in four
orbital bases (with the first-order rounding of every integrand, their spread is each table's rounding tolerance).
The user decided on 2026-10-18 to replace the comparator's earlier records, floors, partial recomputations and
per-release clean-control report by this recomputation: each comparison verifies itself.
:func:`compare_shift` judges a candidate directory against a PyATB reference directory (``Input``,
``inputs.sha256``, ``Out/input.json``, ``Out/running.log`` and ``Out/Shift_Current/shift_current.dat``):

1. **Preconditions** (``not_comparable`` with the reason): the scfout is given (``scfout=``, ``--scfout``; the
   shift-current comparison recomputes the spectrum from it) and its SHA-256 equals both the reference's
   ``inputs.sha256`` entry and the candidate's provenance; the same settings as PyATB resolves them;
   PyATB's printed photon energies are its ``start + i * domega`` to its five decimals; ``nspin`` matches the
   scfout (4 with spin-orbit coupling, else 1); ``method 1`` and ``smearing_method`` 0 or 1 (``method 0`` and
   the adaptive smearing are not transcribed); the work, k-points x bands^2 x (bands + photon energies / 8),
   within the recompute budget (:data:`RECOMPUTE_BUDGET`, ``recompute_budget=``, ``--recompute-budget``: else too
   large to recompute here, run the comparison as a separate job); a scfout the transcription evaluates.
2. **Determined**, judged first: an entry of a table checks A and B judge (QUEER's physical tables and PyATB's
   literal one) is determined when its rounding tolerance without the edge placements lies within
   :data:`DETERMINED_SHARE` of 1e-4 of its table's largest value; every entry is when the table vanishes (its
   values and rounding below :data:`VANISHING_RELATIVE` of the broadened magnitudes it cancels from, and its
   values within :data:`VANISHING_ROUNDING` times its rounding: it is then judged as zero, not in its values below
   that rounding, and the report names it vanishing); not an entry at an outer edge whose transitions' placements
   are not tested exactly (more than ``OUTER_PLACEMENTS`` k-points there, or a k-point's transitions split across
   the edge by the recomputation's own placement).
3. **QUEER's physical value** (check A; a mismatch is ``differs_unexplained``, a QUEER defect, and fails
   whatever PyATB printed): ``sigma_uA_V2``, without smearing ``binned_uA_V2``, and for a collinear scfout
   ``legacy_single_channel`` (PyATB's spin factor times the spin-up channel; ``sigma_uA_V2`` is both channels)
   each equal the recomputation's physical mode at every determined entry within its rounding tolerance, and their
   sums over the photon energies up to each one within the summed tolerance (a transition on an interior bin edge
   may land on either side of it, never vanish or count twice; on the table's first or last edge each k-point's
   transitions lie inside the table or outside it, and the table must meet one combination of those placements,
   nothing in between); an entry the scfout does not determine counts at most its
   tolerance, so it never fails, and makes the comparison ``not_comparable`` (near-degenerate levels; PyATB's own
   value changing with its eigensolver's basis inside degenerate levels); the candidate's photon energies are
   PyATB's; its own ``shift_current.dat`` holds the lattice's handedness times its table.
4. **PyATB's table** (check B): PyATB's printed table equals the recomputation's PyATB-literal mode within its
   print precision (half a unit in the seventh significant digit of each printed value) and the literal mode's
   rounding tolerance (with the same placements at its window's ends); else ``not_comparable`` ("PyATB's table is
   not reproduced", with the gap): a reader or transcription difference is never a pass.
5. **Classes**, per component, with ``tol = 1e-4 max|sigma_P| + print precision + both rounding tolerances``:
   ``agree`` when QUEER's value in PyATB's layout (the lattice's handedness, the legacy channel of a collinear
   scfout, the bins without smearing) meets PyATB's table (``agree_after_convention`` when that layout is not
   the identity); ``agree_after_convention`` when it does once the recomputation's own conventions step (PyATB's
   SI constants and Hartree constant) is applied; otherwise ``differs_documented`` with the measured gap and its
   attribution, computed by the recomputation switching PyATB's ingredients on one after another (its SI
   constants, Hartree constant, window, rows and truncation, scissored derivative kernels, method-1 integrand),
   each step's largest change reported: since checks A and B hold, the steps telescope from QUEER's value to
   PyATB's table, and every documented difference is explained by construction (the user-approved deviation of
   2026-10-17).

The symmetry of the two tables (:func:`symmetry_gate`) and the candidate's self-convergence estimate are
reported as information, not gates. A literature anchor (:func:`literature_anchor`) runs on its own.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .parity import AFTER_CONVENTION, AGREE, DOCUMENTED, MALFORMED, NOT_APPLICABLE, UNEXPLAINED
from .parity_pyatb import PyatbRun, QueerRun, _PyatbComparison
from .pyatb_files import INPUT_FILENAME, SHIFT_COMPONENTS, SHIFT_CURRENT_FILE, parse_shift_current
from .results import ERR_MISSING_INPUT, sha256_file, to_jsonable

if TYPE_CHECKING:
    from .pyatb_shift_integrand import Table

AGREEMENT_RELATIVE = 1e-4
DETERMINED_SHARE = 0.1
# A table vanishes when its values and rounding lie below this share of the broadened magnitudes it cancels from: the
# calibration scfout's inversion-symmetric response is determined to 1.7e-8 of them where four natural near-degenerate
# k-points condition it (18 x 18 x 1, occ_band 16), so the share leaves a margin above that.
VANISHING_RELATIVE = 1e-6
# ... and when its values also lie within this many times its rounding: zero but for rounding and the scfout's own
# asymmetry. Near-degenerate partners' transitions carry magnitudes far above what their sum cancels from, so the share
# alone would call a nonzero table vanishing.
VANISHING_ROUNDING = 1e3
OMEGA_TOLERANCE_EV = 1e-9
OMEGA_PRINT_EV = 5.01e-6
FILE_RELATIVE = 5.01e-7
SYMMETRY_THRESHOLD = 1e-4
SIGN_SIGNIFICANCE = 1e-4
# The work (k-points x bands^2 x (bands + photon energies / 8), summed over the channels; pyatb_shift_integrand.work) a
# comparison recomputes here: one core does about 3e6 a second on the calibration scfout (30 bands, 51 to 2401 photon
# energies, four bases), so the budget is some 45 minutes on one core.
RECOMPUTE_BUDGET = 8e9
NO_SCFOUT = (
    "the shift-current comparison recomputes the spectrum from the scfout (compare with scfout=, or --scfout)"
)
TOO_LARGE = "too large to recompute here: run the comparison as a separate job"
NOT_REPRODUCED = "PyATB's table is not reproduced"
ANCHOR_PEAK_ETA = 2.0
ANCHOR_HEIGHT_RELATIVE = 0.2


def table(columns: Mapping[str, Any] | None) -> np.ndarray | None:
    """``{component: column}`` as a ``(len(omega), 18)`` array in PyATB's order; None for None."""
    if columns is None:
        return None
    return np.column_stack([np.asarray(columns[name], float) for name in SHIFT_COMPONENTS])


def print_precision(values: Any) -> np.ndarray:
    """Half a unit in the seventh significant digit of each value as PyATB prints it (``%15.6e``)."""
    magnitude = np.abs(np.asarray(values, float))
    exponent = np.floor(np.log10(np.where(magnitude > 0, magnitude, 1.0)))
    return np.where(magnitude > 0, 0.5 * 10.0 ** (exponent - 6), 0.0)


def symmetry_gate(reference: np.ndarray, candidate: np.ndarray, domega: float) -> dict[str, Any]:
    """The components above :data:`SYMMETRY_THRESHOLD` of each table's largest value, and the components whose
    integral over omega has opposite signs in the two tables where both integrals exceed
    :data:`SIGN_SIGNIFICANCE` of their integrated magnitudes: information, not a gate."""
    sets, integrals, magnitudes = [], [], []
    for values in (reference, candidate):
        largest = float(np.max(np.abs(values), initial=0.0))
        column = np.max(np.abs(values), axis=0) if values.size else np.zeros(len(SHIFT_COMPONENTS))
        sets.append(
            {name for name, value in zip(SHIFT_COMPONENTS, column) if value > SYMMETRY_THRESHOLD * largest}
        )
        integrals.append(values.sum(axis=0) * domega)
        magnitudes.append(np.abs(values).sum(axis=0) * domega)
    flipped = [
        name
        for index, name in enumerate(SHIFT_COMPONENTS)
        if name in sets[0] & sets[1]
        and all(abs(i[index]) > SIGN_SIGNIFICANCE * m[index] for i, m in zip(integrals, magnitudes))
        and np.sign(integrals[0][index]) != np.sign(integrals[1][index])
    ]
    return {
        "threshold": SYMMETRY_THRESHOLD,
        "reference_components": sorted(sets[0], key=SHIFT_COMPONENTS.index),
        "candidate_components": sorted(sets[1], key=SHIFT_COMPONENTS.index),
        "components_differ": sorted(sets[0] ^ sets[1], key=SHIFT_COMPONENTS.index),
        "integral_sign_differs": flipped,
        "same": sets[0] == sets[1] and not flipped,
    }


def _largest(values: Any) -> float:
    return float(np.max(np.abs(np.asarray(values, float)), initial=0.0))


def placements(values: Table) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """The placements of a table's transitions on its outer edges, as changes to it and to its rounding tolerance
    (the ordinary rounding of the transitions a placement moves into the table), with how many of their groups (one
    per k-point and edge) each moves to the other side: the recomputation's own first, then every other combination;
    only its own when they are not tested exactly (:attr:`Table.placements_exact`)."""
    zero = np.zeros_like(values.value)
    yield 0, zero, zero
    if not values.placements_exact:
        return
    for chosen in range(1, 1 << len(values.placements)):
        moved = [item for bit, item in enumerate(values.placements) if chosen >> bit & 1]
        yield (
            len(moved),
            np.sum([item.change for item in moved], axis=0),
            np.sum([item.rounding for item in moved], axis=0),
        )


def allowances(values: Table) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A table's tolerance at each entry, the edges' part of it, and the edges' allowance in its sums over the rows.
    With its outer edges' placements tested exactly (:func:`placements`) they have no allowance; otherwise their
    entries are not determined (:meth:`_ShiftComparison.determine`) and keep it."""
    if not values.groups:
        return values.rounding, values.edge, values.edge_cumulative
    if values.placements_exact:
        # The interior edges' part (never below zero by the rounding of the difference).
        inner = np.maximum(values.edge - values.outer_edge, 0.0)
        return values.smooth + inner, inner, values.edge_cumulative
    return values.rounding, values.edge, values.edge_cumulative + values.outer_cumulative


def placement_note(values: Table, moved: int) -> str:
    """What a detail says of a table's outer edges."""
    if not values.groups:
        return ""
    if values.placements_exact:
        return (
            f"; the transitions of {values.groups} k-point(s) on its outer edges tested inside and outside the table "
            f"({1 << values.groups} placements" + (f", {moved} moved" if moved else "") + ")"
        )
    return (
        f"; the transitions of {values.groups} k-point(s) on its outer edges are not placed exactly (their entries "
        "are not determined)"
    )


def vanishing_note(item: Mapping[str, Any]) -> str:
    """What a detail says of a table that vanishes (:meth:`_ShiftComparison.determine`)."""
    return (
        f"vanishing (largest {item['largest_uA_V2']:.3g} of the {item['cancelled_scale_uA_V2']:.3g} microA/V^2 that "
        "cancel in it: judged as zero, not its values below that rounding)"
    )


class _ShiftComparison(_PyatbComparison):
    def __init__(
        self,
        reference: PyatbRun,
        candidate: QueerRun,
        scfout: str | Path | None = None,
        *,
        workers: int | None = None,
        recompute_budget: float | None = None,
    ):
        super().__init__("SHIFT_CURRENT", reference, candidate, scfout)
        self.tier = "shift"
        self.workers = workers
        self.budget = RECOMPUTE_BUDGET if recompute_budget is None else float(recompute_budget)
        self.rules: dict[str, Any] = {}
        self.basis: dict[str, Any] = {}
        self.records: Any = None
        self.undetermined: dict[str, np.ndarray] = {}
        self.determination: dict[str, Any] = {}
        self.run_settings: Any = None
        self.recomputation: Any = None
        self.physical_ok = True
        self.vanishing = 0.0

    # ----------------------------------------------------- preconditions
    @property
    def block(self) -> dict[str, Any]:
        return self.candidate.block("shift_current")

    @property
    def effective(self) -> dict[str, Any]:
        return self.notes.get("parameters", {}).get("function", {})

    @property
    def smearing(self) -> int:
        return int(self.effective["smearing_method"])

    @property
    def domega(self) -> float:
        return float(self.effective["domega"])

    def reference_omega(self) -> np.ndarray:
        start, end = (float(value) for value in self.effective["omega"])
        count = int((end - start) / self.domega) + 1
        return start + np.arange(count) * self.domega

    def check_inputs(self) -> bool:
        """The scfout is given and its SHA-256 equals the reference's ``inputs.sha256`` entry and the candidate's
        provenance."""
        model = self.candidate.block("model")
        name = model.get("scfout") or (self.candidate.params or {}).get("scfout") or "scf.scfout"
        recorded = ((self.candidate.document.get("provenance") or {}).get("inputs") or {}).get(name) or {}
        candidate = recorded.get("sha256") or model.get("scfout_sha256")
        code = (self.candidate.document.get("error") or {}).get("code")
        if not candidate and not self.candidate.ok and code == ERR_MISSING_INPUT:
            return self.precondition(
                "inputs", True, f"the candidate failed before it read a scfout ({code}): nothing to recompute"
            )
        if self.scfout is None:
            return self.precondition("inputs", False, NO_SCFOUT)
        given = sha256_file(self.scfout)
        if given is None:
            return self.precondition("inputs", False, f"the scfout {self.scfout} cannot be read")
        if not self.reference.digests:
            return self.precondition(
                "inputs",
                False,
                "the reference recorded no inputs.sha256: the scfout PyATB read is not identified",
            )
        digest, source = self.reference_scfout_digest()
        problems = []
        if digest != given:
            problems.append(
                f"the reference's {source} names {str(digest)[:16]}..., the scfout {given[:16]}..."
            )
        if candidate != given:
            problems.append(
                f"the candidate's provenance names {str(candidate)[:16]}..., the scfout {given[:16]}..."
            )
        if INPUT_FILENAME in self.reference.digests:
            current = sha256_file(self.reference.directory / INPUT_FILENAME)
            if current != self.reference.digests[INPUT_FILENAME]:
                problems.append(f"the reference {INPUT_FILENAME} changed after the run")
        detail = "; ".join(problems) or (
            f"scfout SHA-256 {given[:16]}... equals the reference's {source} and the candidate's provenance"
        )
        return self.precondition("inputs", not problems, detail)

    def check_fermi_energy(self) -> bool:
        return self.precondition("fermi_energy", True, "occupation by band count (occ_band): E_F is not used")

    def check_omega(self) -> bool:
        expected = self.reference_omega()
        printed = np.asarray((self.reference.product or {}).get("omega_eV") or [], float)
        if printed.shape != expected.shape:
            return self.precondition(
                "omega", False, f"PyATB printed {len(printed)} rows for its {len(expected)} photon energies"
            )
        if np.max(np.abs(printed - expected), initial=0.0) > OMEGA_PRINT_EV:
            return self.precondition(
                "omega", False, "PyATB's printed photon energies are not its start + i * domega"
            )
        return self.precondition(
            "omega",
            True,
            f"{len(expected)} photon energies {expected[0]:.5f}..{expected[-1]:.5f} eV, step {self.domega} eV",
        )

    def check_support(self) -> bool:
        method = int(self.effective.get("method", 1))
        if method != 1:
            return self.precondition(
                "method",
                False,
                f"PyATB's method {method} (get_Inm_ik_direct: every denominator of its eigenvector derivatives "
                "regularized) is not transcribed: only method 1 is recomputed",
            )
        if self.smearing not in (0, 1):
            return self.precondition(
                "smearing", False, "the adaptive smearing (smearing_method 2) is not recomputed"
            )
        return self.precondition("method", True, f"method 1, smearing_method {self.smearing}")

    def check_model(self) -> bool:
        """The scfout parsed by the recomputation's own reader, its spin layout as PyATB's ``nspin`` reads it, and
        the budget."""
        from .pyatb_shift_integrand import ShiftRun, work
        from .shift_scfout import read_records

        if self.scfout is None:
            return self.precondition("inputs", False, NO_SCFOUT)
        try:
            self.records = read_records(self.scfout)
            self.run_settings = ShiftRun.from_block(self.effective)
        except (OSError, ValueError, ArithmeticError, KeyError) as error:
            return self.precondition(
                "scfout", False, f"the scfout cannot be read: {type(error).__name__}: {error}"
            )
        nspin = int(self.notes.get("parameters", {}).get("input", {}).get("nspin") or 0)
        mode = self.records.spin_mode
        expected = self.records.nspin
        if nspin and nspin != expected:
            return self.precondition(
                "spin", False, f"PyATB read nspin {nspin}; the scfout's {mode} layout is nspin {expected}"
            )
        self.precondition(
            "spin",
            True,
            f"the scfout is {mode}: PyATB reads "
            + (
                "its spinor bands (spin factor 1)" if mode == "soc" else "its spin-up channel (spin factor 2)"
            ),
        )
        self.sign = self.records.handedness
        self.legacy = mode == "collinear"
        size = work(self.records, self.run_settings)
        ok = size <= self.budget
        detail = (
            f"k-points x bands^2 x (bands + photon energies / 8) {size:.3g} within the recompute budget "
            f"{self.budget:.3g}"
        )
        return self.precondition(
            "recompute_budget", ok, detail if ok else f"{TOO_LARGE} ({detail.replace('within', 'beyond')})"
        )

    def recompute(self) -> bool:
        from .pyatb_shift_integrand import recompute

        try:
            self.recomputation = recompute(self.records, self.run_settings, workers=self.workers)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            return self.precondition(
                "recompute", False, f"the scfout cannot be recomputed here: {type(error).__name__}: {error}"
            )
        self.basis["recomputation"] = self.recomputation.facts
        return True

    # ------------------------------------------------------------ check A
    def queer_tables(self) -> list[tuple[str, np.ndarray | None, Any]]:
        """QUEER's physical tables and the recomputation's: ``(name, candidate, recomputed)``."""
        physical = self.recomputation.physical
        block = self.block
        legacy = (
            block.get("legacy_single_channel")
            if isinstance(block.get("legacy_single_channel"), dict)
            else None
        )
        pairs = [("sigma_uA_V2", table(block.get("sigma_uA_V2")), physical["sigma"])]
        if self.smearing == 0:
            pairs.append(("binned_uA_V2", table(block.get("binned_uA_V2")), physical["binned"]))
        if self.legacy:
            pairs.append(
                (
                    "legacy_single_channel.sigma_uA_V2",
                    table((legacy or {}).get("sigma_uA_V2")),
                    physical["legacy"],
                )
            )
            if self.smearing == 0:
                pairs.append(
                    (
                        "legacy_single_channel.binned_uA_V2",
                        table((legacy or {}).get("binned_uA_V2")),
                        physical["legacy_binned"],
                    )
                )
        return pairs

    def check_physical(self) -> None:
        """Check A: every table of QUEER's physical value against the recomputation's physical mode, at every entry
        the scfout determines (:meth:`determine`); an entry it does not determine counts at most its tolerance (it
        makes the comparison not comparable, never a failure)."""
        omega = np.asarray(self.block.get("omega_eV") or [], float)
        expected = self.reference_omega()
        if (
            omega.shape != expected.shape
            or np.max(np.abs(omega - expected), initial=0.0) > OMEGA_TOLERANCE_EV
        ):
            self.physical_ok = False
            self.quantity(
                "physical.omega_eV",
                UNEXPLAINED,
                detail=f"QUEER's {len(omega)} photon energies are not PyATB's {len(expected)} start + i * domega",
            )
            return
        for name, values, recomputed in self.queer_tables():
            if values is None or values.shape != recomputed.value.shape:
                self.physical_ok = False
                self.quantity(
                    f"physical.{name}", UNEXPLAINED, detail=f"QUEER's results lack {name} of this run"
                )
                continue
            key = f"physical.{name}"
            undetermined = self.undetermined.get(key, np.zeros(recomputed.value.shape, bool))
            tolerance, edge, edge_cumulative = allowances(recomputed)
            # A transition on an interior discontinuity of the kernel (a bin edge) may land on either side of it, so
            # its entries allow either placement; their sum over the rows does not depend on that choice beyond the
            # edge's own rows: QUEER may move such a transition across its edge, never drop it or count it twice. On
            # an outer edge of the table the transitions of each k-point lie inside the table or outside it: each
            # combination is tried, and nothing in between is allowed.
            found: tuple[Any, ...] | None = None
            for moved, change, rounding in placements(recomputed):
                limit = tolerance + rounding
                difference = values - (recomputed.value + change)
                judged = np.where(undetermined, np.clip(difference, -limit, limit), difference)
                gap = np.abs(judged)
                cumulative = np.abs(np.cumsum(judged, axis=0))
                allowed = (
                    np.cumsum(recomputed.smooth + rounding + np.where(undetermined, edge, 0.0), axis=0)
                    + edge_cumulative
                )
                ok = bool(np.all(gap <= limit))
                conserved = bool(np.all(cumulative <= allowed))
                if found is None or (ok and conserved):
                    found = (moved, limit, difference, gap, cumulative, allowed, ok, conserved)
                if ok and conserved:
                    break
            assert found is not None
            moved, tolerance, difference, gap, cumulative, allowed, ok, conserved = found
            ratio = gap / np.maximum(tolerance, 1e-300)
            row, column = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
            beyond = int(np.count_nonzero(gap > tolerance))
            skipped = int(np.count_nonzero(undetermined & (np.abs(difference) > tolerance)))
            facts = self.determination["tables"].get(key) or {}
            detail = (
                (vanishing_note(facts) + ": " if facts.get("vanishing") else "")
                + f"QUEER's {name} against the recomputed physical value: worst {SHIFT_COMPONENTS[column]} at "
                f"{self.reference_omega()[row]:.5f} eV, |Q - T| {gap[row, column]:.4g} vs its rounding "
                f"{tolerance[row, column]:.4g} microA/V^2 ({beyond} entries beyond"
                + (
                    f"; {int(np.count_nonzero(undetermined))} entries the scfout does not determine, {skipped} of "
                    "them beyond their rounding"
                    if undetermined.any()
                    else ""
                )
                + ")"
                + placement_note(recomputed, moved)
            )
            if ok and not conserved:
                excess = cumulative - allowed
                where, axis = np.unravel_index(int(np.argmax(excess)), excess.shape)
                detail += (
                    f"; but its sum over the photon energies up to {self.reference_omega()[where]:.5f} eV differs by "
                    f"{cumulative[where, axis]:.4g} in {SHIFT_COMPONENTS[axis]}, beyond the {allowed[where, axis]:.4g} "
                    "that rounding and the placement of transitions on a bin edge allow"
                )
            ok = ok and conserved
            self.physical_ok &= ok
            self.quantity(
                f"physical.{name}",
                AGREE if ok else UNEXPLAINED,
                float(gap[row, column]),
                float(tolerance[row, column]),
                detail,
            )

    def compare_candidate_file(self) -> None:
        path = self.candidate.directory / SHIFT_CURRENT_FILE
        try:
            written = parse_shift_current(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            self.physical_ok = False
            self.quantity("candidate_file", UNEXPLAINED, detail=f"{SHIFT_CURRENT_FILE}: {error}")
            return
        name = "binned_uA_V2" if self.smearing == 0 else "sigma_uA_V2"
        expected = table(self.block.get(name))
        values = np.asarray(written["sigma"], float)
        omega = np.asarray(written["omega_eV"], float)
        if expected is None or values.shape != expected.shape or omega.shape != (len(expected),):
            self.physical_ok = False
            self.quantity("candidate_file", UNEXPLAINED, detail=f"{SHIFT_CURRENT_FILE} has another shape")
            return
        expected = self.sign * expected
        excess = np.abs(values - expected) - FILE_RELATIVE * np.abs(expected) - 1e-12 * _largest(expected)
        omega_error = _largest(omega - np.asarray(self.block["omega_eV"], float))
        ok = float(np.max(excess, initial=0.0)) <= 0 and omega_error <= OMEGA_PRINT_EV
        self.physical_ok &= ok
        self.quantity(
            "candidate_file",
            AGREE if ok else UNEXPLAINED,
            detail=f"{SHIFT_CURRENT_FILE} holds the lattice's handedness {self.sign:+d} times {name}"
            + ("" if ok else ": it does not"),
        )

    # ------------------------------------------------------- checks B, C
    def printed(self) -> np.ndarray:
        values = np.asarray((self.reference.product or {}).get("sigma") or [], float)
        if values.shape != (len(self.reference_omega()), len(SHIFT_COMPONENTS)):
            raise ValueError(f"PyATB's table has shape {values.shape}")
        return values

    def compared_value(self) -> tuple[np.ndarray, str, bool]:
        """QUEER's value in PyATB's layout, its name and whether that layout is a convention (not the identity)."""
        legacy = self.block.get("legacy_single_channel") if self.legacy else None
        source = legacy if isinstance(legacy, dict) else self.block
        name = ("legacy_single_channel." if self.legacy else "") + (
            "binned_uA_V2" if self.smearing == 0 else "sigma_uA_V2"
        )
        values = table(source.get("binned_uA_V2" if self.smearing == 0 else "sigma_uA_V2"))
        if values is None:
            raise ValueError(f"QUEER's results lack {name}")
        return self.sign * values, name, self.sign < 0 or self.legacy or self.smearing == 0

    def determine(self) -> bool:
        """Step 2, before check A: which entries of every table checks A and B judge (QUEER's physical tables and
        PyATB's literal one) the scfout determines to a share of the agreement tolerance: an entry whose rounding
        without the edge placements (which check A bounds through the rows' sums) lies within
        :data:`DETERMINED_SHARE` x 1e-4 of its table's largest value, or every entry of a table that vanishes (its
        values and rounding below :data:`VANISHING_RELATIVE` of the broadened magnitudes it cancels from), but not
        an entry whose outer edges' placements are not tested exactly beyond that share. Records the facts; True
        when every entry is determined."""
        recomputation = self.recomputation
        tables = [(f"physical.{name}", recomputed) for name, _, recomputed in self.queer_tables()]
        tables.append(("literal", recomputation.literal))
        facts: dict[str, Any] = {}
        for name, values in tables:
            scale = _largest(values.value)
            smooth = values.smooth
            cancelled = _largest(values.absolute)
            vanishing = (
                cancelled > 0
                and max(scale, _largest(smooth)) <= VANISHING_RELATIVE * cancelled
                and scale <= VANISHING_ROUNDING * _largest(smooth)
            )
            allowed = DETERMINED_SHARE * AGREEMENT_RELATIVE * scale
            rounded = np.zeros(values.value.shape, bool) if vanishing else smooth > allowed
            # Transitions on the outer edges whose placements are not tested exactly keep their allowance, so their
            # entries are not determined.
            unplaced = (
                np.zeros(values.value.shape, bool) if values.placements_exact else values.outer_edge > allowed
            )
            undetermined = rounded | unplaced
            self.undetermined[name] = undetermined
            facts[name] = {
                "largest_uA_V2": scale,
                "rounding_uA_V2": _largest(smooth),
                "allowed_uA_V2": allowed,
                "edge_uA_V2": _largest(values.edge),
                "cancelled_scale_uA_V2": cancelled,
                "basis_spread_uA_V2": _largest(values.spread),
                "vanishing": vanishing,
                "outer_edge_groups": values.groups,
                "outer_edge_split": values.split,
                "outer_edge_exact": values.placements_exact,
                "rounding_undetermined_entries": int(np.count_nonzero(rounded)),
                "unplaced_entries": int(np.count_nonzero(unplaced)),
                "undetermined_entries": int(np.count_nonzero(undetermined)),
                "determined": not undetermined.any(),
            }
        compared, literal = recomputation.compared, recomputation.literal
        both = all(facts[name]["vanishing"] for name, _ in tables)
        cancelled = max(_largest(compared.absolute), _largest(literal.absolute))
        self.vanishing = VANISHING_RELATIVE * cancelled if both else 0.0
        self.determination = {"tables": facts, "vanishing": both}
        self.rules["determined"] = {
            "share": DETERMINED_SHARE,
            "vanishing_relative": VANISHING_RELATIVE,
            "vanishing_rounding": VANISHING_ROUNDING,
            "vanishing": both,
            "vanishing_tables": [name for name, item in facts.items() if item["vanishing"]],
            "outer_placements": recomputation.facts["outer_placements"],
            "tables": facts,
        }
        return all(item["determined"] for item in facts.values())

    def record_determined(self, determined: bool) -> bool:
        """The determination as a precondition (check A has passed): an entry the scfout does not determine makes
        the comparison not comparable."""
        facts = self.determination["tables"]
        if determined:
            vanishing = [
                f"{name} {vanishing_note(item)}" for name, item in facts.items() if item["vanishing"]
            ]
            if self.determination["vanishing"]:
                detail = (
                    f"every table vanishes within its rounding (below {VANISHING_RELATIVE:g} of the magnitudes it "
                    "cancels from): " + "; ".join(vanishing)
                )
            else:
                detail = "; ".join(
                    vanishing
                    + [
                        f"every entry's rounding within {DETERMINED_SHARE:g} x 1e-4 of its table's largest value (at "
                        "most "
                        + ", ".join(
                            f"{name} {item['rounding_uA_V2']:.3g}"
                            for name, item in facts.items()
                            if not item["vanishing"]
                        )
                        + " microA/V^2)"
                    ]
                )
            return self.precondition("determined", True, detail)
        causes = []
        limit = self.rules["determined"]["outer_placements"]
        for name, item in facts.items():
            if item["determined"]:
                continue
            if item["rounding_undetermined_entries"]:
                cause = (
                    "PyATB's own value is bounded only to that rounding inside (near-)degenerate levels, where its "
                    "eigensolver's basis enters it"
                    if name == "literal"
                    else "QUEER's physical value is ill-conditioned at near-degenerate levels"
                )
                causes.append(
                    f"{name}: {item['rounding_undetermined_entries']} entries with rounding up to "
                    f"{item['rounding_uA_V2']:.4g} microA/V^2 beyond {DETERMINED_SHARE:g} x 1e-4 of its largest value "
                    f"= {item['allowed_uA_V2']:.4g} (spread over the bases {item['basis_spread_uA_V2']:.4g}; {cause})"
                )
            if item["unplaced_entries"]:
                causes.append(
                    f"{name}: {item['unplaced_entries']} entries at its outer edges, where the transitions of "
                    f"{item['outer_edge_groups']} k-point(s) are not placed exactly ({item['outer_edge_split']} split "
                    f"across the edge by the recomputation's own placement; at most {limit} are tested in "
                    "each placement)"
                )
        return self.precondition(
            "determined",
            False,
            "the scfout does not determine the values to the agreement tolerance: " + "; ".join(causes),
        )

    def check_pyatb(self, printed: np.ndarray) -> bool:
        """Check B: PyATB's printed table against the recomputation's PyATB-literal mode."""
        literal = self.recomputation.literal
        precision = print_precision(printed)
        tolerance, _, edge_cumulative = allowances(literal)
        # As in check A, a transition on one of PyATB's row edges may land on either side, never vanish; on its
        # window's ends each k-point's transitions lie inside or outside it.
        found: tuple[Any, ...] | None = None
        for moved, change, rounding in placements(literal):
            allowed = precision + tolerance + rounding
            difference = printed - (literal.value + change)
            gap = np.abs(difference)
            excess = gap - allowed
            cumulative = np.abs(np.cumsum(difference, axis=0))
            conserved = bool(
                np.all(
                    cumulative <= np.cumsum(precision + literal.smooth + rounding, axis=0) + edge_cumulative
                )
            )
            ok = bool(np.all(excess <= 0)) and conserved
            if found is None or ok:
                found = (moved, allowed, gap, excess, conserved, ok)
            if ok:
                break
        assert found is not None
        moved, allowed, gap, excess, conserved, ok = found
        row, column = np.unravel_index(int(np.argmax(excess)), excess.shape)
        self.rules["pyatb_reproduced"] = {
            "reproduced": ok,
            "worst_component": SHIFT_COMPONENTS[column],
            "worst_omega_eV": float(self.reference_omega()[row]),
            "gap_uA_V2": float(gap[row, column]),
            "allowed_uA_V2": float(allowed[row, column]),
            "largest_gap_uA_V2": _largest(gap),
            "rows_summed_within_tolerance": conserved,
            "outer_edge_groups_moved": moved,
        }
        detail = (
            f"PyATB's table against the recomputed PyATB-literal table: worst {SHIFT_COMPONENTS[column]} at "
            f"{self.reference_omega()[row]:.5f} eV, gap {gap[row, column]:.4g} vs print precision and rounding "
            f"{allowed[row, column]:.4g} microA/V^2"
            + ("" if conserved else "; its sum over the photon energies differs beyond the summed tolerance")
            + placement_note(literal, moved)
        )
        if not self.physical_ok:
            # QUEER's own value is wrong: that fails whatever PyATB printed.
            self.quantity("pyatb_reproduced", NOT_APPLICABLE, detail=detail)
            return ok
        if ok:
            return self.precondition("pyatb_reproduced", True, detail)
        return self.precondition(
            "pyatb_reproduced",
            False,
            f"{NOT_REPRODUCED}: {detail} ({int(np.count_nonzero(excess > 0))} entries beyond)",
        )

    def classify(self, printed: np.ndarray) -> None:
        """Step 5: each component's class, with the attribution of a documented difference."""
        recomputation = self.recomputation
        values, name, layout = self.compared_value()
        steps = recomputation.path
        queer = steps[0][1]
        conventions = dict(steps)["hartree"]
        literal = recomputation.literal.value
        scale = _largest(printed)
        # Two tables that vanish within their rounding agree as zero: each lies below the vanishing scale.
        tolerance = (
            AGREEMENT_RELATIVE * scale
            + print_precision(printed)
            + recomputation.compared.rounding
            + recomputation.literal.rounding
            + 2 * self.vanishing
        )
        moved = values + (conventions - queer)
        attribution: dict[str, Any] = {}
        for index, component in enumerate(SHIFT_COMPONENTS):
            gap = np.abs(values[:, index] - printed[:, index])
            excess = gap - tolerance[:, index]
            row = int(np.argmax(excess))
            changes = {
                step: _largest(current[:, index] - previous[:, index])
                for (_, previous), (step, current) in zip(steps, steps[1:])
            }
            residual = _largest(
                (values[:, index] - printed[:, index]) - (queer[:, index] - literal[:, index])
            )
            if np.all(excess <= 0):
                kind = AFTER_CONVENTION if layout else AGREE
                detail = f"QUEER's {name} meets PyATB's table" + (
                    ": both vanish and agree as zero, not in their values below the rounding of what cancels in them"
                    if self.vanishing
                    else ""
                )
            elif np.all(np.abs(moved[:, index] - printed[:, index]) <= tolerance[:, index]):
                kind = AFTER_CONVENTION
                detail = (
                    f"QUEER's {name} meets PyATB's table with PyATB's SI and Hartree constants (recomputed change up "
                    f"to {changes['constants'] + changes['hartree']:.4g} microA/V^2)"
                )
            else:
                kind = DOCUMENTED
                shown = ", ".join(f"{step} {value:.3g}" for step, value in changes.items())
                detail = (
                    f"QUEER's {name} differs from PyATB's table by up to {_largest(gap):.4g} microA/V^2; the "
                    f"recomputation's steps from QUEER's value to PyATB's table change it by up to: {shown}; "
                    f"residual {residual:.3g} (QUEER's and PyATB's gaps to the recomputation, within checks A and B)"
                )
            attribution[component] = {
                "class": kind,
                "gap_uA_V2": _largest(gap),
                "steps_uA_V2": changes,
                "residual_uA_V2": residual,
            }
            self.quantity(f"sigma_{component}", kind, float(gap[row]), float(tolerance[row, index]), detail)
        self.rules["attribution"] = attribution
        self.rules["path"] = [step for step, _ in steps]
        if self.legacy:
            physical = (
                self.sign * self.recomputation.physical["binned" if self.smearing == 0 else "sigma"].value
            )
            self.rules["physical_gap_uA_V2"] = _largest(physical - printed)
        self.rules["symmetry"] = symmetry_gate(printed, values, self.domega)

    def self_convergence(self) -> None:
        convergence = (self.block.get("diagnostics") or {}).get("self_convergence")
        if isinstance(convergence, Mapping):
            compared = (
                convergence.get("legacy_single_channel") if self.legacy else convergence.get("physical")
            )
            self.rules["self_convergence"] = {
                "grids": [convergence.get("grid"), convergence.get("grid_doubled")],
                "ratio": (compared or {}).get("ratio"),
                "converged": (compared or {}).get("converged"),
                "role": "information: the comparison recomputes the run's own grid",
            }
        else:
            self.rules["self_convergence"] = None

    # ------------------------------------------------------------- driver
    def run(self) -> None:
        if not self.check_documents() or not self.check_family():
            return
        for check in (self.check_inputs, self.check_parameters):
            if not self.guarded(check):
                return
        if not self.reference.ok or not self.candidate.ok:
            self.compare_failures()
            return
        for check in (self.check_fermi_energy, self.check_omega, self.check_support, self.check_model):
            if not self.guarded(check):
                return
        if not self.recompute():
            return
        # What the scfout determines is judged first: check A judges the determined entries only.
        determined = self.determine()
        try:
            self.compare_candidate_file()
            self.check_physical()
        except MALFORMED as error:
            self.physical_ok = False
            self.quantity("physical", UNEXPLAINED, detail=f"QUEER's results could not be compared: {error}")
            return
        self.self_convergence()
        if not self.physical_ok:
            # Check A failed at a determined entry: a QUEER defect, whatever PyATB printed.
            self.rules["determined"]["role"] = "information: check A failed at an entry the scfout determines"
            try:
                self.check_pyatb(self.printed())
            except MALFORMED:
                pass
            return
        # Both preconditions are recorded: an undetermined comparison still says whether PyATB's table is reproduced.
        determined = self.record_determined(determined)
        try:
            printed = self.printed()
        except MALFORMED as error:
            self.precondition("pyatb_reproduced", False, f"PyATB's table cannot be read: {error}")
            return
        reproduced = self.check_pyatb(printed)
        if not (determined and reproduced):
            return
        try:
            self.classify(printed)
        except MALFORMED as error:
            self.quantity("classify", UNEXPLAINED, detail=f"the documents could not be compared: {error}")

    def report(self) -> dict[str, Any]:
        report = super().report()
        report["semantics"] = to_jsonable(
            {
                "profile": "pyatb-openmx-shift-current",
                "key": None,
                "description": (
                    "PyATB's SHIFT_CURRENT on an OpenMX scfout against QUEER's physical value, both recomputed from "
                    "the scfout by an independent transcription with its own parse of the scfout: QUEER's definition "
                    "(exact SI 2019 constants, CODATA's Hartree and Bohr, the records projected onto their "
                    "translation identities, Gaussian eta/sqrt(2) or a histogram tiling its rows, covariant exact "
                    "degeneracies) and PyATB 60fc6b2a's (its reader's Hartree, Bohr and nspin-1 Hermitization, its "
                    "SI constants, window, int() rows, 5 eta truncation, method-1 cut, spin factor and signed "
                    "volume), each in four orbital bases"
                ),
                "rules": self.rules,
                "conversions_applied": self.sign < 0 or self.legacy or self.smearing_or_none() == 0,
                "berry_sign_applied_to_reference": False,
            }
        )
        report["tolerance_basis"] = to_jsonable(
            {
                "tier": "shift",
                "agreement_relative": AGREEMENT_RELATIVE,
                "print_precision": "half a unit in the seventh significant digit (%15.6e)",
                "determined_share": DETERMINED_SHARE,
                "vanishing_relative": VANISHING_RELATIVE,
                "vanishing_rounding": VANISHING_ROUNDING,
                "recompute_budget": self.budget,
                **self.basis,
            }
        )
        return report

    def smearing_or_none(self) -> int | None:
        try:
            return self.smearing
        except (KeyError, TypeError, ValueError):
            return None


def compare_shift(
    reference: PyatbRun,
    candidate: QueerRun,
    *,
    scfout: str | Path | None = None,
    workers: int | None = None,
    recompute_budget: float | None = None,
) -> dict[str, Any]:
    """The report of a shift-current comparison (:mod:`~queer.compat.generator.parity`'s schema). ``scfout`` is
    required (the candidate's and the reference's, by SHA-256); ``workers`` (None: every CPU) recompute it and
    ``recompute_budget`` caps its work, k-points x bands^2 x (bands + photon energies / 8) (default
    :data:`RECOMPUTE_BUDGET`)."""
    comparison = _ShiftComparison(
        reference, candidate, scfout, workers=workers, recompute_budget=recompute_budget
    )
    comparison.run()
    return comparison.report()


# ---------------------------------------------------------------------------
# Literature anchor
# ---------------------------------------------------------------------------

ANCHOR_REQUIRED = ("component", "peak_eV", "height_uA_V2", "scissors_eV", "eta_eV")


def literature_anchor(candidate: str | Path | Mapping[str, Any], anchor: Mapping[str, Any]) -> dict[str, Any]:
    """QUEER's spectrum against a published peak of one component.

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
