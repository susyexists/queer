"""Parity of a Generator OpenMX step (AHC, Chern number) between a PyATB run and a QUEER run.

:func:`compare` judges a candidate directory (``queer_results.json``,
``queer_params.json``, ``Out/``) against a PyATB reference directory (``Input``,
optional ``inputs.sha256``, ``Out/input.json``, ``Out/running.log``, the product
``Out/AHC/ahc.dat`` or ``Out/Chern_Num/chern_number.dat`` and, for an adaptive
integral, ``Out/AHC/integraton.log``). The report has the schema, classes and
verdicts of :mod:`~queer.compat.generator.parity`.

**Preconditions** (``not_comparable`` with the reason when one fails):

- both runs of the requested family (``AHC`` or ``CHERN_NUMBER``);
- the same scfout: the SHA-256 the reference's ``inputs.sha256`` lists (or of the
  file the caller names, when the reference has none) equals the candidate's
  recorded digest; the reference's ``Input`` still matches its digest;
- the same parameters: the function block and ``nspin``/``hall_convention`` of
  ``Out/input.json`` against the candidate's ``queer_params.json``, both read as
  the runner reads them (text vectors and numbers included), PyATB's defaults
  applied, inapplicable and execution-only keys ignored (``hall_convention`` too
  for the FHS engine and a Chern number, which carry no Hall sign). An echo without
  ``hall_convention`` comes from a PyATB build older than the key, which printed
  its native sign: it is ``pyatb`` (a missing key in an ``Input`` is ``auto``);
- the same Fermi point: ``|E_F^Q / HARTREE_EV - E_F^P / PYATB_HARTREE_EV| <=
  1e-6 / PYATB_HARTREE_EV`` (Hartree; E_F^P from the input echo, or the six
  decimals ``running.log`` prints for ``Auto``); the eV difference itself is
  ``agree_after_convention``;
- the same resolved Hall convention (the FHS engine ignores it);
- an integration that can be compared: an adaptive integral starting from at
  least 512 cells, and both codes' error estimates for a refining one (below).

**Quantities.** PyATB's printed conductivity is compared with QUEER's physical
value times the sign table's factor (-1 for the ``pyatb`` convention:
``agree_after_convention``). A scfout without spin-orbit coupling is compared with
QUEER's ``legacy_single_channel`` (``agree_after_convention``, "legacy
single-channel").

- *Tier A* (Grid with the refinement off in both runs, the same Gamma-origin
  nodes): per component ``|d sigma| <= 2e-7 |sigma| + 5e-7 S/cm`` (the base) plus
  the flagged set's occupation window. 2e-7 covers the two codes' physical
  constants (PyATB's e and hbar differ from CODATA's by up to 1.8e-7); 5e-7 S/cm is
  the rounding of PyATB's six decimals. At a flagged k-point PyATB may differ for
  a known reason: its direct engine zeroes the coefficient derivative of a pair
  closer than 1e-4 eV (its Kubo engine has no cut), and either code may occupy a
  level within ``delta_EF`` of E_F. The candidate records each point's
  contribution under PyATB's cut and the occupation window; PyATB's value must
  then lie within the base plus the window of QUEER's value, or of QUEER's value
  with PyATB's cut applied at the flagged points (``differs_documented``, naming
  the points). Agreement that relies on an allowance larger than the base is
  ``differs_documented``, never ``agree``.
- *Tier B* (a Grid threshold below 1e30, or an adaptive integral): outside Tier
  A, ``differs_documented`` only if ``|d sigma| <= 3 max(e_Q, e_P)``. ``e_Q`` is
  the candidate's ``|sigma(N) - sigma(2N)|`` (``self_convergence``) or its
  cubature error estimate. ``e_P`` is PyATB's own discretisation error: for an
  adaptive integral its last-generation error in ``integraton.log`` converted to
  S/cm; for a refined Grid, for which PyATB writes none, ``|sigma_E(N) -
  sigma_E(2N)|`` of QUEER's emulation of PyATB's refined sum
  (:mod:`~queer.compat.generator.pyatb_emulation`), used only when that emulation
  reproduces PyATB's printed value within Tier A (quantity
  ``pyatb_refined_grid``; otherwise ``differs_unexplained``). Without ``e_Q`` or
  ``e_P`` the run is ``not_comparable`` (run with the refinement off for Tier A,
  or with ``self_convergence``).
- *FHS scans*: the energies must match to the printed decimals and PyATB's table
  is compared with QUEER's FHS Chern numbers in PyATB's Cartesian construction
  (``pyatb_construction_S_cm``; ``agree_after_convention`` where it differs from the
  physical vector, on a non-orthogonal cell). At every energy where no plaquette
  vertex of QUEER's scan changes its occupied rank (a gap) each point must meet
  Tier A. Inside a band the two codes' plaquette links differ (PyATB expands
  ``exp(i dk.r)`` to first order at the link's start, QUEER at its midpoint): a gap
  there is ``differs_documented`` with its measured size only when QUEER's
  emulation of PyATB's links reproduces PyATB's value and QUEER's own midpoint
  links on the same eigenvectors reproduce QUEER's, each within Tier A, and
  ``differs_unexplained`` otherwise. A value that is not finite meets no
  tolerance.
- *Chern numbers*: on a gapped plane where both values lie within 0.05 of an
  integer the integers must be equal; otherwise the floats meet Tier A with a
  1e-9 floor (and Tier B when the integral refines).

The candidate's own product file must also hold what its document says (with
the sign table applied). The sign table's factor is the convention's (-1 for
``pyatb`` with the direct/Kubo engine) times, for the direct/Kubo engine, the
lattice's handedness: PyATB's prefactor carries the signed reciprocal volume.

Runs are judged after the scfout and parameter preconditions. Two failed runs
pass when their failures are equivalent: a PyATB failure is classified from its
logs, Slurm's ``slurm-*.err``/``slurm-*.out`` included (an nspin conflict is
``bad_params``, a memory failure ``oom``). A QUEER refusal at its own documented
limit (``queer_validation_failed``, an integration larger than the job's memory or
a zero Adaptive tolerance) against a completed PyATB run is ``not_comparable``.
The candidate records the scfout's digest before it translates its parameters, so
a run that fails early is still compared on the same inputs.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .openmx_params import (
    COUNTERPARTS,
    FUNCTION_BLOCKS,
    PARAMS_FILENAME,
    STEP_AHC,
    STEP_CHERN,
    canonical_openmx_step,
    effective_parameters,
    limit_refusal,
)
from .parity import (
    AFTER_CONVENTION,
    AGREE,
    CLASSES,
    DOCUMENTED,
    FAIL,
    MALFORMED,
    NOT_APPLICABLE,
    NOT_COMPARABLE,
    PASS,
    PASS_DOCUMENTED,
    REPORT_SCHEMA,
    UNEXPLAINED,
)
from .pyatb_conventions import (
    FERMI_TOLERANCE_HARTREE,
    REFINEMENT_OFF_THRESHOLD,
    fermi_difference_hartree,
    hall_export_sign,
    pyatb_input_fermi_energy,
    resolve_hall_convention,
)
from .pyatb_files import (
    AHC_COMPONENTS,
    AHC_FILE,
    CHERN_FILE,
    CHERN_INTEGRATION_LOG,
    INPUT_ECHO,
    INPUT_FILENAME,
    INPUTS_SHA256_FILENAME,
    INTEGRATION_LOG,
    RUNNING_LOG,
    parse_ahc,
    parse_chern,
    parse_input,
    parse_inputs_sha256,
    parse_integration_log,
    parse_running_log,
)
from .results import (
    CHERN_INTEGER_TOL,
    ERR_MISSING_INPUT,
    ERR_VALIDATION,
    RESULTS_FILENAME,
    sha256_file,
    to_jsonable,
)

FAMILIES: tuple[str, ...] = ("AHC", "CHERN_NUMBER")
TIER_A_RELATIVE = 2e-7
AHC_FLOOR_S_CM = 5e-7
CHERN_FLOOR = 1e-9
TIER_B_FACTOR = 3.0
ADAPTIVE_MIN_CELLS = 512
# PyATB prints FHS energies with six decimals.
ENERGY_PRINT_TOL = 5.01e-7
PARAMETER_RELATIVE = 1e-12
FERMI_SLACK = 1e-9
PRODUCT_FILES = {"AHC": AHC_FILE, "CHERN_NUMBER": CHERN_FILE}
# Signatures of PyATB failures in its logs, and the failure they are.
FAILURE_SIGNATURES: tuple[tuple[str, str], ...] = (
    (r"inconsistent with scfout SpinP_switch|supports nspin = 1 or 4", "bad_params"),
    (r"MemoryError|Cannot allocate memory|[Oo]ut of memory|oom[-_]kill", "oom"),
    (r"No such file or directory|FileNotFoundError", "missing_input"),
)
FAILURE_LOGS = ("pyatb.out", "pyatb.err", "job.err", "job.out", RUNNING_LOG)
# The Generator's batch script leaves PyATB's output and tracebacks in Slurm's files.
FAILURE_LOG_PATTERNS = ("slurm-*.err", "slurm-*.out")


def openmx_family(name: Any) -> str | None:
    """``AHC`` or ``CHERN_NUMBER`` for a QUEER, PyATB or bare OpenMX step name; None otherwise."""
    token = str(name or "").strip().upper()
    for prefix in ("QUEER_", "PYATB_"):
        if token.startswith(prefix):
            token = token[len(prefix) :]
            break
    return token if token in FAMILIES else None


def is_pyatb_directory(directory: str | Path) -> bool:
    """Whether a directory is a PyATB run: an ``Input`` and ``Out/``, no QUEER or PythTB results."""
    root = Path(directory)
    results = ("queer_results.json", "pythtb_results.json")
    return (
        (root / INPUT_FILENAME).is_file()
        and (root / "Out").is_dir()
        and not any((root / name).exists() for name in results)
    )


def pyatb_step(directory: str | Path) -> str:
    """The PyATB step of a PyATB run directory, from the function block of its ``Input``."""
    blocks = parse_input((Path(directory) / INPUT_FILENAME).read_text(encoding="utf-8", errors="replace"))
    present = [step for step, block in FUNCTION_BLOCKS.items() if block in blocks]
    if len(present) != 1:
        raise ValueError(f"{directory}/{INPUT_FILENAME} names no single AHC or CHERN_NUMBER block")
    return COUNTERPARTS[present[0]]


# ---------------------------------------------------------------------------
# Run directories
# ---------------------------------------------------------------------------


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


@dataclass
class PyatbRun:
    """What a PyATB run directory recorded."""

    directory: Path
    problems: list[str] = field(default_factory=list)
    blocks: dict[str, dict[str, Any]] = field(default_factory=dict)
    echo: dict[str, Any] | None = None
    running: dict[str, Any] = field(default_factory=dict)
    product: dict[str, Any] | None = None
    failure: str | None = None
    integration_log: dict[str, Any] | None = None
    digests: dict[str, str] | None = None
    family: str | None = None

    @classmethod
    def load(cls, directory: str | Path) -> PyatbRun:
        run = cls(Path(directory))
        text = _read_text(run.directory / INPUT_FILENAME)
        if text is None:
            run.problems.append(f"reference directory {run.directory} holds no PyATB {INPUT_FILENAME}")
            return run
        try:
            run.blocks = parse_input(text)
        except ValueError as error:
            run.problems.append(f"cannot read the reference {INPUT_FILENAME}: {error}")
            return run
        present = [name for name in FAMILIES if name in run.blocks]
        run.family = present[0] if len(present) == 1 else None
        digests = _read_text(run.directory / INPUTS_SHA256_FILENAME)
        run.digests = parse_inputs_sha256(digests) if digests is not None else None
        try:
            echo = json.loads((run.directory / INPUT_ECHO).read_text(encoding="utf-8"))
            run.echo = echo if isinstance(echo, dict) else None
        except (OSError, ValueError):
            run.echo = None
        log = _read_text(run.directory / RUNNING_LOG)
        run.running = parse_running_log(log) if log is not None else {}
        product = _read_text(run.directory / PRODUCT_FILES.get(run.family or "", AHC_FILE))
        if run.echo is None:
            run.failure = f"{INPUT_ECHO} is missing: PyATB stopped before reading its input"
        elif product is None:
            run.failure = f"{PRODUCT_FILES.get(run.family or '', AHC_FILE)} is missing"
        else:
            try:
                run.product = parse_ahc(product) if run.family == "AHC" else parse_chern(product)
            except ValueError as error:
                run.failure = f"the product cannot be read: {error}"
        for name in (INTEGRATION_LOG, CHERN_INTEGRATION_LOG):
            text = _read_text(run.directory / name)
            if text is not None:
                try:
                    run.integration_log = parse_integration_log(text)
                except ValueError:
                    run.integration_log = None
                break
        return run

    @property
    def ok(self) -> bool:
        return self.failure is None and self.product is not None

    def failure_logs(self) -> list[Path]:
        """The logs a failure is read from: PyATB's own, then Slurm's ``slurm-*.err`` and ``slurm-*.out``."""
        paths = [self.directory / name for name in FAILURE_LOGS]
        for pattern in FAILURE_LOG_PATTERNS:
            paths += sorted(self.directory.glob(pattern))
        return paths

    def failure_code(self) -> str:
        """The failure a failed run's logs show (``bad_params``, ``oom``, ``missing_input``) or ``failed``."""
        for path in self.failure_logs():
            text = _read_text(path)
            if not text:
                continue
            for pattern, code in FAILURE_SIGNATURES:
                if re.search(pattern, text):
                    return code
        return "failed"

    def function_block(self) -> dict[str, Any]:
        if self.echo is not None and isinstance(self.echo.get(self.family or ""), dict):
            return dict(self.echo[self.family or ""])
        return dict(self.blocks.get(self.family or "", {}))

    def input_parameters(self) -> dict[str, Any]:
        echoed = (self.echo or {}).get("INPUT_PARAMETERS")
        if isinstance(echoed, dict):
            return dict(echoed)
        return dict(self.blocks.get("INPUT_PARAMETERS", {}))

    def echoed_inputs(self) -> bool:
        """Whether the INPUT_PARAMETERS come from PyATB's resolved echo ``Out/input.json``."""
        return self.echo is not None and isinstance(self.echo.get("INPUT_PARAMETERS"), dict)

    def missing_convention(self) -> str:
        """What an absent ``hall_convention`` means for this run.

        PyATB resolves the key before it writes ``Out/input.json``, so an echo
        without it comes from a build older than the key, which printed its native
        sign (``pyatb``); a missing key in an ``Input`` file is PyATB's default
        ``auto``.
        """
        return "pyatb" if self.echoed_inputs() else "auto"

    def summary(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "results_file": PRODUCT_FILES.get(self.family or "", None),
            "params_file": INPUT_FILENAME,
            "engine": "pyatb",
            "step_type": f"PYATB_{self.family}" if self.family else None,
            "status": "ok" if self.ok else "failed",
            "core_sha256": None,
            "queer_version": None,
            "queer_record_sha256": None,
            "failure": self.failure,
        }


@dataclass
class QueerRun:
    """A QUEER run directory of an OpenMX step."""

    directory: Path
    problems: list[str] = field(default_factory=list)
    document: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] | None = None

    @classmethod
    def load(cls, directory: str | Path) -> QueerRun:
        run = cls(Path(directory))
        try:
            document = json.loads((run.directory / RESULTS_FILENAME).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            run.problems.append(f"cannot read the candidate {RESULTS_FILENAME}: {error}")
            return run
        if not isinstance(document, dict):
            run.problems.append(f"the candidate {RESULTS_FILENAME} does not hold a JSON object")
            return run
        run.document = document
        try:
            params = json.loads((run.directory / PARAMS_FILENAME).read_text(encoding="utf-8"))
            run.params = params if isinstance(params, dict) else None
        except (OSError, ValueError):
            run.params = None
        return run

    @property
    def ok(self) -> bool:
        return self.document.get("status") == "ok"

    def block(self, name: str) -> dict[str, Any]:
        value = self.document.get(name)
        return value if isinstance(value, dict) else {}

    def summary(self) -> dict[str, Any]:
        provenance = self.document.get("provenance") or {}
        return {
            "directory": str(self.directory),
            "results_file": RESULTS_FILENAME if self.document else None,
            "params_file": PARAMS_FILENAME if self.params is not None else None,
            "engine": self.document.get("engine"),
            "step_type": self.document.get("step_type"),
            "status": self.document.get("status"),
            "core_sha256": None,
            "queer_version": provenance.get("queer_version"),
            "queer_record_sha256": provenance.get("queer_record_sha256"),
        }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isinf(a) or math.isinf(b):
            return a == b
        return abs(a - b) <= PARAMETER_RELATIVE * max(abs(a), abs(b), 1e-300)
    return a == b


def _short(text: str, limit: int = 400) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


class _PyatbComparison:
    def __init__(
        self, family: str, reference: PyatbRun, candidate: QueerRun, scfout: str | Path | None = None
    ):
        self.family = family
        self.reference = reference
        self.candidate = candidate
        self.scfout = scfout
        self.preconditions: list[dict[str, Any]] = []
        self.quantities: list[dict[str, Any]] = []
        self.flagged: dict[str, Any] = {}
        self.sign = 1
        self.tier = "A"
        self.legacy = False
        self.e_q: list[float] | None = None
        self.e_p: list[float] | None = None
        self.emulation: dict[str, Any] | None = None
        self.notes: dict[str, Any] = {}

    # ----------------------------------------------------------- records
    def precondition(self, name: str, ok: bool, detail: str) -> bool:
        self.preconditions.append({"name": name, "ok": bool(ok), "detail": _short(detail)})
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
            {"name": name, "class": kind, "metric": metric, "tolerance": tolerance, "detail": _short(detail)}
        )

    # ----------------------------------------------------- preconditions
    def check_documents(self) -> bool:
        problems = self.reference.problems + self.candidate.problems
        return self.precondition("documents", not problems, "; ".join(problems) or "both runs read")

    def check_family(self) -> bool:
        candidate = openmx_family(self.candidate.document.get("step_type"))
        ok = self.reference.family == self.family and candidate == self.family
        detail = f"reference {self.reference.family}, candidate {candidate}: the {self.family} step"
        return self.precondition("step", ok, detail if ok else detail + " is not both runs' step")

    def capability_refusal(self) -> str | None:
        """Why the candidate refused a run QUEER cannot do (its own limit), or None.

        ``queer_validation_failed`` (a refining Chern integral on a metallic plane,
        an integration that ran out of its budget) and a ``queer_bad_params``
        refusal of an integration larger than the job's memory allows or of a zero
        Adaptive tolerance (:func:`~.openmx_params.limit_refusal`).
        """
        if self.candidate.ok:
            return None
        error = self.candidate.document.get("error") or {}
        code, message = str(error.get("code") or ""), str(error.get("message") or "")
        if code == ERR_VALIDATION or limit_refusal(code, message):
            return f"{code}: {message}"
        return None

    def compare_failures(self) -> None:
        reference = None if self.reference.ok else self.reference.failure_code()
        error = (self.candidate.document.get("error") or {}).get("code")
        candidate = None if self.candidate.ok else str(error or "failed").removeprefix("queer_")
        refusal = self.capability_refusal()
        if reference is None and refusal is not None:
            self.precondition(
                "capability",
                False,
                f"QUEER refused a calculation PyATB completed, at a documented limit of QUEER's ({refusal})",
            )
            return
        status = f"reference {'ok' if reference is None else 'failed (' + reference + ')'}, candidate "
        status += f"{self.candidate.document.get('status')} ({error})"
        if (
            reference is not None
            and candidate is not None
            and reference == candidate
            and reference != "failed"
        ):
            self.quantity("status", AGREE, detail=f"both failed with equivalent codes: {status}")
        else:
            self.quantity(
                "status", UNEXPLAINED, detail=status + f"; {self.reference.failure or ''}".rstrip("; ")
            )

    def reference_scfout_digest(self) -> tuple[str | None, str]:
        digests = self.reference.digests
        if digests:
            route = str(self.reference.blocks.get("INPUT_PARAMETERS", {}).get("scfout_route") or "")
            names = [name for name in digests if name != INPUT_FILENAME]
            for name in (route, Path(route).name):
                if name and name in digests:
                    return digests[name], f"{INPUTS_SHA256_FILENAME} ({name})"
            if len(names) == 1:
                return digests[names[0]], f"{INPUTS_SHA256_FILENAME} ({names[0]})"
        if self.scfout is not None:
            digest = sha256_file(self.scfout)
            return digest, f"the caller's scfout {self.scfout}" if digest else f"unreadable {self.scfout}"
        return None, ""

    def check_inputs(self) -> bool:
        digest, source = self.reference_scfout_digest()
        if digest is None:
            why = source or (
                f"the reference has no {INPUTS_SHA256_FILENAME} and no scfout was named to hash"
                if not self.reference.digests
                else f"{INPUTS_SHA256_FILENAME} lists no scfout"
            )
            return self.precondition("inputs", False, why)
        model = self.candidate.block("model")
        name = model.get("scfout") or (self.candidate.params or {}).get("scfout") or "scf.scfout"
        recorded = ((self.candidate.document.get("provenance") or {}).get("inputs") or {}).get(name) or {}
        candidate = recorded.get("sha256") or model.get("scfout_sha256")
        code = (self.candidate.document.get("error") or {}).get("code")
        if not candidate and not self.candidate.ok and code == ERR_MISSING_INPUT:
            return self.precondition(
                "inputs",
                True,
                f"the candidate failed before it read a scfout ({code}); the reference's scfout SHA-256 "
                f"{digest[:16]}... from {source}",
            )
        ok = bool(candidate) and candidate == digest
        detail = f"scfout SHA-256 {digest[:16]}... from {source}; candidate {str(candidate)[:16]}..."
        if ok and self.reference.digests and INPUT_FILENAME in self.reference.digests:
            current = sha256_file(self.reference.directory / INPUT_FILENAME)
            if current != self.reference.digests[INPUT_FILENAME]:
                return self.precondition(
                    "inputs", False, f"the reference {INPUT_FILENAME} changed after the run"
                )
            detail += f"; the reference {INPUT_FILENAME} matches its recorded digest"
        return self.precondition("inputs", ok, detail if ok else "different scfout: " + detail)

    def check_parameters(self) -> bool:
        params = self.candidate.params
        if params is None:
            return self.precondition("parameters", False, f"the candidate holds no {PARAMS_FILENAME}")
        step = STEP_AHC if self.family == "AHC" else STEP_CHERN
        reference = effective_parameters(
            step,
            self.reference.input_parameters(),
            self.reference.function_block(),
            missing_convention=self.reference.missing_convention(),
        )
        inputs = dict(params.get("input_parameters") or {})
        if inputs.get("nspin") is None and self.candidate.block("model").get("nspin") is not None:
            inputs["nspin"] = self.candidate.block("model")["nspin"]
        candidate = effective_parameters(step, inputs, dict(params.get("parameters") or {}))
        differing = []
        for group in ("function", "input"):
            a, b = reference[group], candidate[group]
            for key in sorted(set(a) | set(b)):
                if not _equal(a.get(key), b.get(key)):
                    differing.append(f"{key}: {a.get(key)!r} vs {b.get(key)!r}")
        self.notes["parameters"] = reference
        detail = (
            "; ".join(differing) if differing else f"{len(reference['function'])} function settings equal"
        )
        return self.precondition("parameters", not differing, detail)

    def reference_fermi(self) -> tuple[float | None, str]:
        inputs = self.reference.input_parameters()
        try:
            value = pyatb_input_fermi_energy(
                inputs.get("fermi_energy"), inputs.get("fermi_energy_unit") or "eV"
            )
        except (TypeError, ValueError):
            value = None
        if value is not None:
            return value, "the input echo"
        if "fermi_energy_eV" in self.reference.running:
            return float(self.reference.running["fermi_energy_eV"]), "running.log (six decimals)"
        return None, ""

    def check_fermi_energy(self) -> bool:
        effective = self.notes.get("parameters", {}).get("function", {})
        if self.family == "CHERN_NUMBER" and int(effective.get("occ_band", -1)) >= 0:
            return self.precondition("fermi_energy", True, "occupation by band count: E_F is not used")
        reference, source = self.reference_fermi()
        candidate = self.candidate.block("model").get("fermi_energy")
        if reference is None or candidate is None:
            return self.precondition("fermi_energy", False, "a Fermi energy is missing")
        difference = fermi_difference_hartree(candidate, reference)
        tolerance = FERMI_TOLERANCE_HARTREE * (1 + FERMI_SLACK)
        ok = difference <= tolerance
        detail = (
            f"E_F^Q {candidate:.9f} eV (CODATA Hartree), E_F^P {reference:.9f} eV (PyATB Hartree, {source}): "
            f"{difference:.3g} Ha apart (allowed {FERMI_TOLERANCE_HARTREE:.3g})"
        )
        if ok:
            self.quantity(
                "fermi_energy",
                AFTER_CONVENTION if candidate != reference else AGREE,
                difference,
                tolerance,
                "the same Fermi point; the eV values differ by the Hartree conversion",
            )
        return self.precondition("fermi_energy", ok, detail)

    def candidate_handedness(self) -> int:
        """+1 for a right-handed scfout lattice, -1 for a left-handed one (the candidate's record)."""
        model = self.candidate.block("model")
        recorded = model.get("lattice_handedness")
        if recorded in (1, -1):
            return int(recorded)
        lattice = model.get("lattice_A")
        if lattice is None:
            return 1
        return 1 if float(np.linalg.det(np.asarray(lattice, float))) > 0 else -1

    def check_convention(self) -> bool:
        if self.family != "AHC":
            return self.precondition("hall_convention", True, "a Chern number carries no Hall sign")
        engine = str(self.notes.get("parameters", {}).get("function", {}).get("engine", "kubo"))
        value = self.reference.input_parameters().get("hall_convention")
        reference = resolve_hall_convention(
            value if value is not None else self.reference.missing_convention()
        )
        note = ""
        if value is None and self.reference.echoed_inputs():
            note = (
                "; the input echo names no hall_convention: a PyATB build older than the key, which "
                "printed its native sign (pyatb)"
            )
        elif value is None:
            note = "; the Input names no hall_convention: PyATB's default auto"
        if engine == "fhs":
            self.sign = 1
            return self.precondition("hall_convention", True, "the FHS engine ignores hall_convention")
        handedness = self.candidate_handedness()
        self.sign = hall_export_sign(reference, engine) * handedness
        convention = self.candidate.block("ahc").get("convention") or {}
        candidate = convention.get("resolved")
        ok = candidate == reference
        detail = f"reference {reference}, candidate {candidate}; sign table factor {self.sign:+d}" + note
        if handedness < 0:
            detail += " (a left-handed lattice: PyATB's signed reciprocal volume flips its sign)"
        return self.precondition("hall_convention", ok, detail)

    def check_integration(self) -> bool:
        effective = self.notes.get("parameters", {}).get("function", {})
        if self.family == "AHC" and effective.get("engine") == "fhs":
            self.tier = "FHS"
            return self.precondition("integration", True, "FHS plaquette scan on the Gamma-origin grid")
        mode = effective.get("integrate_mode", "Grid")
        integration = self.candidate.block("integration")
        if (
            mode == "Grid"
            and float(effective.get("adaptive_grid_threshold", 0.0)) >= REFINEMENT_OFF_THRESHOLD
        ):
            refined = self.reference.running.get("refined_kpoints")
            ok = integration.get("refinement") == "off" and refined in (None, 0)
            self.tier = "A"
            detail = (
                f"Grid {effective.get('integrate_grid')} without refinement in both runs "
                f"(reference refined {refined} k-points; candidate refinement {integration.get('refinement')})"
            )
            return self.precondition("integration", ok, detail)
        self.tier = "B"
        if mode == "Adaptive":
            cells = int(np.prod(effective.get("initial_grid", [1, 1, 1])))
            if cells < ADAPTIVE_MIN_CELLS:
                return self.precondition(
                    "integration",
                    False,
                    f"PyATB's adaptive integral starts from {cells} cells (fewer than {ADAPTIVE_MIN_CELLS}): "
                    "it can miss sharp curvature",
                )
        self.e_q = self.candidate_error_estimate(integration)
        if mode == "Grid":
            emulation = self.emulation = self.refined_grid_emulation()
            missing = []
            if self.e_q is None:
                missing.append("QUEER's self-convergence estimate e_Q")
            if emulation is None:
                missing.append(
                    "PyATB's own refinement error e_P (PyATB writes none for a Grid; QUEER emulates its "
                    "refined sum on the grid and its double)"
                )
            if missing or emulation is None:
                return self.precondition(
                    "integration",
                    False,
                    f"Grid integration with refinement (Tier B) needs {' and '.join(missing)}: run with the "
                    "refinement off (adaptive_grid_threshold 1e30) for Tier A, or set self_convergence",
                )
            unit = "_S_cm" if self.family == "AHC" else ""
            self.e_p = [float(value) for value in emulation[f"e_P{unit}"]]
            refined = self.reference.running.get("refined_kpoints")
            detail = (
                f"Grid integration with refinement (Tier B): e_Q {self.e_q}; e_P {self.e_p} from QUEER's "
                f"emulation of PyATB's refined sum ({emulation.get('flagged')} nodes refined; PyATB's "
                f"running.log {refined})"
            )
            return self.precondition("integration", True, detail)
        self.e_p = self.reference_error_estimate()
        if self.e_q is None or self.e_p is None:
            missing = [] if self.e_q is not None else ["QUEER's error estimate e_Q"]
            if self.e_p is None:
                missing.append(f"PyATB's last-generation error e_P ({INTEGRATION_LOG})")
            return self.precondition(
                "integration",
                False,
                f"{mode} integration (Tier B) needs {' and '.join(missing)}",
            )
        detail = f"{mode} integration (Tier B): e_Q {self.e_q}, e_P {self.e_p} from {INTEGRATION_LOG}"
        return self.precondition("integration", True, detail)

    def refined_grid_emulation(self) -> dict[str, Any] | None:
        """QUEER's emulation of PyATB's refined Grid sum, as the candidate recorded it."""
        block = self.candidate.block("ahc" if self.family == "AHC" else "chern")
        emulation = (block.get("pyatb_emulation") or {}).get("refined_grid")
        return emulation if isinstance(emulation, dict) else None

    def candidate_error_estimate(self, integration: Mapping[str, Any]) -> list[float] | None:
        estimates = []
        source = self.compared_block()
        convergence = (source.get("self_convergence") if self.legacy else None) or integration.get(
            "self_convergence"
        )
        if isinstance(convergence, Mapping) and convergence.get("e_Q") is not None:
            value = convergence["e_Q"]
            estimates.append([float(v) for v in value] if isinstance(value, list) else [float(value)])
        error = (source.get("error_estimate") if self.legacy else None) or integration.get("error_estimate")
        if integration.get("refinement") == "cubature" and error is not None:
            estimates.append([float(error)])
        if not estimates:
            return None
        width = max(len(item) for item in estimates)
        return [max(item[i] if len(item) == width else item[0] for item in estimates) for i in range(width)]

    def reference_error_estimate(self) -> list[float] | None:
        log = self.reference.integration_log
        if log is None:
            return None
        errors = [abs(float(value)) for value in log["error"]]
        if self.family == "AHC":
            scale = float(self.candidate.block("ahc").get("raw_curvature_to_S_cm") or 0.0)
            return [value * scale for value in errors] if scale > 0 else None
        return None

    # ----------------------------------------------------------- quantities
    def compared_block(self) -> dict[str, Any]:
        """The candidate block PyATB's numbers correspond to: physical, or the single spin channel."""
        name = "ahc" if self.family == "AHC" else "chern"
        block = self.candidate.block(name)
        legacy = block.get("legacy_single_channel")
        if self.candidate.block("model").get("spin_mode") not in (None, "soc") and isinstance(legacy, dict):
            self.legacy = True
            return legacy
        self.legacy = False
        return block

    def tier_b(self, component: int) -> float | None:
        values = [item[component if len(item) > 1 else 0] for item in (self.e_q, self.e_p) if item]
        return TIER_B_FACTOR * max(values) if values else None

    def judge(
        self,
        name: str,
        difference: float,
        tier_a: float,
        component: int,
        detail: str,
        *,
        base: float | None = None,
        allowance: float = 0.0,
        regularised: float | None = None,
        reason: str = "",
    ) -> None:
        """Class one compared value: Tier A, else Tier B within ``3 max(e_Q, e_P)``, else unexplained.

        ``tier_a`` is ``base`` plus the occupation window of the flagged set.
        ``regularised`` is the gap to QUEER's value with PyATB's own cut at the
        flagged points: within ``tier_a`` it is documented (``reason`` names the
        points and the rule). Agreement that needs an allowance (``allowance``,
        the flagged set's exact one) larger than ``base`` is documented too.
        """
        convention = self.sign < 0 or self.legacy
        if difference <= tier_a:
            if base is not None and allowance > base:
                self.quantity(name, DOCUMENTED, difference, tier_a, f"{detail}; {reason}".rstrip("; "))
            else:
                self.quantity(name, AFTER_CONVENTION if convention else AGREE, difference, tier_a, detail)
            return
        if regularised is not None and regularised <= tier_a:
            self.quantity(
                name,
                DOCUMENTED,
                regularised,
                tier_a,
                f"{detail}; QUEER's value with PyATB's cut at the flagged k-points lies {regularised:.3g} "
                f"from PyATB's; {reason}".rstrip("; "),
            )
            return
        bound = self.tier_b(component) if self.tier == "B" else None
        if bound is not None and difference <= bound:
            self.quantity(
                name,
                DOCUMENTED,
                difference,
                bound,
                f"{detail}; within 3 max(e_Q, e_P) of the refined integrals (Tier B)",
            )
            return
        self.quantity(name, UNEXPLAINED, difference, bound if bound is not None else tier_a, detail)

    def compare_ahc(self) -> None:
        block = self.compared_block()
        product = self.reference.product or {}
        sigma = block.get("sigma_S_cm") or {}
        if self.tier == "FHS":
            self.compare_scan(product, sigma)
            return
        if product.get("format") != "vector":
            self.quantity("sigma", UNEXPLAINED, detail="the reference holds an FHS table, not one AHC vector")
            return
        flagged = block.get("flagged") or {}
        zero = [0.0, 0.0, 0.0]
        allowance = [float(value) for value in flagged.get("allowance_S_cm") or zero]
        shift = [float(value) for value in flagged.get("pair_shift_S_cm") or zero]
        flips = [float(value) for value in flagged.get("flip_allowance_S_cm") or zero]
        reason = self.regularisation(block)
        if self.emulation is not None:
            self.compare_emulation([float(value) for value in product["sigma"]])
        for index, component in enumerate(AHC_COMPONENTS):
            physical = float(sigma[component][0])
            expected = self.sign * physical
            printed = float(product["sigma"][index])
            base = TIER_A_RELATIVE * abs(physical) + AHC_FLOOR_S_CM
            detail = (
                f"PyATB {printed:.6f} vs {self.sign:+d} x QUEER {physical:.9f} S/cm"
                + (" (legacy single-channel)" if self.legacy else "")
                + f"; flagged allowance {allowance[index]:.3g}"
            )
            regularised = abs(printed - self.sign * (physical - shift[index])) if shift[index] else None
            self.judge(
                f"sigma_{component}",
                abs(printed - expected),
                base + flips[index],
                index,
                detail,
                base=base,
                allowance=allowance[index],
                regularised=regularised,
                reason=reason,
            )
        if flagged:
            self.flagged["ahc"] = {
                **{key: flagged.get(key) for key in ("count", "points", "delta_ef_eV", "pair_gap_eV")},
                "allowance_S_cm": allowance,
                "sum_abs_S_cm": flagged.get("sum_abs_S_cm"),
                "summary_line": (
                    f"{flagged.get('count')} of {flagged.get('points')} k-points flagged (delta_EF "
                    f"{float(flagged.get('delta_ef_eV') or 0):.3g} eV, pair gap {flagged.get('pair_gap_eV')} eV); "
                    f"allowance {allowance} S/cm"
                ),
            }

    def regularisation(self, block: Mapping[str, Any]) -> str:
        """Why an agreement needs the flagged allowance: the flagged k-points and PyATB's rule at them."""
        entries = [entry for entry in block.get("per_k_flagged") or [] if isinstance(entry, Mapping)]
        if not entries:
            return ""
        reasons = {
            "level_at_fermi": "a level within delta_EF of E_F, which either code may occupy",
            "pyatb_pair_cut": "an occupied-empty pair closer than 1e-4 eV, whose coefficient derivative "
            "PyATB's direct engine sets to zero",
        }
        found = sorted({reason for entry in entries for reason in entry.get("reasons") or []})
        points = ", ".join(str([round(float(x), 6) for x in entry["k"]]) for entry in entries[:5])
        more = f" and {len(entries) - 5} more" if len(entries) > 5 else ""
        why = "; ".join(reasons.get(reason, reason) for reason in found)
        return f"agreement relies on the allowance of the flagged k-points {points}{more} ({why})"

    def compare_emulation(self, printed: list[float]) -> None:
        """How well QUEER's emulation of PyATB's refined Grid sum reproduces PyATB's printed value.

        Within Tier A plus the emulation's own allowances (threshold ties and flagged
        points) the emulated ``e_P`` bounds PyATB's discretisation error; otherwise
        the emulation does not reproduce PyATB, ``e_P`` is not used, and the gap is
        unexplained.
        """
        emulation = self.emulation or {}
        unit = "_S_cm" if self.family == "AHC" else ""
        emulated = [float(value) for value in emulation[f"printed{unit}"]]
        allowances = [
            float(a) + float(b)
            for a, b in zip(
                emulation[f"tie_allowance{unit}"], emulation[f"flag_allowance{unit}"], strict=True
            )
        ]
        floor = AHC_FLOOR_S_CM if self.family == "AHC" else CHERN_FLOOR
        differences = [abs(p - e) for p, e in zip(printed, emulated, strict=True)]
        tolerances = [
            TIER_A_RELATIVE * abs(e) + floor + allowance
            for e, allowance in zip(emulated, allowances, strict=True)
        ]
        worst = max(range(len(differences)), key=lambda i: differences[i] / tolerances[i])
        ok = all(
            difference <= tolerance for difference, tolerance in zip(differences, tolerances, strict=True)
        )
        refined = self.reference.running.get("refined_kpoints")
        detail = (
            f"PyATB {printed} vs QUEER's emulation of PyATB's refined Grid sum {emulated} "
            f"(|d| {differences[worst]:.3g}, fidelity); nodes refined: emulation {emulation.get('flagged')}, "
            f"PyATB's running.log {refined}; e_P {self.e_p}"
        )
        if ok:
            self.quantity("pyatb_refined_grid", AGREE, differences[worst], tolerances[worst], detail)
        else:
            self.e_p = None
            self.quantity(
                "pyatb_refined_grid",
                UNEXPLAINED,
                differences[worst],
                tolerances[worst],
                detail + "; the emulation does not reproduce PyATB, so its e_P is not used",
            )

    def rank_variation(self, count: int) -> np.ndarray:
        """Per energy, whether QUEER's FHS saw the occupied rank vary across plaquette vertices (a band)."""
        if self.legacy:
            facts = (self.compared_block().get("fhs") or {}).get("metallic_vertex_rank_variation")
        else:
            facts = (self.candidate.block("integration").get("fhs") or {}).get(
                "metallic_vertex_rank_variation"
            )
        flags = np.zeros(count, bool) if facts is None else np.asarray(facts, bool)
        if flags.shape != (count,):
            raise ValueError("the candidate's FHS rank-variation flags do not match its energies")
        return flags

    def compare_scan(self, product: Mapping[str, Any], sigma: Mapping[str, Any]) -> None:
        """PyATB's FHS table against QUEER's, energy by energy.

        Where no plaquette vertex changes its occupied rank (a gap) the two codes'
        plaquette sums are the same integer and every energy must meet Tier A.
        Inside a band the link schemes differ: there the gap is documented with its
        measured size when QUEER's emulation of PyATB's links reproduces PyATB's
        value and QUEER's own links reproduce QUEER's, each within Tier A, and
        unexplained otherwise. A value that is not finite is beyond any tolerance.
        """
        if product.get("format") != "scan":
            self.quantity("sigma", UNEXPLAINED, detail="the reference holds one AHC vector, not an FHS table")
            return
        offsets = self.candidate.block("ahc").get("energy_offsets_eV") or []
        energies = np.asarray(product["energy_eV"], float)
        if len(offsets) != len(energies) or np.max(np.abs(energies - np.asarray(offsets)), initial=0) > (
            ENERGY_PRINT_TOL
        ):
            self.quantity("energies", UNEXPLAINED, detail="the FHS energy grids differ")
            return
        block = self.compared_block()
        construction = block.get("pyatb_construction_S_cm") or sigma
        varying = self.rank_variation(len(energies))
        emulation = (self.candidate.block("ahc").get("pyatb_emulation") or {}).get("fhs_links")
        reference = np.asarray(product["sigma"], float)
        for index, component in enumerate(AHC_COMPONENTS):
            values = np.asarray(construction[component], float)
            physical = np.asarray(sigma[component], float)
            printed = reference[:, index]
            difference = np.abs(printed - values)
            tolerance = TIER_A_RELATIVE * np.abs(values) + AHC_FLOOR_S_CM
            beyond = ~(difference <= tolerance)
            finite = np.where(np.isfinite(difference), difference, np.inf)
            worst = int(np.argmax(finite))
            largest = float(finite[worst]) if len(finite) else 0.0
            detail = (
                f"largest gap {largest:.6g} S/cm at E-E_F {energies[worst]:+.6f} eV; {int(beyond.sum())} of "
                f"{len(values)} energies beyond Tier A ({int((beyond & ~varying).sum())} in a gap, "
                f"{int((beyond & varying).sum())} with a varying occupied rank)"
            )
            if (
                np.isfinite(values).all()
                and np.isfinite(printed).all()
                and np.std(values) * np.std(printed) > 0
            ):
                detail += f"; correlation {float(np.corrcoef(values, printed)[0, 1]):.6f}"
            converted = self.legacy or not np.allclose(values, physical, rtol=1e-12, atol=1e-12)
            if converted and not self.legacy:
                detail += "; PyATB's Cartesian construction of the plane Chern numbers (non-orthogonal cell)"
            name = f"sigma_{component}(E)"
            tolerance_worst = float(tolerance[worst]) if np.isfinite(tolerance[worst]) else None
            if (beyond & ~varying).any():
                self.quantity(
                    name,
                    UNEXPLAINED,
                    largest,
                    tolerance_worst,
                    detail + "; in a gap both codes sum the same integer plaquette fluxes",
                )
                continue
            if not beyond.any():
                self.quantity(
                    name, AFTER_CONVENTION if converted else AGREE, largest, tolerance_worst, detail
                )
                continue
            self.quantity(name, *self.explain_links(emulation, component, printed, values, varying, detail))

    def explain_links(
        self,
        emulation: Mapping[str, Any] | None,
        component: str,
        printed: np.ndarray,
        values: np.ndarray,
        varying: np.ndarray,
        detail: str,
    ) -> tuple[str, float, float | None, str]:
        """The class of a band gap: documented when both link schemes are reproduced at every varying energy."""
        difference = np.abs(printed - values)
        largest = float(np.max(np.where(np.isfinite(difference), difference, np.inf)[varying], initial=0.0))
        if emulation is None:
            return UNEXPLAINED, largest, None, detail + "; no emulation of PyATB's FHS links was recorded"
        emulated = np.asarray(emulation["printed_S_cm"][component], float)
        own = np.asarray(emulation["queer_links_S_cm"][component], float)
        fidelity = np.abs(printed - emulated)
        consistency = np.abs(values - own)
        missed = varying & ~(fidelity <= TIER_A_RELATIVE * np.abs(emulated) + AHC_FLOOR_S_CM)
        strayed = varying & ~(consistency <= TIER_A_RELATIVE * np.abs(own) + AHC_FLOOR_S_CM)
        scheme = np.abs(emulated - own)[varying]
        if missed.any() or strayed.any():
            why = []
            if missed.any():
                residual = float(np.max(np.where(np.isfinite(fidelity), fidelity, np.inf)[missed]))
                why.append(
                    f"QUEER's emulation of PyATB's links misses PyATB's value at {int(missed.sum())} energies "
                    f"(by up to {residual:.3g} S/cm)"
                )
            if strayed.any():
                residual = float(np.max(np.where(np.isfinite(consistency), consistency, np.inf)[strayed]))
                why.append(
                    f"QUEER's value is not its own midpoint links' at {int(strayed.sum())} energies "
                    f"(by up to {residual:.3g} S/cm)"
                )
            return UNEXPLAINED, largest, None, detail + "; " + "; ".join(why)
        explained = (
            f"; inside bands ({int(varying.sum())} energies with a varying occupied rank) PyATB's links, exp(i dk.r) "
            f"to first order at the link's start, and QUEER's midpoint links differ by up to "
            f"{float(np.max(scheme, initial=0.0)):.6g} S/cm on the same eigenvectors; QUEER's emulation "
            f"reproduces PyATB's values within Tier A (up to {float(np.max(fidelity[varying], initial=0.0)):.3g} S/cm)"
        )
        return DOCUMENTED, largest, None, detail + explained

    def compare_chern(self) -> None:
        block = self.compared_block()
        value = block.get("value")
        printed = float((self.reference.product or {}).get("value", math.nan))
        if value is None:
            self.quantity("chern_number", UNEXPLAINED, detail="the candidate recorded no Chern number")
            return
        value = float(value)
        gapped = bool(block.get("gapped", self.candidate.block("chern").get("gapped")))
        near = (
            abs(printed - round(printed)) <= CHERN_INTEGER_TOL
            and abs(value - round(value)) <= CHERN_INTEGER_TOL
        )
        convention = self.legacy
        if gapped and near:
            ok = round(printed) == round(value)
            kind = (AFTER_CONVENTION if convention else AGREE) if ok else UNEXPLAINED
            detail = (
                f"integers {round(printed)} (PyATB {printed!r}) vs {round(value)} (QUEER {value!r}); "
                f"float difference {abs(printed - value):.3g}"
            )
            self.quantity("chern_number", kind, float(abs(round(printed) - round(value))), 0.0, detail)
        else:
            flagged = block.get("flagged") or {}
            allowance = float(flagged.get("allowance") or 0.0)
            shift = float(flagged.get("pair_shift") or 0.0)
            if self.emulation is not None:
                self.compare_emulation([printed])
            base = TIER_A_RELATIVE * abs(value) + CHERN_FLOOR
            detail = (
                f"PyATB {printed!r} vs QUEER {value!r} ({'gapped' if gapped else 'metallic'} plane); "
                f"flagged allowance {allowance:.3g}"
            )
            self.judge(
                "chern_number",
                abs(printed - value),
                base + float(flagged.get("flip_allowance") or 0.0),
                0,
                detail,
                base=base,
                allowance=allowance,
                regularised=abs(printed - (value - shift)) if shift else None,
                reason=self.regularisation(block),
            )
        flagged = block.get("flagged")
        if isinstance(flagged, dict):
            self.flagged["chern"] = {
                **{
                    key: flagged.get(key)
                    for key in ("count", "points", "delta_ef_eV", "pair_gap_eV", "allowance", "sum_abs")
                },
                "summary_line": (
                    f"{flagged.get('count')} of {flagged.get('points')} k-points flagged; allowance "
                    f"{float(flagged.get('allowance') or 0):.3g}"
                ),
            }

    def compare_candidate_file(self) -> None:
        """The candidate's product file holds what its document says, with the sign table applied."""
        path = self.candidate.directory / PRODUCT_FILES[self.family]
        try:
            text = path.read_text(encoding="utf-8")
            product = parse_ahc(text) if self.family == "AHC" else parse_chern(text)
        except (OSError, ValueError) as error:
            self.quantity("candidate_file", UNEXPLAINED, detail=f"{PRODUCT_FILES[self.family]}: {error}")
            return
        if self.family == "CHERN_NUMBER":
            value = float(self.candidate.block("chern").get("value", math.nan))
            ok = product["value"] == value
            self.quantity(
                "candidate_file",
                AGREE if ok else UNEXPLAINED,
                abs(product["value"] - value),
                0.0,
                f"{CHERN_FILE} holds {product['value']!r}",
            )
            return
        block = self.candidate.block("ahc")
        sigma = (block.get("pyatb_construction_S_cm") if block.get("engine") == "fhs" else None) or (
            block.get("sigma_S_cm") or {}
        )
        physical = np.array([sigma[name] for name in AHC_COMPONENTS], float).T
        written = np.atleast_2d(np.asarray(product["sigma"], float))
        if written.shape != physical.shape:
            self.quantity("candidate_file", UNEXPLAINED, detail=f"{AHC_FILE} holds {written.shape[0]} rows")
            return
        difference = float(np.max(np.abs(written - self.sign * physical)))
        tolerance = AHC_FLOOR_S_CM + 1e-12 * float(np.max(np.abs(physical), initial=0))
        kind = AGREE if difference <= tolerance else UNEXPLAINED
        self.quantity(
            "candidate_file",
            kind,
            difference,
            tolerance,
            f"{AHC_FILE} holds {self.sign:+d} x the "
            + ("FHS construction" if block.get("engine") == "fhs" else "physical value")
            + " (sign table)",
        )

    # ------------------------------------------------------------- driver
    def guarded(self, check: Any) -> bool:
        try:
            return bool(check())
        except MALFORMED as error:
            detail = f"the documents could not be checked: {type(error).__name__}: {error}"
            return self.precondition(check.__name__.removeprefix("check_"), False, detail)

    def run(self) -> None:
        if not self.check_documents() or not self.check_family():
            return
        # The same inputs and settings first: a failure is only judged between comparable runs.
        for check in (self.check_inputs, self.check_parameters):
            if not self.guarded(check):
                return
        if not self.reference.ok or not self.candidate.ok:
            self.compare_failures()
            return
        for check in (self.check_fermi_energy, self.check_convention, self.check_integration):
            if not self.guarded(check):
                return
        stage = self.compare_ahc if self.family == "AHC" else self.compare_chern
        for step in (stage, self.compare_candidate_file):
            try:
                step()
            except MALFORMED as error:
                detail = f"the documents could not be compared: {type(error).__name__}: {error}"
                self.quantity(step.__name__, UNEXPLAINED, detail=detail)

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
                    "profile": "pyatb-openmx",
                    "key": None,
                    "description": (
                        "PyATB on an OpenMX scfout: Hartree 27.211324570273 eV, hall_convention-resolved Hall "
                        "sign, Gamma-origin Grid nodes, spin-up Hamiltonian only without spin-orbit coupling"
                    ),
                    "rules": {
                        "tier": self.tier,
                        "sign_table_factor": self.sign,
                        "legacy_single_channel": self.legacy,
                    },
                    "conversions_applied": self.sign < 0 or self.legacy,
                    "berry_sign_applied_to_reference": False,
                },
                "inputs_verified_by_fingerprint": False,
                "tolerance_basis": {
                    "tier": self.tier,
                    "relative": TIER_A_RELATIVE,
                    "floor_S_cm" if self.family == "AHC" else "floor": (
                        AHC_FLOOR_S_CM if self.family == "AHC" else CHERN_FLOOR
                    ),
                    "e_Q": self.e_q,
                    "e_P": self.e_p,
                    "e_P_source": (
                        "QUEER's emulation of PyATB's refined Grid sum"
                        if self.emulation is not None
                        else (INTEGRATION_LOG if self.e_p is not None else None)
                    ),
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


def compare(
    step_type: str,
    reference_dir: str | Path,
    candidate_dir: str | Path,
    *,
    scfout: str | Path | None = None,
) -> dict[str, Any]:
    """Compare a QUEER run of an OpenMX step with the PyATB run of its counterpart.

    ``step_type`` names the family (``QUEER_AHC``, ``PYATB_AHC`` or ``AHC``; likewise
    CHERN_NUMBER); an unknown step raises ``ValueError``. ``scfout`` names the
    scfout to hash when the reference holds no ``inputs.sha256``. Returns the
    report of :func:`~queer.compat.generator.parity.compare`'s schema.
    """
    family = openmx_family(step_type)
    if family is None:
        raise ValueError(f"unknown step {step_type!r}: compare AHC or CHERN_NUMBER (QUEER_ or PYATB_)")
    canonical_openmx_step(f"QUEER_{family}")
    comparison = _PyatbComparison(family, PyatbRun.load(reference_dir), QueerRun.load(candidate_dir), scfout)
    comparison.run()
    return comparison.report()


def reference_values(directory: str | Path) -> dict[str, Any]:
    """The facts a PyATB run directory recorded, as plain JSON (for reports and inspection)."""
    run = PyatbRun.load(directory)
    return to_jsonable(
        {
            "family": run.family,
            "ok": run.ok,
            "failure": run.failure,
            "product": run.product,
            "running": run.running,
            "digests": run.digests,
            "integration_log": run.integration_log,
            "input_parameters": run.input_parameters() if run.blocks else None,
        }
    )
