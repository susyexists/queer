"""The results contract of QUEER's Generator Wannier steps.

``queer_results.json`` follows the schema of the Generator PythTB engine's
``pythtb_results.json`` (version 1): the same blocks, keys, nesting and units,
written on success and on failure. QUEER adds ``engine = "queer"``, the
counterpart PythTB step, its own provenance and citation, and ``warning_codes``:
``warnings`` stays the list of plain messages the Generator parses, and
``warning_codes`` holds one stable code per message, in the same order.

Producer identity is QUEER's: ``provenance.queer_version``,
``queer_record_sha256`` (the installed distribution's ``RECORD``) and
``build_sha`` take the place of the driver's PythTB version and driver hash. The
model block keeps ``pythtb_version``, always null, beside its ``queer_version``,
so readers of the PythTB schema keep working.

Exit codes follow the driver: 0 success, 1 unexpected internal error, 2 failure
reported with a stable code, 3 out of memory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ...physics.validation import NumericalError
from .memory import available_cpus, available_memory_bytes, peak_rss

if TYPE_CHECKING:
    from .pythtb_params import WannierStepParameters

SCHEMA_VERSION = 1
ENGINE = "queer"
RESULTS_FILENAME = "queer_results.json"
LOG_FILENAME = "queer.log"
OUT_DIR = "Out"

# Failure codes; the Generator turns them into the job's failure_reason.
ERR_ENV_MISSING = "queer_env_missing"
ERR_MISSING_INPUT = "queer_missing_input"
ERR_TRANSLATE_HOME_CELL = "queer_translate_home_cell"
ERR_VALIDATION = "queer_validation_failed"
ERR_BAD_PARAMS = "queer_bad_params"
ERR_MEMORY = "queer_oom"
ERR_INTERNAL = "queer_internal_error"
ERR_NUMERICAL = "queer_numerical_error"
ERROR_CODES: tuple[str, ...] = (
    ERR_ENV_MISSING,
    ERR_MISSING_INPUT,
    ERR_TRANSLATE_HOME_CELL,
    ERR_VALIDATION,
    ERR_BAD_PARAMS,
    ERR_MEMORY,
    ERR_INTERNAL,
    ERR_NUMERICAL,
)

EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_FAILED = 2
EXIT_OUT_OF_MEMORY = 3

# Stable warning codes, one per kind of warning message.
WARN_WS_NO_MP_GRID = "ws_no_mp_grid"
WARN_WS_ASSUMED_DEFAULT = "ws_assumed_wannier90_default"
WARN_BAND_DAT_DEVIATION = "band_dat_deviation"
WARN_BAND_DAT_MISSING = "band_dat_missing"
WARN_FERMI_OUTSIDE_FROZEN = "fermi_outside_frozen_window"
WARN_DOS_OUTSIDE_GRID = "dos_outside_grid"
WARN_Z2_FORCED = "z2_without_kramers"
WARN_CHERN_NOT_INTEGER = "chern_kubo_not_integer"
WARN_METRIC_SKIPPED = "quantum_metric_skipped"
WARN_PATH_CURVATURE_SKIPPED = "path_curvature_skipped"
WARN_PLOT_SKIPPED = "plot_skipped"

APPROXIMATION_NOTE = (
    "Tight-binding (diagonal position) approximation, as in the Generator's PythTB engine: the "
    "Wannier centres are the orbital positions and the off-diagonal position matrix elements are "
    "dropped. Band energies and integer invariants are unaffected; Berry-curvature, quantum-metric "
    "and Hall magnitudes are approximate."
)
# The sign conventions every Berry result states.
BERRY_CONVENTIONS: Mapping[str, str] = {
    "omega": "-2 Im Q: Omega_ab = -2 Im sum_{n occ, m unocc} v^a_nm v^b_mn / (E_n - E_m)^2",
    "sigma_xy": "-(e^2/h) (1/2pi) sum_n int f_n Omega_n d^2k, carrier charge -e (-(e^2/h) C in a gap)",
}
# How far a Kubo Chern number may sit from an integer before the mesh is called
# too coarse for a small gap.
CHERN_INTEGER_TOL = 0.05


def citation(version: str | None) -> str:
    """The reference a QUEER Wannier result cites."""
    release = f" {version}" if version else ""
    return f"QUEER Development Team, QUEER{release}, https://github.com/susyexists/queer (MIT License)"


class RunnerError(Exception):
    """A failure with a stable ``code`` for the results contract."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def exit_code_for(code: str) -> int:
    """The process exit code of a failure code: 3 out of memory, 1 internal, else 2."""
    if code == ERR_MEMORY:
        return EXIT_OUT_OF_MEMORY
    if code == ERR_INTERNAL:
        return EXIT_INTERNAL
    return EXIT_FAILED


def describe_failure(error: BaseException) -> tuple[str, str, int]:
    """``(code, message, exit code)`` of an exception that ended a run.

    A :class:`RunnerError` keeps its code; a failed or nonfinite eigensolve is
    ``queer_numerical_error``; ``MemoryError`` is ``queer_oom`` with the driver's
    advice; anything else is ``queer_internal_error`` naming the exception type.
    """
    if isinstance(error, RunnerError):
        return error.code, str(error), exit_code_for(error.code)
    if isinstance(error, NumericalError):
        return ERR_NUMERICAL, str(error), EXIT_FAILED
    if isinstance(error, MemoryError):
        message = f"out of memory ({error}); lower chunk_size/workers or the mesh sizes"
        return ERR_MEMORY, message, EXIT_OUT_OF_MEMORY
    return ERR_INTERNAL, f"{type(error).__name__}: {error}", EXIT_INTERNAL


def to_jsonable(value: Any, digits: int | None = None) -> Any:
    """NumPy and container values as plain JSON values; NaN and infinities become null.

    Floats keep full precision unless ``digits`` rounds them; a complex number
    becomes ``[real, imag]``; a path becomes its string.
    """
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item, digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item, digits) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist(), digits)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return round(number, digits) if digits is not None else number
    if isinstance(value, (complex, np.complexfloating)):
        return [to_jsonable(value.real, digits), to_jsonable(value.imag, digits)]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json_atomic(path: str | Path, payload: Any) -> None:
    """Replace ``path`` with the JSON of ``payload`` at full precision, atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(to_jsonable(payload), stream, indent=1, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path: str | Path) -> str | None:
    """Streamed SHA-256 of a file, or None when it cannot be read."""
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def package_version(name: str) -> str | None:
    """The installed version of a distribution, or None."""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def queer_identity() -> dict[str, Any]:
    """QUEER's version, installed ``RECORD`` digest and release build revision."""
    from ...io.artifacts import distribution_identity

    return distribution_identity()


class RunLog:
    """Timestamped stage lines, appended to ``queer.log`` and echoed on standard output.

    Each line starts with the seconds since the log was opened, as the driver's
    ``pythtb.log`` does. A log that cannot be written never fails the run.
    """

    def __init__(self, path: str | Path, *, echo: bool = True):
        self.path = Path(path)
        self.echo = echo
        self.started = time.time()

    def __call__(self, message: str) -> None:
        line = f"[{time.time() - self.started:8.1f}s] {message}"
        if self.echo:
            print(line, flush=True)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        except OSError:
            pass


class RunRecord:
    """What a run accumulates for its results document.

    ``results`` holds the numerical blocks (``model``, ``validation``, ...);
    ``warnings`` are ``(code, message)`` pairs in order; ``outputs`` the files
    written under ``Out``; ``inputs`` the size and SHA-256 of every model file
    read; ``parallel`` the plan of every parallel stage.
    """

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace)
        self.results: dict[str, Any] = {}
        self.warnings: list[tuple[str, str]] = []
        self.outputs: list[dict[str, str]] = []
        self.inputs: dict[str, dict[str, Any]] = {}
        self.parallel: list[dict[str, Any]] = []

    def warn(self, code: str, message: str) -> None:
        self.warnings.append((code, message))

    def add_output(self, path: str, title: str, kind: str) -> None:
        self.outputs.append({"path": path, "title": title, "kind": kind})

    def record_input(self, name: str) -> None:
        """Size and SHA-256 of a model file just read from the workspace, streamed from disk.

        These are the bytes this run used; a file that cannot be stat'ed is not
        recorded.
        """
        path = self.workspace / name
        try:
            size = path.stat().st_size
        except OSError:
            return
        self.inputs[name] = {"bytes": int(size), "sha256": sha256_file(path)}

    def warning_messages(self) -> list[str]:
        return [message for _, message in self.warnings]

    def warning_codes(self) -> list[str]:
        return [code for code, _ in self.warnings]


def provenance(params_path: str | Path | None) -> dict[str, Any]:
    """The run-independent provenance: producer, parameter file and host."""
    identity = queer_identity()
    return {
        "queer_version": identity.get("version"),
        "queer_record_sha256": identity.get("record_sha256"),
        "build_sha": identity.get("build_sha"),
        "params_sha256": None if params_path is None else sha256_file(params_path),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": package_version("scipy"),
        "matplotlib": package_version("matplotlib"),
        "host": platform.node(),
        "cpus": available_cpus(),
        "memory_available_bytes": available_memory_bytes(),
    }


def new_document(step_type: str | None, params_path: str | Path | None = None) -> dict[str, Any]:
    """A results document in its ``running`` state, before any block is computed."""
    identity = provenance(params_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "engine": ENGINE,
        "status": "running",
        "step_type": step_type,
        "counterpart_step_type": None,
        "provenance": identity,
        "citation": citation(identity.get("queer_version")),
        "approximation": APPROXIMATION_NOTE,
    }


def finish_document(document: dict[str, Any], record: RunRecord, started: float) -> dict[str, Any]:
    """Fold a run's blocks, warnings, outputs, inputs and plans into ``document``."""
    document.update(record.results)
    document["warnings"] = record.warning_messages()
    document["warning_codes"] = record.warning_codes()
    document["outputs"] = list(record.outputs)
    identity = document.setdefault("provenance", {})
    identity["inputs"] = dict(record.inputs)
    identity["parallel"] = list(record.parallel)
    identity["wall_seconds"] = round(time.time() - started, 2)
    identity["peak_rss_bytes"] = peak_rss().get("peak_rss_bytes")
    return document


def read_params(params_path: str | Path) -> Mapping[str, Any]:
    """The JSON object of a parameter file; ``queer_bad_params`` when it cannot be read."""
    try:
        with open(params_path, encoding="utf-8") as stream:
            params = json.load(stream)
    except (OSError, ValueError) as error:
        raise RunnerError(ERR_BAD_PARAMS, f"cannot read {params_path}: {error}") from error
    if not isinstance(params, Mapping):
        raise RunnerError(ERR_BAD_PARAMS, f"{params_path} does not hold a JSON object")
    return params


def unserializable_document(document: Mapping[str, Any], error: Exception) -> dict[str, Any]:
    """A failure document without the result blocks, for a document that is not JSON."""
    kept = (
        "schema_version",
        "engine",
        "step_type",
        "counterpart_step_type",
        "provenance",
        "citation",
        "approximation",
        "warnings",
        "warning_codes",
        "outputs",
    )
    failed = {key: document[key] for key in kept if key in document}
    failed["status"] = "failed"
    failed["error"] = {
        "code": ERR_INTERNAL,
        "message": f"the results document is not JSON ({type(error).__name__}: {error})",
    }
    return failed


StepBody = Callable[["WannierStepParameters", RunRecord, Callable[[str], None]], Any]


def execute(
    workspace: str | Path,
    params_path: str | Path,
    run: StepBody,
    *,
    step: str | None = None,
    log: Callable[[str], None] | None = None,
) -> int:
    """Run one step in ``workspace`` and write ``queer_results.json`` whatever happens.

    The parameter file is resolved for ``step`` (or its own ``step_type``) and
    ``run(parameters, record, log)`` fills ``record``. The document is written on
    success and on every failure, with the blocks computed so far, and the
    return value is the process exit code. Blocks that are not JSON give way to
    a ``queer_internal_error`` document without them. An interrupt propagates
    after the document, still ``running``, is written.
    """
    from .pythtb_params import translate_pythtb

    started = time.time()
    root = Path(workspace)
    write = log if log is not None else RunLog(root / LOG_FILENAME)
    document = new_document(step, params_path)
    record = RunRecord(root)
    exit_code = EXIT_OK
    try:
        params = read_params(params_path)
        if step is None:
            document["step_type"] = params.get("step_type")
        parameters = translate_pythtb(params, step)
        document["step_type"] = parameters.step
        document["counterpart_step_type"] = parameters.counterpart
        (root / OUT_DIR).mkdir(parents=True, exist_ok=True)
        version = document["provenance"].get("queer_version")
        write(f"QUEER {version} - step {parameters.step} ({parameters.counterpart})")
        run(parameters, record, write)
        document["status"] = "ok"
    except Exception as error:  # the results file must exist whatever happened
        code, message, exit_code = describe_failure(error)
        document["status"] = "failed"
        document["error"] = {"code": code, "message": message}
        if code == ERR_INTERNAL:
            write("FAILED with an unexpected error:\n" + traceback.format_exc())
        elif code == ERR_MEMORY:
            write("FAILED [out of memory]")
        else:
            write(f"FAILED [{code}] {message}")
    finally:
        finish_document(document, record, started)
        target = root / RESULTS_FILENAME
        try:
            try:
                write_json_atomic(target, document)
            except (TypeError, ValueError) as error:
                document = unserializable_document(document, error)
                exit_code = EXIT_INTERNAL
                write(f"FAILED [{ERR_INTERNAL}] {document['error']['message']}")
                write_json_atomic(target, document)
        except (OSError, TypeError, ValueError) as error:
            print(f"queer generator: cannot write {target}: {error}", file=sys.stderr)
            exit_code = exit_code or EXIT_INTERNAL
        else:
            write(f"wrote {RESULTS_FILENAME} (status {document['status']})")
    return exit_code
