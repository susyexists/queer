"""Standard-library entry point for ``python -m queer generator``.

The Generator runs QUEER from a staged copy with a cluster host's Python
environment. This module checks that environment before anything imports NumPy,
so a missing or outdated dependency produces a machine-readable failure record
instead of an import traceback. It must import nothing outside the standard
library at module level.

Commands are ``run --step STEP --params FILE [--workspace DIR]``,
``capabilities`` and ``compare ...``. Once the environment check passes, each
is delegated to ``queer.compat.generator.runner.main(argv)`` with the arguments
after ``generator``. Exit codes follow the Generator PythTB driver: 0 success,
1 unexpected internal error, 2 reported failure, 3 out of memory.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import sys
import tempfile
from pathlib import Path
from types import ModuleType

RESULTS_FILENAME = "queer_results.json"
SCHEMA_VERSION = 1
RUNNER_MODULE = "queer.compat.generator.runner"
# Minimum versions of the core dependencies; equal to [project].dependencies.
REQUIREMENTS = {"numpy": "1.26", "scipy": "1.11", "joblib": "1.3", "threadpoolctl": "3.2"}
EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_FAILED = 2
EXIT_OUT_OF_MEMORY = 3

_RELEASE = re.compile(r"v?(\d+(?:\.\d+)*)(.*)", re.IGNORECASE)
_PRE_RELEASE = re.compile(r"[-_.]?(a|b|c|rc|alpha|beta|pre|preview|dev)", re.IGNORECASE)


class _UsageError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise _UsageError(message)


def _release(text: str) -> tuple[tuple[int, ...], bool] | None:
    match = _RELEASE.fullmatch(text.strip())
    if match is None or "!" in text:
        return None
    suffix = match.group(2).split("+", 1)[0]
    return tuple(int(part) for part in match.group(1).split(".")), bool(_PRE_RELEASE.match(suffix))


def version_at_least(installed: str, minimum: str) -> bool:
    """Whether ``installed`` satisfies ``>= minimum`` for PEP 440 release numbers.

    Missing trailing components are zero, a pre- or development release precedes
    its final release, and post or local suffixes do not lower a release.
    Unrecognized versions, including epochs, do not satisfy the minimum.
    """
    have, need = _release(installed), _release(minimum)
    if have is None or need is None:
        return False
    width = max(len(have[0]), len(need[0]))
    left = have[0] + (0,) * (width - len(have[0]))
    right = need[0] + (0,) * (width - len(need[0]))
    if left != right:
        return left > right
    return not have[1]


def environment_gaps(requirements: dict[str, str] | None = None) -> list[str]:
    """Unsatisfied core requirements, without importing them."""
    gaps = []
    for name, minimum in (REQUIREMENTS if requirements is None else requirements).items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            gaps.append(f"{name}>={minimum} is not installed")
            continue
        if not version_at_least(installed, minimum):
            gaps.append(f"{name} {installed} is older than the required {minimum}")
        elif importlib.util.find_spec(name) is None:
            gaps.append(f"{name} {installed} is recorded as installed but cannot be imported")
    return gaps


def _queer_version() -> str:
    from . import __version__

    return __version__


def failure_record(step: str | None, code: str, message: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "step_type": step,
        "engine": "queer",
        "error": {"code": code, "message": message},
        "provenance": {"queer_version": _queer_version(), "python": platform.python_version()},
        "warnings": [],
        "outputs": [],
    }


def write_results(workspace: Path, record: dict) -> Path:
    """Atomically replace ``workspace/queer_results.json``."""
    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / RESULTS_FILENAME
    handle, temporary = tempfile.mkstemp(prefix=f".{RESULTS_FILENAME}.", dir=workspace)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def _environment_message(gaps: list[str]) -> str:
    return (
        "the Python environment cannot run QUEER: "
        + "; ".join(gaps)
        + f" (interpreter {sys.executable}). Install the missing packages in the environment "
        "that the host's Queer settings select."
    )


def _runner() -> ModuleType | None:
    try:
        return importlib.import_module(RUNNER_MODULE)
    except ModuleNotFoundError as error:
        if error.name == RUNNER_MODULE:
            return None
        raise


def _option(arguments: list[str], name: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == name and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith(name + "="):
            return argument.split("=", 1)[1]
    return None


def _fail(workspace: Path, step: str | None, code: str, message: str, exit_code: int) -> int:
    try:
        write_results(workspace, failure_record(step, code, message))
    except OSError as error:
        print(f"queer generator: cannot write {workspace / RESULTS_FILENAME}: {error}", file=sys.stderr)
    print(f"queer generator: [{code}] {message}", file=sys.stderr)
    return exit_code


def _run(arguments: list[str]) -> int:
    parser = _Parser(prog="python -m queer generator run")
    parser.add_argument("--step", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--workspace", default=".")
    try:
        options = parser.parse_args(arguments[1:])
    except _UsageError as error:
        workspace = Path(_option(arguments, "--workspace") or ".")
        message = f"invalid generator run arguments: {error}"
        return _fail(workspace, _option(arguments, "--step"), "queer_bad_params", message, EXIT_FAILED)
    workspace = Path(options.workspace)
    gaps = environment_gaps()
    if gaps:
        return _fail(workspace, options.step, "queer_env_missing", _environment_message(gaps), EXIT_FAILED)
    try:
        runner = _runner()
    except Exception as error:  # the results file must exist whatever happened
        message = f"cannot load the generator runner: {type(error).__name__}: {error}"
        return _fail(workspace, options.step, "queer_internal_error", message, EXIT_INTERNAL)
    if runner is None:
        message = "generator runner not implemented in this build"
        return _fail(workspace, options.step, "queer_internal_error", message, EXIT_INTERNAL)
    return int(runner.main(arguments))


def _capabilities(arguments: list[str]) -> int:
    payload: dict = {"queer_version": _queer_version(), "steps": []}
    gaps = environment_gaps()
    if gaps:
        payload["error"] = {"code": "queer_env_missing", "message": _environment_message(gaps)}
        print(json.dumps(payload, sort_keys=True))
        return EXIT_FAILED
    runner = _runner()
    if runner is not None:
        return int(runner.main(arguments))
    print(json.dumps(payload, sort_keys=True))
    return EXIT_OK


def _compare(arguments: list[str]) -> int:
    gaps = environment_gaps()
    if gaps:
        print(f"queer generator: [queer_env_missing] {_environment_message(gaps)}", file=sys.stderr)
        return EXIT_FAILED
    runner = _runner()
    if runner is None:
        print("queer generator: generator runner not implemented in this build", file=sys.stderr)
        return EXIT_INTERNAL
    return int(runner.main(arguments))


def main(argv: list[str] | None = None) -> int:
    """Dispatch ``run``, ``capabilities`` or ``compare`` (the arguments after ``generator``)."""
    arguments = list(sys.argv[2:] if argv is None else argv)
    commands = {"run": _run, "capabilities": _capabilities, "compare": _compare}
    if not arguments or arguments[0] not in commands:
        print(
            "usage: python -m queer generator {run --step STEP --params FILE [--workspace DIR] | "
            "capabilities | compare ...}",
            file=sys.stderr,
        )
        return EXIT_FAILED
    return commands[arguments[0]](arguments)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
