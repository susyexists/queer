"""The ``python -m queer generator`` commands behind the standard-library boot.

``run --step STEP --params FILE [--workspace DIR]`` resolves the parameter file
for STEP (a QUEER step or its ``PYTHTB_`` name), builds and validates the model
from the workspace's staged ``wannier90`` files, runs the step and writes
``queer_results.json`` in the workspace on success and on every failure. The
exit code follows the Generator PythTB driver: 0 success, 2 a failure reported
with a stable code, 3 out of memory (``queer_oom``), 1 anything unexpected
(``queer_internal_error``, whose message ends with the innermost frames of the
traceback). As in the driver, an interrupt is not caught: the document, still
``running``, is written and the interrupt propagates; SIGTERM keeps Python's
default action and ends the process at once.

``capabilities`` prints ``{"queer_version", "steps"}`` as JSON, the steps being
the ones this build runs.

``compare --step STEP --reference DIR --candidate DIR [--report FILE]
[--markdown FILE]`` compares two run directories of a step
(:func:`~queer.compat.generator.parity.compare`), prints the verdict and the
failed checks, and writes the JSON report and its Markdown page when asked. It
exits 0 for ``pass`` and ``pass_with_documented_differences``, 2 otherwise
(``fail``, ``not_comparable`` or unusable arguments).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .bands import run_bands
from .berry import run_berry
from .pythtb_params import STEP_BANDS, STEP_BERRY, STEP_SLAB, STEP_TOPOLOGY, WannierStepParameters
from .results import (
    ERR_BAD_PARAMS,
    EXIT_FAILED,
    EXIT_OK,
    RESULTS_FILENAME,
    RunnerError,
    RunRecord,
    execute,
    new_document,
    write_json_atomic,
)
from .slab import run_slab
from .topology import run_topology
from .wannier_steps import WannierStepContext, model_stage

STEP_RUNNERS: Mapping[str, Callable[[WannierStepContext], None]] = MappingProxyType(
    {STEP_BANDS: run_bands, STEP_TOPOLOGY: run_topology, STEP_BERRY: run_berry, STEP_SLAB: run_slab}
)
USAGE = (
    "usage: python -m queer generator {run --step STEP --params FILE [--workspace DIR] | "
    "capabilities | compare --step STEP --reference DIR --candidate DIR [--report FILE] [--markdown FILE]}"
)


class _UsageError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message):  # report instead of exiting the interpreter
        raise _UsageError(message)


def step_body(parameters: WannierStepParameters, record: RunRecord, log: Callable[[str], None]) -> None:
    """Build and validate the model of the step, then run it: the body :func:`run` executes."""
    runner = STEP_RUNNERS.get(parameters.step)
    if runner is None:
        steps = ", ".join(sorted(STEP_RUNNERS))
        raise RunnerError(
            ERR_BAD_PARAMS,
            f"{parameters.step} ({parameters.counterpart}) is not available in this build; it runs {steps} "
            "(or their PYTHTB_ names)",
        )
    runner(model_stage(parameters, record, log))


def run(
    step: str | None,
    params_path: str | Path,
    workspace: str | Path = ".",
    *,
    log: Callable[[str], None] | None = None,
) -> int:
    """Run one step in ``workspace``; returns the exit code. ``queer_results.json`` is always written."""
    return execute(workspace, params_path, step_body, step=step, log=log)


def capabilities() -> dict[str, Any]:
    """The QUEER version and the steps this build runs."""
    from ... import __version__

    return {"queer_version": __version__, "steps": sorted(STEP_RUNNERS)}


def _option(arguments: Sequence[str], name: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == name and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith(name + "="):
            return argument.split("=", 1)[1]
    return None


def _usage_failure(arguments: Sequence[str], message: str) -> int:
    """Write a ``queer_bad_params`` document for unusable ``run`` arguments; exit code 2."""
    workspace = Path(_option(arguments, "--workspace") or ".")
    document = new_document(_option(arguments, "--step"))
    document.update(
        status="failed",
        error={"code": ERR_BAD_PARAMS, "message": message},
        warnings=[],
        warning_codes=[],
        outputs=[],
    )
    try:
        write_json_atomic(workspace / RESULTS_FILENAME, document)
    except OSError as error:
        print(f"queer generator: cannot write {workspace / RESULTS_FILENAME}: {error}", file=sys.stderr)
    print(f"queer generator: [{ERR_BAD_PARAMS}] {message}", file=sys.stderr)
    return EXIT_FAILED


def compare_command(arguments: Sequence[str]) -> int:
    """``compare``: print the verdict, write the requested reports; exit 0 when the verdict passes, else 2."""
    from .parity import compare, passing, render_markdown

    parser = _Parser(prog="python -m queer generator compare")
    parser.add_argument("--step", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--report")
    parser.add_argument("--markdown")
    try:
        options = parser.parse_args(list(arguments))
        report = compare(options.step, options.reference, options.candidate)
    except (_UsageError, ValueError) as error:
        print(f"queer generator compare: {error}", file=sys.stderr)
        return EXIT_FAILED
    try:
        if options.report:
            write_json_atomic(options.report, report)
        if options.markdown:
            target = Path(options.markdown)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(render_markdown(report), encoding="utf-8")
    except OSError as error:
        print(f"queer generator compare: cannot write the report: {error}", file=sys.stderr)
        return EXIT_FAILED
    print(report["verdict"])
    for item in report["preconditions"]:
        if not item["ok"]:
            print(f"  precondition {item['name']}: {item['detail']}")
    for item in report["quantities"]:
        if item["class"] == "differs_unexplained":
            print(f"  {item['name']}: {item['detail']}")
    return EXIT_OK if passing(report) else EXIT_FAILED


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch ``run``, ``capabilities`` or ``compare`` (the arguments after ``generator``)."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = arguments[0] if arguments else None
    if command == "run":
        parser = _Parser(prog="python -m queer generator run")
        parser.add_argument("--step", required=True)
        parser.add_argument("--params", required=True)
        parser.add_argument("--workspace", default=".")
        try:
            options = parser.parse_args(arguments[1:])
        except _UsageError as error:
            return _usage_failure(arguments, f"invalid generator run arguments: {error}")
        return run(options.step, options.params, options.workspace)
    if command == "capabilities":
        print(json.dumps(capabilities(), sort_keys=True))
        return EXIT_OK
    if command == "compare":
        return compare_command(arguments[1:])
    print(USAGE, file=sys.stderr)
    return EXIT_FAILED
