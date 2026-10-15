"""The generator runner: exit codes, failure documents, capabilities and ``python -m queer generator``.

Every run writes ``queer_results.json``; the exit code follows the Generator
PythTB driver (0 ok, 2 coded failure, 3 out of memory, 1 unexpected).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tight_binding_models import haldane
from wannier90_files import Wannier90Job

import queer
from queer.compat.generator import bands
from queer.compat.generator.results import (
    EXIT_FAILED,
    EXIT_INTERNAL,
    EXIT_OK,
    EXIT_OUT_OF_MEMORY,
    RESULTS_FILENAME,
)
from queer.compat.generator.runner import COMPARE_UNAVAILABLE, STEP_RUNNERS, main, run

ROOT = Path(__file__).resolve().parents[2]
LAYER_PATH = {
    "segments": [
        ["G", [0, 0, 0], "M", [0.5, 0, 0]],
        ["M", [0.5, 0, 0], "K", [1 / 3, 2 / 3, 0]],
        ["K", [1 / 3, 2 / 3, 0], "G", [0, 0, 0]],
    ]
}
SETTINGS = {
    "step_type": "PYTHTB_BANDS",
    "fermi_energy": 0.0,
    "workers": 1,
    "ws_correction": "off",
    "kpath": LAYER_PATH,
    "kpath_points": 40,
    "dos_mesh": 12,
}


def stage(directory, *, files=True, **settings):
    """A gapped Haldane layer's model files and a parameter file; returns the parameter file."""
    directory.mkdir(parents=True, exist_ok=True)
    if files:
        model = haldane(staggered=0.3)
        job = Wannier90Job(model.translations, model.hopping_matrices, model.lattice, model.centres)
        job.write(directory)
    values = dict(SETTINGS)
    values.update(settings)
    path = directory / "pythtb_params.json"
    path.write_text(json.dumps({key: value for key, value in values.items() if value is not None}))
    return path


def results(directory):
    return json.loads((directory / RESULTS_FILENAME).read_text())


def test_a_run_writes_the_document_and_exits_zero(tmp_path):
    params = stage(tmp_path)
    assert run("QUEER_BANDS", params, tmp_path, log=lambda line: None) == EXIT_OK
    document = results(tmp_path)
    assert document["status"] == "ok" and document["engine"] == "queer"
    assert document["step_type"] == "QUEER_BANDS" and document["counterpart_step_type"] == "PYTHTB_BANDS"
    assert {"model", "validation", "bands", "dos", "electronic"} <= set(document)


@pytest.mark.parametrize(
    ("settings", "files", "code", "fragment"),
    [
        ({}, False, "queer_missing_input", "wannier90.win was not staged"),
        ({"dos_mesh": 1}, True, "queer_bad_params", "dos_mesh"),
        ({"planes": [[2, 0.0]]}, True, "queer_bad_params", "unknown parameters for QUEER_BANDS: planes"),
        ({"fermi_energy": None}, True, "queer_bad_params", "No Fermi energy"),
        (
            {
                "step_type": "PYTHTB_SLAB",
                "kpath": None,
                "kpath_points": None,
                "dos_mesh": None,
                "slab_axis": 2,
            },
            True,
            "queer_bad_params",
            "slab_axis 2 is not a periodic direction of the model",
        ),
        ({"step_type": "PYTHTB_QGEOM"}, True, "queer_bad_params", "unknown step_type"),
    ],
)
def test_coded_failures_exit_two_with_a_failure_document(tmp_path, settings, files, code, fragment):
    params = stage(tmp_path, files=files, **settings)
    assert run(None, params, tmp_path, log=lambda line: None) == EXIT_FAILED
    document = results(tmp_path)
    assert document["status"] == "failed" and document["error"]["code"] == code
    assert fragment in document["error"]["message"]


def test_a_step_missing_from_the_build_is_a_bad_parameter_failure(tmp_path, monkeypatch):
    from types import MappingProxyType

    from queer.compat.generator import runner

    built = {step: body for step, body in STEP_RUNNERS.items() if step != "QUEER_SLAB"}
    monkeypatch.setattr(runner, "STEP_RUNNERS", MappingProxyType(built))
    params = stage(tmp_path, step_type="PYTHTB_SLAB", kpath=None, kpath_points=None, dos_mesh=None)
    assert run(None, params, tmp_path, log=lambda line: None) == EXIT_FAILED
    document = results(tmp_path)
    assert document["error"]["code"] == "queer_bad_params"
    assert document["error"]["message"] == (
        "QUEER_SLAB (PYTHTB_SLAB) is not available in this build; it runs QUEER_BANDS, QUEER_BERRY, "
        "QUEER_TOPOLOGY (or their PYTHTB_ names)"
    )


def test_an_unreadable_parameter_file_is_a_bad_parameter_failure(tmp_path):
    params = tmp_path / "pythtb_params.json"
    params.write_text("{not json")
    assert run("QUEER_BANDS", params, tmp_path, log=lambda line: None) == EXIT_FAILED
    assert results(tmp_path)["error"]["code"] == "queer_bad_params"


def test_an_unexpected_error_exits_one_with_the_traceback_tail(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise ZeroDivisionError("float division by zero")

    monkeypatch.setattr(bands, "dos_divisions", broken)
    params = stage(tmp_path)
    assert run("QUEER_BANDS", params, tmp_path, log=lambda line: None) == EXIT_INTERNAL
    document = results(tmp_path)
    assert document["status"] == "failed" and document["error"]["code"] == "queer_internal_error"
    message = document["error"]["message"]
    assert message.startswith("ZeroDivisionError: float division by zero (traceback tail: ")
    assert "bands.py:" in message and " in run_bands; " in message and message.endswith(" in broken)")
    assert {"model", "validation"} <= set(document) and "bands" not in document


def test_running_out_of_memory_exits_three(tmp_path, monkeypatch):
    def exhausted(*args, **kwargs):
        raise MemoryError("cannot allocate the DOS mesh")

    monkeypatch.setattr(bands, "uniform_mesh", exhausted)
    params = stage(tmp_path)
    assert run("QUEER_BANDS", params, tmp_path, log=lambda line: None) == EXIT_OUT_OF_MEMORY
    error = results(tmp_path)["error"]
    assert error["code"] == "queer_oom" and "lower chunk_size/workers or the mesh sizes" in error["message"]


def test_an_interrupt_propagates_after_the_running_document_is_written(tmp_path, monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(bands, "uniform_mesh", interrupted)
    params = stage(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        run("QUEER_BANDS", params, tmp_path, log=lambda line: None)
    document = results(tmp_path)
    assert document["status"] == "running" and "error" not in document and "model" in document


def test_capabilities_list_the_steps_of_this_build(capsys):
    assert main(["capabilities"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    steps = ["QUEER_BANDS", "QUEER_BERRY", "QUEER_SLAB", "QUEER_TOPOLOGY"]
    assert payload == {"queer_version": queer.__version__, "steps": steps}
    assert payload["steps"] == sorted(STEP_RUNNERS)


def test_compare_writes_nothing_and_exits_two(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["compare", "--reference", "a.json", "--candidate", "b.json"]) == EXIT_FAILED
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.strip() == COMPARE_UNAVAILABLE
    assert list(tmp_path.iterdir()) == []


def test_unusable_arguments_exit_two(tmp_path, capsys):
    assert main(["run", "--params", "p.json", f"--workspace={tmp_path}"]) == EXIT_FAILED
    document = results(tmp_path)
    assert document["status"] == "failed" and document["error"]["code"] == "queer_bad_params"
    assert "--step" in document["error"]["message"] and document["step_type"] is None
    assert main([]) == EXIT_FAILED and main(["deploy"]) == EXIT_FAILED
    assert "usage: python -m queer generator" in capsys.readouterr().err


def entry(arguments, cwd):
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT / "src"), MPLBACKEND="Agg", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    return subprocess.run(
        [sys.executable, "-m", "queer", "generator", *arguments],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_the_module_entry_runs_a_step_in_its_workspace(tmp_path):
    workspace = tmp_path / "job"
    stage(workspace)
    finished = entry(["run", "--step", "QUEER_BANDS", "--params", "pythtb_params.json"], workspace)
    assert finished.returncode == 0, finished.stderr
    document = results(workspace)
    assert document["status"] == "ok" and document["step_type"] == "QUEER_BANDS"
    assert (workspace / "Out" / "dos.json").is_file() and (workspace / "queer.log").is_file()
    assert "wrote queer_results.json (status ok)" in finished.stdout

    empty = tmp_path / "empty"
    stage(empty, files=False)
    finished = entry(["run", "--step", "PYTHTB_BANDS", "--params", "pythtb_params.json"], empty)
    assert finished.returncode == 2
    assert results(empty)["error"]["code"] == "queer_missing_input"

    finished = entry(["capabilities"], tmp_path)
    assert finished.returncode == 0
    assert json.loads(finished.stdout)["steps"] == [
        "QUEER_BANDS",
        "QUEER_BERRY",
        "QUEER_SLAB",
        "QUEER_TOPOLOGY",
    ]
