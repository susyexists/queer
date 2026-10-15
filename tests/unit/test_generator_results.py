"""The QUEER results document is written on success and failure with the driver's codes."""

import hashlib
import json
import math
import re

import numpy as np
import pytest

import queer
from queer.compat.generator import results
from queer.compat.generator.results import (
    ERR_BAD_PARAMS,
    ERR_INTERNAL,
    ERR_MEMORY,
    ERR_NUMERICAL,
    ERR_VALIDATION,
    EXIT_FAILED,
    EXIT_INTERNAL,
    EXIT_OK,
    EXIT_OUT_OF_MEMORY,
    RESULTS_FILENAME,
    RunLog,
    RunnerError,
    RunRecord,
    describe_failure,
    execute,
    to_jsonable,
    write_json_atomic,
)
from queer.physics.validation import NumericalError

PROVENANCE_KEYS = {
    "queer_version",
    "queer_record_sha256",
    "build_sha",
    "params_sha256",
    "python",
    "numpy",
    "scipy",
    "matplotlib",
    "host",
    "cpus",
    "memory_available_bytes",
    "inputs",
    "parallel",
    "wall_seconds",
    "peak_rss_bytes",
}
LOG_LINE = re.compile(r"^\[\s*\d+\.\ds\] ")


def test_numpy_and_nonfinite_values_become_plain_json():
    payload = {
        "nan": float("nan"),
        "inf": np.inf,
        "values": np.array([1.5, np.nan, -np.inf]),
        "integer": np.int64(7),
        "flag": np.bool_(True),
        "single": np.float32(0.25),
        "complex": 1.0 + 2.0j,
        "nested": [(np.float64(1.23456), {"z": np.complex128(np.nan + 1j)})],
        3: "key",
    }
    plain = to_jsonable(payload)
    assert plain == {
        "nan": None,
        "inf": None,
        "values": [1.5, None, None],
        "integer": 7,
        "flag": True,
        "single": 0.25,
        "complex": [1.0, 2.0],
        "nested": [[1.23456, {"z": [None, 1.0]}]],
        "3": "key",
    }
    assert type(plain["integer"]) is int and type(plain["flag"]) is bool
    assert to_jsonable(np.float64(1.23456), digits=2) == 1.23


def test_atomic_json_writes_null_for_nan_and_leaves_no_temporary(tmp_path):
    target = tmp_path / "out" / "doc.json"
    write_json_atomic(target, {"a": 1})
    write_json_atomic(target, {"value": np.float64(np.nan), "array": np.arange(2)})
    assert json.loads(target.read_text()) == {"value": None, "array": [0, 1]}
    assert "NaN" not in target.read_text()
    assert [path.name for path in target.parent.iterdir()] == ["doc.json"]


def test_record_keeps_warnings_in_order_and_hashes_inputs(tmp_path):
    (tmp_path / "wannier90.win").write_bytes(b"num_wann = 2\n")
    record = RunRecord(tmp_path)
    record.warn("band_dat_missing", "first")
    record.warn("plot_skipped", "second")
    record.record_input("wannier90.win")
    record.record_input("absent.dat")
    assert record.warning_messages() == ["first", "second"]
    assert record.warning_codes() == ["band_dat_missing", "plot_skipped"]
    digest = hashlib.sha256(b"num_wann = 2\n").hexdigest()
    assert record.inputs == {"wannier90.win": {"bytes": 13, "sha256": digest}}


@pytest.mark.parametrize(
    ("error", "code", "exit_code"),
    [
        (RunnerError(ERR_VALIDATION, "departs"), ERR_VALIDATION, EXIT_FAILED),
        (RunnerError(ERR_MEMORY, "too large"), ERR_MEMORY, EXIT_OUT_OF_MEMORY),
        (RunnerError(ERR_INTERNAL, "broken"), ERR_INTERNAL, EXIT_INTERNAL),
        (NumericalError("eigensolver failed"), ERR_NUMERICAL, EXIT_FAILED),
        (MemoryError("cannot allocate"), ERR_MEMORY, EXIT_OUT_OF_MEMORY),
        (KeyError("model"), ERR_INTERNAL, EXIT_INTERNAL),
    ],
)
def test_failures_map_to_the_drivers_codes_and_exit_codes(error, code, exit_code):
    described = describe_failure(error)
    assert described[0] == code and described[2] == exit_code
    if isinstance(error, KeyError):
        assert described[1] == "KeyError: 'model'"
    if isinstance(error, MemoryError):
        assert "lower chunk_size/workers or the mesh sizes" in described[1]


def write_params(tmp_path, params):
    path = tmp_path / "pythtb_params.json"
    path.write_text(json.dumps(params))
    return path


def run_document(tmp_path, params, body):
    lines = []
    path = params if isinstance(params, str) else write_params(tmp_path, params)
    code = execute(tmp_path, path, body, log=lines.append)
    return code, json.loads((tmp_path / RESULTS_FILENAME).read_text()), lines


def test_a_successful_run_writes_the_schema_with_queer_provenance(tmp_path):
    def body(parameters, record, log):
        assert parameters.step == "QUEER_BANDS"
        (record.workspace / "wannier90.win").write_text("num_wann = 1\n")
        record.record_input("wannier90.win")
        record.results["model"] = {"num_wann": 1, "lattice": np.eye(3), "gap": float("nan")}
        record.warn("band_dat_missing", "wannier90_band.dat was not staged")
        record.add_output("Out/bands.png", "Band structure", "plot")
        record.parallel.append({"stage": "validation", "k_points": 3})
        log("model: N=1")

    params = {"step_type": "PYTHTB_BANDS", "fermi_energy": 1.0}
    code, document, lines = run_document(tmp_path, params, body)
    assert code == EXIT_OK
    assert document["status"] == "ok" and document["engine"] == "queer" and document["schema_version"] == 1
    assert document["step_type"] == "QUEER_BANDS" and document["counterpart_step_type"] == "PYTHTB_BANDS"
    assert document["warnings"] == ["wannier90_band.dat was not staged"]
    assert document["warning_codes"] == ["band_dat_missing"]
    assert document["outputs"] == [{"path": "Out/bands.png", "title": "Band structure", "kind": "plot"}]
    assert document["model"] == {"num_wann": 1, "lattice": np.eye(3).tolist(), "gap": None}
    assert "error" not in document
    provenance = document["provenance"]
    assert set(provenance) == PROVENANCE_KEYS
    assert provenance["queer_version"] == queer.__version__
    params_bytes = (tmp_path / "pythtb_params.json").read_bytes()
    assert provenance["params_sha256"] == hashlib.sha256(params_bytes).hexdigest()
    assert set(provenance["inputs"]) == {"wannier90.win"}
    assert provenance["parallel"] == [{"stage": "validation", "k_points": 3}]
    assert provenance["cpus"] >= 1 and provenance["memory_available_bytes"] > 0
    assert provenance["peak_rss_bytes"] > 0 and provenance["wall_seconds"] >= 0
    assert queer.__version__ in document["citation"] and "QUEER" in document["citation"]
    assert document["approximation"] == results.APPROXIMATION_NOTE
    assert (tmp_path / "Out").is_dir()
    assert lines[0].startswith(f"QUEER {queer.__version__} - step QUEER_BANDS")
    assert lines[-1] == f"wrote {RESULTS_FILENAME} (status ok)"


@pytest.mark.parametrize(
    ("params", "step_type", "fragment"),
    [
        ("missing.json", None, "cannot read"),
        ({"step_type": "PYTHTB_QGEOM"}, "PYTHTB_QGEOM", "unknown step_type"),
        ({"step_type": "PYTHTB_BANDS", "spin_mode": "collinear"}, "PYTHTB_BANDS", "spin_mode"),
    ],
)
def test_bad_parameters_still_write_a_failure_document(tmp_path, params, step_type, fragment):
    def body(parameters, record, log):
        raise AssertionError("never reached")

    path = str(tmp_path / params) if isinstance(params, str) else params
    code, document, lines = run_document(tmp_path, path, body)
    assert code == EXIT_FAILED
    assert document["status"] == "failed" and document["step_type"] == step_type
    assert document["error"]["code"] == ERR_BAD_PARAMS and fragment in document["error"]["message"]
    assert document["warnings"] == [] and document["warning_codes"] == [] and document["outputs"] == []
    assert set(document["provenance"]) == PROVENANCE_KEYS
    assert any(line.startswith(f"FAILED [{ERR_BAD_PARAMS}]") for line in lines)


@pytest.mark.parametrize(
    ("error", "code", "exit_code", "logged"),
    [
        (RunnerError(ERR_VALIDATION, "departs by 120 meV"), ERR_VALIDATION, EXIT_FAILED, "FAILED [queer_"),
        (NumericalError("nonfinite"), ERR_NUMERICAL, EXIT_FAILED, "FAILED [queer_numerical_error]"),
        (MemoryError("chunk"), ERR_MEMORY, EXIT_OUT_OF_MEMORY, "FAILED [out of memory]"),
        (ValueError("unexpected"), ERR_INTERNAL, EXIT_INTERNAL, "FAILED with an unexpected error:"),
    ],
)
def test_a_failing_step_keeps_its_blocks_and_warnings(tmp_path, error, code, exit_code, logged):
    def body(parameters, record, log):
        record.results["validation"] = {"band_dat": {"in_window_max_meV": math.inf}}
        record.warn("band_dat_deviation", "large deviation")
        raise error

    code_returned, document, lines = run_document(tmp_path, {"step_type": "QUEER_TOPOLOGY"}, body)
    assert code_returned == exit_code
    assert document["status"] == "failed" and document["error"]["code"] == code
    assert document["step_type"] == "QUEER_TOPOLOGY"
    assert document["validation"] == {"band_dat": {"in_window_max_meV": None}}
    assert document["warnings"] == ["large deviation"] and document["warning_codes"] == ["band_dat_deviation"]
    assert any(line.startswith(logged) for line in lines)
    if code == ERR_INTERNAL:
        assert document["error"]["message"] == "ValueError: unexpected"
        assert any("Traceback" in line for line in lines)
    assert lines[-1] == f"wrote {RESULTS_FILENAME} (status failed)"


def test_a_block_that_is_not_json_leaves_an_internal_failure_document(tmp_path):
    def body(parameters, record, log):
        record.results["model"] = {"handle": object()}
        record.warn("plot_skipped", "kept")

    code, document, lines = run_document(tmp_path, {"step_type": "PYTHTB_BANDS"}, body)
    assert code == EXIT_INTERNAL
    assert document["status"] == "failed" and document["error"]["code"] == ERR_INTERNAL
    assert "not JSON" in document["error"]["message"] and "model" not in document
    assert document["step_type"] == "QUEER_BANDS" and document["warnings"] == ["kept"]
    assert set(document["provenance"]) == PROVENANCE_KEYS
    assert lines[-1] == f"wrote {RESULTS_FILENAME} (status failed)"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["Out", "pythtb_params.json", RESULTS_FILENAME]


def test_an_interrupt_propagates_after_the_running_document_is_written(tmp_path):
    def body(parameters, record, log):
        record.results["model"] = {"num_wann": 3}
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute(tmp_path, write_params(tmp_path, {"step_type": "PYTHTB_SLAB"}), body, log=lambda line: None)
    document = json.loads((tmp_path / RESULTS_FILENAME).read_text())
    assert document["status"] == "running" and document["model"] == {"num_wann": 3}


def test_run_log_lines_are_timestamped_and_appended(tmp_path, capsys):
    path = tmp_path / "queer.log"
    path.write_text("[     0.0s] earlier run\n")
    log = RunLog(path)
    log("read wannier90_hr.dat")
    RunLog(path, echo=False)("second")
    lines = path.read_text().splitlines()
    assert lines[0] == "[     0.0s] earlier run"
    assert all(LOG_LINE.match(line) for line in lines)
    assert lines[1].endswith("] read wannier90_hr.dat") and lines[2].endswith("] second")
    assert capsys.readouterr().out.strip().endswith("read wannier90_hr.dat")
