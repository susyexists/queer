"""Scheduler adapters can be checked without an installed Slurm cluster."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "hpc" / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def configuration(tmp_path):
    config = tmp_path / "momentum.toml"
    config.write_text(f'''workflow = "momentum"
[inputs]
material_dir = "."
poscar = "POSCAR"
[parameters]
ef = 0.0
interpolation = "standard"
photon_energy = 21.2
fermi_energy = 0.0
binding_min = 0.0
binding_max = 0.2
binding_step = 0.1
v0_values = [5.0, 12.0]
n_points = 4
sigma = 0.1
[output]
results_dir = "{tmp_path / "results"}"
''')
    profile = tmp_path / "profile.toml"
    profile.write_text('cpus_per_task=2\nconcurrency=1\nmemory="2G"\ntime="00:10:00"\n')
    return config, profile


def test_submission_prepares_logs_and_propagates_failure(tmp_path, monkeypatch):
    adapter = module("submit_sweep")
    config, profile = configuration(tmp_path)
    seen = []

    def fake_sbatch(command, *, env, check):
        dispatch = json.loads(Path(env["QUEER_SWEEP_FILE"]).read_text())
        assert (Path(dispatch["run_root"]) / "logs").is_dir()
        assert dispatch["values"] == [5.0, 12.0]
        snapshot = Path(dispatch["config"])
        assert snapshot != config and snapshot.is_file()
        assert snapshot.read_text() == config.read_text()
        assert dispatch["material_dir"] == str(config.parent)
        assert dispatch["results_dir"] == str(tmp_path / "results")
        assert "--array=0-1%1" in command
        assert "--cpus-per-task=2" in command
        assert Path(env["QUEER_PYTHON"]).is_file()
        seen.append(command)
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(adapter.subprocess, "run", fake_sbatch)
    assert adapter.main(["--config", str(config), "--profile", str(profile)]) == 7
    assert seen


def test_submission_dry_run_creates_no_directories(tmp_path, capsys):
    config, profile = configuration(tmp_path)
    assert module("submit_sweep").main(["--config", str(config), "--profile", str(profile), "--dry-run"]) == 0
    assert not (tmp_path / "results").exists()
    assert "--output=" in capsys.readouterr().out


def test_array_task_propagates_exit_and_caps_native_threads(tmp_path, monkeypatch):
    adapter = module("array_task")
    dispatch = tmp_path / "dispatch.json"
    dispatch.write_text(
        json.dumps(
            {
                "values": [5.0, 12.0],
                "config": "/shared/config.toml",
                "run_root": str(tmp_path),
                "material_dir": "/shared/inputs/material",
                "results_dir": "/shared/results",
            }
        )
    )
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")

    def fake_command(command, *, check):
        assert command[command.index("--v0") + 1] == "12.0"
        assert command[command.index("--workers") + 1] == "3"
        assert command[command.index("--native-threads") + 1] == "1"
        assert command[command.index("--run-dir") + 1].endswith("task-0001")
        assert command[command.index("--material-dir") + 1] == "/shared/inputs/material"
        assert command[command.index("--results-dir") + 1] == "/shared/results"
        return subprocess.CompletedProcess(command, 9)

    monkeypatch.setattr(adapter.subprocess, "run", fake_command)
    assert adapter.main([str(dispatch), "1"]) == 9
    with pytest.raises(ValueError, match="outside"):
        adapter.main([str(dispatch), "-1"])


def test_array_task_rejects_changed_snapshot(tmp_path):
    import hashlib

    adapter = module("array_task")
    snapshot = tmp_path / "snapshot.toml"
    snapshot.write_text("original")
    dispatch = tmp_path / "dispatch.json"
    dispatch.write_text(
        json.dumps(
            {
                "values": [5.0],
                "config": str(snapshot),
                "run_root": str(tmp_path),
                "config_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            }
        )
    )
    snapshot.write_text("modified")
    with pytest.raises(ValueError, match="snapshot was changed"):
        adapter.main([str(dispatch), "0"])


@pytest.mark.parametrize(
    "restart_count,has_manifest,resumes",
    [
        (None, False, False),
        (None, True, False),
        ("0", True, False),
        ("1", False, False),
        ("1", True, True),
        ("2", True, True),
    ],
)
def test_array_task_resumes_only_scheduler_restarts_with_checkpoints(
    tmp_path, monkeypatch, restart_count, has_manifest, resumes
):
    adapter = module("array_task")
    dispatch = tmp_path / "dispatch.json"
    dispatch.write_text(
        json.dumps({"values": [5.0], "config": "/shared/config.toml", "run_root": str(tmp_path)})
    )
    if has_manifest:
        run = tmp_path / "task-0000"
        run.mkdir()
        (run / "manifest.json").write_text("{}")
    if restart_count is None:
        monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    else:
        monkeypatch.setenv("SLURM_RESTART_COUNT", restart_count)

    def fake_command(command, *, check):
        assert ("--resume" in command) is resumes
        assert command[command.index("--run-dir") + 1] == str(tmp_path / "task-0000")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(adapter.subprocess, "run", fake_command)
    assert adapter.main([str(dispatch), "0"]) == 0


@pytest.mark.parametrize("restart_count", ["-1", "invalid", "1.5", ""])
def test_array_task_rejects_invalid_restart_count(tmp_path, monkeypatch, restart_count):
    adapter = module("array_task")
    dispatch = tmp_path / "dispatch.json"
    dispatch.write_text(
        json.dumps({"values": [5.0], "config": "/shared/config.toml", "run_root": str(tmp_path)})
    )
    monkeypatch.setenv("SLURM_RESTART_COUNT", restart_count)

    def prohibited(*args, **kwargs):
        raise AssertionError("Invalid restart metadata must not launch a calculation")

    monkeypatch.setattr(adapter.subprocess, "run", prohibited)
    with pytest.raises(ValueError, match="nonnegative integer"):
        adapter.main([str(dispatch), "0"])
