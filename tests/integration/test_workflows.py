"""Integration coverage for reproducible, independent scientific run artifacts."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from queer.config import config_from_mapping, load_config
from queer.io.artifacts import ALGORITHM_VERSION, RunStore, atomic_npz, digest_file
from queer.workflows import render_run, run_workflow

ROOT = Path(__file__).resolve().parents[2]


def band_config(tmp_path):
    return load_config(ROOT / "configs/examples/bands.toml", overrides={"results_dir": tmp_path})


@pytest.mark.parametrize(
    "workflow,case_count", [("bands", 1), ("susceptibility", 2), ("spectral", 3), ("momentum", 4), ("epw", 2)]
)
def test_fixture_workflows_persist_and_render(workflow, case_count, tmp_path):
    config = load_config(ROOT / f"configs/examples/{workflow}.toml", overrides={"results_dir": tmp_path})
    run = run_workflow(config)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["algorithm_version"] == ALGORITHM_VERSION
    assert len(manifest["cases"]) == case_count
    assert manifest["inputs"] and manifest["versions"]["numpy"]
    attempt = manifest["attempts"][-1]
    assert attempt["elapsed_seconds"] >= 0
    assert attempt["resolved_execution"] == config.execution.resolved().as_dict()
    for record in manifest["cases"].values():
        assert record["completed_at"]
        assert 0 <= record["elapsed_since_previous_checkpoint_seconds"] <= attempt["elapsed_seconds"]
        artifact = run / record["path"]
        assert digest_file(artifact) == record["sha256"]
        with np.load(artifact, allow_pickle=False) as arrays:
            assert all(np.isfinite(value).all() for value in arrays.values() if value.dtype.kind in "fci")
    figures = render_run(run)
    assert list(figures.glob("*.png"))
    assert (figures / "manifest.json").is_file()


def test_run_directories_do_not_collide(tmp_path):
    config = band_config(tmp_path)
    first, second = run_workflow(config), run_workflow(config)
    assert first != second
    with pytest.raises(FileExistsError):
        run_workflow(config, run_dir=first)


def test_completed_resume_skips_energy_and_render_has_no_model_dependency(tmp_path, monkeypatch):
    from queer.models.wannier import WannierModel

    config = band_config(tmp_path)
    run = run_workflow(config)
    numerical = run / "arrays/bands.npz"
    original = numerical.stat().st_mtime_ns

    def prohibited(*args, **kwargs):
        raise AssertionError("Must not recalculate a completed case")

    monkeypatch.setattr(WannierModel, "energies", prohibited)
    assert run_workflow(config, run_dir=run, resume=True) == run
    assert numerical.stat().st_mtime_ns == original
    monkeypatch.setattr(WannierModel, "from_directory", prohibited)
    assert render_run(run).is_dir()


def test_partial_failure_resumes_only_missing_cases(tmp_path, monkeypatch):
    import queer.observables.response as response

    config = load_config(ROOT / "configs/examples/susceptibility.toml", overrides={"results_dir": tmp_path})
    original = response.susceptibility
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("Injected interrupted calculation")
        return original(*args, **kwargs)

    monkeypatch.setattr(response, "susceptibility", fail_second)
    run = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="Injected"):
        run_workflow(config, run_dir=run)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert len(manifest["cases"]) == 1
    first = run / "arrays/q_000000.npz"
    checksum = digest_file(first)
    monkeypatch.setattr(response, "susceptibility", original)
    run_workflow(config, run_dir=run, resume=True)
    assert digest_file(first) == checksum
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert len(manifest["attempts"]) == 2


def test_resume_rejects_changed_science_input_or_artifact(tmp_path):
    config = band_config(tmp_path)
    run = run_workflow(config)
    changed = config.as_dict()
    changed["parameters"]["ef"] = 1.0
    with pytest.raises(ValueError, match="Resume rejected"):
        run_workflow(config_from_mapping(changed), run_dir=run, resume=True)
    numerical = run / "arrays/bands.npz"
    numerical.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="artifact"):
        run_workflow(config, run_dir=run, resume=True)


def test_store_refuses_competing_owner_and_recovers_after_release(tmp_path):
    config = band_config(tmp_path)
    source = config.material_dir / "wannier90_hr.dat"
    run = tmp_path / "locked"
    first = RunStore(run, config.as_dict(), [source])
    try:
        with pytest.raises(ValueError, match="owned"):
            RunStore(run, config.as_dict(), [source], resume=True)
    finally:
        first.release()
    second = RunStore(run, config.as_dict(), [source], resume=True)
    second.release()


def test_atomic_artifacts_reject_python_objects(tmp_path):
    target = tmp_path / "array.npz"
    atomic_npz(target, {"values": [1.0, 2.0]})
    checksum = digest_file(target)
    with pytest.raises(ValueError, match="objects"):
        atomic_npz(target, {"values": [object()]})
    assert digest_file(target) == checksum


def test_resume_rejects_changed_nested_epw_channel(tmp_path):
    copied = tmp_path / "inputs"
    shutil.copytree(ROOT / "tests/fixtures/epw", copied)
    config = load_config(
        ROOT / "configs/examples/epw.toml", overrides={"material_dir": copied, "results_dir": tmp_path}
    )
    run = run_workflow(config)
    branch = next((copied / "gkk_re").iterdir())
    branch.write_text(branch.read_text() + "\n")
    with pytest.raises(ValueError, match="Resume rejected"):
        run_workflow(config, run_dir=run, resume=True)


def test_resume_rejects_changed_algorithm_version(tmp_path, monkeypatch):
    import queer.io.artifacts as artifacts

    config = band_config(tmp_path)
    run = run_workflow(config)
    monkeypatch.setattr(artifacts, "ALGORITHM_VERSION", "next-algorithm")
    with pytest.raises(ValueError, match="Resume rejected"):
        run_workflow(config, run_dir=run, resume=True)


def test_resume_rejects_changed_input_roles_with_identical_file_set(tmp_path):
    config = band_config(tmp_path)
    run = run_workflow(config)
    changed = config.as_dict()
    changed["inputs"]["hr"], changed["inputs"]["poscar"] = (
        changed["inputs"]["poscar"],
        changed["inputs"]["hr"],
    )
    with pytest.raises(ValueError, match="Resume rejected"):
        run_workflow(config_from_mapping(changed), run_dir=run, resume=True)


def test_resume_records_each_environment_and_checkpoint_owner(tmp_path, monkeypatch):
    import queer.io.artifacts as artifacts
    import queer.observables.response as response

    config = load_config(ROOT / "configs/examples/susceptibility.toml", overrides={"results_dir": tmp_path})
    first_versions = {"queer": "1.1.0", "numpy": "1.26.0", "source": {"commit": "first", "dirty": False}}
    second_versions = {"queer": "1.1.1", "numpy": "2.2.0", "source": {"commit": "second", "dirty": False}}
    monkeypatch.setattr(artifacts, "environment_versions", lambda: first_versions)
    monkeypatch.setattr(artifacts.socket, "gethostname", lambda: "first-node")
    monkeypatch.setenv("SLURM_JOB_ID", "101")
    original = response.susceptibility
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("First environment interrupted")
        return original(*args, **kwargs)

    monkeypatch.setattr(response, "susceptibility", fail_second)
    run = tmp_path / "changing-environment"
    with pytest.raises(RuntimeError, match="First environment"):
        run_workflow(config, run_dir=run)
    monkeypatch.setattr(response, "susceptibility", original)
    monkeypatch.setattr(artifacts, "environment_versions", lambda: second_versions)
    monkeypatch.setattr(artifacts.socket, "gethostname", lambda: "second-node")
    monkeypatch.setenv("SLURM_JOB_ID", "202")
    run_workflow(config, run_dir=run, resume=True)
    manifest = json.loads((run / "manifest.json").read_text())
    first, second = manifest["attempts"]
    assert first["versions"] == manifest["versions"] == first_versions
    assert second["versions"] == second_versions
    assert first["host"] == "first-node" and second["host"] == "second-node"
    assert first["scheduler"]["SLURM_JOB_ID"] == "101"
    assert second["scheduler"]["SLURM_JOB_ID"] == "202"
    assert first["status"] == "failed" and second["status"] == "completed"
    assert first["error"]["message"] == "First environment interrupted"
    assert manifest["cases"]["q_000000"]["attempt_index"] == 0
    assert manifest["cases"]["q_000001"]["attempt_index"] == 1
    assert all(
        attempt["elapsed_seconds"] >= 0 and "resolved_execution" in attempt
        for attempt in manifest["attempts"]
    )


def test_advisory_lock_recovers_after_killed_owner(tmp_path):
    run = tmp_path / "killed-owner"
    script = """
import sys, time
from pathlib import Path
from queer.config import load_config
from queer.io.artifacts import RunStore
config = load_config(sys.argv[1])
store = RunStore(Path(sys.argv[2]), config.as_dict(), [config.material_dir / 'wannier90_hr.dat'])
print('locked', flush=True)
time.sleep(30)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(ROOT / "configs/examples/bands.toml"), str(run)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "locked"
        child.kill()
        child.wait(timeout=10)
        config = band_config(tmp_path)
        recovered = RunStore(run, config.as_dict(), [config.material_dir / "wannier90_hr.dat"], resume=True)
        recovered.release()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


@pytest.mark.parametrize("omega", [[0.0, -1.0, 2.5], [-1.0, 0.0, 2.5], [-1.0, 2.5], [0.0]])
def test_curved_spectral_sweep_persists_each_geometry_and_renders(omega, tmp_path, monkeypatch):
    from queer.models.wannier import WannierModel

    data = load_config(ROOT / "configs/examples/spectral_arpes.toml").as_dict()
    data["parameters"]["omega"] = omega
    config = config_from_mapping(data, overrides={"results_dir": tmp_path})
    run = run_workflow(config)
    manifest = json.loads((run / "manifest.json").read_text())
    assert len(manifest["cases"]) == 2 * len(omega)
    for record in manifest["cases"].values():
        assert record["metadata"]["geometry"] == "curved_ARPES"
        with np.load(run / record["path"]) as arrays:
            assert arrays["spectrum"].shape == (1, 3)
            assert np.isfinite(arrays["spectrum"]).all()

    def prohibited(*args, **kwargs):
        raise AssertionError("Must not recalculate completed spectra")

    monkeypatch.setattr(WannierModel, "energies", prohibited)
    run_workflow(config, run_dir=run, resume=True)
    figures = render_run(run)
    assert len(list(figures.glob("*.png"))) == 2


def test_plain_spectral_reuses_fixed_path_eigensolutions(tmp_path, monkeypatch):
    from queer.models.wannier import WannierModel

    original = WannierModel.energies
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(WannierModel, "energies", counted)
    config = load_config(ROOT / "configs/examples/spectral.toml", overrides={"results_dir": tmp_path})
    run_workflow(config)
    assert len(calls) == 1


def test_curved_spectral_commits_each_slice_before_later_failure(tmp_path, monkeypatch):
    import queer.observables.spectral as spectral

    config = load_config(ROOT / "configs/examples/spectral_arpes.toml", overrides={"results_dir": tmp_path})
    original = spectral.iter_spectral_arpes_path_sweep

    def fail_after_first(*args, **kwargs):
        iterator = original(*args, **kwargs)
        yield next(iterator)
        raise RuntimeError("Injected later spectral failure")

    monkeypatch.setattr(spectral, "iter_spectral_arpes_path_sweep", fail_after_first)
    run = tmp_path / "spectral-partial"
    with pytest.raises(RuntimeError, match="spectral failure"):
        run_workflow(config, run_dir=run)
    manifest = json.loads((run / "manifest.json").read_text())
    assert len(manifest["cases"]) == 1
    first = run / "arrays/v0_0000_omega_000000.npz"
    previous = first.stat().st_mtime_ns
    monkeypatch.setattr(spectral, "iter_spectral_arpes_path_sweep", original)
    run_workflow(config, run_dir=run, resume=True)
    assert first.stat().st_mtime_ns == previous
    manifest = json.loads((run / "manifest.json").read_text())
    assert len(manifest["cases"]) == 6
