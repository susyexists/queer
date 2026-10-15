"""Run ownership modes and batched manifest persistence across interruptions."""

import errno
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import queer.io.artifacts as artifacts
from queer.io.artifacts import RunStore, digest_file, owned_run

ROOT = Path(__file__).resolve().parents[2]


def store_config():
    return {
        "workflow": "dos",
        "parameters": {},
        "inputs": {},
        "execution": {"workers": 1, "batch_size": 256, "native_threads": 1, "memory_mb": None},
    }


def disk_manifest(root):
    return json.loads((root / "manifest.json").read_text())


def test_lock_mode_none_skips_advisory_locking_and_is_recorded(tmp_path, monkeypatch):
    def prohibited(*args, **kwargs):
        raise AssertionError("flock must not be called")

    monkeypatch.setenv("QUEER_RUNSTORE_LOCK", "None ")
    monkeypatch.setattr(artifacts.fcntl, "flock", prohibited)
    root = tmp_path / "unlocked"
    with owned_run(RunStore(root, store_config(), [])) as store:
        store.commit("case", {"values": [1.0]})
    manifest = disk_manifest(root)
    assert manifest["status"] == "completed" and manifest["lock"] == "none"
    assert manifest["attempts"][-1]["lock"] == "none"
    assert not (root / ".run.lock").exists()
    # Without locking, ownership is the caller's guarantee rather than a refusal.
    first = RunStore(root, store_config(), [], resume=True)
    second = RunStore(root, store_config(), [], resume=True)
    first.release()
    second.release()


def test_default_lock_mode_is_flock_and_invalid_modes_are_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("QUEER_RUNSTORE_LOCK", raising=False)
    with owned_run(RunStore(tmp_path / "locked", store_config(), [])) as store:
        assert store.lock_stream is not None
    manifest = disk_manifest(tmp_path / "locked")
    assert manifest["lock"] == manifest["attempts"][-1]["lock"] == "flock"
    monkeypatch.setenv("QUEER_RUNSTORE_LOCK", "fcntl")
    with pytest.raises(ValueError, match="QUEER_RUNSTORE_LOCK must be one of flock, none"):
        RunStore(tmp_path / "invalid", store_config(), [])
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("code", [errno.ENOLCK, errno.EOPNOTSUPP])
def test_unsupported_advisory_locking_names_the_override(tmp_path, monkeypatch, code):
    def unsupported(*args, **kwargs):
        raise OSError(code, "lock unavailable")

    monkeypatch.delenv("QUEER_RUNSTORE_LOCK", raising=False)
    monkeypatch.setattr(artifacts.fcntl, "flock", unsupported)
    with pytest.raises(OSError, match="QUEER_RUNSTORE_LOCK=none") as caught:
        RunStore(tmp_path / "nfs", store_config(), [])
    assert caught.value.errno == code


def test_manifest_writes_are_batched_and_flushed_on_release(tmp_path, monkeypatch):
    writes = []
    original = artifacts.atomic_json

    def counting(path, value):
        if path.name == "manifest.json":
            writes.append(len(value["cases"]))
        original(path, value)

    monkeypatch.setattr(artifacts, "atomic_json", counting)
    monkeypatch.setattr(RunStore, "manifest_batch_commits", 4)
    monkeypatch.setattr(RunStore, "manifest_batch_seconds", 3600.0)
    store = RunStore(tmp_path / "batched", store_config(), [])
    for index in range(10):
        store.commit(f"case_{index:02d}", {"values": [float(index)]})
    assert writes == [0, 4, 8]
    assert len(disk_manifest(store.root)["cases"]) == 8
    assert all(store.completed(f"case_{index:02d}") for index in range(10))
    store.release()
    assert writes[-1] == 10 and len(disk_manifest(store.root)["cases"]) == 10
    # A batch also closes once its time budget has passed since the last write.
    monkeypatch.setattr(RunStore, "manifest_batch_seconds", 0.0)
    with owned_run(RunStore(tmp_path / "batched", store_config(), [], resume=True)) as resumed:
        before = len(writes)
        resumed.commit("timed", {"values": [1.0]})
        assert len(writes) == before + 1


def test_large_runs_rewrite_the_manifest_a_bounded_number_of_times(tmp_path, monkeypatch):
    writes = []
    original = artifacts.atomic_json

    def counting(path, value):
        if path.name == "manifest.json":
            writes.append(len(value["cases"]))
        original(path, value)

    monkeypatch.setattr(artifacts, "atomic_json", counting)
    monkeypatch.setattr(RunStore, "manifest_batch_seconds", 3600.0)
    with owned_run(RunStore(tmp_path / "many", store_config(), [])) as store:
        for index in range(1000):
            store.commit(f"cell_{index:05d}", {"values": [float(index)]})
    # Proportional batches keep the total rewritten volume linear in the case count.
    assert len(writes) <= 30 and sum(writes) <= 10_000
    assert writes[-1] == 1000


KILLED_WRITER = """
import os, sys
from pathlib import Path
from queer.io.artifacts import RunStore
RunStore.manifest_batch_commits = 4
RunStore.manifest_batch_seconds = 3600.0
config = {"workflow": "dos", "parameters": {}, "inputs": {}, "execution": {}}
store = RunStore(Path(sys.argv[1]), config, [], resume=sys.argv[2] == "resume")
for index in range(int(sys.argv[3]), int(sys.argv[4])):
    case = f"cell_{index:02d}"
    if not store.completed(case):
        store.commit(case, {"values": [float(index) ** 2]})
if sys.argv[5] == "finish":
    store.finish()
    store.release()
    os._exit(0)
# A killed process runs no cleanup: no finish, release or interpreter exit handlers.
os._exit(9)
"""


def run_writer(root, mode, start, stop, end):
    return subprocess.run(
        [sys.executable, "-c", KILLED_WRITER, str(root), mode, str(start), str(stop), end],
        capture_output=True,
        text=True,
        check=False,
    )


def test_killed_writer_loses_only_its_unsaved_batch_and_resumes(tmp_path):
    root = tmp_path / "killed"
    killed = run_writer(root, "new", 0, 10, "kill")
    assert killed.returncode == 9, killed.stderr
    manifest = disk_manifest(root)
    assert sorted(manifest["cases"]) == [f"cell_{index:02d}" for index in range(8)]
    assert manifest["status"] == "running"
    saved = {case: digest_file(root / record["path"]) for case, record in manifest["cases"].items()}
    # The arrays of the two unsaved commits exist but are not trusted on resume.
    assert (root / "arrays/cell_09.npz").is_file()
    resumed = run_writer(root, "resume", 0, 12, "finish")
    assert resumed.returncode == 0, resumed.stderr
    manifest = disk_manifest(root)
    assert manifest["status"] == "completed"
    assert sorted(manifest["cases"]) == [f"cell_{index:02d}" for index in range(12)]
    assert [record["attempt_index"] for record in manifest["cases"].values()] == [0] * 8 + [1] * 4
    for case, checksum in saved.items():
        assert digest_file(root / manifest["cases"][case]["path"]) == checksum
    for index in range(12):
        with np.load(root / f"arrays/cell_{index:02d}.npz") as arrays:
            assert arrays["values"].tolist() == [float(index) ** 2]
