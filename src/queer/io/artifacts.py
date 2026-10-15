"""Atomic numerical artifacts and resumable run manifests."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import tempfile
import time
import tomllib
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO, cast
from uuid import uuid4

import numpy as np

from ..execution import ExecutionOptions

ALGORITHM_VERSION = "1.4.1-refactor-v1"
RUNSTORE_LOCK_MODES = ("flock", "none")
# The repository root when this module runs from a source checkout.
_SOURCE_ROOT = Path(__file__).resolve().parents[3]


def scheduler_context() -> dict[str, str]:
    return {
        key: os.environ[key]
        for key in (
            "SLURM_JOB_ID",
            "SLURM_ARRAY_JOB_ID",
            "SLURM_ARRAY_TASK_ID",
            "SLURM_CPUS_PER_TASK",
        )
        if key in os.environ
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    """Commit arrays without pickle; rejected object arrays cannot enter artifacts."""
    converted = {name: np.asarray(value) for name, value in arrays.items()}
    if set(converted) & {"file", "allow_pickle"}:
        raise ValueError("Array names file and allow_pickle are reserved by the NumPy writer")
    if any(value.dtype.hasobject for value in converted.values()):
        raise ValueError("Run arrays must not contain Python objects")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            # NumPy 2 added allow_pickle; object rejection keeps this compatible
            # with 1.26 without storing an unwanted allow_pickle array there.
            cast(Callable[..., Any], np.savez_compressed)(stream, **converted)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _source_checkout(root: Path) -> bool:
    """Whether ``root`` is a QUEER source checkout: its project file and Git metadata."""
    try:
        with (root / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream).get("project")
    except (OSError, tomllib.TOMLDecodeError):
        return False
    return isinstance(project, dict) and project.get("name") == "queer" and (root / ".git").exists()


def _source_revision(root: Path) -> dict[str, Any] | None:
    if not _source_checkout(root):
        return None
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"commit": commit, "dirty": dirty}


def distribution_identity() -> dict[str, Any]:
    """Version, installed RECORD digest and release build revision of QUEER.

    ``record_sha256`` hashes the raw bytes of the installed ``RECORD`` and is
    ``None`` without an installed distribution. ``build_sha`` is the revision a
    release build wrote into ``queer._build``; source checkouts carry ``None``.
    """
    version = None
    record = None
    try:
        distribution = importlib.metadata.distribution("queer")
    except importlib.metadata.PackageNotFoundError:
        distribution = None
    if distribution is not None:
        version = distribution.version
        for entry in distribution.files or ():
            if entry.name == "RECORD" and entry.parent.name.endswith(".dist-info"):
                try:
                    content = Path(str(distribution.locate_file(entry))).read_bytes()
                except OSError:
                    break
                record = hashlib.sha256(content).hexdigest()
                break
    if version is None:
        from .. import __version__ as version
    try:
        from .._build import BUILD_SHA
    except ImportError:
        BUILD_SHA = None
    build = BUILD_SHA if isinstance(BUILD_SHA, str) and BUILD_SHA else None
    return {"version": version, "record_sha256": record, "build_sha": build}


def environment_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for package in (
        "queer",
        "numpy",
        "scipy",
        "joblib",
        "spglib",
        "mpmath",
        "numba",
        "torch",
        "mace-torch",
        "torch-geometric",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "uninstalled"
    # Git describes this code only when it runs from a QUEER checkout; an
    # installed copy may sit inside an unrelated repository.
    versions["source"] = _source_revision(_SOURCE_ROOT)
    versions["queer_dist"] = distribution_identity()
    return versions


def runstore_lock_mode() -> str:
    """Run ownership from ``QUEER_RUNSTORE_LOCK``: ``flock`` (default) or ``none``."""
    mode = os.environ.get("QUEER_RUNSTORE_LOCK", "flock").strip().lower() or "flock"
    if mode not in RUNSTORE_LOCK_MODES:
        raise ValueError(f"QUEER_RUNSTORE_LOCK must be one of {', '.join(RUNSTORE_LOCK_MODES)}, not {mode!r}")
    return mode


class RunStore:
    """An exclusively owned run with validated incremental completion records.

    Ownership uses an advisory ``flock`` on ``.run.lock``; with
    ``QUEER_RUNSTORE_LOCK=none`` the caller guarantees a single writer instead,
    which each attempt records as its ``lock``. Case commits are durable when
    they return, while the manifest listing them is rewritten in batches: after
    ``manifest_batch_commits`` pending commits (or an eighth of the recorded
    cases, whichever is larger), once ``manifest_batch_seconds`` have passed
    since the last write, and on every explicit save, finish or release. A
    killed process therefore loses at most the commits of its unsaved batch;
    resuming recomputes them and atomically replaces their arrays.
    """

    manifest_batch_commits = 64
    manifest_batch_seconds = 30.0

    def __init__(self, root: Path, config: dict[str, Any], input_files: list[Path], *, resume: bool = False):
        self.lock_mode = runstore_lock_mode()
        self.started_at = time.monotonic()
        self.previous_checkpoint_at = self.started_at
        self.saved_at = self.started_at
        self.pending_commits = 0
        self.root = root
        self.lock = root / ".run.lock"
        self.owned = False
        self.lock_stream: TextIO | None = None
        self.config = config
        attempt = {
            "started_at": utc_now(),
            "status": "running",
            "execution": config["execution"],
            "resolved_execution": ExecutionOptions(**config["execution"]).resolved().as_dict(),
            "versions": environment_versions(),
            "host": socket.gethostname(),
            "scheduler": scheduler_context(),
            "lock": self.lock_mode,
        }
        inputs = [
            {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": digest_file(path)}
            for path in sorted(set(input_files))
        ]
        # Execution, output placement and reusable cache placement do not alter
        # scientific identity. Cache contents are validated independently.
        scientific_parameters = {
            key: value for key, value in config["parameters"].items() if key != "cache_dir"
        }
        identity = {
            "algorithm_version": ALGORITHM_VERSION,
            "workflow": config["workflow"],
            "parameters": scientific_parameters,
            "input_options": config["inputs"],
            "inputs": inputs,
        }
        self.fingerprint = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
        if resume:
            if not (root / "manifest.json").is_file():
                raise ValueError("--resume requires an existing run manifest")
        else:
            root.mkdir(parents=True, exist_ok=False)
        try:
            self._acquire()
            if resume:
                self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
                if (
                    self.manifest.get("schema_version") != 1
                    or self.manifest.get("fingerprint") != self.fingerprint
                ):
                    raise ValueError("Resume rejected: scientific configuration or input checksums changed")
                self.manifest.setdefault("attempts", []).append(attempt)
            else:
                self.manifest = {
                    "schema_version": 1,
                    "algorithm_version": ALGORITHM_VERSION,
                    "workflow": config["workflow"],
                    "fingerprint": self.fingerprint,
                    "created_at": utc_now(),
                    "config": config,
                    "inputs": inputs,
                    "versions": attempt["versions"],
                    "host": attempt["host"],
                    "scheduler": attempt["scheduler"],
                    "cases": {},
                    "attempts": [attempt],
                }
                atomic_json(root / "config.json", config)
            self.manifest["status"] = "running"
            self.manifest["lock"] = self.lock_mode
            self.manifest.pop("error", None)
            self.save()
        except BaseException:
            self.release()
            raise

    def _acquire(self) -> None:
        if self.lock_mode == "none":
            self.owned = True
            return
        stream = self.lock.open("a+")
        self.lock_stream = stream
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.close()
            self.lock_stream = None
            raise ValueError(f"Run is owned by another process: {self.root}") from exc
        except OSError as exc:
            stream.close()
            self.lock_stream = None
            raise OSError(
                exc.errno,
                f"Run storage at {self.root} does not support advisory file locking ({exc.strerror}); "
                "set QUEER_RUNSTORE_LOCK=none if no other process writes this run",
            ) from exc
        self.owned = True
        stream.seek(0)
        stream.truncate()
        json.dump({"pid": os.getpid(), "host": socket.gethostname()}, stream)
        stream.flush()

    def save(self) -> None:
        self.manifest["updated_at"] = utc_now()
        atomic_json(self.root / "manifest.json", self.manifest)
        self.pending_commits = 0
        self.saved_at = time.monotonic()

    def flush(self) -> None:
        """Write the manifest if commits are pending."""
        if self.pending_commits:
            self.save()

    def completed(self, case: str) -> bool:
        record = self.manifest["cases"].get(case)
        if not record:
            return False
        artifact = self.root / record["path"]
        if not artifact.is_file() or digest_file(artifact) != record["sha256"]:
            raise ValueError(f"Resume rejected: completed artifact was changed or removed: {case}")
        return True

    def commit(
        self, case: str, arrays: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None
    ) -> None:
        relative = Path("arrays") / f"{case}.npz"
        artifact = self.root / relative
        atomic_npz(artifact, arrays)
        checkpoint_at = time.monotonic()
        self.manifest["cases"][case] = {
            "path": str(relative),
            "sha256": digest_file(artifact),
            "metadata": dict(metadata or {}),
            "attempt_index": len(self.manifest["attempts"]) - 1,
            "completed_at": utc_now(),
            "elapsed_since_previous_checkpoint_seconds": checkpoint_at - self.previous_checkpoint_at,
        }
        self.previous_checkpoint_at = checkpoint_at
        self.pending_commits += 1
        batch = max(self.manifest_batch_commits, len(self.manifest["cases"]) // 8)
        if self.pending_commits >= batch or checkpoint_at - self.saved_at >= self.manifest_batch_seconds:
            self.save()

    def finish(self, *, status: str = "completed", error: BaseException | None = None) -> None:
        self.manifest["status"] = status
        self.manifest["attempts"][-1]["status"] = status
        self.manifest["attempts"][-1]["finished_at"] = utc_now()
        self.manifest["attempts"][-1]["elapsed_seconds"] = time.monotonic() - self.started_at
        if error:
            self.manifest["error"] = {"type": type(error).__name__, "message": str(error)}
            self.manifest["attempts"][-1]["error"] = self.manifest["error"]
        elif status == "completed":
            self.manifest.pop("error", None)
        self.save()

    def release(self) -> None:
        if not self.owned:
            return
        try:
            self.flush()
        finally:
            if self.lock_stream is not None:
                fcntl.flock(self.lock_stream, fcntl.LOCK_UN)
                self.lock_stream.close()
                self.lock_stream = None
            self.owned = False


def new_run_directory(results_dir: Path, workflow: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return results_dir / workflow / f"{stamp}_{uuid4().hex[:12]}"


def load_manifest(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("status") != "completed":
        raise ValueError("Rendering requires a completed, supported run manifest")
    for case, record in manifest["cases"].items():
        artifact = root / record["path"]
        if not artifact.is_file() or digest_file(artifact) != record["sha256"]:
            raise ValueError(f"Artifact was changed or removed: {case}")
    return manifest


@contextmanager
def owned_run(store):
    """Finish and release one exclusively owned run on success or interruption."""
    try:
        yield store
        store.finish()
    except BaseException as error:
        store.finish(
            status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
            error=error,
        )
        raise
    finally:
        store.release()
