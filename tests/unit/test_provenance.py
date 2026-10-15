"""Run provenance identifies source checkouts and installed distributions without guessing."""

import hashlib
import importlib.metadata
import subprocess

import pytest

import queer
import queer._build
import queer.io.artifacts as artifacts


class FakeDistribution:
    def __init__(self, root, version="9.8.7", record=b"queer/__init__.py,sha256=x,1\r\n"):
        self.version = version
        self.root = root
        info = root / f"queer-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "RECORD").write_bytes(record)
        self.files = [
            importlib.metadata.PackagePath("queer/__init__.py"),
            importlib.metadata.PackagePath(f"queer-{version}.dist-info/RECORD"),
        ]

    def locate_file(self, path):
        return self.root / path


def git_only(handler):
    """Route git commands to ``handler``; platform probes keep the real subprocess."""
    original = subprocess.run

    def run(command, *args, **kwargs):
        if command and command[0] == "git":
            return handler(command)
        return original(command, *args, **kwargs)

    return run


def project(root, name, *, git):
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "0"\n')
    if git:
        (root / ".git").write_text("gitdir: elsewhere\n")
    return root


@pytest.mark.parametrize(
    "layout",
    ["no_project", "other_project", "queer_without_git", "unreadable_project"],
)
def test_git_runs_only_in_a_queer_source_checkout(tmp_path, monkeypatch, layout):
    root = tmp_path / layout
    if layout == "no_project":
        root.mkdir()
        (root / ".git").mkdir()
    elif layout == "other_project":
        project(root, "other", git=True)
    elif layout == "queer_without_git":
        project(root, "queer", git=False)
    else:
        project(root, "queer", git=True)
        (root / "pyproject.toml").write_text("[project\nname = 'queer'\n")

    def prohibited(command):
        raise AssertionError("git must not describe an unrelated directory")

    monkeypatch.setattr(artifacts, "_SOURCE_ROOT", root)
    monkeypatch.setattr(artifacts.subprocess, "run", git_only(prohibited))
    versions = artifacts.environment_versions()
    assert versions["source"] is None
    assert {"python", "platform", "queer", "numpy", "scipy", "joblib", "queer_dist"} <= set(versions)


def test_source_checkout_records_commit_and_dirty_state(tmp_path, monkeypatch):
    root = project(tmp_path / "checkout", "queer", git=True)
    calls = []

    def fake_git(command):
        calls.append(command)
        output = "0123abcd\n" if command[-1] == "HEAD" else " M src/queer/io/artifacts.py\n"
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(artifacts, "_SOURCE_ROOT", root)
    monkeypatch.setattr(artifacts.subprocess, "run", git_only(fake_git))
    assert artifacts.environment_versions()["source"] == {"commit": "0123abcd", "dirty": True}
    assert calls and all(command[:3] == ["git", "-C", str(root)] for command in calls)

    def failing_git(command):
        raise subprocess.CalledProcessError(128, command)

    monkeypatch.setattr(artifacts.subprocess, "run", git_only(failing_git))
    assert artifacts.environment_versions()["source"] is None


def test_distribution_identity_hashes_record_bytes_and_reads_the_build_revision(tmp_path, monkeypatch):
    record = b"queer/__init__.py,sha256=abc,10\r\nqueer-9.8.7.dist-info/RECORD,,\r\n"
    fake = FakeDistribution(tmp_path / "site", record=record)
    monkeypatch.setattr(artifacts.importlib.metadata, "distribution", lambda name: fake)
    monkeypatch.setattr(queer._build, "BUILD_SHA", "f" * 40)
    identity = artifacts.environment_versions()["queer_dist"]
    assert identity == {
        "version": "9.8.7",
        "record_sha256": hashlib.sha256(record).hexdigest(),
        "build_sha": "f" * 40,
    }
    monkeypatch.setattr(queer._build, "BUILD_SHA", None)
    assert artifacts.distribution_identity()["build_sha"] is None

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(artifacts.importlib.metadata, "distribution", missing)
    assert artifacts.distribution_identity() == {
        "version": queer.__version__,
        "record_sha256": None,
        "build_sha": None,
    }


def test_installed_distribution_record_matches_its_file():
    distribution = importlib.metadata.distribution("queer")
    entry = next(
        path
        for path in distribution.files or ()
        if path.name == "RECORD" and path.parent.suffix == ".dist-info"
    )
    expected = hashlib.sha256(distribution.locate_file(entry).read_bytes()).hexdigest()
    identity = artifacts.distribution_identity()
    assert identity["record_sha256"] == expected
    assert identity["version"] == distribution.version
    assert queer._build.BUILD_SHA is None or isinstance(queer._build.BUILD_SHA, str)
