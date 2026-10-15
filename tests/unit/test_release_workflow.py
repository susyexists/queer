"""The tag workflow publishes exactly the CI-built distributions with their checksums."""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
JOBS = WORKFLOW["jobs"]


def scripts(job):
    return [step.get("run", "") for step in JOBS[job]["steps"]]


def first(job, text):
    return next(index for index, script in enumerate(scripts(job)) if text in script)


def test_release_job_records_the_tagged_commit_before_testing_and_building():
    record = first("release", "src/queer/_build.py")
    assert "GITHUB_SHA" in scripts("release")[record]
    assert record < first("release", "pytest -m packaging") < first("release", "-m build")
    packaging = JOBS["release"]["steps"][first("release", "pytest -m packaging")]
    assert packaging["env"]["QUEER_EXPECTED_BUILD_SHA"] == "${{ github.sha }}"
    assert first("release", "-m build") < first("release", 'read("queer/_build.py")')
    upload = JOBS["release"]["steps"][-1]
    assert upload["uses"].startswith("actions/upload-artifact@")
    assert upload["with"]["name"] == "queer-${{ github.ref_name }}"


def test_publish_job_releases_the_built_files_and_their_checksums():
    publish = JOBS["publish"]
    assert WORKFLOW["permissions"] == {"contents": "read"}
    assert publish["permissions"] == {"contents": "write"}
    assert {"acceptance", "backend-acceptance", "release"} <= set(publish["needs"])
    download = publish["steps"][0]
    assert download["uses"].startswith("actions/download-artifact@")
    assert download["with"]["name"] == JOBS["release"]["steps"][-1]["with"]["name"]
    checksum = scripts("publish")[first("publish", "SHA256SUMS")]
    assert "sha256sum --" in checksum and "sha256sum --check SHA256SUMS" in checksum
    release = publish["steps"][first("publish", "gh release create")]
    assert release["env"]["GH_TOKEN"] == "${{ secrets.GITHUB_TOKEN }}"
    for asset in ("dist/queer-*.whl", "dist/queer-*.tar.gz", "dist/SHA256SUMS", "--verify-tag"):
        assert asset in release["run"]
    assert not any("-m build" in script for script in scripts("publish"))


def test_acceptance_matrix_is_unchanged():
    assert JOBS["acceptance"]["strategy"]["matrix"]["python"] == ["3.11", "3.12", "3.13", "3.14"]
    assert JOBS["release"]["needs"] == ["acceptance", "backend-acceptance"]
