"""Every Generator step runs from a staged QUEER distribution, as on a cluster host.

The Generator ships ``src/queer`` and a ``queer-<version>.dist-info`` as one zip,
extracts it with ``python -m zipfile -e`` and runs ``python -m queer generator``
with only the extracted directory on PYTHONPATH, in a job workspace. The test
builds that zip from this checkout, extracts it the same way and runs
``capabilities`` and each step in subprocesses whose working directory is
outside the checkout, so the checkout's ``src`` cannot stand in for the copy.
"""

import base64
import hashlib
import json
import os
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "queer"
sys.path.insert(0, str(ROOT / "tests" / "numerical"))

from tight_binding_models import haldane  # noqa: E402
from wannier90_files import Wannier90Job  # noqa: E402

LAYER_PATH = {
    "segments": [
        ["G", [0, 0, 0], "M", [0.5, 0, 0]],
        ["M", [0.5, 0, 0], "K", [1 / 3, 2 / 3, 0]],
        ["K", [1 / 3, 2 / 3, 0], "G", [0, 0, 0]],
    ]
}
MODEL = {"fermi_energy": 0.0, "workers": 1, "ws_correction": "off"}
STEPS = {
    "QUEER_BANDS": {"kpath": LAYER_PATH, "kpath_points": 24, "dos_mesh": 8},
    "QUEER_TOPOLOGY": {"kpath": LAYER_PATH, "loop_points": 11, "transverse_points": 6},
    "QUEER_BERRY": {"kpath": LAYER_PATH, "kpath_points": 24, "berry_mesh": 8},
    "QUEER_SLAB": {"slab_kpoints": 12, "slab_cells": 4},
}
STEP_BLOCKS = {
    "QUEER_BANDS": "bands",
    "QUEER_TOPOLOGY": "topology",
    "QUEER_BERRY": "berry",
    "QUEER_SLAB": "slab",
}


def record_digest(data):
    """A wheel RECORD hash: urlsafe base64 SHA-256 without padding."""
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"sha256={digest}"


def build_staged_zip(target):
    """``src/queer/**`` (no ``__pycache__``) plus a minimal dist-info; returns the version and RECORD bytes."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    version = project["version"]
    info = f"queer-{version}.dist-info"
    rows = []
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(PACKAGE.rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts:
                continue
            name = path.relative_to(PACKAGE.parent).as_posix()
            data = path.read_bytes()
            archive.writestr(name, data)
            rows.append(f"{name},{record_digest(data)},{len(data)}")
        requirements = "".join(f"Requires-Dist: {requirement}\n" for requirement in project["dependencies"])
        metadata = f"Metadata-Version: 2.1\nName: queer\nVersion: {version}\n{requirements}".encode()
        archive.writestr(f"{info}/METADATA", metadata)
        rows.append(f"{info}/METADATA,{record_digest(metadata)},{len(metadata)}")
        rows.append(f"{info}/RECORD,,")
        record = ("\n".join(rows) + "\n").encode()
        archive.writestr(f"{info}/RECORD", record)
    return version, record


def environment(site):
    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
    env.update(
        PYTHONPATH=str(site),
        MPLBACKEND="Agg",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONNOUSERSITE="1",
    )
    return env


def python(arguments, cwd, env):
    return subprocess.run(
        [sys.executable, *arguments], cwd=cwd, env=env, capture_output=True, text=True, timeout=120
    )


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    """The extracted distribution, its version and RECORD digest, and an environment that sees only it."""
    root = tmp_path_factory.mktemp("staged")
    archive = root / "queer-staged.zip"
    version, record = build_staged_zip(archive)
    site = root / "site"
    extracted = subprocess.run(
        [sys.executable, "-m", "zipfile", "-e", str(archive), str(site)], capture_output=True, text=True
    )
    assert extracted.returncode == 0, extracted.stderr
    env = environment(site)
    probe = python(
        [
            "-c",
            "import importlib.metadata as m, json, queer; "
            "print(json.dumps([queer.__file__, str(m.distribution('queer').locate_file('')), "
            "m.version('queer'), m.requires('queer')]))",
        ],
        root,
        env,
    )
    assert probe.returncode == 0, probe.stderr
    module, located, installed, requires = json.loads(probe.stdout)
    assert Path(module).resolve() == (site / "queer" / "__init__.py").resolve()
    assert Path(located).resolve() == site.resolve()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert installed == version and requires == project["dependencies"]
    return {"root": root, "site": site, "env": env, "version": version, "record": record}


def test_capabilities_of_the_staged_copy_list_every_step(staged):
    finished = python(["-m", "queer", "generator", "capabilities"], staged["root"], staged["env"])
    assert finished.returncode == 0, finished.stderr
    assert json.loads(finished.stdout) == {"queer_version": staged["version"], "steps": sorted(STEPS)}


@pytest.mark.parametrize("step", sorted(STEPS))
def test_each_step_runs_from_the_staged_copy(staged, step):
    job = staged["root"] / step.lower()
    workspace = job / "ws"
    workspace.mkdir(parents=True)
    model = haldane(staggered=0.2)
    Wannier90Job(model.translations, model.hopping_matrices, model.lattice, model.centres).write(workspace)
    (job / "params.json").write_text(json.dumps({**MODEL, **STEPS[step]}))
    finished = python(
        ["-m", "queer", "generator", "run", "--step", step, "--params", "params.json", "--workspace", "ws"],
        job,
        staged["env"],
    )
    document = json.loads((workspace / "queer_results.json").read_text())
    assert finished.returncode == 0, (finished.stderr[-2000:], document.get("error"))
    assert document["status"] == "ok" and document["engine"] == "queer"
    assert document["step_type"] == step and document["counterpart_step_type"] == step.replace(
        "QUEER", "PYTHTB"
    )
    provenance = document["provenance"]
    assert provenance["queer_version"] == staged["version"]
    assert provenance["queer_record_sha256"] == hashlib.sha256(staged["record"]).hexdigest()
    assert document["model"]["queer_version"] == staged["version"]
    assert STEP_BLOCKS[step] in document
    paths = [entry["path"] for entry in document["outputs"]]
    assert paths and all((workspace / path).is_file() for path in paths), paths
    assert any(path.endswith(".json") for path in paths), paths
