"""Verify the distribution in an isolated environment outside the source tree."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.packaging
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def distributions(tmp_path_factory):
    """Build and validate both artifacts once for the isolated install matrix."""
    destination = tmp_path_factory.mktemp("distributions")
    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(destination)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [sys.executable, "-m", "twine", "check", *map(str, destination.iterdir())],
        check=True,
        capture_output=True,
    )
    return {"wheel": next(destination.glob("*.whl")), "sdist": next(destination.glob("*.tar.gz"))}


@pytest.mark.parametrize("artifact_kind", ["wheel", "sdist"])
def test_distribution_install_without_optional_dependencies(distributions, artifact_kind, tmp_path):
    venv = tmp_path / "installed"
    subprocess.run(["uv", "venv", "--python", sys.executable, str(venv)], check=True, capture_output=True)
    interpreter = venv / "bin" / "python"
    subprocess.run(
        ["uv", "pip", "install", "--python", str(interpreter), str(distributions[artifact_kind])],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("QUEER_PROJECT_ROOT", None)
    env.pop("QUEER_DATA_DIR", None)
    env["QUEER_RESULTS_DIR"] = str(tmp_path / "results")
    probe = """
import sys
from importlib.resources import files
from pathlib import Path
import queer
from queer import WannierModel, EPWModel
assert queer.__version__ == "1.1.0"
assert Path(queer.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert files("queer").joinpath("resources/neon.mplstyle").is_file()
assert not any(name in sys.modules for name in ("matplotlib", "pandas", "seekpath", "nbformat"))
"""
    subprocess.run([str(interpreter), "-c", probe], cwd=tmp_path, env=env, check=True, capture_output=True)
    subprocess.run(
        [str(venv / "bin" / "queer"), "--help"], cwd=tmp_path, env=env, check=True, capture_output=True
    )
    # Inputs also live outside the checkout, so execution cannot rely on source-tree paths.
    material = tmp_path / "material"
    shutil.copytree(ROOT / "tests/fixtures/wannier/chain", material)
    config = tmp_path / "bands.toml"
    config.write_text(f'''workflow="bands"
[inputs]
material_dir="{material}"
[parameters]
ef=0.0
interpolation="standard"
kpoints=[[0.0,0.0,0.0],[0.125,0.0,0.0]]
''')
    subprocess.run(
        [str(venv / "bin" / "queer"), "bands", "--config", str(config)],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
    )
    manifests = list((tmp_path / "results").glob("bands/*/manifest.json"))
    assert len(manifests) == 1
    numerical_probe = f"""
import json
from pathlib import Path
import numpy as np
manifest_path = Path({str(manifests[0])!r})
manifest = json.loads(manifest_path.read_text())
assert manifest["status"] == "completed"
with np.load(manifest_path.parent / manifest["cases"]["bands"]["path"], allow_pickle=False) as arrays:
    np.testing.assert_allclose(arrays["energies"], [[0.3, 0.07071067811865475]], atol=1e-12, rtol=1e-12)
"""
    subprocess.run(
        [str(interpreter), "-c", numerical_probe], cwd=tmp_path, env=env, check=True, capture_output=True
    )
