"""Execute every active notebook cell with committed scientific fixtures."""

import json
import os
import sys
from pathlib import Path

import nbformat
import pytest
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.notebook
def test_active_notebook_executes_from_fresh_kernel(tmp_path, monkeypatch):
    config = tmp_path / "notebook.toml"
    config.write_text(f'''workflow="bands"
[inputs]
material_dir="{ROOT / "tests/fixtures/wannier/fcc"}"
poscar="POSCAR"
[parameters]
ef=0.0
interpolation="standard"
kpath="GAMMA-X-W-K-GAMMA-L"
n_points=16
''')
    monkeypatch.setenv("QUEER_NOTEBOOK_CONFIG", str(config))
    monkeypatch.setenv("QUEER_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
    specdir = tmp_path / "kernels" / "queer-test"
    specdir.mkdir(parents=True)
    (specdir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "QUEER test",
                "language": "python",
            }
        )
    )
    manager = KernelManager(
        kernel_name="queer-test", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(specdir.parent)])
    )
    notebook = nbformat.read(ROOT / "notebooks/active/Ag_primitive_single.ipynb", as_version=4)
    NotebookClient(
        notebook, km=manager, timeout=180, resources={"metadata": {"path": str(tmp_path)}}
    ).execute()
    assert all(
        output.get("output_type") != "error" for cell in notebook.cells for output in cell.get("outputs", [])
    )
    manifests = list((tmp_path / "results").glob("bands/*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["status"] == "completed"
    assert (manifests[0].parent / "arrays" / "bands.npz").is_file()
    assert list((manifests[0].parent / "figures").glob("*.png"))
