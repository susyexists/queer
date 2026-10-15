"""Verify the distribution in an isolated environment outside the source tree."""

import os
import shutil
import subprocess
import sys
import tarfile
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


def test_source_distribution_contains_reproducible_native_validation(distributions):
    with tarfile.open(distributions["sdist"]) as archive:
        members = {Path(name).relative_to(Path(name).parts[0]).as_posix() for name in archive.getnames()}
    assert {
        "docs/openmx_workflows.md",
        "docs/scientific_conventions.md",
        "docs/equations.toml",
        "docs/development.md",
        "src/queer/physics/geometry.py",
        "src/queer/models/contracts.py",
        "configs/examples/openmx_bands.toml",
        "configs/examples/openmx_dos.toml",
        "examples/openmx.py",
        "scripts/runs/openmx_benchmark.py",
        "scripts/validation/generator_replay.py",
        "scripts/validation/generator_response_replay.py",
        "scripts/validation/generator_shift_replay.py",
        "scripts/validation/generator_omx_replay.py",
        "scripts/validation/generator_metric_replay.py",
        "scripts/hpc/submit_sweep.py",
        "tests/fixtures/openmx/chain/chain.scfout",
        "tests/numerical/test_openmx.py",
        "tests/numerical/test_optical_nonlinear.py",
        "tests/integration/test_openmx_execution_agreement.py",
        "tests/integration/test_openmx_global_refinement.py",
        "data/references/openmx/manifest.tsv",
        "docs/generator_services.md",
        "docs/native_structures.md",
        "docs/native_atomistics.md",
        "docs/native_responses.md",
        "configs/examples/native_nvt.toml",
        "examples/native_services.py",
        "tests/fixtures/atomistic/POSCAR",
        "docs/finite_field.md",
        "examples/finite_field.py",
        "configs/examples/finite_field_paper_fig4d.toml",
        "scripts/hpc/aggregate_sweep.py",
        "src/queer/finite_field/SOURCE_MANIFEST.json",
    } <= members
    assert "data/references/openmx/FeCl2.scfout" not in members
    assert "data/references/openmx/GaAs_soc.scfout" not in members
    assert not any(Path(name).suffix in {".pyc", ".pyo"} for name in members)


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
from queer import WannierModel, EPWModel, OpenMXModel, ObservableResult
from queer.models.contracts import EigenstateBatch
from queer.physics.broadening import gaussian, lorentzian
from queer.physics.statistics import occupations, thermal_occupation
from queer.io.numerical import stable_hash
assert thermal_occupation(0., 0.) == .5
assert occupations(0., 0., 0.) == 0.
assert gaussian(0., .1) > 0 and lorentzian(0., .1) > 0
assert len(stable_hash({'probe': 1})) == 64
assert queer.__version__ == "1.4.1"
assert Path(queer.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert files("queer").joinpath("resources/neon.mplstyle").is_file()
assert files("queer").joinpath("resources/symmetry/hpkot.json").is_file()
assert files("queer").joinpath("resources/symmetry/LICENSE.txt").is_file()
assert files("queer").joinpath("finite_field/SOURCE_MANIFEST.json").is_file()
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
    native = tmp_path / "native"
    shutil.copytree(ROOT / "tests/fixtures/openmx/chain", native)
    native_config = tmp_path / "native.toml"
    native_config.write_text(f'''workflow="fat_band"
[inputs]
model="openmx"
material_dir="{native}"
scfout="chain.scfout"
[parameters]
kpoints=[[0.0,0.0,0.0],[0.25,0.0,0.0]]
''')
    subprocess.run(
        [str(venv / "bin" / "queer"), "fat_band", "--config", str(native_config)],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
    )
    native_probe = f"""
import json
from pathlib import Path
import numpy as np
from queer import OpenMXModel
model=OpenMXModel.from_scfout({str(native / "chain.scfout")!r})
np.testing.assert_allclose(model.energies([[0,0,0],[.25,0,0]]),
                           [[1.4/1.2-.2,1-.2],[2.4/1.2-.2,2-.2]],atol=1e-12)
manifest_path=next(Path({str(tmp_path / "results")!r}).glob("fat_band/*/manifest.json"))
manifest=json.loads(manifest_path.read_text())
with np.load(manifest_path.parent/manifest["cases"]["result"]["path"]) as arrays:
    np.testing.assert_allclose(arrays["weights"].sum(axis=0),1,atol=1e-12)
assert manifest["cases"]["result"]["metadata"]["projection"] == "mulliken"
assert not any(x in __import__("sys").modules for x in ("matplotlib","pandas","seekpath"))
"""
    subprocess.run(
        [str(interpreter), "-c", native_probe],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
    )
    services_probe = """
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
from queer import Evaluation, Structure, run_workflow
from queer.compat.generator import export_run
from queer.io import write_structure, read_frames

assert all(importlib.util.find_spec(name) is None for name in ("ase", "pyatb", "seekpath", "Dans_Diffraction"))
atomistic=Path.cwd()/"atomistic"
atomistic.mkdir()
structure=Structure(np.eye(3)*5, [[0,0,0]], [1])
write_structure(atomistic/"POSCAR", structure)
config={"workflow":"nvt","inputs":{"material_dir":str(atomistic),"structure":"POSCAR"},
        "parameters":{"provider":{"kind":"harmonic"},"n_steps":4,"seed":7,
                      "remove_com":False,"remove_rotation":False,"log_interval":2,"traj_interval":3},
        "output":{"results_dir":str(atomistic/"runs")}}
run=run_workflow(config)
assert run_workflow(config,run_dir=run,resume=True)==run
target=export_run(run)
assert [frame.step for frame in read_frames(target/"trajectory.traj")] == [0,3,4]
assert json.loads((target/"nvt_results.json").read_text())["completed"] is True
forces=np.zeros((6,1,3))
for axis in range(3):
    forces[2*axis,0,axis]=.01
    forces[2*axis+1,0,axis]=-.01
np.save(atomistic/"forces.npy",forces)
config["workflow"]="phonons"
config["inputs"]["forces"]="forces.npy"
config["parameters"]={"supercell":[1,1,1],"qpoints":[[0,0,0]],"frederiksen":False,"acoustic":False}
assert (export_run(run_workflow(config))/"phonon_results.json").is_file()
assert not any(name in sys.modules for name in ("ase","pyatb","seekpath","Dans_Diffraction","torch"))
"""
    subprocess.run(
        [str(interpreter), "-c", services_probe], cwd=tmp_path, env=env, check=True, capture_output=True
    )
    finite_field_probe = """
import importlib.util
import sys
import numpy as np
from queer import FieldPlane, FiniteFieldOptions
from queer.finite_field import FourBandModel, FourBandParams, crossing_scan
from queer.finite_field.figures import compute_figure
assert importlib.util.find_spec('hhh') is None
assert importlib.util.find_spec('mpmath') is None
assert importlib.util.find_spec('numba') is None
scan=crossing_scan(FourBandModel(FourBandParams().at(.9)), [0.], options='physical_eq8')
assert scan.metadata['options']['conventions']['charge_sign'] == -1
assert scan['critical_field'][0] > 0
result=compute_figure('fig4b', fields=[1000.,2000.])
np.testing.assert_allclose(result['probability'], np.exp(-result.metadata['critical_field']/result['fields']))
assert not any(name in sys.modules for name in ('hhh','mpmath','numba','matplotlib'))
"""
    subprocess.run(
        [str(interpreter), "-c", finite_field_probe], cwd=tmp_path, env=env, check=True, capture_output=True
    )
    example = tmp_path / "finite_field.py"
    shutil.copyfile(ROOT / "examples/finite_field.py", example)
    figure_config = tmp_path / "figure.toml"
    figure_config.write_text("""workflow="paper_figure"
[inputs]
model="fourband"
[parameters]
figure="fig4b"
[parameters.recipe]
fields=[1000.0,2000.0]
""")
    subprocess.run(
        [str(interpreter), str(example), "--config", str(figure_config)],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
    )
