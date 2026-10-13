"""Native finite-field CLI, scientific identity, artifact, and resume contracts."""

import json
from pathlib import Path

import numpy as np
import pytest

from queer.config import config_from_mapping, load_config
from queer.io.artifacts import RunStore
from queer.workflows import render_run, run_workflow

ROOT = Path(__file__).resolve().parents[2]


def configuration(tmp_path, workflow="landau_zener_hall", **parameters):
    return config_from_mapping(
        {
            "workflow": workflow,
            "inputs": {"model": "fourband"},
            "parameters": {
                "model": {"x": 0.5},
                "preset": "physical",
                "fields": [1e4, 2e4],
                "drive": {"E_min": 1e3, "E_max": 3e4},
                "crossings": {"window": 0.002, "scan_points": 65},
                "quadrature": {"t_max": 0.0002, "du": 2.0, "dmin": 1e-7, "lam_per_decade": 2},
                **parameters,
            },
            "output": {"results_dir": str(tmp_path / "results")},
        }
    )


def test_analytical_config_needs_no_material_directory(tmp_path):
    config = configuration(tmp_path)
    assert "material_dir" not in config.inputs
    assert config.parameters["preset"] == "physical"
    from queer.workflows.runner import _input_files

    assert _input_files(config) == []


def test_finite_field_configuration_rejects_unknown_and_missing_sources(tmp_path):
    data = configuration(tmp_path).as_dict()
    data["parameters"]["metric_typo"] = "cv"
    with pytest.raises(ValueError, match="metric_typo"):
        config_from_mapping(data)
    data = configuration(tmp_path).as_dict()
    data["inputs"]["geometry_grid"] = "sample.npz"
    with pytest.raises(ValueError, match="exactly one"):
        config_from_mapping(data)


def test_split_wannier_family_preserves_images_and_resolved_member_model(tmp_path, monkeypatch):
    from queer.finite_field.api import PreparedHallResponse
    from queer.models.wannier import WannierModel
    from queer.workflows.finite_field import (
        _crossing_map,
        _prepare,
        load_finite_field_source,
        split_harmonic_map,
    )

    plus = np.array([[0.04 + 0.01j, 0.013 - 0.02j, 0], [0, -0.03 - 0.02j, 0], [0, 0, 0.07]])
    blocks = [plus.conj().T, np.diag([-0.4, 0.3, 1.5]), plus]
    hr, ws, centres = (
        tmp_path / "member_hr.dat",
        tmp_path / "member_wsvec.dat",
        tmp_path / "member_centres.xyz",
    )
    with hr.open("w") as stream:
        stream.write("test\n3\n3\n2 1 2\n")
        for R, block in zip((-1, 0, 1), blocks, strict=True):
            for m in range(3):
                for n in range(3):
                    value = block[m, n]
                    stream.write(f"{R} 0 0 {m + 1} {n + 1} {value.real} {value.imag}\n")
    with ws.open("w") as stream:
        stream.write("use_ws_distance=.true.\n")
        for R in (-1, 0, 1):
            for m in range(3):
                for n in range(3):
                    shifts = [0, 2 * R] if R else [0]
                    stream.write(f"{R} 0 0 {m + 1} {n + 1}\n{len(shifts)}\n")
                    for shift in shifts:
                        stream.write(f"{shift} 0 0\n")
    centres.write_text("3\ncentres\nX 0.2 0 -2\nX -0.1 0 0\nX 0.3 0 2\n")
    member_plane = {"origin": [0.07, -0.02, 0], "direction": [0, 1, 0], "transverse": [-1, 0, 0]}
    lattice = np.diag([4, 5, 20]).tolist()
    family_file = tmp_path / "family.json"
    family_file.write_text(
        json.dumps(
            {
                "lattice": lattice,
                "members": [
                    {
                        "label": "member",
                        "D": 0.1,
                        "n_occ": 1,
                        "band_pair": None,
                        "plane": member_plane,
                        "files": {"hr": hr.name, "wsvec": ws.name, "centres": centres.name},
                    }
                ],
            }
        )
    )
    projector = np.diag([1, 0, 0]).tolist()
    config = config_from_mapping(
        {
            "workflow": "harmonic_map",
            "inputs": {"family_manifest": str(family_file)},
            "parameters": {
                "amplitudes": [1e4],
                "ef": 7.0,
                "interpolation": "standard",
                "occupied_bands": 2,
                "band_pair": [1, 2],
                "plane": {"origin": [-0.11, 0.03, 0]},
                "lattice": np.diag([7, 7, 7]).tolist(),
                "layer_projector": projector,
            },
        }
    )
    family = load_finite_field_source(config)
    children, _ = split_harmonic_map(config)
    child = children[0]
    assert child.parameters["interpolation"] == "ws_distance"
    assert child.parameters["occupied_bands"] == 1
    assert "ef" not in child.parameters and "band_pair" not in child.parameters
    assert child.parameters["lattice"] == lattice
    loaded = load_finite_field_source(child)
    assert loaded.interpolation == "ws_distance"
    assert loaded.fermi_energy == family.model(0).source_model.fermi_energy == 0

    class CompletedStore:
        @staticmethod
        def completed(case):
            return True

    # Inspect the actual views used by both preparation paths without running
    # the expensive response stage; these settings determine their Hamiltonians.
    monkeypatch.setattr(PreparedHallResponse, "prepare", lambda self, **kwargs: self)
    unsplit = _prepare(config, family.model(0), CompletedStore()).model
    split = _prepare(child, loaded, CompletedStore()).model
    assert split.band_pair() == unsplit.band_pair() == (0, 1)
    assert split.plane.describe() == unsplit.plane.describe()
    np.testing.assert_array_equal(split.layer_projector(), projector)
    np.testing.assert_array_equal(unsplit.layer_projector(), projector)
    np.testing.assert_array_equal(split.lattice, unsplit.lattice)
    points = np.array([0.01, -0.03, 0.07])
    np.testing.assert_array_equal(split.hamiltonian(points, 0.02), unsplit.hamiltonian(points, 0.02))
    from queer import finite_field

    original_scan = finite_field.crossing_scan
    crossing_views = []

    def inspect_scan(model, offsets, **kwargs):
        crossing_views.append(PreparedHallResponse(model, **kwargs).model)
        return original_scan(model, offsets, **kwargs)

    class FreshStore:
        @staticmethod
        def completed(case):
            return False

        @staticmethod
        def commit(case, arrays, *, metadata=None):
            pass

    monkeypatch.setattr(finite_field, "crossing_scan", inspect_scan)
    _crossing_map(config, family, FreshStore())
    assert len(crossing_views) == 1
    assert crossing_views[0].band_pair() == unsplit.band_pair()
    assert crossing_views[0].plane.describe() == unsplit.plane.describe()
    np.testing.assert_array_equal(crossing_views[0].layer_projector(), projector)
    # The fixture distinguishes corrected interpolation from ordinary HR sums.
    standard = WannierModel.from_directory(tmp_path, hr=hr.name, centres=centres.name)
    fractional = np.array([[0.07, 0, 0], [0.13, 0, 0]])
    assert not np.allclose(standard.energies(fractional), loaded.energies(fractional))


def test_geometry_grid_roundtrip_render_and_export_need_no_source(tmp_path):
    config = config_from_mapping(
        {
            "workflow": "geometry_grid",
            "inputs": {"model": "fourband"},
            "parameters": {
                "model": {"x": 0.5},
                "kx": np.linspace(-0.001, 0.001, 5).tolist(),
                "ky": np.linspace(-0.001, 0.001, 5).tolist(),
            },
            "output": {"results_dir": str(tmp_path)},
        }
    )
    run = run_workflow(config, run_dir=tmp_path / "grid")
    manifest = json.loads((run / "manifest.json").read_text())
    from queer.finite_field import GriddedGeometry

    grid = GriddedGeometry.from_npz(run / manifest["cases"]["result"]["path"])
    assert grid.geometry(0.0001, 0.0002).g_cv.shape == (3,)
    assert (render_run(run) / "geometry_grid.png").is_file()
    from queer.compat.finite_field import export_run

    exported = export_run(run)
    with np.load(exported / "result.npz", allow_pickle=False) as arrays:
        assert not any(arrays[key].dtype.hasobject for key in arrays.files)
    assert json.loads((exported / "manifest.json").read_text())["profile"] == "finite_field"


def test_interrupted_line_checkpoint_is_reused_and_result_matches(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    run = tmp_path / "interrupted"
    original = RunStore.commit
    interrupted = False

    def interrupt_after_line(store, case, arrays, *, metadata=None):
        nonlocal interrupted
        original(store, case, arrays, metadata=metadata)
        if "/lines/" in case and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("checkpoint interruption")

    monkeypatch.setattr(RunStore, "commit", interrupt_after_line)
    with pytest.raises(KeyboardInterrupt):
        run_workflow(config, run_dir=run)
    manifest = json.loads((run / "manifest.json").read_text())
    saved = {key: value["sha256"] for key, value in manifest["cases"].items() if "/lines/" in key}
    assert saved and "stages/layout" in manifest["cases"]
    monkeypatch.setattr(RunStore, "commit", original)
    from queer.finite_field.transport.hall import HallCalculator

    original_layout = HallCalculator.transverse_layout

    def layout_was_saved(*args, **kwargs):
        raise AssertionError("A committed transverse layout was recomputed")

    monkeypatch.setattr(HallCalculator, "transverse_layout", layout_was_saved)
    run_workflow(config, run_dir=run, resume=True)
    resumed = json.loads((run / "manifest.json").read_text())
    assert resumed["status"] == "completed"
    for case, checksum in saved.items():
        assert resumed["cases"][case]["sha256"] == checksum
        assert resumed["cases"][case]["attempt_index"] == 0
    monkeypatch.setattr(HallCalculator, "transverse_layout", original_layout)
    fresh = run_workflow(config, run_dir=tmp_path / "fresh")
    with np.load(run / resumed["cases"]["result"]["path"], allow_pickle=False) as a:
        fresh_manifest = json.loads((fresh / "manifest.json").read_text())
        with np.load(fresh / fresh_manifest["cases"]["result"]["path"], allow_pickle=False) as b:
            np.testing.assert_array_equal(a["current"], b["current"])


def test_flux_workflow_persists_refinement_and_renders(tmp_path):
    config = config_from_mapping(
        {
            "workflow": "berry_flux",
            "inputs": {"model": "fourband"},
            "parameters": {"model": {"x": 0.5}, "n": 5, "window": 0.002},
        }
    )
    run = run_workflow(config, run_dir=tmp_path / "flux")
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["cases"]["result"]["metadata"]["quantized_chern_certificate"] is False
    with np.load(run / manifest["cases"]["result"]["path"], allow_pickle=False) as result:
        np.testing.assert_array_equal(result["grid_size"], [5, 9])
        assert np.isfinite(result["flux"]).all()
    assert (render_run(run) / "berry_flux.png").is_file()


def test_lindblad_workflow_capability_failure_is_explicit(tmp_path):
    from queer.capabilities import CapabilityError

    config = config_from_mapping(
        {"workflow": "lindblad_line", "inputs": {"model": "fourband"}, "parameters": {"fields": [10000.0]}}
    )
    with pytest.raises(CapabilityError, match="two-band"):
        run_workflow(config, run_dir=tmp_path / "lindblad")


def test_paper_probability_uses_resolved_recipe_options(tmp_path):
    config = config_from_mapping(
        {
            "workflow": "paper_figure",
            "inputs": {"model": "fourband"},
            "parameters": {"figure": "fig4b", "recipe": {"fields": [1000.0, 10000.0]}},
        }
    )
    run = run_workflow(config, run_dir=tmp_path / "paper")
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["scientific_model"]["preset"] == "fig4b"
    with np.load(run / manifest["cases"]["result"]["path"], allow_pickle=False) as result:
        assert result["probability"][0] < result["probability"][1]
    assert (render_run(run) / "fig4b.png").is_file()


def test_crossing_ensemble_retains_ragged_line_and_segment_ownership(tmp_path):
    config = config_from_mapping(
        {
            "workflow": "crossing_scan",
            "inputs": {"model": "fourband"},
            "parameters": {"preset": "paper", "model_values": [0.5, 1.5], "offsets": [0.0, 0.0001]},
        }
    )
    run = run_workflow(config, run_dir=tmp_path / "crossings")
    manifest = json.loads((run / "manifest.json").read_text())
    with np.load(run / manifest["cases"]["result"]["path"], allow_pickle=False) as result:
        np.testing.assert_array_equal(result["line_member_index"], [0, 0, 1, 1])
        np.testing.assert_array_equal(
            result["line_member_index"][result["segment_line_index"]], result["segment_member_index"]
        )
        np.testing.assert_array_equal(
            result["line_member_index"][result["line_index"]], result["member_index"]
        )


@pytest.mark.parametrize("figure", ["fig5c", "ext9b"])
def test_paper_harmonics_use_owner_written_preparation_checkpoints(tmp_path, figure, monkeypatch):
    recipe = {
        "orders": [3],
        "options": {
            "drive": {"E_min": 1000.0, "E_max": 30000.0},
            "crossings": {"window": 0.002, "scan_points": 65},
            "quadrature": {"t_max": 0.0002, "du": 2.0, "dmin": 1e-7, "lam_per_decade": 2},
            "fourier": {"n_theta": 16, "adaptive": False},
        },
    }
    if figure == "fig5c":
        recipe.update(xs=[0.5, 1.5], amplitudes=[10000.0, 20000.0])
    else:
        recipe.update(scaled_amplitudes=[1000.0, 3000.0])
    config = config_from_mapping(
        {
            "workflow": "paper_figure",
            "inputs": {"model": "fourband"},
            "parameters": {"figure": figure, "recipe": recipe},
        }
    )
    run = tmp_path / figure
    original = RunStore.commit
    interrupted = False

    def interrupt_after_fourier(store, case, arrays, *, metadata=None):
        nonlocal interrupted
        original(store, case, arrays, metadata=metadata)
        if case.endswith("/Fourier") and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("Fourier stage committed")

    monkeypatch.setattr(RunStore, "commit", interrupt_after_fourier)
    with pytest.raises(KeyboardInterrupt):
        run_workflow(config, run_dir=run)
    interrupted_manifest = json.loads((run / "manifest.json").read_text())
    committed = {case: record["sha256"] for case, record in interrupted_manifest["cases"].items()}
    monkeypatch.setattr(RunStore, "commit", original)
    run_workflow(config, run_dir=run, resume=True)
    manifest = json.loads((run / "manifest.json").read_text())
    assert any(case.endswith("/layout") for case in manifest["cases"])
    assert any("/lines/" in case for case in manifest["cases"])
    assert any(case.endswith("/Fourier") for case in manifest["cases"])
    assert manifest["cases"]["result"]["metadata"]["figure"] == figure
    for case, checksum in committed.items():
        assert manifest["cases"][case]["sha256"] == checksum
    expected_dims = ["member", "amplitude", "harmonic"] if figure == "fig5c" else ["harmonic", "amplitude"]
    product = "b" if figure == "fig5c" else "scaled_sine"
    assert manifest["cases"]["result"]["metadata"]["dims"][product] == expected_dims
    assert (render_run(run) / f"{figure}.png").is_file()


def test_preparation_cache_reused_across_field_scans(tmp_path, monkeypatch):
    config = configuration(tmp_path, cache_dir=str(tmp_path / "cache"))
    run_workflow(config, run_dir=tmp_path / "first")
    from queer.finite_field.transport import hall

    def no_new_line(*args, **kwargs):
        raise AssertionError("A cached line was recomputed")

    monkeypatch.setattr(hall, "solve_line", no_new_line)
    data = config.as_dict()
    data["parameters"]["fields"] = [15000.0]
    second = run_workflow(config_from_mapping(data), run_dir=tmp_path / "second")
    manifest = json.loads((second / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert any("/lines/" in case for case in manifest["cases"])


def test_auto_discovered_wannier_sidecar_enters_scientific_hash(tmp_path):
    import shutil

    fixture = ROOT / "tests/fixtures/wannier/chain"
    shutil.copytree(fixture, tmp_path / "model")
    sidecar = tmp_path / "model/wannier90_centres.xyz"
    sidecar.write_text("1\ncentres\nX 0.1 0.2 0.3\n")
    config = config_from_mapping(
        {
            "workflow": "geometry_grid",
            "inputs": {"model": "wannier", "material_dir": str(tmp_path / "model"), "poscar": "POSCAR"},
            "parameters": {
                "occupied_bands": 1,
                "kx": [-0.01, -0.003, 0.003, 0.01],
                "ky": [-0.01, -0.003, 0.003, 0.01],
            },
        }
    )
    from queer.workflows.runner import _input_files

    assert sidecar in _input_files(config)


def test_family_hashes_auto_sidecars_in_member_directories(tmp_path):
    import shutil

    shutil.copytree(ROOT / "tests/fixtures/wannier/chain", tmp_path / "member")
    sidecar = tmp_path / "member/wannier90_centres.xyz"
    sidecar.write_text("1\ncentres\nX 0.1 0.2 0.3\n")
    manifest = tmp_path / "family.json"
    manifest.write_text(
        json.dumps(
            {
                "n_occ": 1,
                "members": [
                    {
                        "label": "a",
                        "D": 0.0,
                        "files": {"hr": "member/wannier90_hr.dat", "poscar": "member/POSCAR"},
                    },
                ],
            }
        )
    )
    config = config_from_mapping({"workflow": "family_report", "inputs": {"family_manifest": str(manifest)}})
    from queer.workflows.runner import _input_files

    assert sidecar in _input_files(config)


def test_all_analytical_examples_validate_from_unrelated_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sources = list((ROOT / "configs/examples").glob("finite_field*.toml"))
    assert sources
    for source in sources:
        load_config(source)


def test_split_members_submission_and_verified_aggregation(tmp_path, monkeypatch):
    import importlib.util
    import subprocess

    module_spec = importlib.util.spec_from_file_location(
        "finite_field_submit", ROOT / "scripts/hpc/submit_sweep.py"
    )
    submit = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(submit)
    source = tmp_path / "map.toml"
    source.write_text("""workflow = "harmonic_map"
[inputs]
model = "fourband"
[parameters]
model_values = [0.5, 1.5]
amplitudes = [10000.0, 20000.0]
n_max = 3
[parameters.drive]
E_min = 1000.0
E_max = 30000.0
[parameters.crossings]
window = 0.002
scan_points = 65
[parameters.quadrature]
t_max = 0.0002
du = 2.0
dmin = 1e-7
lam_per_decade = 2
[parameters.fourier]
n_theta = 16
adaptive = false
[output]
results_dir = "results"
""")
    profile = tmp_path / "profile.toml"
    profile.write_text("cpus_per_task=1\nconcurrency=2\n")
    dispatch_path = None

    def submitted(command, *, env, check):
        nonlocal dispatch_path
        assert "--array=0-1%2" in command
        dispatch_path = Path(env["QUEER_SWEEP_FILE"])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(submit.subprocess, "run", submitted)
    assert submit.main(["--config", str(source), "--profile", str(profile), "--split-members"]) == 0
    monkeypatch.undo()
    dispatch = json.loads(dispatch_path.read_text())
    for index, case in enumerate(dispatch["cases"]):
        child = load_config(case["config"], base_dir=Path(case["config_base_dir"]))
        assert child.workflow == "hall_harmonics"
        assert child.parameters["model"]["x"] == dispatch["aggregation"]["parameters"][index]
        run_workflow(child, run_dir=Path(dispatch["run_root"]) / f"task-{index:04d}")
    from queer.workflows.finite_field import aggregate_member_runs

    aggregate = aggregate_member_runs(dispatch_path)
    manifest = json.loads((aggregate / "manifest.json").read_text())
    with np.load(aggregate / manifest["cases"]["result"]["path"], allow_pickle=False) as result:
        assert result["b"].shape == (2, 2, 4)
        np.testing.assert_array_equal(result["parameter"], [0.5, 1.5])
    assert aggregate_member_runs(dispatch_path) == aggregate
    child_manifest = json.loads((Path(dispatch["run_root"]) / "task-0000/manifest.json").read_text())
    product = Path(dispatch["run_root"]) / "task-0000" / child_manifest["cases"]["result"]["path"]
    product.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed or removed"):
        aggregate_member_runs(dispatch_path)
