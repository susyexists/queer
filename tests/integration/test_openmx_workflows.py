"""Native SCFOUT workflows, scientific reductions and durable auxiliary inputs."""

import inspect
import json
import struct
from functools import wraps
from pathlib import Path

import numpy as np
import pytest

from queer.config import config_from_mapping, load_config
from queer.io.artifacts import digest_file
from queer.io.openmx import BOHR_ANGSTROM, HARTREE_EV
from queer.models.openmx import OpenMXModel
from queer.workflows import render_run, run_workflow
from queer.workflows.registry import REGISTRY


def native_input(directory):
    """An independent one-cell, one-spatial-orbital SOC binary fixture."""
    directory.mkdir()
    records = []

    def write(kind, values):
        records.append(struct.pack("<" + str(len(values)) + kind, *values))

    write("i", [1, 15, 0, 0, 0, 0])
    write("i", [1])
    write("d", [0, 0, 0, 0])
    write("i", [0, 0, 0, 0])
    for value in (1, 0, 1, 0):
        write("i", [value])
    for matrix in (np.eye(3) / BOHR_ANGSTROM, np.eye(3) * 2 * np.pi * BOHR_ANGSTROM):
        write("d", [value for row in matrix for value in (0, *row)])
    write("d", [0, 0, 0, 0])
    for value in (-0.5, 0.5, 0.2, 0.1):
        write("d", [value / HARTREE_EV])
    for _ in range(3):
        write("d", [0])
    write("d", [1])
    for _ in range(3 + 3 + 4 + 2):
        write("d", [0])
    write("i", [3])
    write("d", [0, 300, 0, 0, 0, 0, 0, 0, 1, 0])
    lines = [
        "<Definition.of.Atomic.Species",
        "X X2.0-s1 X_PBE19",
        "Definition.of.Atomic.Species>",
        "<Atoms.SpeciesAndCoordinates",
        "1 X 0 0 0",
        "Atoms.SpeciesAndCoordinates>",
    ]
    write("i", [len(lines)])
    records.extend((line.encode() + b"\n").ljust(256, b"\0") for line in lines)
    (directory / "toy.scfout").write_bytes(b"".join(records))
    radii = np.geomspace(0.001, 0.1, 101)
    rows = [f"{np.log(radius):.17g} {radius:.17g} {np.sqrt(3 / 0.1**3):.17g}" for radius in radii]
    (directory / "X2.0.pao").write_text(
        "PAO.Lmax 0\nPAO.Mul 1\ngrid.num.output 101\n<pseudo.atomic.orbitals.L=0\n"
        + "\n".join(rows)
        + "\npseudo.atomic.orbitals.L=0>\n"
    )
    return directory


def parameters(name):
    signature = inspect.signature(REGISTRY[name].calculator())
    values = {}
    if "kpoints" in signature.parameters:
        values["kpoints"] = [[0, 0, 0], [0.25, 0, 0]]
    if "grid" in signature.parameters:
        values["grid"] = [2, 2, 1]
    if "omega" in signature.parameters:
        values["omega"] = [0.1, 0.5, 1.1]
    if "occupied_bands" in signature.parameters:
        values["occupied_bands"] = 1
    if name == "fermi_energy":
        values["electron_num"] = 1
    if name.startswith("bandunfolding"):
        values.update(supercell_matrix=np.eye(3, dtype=int).tolist(), ecut=0.1)
    if name == "find_nodes":
        values["initial_grid"] = [2, 1, 1]
    if name == "chirality":
        values.update(kpoint=[0, 0, 0], polar_points=4, azimuth_points=4)
    if name == "wilson_loop":
        values["kpoints"] = [[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0], [1, 0, 0]]
    if name == "polarization":
        values["grid"] = [2, 2, 2]
        values["ionic_charges"] = [1]
    if name == "boltz_transport":
        values.update(relaxation_time=10, temperature_kelvin=300)
    return values


def native_config(directory, name, results, **overrides):
    inputs = {"model": "openmx", "material_dir": directory, "scfout": "toy.scfout"}
    if name.startswith("bandunfolding"):
        inputs["pao_files"] = {"X2.0": "X2.0.pao"}
    values = parameters(name)
    values.update(overrides)
    return config_from_mapping(
        {
            "workflow": name,
            "inputs": inputs,
            "parameters": values,
            "execution": {"batch_size": 1, "native_threads": 1},
            "output": {"results_dir": results},
        }
    )


@pytest.mark.parametrize("name", tuple(REGISTRY))
def test_all_native_registry_workflows_persist_render_and_resume(tmp_path, monkeypatch, name):
    directory = native_input(tmp_path / "material")
    config = native_config(directory, name, tmp_path / "results")
    run = run_workflow(config)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["scientific_model"]["spin_mode"] == "soc"
    record = manifest["cases"]["result"]
    artifact = run / record["path"]
    checksum, timestamp = digest_file(artifact), artifact.stat().st_mtime_ns
    with np.load(artifact, allow_pickle=False) as data:
        assert set(data.files) <= record["metadata"]["dims"].keys()
        assert set(data.files) <= record["metadata"]["units"].keys()
        for key in data.files:
            assert len(record["metadata"]["dims"][key]) == data[key].ndim
            if data[key].dtype.kind in "fci":
                assert np.isfinite(data[key]).all(), key
        if name == "pdos":
            np.testing.assert_allclose(data["pdos"].sum(axis=0), data["dos"], atol=1e-12)
        if name in {"fat_band", "spin_texture"}:
            assert data["energies"].shape == (2, 2)

    def prohibited(*args, **kwargs):
        raise AssertionError("completed scientific case must not be recomputed")

    monkeypatch.setattr(OpenMXModel, "energies", prohibited)
    monkeypatch.setattr(OpenMXModel, "eigenstates", prohibited)
    run_workflow(config, run_dir=run, resume=True)
    assert artifact.stat().st_mtime_ns == timestamp
    assert digest_file(artifact) == checksum
    monkeypatch.setattr(OpenMXModel, "from_directory", prohibited)
    figures = render_run(run)
    assert (figures / f"{name}.png").is_file()
    assert (
        json.loads((figures / "manifest.json").read_text())["source_fingerprint"] == manifest["fingerprint"]
    )


@pytest.mark.parametrize("name", ["dos", "pdos", "jdos", "fat_band", "spin_texture", "boltz_transport"])
def test_multiple_tiles_equal_complete_nonuniform_mesh(tmp_path, monkeypatch, name):
    import queer.workflows.registry as registry
    from queer.workflows.runner import _load_model

    directory = native_input(tmp_path / "material")
    points = [[value, 0, 0] for value in (0, 0.1, 0.25, 0.5, 0.9)]
    overrides = {"kpoints": points}
    if "weights" in inspect.signature(REGISTRY[name].calculator()).parameters:
        overrides["weights"] = [0.1, 0, 0.2, 0.3, 0.4]
    config = native_config(directory, name, tmp_path / "results", **overrides)
    direct_parameters = dict(config.parameters)
    direct = REGISTRY[name].calculator()(_load_model(config), **direct_parameters)
    monkeypatch.setattr(registry, "_TILE_SIZE", 2)
    run = run_workflow(config)
    manifest = json.loads((run / "manifest.json").read_text())
    assert len(manifest["cases"]) == 4
    metadata = manifest["cases"]["result"]["metadata"]
    assert metadata["kpoint_count"] == len(points)
    assert metadata["sample_count"] == len(points)
    recorded_weights = np.concatenate(
        [manifest["cases"][case]["metadata"]["integration_weights"] for case in metadata["partial_cases"]]
    )
    expected_weights = np.asarray(overrides.get("weights", np.ones(len(points))), dtype=float)
    np.testing.assert_allclose(recorded_weights, expected_weights / expected_weights.sum(), atol=1e-15)
    with np.load(run / manifest["cases"]["result"]["path"], allow_pickle=False) as data:
        for key, expected in direct.arrays.items():
            if expected.dtype.kind in "fci":
                np.testing.assert_allclose(data[key], expected, atol=1e-12, rtol=1e-12)
            else:
                np.testing.assert_array_equal(data[key], expected)


def test_pao_auxiliary_hashing_rejects_changed_radial_input(tmp_path):
    directory = native_input(tmp_path / "material")
    config = native_config(directory, "bandunfolding", tmp_path / "results")
    run = run_workflow(config)
    manifest = json.loads((run / "manifest.json").read_text())
    assert any(Path(record["path"]).name == "X2.0.pao" for record in manifest["inputs"])
    source = directory / "X2.0.pao"
    source.write_text(source.read_text() + "\n# changed radial input\n")
    with pytest.raises(ValueError, match="Resume rejected"):
        run_workflow(config, run_dir=run, resume=True)


def test_interrupted_density_resumes_only_missing_scientific_tiles(tmp_path, monkeypatch):
    import queer.observables.electronic as electronic
    import queer.workflows.registry as registry

    directory = native_input(tmp_path / "material")
    config = native_config(directory, "pdos", tmp_path / "results", grid=[5, 1, 1])
    values = dict(config.parameters)
    values.pop("kpoints")
    config = config_from_mapping({**config.as_dict(), "parameters": values})
    monkeypatch.setattr(registry, "_TILE_SIZE", 2)
    original = electronic.pdos
    calls = []

    @wraps(original)
    def fail_second(model, kpoints, **kwargs):
        calls.append(len(kpoints))
        if len(calls) == 2:
            raise RuntimeError("injected electronic tile interruption")
        return original(model, kpoints, **kwargs)

    monkeypatch.setattr(electronic, "pdos", fail_second)
    run = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="injected electronic"):
        run_workflow(config, run_dir=run)
    manifest = json.loads((run / "manifest.json").read_text())
    assert list(manifest["cases"]) == ["level_00_tile_00000000"]
    first = run / manifest["cases"]["level_00_tile_00000000"]["path"]
    checksum, timestamp = digest_file(first), first.stat().st_mtime_ns
    resumed_calls = []

    @wraps(original)
    def record(model, kpoints, **kwargs):
        resumed_calls.append(len(kpoints))
        return original(model, kpoints, **kwargs)

    monkeypatch.setattr(electronic, "pdos", record)
    run_workflow(config, run_dir=run, resume=True)
    assert resumed_calls == [2, 1]
    assert digest_file(first) == checksum and first.stat().st_mtime_ns == timestamp
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "completed" and len(manifest["attempts"]) == 2
    with np.load(run / "arrays/result.npz", allow_pickle=False) as data:
        np.testing.assert_allclose(data["pdos"].sum(axis=0), data["dos"], atol=1e-12)


def test_density_ignores_zero_quadrature_tiles_and_adapts_with_diagnostics(tmp_path, monkeypatch):
    import queer.workflows.registry as registry

    directory = native_input(tmp_path / "material")
    monkeypatch.setattr(registry, "_TILE_SIZE", 1)
    config = native_config(directory, "dos", tmp_path / "results", weights=[0, 1])
    run = run_workflow(config)
    manifest = json.loads((run / "manifest.json").read_text())
    assert set(manifest["cases"]) == {"level_00_tile_00000001", "result"}
    assert manifest["cases"]["result"]["metadata"]["kpoint_count"] == 2
    assert manifest["cases"]["result"]["metadata"]["evaluated_kpoint_count"] == 1
    values = dict(config.parameters)
    values.pop("kpoints")
    values.pop("weights")
    values.update(
        grid=[2, 1, 1],
        integration={
            "method": "adaptive",
            "max_levels": 2,
            "absolute_tolerance": 1e-10,
            "relative_tolerance": 1e-10,
        },
    )
    adaptive = config_from_mapping({**config.as_dict(), "parameters": values})
    run = run_workflow(adaptive)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["integration"]["converged"] and manifest["integration"]["level"] == 1
    assert manifest["integration"]["points"] == 4
    assert manifest["integration"]["diagnostics"]["dos"]["error_estimate"] < 1e-12


def test_native_toml_density_workflow_matches_analytical_levels(tmp_path):
    directory = native_input(tmp_path / "material")
    source = tmp_path / "pdos.toml"
    source.write_text(
        'workflow = "pdos"\n[inputs]\nmodel = "openmx"\nmaterial_dir = "material"\n'
        'scfout = "toy.scfout"\n[parameters]\ngrid = [2, 1, 1]\nomega = [-0.5, 0.0, 0.5]\n'
        'eta = 0.1\n[output]\nresults_dir = "results"\n'
    )
    config = load_config(source)
    assert config.material_dir == directory
    run = run_workflow(config)
    eigenvalue = np.sqrt(0.3)
    omega = np.asarray(config.parameters["omega"])
    expected = sum(
        np.exp(-0.5 * ((omega - value) / 0.1) ** 2) / (0.1 * np.sqrt(2 * np.pi))
        for value in (-eigenvalue, eigenvalue)
    )
    with np.load(run / "arrays/result.npz", allow_pickle=False) as data:
        np.testing.assert_allclose(data["dos"], expected, atol=1e-12)
        np.testing.assert_allclose(data["pdos"].sum(axis=0), expected, atol=1e-12)


def test_surface_band_workflow_needs_no_frequency_axis(tmp_path):
    directory = native_input(tmp_path / "material")
    config = native_config(directory, "surface_state", tmp_path / "results", method="bands", slab_layers=4)
    values = dict(config.parameters)
    values.pop("omega")
    config = config_from_mapping({**config.as_dict(), "parameters": values})
    run = run_workflow(config)
    with np.load(run / "arrays/result.npz", allow_pickle=False) as data:
        assert data["energies"].shape == (8, 2)
        np.testing.assert_allclose(data["layer_weights"].sum(axis=0), 1, atol=1e-12)
        assert data["channel_labels"].tolist() == ["soc"] * 8
    assert (render_run(run) / "surface_state.png").is_file()
