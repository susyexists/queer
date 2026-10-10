"""Whole-mesh refinement must retain FHS topology and EMPC scattering physics."""

import json
import struct
from functools import wraps

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.constants import Boltzmann, elementary_charge, h, hbar

from queer.config import config_from_mapping
from queer.io.artifacts import digest_file
from queer.io.openmx import BOHR_ANGSTROM, HARTREE_EV
from queer.models.openmx import OpenMXModel
from queer.observables.transport import boltz_transport
from queer.workflows import run_workflow


def _write_spinor(directory, translations, hopping):
    """Write the v3 specification for one atom and one spinor spatial orbital."""
    directory.mkdir()
    translations, hopping = np.asarray(translations), np.asarray(hopping)
    cells = len(translations)
    records = []

    def write(kind, values):
        records.append(struct.pack("<" + str(len(values)) + kind, *values))

    write("i", [1, 15, 0, 0, 0, cells - 1])
    write("i", [1])
    write("d", [value for row in translations / BOHR_ANGSTROM for value in (0, *row)])
    write("i", [int(value) for row in translations for value in (0, *row)])
    write("i", [1])
    write("i", [cells - 1])
    write("i", [1] * cells)
    write("i", list(range(cells)))
    for matrix in (np.eye(3) / BOHR_ANGSTROM, 2 * np.pi * BOHR_ANGSTROM * np.eye(3)):
        write("d", [value for row in matrix for value in (0, *row)])
    write("d", [0, 0, 0, 0])
    for component in (
        hopping[:, 0, 0].real,
        hopping[:, 1, 1].real,
        hopping[:, 0, 1].real,
        hopping[:, 0, 1].imag,
        hopping[:, 0, 0].imag,
        hopping[:, 1, 1].imag,
        np.zeros(cells),
    ):
        write("d", component / HARTREE_EV)
    write("d", [1] + [0] * (cells - 1))
    for _ in range(3 + 3 + 4 + 2):
        write("d", np.zeros(cells))
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
    source = directory / "toy.scfout"
    source.write_bytes(b"".join(records))
    return source


def _config(directory, workflow, parameters, results):
    return config_from_mapping(
        {
            "workflow": workflow,
            "inputs": {"model": "openmx", "material_dir": directory, "scfout": "toy.scfout"},
            "parameters": parameters,
            "execution": {"batch_size": 8, "native_threads": 1},
            "output": {"results_dir": results},
        }
    )


def _adaptive(max_levels):
    return {
        "method": "adaptive",
        "max_levels": max_levels,
        "absolute_tolerance": 1e-12,
        "relative_tolerance": 1e-10,
    }


def _write_qwz(directory):
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.diag([1, -1])
    translations = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
    hopping = [
        -0.1 * sigma_z,
        0.5 * sigma_z - 0.5j * sigma_x,
        0.5 * sigma_z + 0.5j * sigma_x,
        0.5 * sigma_z - 0.5j * sigma_y,
        0.5 * sigma_z + 0.5j * sigma_y,
    ]
    return _write_spinor(directory, translations, hopping)


def test_fhs_refinement_corrects_wrapped_chern_and_resumes_global_cases(tmp_path, monkeypatch):
    import queer.observables.integrated as integrated

    # For -2 < m < 0 the lower QWZ band has C=+1 in Queer's convention.
    # At m=-0.1 a3x3 mesh wraps its concentrated curvature into C=-1.
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.diag([1, -1])
    directory = tmp_path / "material"
    source = _write_qwz(directory)
    model = OpenMXModel.from_scfout(source)
    # Check the independently packed native model against the analytic QWZ d(k).
    point = [0.17, 0.31, 0]
    qx, qy = 2 * np.pi * np.asarray(point[:2])
    expected_h = np.sin(qx) * sigma_x + np.sin(qy) * sigma_y + (-0.1 + np.cos(qx) + np.cos(qy)) * sigma_z
    assert_allclose(model.hamiltonian(point), expected_h, atol=2e-15)
    config = _config(
        directory,
        "ahc",
        {
            "engine": "fhs",
            "dimension": 2,
            "plane": [0, 1],
            "grid": [3, 3, 1],
            "chemical_potentials": [0],
            "integration": _adaptive(3),
        },
        tmp_path / "results",
    )
    original = integrated.ahc
    calls = []

    @wraps(original)
    def interrupt_third(model, **kwargs):
        calls.append(np.asarray(kwargs["grid"]).tolist())
        if len(calls) == 3:
            raise KeyboardInterrupt("injected whole-mesh refinement interruption")
        return original(model, **kwargs)

    monkeypatch.setattr(integrated, "ahc", interrupt_third)
    run = tmp_path / "partial"
    with pytest.raises(KeyboardInterrupt, match="injected whole-mesh"):
        run_workflow(config, run_dir=run)
    assert calls == [[3, 3, 1], [6, 6, 1], [12, 12, 1]]
    interrupted = json.loads((run / "manifest.json").read_text())
    assert interrupted["status"] == "interrupted"
    assert set(interrupted["cases"]) == {
        "level_00_global",
        "level_00_reduction",
        "level_01_global",
        "level_01_reduction",
    }
    coarse, corrected = (interrupted["cases"][f"level_{level:02d}_global"] for level in (0, 1))
    assert_allclose(coarse["metadata"]["fhs_diagnostics"]["chern_vector"], [[0, 0, -1]], atol=2e-13)
    assert_allclose(corrected["metadata"]["fhs_diagnostics"]["chern_vector"], [[0, 0, 1]], atol=2e-13)
    assert interrupted["integration"]["converged"] is False
    assert_allclose(
        interrupted["integration"]["diagnostics"]["ahc"]["error_estimate"],
        2 * elementary_charge**2 / h,
        atol=2e-18,
    )
    snapshots = {
        case: (record, digest_file(run / record["path"]), (run / record["path"]).stat().st_mtime_ns)
        for case, record in interrupted["cases"].items()
    }
    resumed_calls = []

    @wraps(original)
    def record_refinement(model, **kwargs):
        resumed_calls.append(np.asarray(kwargs["grid"]).tolist())
        return original(model, **kwargs)

    monkeypatch.setattr(integrated, "ahc", record_refinement)
    run_workflow(config, run_dir=run, resume=True)
    assert resumed_calls == [[12, 12, 1]]
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert [attempt["status"] for attempt in manifest["attempts"]] == ["interrupted", "completed"]
    for case, (record, checksum, modified) in snapshots.items():
        artifact = run / record["path"]
        assert manifest["cases"][case] == record
        assert digest_file(artifact) == checksum and artifact.stat().st_mtime_ns == modified
    assert not any("tile" in case for case in manifest["cases"])
    assert [
        manifest["cases"][f"level_{level:02d}_global"]["metadata"]["sample_count"] for level in range(3)
    ] == [9, 36, 144]
    integration = manifest["integration"]
    assert integration["converged"] and integration["level"] == 2
    assert integration["points"] == 144
    assert integration["active_axes"] == [True, True, False]
    assert integration["diagnostics"]["ahc"]["error_estimate"] < 1e-18
    result = manifest["cases"]["result"]
    assert result["metadata"]["partial_cases"] == ["level_02_global"]
    assert result["metadata"]["units"]["ahc"] == "S"
    assert result["metadata"]["fhs_diagnostics"]["grid"] == [12, 12, 1]
    with np.load(run / result["path"], allow_pickle=False) as arrays:
        expected = np.zeros((1, 3, 3))
        expected[0, 0, 1] = -(elementary_charge**2) / h
        expected[0, 1, 0] = elementary_charge**2 / h
        assert_allclose(arrays["ahc"], expected, atol=2e-18)


def test_all_one_sheet_fhs_grid_refines_only_the_selected_plane(tmp_path):
    directory = tmp_path / "material"
    _write_qwz(directory)
    config = _config(
        directory,
        "ahc",
        {
            "engine": "fhs",
            "dimension": 2,
            "plane": [0, 1],
            "grid": [1, 1, 1],
            "chemical_potentials": [0],
            "integration": _adaptive(4),
        },
        tmp_path / "results",
    )
    run = run_workflow(config)
    manifest = json.loads((run / "manifest.json").read_text())
    integration = manifest["integration"]
    assert integration["converged"] and integration["level"] == 3
    assert integration["base_grid"] == [1, 1, 1]
    assert integration["active_axes"] == [True, True, False]
    assert integration["points"] == 64
    for level in range(4):
        metadata = manifest["cases"][f"level_{level:02d}_global"]["metadata"]
        assert metadata["fhs_diagnostics"]["grid"] == [2**level, 2**level, 1]
        assert metadata["sample_count"] == 4**level
    with np.load(run / manifest["cases"]["result"]["path"], allow_pickle=False) as arrays:
        assert_allclose(arrays["ahc"][0, 0, 1], -(elementary_charge**2) / h, atol=2e-18)
        assert_allclose(arrays["ahc"][0, 1, 0], elementary_charge**2 / h, atol=2e-18)


def test_empc_refines_native_whole_mesh_and_matches_analytic_lifetime(tmp_path, monkeypatch):
    import queer.observables.transport as transport

    directory = tmp_path / "material"
    source = _write_spinor(directory, [[0, 0, 0]], [[[-0.5, 0.2 + 0.1j], [0.2 - 0.1j, 0.5]]])
    config = _config(
        directory,
        "boltz_transport",
        {
            "grid": [2, 2, 2],
            "transport_method": "EMPC",
            "deformation_potential": 2,
            "young_modulus": 100,
            "temperature_kelvin": 1000,
            "chemical_potentials": [0],
            "eta": 0.1,
            "integration": _adaptive(2),
        },
        tmp_path / "results",
    )
    original = transport.boltz_transport
    calls = []

    @wraps(original)
    def record_global(model, kpoints, **kwargs):
        calls.append(len(kpoints))
        assert "grid" not in kwargs
        assert_allclose(np.asarray(kwargs["weights"]).sum(), 1, atol=1e-15)
        return original(model, kpoints, **kwargs)

    monkeypatch.setattr(transport, "boltz_transport", record_global)
    run = run_workflow(config)
    assert calls == [8, 64]
    manifest = json.loads((run / "manifest.json").read_text())
    assert set(manifest["cases"]) == {"level_00_global", "level_00_reduction", "level_01_global", "result"}
    for level, count in [(0, 8), (1, 64)]:
        metadata = manifest["cases"][f"level_{level:02d}_global"]["metadata"]
        assert metadata["sample_count"] == count and metadata["refinement_level"] == level
        assert_allclose(metadata["integration_weights"], np.full(count, 1 / count), atol=1e-15)
        assert metadata["transport_method"] == "EMPC"
    integration = manifest["integration"]
    assert integration["converged"] and integration["level"] == 1
    assert integration["points"] == 64 and integration["active_axes"] == [True, True, True]
    assert "electron_count" in integration["diagnostics"]
    assert "conductivity" in integration["diagnostics"]
    # Point-dependent energies/lifetimes have different shapes at successive
    # meshes; convergence compares the integrated response and carrier counts.
    assert "energies_absolute" not in integration["diagnostics"]
    assert "relaxation_time" not in integration["diagnostics"]
    result = manifest["cases"]["result"]
    assert result["metadata"]["partial_cases"] == ["level_01_global"]
    points = np.stack(np.meshgrid(*(np.arange(4) / 4 for _ in range(3)), indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    direct = boltz_transport(
        OpenMXModel.from_scfout(source),
        points,
        transport_method="EMPC",
        deformation_potential=2,
        young_modulus=100,
        temperature_kelvin=1000,
        chemical_potentials=[0],
        eta=0.1,
    )
    with np.load(run / result["path"], allow_pickle=False) as arrays:
        for key, expected in direct.arrays.items():
            if expected.dtype.kind in "fc":
                assert_allclose(arrays[key], expected, rtol=2e-13, atol=1e-12, err_msg=key)
            else:
                assert_array_equal(arrays[key], expected)
        # This native model has k-independent E=+-sqrt(.3) and zero velocity.
        # The two Gaussian levels give a mesh-independent whole-mesh DOS and
        # acoustic lifetime, derived here without the transport implementation.
        splitting = np.sqrt(0.3)
        dos_per_eV = (1 + np.exp(-0.5 * (2 * splitting / 0.1) ** 2)) / (0.1 * np.sqrt(2 * np.pi))
        dos_per_J_m3 = dos_per_eV / (1e-30 * elementary_charge)
        expected_tau = (
            hbar * 100e9 / (2 * np.pi * Boltzmann * 1000 * (2 * elementary_charge) ** 2 * dos_per_J_m3)
        )
        assert arrays["relaxation_time"].shape == (1, 2, 64)
        assert_allclose(arrays["scattering_dos"], dos_per_J_m3, rtol=3e-14)
        assert_allclose(arrays["relaxation_time"], expected_tau, rtol=3e-14, atol=0)
        assert_allclose(arrays["electron_count"], 1, atol=2e-14)
        for name in ("velocities", "conductivity", "seebeck", "thermal_conductivity", "mobility"):
            assert_allclose(arrays[name], 0, atol=0)
