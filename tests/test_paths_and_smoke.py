import json
from pathlib import Path

import numpy as np
import pytest

import queer
from queer.arpes import arpes_mesh, arpes_path, binding_k
from queer.kpath import named_k_path, parse_point_names, seekpath_points
from queer.mesh import angstrom2reciprocal, crystal2cartesian, reciprocal2angstrom
from queer.paths import data_path, project_root, results_path


def test_project_paths_resolve(monkeypatch, tmp_path):
    root = project_root()
    assert (root / "src" / "queer").is_dir()

    monkeypatch.setenv("QUEER_DATA_DIR", str(tmp_path / "data-root"))
    monkeypatch.setenv("QUEER_RESULTS_DIR", str(tmp_path / "results-root"))

    assert data_path("materials", "ag").parent == tmp_path / "data-root" / "materials"
    assert results_path("v0_sweep").parent == tmp_path / "results-root"


def test_mesh_conversion_legacy_helpers_round_trip():
    g_vec = np.eye(3) * 2.0
    points = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    angstrom = reciprocal2angstrom(points, g_vec)
    reciprocal = angstrom2reciprocal(angstrom, g_vec)
    np.testing.assert_allclose(reciprocal, points)

    legacy = crystal2cartesian(angstrom.T, g_vec)
    np.testing.assert_allclose(legacy, reciprocal)


def test_arpes_mesh_and_path_helpers_accept_current_and_legacy_calls():
    mesh = arpes_mesh(21.2, 0.0, 0.0, 12.0, 4, 2, align=[[0, 0, 1], [1, 1, 1]])
    assert mesh.shape == (16, 3)

    kmesh = binding_k(21.2, 0.0, [4.7, 4.8], 0.5, 12.0, 4, 2)
    assert list(kmesh.keys())

    path = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    g_vec = np.eye(3)
    assert arpes_path(path, g_vec, Ek=21.2, V0=15).shape == path.shape
    assert arpes_path(path, g_vec, photon_energy=21.2, binding_energy=0.0, fermi_energy=0.0).shape == path.shape


def test_seekpath_named_k_path_for_ag_primitive():
    poscar = data_path("materials", "ag_primitive", "POSCAR")
    if not poscar.exists():
        pytest.skip("local Ag primitive POSCAR is not available")

    points = seekpath_points(poscar)
    assert {"GAMMA", "X", "W", "K", "L"}.issubset(points)

    sym, path, labels = named_k_path(["GAMMA", "X", "W", "K", "GAMMA", "L"], 200, poscar)
    assert labels == ["Γ", "X", "W", "K", "Γ", "L"]
    assert sym.tolist() == [0, 47, 70, 82, 143, 200]
    assert path.shape == (201, 3)
    np.testing.assert_allclose(path[0], points["GAMMA"])
    np.testing.assert_allclose(path[sym[1]], points["X"])
    assert parse_point_names("GAMMA-X-W-K-GAMMA-L") == ["GAMMA", "X", "W", "K", "GAMMA", "L"]


def test_ag_primitive_model_smoke():
    material = data_path("materials", "ag_primitive")
    if not (material / "wannier90_hr.dat").exists():
        pytest.skip("local Ag primitive Wannier data is not available")

    model = queer.model(path=material, ef=8.310342, poscar="POSCAR", num_core=1)
    energy = model.calculate_energy(np.array([[0.0, 0.0, 0.0]]))
    assert energy.shape == (model.nbnd, 1)
    assert np.isfinite(energy).all()

    sym, path, labels = model.kpath("GAMMA-X-W-K-GAMMA-L", 200)
    assert labels == ["Γ", "X", "W", "K", "Γ", "L"]
    assert sym.tolist() == [0, 47, 70, 82, 143, 200]
    assert path.shape == (201, 3)


def test_active_notebook_uses_organized_paths():
    notebook = project_root() / "notebooks" / "active" / "Ag_primitive_single.ipynb"
    data = json.loads(notebook.read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in data["cells"])

    assert 'data_path("materials", "ag_primitive")' in source
    assert "model.kpath(" in source
    assert "results_path(" in source
    assert "custom_points = [" not in source
    assert "path_create(" not in source
    assert "named_k_path(" not in source
    assert "./Ag_primitive/" not in source
