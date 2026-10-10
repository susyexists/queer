"""Analytic and nonorthogonal acceptance references for pair geometry."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from queer.models.openmx import OpenMXModel
from queer.models.wannier import WannierModel
from queer.observables.geometry import band_pair_geometry, berry_curvature, quantum_metric

PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
TRANSLATIONS = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])


def qwz(lattice=np.eye(3), mass=-1):
    x, y, z = PAULI
    hopping = [mass * z, (z - 1j * x) / 2, (z + 1j * x) / 2, (z - 1j * y) / 2, (z + 1j * y) / 2]
    return WannierModel(TRANSLATIONS, hopping, lattice=lattice, fermi_energy=0.3)


def test_analytic_pair_total_metric_curvature_velocity_and_units():
    model = qwz()
    result = band_pair_geometry(model, [[0, 0, 0]], bands=(0, 1))
    assert_allclose(result["energies"][:, 0], [-1, 1])
    assert_allclose(result["gap"], [2])
    assert_allclose(result["pair_metric"][0], np.diag([0.25, 0.25, 0]))
    assert_allclose(result["total_metric"][:, 0], np.broadcast_to(result["pair_metric"][0], (2, 3, 3)))
    assert_allclose(result["curvature"][:, 0, 0, 1], [0.5, -0.5])
    assert_allclose(abs(result["velocity_cv"])[0], [1, 1, 0])
    assert result["valid"].all() and result["band_valid"].all()
    assert result.units["gap"] == "eV"
    assert result.units["velocity_cv"] == "eV Angstrom"
    assert result.metadata["bands"] == [0, 1]
    assert_allclose(result["pair_metric"], quantum_metric(model, [[0, 0, 0]], occupied_bands=1)["metric"])
    assert_allclose(result["curvature"], berry_curvature(model, [[0, 0, 0]])["curvature"])
    inferred = band_pair_geometry(model, [[0, 0, 0]], occupied_bands=1)
    assert_allclose(inferred["pair_metric"], result["pair_metric"])


def test_metric_cartesian_rotation_and_skew_cell():
    lattice = np.array([[2, 0.4, 0.2], [0.1, 1.5, 0.3], [0, 0, 8]])
    angle = 0.63
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    metric = band_pair_geometry(qwz(lattice), [[0, 0, 0]], bands=(0, 1))["pair_metric"][0]
    expected = 0.25 * (np.outer(lattice[0], lattice[0]) + np.outer(lattice[1], lattice[1]))
    assert_allclose(metric, expected, atol=1e-13)
    rotated = band_pair_geometry(qwz(lattice @ rotation), [[0, 0, 0]], bands=(0, 1))["pair_metric"][0]
    assert_allclose(rotated, rotation.T @ metric @ rotation, atol=1e-13)


def test_pair_metric_is_distinct_from_all_band_total():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    hopping = np.stack([np.diag([-3, -1, 1, 4]), x * 0.1, x.conj().T * 0.1])
    model = WannierModel([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], hopping, lattice=np.eye(3))
    result = band_pair_geometry(model, [[0.17, 0, 0]], bands=(1, 2))
    assert result["total_metric"][0, 0, 0, 0] > result["pair_metric"][0, 0, 0]
    assert result["total_metric"][1, 0, 0, 0] > result["pair_metric"][0, 0, 0]


def test_independent_gap_survives_large_absolute_energy_offset():
    coupling = 1e-12
    h = 5 * np.eye(2) + coupling * PAULI[0]
    model = WannierModel([[0, 0, 0]], h[None], lattice=np.eye(3))
    result = band_pair_geometry(model, [[0, 0, 0]], bands=(0, 1), degeneracy_tol=1e-14)
    assert_allclose(result["gap"], 2 * coupling, rtol=1e-14)
    assert abs(float(np.diff(result["energies"][:, 0])[0]) - 2 * coupling) > 1e-17
    assert result["gap_valid"].all()
    assert result["precision_floor"][0] < coupling


def test_degeneracies_are_undefined_and_masked_without_fake_zeros():
    model = WannierModel([[0, 0, 0]], np.diag([0, 0, 1])[None], lattice=np.eye(3))
    degenerate_pair = band_pair_geometry(model, [[0, 0, 0]], bands=(0, 1))
    assert not degenerate_pair["valid"].any()
    assert not degenerate_pair["gap_valid"].any()
    assert np.isnan(degenerate_pair["pair_metric"]).all()
    assert np.isnan(degenerate_pair["total_metric"]).all()
    assert np.isnan(degenerate_pair["curvature"]).all()
    remote = band_pair_geometry(model, [[0, 0, 0]], bands=(1, 2))
    assert remote["gap_valid"].all()
    assert remote["band_valid"][:, 0].tolist() == [False, True]
    assert np.isnan(remote["pair_metric"]).all()
    assert np.isnan(remote["total_metric"][0]).all()
    assert np.isfinite(remote["total_metric"][1]).all()


def test_nonorthogonal_covariance_and_same_channel_selection():
    model = qwz()
    transform = np.array([[1.2 + 0.1j, 0.2 - 0.3j], [-0.1 + 0.2j, 0.8]])
    h = np.einsum("in,rij,jm->rnm", transform.conj(), model.hopping_matrices, transform)
    s = np.zeros_like(h)
    s[0] = transform.conj().T @ transform
    changed = OpenMXModel(
        TRANSLATIONS, h, s, position=np.zeros((3, 5, 2, 2)), lattice=np.eye(3), spin_degeneracy=1
    )
    points = [[0.1, 0.2, 0], [0.27, 0.36, 0]]
    first = band_pair_geometry(model, points, bands=(0, 1))
    second = band_pair_geometry(changed, points, bands=(0, 1))
    for key in ("energies", "gap", "pair_metric", "total_metric", "curvature"):
        assert_allclose(second[key], first[key], atol=3e-12)
    assert_allclose(abs(second["velocity_cv"]), abs(first["velocity_cv"]), atol=3e-12)
    full_h = np.zeros((1, 4, 4), complex)
    full_h[0] = np.diag([-1, 1, -0.5, 2])
    collinear = OpenMXModel(
        [[0, 0, 0]],
        full_h,
        np.eye(4)[None],
        position=np.zeros((3, 1, 4, 4)),
        lattice=np.eye(3),
        spin_mode="collinear",
    )
    assert_allclose(band_pair_geometry(collinear, [[0, 0, 0]], bands=(2, 3))["gap"], 2.5)
    with pytest.raises(ValueError, match="same spin channel"):
        band_pair_geometry(collinear, [[0, 0, 0]], bands=(0, 3))
    with pytest.raises(ValueError, match="same spin channel"):
        band_pair_geometry(collinear, [[0, 0, 0]], occupied_bands=2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"bands": (0, 1), "occupied_bands": 1},
        {"bands": (1, 0)},
        {"bands": (0, 0)},
        {"bands": (0.0, 1.0)},
        {"occupied_bands": True},
        {"occupied_bands": 0},
    ],
)
def test_pair_selection_is_explicit_and_validated(kwargs):
    with pytest.raises(ValueError):
        band_pair_geometry(qwz(), [[0, 0, 0]], **kwargs)
