"""Surface decimation checked against analytic generalized chains and exact slabs."""

import numpy as np
import pytest

from queer.execution import ExecutionOptions
from queer.models.openmx import OpenMXModel
from queer.models.wannier import NumericalError
from queer.observables.surface import surface_state


def chain(overlap=0.0, translation=1):
    r = [[0, 0, 0], [0, 0, translation], [0, 0, -translation]]
    return OpenMXModel(r, [[[0.1]], [[0.4]], [[0.4]]], [[[1]], [[overlap]], [[overlap]]])


def test_nonorthogonal_surface_and_bulk_analytic_reference():
    model = chain(0.05)
    omega, eta = np.array([-0.3, 0, 0.4, 1.2]), 0.02
    result = surface_state(model, [[0, 0, 0]], omega, eta=eta)
    z = omega + 1j * eta
    onsite, coupling = z - 0.1, z * 0.05 - 0.4
    root = np.sqrt(onsite**2 - 4 * coupling**2)
    first, second = (onsite - root) / (2 * coupling**2), (onsite + root) / (2 * coupling**2)
    green = np.where(np.abs(first) < np.abs(second), first, second)
    surface_density = -2 * (green - 0.05 * coupling * green**2).imag / np.pi
    bulk_green = 1 / (onsite - 2 * coupling**2 * green)
    bulk_density = -2 * (bulk_green - 0.1 * coupling * green * bulk_green).imag / np.pi
    np.testing.assert_allclose(result["spectrum"][:, 0], surface_density, atol=1e-11)
    np.testing.assert_allclose(result["bottom_spectrum"][:, 0], surface_density, atol=1e-11)
    np.testing.assert_allclose(result["bulk_spectrum"][:, 0], bulk_density, atol=1e-11)


def test_direct_nonorthogonal_slab_green_matches_generalized_projection():
    arguments = (chain(0.1), [[0.1, 0.2, 0]], [-0.1, 0.1, 0.6])
    diagonal = surface_state(*arguments, method="slab", slab_layers=11, eta=0.04)
    inverse = surface_state(*arguments, method="direct_green", slab_layers=11, eta=0.04)
    for name in ("spectrum", "bottom_spectrum", "bulk_spectrum"):
        np.testing.assert_allclose(diagonal[name], inverse[name], atol=1e-11)


def test_long_range_hopping_groups_complete_principal_layers():
    short = surface_state(chain(), [[0, 0, 0]], [0.2])
    long = surface_state(chain(translation=2), [[0, 0, 0]], [0.2])
    assert long.metadata["coupling_layers"] == 2
    np.testing.assert_allclose(long["spectrum"], 2 * short["spectrum"], atol=1e-11)
    with pytest.raises(ValueError, match="at least 2"):
        surface_state(chain(translation=2), [[0, 0, 0]], [0.2], coupling_layers=1)


def test_surface_convergence_and_memory_failures_are_explicit():
    with pytest.raises(NumericalError, match="converge"):
        surface_state(chain(), [[0, 0, 0]], [0.2], max_iterations=1)
    with pytest.raises(ValueError, match="too small"):
        surface_state(chain(), [[0, 0, 0]], [0.2], execution=ExecutionOptions(memory_mb=1e-10))


def test_generalized_finite_slab_bands_and_analytic_layer_projections():
    layers = 7
    theta = np.arange(1, layers + 1) * np.pi / (layers + 1)
    energy = (0.1 + 0.8 * np.cos(theta)) / (1 + 0.1 * np.cos(theta))
    order = np.argsort(energy)
    result = surface_state(chain(0.05), [[0, 0, 0], [0.2, 0.3, 0]], method="bands", slab_layers=layers)
    np.testing.assert_allclose(result["energies"], np.tile(energy[order, None], (1, 2)), atol=1e-12)
    expected = 2 / (layers + 1) * np.sin(np.arange(1, layers + 1)[:, None] * theta[None, order]) ** 2
    np.testing.assert_allclose(
        result["layer_weights"], np.repeat(expected[:, :, None], 2, axis=2), atol=1e-12
    )
    np.testing.assert_allclose(result["layer_weights"].sum(axis=0), 1, atol=1e-12)
    np.testing.assert_allclose(result["surface_weights"], result["layer_weights"][[0, -1]])
    np.testing.assert_allclose(result["spin"], 0)
    assert result["band_indices"].tolist() == list(range(layers))


@pytest.mark.parametrize("mode", ["collinear", "soc"])
def test_spinful_slab_bands_preserve_channel_order_and_physical_spin(mode):
    translations = [[0, 0, 0], [0, 0, 1], [0, 0, -1]]
    onsite = np.diag([2, -1]) if mode == "collinear" else np.array([[0, 0.4], [0.4, 0]])
    model = OpenMXModel(translations, [onsite, np.eye(2) * 0.1, np.eye(2) * 0.1], spin_mode=mode)
    layers = 5
    dispersion = np.sort(0.2 * np.cos(np.arange(1, layers + 1) * np.pi / (layers + 1)))
    result = surface_state(model, [[0, 0, 0]], method="bands", slab_layers=layers)
    if mode == "collinear":
        expected = np.concatenate([dispersion + 2, dispersion - 1])
        assert result["band_channels"].tolist() == [0] * layers + [1] * layers
        assert result["channel_labels"].tolist() == ["up"] * layers + ["down"] * layers
        np.testing.assert_allclose(result["spin"][:, 0, 2], [0.5] * layers + [-0.5] * layers)
        np.testing.assert_allclose(result["spin"][:, 0, :2], 0)
    else:
        expected = np.concatenate([dispersion - 0.4, dispersion + 0.4])
        assert result["channel_labels"].tolist() == ["soc"] * (2 * layers)
        np.testing.assert_allclose(result["spin"][:, 0, 0], [-0.5] * layers + [0.5] * layers, atol=1e-12)
        np.testing.assert_allclose(result["spin"][:, 0, 1:], 0, atol=1e-12)
    np.testing.assert_allclose(result["energies"][:, 0], expected, atol=1e-12)
    np.testing.assert_allclose(result["layer_weights"].sum(axis=0), 1, atol=1e-12)


def test_degenerate_slab_layer_and_spin_means_ignore_eigenvector_gauge(monkeypatch):
    import queer.observables.surface as surface

    model = OpenMXModel([[0, 0, 0]], [np.zeros((2, 2))], spin_mode="soc")
    arguments = {"kpoints": [[0, 0, 0]], "method": "bands", "slab_layers": 3}
    ordinary = surface_state(model, **arguments)
    original = surface.eigh
    rng = np.random.default_rng(8104)
    rotation = np.linalg.qr(rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)))[0]

    def rotate(*args, **kwargs):
        energies, coefficients = original(*args, **kwargs)
        return energies, coefficients @ rotation

    monkeypatch.setattr(surface, "eigh", rotate)
    changed = surface_state(model, **arguments)
    np.testing.assert_allclose(ordinary["layer_weights"], 1 / 3, atol=1e-12)
    np.testing.assert_allclose(changed["layer_weights"], ordinary["layer_weights"], atol=1e-12)
    np.testing.assert_allclose(ordinary["spin"], 0, atol=1e-12)
    np.testing.assert_allclose(changed["spin"], ordinary["spin"], atol=1e-12)
    assert ordinary["manifold_sizes"].tolist() == [[6]] * 6
