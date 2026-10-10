"""Independent numerical references for native electronic/projection observables."""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from queer.execution import ExecutionOptions
from queer.models.openmx import OpenMXModel
from queer.observables.electronic import (
    band_structure,
    dos,
    fat_band,
    fermi_energy,
    fermi_surface,
    find_nodes,
    jdos,
    pdos,
    reduce_basis,
    spin_texture,
)


def onsite(values=(-0.5, 0.5), **kwargs):
    return OpenMXModel([[0, 0, 0]], [np.diag(values)], **kwargs)


def test_complex_nonorthogonal_mulliken_projection_and_dos_sum_rule():
    transform = np.array([[1, 0.2j], [0.3, 1]])
    overlap = transform.conj().T @ transform
    hamiltonian = transform.conj().T @ np.diag([-0.5, 0.5]) @ transform
    model = OpenMXModel([[0, 0, 0]], [hamiltonian], [overlap])
    points = np.array([[0, 0, 0], [0.2, 0.3, 0.1]])
    coefficients = np.linalg.inv(transform)
    expected = (coefficients.conj() * (overlap @ coefficients)).real
    result = fat_band(model, points)
    np.testing.assert_allclose(result["weights"], np.repeat(expected[:, :, None], 2, axis=2), atol=1e-12)
    np.testing.assert_allclose(result["weights"].sum(axis=0), 1, atol=1e-12)
    omega = np.linspace(-1, 1, 1001)
    projected = pdos(
        model, points, omega, eta=0.03, execution=ExecutionOptions(batch_size=1, memory_mb=0.004)
    )
    total = dos(model, points, omega, eta=0.03)
    np.testing.assert_allclose(projected["pdos"].sum(axis=0), total["dos"], atol=1e-12)
    assert trapezoid(total["dos"], omega) == pytest.approx(4, abs=1e-10)


def test_projection_batches_bound_temporary_eigenstates(monkeypatch):
    model = onsite()
    original = model.eigenstates
    calls = []

    def bounded(points, **kwargs):
        calls.append(len(points))
        assert len(points) <= 2
        return original(points, **kwargs)

    monkeypatch.setattr(model, "eigenstates", bounded)
    points = np.zeros((11, 3))
    options = ExecutionOptions(batch_size=20, memory_mb=0.003)
    fat = fat_band(model, points, execution=options)
    projected = pdos(model, points, [0], execution=options)
    spin = spin_texture(model, points, execution=options)
    np.testing.assert_allclose(fat["weights"].sum(axis=0), 1)
    np.testing.assert_allclose(projected["pdos"].sum(axis=0), projected["dos"])
    np.testing.assert_allclose(spin["spin"], 0)
    assert len(calls) >= 18


def test_fermi_energy_absolute_reference_and_gaussian_jdos():
    model = onsite((3, 4), fermi_energy=3.5)
    points = [[0, 0, 0]]
    assert fermi_energy(model, points, 2)["fermi_energy"] == pytest.approx(3.5)
    assert fermi_energy(model, points, 2, temperature_kelvin=300)["fermi_energy"] == pytest.approx(3.5)
    eta = 0.02
    frequency = np.array([0.9, 1, 1.1])
    expected = 2 * np.exp(-0.5 * ((frequency - 1) / eta) ** 2) / (eta * np.sqrt(2 * np.pi))
    np.testing.assert_allclose(jdos(model, points, frequency, eta=eta)["jdos"], expected, atol=1e-12)
    np.testing.assert_allclose(jdos(model, points, frequency, eta=eta, chemical_potential=5)["jdos"], 0)
    assert fermi_surface(model, points, chemical_potential=3, width=0.01)["mask"].tolist() == [
        [True],
        [False],
    ]


def test_scissors_selection_and_basis_reduction_threshold():
    model = onsite((-0.5, 5))
    result = band_structure(model, [[0, 0, 0]], bands=1, scissors=0.25, n_occ=1)
    np.testing.assert_allclose(result["energies"], [[5.25]])
    reduced = reduce_basis(model, [[0, 0, 0]], energy_range=(-1, 1))
    np.testing.assert_allclose(reduced["scores"], [1, 0])
    assert reduced["keep_indices"].tolist() == [0]
    assert reduced["remove_indices"].tolist() == [1]


def test_soc_and_collinear_physical_spin_order():
    soc = OpenMXModel([[0, 0, 0]], [[[0, 1], [1, 0]]], spin_mode="soc")
    spin = spin_texture(soc, [[0, 0, 0]])["spin"]
    np.testing.assert_allclose(spin[:, 0], [[-0.5, 0, 0], [0.5, 0, 0]], atol=1e-12)
    collinear = onsite((2, -1), spin_mode="collinear")
    result = spin_texture(collinear, [[0, 0, 0]])
    np.testing.assert_allclose(result["energies"][:, 0], [2, -1])
    np.testing.assert_allclose(result["spin"][:, 0], [[0, 0, 0.5], [0, 0, -0.5]])
    resolved = onsite((-1, 2, -3, 4), spin_mode="collinear")
    band = band_structure(resolved, [[0, 0, 0]], bands=[1], spin="down", scissors=0.5, n_occ=1)
    np.testing.assert_allclose(band["energies"], [[4.5]])
    assert band["band_indices"].tolist() == [3]
    assert band["band_channels"].tolist() == [1]
    assert band["channel_labels"].tolist() == ["down"]


def test_periodic_three_dimensional_node_refinement():
    pauli = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
    translations, hopping = [[0, 0, 0]], [np.zeros((2, 2), complex)]
    for axis, matrix in enumerate(pauli):
        vector = np.eye(3, dtype=int)[axis]
        translations.extend((vector, -vector))
        hopping.extend((matrix / (2j), -matrix / (2j)))
    model = OpenMXModel(translations, hopping, spin_degeneracy=1)
    result = find_nodes(model, initial_grid=(2, 2, 2), threshold=1e-8, initial_threshold=1e-7)
    assert len(result["kpoints"]) == 8
    np.testing.assert_allclose(result["gaps"], 0, atol=1e-12)
    np.testing.assert_allclose((2 * result["kpoints"]) % 1, 0, atol=1e-9)


@pytest.mark.parametrize("function,args", [(dos, ([],)), (pdos, ([],)), (fermi_energy, (1,))])
def test_density_empty_mesh_rejected(function, args):
    with pytest.raises(ValueError, match="k-point"):
        function(onsite(), np.empty((0, 3)), *args)


def test_complete_degenerate_projection_means_are_gauge_invariant_and_normalized():
    angle, phase = 0.37, 0.61
    rotation = np.array(
        [
            [np.cos(angle), -np.exp(-1j * phase) * np.sin(angle)],
            [np.exp(1j * phase) * np.sin(angle), np.cos(angle)],
        ]
    )

    class RotatingModel(OpenMXModel):
        def __init__(self, gauge):
            super().__init__([[0, 0, 0]], [np.zeros((2, 2))], spin_mode="soc")
            self.gauge = gauge

        def eigenstates(self, points, **kwargs):
            result = super().eigenstates(points, **kwargs)
            return type(result)(
                result.energies, result.coefficients @ self.gauge, result.overlaps, result.band_channels
            )

    first, rotated = RotatingModel(np.eye(2)), RotatingModel(rotation)
    points = [[0, 0, 0], [0.2, 0.3, 0.4]]
    fat = fat_band(first, points)
    changed = fat_band(rotated, points)
    np.testing.assert_allclose(fat["weights"], 0.5, atol=1e-12)
    np.testing.assert_allclose(changed["weights"], fat["weights"], atol=1e-12)
    np.testing.assert_allclose(fat["weights"].sum(axis=0), 1, atol=1e-12)
    assert fat["manifold_sizes"].tolist() == [[2, 2], [2, 2]]
    spin = spin_texture(first, points)
    np.testing.assert_allclose(spin["spin"], 0, atol=1e-12)
    np.testing.assert_allclose(spin_texture(rotated, points)["spin"], spin["spin"], atol=1e-12)
    np.testing.assert_allclose(reduce_basis(rotated, points)["scores"], [0.5, 0.5], atol=1e-12)
    for function in (fat_band, spin_texture, reduce_basis):
        with pytest.raises(ValueError, match="cuts a degenerate subspace"):
            function(first, points, bands=[0])
    np.testing.assert_allclose(fat_band(rotated, points, bands=[1, 0])["weights"], 0.5, atol=1e-12)
    assert spin.metadata["degenerate_state_interpretation"] == "complete_subspace_mean_per_state"
    with pytest.raises(ValueError, match="occupied boundary cuts"):
        band_structure(first, points, scissors=0.1, n_occ=1)
    # Coincident energies in distinct collinear channels remain resolved.
    collinear = onsite((0, 0), spin_mode="collinear")
    np.testing.assert_allclose(spin_texture(collinear, points, bands=[0])["spin"][:, :, 2], 0.5)
