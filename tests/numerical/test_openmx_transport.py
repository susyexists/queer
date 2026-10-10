"""Boltzmann transport checked against dilute parabolic-band limits."""

import numpy as np
import pytest
from scipy.constants import Boltzmann, electron_mass, elementary_charge, hbar

from queer.execution import ExecutionOptions
from queer.models.openmx import OpenMXModel
from queer.observables.transport import boltz_transport, finalize_transport


class ParabolicModel(OpenMXModel):
    def __init__(self):
        super().__init__([[0, 0, 0]], [[[0]]], lattice=np.eye(3) * 10, position=[[[[0]]], [[[0]]], [[[0]]]])
        self.curvature = hbar**2 / (2 * electron_mass) / elementary_charge / 1e-20

    def hamiltonian(self, k):
        q = np.asarray(k) @ self.g_vec
        return np.asarray([[self.curvature * (q @ q)]])

    def derivatives(self, k, order=1):
        q = np.asarray(k) @ self.g_vec
        value = (
            2 * self.curvature * q[:, None, None]
            if order == 1
            else 2 * self.curvature * np.eye(3)[:, :, None, None]
        )
        return {"H": value, "S": np.zeros_like(value), "r": np.zeros((3, *value.shape), complex)}


def test_crta_parabolic_conductivity_seebeck_and_lorenz_reference():
    model = ParabolicModel()
    q = np.linspace(-0.3, 0.3, 17)
    points = np.stack(np.meshgrid(q, q, q, indexing="ij"), axis=-1).reshape(-1, 3) @ np.linalg.inv(
        model.g_vec
    )
    temperature, mu, tau = 300, -0.3, 12
    result = boltz_transport(model, points, [mu], [temperature], relaxation_time=tau)
    density = result["electron_density"][0, 0]
    sigma = result["conductivity"][0, 0]
    expected_sigma = density * elementary_charge**2 * tau * 1e-15 / electron_mass
    np.testing.assert_allclose(np.diag(sigma), expected_sigma, rtol=2e-4)
    np.testing.assert_allclose(sigma - np.diag(np.diag(sigma)), 0, atol=1e-8)
    expected_seebeck = (
        -Boltzmann / elementary_charge * (2.5 - mu * elementary_charge / (Boltzmann * temperature))
    )
    np.testing.assert_allclose(np.diag(result["seebeck"][0, 0]), expected_seebeck, rtol=3e-5)
    lorenz = np.diag(result["thermal_conductivity"][0, 0]) / (np.diag(sigma) * temperature)
    np.testing.assert_allclose(lorenz, 2.5 * (Boltzmann / elementary_charge) ** 2, rtol=2e-4)
    np.testing.assert_allclose(
        np.diag(result["mobility"][0, 0]), elementary_charge * tau * 1e-15 / electron_mass, rtol=2e-4
    )


def test_effective_mass_curvature_and_empc_parameter_scaling():
    model = ParabolicModel()
    points = [[0.1, 0.1, 0.1], [0.2, 0, 0], [0, 0.2, 0]]
    mass = boltz_transport(model, points, [0.01], [100], effective_mass=True, relaxation_time=10)
    expected = np.broadcast_to(np.eye(3) / electron_mass, (1, 3, 3, 3))
    np.testing.assert_allclose(mass["inverse_mass"], expected, rtol=1e-10, atol=1e-10 / electron_mass)
    np.testing.assert_allclose(mass["effective_mass"][0, 0], np.eye(3) * electron_mass, rtol=1e-10)
    first = boltz_transport(
        model, points, [0.01], [100, 200], transport_method="EMPC", deformation_potential=2, young_modulus=100
    )
    second = boltz_transport(
        model, points, [0.01], [100], transport_method="EMPC", deformation_potential=4, young_modulus=100
    )
    np.testing.assert_allclose(first["relaxation_time"][1], first["relaxation_time"][0] / 2)
    np.testing.assert_allclose(second["relaxation_time"][0], first["relaxation_time"][0] / 4)
    assert np.all(first["relaxation_time"] > 0)
    with pytest.raises(ValueError, match="requires"):
        boltz_transport(model, points, transport_method="EMPC")


def test_crta_additive_moments_reconstruct_complete_mesh_response():
    model = ParabolicModel()
    points = np.array([[0.1, 0.1, 0.1], [0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]])
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    settings = {
        "chemical_potentials": [0.01, 0.02],
        "temperature_kelvin": [100, 200],
        "relaxation_time": 10,
        "effective_mass": True,
    }
    complete = boltz_transport(model, points, weights=weights, **settings)
    additive = {f"transport_moment_{order}" for order in range(3)} | {
        "electron_density",
        "hole_density",
        "carrier_dos",
        "electron_count",
        "inverse_mass_numerator",
        "inverse_mass_normalization",
    }
    arrays = {key: value.copy() for key, value in complete.arrays.items()}
    for key in additive:
        arrays[key] = np.zeros_like(arrays[key])
    for selection in (slice(0, 1), slice(1, 4)):
        tile = boltz_transport(model, points[selection], weights=weights[selection], **settings)
        for key in additive:
            arrays[key] += weights[selection].sum() * tile[key]
    reconstructed = finalize_transport(arrays)
    for key in ("conductivity", "seebeck", "thermal_conductivity", "mobility", "effective_mass"):
        np.testing.assert_allclose(reconstructed[key], complete[key], rtol=1e-12, atol=1e-30)
    with pytest.raises(ValueError, match="explicit relaxation_time"):
        boltz_transport(model, points)


def test_parallel_transport_preserves_bounded_serial_inner_solves():
    model = ParabolicModel()
    points = [[0.1, 0.1, 0.1], [0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]]
    original = model.execution
    settings = {"chemical_potentials": [0.01], "temperature_kelvin": 100, "relaxation_time": 10}
    serial = boltz_transport(model, points, **settings)
    parallel = boltz_transport(
        model, points, **settings, execution=ExecutionOptions(workers=2, batch_size=3, memory_mb=0.003)
    )
    for name in ("conductivity", "seebeck", "thermal_conductivity", "mobility", "velocities"):
        np.testing.assert_allclose(parallel[name], serial[name], rtol=1e-12, atol=1e-12)
    assert model.execution is original
    with pytest.raises(ValueError, match="too small"):
        boltz_transport(model, points, **settings, execution=ExecutionOptions(memory_mb=0.0001))


def test_degenerate_transport_block_trace_is_isotropic_and_gauge_invariant():
    pauli = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])

    class DiracModel(OpenMXModel):
        def __init__(self, rotation):
            super().__init__([[0, 0, 0]], [np.zeros((2, 2))], spin_mode="soc")
            self.slopes = rotation.conj().T @ pauli @ rotation

        def hamiltonian(self, k):
            return np.einsum("a,aij->ij", np.asarray(k) @ self.g_vec, self.slopes)

        def derivatives(self, k, order=1):
            matrix = self.slopes if order == 1 else np.zeros((3, 3, 2, 2), complex)
            return {"H": matrix, "S": np.zeros_like(matrix), "r": np.zeros((3, *matrix.shape), complex)}

    rotation = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    settings = {
        "kpoints": [[0, 0, 0]],
        "temperature_kelvin": 100,
        "relaxation_time": 10,
        "effective_mass": True,
    }
    first = boltz_transport(DiracModel(np.eye(2)), **settings)
    rotated = boltz_transport(DiracModel(rotation), **settings)
    velocity_unit = elementary_charge * 1e-10 / hbar
    derivative_eV = elementary_charge / (4 * Boltzmann * 100)
    sigma = elementary_charge * 10e-15 * 2 * velocity_unit**2 * derivative_eV / 1e-30
    np.testing.assert_allclose(first["conductivity"][0, 0], np.eye(3) * sigma, rtol=1e-12)
    np.testing.assert_allclose(rotated["conductivity"], first["conductivity"], rtol=1e-12, atol=1)
    assert not first["resolved_band_mask"].any()
    assert not first["effective_mass_valid_mask"].any()
    assert np.isnan(first["inverse_mass"]).all() and np.isnan(first["effective_mass"]).all()
