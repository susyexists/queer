"""Independent dipole, finite-field, Rice--Mele, and Weyl references."""

import numpy as np
import pytest
from scipy.constants import elementary_charge as E
from scipy.constants import epsilon_0, hbar

from queer.execution import ExecutionOptions
from queer.observables.nonlinear import cpge, pockels, second_order_static, shg, shift_current
from queer.observables.optical import dielectric_function, optical_conductivity

PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], complex)


class DipoleModel:
    """Isolated finite-level molecules repeated without hopping."""

    def __init__(self, energies=(-1.0, 1.0), dipole=None, spin_degeneracy=1):
        self.nbnd = len(energies)
        self.lattice = 10 * np.eye(3)
        self.g_vec = 2 * np.pi * np.linalg.inv(self.lattice).T
        self.fermi_energy = 0.0
        self.spin_degeneracy = spin_degeneracy
        self.h = np.diag(energies).astype(complex)
        self.r = np.zeros((3, self.nbnd, self.nbnd), complex)
        self.r[0] = [[0, 0.7], [0.7, 0.3]] if dipole is None else dipole

    def hamiltonian(self, k):
        return self.h

    def overlap(self, k):
        return np.eye(self.nbnd)

    def position(self, k):
        return self.r

    def derivatives(self, k, order=1):
        shape = (3,) * order + (self.nbnd, self.nbnd)
        return {
            "H": np.zeros(shape, complex),
            "S": np.zeros(shape, complex),
            "r": np.zeros((3, *shape), complex),
        }


def molecular_chi2(z1, z2, gap=2.0, dipole=0.7, permanent=0.3, volume=1000.0):
    """Closed two-level polarization, including population redistribution."""
    zout = z1 + z2
    coherence = 0.5 * (
        (1 / (z1 + gap) + 1 / (z2 + gap)) / (zout + gap) + (1 / (z1 - gap) + 1 / (z2 - gap)) / (zout - gap)
    )
    population = -0.5 / zout * (1 / (z1 - gap) + 1 / (z1 + gap) + 1 / (z2 - gap) + 1 / (z2 + gap))
    return -E / (epsilon_0 * volume) * permanent * dipole**2 * (coherence + population) * 1e12


def test_optical_two_level_closed_line_shape_dielectric_and_passivity():
    model = DipoleModel()
    omega = np.array([0.0, 0.4, 2.0, 3.2])
    eta = 0.03
    z = omega + 1j * eta
    expected = 1j * E**2 / hbar * 1e8 / 1000 * 2 * 0.7**2 * (1 / (z - 2) + 1 / (z + 2))
    actual = optical_conductivity(model, omega, eta=eta, grid=(1, 1, 1))
    np.testing.assert_allclose(actual["conductivity"][:, 0, 0], expected, rtol=1e-12, atol=1e-12)
    expected_dielectric = 1 + E * 1e10 / (epsilon_0 * 1000) * 4 * 0.7**2 / (4 - z**2)
    np.testing.assert_allclose(actual["dielectric"][:, 0, 0], expected_dielectric, rtol=1e-12)
    assert np.all(actual["conductivity"][:, 0, 0].real >= 0)
    np.testing.assert_array_equal(actual["conductivity"][:, 1:, :], 0)
    np.testing.assert_allclose(
        dielectric_function(model, omega, eta=eta, grid=(1, 1, 1))["dielectric"], actual["dielectric"]
    )
    assert actual.units["conductivity"] == "S/cm"


def test_optical_spin_occupation_and_scissors_have_quantitative_effects():
    omega = [1.7, 2.0, 2.5]
    base = optical_conductivity(DipoleModel(), omega, grid=(1, 1, 1))["conductivity"]
    doubled = optical_conductivity(DipoleModel(spin_degeneracy=2), omega, grid=(1, 1, 1))["conductivity"]
    np.testing.assert_allclose(doubled, 2 * base, rtol=1e-12)
    empty = optical_conductivity(DipoleModel(), omega, chemical_potential=-3, grid=(1, 1, 1))
    np.testing.assert_array_equal(empty["conductivity"], 0)
    shifted = optical_conductivity(DipoleModel(), omega, scissors=0.5, eta=0.03, grid=(1, 1, 1))
    z = np.asarray(omega) + 0.03j
    expected = 1j * E**2 / hbar * 1e8 / 1000 * 2.5 * 0.7**2 * (1 / (z - 2.5) + 1 / (z + 2.5))
    np.testing.assert_allclose(shifted["conductivity"][:, 0, 0], expected, rtol=1e-12)
    temperature = 10000.0
    thermal = 8.617333262145e-5 * temperature
    warm = optical_conductivity(DipoleModel(), omega, temperature_kelvin=temperature, grid=(1, 1, 1))
    np.testing.assert_allclose(warm["conductivity"], np.tanh(1 / (2 * thermal)) * base, rtol=1e-12)


def test_shg_and_pockels_nonzero_two_level_frequency_reference():
    model = DipoleModel()
    omega = np.array([0.2, 0.7, 1.8])
    eta = 0.02
    harmonic = shg(model, omega, eta=eta, grid=(1, 1, 1))
    np.testing.assert_allclose(
        harmonic["shg"][:, 0, 0, 0], molecular_chi2(omega + 1j * eta, omega + 1j * eta), rtol=1e-11
    )
    electrooptic = pockels(model, omega, eta=eta, grid=(1, 1, 1))
    reference_chi = molecular_chi2(omega + 1j * eta, 1j * eta)
    np.testing.assert_allclose(electrooptic["chi2"][:, 0, 0, 0], reference_chi, rtol=1e-11)
    expected_r = -2 * reference_chi / electrooptic["dielectric"][:, 0, 0] ** 2
    np.testing.assert_allclose(electrooptic["pockels"][:, 0, 0, 0], expected_r, rtol=1e-12)
    assert np.max(np.abs(harmonic["shg"])) > 1
    assert electrooptic.units["chi2"] == "pm/V"


def test_static_three_level_rayleigh_and_independent_finite_field_reference():
    energy = np.array([-1.0, 1.1, 2.4])
    dipole = np.array([[0.15, 0.6, -0.2], [0.6, 0.4, 0.3], [-0.2, 0.3, -0.1]])
    model = DipoleModel(energy, dipole)
    denominator = energy[0] - energy[1:]
    cubic = np.sum(
        dipole[0, 1:, None]
        * dipole[1:, 1:]
        * dipole[None, 1:, 0]
        / denominator[:, None]
        / denominator[None, :]
    )
    cubic -= dipole[0, 0] * np.sum(dipole[0, 1:] ** 2 / denominator**2)
    reference = -3 * E / (epsilon_0 * 1000) * cubic * 1e12
    actual = second_order_static(model, eta=1e-5, grid=(1, 1, 1))["second_order_static"][0, 0, 0]
    assert actual == pytest.approx(reference, rel=2e-8)
    fields = np.linspace(-0.005, 0.005, 15)
    ground_energies = np.array([np.linalg.eigvalsh(np.diag(energy) + field * dipole)[0] for field in fields])
    coefficients = np.polynomial.polynomial.polyfit(fields, ground_energies, 5)
    finite_field = -3 * E / (epsilon_0 * 1000) * coefficients[3] * 1e12
    assert actual == pytest.approx(finite_field, rel=2e-6)
    assert abs(actual) > 0.1


def test_static_full_cartesian_tensor_matches_independent_finite_field_polarization():
    model = DipoleModel((-1.0, 1.1, 2.4), [[0.15, 0.6, -0.2], [0.6, 0.4, 0.3], [-0.2, 0.3, -0.1]])
    model.r[1] = [[-0.2, 0.4, 0.7], [0.4, 0.1, -0.1], [0.7, -0.1, 0.2]]
    model.r[2] = [[0.1, 0.1, -0.3], [0.1, 0.2, 0.5], [-0.3, 0.5, 0.4]]

    def polarization(field):
        _, vectors = np.linalg.eigh(model.h + np.einsum("a,aij->ij", field, model.r))
        ground = vectors[:, 0]
        return np.einsum("i,aij,j->a", ground.conj(), model.r, ground).real

    step = 1e-4
    expected = np.zeros((3, 3, 3))
    for b in range(3):
        for c in range(3):
            fb, fc = np.eye(3)[b] * step, np.eye(3)[c] * step
            expected[:, b, c] = (
                polarization(fb + fc)
                - polarization(fb - fc)
                - polarization(-fb + fc)
                + polarization(-fb - fc)
            ) / (4 * step**2)
    expected *= -E / (epsilon_0 * 1000) * 1e12 / 2
    actual = second_order_static(model, eta=1e-6, grid=(1, 1, 1))["second_order_static"]
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(actual, actual.transpose(1, 0, 2), atol=1e-10)
    assert np.max(np.abs(model.r[0] @ model.r[1] - model.r[1] @ model.r[0])) > 0.1


def test_optical_projected_coordinate_contact_matches_direct_dipole_polarization():
    model = DipoleModel()
    model.r[1] = [[0.1, 0.4j], [-0.4j, -0.2]]
    omega = np.array([0.0, 0.6, 1.9])
    eta = 0.02
    z = omega + 1j * eta
    energy = np.diag(model.h).real
    occupied = np.array([1.0, 0.0])
    density = model.r[None] * (occupied[None, :] - occupied[:, None])[None, None]
    density = density / (z[:, None, None, None] - (energy[:, None] - energy[None, :])[None, None])
    dipole_response = np.einsum("anm,wbmn->wab", model.r, density)
    expected_sigma = 1j * E**2 / hbar * 1e8 / 1000 * z[:, None, None] * dipole_response
    expected_dielectric = np.eye(3)[None] - E / (epsilon_0 * 1000) * 1e10 * dipole_response
    result = optical_conductivity(model, omega, eta=eta, grid=(1, 1, 1))
    np.testing.assert_allclose(result["conductivity"], expected_sigma, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(result["dielectric"], expected_dielectric, rtol=1e-11, atol=1e-11)


class RiceMeleModel:
    nbnd = 2
    fermi_energy = 0.0
    spin_degeneracy = 1
    lattice = np.eye(3)
    g_vec = 2 * np.pi * np.eye(3)

    def __init__(self, t1=1.1, t2=0.7, staggering=0.4):
        self.t1, self.t2, self.staggering = t1, t2, staggering

    def hamiltonian(self, k):
        x = 2 * np.pi * np.asarray(k)[0]
        return (
            (self.t1 + self.t2 * np.cos(x)) * PAULI[0]
            + self.t2 * np.sin(x) * PAULI[1]
            + self.staggering * PAULI[2]
        )

    def overlap(self, k):
        return np.eye(2)

    def position(self, k):
        r = np.zeros((3, 2, 2))
        r[0] = np.diag([0.0, 0.5])
        return r

    def derivatives(self, k, order=1):
        x = 2 * np.pi * np.asarray(k)[0]
        shape = (3,) * order + (2, 2)
        h = np.zeros(shape, complex)
        if order == 1:
            h[0] = -self.t2 * np.sin(x) * PAULI[0] + self.t2 * np.cos(x) * PAULI[1]
        else:
            h[0, 0] = -self.t2 * np.cos(x) * PAULI[0] - self.t2 * np.sin(x) * PAULI[1]
        return {"H": h, "S": np.zeros_like(h), "r": np.zeros((3, *shape), complex)}


def rice_mele_reference(model, points, omega, eta):
    """Differentiate wavefunctions directly, independently of response kernels."""
    values = []
    h = 2e-4  # Cartesian inverse Angstrom; nested central differences.
    for k in points:
        energy, reference = np.linalg.eigh(model.hamiltonian(k))

        def coeff(x):
            _, vectors = np.linalg.eigh(model.hamiltonian(k + [x / (2 * np.pi), 0, 0]))
            phase = np.diag(reference.conj().T @ vectors)
            return vectors * (phase.conj() / np.abs(phase))[None]

        def connection(x):
            c = coeff(x)
            dc = (coeff(x + h) - coeff(x - h)) / (2 * h)
            return 1j * c.conj().T @ dc + c.conj().T @ model.position(k)[0] @ c

        a = connection(0)
        da = (connection(h) - connection(-h)) / (2 * h)
        generalized = da[0, 1] - 1j * (a[0, 0] - a[1, 1]) * a[0, 1]
        integrand = (a[1, 0] * generalized).imag
        line = eta / (np.pi * ((omega - (energy[1] - energy[0])) ** 2 + eta**2))
        values.append(integrand * line)
    return -np.pi * E**2 / hbar * 1e6 * np.mean(values, axis=0)


def test_shift_current_rice_mele_independent_wavefunction_derivatives_and_inversion():
    points = np.column_stack((np.arange(40) / 40, np.zeros((40, 2))))
    model = RiceMeleModel()
    omega = np.array([1.0, 2.0, 3.0])
    eta = 0.07
    actual = shift_current(model, omega, eta=eta, kpoints=points)
    expected = rice_mele_reference(model, points, omega, eta)
    np.testing.assert_allclose(actual["shift_current"][:, 0, 0, 0], expected, rtol=2e-6, atol=2e-5)
    assert np.max(np.abs(expected)) > 1
    symmetric = shift_current(RiceMeleModel(staggering=0.0), omega, eta=eta, kpoints=points)
    np.testing.assert_allclose(symmetric["shift_current"], 0, atol=1e-8)


class WeylModel(RiceMeleModel):
    """Linear chiral Weyl cone over a spherical reciprocal integration domain."""

    def __init__(self, radius=3.0):
        integration_volume = 4 * np.pi * radius**3 / 3
        self.lattice = ((2 * np.pi) ** 3 / integration_volume) ** (1 / 3) * np.eye(3)
        self.g_vec = 2 * np.pi * np.linalg.inv(self.lattice).T

    def hamiltonian(self, k):
        return np.einsum("a,aij->ij", np.asarray(k) @ self.g_vec, PAULI)

    def position(self, k):
        return np.zeros((3, 2, 2), complex)

    def derivatives(self, k, order=1):
        h = PAULI.copy() if order == 1 else np.zeros((3, 3, 2, 2), complex)
        return {"H": h, "S": np.zeros_like(h), "r": np.zeros((3, *h.shape), complex)}


def test_cpge_weyl_quantized_trace_and_steady_conversion():
    model = WeylModel()
    nodes, radial_weights = np.polynomial.legendre.leggauss(256)
    radii = 1.5 * (nodes + 1)
    directions = np.vstack((np.eye(3), -np.eye(3)))
    cartesian = (radii[:, None, None] * directions[None]).reshape(-1, 3)
    points = cartesian @ np.linalg.inv(model.g_vec)
    weights = np.repeat(radial_weights * 1.5 * radii**2 / 6, 6)
    result = cpge(
        model,
        [2.0],
        eta=0.04,
        smearing="gaussian",
        kpoints=points,
        weights=weights,
        relaxation_time_seconds=1e-12,
    )
    quantum = -np.pi * E**3 / (2 * np.pi * hbar) ** 2
    assert np.trace(result["helicity_rate"][0]) == pytest.approx(quantum, rel=2e-7)
    np.testing.assert_allclose(result["cpge"] + result["cpge"].swapaxes(2, 3), 0, atol=1e-4)
    np.testing.assert_allclose(result["steady_cpge"], result["cpge"] * 1e-6, rtol=1e-12)
    assert result.units["cpge"] == "A/(V^2 s)"


def test_invalid_optical_domains_and_static_metal_are_rejected():
    model = DipoleModel()
    for kwargs in ({"eta": 0}, {"temperature_kelvin": -1}, {"scissors": -1}, {"chemical_potential": np.nan}):
        with pytest.raises(ValueError):
            optical_conductivity(model, [1.0], grid=(1, 1, 1), **kwargs)
    with pytest.raises(ValueError, match="omega"):
        shg(model, [-1], grid=(1, 1, 1))
    with pytest.raises(ValueError, match="insulating"):
        second_order_static(model, chemical_potential=-1, grid=(1, 1, 1))
    with pytest.raises(ValueError, match="insulating occupations"):
        second_order_static(model, temperature_kelvin=10000, grid=(1, 1, 1))
    with pytest.raises(ValueError, match="smearing"):
        shift_current(model, [1], smearing="unknown", grid=(1, 1, 1))
    with pytest.raises(ValueError, match="relaxation_time"):
        cpge(model, [1], relaxation_time_seconds=0, grid=(1, 1, 1))


def test_optical_and_nonlinear_covariance_under_nonorthogonal_basis_change():
    original = DipoleModel()
    transformed = DipoleModel()
    change = np.array([[1.2, 0.2j], [0.3, 0.8]])
    transformed.h = change.conj().T @ original.h @ change
    transformed.r = np.einsum("in,aij,jm->anm", change.conj(), original.r, change)
    metric = change.conj().T @ change
    transformed.overlap = lambda k: metric
    kwargs = {"eta": 0.03, "grid": (1, 1, 1)}
    for function, key in [
        (optical_conductivity, "conductivity"),
        (shg, "shg"),
        (shift_current, "shift_current"),
    ]:
        reference = function(original, [0.4, 1.8, 2.3], **kwargs)
        actual = function(transformed, [0.4, 1.8, 2.3], **kwargs)
        np.testing.assert_allclose(actual[key], reference[key], rtol=1e-11, atol=1e-10)


def test_cartesian_rotation_origin_and_permutation_contracts():
    original = DipoleModel()
    rotated = DipoleModel()
    direction = np.array([0.6, 0.8, 0.0])
    rotated.r = direction[:, None, None] * original.r[0]
    origin = np.array([2.0, -0.7, 0.2])
    rotated.r += origin[:, None, None] * np.eye(2)
    omega = [0.3, 0.8]
    expected = shg(original, omega, grid=(1, 1, 1))["shg"][:, 0, 0, 0]
    actual = shg(rotated, omega, grid=(1, 1, 1))
    np.testing.assert_allclose(
        actual["shg"], np.einsum("w,a,b,c->wabc", expected, direction, direction, direction), atol=1e-11
    )
    np.testing.assert_allclose(actual["shg"], actual["shg"].swapaxes(2, 3), atol=1e-12)
    assert actual.dims["shg"] == ("omega", "cartesian_a", "cartesian_b", "cartesian_c")


def test_response_memory_batches_and_process_workers_agree():
    points = np.column_stack((np.arange(12) / 12, np.zeros((12, 2))))
    model = RiceMeleModel()
    omega = np.linspace(0.2, 3, 17)
    serial = shg(model, omega, kpoints=points, execution=ExecutionOptions(batch_size=12))["shg"]
    for execution in (
        ExecutionOptions(batch_size=2, memory_mb=0.05),
        ExecutionOptions(workers=2, batch_size=4),
    ):
        actual = shg(model, omega, kpoints=points, execution=execution)["shg"]
        np.testing.assert_allclose(actual, serial, rtol=1e-10, atol=1e-9)
    with pytest.raises(ValueError, match="too small"):
        shg(model, omega, kpoints=points, execution=ExecutionOptions(memory_mb=1e-6))


def test_all_optical_responses_are_covariant_at_degenerate_manifold_crossings():
    generator = np.random.default_rng(60)
    original = DipoleModel((-1.0, -1.0, 2.0), np.eye(3))
    random = generator.normal(size=(3, 3, 3)) + 1j * generator.normal(size=(3, 3, 3))
    original.r = (random + random.conj().transpose(0, 2, 1)) / 2
    random = generator.normal(size=(3, 3, 3)) + 1j * generator.normal(size=(3, 3, 3))
    slope = (random + random.conj().transpose(0, 2, 1)) / 2
    original.hamiltonian = lambda k: (
        original.h + np.einsum("a,aij->ij", np.asarray(k) @ original.g_vec, slope)
    )
    original.derivatives = lambda k, order=1: {
        "H": slope if order == 1 else np.zeros((3, 3, 3, 3)),
        "S": np.zeros((3,) * order + (3, 3)),
        "r": np.zeros((3,) * (order + 1) + (3, 3)),
    }
    rotated = DipoleModel((-1.0, -1.0, 2.0), np.eye(3))
    angle = 0.43
    change = np.array(
        [[np.cos(angle), np.sin(angle), 0], [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]], complex
    )
    rotated.r = np.einsum("in,aij,jm->anm", change.conj(), original.r, change)
    rotated.hamiltonian = lambda k: change.conj().T @ original.hamiltonian(k) @ change
    rotated.derivatives = lambda k, order=1: {
        key: np.einsum("in,...ij,jm->...nm", change.conj(), value, change)
        for key, value in original.derivatives(k, order).items()
    }
    for function, key in [
        (optical_conductivity, "conductivity"),
        (shg, "shg"),
        (pockels, "pockels"),
        (shift_current, "shift_current"),
        (cpge, "cpge"),
    ]:
        for temperature in (0, 10000):
            kwargs = {"kpoints": [[0, 0, 0]], "temperature_kelvin": temperature}
            expected = function(original, [0.3, 1.4, 3.0], **kwargs)[key]
            actual = function(rotated, [0.3, 1.4, 3.0], **kwargs)[key]
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-9)


def test_optical_and_nonlinear_are_covariant_under_moving_nonorthogonal_basis():
    original = DipoleModel()
    original.r[1] = [[0.1, 0.4j], [-0.4j, -0.2]]
    tx = np.array([[0.13 + 0.1j, 0.05], [0.02j, -0.06]])
    ty = np.array([[0.08j, 0.07], [-0.04, 0.05j]])

    def transformation(k):
        q = np.asarray(k) @ original.g_vec
        t = np.eye(2) + np.sin(q[0]) * tx + np.cos(q[1]) * ty
        first = np.array([np.cos(q[0]) * tx, -np.sin(q[1]) * ty, np.zeros((2, 2))])
        second = np.zeros((3, 3, 2, 2), complex)
        second[0, 0], second[1, 1] = -np.sin(q[0]) * tx, -np.cos(q[1]) * ty
        return t, first, second

    changed = DipoleModel()
    changed.hamiltonian = lambda k: transformation(k)[0].conj().T @ original.h @ transformation(k)[0]
    changed.overlap = lambda k: transformation(k)[0].conj().T @ transformation(k)[0]

    def position(k):
        t, first, _ = transformation(k)
        return np.array([t.conj().T @ original.r[a] @ t - 1j * first[a].conj().T @ t for a in range(3)])

    def derivative(k, order=1):
        t, first, second = transformation(k)
        if order == 1:
            h = np.array([a.conj().T @ original.h @ t + t.conj().T @ original.h @ a for a in first])
            s = np.array([a.conj().T @ t + t.conj().T @ a for a in first])
            r = np.array(
                [
                    [
                        first[a].conj().T @ original.r[b] @ t
                        + t.conj().T @ original.r[b] @ first[a]
                        - 1j * second[a, b].conj().T @ t
                        - 1j * first[b].conj().T @ first[a]
                        for b in range(3)
                    ]
                    for a in range(3)
                ]
            )
            return {"H": h, "S": s, "r": r}
        h, s = np.empty_like(second), np.empty_like(second)
        for a in range(3):
            for b in range(3):
                h[a, b] = (
                    second[a, b].conj().T @ original.h @ t
                    + first[a].conj().T @ original.h @ first[b]
                    + first[b].conj().T @ original.h @ first[a]
                    + t.conj().T @ original.h @ second[a, b]
                )
                s[a, b] = (
                    second[a, b].conj().T @ t
                    + first[a].conj().T @ first[b]
                    + first[b].conj().T @ first[a]
                    + t.conj().T @ second[a, b]
                )
        return {"H": h, "S": s, "r": np.zeros((3, 3, 3, 2, 2))}

    changed.position, changed.derivatives = position, derivative
    for function, key in [
        (optical_conductivity, "conductivity"),
        (shg, "shg"),
        (pockels, "pockels"),
        (shift_current, "shift_current"),
    ]:
        kwargs = {"kpoints": [[0.13, 0.22, 0], [0.37, 0.41, 0]], "eta": 0.03}
        expected = function(original, [0.3, 1.4, 2.3], **kwargs)[key]
        actual = function(changed, [0.3, 1.4, 2.3], **kwargs)[key]
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-9)


@pytest.mark.parametrize(
    "name", ["optical_conductivity", "shift_current", "cpge", "shg", "pockels", "second_order_static"]
)
def test_optical_registered_tiles_equal_direct_weighted_result(tmp_path, name):
    from types import SimpleNamespace

    from queer.io.artifacts import RunStore
    from queer.workflows.registry import REGISTRY, calculate_registered

    points = np.column_stack((np.arange(147) / 147, np.full((147, 2), 0.137)))
    model = WeylModel() if name == "cpge" else RiceMeleModel()
    params = {"kpoints": points.tolist(), "weights": np.linspace(0.5, 1.5, len(points)).tolist()}
    if name != "second_order_static":
        params.update(omega=[0.3, 1.4, 3.0], eta=0.1)
    if name == "cpge":
        params["relaxation_time_seconds"] = 1e-12
    execution = ExecutionOptions()
    config = SimpleNamespace(
        workflow=name, parameters=params, execution=execution, inputs={}, material_dir=tmp_path
    )
    serialized = {"workflow": name, "parameters": params, "inputs": {}, "execution": execution.as_dict()}
    store = RunStore(tmp_path / name, serialized, [])
    try:
        expected = REGISTRY[name].calculator()(model, **params).arrays
        calculate_registered(config, model, store)
        assert len([case for case in store.manifest["cases"] if case != "result"]) == 2
        with np.load(store.root / store.manifest["cases"]["result"]["path"], allow_pickle=False) as actual:
            assert set(actual.files) == set(expected)
            for key, value in expected.items():
                roundoff = 64 * np.finfo(float).eps * max(1, np.max(np.abs(value), initial=0))
                np.testing.assert_allclose(actual[key], value, rtol=3e-12, atol=roundoff)
        records = dict(store.manifest["cases"])
        calculate_registered(config, model, store)
        assert store.manifest["cases"] == records
    finally:
        store.release()


def test_static_workflow_rejects_rank_changes_across_tile_boundaries(tmp_path):
    from types import SimpleNamespace

    from queer.io.artifacts import RunStore
    from queer.workflows.registry import calculate_registered

    model = DipoleModel()
    model.hamiltonian = lambda k: np.diag([-1.0, 1.0] if k[0] < 0.5 else [-2.0, -1.0])
    points = np.zeros((147, 3))
    points[128:, 0] = 0.75
    params = {"kpoints": points.tolist()}
    execution = ExecutionOptions()
    config = SimpleNamespace(
        workflow="second_order_static",
        parameters=params,
        execution=execution,
        inputs={},
        material_dir=tmp_path,
    )
    serialized = {
        "workflow": config.workflow,
        "parameters": params,
        "inputs": {},
        "execution": execution.as_dict(),
    }
    store = RunStore(tmp_path / "rank_change", serialized, [])
    try:
        with pytest.raises(ValueError, match="occupied rank"):
            calculate_registered(config, model, store)
        assert "result" not in store.manifest["cases"]
    finally:
        store.release()
