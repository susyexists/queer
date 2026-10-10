"""Native PAOs and plane-wave projections checked against compact sphere orbitals."""

import numpy as np
import pytest
from scipy.integrate import simpson

from queer.io.pao import BOHR_ANGSTROM, PAORadialData, read_pao, real_harmonics
from queer.models.openmx import OpenMXModel
from queer.observables.unfolding import bandunfolding, bandunfolding_spin_texture


def radial_sphere(radius=1.0):
    radii = np.linspace(0, radius, 1001)
    return PAORadialData(radii, {0: np.full((len(radii), 1), np.sqrt(3 / radius**3))})


def orbitals(centers):
    return [
        {
            "atom_index": i,
            "species": "X",
            "l": 0,
            "zeta": 0,
            "m": 0,
            "harmonic": "s",
            "pao_key": "X",
            "center": list(center),
        }
        for i, center in enumerate(centers)
    ]


def test_native_pao_parser_validates_table_counts_and_grids(tmp_path):
    source = tmp_path / "X.pao"
    source.write_text(
        "PAO.Lmax 1\nPAO.Mul 1\ngrid.num.output 3\n"
        "<pseudo.atomic.orbitals.L=0\n-2 0.1 1\n-1 0.2 2\n0 0.3 3\n"
        "pseudo.atomic.orbitals.L=0>\n"
        "<pseudo.atomic.orbitals.L=1\n-2 0.1 4\n-1 0.2 5\n0 0.3 6\n"
        "pseudo.atomic.orbitals.L=1>\n"
    )
    result = read_pao(source, required_counts={0: 1, 1: 1})
    np.testing.assert_allclose(result.radii, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(result.radial[1][:, 0], [4, 5, 6])
    with pytest.raises(ValueError, match="insufficient"):
        read_pao(source, required_counts={1: 2})
    source.write_text(source.read_text().replace("-1 0.2 5", "-1 0.25 5"))
    with pytest.raises(ValueError, match="different radial grids"):
        read_pao(source)


def test_openmx_real_harmonics_order_and_orthonormality():
    np.testing.assert_allclose(real_harmonics(1, np.eye(3)), np.eye(3) * np.sqrt(3 / (4 * np.pi)))
    dz = real_harmonics(2, [[0, 0, 1]])[0]
    np.testing.assert_allclose(dz, [np.sqrt(5 / (4 * np.pi)), 0, 0, 0, 0], atol=1e-14)
    z, wz = np.polynomial.legendre.leggauss(24)
    phi = np.arange(64) * 2 * np.pi / 64
    zz, pp = np.meshgrid(z, phi, indexing="ij")
    directions = np.column_stack(
        (
            np.sqrt(1 - zz.ravel() ** 2) * np.cos(pp.ravel()),
            np.sqrt(1 - zz.ravel() ** 2) * np.sin(pp.ravel()),
            zz.ravel(),
        )
    )
    integration_weights = np.repeat(wz, 64) * 2 * np.pi / 64
    for ell in range(7):
        harmonics = real_harmonics(ell, directions)
        np.testing.assert_allclose(
            harmonics.T @ (integration_weights[:, None] * harmonics), np.eye(2 * ell + 1), atol=2e-13
        )


def test_sphere_radial_fourier_transform_in_physical_units():
    data = radial_sphere(2)
    radius = 2 * BOHR_ANGSTROM
    radial_amplitude = np.sqrt(3 / 2**3) / BOHR_ANGSTROM**1.5
    q = np.array([0, 0.2, 1.5])
    expected = np.array(
        [
            radial_amplitude * radius**3 / 3,
            radial_amplitude * (np.sin(q[1] * radius) - q[1] * radius * np.cos(q[1] * radius)) / q[1] ** 3,
            radial_amplitude * (np.sin(q[2] * radius) - q[2] * radius * np.cos(q[2] * radius)) / q[2] ** 3,
        ]
    )
    np.testing.assert_allclose(data.fourier(q, 0, 0), expected, atol=1e-12)
    assert simpson(data.radii**2 * data.radial[0][:, 0] ** 2, x=data.radii) == pytest.approx(1)


def test_plane_wave_normalization_and_pristine_double_cell_selection():
    data = radial_sphere()
    centers = [[0, 0, 0], [5, 0, 0]]
    model = OpenMXModel(
        [[0, 0, 0]], [[[0, 0.2], [0.2, 0]]], lattice=np.diag([10, 10, 10]), orbital_metadata=orbitals(centers)
    )
    matrix = np.diag([2, 1, 1])
    result = bandunfolding(model, [[0, 0, 0], [0.5, 0, 0]], matrix, ecut=2, radial_data={"X": data})
    # The bonding state (band1) is primitive Gamma; the antibonding state is the folded edge.
    assert result["weights"][0, 0] < 1e-25
    assert result["weights"][1, 1] < 1e-25
    sphere_volume = 4 * np.pi * BOHR_ANGSTROM**3 / 3
    # Only primitive G=0 enters at Gamma for this cutoff along x; y/z satellites contribute.
    momenta = []
    for gx in range(-2, 3):
        for gy in range(-3, 4):
            for gz in range(-3, 4):
                q = np.array([gx * 2 * np.pi / 5, gy * 2 * np.pi / 10, gz * 2 * np.pi / 10])
                if 3.80998211615486 * (q @ q) <= 2:
                    momenta.append(np.linalg.norm(q))
    expected = 2 * 4 * np.pi * np.sum(data.fourier(momenta, 0, 0) ** 2) / 1000
    assert result["weights"][1, 0] == pytest.approx(expected, rel=1e-12)
    assert result["weights"][1, 0] >= 2 * sphere_volume / 1000


def test_actual_radial_unfolded_spin_uses_all_up_all_down_order():
    data = radial_sphere()
    model = OpenMXModel(
        [[0, 0, 0]],
        [[[0, 1], [1, 0]]],
        spin_mode="soc",
        lattice=np.eye(3) * 10,
        orbital_metadata=orbitals([[0, 0, 0]]) * 2,
    )
    result = bandunfolding_spin_texture(
        model, [[0, 0, 0]], np.eye(3, dtype=int), ecut=0.1, radial_data={"X": data}
    )
    sphere_fraction = 4 * np.pi * BOHR_ANGSTROM**3 / 3 / 1000
    np.testing.assert_allclose(result["weights"][:, 0], sphere_fraction, rtol=1e-12)
    np.testing.assert_allclose(
        result["spin"][:, 0, 0], [-sphere_fraction / 2, sphere_fraction / 2], rtol=1e-12
    )
    np.testing.assert_allclose(result["spin"][:, 0, 1:], 0, atol=1e-16)
    with pytest.raises(ValueError, match="missing actual"):
        bandunfolding(model, [[0, 0, 0]], np.eye(3, dtype=int))


def test_radial_cutoff_convergence_to_normalized_gaussian_plane_wave_sum():
    radii_angstrom = np.linspace(0, 8, 1001)
    amplitude = 2 / np.pi**0.25
    radial = amplitude * np.exp(-(radii_angstrom**2) / 2) * BOHR_ANGSTROM**1.5
    data = PAORadialData(radii_angstrom / BOHR_ANGSTROM, {0: radial[:, None]})
    model = OpenMXModel([[0, 0, 0]], [[[0]]], lattice=np.eye(3) * 10, orbital_metadata=orbitals([[0, 0, 0]]))
    low = bandunfolding(model, [[0, 0, 0]], np.eye(3, dtype=int), ecut=10, radial_data={"X": data})
    high = bandunfolding(model, [[0, 0, 0]], np.eye(3, dtype=int), ecut=100, radial_data={"X": data})
    expected_low = 0.0
    for gx in range(-3, 4):
        for gy in range(-3, 4):
            for gz in range(-3, 4):
                square = (2 * np.pi / 10) ** 2 * (gx * gx + gy * gy + gz * gz)
                if 3.80998211615486 * square <= 10:
                    expected_low += 8 * np.pi**1.5 / 1000 * np.exp(-square)
    assert low["weights"][0, 0] == pytest.approx(expected_low, abs=1e-10)
    assert low["weights"][0, 0] < high["weights"][0, 0]
    assert high["weights"][0, 0] == pytest.approx(1, abs=2e-9)


def test_later_contracted_angular_channel_requires_explicit_radial_functions(tmp_path):
    metadata = orbitals([[0, 0, 0]])
    metadata.extend([{**metadata[0], "l": 1, "m": m, "contracted": True} for m in range(3)])
    model = OpenMXModel([[0, 0, 0]], [np.eye(4)], orbital_metadata=metadata)
    source = tmp_path / "X.pao"
    source.write_text(
        "PAO.Mul 1\nPAO.Lmax 1\ngrid.num.output 3\n"
        "<pseudo.atomic.orbitals.L=0\n-2 0.1 1\n-1 0.2 1\n0 0.3 1\n"
        "pseudo.atomic.orbitals.L=0>\n"
        "<pseudo.atomic.orbitals.L=1\n-2 0.1 1\n-1 0.2 1\n0 0.3 1\n"
        "pseudo.atomic.orbitals.L=1>\n"
    )
    with pytest.raises(ValueError, match="explicit matching"):
        bandunfolding(model, [[0, 0, 0]], np.eye(3, dtype=int), pao_dir=tmp_path, ecut=0.1)
    result = bandunfolding(model, [[0, 0, 0]], np.eye(3, dtype=int), radial_data={"X": source}, ecut=0.1)
    assert result["weights"].shape == (4, 1)


@pytest.mark.parametrize("mode", ["nonmagnetic", "soc"])
def test_degenerate_radial_projection_and_spin_means_ignore_eigenvector_gauge(mode):
    angle, phase = 0.47, 0.68
    rotation = np.array(
        [
            [np.cos(angle), -np.exp(-1j * phase) * np.sin(angle)],
            [np.exp(1j * phase) * np.sin(angle), np.cos(angle)],
        ]
    )
    metadata = orbitals([[0, 0, 0], [5, 0, 0]]) if mode == "nonmagnetic" else orbitals([[0, 0, 0]]) * 2

    class RotatingModel(OpenMXModel):
        def __init__(self, gauge):
            super().__init__(
                [[0, 0, 0]],
                [np.zeros((2, 2))],
                spin_mode=mode,
                lattice=np.eye(3) * 10,
                orbital_metadata=metadata,
            )
            self.gauge = gauge

        def eigenstates(self, points, **kwargs):
            result = super().eigenstates(points, **kwargs)
            return type(result)(
                result.energies, result.coefficients @ self.gauge, result.overlaps, result.band_channels
            )

    arguments = {
        "kpoints": [[0, 0, 0]],
        "supercell_matrix": np.eye(3, dtype=int),
        "ecut": 0.1,
        "radial_data": {"X": radial_sphere()},
    }
    ordinary = bandunfolding_spin_texture(RotatingModel(np.eye(2)), **arguments)
    changed = bandunfolding_spin_texture(RotatingModel(rotation), **arguments)
    sphere_fraction = 4 * np.pi * BOHR_ANGSTROM**3 / 3 / 1000
    np.testing.assert_allclose(ordinary["weights"], sphere_fraction, rtol=1e-12)
    np.testing.assert_allclose(changed["weights"], ordinary["weights"], rtol=1e-12)
    np.testing.assert_allclose(ordinary["spin"], 0, atol=1e-16)
    np.testing.assert_allclose(changed["spin"], ordinary["spin"], atol=1e-16)
    assert ordinary["manifold_sizes"].tolist() == [[2], [2]]
    with pytest.raises(ValueError, match="cuts a degenerate subspace"):
        bandunfolding(RotatingModel(np.eye(2)), **arguments, bands=[0])
