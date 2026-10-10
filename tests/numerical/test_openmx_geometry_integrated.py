"""Independent analytic references for localized-orbital geometry/responses."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import block_diag

from queer.execution import ExecutionOptions
from queer.models.openmx import OpenMXModel
from queer.models.wannier import WannierModel
from queer.observables._tb import E2_H, E_CHARGE, HBAR, KB_EV, band_geometry, basis_curvature, velocities
from queer.observables.geometry import (
    berry_curvature,
    chern_number,
    chirality,
    polarization,
    quantum_metric,
    wilson_loop,
)
from queer.observables.integrated import (
    ahc,
    anc,
    berry_curvature_dipole,
    drude_weight,
    orbital_magnetization,
    shc,
)

PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
TRANSLATIONS = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])


def qwz(mass=-1, lattice=None):
    x, y, z = PAULI
    hopping = np.array(
        [mass * z, 0.5 * z - 0.5j * x, 0.5 * z + 0.5j * x, 0.5 * z - 0.5j * y, 0.5 * z + 0.5j * y]
    )
    lattice = np.eye(3) if lattice is None else np.asarray(lattice)
    return WannierModel(TRANSLATIONS, hopping, reciprocal=2 * np.pi * np.linalg.inv(lattice).T)


@pytest.mark.parametrize("mass,expected", [(-3, 0), (-1, 1), (1, -1), (3, 0)])
def test_chern_analytic_phase_diagram(mass, expected):
    result = chern_number(qwz(mass), grid=(10, 10, 1))
    assert_allclose(result["chern_number"], expected, atol=2e-13)


def test_qwz_cartesian_curvature_metric_and_integration():
    model = qwz()
    curvature = berry_curvature(model, [[0, 0, 0]])["curvature"]
    metric = quantum_metric(model, [[0, 0, 0]], occupied_bands=1)["metric"][0]
    assert curvature.shape == (2, 1, 3, 3)
    assert_allclose(curvature[:, 0, 0, 1], [0.5, -0.5], atol=1e-14)
    assert_allclose(metric, np.diag([0.25, 0.25, 0]), atol=1e-14)
    assert_allclose(np.linalg.det(metric[:2, :2]), (curvature[0, 0, 0, 1] / 2) ** 2)
    assert_allclose(chern_number(model, grid=(32, 32, 1), engine="direct")["chern_number"], 1, atol=2e-8)
    assert_allclose(
        berry_curvature(model, [[0.1, 0.2, 0]], engine="direct")["curvature"],
        berry_curvature(model, [[0.1, 0.2, 0]], engine="kubo")["curvature"],
        atol=1e-13,
    )
    assert_allclose(quantum_metric(model, [[0, 0, 0]], occupied_bands=2)["metric"], 0, atol=1e-14)
    with pytest.raises(ValueError, match="second moments"):
        quantum_metric(model, [[0, 0, 0]], definition="physical")


def test_oblique_metric_cartesian_transformation():
    lattice = np.array([[2, 0.4, 0.2], [0.1, 1.5, 0.3], [0, 0, 8]])
    result = quantum_metric(qwz(lattice=lattice), [[0, 0, 0]], occupied_bands=1)["metric"][0]
    # dH/dk_cart_a = sum_l R_cart_l,a dH/dq_l; metric transforms with real lattice rows.
    expected = 0.25 * (np.outer(lattice[0], lattice[0]) + np.outer(lattice[1], lattice[1]))
    assert_allclose(result, expected, atol=2e-13)
    fhs = chern_number(qwz(lattice=lattice), grid=(10, 10, 1))["chern_number"]
    direct = chern_number(qwz(lattice=lattice), grid=(32, 32, 1), engine="direct")["chern_number"]
    assert_allclose([fhs, direct], [1, 1], atol=2e-8)


def test_nonorthogonal_basis_covariance():
    model = qwz()
    transform = np.array([[1.2 + 0.1j, 0.2 - 0.3j], [-0.1 + 0.2j, 0.8]])
    transformed_h = np.einsum("in,rij,jm->rnm", transform.conj(), model.hopping_matrices, transform)
    overlap = np.zeros_like(transformed_h)
    overlap[0] = transform.conj().T @ transform
    changed = OpenMXModel(
        TRANSLATIONS,
        transformed_h,
        overlap,
        position=np.zeros((3, 5, 2, 2)),
        lattice=np.eye(3),
        spin_degeneracy=1,
    )
    points = [[0.1, 0.2, 0], [0.27, 0.36, 0]]
    assert_allclose(
        berry_curvature(changed, points)["curvature"], berry_curvature(model, points)["curvature"], atol=2e-12
    )
    assert_allclose(
        quantum_metric(changed, points)["metric"], quantum_metric(model, points)["metric"], atol=2e-12
    )
    assert_allclose(velocities(changed, points[0])[0], velocities(model, points[0])[0], atol=1e-13)
    # Physical band matrix magnitudes survive arbitrary eigenvector phases.
    assert_allclose(abs(velocities(changed, points[0])[1]), abs(velocities(model, points[0])[1]), atol=2e-12)


class MovingBasis:
    """Analytic change of a complete orthogonal basis, including its dipoles.

    chi(k)=orthogonal_basis*T(k), so S=T†T and r_a=-i T_a†T.
    The physical Hilbert space stays fixed while the generalized eigenproblem,
    coefficient derivative, and orbital connection each acquire k dependence.
    """

    nbnd = 2
    lattice = np.eye(3)
    fermi_energy = 0.0
    spin_degeneracy = 1
    base = qwz()
    tx = np.array([[0.13 + 0.1j, 0.05], [0.02j, -0.06]])
    ty = np.array([[0.08j, 0.07], [-0.04, 0.05j]])

    def transformation(self, k):
        q = 2 * np.pi * np.asarray(k)
        return np.eye(2) + np.sin(q[0]) * self.tx + np.cos(q[1]) * self.ty

    def transformation_derivatives(self, k):
        q = 2 * np.pi * np.asarray(k)
        first = np.array([np.cos(q[0]) * self.tx, -np.sin(q[1]) * self.ty, np.zeros((2, 2))])
        second = np.zeros((3, 3, 2, 2), complex)
        second[0, 0] = -np.sin(q[0]) * self.tx
        second[1, 1] = -np.cos(q[1]) * self.ty
        return first, second

    def hamiltonian(self, k):
        t = self.transformation(k)
        return t.conj().T @ self.base.hamiltonian(k) @ t

    def overlap(self, k):
        t = self.transformation(k)
        return t.conj().T @ t

    def position(self, k):
        t = self.transformation(k)
        first, _ = self.transformation_derivatives(k)
        return np.array([-1j * a.conj().T @ t for a in first])

    def basis_overlap(self, k, kp):
        return self.transformation(k).conj().T @ self.transformation(kp)

    def derivatives(self, k, order=1):
        if order != 1:
            raise ValueError("this independent analytic fixture needs first derivatives only")
        from queer.observables._tb import derivatives

        t = self.transformation(k)
        first, second = self.transformation_derivatives(k)
        h0 = self.base.hamiltonian(k)
        dh0 = derivatives(self.base, k)["H"]
        h = np.array(
            [a.conj().T @ h0 @ t + t.conj().T @ dh0[i] @ t + t.conj().T @ h0 @ a for i, a in enumerate(first)]
        )
        s = np.array([a.conj().T @ t + t.conj().T @ a for a in first])
        r = np.array(
            [
                [-1j * (second[b, a].conj().T @ t + first[a].conj().T @ first[b]) for a in range(3)]
                for b in range(3)
            ]
        )
        return {"H": h, "S": s, "r": r}


def test_k_dependent_nonorthogonal_basis_covariance_with_dipoles():
    original, changed = qwz(), MovingBasis()
    points = [[0.13, 0.22, 0], [0.37, 0.41, 0]]
    for engine in ("direct", "kubo"):
        assert_allclose(
            berry_curvature(changed, points, engine=engine)["curvature"],
            berry_curvature(original, points, engine=engine)["curvature"],
            atol=3e-12,
        )
    assert_allclose(
        quantum_metric(changed, points)["metric"], quantum_metric(original, points)["metric"], atol=3e-12
    )
    assert_allclose(
        abs(velocities(changed, points[0])[1]), abs(velocities(original, points[0])[1]), atol=3e-12
    )
    path = np.zeros((33, 3))
    path[:, 0] = np.linspace(0, 1, len(path))
    path[:, 1] = 0.2
    assert_allclose(
        wilson_loop(changed, path, occupied_bands=1)["phases"],
        wilson_loop(original, path, occupied_bands=1)["phases"],
        atol=3e-12,
    )
    assert_allclose(basis_curvature(changed, points[0]), 0, atol=3e-12)


def test_occupied_degenerate_subspaces_are_gated():
    original = qwz()
    hopping = np.array([block_diag(h, h) for h in original.hopping_matrices])
    model = OpenMXModel(TRANSLATIONS, hopping, lattice=np.eye(3), spin_mode="soc")
    with pytest.raises(ValueError, match="individual-band"):
        berry_curvature(model, [[0, 0, 0]])
    with pytest.raises(ValueError, match="degenerate subspace"):
        quantum_metric(model, [[0, 0, 0]], occupied_bands=1)
    assert_allclose(berry_curvature(model, [[0, 0, 0]], occupied_bands=2)["curvature"][0, 0, 1], 1)
    assert_allclose(quantum_metric(model, [[0, 0, 0]], occupied_bands=2)["metric"][0], np.diag([0.5, 0.5, 0]))


def test_hall_sign_sheet_bulk_units_and_empty_full_manifolds():
    model = qwz(lattice=np.diag([1.0, 1.0, 10.0]))
    fhs = ahc(model, grid=(10, 10, 1), engine="fhs", chemical_potentials=[-5, 0, 5])
    assert_allclose(fhs["ahc"][:, 0, 1], [0, -E2_H * 1e7, 0], atol=2e-10)
    assert_allclose(fhs["ahc"], -fhs["ahc"].swapaxes(-1, -2), atol=1e-13)
    sheet = ahc(model, grid=(10, 10, 1), engine="fhs", dimension=2)
    assert_allclose(sheet["ahc"][0, 0, 1], -E2_H, atol=1e-16)
    direct = ahc(model, grid=(32, 32, 1))
    kubo = ahc(model, grid=(32, 32, 1), engine="kubo")
    assert_allclose(direct["ahc"][0, 0, 1], -E2_H * 1e7, rtol=2e-8)
    assert_allclose(kubo["ahc"], direct["ahc"], rtol=2e-13, atol=1e-12)
    with pytest.raises(ValueError, match="zero temperature"):
        ahc(model, grid=(4, 4, 1), engine="fhs", temperature_kelvin=1)


def test_metallic_fhs_keeps_collinear_energy_manifolds_independent():
    # E_up=sin(qx), E_down=-sin(qx) cross at EF and switch global ordering.
    # Each one-dimensional spin basis has no Berry curvature, so Hall=0.
    translations = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    hopping = np.array([np.zeros((2, 2)), -0.5j * PAULI[2], 0.5j * PAULI[2]])
    model = OpenMXModel(translations, hopping, lattice=np.eye(3), spin_mode="collinear")
    result = ahc(model, grid=(5, 5, 1), engine="fhs")
    assert_allclose(result["ahc"], 0, atol=1e-14)
    diagnostics = result.metadata["fhs_diagnostics"]
    assert diagnostics["metallic_vertex_rank_variation_detected"] == [True]
    assert diagnostics["maximum_channel_occupation_rank_range_by_plane_and_mu"][2] == [1]
    assert diagnostics["continuum_convergence_established"] is False


def test_fhs_flux_diagnostics_and_refinement_detect_wrapped_coarse_topology():
    # Near a phase boundary a3x3 mesh wraps the concentrated true curvature.
    # Every principal flux lies inside(-pi,pi), yet the coarse Chern is wrong.
    # Refinement recovers the known QWZ C=+1 and exposes the difference of2.
    result = chern_number(qwz(-0.1), grid=(3, 3, 1), refinement=True)
    flux = result["plaquette_flux"]
    assert flux.shape == (9,)
    assert result["kpoints"].shape == (9, 3)
    assert result.units["plaquette_flux"] == "1"
    assert_allclose(flux.sum() / (2 * np.pi), result["chern_number"], atol=2e-14)
    assert_allclose(result["chern_number"], -1, atol=2e-13)
    diagnostics = result.metadata
    assert_allclose(diagnostics["max_abs_plaquette_flux"], np.max(abs(flux)), atol=2e-14)
    assert_allclose(diagnostics["minimum_branch_margin"], np.pi - np.max(abs(flux)), atol=2e-14)
    assert diagnostics["principal_branch_admissible"] is True
    assert diagnostics["continuum_convergence_established"] is False
    assert diagnostics["refinement"]["grid"] == [6, 6, 1]
    assert_allclose(diagnostics["refinement"]["chern_number"], 1, atol=2e-13)
    assert_allclose(diagnostics["refinement"]["absolute_difference"], 2, atol=2e-13)


def test_chern_paired_geometry_engines_and_hall_insulating_diagnostics():
    model = qwz()
    result = chern_number(model, grid=(10, 10, 1), engine="direct", refinement=True)
    comparison = result.metadata["engine_comparison"]
    assert_allclose(comparison["direct_chern_number"], result["chern_number"], atol=2e-13)
    assert_allclose(comparison["direct_minus_kubo"], 0, atol=2e-13)
    assert_allclose(result["quadrature_flux"].sum() / (2 * np.pi), result["chern_number"], atol=2e-13)
    assert result.metadata["refinement"]["absolute_difference"] > 1e-4
    hall = ahc(model, grid=(10, 10, 1), engine="fhs", dimension=2)
    diagnostics = hall.metadata["fhs_diagnostics"]
    assert_allclose(diagnostics["chern_vector"][0], [0, 0, 1], atol=2e-13)
    assert diagnostics["metallic_vertex_rank_variation_detected"] == [False]
    assert diagnostics["grid"] == [10, 10, 1]
    assert diagnostics["evaluated_reciprocal_planes"] == [[0, 1]]
    assert diagnostics["principal_branch_admissible"] is True
    assert diagnostics["continuum_convergence_established"] is False


def test_chern_engine_difference_matches_independent_basis_curvature_integral():
    original = qwz()
    position = np.zeros((3, len(TRANSLATIONS), 2, 2), complex)
    position[0, 0], position[1, 0] = 0.5 * PAULI[0], -0.5 * PAULI[1]
    model = OpenMXModel(
        TRANSLATIONS, original.hopping_matrices, position=position, lattice=np.eye(3), spin_degeneracy=1
    )
    result = chern_number(model, grid=(10, 10, 1), engine="direct")
    qx, qy = np.meshgrid(2 * np.pi * np.arange(10) / 10, 2 * np.pi * np.arange(10) / 10, indexing="ij")
    dz = -1 + np.cos(qx) + np.cos(qy)
    norm = np.sqrt(np.sin(qx) ** 2 + np.sin(qy) ** 2 + dz**2)
    # F_xy=-i[r_x,r_y]=-.5 sigma_z and <sigma_z>_lower=-dz/|d|.
    # Its Chern contribution is(2pi)*mean(.5 dz/|d|).
    expected = np.pi * np.mean(dz / norm)
    assert abs(expected) > 1
    assert_allclose(result.metadata["engine_comparison"]["direct_minus_kubo"], expected, atol=2e-12)


def test_oblique_bulk_and_projected_sheet_hall_cartesian_tensors():
    lattice = np.array([[2.0, 0.4, 0.2], [0.1, 1.5, 0.3], [0.2, -0.1, 8.0]])
    model = qwz(lattice=lattice)
    g = 2 * np.pi * np.linalg.inv(lattice).T
    normal = np.cross(g[0], g[1])
    normal /= np.linalg.norm(normal)
    for dimension in (2, 3):
        vector = -E2_H * normal if dimension == 2 else -E2_H * g[2] / (2 * np.pi) * 1e8
        expected = np.array(
            [[0, vector[2], -vector[1]], [-vector[2], 0, vector[0]], [vector[1], -vector[0], 0]]
        )
        direct = ahc(model, grid=(32, 32, 1), dimension=dimension)["ahc"][0]
        fhs = ahc(model, grid=(10, 10, 1), engine="fhs", dimension=dimension)["ahc"][0]
        assert_allclose(direct, expected, rtol=2e-8, atol=1e-13)
        assert_allclose(fhs, expected, rtol=2e-13, atol=1e-13)
    drude = drude_weight(model, grid=(8, 8, 1), dimension=2, temperature_kelvin=4000.0)["drude_weight"][0]
    assert np.trace(drude) > 0
    assert_allclose(drude @ normal, 0, atol=1e-15)


def test_native_precision_controls_resolved_band_geometry():
    model = WannierModel([[0, 0, 0]], [np.diag([0.0, 1e-6])], reciprocal=2 * np.pi * np.eye(3))
    model.degeneracy_tolerance = 2e-6
    with pytest.raises(ValueError, match="individual-band"):
        berry_curvature(model, [[0, 0, 0]])
    with pytest.raises(ValueError, match="degenerate subspace"):
        quantum_metric(model, [[0, 0, 0]], occupied_bands=1)
    with pytest.raises(ValueError, match="degenerate subspace"):
        wilson_loop(model, [[0, 0, 0], [1, 0, 0]], occupied_bands=1)
    assert_allclose(berry_curvature(model, [[0, 0, 0]], degeneracy_tol=1e-8)["curvature"], 0)
    assert_allclose(quantum_metric(model, [[0, 0, 0]], occupied_bands=1, degeneracy_tol=1e-8)["metric"], 0)
    with pytest.raises(ValueError, match="integer indices"):
        berry_curvature(model, [[0, 0, 0]], bands=[0.5])


class Weyl:
    nbnd = 2
    lattice = np.eye(3)
    fermi_energy = 0.0
    spin_degeneracy = 1

    def hamiltonian(self, k):
        return np.einsum("a,aij->ij", 2 * np.pi * np.asarray(k), PAULI)

    def derivatives(self, k, order=1):
        h = PAULI if order == 1 else np.zeros((3, 3, 2, 2), complex)
        return {"H": h, "S": np.zeros_like(h), "r": np.zeros((3, *h.shape), complex)}


def test_weyl_chirality_outward_cartesian_sphere():
    assert_allclose(
        chirality(Weyl(), [0, 0, 0], radius=0.12, polar_points=8, azimuth_points=12)["chirality"],
        1,
        atol=2e-13,
    )

    class TiltedWeyl(Weyl):
        def hamiltonian(self, k):
            return super().hamiltonian(k) + 4 * np.pi * k[2] * np.eye(2)

        def derivatives(self, k, order=1):
            raw = super().derivatives(k, order)
            if order == 1:
                raw["H"] = raw["H"].copy()
                raw["H"][2] += 2 * np.eye(2)
            return raw

    with pytest.raises(ValueError, match="constant occupied rank"):
        chirality(TiltedWeyl(), [0, 0, 0], radius=0.12, polar_points=8, azimuth_points=12)
    assert_allclose(
        chirality(TiltedWeyl(), [0, 0, 0], occupied_bands=1, radius=0.12, polar_points=8, azimuth_points=12)[
            "chirality"
        ],
        1,
        atol=2e-13,
    )


def test_drude_degenerate_velocity_block_is_basis_invariant():
    class RotatedWeyl(Weyl):
        rotation = np.array([[np.cos(0.31), -np.sin(0.31)], [np.sin(0.31), np.cos(0.31)]])

        def hamiltonian(self, k):
            return self.rotation.conj().T @ super().hamiltonian(k) @ self.rotation

        def derivatives(self, k, order=1):
            raw = super().derivatives(k, order)
            return {name: self.rotation.conj().T @ array @ self.rotation for name, array in raw.items()}

    # At the node all Pauli velocities lie within one equal-energy block.
    # Tr(sigma_a sigma_b)=2delta_ab; using diagonal slopes alone loses two axes.
    width = 0.1
    factor = E_CHARGE**2 / HBAR * 1e8 * 2 / (np.sqrt(2 * np.pi) * width)
    expected = factor * np.eye(3)
    first = drude_weight(Weyl(), kpoints=[[0, 0, 0]], broadening=width)["drude_weight"][0]
    rotated = drude_weight(RotatedWeyl(), kpoints=[[0, 0, 0]], broadening=width)["drude_weight"][0]
    assert_allclose(first, expected, rtol=2e-14)
    assert_allclose(rotated, expected, rtol=2e-14, atol=1e-9)
    with pytest.raises(ValueError, match="unresolved dispersive degeneracy"):
        berry_curvature_dipole(Weyl(), kpoints=[[0, 0, 0]], broadening=width)


def test_wilson_winding_and_atomic_polarization():
    model = qwz()
    paths = np.zeros((17, 65, 3))
    paths[:, :, 0] = np.linspace(0, 1, 65)
    paths[:, :, 1] = np.linspace(0, 1, 17)[:, None]
    phases = wilson_loop(model, paths, occupied_bands=1)["phases"][:, 0]
    winding = np.unwrap(2 * np.pi * phases)
    assert_allclose((winding[-1] - winding[0]) / (2 * np.pi), -1, atol=1e-12)
    center = np.array([0.2, 0.1, -0.15])
    position = center[:, None, None, None].astype(complex)
    atom = OpenMXModel(
        [[0, 0, 0]],
        [[[-1.0]]],
        position=position,
        lattice=np.eye(3),
        spin_degeneracy=1,
        atom_positions=center[None],
    )
    # First-order operator links converge quadratically; do not claim exact finite-step fidelity.
    coarse = polarization(atom, grid=(12, 12, 12), occupied_bands=1)["reduced_electronic"]
    fine = polarization(atom, grid=(24, 24, 24), occupied_bands=1, ionic_charges=[1])
    assert_allclose(fine["reduced_electronic"], -center, atol=2e-4)
    assert np.linalg.norm(fine["reduced_electronic"] + center) < 0.26 * np.linalg.norm(coarse + center)
    assert_allclose(fine["ionic"], E_CHARGE * 1e20 * center, atol=1e-12)
    assert_allclose(fine["total"], fine["ionic"] + fine["electronic"], atol=1e-12)


class SmoothOrbital:
    """One isolated retained band with a known periodic Berry connection."""

    nbnd = 1
    lattice = np.eye(3)
    fermi_energy = 0.0
    spin_degeneracy = 1

    def hamiltonian(self, k):
        return np.array([[-1.0]])

    def position(self, k):
        y = 2 * np.pi * k[1]
        return np.array([0.32 + 0.27 * np.cos(y) + 0.11 * np.cos(2 * y), 0, 0])[:, None, None]

    def basis_overlap(self, k, kp):
        delta = (np.asarray(kp) - k) * 2 * np.pi
        connection = self.position((np.asarray(k) + kp) / 2)[:, 0, 0]
        return np.exp(-1j * np.dot(delta, connection)).reshape(1, 1)


def test_polarization_averages_unwrapped_phases_and_gates_chern_manifold():
    # Integrating A_x=.32+.27cos(2pi ky)+.11cos(4pi ky) gives mean .32.
    # Its asymmetric phase distribution crosses a branch; circular averaging
    # would return a biased mean, whereas a continuous branch is exact.
    result = polarization(SmoothOrbital(), grid=(4, 24, 2), occupied_bands=1)
    assert_allclose(result["reduced_electronic"], [-0.32, 0, 0], atol=2e-14)
    with pytest.raises(ValueError, match="zero-Chern"):
        polarization(qwz(), grid=(8, 8, 2), occupied_bands=1)


def test_wilson_bounded_blocks_and_parallel_paths_preserve_gauge():
    model = qwz()
    path = np.zeros((65, 3))
    path[:, 0] = np.linspace(0, 1, len(path))
    path[:, 1] = 0.2
    reference = wilson_loop(model, path, occupied_bands=1)["phases"]
    bounded = wilson_loop(
        model, path, occupied_bands=1, execution=ExecutionOptions(batch_size=2, memory_mb=0.1)
    )["phases"]
    assert_allclose(bounded, reference, atol=2e-14)
    paths = np.array([path, path])
    parallel = wilson_loop(
        model, paths, occupied_bands=1, execution=ExecutionOptions(workers=2, batch_size=2, memory_mb=0.1)
    )["phases"]
    assert_allclose(parallel, np.repeat(reference, 2, axis=0), atol=2e-14)


class ResponsePatch:
    """An independently prescribed retained interband dipole geometry patch.

    Lower energy qx has constant DOS and Berry curvature +1/2; upper energy10.
    It deliberately tests the Kubo/finite-basis response definition, not a
    periodic full-continuum Direct curvature claim.
    """

    nbnd = 2
    lattice = np.eye(3)
    fermi_energy = 0.0
    spin_degeneracy = 1

    def hamiltonian(self, k):
        return np.diag([2 * np.pi * k[0], 10.0])

    def position(self, k):
        return np.array([0.5 * PAULI[0], -0.5 * PAULI[1], np.zeros((2, 2))])

    def derivatives(self, k, order=1):
        h = np.zeros((3, 2, 2), complex) if order == 1 else np.zeros((3, 3, 2, 2), complex)
        if order == 1:
            h[0, 0, 0] = 1
        return {"H": h, "S": np.zeros_like(h), "r": np.zeros((3, *h.shape), complex)}


def test_projected_coordinate_commutator_and_direct_kubo_difference():
    model = ResponsePatch()
    point = np.array([0.01, 0.1, 0.2])
    position = model.position(point)
    curvature = basis_curvature(model, point)
    for a in range(3):
        for b in range(3):
            assert_allclose(
                curvature[a, b], -1j * (position[a] @ position[b] - position[b] @ position[a]), atol=1e-14
            )
    direct = berry_curvature(model, [point], engine="direct")["curvature"][:, 0]
    kubo = berry_curvature(model, [point], engine="kubo")["curvature"][:, 0]
    assert_allclose(direct, 0, atol=1e-14)
    assert_allclose(direct - kubo, curvature.diagonal(axis1=2, axis2=3).transpose(2, 0, 1), atol=1e-14)


def test_entropy_anc_mott_limit_bcd_and_drude_normalization():
    model = ResponsePatch()
    n = 1000
    points = np.zeros((n, 3))
    points[:, 0] = ((np.arange(n) + 0.5) / n - 0.5) / (2 * np.pi)
    sigma_factor = E_CHARGE**2 / HBAR * 1e8
    # A_x=.5 sigma_x,A_y=-.5 sigma_y, so Omega_xy=+.5 in the lower band.
    thermal = anc(model, kpoints=points, temperature_kelvin=100, engine="kubo")["anc"][0, 0, 1]
    expected = 0.5 * sigma_factor * np.pi**2 / 3 * KB_EV**2 * 100
    assert_allclose(thermal, expected, rtol=2e-10)
    assert_allclose(anc(model, kpoints=points[:1], temperature_kelvin=0, engine="kubo")["anc"], 0)
    bcd = berry_curvature_dipole(model, kpoints=points, engine="kubo", broadening=0.02)[
        "berry_curvature_dipole"
    ][0]
    assert_allclose(bcd, [[0, 0, 0.5], [0, 0, 0], [0, 0, 0]], atol=2e-12)
    weight = drude_weight(model, kpoints=points, broadening=0.02)["drude_weight"][0]
    assert_allclose(weight, np.diag([sigma_factor, 0, 0]), rtol=1e-12, atol=1e-12)


def test_orbital_magnetization_particle_hole_and_streda_identity():
    model = qwz(lattice=np.diag([1.0, 1.0, 10.0]))
    mus = np.array([-0.2, 0, 0.2])
    result = orbital_magnetization(model, grid=(32, 32, 1), chemical_potentials=mus)
    assert_allclose(result["orbital_magnetization"][:, 2], E2_H * 1e9 * mus, rtol=2e-8, atol=1e-8)
    assert_allclose(result["orbital_magnetization"], result["local"] + result["itinerant"], atol=1e-12)


def test_conserved_spin_hall_equals_difference_of_charge_blocks():
    up, down = qwz(-1), qwz(1)
    hopping = np.array(
        [block_diag(a, b) for a, b in zip(up.hopping_matrices, down.hopping_matrices, strict=True)]
    )
    model = OpenMXModel(TRANSLATIONS, hopping, lattice=np.diag([1.0, 1.0, 10.0]), spin_mode="collinear")
    result = shc(model, grid=(32, 32, 1))["shc"][0]
    assert_allclose(result[2, 0, 1], -2 * E2_H * 1e7, rtol=2e-8)
    assert_allclose(result[:2], 0, atol=1e-12)
    assert_allclose(result[2, 1, 0], -result[2, 0, 1], atol=1e-12)


def test_geometry_parallel_execution_agrees():
    model = qwz()
    points = [[0, 0, 0], [0.1, 0.2, 0], [0.23, 0.31, 0]]
    serial = berry_curvature(model, points)["curvature"]
    parallel = berry_curvature(model, points, execution=ExecutionOptions(workers=2, batch_size=1))[
        "curvature"
    ]
    assert_allclose(parallel, serial, atol=1e-13)
    assert_allclose(band_geometry(model, points[0])["velocities"][0].diagonal().imag, 0, atol=1e-14)


@pytest.mark.parametrize(
    "name",
    [
        "ahc",
        "anc",
        "berry_curvature_dipole",
        "drude_weight",
        "orbital_magnetization",
        "shc",
        "berry_curvature",
        "quantum_metric",
    ],
)
def test_geometry_registered_tiles_equal_direct_weighted_result(tmp_path, name):
    """Two durable tiles must preserve physical prefactors, axes and weights."""
    from types import SimpleNamespace

    from queer.io.artifacts import RunStore
    from queer.observables._tb import kmesh
    from queer.workflows.registry import REGISTRY, calculate_registered

    original = qwz()
    up, down = qwz(-1), qwz(1)
    model = (
        OpenMXModel(
            TRANSLATIONS,
            np.array(
                [block_diag(a, b) for a, b in zip(up.hopping_matrices, down.hopping_matrices, strict=True)]
            ),
            lattice=np.eye(3),
            spin_mode="collinear",
        )
        if name == "shc"
        else original
    )
    points = kmesh((7, 7, 3))
    params = {"kpoints": points.tolist()}
    if name not in {"berry_curvature", "quantum_metric"}:
        params.update(
            weights=np.linspace(0.5, 1.5, len(points)).tolist(),
            chemical_potentials=[-0.3, 0, 0.3],
            temperature_kelvin=180.0,
        )
    if name == "quantum_metric":
        params["occupied_bands"] = 1
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
        with np.load(store.root / store.manifest["cases"]["result"]["path"], allow_pickle=False) as result:
            assert set(result.files) == set(expected)
            for key in expected:
                roundoff = (
                    32 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(expected[key]), initial=0)))
                )
                assert_allclose(result[key], expected[key], rtol=3e-12, atol=roundoff)
        # A complete result must be reused without touching its case records.
        records = dict(store.manifest["cases"])
        calculate_registered(config, model, store)
        assert store.manifest["cases"] == records
    finally:
        store.release()


def test_real_gaas_time_reversal_geometry_and_hall_cancellation():
    """SOC scfout obeys TR parity to its native grid-integration precision."""
    from pathlib import Path

    from threadpoolctl import threadpool_limits

    source = Path(__file__).resolve().parents[2] / "data/references/openmx/GaAs_soc.scfout"
    if not source.exists():
        pytest.skip("optional GaAs OpenMX reference fixture is not installed")
    with threadpool_limits(limits=1):
        model = OpenMXModel.from_scfout(source)
        point = np.array([0.137, 0.217, 0.319])
        points = np.array([point, -point])
        curvature = berry_curvature(model, points, occupied_bands=28)["curvature"]
        metric = quantum_metric(model, points, occupied_bands=28)["metric"]
        assert_allclose(curvature[0], -curvature[1], atol=1e-8)
        assert_allclose(metric[0], metric[1], atol=2e-8)
        assert np.min(np.linalg.eigvalsh(metric)) > 0
        assert_allclose(ahc(model, kpoints=points)["ahc"], 0, atol=1e-5)
        assert_allclose(orbital_magnetization(model, kpoints=points)["orbital_magnetization"], 0, atol=1e-3)


def test_real_fecl2_wilson_flux_converges_to_cartesian_direct_curvature():
    from pathlib import Path

    from threadpoolctl import threadpool_limits

    from queer.observables.geometry import _plaquette

    source = Path(__file__).resolve().parents[2] / "data/references/openmx/FeCl2.scfout"
    if not source.exists():
        pytest.skip("optional FeCl2 OpenMX reference fixture is not installed")
    with threadpool_limits(limits=1):
        model = OpenMXModel.from_scfout(source)
        point = np.array([0.137, 0.217, 0.319])
        tensor = berry_curvature(model, [point], occupied_bands=28)["curvature"][0]
        exact = np.einsum("ab,a,b", tensor, model.g_vec[0], model.g_vec[1])
        errors = []
        for delta in (0.005, 0.0025, 0.00125):
            u, v = delta * np.eye(3)[:2]
            corners = np.array(
                [point - u / 2 - v / 2, point + u / 2 - v / 2, point + u / 2 + v / 2, point - u / 2 + v / 2]
            )
            finite_flux = -_plaquette(model, corners, 28) / delta**2
            errors.append(abs(finite_flux - exact))
        assert errors[-1] < 5e-4 * abs(exact)
        assert errors[1] < 0.26 * errors[0]
        assert errors[2] < 0.26 * errors[1]


@pytest.mark.parametrize("name", ["chern_number", "chirality", "wilson_loop", "polarization", "ahc"])
def test_geometry_registered_global_results_equal_python(tmp_path, name):
    from types import SimpleNamespace

    from queer.io.artifacts import RunStore
    from queer.workflows.registry import REGISTRY, calculate_registered

    model = qwz()
    if name == "chern_number":
        params = {"grid": [8, 8, 1], "occupied_bands": 1}
    elif name == "chirality":
        model = Weyl()
        params = {"kpoint": [0, 0, 0], "radius": 0.12, "polar_points": 4, "azimuth_points": 8}
    elif name == "wilson_loop":
        path = np.zeros((17, 3))
        path[:, 0] = np.linspace(0, 1, len(path))
        path[:, 1] = 0.2
        params = {"kpoints": path.tolist(), "occupied_bands": 1}
    elif name == "polarization":
        center = np.array([0.2, 0.1, -0.15])
        model = OpenMXModel(
            [[0, 0, 0]],
            [[[-1.0]]],
            position=center[:, None, None, None],
            lattice=np.eye(3),
            spin_degeneracy=1,
            atom_positions=center[None],
        )
        params = {"grid": [4, 3, 2], "occupied_bands": 1, "ionic_charges": [1]}
    else:
        params = {"grid": [8, 8, 1], "engine": "fhs"}
    execution = ExecutionOptions()
    config = SimpleNamespace(
        workflow=name, parameters=params, execution=execution, inputs={}, material_dir=tmp_path
    )
    serialized = {"workflow": name, "parameters": params, "inputs": {}, "execution": execution.as_dict()}
    store = RunStore(tmp_path / name, serialized, [])
    try:
        expected = REGISTRY[name].calculator()(model, **params).arrays
        calculate_registered(config, model, store)
        assert set(store.manifest["cases"]) == {"result"}
        with np.load(store.root / store.manifest["cases"]["result"]["path"], allow_pickle=False) as result:
            for key in expected:
                assert_allclose(result[key], expected[key], rtol=3e-12, atol=3e-12)
    finally:
        store.release()
