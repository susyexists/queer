"""Native plane covariance and generic-model capability boundaries."""

import numpy as np
import pytest

from queer import OpenMXModel, WannierModel
from queer.finite_field.native import FieldPlane, NativePlaneModel
from queer.observables._tb import band_geometry


def model():
    # Periodic two-band model with three independent Cartesian derivatives.
    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.diag([1, -1])
    translations = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    hopping = np.array([0.3 * sz, sx / 2j, -sx / 2j, sy / 2j, -sy / 2j])
    return WannierModel(translations, hopping, lattice=np.diag([2, 3, 4]))


def test_plane_skew_conversion_and_orientation():
    m = model()
    m = WannierModel(m.translations, m.hopping_matrices, lattice=[[2, 0.2, 0], [0, 3, 0.4], [0.3, 0, 4]])
    plane = FieldPlane(origin=(0.1, 0.2, 0.3), direction=(0, 0, 1), transverse=(1, 0, 0))
    points = plane.fractional(m, [0.2, -0.3], [0.4, 0.1])
    np.testing.assert_allclose((points - plane.origin) @ m.g_vec, [[0.4, 0, 0.2], [0.1, 0, -0.3]], atol=1e-15)
    np.testing.assert_allclose(plane.normal, [0, 1, 0])


@pytest.mark.parametrize("axes", [((2, 0, 0), (0, 1, 0)), ((1, 0, 0), (1, 0, 0))])
def test_plane_rejects_nonorthonormal_axes(axes):
    with pytest.raises(ValueError, match="orthonormal"):
        FieldPlane(direction=axes[0], transverse=axes[1])


def test_pair_metrics_and_curvature_rotate_as_tensors():
    m = model()
    first = NativePlaneModel(m, band_pair=(0, 1))
    swapped = NativePlaneModel(m, plane={"direction": [0, 1, 0], "transverse": [1, 0, 0]}, band_pair=(0, 1))
    x, y = np.array([0.07, 0.13]), np.array([0.04, -0.02])
    a, b = first.geometry(x, y), swapped.geometry(y, x)
    np.testing.assert_allclose(a.omega_c, -b.omega_c, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(a.g_cv[:, [2, 1, 0]], b.g_cv, rtol=1e-12, atol=1e-12)
    k = first.plane.fractional(m, x[0], y[0])
    native = band_geometry(m, k)
    expected = np.outer(native["connections"][:2, 0, 1], native["connections"][:2, 0, 1].conj()).real
    np.testing.assert_allclose(a.g_cv[0], expected[[0, 0, 1], [0, 1, 1]])


def test_nonorthogonal_geometry_uses_generalized_solution():
    m = model()
    transform = np.array([[1.4, 0.2j], [0, 0.6]], complex)
    h = np.einsum("ai,raj,jb->rib", transform.conj(), m.hopping_matrices, transform)
    s = np.zeros_like(h)
    s[0] = transform.conj().T @ transform
    position = np.zeros((3, *h.shape), complex)
    generalized = OpenMXModel(
        m.translations, h, s, lattice=np.diag([2, 3, 4]), position=position, spin_degeneracy=1
    )
    a = NativePlaneModel(m, band_pair=(0, 1)).geometry([0.03], [0.05])
    view = NativePlaneModel(generalized, band_pair=(0, 1))
    b = view.geometry([0.03], [0.05])
    np.testing.assert_allclose(a.g_cv, b.g_cv, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(a.omega_c, b.omega_c, rtol=1e-12, atol=1e-12)
    assert not view.supports_lowdin_patch
    assert not view.complex_k
    assert not view.supports_tdse


def test_complex_continuation_preserves_nonhermitian_matrix():
    m = model()
    view = NativePlaneModel(m, band_pair=(0, 1))
    k = view.plane.fractional(m, 0.05 + 0.02j, 0.03)
    h = view.hamiltonian(0.05 + 0.02j, 0.03)
    np.testing.assert_allclose(h, m.analytic_hamiltonian(k), atol=1e-15)
    assert np.max(np.abs(h - h.conj().T)) > 1e-3


def test_openmx_fingerprints_depend_on_operator_contents():
    m = OpenMXModel([[0, 0, 0]], [np.diag([-1, 1])], position=np.zeros((3, 1, 2, 2)))
    n = OpenMXModel([[0, 0, 0]], [np.diag([-2, 2])], position=np.zeros((3, 1, 2, 2)))
    assert (
        NativePlaneModel(m, band_pair=(0, 1)).fingerprint()
        != NativePlaneModel(n, band_pair=(0, 1)).fingerprint()
    )


def test_degenerate_pair_is_masked():
    m = WannierModel([[0, 0, 0]], [np.zeros((2, 2))], lattice=np.eye(3))
    result = NativePlaneModel(m, band_pair=(0, 1)).geometry([0.0], [0.0])
    assert result.deg_flag[0]
    assert np.isnan(result.g_cv).all()


def test_collinear_pair_is_explicit_and_stays_in_channel():
    m = OpenMXModel([[0, 0, 0]], [np.diag([-2, 1, -1, 2])], spin_mode="collinear")
    with pytest.raises(ValueError, match="explicit"):
        NativePlaneModel(m, occupied_bands=2)
    with pytest.raises(ValueError, match="spin channel"):
        NativePlaneModel(m, band_pair=(1, 2))


def test_generalized_eq8_hall_and_harmonics_match_orthogonal_basis():
    from queer.capabilities import CapabilityError
    from queer.finite_field import PreparedHallResponse, resolve_options

    base = model()
    base = WannierModel(base.translations, base.hopping_matrices * 1e-3, lattice=base.lattice)
    transform = np.array([[1.4, 0.2j], [0.0, 0.6]], complex)
    hopping = np.einsum("ai,raj,jb->rib", transform.conj(), base.hopping_matrices, transform)
    overlap = np.zeros_like(hopping)
    overlap[0] = transform.conj().T @ transform
    generalized = OpenMXModel(
        base.translations,
        hopping,
        overlap,
        lattice=base.lattice,
        position=np.zeros((3, *hopping.shape)),
        spin_degeneracy=1,
    )
    cfg = resolve_options(
        {
            "preset": "physical_eq8",
            "drive": {"E_min": 1e5, "E_max": 1e7},
            "crossings": {"window": 0.2, "scan_points": 65},
            "quadrature": {"t_max": 0.02, "du": 1.0, "dmin": 1e-4},
            "fourier": {"n_theta": 32, "n_cap": 128, "tol": 1e-6},
        }
    )
    responses = [
        PreparedHallResponse(NativePlaneModel(m, band_pair=(0, 1)), cfg) for m in (base, generalized)
    ]
    fields = [-4e6, 0, 4e6]
    reference, actual = [response.J(fields) for response in responses]
    assert abs(reference[-1]) > 1e-8
    # Independent avoided-crossing minimizers converge at slightly different
    # floating-point locations; compare the integrated response below its
    # 1e-7 table/quadrature acceptance floor.
    np.testing.assert_allclose(actual, reference, rtol=2e-8, atol=1e-14)
    reference, actual = [response.harmonics([4e6], n_max=3)["b"] for response in responses]
    np.testing.assert_allclose(actual, reference, rtol=2e-8, atol=1e-13)
    with pytest.raises(CapabilityError, match="complex-k"):
        PreparedHallResponse(NativePlaneModel(generalized, band_pair=(0, 1)), options="physical")
