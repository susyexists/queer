import math

import numpy as np
import pytest

from queer.finite_field.geometry import band_geometry
from queer.finite_field.models import FourBandModel, FourBandParams
from queer.finite_field.models.toy import MassiveDirac

P = FourBandParams()


def test_massive_dirac_closed_forms(rng):
    m = MassiveDirac(v=0.3, m=-1.3e-3)
    kx, ky = rng.uniform(-0.02, 0.02, (2, 300))
    g = band_geometry(m, kx, ky)
    assert np.allclose(g.omega_c, m.omega_c(kx, ky), rtol=1e-10, atol=0)
    assert np.allclose(g.omega_v, -g.omega_c, rtol=1e-10, atol=0)
    assert np.allclose(g.g_cv[:, 0], m.g_xx(kx, ky), rtol=1e-10, atol=0)
    assert np.allclose(g.g_cv[:, 1], m.g_xy(kx, ky), rtol=1e-9, atol=1e-12 * np.max(g.g_cv[:, 0]))
    # two-band identity: the band-resolved metric is the total metric of either band
    assert np.allclose(g.g_cv, g.g_v_total, rtol=1e-12) and np.allclose(g.g_cv, g.g_c_total, rtol=1e-12)
    # |<c|dH/dx|v>|^2 / eps_g^2 = g_xx (Methods 6: |A_x,12| = sqrt(g_xx))
    assert np.allclose(np.abs(g.v_cv[:, 0]) ** 2 / g.eps_g**2, g.g_cv[:, 0], rtol=1e-12)


def test_field_frame_rotation(rng):
    m = FourBandModel(P.at(1.4)).traceless()
    kx, ky = rng.uniform(-0.01, 0.01, (2, 50))
    g0 = band_geometry(m, kx, ky, angle=0.0)
    phi = 0.73
    g1 = band_geometry(m, kx, ky, angle=phi)
    assert np.allclose(g1.omega_c, g0.omega_c, rtol=1e-12)
    s = np.array([math.cos(phi), math.sin(phi)])
    t = np.array([-s[1], s[0]])
    G = np.stack(
        [np.stack([g0.g_cv[:, 0], g0.g_cv[:, 1]], -1), np.stack([g0.g_cv[:, 1], g0.g_cv[:, 2]], -1)], -2
    )
    assert np.allclose(g1.g_cv[:, 0], np.einsum("i,nij,j->n", s, G, s), rtol=1e-10)
    assert np.allclose(g1.g_cv[:, 1], np.einsum("i,nij,j->n", s, G, t), atol=1e-10 * np.max(G))
    assert np.allclose(g1.g_cv[:, 2], np.einsum("i,nij,j->n", t, G, t), rtol=1e-10)


def test_particle_hole_relations(rng):
    m = FourBandModel(P.at(0.6))
    kx, ky = rng.uniform(-0.02, 0.02, (2, 400))
    g = band_geometry(m, kx, ky)
    assert np.max(np.abs(g.omega_c + g.omega_v)) < 1e-12 * np.max(np.abs(g.omega_c))
    outer = band_geometry(m, kx, ky, bands=(0, 3))
    assert np.max(np.abs(outer.omega_c + outer.omega_v)) < 1e-12 * np.max(np.abs(outer.omega_v))


def test_critical_berry_curvature_limit():
    m = FourBandModel(P.at(1.0)).traceless()
    k = np.geomspace(1e-7, 1e-6, 5)
    g = band_geometry(m, k, 0 * k)
    assert np.allclose(k * g.omega_c, -P.at(1.0).C_critical, rtol=1e-5)
    assert np.allclose(g.eps_g / (2 * k), P.at(1.0).hbar_v, rtol=1e-5)
    # isotropic: same limit along another direction
    g2 = band_geometry(m, k * math.cos(0.4), k * math.sin(0.4))
    assert np.allclose(k * g2.omega_c, -P.at(1.0).C_critical, rtol=1e-5)


def test_metric_at_gamma_closed_form():
    for x in (0.0, 0.5, 0.9, 1.1):
        p = P.at(x)
        g = band_geometry(FourBandModel(p), 0.0, 0.0)
        assert g.eps_g == pytest.approx(p.gap_k0, rel=1e-12)
        assert g.g_cv[0] == pytest.approx(p.gcv_k0, rel=1e-10)
    assert band_geometry(FourBandModel(P), 0.0, 0.0).omega_c == pytest.approx(1.6971734e5, rel=1e-7)


def test_degeneracy_guard_on_regularization_ring():
    # VL = 0: the t_x sectors cross exactly on |k| = sqrt(M/B); sum-over-states must stay finite
    m = FourBandModel(P)
    k = P.k_inv
    g = band_geometry(m, np.array([k, k * math.cos(0.3)]), np.array([0.0, k * math.sin(0.3)]))
    assert np.all(np.isfinite(g.omega_c)) and np.all(np.isfinite(g.g_cv))
    assert np.all(np.abs(g.omega_c) < 1e3)
    assert g.deg_flag.any()
    # a tiny but physical avoided crossing (VL > 0) must not be dropped
    g_small = band_geometry(FourBandModel(P.at(1e-4)), k, 0.0)
    assert not g_small.deg_flag and np.isfinite(g_small.omega_c)


def test_chunking_does_not_change_results(rng):
    m = FourBandModel(P.at(1.3))
    kx, ky = rng.uniform(-0.01, 0.01, (2, 101))
    a = band_geometry(m, kx, ky)
    b = band_geometry(m, kx, ky, max_chunk=7)
    assert np.array_equal(a.omega_c, b.omega_c) and np.array_equal(a.g_cv, b.g_cv)


def test_shape_broadcasting():
    m = FourBandModel(P.at(0.5))
    s = np.linspace(-1e-3, 1e-3, 5)
    t = np.linspace(0, 1e-3, 3)
    g = band_geometry(m, s[:, None], t[None, :])
    assert g.omega_c.shape == (5, 3) and g.g_cv.shape == (5, 3, 3) and g.v_cv.shape == (5, 3, 2)


@pytest.mark.precision
def test_degeneracy_guard_drops_the_pair_at_a_gap_closing():
    """Documented limitation (kubo.py): within eps_g <= deg_tol of a touching point the guard drops the
    physical (v, c) pair (|<c|dH|v>| = hbar v): g^{cv} = 0 and the point is flagged. ModelGeometry's patch
    serves these points instead (g_rr = C^2)."""
    pytest.importorskip("mpmath")
    from queer.finite_field.geometry.provider import ModelGeometry

    m = FourBandModel(P.at(1.0)).traceless()
    g = band_geometry(m, np.array([1e-12, 3e-12]), np.zeros(2))
    assert g.deg_flag.all() and np.all(g.g_cv[:, 0] == 0.0)
    gp = ModelGeometry(m).geometry(np.array([1e-12, 3e-12]), np.zeros(2))
    assert not gp.deg_flag.any() and np.allclose(gp.g_cv[:, 0], P.at(1.0).C_critical ** 2, rtol=1e-9)
