import math

import numpy as np
import pytest

from queer.finite_field.geometry import band_geometry
from queer.finite_field.geometry.patch import RadialPatch
from queer.finite_field.models import FourBandModel, FourBandParams

pytest.importorskip("mpmath", reason="extended-precision reference extra")
pytestmark = pytest.mark.precision

P = FourBandParams()


@pytest.fixture(scope="module")
def critical_patch():
    return RadialPatch(P.at(1.0))


def test_patch_removes_float64_floor(critical_patch):
    assert critical_patch.exact_critical
    k = np.geomspace(1e-20, 1e-9, 12)  # includes extrapolation below the table (k_min = 1e-14)
    g = critical_patch.evaluate(k, 0 * k)
    assert np.allclose(k * g.omega_c, -P.at(1.0).C_critical, rtol=1e-7)
    assert np.allclose(g.eps_g / (2 * k), P.at(1.0).hbar_v, rtol=1e-7)
    # float64 is already wrong at k = 1e-10 (sign/magnitude garbage), the patch is not
    fp = band_geometry(FourBandModel(P.at(1.0)).traceless(), 1e-10, 0.0)
    assert abs(1e-10 * fp.omega_c + P.at(1.0).C_critical) > 1.0


def test_patch_overlaps_float64(critical_patch):
    m = FourBandModel(P.at(1.0)).traceless()
    assert critical_patch.overlap_error(m, 1e-7, 1e-6) < 1e-5


def test_patch_tensor_orientation(critical_patch):
    m = FourBandModel(P.at(1.0)).traceless()
    k = 5e-7
    for alpha in (0.0, 0.4, 1.2):
        kx, ky = k * math.cos(alpha), k * math.sin(alpha)
        for angle in (0.0, 0.9):
            a = critical_patch.evaluate(kx, ky, angle)
            b = band_geometry(m, kx, ky, angle=angle)
            assert np.allclose(a.g_cv, b.g_cv, rtol=1e-5, atol=1e-5 * np.max(np.abs(b.g_cv)))
            assert a.omega_c == pytest.approx(float(b.omega_c), rel=1e-5)


def test_patch_off_critical_finite_limit():
    p = P.at(1.0 + 1e-6)
    patch = RadialPatch(p)
    assert not patch.exact_critical
    m = FourBandModel(p).traceless()
    assert patch.overlap_error(m, 1e-7, 1e-6) < 1e-5
    g = patch.evaluate(np.array([1e-22, 1e-20]), np.zeros(2))
    assert np.all(np.isfinite(g.omega_c)) and abs(g.omega_c[0] / g.omega_c[1] - 1) < 1e-6
    # massive limit: eps_g -> 2|m0|
    assert g.eps_g[0] == pytest.approx(2 * abs(p.m0), rel=1e-6)


def test_patch_and_provider_pickle(critical_patch):
    """scipy >= 1.18 splines are not picklable; the patch rebuilds them lazily (worker pools)."""
    import pickle

    from queer.finite_field.geometry.provider import ModelGeometry

    back = pickle.loads(pickle.dumps(critical_patch))
    k = np.geomspace(1e-13, 1e-6, 7)
    assert np.array_equal(back.evaluate(k, 0 * k).omega_c, critical_patch.evaluate(k, 0 * k).omega_c)
    prov = ModelGeometry(FourBandModel(P.at(1.0)))
    prov2 = pickle.loads(pickle.dumps(prov))
    s = np.concatenate([-k, k, [3e-6, 2e-3]])
    a, b = prov.geometry(s, 1e-9 + 0 * s), prov2.geometry(s, 1e-9 + 0 * s)
    for f in ("eps_v", "eps_c", "omega_v", "omega_c", "g_cv", "g_v_total", "g_c_total", "deg_flag"):
        assert np.array_equal(getattr(a, f), getattr(b, f))


def test_extrapolation_below_table_is_exact():
    """Below k_min the surrogates are constant: exact at x = 1 (g_rr -> C^2, tabulated without k^2 weight)
    and for nearly critical unsnapped models (table extended to 1e-4 k_s)."""
    from queer.finite_field.geometry.patch import _mp_geometry

    for x, k_test, tol in ((1.0, (1e-16, 1e-22), 1e-12), (1 - 1e-12, (1e-22,), 1e-7)):
        p = P.at(x)
        patch = RadialPatch(p)
        assert patch.k_min <= min(1e-14, 1e-4 * patch.k_s) or patch.k_s == 0
        for k in k_test:
            ref = _mp_geometry(p, k, 60, patch.exact_critical)
            g = patch.evaluate(np.array([k]), np.array([0.0]))
            assert g.g_cv[0, 0] == pytest.approx(ref["g_cv_rr"], rel=tol)
            assert g.g_v_total[0, 0] == pytest.approx(ref["g_v_rr"], rel=tol)
            assert g.omega_c[0] == pytest.approx(ref["omega_c"], rel=tol)


def test_patched_points_flags_and_v_cv():
    """At patched points deg_flag and v_cv come from the patch (no float64 'near_degenerate' at x = 1)."""
    from queer.finite_field.geometry.provider import ModelGeometry

    prov = ModelGeometry(FourBandModel(P.at(1.0)), angle=0.6)
    s = np.array([1e-12, 3e-12, -2e-9, 5e-7])
    g = prov.geometry(s, np.array([0.0, 1e-12, 4e-10, -1e-7]))
    assert g.extras["patched"].all() and not g.deg_flag.any()
    assert np.allclose(np.abs(g.v_cv[:, 0]) ** 2 / g.eps_g**2, g.g_cv[:, 0], rtol=1e-10)
    assert np.allclose(np.abs(g.v_cv[:, 1]) ** 2 / g.eps_g**2, g.g_cv[:, 2], rtol=1e-10)
    q = g.v_cv[:, 0] * np.conj(g.v_cv[:, 1])  # X^s X^t*: Re = g_st eps_g^2, Im = pair part of Omega_c
    assert np.allclose(q.real, g.g_cv[:, 1] * g.eps_g**2, rtol=1e-8, atol=1e-12 * np.abs(q).max())
    m = FourBandModel(P.at(1.0)).traceless()
    k = 5e-7  # float64 reliable: same |v_cv| as the Kubo sum
    assert np.allclose(
        np.abs(prov.at_cartesian(k, 0.0).v_cv), np.abs(band_geometry(m, k, 0.0).v_cv), rtol=1e-6
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        g0 = prov.geometry(0.0, 0.0)
    assert g0.eps_g == 0.0 and g0.deg_flag  # the touching point itself


def test_at_cartesian_applies_patch():
    """GriddedGeometry.sample uses at_cartesian: r[0] ~ 1e-12 must be exact (model frame, angle 0)."""
    from queer.finite_field.geometry.provider import ModelGeometry

    prov = ModelGeometry(FourBandModel(P.at(1.0)), angle=0.9)
    k = np.geomspace(1e-12, 1e-8, 5)
    for alpha in (0.0, 2.2):
        g = prov.at_cartesian(k * math.cos(alpha), k * math.sin(alpha))
        assert g.extras["patched"].all()
        assert np.allclose(k * g.omega_c, -P.at(1.0).C_critical, rtol=1e-7)
        # model-frame tensor: g = C^2 k_hat k_hat + 1/(4 k^2) theta_hat theta_hat
        assert np.allclose(
            g.g_cv[:, 0],
            P.at(1.0).C_critical ** 2 * math.cos(alpha) ** 2 + math.sin(alpha) ** 2 / (4 * k**2),
            rtol=1e-6,
        )


def test_radial_extrema_prominence_matches_scan_path():
    """Shallow ring just above x* = 1.21347: the radial path (prominence_rel = 1e-12 with basin merging, as
    lz.crossings.gap_minima) finds the same minima as the scan; nothing changes away from x*."""
    from queer.finite_field.config import PAPER
    from queer.finite_field.geometry.provider import ModelGeometry
    from queer.finite_field.lz.crossings import gap_minima

    pol = PAPER.crossings
    xs = P.ring_threshold_x()
    for x, rtol in ((1.2140, 1e-4), (1.2150, 1e-4), (xs + 1e-5, 1e-3), (xs + 3e-5, 1e-3)):
        prov = ModelGeometry(FourBandModel(P.at(x)))
        for t in (0.0, 1e-6):
            rad = prov.line_minima_radial(t, pol.window)
            scan = gap_minima(prov, t, pol.window, pol.scan_points, pol.prominence_rel)[0]
            assert len(rad) == len(scan) == 2 and np.allclose(rad, scan, rtol=rtol, atol=0)
    prov = ModelGeometry(FourBandModel(P.at(xs + 1e-5)))
    assert len(prov.line_minima_radial(0.0, pol.window)) == 2
    assert prov.line_minima_radial(0.0, pol.window, prominence_rel=1e-9) == [
        0.0
    ]  # the old threshold missed it
    assert len(prov.__dict__["_radial_cache"]) == 2  # the cache key includes prominence_rel
    for x in (0.5, 1.0, 1.5, 2.0):
        prov = ModelGeometry(FourBandModel(P.at(x)))
        for t in (0.0, 1e-6, 1e-4):
            new = prov.line_minima_radial(t, pol.window)
            assert new == prov.line_minima_radial(t, pol.window, prominence_rel=1e-9)
            scan = gap_minima(prov, t, pol.window, pol.scan_points, pol.prominence_rel)[0]
            assert len(new) == len(scan) and np.allclose(new, scan, rtol=1e-6, atol=1e-9)
