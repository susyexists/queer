import math

import numpy as np
import pytest

from queer.finite_field.geometry import band_geometry
from queer.finite_field.models import FourBandModel, FourBandParams
from queer.finite_field.units import critical_field

P = FourBandParams()


def test_vlc_and_paper_matched_variant():
    assert P.vlc == pytest.approx(0.913980695e-3, rel=1e-9)
    assert FourBandParams.paper_vlc_matched().vlc == pytest.approx(0.913422e-3, rel=1e-6)


@pytest.mark.parametrize("x", [0.0, 0.5, 1.0, 1.5])
def test_k0_spectrum(x):
    p = P.at(x)
    ev = np.linalg.eigvalsh(FourBandModel(p).traceless().hamiltonian(0.0, 0.0))
    R = p.R
    expect = np.sort([-abs(p.M) - R, abs(p.M) - R, R - abs(p.M), abs(p.M) + R])
    assert np.allclose(ev, expect, atol=1e-16)
    assert np.allclose(ev, p.spectrum_closed_form(0.0), atol=1e-16)


def test_vl0_spectrum_values():
    ev = np.linalg.eigvalsh(FourBandModel(P).traceless().hamiltonian(0.0, 0.0)) * 1e3
    assert np.allclose(ev, [-1.4793, -0.5647, 0.5647, 1.4793], atol=1e-12)


def test_gap_closes_only_at_gamma_at_vlc():
    m = FourBandModel(P.at(1.0))
    assert band_geometry(m, 0.0, 0.0).eps_g < 1e-15
    k = np.geomspace(1e-6, 0.2, 400)
    gap = band_geometry(m, k, 0 * k).eps_g
    assert np.all(gap > 1.5 * 0.147 * k)  # linear Dirac closing, no other touching point


@pytest.mark.parametrize("x", [0.0, 0.3, 1.0, 1.7])
def test_closed_form_gap_matches_numerics(x, rng):
    m = FourBandModel(P.at(x))
    kx, ky = rng.uniform(-0.03, 0.03, (2, 500))
    g = band_geometry(m, kx, ky)
    assert np.max(np.abs(g.eps_g - P.at(x).gap_closed_form(np.hypot(kx, ky)))) < 1e-15


def test_effective_model_constants():
    pc = P.at(1.0)
    assert pc.hbar_v == pytest.approx(0.14721301, rel=1e-7)
    assert pc.m2 == pytest.approx(-41.3535005, rel=1e-7)
    assert pc.C_critical == pytest.approx(140.454636, rel=1e-7)


def test_ec_closed_form_values_and_consistency():
    for x, expect in [(0.0, 30450.158), (0.5, 19034.420), (1.2, 6861.673)]:
        p = P.at(x)
        assert p.ec_k0 == pytest.approx(expect, rel=1e-7)
        assert critical_field(p.gap_k0, p.gcv_k0) == pytest.approx(p.ec_k0, rel=1e-12)
    assert P.ec_k0 == pytest.approx(math.pi * (abs(P.M) - P.Bl) ** 2 / P.A * 1e10, rel=1e-12)


def test_ring_threshold_and_diabatic_onset():
    xs = P.ring_threshold_x()
    assert xs == pytest.approx(1.2134745, abs=1e-6)
    # curvature of eps_g^2 at k = 0 changes sign at the threshold
    for x, sign in [(xs - 1e-3, +1), (xs + 1e-3, -1)]:
        g = P.at(x).gap_closed_form(np.array([0.0, 1e-5]))
        assert np.sign(g[1] ** 2 - g[0] ** 2) == sign
    assert P.x_diabatic == pytest.approx(1.1181855, abs=1e-6)
    assert P.at(1.1).k_diabatic() is None
    p2 = P.at(2.0)
    k = p2.k_diabatic()
    assert k == pytest.approx(4.650223e-3, rel=1e-6)
    assert p2.A**2 * k**2 + (p2.M - p2.B * k**2) ** 2 == pytest.approx(p2.VL**2, rel=1e-12)
    assert P.k_inv == pytest.approx(0.0319783297, rel=1e-8)


@pytest.mark.parametrize(
    "opts",
    [{}, {"include_identity": False}, {"d3_layer": "tz"}, {"warping": 200.0, "warping_matrix": "s0_tz"}],
)
def test_dhamiltonian_matches_finite_differences(opts, rng):
    m = FourBandModel(P.at(0.7), **opts)
    kx, ky = rng.uniform(-0.03, 0.03, (2, 20))
    h = 1e-7
    dHx, dHy = m.dhamiltonian(kx, ky)
    fx = (m.hamiltonian(kx + h, ky) - m.hamiltonian(kx - h, ky)) / (2 * h)
    fy = (m.hamiltonian(kx, ky + h) - m.hamiltonian(kx, ky - h)) / (2 * h)
    assert np.max(np.abs(dHx - fx)) < 1e-6 * np.max(np.abs(dHx))
    assert np.max(np.abs(dHy - fy)) < 1e-6 * np.max(np.abs(dHy))


def test_geometry_independent_of_identity_terms(rng):
    kx, ky = rng.uniform(-0.03, 0.03, (2, 2000))
    for x in (0.3, 1.0000001, 1.7):
        full = band_geometry(FourBandModel(P.at(x)), kx, ky)
        tl = band_geometry(FourBandModel(P.at(x)).traceless(), kx, ky)
        assert np.max(np.abs(full.eps_g - tl.eps_g)) < 1e-15
        assert np.max(np.abs(full.omega_c - tl.omega_c)) < 1e-12 * np.max(np.abs(tl.omega_c))
        assert np.max(np.abs(full.g_cv - tl.g_cv)) < 1e-12 * np.max(np.abs(tl.g_cv))


@pytest.mark.parametrize("opts", [{}, {"include_identity": False}, {"d3_layer": "tz"}, {"warping": 150.0}])
def test_complex_k_hamiltonian_is_the_analytic_continuation(opts, rng):
    """H accepts complex k (branch points): entire in k, unchanged on the real axis."""
    m = FourBandModel(P.at(1.3), **opts)
    assert m.complex_k
    kx, ky = rng.uniform(-3e-3, 3e-3, (2, 16))
    H = m.hamiltonian(kx, ky)
    assert H.dtype == complex and np.allclose(m.hamiltonian(kx + 0j, ky + 0j), H, rtol=0, atol=1e-18)
    # Cauchy-Riemann: d/d(kx) along the real and the imaginary direction agree with dhamiltonian
    h = 1e-7
    dx = (m.hamiltonian(kx + h, ky) - m.hamiltonian(kx - h, ky)) / (2 * h)
    dix = (m.hamiltonian(kx + 1j * h, ky) - m.hamiltonian(kx - 1j * h, ky)) / (2j * h)
    dHx = m.dhamiltonian(kx, ky)[0]
    assert np.allclose(dx, dHx, rtol=0, atol=1e-8) and np.allclose(dix, dHx, rtol=0, atol=1e-8)


@pytest.mark.parametrize("x, t", [(0.5, 0.0), (1.5, 1e-3), (2.0, 4e-3)])
def test_v_and_c_coalesce_at_the_exact_branch_point(x, t):
    from queer.finite_field.lz.landau_zener import fourband_ddp_t

    k_c = fourband_ddp_t(P.at(x), t)[0][0]
    ev = np.sort_complex(np.linalg.eigvals(FourBandModel(P.at(x)).traceless().hamiltonian(k_c, t)))
    lo, hi = abs(ev[2] - ev[1]), abs(ev[3] - ev[2])
    assert lo < 1e-6 * hi  # v and c meet (a square-root branch point: |split| ~ sqrt(rounding))
