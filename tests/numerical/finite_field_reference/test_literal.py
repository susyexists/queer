"""Literal Eqs. 7, 8, 12 and 13 (queer.finite_field.literal): limits, broadcasting and the Fourier forms."""

import math
import warnings

import numpy as np
import pytest

from queer.finite_field import literal as L
from queer.finite_field import units
from queer.finite_field.constants import HBAR_EVS
from queer.finite_field.geometry import band_geometry
from queer.finite_field.models import FourBandModel, FourBandParams
from queer.finite_field.models.toy import LinearLZ

PS = 1e-12


def _decay_by_hand(dk, E, T):
    """exp(-hbar dk / (e|E| T)) with explicit constants: e|E| [eV/A] = 1e-10 E [V/m]."""
    return math.exp(-HBAR_EVS * dk / (1e-10 * abs(E) * T))


# --------------------------------------------------------------------------------------------- Eq. 7


def test_p_lz_values_and_limits():
    Ec = 2.0e4
    assert L.p_lz(Ec, Ec) == pytest.approx(math.exp(-1.0), rel=1e-15)
    assert L.p_lz(Ec, -3e3) == L.p_lz(Ec, 3e3) == pytest.approx(math.exp(-Ec / 3e3), rel=1e-15)
    assert L.p_lz(Ec, 0.0) == 0.0  # no field, finite gap: no tunnelling
    assert L.p_lz(0.0, 0.0) == 1.0  # touching point at zero field
    assert L.p_lz(0.0, 3e3) == 1.0 and L.p_lz(0.0, -3e3) == 1.0  # touching point: E_c = 0 gives 1
    assert L.p_lz(Ec, math.inf) == 1.0
    assert L.p_lz(math.inf, 1e4) == 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert L.p_lz(Ec, 1e-310) == 0.0  # subnormal field: exact limit, no overflow warning
        assert L.p_lz(Ec, 0.0) == 0.0
    assert np.isnan(L.p_lz(np.nan, 1e4))
    E = np.geomspace(1e2, 1e6, 50)
    P = L.p_lz(Ec, E)
    assert np.all(np.diff(P) > 0) and np.all((P > 0) & (P < 1))
    with pytest.raises(ValueError):
        L.p_lz(-1.0, 1e4)


def test_p_lz_broadcasting_and_scalars():
    Ec = np.array([0.0, 1e3, 1e4])[:, None]
    E = np.array([-2e3, 0.0, 5e3, 1e5])[None, :]
    P = L.p_lz(Ec, E)
    assert P.shape == (3, 4)
    for i in range(3):
        for j in range(4):
            assert P[i, j] == L.p_lz(float(Ec[i, 0]), float(E[0, j]))
    assert np.all(P[0] == 1.0) and P[1, 1] == 0.0 and P[2, 1] == 0.0
    assert isinstance(L.p_lz(1e3, 2e3), float)  # numpy scalar for scalar input


# --------------------------------------------------------------------------------------------- Eq. 8


def test_critical_field_is_eq8():
    eps_g = np.array([1e-4, 3e-4, 2e-3])
    g = np.array([1e4, 2.5e3, 8.4859e4])
    expected = math.pi * eps_g / (4.0 * np.sqrt(g)) * 1e10
    assert np.allclose(L.critical_field(eps_g, g), expected, rtol=1e-15)
    assert np.array_equal(L.critical_field(eps_g, g), units.critical_field(eps_g, g))
    assert L.critical_field(0.0, 1e4) == 0.0  # touching point -> E_c = 0 -> P_LZ = 1


def test_critical_field_exact_for_linear_lz():
    """Eq. 8 with the Kubo gap and metric at the crossing equals the exact LZ field pi delta^2/alpha."""
    m = LinearLZ(alpha=0.15, delta=0.3e-3)
    g = band_geometry(m, 0.0, 0.0)
    Ec = L.critical_field(g.eps_g, g.g_cv[..., 0])
    assert Ec == pytest.approx(1e10 * math.pi * m.delta**2 / m.alpha, rel=1e-12)


@pytest.mark.paper
def test_fig4b_lz_probability_chain():
    """Fig. 4b (x = 0.9, k_y = 0 line, gap minimum at k = 0): P = 0.277 / 0.773 / 0.879 at 1 / 5 / 10 kV/m."""
    p = FourBandParams().at(0.9)
    g = band_geometry(FourBandModel(p).traceless(), 0.0, 0.0)
    Ec = L.critical_field(g.eps_g, g.g_cv[..., 0])
    assert Ec == pytest.approx(p.ec_k0, rel=1e-10)
    assert np.allclose(L.p_lz(Ec, [1e3, 5e3, 1e4]), [0.277, 0.773, 0.879], atol=6e-4)


# -------------------------------------------------------------------------------------- Eq. 12: Phi


def test_phase():
    assert L.phase(1.31e4, 1.31e4) == pytest.approx(2.0 * math.pi, rel=1e-15)
    assert L.phase(1.31e4, -2.0e3) == L.phase(1.31e4, 2.0e3) == pytest.approx(2.0 * math.pi * 6.55, rel=1e-15)
    assert L.phase(1.0, 0.0) == math.inf
    assert L.phase(0.0, 0.0) == 0.0 and L.phase(0.0, 5e3) == 0.0  # no separation, no phase
    Lam = np.array([1e3, 2e4])[:, None]
    E = np.array([1e3, -4e3, 1e4])
    Phi = L.phase(Lam, E)
    assert Phi.shape == (2, 3)
    assert np.allclose(Phi, 2.0 * math.pi * Lam / np.abs(E), rtol=1e-15)
    # the phase advances by 2 pi between consecutive maxima in 1/E: Phi(1/E + 1/Lambda) = Phi + 2 pi
    assert L.phase(1.31e4, 1.0 / (1.0 / 7e3 + 1.0 / 1.31e4)) - L.phase(1.31e4, 7e3) == pytest.approx(
        2 * math.pi
    )


# ------------------------------------------------------------------------------------- Eq. 12: P_int


def test_p_int_generic_value_by_hand():
    P, k0, E, T1, T2, Phi = 0.37, 2.6e-3, -1.4e4, 2 * PS, 3 * PS, 1.1
    pop = _decay_by_hand(2 * k0, E, T1)
    coh = _decay_by_hand(2 * k0, E, 2 * T1) * _decay_by_hand(2 * k0, E, T2)
    expected = P * (1 + (1 - 2 * P) * pop + 2 * (1 - P) * coh * math.cos(Phi))
    assert L.p_int(P, k0, E, T1, T2, Phi) == pytest.approx(expected, rel=1e-13)
    assert L.p_int(P, k0, -E, T1, T2, Phi) == L.p_int(P, k0, E, T1, T2, Phi)  # |E|


def test_p_int_limits():
    k0, E, T1, T2 = 2.6e-3, 1.4e4, 2 * PS, 2 * PS
    Phi = np.linspace(0.0, 2.0 * math.pi, 7)
    P = np.linspace(0.0, 1.0, 11)[:, None]
    # P -> 0: no tunnelling at either crossing
    assert np.all(L.p_int(0.0, k0, E, T1, T2, Phi) == 0.0)
    assert np.allclose(L.p_int(1e-12, k0, E, T1, T2, Phi), 0.0, atol=1e-11)
    # P = 1: full transfer at the first crossing, back-transfer weighted by the population decay
    assert np.allclose(L.p_int(1.0, k0, E, T1, T2, Phi), 1.0 - _decay_by_hand(2 * k0, E, T1), rtol=1e-13)
    # T1 = T2 = inf: coherent beam splitter, 2P(1 - P)(1 + cos Phi)
    coherent = L.p_int(P, k0, E, math.inf, math.inf, Phi)
    assert np.allclose(coherent, 2 * P * (1 - P) * (1 + np.cos(Phi)), rtol=1e-13, atol=1e-15)
    # k0 -> inf: both decays complete, only the first crossing matters
    assert np.allclose(L.p_int(P, 1e3, E, T1, T2, Phi), P * np.ones_like(Phi), rtol=1e-15)
    # k0 = 0 (merged crossings, Phi = 0): 4P(1 - P)
    assert np.allclose(L.p_int(P, 0.0, E, T1, T2, 0.0), 4 * P * (1 - P), rtol=1e-14)
    # T1 = inf alone: no population decay, coherence limited by T2
    expected = P * (1 + (1 - 2 * P) + 2 * (1 - P) * _decay_by_hand(2 * k0, E, T2) * np.cos(Phi))
    assert np.allclose(L.p_int(P, k0, E, math.inf, T2, Phi), expected, rtol=1e-13, atol=1e-15)


def test_p_int_zero_field_is_finite():
    Ec, k0, Lam = 1.6e4, 2.6e-3, 1.3e4
    for T in (2 * PS, math.inf):
        P = L.p_lz(Ec, 0.0)
        assert L.p_int(P, k0, 0.0, T, T, L.phase(Lam, 0.0)) == 0.0
    # finite T at E = 0: the decays vanish, the cos Phi term drops out although Phi = inf
    assert L.p_int(0.5, k0, 0.0, 2 * PS, 2 * PS, math.inf) == 0.5


def test_p_int_broadcasting():
    P = np.array([0.1, 0.5, 0.9])[:, None, None]
    E = np.array([3e3, -1e4, 5e4, 2e5])[None, :, None]
    Phi = np.linspace(0.0, 3.0, 5)[None, None, :]
    out = L.p_int(P, 1.7e-3, E, 2 * PS, 3 * PS, Phi)
    assert out.shape == (3, 4, 5)
    for i, j, k in [(0, 0, 0), (2, 1, 4), (1, 3, 2)]:
        assert out[i, j, k] == L.p_int(
            float(P[i, 0, 0]), 1.7e-3, float(E[0, j, 0]), 2 * PS, 3 * PS, float(Phi[0, 0, k])
        )
    assert isinstance(L.p_int(0.3, 1e-3, 1e4, 2 * PS, 2 * PS, 0.5), float)
    with pytest.raises(ValueError):
        L.p_int(0.3, -1e-3, 1e4, 2 * PS, 2 * PS, 0.5)
    with pytest.raises(ValueError):
        L.p_int(0.3, 1e-3, 1e4, -PS, 2 * PS, 0.5)


def test_p_int_uses_units_kappa():
    """hbar k0/(e|E|T) = k0/kappa with kappa(1e4 V/m, 2 ps) = 3.0385e-3 1/A."""
    k0 = units.kappa(1e4, 2 * PS) / 2.0  # 2 k0 = kappa: population factor e^{-1}
    assert L.p_int(1.0, k0, 1e4, 2 * PS, math.inf, 0.0) == pytest.approx(1.0 - math.exp(-1.0), rel=1e-14)


# --------------------------------------------------------------------------------------------- Eq. 13


def test_jn_literal_polynomials():
    E0 = np.array([0.5, 1.0, 2.0])
    N = np.arange(6)
    b = L.jn_main_text(lambda E: E**3, E0, N, n_theta=64)
    assert b.shape == (3, 6)
    expected = np.zeros((3, 6))
    expected[:, 1], expected[:, 3] = 0.75 * E0**3, -0.25 * E0**3
    assert np.allclose(b, expected, atol=1e-15 * 8)
    c = L.jn_methods(lambda E: E**3, E0, N, n_theta=64)
    assert np.allclose(c, -0.5j * expected, atol=1e-15 * 8)
    c2 = L.jn_methods(lambda E: E**2, E0, N, n_theta=64)  # E0^2 sin^2 = E0^2 (1 - cos 2 theta)/2
    assert np.allclose(c2[:, 0], E0**2 / 2) and np.allclose(c2[:, 2], -(E0**2) / 4)
    assert np.allclose(np.delete(c2, [0, 2], axis=1), 0.0, atol=1e-15 * 4)


def test_jn_literal_shapes_and_negative_orders():
    J = lambda E: np.tanh(E) + 0.3 * E**2  # noqa: E731
    E0 = np.array([[0.4, 1.0, 1.7], [2.0, 0.1, 3.0]])
    N = np.array([-3, -1, 0, 1, 3])
    c = L.jn_methods(J, E0, N, n_theta=512)
    b = L.jn_main_text(J, E0, N, n_theta=512)
    assert c.shape == b.shape == (2, 3, 5)
    assert np.allclose(c[..., 0], np.conj(c[..., 4])) and np.allclose(c[..., 1], np.conj(c[..., 3]))
    assert np.allclose(b[..., 0], -b[..., 4]) and np.all(b[..., 2] == 0.0)
    assert np.allclose(b, -2.0 * c.imag, rtol=1e-15, atol=0)  # b_N = -2 Im c_N for every N
    scalar = L.jn_methods(J, 1.0, 1, n_theta=256)
    assert np.ndim(scalar) == 0 and np.iscomplexobj(scalar)


def test_jn_literal_validation():
    with pytest.raises(ValueError):
        L.jn_methods(np.sin, 1.0, 64, n_theta=128)  # aliased: |N| must be < n_theta/2
    with pytest.raises(ValueError):
        L.jn_main_text(np.sin, 1.0, 1.5)
    # the half-shifted grid never samples E = 0, so a J undefined there still works ...
    c1 = L.jn_methods(lambda E: np.where(E == 0.0, np.nan, E), 1.0, 1)
    assert c1 == pytest.approx(-0.5j, abs=1e-14)
    # ... while non-finite values elsewhere are rejected
    with pytest.raises(ValueError):
        L.jn_methods(lambda E: np.where(E > 0.5, np.nan, E), 1.0, 1)
    with pytest.raises(ValueError):
        L.jn_methods(lambda E: np.float64(1.0), 1.0, 1)  # not vectorized
    with pytest.raises(TypeError):
        L.jn_methods(lambda E: E + 0j, 1.0, 1)
