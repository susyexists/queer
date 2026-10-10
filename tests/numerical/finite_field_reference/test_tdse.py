import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

import queer.finite_field.tdse.houston as ho
from queer.finite_field.geometry import band_geometry
from queer.finite_field.lz.propagation import propagate_crossings, stokes_phase_from_P
from queer.finite_field.models import FourBandModel, FourBandParams
from queer.finite_field.models.toy import LinearLZ, ParabolicDoubleCrossing
from queer.finite_field.tdse.houston import houston_line, lindblad_two_level
from queer.finite_field.units import critical_field, kappa

LZ = LinearLZ(0.15, 0.3e-3)
LZ_EC = 1e10 * math.pi * LZ.delta**2 / LZ.alpha  # Eq. 8, exact for this model
LZ_E = np.array([5e3, 2e4, 8e4])

_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)


class Mirrored:
    """H'(kx, ky) = H(-kx, ky): a forward pass through H' is a backward pass through H."""

    def __init__(self, model):
        self.model, self.n_orb = model, model.n_orb

    def hamiltonian(self, kx, ky):
        return self.model.hamiltonian(-np.asarray(kx, dtype=float), ky)

    def dhamiltonian(self, kx, ky):
        dHx, dHy = self.model.dhamiltonian(-np.asarray(kx, dtype=float), ky)
        return -dHx, dHy

    def band_pair(self):
        return self.model.band_pair()


class Curved:
    """H = alpha kx sz + delta sx + beta kx^2 sy: a twisted crossing, P(E) != P(-E)."""

    n_orb = 2

    def __init__(self, alpha=0.15, delta=0.3e-3, beta=3.0):
        self.alpha, self.delta, self.beta = alpha, delta, beta

    def hamiltonian(self, kx, ky):
        kx = np.asarray(kx, dtype=float)[..., None, None]
        return self.alpha * kx * _SZ + self.delta * _SX + self.beta * kx * kx * _SY

    def dhamiltonian(self, kx, ky):
        kx = np.asarray(kx, dtype=float)[..., None, None]
        return self.alpha * _SZ + 2.0 * self.beta * kx * _SY, 0.0 * kx * _SX

    def band_pair(self):
        return 0, 1


@pytest.fixture(scope="module")
def lz_ref():
    return houston_line(LZ, 0.0, LZ_E, -0.05, 0.05)


# ---- fast: linear Landau-Zener -------------------------------------------------------------------------
def test_linear_lz_reproduces_exact_probabilities(lz_ref):
    r = lz_ref
    assert np.allclose(r.P, [0.02305, 0.38966, 0.79008], atol=2e-4, rtol=0)
    assert np.allclose(r.P, np.exp(-LZ_EC / LZ_E), atol=2e-5, rtol=0)
    assert np.array_equal(r.n_exc, r.P)  # two bands: v -> c is the only excitation
    assert r.T.shape == (3, 2, 2) and (r.v, r.c) == (0, 1)
    assert np.allclose(r.T.sum(axis=1), 1, atol=1e-12) and np.allclose(r.T.sum(axis=2), 1, atol=1e-12)
    assert r.info["unitarity"] < 1e-11
    assert (
        r.info["window"] == (-0.05, 0.05) and r.info["basis"] == "superadiabatic" and r.info["steps"] > 1000
    )


def test_window_convergence_and_superadiabatic_sign(lz_ref, monkeypatch):
    wide = houston_line(LZ, 0.0, LZ_E, -0.1, 0.08)
    assert np.max(np.abs(wide.P - lz_ref.P)) < 2e-5
    # in a tight window the first-order dressing removes most of the adiabatic-basis error ...
    exact = np.exp(-LZ_EC / LZ_E)
    err_sa = np.abs(houston_line(LZ, 0.0, LZ_E, -0.02, 0.02).P - exact)
    err_ad = np.abs(houston_line(LZ, 0.0, LZ_E, -0.02, 0.02, basis="adiabatic").P - exact)
    assert err_ad[2] > 10 * err_sa[2]
    # ... and only with the documented sign; the opposite sign doubles the error
    monkeypatch.setattr(ho, "SUPERADIABATIC_SIGN", +1)
    err_wrong = np.abs(houston_line(LZ, 0.0, LZ_E, -0.02, 0.02).P - exact)
    assert err_wrong[2] > 1.5 * err_ad[2]


def test_negative_field_by_symmetry(lz_ref):
    neg = houston_line(LZ, 0.0, -LZ_E, -0.05, 0.05)
    assert np.allclose(neg.P, lz_ref.P, atol=1e-10, rtol=0)
    mixed = houston_line(LZ, 0.0, [2e4, -2e4], -0.05, 0.05)
    assert mixed.P[0] == pytest.approx(mixed.P[1], abs=1e-10)
    assert mixed.P[0] == pytest.approx(lz_ref.P[1], abs=1e-10)


def test_reversed_traversal_equals_mirrored_forward_pass():
    # E < 0 runs the CF4 steps backwards; on asymmetric models it must equal the forward pass of H(-s)
    c = Curved()
    a = houston_line(c, 0.0, [-2e4, 2e4], -0.03, 0.05)
    b = houston_line(Mirrored(c), 0.0, 2e4, -0.05, 0.03)
    assert np.max(np.abs(a.T[0] - b.T[0])) < 1e-10
    assert a.P[0] - a.P[1] > 0.05  # twisted crossing: genuinely nonreciprocal
    rho_a = lindblad_two_level(c, 0.0, [-2e4, 2e4], -0.03, 0.05, 10e-12, 20e-12)
    rho_b = lindblad_two_level(Mirrored(c), 0.0, 2e4, -0.05, 0.03, 10e-12, 20e-12)
    assert rho_a[0] == pytest.approx(rho_b[0], abs=1e-10)
    m = FourBandModel(FourBandParams().at(0.5), warping=200.0)  # 3m' emulator, off the mirror line
    a = houston_line(m, 0.002, -1.5e4, -0.006, 0.01)
    b = houston_line(Mirrored(m), 0.002, 1.5e4, -0.01, 0.006)
    assert np.max(np.abs(a.T - b.T)) < 1e-10


def test_step_size_convergence(lz_ref):
    coarse = houston_line(LZ, 0.0, LZ_E, -0.05, 0.05, dphi=0.1)
    assert coarse.info["steps"] < lz_ref.info["steps"] / 4
    assert np.max(np.abs(coarse.P - lz_ref.P)) < 1e-7


def test_identity_terms_and_field_angle_drop_out():
    p = FourBandParams().at(0.5)
    full = houston_line(FourBandModel(p), 0.001, [1.5e4, -1.5e4], -0.01, 0.01)
    bare = houston_line(FourBandModel(p).traceless(), 0.001, [1.5e4, -1.5e4], -0.01, 0.01)
    assert np.max(np.abs(full.T - bare.T)) < 1e-9
    # the traceless model is isotropic: the same line along another direction gives the same dynamics
    rot = houston_line(FourBandModel(p).traceless(), 0.001, [1.5e4, -1.5e4], -0.01, 0.01, angle=0.7)
    assert np.max(np.abs(rot.T - bare.T)) < 1e-9
    assert 0.1 < bare.P[0] < 0.9 and bare.n_exc[0] >= bare.P[0]


def test_lindblad_coherent_limit_equals_tdse(lz_ref):
    rho = lindblad_two_level(LZ, 0.0, LZ_E, -0.05, 0.05, math.inf, math.inf)
    assert np.allclose(rho, lz_ref.P, atol=1e-10, rtol=0)


@pytest.mark.parametrize("coherence", ["paper", "bloch"])
def test_lindblad_population_decay_rate(coherence):
    # after the crossing rho_cc decays as exp(-ds/kappa1) with kappa1 = e|E|T1/hbar
    E, T1 = np.array([2e4, -2e4]), 20e-12
    near = lindblad_two_level(LZ, 0.0, E, -0.05, 0.05, T1, T1, coherence=coherence)
    far = lindblad_two_level(LZ, 0.0, E[:1], -0.05, 0.08, T1, T1, coherence=coherence)
    assert far[0] / near[0] == pytest.approx(math.exp(-0.03 / kappa(2e4, T1)), rel=1e-3)
    assert near[1] == pytest.approx(near[0], rel=1e-9)  # symmetric model, reversed traversal


def test_validation():
    with pytest.raises(ValueError):
        houston_line(LZ, 0.0, 1e4, 0.05, -0.05)
    with pytest.raises(ValueError):
        houston_line(LZ, 0.0, [1e4, 0.0], -0.05, 0.05)
    with pytest.raises(ValueError):
        houston_line(LZ, 0.0, 1e4, -0.05, 0.05, basis="diabatic")
    with pytest.raises(ValueError):
        lindblad_two_level(FourBandModel(), 0.0, 1e4, -0.01, 0.01, 1e-12, 1e-12)
    with pytest.raises(ValueError):
        lindblad_two_level(LZ, 0.0, 1e4, -0.05, 0.05, 1e-12, 3e-12, coherence="bloch")


# ---- slow: cross-validation of Eqs. 7, 12 and the propagation model ------------------------------------
def _parabolic_setup():
    m = ParabolicDoubleCrossing()
    k0 = m.k0
    g = band_geometry(m, k0, 0.0)
    Ec = float(critical_field(g.eps_g, g.g_cv[0]))  # Eq. 8 at the gap minimum

    def gap(s):
        return 2.0 * math.sqrt((m.a * s * s - m.b) ** 2 + m.delta**2)

    gap_integral = quad(gap, -k0, k0, epsabs=0, epsrel=1e-13, limit=200)[0]  # eV/A
    return m, k0, Ec, gap_integral


@pytest.mark.slow
def test_fourband_vl0_lz_probability():
    m = FourBandModel(FourBandParams().at(0.0)).traceless()
    Ec = FourBandParams().at(0.0).ec_k0
    assert Ec == pytest.approx(30450.16, rel=1e-6)
    P = houston_line(m, 0.0, Ec, -0.019, 0.019).P[0]  # T[c=2, v=1]: bands 2 -> 3 (1-based)
    assert P == pytest.approx(0.3659, abs=3e-3)
    assert math.exp(-1) - P > 1e-3  # Eq. 7 (0.3679) misses the k^2 dispersion of the mass
    assert houston_line(m, 0.0, Ec, -0.015, 0.015).P[0] == pytest.approx(P, abs=5e-4)


@pytest.mark.slow
def test_double_crossing_stokes_and_geometric_phase():
    """TDSE = 2P(1-P)(1 + cos(Phi + 2 phi_S + gamma)), gamma = pi: theta = Phi + gamma, phiS = +phi_S."""
    m, k0, Ec, gap_integral = _parabolic_setup()
    E = np.geomspace(300.0, 5000.0, 80)  # 2 k0 >= 8 k_LZ: well separated crossings
    y = houston_line(m, 0.0, E, -0.012, 0.012, dphi=0.05).P
    P = np.exp(-Ec / E)
    Phi = gap_integral / (1e-10 * E)
    phi_s = stokes_phase_from_P(P)

    def rms(form):
        return float(np.sqrt(np.mean((y - form) ** 2)))

    fit = rms(2 * P * (1 - P) * (1 + np.cos(Phi + 2 * phi_s + math.pi)))
    assert fit < 0.02
    assert rms(2 * P * (1 - P) * (1 + np.cos(Phi))) > max(0.2, 10 * fit)  # Eq. 12 as printed
    assert rms(2 * P * (1 - P) * (1 + np.cos(Phi - 2 * phi_s + math.pi))) > 0.2  # other Stokes sign
    assert rms(2 * P * (1 - P) * (1 + np.cos(Phi + 2 * phi_s))) > 0.2  # no geometric phase
    # the same convention in the propagation model
    model = propagate_crossings(
        np.stack([P, P]),
        np.full((1, E.size), 2 * k0),
        (Phi + math.pi)[None],
        math.inf,
        math.inf,
        phiS=np.stack([phi_s, phi_s]),
    )[1]
    assert rms(model) == pytest.approx(fit, abs=1e-12)


@pytest.mark.slow
def test_fourband_interference_period_matches_lambda():
    p = FourBandParams().at(1.8)
    k0 = minimize_scalar(
        lambda k: float(p.gap_closed_form(k)), bounds=(1e-4, 0.01), method="bounded", options={"xatol": 1e-12}
    ).x
    lam = quad(lambda k: float(p.gap_closed_form(k)), -k0, k0, epsrel=1e-12)[0] * 1e10 / (2 * math.pi)
    assert lam == pytest.approx(13145.0, abs=2.0)  # V/m, between the gap minima
    u = np.linspace(1 / 12000.0, 1 / 1250.0, 140)  # 1/E, about 9 periods
    y = houston_line(FourBandModel(p).traceless(), 0.0, 1 / u, -0.015, 0.015, dphi=0.05).P
    # equal crossings (ky = 0 mirror): y = 4P(1-P) cos^2(psi/2) vanishes once per period
    i = np.flatnonzero((y[1:-1] < y[:-2]) & (y[1:-1] <= y[2:])) + 1
    h = u[1] - u[0]
    curv = y[i - 1] - 2 * y[i] + y[i + 1]
    zeros = u[i] + 0.5 * h * (y[i - 1] - y[i + 1]) / curv  # vertex of the 3-point parabola
    vertex = y[i] - (y[i - 1] - y[i + 1]) ** 2 / (8 * curv)
    assert zeros.size >= 8 and np.all(np.abs(vertex) < 0.05 * (y[i - 1] + y[i + 1]))  # complete destruction
    period = np.polyfit(np.arange(zeros.size), zeros, 1)[0]
    assert 1.0 / period == pytest.approx(lam, rel=0.03)  # TDSE: +0.5% (branch points lie beyond the minima)


@pytest.mark.slow
def test_lindblad_reproduces_propagation_model():
    m, k0, Ec, gap_integral = _parabolic_setup()
    E = np.geomspace(500.0, 5000.0, 40)
    T, S = 30e-12, 0.012  # d/kappa1 = 0.4 ... 4.4 between the crossings
    rho = lindblad_two_level(m, 0.0, E, -S, S, T, T, dphi=0.05)
    P = np.exp(-Ec / E)
    phi_s = stokes_phase_from_P(P)
    k = kappa(E, T)
    # crossings at -k0, +k0, then a transparent "crossing" at the window end (relaxation only)
    Pc = np.stack([P, P, 0 * P])
    d = np.stack([np.full_like(E, 2 * k0), np.full_like(E, S - k0)])
    theta = np.stack([gap_integral / (1e-10 * E) + math.pi, 0 * E])
    model = propagate_crossings(Pc, d, theta, k, k, phiS=np.stack([phi_s, phi_s, 0 * P]))[-1]
    assert np.max(np.abs(rho - model)) < 0.03
    printed = propagate_crossings(Pc, d, theta - math.pi, k, k)[-1]  # no Stokes or geometric phase
    assert np.max(np.abs(rho - printed)) > 3 * np.max(np.abs(rho - model))
