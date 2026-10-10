import math

import numpy as np
import pytest

from queer.finite_field.lz.propagation import (
    p_int_two,
    propagate_crossings,
    stokes_phase,
    stokes_phase_from_P,
)

INF = math.inf


def paper_p_int(P, k0, kappa1, kappa2, Phi):
    """Eq. 12 exactly as printed (written out here; queer.finite_field.literal is tested separately)."""
    return P * (
        1
        + (1 - 2 * P) * np.exp(-2 * k0 / kappa1)
        + 2 * (1 - P) * np.exp(-2 * k0 / (2 * kappa1)) * np.exp(-2 * k0 / kappa2) * np.cos(Phi)
    )


def mp_stokes(delta):
    mpmath = pytest.importorskip("mpmath")
    with mpmath.workdps(50):
        d = mpmath.mpf(delta)
        return float(mpmath.pi / 4 + d * (mpmath.log(d) - 1) + mpmath.im(mpmath.loggamma(1 - 1j * d)))


# ---- Stokes phase ------------------------------------------------------------------------------------
@pytest.mark.precision
def test_stokes_phase_reference_value():
    d = math.log(2) / (2 * math.pi)  # P = 1/2
    assert stokes_phase(d) == pytest.approx(0.495, abs=5e-4)
    assert stokes_phase(d) == pytest.approx(mp_stokes(d), abs=1e-14)
    assert stokes_phase_from_P(0.5) == pytest.approx(stokes_phase(d), abs=1e-15)


@pytest.mark.parametrize("delta", [1e-9, 1e-3, 0.05, 0.3, 1.0, 3.0, 7.99, 8.0, 8.01, 15.0, 1e3, 1e6])
@pytest.mark.precision
def test_stokes_phase_against_mpmath(delta):
    # both branches (direct loggamma below delta = 8, Stirling series above) agree with 50-digit mpmath
    assert stokes_phase(delta) == pytest.approx(mp_stokes(delta), abs=1e-14, rel=1e-12)


def test_stokes_phase_limits_and_monotonicity():
    assert stokes_phase(0.0) == pytest.approx(math.pi / 4, abs=1e-16)  # sudden limit
    assert stokes_phase(INF) == 0.0  # adiabatic limit
    assert stokes_phase(1e9) == pytest.approx(1 / 12e9, rel=1e-12)
    assert stokes_phase_from_P(1.0) == pytest.approx(math.pi / 4, abs=1e-16)
    assert stokes_phase_from_P(0.0) == 0.0
    d = np.geomspace(1e-6, 1e4, 400)
    phi = stokes_phase(d)
    assert phi.shape == d.shape and np.all(np.diff(phi) < 0) and np.all(phi > 0)
    with pytest.raises(ValueError):
        stokes_phase(-0.1)
    with pytest.raises(ValueError):
        stokes_phase_from_P(1.5)


# ---- reductions ----------------------------------------------------------------------------------------
def test_single_crossing_is_P(rng):
    P = rng.uniform(0, 1, (1, 7))
    for k1, k2 in ((INF, INF), (1e-3, 2e-3)):
        assert np.array_equal(propagate_crossings(P, None, None, k1, k2), P)
        out = propagate_crossings(
            P, np.zeros((0, 7)), np.zeros((0, 7)), k1, k2, phiS=rng.uniform(0, 1, (1, 7))
        )
        assert np.allclose(out, P, atol=1e-15)


def test_two_equal_crossings_reproduce_eq12(rng):
    n = 2000
    P = rng.uniform(0, 1, n)
    k0 = rng.uniform(1e-4, 1e-2, n)
    k1 = rng.uniform(1e-4, 1e-2, n)
    k2 = rng.uniform(1e-4, 1e-2, n)
    Phi = rng.uniform(-50, 50, n)
    out = propagate_crossings(np.stack([P, P]), (2 * k0)[None], Phi[None], k1, k2)
    assert out.shape == (2, n)
    assert np.max(np.abs(out[0] - P)) < 1e-15
    assert np.max(np.abs(out[1] - paper_p_int(P, k0, k1, k2, Phi))) < 1e-12
    assert np.max(np.abs(out[1] - p_int_two(P, P, 2 * k0, k1, k2, Phi))) < 1e-12
    # T1 = T2 = inf: 2P(1-P)(1 + cos Phi)
    coh = propagate_crossings(np.stack([P, P]), (2 * k0)[None], Phi[None], INF, INF)[1]
    assert np.max(np.abs(coh - 2 * P * (1 - P) * (1 + np.cos(Phi)))) < 1e-12


@pytest.mark.parametrize("coherence", ["paper", "bloch"])
def test_closed_form_unequal_P(rng, coherence):
    n = 500
    P1, P2 = rng.uniform(0, 1, (2, n))
    d = rng.uniform(0, 0.02, n)
    k1 = rng.uniform(2e-3, 2e-2, n)
    k2 = rng.uniform(1e-3, 2 * k1)  # bloch needs kappa2 <= 2 kappa1
    th = rng.uniform(-20, 20, n)
    out = propagate_crossings(np.stack([P1, P2]), d[None], th[None], k1, k2, coherence=coherence)[1]
    assert np.max(np.abs(out - p_int_two(P1, P2, d, k1, k2, th, coherence=coherence))) < 1e-12


def test_stokes_phases_add_to_theta(rng):
    P1, P2, phi1, phi2 = rng.uniform(0, 1, (4, 50))
    d, k1, k2, th = 0.01, 0.02, 0.03, rng.uniform(-5, 5, 50)
    out = propagate_crossings(np.stack([P1, P2]), [[d]], th[None], k1, k2, phiS=np.stack([phi1, phi2]))[1]
    assert np.allclose(out, p_int_two(P1, P2, d, k1, k2, th + phi1 + phi2), atol=1e-13, rtol=0)


def test_limits(rng):
    th = rng.uniform(-10, 10, 20)
    one, zero = np.ones(20), np.zeros(20)
    d = 0.01
    # adiabatic crossings: nothing is excited
    assert np.all(propagate_crossings(np.stack([zero, zero]), [[d]], th[None], 0.01, 0.01) == 0)
    # sudden crossings: v -> c -> v; only the relaxed part stays in v after the second crossing
    out = propagate_crossings(np.stack([one, one]), [[d]], th[None], 0.004, 0.01)
    assert np.allclose(out[0], 1) and np.allclose(out[1], 1 - math.exp(-d / 0.004), atol=1e-15)
    assert np.allclose(propagate_crossings(np.stack([one, one]), [[d]], th[None], INF, INF)[1], 0, atol=1e-15)
    P1, P2 = rng.uniform(0, 1, (2, 20))
    PP = np.stack([P1, P2])
    # full relaxation between the crossings: only the second crossing counts
    assert np.allclose(propagate_crossings(PP, [[d]], th[None], d / 800, INF)[1], P2, atol=1e-15)
    # pure dephasing only (T1 = inf, T2 -> 0): incoherent rate-equation result
    assert np.allclose(
        propagate_crossings(PP, [[d]], th[None], INF, d / 800)[1], P1 + P2 - 2 * P1 * P2, atol=1e-15
    )
    # no relaxation: coherent two-path interference
    coh = np.sqrt(P1 * P2 * (1 - P1) * (1 - P2))
    expect = P1 + P2 - 2 * P1 * P2 + 2 * coh * np.cos(th)
    assert np.allclose(propagate_crossings(PP, [[d]], th[None], INF, INF)[1], expect, atol=1e-14)


@pytest.mark.parametrize("coherence", ["paper", "bloch"])
def test_matches_explicit_density_matrix_products(rng, coherence):
    n = 5
    P = rng.uniform(0, 1, n)
    phi = rng.uniform(0, math.pi / 4, n)
    d = rng.uniform(0, 0.01, n - 1)
    th = rng.uniform(-10, 10, n - 1)
    k1, k2 = 0.008, 0.005
    gamma = 1 / (2 * k1) + 1 / k2 if coherence == "paper" else 1 / k2
    rho = np.array([[0, 0], [0, 1]], dtype=complex)  # (c, v) basis, start in v
    expect = []
    for j in range(n):
        q = math.sqrt(1 - P[j])
        S = np.array(
            [[q * np.exp(-1j * phi[j]), -math.sqrt(P[j])], [math.sqrt(P[j]), q * np.exp(1j * phi[j])]]
        )
        rho = S @ rho @ S.conj().T
        expect.append(rho[0, 0].real)
        if j < n - 1:
            cc = rho[0, 0].real * math.exp(-d[j] / k1)
            cv = rho[0, 1] * np.exp(-gamma * d[j] - 1j * th[j])
            rho = np.array([[cc, cv], [np.conj(cv), 1 - cc]])
    out = propagate_crossings(P, d, th, k1, k2, phiS=phi, coherence=coherence)
    assert np.allclose(out, expect, atol=1e-14, rtol=0)


def test_transparent_third_crossing_only_relaxes(rng):
    P1, P2 = rng.uniform(0, 1, (2, 30))
    d = rng.uniform(0, 0.01, (2, 30))
    th = rng.uniform(-10, 10, (2, 30))
    k1, k2 = 0.006, 0.009
    phi = rng.uniform(0, 1, (3, 30))  # the Stokes phase of a P = 0 crossing only rotates rho_cv
    two = propagate_crossings(np.stack([P1, P2]), d[:1], th[:1], k1, k2, phiS=phi[:2])
    three = propagate_crossings(np.stack([P1, P2, 0 * P1]), d, th, k1, k2, phiS=phi)
    assert np.allclose(three[:2], two, atol=1e-15)
    assert np.allclose(three[2], two[1] * np.exp(-d[1] / k1), atol=1e-15)


@pytest.mark.parametrize("coherence", ["paper", "bloch"])
def test_positivity(rng, coherence):
    n, m = 8, 4000
    P = rng.uniform(0, 1, (n, m))
    P[:, :200] = rng.choice([0.0, 1.0], (n, 200))  # include the extremes
    d = rng.exponential(0.005, (n - 1, m))
    th = rng.uniform(-30, 30, (n - 1, m))
    k1 = rng.choice([INF, 1e-3, 1e-2], m)
    k2 = np.where(np.isinf(k1), rng.uniform(1e-3, 1e-1, m), k1 * rng.uniform(0.05, 2.0, m))
    phi = rng.uniform(0, math.pi / 4, (n, m))
    rcc, rcv = propagate_crossings(P, d, th, k1, k2, phiS=phi, coherence=coherence, return_coherence=True)
    assert np.all(rcc >= -1e-15) and np.all(rcc <= 1 + 1e-15)
    assert np.all(np.abs(rcv) ** 2 <= rcc * (1 - rcc) + 1e-14)


def test_bloch_vs_paper(rng):
    P = np.stack(rng.uniform(0.1, 0.9, (2, 40)))
    th = rng.uniform(-10, 10, (1, 40))
    d = 0.01
    # without T1 the two coherence models coincide
    a = propagate_crossings(P, [[d]], th, INF, 0.02, coherence="paper")
    b = propagate_crossings(P, [[d]], th, INF, 0.02, coherence="bloch")
    assert np.allclose(a, b, atol=1e-15)
    # with T1 the paper model has the extra e^{-d/(2 kappa1)} coherence decay
    a = propagate_crossings(P, [[d]], th, 0.01, 0.02, coherence="paper")[1]
    b = propagate_crossings(P, [[d]], th, 0.01, 0.02, coherence="bloch")[1]
    incoh = P[1] + P[0] * (1 - 2 * P[1]) * math.exp(-d / 0.01)
    assert np.allclose(a - incoh, (b - incoh) * math.exp(-d / (2 * 0.01)), atol=1e-14)
    with pytest.raises(ValueError):  # T2 > 2 T1 is not a positive map
        propagate_crossings(P, [[d]], th, 0.01, 0.03, coherence="bloch")
    with pytest.raises(ValueError):
        propagate_crossings(P, [[d]], th, 0.01, 0.01, coherence="other")


def test_broadcasting_and_validation():
    P = np.full((2, 4), 0.3)
    out = propagate_crossings(P, [0.01], [[0.0, 1.0, 2.0, 3.0]], np.array([[1e-2], [2e-2], [INF]]), INF)
    assert out.shape == (2, 3, 4)
    assert np.allclose(out[1, 2], 2 * 0.3 * 0.7 * (1 + np.cos([0.0, 1.0, 2.0, 3.0])), atol=1e-15)
    with pytest.raises(ValueError):
        propagate_crossings(P, [0.01, 0.02], [0.0, 0.0], INF, INF)  # wrong number of segments
    with pytest.raises(ValueError):
        propagate_crossings(np.array([[0.3], [1.2]]), [0.01], [0.0], INF, INF)
    with pytest.raises(ValueError):
        propagate_crossings(P, [0.01], [0.0], 0.0, INF)
    with pytest.raises(ValueError):
        propagate_crossings(0.3, None, None, INF, INF)


def test_review_edge_cases():
    import math

    from queer.finite_field.lz.propagation import p_int_two, propagate_crossings, stokes_phase_from_P

    with pytest.raises(ValueError):
        p_int_two(0.5, 0.1, d=-0.01, kappa1=0.002, kappa2=0.004, theta=0.0)
    assert np.isfinite(p_int_two(0.3, 0.4, math.inf, math.inf, math.inf, 0.0))
    rho = propagate_crossings(np.array([0.3, 0.4]), np.array([math.inf]), np.array([1.0]), math.inf, math.inf)
    assert np.all(np.isfinite(rho))
    assert stokes_phase_from_P(1 + 1e-13) == pytest.approx(stokes_phase_from_P(1.0))
