"""Harmonic projector of Eq. 13 (queer.finite_field.harmonics): exact cases, power-law closed forms, convergence."""

import math

import numpy as np
import pytest
from scipy import special as sp

from queer.finite_field import literal as L
from queer.finite_field.harmonics import (
    C_F,
    HarmonicResult,
    _gauss_legendre_01,
    harmonics,
    local_exponent,
    power_law_bN,
    power_law_cf,
    theta_nodes,
)

E0S = np.array([0.3, 1.0, 2.5, 7.5e3])


def power_law(p):
    return lambda E: np.sign(E) * np.abs(E) ** p


def activated(Ec, Lam):
    """Oscillatory activated J = sgn(E) exp(-Ec/|E|) (1 + cos(2 pi Lam/|E|)), J(0) = 0."""

    def J(E):
        E = np.asarray(E, dtype=float)
        a = np.abs(E)
        out = np.zeros_like(E)
        m = a > 0
        out[m] = np.sign(E[m]) * np.exp(-Ec / a[m]) * (1.0 + np.cos(2.0 * math.pi * Lam / a[m]))
        return out

    return J


def midpoint_reference(J, E0, n_max, M=2**16, chunk=2**14):
    """Uniform midpoint rule over the full period [0, 2 pi): b_N, a_N = (2/M) sum_k J(E0 sin th_k) (sin, cos)(N th_k)."""
    N = np.arange(n_max + 1)
    b = np.zeros((E0.size, N.size))
    a = np.zeros_like(b)
    for start in range(0, M, chunk):
        th = (np.arange(start, min(start + chunk, M)) + 0.5) * 2.0 * math.pi / M
        f = J((E0[:, None] * np.sin(th)).ravel()).reshape(E0.size, th.size)
        b += f @ np.sin(np.outer(th, N))
        a += f @ np.cos(np.outer(th, N))
    return 2.0 * b / M, 2.0 * a / M


def zero_except(res_arr, keep):
    """Copy of a coefficient array with the listed columns zeroed (for 'all others vanish' checks)."""
    out = np.array(res_arr, copy=True)
    out[:, keep] = 0.0
    return out


# ----------------------------------------------------------------------------- exact polynomial cases


def test_linear_J_has_only_b1():
    res = harmonics(lambda E: E, E0S)
    assert isinstance(res, HarmonicResult) and res.converged
    assert res.b.shape == res.a.shape == res.c.shape == (E0S.size, 42) and res.c0.shape == (E0S.size,)
    assert np.array_equal(res.N, np.arange(42))
    assert np.allclose(res.b[:, 1], E0S, rtol=1e-14)
    assert np.max(np.abs(zero_except(res.b, [1]))) < 1e-14 * E0S.max()
    assert np.all(res.a == 0.0) and np.all(res.c0 == 0.0)  # J_odd exactly: J(E) + J(-E) = 0


def test_cubic_J():
    res = harmonics(lambda E: E**3, E0S)  # sin^3 = (3 sin - sin 3 theta)/4
    assert np.allclose(res.b[:, 1], 0.75 * E0S**3, rtol=1e-14)
    assert np.allclose(res.b[:, 3], -0.25 * E0S**3, rtol=1e-14)
    assert np.all(np.abs(zero_except(res.b, [1, 3])) < 1e-14 * E0S[:, None] ** 3)
    assert np.all(np.abs(res.a) < 1e-14 * E0S[:, None] ** 3)  # x**3 is odd only to rounding (pow)


def test_quadratic_J_is_dc_plus_second_harmonic():
    res = harmonics(lambda E: E**2, E0S)  # sin^2 = (1 - cos 2 theta)/2
    assert np.allclose(res.c0, E0S**2 / 2, rtol=1e-14)
    assert np.allclose(res.a[:, 0], E0S**2, rtol=1e-14)  # standard convention a_0 = 2 c0
    assert np.allclose(res.a[:, 2], -(E0S**2) / 2, rtol=1e-14)
    assert np.all(np.abs(zero_except(res.a, [0, 2])) < 1e-14 * E0S[:, None] ** 2)
    assert np.all(res.b == 0.0)
    assert np.allclose(res.c[:, 2], -(E0S**2) / 4, rtol=1e-14) and np.allclose(res.c[:, 0], res.c0, rtol=0)


def test_mixed_J():
    res = harmonics(lambda E: E + E**2, E0S)
    assert np.allclose(res.b[:, 1], E0S, rtol=1e-14)
    assert np.allclose(res.c0, E0S**2 / 2, rtol=1e-14)
    assert np.allclose(res.a[:, 2], -(E0S**2) / 2, rtol=1e-14)
    assert np.allclose(res.c[:, 1], -0.5j * E0S, rtol=1e-14)
    assert np.allclose(res.c[:, 2], -(E0S**2) / 4, rtol=1e-14)
    scale = (E0S + E0S**2)[:, None]
    assert np.all(np.abs(zero_except(res.b, [1])) < 1e-14 * scale)
    assert np.all(np.abs(zero_except(res.a, [0, 2])) < 1e-14 * scale)


def test_parity_selection_is_structural():
    J = lambda E: np.exp(0.7 * E) * np.cos(1.3 * E)  # noqa: E731  (neither odd nor even)
    res = harmonics(J, [0.5, 1.5, 3.0])
    assert np.all(res.b[:, 0::2] == 0.0) and np.all(res.a[:, 1::2] == 0.0)
    assert np.max(np.abs(res.b[:, 1::2])) > 0.1 and np.max(np.abs(res.a[:, 2::2])) > 0.1
    assert np.allclose(res.c, 0.5 * (res.a - 1j * res.b), rtol=0, atol=0)


def test_negative_amplitude_flips_odd_harmonics():
    J = lambda E: np.tanh(E) + 0.4 * E**2 / (1 + E**2)  # noqa: E731
    E0 = np.array([0.4, 1.3, 2.2])
    pos, neg = harmonics(J, E0), harmonics(J, -E0)
    assert np.allclose(neg.b, -pos.b, rtol=1e-13, atol=1e-15)
    assert np.allclose(neg.a, pos.a, rtol=1e-13, atol=1e-15)


# ----------------------------------------------------------------------------- power-law closed forms


B_REF_15 = {1: 0.9153117, 3: -0.1017013, 5: -0.02346953, 7: -0.009663925, 21: -5.951728e-4, 41: -1.113364e-4}


def test_power_law_reference_values():
    N = np.array(list(B_REF_15))
    ref = np.array(list(B_REF_15.values()))
    for form in ("ratio", "beta"):
        assert np.allclose(power_law_bN(N, 1.5, form=form), ref, rtol=6e-7, atol=0)
    # plan form for p = 3/2: -(3/(8 sqrt(pi))) Gamma(N/2 - 3/4)/Gamma(N/2 + 7/4)
    Nodd = np.arange(1, 42, 2)
    plan = -3.0 / (8.0 * math.sqrt(math.pi)) * sp.gamma(Nodd / 2 - 0.75) / sp.gamma(Nodd / 2 + 1.75)
    assert np.allclose(power_law_bN(Nodd, 1.5), plan, rtol=1e-13, atol=0)


@pytest.mark.parametrize("p", [0.5, 1.5, 2.5])
def test_closed_forms_agree(p):
    N = np.arange(0, 42)
    ratio, beta = power_law_bN(N, p, form="ratio"), power_law_bN(N, p, form="beta")
    assert np.all(ratio[0::2] == 0.0) and np.all(beta[0::2] == 0.0)
    assert np.allclose(ratio[1::2], beta[1::2], rtol=1e-13, atol=0)
    assert np.all(ratio[1::2] != 0.0)
    big = np.array([101, 341, 1001, 10001])  # log-gamma branch beyond the range of Gamma
    assert np.allclose(power_law_bN(big, p, "ratio"), power_law_bN(big, p, "beta"), rtol=1e-9, atol=0)


def test_closed_forms_at_gamma_poles():
    """Odd-integer p: finitely many harmonics (sin^p expands in sin N theta, N <= p)."""
    N = np.arange(0, 12)
    exact = {1.0: {1: 1.0}, 3.0: {1: 0.75, 3: -0.25}, 5.0: {1: 0.625, 3: -0.3125, 5: 0.0625}}
    for p, coeffs in exact.items():
        expected = np.zeros(N.size)
        for n, v in coeffs.items():
            expected[n] = v
        for form in ("ratio", "beta"):
            assert np.allclose(power_law_bN(N, p, form=form), expected, rtol=1e-14, atol=1e-16)
    # p = 0 (sgn E): b_N = 4/(pi N) for odd N
    assert np.allclose(
        power_law_bN(np.arange(1, 42, 2), 0.0), 4.0 / (math.pi * np.arange(1, 42, 2)), rtol=1e-14
    )


def test_closed_form_validation():
    with pytest.raises(ValueError):
        power_law_bN(3, -1.0)
    with pytest.raises(ValueError):
        power_law_bN(2.5, 1.5)
    with pytest.raises(ValueError):
        power_law_bN(-1, 1.5)
    with pytest.raises(ValueError):
        power_law_bN(3, 1.5, form="other")


def test_power_law_projection_matches_closed_form():
    J = power_law(1.5)
    N = np.arange(42)
    ref = power_law_bN(N, 1.5)
    E0 = np.array([1.0, 2.0, 1e4])
    fixed = harmonics(J, E0, n_theta=256, adaptive=False)
    adaptive = harmonics(J, E0)
    for res in (fixed, adaptive):
        assert np.allclose(res.b[:, 1::2], E0[:, None] ** 1.5 * ref[1::2], rtol=1e-9, atol=0)
        assert np.all(res.b[:, 0::2] == 0.0) and np.all(res.a == 0.0)
    assert not fixed.converged and fixed.n_theta == 256 and fixed.info["n_calls"] == 1
    assert adaptive.converged and adaptive.n_theta >= 768


def test_cf_large_n_limit():
    assert C_F == pytest.approx(1.196827, abs=5e-7) and C_F == 3.0 / math.sqrt(2.0 * math.pi)
    assert power_law_cf(1.5) == pytest.approx(C_F, rel=1e-15)
    assert 41**2.5 * abs(power_law_bN(41, 1.5)) == pytest.approx(1.19839, abs=5e-6)
    N = np.array([41, 101, 1001, 10001])
    dev = N**2.5 * np.abs(power_law_bN(N, 1.5)) / C_F - 1.0
    assert np.all(dev > 0) and np.all(np.diff(dev) < 0) and dev[-1] < 1e-7  # O(1/N^2) approach from above
    for p in (0.5, 2.5):
        assert 10001 ** (p + 1) * abs(power_law_bN(10001, p)) == pytest.approx(power_law_cf(p), rel=1e-6)
    # the projected harmonics reach the same law, also at high orders
    res = harmonics(power_law(1.5), 1.0, n_max=401)
    assert 41**2.5 * abs(res.b[0, 41]) == pytest.approx(1.19839, abs=5e-6)
    assert np.allclose(res.b[0], power_law_bN(res.N, 1.5), rtol=1e-6, atol=1e-13)
    assert 401**2.5 * abs(res.b[0, 401]) == pytest.approx(C_F, rel=2e-5)


# ----------------------------------------------------------------------------- Methods form and the literal cross-check


def test_methods_form_of_odd_J():
    J = lambda E: np.sign(E) * np.abs(E) ** 1.5 + np.tanh(3.0 * E)  # noqa: E731  (odd)
    res = harmonics(J, [0.2, 1.0, 3.0])
    assert np.all(res.c.real == 0.0)
    assert np.array_equal(res.c.imag, -0.5 * res.b)


def nontrivial_J(E):
    """Smooth, neither odd nor even: activated odd part plus an activated even part and a linear term."""
    E = np.asarray(E, dtype=float)
    a = np.abs(E)
    out = 0.05 * E
    m = a > 0
    act = np.exp(-1.0 / a[m])
    out[m] += np.sign(E[m]) * act * (1.0 + 0.5 * np.cos(0.8 / a[m])) + 0.3 * act**2
    return out


def test_agreement_with_literal_forms():
    E0 = np.array([0.4, 1.0, 2.0, 5.0])
    N = np.arange(42)
    res = harmonics(nontrivial_J, E0)
    assert res.converged
    c_lit = L.jn_methods(nontrivial_J, E0, N, n_theta=2**14)
    b_lit = L.jn_main_text(nontrivial_J, E0, N, n_theta=2**14)
    scale = np.max(np.abs(res.c))
    assert np.max(np.abs(res.c - c_lit)) < 1e-11 * scale
    assert np.max(np.abs(res.b - b_lit)) < 2e-11 * scale
    assert np.max(np.abs(res.a[:, 2::2])) > 1e-2 * scale  # the even part is really exercised


def test_oscillatory_activated_convergence():
    """Ec = 16.1 kV/m, Lambda = 13.1 kV/m, E0 in [3e3, 5e4] V/m vs a 2^16-point uniform-midpoint reference."""
    J = activated(16.1e3, 13.1e3)
    E0 = np.geomspace(3e3, 5e4, 24)
    res = harmonics(J, E0)
    b_ref, a_ref = midpoint_reference(J, E0, 41)
    assert res.converged and res.info["err_rel"] <= 1e-8
    assert np.max(np.abs(res.b - b_ref)) < 1e-6 * np.max(np.abs(b_ref))
    assert np.max(np.abs(res.a - a_ref)) < 1e-6 * np.max(np.abs(b_ref))


def test_weakly_activated_needs_doublings():
    """Ec << Lambda: fast oscillations survive near E = 0; adaptive doubling must resolve them."""
    J = activated(4.0e3, 13.1e3)
    E0 = np.geomspace(3e3, 5e4, 12)
    res = harmonics(J, E0)
    b_ref, _ = midpoint_reference(J, E0, 41, M=2**17)
    assert res.converged and len(res.info["n_history"]) >= 3
    assert res.info["n_history"] == [384 * 2**i for i in range(len(res.info["n_history"]))]
    assert np.max(np.abs(res.b - b_ref)) < 1e-6 * np.max(np.abs(b_ref))
    coarse = harmonics(J, E0, adaptive=False)
    assert np.max(np.abs(coarse.b - b_ref)) > 1e-6 * np.max(np.abs(b_ref))  # 384 nodes alone are not enough


@pytest.mark.slow
@pytest.mark.parametrize("Ec", [500.0, 1000.0])
def test_deep_adaptive_convergence(Ec):
    """Weakly activated cases that need 6144-12288 nodes, vs a 2^20-point midpoint reference."""
    J = activated(Ec, 13.1e3)
    E0 = np.geomspace(3e3, 5e4, 8)
    res = harmonics(J, E0)
    b_ref, a_ref = midpoint_reference(J, E0, 41, M=2**20, chunk=2**15)
    assert res.converged and res.n_theta >= 6144
    assert np.max(np.abs(res.b - b_ref)) < 1e-7 * np.max(np.abs(b_ref))
    assert np.max(np.abs(res.a - a_ref)) < 1e-7 * np.max(np.abs(b_ref))


def test_n_cap_reports_non_convergence():
    res = harmonics(activated(200.0, 13.1e3), np.geomspace(3e3, 5e4, 6), n_cap=1536)
    assert not res.converged and res.n_theta == 1536
    assert res.info["n_history"] == [384, 768, 1536] and res.info["err_rel"] > 1e-8
    capped = harmonics(lambda E: E**3, 1.0, n_theta=500, n_cap=700)  # the last step clamps to the cap
    assert capped.info["n_history"] == [500, 700] and capped.converged and capped.n_theta == 700
    assert not harmonics(lambda E: E**3, 1.0, n_theta=500, n_cap=400).converged  # no room to double


# ----------------------------------------------------------------------------- evaluation contract


class Spy:
    def __init__(self, J):
        self.J, self.calls = J, []

    def __call__(self, E):
        assert isinstance(E, np.ndarray) and E.ndim == 1
        self.calls.append(E.copy())
        return self.J(E)


def test_one_vectorized_call_per_node_set():
    spy = Spy(lambda E: E**3 + 0.1 * E**2)
    E0 = np.array([0.5, 1.0, 2.0])
    res = harmonics(spy, E0, n_theta=128, tol=1e-12)
    assert len(spy.calls) == res.info["n_calls"] == len(res.info["n_history"]) >= 2
    for E, n in zip(spy.calls, res.info["n_history"]):
        assert E.size == 2 * E0.size * n
        assert np.allclose(np.sort(E[E > 0]), np.sort(-E[E < 0]))  # explicit +E and -E evaluations
    assert res.info["n_evals"] == sum(c.size for c in spy.calls)


def test_odd_only_mode():
    J = power_law(1.5)
    spy = Spy(J)
    E0 = np.array([0.5, 2.0])
    odd = harmonics(spy, E0, odd_only=True)
    full = harmonics(J, E0)
    assert all(np.all(E > 0) for E in spy.calls)  # J(+E) only
    assert all(E.size == E0.size * n for E, n in zip(spy.calls, odd.info["n_history"]))
    assert np.allclose(odd.b, full.b, rtol=1e-14, atol=1e-16)
    assert np.all(odd.a == 0.0) and np.all(odd.c0 == 0.0) and odd.info["odd_only"]


def test_input_validation():
    with pytest.raises(ValueError):
        harmonics(np.sin, np.ones((2, 2)))
    with pytest.raises(ValueError):
        harmonics(np.sin, [])
    with pytest.raises(ValueError):
        harmonics(lambda E: np.where(np.abs(E) < 0.1, np.nan, E), 1.0)
    with pytest.raises(TypeError):
        harmonics(lambda E: E + 0j, 1.0)
    res = harmonics(lambda E: np.full_like(E, 2.0), [1.0, 3.0], n_max=4)  # constant J: DC only
    assert np.allclose(res.c0, 2.0, rtol=1e-14) and np.all(res.b == 0.0)
    assert np.max(np.abs(res.a[:, 1:])) < 1e-14


# ----------------------------------------------------------------------------- quadrature rule


def test_quarter_wave_rule():
    theta, w = theta_nodes(64)
    assert theta.shape == w.shape == (64,) and np.all(np.diff(theta) > 0)
    assert 0.0 < theta[0] and theta[-1] < math.pi / 2
    assert w.sum() == pytest.approx(math.pi / 2, rel=1e-15)
    # Int_0^{pi/2} sin^{3/2} = sqrt(pi) Gamma(5/4) / (2 Gamma(7/4)): analytic in tau, exact at modest n
    exact = math.sqrt(math.pi) * math.gamma(1.25) / (2.0 * math.gamma(1.75))
    assert np.dot(w, np.sin(theta) ** 1.5) == pytest.approx(exact, rel=1e-14)
    assert not theta.flags.writeable and theta_nodes(64)[0] is theta  # cached and read-only


@pytest.mark.parametrize("n", [1, 2, 7, 64, 385, 4096])
def test_gauss_legendre_nodes(n):
    tau, w = _gauss_legendre_01(n)
    assert tau.size == n and np.all(np.diff(tau) > 0) and 0.0 < tau[0] and tau[-1] < 1.0
    assert np.allclose(tau + tau[::-1], 1.0, rtol=0, atol=1e-15) and np.allclose(w, w[::-1], rtol=1e-12)
    for j in range(min(2 * n, 40)):  # exact for polynomials of degree 2n - 1
        assert np.dot(w, tau**j) == pytest.approx(1.0 / (j + 1), rel=2e-14)


# ----------------------------------------------------------------------------- local exponent


def test_local_exponent():
    E = np.geomspace(1e2, 1e5, 40)
    assert np.allclose(local_exponent(E, E**1.5), 1.5, rtol=0, atol=1e-12)
    rng = np.random.default_rng(7)
    E_irregular = np.sort(rng.uniform(1e2, 1e5, 60))
    assert np.allclose(local_exponent(E_irregular, lambda x: 3.0 * x**1.5), 1.5, rtol=0, atol=1e-10)
    assert np.allclose(local_exponent(-E, np.sign(-E) * E**1.5), 1.5, rtol=0, atol=1e-12)  # E < 0 side
    two_rows = np.stack([E**1.5, -(E**2.5)])  # E along the last axis; |J| is used
    assert np.allclose(local_exponent(E, two_rows), [[1.5], [2.5]], rtol=0, atol=1e-11)
    # activated power law: p(E) = 1.5 + Ec/E (second-order differences on a fine grid)
    Ec, Ef = 1e3, np.geomspace(1e2, 1e5, 2000)
    p = local_exponent(Ef, Ef**1.5 * np.exp(-Ec / Ef))
    assert np.allclose(p, 1.5 + Ec / Ef, rtol=2e-5, atol=0)


@pytest.mark.slow
def test_per_row_convergence_over_decades():
    """Review regression: rows with small amplitude must converge relative to their own size."""

    def J(E):
        a = np.abs(E)
        with np.errstate(divide="ignore", invalid="ignore"):
            osc = np.where(a > 0, np.cos(2 * np.pi / np.where(a > 0, a, 1.0)), 0.0)
        return np.sign(E) * a**1.5 * (1.0 + 0.5 * osc)  # essential oscillatory singularity at E = 0

    E0 = np.geomspace(1e-2, 1e2, 9)
    n = 2**21  # reference: half-shifted periodic trapezoid (never samples E = 0)
    theta = 2 * np.pi * (np.arange(n) + 0.5) / n
    b_ref = np.array(
        [[2.0 * np.mean(J(e0 * np.sin(theta)) * np.sin(N * theta)) for N in range(10)] for e0 in E0]
    )
    row = np.max(np.abs(b_ref), axis=1)
    res = harmonics(J, E0, n_max=9, tol=1e-6, n_cap=65536)
    assert res.converged
    assert np.all(np.max(np.abs(res.b - b_ref), axis=1) < 1e-6 * row)
    # an unreachable tolerance is reported honestly instead of certified against the global scale
    hard = harmonics(J, E0, n_max=9, tol=1e-9, n_cap=8192)
    assert not hard.converged and hard.info["err_rel_rows_max"] > 1e-9
