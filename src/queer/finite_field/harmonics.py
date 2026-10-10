"""Harmonic projector for Eq. 13 (Methods 8) and the main-text sine form (l. 178).

The drive E(theta) = E0 sin(theta) gives the periodic signal J(theta) = J_H(E0 sin theta) with

    J(theta) = a_0/2 + sum_{N>=1} [a_N cos(N theta) + b_N sin(N theta)],
    a_N = (1/pi) Int_0^{2pi} J cos(N theta) d theta,     b_N = (1/pi) Int_0^{2pi} J sin(N theta) d theta,
    c_N = (1/2pi) Int_0^{2pi} J e^{-i N theta} d theta = (a_N - i b_N)/2,     c_0 = a_0/2 = <J>.

b_N is the main-text J^{N w}_y and c_N the Methods form of Eq. 13. Write J_odd(E) = [J(E) - J(-E)]/2
and J_even(E) = [J(E) + J(-E)]/2. The drive satisfies f(pi - theta) = f(theta) and
f(theta + pi) = J(-E0 sin theta), which reduces the period to a quarter wave. This is exact for any J:

    b_N = (4/pi) Int_0^{pi/2} J_odd(E0 sin theta) sin(N theta) d theta    (odd N; b_N = 0 for even N)
    a_N = (4/pi) Int_0^{pi/2} J_even(E0 sin theta) cos(N theta) d theta   (even N; a_N = 0 for odd N)
    c_0 = (2/pi) Int_0^{pi/2} J_even(E0 sin theta) d theta.

J is evaluated explicitly at +E and -E. Oddness is assumed only when odd_only=True.

Quadrature: Gauss-Legendre in tau on [0, 1] with theta = (pi/2) tau^2 (d theta = pi tau d tau). Near
theta = 0, |sin theta|^p = (pi tau^2/2)^p times an analytic factor, so the critical J ~ |E|^{3/2} is
analytic in tau and the rule converges exponentially. An activated, oscillatory J such as
exp(-E_c/|E|) cos(2 pi Lambda/|E|) needs more nodes. Adaptive mode doubles the node count until no
coefficient changes by more than tol * max(|b_1|, |c_0|).

Closed forms for J = sgn(E)|E|^p at E0 = 1 (b_N scales as E0^p): b_N(p), with the large-N law
N^{p+1}|b_N| -> (4/pi) Gamma(p+1)|cos(pi p/2)|. For p = 3/2 this limit is c_F = 3/sqrt(2 pi) = 1.196827.

Units: E0 in V/m, as are the arguments of J; the coefficients carry the unit of J (A/m in the pipeline).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy import special as sp

C_F = 3.0 / math.sqrt(2.0 * math.pi)  # lim N^{5/2}|b_N| for sgn(E)|E|^{3/2}; Methods 5 quotes c_F = 1.1968


@dataclass(frozen=True)
class HarmonicResult:
    """Fourier coefficients of J(E0 sin theta): rows are amplitudes E0, columns are orders N = 0..n_max.

    The convention is the standard one, a_0 = 2 c0, so c = (a - i b)/2 holds in every column.
    By construction b_N = 0 for even N and a_N = 0 for odd N.
    """

    E0: np.ndarray  # (nE0,) drive amplitudes, V/m
    N: np.ndarray  # (n_max + 1,) harmonic orders 0..n_max
    b: np.ndarray  # (nE0, n_max + 1) sine coefficients (main text, l. 178)
    a: np.ndarray  # (nE0, n_max + 1) cosine coefficients; a[:, 0] = 2 c0
    c: np.ndarray  # (nE0, n_max + 1) complex Methods-form coefficients (Eq. 13), c[:, 0] = c0
    c0: np.ndarray  # (nE0,) DC part <J>
    n_theta: int  # quarter-wave nodes behind the returned coefficients
    converged: bool  # True only if a doubling check met the tolerance
    info: dict = field(default_factory=dict)
    converged_rows: np.ndarray | None = None
    relative_error: np.ndarray | None = None


# ---------------------------------------------------------------------------------------------------
# quarter-wave rule
# ---------------------------------------------------------------------------------------------------


def _legendre_pair(n: int, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(P_n(x), P_{n-1}(x)) by the upward three-term recurrence (stable on [-1, 1])."""
    p_prev = np.ones_like(x)
    p = x.copy()
    tmp = np.empty_like(x)
    for k in range(2, n + 1):  # k P_k = (2k - 1) x P_{k-1} - (k - 1) P_{k-2}
        np.multiply(x, p, out=tmp)
        tmp *= (2.0 * k - 1.0) / k
        p_prev *= (k - 1.0) / k
        tmp -= p_prev
        p_prev, p, tmp = p, tmp, p_prev
    return p, p_prev


@functools.lru_cache(maxsize=64)
def _gauss_legendre_01(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes (ascending) and weights on [0, 1] in O(n^2) time and O(n) memory.

    The method is Newton iteration on P_n(cos phi) in the angle phi, from Tricomi's estimate.
    1 - x^2 = sin^2 phi and tau = sin^2(phi/2) | cos^2(phi/2) avoid cancellation at the ends. Nodes and
    weights are good to ~1e-16 absolute: 5e-10 relative on the outermost weight at n = 6144, against
    6e-6 for scipy's roots_legendre, which also takes ~4 s at n = 16384 (this takes ~0.3 s).
    """
    m = (n + 1) // 2
    k = np.arange(1, m + 1, dtype=float)
    phi = math.pi * (4.0 * k - 1.0) / (4.0 * n + 2.0)
    phi += (1.0 / (8.0 * n * n) - 1.0 / (8.0 * n**3)) / np.tan(phi)  # x ~ (1 - 1/8n^2 + 1/8n^3) cos(phi)
    for _ in range(50):
        x = np.cos(phi)
        p, p_prev = _legendre_pair(n, x)
        step = p * np.sin(phi) / (n * (p_prev - x * p))  # -g/g' for g(phi) = P_n(cos phi)
        phi += step
        if np.max(np.abs(step)) < 1e-11:  # quadratic convergence: the next step is far below 1e-16
            break
    else:  # pragma: no cover - the iteration converges in 2-3 steps for every n
        raise RuntimeError(f"Gauss-Legendre nodes did not converge for n = {n}")
    x = np.cos(phi)
    p, p_prev = _legendre_pair(n, x)
    w = np.sin(phi) ** 2 / (n * (p_prev - x * p)) ** 2  # [0, 1] weight = half of 2(1 - x^2)/(n P_{n-1})^2
    lo, hi = np.sin(0.5 * phi) ** 2, np.cos(0.5 * phi) ** 2  # tau = (1 -+ cos phi)/2 without cancellation
    mirror = slice(-2, None, -1) if n % 2 else slice(None, None, -1)  # odd n: tau = 1/2 only once
    tau = np.concatenate([lo, hi[mirror]])
    wt = np.concatenate([w, w[mirror]])
    tau.flags.writeable = False
    wt.flags.writeable = False
    return tau, wt


@functools.lru_cache(maxsize=64)
def _theta_nodes_cached(n: int) -> tuple[np.ndarray, np.ndarray]:
    tau, w = _gauss_legendre_01(n)
    theta = 0.5 * math.pi * tau * tau
    weight = math.pi * tau * w
    theta.flags.writeable = False
    weight.flags.writeable = False
    return theta, weight


def theta_nodes(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Quarter-wave rule: nodes theta_j in (0, pi/2) and weights W_j with sum_j W_j g(theta_j) ~ Int_0^{pi/2} g.

    The rule is n-point Gauss-Legendre in tau on [0, 1] with theta = (pi/2) tau^2 and W = pi tau w_tau.
    The arrays are cached and read-only.
    """
    n = int(n)
    if n < 1:
        raise ValueError("n must be a positive integer")
    return _theta_nodes_cached(n)


# ---------------------------------------------------------------------------------------------------
# projector
# ---------------------------------------------------------------------------------------------------


def _sample(J: Callable, E: np.ndarray) -> np.ndarray:
    """J at every entry of E with one vectorized call on the flattened array, reshaped to E.shape."""
    flat = E.reshape(-1)
    vals = np.asarray(J(flat))
    if np.iscomplexobj(vals):
        raise TypeError("J must be real-valued")
    if vals.shape != flat.shape:
        raise ValueError(f"J must be vectorized: returned shape {vals.shape} for input shape {flat.shape}")
    vals = vals.astype(float, copy=False)
    if not np.all(np.isfinite(vals)):
        raise ValueError("J returned non-finite values")
    return vals.reshape(E.shape)


def harmonics(
    J: Callable,
    E0: ArrayLike,
    n_max: int = 41,
    n_theta: int = 384,
    adaptive: bool = True,
    tol: float = 1e-8,
    n_cap: int = 16384,
    odd_only: bool = False,
) -> HarmonicResult:
    """Harmonics a_N, b_N, c_N (N = 0..n_max) of J(E0 sin theta) by the exact quarter-wave reduction.

    J: vectorized callable of the field (V/m, either sign) that takes a 1-D array. It is called once per
    node set, on all E0 sin(theta_j) and -E0 sin(theta_j); with odd_only=True it is called only on
    +E0 sin(theta_j), and J_even = 0 is assumed.
    E0: scalar or 1-D array of drive amplitudes (V/m).
    n_theta: Gauss-Legendre nodes in tau. With adaptive=True the node count doubles (capped at n_cap)
    until, for every E0 row, max_N |change of a and b| <= tol * max(|b_1|, |c0|, floor) with the row's
    own amplitudes and an absolute floor of 1e-14 times the global scale (rows whose response is
    numerically zero do not block convergence). `converged` is True only when that check passes; with
    adaptive=False it is False (no check is made).
    """
    E0 = np.atleast_1d(np.asarray(E0, dtype=float))
    if E0.ndim != 1 or E0.size == 0 or not np.isfinite(E0).all():
        raise ValueError("E0 must be a scalar or a non-empty 1-D array")
    if any(isinstance(value, (bool, np.bool_)) or int(value) != value for value in (n_max, n_theta, n_cap)):
        raise ValueError("n_max, n_theta and n_cap must be integers")
    if not np.isfinite(tol) or tol <= 0 or n_cap < 1:
        raise ValueError("tol and n_cap must be positive")
    n_max = int(n_max)
    n = int(n_theta)
    if n_max < 0 or n < 1:
        raise ValueError("need n_max >= 0 and n_theta >= 1")
    N = np.arange(n_max + 1)
    odd_N, even_N = N[1::2], N[0::2]
    calls, evals = 0, 0

    def project(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
        nonlocal calls, evals
        theta, w = theta_nodes(n_nodes)
        E = E0[:, None] * np.sin(theta)
        if odd_only:
            j_odd, j_even = _sample(J, E), None
        else:
            jpm = _sample(J, np.stack([E, -E]))
            j_odd, j_even = 0.5 * (jpm[0] - jpm[1]), 0.5 * (jpm[0] + jpm[1])
        calls += 1
        evals += E.size * (1 if odd_only else 2)
        a = np.zeros((E0.size, n_max + 1))
        b = np.zeros_like(a)
        w4 = (4.0 / math.pi) * w[:, None]
        if odd_N.size:
            b[:, 1::2] = j_odd @ (w4 * np.sin(np.outer(theta, odd_N)))
        if j_even is not None:
            a[:, 0::2] = j_even @ (w4 * np.cos(np.outer(theta, even_N)))  # column 0: a_0 = 2 c0
        return a, b

    def scale_of(a: np.ndarray, b: np.ndarray) -> float:
        b1 = float(np.max(np.abs(b[:, 1]))) if n_max >= 1 else 0.0
        return max(b1, float(np.max(np.abs(a[:, 0]))) / 2.0, np.finfo(float).tiny)

    def row_scales(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        b1 = np.abs(b[:, 1]) if n_max >= 1 else np.zeros(a.shape[0])
        return np.maximum(np.maximum(b1, 0.5 * np.abs(a[:, 0])), 1e-14 * scale_of(a, b))

    a, b = project(n)
    n_history, err_history = [n], []
    converged = False
    err_rel_rows = np.full(E0.shape, np.nan)
    if adaptive:
        while True:
            n_next = min(2 * n, int(n_cap))
            if n_next <= n:
                break
            a2, b2 = project(n_next)
            err_rows = np.maximum(np.max(np.abs(a2 - a), axis=1), np.max(np.abs(b2 - b), axis=1))
            err = float(np.max(err_rows))
            a, b, n = a2, b2, n_next
            n_history.append(n)
            err_history.append(err)
            err_rel_rows = err_rows / row_scales(a, b)
            if np.all(err_rel_rows <= tol):
                converged = True
                break
    scale = scale_of(a, b)
    info = {
        "rule": "gauss_legendre_tau2",
        "n_history": n_history,
        "err_history": err_history,  # max |change| of (a, b) at each doubling
        "err_rel": err_history[-1] / scale if err_history else math.nan,
        "err_rel_rows_max": float(np.max(err_rel_rows)) if err_history else math.nan,
        "scale": scale,
        "tol": float(tol),
        "n_cap": int(n_cap),
        "adaptive": bool(adaptive),
        "odd_only": bool(odd_only),
        "n_calls": calls,
        "n_evals": evals,
    }
    return HarmonicResult(
        E0=E0.copy(),
        N=N,
        b=b,
        a=a,
        c=0.5 * (a - 1j * b),
        c0=0.5 * a[:, 0],
        n_theta=n,
        converged=converged,
        info=info,
        converged_rows=err_rel_rows <= tol,
        relative_error=err_rel_rows,
    )


# ---------------------------------------------------------------------------------------------------
# power-law closed forms (critical scaling J ~ sgn(E)|E|^p)
# ---------------------------------------------------------------------------------------------------


def _is_pole(x: np.ndarray) -> np.ndarray:
    """True at the poles of Gamma (non-positive integers)."""
    return (x <= 0.0) & (x == np.floor(x))


def _gamma_quotient(num, den) -> np.ndarray:
    """prod_i Gamma(num_i) / prod_j Gamma(den_j), elementwise with broadcasting.

    The product is evaluated directly when every argument is moderate (|x| < 170) and in log-gamma form
    with explicit signs otherwise. A pole in `den` gives exactly 0. The caller excludes poles in `num`.
    """
    args = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (*num, *den)))
    num_b, den_b = args[: len(num)], args[len(num) :]
    with np.errstate(all="ignore"):
        direct = np.ones(args[0].shape)
        sign = np.ones(args[0].shape)
        log_q = np.zeros(args[0].shape)
        for x in num_b:
            direct = direct * sp.gamma(x)
            sign = sign * sp.gammasgn(x)
            log_q = log_q + sp.gammaln(x)
        for x in den_b:
            direct = direct * sp.rgamma(x)
            sign = sign * sp.gammasgn(x)
            log_q = log_q - sp.gammaln(x)
        via_log = sign * np.exp(log_q)
    moderate = (
        np.all([np.abs(x) < 170.0 for x in args], axis=0) & np.isfinite(direct) & (np.abs(direct) > 1e-280)
    )
    out = np.where(moderate, direct, via_log)
    return np.where(np.any([_is_pole(x) for x in den_b], axis=0), 0.0, out)


def _cos_half_pi(p: np.ndarray) -> np.ndarray:
    """cos(pi p/2), with full relative accuracy near (and exactly 0 at) odd integers p.

    With k = round((p - 1)/2) and d = p - (2k + 1) (exact by Sterbenz's lemma near odd integers):
    cos(pi p/2) = (-1)^(k+1) sin(pi d/2).
    """
    p = np.asarray(p, dtype=float)
    k = np.round(0.5 * (p - 1.0))
    d = p - (2.0 * k + 1.0)
    sign = np.where(np.mod(k, 2.0) == 0.0, -1.0, 1.0)
    return sign * np.sin(0.5 * math.pi * d)


def power_law_bN(N: ArrayLike, p: ArrayLike, form: str = "ratio") -> np.ndarray:
    """Closed-form b_N of J = sgn(E)|E|^p at E0 = 1 (b_N is proportional to E0^p); b_N = 0 for even N.

    form="ratio": (2^{1-p}/pi) Gamma(p+1) cos(pi p/2) Gamma((N-p)/2) / Gamma((N+p)/2 + 1)
    form="beta":  2^{1-p} sin(N pi/2) Gamma(p+1) / [Gamma(1 + (p+N)/2) Gamma(1 + (p-N)/2)]
    The two agree for odd N by the reflection formula. At the poles of Gamma((N-p)/2) (p = N + 2m an odd
    integer, where cos(pi p/2) = 0) the ratio form takes the limit 2^{1-p} sin(N pi/2) Gamma(p+1) /
    [m! Gamma((N+p)/2 + 1)]. The beta form vanishes at the poles of Gamma(1 + (p-N)/2), which gives
    b_N = 0 for N > p when p is an odd integer. For p = 3/2:
    b_N = -(3/(8 sqrt(pi))) Gamma(N/2 - 3/4)/Gamma(N/2 + 7/4) with b_1 = 0.9153117 and b_3 = -0.1017013.
    Requires p > -1 (integrability) and integer N >= 0.
    """
    N, p = np.broadcast_arrays(np.asarray(N), np.asarray(p, dtype=float))
    if not np.all(N == np.round(N)) or np.any(N < 0):
        raise ValueError("N must be non-negative integers")
    if np.any(p <= -1.0):
        raise ValueError("p must exceed -1")
    Ni = np.round(N).astype(np.int64)
    Nf = Ni.astype(float)
    odd = Ni % 2 == 1
    s_N = np.where(odd, np.where((Ni // 2) % 2 == 0, 1.0, -1.0), 0.0)  # sin(N pi/2) for integer N
    two = 2.0 ** (1.0 - p)
    zb = 0.5 * (Nf + p) + 1.0
    with np.errstate(all="ignore"):  # entries masked below (even N) may overflow harmlessly
        if form == "ratio":
            z = 0.5 * (Nf - p)
            pole = _is_pole(z)
            z_safe = np.where(pole, 0.5, z)  # keep the excluded numerator poles out of Gamma
            regular = two / math.pi * _cos_half_pi(p) * _gamma_quotient((p + 1.0, z_safe), (zb,))
            limit = (
                two * s_N * _gamma_quotient((p + 1.0,), (1.0 - z, zb))
            )  # cos(pi p/2) Gamma(-m) -> pi s_N/m!
            out = np.where(pole, limit, regular)
        elif form == "beta":
            out = two * s_N * _gamma_quotient((p + 1.0,), (zb, 1.0 + 0.5 * (p - Nf)))
        else:
            raise ValueError(f"unknown form {form!r}; use 'ratio' or 'beta'")
    return (np.where(odd, out, 0.0) + 0.0)[()]  # + 0.0 turns the -0.0 of vanishing terms into 0.0


def power_law_cf(p: ArrayLike) -> np.ndarray:
    """Large-N limit of N^{p+1}|b_N| for J = sgn(E)|E|^p at E0 = 1: (4/pi) Gamma(p+1)|cos(pi p/2)|.

    Equals C_F = 3/sqrt(2 pi) for p = 3/2.
    """
    p = np.asarray(p, dtype=float)
    if np.any(p <= -1.0):
        raise ValueError("p must exceed -1")
    return (4.0 / math.pi * sp.gamma(p + 1.0) * np.abs(_cos_half_pi(p)))[()]


def local_exponent(E: ArrayLike, J: ArrayLike | Callable, axis: int = -1) -> np.ndarray:
    """Local power-law exponent p(E) = d ln|J| / d ln|E| on the given grid.

    Uses centered differences (second order on non-uniform grids) and one-sided differences at the ends
    (second order if the grid has at least 3 points). E is a 1-D grid without zeros. J holds the values
    on the grid, with E along `axis`, or is a callable evaluated on E.
    """
    E = np.asarray(E, dtype=float)
    values = np.asarray(J(E) if callable(J) else J, dtype=float)
    x = np.log(np.abs(E))
    with np.errstate(divide="ignore"):
        y = np.log(np.abs(values))
    return np.gradient(y, x, axis=axis, edge_order=2 if E.size >= 3 else 1)
