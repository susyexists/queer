"""Interference propagation through a sequence of avoided crossings (Eq. 12 and its generalization).

Eq. 12 (unnumbered, after Eq. 11) is the conduction-band population after two crossings at -+k0,

    P_int = P (1 + (1 - 2P) e^{-d/kappa1} + 2 (1 - P) e^{-d/(2 kappa1)} e^{-d/kappa2} cos Phi),   d = 2 k0,

with the relaxation lengths kappa_i = e|E|T_i/hbar (1/A; units.kappa) and Phi = 2 pi Lambda/|E|. It is the
n = 2 case of a sequential two-level density-matrix model (adiabatic-impulse model) that this module
evaluates for any number of crossings in traversal order:

  * rho is 2x2 in the (c, v) basis and starts in the valence band, rho_vv = 1.
  * Crossing j (LZ probability P_j, Stokes phase phi_j) acts as rho <- S rho S^dag with
        S = [[sqrt(1-P) e^{-i phi}, -sqrt(P)], [sqrt(P), sqrt(1-P) e^{+i phi}]].
  * A segment of length d (1/A) between crossings relaxes populations, rho_cc <- rho_cc e^{-d/kappa1}
    (rho_vv = 1 - rho_cc), and coherences, rho_cv <- rho_cv exp(-Gamma d - i theta) with
        "paper":  Gamma = 1/(2 kappa1) + 1/kappa2   (amplitude damping plus pure dephasing; Eq. 12)
        "bloch":  Gamma = 1/kappa2                   (T2 is the total coherence time; needs T2 <= 2 T1)
    theta is the interband phase accumulated on the segment: the dynamical phase Int eps_g ds/(e|E|)
    plus, optionally, the geometric (shift-vector) phase gamma_R. The caller assembles it as
    theta = Phi + gamma_R, with gamma_R the gauge-invariant link sum (geometry/wilson.py) accumulated
    from the entry crossing to the exit crossing along the direction of motion (reversed path for
    E < 0); the opposite sign is ruled out by the Houston TDSE (tests/test_tdse.py).

Two crossings give P_2 + P_1 (1 - 2 P_2) e^{-d/kappa1} + 2 sqrt(P_1 P_2 (1-P_1)(1-P_2)) e^{-Gamma d}
cos(theta + phi_1 + phi_2), so P_1 = P_2 = P with phi = 0 is Eq. 12 exactly.

The Stokes phase of a linear crossing (Shevchenko, Ashhab & Nori, Phys. Rep. 492, 1 (2010)),

    phi_S(delta) = pi/4 + delta (ln delta - 1) + arg Gamma(1 - i delta),     P = exp(-2 pi delta),

runs from pi/4 (sudden, P -> 1) to 0 (adiabatic, P -> 0; phi_S ~ 1/(12 delta)). For the real
ParabolicDoubleCrossing toy model the Houston TDSE gives the final c-population
2 P (1 - P)(1 + cos(Phi + 2 phi_S + gamma)) with gamma = pi between the gap minima (tests/test_tdse.py),
i.e. theta = Phi + gamma and phiS = +phi_S in the S matrix above; the printed Eq. 12 is the
phi_S = gamma = 0 case.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import loggamma

_DELTA_ASYMPTOTIC = 8.0  # above: Stirling series (the direct form cancels ~delta ln delta digits)
# |B_2n| / (2n (2n - 1)) for n = 1..7: phi_S = sum_n c_n delta^{1 - 2n} (all terms positive)
_STIRLING = (1.0 / 12.0, 1.0 / 360.0, 1.0 / 1260.0, 1.0 / 1680.0, 1.0 / 1188.0, 691.0 / 360360.0, 1.0 / 156.0)


def stokes_phase(delta):
    """Stokes phase phi_S(delta) = pi/4 + delta (ln delta - 1) + arg Gamma(1 - i delta), rad.

    delta >= 0 is the adiabaticity parameter (P_LZ = exp(-2 pi delta)); inf is allowed (phi_S = 0).
    For delta >= 8 the asymptotic series sum_n |B_2n| / (2n (2n-1) delta^{2n-1}) is used (error < 1e-14).
    """
    d = np.asarray(delta, dtype=float)
    if np.any(np.isnan(d)) or np.any(d < 0):
        raise ValueError("delta must be >= 0")
    out = np.empty(d.shape)
    small = d < _DELTA_ASYMPTOTIC
    ds = d[small]
    with np.errstate(divide="ignore", invalid="ignore"):
        dlog = np.where(ds > 0, ds * (np.log(ds) - 1.0), 0.0)
    out[small] = 0.25 * math.pi + dlog + loggamma(1.0 - 1j * ds).imag
    inv = 1.0 / d[~small]
    inv2 = inv * inv
    series = np.zeros_like(inv)
    for coef in reversed(_STIRLING):
        series = coef + inv2 * series
    out[~small] = inv * series
    return out[()] if out.ndim == 0 else out


def _check_P(P):
    P = np.asarray(P, dtype=float)
    if np.any(np.isnan(P)) or np.any(P < -1e-12) or np.any(P > 1.0 + 1e-12):
        raise ValueError("LZ probabilities must lie in [0, 1]")
    return np.clip(P, 0.0, 1.0)


def stokes_phase_from_P(P):
    """Stokes phase for an LZ probability P in [0, 1]: phi_S(delta) with delta = -ln(P)/(2 pi)."""
    P = _check_P(P)  # same rounding tolerance as propagate_crossings
    with np.errstate(divide="ignore"):
        delta = -np.log(P) / (2.0 * math.pi)
    return stokes_phase(np.maximum(delta, 0.0))


def _ratio(d, k):
    """d / kappa with kappa = inf meaning no relaxation (0, also for d = inf)."""
    with np.errstate(invalid="ignore"):
        return np.where(np.isinf(k), 0.0, np.asarray(d, dtype=float) / k)


def _coherence_rate(r1, r2, coherence: str):
    """Coherence decay exponent Gamma*d from r1 = d/kappa1 and r2 = d/kappa2."""
    if coherence == "paper":
        return 0.5 * r1 + r2
    if coherence == "bloch":
        if np.any(r2 < 0.5 * r1 * (1.0 - 1e-12)):
            raise ValueError(
                "coherence='bloch' requires kappa2 <= 2 kappa1 (T2 <= 2 T1) to keep rho positive"
            )
        return r2
    raise ValueError(f"coherence must be 'paper' or 'bloch', got {coherence!r}")


def _check_kappa(kappa):
    k = np.asarray(kappa, dtype=float)
    if np.any(np.isnan(k)) or np.any(k <= 0):
        raise ValueError("relaxation lengths must be > 0 (inf allowed)")
    return k


def _leading(a, n: int, name: str):
    """Array with leading axis of length n (crossings or segments)."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 0 or a.shape[0] != n:
        raise ValueError(f"{name} must have leading dimension {n}, got shape {a.shape}")
    return a


def _expand(a, shape):
    """Broadcast (L, *s) to (L, *shape), aligning s with the trailing axes."""
    a = a.reshape((a.shape[0],) + (1,) * (len(shape) - (a.ndim - 1)) + a.shape[1:])
    return np.broadcast_to(a, (a.shape[0],) + tuple(shape))


def propagate_crossings(
    P, dseg, theta, kappa1, kappa2, phiS=None, coherence: str = "paper", return_coherence: bool = False
):
    """Conduction-band population rho_cc right after each of n crossings, shape (n, ...).

    P: (n, ...) LZ probabilities in traversal order. dseg: (n-1, ...) distances between consecutive
    crossings (1/A). theta: (n-1, ...) interband phases of the segments (rad). kappa1, kappa2: relaxation
    lengths e|E|T/hbar (1/A, inf allowed), broadcastable to the trailing shape `...`. phiS: (n, ...)
    Stokes phases or None (0). coherence: "paper" or "bloch" (module docstring). For n = 1, dseg and theta
    may be None. return_coherence: also return rho_cv (complex, (n, ...)) right after each crossing.

    The map is completely positive ("paper" always; "bloch" for kappa2 <= 2 kappa1, checked), so
    0 <= rho_cc <= 1 and |rho_cv|^2 <= rho_cc rho_vv hold throughout.
    """
    P = _check_P(P)
    if P.ndim == 0:
        raise ValueError("P must have a leading crossing axis")
    n = P.shape[0]
    if n == 0:
        raise ValueError("need at least one crossing")
    empty = np.zeros((0,) + P.shape[1:])
    dseg = empty if (dseg is None and n == 1) else _leading(dseg, n - 1, "dseg")
    theta = empty if (theta is None and n == 1) else _leading(theta, n - 1, "theta")
    if np.any(dseg < 0):
        raise ValueError("segment lengths must be >= 0")
    phi = np.zeros((n,) + P.shape[1:]) if phiS is None else _leading(phiS, n, "phiS")
    k1, k2 = _check_kappa(kappa1), _check_kappa(kappa2)
    shape = np.broadcast_shapes(
        P.shape[1:], phi.shape[1:], dseg.shape[1:], theta.shape[1:], k1.shape, k2.shape
    )
    P, phi, dseg, theta = (_expand(a, shape) for a in (P, phi, dseg, theta))

    r1 = _ratio(dseg, k1)  # kappa = inf: no decay, also for d = inf
    r2 = _ratio(dseg, k2)
    pop = np.exp(-r1)  # (n-1, *shape)
    coh = np.exp(-_coherence_rate(r1, r2, coherence) - 1j * theta)

    out = np.empty((n,) + tuple(shape))
    out_cv = np.empty((n,) + tuple(shape), dtype=complex)
    rcc = np.zeros(shape)
    rcv = np.zeros(shape, dtype=complex)
    for j in range(n):
        p = P[j]
        q = 1.0 - p
        sq = np.sqrt(p * q)
        ph = np.exp(-1j * phi[j])
        rvv = 1.0 - rcc
        # rho <- S rho S^dag with S = [[sqrt(q) ph, -sqrt(p)], [sqrt(p), sqrt(q) conj(ph)]]
        rcc, rcv = (
            q * rcc + p * rvv - 2.0 * sq * np.real(ph * rcv),
            sq * ph * (rcc - rvv) + q * ph * ph * rcv - p * np.conj(rcv),
        )
        out[j], out_cv[j] = rcc, rcv
        if j < n - 1:
            rcc = rcc * pop[j]
            rcv = rcv * coh[j]
    return (out, out_cv) if return_coherence else out


def p_int_two(P1, P2, d, kappa1, kappa2, theta, coherence: str = "paper"):
    """Closed form of propagate_crossings for n = 2 (rho_cc after the second crossing).

    P2 + P1 (1 - 2 P2) e^{-d/kappa1} + 2 sqrt(P1 P2 (1-P1)(1-P2)) e^{-Gamma d} cos(theta), with
    Gamma = 1/(2 kappa1) + 1/kappa2 ("paper") or 1/kappa2 ("bloch"). Stokes phases enter as
    theta -> theta + phi_1 + phi_2. P1 = P2 = P with d = 2 k0 and theta = Phi is Eq. 12.
    """
    P1, P2 = _check_P(P1), _check_P(P2)
    d = np.asarray(d, dtype=float)
    if np.any(np.isnan(d)) or np.any(d < 0):
        raise ValueError("segment lengths must be >= 0")
    k1, k2 = _check_kappa(kappa1), _check_kappa(kappa2)
    r1, r2 = _ratio(d, k1), _ratio(d, k2)
    amp = np.sqrt(P1 * P2 * (1.0 - P1) * (1.0 - P2))
    return (
        P2
        + P1 * (1.0 - 2.0 * P2) * np.exp(-r1)
        + 2.0 * amp * np.exp(-_coherence_rate(r1, r2, coherence)) * np.cos(np.asarray(theta, dtype=float))
    )
