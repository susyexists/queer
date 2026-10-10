"""Naive reference for Eqs. 10-11: nested adaptive quadrature with per-point diagonalization.

Independent of the optimized pipeline: no logistic nodes, no lambda-tables, no kernels, its own
transverse range and tangency breakpoints (both sides scanned separately, no mirroring) and its own
density-matrix propagation. It shares only the crossing finder and the band-geometry kernel. Used to
validate HallCalculator and as the baseline of the performance benchmarks (seconds per J_H(E) value).

Supported: every PAPER-type convention including ec_method = "ddp". The PHYSICAL phases (geometric,
Stokes, twisted LZ) are not implemented here and raise.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad

from ..config import PAPER, RunConfig
from ..geometry.provider import ModelGeometry
from ..lz.crossings import find_crossings
from ..units import hall_prefactor, kappa


def _omega(provider, s, t, mode):
    geo = provider.geometry(np.array([s]), np.array([t]))
    return float(geo.omega(mode)[0])


def _check(cfg: RunConfig):
    ph = cfg.phase
    if ph.geometric or ph.stokes or ph.twisted_lz:
        raise NotImplementedError("the naive reference implements the dynamical phase only")


def _crossings(provider, t, cfg):
    ddp = cfg.conventions.ec_method == "ddp"
    lc = find_crossings(
        provider, t, cfg.crossings, want_ddp=ddp, e_cut=cfg.quadrature.p_cut * cfg.drive.E_max
    )
    cr = sorted(lc.crossings, key=lambda c: c.s)
    Ec = np.array([(c.Ec_ddp if ddp else c.Ec[cfg.conventions.metric]) for c in cr])
    return np.array([c.s for c in cr]), Ec


def _populations(P, d, theta, lam1, lam2, coherence):
    """rho_cc after each crossing: explicit 2x2 density matrices, rho <- S rho S^dag at a crossing."""
    rho = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)  # basis (c, v); start in v
    out = []
    for j, p in enumerate(P):
        if j > 0:
            pop = math.exp(-d[j - 1] / lam1) if math.isfinite(lam1) else 1.0
            rate = (d[j - 1] / lam2 if math.isfinite(lam2) else 0.0) + (
                0.5 * d[j - 1] / lam1 if (coherence == "paper" and math.isfinite(lam1)) else 0.0
            )
            rcc = rho[0, 0].real * pop
            rcv = rho[0, 1] * math.exp(-rate) * np.exp(-1j * theta[j - 1])
            rho = np.array([[rcc, rcv], [np.conj(rcv), 1.0 - rcc]])
        S = np.array([[math.sqrt(1.0 - p), -math.sqrt(p)], [math.sqrt(p), math.sqrt(1.0 - p)]])
        rho = S @ rho @ S.T
        out.append(rho[0, 0].real)
    return np.array(out)


def line_integrand(provider, t: float, E: float, cfg: RunConfig = PAPER, epsrel: float = 1e-10) -> float:
    """F(t, E) = sum_j rho_j Int_seg_j exp(-|s - s_j|/lambda) Omega(s, t) ds by adaptive quadrature."""
    _check(cfg)
    conv, rel, pol = cfg.conventions, cfg.relaxation, cfg.crossings
    s, Ec = _crossings(provider, t, cfg)
    if s.size == 0:
        return 0.0
    direction = conv.charge_sign * int(np.sign(E))
    Ea = abs(E)
    if np.all(Ec / Ea > 745.0):  # every P underflows: no injection on this line
        return 0.0
    lam1 = float(kappa(Ea, rel.T1))
    lam2 = float(kappa(Ea, rel.T2))
    order = s if direction > 0 else s[::-1]
    Ec = Ec if direction > 0 else Ec[::-1]
    P = np.exp(-Ec / Ea)
    gaps = []
    for a, b in zip(order[:-1], order[1:]):
        lo, hi = min(a, b), max(a, b)
        gaps.append(
            quad(
                lambda x: float(provider.gap(np.array([x]), np.array([t]))[0]),
                lo,
                hi,
                epsrel=epsrel,
                limit=200,
            )[0]
        )
    theta = np.array(gaps) * 1e10 / Ea
    rho = _populations(P, np.abs(np.diff(order)), theta, lam1, lam2, rel.coherence)
    W = pol.window
    total = 0.0
    for j, start in enumerate(order):
        end = order[j + 1] if j + 1 < len(order) else direction * W

        def f(x, start=start):
            w = math.exp(-abs(x - start) / lam1) if math.isfinite(lam1) else 1.0
            return w * _omega(provider, x, t, conv.berry)

        lo, hi = min(start, end), max(start, end)
        pts = [p for p in (0.0,) if lo < p < hi]
        val = quad(f, lo, hi, points=pts or None, epsrel=epsrel, epsabs=0.0, limit=400)[0]
        total += rho[j] * val
    return total


def transverse_breaks(provider, cfg: RunConfig, n_scan: int = 400) -> list:
    """Tangency points (where the number of crossings changes) on both sides, scanned independently."""
    W = cfg.crossings.window
    out = []
    for sign in (+1, -1):
        mags = np.geomspace(1e-9 * W, W, n_scan)
        n = [_crossings(provider, sign * m, cfg)[0].size for m in mags]
        for i in range(mags.size - 1):
            if n[i] == n[i + 1]:
                continue
            lo, hi = mags[i], mags[i + 1]
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if _crossings(provider, sign * mid, cfg)[0].size == n[i]:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 1e-14 * W:
                    break
            out.append(sign * 0.5 * (lo + hi))
    return sorted(out)


def naive_jh(model, E: float, cfg: RunConfig = PAPER, t_breaks=None, epsrel: float = 1e-8):
    """J_H(E) in A/m by nested quad over the whole transverse window [-W, W].

    t_breaks: tangency points where the integrand jumps; found by `transverse_breaks` when None.
    """
    _check(cfg)
    provider = ModelGeometry(model, cfg.drive.angle, k_patch=cfg.quadrature.k_patch)
    G = hall_prefactor(cfg.conventions.prefactor) * cfg.conventions.figure_scale * provider.spin_degeneracy
    W = cfg.crossings.window
    t_breaks = transverse_breaks(provider, cfg) if t_breaks is None else list(t_breaks)
    edges = sorted({-W, W, 0.0, *t_breaks})
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        total += quad(
            lambda t: line_integrand(provider, t, E, cfg, epsrel=1e-10),
            a,
            b,
            epsrel=epsrel,
            epsabs=0.0,
            limit=200,
        )[0]
    return G * E * total
