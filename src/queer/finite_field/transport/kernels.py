"""Fused numba kernels for the response stage of Eqs. 10-11 (one and two crossings per line).

For every field E the kernel loops over lines, evaluates P_LZ (Eq. 7), the two-crossing population
(Eq. 12 generalized to P1 != P2) and the quintic-Hermite lambda-tables without temporaries. They
reproduce the numpy path of HallCalculator bit-for-bit up to floating-point reassociation
(fastmath is off) and are used when numba is available.
"""

from __future__ import annotations

import math

import numpy as np

from ..lz.propagation import two_crossing_population

try:
    import numba
    from numba import prange

    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    prange = range
    _HAVE_NUMBA = False


def _interp(tab, finf, m1, x0, h, K, lam_lo, lam_hi, lam):
    if not math.isfinite(lam):
        return finf
    if lam > lam_hi:
        return finf - m1 / lam
    if lam < lam_lo:
        return tab[0, 0] * (lam / lam_lo)
    pos = (math.log(lam) - x0) / h
    k = int(math.floor(pos))
    if k < 0:
        k = 0
    elif k > K - 2:
        k = K - 2
    t = pos - k
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    h0 = 1.0 - 10.0 * t3 + 15.0 * t4 - 6.0 * t5
    h1 = t - 6.0 * t3 + 8.0 * t4 - 3.0 * t5
    h2 = 0.5 * (t2 - 3.0 * t3 + 3.0 * t4 - t5)
    h3 = 0.5 * (t3 - 2.0 * t4 + t5)
    h4 = -4.0 * t3 + 7.0 * t4 - 3.0 * t5
    h5 = 10.0 * t3 - 15.0 * t4 + 6.0 * t5
    return (
        tab[k, 0] * h0
        + tab[k + 1, 0] * h5
        + h * (tab[k, 1] * h1 + tab[k + 1, 1] * h4)
        + h * h * (tab[k, 2] * h2 + tab[k + 1, 2] * h3)
    )


def _kernel_n1(Ea, lam1, w, Ec, tab, finf, m1, x0, h, K, lam_lo, lam_hi, out):
    nE = Ea.shape[0]
    L = w.shape[0]
    for e in prange(nE):
        acc = 0.0
        E = Ea[e]
        for line in range(L):
            ec = Ec[line, 0]
            if ec > 700.0 * E:
                continue
            P = math.exp(-ec / E)
            acc += (
                w[line]
                * P
                * _interp_j(tab[line, 0], finf[line, 0], m1[line, 0], x0, h, K, lam_lo, lam_hi, lam1[e])
            )
        out[e] = acc


def _kernel_n2(Ea, lam1, lam2, w, Ec, gaps, dseg, tab, finf, m1, x0, h, K, lam_lo, lam_hi, paper_coh, out):
    nE = Ea.shape[0]
    L = w.shape[0]
    for e in prange(nE):
        acc = 0.0
        E = Ea[e]
        l1 = lam1[e]
        l2 = lam2[e]
        for line in range(L):
            P1 = math.exp(-Ec[line, 0] / E) if Ec[line, 0] < 700.0 * E else 0.0
            P2 = math.exp(-Ec[line, 1] / E) if Ec[line, 1] < 700.0 * E else 0.0
            if P1 == 0.0 and P2 == 0.0:
                continue
            d = dseg[line, 0]
            pop = math.exp(-d / l1)
            coh = math.exp(-d / l2)
            if paper_coh:
                coh *= math.exp(-d / (2.0 * l1))
            theta = gaps[line, 0] * 1e10 / E
            rho2 = _population_j(P1, P2, pop, coh, theta)
            F1 = _interp_j(tab[line, 0], finf[line, 0], m1[line, 0], x0, h, K, lam_lo, lam_hi, l1)
            F2 = _interp_j(tab[line, 1], finf[line, 1], m1[line, 1], x0, h, K, lam_lo, lam_hi, l1)
            acc += w[line] * (P1 * F1 + rho2 * F2)
        out[e] = acc


if _HAVE_NUMBA:
    _interp_j = numba.njit(cache=True, fastmath=False)(_interp)
    _population_j = numba.njit(cache=True, fastmath=False)(two_crossing_population)
    kernel_n1 = numba.njit(cache=True, parallel=True, fastmath=False)(_kernel_n1)
    kernel_n2 = numba.njit(cache=True, parallel=True, fastmath=False)(_kernel_n2)
else:  # pragma: no cover
    _interp_j = _interp
    _population_j = two_crossing_population
    kernel_n1 = _kernel_n1
    kernel_n2 = _kernel_n2

HAVE_NUMBA = _HAVE_NUMBA


def run_group(n, Ea, lam1, lam2, w, Ec, gaps, dseg, tab, finf, m1, grid, coherence: str) -> np.ndarray:
    """Sum over the lines of one group for all fields; returns (nE,)."""
    out = np.empty(Ea.size)
    args = (grid.x0, grid.h, grid.K, grid.lam_lo, grid.lam_hi)
    tab = np.ascontiguousarray(tab)
    if n == 1:
        kernel_n1(
            Ea,
            lam1,
            w,
            np.ascontiguousarray(Ec),
            tab,
            np.ascontiguousarray(finf),
            np.ascontiguousarray(m1),
            *args,
            out,
        )
    elif n == 2:
        kernel_n2(
            Ea,
            lam1,
            lam2,
            w,
            np.ascontiguousarray(Ec),
            np.ascontiguousarray(gaps),
            np.ascontiguousarray(dseg),
            tab,
            np.ascontiguousarray(finf),
            np.ascontiguousarray(m1),
            *args,
            coherence == "paper",
            out,
        )
    else:
        raise ValueError("fused kernels cover one and two crossings per line")
    return out
