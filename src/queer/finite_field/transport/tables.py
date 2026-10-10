"""Laplace lambda-tables for the relaxation-weighted Berry-curvature integrals of Eqs. 10-11.

For a segment that starts at the crossing s_j, the line integral with relaxation length
lambda = e|E|T1/hbar is

    F(lambda) = Int_seg exp(-|s - s_j|/lambda) Omega(s) ds  ~=  sum_i c_i exp(-r_i/lambda),
    c_i = w_i Omega(s_i), r_i = |s_i - s_j|.

F depends on E and T1 only through lambda, so it is tabulated once per line on a uniform grid in
x = ln(lambda) together with the exact derivatives
    dF/dx = sum c e^{-q} q,   d2F/dx2 = sum c e^{-q} q (q - 1),   q = r/lambda,
and interpolated with C2 quintic Hermite polynomials (smooth J_H(E) keeps the harmonic projection
spectrally accurate). Outside the grid: F ~ F(lambda_lo) lambda/lambda_lo for lambda < lambda_lo
(Watson's lemma, F ~ lambda Omega(s_j+)), and F = F_inf - M1/lambda + O(lambda^-2) above, with
F_inf = sum c and M1 = sum c r.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LambdaGrid:
    x0: float  # ln(lambda_lo)
    h: float  # step in ln(lambda)
    K: int

    @classmethod
    def make(cls, lam_lo: float, lam_hi: float, per_decade: int = 12) -> "LambdaGrid":
        n_dec = math.log10(lam_hi / lam_lo)
        K = max(4, int(math.ceil(n_dec * per_decade)) + 1)
        x0 = math.log(lam_lo)
        h = (math.log(lam_hi) - x0) / (K - 1)
        return cls(x0, h, K)

    @property
    def lam(self) -> np.ndarray:
        return np.exp(self.x0 + self.h * np.arange(self.K))

    @property
    def lam_lo(self) -> float:
        return math.exp(self.x0)

    @property
    def lam_hi(self) -> float:
        return math.exp(self.x0 + self.h * (self.K - 1))


def build_table(r: np.ndarray, c: np.ndarray, grid: LambdaGrid):
    """(table (K, 3) = [F, dF/dx, d2F/dx2], F_inf, M1) for one segment."""
    lam = grid.lam
    if r.size == 0:
        return np.zeros((grid.K, 3)), 0.0, 0.0
    q = r[:, None] / lam[None, :]
    e = np.exp(-q)
    ce = c[:, None] * e
    F = ce.sum(0)
    F1 = (ce * q).sum(0)
    F2 = (ce * q * (q - 1.0)).sum(0)
    return np.stack([F, F1, F2], axis=-1), float(c.sum()), float((c * r).sum())


def eval_tables(tab: np.ndarray, F_inf: np.ndarray, M1: np.ndarray, grid: LambdaGrid, lam) -> np.ndarray:
    """Evaluate tables (..., K, 3) at lambda (nE,), returning (..., nE). lambda may be inf."""
    lam = np.asarray(lam, dtype=float)
    lead = tab.shape[:-2]
    tab2 = tab.reshape((-1, grid.K, 3))
    Finf = np.asarray(F_inf, dtype=float).reshape(-1)
    m1 = np.asarray(M1, dtype=float).reshape(-1)
    out = np.empty((tab2.shape[0], lam.size))
    inf = ~np.isfinite(lam)
    lo = (~inf) & (lam < grid.lam_lo)
    hi = (~inf) & (lam > grid.lam_hi)
    mid = ~(inf | lo | hi)
    if np.any(inf):
        out[:, inf] = Finf[:, None]
    if np.any(hi):
        out[:, hi] = Finf[:, None] - m1[:, None] / lam[hi][None, :]
    if np.any(lo):
        out[:, lo] = tab2[:, 0, 0][:, None] * (lam[lo] / grid.lam_lo)[None, :]
    if np.any(mid):
        pos = (np.log(lam[mid]) - grid.x0) / grid.h
        k = np.clip(np.floor(pos).astype(int), 0, grid.K - 2)
        t = pos - k
        t2, t3 = t * t, t * t * t
        t4, t5 = t3 * t, t3 * t2
        h0 = 1 - 10 * t3 + 15 * t4 - 6 * t5
        h1 = t - 6 * t3 + 8 * t4 - 3 * t5
        h2 = 0.5 * (t2 - 3 * t3 + 3 * t4 - t5)
        h3 = 0.5 * (t3 - 2 * t4 + t5)
        h4 = -4 * t3 + 7 * t4 - 3 * t5
        h5 = 10 * t3 - 15 * t4 + 6 * t5
        a, b = tab2[:, k, :], tab2[:, k + 1, :]  # (n_tab, n_mid, 3)
        h = grid.h
        out[:, mid] = (
            a[..., 0] * h0
            + b[..., 0] * h5
            + h * (a[..., 1] * h1 + b[..., 1] * h4)
            + h * h * (a[..., 2] * h2 + b[..., 2] * h3)
        )
    return out.reshape(lead + (lam.size,))
