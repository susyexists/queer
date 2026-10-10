"""Eqs. 7, 8, 12 and 13 of HHH_v13 exactly as printed (pure, vectorized reference formulas).

    Eq. 7    P_LZ  = exp(-E_c / |E|)
    Eq. 8    E_c   = pi eps_g / (4 |e| sqrt(g_xx))                          (at the avoided crossing)
    Eq. 12   P_int = P [1 + (1 - 2P) e^{-2 hbar k0/(e|E| T1)}
                          + 2 (1 - P) e^{-2 hbar k0/(2 e|E| T1)} e^{-2 hbar k0/(e|E| T2)} cos Phi],
             Phi   = 2 pi Lambda / |E|                     (unnumbered, after Eq. 11; main text l. 175)
    Eq. 13   J^{N w}   = (1/2pi) Int_0^{2pi} d theta J_H(E0 sin theta) e^{-i N theta}    (Methods 8)
             J^{N w}_y = (1/pi)  Int_0^{2pi} d theta J_H(E0 sin theta) sin(N theta)   (main text l. 178)

Units: fields E, E_c and Lambda in V/m; eps_g in eV; g_xx in A^2; k0 (half-separation of the two
crossings at +-k0) in 1/A; T1, T2 in s (math.inf allowed). hbar k0 / (e|E| T) = k0 / kappa with
kappa = e|E|T/hbar from `units.kappa`. The printed formulas contain E_x; its magnitude is used in
the exponents and in the phase, so every function accepts either sign of E.

All functions broadcast their arguments and return numpy scalars for scalar input. The two Fourier
forms use the periodic trapezoid rule on [0, 2 pi) (one real FFT), which is spectrally accurate for
smooth J. They are cross-checks of the production projector in `harmonics.py`.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from . import units


def p_lz(E_c: ArrayLike, E: ArrayLike) -> np.ndarray:
    """Eq. 7: Landau-Zener tunnelling probability exp(-E_c/|E|) (E_c, E in V/m).

    Limits: E = 0 gives 0 if E_c > 0 and 1 if E_c = 0; a touching point (E_c = 0) gives 1 for every E.
    """
    E_c, E = np.broadcast_arrays(np.asarray(E_c, dtype=float), np.asarray(E, dtype=float))
    if np.any(E_c < 0):
        raise ValueError("E_c must be non-negative")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = np.exp(-E_c / np.abs(E))  # E = 0 < E_c: exp(-inf) = 0
    return np.where(E_c == 0.0, 1.0, out)[()]  # touching point, including E = 0


def critical_field(eps_g: ArrayLike, g_xx: ArrayLike) -> np.ndarray:
    """Eq. 8: E_c = pi eps_g / (4 |e| sqrt(g_xx)) in V/m, eps_g in eV and g_xx in A^2 at the crossing."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return units.critical_field(eps_g, g_xx)[()]


def phase(Lambda: ArrayLike, E: ArrayLike) -> np.ndarray:
    """Interference phase Phi = 2 pi Lambda / |E| (Lambda, E in V/m); inf at E = 0, and 0 where Lambda = 0."""
    Lambda, E = np.broadcast_arrays(np.asarray(Lambda, dtype=float), np.asarray(E, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 2.0 * math.pi * Lambda / np.abs(E)
    return np.where(Lambda == 0.0, 0.0, out)[()]


def _decay(dk, E, T):
    """exp(-hbar dk / (e|E| T)) = exp(-dk / kappa(E, T)) for a k-distance dk >= 0 (1/A).

    Limits: 1 for T = inf (as printed: no relaxation) or dk = 0; 0 for E = 0 < dk with finite T.
    """
    dk, E, T = np.broadcast_arrays(
        np.asarray(dk, dtype=float), np.asarray(E, dtype=float), np.asarray(T, dtype=float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.exp(-dk / units.kappa(E, T))
    return np.where(np.isposinf(T) | (dk == 0.0), 1.0, out)


def p_int(
    P: ArrayLike, k0: ArrayLike, E: ArrayLike, T1: ArrayLike, T2: ArrayLike, Phi: ArrayLike
) -> np.ndarray:
    """Eq. 12 (unnumbered, after Eq. 11) exactly as printed: conduction population behind two crossings.

    P_int = P [1 + (1 - 2P) e^{-2 hbar k0/(e|E| T1)} + 2(1 - P) e^{-2 hbar k0/(2 e|E| T1)}
               e^{-2 hbar k0/(e|E| T2)} cos Phi]

    P: LZ probability of each crossing (Eq. 7); k0: half-separation of the crossings at +-k0 (1/A);
    E in V/m; T1, T2 in s (math.inf allowed, the exponential is then 1); Phi: interference phase (rad).
    Limits: 0 for P = 0, P for k0 -> inf, 2P(1 - P)(1 + cos Phi) for T1 = T2 = inf, and
    1 - e^{-2 hbar k0/(e|E| T1)} for P = 1. A vanishing coefficient of cos Phi removes the term even
    when Phi is not finite (Phi = inf at E = 0).
    """
    P, k0, E, T1, T2, Phi = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (P, k0, E, T1, T2, Phi))
    )
    if np.any(k0 < 0):
        raise ValueError("k0 (half-separation of the crossings) must be non-negative")
    if np.any(T1 < 0) or np.any(T2 < 0):
        raise ValueError("T1 and T2 must be non-negative (math.inf for no relaxation)")
    population = _decay(2.0 * k0, E, T1)  # e^{-2 hbar k0/(e|E| T1)}
    coherence = _decay(2.0 * k0, E, 2.0 * T1) * _decay(2.0 * k0, E, T2)  # e^{-2hk0/(2eET1)} e^{-2hk0/(eET2)}
    coeff = 2.0 * (1.0 - P) * coherence
    with np.errstate(invalid="ignore"):
        interference = np.where(coeff == 0.0, 0.0, coeff * np.cos(Phi))
    out = P * (1.0 + (1.0 - 2.0 * P) * population + interference)
    return np.where(P == 0.0, 0.0, out)[()]


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


def _trapezoid_spectrum(
    J: Callable, E0: ArrayLike, N: ArrayLike, n_theta: int
) -> tuple[np.ndarray, np.ndarray]:
    """(1/n) sum_k J(E0 sin theta_k) e^{-i |N| theta_k} on the half-shifted periodic grid
    theta_k = 2 pi (k + 1/2)/n (spectrally accurate like the unshifted rule, but never samples E = 0,
    where a naively written activated J is undefined); shape E0.shape + N.shape."""
    E0 = np.asarray(E0, dtype=float)
    N = np.asarray(N)
    if not np.all(N == np.round(N)):
        raise ValueError("harmonic orders N must be integers")
    absN = np.abs(N).astype(int)
    n_theta = int(n_theta)
    if absN.size and np.max(absN) >= n_theta // 2:
        raise ValueError(f"|N| must be below n_theta/2 = {n_theta // 2}")
    theta = 2.0 * math.pi * (np.arange(n_theta) + 0.5) / n_theta
    f = _sample(J, E0[..., None] * np.sin(theta))
    spectrum = np.fft.rfft(f, axis=-1) / n_theta  # (E0.shape..., n_theta//2 + 1)
    shift = np.exp(-1j * math.pi * np.arange(spectrum.shape[-1]) / n_theta)  # e^{-i N pi/n}: grid offset
    return (spectrum * shift)[..., absN], N


def jn_methods(J: Callable, E0: ArrayLike, N: ArrayLike, n_theta: int = 4096) -> np.ndarray:
    """Eq. 13 literally: c_N = (1/2pi) Int_0^{2pi} J(E0 sin theta) e^{-i N theta} d theta (complex).

    J: vectorized callable of the field (V/m, either sign). E0 in V/m. Periodic trapezoid rule with
    n_theta points (|N| < n_theta/2); c_{-N} = conj(c_N). Returns shape E0.shape + N.shape.
    """
    c, N = _trapezoid_spectrum(J, E0, N, n_theta)
    return np.where(N < 0, np.conj(c), c)[()]


def jn_main_text(J: Callable, E0: ArrayLike, N: ArrayLike, n_theta: int = 4096) -> np.ndarray:
    """Main text l. 178: b_N = (1/pi) Int_0^{2pi} J(E0 sin theta) sin(N theta) d theta (real; = -2 Im c_N).

    Same rule and shapes as `jn_methods`; b_{-N} = -b_N.
    """
    c, N = _trapezoid_spectrum(J, E0, N, n_theta)
    return (-2.0 * np.sign(N) * c.imag)[()]
