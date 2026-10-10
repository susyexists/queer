"""Methods 4.1-4.2 analysis: harmonic onset fields E_c^N, oscillation frequencies f_N, and E_c, Lambda.

Methods 4.1 fits each harmonic with a smooth activated form plus a 1/E-periodic oscillation:
    V_smooth = A exp(-E_c^N/E) E^{3/2} [1 + (E/E_s)^p]^{-3/(2p)}   (saturation only if requested)
    V_osc    = V - V_smooth = A_osc cos(2 pi f_N / E + phi_N).
Methods 4.2 then uses E_c^N = a_F E_c N (linear in N) and f_N = Lambda + A_f N^2 (small N).

The analysis workflow showed that both extractions depend on the fit windows: a_F runs from 0.40 to
0.05 as the window grows from 0.5 to 2 N E_c (the paper quotes 0.14; its own Ext. Data Fig. 7d data
imply 0.110), and f_N follows the stationary-phase law f_N = Lambda / sin(theta*) with
cos(theta*)/sin(theta*)^2 = N E0 / (2 pi Lambda), quadratic in N only for N << 2 pi Lambda/E0. Fit
windows are therefore explicit inputs here, and the stationary-phase prediction is provided.
Works for fields (E0, V/m) or currents (I, A) alike.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import OptimizeWarning, brentq
from scipy.optimize import curve_fit as _curve_fit


def curve_fit(*args, **kw):
    """scipy curve_fit without the (unused) covariance warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        return _curve_fit(*args, **kw)


@dataclass
class SmoothFit:
    A: float
    Ec_N: float
    E_s: float | None
    p: float | None
    window: tuple
    residual_rms: float


def _smooth(E, A, Ec, E_s=None, p=None):
    out = A * np.exp(-Ec / E) * E**1.5
    if E_s is not None:
        out = out * (1.0 + (E / E_s) ** p) ** (-1.5 / p)
    return out


def fit_smooth(E, V, window, saturation: bool = False, Ec_guess: float | None = None) -> SmoothFit:
    """Fit the Methods 4.1 smooth form on E in [window[0], window[1]] (V may have either sign)."""
    E = np.asarray(E, float)
    V = np.asarray(V, float)
    m = (E >= window[0]) & (E <= window[1])
    e, v = E[m], V[m]
    sign = np.sign(np.sum(v)) or 1.0
    v = sign * v
    Ec0 = Ec_guess if Ec_guess is not None else 0.3 * float(np.median(e))
    A0 = float(np.max(np.abs(v)) / (np.exp(-Ec0 / e[-1]) * e[-1] ** 1.5))
    if saturation:
        f = lambda x, A, Ec, Es, p: _smooth(x, A, Ec, Es, p)  # noqa: E731
        p0 = [A0, Ec0, float(e[-1]), 2.0]
        bounds = ([0, 0, 0, 0.3], [np.inf, np.inf, np.inf, 20])
    else:
        f = lambda x, A, Ec: _smooth(x, A, Ec)  # noqa: E731
        p0 = [A0, Ec0]
        bounds = ([0, 0], [np.inf, np.inf])
    popt, _ = curve_fit(f, e, v, p0=p0, bounds=bounds, maxfev=20000)
    res = v - f(e, *popt)
    return SmoothFit(
        float(sign * popt[0]),
        float(popt[1]),
        float(popt[2]) if saturation else None,
        float(popt[3]) if saturation else None,
        tuple(window),
        float(np.sqrt(np.mean(res**2)) / np.max(np.abs(v))),
    )


def fit_oscillation(
    E, V_osc, f_guess: float | None = None, f_bounds: tuple | None = None, n_scan: int = 4000
):
    """f_N, amplitude and phase of V_osc = A cos(2 pi f/E + phi) + c (u = 1/E).

    Without f_guess, f is chosen by a least-squares scan over f_bounds (default: from a quarter period
    to the Nyquist limit of the u sampling), solving the linear amplitudes at each f; this is robust
    when the window holds only a few periods, where periodogram peaks are broad. Then refined.
    """
    E = np.asarray(E, float)
    u = 1.0 / E
    y = np.asarray(V_osc, float)
    if f_guess is None:
        lo, hi = (
            f_bounds
            if f_bounds is not None
            else (0.25 / np.ptp(u), 0.5 / np.min(np.abs(np.diff(np.sort(u)))))
        )
        fs = np.linspace(lo, hi, n_scan)
        best, f_guess = np.inf, fs[0]
        for fq in fs:
            Amat = np.stack(
                [np.cos(2 * math.pi * fq * u), np.sin(2 * math.pi * fq * u), np.ones_like(u)], axis=1
            )
            coef, *_ = np.linalg.lstsq(Amat, y, rcond=None)
            r = float(np.sum((Amat @ coef - y) ** 2))
            if r < best:
                best, f_guess = r, fq
    c = np.sum(y * np.cos(2 * math.pi * f_guess * u))
    s = np.sum(y * np.sin(2 * math.pi * f_guess * u))
    A0, ph0 = math.hypot(c, s) * 2 / y.size, math.atan2(-s, c)
    popt, _ = curve_fit(
        lambda uu, A, f, ph: A * np.cos(2 * math.pi * f * uu + ph), u, y, p0=[A0, f_guess, ph0], maxfev=20000
    )
    A, f, ph = popt
    if A < 0:
        A, ph = -A, ph + math.pi
    return {"f": float(f), "amplitude": float(A), "phase": float(math.remainder(ph, 2 * math.pi))}


def fit_harmonic(E, V, window, f_bounds=None):
    """Joint Methods 4.1 fit of one harmonic: A exp(-E_c^N/E) E^{3/2} + B cos(2 pi f_N/E + phi).

    Initialized from fit_smooth and a least-squares f scan of its residual; fitting both parts
    together keeps the smooth form from absorbing the oscillation. Returns a dict.
    """
    E = np.asarray(E, float)
    V = np.asarray(V, float)
    m = (E >= window[0]) & (E <= window[1])
    e, v = E[m], V[m]
    sm = fit_smooth(e, v, (e[0], e[-1]))
    osc = fit_oscillation(e, v - _smooth(e, sm.A, sm.Ec_N), f_bounds=f_bounds)
    scale = float(np.max(np.abs(v)))

    def model(x, A, Ec, B, f, ph):
        return A * np.exp(-Ec / x) * x**1.5 + B * np.cos(2 * math.pi * f / x + ph)

    p0 = [sm.A / scale, sm.Ec_N, osc["amplitude"] / scale, osc["f"], osc["phase"]]
    popt, _ = curve_fit(
        lambda x, A, Ec, B, f, ph: model(x, A * scale, Ec, B * scale, f, ph) / scale,
        e,
        v / scale,
        p0=p0,
        maxfev=40000,
    )
    A, Ec, B, f, ph = popt
    if B < 0:
        B, ph = -B, ph + math.pi
    res = v / scale - model(e, A * scale, Ec, B * scale, f, ph) / scale
    return {
        "A": float(A * scale),
        "Ec_N": float(Ec),
        "B": float(B * scale),
        "f": float(f),
        "phase": float(math.remainder(ph, 2 * math.pi)),
        "residual_rms": float(np.sqrt(np.mean(res**2))),
        "window": tuple(window),
    }


def onset_slope(N, Ec_N):
    """Linear fit E_c^N = a + slope N; slope = a_F E_c (Methods 4.2)."""
    slope, intercept = np.polyfit(np.asarray(N, float), np.asarray(Ec_N, float), 1)
    return {"slope": float(slope), "intercept": float(intercept)}


def lambda_intercept(N, f_N, small=(3, 5, 7)):
    """Quadratic fit f_N = Lambda + A N^2 on the small-N subset (Methods 4.2)."""
    N = np.asarray(N, float)
    f_N = np.asarray(f_N, float)
    m = np.isin(N, small)
    A, Lam = np.polyfit(N[m] ** 2, f_N[m], 1)
    return {"Lambda": float(Lam), "A": float(A)}


def stationary_phase_fN(Lambda: float, N, E0: float):
    """f_N = Lambda / sin(theta*) with cos(theta*)/sin^2(theta*) = N E0/(2 pi Lambda)."""
    out = []
    for n in np.atleast_1d(N):
        c = n * E0 / (2 * math.pi * Lambda)
        th = brentq(lambda t: math.cos(t) / math.sin(t) ** 2 - c, 1e-9, 0.5 * math.pi)
        out.append(Lambda / math.sin(th))
    return np.array(out)
