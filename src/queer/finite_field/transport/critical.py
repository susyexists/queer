"""Critical-point analysis: harmonic universality and the critical Berry-curvature strength C.

At VL = VLc the gap closes at Gamma with eps_g ~ 2 hbar_v |k| and Omega_c ~ C_s/|k| (signed; the
four-band model has C_s = -140.45 A). On the line at offset t the crossing sits at s = 0 with
E_c(t) = pi hbar_v t^2 / e, so P_LZ = exp(-t^2/delta^2), delta = dky/sqrt(pi), dky = sqrt(e|E|/(hbar_v)).

Main text (SI.VI.3):   J_H = C dky (e^2/h) E   ->   J ~ |E|^{3/2}, N J^{Nw} universal in E/N.
With the full 1/k curvature and no relaxation (T1 = inf) the line integral is C_s ln(2 k_c/|t|), giving
    J_H ~= (e^2/h) C_s dky E [ln(2 k_c/delta) + (gamma_E + 2 ln 2)/2],
a log-corrected 3/2 law (local exponent ~1.4, as in Ext. Data Fig. 9b). With finite T1 the relaxation
length lambda = e|E|T1/hbar replaces k_c; for lambda << delta the response crosses over to ~E^2 ln.

Methods 5 inverts the critical harmonics for C. The asymptotic constant c_F = 1.1968 is used in the
paper and by default; `finite_N=True` uses the exact beta_N(3/2) (c_F is 32% off at N = 3). The
inversions need the normalization that produced b_N (prefactor and conventions.figure_scale; pass cfg).
"""

from __future__ import annotations

import math

import numpy as np

from ..constants import E2_H
from ..units import V_PER_M_TO_EV_PER_A, delta_ky, hall_prefactor
from .quadrature import logistic_nodes

EULER_GAMMA = 0.5772156649015329


def critical_constants(provider, k=None, angles=(0.0, 0.7)) -> dict:
    """hbar_v = lim eps_g/(2k) and C_s = lim k * Omega_c at small k (provider at the critical point).

    Returns the values at the smallest k, their spread over k (convergence) and over directions
    (isotropy), in eV*A and A.
    """
    k = np.geomspace(1e-7, 1e-6, 5) if k is None else np.asarray(k, dtype=float)
    hv, cs = [], []
    for a in angles:
        kx, ky = k * math.cos(a), k * math.sin(a)
        geo = provider.at_cartesian(kx, ky)
        hv.append(geo.eps_g / (2.0 * k))
        cs.append(k * geo.omega_c)
    hv, cs = np.array(hv), np.array(cs)
    return {
        "hbar_v": float(hv[0, 0]),
        "C_s": float(cs[0, 0]),
        "C_paper": -float(cs[0, 0]),
        "hbar_v_spread": float(np.ptp(hv) / abs(hv[0, 0])),
        "C_spread": float(np.ptp(cs) / abs(cs[0, 0])),
    }


def critical_kc(provider, C_s: float, t: float = 1e-8, window: float = 0.03, du: float = 0.1) -> float:
    """Effective cutoff k_c of the 1/k curvature: Int_0^W Omega_c(s, t) ds = C_s ln(2 k_c / t) for t -> 0."""
    s, w = logistic_nodes(0.0, window, du, d_a=1e-3 * t, d_b=1e-6 * window)
    geo = provider.geometry(s, np.full_like(s, t))
    integral = float(np.sum(w * geo.omega_c))
    return 0.5 * t * math.exp(integral / C_s)


def j_critical_analytic(E, C_s: float, hbar_v: float, prefactor: str = "e2_h") -> np.ndarray:
    """Main-text critical law J = C dky (e^2/h) E (signed C_s, sgn E), A/m."""
    E = np.asarray(E, dtype=float)
    return hall_prefactor(prefactor) * C_s * delta_ky(E, hbar_v) * E


def j_critical_log(E, C_s: float, hbar_v: float, k_c: float, prefactor: str = "e2_h") -> np.ndarray:
    """Log-corrected T1 = inf asymptote (valid for delta << k_c), A/m."""
    E = np.asarray(E, dtype=float)
    dk = delta_ky(E, hbar_v)
    delta = dk / math.sqrt(math.pi)
    return (
        hall_prefactor(prefactor)
        * C_s
        * dk
        * E
        * (np.log(2.0 * k_c / delta) + 0.5 * (EULER_GAMMA + 2.0 * math.log(2.0)))
    )


def critical_harmonics_analytic(E0, N, C_s: float, hbar_v: float, prefactor: str = "e2_h") -> np.ndarray:
    """Sine harmonics b_N(E0) of the main-text critical law (exact finite-N factors), shape (nE0, nN)."""
    from ..harmonics import power_law_bN

    E0 = np.atleast_1d(np.asarray(E0, dtype=float))
    N = np.atleast_1d(np.asarray(N))
    K = hall_prefactor(prefactor) * C_s * math.sqrt(V_PER_M_TO_EV_PER_A / hbar_v)  # J = K sgn(E)|E|^{3/2}
    return K * E0[:, None] ** 1.5 * power_law_bN(N, 1.5)[None, :]


def _odd_orders(N) -> np.ndarray:
    N = np.asarray(N)
    if np.any(N != np.round(N)) or np.any(N < 1) or np.any(np.round(N) % 2 == 0):
        raise ValueError(f"critical harmonics are odd orders N >= 1 (b_N = 0 for even N), got {N.tolist()}")
    return N


def c_from_harmonic(
    N,
    b_N,
    E0,
    hbar_v: float,
    prefactor: str = "e2_h",
    finite_N: bool = False,
    figure_scale: float = 1.0,
    cfg=None,
) -> np.ndarray:
    """Methods 5 inversion of a critical harmonic: C = b_N / (s beta_N sqrt(e/(hbar v)) G E0^{3/2}).

    G = e^2/h ("e2_h") or e^2/(2 pi h) ("e2_2pi_h") and s = figure_scale must be those that produced b_N;
    cfg (a RunConfig) supplies both from cfg.conventions (b_N of PAPER_FIGURES-based presets carry
    s = 1/(4 pi)). Odd N only. With finite_N=False the paper's asymptotic factor c_F N^{-5/2} replaces
    |beta_N| (beta_1 is positive and every odd beta_N for N >= 3 is negative); C then comes out
    32% too large at N = 3 and 24% too small at N = 1.
    """
    from ..harmonics import C_F, power_law_bN

    berry = 1.0
    if cfg is not None:
        prefactor, figure_scale = cfg.conventions.prefactor, cfg.conventions.figure_scale
        berry = {"c": 1.0, "c_minus_v": 2.0, "v": -1.0}[cfg.conventions.berry]
    N = _odd_orders(N)
    exact = power_law_bN(N, 1.5)
    beta = exact if finite_N else np.sign(exact) * C_F * np.asarray(N, dtype=float) ** -2.5
    K = berry * hall_prefactor(prefactor) * figure_scale * math.sqrt(V_PER_M_TO_EV_PER_A / hbar_v)
    return np.asarray(b_N, dtype=float) / (beta * K * np.asarray(E0, dtype=float) ** 1.5)


def c_from_slope(
    slope: float,
    L: float,
    W: float,
    R_xx: float | None,
    hbar_v: float,
    *,
    rho_xx: float | None = None,
) -> float:
    """Methods 5 |C| in Å from the fitted slope of N V_N against (I/N)**1.5.

    The fitted intercept is excluded. Resistance and geometry use the same SI
    conventions as c_from_device; hbar_v is in eV Å.
    """
    from ..harmonics import C_F

    if rho_xx is not None:
        if R_xx is not None:
            raise ValueError("give R_xx or rho_xx, not both")
        R_xx = rho_xx * L / W
    elif R_xx is None:
        raise ValueError("R_xx (longitudinal resistance) or rho_xx (sheet resistance) is required")
    sqrt_e_hv = math.sqrt(1.0 / (hbar_v * 1e-10))
    return float(abs(slope) * L**2.5 / (C_F * W**2 * R_xx**2.5) / (sqrt_e_hv * E2_H) * 1e10)


def c_from_device(
    N,
    V_N,
    current,
    L: float,
    W: float,
    R_xx: float | None,
    hbar_v: float,
    finite_N: bool = False,
    *,
    rho_xx: float | None = None,
) -> np.ndarray:
    """Experimental |C| (A) from the N-th harmonic Hall voltage amplitude (Methods 5).

    |C| = [1/(sqrt(e/(hbar v)) e^2/h)] (|V_N| N)/((current/N)^{3/2}) L^{5/2} / (c_F W^2 R_xx^{5/2}), SI inputs:
    V_N in V, current in A (current amplitude), L (voltage-probe distance) and W (channel width) in m, hbar_v
    in eV*A. R_xx is the longitudinal resistance V_xx/current at the critical point in Ohm (= rho_xx L/W, not
    the sheet resistance); alternatively give the sheet resistance rho_xx (Ohm per square) and R_xx=None.
    Uses J_N = V_N L/(R_xx W^2) and E = current R_xx / L. Like the paper it works with amplitudes (critical
    harmonics N >= 3 carry the opposite sign to C). Odd N only. finite_N=True replaces c_F by
    |beta_N(3/2)| N^{5/2}.
    """
    from ..harmonics import C_F, power_law_bN

    if rho_xx is not None:
        if R_xx is not None:
            raise ValueError("give R_xx or rho_xx, not both")
        R_xx = rho_xx * L / W
    elif R_xx is None:
        raise ValueError(
            "R_xx (longitudinal resistance, Ohm) or rho_xx (sheet resistance, Ohm/sq) is required"
        )
    N = np.asarray(_odd_orders(N), dtype=float)
    cf = np.abs(power_law_bN(N.astype(int), 1.5)) * N**2.5 if finite_N else C_F
    hbar_v_si = hbar_v * 1e-10  # eV*m: e/(hbar v) = 1/hbar_v_si in 1/(V*m) with e = 1 per eV
    sqrt_e_hv = math.sqrt(1.0 / hbar_v_si)  # (V m)^{-1/2}
    C_m = (
        (np.abs(np.asarray(V_N)) * N / (np.asarray(current) / N) ** 1.5)
        * L**2.5
        / (cf * W**2 * R_xx**2.5)
        / (sqrt_e_hv * E2_H)
    )
    return C_m * 1e10  # m -> A
