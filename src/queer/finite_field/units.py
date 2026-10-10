"""Unit conversions between the internal units and SI.

Internal units: energy eV, length Angstrom, wave vector 1/Angstrom. Fields are given in V/m and
times in s at the API boundary. With energies in eV the elementary charge is 1, so e*E [eV/m]
equals E [V/m], and e*E [eV/A] = 1e-10 * E [V/m].
"""

import math

import numpy as np

from .constants import E2_H, HBAR_EVS

V_PER_M_TO_EV_PER_A = 1e-10  # e*E [eV/A] per E [V/m]
EV_PER_A_TO_V_PER_M = 1e10


def eE(E):
    """Force e*E in eV/A for a field E in V/m (sign preserved)."""
    return np.asarray(E, dtype=float) * V_PER_M_TO_EV_PER_A


def kappa(E, T):
    """Relaxation length in k-space, e|E|T/hbar, in 1/A (E in V/m, T in s).

    kappa(1e4 V/m, 2 ps) = 3.0385e-3 1/A. T = inf gives inf.
    """
    return np.abs(np.asarray(E, dtype=float)) * V_PER_M_TO_EV_PER_A * np.asarray(T, dtype=float) / HBAR_EVS


def critical_field(eps_g, g):
    """Eq. 8: E_c = pi eps_g / (4 sqrt(g)) in V/m, for eps_g in eV and g (metric) in A^2."""
    return (
        math.pi
        * np.asarray(eps_g, dtype=float)
        / (4.0 * np.sqrt(np.asarray(g, dtype=float)))
        * EV_PER_A_TO_V_PER_M
    )


def interference_frequency(gap_integral):
    """Lambda = Int eps_g dk / (2 pi e) in V/m, for gap_integral = Int eps_g dk in eV/A."""
    return np.asarray(gap_integral, dtype=float) * EV_PER_A_TO_V_PER_M / (2.0 * math.pi)


def dynamical_phase(gap_integral, E):
    """Phi = (1/(e|E|)) Int eps_g dk (dimensionless), gap_integral in eV/A, E in V/m."""
    return np.asarray(gap_integral, dtype=float) / (np.abs(np.asarray(E, dtype=float)) * V_PER_M_TO_EV_PER_A)


def delta_ky(E, hbar_v):
    """Critical LZ window sqrt(e|E|/(hbar v)) in 1/A (E in V/m, hbar_v in eV*A)."""
    return np.sqrt(np.abs(np.asarray(E, dtype=float)) * V_PER_M_TO_EV_PER_A / hbar_v)


def hall_prefactor(mode="e2_h"):
    """Prefactor G of J_H = G * E * Int dk Int dk (...) in S.

    "e2_h": e^2/h as printed in Eqs. 10-11; "e2_2pi_h": e^2/h/(2 pi), the Kubo normalization
    of Int d^2k/(2 pi)^2 with e^2/hbar.
    """
    if mode == "e2_h":
        return E2_H
    if mode == "e2_2pi_h":
        return E2_H / (2.0 * math.pi)
    raise ValueError(f"unknown prefactor mode {mode!r}")


def velocity_to_hbar_v(v):
    """hbar*v in eV*A for a velocity v in m/s."""
    return HBAR_EVS * v * 1e10


def hbar_v_to_velocity(hbar_v):
    """Velocity in m/s for hbar*v in eV*A."""
    return hbar_v * 1e-10 / HBAR_EVS
