"""Two-band toy models with closed-form geometry or LZ physics (test fixtures)."""

from __future__ import annotations

import numpy as np

_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def _k(kx, ky):
    """Broadcast (kx, ky) as float arrays, or complex ones if either is complex (analytic continuation)."""
    dt = complex if (np.iscomplexobj(kx) or np.iscomplexobj(ky)) else float
    return np.broadcast_arrays(np.asarray(kx, dt), np.asarray(ky, dt))


class _TwoBand:
    n_orb = 2
    complex_k = True  # hamiltonian() accepts complex k (polynomial in k)

    def band_pair(self):
        return 0, 1


class MassiveDirac(_TwoBand):
    """H = v (kx sx + ky sy) + m sz.  Omega_c = -m v^2 / (2 E^3), g_xx = v^2 (v^2 ky^2 + m^2) / (4 E^4)."""

    def __init__(self, v: float = 0.3, m: float = 1e-3):
        self.v, self.m = v, m

    def hamiltonian(self, kx, ky):
        kx, ky = _k(kx, ky)
        return (self.v * kx)[..., None, None] * _SX + (self.v * ky)[..., None, None] * _SY + self.m * _SZ

    def dhamiltonian(self, kx, ky):
        kx, ky = _k(kx, ky)
        one = np.ones_like(kx)[..., None, None]
        return self.v * one * _SX, self.v * one * _SY

    def energy(self, kx, ky):
        return np.sqrt(self.v**2 * (np.asarray(kx) ** 2 + np.asarray(ky) ** 2) + self.m**2)

    def omega_c(self, kx, ky):
        return -self.m * self.v**2 / (2.0 * self.energy(kx, ky) ** 3)

    def g_xx(self, kx, ky):
        E = self.energy(kx, ky)
        return self.v**2 * (self.v**2 * np.asarray(ky) ** 2 + self.m**2) / (4.0 * E**4)

    def g_xy(self, kx, ky):
        E = self.energy(kx, ky)
        return -(self.v**4) * np.asarray(kx) * np.asarray(ky) / (4.0 * E**4)


class LinearLZ(_TwoBand):
    """Landau-Zener model along kx: H = alpha kx sz + delta sx (ky ignored).

    Eq. 8 is exact here: E_c = pi delta^2 / alpha (eV/A), i.e. 1e10 * pi delta^2 / alpha V/m.
    """

    def __init__(self, alpha: float = 0.15, delta: float = 0.3e-3):
        self.alpha, self.delta = alpha, delta

    def hamiltonian(self, kx, ky):
        kx, ky = _k(kx, ky)
        return (self.alpha * kx)[..., None, None] * _SZ + self.delta * _SX + 0j * ky[..., None, None]

    def dhamiltonian(self, kx, ky):
        kx, ky = _k(kx, ky)
        one = np.ones_like(kx)[..., None, None]
        return self.alpha * one * _SZ, 0.0 * one * _SX


class ParabolicDoubleCrossing(_TwoBand):
    """Two avoided crossings at kx = +-sqrt(b/a): H = (a kx^2 - b) sz + delta sx + v ky sy.

    Real for ky = 0, where the gauge-invariant geometric phase between the crossings is pi.
    """

    def __init__(self, a: float = 40.0, b: float = 1e-3, delta: float = 0.1e-3, v: float = 0.0):
        self.a, self.b, self.delta, self.v = a, b, delta, v

    def hamiltonian(self, kx, ky):
        kx, ky = _k(kx, ky)
        return (
            (self.a * kx * kx - self.b)[..., None, None] * _SZ
            + self.delta * _SX
            + (self.v * ky)[..., None, None] * _SY
        )

    def dhamiltonian(self, kx, ky):
        kx, ky = _k(kx, ky)
        one = np.ones_like(kx)[..., None, None]
        return (2.0 * self.a * kx)[..., None, None] * _SZ, self.v * one * _SY

    @property
    def k0(self) -> float:
        return float(np.sqrt(self.b / self.a))
