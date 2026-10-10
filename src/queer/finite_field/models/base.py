"""Model protocol and field-frame helpers.

A band model supplies the Bloch Hamiltonian and its k-derivatives on arrays of Cartesian k
(1/A, model frame). Geometry is evaluated in the field frame (s, t): s along the field direction
e = (cos phi, sin phi), t along z x e, so k = s e + t (z x e).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BandModel(Protocol):
    n_orb: int

    def hamiltonian(self, kx, ky) -> np.ndarray:
        """H(k), shape (..., n, n), eV."""

    def dhamiltonian(self, kx, ky) -> tuple[np.ndarray, np.ndarray]:
        """(dH/dkx, dH/dky), each shape (..., n, n), eV*A."""

    def band_pair(self) -> tuple[int, int]:
        """0-based (v, c) band indices (bands sorted ascending)."""


def field_frame(s, t, angle: float = 0.0):
    """Cartesian (kx, ky) of field-frame coordinates (s along the field, t perpendicular)."""
    c, sn = np.cos(angle), np.sin(angle)
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)
    return s * c - t * sn, s * sn + t * c


def rotate_derivatives(dHx, dHy, angle: float = 0.0):
    """(dH/ds, dH/dt) from Cartesian derivatives for a field at `angle` (rad)."""
    if angle == 0.0:
        return dHx, dHy
    c, sn = np.cos(angle), np.sin(angle)
    return c * dHx + sn * dHy, -sn * dHx + c * dHy
