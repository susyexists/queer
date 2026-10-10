"""Compatibility exports for statistical occupations and rotations."""

import numpy as np
from scipy.special import expit

from .geometry.rotations import rotation_matrix as get_rotation_matrix


def fd(energy, T=0.025):
    """Fermi occupation for energies and temperature in eV; T=0 uses a step."""
    values = np.asarray(energy, dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(T) or T < 0:
        raise ValueError("energy must be finite and T must be nonnegative and finite")
    if T == 0:
        return np.where(values < 0, 1.0, np.where(values > 0, 0.0, 0.5))
    return expit(-values / T)


__all__ = ["fd", "get_rotation_matrix"]
