"""Compatibility exports for statistical occupations and rotations."""

from .geometry.rotations import rotation_matrix as get_rotation_matrix
from .physics.statistics import thermal_occupation


def fd(energy, T=0.025):
    """Fermi occupation for energies and thermal energy in eV; equality is half."""
    return thermal_occupation(energy, T)


__all__ = ["fd", "get_rotation_matrix"]
