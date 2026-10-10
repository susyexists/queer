"""Coordinate transformations with lattice vectors stored as matrix rows."""

import numpy as np


def validated_lattice(lattice) -> np.ndarray:
    array = np.asarray(lattice, dtype=np.float64)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise ValueError("lattice must be a finite (3, 3) array of row vectors")
    if np.linalg.cond(array) > 1e12:
        raise ValueError("lattice is singular or numerically ill-conditioned")
    return array


def validated_points(points) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("k-points must have shape (N, 3)")
    if not np.isfinite(array).all():
        raise ValueError("k-points must be finite")
    return array


def reciprocal_lattice(direct_lattice) -> np.ndarray:
    """Return B in Å⁻¹ satisfying A @ B.T = 2π I for A in Å."""
    direct = validated_lattice(direct_lattice)
    return 2 * np.pi * np.linalg.solve(direct, np.eye(3)).T


def fractional_to_cartesian(points, reciprocal) -> np.ndarray:
    """Map (N,3) fractional reciprocal points into Cartesian Å⁻¹."""
    return validated_points(points) @ validated_lattice(reciprocal)


def cartesian_to_fractional(points, reciprocal) -> np.ndarray:
    """Map Cartesian Å⁻¹ points into fractional reciprocal coordinates."""
    lattice = validated_lattice(reciprocal)
    return np.linalg.solve(lattice.T, validated_points(points).T).T
