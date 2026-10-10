"""Compatibility mesh helpers; modern geometry uses (N,3) arrays throughout."""

import numpy as np

from .geometry.lattice import cartesian_to_fractional, fractional_to_cartesian, validated_lattice
from .geometry.mesh import cartesian_mesh


def mesh_cartesian(N, factor=1, center=True):
    return cartesian_mesh(N, factor, center)


def mesh_crystal(N, g_vec, dimension=3, factor=1):
    """Legacy Cartesian lattice mesh, returned as (3,N)."""
    if dimension not in (2, 3):
        raise ValueError("dimension must be 2 or 3")
    if dimension == 2:
        counts = [N, N, 1] if isinstance(N, (int, np.integer)) else [*list(N)[:2], 1]
    else:
        counts = N
    return fractional_to_cartesian(cartesian_mesh(counts, factor), g_vec).T


def hexagon_crystal(N, g_vec):
    """Return a planar reciprocal mesh clipped to the nearest-cell region."""
    lattice = validated_lattice(g_vec)
    points = fractional_to_cartesian(cartesian_mesh([N, N, 1], factor=2), lattice)
    inside = np.ones(len(points), dtype=bool)
    for n1 in (-1, 0, 1):
        for n2 in (-1, 0, 1):
            if n1 or n2:
                vector = n1 * lattice[0] + n2 * lattice[1]
                inside &= points @ vector <= vector @ vector / 2 + 1e-12
    return points[inside]


def cartesian2crystal(cartesian, g_vec):
    """Historical name: fractional → Cartesian, returning (3,N)."""
    return fractional_to_cartesian(cartesian, g_vec).T


def crystal2cartesian(crystal, g_vec):
    """Historical name: Cartesian (3,N) → fractional (N,3)."""
    return cartesian_to_fractional(np.asarray(crystal).T, g_vec)


def reciprocal2angstrom(reciprocal, g_vec):
    return fractional_to_cartesian(reciprocal, g_vec)


def angstrom2reciprocal(angstrom, g_vec):
    return cartesian_to_fractional(angstrom, g_vec)
