"""Compatibility reexports for shared tight-binding response primitives.

New implementation belongs to physics kernels, model evaluation and observable
sampling. This module retains historical internal imports.
"""

from ..models.evaluation import (
    band_connections,
    band_geometry,
    basis_curvature,
    basis_overlap,
    cell,
    chemical_potentials,
    degeneracy_tolerance,
    derivatives,
    overlap_matrix,
    position_matrix,
    reciprocal,
    spin_degeneracy,
    states,
    velocities,
    volume,
)
from ..physics.constants import E2_H, E_CHARGE, HBAR, KB_EV
from ..physics.geometry import energy_blocks, velocity_products
from ..physics.statistics import fermi_derivative, occupations
from ._sampling import kmesh, map_points, normalized_weights

__all__ = [
    "E2_H",
    "E_CHARGE",
    "HBAR",
    "KB_EV",
    "occupations",
    "fermi_derivative",
    "energy_blocks",
    "velocity_products",
    "band_connections",
    "band_geometry",
    "basis_curvature",
    "basis_overlap",
    "cell",
    "chemical_potentials",
    "degeneracy_tolerance",
    "derivatives",
    "overlap_matrix",
    "position_matrix",
    "reciprocal",
    "spin_degeneracy",
    "states",
    "velocities",
    "volume",
    "kmesh",
    "normalized_weights",
    "map_points",
]
