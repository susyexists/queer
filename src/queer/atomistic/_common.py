"""Shared force-producer boundaries and atomistic energy calculations."""

from __future__ import annotations

import numpy as np

from ..physics.constants import ACCELERATION
from ..structures import Evaluation


def integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def positive(value, name, *, zero=False):
    value = float(value)
    if not np.isfinite(value) or value < 0 or (value == 0 and not zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def evaluate(provider, structure):
    """Evaluate an isolated structure copy and validate the producer's result."""
    value = (
        provider.evaluate(structure.copy()) if hasattr(provider, "evaluate") else provider(structure.copy())
    )
    if not isinstance(value, Evaluation):
        raise TypeError("force provider must return Evaluation")
    # Producers may mutate an Evaluation after construction; validate again.
    value = Evaluation(value.energy_eV, value.forces_eV_A, value.stress_eV_A3)
    if value.forces_eV_A.shape != (structure.natoms, 3):
        raise ValueError("force provider changed the atom count or force shape")
    return value


def kinetic_energy(masses, velocities):
    """Kinetic energy in eV for masses in amu and velocities in Å/fs."""
    return float(np.sum(np.asarray(masses)[:, None] * np.asarray(velocities) ** 2) / (2 * ACCELERATION))
