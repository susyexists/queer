"""Implicit spin copies contribute to currents, independently of pair geometry."""

import numpy as np


def validated_spin_degeneracy(value=1):
    """Accept the native single-channel or implicit nonmagnetic double copy."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value not in (1, 2):
        raise ValueError("spin_degeneracy must be integer one or two")
    return int(value)
