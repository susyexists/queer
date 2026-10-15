"""Validated vectorized current sampling shared by Fourier projectors."""

from collections.abc import Callable

import numpy as np


def sample_current(J: Callable, E: np.ndarray) -> np.ndarray:
    """J at every entry of E with one vectorized call on the flattened array, reshaped to E.shape."""
    flat = E.reshape(-1)
    vals = np.asarray(J(flat))
    if np.iscomplexobj(vals):
        raise TypeError("J must be real-valued")
    if vals.shape != flat.shape:
        raise ValueError(f"J must be vectorized: returned shape {vals.shape} for input shape {flat.shape}")
    vals = vals.astype(float, copy=False)
    if not np.all(np.isfinite(vals)):
        raise ValueError("J returned non-finite values")
    return vals.reshape(E.shape)
