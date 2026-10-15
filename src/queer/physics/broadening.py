"""Normalized line shapes; Gaussian widths are sigma, Lorentzian widths HWHM."""

import numpy as np


def _width(width):
    values = np.asarray(width, dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("width must be finite and positive")
    return values


def gaussian(difference, width):
    """Unit-area Gaussian with standard deviation ``width``."""
    width = _width(width)
    return np.exp(-0.5 * (np.asarray(difference) / width) ** 2) / (width * np.sqrt(2 * np.pi))


def lorentzian(difference, width):
    """Unit-area Lorentzian with half width at half maximum ``width``."""
    width = _width(width)
    return width / np.pi / (np.asarray(difference) ** 2 + width**2)
