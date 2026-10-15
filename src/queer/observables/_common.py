"""Shared validation for numerical observables."""

import numpy as np

from ..physics.validation import positive as positive


def points_array(points):
    points = np.asarray(getattr(points, "path", points), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if np.isinf(points).any():
        raise ValueError("points must not contain infinities; NaNs denote invalid geometry gaps")
    return points


def model_energies(model, points, execution=None):
    """Evaluate finite points and preserve NaN gaps in invalid ARPES paths."""
    points = points_array(points)
    valid = np.isfinite(points).all(axis=1)
    if hasattr(model, "energies"):
        bands = model.energies(points[valid], execution=execution)
    else:
        bands = model.calculate_energy(points[valid])
    bands = np.asarray(bands, dtype=float)
    if bands.ndim != 2 or bands.shape[1] != int(valid.sum()):
        raise ValueError("model energies must have shape (Nb, Nk)")
    if not np.isfinite(bands).all():
        raise ValueError("model energies at valid points must be finite")
    result = np.full((bands.shape[0], len(points)), np.nan)
    result[:, valid] = bands
    return result
