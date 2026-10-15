"""Shared numerical convergence arithmetic, independent of grids and storage."""

import numpy as np


def convergence_diagnostic(current, previous, options):
    """Compare values using the established maximum absolute component error."""
    error = float(np.max(np.abs(current - previous), initial=0))
    scale = float(np.max(np.abs(current), initial=0))
    threshold = options.get("absolute_tolerance", 1e-6) + options.get("relative_tolerance", 1e-4) * scale
    return {"error_estimate": error, "threshold": threshold}


def refined_grid(base, active, level):
    """Refine only the caller-selected axes; callers retain their domain policy."""
    return np.asarray(base) * np.where(active, 2**level, 1)
