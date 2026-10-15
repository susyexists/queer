"""Historical EPW branch heuristics, keeping phonons and couplings aligned."""

import numpy as np


def _crossing_input(values, tolerance, offset):
    array = np.asarray(values)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("crossing values must be a finite (Nbranch,Nq) array")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be nonnegative and finite")
    if isinstance(offset, bool) or not isinstance(offset, (int, np.integer)) or offset < 0:
        raise ValueError("offset must be a nonnegative integer")
    return array


def ph_cross(ph, tolerance, offset=50):
    ph = _crossing_input(ph, tolerance, offset)
    crossings, previous = [], {}
    for q in range(offset, ph.shape[1] - offset):
        for i in range(ph.shape[0]):
            for j in range(i + 1, ph.shape[0]):
                if abs(ph[i, q] - ph[j, q]) < tolerance:
                    if previous.get((i, j)) != q - 1:
                        crossings.append([i, j, q])
                    previous[i, j] = q
    return crossings


def g_cross(g, tolerance, offset=50):
    g = _crossing_input(g, tolerance, offset)
    crossings = []
    for q in range(max(1, offset), g.shape[1] - offset):
        for i in range(g.shape[0]):
            for j in range(i + 1, g.shape[0]):
                if abs(g[i, q] - g[i, q - 1]) > tolerance and abs(g[j, q] - g[j, q - 1]) > tolerance:
                    crossings.append([i, j, q])
    return crossings


def untangle(band, cross):
    fixed = np.asarray(band).copy()
    if fixed.ndim < 2:
        raise ValueError("band must have branch and q axes")
    for i, j, point in cross:
        if not all(isinstance(v, (int, np.integer)) for v in (i, j, point)) or not (
            0 <= i < len(fixed) and 0 <= j < len(fixed) and 0 <= point <= fixed.shape[1]
        ):
            raise ValueError("crossing indices outside the array")
        fixed[[i, j], :point] = fixed[[j, i], :point]
    return fixed
