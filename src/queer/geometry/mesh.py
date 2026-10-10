"""Deterministic meshes with shape (N,3), including planar meshes."""

import numpy as np


def cartesian_mesh(n_points, factor: float = 1.0, center: bool = True) -> np.ndarray:
    if isinstance(n_points, (int, np.integer)) and not isinstance(n_points, bool):
        counts = [int(n_points)] * 3
    else:
        try:
            counts = list(n_points)
        except TypeError as error:
            raise ValueError("n_points must be an integer or three axis counts") from error
    if len(counts) != 3 or any(
        isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1 for n in counts
    ):
        raise ValueError("each of the three mesh axis counts must be a positive integer")
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("factor must be positive and finite")
    axes = [np.arange(n, dtype=float) / n for n in counts]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3) * factor
    if center:
        mesh -= mesh.mean(axis=0)
    return mesh
