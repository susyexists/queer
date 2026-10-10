"""Validated proper rotations aligning Cartesian vectors."""

import numpy as np


def rotation_matrix(from_vector, to_vector) -> np.ndarray:
    vectors = [np.asarray(vector, dtype=float) for vector in (from_vector, to_vector)]
    if any(
        vector.shape != (3,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0
        for vector in vectors
    ):
        raise ValueError("rotation vectors must be finite nonzero 3-vectors")
    u, v = (vector / np.linalg.norm(vector) for vector in vectors)
    cosine = float(np.clip(u @ v, -1, 1))
    cross = np.cross(u, v)
    sine = np.linalg.norm(cross)
    if sine < 1e-14:
        if cosine > 0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis -= u * (axis @ u)
        axis /= np.linalg.norm(axis)
    else:
        axis = cross / sine
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + sine * skew + (1 - cosine) * (skew @ skew)
