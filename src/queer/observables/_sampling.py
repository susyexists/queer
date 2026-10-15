"""Ordered bounded observable sampling and normalized reciprocal meshes."""

from __future__ import annotations

import numpy as np

from ..execution import ExecutionOptions, execution_session
from ..models.evaluation import point_execution


def kmesh(grid, k_start=None, k_vectors=None):
    """Uniform Gamma-origin fractional mesh; the upper boundary is excluded."""
    dimensions = np.asarray(grid)
    if dimensions.shape != (3,) or not np.equal(dimensions, np.round(dimensions)).all():
        raise ValueError("grid must contain three positive integers")
    if np.any(dimensions < 1) or not np.isfinite(dimensions).all():
        raise ValueError("grid must contain three positive integers")
    start = np.zeros(3) if k_start is None else np.asarray(k_start, dtype=float)
    vectors = np.eye(3) if k_vectors is None else np.asarray(k_vectors, dtype=float)
    if (
        start.shape != (3,)
        or vectors.shape != (3, 3)
        or not np.isfinite(start).all()
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("k_start/k_vectors must be finite (3,)/(3,3) arrays")
    axes = [np.arange(int(n), dtype=float) / n for n in dimensions]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return start + mesh @ vectors


def normalized_weights(points, weights=None):
    n = len(points)
    if n < 1:
        raise ValueError("at least one k-point is required")
    values = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if values.shape != (n,) or not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("weights must be finite nonnegative values with positive sum")
    return values / values.sum()


def map_points(model, points, function, execution=None):
    """Ordered bounded process evaluation with the shared execution budget."""
    options = (execution or getattr(model, "execution", None) or ExecutionOptions()).resolved()
    options.bounded_batch_size(1024 * model.nbnd**2)
    inner = ExecutionOptions(
        workers=1,
        batch_size=options.batch_size,
        native_threads=options.native_threads,
        memory_mb=None if options.memory_mb is None else options.memory_mb / options.workers,
    )
    from threadpoolctl import threadpool_limits

    if options.workers == 1:
        with threadpool_limits(limits=options.native_threads):
            return [_evaluate_point(function, point, inner) for point in points]
    from joblib import delayed

    with execution_session(options) as session:
        return list(session.parallel()(delayed(_evaluate_point)(function, point, inner) for point in points))


def _evaluate_point(function, point, execution):
    with point_execution(execution):
        return function(point)


def _weighted_sum(model, points, weight, function, shape, options, bytes_per_point):
    """Reduce bounded k-point batches, retaining only the final scan tensor."""
    batch = options.bounded_batch_size(bytes_per_point)
    answer = np.zeros(shape, complex)
    with execution_session(options):
        for start in range(0, len(points), batch):
            stop = start + batch
            values = np.asarray(map_points(model, points[start:stop], function, options))
            answer += np.tensordot(weight[start:stop], values, axes=(0, 0))
    return answer
