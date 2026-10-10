"""Independent local quadrature with deterministic, resumable cell evaluations.

The integration domain is the unit box. Weights integrate to one, including
after local subdivision. Error estimates are differences between embedded
midpoint and tensor Gauss rules, and are not proofs of continuum convergence.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import product

import numpy as np

_STORE = ContextVar("queer_quadrature_store", default=None)


@contextmanager
def quadrature_checkpoint(store):
    """Commit immutable cell evaluations through the enclosing run store."""
    token = _STORE.set(store)
    try:
        yield
    finally:
        _STORE.reset(token)


def checkpoint_samples(function, points, *, namespace, tag, value_units="integrand"):
    """Evaluate one deterministic node group, restoring verified saved arrays."""
    points = np.asarray(points, float)
    store = _STORE.get()
    identity = namespace.encode() + b"\0" + tag.encode() + b"\0" + points.astype("<f8").tobytes()
    case = "quadrature_" + hashlib.sha256(identity).hexdigest()[:24]
    if store is not None and store.completed(case):
        record = store.manifest["cases"][case]
        with np.load(store.root / record["path"], allow_pickle=False) as saved:
            if not np.array_equal(saved["points"], points):
                raise ValueError("Quadrature checkpoint points changed")
            return saved["values"]
    values = np.asarray(function(points))
    if values.shape[:1] != (len(points),) or not np.isfinite(values).all():
        raise ValueError("Quadrature function must return finite arrays with a leading point axis")
    if store is not None:
        store.commit(
            case,
            {"points": points, "values": values},
            metadata={
                "quadrature_namespace": namespace,
                "rule": tag,
                "sample_count": len(points),
                "dims": {
                    "points": ["sample", "fractional"],
                    "values": ["sample", *[f"response_{i}" for i in range(values.ndim - 1)]],
                },
                "units": {"points": "fractional", "values": value_units},
            },
        )
    return values


@dataclass(frozen=True)
class QuadratureResult:
    value: np.ndarray
    error: np.ndarray
    metadata: dict


def nested_observable(function, grid, options, key, *, active_axes=None):
    """Nested uniform refinement for direct Python callers and whole-mesh cases."""
    from .result import ObservableResult

    controls = validate_integration(options)
    base = np.asarray(controls.get("grid", grid))
    if (
        base.shape != (3,)
        or not np.isfinite(base).all()
        or np.any(base < 1)
        or not np.equal(base, np.round(base)).all()
    ):
        raise ValueError("nested integration grid must contain three positive integers")
    base = base.astype(int)
    active = base > 1 if active_axes is None else np.isin(np.arange(3), active_axes)
    if not active.any():
        active[:] = True
    previous = None
    store = _STORE.get()
    for level in range(controls.get("max_levels", 4)):
        mesh = base * np.where(active, 2**level, 1)
        if int(np.prod(mesh)) > controls.get("max_points", 1_000_000):
            raise RuntimeError("Adaptive integration exhausted max_points before convergence")
        case = f"quadrature_{key}_level_{level:04d}"
        if store is not None and store.completed(case):
            record = store.manifest["cases"][case]
            with np.load(store.root / record["path"], allow_pickle=False) as saved:
                result = ObservableResult(
                    {name: saved[name] for name in saved.files},
                    record["metadata"]["dims"],
                    record["metadata"]["units"],
                    record["metadata"],
                )
        else:
            result = function(mesh)
            if store is not None:
                store.commit(case, result.arrays, metadata=result.artifact_metadata())
        value = result[key]
        if previous is not None:
            error = float(np.max(np.abs(value - previous), initial=0))
            scale = float(np.max(np.abs(value), initial=0))
            threshold = (
                controls.get("absolute_tolerance", 1e-6) + controls.get("relative_tolerance", 1e-4) * scale
            )
            if error <= threshold:
                return ObservableResult(
                    result.arrays,
                    result.dims,
                    result.units,
                    {
                        **result.metadata,
                        "integration": {
                            "method": "nested_grid_refinement",
                            "level": level,
                            "grid": mesh.tolist(),
                            "points": int(np.prod(mesh)),
                            "error_estimate": error,
                            "threshold": threshold,
                            "converged": True,
                            "continuum_convergence_established": False,
                        },
                    },
                )
        previous = value
    raise RuntimeError("Adaptive integration exhausted max_levels before convergence")


def validate_integration(options):
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise ValueError("integration must be a mapping")
    allowed = {
        "method",
        "grid",
        "subdivision",
        "threshold",
        "absolute_tolerance",
        "relative_tolerance",
        "max_points",
        "max_cells",
        "max_levels",
        "threshold_quantity",
    }
    if set(options) - allowed:
        raise ValueError(f"Unknown integration options: {sorted(set(options) - allowed)}")
    if options.get("method", "grid") not in {"grid", "adaptive", "threshold", "cubature"}:
        raise ValueError("integration.method must be grid, adaptive, threshold, or cubature")
    if options.get("threshold_quantity", "response") not in {"response", "curvature"}:
        raise ValueError("threshold_quantity must be response or curvature (in square angstroms)")
    for name in ("threshold", "absolute_tolerance", "relative_tolerance"):
        if name in options and (not np.isfinite(options[name]) or options[name] < 0):
            raise ValueError(f"integration.{name} must be finite and nonnegative")
    for name in ("max_points", "max_cells", "max_levels"):
        if name in options and (
            isinstance(options[name], bool) or int(options[name]) != options[name] or options[name] < 1
        ):
            raise ValueError(f"integration.{name} must be a positive integer")
    result = dict(options)
    for name in ("max_points", "max_cells", "max_levels"):
        if name in result:
            result[name] = int(result[name])
    return result


def integrate_unit_box(
    function, grid, options, *, namespace="integral", active_axes=None, threshold_selector=None
):
    """Integrate array-valued ``function(points)`` over a normalized unit box.

    Threshold mode replaces large midpoint cells with a midpoint subgrid.
    Cubature adaptively bisects the largest embedded-error cell using a
    two-node tensor Gauss rule. Every component must satisfy its tolerance.
    ``max_points`` includes all evaluated embedded nodes; exhausted budgets
    raise, leaving any completed checkpoint cells available for resumption.
    """
    opts = validate_integration(options)
    method = opts.get("method", "cubature")
    if method not in {"threshold", "cubature"}:
        raise ValueError("local quadrature requires method threshold or cubature")
    mesh = np.asarray(opts.get("grid", grid))
    if (
        mesh.ndim != 1
        or not len(mesh)
        or not np.isfinite(mesh).all()
        or np.any(mesh < 1)
        or (not np.equal(mesh, np.round(mesh)).all())
    ):
        raise ValueError("integration grid must contain positive integers")
    mesh = mesh.astype(int)
    axes = np.flatnonzero(mesh > 1) if active_axes is None else np.asarray(active_axes, int)
    if not len(axes):
        axes = np.arange(len(mesh))
    if len(np.unique(axes)) != len(axes) or np.any(axes < 0) or np.any(axes >= len(mesh)):
        raise ValueError("active integration axes must be distinct and in range")
    if np.any(mesh[np.setdiff1d(np.arange(len(mesh)), axes)] != 1):
        raise ValueError("inactive integration axes require exactly one grid point")
    limit = opts.get("max_points", 1_000_000)
    max_cells = opts.get("max_cells", 100_000)
    evaluations = 0

    def evaluate(points, tag):
        nonlocal evaluations
        points = np.asarray(points, float)
        if evaluations + len(points) > limit:
            raise RuntimeError("Local integration exhausted max_points before convergence")
        evaluations += len(points)
        return checkpoint_samples(function, points, namespace=namespace, tag=tag)

    cells = []
    for index in product(*(range(int(n)) for n in mesh)):
        lower = np.asarray(index, float) / mesh
        width = np.zeros(len(mesh))
        width[axes] = 1 / mesh[axes]
        cells.append((lower, width))
    if len(cells) > max_cells:
        raise RuntimeError("Initial integration grid exceeds max_cells")

    def measure(width):
        return float(np.prod(width[axes]))

    if method == "threshold":
        centers = np.asarray([lower + width / 2 for lower, width in cells])
        coarse = evaluate(centers, "midpoint")
        answer = np.zeros_like(coarse[0])
        error = np.zeros_like(np.abs(coarse[0]), dtype=float)
        subdivision = np.asarray(opts.get("subdivision", np.repeat(4, len(mesh))))
        if (
            subdivision.shape != mesh.shape
            or np.any(subdivision < 1)
            or not np.equal(subdivision, np.round(subdivision)).all()
        ):
            raise ValueError("integration.subdivision must contain one positive integer per axis")
        subdivision = subdivision.astype(int)
        subdivision[np.setdiff1d(np.arange(len(mesh)), axes)] = 1
        selected, leaf_cells = 0, len(cells)
        for (lower, width), value in zip(cells, coarse, strict=True):
            cell_measure = measure(width)
            magnitude = (
                np.linalg.norm(np.ravel(value)) if threshold_selector is None else threshold_selector(value)
            )
            if float(magnitude) >= opts.get("threshold", 50.0):
                leaf_cells += int(np.prod(subdivision)) - 1
                if leaf_cells > max_cells:
                    raise RuntimeError("Threshold integration exhausted max_cells during subdivision")
                offsets = np.array(list(product(*((np.arange(n) + 0.5) / n for n in subdivision))))
                refined = evaluate(lower + offsets * width, "subgrid").mean(axis=0)
                answer += refined * cell_measure
                error += np.abs(refined - value) * cell_measure
                selected += 1
            else:
                answer += value * cell_measure
        return QuadratureResult(
            answer,
            error,
            {
                "method": "threshold",
                "points": evaluations,
                "cells": leaf_cells,
                "coarse_cells": len(cells),
                "refined_cells": selected,
                "normalized_weight_sum": 1.0,
                "error_estimate": float(np.max(error, initial=0)),
                "converged": None,
                "continuum_convergence_established": False,
            },
        )

    nodes = np.zeros((2 ** len(axes), len(mesh)))
    nodes[:, axes] = np.asarray(list(product(*([[-1, 1]] * len(axes))))) / np.sqrt(3)

    def cell_rule(lower, width):
        center = lower + width / 2
        points = np.vstack((center[None], center + nodes * width / 2))
        values = evaluate(points, "embedded-gauss2")
        integral = values[1:].mean(axis=0) * measure(width)
        error = np.abs(integral - values[0] * measure(width))
        return [lower, width, integral, error]

    active = [cell_rule(lower, width) for lower, width in cells]
    atol = opts.get("absolute_tolerance", 1e-6)
    rtol = opts.get("relative_tolerance", 1e-4)
    if atol == 0 and rtol == 0:
        raise ValueError("At least one integration tolerance must be positive")
    while True:
        answer = sum((cell[2] for cell in active), np.zeros_like(active[0][2]))
        error = sum((cell[3] for cell in active), np.zeros_like(active[0][3]))
        tolerance = atol + rtol * np.abs(answer)
        if np.all(error <= tolerance):
            return QuadratureResult(
                answer,
                error,
                {
                    "method": "cubature",
                    "points": evaluations,
                    "cells": len(active),
                    "normalized_weight_sum": 1.0,
                    "converged": True,
                    "error_estimate": float(np.max(error, initial=0)),
                    "absolute_tolerance": atol,
                    "relative_tolerance": rtol,
                    "continuum_convergence_established": False,
                },
            )
        if len(active) + 1 > max_cells:
            raise RuntimeError("Local integration exhausted max_cells before convergence")
        worst = max(
            range(len(active)),
            key=lambda i: float(np.max(active[i][3] / np.maximum(tolerance, np.finfo(float).tiny))),
        )
        lower, width, _, _ = active.pop(worst)
        axis = int(axes[np.argmax(width[axes])])
        half = width.copy()
        half[axis] /= 2
        second = lower.copy()
        second[axis] += half[axis]
        active.extend((cell_rule(lower, half), cell_rule(second, half)))
