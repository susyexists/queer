"""Compare user-supplied digitized Hall maps with native model results."""

from __future__ import annotations

import numpy as np

from ...execution import ExecutionOptions, execution_session
from .._storage import json_safe
from ..api import PreparedHallResponse
from ..config import resolve_options
from ..models.fourband import FourBandModel, FourBandParams


def load_paper_maps(path):
    """Load explicit digitized data; no installed-package or source-tree data dependency."""
    with np.load(path, allow_pickle=False) as arrays:
        orders = np.asarray(arrays["N"], dtype=int)
        return {
            "x": arrays["x"],
            "E0": arrays["E0"],
            "N": orders.tolist(),
            "maps": {int(n): arrays[f"J_{n}"] for n in orders},
            "clim": dict(zip(orders.tolist(), arrays["clim"].tolist())),
        }


def model_maps(definition, x, amplitudes, orders, *, execution=None, model_parameters=None):
    """Native Ext. 7a comparison maps in mA/m; signed Fourier coefficients are retained."""
    options = (execution or ExecutionOptions()).resolved()
    orders = np.asarray(orders, dtype=int)
    if orders.ndim != 1 or not len(orders) or np.any(orders < 1):
        raise ValueError("orders must be a nonempty sequence of positive harmonic indices")
    amplitudes = np.asarray(amplitudes, float)
    cfg = resolve_options(
        {
            "preset": "ext7a",
            "crossings.definition": definition,
            "drive.E_max": max(float(np.max(amplitudes)), 1.0),
            "use_symmetry_shortcuts": True,
        }
    )

    def calculate(ratio):
        model = FourBandModel(FourBandParams(**(model_parameters or {})).at(float(ratio)))
        inner = ExecutionOptions(
            native_threads=options.native_threads,
            memory_mb=None if options.memory_mb is None else options.memory_mb / options.workers,
        )
        return PreparedHallResponse(model, cfg, execution=inner).harmonics(amplitudes, int(max(orders)))["b"]

    if options.workers == 1:
        values = [calculate(ratio) for ratio in x]
    else:
        from joblib import delayed

        with execution_session(options) as session:
            values = list(session.parallel()(delayed(calculate)(ratio) for ratio in x))
    coefficients = np.asarray(values)
    return {int(n): coefficients[..., n].T * 1e3 for n in orders}


def compare_maps(model, paper, *, window=(1.12, 1.25), min_fraction=0.08):
    """Sign agreement and scale errors on finite nonsaturated pixels, with explicit masks."""
    if not 0 <= min_fraction < 1 or len(window) != 2 or window[1] < window[0]:
        raise ValueError("Invalid comparison threshold/window")
    x = np.asarray(paper["x"])
    output = {}
    for order in paper["N"]:
        target, predicted = np.asarray(paper["maps"][order]), np.asarray(model[order])
        if target.shape != predicted.shape or target.ndim != 2 or target.shape[1] != len(x):
            raise ValueError("Model and digitized maps must share (amplitude, parameter) axes")
        eligible = np.isfinite(target) & (np.abs(target) > min_fraction * paper["clim"][order])
        for name, columns in (
            ("all", np.ones_like(x, dtype=bool)),
            ("window", (x >= window[0]) & (x <= window[1])),
        ):
            selected = eligible & columns[None, :]
            valid = selected & np.isfinite(predicted)
            ratio = predicted[valid] / target[valid]
            positive = ratio > 0
            output[f"{order}/{name}"] = {
                "pixels": int(valid.sum()),
                "invalid_model_pixels": int(np.count_nonzero(selected & ~valid)),
                "sign_agreement": float(positive.mean()) if ratio.size else None,
                "median_ratio": float(np.median(ratio[positive])) if positive.any() else None,
                "iqr": np.percentile(ratio[positive], [25, 75]).tolist() if positive.any() else None,
                "rms_log_ratio": float(np.sqrt(np.mean(np.log(ratio[positive]) ** 2)))
                if positive.any()
                else None,
            }
    return json_safe({"window": list(window), "min_fraction": min_fraction, "harmonics": output})
