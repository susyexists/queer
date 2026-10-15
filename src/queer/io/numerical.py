"""Shared deterministic identities and metadata-bearing numerical archives."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .artifacts import atomic_npz, canonical_json


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf" if value < 0 else None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def stable_hash(*values):
    def canonical(value):
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            return {
                "array_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            }
        if isinstance(value, dict):
            return {str(k): canonical(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [canonical(v) for v in value]
        return json_safe(value)

    return hashlib.sha256(canonical_json([canonical(v) for v in values]).encode()).hexdigest()


def save_npz(path, arrays, meta=None):
    path = Path(path)
    if "__meta__" in arrays:
        raise ValueError("__meta__ is reserved")
    atomic_npz(path, {**arrays, "__meta__": np.asarray(canonical_json(json_safe(meta or {})))})
    return path


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return (
            {k: data[k] for k in data.files if k != "__meta__"},
            json.loads(str(data["__meta__"])) if "__meta__" in data else {},
        )
