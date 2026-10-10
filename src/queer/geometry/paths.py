"""Named reciprocal paths expressed in the original model lattice basis."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..io.structure import read_poscar
from .lattice import fractional_to_cartesian, reciprocal_lattice, validated_points

GAMMA_ALIASES = {"G", "GAMMA", "Γ", "\\GAMMA"}


def canonical_label(label) -> str:
    upper = str(label).strip().upper()
    return "GAMMA" if upper in GAMMA_ALIASES else upper


def parse_point_names(point_names) -> list[str]:
    if isinstance(point_names, str):
        names = [part for part in re.split(r"\s*(?:->|→|-|,)\s*|\s+", point_names.strip()) if part]
    else:
        names = list(point_names)
    return [canonical_label(label) for label in names]


@dataclass
class KPath:
    """Fractional (N,3) points, tick indices, and Cartesian path distances."""

    sym: np.ndarray
    path: np.ndarray
    labels: list[str]
    names: list[str] = field(default_factory=list)
    point_coords: dict = field(default_factory=dict)
    distances: np.ndarray | None = None

    def __post_init__(self):
        self.path = validated_points(self.path)
        self.sym = np.asarray(self.sym, dtype=int)
        self.labels = list(self.labels)
        self.names = list(self.names)
        self.point_coords = dict(self.point_coords)
        if self.sym.ndim != 1 or len(self.sym) != len(self.labels):
            raise ValueError("path tick indices and labels must have equal lengths")
        if len(self.sym) and (
            self.sym.min() < 0 or self.sym.max() >= len(self.path) or np.any(np.diff(self.sym) < 0)
        ):
            raise ValueError("path tick indices must be ordered and in range")
        if self.distances is not None:
            self.distances = np.asarray(self.distances, dtype=float)
            if self.distances.shape != (len(self.path),) or not np.isfinite(self.distances).all():
                raise ValueError("path distances must be finite and aligned with k-points")

    def __iter__(self):
        yield self.sym
        yield self.path
        yield self.labels

    def __array__(self, dtype=None, copy=None):
        if copy is None:
            return np.asarray(self.path, dtype=dtype)
        return np.array(self.path, dtype=dtype, copy=copy)

    def as_tuple(self):
        return self.sym, self.path, self.labels


def interpolate_path(n_points: int, corners, reciprocal=None) -> tuple[np.ndarray, np.ndarray]:
    """Allocate exactly n_points intervals, preserving every segment endpoint."""
    points = validated_points(corners)
    if len(points) < 2:
        raise ValueError("a path requires at least two corners")
    if isinstance(n_points, bool) or not isinstance(n_points, (int, np.integer)) or n_points < 1:
        raise ValueError("n_points must be a positive integer")
    steps = np.diff(points, axis=0)
    physical_steps = steps if reciprocal is None else fractional_to_cartesian(steps, reciprocal)
    lengths = np.linalg.norm(physical_steps, axis=1)
    positive = lengths > 1e-14
    if not positive.any():
        raise ValueError("a path must contain at least one nonzero segment")
    if n_points < positive.sum():
        raise ValueError("n_points must provide at least one interval per nonzero segment")
    allocation = positive.astype(int)
    remaining = n_points - allocation.sum()
    quotas = remaining * lengths / lengths.sum()
    allocation += np.floor(quotas).astype(int)
    extra = n_points - allocation.sum()
    order = np.argsort(-(quotas - np.floor(quotas)), kind="stable")
    allocation[order[:extra]] += 1
    tick_indices = np.concatenate(([0], np.cumsum(allocation)))
    pieces = [
        np.linspace(points[index], points[index + 1], count + 1)[:-1]
        for index, count in enumerate(allocation)
        if count
    ]
    path = np.vstack([*pieces, points[-1:]])
    return tick_indices, path


def seekpath_points(poscar: str | Path, **kwargs) -> dict:
    """Compatibility name for native HPKOT points in the input cell basis."""
    from .symmetry import compute_band_path

    return compute_band_path(read_poscar(poscar), **kwargs)["point_coords"]


def named_k_path(point_names, n_points: int, poscar: str | Path, **kwargs) -> KPath:
    return named_k_path_from_structure(point_names, n_points, read_poscar(poscar), **kwargs)


def named_k_path_from_structure(point_names, n_points: int, structure, **kwargs) -> KPath:
    """Named path for native Structure, Generator dict or a symmetry-cell tuple."""
    from .symmetry import _cell_tuple, compute_band_path

    names = parse_point_names(point_names)
    coordinates = compute_band_path(structure, **kwargs)["point_coords"]
    missing = [name for name in names if name not in coordinates]
    if missing:
        raise KeyError(f"Unknown SeekPath points {missing}; available: {', '.join(sorted(coordinates))}")
    lattice, _, _ = _cell_tuple(structure)
    reciprocal = reciprocal_lattice(lattice)
    sym, points = interpolate_path(n_points, [coordinates[name] for name in names], reciprocal)
    cartesian = fractional_to_cartesian(points, reciprocal)
    distances = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(cartesian, axis=0), axis=1))))
    labels = ["Γ" if name == "GAMMA" else name for name in names]
    return KPath(sym, points, labels, names, coordinates, distances)
