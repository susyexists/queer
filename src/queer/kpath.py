"""High-symmetry k-path helpers backed by SeekPath."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union
import re

import numpy as np
import seekpath

from .utils import path_create


GAMMA_ALIASES = {"G", "GAMMA", "Γ", "\\GAMMA"}


def _canonical_label(label: str) -> str:
    clean = str(label).strip()
    upper = clean.upper()
    if upper in GAMMA_ALIASES:
        return "GAMMA"
    return upper


def _display_label(label: str) -> str:
    return "Γ" if _canonical_label(label) == "GAMMA" else str(label).strip()


PathLike = Union[str, Path]


@dataclass
class KPath:
    """Named high-symmetry path with plotting metadata."""

    sym: np.ndarray
    path: np.ndarray
    labels: List[str]
    names: List[str] = field(default_factory=list)
    point_coords: Dict[str, Sequence[float]] = field(default_factory=dict)

    def __post_init__(self):
        self.sym = np.asarray(self.sym, dtype=int)
        self.path = np.asarray(self.path, dtype=float)
        self.labels = list(self.labels)
        self.names = list(self.names)
        self.point_coords = dict(self.point_coords)

    def __iter__(self):
        """Keep legacy ``sym, path, labels = model.kpath(...)`` unpacking working."""
        yield self.sym
        yield self.path
        yield self.labels

    def __array__(self, dtype=None, copy=None):
        """Allow numpy-based routines to consume a KPath as its k-point array."""
        if copy is None:
            return np.asarray(self.path, dtype=dtype)
        return np.array(self.path, dtype=dtype, copy=copy)

    def as_tuple(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Return ``(sym, path, labels)`` for older call sites."""
        return self.sym, self.path, self.labels


def parse_point_names(point_names: Union[str, Iterable[str]]) -> List[str]:
    """Normalize point-name input from a string or iterable.

    Examples
    --------
    ``"GAMMA-X-W-K-GAMMA-L"`` and
    ``["GAMMA", "X", "W", "K", "GAMMA", "L"]`` are equivalent.
    """
    if isinstance(point_names, str):
        names = [part for part in re.split(r"\s*(?:->|→|-|,)\s*|\s+", point_names.strip()) if part]
    else:
        names = list(point_names)
    return [_canonical_label(label) for label in names]


def read_poscar_seekpath_structure(poscar: PathLike):
    """Read a POSCAR into the ``(cell, positions, numbers)`` tuple SeekPath expects."""
    lines = [line.strip() for line in Path(poscar).read_text().splitlines() if line.strip()]
    scale = float(lines[1].split()[0])
    cell = np.array([[float(value) for value in lines[index].split()[:3]] for index in range(2, 5)])
    cell *= scale

    symbol_or_count = lines[5].split()
    try:
        counts = [int(value) for value in symbol_or_count]
        coord_line_index = 6
    except ValueError:
        counts = [int(value) for value in lines[6].split()]
        coord_line_index = 7

    if lines[coord_line_index].lower().startswith("s"):
        coord_line_index += 1

    coord_mode = lines[coord_line_index].lower()
    first_position = coord_line_index + 1
    natoms = sum(counts)
    positions = np.array(
        [[float(value) for value in lines[first_position + index].split()[:3]] for index in range(natoms)]
    )

    if coord_mode.startswith(("c", "k")):
        positions = positions * scale @ np.linalg.inv(cell)

    numbers = []
    for species_index, count in enumerate(counts, start=1):
        numbers.extend([species_index] * count)

    return cell, positions, numbers


def seekpath_points(poscar: PathLike, **seekpath_kwargs) -> Dict[str, Sequence[float]]:
    """Return SeekPath high-symmetry point coordinates for a POSCAR."""
    structure = read_poscar_seekpath_structure(poscar)
    return seekpath.get_path(structure, **seekpath_kwargs)["point_coords"]


def named_k_path(
    point_names: Union[str, Iterable[str]],
    n_points: int,
    poscar: PathLike,
    **seekpath_kwargs,
) -> KPath:
    """Build a k-path from SeekPath point names.

    Returns a :class:`KPath` containing ``sym``, ``path``, and ``labels``.
    The object can still be unpacked as ``sym, path, labels`` for legacy code.
    """
    requested = parse_point_names(point_names)
    point_coords = seekpath_points(poscar, **seekpath_kwargs)
    missing = [label for label in requested if label not in point_coords]
    if missing:
        available = ", ".join(sorted(point_coords))
        raise KeyError(f"Unknown SeekPath point(s) {missing}. Available points: {available}")

    custom_points = [point_coords[label] for label in requested]
    sym, path = path_create(n_points, custom_points)
    labels = [_display_label(label) for label in requested]
    return KPath(sym=sym, path=path, labels=labels, names=requested, point_coords=point_coords)
