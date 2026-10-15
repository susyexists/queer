"""One POSCAR record parser, independent of chemical-species interpretation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoscarRecord:
    comment: str
    cell: np.ndarray
    positions: np.ndarray
    fractional_positions: np.ndarray
    counts: tuple[int, ...]
    symbols: tuple[str, ...] | None
    selective_flags: tuple[tuple[str, ...], ...] | None


def parse_poscar_record(text) -> PoscarRecord:
    """Parse scaling and coordinates without guessing VASP 4 element symbols."""
    lines = str(text).splitlines()
    scales = np.asarray([float(value) for value in lines[1].split()])
    raw_cell = np.asarray([[float(value) for value in line.split()[:3]] for line in lines[2:5]])
    if raw_cell.shape != (3, 3) or not np.isfinite(raw_cell).all():
        raise ValueError("invalid lattice")
    if len(scales) == 1:
        if not np.isfinite(scales[0]) or scales[0] == 0:
            raise ValueError("scale must be nonzero and finite")
        scale = scales[0]
        if scale < 0:
            volume = abs(np.linalg.det(raw_cell))
            if volume == 0:
                raise ValueError("singular lattice")
            scale = (-scale / volume) ** (1 / 3)
        scale = np.repeat(scale, 3)
    elif len(scales) == 3 and np.isfinite(scales).all() and np.all(scales > 0):
        scale = scales
    else:
        raise ValueError("expected one nonzero or three positive scales")
    cell = raw_cell * scale
    if not np.isfinite(cell).all() or abs(np.linalg.det(cell)) < 1e-14:
        raise ValueError("singular or nonfinite lattice")
    line = 5
    tokens = lines[line].split()
    symbols = None
    try:
        counts = tuple(int(value) for value in tokens)
    except ValueError:
        symbols = tuple(tokens)
        line += 1
        counts = tuple(int(value) for value in lines[line].split())
    if not counts or min(counts) < 0 or (symbols is not None and len(symbols) != len(counts)):
        raise ValueError("invalid species or atom counts")
    line += 1
    selective = lines[line].strip().lower().startswith("s")
    if selective:
        line += 1
    mode = lines[line].strip().lower()
    if not mode.startswith(("d", "c", "k")):
        raise ValueError("expected Direct or Cartesian coordinates")
    rows = [lines[line + 1 + index].split() for index in range(sum(counts))]
    coordinates = np.asarray([[float(value) for value in row[:3]] for row in rows]).reshape(-1, 3)
    if coordinates.shape != (sum(counts), 3) or not np.isfinite(coordinates).all():
        raise ValueError("invalid positions")
    if mode.startswith("d"):
        fractional, positions = coordinates, coordinates @ cell
    else:
        positions = coordinates * scale
        fractional = np.linalg.solve(cell.T, positions.T).T
    return PoscarRecord(
        lines[0],
        cell,
        positions,
        fractional,
        counts,
        symbols,
        tuple(tuple(row[3:6]) for row in rows) if selective else None,
    )
