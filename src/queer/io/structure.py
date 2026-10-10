"""POSCAR and Quantum ESPRESSO structure/energy readers."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def read_poscar(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Read VASP POSCAR lattice, fractional positions, and species identifiers.

    Supports scalar/three-component scales, a negative target-volume scale,
    VASP 4/5 species lines, and selective dynamics. Species identifiers label
    equivalent atoms for symmetry finding and are not atomic numbers.
    """
    source = Path(path)
    lines = source.read_text(encoding="utf-8").splitlines()
    try:
        scales = np.asarray([float(value) for value in lines[1].split()])
        raw_cell = np.asarray([[float(value) for value in line.split()[:3]] for line in lines[2:5]])
        if raw_cell.shape != (3, 3) or not np.isfinite(raw_cell).all():
            raise ValueError("invalid lattice")
        if len(scales) == 1:
            if scales[0] == 0:
                raise ValueError("zero scale")
            scale = scales[0]
            if scale < 0:
                volume = abs(np.linalg.det(raw_cell))
                if volume == 0:
                    raise ValueError("singular lattice")
                scale = (-scale / volume) ** (1 / 3)
            scale = np.repeat(scale, 3)
        elif len(scales) == 3 and np.all(scales > 0):
            scale = scales
        else:
            raise ValueError("expected one nonzero or three positive scales")
        cell = raw_cell * scale
        if not np.isfinite(cell).all() or abs(np.linalg.det(cell)) < 1e-14:
            raise ValueError("singular or nonfinite lattice")
        count_index = 5
        try:
            counts = [int(value) for value in lines[count_index].split()]
        except ValueError:
            count_index += 1
            counts = [int(value) for value in lines[count_index].split()]
        if not counts or min(counts) < 0 or sum(counts) < 1:
            raise ValueError("invalid atom counts")
        mode_index = count_index + 1
        if lines[mode_index].strip().lower().startswith("s"):
            mode_index += 1
        mode = lines[mode_index].strip().lower()
        if not mode.startswith(("d", "c", "k")):
            raise ValueError("expected Direct or Cartesian coordinates")
        positions = np.asarray(
            [
                [float(value) for value in lines[mode_index + 1 + index].split()[:3]]
                for index in range(sum(counts))
            ]
        )
        if positions.shape != (sum(counts), 3) or not np.isfinite(positions).all():
            raise ValueError("invalid positions")
        if mode.startswith(("c", "k")):
            positions = np.linalg.solve(cell.T, (positions * scale).T).T
    except (IndexError, ValueError, np.linalg.LinAlgError) as error:
        raise ValueError(f"{source}: malformed POSCAR ({error})") from error
    types = [species for species, count in enumerate(counts, 1) for _ in range(count)]
    return cell, positions, types


def read_fermi_energy(path: str | Path) -> float:
    """Read the last explicit QE Fermi-energy record in eV."""
    energies = []
    pattern = re.compile(r"the Fermi energy is\s+([-+\d.eEdD]+)")
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            match = pattern.search(line)
            if match:
                energies.append(float(match.group(1).replace("D", "E").replace("d", "e")))
    if not energies or not np.isfinite(energies[-1]):
        raise ValueError(f"{path}: no finite Fermi-energy record")
    return energies[-1]


def read_qe_reciprocal(path: str | Path) -> np.ndarray:
    """Read QE reciprocal vectors (2π/alat) into row vectors in Å⁻¹."""
    alat = None
    vectors = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if "lattice parameter (alat)" in line:
                match = re.search(r"=\s*([-+\d.eEdD]+)", line)
                if match:
                    alat = float(match.group(1).replace("D", "E").replace("d", "e"))
            match = re.search(r"b\(\s*([123])\s*\)\s*=\s*\(\s*([^)]*)\)", line)
            if match:
                vectors[int(match.group(1))] = [float(value) for value in match.group(2).split()]
                if len(vectors) == 3 and alat is not None:
                    break
    if alat is None or not np.isfinite(alat) or alat <= 0 or len(vectors) != 3:
        raise ValueError(f"{path}: incomplete QE reciprocal lattice or lattice parameter")
    lattice = np.asarray([vectors[index] for index in (1, 2, 3)], dtype=float)
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        raise ValueError(f"{path}: invalid QE reciprocal lattice")
    return lattice * (2 * np.pi / (alat * 0.529177210903))
