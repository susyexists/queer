"""POSCAR and Quantum ESPRESSO structure/energy readers."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..physics.constants import WANNIER_BOHR_ANGSTROM
from ._poscar import parse_poscar_record


def read_poscar(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Read VASP POSCAR lattice, fractional positions, and species identifiers.

    Supports scalar/three-component scales, a negative target-volume scale,
    VASP 4/5 species lines, and selective dynamics. Species identifiers label
    equivalent atoms for symmetry finding and are not atomic numbers.
    """
    source = Path(path)
    try:
        record = parse_poscar_record(source.read_text(encoding="utf-8"))
        if sum(record.counts) < 1:
            raise ValueError("invalid atom counts")
    except (IndexError, ValueError, np.linalg.LinAlgError) as error:
        raise ValueError(f"{source}: malformed POSCAR ({error})") from error
    types = [species for species, count in enumerate(record.counts, 1) for _ in range(count)]
    return record.cell, record.fractional_positions, types


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
    return lattice * (2 * np.pi / (alat * WANNIER_BOHR_ANGSTROM))
