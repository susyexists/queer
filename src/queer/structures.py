"""Shared atomistic value types with explicit Å/eV/amu/fs units."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

from .elements import ATOMIC_MASSES, SYMBOLS


@dataclass(init=False)
class Structure:
    cell: np.ndarray
    positions: np.ndarray
    atomic_numbers: np.ndarray
    masses: np.ndarray
    pbc: tuple[bool, bool, bool] = (True, True, True)
    magnetic_moments: np.ndarray | None = None
    fixed_coordinates: np.ndarray | None = None
    arrays: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __init__(
        self,
        cell: ArrayLike,
        positions: ArrayLike,
        atomic_numbers: ArrayLike,
        pbc: bool | Iterable[bool] = (True, True, True),
        masses: ArrayLike | None = None,
        magnetic_moments: ArrayLike | None = None,
        fixed_coordinates: ArrayLike | None = None,
        arrays: Mapping[str, ArrayLike] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ):
        self.cell = np.array(cell, dtype=float, copy=True)
        self.positions = np.array(positions, dtype=float, copy=True).reshape(-1, 3)
        raw = np.asarray(atomic_numbers)
        if raw.ndim != 1 or not np.equal(raw, np.floor(raw)).all():
            raise ValueError("atomic_numbers must be a one-dimensional integer array")
        self.atomic_numbers = raw.astype(int, copy=True)
        if self.cell.shape != (3, 3) or not np.isfinite(self.cell).all():
            raise ValueError("cell must be a finite (3,3) array of row vectors")
        if not np.isfinite(self.positions).all() or len(self.positions) != len(raw):
            raise ValueError("positions must be finite and match atomic_numbers")
        if np.any((raw < 1) | (raw >= len(SYMBOLS))):
            raise ValueError("atomic_numbers must be between 1 and 118")
        if isinstance(pbc, (bool, np.bool_)):
            self.pbc = (bool(pbc),) * 3
        else:
            values = tuple(bool(value) for value in pbc)
            if len(values) != 3:
                raise ValueError("pbc must have three components")
            self.pbc = (values[0], values[1], values[2])
        if any(self.pbc) and (abs(np.linalg.det(self.cell)) < 1e-14 or np.linalg.cond(self.cell) > 1e12):
            raise ValueError("periodic structures require a nonsingular cell")
        self.masses = ATOMIC_MASSES[self.atomic_numbers].copy() if masses is None else np.array(masses, float)
        if (
            self.masses.shape != (self.natoms,)
            or not np.isfinite(self.masses).all()
            or np.any(self.masses <= 0)
        ):
            raise ValueError("masses must be finite positive amu, one per atom")
        self.magnetic_moments = None
        self.fixed_coordinates = None
        for name, value, dtype in (
            ("magnetic_moments", magnetic_moments, float),
            ("fixed_coordinates", fixed_coordinates, bool),
        ):
            if value is not None:
                value = np.array(value, dtype=dtype, copy=True)
                if value.shape != (self.natoms, 3) or not np.isfinite(value).all():
                    raise ValueError(f"{name} must have shape (natoms,3)")
                setattr(self, name, value)
        self.arrays = {key: np.array(value, copy=True) for key, value in (arrays or {}).items()}
        if any(value.ndim == 0 or len(value) != self.natoms for value in self.arrays.values()):
            raise ValueError("per-atom arrays must have natoms rows")
        self.metadata = deepcopy(dict(metadata or {}))

    @property
    def natoms(self):
        return len(self.atomic_numbers)

    @property
    def symbols(self):
        return [SYMBOLS[int(number)] for number in self.atomic_numbers]

    @property
    def fractional_positions(self):
        return np.linalg.solve(self.cell.T, self.positions.T).T

    def copy(self, **changes):
        return replace(self, **changes)


@dataclass
class Evaluation:
    energy_eV: float
    forces_eV_A: np.ndarray
    stress_eV_A3: np.ndarray | None = None

    def __post_init__(self):
        self.energy_eV = float(self.energy_eV)
        self.forces_eV_A = np.array(self.forces_eV_A, float, copy=True)
        if not np.isfinite(self.energy_eV) or self.forces_eV_A.ndim != 2 or self.forces_eV_A.shape[1] != 3:
            raise ValueError("evaluation needs a finite energy and (natoms,3) forces")
        if not np.isfinite(self.forces_eV_A).all():
            raise ValueError("forces must be finite")
        if self.stress_eV_A3 is not None:
            self.stress_eV_A3 = np.array(self.stress_eV_A3, float, copy=True)
            if self.stress_eV_A3.shape != (3, 3) or not np.isfinite(self.stress_eV_A3).all():
                raise ValueError("stress must be a finite (3,3) tensor in eV/Å³")


@dataclass
class Frame:
    structure: Structure
    velocities: np.ndarray | None = None
    step: int | None = None
    time_fs: float | None = None
    evaluation: Evaluation | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.velocities is not None:
            self.velocities = np.array(self.velocities, float, copy=True)
            if self.velocities.shape != (self.structure.natoms, 3) or not np.isfinite(self.velocities).all():
                raise ValueError("velocities must be finite (natoms,3) in Å/fs")
        if self.evaluation is not None and len(self.evaluation.forces_eV_A) != self.structure.natoms:
            raise ValueError("evaluation forces must match frame atoms")
        if self.step is not None:
            if isinstance(self.step, bool) or not np.isfinite(self.step) or int(self.step) != self.step:
                raise ValueError("frame step must be an integer")
            self.step = int(self.step)
        if self.time_fs is not None:
            self.time_fs = float(self.time_fs)
            if not np.isfinite(self.time_fs):
                raise ValueError("frame time_fs must be finite")
        self.metadata = dict(self.metadata or {})


@runtime_checkable
class ForceProvider(Protocol):
    def evaluate(self, structure: Structure) -> Evaluation: ...
