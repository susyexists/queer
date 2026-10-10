"""OpenMX PAO radial data and its real spherical-harmonic conventions.

PAO tables contain log(r), r in Bohr, and successive radial functions R_l(r).
They are read directly, without synthesizing orbitals from angular labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import factorial
from pathlib import Path

import numpy as np
from scipy.integrate import simpson
from scipy.special import lpmv, spherical_jn

from .openmx import BOHR_ANGSTROM

HARMONIC_NAMES = {
    0: ("s",),
    1: ("px", "py", "pz"),
    2: ("dz2", "dx2-y2", "dxy", "dxz", "dyz"),
    3: ("fz3", "fxz2", "fyz2", "fz(x2-y2)", "fxyz", "fx(x2-3y2)", "fy(3x2-y2)"),
}
for _l in range(4, 11):
    HARMONIC_NAMES[_l] = (f"Y{_l},0",) + tuple(
        name for m in range(1, _l + 1) for name in (f"Y{_l},{m},cos", f"Y{_l},{m},sin")
    )


@dataclass(frozen=True)
class PAORadialData:
    """Numerical radial functions indexed by angular momentum and zero-based zeta."""

    radii: np.ndarray
    radial: dict[int, np.ndarray]
    source: Path | None = None

    def __post_init__(self):
        radii = np.array(self.radii, dtype=float, copy=True)
        if radii.ndim != 1 or len(radii) < 3 or not np.isfinite(radii).all():
            raise ValueError("PAO radii must contain at least three finite points")
        if np.any(radii < 0) or np.any(np.diff(radii) <= 0):
            raise ValueError("PAO radii must be nonnegative and strictly increasing")
        radial = {}
        for ell, functions in self.radial.items():
            array = np.array(functions, dtype=float, copy=True)
            if not isinstance(ell, int) or ell < 0 or array.ndim != 2 or array.shape[0] != len(radii):
                raise ValueError("PAO radial functions must have shape (Nr,Nzeta) for nonnegative l")
            if array.shape[1] < 1 or not np.isfinite(array).all():
                raise ValueError("PAO radial functions must be nonempty and finite")
            array.setflags(write=False)
            radial[ell] = array
        if not radial:
            raise ValueError("PAO data must contain radial functions")
        radii.setflags(write=False)
        object.__setattr__(self, "radii", radii)
        object.__setattr__(self, "radial", radial)

    def fourier(self, q, ell, zeta):
        """Return integral r² R_l(r) j_l(qr) dr; q is in inverse angstroms.

        Stored atomic-unit radial functions are converted to Å^(-3/2), preserving
        their actual normalization and signs. The result has units Å^(3/2).
        """
        q = np.atleast_1d(np.asarray(q, dtype=float))
        if q.ndim != 1 or not np.isfinite(q).all() or np.any(q < 0):
            raise ValueError("radial Fourier momenta must be finite and nonnegative")
        if ell not in self.radial or not 0 <= zeta < self.radial[ell].shape[1]:
            raise ValueError(f"PAO lacks l={ell}, zeta={zeta}")
        radii = self.radii * BOHR_ANGSTROM
        radial = self.radial[ell][:, zeta] / BOHR_ANGSTROM**1.5
        return np.asarray(
            [simpson(radii**2 * radial * spherical_jn(ell, value * radii), x=radii) for value in q]
        )


def read_pao(path, required_counts=None):
    """Read native pseudo.atomic.orbitals.L blocks and validate requested counts."""
    source = Path(path).expanduser().resolve()
    lines = source.read_text(encoding="utf-8").splitlines()
    tables, grid, declared_count, declared_lmax, declared_rows = {}, None, None, None, None
    block = None
    rows: list[list[float]] = []
    for line_number, raw in enumerate(lines, 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if fields[0].lower() in {"pao.mul", "pao.lmax", "grid.num.output"} and len(fields) == 2:
            try:
                number = int(fields[1])
            except ValueError as error:
                raise ValueError(f"{source}:{line_number}: invalid PAO header count") from error
            if fields[0].lower() == "pao.mul":
                declared_count = number
            elif fields[0].lower() == "pao.lmax":
                declared_lmax = number
            else:
                declared_rows = number
        opening = re.fullmatch(r"<pseudo\.atomic\.orbitals\.L\s*=\s*(\d+)", line, re.I)
        if opening:
            if block is not None:
                raise ValueError(f"{source}:{line_number}: nested PAO radial blocks")
            block, rows = int(opening.group(1)), []
            if block in tables:
                raise ValueError(f"{source}:{line_number}: duplicate PAO angular block")
            continue
        closing = re.fullmatch(r"pseudo\.atomic\.orbitals\.L\s*=\s*(\d+)>", line, re.I)
        if closing:
            if block is None or int(closing.group(1)) != block or not rows:
                raise ValueError(f"{source}:{line_number}: mismatched or empty PAO block")
            table = np.asarray(rows)
            if table.ndim != 2 or table.shape[1] < 3 or not np.isfinite(table).all():
                raise ValueError(f"{source}: invalid PAO radial table")
            if grid is None:
                grid = table[:, 1]
            elif table.shape[0] != len(grid) or not np.allclose(table[:, 1], grid, rtol=1e-12, atol=0):
                raise ValueError(f"{source}: PAO angular blocks use different radial grids")
            if declared_rows is not None and table.shape[0] != declared_rows:
                raise ValueError(f"{source}: PAO table length disagrees with grid.num.output")
            if declared_count is not None and table.shape[1] - 2 != declared_count:
                raise ValueError(f"{source}: PAO radial count disagrees with PAO.Mul")
            tables[block] = table[:, 2:]
            block = None
            continue
        if block is not None:
            try:
                row = [float(value.replace("D", "E").replace("d", "e")) for value in fields]
            except ValueError as error:
                raise ValueError(f"{source}:{line_number}: invalid PAO radial value") from error
            if rows and len(row) != len(rows[0]):
                raise ValueError(f"{source}:{line_number}: inconsistent PAO radial row width")
            rows.append(row)
    if block is not None:
        raise ValueError(f"{source}: truncated PAO radial block")
    if grid is None:
        raise ValueError(f"{source}: no pseudo.atomic.orbitals.L tables")
    if declared_lmax is not None and set(tables) != set(range(declared_lmax + 1)):
        raise ValueError(f"{source}: PAO angular blocks disagree with PAO.Lmax")
    result = PAORadialData(grid, tables, source)
    for ell, count in (required_counts or {}).items():
        if ell not in result.radial or result.radial[ell].shape[1] < count:
            raise ValueError(f"{source}: insufficient radial functions for l={ell}")
    return result


def real_harmonics(ell, directions):
    """Normalized real harmonics through l=10 in OpenMX's basis order."""
    vectors = np.asarray(directions, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or not np.isfinite(vectors).all():
        raise ValueError("harmonic directions must have shape (N,3) and be finite")
    if ell not in HARMONIC_NAMES:
        raise ValueError("native OpenMX angular momentum must lie within 0..10")
    norm = np.linalg.norm(vectors, axis=1)
    unit = np.divide(vectors, norm[:, None], out=np.zeros_like(vectors), where=norm[:, None] != 0)
    x, y, z = unit.T
    if ell == 0:
        values = np.ones((len(vectors), 1)) / np.sqrt(4 * np.pi)
    elif ell == 1:
        values = np.column_stack((x, y, z)) * np.sqrt(3 / (4 * np.pi))
    elif ell == 2:
        values = np.column_stack(
            (
                np.sqrt(5 / (16 * np.pi)) * (3 * z * z - 1),
                np.sqrt(15 / (16 * np.pi)) * (x * x - y * y),
                np.sqrt(15 / (4 * np.pi)) * x * y,
                np.sqrt(15 / (4 * np.pi)) * x * z,
                np.sqrt(15 / (4 * np.pi)) * y * z,
            )
        )
    elif ell == 3:
        values = np.column_stack(
            (
                np.sqrt(7 / (16 * np.pi)) * z * (5 * z * z - 3),
                np.sqrt(21 / (32 * np.pi)) * x * (5 * z * z - 1),
                np.sqrt(21 / (32 * np.pi)) * y * (5 * z * z - 1),
                np.sqrt(105 / (16 * np.pi)) * z * (x * x - y * y),
                np.sqrt(105 / (4 * np.pi)) * x * y * z,
                np.sqrt(35 / (32 * np.pi)) * x * (x * x - 3 * y * y),
                np.sqrt(35 / (32 * np.pi)) * y * (3 * x * x - y * y),
            )
        )
    else:
        phi = np.arctan2(y, x)
        columns = [np.sqrt((2 * ell + 1) / (4 * np.pi)) * lpmv(0, ell, z)]
        for m in range(1, ell + 1):
            radial_angle = (
                (-1) ** m
                * np.sqrt(2 * (2 * ell + 1) / (4 * np.pi) * factorial(ell - m) / factorial(ell + m))
                * lpmv(m, ell, z)
            )
            columns.extend((radial_angle * np.cos(m * phi), radial_angle * np.sin(m * phi)))
        values = np.column_stack(columns)
    if ell:
        values[norm == 0] = 0
    return values
