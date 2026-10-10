"""Validated native readers for Wannier90 Hamiltonians and position operators."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..geometry.lattice import validated_lattice
from .wannier_ws import ComputedWignerSeitzShifts
from .wannier_ws import recompute_wsvec as recompute_wsvec

BOHR_ANGSTROM = 0.529177210903


@dataclass(frozen=True)
class WannierData:
    translations: np.ndarray
    hopping: np.ndarray
    degeneracies: np.ndarray
    position: np.ndarray | None = None
    lattice: np.ndarray | None = None

    @property
    def nbnd(self) -> int:
        return self.hopping.shape[1]


@dataclass(frozen=True)
class WignerSeitzData:
    shifts: Mapping[tuple[int, int, int, int, int], np.ndarray]
    use_ws_distance: bool
    ndegen_applied: bool


def resolve_wannier_files(
    directory,
    hr="wannier90_hr.dat",
    *,
    tb=None,
    centres=None,
    wout=None,
    poscar=None,
    wsvec=None,
    interpolation="standard",
) -> dict[str, Path]:
    """Inventory effective native inputs without parsing large operator files.

    This resolver mirrors loader sidecar discovery so durable cache keys include
    every consumed file, including companions absent from a family manifest.
    Explicit missing paths are retained to produce actionable file errors.
    """
    root = Path(directory).expanduser().resolve()
    if hr is None:
        if tb is None:
            raise ValueError("Wannier loading requires hr or tb")
        hr = tb
    hr_path = (root / hr).resolve()
    seed = hr_path.name.removesuffix("_hr.dat").removesuffix("_tb.dat")
    tb_path = (root / tb).resolve() if tb is not None else root / f"{seed}_tb.dat"
    if tb is not None and not hr_path.exists():
        seed = tb_path.name.removesuffix("_tb.dat")
    if hr_path.name.endswith("_tb.dat"):
        tb_path = hr_path
    files = {}
    if hr_path.exists() and not hr_path.name.endswith("_tb.dat"):
        files["hr"] = hr_path
    if tb is not None or tb_path.exists():
        files["tb"] = tb_path.resolve()
    if not files:
        files["hr"] = hr_path
    if poscar is not None:
        files["poscar"] = (root / poscar).resolve()
    wout_path = (root / wout).resolve() if wout is not None else root / f"{seed}.wout"
    if wout is not None or ("tb" not in files and poscar is None and wout_path.exists()):
        files["wout"] = wout_path.resolve()
    centres_path = (root / centres).resolve() if centres is not None else root / f"{seed}_centres.xyz"
    if centres is not None or centres_path.exists():
        files["centres"] = centres_path.resolve()
    if interpolation in {"ws_distance", "auto"}:
        files["wsvec"] = ((root / wsvec) if wsvec is not None else root / f"{seed}_wsvec.dat").resolve()
    return files


def _integer(token: str, context: str) -> int:
    try:
        return int(token)
    except ValueError as error:
        raise ValueError(f"{context}: expected an integer, got {token!r}") from error


def read_hr(path: str | Path) -> WannierData:
    """Read by explicit orbital indices, preserving the HR degeneracy block."""
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        try:
            next(stream)  # Human-readable creation header.
            nbnd = _integer(next(stream).strip(), str(source))
            nrpts = _integer(next(stream).strip(), str(source))
        except StopIteration as error:
            raise ValueError(f"{source}: incomplete HR header") from error
        if nbnd < 1 or nrpts < 1:
            raise ValueError(f"{source}: band and translation counts must be positive")
        degeneracies: list[int] = []
        while len(degeneracies) < nrpts:
            line = stream.readline()
            if not line:
                raise ValueError(f"{source}: truncated HR degeneracy block")
            degeneracies.extend(_integer(value, str(source)) for value in line.split())
        if len(degeneracies) != nrpts or min(degeneracies) < 1:
            raise ValueError(f"{source}: expected {nrpts} positive degeneracies")

        translations: list[tuple[int, ...]] = []
        hopping = np.empty((nrpts, nbnd, nbnd), dtype=np.complex128)
        seen = np.zeros((nrpts, nbnd, nbnd), dtype=bool)
        lookup = {}
        rows = 0
        for line_number, line in enumerate(stream, start=4 + int(np.ceil(nrpts / 15))):
            if not line.strip():
                continue
            fields = line.split()
            context = f"{source}:{line_number}"
            if len(fields) != 7:
                raise ValueError(f"{context}: expected R1 R2 R3 m n real imaginary")
            translation = tuple(_integer(value, context) for value in fields[:3])
            m, n = (_integer(value, context) - 1 for value in fields[3:5])
            if not 0 <= m < nbnd or not 0 <= n < nbnd:
                raise ValueError(f"{context}: orbital index outside 1..{nbnd}")
            if translation not in lookup:
                if len(translations) >= nrpts:
                    raise ValueError(f"{context}: too many distinct lattice translations")
                lookup[translation] = len(translations)
                translations.append(translation)
            index = lookup[translation]
            if seen[index, m, n]:
                raise ValueError(f"{context}: duplicate translation/orbital entry")
            try:
                value = complex(
                    float(fields[5].replace("D", "E").replace("d", "e")),
                    float(fields[6].replace("D", "E").replace("d", "e")),
                )
            except ValueError as error:
                raise ValueError(f"{context}: invalid hopping matrix element") from error
            if not np.isfinite(value):
                raise ValueError(f"{context}: nonfinite hopping matrix element")
            hopping[index, m, n] = value
            seen[index, m, n] = True
            rows += 1
        if rows != nrpts * nbnd**2 or not seen.all():
            raise ValueError(f"{source}: incomplete HR matrix entries")
    arrays = (np.asarray(translations, dtype=np.int64), hopping, np.asarray(degeneracies, dtype=np.int64))
    for array in arrays:
        array.setflags(write=False)
    return WannierData(*arrays)


def read_wsvec(path: str | Path) -> WignerSeitzData:
    """Read integer superlattice shifts keyed by (R1,R2,R3,m,n)."""
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        header = stream.readline().lower()
        match = re.search(r"use_ws_distance\s*=\s*\.?(true|false)\.?", header)
        if not match:
            raise ValueError(f"{source}: wsvec header must specify use_ws_distance")
        applied = re.search(r"write_ndegen_applied\s*=\s*\.?true\.?", header) is not None
        lines = iter(
            (line_number, line.strip()) for line_number, line in enumerate(stream, 2) if line.strip()
        )
        shifts = {}
        for line_number, line in lines:
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{source}:{line_number}: expected R1 R2 R3 m n")
            r1, r2, r3, m, n = (_integer(value, str(source)) for value in fields)
            key = (r1, r2, r3, m, n)
            if key in shifts:
                raise ValueError(f"{source}:{line_number}: duplicate wsvec entry")
            try:
                _, count_line = next(lines)
                count = _integer(count_line, str(source))
                if count < 1:
                    raise ValueError(f"{source}: wsvec shift count must be positive")
                entries = []
                for _ in range(count):
                    shift_number, shift_line = next(lines)
                    values = shift_line.split()
                    if len(values) != 3:
                        raise ValueError(f"{source}:{shift_number}: expected three shift integers")
                    entries.append([_integer(value, str(source)) for value in values])
            except StopIteration as error:
                raise ValueError(f"{source}: truncated wsvec entry") from error
            if len(set(map(tuple, entries))) != count:
                raise ValueError(f"{source}:{line_number}: duplicate equivalent shifts")
            shifts[key] = np.asarray(entries, dtype=np.int64)
    return WignerSeitzData(shifts, match.group(1) == "true", applied)


def _number(token: str, context: str) -> float:
    try:
        value = float(token.replace("D", "E").replace("d", "e"))
    except ValueError as error:
        raise ValueError(f"{context}: invalid number {token!r}") from error
    if not np.isfinite(value):
        raise ValueError(f"{context}: nonfinite number")
    return value


def read_tb(path: str | Path) -> WannierData:
    """Read lattice, HR blocks and optional full Cartesian position blocks.

    Both operators remain raw: degeneracy and WSVec weights are applied by
    :func:`normalized_operators`. Orbital rows and position blocks may be shuffled.
    A present but incomplete position section is an error.
    """
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        stream.readline()  # creation header
        lines = iter((i, line.strip()) for i, line in enumerate(stream, 2) if line.strip())

        def take(what):
            try:
                number, line = next(lines)
            except StopIteration as error:
                raise ValueError(f"{source}: incomplete TB {what}") from error
            return f"{source}:{number}", line.split()

        lattice_rows = []
        for _ in range(3):
            context, fields = take("lattice")
            if len(fields) != 3:
                raise ValueError(f"{context}: expected three lattice components")
            lattice_rows.append([_number(field, context) for field in fields])
        lattice = validated_lattice(lattice_rows).copy()
        context, fields = take("band count")
        if len(fields) != 1:
            raise ValueError(f"{context}: expected one band count")
        nbnd = _integer(fields[0], context)
        context, fields = take("translation count")
        if len(fields) != 1:
            raise ValueError(f"{context}: expected one translation count")
        nrpts = _integer(fields[0], context)
        if min(nbnd, nrpts) < 1:
            raise ValueError(f"{source}: band and translation counts must be positive")
        degeneracies: list[int] = []
        while len(degeneracies) < nrpts:
            context, fields = take("degeneracy block")
            degeneracies.extend(_integer(field, context) for field in fields)
        if len(degeneracies) != nrpts or min(degeneracies) < 1:
            raise ValueError(f"{source}: expected {nrpts} positive degeneracies")
        translations = []
        lookup = {}
        hopping = np.empty((nrpts, nbnd, nbnd), complex)

        def block(array, index, columns):
            seen = np.zeros((nbnd, nbnd), bool)
            for _ in range(nbnd**2):
                context, fields = take("matrix entries")
                if len(fields) != columns:
                    raise ValueError(f"{context}: expected {columns} fields per matrix entry")
                m, n = (_integer(field, context) - 1 for field in fields[:2])
                if not (0 <= m < nbnd and 0 <= n < nbnd):
                    raise ValueError(f"{context}: orbital index outside 1..{nbnd}")
                if seen[m, n]:
                    raise ValueError(f"{context}: duplicate orbital entry")
                seen[m, n] = True
                values = [_number(field, context) for field in fields[2:]]
                complex_values = np.asarray(values[::2]) + 1j * np.asarray(values[1::2])
                array[index, m, n] = complex_values[0] if columns == 4 else complex_values

        for index in range(nrpts):
            context, fields = take("Hamiltonian block header")
            if len(fields) != 3:
                raise ValueError(f"{context}: expected three translation integers")
            translation = tuple(_integer(field, context) for field in fields)
            if translation in lookup:
                raise ValueError(f"{context}: duplicate Hamiltonian translation")
            lookup[translation] = index
            translations.append(translation)
            block(hopping, index, 4)
        position = None
        first_position = next(lines, None)
        if first_position is not None:
            position = np.empty((nrpts, nbnd, nbnd, 3), complex)
            seen_translations = set()
            for count in range(nrpts):
                if count == 0:
                    line_number, line = first_position
                    context, fields = f"{source}:{line_number}", line.split()
                else:
                    context, fields = take("position block header")
                if len(fields) != 3:
                    raise ValueError(f"{context}: expected three translation integers")
                translation = tuple(_integer(field, context) for field in fields)
                if translation not in lookup or translation in seen_translations:
                    raise ValueError(f"{context}: unexpected or duplicate position translation")
                seen_translations.add(translation)
                block(position, lookup[translation], 8)
            if next(lines, None) is not None:
                raise ValueError(f"{source}: unexpected trailing TB entries")
    translation_array = np.asarray(translations, np.int64)
    degeneracy_array = np.asarray(degeneracies, np.int64)
    for array in (translation_array, hopping, degeneracy_array, lattice, position):
        if array is not None:
            array.setflags(write=False)
    return WannierData(translation_array, hopping, degeneracy_array, position, lattice)


def read_centres_xyz(path: str | Path) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    """Read all XYZ records, returning ``X`` Wannier centres and atom records in Å."""
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        count = _integer(stream.readline().strip(), str(source))
        if count < 1 or not stream.readline():
            raise ValueError(f"{source}: incomplete centres XYZ header")
        centres, atoms = [], []
        for index in range(count):
            fields = stream.readline().split()
            context = f"{source}:{index + 3}"
            if len(fields) != 4:
                raise ValueError(f"{context}: expected symbol and three Cartesian coordinates")
            point = np.asarray([_number(field, context) for field in fields[1:]])
            if fields[0] == "X":
                centres.append(point)
            else:
                atoms.append((fields[0], point))
        if any(line.strip() for line in stream):
            raise ValueError(f"{source}: extra XYZ records exceed the announced count")
    result = np.asarray(centres, float).reshape(-1, 3)
    result.setflags(write=False)
    return result, atoms


def read_wout_lattice(path: str | Path) -> np.ndarray:
    """Read the labelled lattice block in Å, converting explicit Bohr blocks."""
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if "lattice vectors" not in line.lower():
                continue
            if "bohr" in line.lower():
                scale = BOHR_ANGSTROM
            elif "ang" in line.lower():
                scale = 1.0
            else:
                raise ValueError(f"{source}: lattice block must declare Angstrom or Bohr units")
            rows: list[list[float]] = []
            for line in stream:
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 4 or fields[0].lower() != f"a_{len(rows) + 1}":
                    raise ValueError(f"{source}: incomplete or unordered lattice vectors block")
                rows.append([_number(field, str(source)) for field in fields[1:]])
                if len(rows) == 3:
                    return validated_lattice(np.asarray(rows) * scale).copy()
            raise ValueError(f"{source}: incomplete lattice vectors block")
    raise ValueError(f"{source}: no lattice vectors block found")


def normalized_hopping(
    data: WannierData, mode: str, wsvec: WignerSeitzData | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Apply HR/WS weights once and produce a plain positive-phase Fourier sum."""
    return _normalized_matrix(data, data.hopping, mode, wsvec)


def normalized_operators(
    data: WannierData, mode: str, wsvec: WignerSeitzData | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Apply identical translation/image weights to Hamiltonian and position data."""
    if data.position is None:
        translations, hopping = normalized_hopping(data, mode, wsvec)
        return translations, hopping, None
    joined = np.concatenate((data.hopping[..., None], data.position), axis=-1)
    translations, operators = _normalized_matrix(data, joined, mode, wsvec)
    return translations, operators[..., 0], operators[..., 1:]


def _normalized_matrix(data, matrices, mode, wsvec):
    if mode not in {"standard", "ws_distance", "ndegen_applied"}:
        raise ValueError(f"Unknown interpolation mode {mode!r}")
    if mode == "ndegen_applied":
        if not np.all(data.degeneracies == 1):
            raise ValueError("ndegen_applied HR exports must contain unit degeneracies")
        return data.translations, matrices
    if mode == "standard":
        weights = data.degeneracies.reshape((-1,) + (1,) * (matrices.ndim - 1))
        return data.translations, matrices / weights
    if wsvec is None or not wsvec.use_ws_distance:
        raise ValueError("ws_distance interpolation requires a matching enabled wsvec file")
    if wsvec.ndegen_applied:
        raise ValueError("wsvec shifts have already been applied; select ndegen_applied")
    if isinstance(wsvec.shifts, ComputedWignerSeitzShifts):
        return wsvec.shifts.normalized_matrices(data, matrices)
    expected = {
        (*map(int, translation), m + 1, n + 1)
        for translation in data.translations
        for m in range(data.nbnd)
        for n in range(data.nbnd)
    }
    if set(wsvec.shifts) != expected:
        raise ValueError("wsvec translation/orbital keys do not match the HR file")
    expanded = {}
    for index, translation in enumerate(data.translations):
        for m in range(data.nbnd):
            for n in range(data.nbnd):
                shifts = wsvec.shifts[(*map(int, translation), m + 1, n + 1)]
                value = matrices[index, m, n] / (data.degeneracies[index] * len(shifts))
                for shift in shifts:
                    key = tuple(map(int, translation + shift))
                    if key not in expanded:
                        expanded[key] = np.zeros(matrices.shape[1:], dtype=np.complex128)
                    expanded[key][m, n] += value
    keys = sorted(expanded)
    return np.asarray(keys, dtype=np.int64), np.asarray([expanded[key] for key in keys])
