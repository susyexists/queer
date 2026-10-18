"""Validated native readers for Wannier90 Hamiltonians and position operators."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from types import MappingProxyType

import numpy as np

from ..geometry.lattice import validated_lattice
from ..physics.constants import WANNIER_BOHR_ANGSTROM as BOHR_ANGSTROM
from .wannier_ws import ComputedWignerSeitzShifts, PackedWignerSeitzShifts
from .wannier_ws import fold_vacuum_axes as fold_vacuum_axes
from .wannier_ws import periodic_dirs_from_table as periodic_dirs_from_table
from .wannier_ws import recompute_wsvec as recompute_wsvec
from .wannier_ws import unfold_from_wsvec as unfold_from_wsvec

_FLOAT_TOKEN = r"[+\-]?(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+\-]?[0-9]+)?|inf(?:inity)?|nan)"
_WIN_COMMENT = re.compile(r"[!#].*$")
_WIN_KEYWORD = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*[=:]?\s*(.*)$")
_WIN_TRUE = frozenset({"true", ".true.", "t", ".t.", "1", "yes"})
_WIN_FALSE = frozenset({"false", ".false.", "f", ".f.", "0", "no"})
_WIN_UNITS = ("bohr", "ang", "angstrom")


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
    position="auto",
    win=None,
) -> dict[str, Path]:
    """Inventory effective native inputs without parsing large operator files.

    This resolver mirrors loader sidecar discovery so durable cache keys include
    every consumed file, including companions absent from a family manifest.
    Explicit missing paths are retained to produce actionable file errors.
    With ``position="diagonal"`` a TB file is consumed only as the Hamiltonian
    source, so a companion ``_tb.dat`` is neither discovered nor accepted. A
    ``.win`` file is consumed only when named and then supplies the lattice.
    """
    if position not in {"auto", "full", "diagonal"}:
        raise ValueError("position must be auto, full or diagonal")
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
    hr_is_tb = hr_path.name.endswith("_tb.dat")
    if hr_is_tb:
        tb_path = hr_path
    if position == "diagonal":
        if tb is not None and hr_path.exists() and not hr_is_tb:
            raise ValueError("position='diagonal' reads no TB position data; omit tb when an HR file is used")
        consume_tb = hr_is_tb or (tb is not None and not hr_path.exists())
    else:
        consume_tb = tb is not None or tb_path.exists()
    files = {}
    if hr_path.exists() and not hr_is_tb:
        files["hr"] = hr_path
    if consume_tb:
        files["tb"] = tb_path.resolve()
    if not files:
        files["hr"] = hr_path
    if poscar is not None:
        files["poscar"] = (root / poscar).resolve()
    if win is not None:
        files["win"] = (root / win).resolve()
    wout_path = (root / wout).resolve() if wout is not None else root / f"{seed}.wout"
    if wout is not None or ("tb" not in files and win is None and poscar is None and wout_path.exists()):
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


def _integer_tokens(body, context):
    """Strict bulk integer conversion, avoiding numpy's permissive edge cases."""
    if isinstance(body, str):
        try:
            body = body.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError(f"{context}: expected integer data") from error
    if re.search(rb"[^0-9+\-\s]|(?<=[0-9])[+\-]|[+\-](?![0-9])", body):
        raise ValueError(f"{context}: expected integer data")
    for token in re.findall(rb"[+\-]?[0-9]{19,}", body):
        value = int(token)
        if not np.iinfo(np.int64).min <= value <= np.iinfo(np.int64).max:
            raise ValueError(f"{context}: integer outside int64 range")
    try:
        return np.fromstring(body.decode("ascii"), dtype=np.int64, sep=" ")
    except ValueError as error:
        raise ValueError(f"{context}: expected integer data") from error


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
        line_number = 3
        while len(degeneracies) < nrpts:
            line = stream.readline()
            line_number += 1
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
        while lines := list(islice(stream, 32768)):
            fields = [line.split() for line in lines if line.strip()]
            context = f"{source}:{line_number + 1}"
            if any(len(row) != 7 for row in fields):
                raise ValueError(f"{context}: expected R1 R2 R3 m n real imaginary")
            line_number += len(lines)
            if not fields:
                continue
            integers = _integer_tokens(" ".join(" ".join(row[:5]) for row in fields), context).reshape(-1, 5)
            number_text = " ".join(" ".join(row[5:]) for row in fields).replace("D", "E").replace("d", "e")
            if re.search(r"(?<!\S)(?!" + _FLOAT_TOKEN + r"(?=\s|$))\S+", number_text, re.IGNORECASE):
                raise ValueError(f"{context}: invalid hopping matrix element")
            try:
                numbers = np.fromstring(
                    number_text,
                    dtype=float,
                    sep=" ",
                ).reshape(-1, 2)
            except ValueError as error:
                raise ValueError(f"{context}: invalid hopping matrix element") from error
            if numbers.shape != (len(fields), 2):
                raise ValueError(f"{context}: invalid hopping matrix element")
            if not np.isfinite(numbers).all():
                raise ValueError(f"{context}: nonfinite hopping matrix element")
            m, n = integers[:, 3] - 1, integers[:, 4] - 1
            if np.any(m < 0) or np.any(m >= nbnd) or np.any(n < 0) or np.any(n >= nbnd):
                raise ValueError(f"{context}: orbital index outside 1..{nbnd}")
            vectors, first, inverse = np.unique(
                integers[:, :3], axis=0, return_index=True, return_inverse=True
            )
            for item in np.argsort(first):
                translation = tuple(vectors[item])
                if translation not in lookup:
                    if len(translations) >= nrpts:
                        raise ValueError(f"{context}: too many distinct lattice translations")
                    lookup[translation] = len(translations)
                    translations.append(translation)
            index = np.array([lookup[tuple(row)] for row in vectors], dtype=np.int64)[inverse]
            flat = (index * nbnd + m) * nbnd + n
            if len(np.unique(flat)) != len(flat) or np.any(seen.reshape(-1)[flat]):
                raise ValueError(f"{context}: duplicate translation/orbital entry")
            values = np.empty(len(fields), complex)
            values.real, values.imag = numbers[:, 0], numbers[:, 1]
            hopping.reshape(-1)[flat] = values
            seen.reshape(-1)[flat] = True
            rows += len(fields)
        if rows != nrpts * nbnd**2 or not seen.all():
            raise ValueError(f"{source}: incomplete HR matrix entries")
    arrays = (np.asarray(translations, dtype=np.int64), hopping, np.asarray(degeneracies, dtype=np.int64))
    for array in arrays:
        array.setflags(write=False)
    return WannierData(*arrays)


def read_wsvec(path: str | Path) -> WignerSeitzData:
    """Read integer superlattice shifts keyed by (R1,R2,R3,m,n)."""
    source = Path(path)
    entries, images = [], []
    current, remaining = None, 0
    line_offset = 1
    with source.open("rb") as stream:
        header = stream.readline().decode("utf-8").lower()
        match = re.search(r"use_ws_distance\s*=\s*\.?(true|false)\.?", header)
        if not match:
            raise ValueError(f"{source}: wsvec header must specify use_ws_distance")
        applied = re.search(r"write_ndegen_applied\s*=\s*\.?true\.?", header) is not None
        while lines := list(islice(stream, 65536)):
            widths = np.fromiter((len(line.split()) for line in lines), dtype=np.int64)
            active = np.flatnonzero(widths)
            numbers = active + line_offset + 1
            widths = widths[active]
            body = b"".join(lines)
            tokens = _integer_tokens(body, source)
            records = np.empty((np.count_nonzero(widths == 5) + 1, 6), dtype=np.int64)
            shifts = np.empty((np.count_nonzero(widths == 3), 3), dtype=np.int64)
            row = token = record_count = shift_count = 0
            while row < len(widths):
                context = f"{source}:{numbers[row]}"
                if remaining:
                    count = min(remaining, len(widths) - row)
                    if np.any(widths[row : row + count] != 3):
                        raise ValueError(f"{context}: expected three shift integers")
                    shifts[shift_count : shift_count + count] = tokens[token : token + 3 * count].reshape(
                        -1, 3
                    )
                    shift_count += count
                    token += 3 * count
                    row += count
                    remaining -= count
                elif current is None:
                    if widths[row] != 5:
                        raise ValueError(f"{context}: expected R1 R2 R3 m n")
                    current = tokens[token : token + 5].copy()
                    row += 1
                    token += 5
                else:
                    if widths[row] != 1:
                        raise ValueError(f"{context}: expected one shift count integer")
                    remaining = int(tokens[token])
                    if remaining < 1:
                        raise ValueError(f"{context}: wsvec shift count must be positive")
                    records[record_count, :5] = current
                    records[record_count, 5] = remaining
                    record_count += 1
                    current = None
                    row += 1
                    token += 1
            entries.append(records[:record_count].copy())
            images.append(shifts[:shift_count].copy())
            line_offset += len(lines)
    if current is not None or remaining:
        raise ValueError(f"{source}: truncated wsvec entry")
    entry_array = np.concatenate(entries) if entries else np.empty((0, 6), np.int64)
    entries.clear()
    image_array = np.concatenate(images) if images else np.empty((0, 3), np.int64)
    images.clear()
    shifts = PackedWignerSeitzShifts(entry_array, image_array, source=source)
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


KPathSegment = tuple[str, tuple[float, float, float], str, tuple[float, float, float]]


@dataclass(frozen=True)
class WinData:
    """Settings of a Wannier90 ``.win`` input that post-processing consumes.

    ``lattice`` holds Cartesian row vectors in Å. ``cell_unit`` is the lower-case
    unit line of ``unit_cell_cart``, or ``None`` when the block has none, which
    Wannier90 reads as Å. ``mp_grid`` comes from its keyword or, when that is
    absent or malformed, from the distinct coordinates of the ``kpoints`` block
    (``mp_grid_inferred``). ``kpoints`` are fractional, and ``kpoint_path`` rows
    are ``(label, start, label, end)`` in fractional coordinates. Absent or
    unparsable scalars are ``None``. ``keywords`` maps lower-case keywords to their
    raw values; ``blocks`` maps lower-case block names to their comment-free lines.
    """

    lattice: np.ndarray | None
    cell_unit: str | None
    mp_grid: tuple[int, int, int] | None
    mp_grid_inferred: bool
    kpoints: np.ndarray
    kpoint_path: tuple[KPathSegment, ...]
    num_wann: int | None
    num_bands: int | None
    spinors: bool | None
    use_ws_distance: bool | None
    translate_home_cell: bool | None
    write_tb: bool | None
    write_hr: bool | None
    write_xyz: bool | None
    bands_plot: bool | None
    dis_win_min: float | None
    dis_win_max: float | None
    dis_froz_min: float | None
    dis_froz_max: float | None
    ws_search_size: int | None
    ws_distance_tol: float | None
    fermi_energy: float | None
    keywords: Mapping[str, str]
    blocks: Mapping[str, tuple[str, ...]]

    @property
    def n_kpoints(self) -> int:
        return len(self.kpoints)


def _fortran_float(token: str) -> float:
    return float(token.strip().lower().replace("d", "e"))


def _win_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    token = value.split()[0].lower() if value.strip() else ""
    if token in _WIN_TRUE:
        return True
    if token in _WIN_FALSE:
        return False
    return None


def _win_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        return _fortran_float(value.split()[0])
    except ValueError:
        return None


def _win_int(value: str | None) -> int | None:
    number = _win_float(value)
    return None if number is None or not np.isfinite(number) else int(round(number))


def infer_mp_grid(kpoints) -> tuple[int, int, int] | None:
    """Monkhorst-Pack divisions of a complete uniform fractional k-point list.

    Each axis counts the distinct coordinates modulo one, after rounding to six
    and then five decimals.
    """
    points = np.asarray(kpoints, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3 or not len(points):
        return None
    counts = []
    for axis in range(3):
        fractional = np.mod(np.round(points[:, axis], 6), 1.0)
        counts.append(int(len(np.unique(np.round(fractional, 5)))))
    return counts[0], counts[1], counts[2]


def read_win(path: str | Path) -> WinData:
    """Read the ``.win`` keywords and blocks consumed by Wannier post-processing.

    Keywords and block names are case-insensitive; ``key = value``, ``key : value``
    and ``key value`` are accepted, and ``!`` or ``#`` start comments. A later
    keyword or block replaces an earlier one, an ``end`` line closes whichever block
    is open, and an unterminated block is ignored. Fortran ``d`` exponents and
    logical spellings (``.true.``, ``T``, ``yes``, ``1`` and their negations) are
    recognized. Malformed rows of the ``kpoints`` and ``kpoint_path`` blocks are
    skipped; a malformed ``unit_cell_cart`` value is an error.
    """
    source = Path(path)
    keywords: dict[str, str] = {}
    blocks: dict[str, tuple[str, ...]] = {}
    current: str | None = None
    lines: list[str] = []
    for raw in source.read_text(encoding="utf-8", errors="replace").splitlines():
        line = _WIN_COMMENT.sub("", raw).strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("begin") and (len(low) == 5 or low[5] in " \t"):
            parts = low.split()
            current = parts[1] if len(parts) > 1 else ""
            lines = []
            continue
        if low.startswith("end") and (len(low) == 3 or low[3] in " \t"):
            if current is not None:
                blocks[current] = tuple(lines)
            current = None
            continue
        if current is not None:
            lines.append(line)
            continue
        match = _WIN_KEYWORD.match(line)
        if match:
            keywords[match.group(1).lower()] = match.group(2).strip()

    lattice = None
    cell_unit = None
    cell = blocks.get("unit_cell_cart")
    if cell:
        rows = list(cell)
        scale = 1.0
        first = rows[0].split()[0].lower()
        if first in _WIN_UNITS:
            cell_unit = first
            scale = BOHR_ANGSTROM if first == "bohr" else 1.0
            rows = rows[1:]
        try:
            vectors = [[_fortran_float(value) * scale for value in row.split()[:3]] for row in rows[:3]]
        except ValueError as error:
            raise ValueError(f"{source}: invalid unit_cell_cart value ({error})") from error
        if len(vectors) == 3 and all(len(vector) == 3 for vector in vectors):
            lattice = np.asarray(vectors, dtype=float)
            lattice.setflags(write=False)

    grid: list[int] | tuple[int, int, int] | None = None
    if "mp_grid" in keywords:
        try:
            grid = [int(value) for value in keywords["mp_grid"].replace(",", " ").split()[:3]]
        except ValueError:
            grid = None
    kpoints = []
    for row in blocks.get("kpoints", ()):
        fields = row.split()
        if len(fields) >= 3:
            try:
                kpoints.append([_fortran_float(value) for value in fields[:3]])
            except ValueError:
                continue
    inferred = False
    if (not grid or len(grid) != 3) and kpoints:
        grid = infer_mp_grid(kpoints)
        inferred = grid is not None
    mp_grid = (int(grid[0]), int(grid[1]), int(grid[2])) if grid and len(grid) == 3 else None
    kpoint_array = np.asarray(kpoints, dtype=float).reshape(-1, 3)
    kpoint_array.setflags(write=False)

    segments: list[KPathSegment] = []
    for row in blocks.get("kpoint_path", ()):
        fields = row.replace(",", " ").split()
        if len(fields) < 8:
            continue
        try:
            start = [_fortran_float(value) for value in fields[1:4]]
            end = [_fortran_float(value) for value in fields[5:8]]
        except ValueError:
            continue
        segments.append((fields[0], (start[0], start[1], start[2]), fields[4], (end[0], end[1], end[2])))

    return WinData(
        lattice=lattice,
        cell_unit=cell_unit,
        mp_grid=mp_grid,
        mp_grid_inferred=inferred and mp_grid is not None,
        kpoints=kpoint_array,
        kpoint_path=tuple(segments),
        num_wann=_win_int(keywords.get("num_wann")),
        num_bands=_win_int(keywords.get("num_bands")),
        spinors=_win_bool(keywords.get("spinors")),
        use_ws_distance=_win_bool(keywords.get("use_ws_distance")),
        translate_home_cell=_win_bool(keywords.get("translate_home_cell")),
        write_tb=_win_bool(keywords.get("write_tb")),
        write_hr=_win_bool(keywords.get("write_hr")),
        write_xyz=_win_bool(keywords.get("write_xyz")),
        bands_plot=_win_bool(keywords.get("bands_plot")),
        dis_win_min=_win_float(keywords.get("dis_win_min")),
        dis_win_max=_win_float(keywords.get("dis_win_max")),
        dis_froz_min=_win_float(keywords.get("dis_froz_min")),
        dis_froz_max=_win_float(keywords.get("dis_froz_max")),
        ws_search_size=_win_int(keywords.get("ws_search_size")),
        ws_distance_tol=_win_float(keywords.get("ws_distance_tol")),
        fermi_energy=_win_float(keywords.get("fermi_energy")),
        keywords=MappingProxyType(dict(keywords)),
        blocks=MappingProxyType(dict(blocks)),
    )


def read_eig(path: str | Path) -> np.ndarray:
    """Read Wannier90 ``seed.eig`` rows ``band kpoint energy`` as (NK,NB) eV.

    Indices are one-based; columns after the energy are ignored. Pairs absent from
    the file are NaN, and a repeated band/k-point pair is an error.
    """
    source = Path(path)
    bands: list[int] = []
    kpoints: list[int] = []
    energies: list[float] = []
    with source.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            context = f"{source}:{number}"
            if len(fields) < 3:
                raise ValueError(f"{context}: expected band index, k-point index and energy")
            band, kpoint = _integer(fields[0], context), _integer(fields[1], context)
            if band < 1 or kpoint < 1:
                raise ValueError(f"{context}: band and k-point indices are one-based")
            bands.append(band - 1)
            kpoints.append(kpoint - 1)
            energies.append(_number(fields[2], context))
    if not energies:
        raise ValueError(f"{source}: no eigenvalue records")
    band_index, kpoint_index = np.asarray(bands), np.asarray(kpoints)
    shape = (max(kpoints) + 1, max(bands) + 1)
    if len(np.unique(kpoint_index * shape[1] + band_index)) != len(energies):
        raise ValueError(f"{source}: repeated band/k-point record")
    result = np.full(shape, np.nan)
    result[kpoint_index, band_index] = energies
    result.setflags(write=False)
    return result


def read_band_kpt(path: str | Path) -> np.ndarray:
    """Read ``seed_band.kpt`` fractional k-points (NK,3), skipping its count line.

    Rows with fewer than three numeric fields are skipped; weights are ignored.
    """
    rows = [line.split() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if rows and len(rows[0]) == 1:
        rows = rows[1:]
    points = []
    for fields in rows:
        if len(fields) >= 3:
            try:
                points.append([_fortran_float(value) for value in fields[:3]])
            except ValueError:
                continue
    result = np.asarray(points, dtype=float).reshape(-1, 3)
    result.setflags(write=False)
    return result


def read_labelinfo(path: str | Path) -> tuple[list[int], list[str]]:
    """Read ``seed_band.labelinfo.dat`` rows ``LABEL index ...`` as zero-based ticks.

    Rows without an integer second field are skipped.
    """
    indices: list[int] = []
    labels: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            index = int(fields[1])
        except ValueError:
            continue
        indices.append(index - 1)
        labels.append(fields[0])
    return indices, labels


def read_band_dat(path: str | Path, num_bands: int) -> np.ndarray:
    """Read ``seed_band.dat`` band-major ``distance energy`` rows as (NK,NB) eV.

    Blank-line separation is not required: every row with a numeric second field
    contributes, in order, and the count must be a multiple of ``num_bands``.
    """
    values = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2:
            try:
                values.append(_fortran_float(fields[1]))
            except ValueError:
                continue
    data = np.asarray(values, dtype=float)
    if num_bands <= 0 or data.size % num_bands:
        raise ValueError(f"{path}: {data.size} band values are not a multiple of {num_bands} bands")
    result = np.ascontiguousarray(data.reshape(num_bands, -1).T)
    result.setflags(write=False)
    return result


def band_path_breaks(kpoints, tick_indices: Sequence[int]) -> list[int]:
    """Tick indices where a sampled path jumps instead of continuing.

    Wannier90 does not repeat the point joining two segments, so a discontinuity
    appears as a step more than three times longer than the step before it.
    """
    points = np.asarray(kpoints, dtype=float)
    breaks: list[int] = []
    for index in tick_indices:
        if index < 2 or index >= len(points) or index in breaks:
            continue
        step = float(np.linalg.norm(points[index] - points[index - 1]))
        reference = float(np.linalg.norm(points[index - 1] - points[index - 2]))
        if reference > 0 and step > 3.0 * reference:
            breaks.append(int(index))
    return breaks


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
    if mode == "ws_distance" and wsvec is not None:
        if not wsvec.use_ws_distance or wsvec.ndegen_applied:
            raise ValueError("ws_distance interpolation requires unapplied enabled wsvec shifts")
        if isinstance(wsvec.shifts, ComputedWignerSeitzShifts):
            translations, hopping = wsvec.shifts.normalized_matrices(data, data.hopping)
            _, position = wsvec.shifts.normalized_matrices(data, data.position)
            return translations, hopping, position
        shifts = PackedWignerSeitzShifts.from_mapping(wsvec.shifts)
        translations, operators = shifts.normalized_operator_arrays(data, (data.hopping, data.position))
        return translations, operators[0], operators[1]
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
    if isinstance(wsvec.shifts, (ComputedWignerSeitzShifts, PackedWignerSeitzShifts)):
        return wsvec.shifts.normalized_matrices(data, matrices)
    return PackedWignerSeitzShifts.from_mapping(wsvec.shifts).normalized_matrices(data, matrices)


@dataclass(frozen=True)
class TbLayout:
    """The header of a wannier90 ``_tb.dat`` and the exact size its body must have.

    ``h_line_bytes`` is the measured length of one Hamiltonian entry line and
    ``r_line_bytes`` the position entry line length derived from the file size;
    ``lines_ok`` holds when they have wannier90 3.1's widths (45 or 46 bytes and 64
    more), and the file is complete when ``actual_bytes == expected_bytes``.
    """

    lattice: np.ndarray
    num_wann: int
    nrpts: int
    ndegen: np.ndarray
    header_bytes: int
    h_line_bytes: int
    r_line_bytes: int
    expected_bytes: int
    actual_bytes: int
    lines_ok: bool


@dataclass(frozen=True)
class TbRecords:
    """``_tb.dat`` blocks as written, not divided by ``ndegen``, indexed [j-1, i-1] = <0j|X|Ri>.

    ``R`` (nR, 3), ``H`` (nR, N, N) eV and ``r`` (nR, N, N, 2) Å with the Cartesian x
    and y components; ``rz2_offdiag`` is sum |r_z|^2 / ndegen over every entry but the
    (0, n, n) diagonal (r_z itself is not kept).
    """

    R: np.ndarray
    H: np.ndarray
    r: np.ndarray
    rz2_offdiag: float


def read_tb_layout(path: str | Path) -> TbLayout:
    """The header and size check of a wannier90 3.1 ``_tb.dat`` (``hamiltonian_write_tb``).

    Layout: a date line; three lattice rows (Å, list-directed, so read by tokens);
    num_wann; nrpts; ndegen at 15 per line; then per R a blank line, the R line and N^2
    lines ``j i Re Im`` of H, and then the same for r with ``j i xRe xIm yRe yIm zRe zIm``.
    A format's trailing ``1x`` writes nothing at the end of a record, so the H line
    length is measured and the r line length derived from the size. A truncated header
    raises ``ValueError``.
    """
    source = Path(path)
    with source.open("rb") as handle:
        handle.readline()
        reals: list[float] = []
        while len(reals) < 9:
            line = handle.readline()
            if not line:
                raise ValueError("truncated _tb.dat header")
            reals += [float(x) for x in line.split()]
        num_wann = int(handle.readline().split()[0])
        nrpts = int(handle.readline().split()[0])
        ndegen: list[int] = []
        while len(ndegen) < nrpts:
            line = handle.readline()
            if not line:
                raise ValueError("truncated _tb.dat degeneracies")
            ndegen += [int(x) for x in line.split()]
        header = handle.tell()
        handle.readline()  # blank
        handle.readline()  # first R
        h_line = len(handle.readline())
    actual = source.stat().st_size
    nn = num_wann * num_wann
    body = actual - header - nrpts * 34
    r_line = body // (nrpts * nn) - h_line if body > 0 and nrpts and nn else 0
    expected = header + nrpts * (34 + (h_line + r_line) * nn)
    return TbLayout(
        lattice=np.asarray(reals[:9], float).reshape(3, 3),
        num_wann=num_wann,
        nrpts=nrpts,
        ndegen=np.asarray(ndegen[:nrpts], int),
        header_bytes=header,
        h_line_bytes=h_line,
        r_line_bytes=r_line,
        expected_bytes=expected,
        actual_bytes=actual,
        lines_ok=h_line in (45, 46) and r_line == h_line + 64,
    )


def read_tb_records(path: str | Path, layout: TbLayout) -> TbRecords:
    """Stream the blocks of a ``_tb.dat`` whose :func:`read_tb_layout` checked out.

    Each block is read as one fixed-size record, so a 1 GB file costs one pass. An r
    block whose translation differs from its H block raises ``ValueError``.
    """
    n, nr = layout.num_wann, layout.nrpts
    nn = n * n
    lh, lr = layout.h_line_bytes, layout.r_line_bytes
    R = np.zeros((nr, 3), int)
    H = np.zeros((nr, n, n), complex)
    r = np.zeros((nr, n, n, 2), complex)
    rz2 = 0.0
    with Path(path).open("rb") as handle:
        handle.seek(layout.header_bytes)
        for k in range(nr):
            handle.readline()
            R[k] = [int(x) for x in handle.readline().split()]
            v = np.array(handle.read(lh * nn).split(), dtype=float).reshape(n, n, 4)
            H[k] = (v[:, :, 2] + 1j * v[:, :, 3]).T
        for k in range(nr):
            handle.readline()
            Rk = [int(x) for x in handle.readline().split()]
            if Rk != R[k].tolist():
                raise ValueError(f"the r block's R {Rk} differs from the H block's {R[k].tolist()}")
            v = np.array(handle.read(lr * nn).split(), dtype=float).reshape(n, n, 8)
            r[k, :, :, 0] = (v[:, :, 2] + 1j * v[:, :, 3]).T
            r[k, :, :, 1] = (v[:, :, 4] + 1j * v[:, :, 5]).T
            rz = (v[:, :, 6] + 1j * v[:, :, 7]).T
            if not R[k].any():
                rz = rz - np.diag(np.diag(rz))
            rz2 += float(np.sum(np.abs(rz) ** 2)) / float(layout.ndegen[k])
    return TbRecords(R=R, H=H, r=r, rz2_offdiag=rz2)
