"""Validated readers for Wannier90 HR and Wigner-Seitz translation files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WannierData:
    translations: np.ndarray
    hopping: np.ndarray
    degeneracies: np.ndarray

    @property
    def nbnd(self) -> int:
        return self.hopping.shape[1]


@dataclass(frozen=True)
class WignerSeitzData:
    shifts: dict[tuple[int, int, int, int, int], np.ndarray]
    use_ws_distance: bool
    ndegen_applied: bool


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
        degeneracies = []
        while len(degeneracies) < nrpts:
            line = stream.readline()
            if not line:
                raise ValueError(f"{source}: truncated HR degeneracy block")
            degeneracies.extend(_integer(value, str(source)) for value in line.split())
        if len(degeneracies) != nrpts or min(degeneracies) < 1:
            raise ValueError(f"{source}: expected {nrpts} positive degeneracies")

        translations = []
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
            key = tuple(_integer(value, str(source)) for value in fields)
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


def normalized_hopping(
    data: WannierData, mode: str, wsvec: WignerSeitzData | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Apply HR/WS weights once and produce a plain positive-phase Fourier sum."""
    if mode not in {"standard", "ws_distance", "ndegen_applied"}:
        raise ValueError(f"Unknown interpolation mode {mode!r}")
    if mode == "ndegen_applied":
        if not np.all(data.degeneracies == 1):
            raise ValueError("ndegen_applied HR exports must contain unit degeneracies")
        return data.translations, data.hopping
    if mode == "standard":
        return data.translations, data.hopping / data.degeneracies[:, None, None]
    if wsvec is None or not wsvec.use_ws_distance:
        raise ValueError("ws_distance interpolation requires a matching enabled wsvec file")
    if wsvec.ndegen_applied:
        raise ValueError("wsvec shifts have already been applied; select ndegen_applied")
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
                value = data.hopping[index, m, n] / (data.degeneracies[index] * len(shifts))
                for shift in shifts:
                    key = tuple(map(int, translation + shift))
                    if key not in expanded:
                        expanded[key] = np.zeros((data.nbnd, data.nbnd), dtype=np.complex128)
                    expanded[key][m, n] += value
    keys = sorted(expanded)
    return np.asarray(keys, dtype=np.int64), np.asarray([expanded[key] for key in keys])
