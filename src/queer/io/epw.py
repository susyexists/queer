"""Deterministic readers for EPW text metadata and per-mode array directories.

The supported directory has gkk.out and omega/, gkk_re/, gkk_im/, enk/,
enkq/ channels. Mode files end in a numeric branch index. Frequencies and
couplings are in meV, electronic energies and Fermi energy are in eV.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EPWMetadata:
    nk: int
    nq: int
    nph: int
    ef: float


def read_epw_metadata(path):
    """Scan headers without fixed line limits or positional token assumptions."""
    values = {}
    max_mode = 0
    with Path(path).open() as stream:
        for line in stream:
            lower = line.lower()
            if re.match(r"\s*ik\b", lower) and max_mode and all(key in values for key in ("nk", "nq", "ef")):
                break
            numbers = re.findall(r"[-+]?\d+(?:\.\d*)?(?:[eEdD][-+]?\d+)?", line)
            if not numbers:
                continue
            if "size" in lower and re.search(r"\bk\b", lower):
                values["nk"] = int(float(numbers[-1]))
            elif "size" in lower and re.search(r"\bq\b", lower):
                values["nq"] = int(float(numbers[-1]))
            elif "fermi" in lower:
                values["ef"] = float(numbers[-1].replace("D", "E").replace("d", "e"))
            elif "phonon" in lower and ("mode" in lower or "branch" in lower):
                values["nph"] = int(float(numbers[-1]))
            elif len(numbers) >= 6 and re.match(r"\s*\d+\s+\d+\s+\d+\s", line):
                max_mode = max(max_mode, int(numbers[2]))
            if all(key in values for key in ("nk", "nq", "nph", "ef")):
                break
    if "nph" not in values and max_mode:
        values["nph"] = max_mode
    return values


def _branch_files(directory, channel):
    folder = directory / channel
    if not folder.is_dir():
        raise ValueError(f"missing EPW channel: {folder}")
    branches = {}
    for path in folder.iterdir():
        if not path.is_file() or path.name.startswith("."):
            continue
        match = re.search(r"(\d+)(?:\.[^.]+)?$", path.name)
        if match is None:
            raise ValueError(f"EPW mode filename must end in a branch index: {path}")
        index = int(match.group(1))
        if index in branches:
            raise ValueError(f"duplicate EPW branch {index} in {folder}")
        branches[index] = path
    if not branches:
        raise ValueError(f"empty EPW channel: {folder}")
    return branches


def _load(path):
    try:
        values = np.loadtxt(path, dtype=float, ndmin=1).reshape(-1)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read EPW numeric file {path}: {exc}") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite EPW values in {path}")
    return values


def read_epw_directory(directory, nk=None, nq=None, nph=None, ef=None):
    """Read validated arrays and metadata; explicit dimensions override headers."""
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise ValueError(f"EPW directory does not exist: {directory}")
    header = directory / "gkk.out"
    values = read_epw_metadata(header) if header.is_file() else {}
    mode_files = {key: _branch_files(directory, key) for key in ("omega", "gkk_re", "gkk_im")}
    branches = sorted(mode_files["omega"])
    for key, files in mode_files.items():
        if sorted(files) != branches:
            raise ValueError(f"EPW branch indices differ between omega and {key}")
    overrides = {"nk": nk, "nq": nq, "nph": nph, "ef": ef}
    values.update({key: value for key, value in overrides.items() if value is not None})
    values.setdefault("nph", len(branches))
    for key in ("nk", "nq", "nph"):
        value = values.get(key)
        if value is None or isinstance(value, bool) or int(value) != value or int(value) <= 0:
            raise ValueError(f"EPW {key} must be a positive integer supplied explicitly or in gkk.out")
        values[key] = int(value)
    if len(branches) != values["nph"]:
        raise ValueError("EPW mode file count differs from nph")
    if "ef" not in values or not np.isfinite(values["ef"]):
        raise ValueError("EPW ef must be finite and supplied explicitly or in gkk.out")
    meta = EPWMetadata(**values)
    shape = (meta.nq, meta.nk)
    arrays = {}
    for channel in ("gkk_re", "gkk_im"):
        data = []
        for branch in branches:
            vector = _load(mode_files[channel][branch])
            if vector.size != meta.nq * meta.nk:
                raise ValueError(f"EPW {channel} branch {branch} size differs from nq*nk")
            data.append(vector.reshape(shape))
        arrays[channel] = np.asarray(data)
    frequencies = []
    for branch in branches:
        vector = _load(mode_files["omega"][branch])
        if vector.size == meta.nq * meta.nk:
            repeated = vector.reshape(shape)
            if not np.allclose(repeated, repeated[:, :1]):
                raise ValueError("EPW phonon frequency varies with k within one q point")
            vector = repeated[:, 0]
        if vector.size != meta.nq:
            raise ValueError(f"EPW omega branch {branch} size differs from nq")
        frequencies.append(vector)
    arrays["ph"] = np.asarray(frequencies)
    for channel in ("enk", "enkq"):
        files = _branch_files(directory, channel)
        if len(files) != 1:
            raise ValueError(f"EPW {channel} must contain one electronic-energy file")
        vector = _load(next(iter(files.values())))
        if vector.size != meta.nq * meta.nk:
            raise ValueError(f"EPW {channel} size differs from nq*nk")
        arrays[channel] = vector.reshape(shape)
    return meta, arrays
