"""Explicit compatibility helpers; numerical domains live in dedicated modules."""

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

__all__ = [
    "dotdict",
    "read_data",
    "read_parallel",
    "hexagon",
    "suscep_epw",
    "read_poscar",
    "crystal2reciprocal",
    "write_data",
    "path_create",
    "read_gvec",
    "fermi_bands",
    "ph_cross",
    "g_cross",
    "untangle",
    "phselfen",
]


class dotdict(dict):
    """Historical dictionary attribute access."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def read_data(path):
    """Read a whitespace DataFrame, propagating failures with their filename."""
    import pandas as pd

    source = Path(path)
    return [source.parent.name, pd.read_csv(source, sep=r"\s+", header=None)]


def _natural_key(path):
    return tuple(int(piece) if piece.isdigit() else piece for piece in re.split(r"(\d+)", str(path)))


def read_parallel(path, workers=1):
    """Return channel DataFrames in deterministic natural filename order."""
    from .execution import ExecutionOptions

    root = Path(path)
    count = ExecutionOptions(workers=workers).resolved().workers
    files = [
        file
        for folder in sorted(root.iterdir(), key=_natural_key)
        if folder.is_dir()
        for file in sorted(folder.iterdir(), key=_natural_key)
        if file.is_file() and not file.name.startswith(".")
    ]
    if count == 1:
        frames = map(read_data, files)
        return _collect_frames(frames)
    with ThreadPoolExecutor(max_workers=count) as pool:
        return _collect_frames(pool.map(read_data, files))


def _collect_frames(frames):
    result = {}
    for channel, frame in frames:
        result.setdefault(channel, []).append(frame)
    return result


def hexagon():
    return np.array(
        [
            [[-1 / np.sqrt(3), 1 / np.sqrt(3)], [1, 1]],
            [[1 / np.sqrt(3), 2 / np.sqrt(3)], [1, 0]],
            [[2 / np.sqrt(3), 1 / np.sqrt(3)], [0, -1]],
            [[1 / np.sqrt(3), -1 / np.sqrt(3)], [-1, -1]],
            [[-1 / np.sqrt(3), -2 / np.sqrt(3)], [-1, 0]],
            [[-2 / np.sqrt(3), -1 / np.sqrt(3)], [0, 1]],
        ]
    )


def suscep_epw(self, point, mesh, mesh_energy, mesh_fermi, epw1, epw2=None):
    if epw2 is None or isinstance(epw2, list) and not epw2:
        epw2 = epw1
    shifted_energy = self.calculate_energy(np.asarray(point) + mesh)[6]
    shifted_fermi = self.fermi(shifted_energy)
    return np.average(
        -np.asarray(epw1)
        * np.conj(epw2)
        * (mesh_fermi - shifted_fermi)
        / (mesh_energy - shifted_energy)
        * 1e-3
    )


def read_poscar(filename):
    from .io.structure import read_poscar as reader

    return reader(filename)[0]


def crystal2reciprocal(a):
    from .geometry.lattice import reciprocal_lattice

    return reciprocal_lattice(a)


def write_data(path, data, data_label):
    np.savetxt(path, np.asarray(data), header=" ".join(data_label), comments="#")


def path_create(n_points, corners):
    from .geometry.paths import interpolate_path

    return interpolate_path(n_points, corners)


def read_gvec(path, dim=3):
    from .io.structure import read_qe_reciprocal

    return read_qe_reciprocal(path)


def fermi_bands(bands, window=0.5):
    if not np.isfinite(window) or window < 0:
        raise ValueError("window must be nonnegative and finite")
    bands = np.asarray(bands)
    if bands.ndim != 2:
        raise ValueError("bands must have shape (Nb,Nk)")
    return set(np.flatnonzero(np.any(np.abs(bands) <= window, axis=1)).tolist())


def _crossing_input(values, tolerance, offset):
    array = np.asarray(values)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("crossing values must be a finite (Nbranch,Nq) array")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be nonnegative and finite")
    if isinstance(offset, bool) or not isinstance(offset, (int, np.integer)) or offset < 0:
        raise ValueError("offset must be a nonnegative integer")
    return array


def ph_cross(ph, tolerance, offset=50):
    ph = _crossing_input(ph, tolerance, offset)
    crossings, previous = [], {}
    for q in range(offset, ph.shape[1] - offset):
        for i in range(ph.shape[0]):
            for j in range(i + 1, ph.shape[0]):
                if abs(ph[i, q] - ph[j, q]) < tolerance:
                    if previous.get((i, j)) != q - 1:
                        crossings.append([i, j, q])
                    previous[i, j] = q
    return crossings


def g_cross(g, tolerance, offset=50):
    g = _crossing_input(g, tolerance, offset)
    crossings = []
    for q in range(max(1, offset), g.shape[1] - offset):
        for i in range(g.shape[0]):
            for j in range(i + 1, g.shape[0]):
                if abs(g[i, q] - g[i, q - 1]) > tolerance and abs(g[j, q] - g[j, q - 1]) > tolerance:
                    crossings.append([i, j, q])
    return crossings


def untangle(band, cross):
    fixed = np.asarray(band).copy()
    if fixed.ndim < 2:
        raise ValueError("band must have branch and q axes")
    for i, j, point in cross:
        if not all(isinstance(v, (int, np.integer)) for v in (i, j, point)) or not (
            0 <= i < len(fixed) and 0 <= j < len(fixed) and 0 <= point <= fixed.shape[1]
        ):
            raise ValueError("crossing indices outside the array")
        fixed[[i, j], :point] = fixed[[j, i], :point]
    return fixed


def phselfen(self, λ=None, λ_real=None, γ=None, γ_real=None, ω=None):
    return tuple(
        np.loadtxt(Path(self.path) / name).reshape(-1, 9).T
        for name in ("lambda.dat", "lambda_re.dat", "gamma.dat", "gamma_re.dat", "omega.dat")
    )
