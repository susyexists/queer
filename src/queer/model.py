"""Compatibility facade for the validated Wannier model and plotting API."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from .execution import cpu_budget
from .geometry.mesh import cartesian_mesh
from .geometry.paths import KPath as KPath
from .io.structure import read_fermi_energy, read_qe_reciprocal
from .io.wannier import read_hr as _read_hr
from .models.wannier import WannierModel
from .physics.broadening import lorentzian
from .physics.electronic import lindhard
from .physics.statistics import thermal_occupation as fd


class model(WannierModel):
    """Legacy callable preserved for notebooks; prefer WannierModel for new code."""

    def __init__(
        self,
        hr="wannier90_hr.dat",
        path="./",
        nscf=False,
        poscar=False,
        ef=0,
        read_ef=False,
        shift=0,
        num_core=False,
        interpolation="standard",
        wsvec=None,
    ):
        root = Path(path).expanduser().resolve()
        if read_ef and not nscf:
            raise ValueError("read_ef requires an nscf output file")
        reference = read_fermi_energy(root / nscf) if read_ef else ef
        loaded = WannierModel.from_directory(
            root, hr=hr, poscar=poscar or None, ef=reference + shift, interpolation=interpolation, wsvec=wsvec
        )
        self.__dict__.update(loaded.__dict__)
        self.shift = shift
        self.nscf = Path(nscf) if nscf else None
        if self.nscf is not None and not poscar:
            self.g_vec = read_qe_reciprocal(root / self.nscf)
        self.num_cores = 1 if num_core is False or num_core is None else num_core
        self.points = len(self.raw_data.translations)
        self.sym = self.raw_data.degeneracies
        self.h = self.raw_data.hopping.reshape(self.points, -1)
        self.hopping = self.h.reshape(-1)

    @property
    def num_cores(self):
        return self.execution.workers

    @num_cores.setter
    def num_cores(self, value):
        workers = cpu_budget() if value == -1 else int(value)
        self.execution = replace(self.execution, workers=workers)

    @property
    def x(self):
        return np.broadcast_to(self.raw_data.translations.T[:, :, None], (3, self.points, self.nbnd**2))

    @property
    def data(self):
        return self.hopping, self.x.reshape(3, -1), self.sym

    def suscep(self, point, mesh, mesh_energy, mesh_fermi, bands, T=1, delta=1e-7, fermi_shift=0):
        indices = self._band_indices(bands)
        if delta <= 0 or not np.isfinite(delta):
            raise ValueError("delta must be positive and finite")
        cached = np.asarray(mesh_energy, dtype=float)
        occupations = np.asarray(mesh_fermi, dtype=float)
        if cached.ndim == 1 and len(indices) == 1:
            cached = cached[None, :]
            occupations = occupations[None, :]
        elif cached.ndim == 2 and cached.shape[0] == self.nbnd:
            cached = cached[indices]
            occupations = occupations[indices]
        if cached.shape != occupations.shape or cached.shape != (len(indices), len(mesh)):
            raise ValueError("cached energies/occupations must align with selected bands and mesh")
        shifted = self.energies(np.asarray(mesh) + point, bands=indices) - fermi_shift
        cached = cached - fermi_shift
        if fermi_shift:
            occupations = fd(cached, T)
        shifted_occupations = fd(shifted, T)
        response = lindhard(cached, shifted, occupations, shifted_occupations, delta)
        joint_density = 0.0
        for i in range(len(indices)):
            for j in range(len(indices)):
                joint_density += np.mean(delta_function(cached[i]) * delta_function(shifted[j]))
        return [np.real(response), joint_density]

    def suscep_path(self, q_path, k_mesh, band_index, T=1):
        energies = self.energies(k_mesh)
        occupations = fd(energies, T)
        return np.asarray([self.suscep(q, k_mesh, energies, occupations, band_index, T=T) for q in q_path]).T

    def plot_electron_path(self, *args, **kwargs):
        from .plotting.bands import plot_electron_path

        return plot_electron_path(self, *args, **kwargs)

    def plot_band_path(self, *args, **kwargs):
        from .plotting.bands import plot_band_path

        return plot_band_path(self, *args, **kwargs)

    def plot_brillouin_zone(self, *args, **kwargs):
        from .plotting.bands import plot_brillouin_zone

        return plot_brillouin_zone(self, *args, **kwargs)


def read_hr(path):
    """Legacy tuple adapter for the validated, index-aware HR reader."""
    data = _read_hr(path)
    return (
        data.hopping.reshape(-1),
        np.broadcast_to(data.translations.T[:, :, None], (3, len(data.translations), data.nbnd**2)).reshape(
            3, -1
        ),
        data.degeneracies,
    )


def read_efermi(path):
    return read_fermi_energy(path)


def Symmetries(fstring):
    with Path(fstring).open(encoding="utf-8") as stream:
        return np.asarray([float(line.split()[-1]) for line in stream if line.strip()])


def delta_function(x, epsilon=0.00001):
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    values = np.asarray(x)
    return lorentzian(values, epsilon)


def density_of_states(energy, band_index=False, dE=1e-2):
    """Legacy per-sample window counts (not a normalized energy-axis DOS)."""
    if not np.isfinite(dE) or dE <= 0:
        raise ValueError("dE must be positive and finite")
    values = np.asarray(energy)
    if band_index is not False and band_index is not None:
        values = values[band_index]
    if values.ndim != 1:
        raise ValueError("density_of_states requires one-dimensional energy samples")
    if not np.isfinite(values).all():
        raise ValueError("energy samples must be finite")
    return np.asarray([np.mean(abs(value - values) < dE) for value in values])


def rotate(vector, angle):
    matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return matrix @ np.asarray(vector).T


def triangle_mesh(N):
    grid = cartesian_mesh([N, N, 1], center=False)
    x, y = grid[:, :2].T
    return grid[(y <= np.sqrt(3) * x) & (y <= -np.sqrt(3) * x + np.sqrt(3)), :2].T


def find_cross(band, parameter):
    values = np.asarray(band)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("bands must have shape (Nb,Nk) with at least two k-points")
    flagged = np.abs(np.diff(np.gradient(values, axis=1), axis=1)) > parameter
    crossings = [
        (i, j, point + 1)
        for point in range(flagged.shape[1])
        for i in range(len(values))
        for j in range(i + 1, len(values))
        if flagged[i, point] and flagged[j, point]
    ]
    return np.asarray(crossings, dtype=int).reshape(-1, 3)


def plot_fs(band, fs_thickness=0.01, title=None, mesh=None):
    import matplotlib.pyplot as plt

    values = np.asarray(band)
    if values.ndim != 1:
        raise ValueError("plot_fs requires one band")
    if mesh is None:
        side = int(np.sqrt(len(values)))
        if side**2 != len(values):
            raise ValueError("provide mesh for a non-square band grid")
        mesh = cartesian_mesh([side, side, 1])
    points = np.asarray(mesh)
    if points.shape != (len(values), 3):
        raise ValueError("mesh and band must align")
    selected = points[np.abs(values) <= fs_thickness]
    _, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(selected[:, 0], selected[:, 1])
    if title is not None:
        axis.set_title(title)
    return axis


def ram_check():
    """Print the current process peak resident memory without an optional dependency."""
    import resource
    import sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1e9 if sys.platform == "darwin" else 1e6
    print(f"Process peak resident memory (GB): {peak / divisor:.3f}")
