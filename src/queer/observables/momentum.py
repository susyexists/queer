"""Streaming momentum microscopy surfaces using compact NumPy arrays."""

import logging

import numpy as np

from ..geometry.lattice import cartesian_to_fractional as angstrom2reciprocal
from ._common import model_energies, points_array, positive
from .arpes import arpes_mesh

SURFACE_COLUMNS = ("kx", "ky", "kz", "e", "I")
logger = logging.getLogger(__name__)


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    if seconds < 3600:
        return f"{seconds / 60:.2f} minutes"
    return f"{seconds / 3600:.2f} hours"


def select_surface(mesh, bands, binding_energy, sigma=0.1):
    """Choose maximal intensity per xy, preserving band-major first-row ties.

    Returns (N,5), ordered by kx,ky as the historical DataFrame groupby result.
    Coordinates are never repeated for each band.
    """
    sigma = positive(sigma, "sigma")
    mesh = points_array(mesh)
    bands = np.asarray(bands, dtype=float)
    if bands.ndim != 2 or bands.shape[1] != len(mesh):
        raise ValueError("bands must have shape (Nb,Nk)")
    if not np.isfinite(mesh).all() or not np.isfinite(bands).all() or not np.isfinite(binding_energy):
        raise ValueError("surface coordinates, band energies and binding energy must be finite")
    if not len(mesh) or not len(bands):
        return np.empty((0, 5))
    intensity = np.exp(-((bands + binding_energy) ** 2) / (2 * sigma**2))
    best_band = np.argmax(intensity, axis=0)
    columns = np.arange(len(mesh))
    best_intensity = intensity[best_band, columns]
    best_energy = bands[best_band, columns]
    original_index = best_band * len(mesh) + columns
    order = np.lexsort((original_index, -best_intensity, mesh[:, 1], mesh[:, 0]))
    sorted_xy = mesh[order, :2]
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.any(sorted_xy[1:] != sorted_xy[:-1], axis=1)
    chosen = order[first]
    return np.column_stack((mesh[chosen], best_energy[chosen], best_intensity[chosen]))


def iter_binding_energy_surfaces(
    model,
    g_vec,
    photon_energy,
    fermi_energy,
    binding_range,
    binding_step,
    v0,
    n_points,
    factor=None,
    align=None,
    energy_shift=0.0,
    sigma=0.1,
    execution=None,
):
    """Yield (binding_energy, surface) one slice at a time; arrays use SURFACE_COLUMNS."""
    positive(sigma, "sigma")
    if not np.isfinite(energy_shift):
        raise ValueError("energy_shift must be finite")
    for binding_energy, mesh, bands in _iter_binding_geometry(
        model,
        g_vec,
        photon_energy,
        fermi_energy,
        binding_range,
        binding_step,
        v0,
        n_points,
        factor,
        align,
        execution,
    ):
        yield binding_energy, select_surface(mesh, bands + energy_shift, binding_energy, sigma)


def _iter_binding_geometry(
    model,
    g_vec,
    photon_energy,
    fermi_energy,
    binding_range,
    binding_step,
    v0,
    n_points,
    factor,
    align,
    execution,
):
    """Keep only one slice's valid coordinates and unshifted band energies."""
    binding_step = positive(binding_step, "binding_step")
    if len(binding_range) != 2 or not np.isfinite(binding_range).all() or binding_range[1] < binding_range[0]:
        raise ValueError("binding_range must have ordered finite endpoints")
    for binding_energy in np.arange(*binding_range, binding_step):
        mesh, valid = arpes_mesh(
            photon_energy,
            fermi_energy,
            binding_energy,
            v0,
            n_points,
            factor,
            align,
            return_mask=True,
        )
        mesh = mesh[valid]
        if len(mesh):
            bands = model_energies(model, angstrom2reciprocal(mesh, g_vec), execution)
        else:
            bands = np.empty((0, 0))
        yield float(binding_energy), mesh, bands


def binding_energy_surfaces(*args, **kwargs):
    return dict(iter_binding_energy_surfaces(*args, **kwargs))


def iter_surface_sweep(
    model,
    g_vec,
    photon_energy,
    fermi_energy,
    binding_range,
    binding_step,
    v0_list,
    n_points,
    factor=None,
    align=None,
    energy_shifts=(0.0,),
    sigma=0.1,
    execution=None,
):
    """Yield geometry-first slices, reusing raw bands across energy shifts.

    Only the current (V0,binding-energy) geometry is retained. Dictionary
    collectors preserve the existing {(V0,shift): {binding_energy: surface}} shape.
    """
    positive(sigma, "sigma")
    shifts = tuple(energy_shifts)
    if any(not np.isfinite(shift) for shift in shifts):
        raise ValueError("energy_shifts must be finite")
    if not shifts:
        return
    for v0 in v0_list:
        for energy, mesh, bands in _iter_binding_geometry(
            model,
            g_vec,
            photon_energy,
            fermi_energy,
            binding_range,
            binding_step,
            v0,
            n_points,
            factor,
            align,
            execution,
        ):
            for shift in shifts:
                yield (float(v0), float(shift), energy), select_surface(mesh, bands + shift, energy, sigma)


def sweep_binding_energy_surfaces(
    model,
    g_vec,
    photon_energy,
    fermi_energy,
    binding_range,
    binding_step,
    v0_list,
    n_points,
    factor=None,
    align=None,
    energy_shifts=(0.0,),
    sigma=0.1,
    verbose=True,
    execution=None,
):
    results = {}
    for (v0, shift, energy), surface in iter_surface_sweep(
        model,
        g_vec,
        photon_energy,
        fermi_energy,
        binding_range,
        binding_step,
        v0_list,
        n_points,
        factor,
        align,
        energy_shifts,
        sigma,
        execution,
    ):
        results.setdefault((v0, shift), {})[energy] = surface
        if verbose:
            logger.info("Computed V0=%s shift=%s binding_energy=%s", v0, shift, energy)
    return results


def fermi_surface_points(mesh, model, g_vec, delta=0.05, execution=None):
    delta = positive(delta, "delta")
    mesh = points_array(mesh)
    valid_mesh = mesh[np.isfinite(mesh).all(axis=1)]
    if not len(valid_mesh):
        return np.empty(0, dtype=bool), np.empty(0), np.empty(0)
    bands = model_energies(model, angstrom2reciprocal(valid_mesh, g_vec), execution)
    mask = np.any(np.abs(bands) <= delta, axis=0)
    return mask, valid_mesh[mask, 0], valid_mesh[mask, 1]
