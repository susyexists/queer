"""ARPES geometry, with NaN marking points outside the real hemisphere."""

import numpy as np

from ..geometry.lattice import cartesian_to_fractional as angstrom2reciprocal
from ..geometry.lattice import fractional_to_cartesian as reciprocal2angstrom
from ..geometry.mesh import cartesian_mesh as mesh_cartesian
from ..geometry.rotations import rotation_matrix as get_rotation_matrix
from ..physics.constants import ARPES_FACTOR as ARPES_FACTOR
from ._common import points_array, positive


def arpes_equation(Ek, V0, k_xy):
    coordinates = np.asarray(k_xy, dtype=float)
    if np.isinf(coordinates).any() or not np.isfinite(Ek).all() or not np.isfinite(V0).all():
        raise ValueError("ARPES energies must be finite and coordinates cannot contain infinities")
    kx, ky = coordinates
    argument = ARPES_FACTOR * (np.asarray(Ek) + np.asarray(V0)) - kx**2 - ky**2
    result = np.full(np.shape(argument), np.nan, dtype=float)
    np.sqrt(argument, out=result, where=argument >= 0)
    return result.item() if result.ndim == 0 else result


def arpes_mesh(
    photon_energy, fermi_energy, binding_energy, V0, N, factor=None, align=None, *, return_mask=False
):
    """Return an (N*N,3) mesh, optionally paired with its aligned valid-row mask."""
    if isinstance(N, bool) or int(N) != N or N <= 0:
        raise ValueError("N must be a positive integer")
    kinetic = float(photon_energy) - float(binding_energy) - float(fermi_energy)
    if not all(np.isfinite(v) for v in (kinetic, V0)):
        raise ValueError("ARPES energies must be finite")
    if factor is None:
        factor = 2 * np.sqrt(ARPES_FACTOR * (kinetic + V0)) if kinetic + V0 > 0 else 1.0
    elif not np.isfinite(factor) or factor <= 0:
        raise ValueError("factor must be finite and positive")
    mesh = mesh_cartesian([int(N), int(N), 1], factor=factor, center=True)
    mesh[:, 2] = arpes_equation(kinetic, V0, mesh[:, :2].T)
    if align is not None:
        mesh = aligned_arpes_mesh(align, mesh.T).T
    return (mesh, np.isfinite(mesh).all(axis=1)) if return_mask else mesh


def aligned_arpes_mesh(align, k_points):
    initial, final = align
    return get_rotation_matrix(initial, final) @ np.asarray(k_points)


def binding_k(photon_energy, fermi_energy, binding_range, binding_step, V0, N, factor=None, align=None):
    binding_step = positive(binding_step, "binding_step")
    if len(binding_range) != 2 or not np.isfinite(binding_range).all() or binding_range[1] < binding_range[0]:
        raise ValueError("binding_range must contain ordered finite endpoints")
    result = {}
    for binding_energy in np.arange(binding_range[0], binding_range[1], binding_step):
        mesh = arpes_mesh(photon_energy, fermi_energy, binding_energy, V0, N, factor, align)
        result[binding_energy] = tuple(mesh[np.isfinite(mesh).all(axis=1)].T)
    return result


def arpes_path(
    path, g_vec, photon_energy=None, binding_energy=0, fermi_energy=0, V0=10, Ek=None, *, return_mask=False
):
    """Curve an original-row-aligned path; optionally return its valid-point mask."""
    if Ek is None:
        Ek = (21 if photon_energy is None else photon_energy) - binding_energy - fermi_energy
    points = points_array(path)
    valid = np.isfinite(points).all(axis=1)
    flat = reciprocal2angstrom(points[valid], reciprocal=g_vec)
    curved = flat.copy()
    curved[:, 2] += arpes_equation(Ek, V0, flat[:, :2].T)
    domain = np.isfinite(curved).all(axis=1)
    result = np.full(points.shape, np.nan)
    valid_indices = np.flatnonzero(valid)[domain]
    result[valid_indices] = angstrom2reciprocal(curved[domain], reciprocal=g_vec)
    return (result, np.isfinite(result).all(axis=1)) if return_mask else result
