"""Green functions, Lorentzian spectra, and electronic susceptibility."""

import numpy as np

from ..functions import fd
from ._common import model_energies, points_array, positive


def green_function(H, omega, eta):
    """Retarded Green function; scalar or arbitrary frequency arrays broadcast."""
    eta = positive(eta, "eta")
    H = np.asarray(H, dtype=complex)
    if H.ndim != 2 or H.shape[0] != H.shape[1] or not np.isfinite(H).all():
        raise ValueError("H must be a finite square matrix")
    omega = np.asarray(omega, dtype=float)
    if not np.isfinite(omega).all():
        raise ValueError("omega must be finite")
    identity = np.eye(H.shape[0])
    return np.linalg.inv((omega[..., None, None] + 1j * eta) * identity - H)


def spectral_from_green(H, omega_list, eta):
    green = green_function(H, omega_list, eta)
    return -np.trace(green.imag, axis1=-2, axis2=-1) / np.pi


def spectral_from_eig(evals, omega_list, eta):
    """Sum Lorentzian bands; frequencies are leading axes, bands the last axis."""
    eta = positive(eta, "eta")
    eigenvalues = np.asarray(evals, dtype=float)
    omega = np.asarray(omega_list, dtype=float)
    if eigenvalues.ndim == 0:
        eigenvalues = eigenvalues.reshape(1)
    if not np.isfinite(omega).all():
        raise ValueError("omega must be finite")
    if np.isinf(eigenvalues).any():
        raise ValueError("eigenvalues must not contain infinities; NaNs denote spectral gaps")
    frequency = omega.reshape(omega.shape + (1,) * (eigenvalues.ndim - 1))
    result = np.zeros(omega.shape + eigenvalues.shape[:-1])
    for band in range(eigenvalues.shape[-1]):
        difference = frequency - eigenvalues[..., band]
        result += eta / np.pi / (difference**2 + eta**2)
    return result


def susceptibility(
    model,
    q_path,
    k_mesh,
    band_indices=None,
    temperature=0.025,
    broadening=1e-7,
    execution=None,
    mesh_energies=None,
):
    """Lindhard response in 1/eV at q points, with fractional reciprocal inputs.

    Returns a complex array (Nq,). This conventional response differs from the
    historical model.suscep pair, whose second component is a joint DOS.
    """
    broadening = positive(broadening, "broadening")
    q_path = points_array(q_path)
    k_mesh = points_array(k_mesh)
    if not len(k_mesh) or not np.isfinite(k_mesh).all() or not np.isfinite(q_path).all():
        raise ValueError("k_mesh must be nonempty and all momentum points finite")
    energies = (
        model_energies(model, k_mesh, execution)
        if mesh_energies is None
        else np.asarray(mesh_energies, dtype=float)
    )
    if energies.ndim != 2 or energies.shape[1] != len(k_mesh) or not np.isfinite(energies).all():
        raise ValueError("mesh_energies must be a finite (Nb,Nk) array")
    indices = np.arange(len(energies)) if band_indices is None else np.asarray(band_indices)
    if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
        raise ValueError("band_indices must be a one-dimensional integer list")
    indices = indices.astype(int)
    if np.any(indices < 0) or np.any(indices >= len(energies)):
        raise ValueError("band index outside the model")
    energies = energies[indices]
    occupations = fd(energies, temperature)
    result = np.empty(len(q_path), dtype=complex)
    for i, point in enumerate(q_path):
        shifted = model_energies(model, k_mesh + point, execution)[indices]
        shifted_fd = fd(shifted, temperature)
        # Sum band pairs without allocating an Nb*Nb*Nk tensor.
        result[i] = -sum(
            np.mean((occupations[m] - shifted_fd[n]) / (energies[m] - shifted[n] + 1j * broadening))
            for m in range(len(indices))
            for n in range(len(indices))
        )
    return result


def susceptibility_point(model, point, k_mesh, **kwargs):
    return susceptibility(model, np.asarray(point).reshape(1, 3), k_mesh, **kwargs)[0]
