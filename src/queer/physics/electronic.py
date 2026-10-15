"""Pure generalized Green spectra and bounded Lindhard band-pair contractions."""

import numpy as np

from .broadening import lorentzian
from .validation import positive


def green_function(H, omega, eta, overlap=None, *, fermi_energy=0.0):
    """Retarded Green function; scalar or arbitrary frequency arrays broadcast."""
    eta = positive(eta, "eta")
    H = np.asarray(H, dtype=complex)
    if H.ndim != 2 or H.shape[0] != H.shape[1] or not np.isfinite(H).all():
        raise ValueError("H must be a finite square matrix")
    omega = np.asarray(omega, dtype=float)
    if not np.isfinite(omega).all():
        raise ValueError("omega must be finite")
    metric = np.eye(H.shape[0]) if overlap is None else np.asarray(overlap, dtype=complex)
    if metric.shape != H.shape or not np.isfinite(metric).all():
        raise ValueError("overlap must be finite and have the same shape as H")
    if not np.allclose(metric, metric.conj().T, rtol=1e-10, atol=1e-10):
        raise ValueError("overlap must be Hermitian")
    try:
        np.linalg.cholesky(metric)
    except np.linalg.LinAlgError as error:
        raise ValueError("overlap must be positive definite") from error
    if not np.isfinite(fermi_energy):
        raise ValueError("fermi_energy must be finite")
    return np.linalg.inv((omega[..., None, None] + fermi_energy + 1j * eta) * metric - H)


def spectral_from_green(H, omega_list, eta, overlap=None, *, fermi_energy=0.0):
    green = green_function(H, omega_list, eta, overlap, fermi_energy=fermi_energy)
    if overlap is not None:
        green = np.asarray(overlap) @ green
    return -np.trace(green, axis1=-2, axis2=-1).imag / np.pi


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
        result += lorentzian(difference, eta)
    return result


def lindhard(energies, shifted, occupations, shifted_occupations, broadening):
    """Conventional response from (band,k) arrays without a band²×k temporary."""
    return -sum(
        np.mean((occupations[m] - shifted_occupations[n]) / (energies[m] - shifted[n] + 1j * broadening))
        for m in range(len(energies))
        for n in range(len(shifted))
    )
