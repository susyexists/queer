"""Array-based spectral functions along reciprocal and ARPES paths."""

import numpy as np

from ._common import model_energies, points_array, positive
from .arpes import arpes_path
from .response import spectral_from_eig


def spectral_along_path(model, path, omega_list, eta=0.1, execution=None):
    """Return A(omega,k) of shape (Nw,Nk), preserving invalid-point NaN gaps."""
    positive(eta, "eta")
    omega = np.atleast_1d(np.asarray(omega_list, dtype=float))
    if omega.ndim != 1:
        raise ValueError("omega_list must be scalar or one dimensional")
    bands = model_energies(model, path, execution)
    return spectral_from_eig(bands.T, omega, eta)


def iter_spectral_arpes_path_sweep(
    model, band_path, g_vec, photon_energy, v0_list, omega_list, eta=0.1, align_zero=True, execution=None
):
    """Yield (v0, omega_index, spectrum_row) without retaining the sweep.

    Frequencies are signed omega=E-EF in eV (occupied states are negative).
    With the existing zero work-function/reference offset, Ek=photon_energy+omega.
    The zero-alignment reference is calculated once per V0 at omega=0.
    """
    positive(eta, "eta")
    if not np.isfinite(photon_energy):
        raise ValueError("photon_energy must be finite")
    points = points_array(band_path)
    omega = np.atleast_1d(np.asarray(omega_list, dtype=float))
    if omega.ndim != 1 or not np.isfinite(omega).all():
        raise ValueError("omega_list must be scalar or a finite one-dimensional array")
    for v0 in v0_list:
        if not np.isfinite(v0):
            raise ValueError("V0 values must be finite")
        offset = 0.0
        if align_zero:
            reference = model_energies(model, arpes_path(points, g_vec, Ek=photon_energy, V0=v0), execution)
            negative = reference[reference < 0]
            if negative.size:
                offset = negative.max()
        for index, frequency in enumerate(omega):
            bands = model_energies(
                model, arpes_path(points, g_vec, Ek=photon_energy + frequency, V0=v0), execution
            )
            yield v0, index, spectral_from_eig((bands - offset).T, frequency, eta)


def spectral_arpes_path_sweep(
    model, band_path, g_vec, photon_energy, v0_list, omega_list, eta=0.1, align_zero=True, execution=None
):
    """Collect the signed-frequency iterator into {V0: array(Nw,Nk)}."""
    points = points_array(band_path)
    omega = np.atleast_1d(np.asarray(omega_list, dtype=float))
    v0_list = list(v0_list)
    results = {v0: np.empty((len(omega), len(points))) for v0 in v0_list}
    for v0, index, row in iter_spectral_arpes_path_sweep(
        model, points, g_vec, photon_energy, v0_list, omega, eta, align_zero, execution
    ):
        results[v0][index] = row
    return results
