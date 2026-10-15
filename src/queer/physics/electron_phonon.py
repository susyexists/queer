"""Reduced-coupling EPW contractions with their historical normalization.

Electron energies and thermal energy are eV; phonons and input couplings are
meV. The negative imaginary regulator is intentional and is not the retarded
electronic Green-function convention.
"""

import numpy as np

from .statistics import thermal_occupation


def reduced_coupling(phonons, coupling):
    if np.any(phonons < 0):
        raise ValueError("reduced coupling requires nonnegative phonon frequencies")
    return coupling * np.sqrt(phonons)[..., None]


def electron_hole_response(e_k, e_kq, phonons, ef, temperature, broadening):
    numerator = thermal_occupation(e_k - ef, temperature) - thermal_occupation(e_kq - ef, temperature)
    return numerator / (e_k - e_kq - 1j * broadening - phonons * 1e-3)


def self_energy(phonons, reduced, e_k, e_kq, ef, temperature, broadening):
    numerator = thermal_occupation(e_k - ef, temperature) - thermal_occupation(e_kq - ef, temperature)
    denominator = (e_k - e_kq)[None, ...] - 1j * broadening - phonons[..., None] * 1e-3
    return np.mean(np.abs(reduced) ** 2 * 1e-6 * numerator[None, ...] / denominator, axis=2).T
