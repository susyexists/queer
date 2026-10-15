"""Stable Fermi functions with explicit temperature and equality conventions."""

import numpy as np
from scipy.special import expit

from .broadening import gaussian
from .constants import KB_EV


def _fermi(energy, thermal_energy, mu, equality):
    if thermal_energy == 0:
        return np.where(energy < mu, 1.0, np.where(energy > mu, 0.0, equality))
    return expit((mu - energy) / thermal_energy)


def thermal_occupation(energy, T=0.025):
    """Legacy thermal energy in eV; equality at zero temperature is one half."""
    values = np.asarray(energy, dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(T) or T < 0:
        raise ValueError("energy must be finite and T must be nonnegative and finite")
    return _fermi(values, T, 0.0, 0.5)


def occupations(energies, mu=0.0, temperature=0.0):
    """Native energies/mu in eV, temperature in Kelvin; equality is unoccupied."""
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature < 0 or not np.isfinite(mu).all():
        raise ValueError("temperature must be nonnegative Kelvin; chemical potential must be finite")
    energy = np.asarray(energies, dtype=float)
    if temperature == 0:
        return (energy < mu).astype(float)
    return _fermi(energy, KB_EV * temperature, mu, 0.0)


def fermi_derivative(energies, mu=0.0, temperature=0.0, broadening=0.05):
    """Positive -df/dE in eV^-1, with normalized Gaussian regularization at T=0."""
    temperature = float(temperature)
    f = occupations(energies, mu, temperature)
    if temperature > 0:
        return f * (1 - f) / (KB_EV * temperature)
    if not np.isfinite(broadening) or broadening <= 0:
        raise ValueError("zero-temperature broadening must be positive eV")
    return gaussian(np.asarray(energies) - mu, broadening)


def occupation_derivatives(energy, mu, temperature, eta):
    """Return f, df/dE and d²f/dE² using the native Kelvin convention."""
    energy = np.asarray(energy)
    f = occupations(energy, mu, temperature)
    if temperature > 0:
        thermal = KB_EV * temperature
        first = -f * (1 - f) / thermal
        second = f * (1 - f) * (1 - 2 * f) / thermal**2
    else:
        shape = gaussian(energy - mu, eta)
        first = -shape
        second = (energy - mu) * shape / eta**2
    return f, first, second
