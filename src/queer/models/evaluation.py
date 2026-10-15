"""Native model evaluation with explicit Cartesian operator conventions."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
from scipy.linalg import eigh

from ..geometry.lattice import validated_points
from ..physics.geometry import band_geometry_from_operators, projected_basis_curvature
from .contracts import EigenstateBatch

_POINT_EXECUTION = ContextVar("queer_observable_point_execution", default=None)


@contextmanager
def point_execution(execution):
    """Scope the nested eigensolver budget for one observable point evaluation."""
    token = _POINT_EXECUTION.set(execution)
    try:
        yield
    finally:
        _POINT_EXECUTION.reset(token)


def cell(model):
    """Return the real-space cell with lattice vectors in rows, in Angstrom."""
    lattice = getattr(model, "lattice", None)
    if lattice is None:
        reciprocal = getattr(model, "g_vec", None)
        if reciprocal is None:
            raise ValueError("Cartesian observables require lattice/reciprocal vectors")
        lattice = 2 * np.pi * np.linalg.inv(reciprocal).T
    lattice = np.asarray(lattice, dtype=float)
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all() or abs(np.linalg.det(lattice)) < 1e-14:
        raise ValueError("lattice must be a finite nonsingular (3,3) Angstrom cell")
    return lattice


def reciprocal(model):
    return 2 * np.pi * np.linalg.inv(cell(model)).T


def volume(model):
    return float(abs(np.linalg.det(cell(model))))


def states(model, points, bands=None, execution=None, spin=None):
    """Absolute eigenvalues (Nb,Nk), coefficients (Nk,N,Nb), metric overlaps."""
    if execution is None:
        execution = _POINT_EXECUTION.get()
    points = validated_points(getattr(points, "path", points))
    if hasattr(model, "eigenstates"):
        kwargs = {"bands": bands, "execution": execution}
        if spin is not None:
            kwargs["spin"] = spin
        return model.eigenstates(points, **kwargs)
    if spin is not None:
        raise ValueError("spin channel selection requires a model with eigenstates support")
    selection = model._band_indices(bands) if hasattr(model, "_band_indices") else np.arange(model.nbnd)
    energies = np.empty((len(selection), len(points)))
    coefficients = np.empty((len(points), model.nbnd, len(selection)), complex)
    overlaps = np.empty((len(points), model.nbnd, model.nbnd), complex)
    for i, point in enumerate(points):
        overlap = overlap_matrix(model, point)
        energy, coeff = eigh(model.hamiltonian(point), overlap)
        energies[:, i] = energy[selection]
        coefficients[i] = coeff[:, selection]
        overlaps[i] = overlap
    return EigenstateBatch(
        energies=energies,
        coefficients=coefficients,
        overlaps=overlaps,
        band_channels=np.zeros(len(selection), dtype=int),
    )


def overlap_matrix(model, k):
    return model.overlap(k) if hasattr(model, "overlap") else np.eye(model.nbnd, dtype=complex)


def position_matrix(model, k):
    return model.position(k) if hasattr(model, "position") else np.zeros((3, model.nbnd, model.nbnd), complex)


def basis_overlap(model, k, kprime):
    if hasattr(model, "basis_overlap"):
        return model.basis_overlap(k, kprime)
    return np.eye(model.nbnd, dtype=complex)


def derivatives(model, k, order=1):
    """Cartesian derivatives, using analytic Fourier moments for Wannier models."""
    if order not in (1, 2):
        raise ValueError("derivative order must be one or two")
    if hasattr(model, "derivatives"):
        return model.derivatives(k, order=order)
    if not hasattr(model, "hopping_matrices"):
        raise ValueError("model must provide Cartesian matrix derivatives")
    vectors = model.translations @ cell(model)
    phase = np.exp(2j * np.pi * (model.translations @ np.asarray(k)))
    if order == 1:
        h = np.einsum("r,ra,rij->aij", phase * 1j, vectors, model.hopping_matrices)
        return {"H": h, "S": np.zeros_like(h), "r": np.zeros((3, *h.shape), complex)}
    h = -np.einsum("r,ra,rb,rij->abij", phase, vectors, vectors, model.hopping_matrices)
    return {"H": h, "S": np.zeros_like(h), "r": np.zeros((3, *h.shape), complex)}


def degeneracy_tolerance(model, value=None):
    value = float(getattr(model, "degeneracy_tolerance", 1e-9) if value is None else value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("degeneracy_tol must be positive")
    return value


def band_connections(model, k, degeneracy_tol=None):
    data = band_geometry(model, k, degeneracy_tol)
    return data["energies"], data["connections"]


def velocities(model, k, degeneracy_tol=None):
    data = band_geometry(model, k, degeneracy_tol)
    return data["energies"], data["velocities"]


def chemical_potentials(model, fermi_energies=None):
    values = np.atleast_1d(model.fermi_energy if fermi_energies is None else fermi_energies).astype(float)
    if values.ndim != 1 or not np.isfinite(values).all() or not values.size:
        raise ValueError("fermi_energies must be a nonempty finite eV sequence")
    return values


def spin_degeneracy(model):
    return int(getattr(model, "spin_degeneracy", 1))


def band_geometry(model, k, degeneracy_tol=None):
    """Evaluate the full retained-basis covariant geometry at fractional k."""
    tolerance = degeneracy_tolerance(model, degeneracy_tol)
    solution = states(model, np.asarray(k)[None, :])
    return band_geometry_from_operators(solution, derivatives(model, k), position_matrix(model, k), tolerance)


def basis_curvature(model, k, data=None):
    data = band_geometry(model, k) if data is None else data
    return projected_basis_curvature(data, derivatives(model, k)["r"])
