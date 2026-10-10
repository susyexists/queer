"""Shared Cartesian, nonorthogonal tight-binding response primitives.

Energy matrices use eV, real-space vectors use Angstrom, and k-points are
fractional reciprocal coordinates. ``velocities`` returns hbar times velocity,
in eV Angstrom. Berry connections use A = i <u|partial_k u>.
"""

from __future__ import annotations

from contextvars import ContextVar
from types import SimpleNamespace

import numpy as np
from scipy.constants import hbar as HBAR
from scipy.linalg import eigh
from scipy.special import expit

from ..execution import ExecutionOptions, execution_session
from ..geometry.lattice import validated_points

KB_EV = 8.617333262145e-5
E_CHARGE = 1.602176634e-19
E2_H = E_CHARGE**2 / (2 * np.pi * HBAR)
_POINT_EXECUTION = ContextVar("queer_observable_point_execution", default=None)


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


def kmesh(grid, k_start=None, k_vectors=None):
    """Uniform Gamma-origin fractional mesh; the upper boundary is excluded."""
    dimensions = np.asarray(grid)
    if dimensions.shape != (3,) or not np.equal(dimensions, np.round(dimensions)).all():
        raise ValueError("grid must contain three positive integers")
    if np.any(dimensions < 1) or not np.isfinite(dimensions).all():
        raise ValueError("grid must contain three positive integers")
    start = np.zeros(3) if k_start is None else np.asarray(k_start, dtype=float)
    vectors = np.eye(3) if k_vectors is None else np.asarray(k_vectors, dtype=float)
    if (
        start.shape != (3,)
        or vectors.shape != (3, 3)
        or not np.isfinite(start).all()
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("k_start/k_vectors must be finite (3,)/(3,3) arrays")
    axes = [np.arange(int(n), dtype=float) / n for n in dimensions]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return start + mesh @ vectors


def normalized_weights(points, weights=None):
    n = len(points)
    if n < 1:
        raise ValueError("at least one k-point is required")
    values = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if values.shape != (n,) or not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("weights must be finite nonnegative values with positive sum")
    return values / values.sum()


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
    return SimpleNamespace(
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


def band_geometry(model, k, degeneracy_tol=None):
    """Covariant connections and hbar velocities in the complete retained basis.

    Inside a degenerate block, the coefficient connection is the symmetric
    normalization gauge. Individual-band geometry at such a degeneracy is not
    defined; callers must sum complete equally occupied blocks.
    """
    degeneracy_tol = degeneracy_tolerance(model, degeneracy_tol)
    solution = states(model, np.asarray(k)[None, :])
    energies = np.asarray(solution.energies)[:, 0]
    coeff = np.asarray(solution.coefficients)[0]
    metric = np.asarray(solution.overlaps)[0]
    channels = np.asarray(getattr(solution, "band_channels", np.zeros(len(energies), int)))
    derivative = derivatives(model, k)
    dh = np.einsum("in,aij,jm->anm", coeff.conj(), derivative["H"], coeff, optimize=True)
    ds = np.einsum("in,aij,jm->anm", coeff.conj(), derivative["S"], coeff, optimize=True)
    r = np.einsum("in,aij,jm->anm", coeff.conj(), position_matrix(model, k), coeff, optimize=True)
    delta = energies[None, :] - energies[:, None]
    same_channel = channels[:, None] == channels[None, :]
    separated = (np.abs(delta) > degeneracy_tol) & same_channel
    d = -0.5 * ds
    numerator = dh - ds * energies[None, None, :]
    np.divide(numerator, delta[None], out=d, where=separated[None])
    d[:, ~same_channel] = 0
    connections = 1j * d + r.conj().transpose(0, 2, 1)
    # Covariance identity makes this Hermitian; averaging removes roundoff only.
    residual = np.max(np.abs(connections - connections.conj().transpose(0, 2, 1)), initial=0)
    scale = np.max(np.abs(connections), initial=0)
    if residual > 1e-7 * (1 + scale):
        raise ValueError("position/overlap derivatives violate dipole covariance")
    connections = (connections + connections.conj().transpose(0, 2, 1)) * 0.5
    velocity = dh - 0.5 * ds * (energies[:, None] + energies[None, :])[None]
    velocity += -1j * delta[None] * (r + r.conj().transpose(0, 2, 1)) * 0.5
    return {
        "energies": energies,
        "coefficients": coeff,
        "overlap": metric,
        "connections": connections,
        "velocities": velocity,
        "band_channels": channels,
        "coefficient_derivatives": d,
        "projected_derivatives": {"H": dh, "S": ds, "r": r},
        "degeneracy_tol": degeneracy_tol,
    }


def band_connections(model, k, degeneracy_tol=None):
    data = band_geometry(model, k, degeneracy_tol)
    return data["energies"], data["connections"]


def basis_curvature(model, k, data=None):
    """Retained-space F_ab=partial_a A_b-partial_b A_a-i[A_a,A_b].

    This covariant matrix measures the curvature of the projected position
    operators. A complete orthogonal fixed basis has F=0. Localized continuum
    orbitals need not span a complete Hilbert space, so their F can be nonzero.
    Only first position/overlap derivatives are required.
    """
    data = band_geometry(model, k) if data is None else data
    c = data["coefficients"]
    d = data["coefficient_derivatives"]
    ds = data["projected_derivatives"]["S"]
    p = data["projected_derivatives"]["r"].conj().transpose(0, 2, 1)
    a = data["connections"]
    dr = derivatives(model, k)["r"]
    n = len(data["energies"])
    if np.shape(dr) != (3, 3, model.nbnd, model.nbnd):
        raise ValueError("basis curvature needs position derivatives shaped (3,3,N,N)")
    result = np.zeros((3, 3, n, n), complex)
    for i in range(3):
        for j in range(i + 1, 3):
            dp = dr[i, j].conj().T - dr[j, i].conj().T
            f = (
                c.conj().T @ dp @ c
                - 1j * (d[i] @ d[j] - d[j] @ d[i])
                - d[i] @ p[j]
                + p[j] @ d[i]
                + d[j] @ p[i]
                - p[i] @ d[j]
                - ds[i] @ p[j]
                + ds[j] @ p[i]
                - 1j * (a[i] @ a[j] - a[j] @ a[i])
            )
            result[i, j] = (f + f.conj().T) / 2
            result[j, i] = -result[i, j]
    return result


def velocities(model, k, degeneracy_tol=None):
    data = band_geometry(model, k, degeneracy_tol)
    return data["energies"], data["velocities"]


def energy_blocks(data):
    """Complete same-channel energy manifolds at the source resolution."""
    energy = data["energies"]
    channels = data["band_channels"]
    blocks = []
    for channel in np.unique(channels):
        indices = np.flatnonzero(channels == channel)
        indices = indices[np.argsort(energy[indices], kind="stable")]
        cuts = np.flatnonzero(np.diff(energy[indices]) > data["degeneracy_tol"]) + 1
        blocks.extend(np.split(indices, cuts))
    return blocks


def velocity_products(data):
    """Per-band real velocity products including degenerate-block coherences.

    Summing complete equally weighted degenerate manifolds gives the invariant
    trace V_a V_b. Resolved bands reduce to partial_a E partial_b E.
    """
    degenerate = np.zeros((len(data["energies"]), len(data["energies"])), bool)
    for block in energy_blocks(data):
        degenerate[np.ix_(block, block)] = True
    v = data["velocities"]
    return np.einsum("anm,bmn,nm->nab", v, v, degenerate).real


def occupations(energies, mu=0.0, temperature=0.0):
    """Fermi occupations; energies/mu in eV, temperature in Kelvin."""
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature < 0 or not np.isfinite(mu).all():
        raise ValueError("temperature must be nonnegative Kelvin; chemical potential must be finite")
    energy = np.asarray(energies, dtype=float)
    if temperature == 0:
        return (energy < mu).astype(float)
    return expit((mu - energy) / (KB_EV * temperature))


def fermi_derivative(energies, mu=0.0, temperature=0.0, broadening=0.05):
    """Positive -df/dE in eV^-1; a normalized Gaussian represents T=0."""
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature < 0 or not np.isfinite(mu).all():
        raise ValueError("temperature must be nonnegative Kelvin; chemical potential must be finite")
    if temperature > 0:
        f = occupations(energies, mu, temperature)
        return f * (1 - f) / (KB_EV * temperature)
    if not np.isfinite(broadening) or broadening <= 0:
        raise ValueError("zero-temperature broadening must be positive eV")
    return np.exp(-0.5 * ((np.asarray(energies) - mu) / broadening) ** 2) / (np.sqrt(2 * np.pi) * broadening)


def chemical_potentials(model, fermi_energies=None):
    values = np.atleast_1d(model.fermi_energy if fermi_energies is None else fermi_energies).astype(float)
    if values.ndim != 1 or not np.isfinite(values).all() or not values.size:
        raise ValueError("fermi_energies must be a nonempty finite eV sequence")
    return values


def spin_degeneracy(model):
    return int(getattr(model, "spin_degeneracy", 1))


def map_points(model, points, function, execution=None):
    """Ordered bounded process evaluation with the shared execution budget."""
    options = (execution or getattr(model, "execution", None) or ExecutionOptions()).resolved()
    options.bounded_batch_size(1024 * model.nbnd**2)
    inner = ExecutionOptions(
        workers=1,
        batch_size=options.batch_size,
        native_threads=options.native_threads,
        memory_mb=None if options.memory_mb is None else options.memory_mb / options.workers,
    )
    from threadpoolctl import threadpool_limits

    if options.workers == 1:
        with threadpool_limits(limits=options.native_threads):
            return [_evaluate_point(function, point, inner) for point in points]
    from joblib import delayed

    with execution_session(options) as session:
        return list(session.parallel()(delayed(_evaluate_point)(function, point, inner) for point in points))


def _evaluate_point(function, point, execution):
    token = _POINT_EXECUTION.set(execution)
    try:
        return function(point)
    finally:
        _POINT_EXECUTION.reset(token)
