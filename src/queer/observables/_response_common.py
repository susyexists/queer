"""Shared optical response configuration and model-aware sampling.

Numerical density recursion is owned by physics.response; this module owns
parameter resolution, physical response prefactors and returned metadata.
"""

import numpy as np
from scipy.constants import epsilon_0

from ..execution import ExecutionOptions
from ..geometry.lattice import validated_points
from ..models.evaluation import band_geometry, basis_curvature, spin_degeneracy, volume
from ..physics.constants import E_CHARGE, HBAR
from ..physics.response import linear_density as _linear_density
from ..physics.statistics import occupations
from ._sampling import _weighted_sum, kmesh, normalized_weights


def _settings(
    model,
    omega,
    eta,
    chemical_potential,
    temperature_kelvin,
    grid,
    kpoints,
    weights,
    scissors,
    degeneracy_tol=None,
):
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    if omega.ndim != 1 or not len(omega) or not np.isfinite(omega).all() or np.any(omega < 0):
        raise ValueError("omega must be a nonempty finite nonnegative photon-energy sequence in eV")
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be positive and finite in eV")
    if not np.isfinite(temperature_kelvin) or temperature_kelvin < 0:
        raise ValueError("temperature_kelvin must be finite and nonnegative")
    mu = float(getattr(model, "fermi_energy", 0.0) if chemical_potential is None else chemical_potential)
    if not np.isfinite(mu):
        raise ValueError("chemical_potential must be finite in eV")
    if not np.isfinite(scissors) or scissors < 0:
        raise ValueError("scissors must be a finite nonnegative conduction-band shift in eV")
    degeneracy_tol = (
        getattr(model, "degeneracy_tolerance", 1e-9) if degeneracy_tol is None else degeneracy_tol
    )
    if not np.isfinite(degeneracy_tol) or degeneracy_tol <= 0:
        raise ValueError("degeneracy_tol must be positive and finite in eV")
    points = kmesh(grid) if kpoints is None else validated_points(kpoints)
    weight = normalized_weights(points, weights)
    metadata = {
        "chemical_potential_eV": mu,
        "temperature_kelvin": float(temperature_kelvin),
        "eta_eV": float(eta),
        "scissors_eV": float(scissors),
        "scissors_definition": "rigid upward shift of states above the chemical potential",
        "frequency_convention": "exp(-i omega t); retarded z=omega+i eta",
        "spin_degeneracy": spin_degeneracy(model),
        "cell_volume_angstrom3": volume(model),
        "kpoint_count": len(points),
        "cartesian_order": ["x", "y", "z"],
        "basis": "complete retained tight-binding basis",
        "degeneracy_tolerance_eV": float(degeneracy_tol),
    }
    return omega, mu, points, weight, metadata


def _occupied_manifold(data, occupied_bands, mu):
    """A complete energy manifold, with optional explicit per-channel counts.

    A scalar counts states in the complete model; a mapping counts states in
    each channel independently. All indices remain zero-based in native APIs.
    """
    from ..physics.geometry import occupied_mask as _occupied

    if not isinstance(occupied_bands, dict):
        return _occupied(data, occupied_bands, mu)
    channels = np.asarray(data["band_channels"])
    mask = np.zeros(len(channels), bool)
    unknown = set(map(str, occupied_bands)) - {str(int(c)) for c in np.unique(channels)}
    if unknown:
        raise ValueError(f"Unknown occupied channel keys: {sorted(unknown)}")
    for channel in np.unique(channels):
        indices = np.flatnonzero(channels == channel)
        if str(int(channel)) not in occupied_bands and int(channel) not in occupied_bands:
            raise ValueError("occupied_bands must provide a count for every channel")
        count = occupied_bands.get(str(int(channel)), occupied_bands.get(int(channel)))
        local = {**data, "energies": data["energies"][indices], "band_channels": channels[indices]}
        mask[indices] = _occupied(local, count, mu)
    return mask


def _scissor_geometry(data, mu, scissors, occupied_bands=None):
    if not scissors:
        return data
    result = dict(data)
    original = data["energies"]
    conduction = original >= mu if occupied_bands is None else ~_occupied_manifold(data, occupied_bands, mu)
    shifted = original + scissors * conduction
    gap = shifted[:, None] - shifted[None, :]
    velocity = 1j * gap[None] * data["connections"]
    diagonal = np.arange(len(original))
    velocity[:, diagonal, diagonal] = data["velocities"][:, diagonal, diagonal]
    result["energies"] = shifted
    result["velocities"] = velocity
    return result


def _linear_response(
    model, omega, eta, mu, temperature, points, weight, scissors, execution, degeneracy_tol=None
):
    z = omega + 1j * eta
    options = (execution or getattr(model, "execution", None) or ExecutionOptions()).resolved()
    frequency_batch = _frequency_batch(model, len(z), options, 48, 6, 32)

    def point_response(k):
        raw = band_geometry(model, k, degeneracy_tol)
        data = _scissor_geometry(raw, mu, scissors)
        f = occupations(data["energies"], mu, temperature)
        contact = np.einsum("abnn,n->ab", basis_curvature(model, k, raw), f)
        answer = np.empty((len(z), 3, 3), complex)
        for start in range(0, len(z), frequency_batch):
            stop = start + frequency_batch
            density = _linear_density(data, z[start:stop], mu, temperature, eta)
            answer[start:stop] = np.einsum("anm,wbmn->wab", data["velocities"], density)
            answer[start:stop] += contact
        return answer

    bytes_per_point = 16 * (model.nbnd**2 * (48 + 6 * frequency_batch) + 9 * len(z))
    reduced = _weighted_sum(model, points, weight, point_response, (len(z), 3, 3), options, bytes_per_point)
    # hbar*v is eV Angstrom and rho/E is Angstrom/eV. Convert Angstrom^2
    # per cell volume to inverse metres, then S/m to S/cm.
    sigma = -spin_degeneracy(model) * E_CHARGE**2 / HBAR * 1e8 / volume(model) * reduced
    dielectric = np.eye(3)[None] + 1j * sigma * 100 / (epsilon_0 * (z * E_CHARGE / HBAR)[:, None, None])
    return sigma, dielectric


def _frequency_batch(model, count, options, fixed_matrices, frequency_matrices, default):
    limit = min(count, default)
    if options.memory_mb is not None:
        budget = options.memory_mb * 1024**2 * 0.8 / options.workers
        available = budget - 16 * model.nbnd**2 * fixed_matrices - 16 * count * 27
        limit = min(limit, int(available / (16 * model.nbnd**2 * frequency_matrices)))
        if limit < 1:
            raise ValueError("memory_mb is too small for one optical k-point and photon energy per worker")
    return limit
