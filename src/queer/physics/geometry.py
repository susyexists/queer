"""Pure Cartesian geometry kernels for retained localized-orbital bases.

Inputs use absolute eV, Cartesian Angstrom derivatives and physical metric
normalization. Model evaluation, execution and artifact formatting are separate.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np


class BandGeometryData(TypedDict):
    """One point in a complete retained basis; matrix axes are (band,band)."""

    energies: np.ndarray
    coefficients: np.ndarray
    overlap: np.ndarray
    connections: np.ndarray
    velocities: np.ndarray
    band_channels: np.ndarray
    coefficient_derivatives: np.ndarray
    projected_derivatives: dict[str, np.ndarray]
    degeneracy_tol: float


def band_geometry_from_operators(solution, derivative, position, degeneracy_tol) -> BandGeometryData:
    """Covariant connections and hbar velocities in the complete retained basis.

    Inside a degenerate block, the coefficient connection is the symmetric
    normalization gauge. Individual-band geometry at such a degeneracy is not
    defined; callers must sum complete equally occupied blocks.
    """
    if np.shape(solution.energies)[1] != 1:
        raise ValueError("operator geometry requires exactly one eigenstate point")
    energies = np.asarray(solution.energies)[:, 0]
    coeff = np.asarray(solution.coefficients)[0]
    metric = np.asarray(solution.overlaps)[0]
    channels = np.asarray(getattr(solution, "band_channels", np.zeros(len(energies), int)))
    dh = np.einsum("in,aij,jm->anm", coeff.conj(), derivative["H"], coeff, optimize=True)
    ds = np.einsum("in,aij,jm->anm", coeff.conj(), derivative["S"], coeff, optimize=True)
    r = np.einsum("in,aij,jm->anm", coeff.conj(), position, coeff, optimize=True)
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


def projected_basis_curvature(data, dr):
    """Retained-space F_ab=partial_a A_b-partial_b A_a-i[A_a,A_b].

    This covariant matrix measures the curvature of the projected position
    operators. A complete orthogonal fixed basis has F=0. Localized continuum
    orbitals need not span a complete Hilbert space, so their F can be nonzero.
    Only first position/overlap derivatives are required.
    """
    c = data["coefficients"]
    d = data["coefficient_derivatives"]
    ds = data["projected_derivatives"]["S"]
    p = data["projected_derivatives"]["r"].conj().transpose(0, 2, 1)
    a = data["connections"]
    n = len(data["energies"])
    if np.shape(dr) != (3, 3, c.shape[0], c.shape[0]):
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


def band_curvatures(data, engine="direct", projected_curvature=None):
    """Per-band curvature (band,Cartesian,Cartesian), with block gauge."""
    if engine not in {"direct", "kubo"}:
        raise ValueError("curvature engine must be 'direct' or 'kubo'")
    a = data["connections"]
    n = len(data["energies"])
    result = np.zeros((n, 3, 3))
    if engine == "kubo":
        for i in range(3):
            for j in range(i + 1, 3):
                q = np.einsum("nm,mn->n", a[i], a[j])
                result[:, i, j] = -2 * q.imag
                result[:, j, i] = -result[:, i, j]
        return result
    if projected_curvature is None:
        raise ValueError("direct curvature requires projected-coordinate curvature")
    f = projected_curvature
    for i in range(3):
        for j in range(i + 1, 3):
            curl = f[i, j] + 1j * (a[i] @ a[j] - a[j] @ a[i])
            result[:, i, j] = curl.diagonal().real
            result[:, j, i] = -result[:, i, j]
    return result


def occupied_mask(data, occupied_bands=None, fermi_energy=0.0):
    energies = data["energies"]
    if occupied_bands is None:
        mask = energies < fermi_energy
    else:
        count = int(occupied_bands)
        if count != occupied_bands or count < 0 or count > len(energies):
            raise ValueError("occupied_bands must be an integer within the band count")
        mask = np.zeros(len(energies), dtype=bool)
        mask[np.argsort(energies, kind="stable")[:count]] = True
    equal = np.abs(energies[:, None] - energies[None, :]) < data["degeneracy_tol"]
    same = data["band_channels"][:, None] == data["band_channels"][None, :]
    if np.any(equal & same & (mask[:, None] != mask[None, :])):
        raise ValueError("occupied/unoccupied boundary splits a degenerate subspace")
    return mask


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


def differentiated_geometry(data, first, second):
    """Cartesian connection gradients in the local normalization gauge.

    Differentiate the generalized eigenvalue perturbation identity analytically;
    this avoids numerical eigenvector phases and k-space difference steps.
    Complete degenerate blocks use the same normalization gauge as band_geometry.
    """
    coeff = data["coefficients"]
    d = data["coefficient_derivatives"]
    projected = data["projected_derivatives"]
    energy = data["energies"]
    gap = energy[None, :] - energy[:, None]
    same_channel = data["band_channels"][:, None] == data["band_channels"][None, :]
    block = (np.abs(gap) <= data["degeneracy_tol"]) & same_channel
    separated = ~block & same_channel
    # Within a degenerate manifold the energy derivative is a Hermitian
    # matrix, rather than a list of diagonal band slopes. Retain it to avoid
    # choosing arbitrary eigenvectors at Kramers/accidental crossings.
    grad_energy = projected["H"] + np.einsum("anm,m->anm", d.conj().transpose(0, 2, 1), energy)
    grad_energy += np.einsum("n,anm->anm", energy, d)
    grad_energy[:, ~block] = 0
    connection_gradient = np.empty((3, 3, len(energy), len(energy)), complex)
    hessian = np.empty_like(connection_gradient)
    for a in range(3):
        for b in range(3):
            ddh = (
                d[a].conj().T @ projected["H"][b]
                + projected["H"][b] @ d[a]
                + coeff.conj().T @ second["H"][a, b] @ coeff
            )
            dds = (
                d[a].conj().T @ projected["S"][b]
                + projected["S"][b] @ d[a]
                + coeff.conj().T @ second["S"][a, b] @ coeff
            )
            dr = (
                d[a].conj().T @ projected["r"][b]
                + projected["r"][b] @ d[a]
                + coeff.conj().T @ first["r"][a, b] @ coeff
            )
            numerator = ddh - projected["S"][b] @ grad_energy[a]
            numerator -= energy[None, :] * dds
            numerator -= d[b] @ grad_energy[a] - grad_energy[a] @ d[b]
            dd = -0.5 * dds
            np.divide(numerator, gap, out=dd, where=separated)
            dd[data["band_channels"][:, None] != data["band_channels"][None, :]] = 0
            connection_gradient[a, b] = 1j * dd + dr.conj().T
            hessian[a, b] = (
                dd.conj().T * energy[None, :]
                + d[b].conj().T @ grad_energy[a]
                + ddh
                + grad_energy[a] @ d[b]
                + energy[:, None] * dd
            )
            hessian[a, b, ~block] = 0
    return connection_gradient, grad_energy, hessian


def pair_metrics(connections, pair, channels):
    """Transition metric and same-channel total metrics for a selected pair."""
    v, c = pair
    transition = connections[:, v, c]
    pair_metric = np.outer(transition, transition.conj()).real
    total_metric = np.empty((2, connections.shape[0], connections.shape[0]))
    for index, band in enumerate(pair):
        other = (np.arange(len(channels)) != band) & (channels == channels[band])
        links = connections[:, band, other]
        total_metric[index] = np.einsum("am,bm->ab", links, links.conj()).real
    return pair_metric, total_metric


def interband_products(xs, xt, inverse_gap_squared):
    """Selected-row Berry and metric products for a Cartesian two-axis frame.

    Arrays end in (selected_band, other_band); leading point axes broadcast.
    Zero inverse gaps denote explicitly excluded unresolved contributions.
    """
    cross = xs * np.conj(xt)
    omega = -2.0 * np.sum(cross.imag * inverse_gap_squared, axis=-1)
    gss = np.abs(xs) ** 2 * inverse_gap_squared
    gst = cross.real * inverse_gap_squared
    gtt = np.abs(xt) ** 2 * inverse_gap_squared
    return omega, gss, gst, gtt
