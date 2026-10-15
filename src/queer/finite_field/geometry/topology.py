"""Topology diagnostics: Chern numbers and Berry flux in the Omega = -2 Im<d_x u|d_y u> convention."""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad

from .kubo import band_geometry


def hall_chern(flux: float) -> float:
    """Hall Chern number C=-flux for Berry flux normalized by 2π.

    A finite-window flux alone does not certify a whole-Brillouin-zone Chern number.
    """
    return -flux


def chern_radial(model, band: int, k_max: float = 20.0, breakpoints=(), epsabs: float = 1e-10) -> float:
    """Berry flux (1/2pi) Int Omega_band d^2k for an isotropic model (k along x).

    Convert to the Hall convention with hall_chern.

    band: 0-based band index. Use `breakpoints` for sharp features such as the regularization ring
    k_inv of Eq. 9.
    """
    partner = (band + 1) % model.n_orb

    def f(k):
        g = band_geometry(model, k, 0.0, bands=(band, partner))
        return float(g.omega_v * k)  # omega_v holds the first band of the pair

    edges = sorted({0.0, *[b for b in breakpoints if 0 < b < k_max], k_max})
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        # graded sub-intervals: geometric from a small offset to resolve 1/k structure at the origin
        pts = np.geomspace(max(a, 1e-9), b, 12) if a == 0.0 else np.linspace(a, b, 6)
        if a == 0.0:
            pts = np.concatenate([[0.0], pts])
        for lo, hi in zip(pts[:-1], pts[1:]):
            val, _ = quad(f, lo, hi, limit=200, epsabs=epsabs, epsrel=1e-10)
            total += val
    return total


def fhs_flux(model, bands, kx_edges, ky_edges) -> float:
    """Berry flux / 2pi of the band subspace `bands` (indices) through a rectangular, possibly graded grid.

    Fukui-Hatsugai-Suzuki plaquettes with U(1) links det<u(k)|u(k')>; the sign is converted to the
    Omega = -2 Im<d_x u|d_y u> convention (FHS link phases give -Omega dkx dky).
    """
    axes = [np.asarray(edges, float) for edges in (kx_edges, ky_edges)]
    if any(
        axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0)
        for axis in axes
    ):
        raise ValueError("Flux axes must be finite strictly increasing one-dimensional grids")
    bands = np.asarray(bands)
    if (
        bands.ndim != 1
        or not len(bands)
        or bands.dtype.kind not in "iu"
        or len(np.unique(bands)) != len(bands)
        or np.any(bands < 0)
        or np.any(bands >= model.n_orb)
    ):
        raise ValueError("Flux bands must be nonempty unique valid integer band indices")
    KX, KY = np.meshgrid(*axes, indexing="ij")

    if getattr(model, "native_band_geometry", False):
        from ...models.evaluation import basis_overlap, states

        source = model.source_model
        points = model.plane.fractional(source, KX, KY).reshape(-1, 3)
        solution = states(source, points)
        energies = np.asarray(solution.energies)
        channels = np.asarray(getattr(solution, "band_channels", np.zeros(model.n_orb, int)))
        outside = np.setdiff1d(np.arange(model.n_orb), bands)
        for selected in bands:
            for other in outside[channels[outside] == channels[selected]]:
                if np.any(np.abs(energies[selected] - energies[other]) <= model.degeneracy_tol):
                    raise ValueError(
                        "Flux subspace cuts a degenerate block; select a complete separated manifold"
                    )
        V = np.asarray(solution.coefficients)[:, :, bands].reshape((*KX.shape, model.n_orb, len(bands)))
        points = points.reshape((*KX.shape, 3))

        def physical_link(i, j, ip, jp):
            overlap = V[i, j].conj().T @ basis_overlap(source, points[i, j], points[ip, jp]) @ V[ip, jp]
            sign, logabs = np.linalg.slogdet(overlap)
            if not np.isfinite(logabs) or not np.isfinite(sign):
                raise ValueError("Singular physical subspace overlap; refine the flux grid")
            return sign

        Ux = np.array(
            [[physical_link(i, j, i + 1, j) for j in range(KX.shape[1])] for i in range(KX.shape[0] - 1)]
        )
        Uy = np.array(
            [[physical_link(i, j, i, j + 1) for j in range(KX.shape[1] - 1)] for i in range(KX.shape[0])]
        )
    else:
        _, U = np.linalg.eigh(model.hamiltonian(KX, KY))
        V = U[..., :, bands]

        def link(a, b):
            ov = np.conj(np.swapaxes(a, -1, -2)) @ b
            sign, logabs = np.linalg.slogdet(ov)
            if not np.isfinite(logabs).all() or not np.isfinite(sign).all():
                raise ValueError("Singular subspace overlap; refine the flux grid")
            return sign

        Ux = link(V[:-1, :], V[1:, :])
        Uy = link(V[:, :-1], V[:, 1:])
    plaq = Ux[:, :-1] * Uy[1:, :] / (Ux[:, 1:] * Uy[:-1, :])
    return float(-np.sum(np.angle(plaq)) / (2.0 * math.pi))


def graded_axis(k_max: float, n: int, a: float = 1e-5) -> np.ndarray:
    """Symmetric sinh-graded grid on [-k_max, k_max] with spacing ~a near zero (n odd -> includes 0)."""
    if not np.isfinite(k_max) or k_max <= 0 or not np.isfinite(a) or a <= 0:
        raise ValueError("Flux window and grading scale must be finite and positive")
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("Flux grid size must be an integer >= 2")
    u = np.linspace(-1.0, 1.0, n)
    s = np.arcsinh(k_max / a)
    return a * np.sinh(s * u)
