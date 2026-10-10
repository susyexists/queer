"""Gauge-invariant geometric (shift-vector) phase along a field line (PHYSICAL preset).

Between two avoided crossings the interband coherence acquires, besides the dynamical phase
Phi = (1/(e|E|)) Int eps_g ds, the gauge-invariant phase

    gamma_R = Int_{s_a}^{s_b} R_s ds,    R_s = d/ds arg<c|dH/ds|v> - (A_c - A_v)   (shift vector),

evaluated with the link formula

    gamma_R = sum_i arg[ v_cv(s_{i+1}) / v_cv(s_i) * <c_i|c_{i+1}> <v_{i+1}|v_i> ],  v_cv = <c|dH/ds|v>,

which is invariant under independent U(1) gauge changes of |c> and |v> at every s. A sign change of
the real interband element of a real Hamiltonian gives gamma_R = pi (Stueckelberg sin^2 vs cos^2).
The sign with which gamma_R enters the interference term is fixed against the TDSE validator.
"""

from __future__ import annotations

import numpy as np

from ..models.base import field_frame, rotate_derivatives


def _states(model, s, t, angle, bands):
    if hasattr(model, "phase_states"):
        return model.phase_states(s, t, angle, bands)
    kx, ky = field_frame(s, np.full_like(s, t), angle)
    eps, U = np.linalg.eigh(model.hamiltonian(kx, ky))
    dHx, dHy = model.dhamiltonian(kx, ky)
    dHs, _ = rotate_derivatives(dHx, dHy, angle)
    v, c = bands
    uc, uv = U[:, :, c], U[:, :, v]
    vcv = np.einsum("ki,kij,kj->k", np.conj(uc), dHs, uv)
    return uc, uv, vcv


def geometric_phase(
    model,
    s_a: float,
    s_b: float,
    t: float,
    angle: float = 0.0,
    n: int = 2001,
    bands: tuple[int, int] | None = None,
) -> float:
    """gamma_R (rad) accumulated from s_a to s_b on the line at transverse coordinate t."""
    bands = model.band_pair() if bands is None else bands
    s = np.linspace(s_a, s_b, n)
    uc, uv, vcv = _states(model, s, t, angle, bands)
    tiny = 1e-12 * np.max(np.abs(vcv))
    zero = np.abs(vcv) <= tiny
    if np.any(zero):  # a node exactly on a zero of v_cv: nudge it (the link phase is +-pi across the zero)
        s[zero] += 1e-6 * (s_b - s_a) / (n - 1)
        uc2, uv2, vcv2 = _states(model, s[zero], t, angle, bands)
        uc[zero], uv[zero], vcv[zero] = uc2, uv2, vcv2
    if hasattr(model, "phase_links"):
        occ, ovv = model.phase_links(s, t, angle, bands=bands, uc=uc, uv=uv)
    else:
        occ = np.einsum("ki,ki->k", np.conj(uc[:-1]), uc[1:])
        ovv = np.einsum("ki,ki->k", np.conj(uv[1:]), uv[:-1])
    links = vcv[1:] / vcv[:-1] * occ * ovv
    return float(np.sum(np.angle(links)))


def shift_vector(
    model, s: float, t: float, angle: float = 0.0, h: float = 1e-7, bands: tuple[int, int] | None = None
) -> float:
    """Shift vector R_s (A) at a point by a symmetric two-link difference."""
    return geometric_phase(model, s - h, s + h, t, angle, n=3, bands=bands) / (2.0 * h)


def geometric_phase_refined(
    model,
    s_a: float,
    s_b: float,
    t: float,
    angle: float = 0.0,
    *,
    n: int = 201,
    n_max: int = 3201,
    tolerance: float = 1e-5,
    bands: tuple[int, int] | None = None,
) -> tuple[float, dict]:
    """Refine physical links until their circular phase difference converges.

    An exhausted node budget returns the finest phase and an explicit failed
    convergence flag. Phases are compared modulo 2pi, preserving the returned
    continuous sum used by the interference calculation.
    """
    for name, value in (("n", n), ("n_max", n_max)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 3:
            raise ValueError(f"{name} must be an integer of at least three nodes")
    if n_max < n:
        raise ValueError("n_max must be at least n")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("phase tolerance must be finite and positive")
    if not np.isfinite([s_a, s_b, t, angle]).all():
        raise ValueError("phase coordinates and angle must be finite")
    nodes = int(n)
    phase = geometric_phase(model, s_a, s_b, t, angle, n=nodes, bands=bands)
    error: float | None = None
    refinements = 0
    converged = False
    while nodes < n_max:
        refined_nodes = min(2 * nodes - 1, int(n_max))
        fine = geometric_phase(model, s_a, s_b, t, angle, n=refined_nodes, bands=bands)
        refinements += 1
        if np.isfinite([phase, fine]).all():
            error = float(abs(np.angle(np.exp(1j * (fine - phase)))))
            converged = error <= tolerance
        else:
            error = None
        phase, nodes = fine, refined_nodes
        if converged:
            break
    return phase, {
        "error": error,
        "converged": bool(converged),
        "nodes": nodes,
        "refinements": refinements,
        "tolerance": tolerance,
        "node_budget": int(n_max),
    }


def shift_vector_refined(
    model,
    s: float,
    t: float,
    angle: float = 0.0,
    *,
    h: float,
    tolerance: float = 1e-5,
    bands: tuple[int, int] | None = None,
) -> tuple[float, dict]:
    """Check a physical shift vector against a halved symmetric step.

    ``error`` is the relative difference normalized by max(|R_h|, |R_h/2|,
    1 Angstrom), so an almost zero shift has a useful absolute error criterion.
    """
    if not np.isfinite(h) or h <= 0:
        raise ValueError("shift-vector h must be finite and positive")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("shift-vector tolerance must be finite and positive")
    if not np.isfinite([s, t, angle]).all():
        raise ValueError("shift-vector coordinates and angle must be finite")
    coarse = shift_vector(model, s, t, angle, h=h, bands=bands)
    fine = shift_vector(model, s, t, angle, h=h / 2, bands=bands)
    finite = np.isfinite([coarse, fine]).all()
    error = float(abs(fine - coarse) / max(abs(fine), abs(coarse), 1.0)) if finite else None
    return fine, {
        "error": error,
        "converged": bool(error is not None and error <= tolerance),
        "h": h / 2,
        "coarse_h": h,
        "nodes": 3,
        "tolerance": tolerance,
    }


def shift_field(
    model, kx, ky, angle: float = 0.0, rel_h: float = 1e-3, bands: tuple[int, int] | None = None
) -> np.ndarray:
    """Shift vector R_s (A) for motion along the field direction `angle` at Cartesian k (vectorized).

    Same two-link difference as `shift_vector`, with step h = rel_h * max(|k|, 1e-8) per point. Used to
    certify symmetries at the wavefunction level: a unitary operation with R e = sigma e (sigma = +-1)
    maps R_s(k) to R_s(R k) = sigma R_s(k).
    """
    bands = model.band_pair() if bands is None else bands
    if hasattr(model, "phase_states"):
        kx, ky = np.broadcast_arrays(np.asarray(kx, float), np.asarray(ky, float))
        cs, sn = np.cos(angle), np.sin(angle)
        shifts = [
            shift_vector(
                model,
                x * cs + y * sn,
                -x * sn + y * cs,
                angle,
                h=rel_h * max(np.hypot(x, y), 1e-8),
                bands=bands,
            )
            for x, y in zip(kx.ravel(), ky.ravel())
        ]
        return np.asarray(shifts).reshape(kx.shape)
    kx, ky = np.broadcast_arrays(np.asarray(kx, dtype=float), np.asarray(ky, dtype=float))
    shape = kx.shape
    kx, ky = kx.ravel(), ky.ravel()
    e = np.array([np.cos(angle), np.sin(angle)])
    h = rel_h * np.maximum(np.hypot(kx, ky), 1e-8)
    pts = [(kx + j * h * e[0], ky + j * h * e[1]) for j in (-1, 0, 1)]
    v, c = bands
    uc, uv, vcv = [], [], []
    for px, py in pts:
        _, U = np.linalg.eigh(model.hamiltonian(px, py))
        dHx, dHy = model.dhamiltonian(px, py)
        dHs, _ = rotate_derivatives(dHx, dHy, angle)
        uc.append(U[:, :, c])
        uv.append(U[:, :, v])
        vcv.append(np.einsum("ki,kij,kj->k", np.conj(U[:, :, c]), dHs, U[:, :, v]))
    gam = np.zeros(kx.size)
    for a, b in ((0, 1), (1, 2)):
        occ = np.einsum("ki,ki->k", np.conj(uc[a]), uc[b])
        ovv = np.einsum("ki,ki->k", np.conj(uv[b]), uv[a])
        gam += np.angle(vcv[b] / vcv[a] * occ * ovv)
    return (gam / (2.0 * h)).reshape(shape)
