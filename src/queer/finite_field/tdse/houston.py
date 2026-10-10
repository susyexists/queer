"""Houston (accelerated Bloch state) TDSE along one line k = s e + t (z x e): a per-line validator.

With the paper convention hbar dk/dt = +eE, the Bloch state on the line obeys

    i F d psi/ds = H(s, t) psi,        F = eE [eV/A] = 1e-10 E [V/m],

so for E > 0 the electron enters at s_start and leaves at s_end, and for E < 0 it enters at s_end and
leaves at s_start. All bands of the model are kept, so leakage to remote bands is visible.

Integrator: 4th-order commutator-free Magnus (Blanes & Moan 2006). With Gauss nodes
H_{1,2} = H(s_j + c_{1,2} h), c_{1,2} = 1/2 -+ sqrt(3)/6 and alpha_{1,2} = (3 -+ 2 sqrt 3)/12, a step is

    psi <- exp(-i h (a1 H1 + a2 H2)/|F|) exp(-i h (a2 H1 + a1 H2)/|F|) psi.

The exponents do not depend on E: each is eigendecomposed once and exponentiated for every |E| at once.
A reversed step (E < 0) uses the same two matrices in the opposite order. Step products are reduced
pairwise in chunks (O(log N) vectorized levels; no Python loop over steps).

Step grid (shared by all E): the local step equidistributes the phase, h(s) width(s)/min|F| <= dphi, where
width is the spectral width of H(s) (max - min eigenvalue). Two safeguards are added: h^2 width'(s)/min|F|
<= dphi, where width' is the spectral width of dH/ds (this binds only for very sudden crossings), and at
least 64 steps.

Start/end states: adiabatic eigenstates, or the first-order superadiabatic states

    |n~> = |n> - i F sum_{m != n} |m> <m|dH/ds|n> / (eps_m - eps_n)^2       (F = eE, signed),

Lowdin-orthonormalized. The sign follows from the quasi-static solution of i F d psi/ds = H psi (the
admixture c_m = i F <m|d_s n>/(eps_m - eps_n) with <m|d_s n> = <m|dH/ds|n>/(eps_n - eps_m)). It is the
sign for which the final populations converge with the window: the dressing residual falls from
~F |dH/ds|/eps_g^2 (adiabatic basis) to second order. The opposite sign doubles the adiabatic-basis
error (tests/test_tdse.py checks both).

Outputs: T[E, m, n] = |<m~_end|U|n~_start>|^2 (bands sorted ascending at each end) and the many-body
excitation n_exc = sum_{n <= v, m >= c} T[m, n] = Tr[P_empty U P_filled U^dag]. Band labels are sorted
at each end; across an exact degeneracy (e.g. the k_inv ring of Eq. 9 at VL = 0) use n_exc.

lindblad_two_level adds relaxation for two-band models: amplitude damping c -> v at the rate 1/T1 in
time, i.e. 1/kappa1 = hbar/(e|E| T1) per unit s, and pure dephasing in the instantaneous adiabatic basis.
The coherence decays at 1/(2 T1) + 1/T2 ("paper", as in Eq. 12) or at 1/T2 ("bloch"). The dissipator
and the coherent Magnus step are Strang-split, D(h/2) U D(h/2), in Liouville space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..models.base import field_frame, rotate_derivatives
from ..units import eE, kappa

_SQ3 = math.sqrt(3.0)
_C1, _C2 = 0.5 - _SQ3 / 6.0, 0.5 + _SQ3 / 6.0  # Gauss-Legendre nodes
_A1, _A2 = (3.0 - 2.0 * _SQ3) / 12.0, (3.0 + 2.0 * _SQ3) / 12.0  # CF4 weights
_PRE_GRID = 4097  # points used to equidistribute the step phase
_MIN_STEPS = 64
_MAX_STEPS = 20_000_000
_CHUNK_BYTES = 32e6  # working memory per chunk of steps
_DEG_TOL = 1e-12  # eV; band pairs closer than this get no superadiabatic dressing
SUPERADIABATIC_SIGN = -1  # |n~> = |n> + SIGN * i F sum_m |m><m|dH/ds|n>/(eps_m - eps_n)^2


@dataclass(frozen=True)
class TDSEResult:
    """Transition probabilities of one line; T[E, final, initial] with bands sorted ascending."""

    E: np.ndarray  # (nE,) V/m
    T: np.ndarray  # (nE, n, n)
    n_exc: np.ndarray  # (nE,) many-body excitation sum_{n <= v, m >= c} T[m, n]
    v: int
    c: int
    info: dict = field(default_factory=dict)

    @property
    def P(self) -> np.ndarray:
        """v -> c transition probability T[:, c, v], shape (nE,)."""
        return self.T[:, self.c, self.v]


# ---- line operators and step grid ---------------------------------------------------------------
def _hamiltonian(model, s, t, angle):
    kx, ky = field_frame(s, t, angle)
    return np.asarray(model.hamiltonian(kx, ky), dtype=complex)


def _line_operators(model, s, t, angle):
    """H and dH/ds on the line, shape (len(s), n, n)."""
    kx, ky = field_frame(s, t, angle)
    H = np.asarray(model.hamiltonian(kx, ky), dtype=complex)
    dHx, dHy = model.dhamiltonian(kx, ky)
    dHs, _ = rotate_derivatives(dHx, dHy, angle)
    return H, np.broadcast_to(np.asarray(dHs, dtype=complex), H.shape)


def _step_grid(model, t, angle, s_start, s_end, F_min, dphi, chunk_bytes=None):
    """Step boundaries s_start = s_0 < ... < s_N = s_end equidistributing the phase per step."""
    s_pre = np.linspace(s_start, s_end, _PRE_GRID)
    scale = dphi * F_min
    rho = np.empty_like(s_pre)
    chunk = _chunk_size(1, model.n_orb, chunk_bytes)
    for start in range(0, len(s_pre), chunk):
        stop = min(start + chunk, len(s_pre))
        H, dHs = _line_operators(model, s_pre[start:stop], t, angle)
        w = np.linalg.eigvalsh(H)
        wd = np.linalg.eigvalsh(dHs)
        rho[start:stop] = np.maximum((w[:, -1] - w[:, 0]) / scale, np.sqrt((wd[:, -1] - wd[:, 0]) / scale))
    rho = np.maximum(rho, _MIN_STEPS / (s_end - s_start))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (rho[1:] + rho[:-1]) * np.diff(s_pre))])
    n_steps = max(_MIN_STEPS, math.ceil(cum[-1]))
    if n_steps > _MAX_STEPS:
        raise ValueError(f"{n_steps} steps needed; shrink the window, raise dphi or min|E|")
    s = np.interp(np.linspace(0.0, cum[-1], n_steps + 1), cum, s_pre)
    s[0], s[-1] = s_start, s_end
    return s


def _validate(E, s_start, s_end, dphi, basis):
    E = np.asarray(E, dtype=float).reshape(-1)
    if E.size == 0 or np.any(~np.isfinite(E)) or np.any(E == 0):
        raise ValueError("E must be finite and nonzero")
    if not s_end > s_start:
        raise ValueError("need s_start < s_end (the direction of motion follows the sign of E)")
    if not dphi > 0:
        raise ValueError("dphi must be > 0")
    if basis not in ("adiabatic", "superadiabatic"):
        raise ValueError(f"basis must be 'adiabatic' or 'superadiabatic', got {basis!r}")
    return E


# ---- Magnus steps ---------------------------------------------------------------------------------
def _magnus_eig(model, t, angle, s_lo, h):
    """Eigen-decompositions of the two CF4 exponents of each step (E independent), trace removed."""
    m = s_lo.size
    H = _hamiltonian(model, np.concatenate([s_lo + _C1 * h, s_lo + _C2 * h]), t, angle)
    H1, H2 = H[:m], H[m:]
    l1, V1 = np.linalg.eigh(_A2 * H1 + _A1 * H2)  # applied first in forward steps
    l2, V2 = np.linalg.eigh(_A1 * H1 + _A2 * H2)
    l1 -= l1.mean(axis=-1, keepdims=True)  # a global phase only
    l2 -= l2.mean(axis=-1, keepdims=True)
    return l1, V1, l2, V2, np.conj(np.swapaxes(V2, -1, -2)) @ V1  # last: V2^dag V1


def _step_unitaries(h, eig, Fabs, forward: bool):
    """Step propagators, shape (m, nE, n, n): exp(-ih B2/|F|) exp(-ih B1/|F|) (forward) or reversed."""
    l1, V1, l2, V2, W = eig
    x = h[:, None, None] / Fabs[None, :, None]
    p1 = np.exp(-1j * x * l1[:, None, :])  # (m, nE, n)
    p2 = np.exp(-1j * x * l2[:, None, :])
    if forward:
        mid = W[:, None] * p2[..., :, None] * p1[..., None, :]
        return V2[:, None] @ mid @ np.conj(np.swapaxes(V1, -1, -2))[:, None]
    mid = np.conj(np.swapaxes(W, -1, -2))[:, None] * p1[..., :, None] * p2[..., None, :]
    return V1[:, None] @ mid @ np.conj(np.swapaxes(V2, -1, -2))[:, None]


def _chain(m, reverse: bool = False):
    """Ordered product over axis 0: m[L-1] ... m[1] m[0], or m[0] m[1] ... m[L-1] if reverse."""
    while m.shape[0] > 1:
        tail = m[-1:] if m.shape[0] % 2 else None
        a, b = m[0 : m.shape[0] - (tail is not None) : 2], m[1::2]
        m = a @ b if reverse else b @ a
        if tail is not None:
            m = np.concatenate([m, tail])
    return m[0]


def _chunk_size(n_e: int, d: int, chunk_bytes: float | None = None) -> int:
    """Bound the concurrent step arrays, reserving six complex matrices per field."""
    budget = _CHUNK_BYTES if chunk_bytes is None else chunk_bytes
    if isinstance(budget, bool) or not math.isfinite(budget) or budget <= 0:
        raise ValueError("chunk_bytes must be a positive finite memory budget")
    per_step = 16 * 6 * max(n_e, 1) * d * d
    count = int(budget // per_step)
    if count < 1:
        raise ValueError(f"chunk_bytes is too small for one propagation step ({per_step} bytes)")
    return count


def _propagators(model, t, angle, s, F, chunk_bytes=None):
    """U(F) for every signed force F, shape (nE, n, n), traversing s forward (F > 0) or backward."""
    n = model.n_orb
    pos, neg = F > 0, F < 0
    U = np.empty((F.size, n, n), dtype=complex)
    Up = np.broadcast_to(np.eye(n, dtype=complex), (int(pos.sum()), n, n)).copy()
    Un = np.broadcast_to(np.eye(n, dtype=complex), (int(neg.sum()), n, n)).copy()
    h_all = np.diff(s)
    chunk = _chunk_size(F.size, n, chunk_bytes)
    for j0 in range(0, h_all.size, chunk):
        h = h_all[j0 : j0 + chunk]
        eig = _magnus_eig(model, t, angle, s[j0 : j0 + h.size], h)
        if pos.any():
            Up = _chain(_step_unitaries(h, eig, F[pos], True)) @ Up
        if neg.any():
            Un = Un @ _chain(_step_unitaries(h, eig, -F[neg], False), reverse=True)
    U[pos], U[neg] = Up, Un
    return U


# ---- start and end bases ----------------------------------------------------------------------------
def _endpoint_basis(model, s, t, angle, F, basis: str):
    """Columns |n~> at the points s (nE,) for signed forces F (nE,), shape (nE, n, n), bands ascending.

    Also returns the smallest level spacing and the largest first-order admixture |F C_mn| (the window
    is adequate when it is small; it bounds the adiabatic-basis error of the populations).
    """
    H, dHs = _line_operators(model, s, t, angle)
    eps, N = np.linalg.eigh(H)
    gaps = np.diff(eps, axis=-1)
    min_gap = float(gaps.min()) if gaps.size else math.inf
    n = N.shape[-1]
    X = np.conj(np.swapaxes(N, -1, -2)) @ dHs @ N  # <m|dH/ds|n>
    d = eps[:, :, None] - eps[:, None, :]  # eps_m - eps_n
    ok = np.abs(d) > _DEG_TOL
    FC = F[:, None, None] * np.where(ok, SUPERADIABATIC_SIGN * 1j * X / np.where(ok, d, 1.0) ** 2, 0.0)
    dressing = float(np.max(np.abs(FC)))
    if basis == "adiabatic":
        return N, min_gap, dressing
    Nt = N @ (np.eye(n) + FC)
    S = np.conj(np.swapaxes(Nt, -1, -2)) @ Nt  # Lowdin: Nt S^{-1/2}
    w, Q = np.linalg.eigh(S)
    return Nt @ (Q * w[:, None, :] ** -0.5) @ np.conj(np.swapaxes(Q, -1, -2)), min_gap, dressing


def _endpoints(model, t, angle, s_start, s_end, F, basis):
    s_a = np.where(F > 0, s_start, s_end)  # entry point
    s_b = np.where(F > 0, s_end, s_start)  # exit point
    Ba, ga, da = _endpoint_basis(model, s_a, t, angle, F, basis)
    Bb, gb, db = _endpoint_basis(model, s_b, t, angle, F, basis)
    return Ba, Bb, {"endpoint_min_gap": min(ga, gb), "endpoint_dressing": max(da, db)}


def houston_line(
    model,
    t,
    E,
    s_start,
    s_end,
    angle: float = 0.0,
    dphi: float = 0.02,
    basis: str = "superadiabatic",
    bands: tuple[int, int] | None = None,
    *,
    chunk_bytes: float | None = None,
) -> TDSEResult:
    """Transition probabilities of the Houston TDSE on the line k = s e + t (z x e), s in [s_start, s_end].

    t in 1/A, angle (rad) is the field direction e in the model frame, E (V/m, scalar or 1-D, either sign;
    all |E| share the step grid, set by min|E|). Bands (v, c) default to model.band_pair().
    """
    E = _validate(E, s_start, s_end, dphi, basis)
    _chunk_size(E.size, model.n_orb, chunk_bytes)
    v, c = bands if bands is not None else model.band_pair()
    F = eE(E)
    s = _step_grid(model, t, angle, s_start, s_end, float(np.min(np.abs(F))), dphi, chunk_bytes)
    U = _propagators(model, t, angle, s, F, chunk_bytes)
    Ba, Bb, ends = _endpoints(model, t, angle, s_start, s_end, F, basis)
    A = np.conj(np.swapaxes(Bb, -1, -2)) @ U @ Ba
    T = np.abs(A) ** 2
    n = model.n_orb
    unitarity = float(np.max(np.abs(np.conj(np.swapaxes(U, -1, -2)) @ U - np.eye(n))))
    info = {
        "steps": s.size - 1,
        "window": (float(s_start), float(s_end)),
        "basis": basis,
        "dphi": dphi,
        "t": float(t),
        "angle": float(angle),
        "h_min": float(np.diff(s).min()),
        "h_max": float(np.diff(s).max()),
        "unitarity": unitarity,
        **ends,
        "superadiabatic_sign": SUPERADIABATIC_SIGN,
        "chunk_bytes": _CHUNK_BYTES if chunk_bytes is None else chunk_bytes,
    }
    return TDSEResult(E, T, T[:, c:, : v + 1].sum(axis=(1, 2)), v, c, info)


# ---- two-band Lindblad --------------------------------------------------------------------------------
def _kron(A, B):
    """Batched Kronecker product of square matrices (..., d, d) x (..., d, d)."""
    d = A.shape[-1]
    return (A[..., :, None, :, None] * B[..., None, :, None, :]).reshape(A.shape[:-2] + (d * d, d * d))


def _dissipators(Vb, tau, g1, g2):
    """Liouville-space relaxation over lengths tau (m,) in the eigenbases Vb (m, 2, 2) [v, c columns].

    In the adiabatic basis (vec order vv, vc, cv, cc): rho_cc *= e1, rho_vv += (1 - e1) rho_cc,
    coherences *= e2, with e1 = exp(-g1 tau), e2 = exp(-g2 tau); g1, g2 (nE,) per unit s.
    """
    e1 = np.exp(-tau[:, None] * g1[None, :])  # (m, nE)
    e2 = np.exp(-tau[:, None] * g2[None, :])
    D = np.zeros(e1.shape + (4, 4))
    D[..., 0, 0] = 1.0
    D[..., 0, 3] = 1.0 - e1
    D[..., 1, 1] = D[..., 2, 2] = e2
    D[..., 3, 3] = e1
    Vk = _kron(Vb, np.conj(Vb))[:, None]  # rho -> V rho V^dag
    return Vk @ D @ np.conj(np.swapaxes(Vk, -1, -2))


def lindblad_two_level(
    model,
    t,
    E,
    s_start,
    s_end,
    T1,
    T2,
    angle: float = 0.0,
    dphi: float = 0.02,
    coherence: str = "paper",
    basis: str = "superadiabatic",
    *,
    chunk_bytes: float | None = None,
) -> np.ndarray:
    """Final conduction-band population rho_cc (nE,) of a two-band model with T1/T2 relaxation.

    The electron enters in the valence state (`basis` as in houston_line) and rho_cc is measured in the
    same basis at the exit. T1, T2 in s (inf allowed). coherence: "paper" (rho_cv decays at
    1/(2 T1) + 1/T2, pure dephasing 1/T2) or "bloch" (rho_cv decays at 1/T2; needs T2 <= 2 T1).
    """
    if model.n_orb != 2:
        raise ValueError("lindblad_two_level needs a two-band model")
    E = _validate(E, s_start, s_end, dphi, basis)
    chunk = _chunk_size(E.size, 4, chunk_bytes)
    if not (T1 > 0 and T2 > 0):
        raise ValueError("T1 and T2 must be > 0 (inf allowed)")
    F = eE(E)
    g1 = 1.0 / kappa(E, T1)  # amplitude damping per unit s
    if coherence == "paper":
        g2 = 0.5 * g1 + 1.0 / kappa(E, T2)
    elif coherence == "bloch":
        if T2 > 2.0 * T1:
            raise ValueError("coherence='bloch' requires T2 <= 2 T1")
        g2 = 1.0 / kappa(E, T2)
    else:
        raise ValueError(f"coherence must be 'paper' or 'bloch', got {coherence!r}")

    s = _step_grid(model, t, angle, s_start, s_end, float(np.min(np.abs(F))), dphi, chunk_bytes)
    h_all = np.diff(s)
    tau = 0.5 * (np.concatenate([[0.0], h_all]) + np.concatenate([h_all, [0.0]]))  # per boundary
    pos, neg = F > 0, F < 0
    Mp = np.broadcast_to(np.eye(4, dtype=complex), (int(pos.sum()), 4, 4)).copy()
    Mn = np.broadcast_to(np.eye(4, dtype=complex), (int(neg.sum()), 4, 4)).copy()
    for j0 in range(0, h_all.size, chunk):
        h = h_all[j0 : j0 + chunk]
        eig = _magnus_eig(model, t, angle, s[j0 : j0 + h.size], h)
        _, Vb = np.linalg.eigh(_hamiltonian(model, s[j0 : j0 + h.size], t, angle))
        for sel, forward in ((pos, True), (neg, False)):
            if not sel.any():
                continue
            U = _step_unitaries(h, eig, np.abs(F[sel]), forward)
            K = _kron(U, np.conj(U))
            D = _dissipators(Vb, tau[j0 : j0 + h.size], g1[sel], g2[sel])
            if forward:  # ... K_j D_j ... : relax over the half steps around s_j, then step j
                Mp = _chain(K @ D) @ Mp
            else:  # ... D_j K_j^- ...: step j backwards, then relax around s_j
                Mn = Mn @ _chain(D @ K, reverse=True)
    _, Vend = np.linalg.eigh(_hamiltonian(model, s[-1:], t, angle))
    for sel, forward in ((pos, True), (neg, False)):  # relaxation over the half step at s_end
        if sel.any():
            D = _dissipators(Vend, tau[-1:], g1[sel], g2[sel])[0]
            if forward:
                Mp = D @ Mp
            else:
                Mn = Mn @ D
    M = np.empty((F.size, 4, 4), dtype=complex)
    M[pos], M[neg] = Mp, Mn

    Ba, Bb, _ = _endpoints(model, t, angle, s_start, s_end, F, basis)
    v, c = model.band_pair()
    psi = Ba[:, :, v]
    rho = (M @ (psi[:, :, None] * np.conj(psi[:, None, :])).reshape(-1, 4, 1)).reshape(-1, 2, 2)
    cvec = Bb[:, :, c]
    return np.real(np.einsum("ei,eij,ej->e", np.conj(cvec), rho, cvec))
