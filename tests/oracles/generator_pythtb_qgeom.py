"""Quantum geometry of a 2D Wannier model: the numpy kernel of the PYTHTB_QGEOM step.

Numpy only at import, like ``pythtb_core``: the backend and its CI import it
without PythTB, and the step stages it byte for byte next to the driver as
``_pythtb_qgeom.py``.

Conventions (``docs/ENGINE-PYTHTB.md`` carries the derivations):

- Bloch sums in wannier90's convention II on the Wigner-Seitz-corrected,
  folded table, E_F already subtracted: H(k) = sum_R e^{ik.R} H(R).
- k is Cartesian in rad/Angstrom (2 pi included) and hbar = 1 inside band
  formulas, so v = dE/dk is in eV Angstrom, positions and A in Angstrom, g and
  Omega in Angstrom^2, G in Angstrom^2/eV, and [dk] = d^2k/(2 pi)^2.
- Vector components are taken in the layer frame (e1, e2): e1 along a1
  projected on the layer plane, e2 = n x e1, n along a1 x a2.
- Positions: A^W_a(k) = sum_R e^{ik.R} r_a(R) with r_a,ij(R) = <0i|r_a|Rj>, and
  Omega^W(k) = sum_R e^{ik.R} i(R_1 r_2(R) - R_2 r_1(R)) (Wang, Yates, Souza,
  Vanderbilt, PRB 74, 195118 (2006), Eqs. 25 and 37). The diagonal level is
  r_a(R) = diag(tau_a) delta_R0, which is PythTB's tight-binding model.
- In the eigenbasis H = U E U^dag: V_a = U^dag dH_a U, Abar_a = U^dag A^W_a U,
  and A^a_nm = Abar^a_nm + i V^a_nm / (E_m - E_n) between bands of different
  degenerate clusters.
- Omega = -2 Im Q (PythTB, Xiao-Chang-Niu); carriers have charge -e.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# CODATA 2018 exact values.
E_CHARGE = 1.602176634e-19  # C
H_PLANCK = 6.62607015e-34  # J s
HBAR = H_PLANCK / (2.0 * math.pi)
K_B_EV = 8.617333262e-5  # eV / K

E2_OVER_HBAR_S = E_CHARGE**2 / HBAR  # 2.434135e-4 S
E2_OVER_H_S = E_CHARGE**2 / H_PLANCK  # 3.874046e-5 S
# (e^3/hbar) times a dipole of 1 Angstrom/eV, in A m / V^2: the eV in the
# denominator cancels one e, the Angstrom is 1e-10 m.
CHI_SI_PER_A_PER_EV = E2_OVER_HBAR_S * 1e-10
# The same in mA nm / V^2, the unit the results and the UI quote.
CHI_MA_NM_PER_V2_PER_A_PER_EV = CHI_SI_PER_A_PER_EV * 1e12

# The published (c_out, c_in) coefficient sets of the intrinsic second-order
# response chi_abc = -(e^3/hbar)[c_out M_abc + c_in (M_bac + M_cab)].
CONVENTION_SETS: Dict[str, Tuple[float, float]] = {
    # wave-packet dynamics: Gao, Yang, Niu PRL 112, 166601 (2014)
    "wpd_1_m05": (1.0, -0.5),
    # Luttinger-Kohn: Kaplan, Holder, Yan, Eq. 12c
    "lka_2_m05": (2.0, -0.5),
    # quantum kinetics: Das et al. PRB 108, L201405 (2023)
    "qk_05_m1": (0.5, -1.0),
}

# Partner-gap bins (meV) of the BCP sources: <1, 1-10, 10-100, >=100.
GAP_BIN_EDGES_MEV: Tuple[float, ...] = (1.0, 10.0, 100.0)

LEVEL_FULL = "full"
LEVEL_DIAGONAL = "diagonal"


# ----------------------------------------------------------------- geometry
def layer_frame(lattice: Sequence[Sequence[float]]) -> Dict[str, np.ndarray]:
    """The in-plane frame (e1, e2) and the layer normal n of a 2D cell.

    n is along a1 x a2, e1 along a1 projected on the layer plane, e2 = n x e1.
    For a cell with a1 along x and a3 along z this is the lab frame.
    """
    lat = np.asarray(lattice, float).reshape(3, 3)
    normal = np.cross(lat[0], lat[1])
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        raise ValueError("a1 and a2 are parallel")
    normal = normal / norm
    e1 = lat[0] - (lat[0] @ normal) * normal
    e1 = e1 / float(np.linalg.norm(e1))
    e2 = np.cross(normal, e1)
    return {"e1": e1, "e2": e2, "normal": normal}


def a3_misalignment(lattice: Sequence[Sequence[float]]) -> float:
    """|a3_hat x n_hat|: zero when a3 is along the layer normal."""
    lat = np.asarray(lattice, float).reshape(3, 3)
    normal = np.cross(lat[0], lat[1])
    a3 = lat[2] / float(np.linalg.norm(lat[2]))
    return float(np.linalg.norm(np.cross(a3, normal / float(np.linalg.norm(normal)))))


def cell_area(lattice: Sequence[Sequence[float]]) -> float:
    """|a1 x a2| in Angstrom^2."""
    lat = np.asarray(lattice, float).reshape(3, 3)
    return float(np.linalg.norm(np.cross(lat[0], lat[1])))


def zone_area(lattice: Sequence[Sequence[float]]) -> float:
    """The in-plane Brillouin-zone area (2 pi)^2 / A_cell, in Angstrom^-2."""
    return (2.0 * math.pi) ** 2 / cell_area(lattice)


def reciprocal_in_plane(
    lattice: Sequence[Sequence[float]], frame: Dict[str, np.ndarray]
) -> np.ndarray:
    """b1, b2 (2 pi included) as rows of frame components (2, 2)."""
    lat = np.asarray(lattice, float).reshape(3, 3)
    recip = 2.0 * math.pi * np.linalg.inv(lat).T
    return np.stack([recip[:2] @ frame["e1"], recip[:2] @ frame["e2"]], axis=1)


def kcart(kfrac: np.ndarray, binv: np.ndarray) -> np.ndarray:
    """Frame components of k (rad/Angstrom) from in-plane fractional k (nk, 2)."""
    return np.asarray(kfrac, float)[:, :2] @ binv


def is_hexagonal(lattice: Sequence[Sequence[float]], tol: float = 1e-4) -> bool:
    """|a1| = |a2| with a 60 or 120 degree angle."""
    lat = np.asarray(lattice, float).reshape(3, 3)
    a, b = float(np.linalg.norm(lat[0])), float(np.linalg.norm(lat[1]))
    if abs(a - b) > tol * a:
        return False
    cosang = float(lat[0] @ lat[1]) / (a * b)
    return abs(abs(cosang) - 0.5) < tol


def c3_rotate_frac(kfrac: np.ndarray, lattice: Sequence[Sequence[float]]) -> np.ndarray:
    """Fractional k rotated by 120 degrees about the layer normal."""
    lat = np.asarray(lattice, float).reshape(3, 3)
    frame = layer_frame(lat)
    b = reciprocal_in_plane(lat, frame)  # rows b1, b2 in the frame
    c, s = math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3)
    rot = np.array([[c, -s], [s, c]])
    kc = np.asarray(kfrac, float)[:, :2] @ b
    out = (kc @ rot.T) @ np.linalg.inv(b)
    return np.concatenate([out, np.zeros((len(out), 1))], axis=1)


# ----------------------------------------------------------------- the table
class Table:
    """One level's stacked Fourier table.

    Per R, in this order: H, i R_1 H, i R_2 H and, at the full level, r_1, r_2
    and i (R_1 r_2 - R_2 r_1). One GEMM per chunk of k gives every matrix.
    The full level's r must sit on the same R set as H (unfold them together).
    """

    def __init__(
        self,
        R: np.ndarray,
        H: np.ndarray,
        lattice: Sequence[Sequence[float]],
        frame: Dict[str, np.ndarray],
        *,
        tau: Optional[np.ndarray] = None,
        r: Optional[np.ndarray] = None,
    ):
        self.R = np.asarray(R, int).reshape(-1, 3)
        H = np.asarray(H, complex)
        self.lattice = np.asarray(lattice, float).reshape(3, 3)
        self.frame = frame
        self.n = int(H.shape[1])
        Rc = self.R @ self.lattice
        self.R12 = np.stack([Rc @ frame["e1"], Rc @ frame["e2"]], axis=1)
        r1 = self.R12[:, 0, None, None]
        r2 = self.R12[:, 1, None, None]
        blocks = [H, 1j * r1 * H, 1j * r2 * H]
        self.level = LEVEL_DIAGONAL
        self.tau = None
        if r is not None:
            r = np.asarray(r, complex)
            if r.shape != H.shape + (2,):
                raise ValueError(f"r has shape {r.shape}, expected {H.shape + (2,)}")
            blocks += [r[..., 0], r[..., 1], 1j * (r1 * r[..., 1] - r2 * r[..., 0])]
            self.level = LEVEL_FULL
        else:
            if tau is None:
                raise ValueError("the diagonal level needs the centres tau (N, 2)")
            self.tau = np.asarray(tau, float).reshape(self.n, 2)
        self.T = len(blocks)
        self.stack = np.stack(blocks, axis=1).reshape(
            len(self.R), self.T * self.n * self.n
        )
        self.binv = reciprocal_in_plane(self.lattice, frame)

    @property
    def nbytes(self) -> int:
        return int(self.stack.nbytes)

    def energies(self, kfrac: np.ndarray) -> np.ndarray:
        """Eigenvalues (nk, N) at fractional k: the H block of the stack only."""
        kfrac = np.asarray(kfrac, float).reshape(-1, 3)
        phases = np.exp(2j * math.pi * (kfrac @ self.R.T))
        nn = self.n * self.n
        rows = self.stack.reshape(len(self.R), self.T, nn)[:, 0, :]
        Hk = (phases @ rows).reshape(len(kfrac), self.n, self.n)
        return np.linalg.eigvalsh(0.5 * (Hk + np.conj(np.swapaxes(Hk, -1, -2))))

    def fourier(self, kfrac: np.ndarray) -> np.ndarray:
        """(nk, T, N, N) Hermitized matrices at fractional k (nk, 3) with k3 = 0."""
        kfrac = np.asarray(kfrac, float).reshape(-1, 3)
        phases = np.exp(2j * math.pi * (kfrac @ self.R.T))
        out = (phases @ self.stack).reshape(len(kfrac), self.T, self.n, self.n)
        return 0.5 * (out + np.conj(np.swapaxes(out, -1, -2)))


def degenerate_clusters(E: np.ndarray, tol: float) -> np.ndarray:
    """Cluster label per band (nk, N): consecutive bands closer than tol share one."""
    E = np.asarray(E, float)
    gaps = np.diff(E, axis=-1) > tol
    lead = np.zeros(E.shape[:-1] + (1,), int)
    return np.concatenate([lead, np.cumsum(gaps, axis=-1)], axis=-1)


def _rows(U: np.ndarray, M: np.ndarray, rows: slice) -> np.ndarray:
    """(U^dag M U)[rows, :] for every k: (nk, |rows|, N)."""
    return np.matmul(np.conj(np.swapaxes(U[:, :, rows], -1, -2)), np.matmul(M, U))


def _rows_diag(U: np.ndarray, tau_a: np.ndarray, rows: slice) -> np.ndarray:
    """(U^dag diag(tau_a) U)[rows, :]."""
    left = np.conj(np.swapaxes(U[:, :, rows], -1, -2)) * tau_a[None, None, :]
    return np.matmul(left, U)


def _diag_sandwich(U: np.ndarray, M: np.ndarray, rows: slice) -> np.ndarray:
    """diag(U^dag M U)[rows] for every k: (nk, |rows|)."""
    Ur = U[:, :, rows]
    return np.real(np.einsum("kin,kij,kjn->kn", np.conj(Ur), M, Ur))


def evaluate(table: Table, kfrac: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
    """``geometry`` at fractional k (nk, 3) of a table's level."""
    F = table.fourier(kfrac)
    if table.level == LEVEL_FULL:
        return geometry(
            F[:, 0], (F[:, 1], F[:, 2]), AW=(F[:, 3], F[:, 4]), OW=F[:, 5], **kwargs
        )
    return geometry(F[:, 0], (F[:, 1], F[:, 2]), tau=table.tau, **kwargs)


def geometry(
    H: np.ndarray,
    dH: Sequence[np.ndarray],
    *,
    AW: Optional[Sequence[np.ndarray]] = None,
    OW: Optional[np.ndarray] = None,
    tau: Optional[np.ndarray] = None,
    window: Tuple[int, int],
    occupied: Iterable[int] = (),
    deg_tol: float = 1e-6,
    gap_bins: bool = False,
    frozen: Optional[Tuple[float, float]] = None,
    eig: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Band and manifold geometry from Hermitian H(k) (nk, N, N) and its derivatives.

    Positions: ``AW`` = (A^W_1, A^W_2) and ``OW`` = Omega^W at the full level,
    or the centres ``tau`` (N, 2) at the diagonal level; neither means none.
    ``window`` = (lo, hi): the bands n in [lo, hi) get per-band velocity v (nk,
    nW, 2), in-W metric g and BCP G (nk, nW, 2, 2) and the complete curvature
    omega (nk, nW), each summing over every partner m outside n's degenerate
    cluster (``deg_tol``, eV). ``occupied`` lists manifold cuts c: the bands
    [0, c) get their in-W tensor Q_in and the complete curvature, under
    ``blocks[c]``. ``gap_bins`` splits G by partner gap (GAP_BIN_EDGES_MEV);
    ``frozen`` = (lo, hi) eV adds the share from partners outside that window.
    ``eig`` = (E, U) reuses an eigensystem of H (two levels share one).
    """
    lo, hi = int(window[0]), int(window[1])
    n_bands = int(np.shape(H)[-1])
    cuts = sorted({int(c) for c in occupied if 0 < int(c) < n_bands})
    E, U = np.linalg.eigh(H) if eig is None else eig
    nk, n = E.shape
    top = max([hi] + cuts)
    rows = slice(0, top)
    V = [_rows(U, dH[0], rows), _rows(U, dH[1], rows)]
    omega_w = None
    if AW is not None:
        Ab = [_rows(U, AW[0], rows), _rows(U, AW[1], rows)]
        if OW is not None:
            omega_w = _diag_sandwich(U, OW, rows)
    else:
        centres = (
            np.zeros((n, 2)) if tau is None else np.asarray(tau, float).reshape(n, 2)
        )
        Ab = [_rows_diag(U, centres[:, 0], rows), _rows_diag(U, centres[:, 1], rows)]
    cid = degenerate_clusters(E, deg_tol)
    out: Dict[str, Any] = {"E": E, "cluster": cid}

    # Omega-bar' diagonal: [U^dag Omega^W U]_nn + 2 Im sum_m Abar_x,nm conj(Abar_y,nm).
    # Exactly zero at the diagonal level, where diag(tau_x) and diag(tau_y) commute.
    if AW is not None:
        bar = 2.0 * np.imag(np.sum(Ab[0] * np.conj(Ab[1]), axis=-1))
        if omega_w is not None:
            bar = bar + omega_w
    else:
        bar = np.zeros((nk, top))

    # Per-band window quantities.
    w = slice(lo, hi)
    Ew = E[:, w]
    same = cid[:, w, None] == cid[:, None, :]
    dE = E[:, None, :] - Ew[:, :, None]  # [k, n, m] = E_m - E_n
    inv = np.where(same, 0.0, 1.0 / np.where(same, 1.0, dE))
    A = [np.where(same, 0.0, Ab[a][:, w] + 1j * V[a][:, w] * inv) for a in range(2)]
    pair = np.empty((nk, hi - lo, n, 2, 2), complex)
    for a in range(2):
        for b in range(2):
            pair[..., a, b] = A[a] * np.conj(A[b])  # A^a_nm A^b_mn
    out["v"] = np.stack(
        [np.real(np.diagonal(V[a][:, w, w], axis1=1, axis2=2)) for a in range(2)],
        axis=-1,
    )
    # |V_{n, n+1}|: the interband speed of each window band and the band above it.
    nxt = np.minimum(np.arange(lo, hi) + 1, n - 1)
    out["v_pair"] = np.sqrt(
        sum(
            np.abs(np.take_along_axis(V[a][:, w], nxt[None, :, None], axis=2)[..., 0])
            ** 2
            for a in range(2)
        )
    )
    out["g"] = np.real(pair.sum(axis=2))
    out["omega_in"] = -2.0 * np.imag(pair[..., 0, 1].sum(axis=2))
    out["omega_bar"] = bar[:, w]
    out["omega"] = out["omega_in"] + out["omega_bar"]
    gw = -inv  # 1 / (E_n - E_m), zero inside the cluster
    contrib = 2.0 * np.real(pair) * gw[..., None, None]
    out["G"] = contrib.sum(axis=2)
    if gap_bins:
        gap_mev = np.abs(dE) * 1e3
        edges = (0.0,) + GAP_BIN_EDGES_MEV + (np.inf,)
        bins = []
        for low, high in zip(edges[:-1], edges[1:]):
            sel = (~same) & (gap_mev >= low) & (gap_mev < high)
            bins.append(np.where(sel[..., None, None], contrib, 0.0).sum(axis=2))
        out["G_bins"] = np.stack(bins, axis=-1)
        if frozen is not None:
            outside = (E < frozen[0]) | (E > frozen[1])
            sel = (~same) & outside[:, None, :]
            out["G_outside_frozen"] = np.where(sel[..., None, None], contrib, 0.0).sum(
                axis=2
            )

    # Manifold blocks [0, c).
    blocks: Dict[int, Dict[str, np.ndarray]] = {}
    for c in cuts:
        S = slice(0, c)
        C = slice(c, n)
        dSC = E[:, None, C] - E[:, S, None]  # E_m - E_n, n in S, m above
        # A closed gap at the cut makes the block inapplicable there (the caller
        # judges ``gap``); never divide by it.
        safe = np.abs(dSC) > 1e-12
        invSC = np.where(safe, 1.0 / np.where(safe, dSC, 1.0), 0.0)
        ASC = [Ab[a][:, S, C] + 1j * V[a][:, S, C] * invSC for a in range(2)]
        Q = np.empty((nk, 2, 2), complex)
        for a in range(2):
            for b in range(2):
                Q[:, a, b] = np.sum(ASC[a] * np.conj(ASC[b]), axis=(1, 2))
        omega_in = -2.0 * np.imag(Q[:, 0, 1])
        omega_bar = bar[:, S].sum(axis=1)
        trace = np.real(Q[:, 0, 0] + Q[:, 1, 1])
        lam = np.linalg.eigvalsh(0.5 * (Q + np.conj(np.swapaxes(Q, -1, -2))))
        blocks[c] = {
            "g_in": np.real(Q),
            "omega_in": omega_in,
            "omega": omega_in + omega_bar,
            "gap": E[:, c] - E[:, c - 1],
            "v_cut": np.sqrt(sum(np.abs(V[a][:, c - 1, c]) ** 2 for a in range(2))),
            "psd_min_rel": lam[:, 0] / np.maximum(trace, 1e-300),
        }
    out["blocks"] = blocks
    return out


def sum_rule_residual(table: Table, kfrac: np.ndarray, deg_tol: float = 1e-6) -> float:
    """max |sum_n G^n| over all bands, relative to max |G^n|: zero exactly."""
    res = evaluate(table, kfrac, window=(0, table.n), deg_tol=deg_tol)
    total = np.abs(res["G"].sum(axis=1)).max()
    scale = np.abs(res["G"]).max()
    return float(total / scale) if scale > 0 else 0.0


# ----------------------------------------------------------------- uniform grid
def uniform_grid(n: int) -> np.ndarray:
    """A Gamma-centred n x n grid of fractional k (n^2, 3) on [-1/2, 1/2)."""
    f = (np.arange(n) - n // 2) / float(n)
    F1, F2 = np.meshgrid(f, f, indexing="ij")
    return np.stack([F1.ravel(), F2.ravel(), np.zeros(n * n)], axis=1)


def fermi(x: np.ndarray) -> np.ndarray:
    """1 / (e^x + 1), overflow-safe."""
    x = np.clip(np.asarray(x, float), -700.0, 700.0)
    return 0.5 * (1.0 - np.tanh(0.5 * x))


def minus_fprime(e: np.ndarray, kT: float) -> np.ndarray:
    """-df/dE at energy e (eV, relative to mu) and kT (eV)."""
    x = np.clip(np.asarray(e, float) / kT, -700.0, 700.0)
    return 0.25 / kT / np.cosh(0.5 * x) ** 2


def evaluate_chunks(
    table: Table, kfrac: np.ndarray, chunk: int = 256, **kwargs: Any
) -> Dict[str, Any]:
    """``evaluate`` over chunks, concatenated."""
    parts = [
        evaluate(table, kfrac[s : s + chunk], **kwargs)
        for s in range(0, len(kfrac), chunk)
    ]
    out: Dict[str, Any] = {}
    for key in parts[0]:
        if key == "blocks":
            out["blocks"] = {
                c: {
                    k: np.concatenate([p["blocks"][c][k] for p in parts])
                    for k in parts[0]["blocks"][c]
                }
                for c in parts[0]["blocks"]
            }
        else:
            out[key] = np.concatenate([p[key] for p in parts])
    return out


def uniform_response(
    table: Table,
    n: int,
    *,
    window: Tuple[int, int],
    mus: Sequence[float],
    kT: float = 0.0,
    occupied: Iterable[int] = (),
    deg_tol: float = 1e-6,
    chunk: int = 256,
) -> Dict[str, Any]:
    """Zone integrals on the Gamma-centred n x n grid, direct weights.

    kT = 0: Fermi-sea sums with theta(mu - E) (gapped mu only); kT > 0: -f'
    weights. Returns per mu: M (2, 2, 2) in Angstrom/eV, BCD (2) in Angstrom,
    sigma_xy_e2_h of the window bands plus the always-occupied bands [0, lo)
    (included when lo > 0), DOS (states/eV/cell), the linear Drude weight (2, 2)
    in eV; and per manifold cut its C, C_in and K_in.
    """
    k = uniform_grid(n)
    lo, hi = int(window[0]), int(window[1])
    cuts = set(int(c) for c in occupied)
    if lo > 0:
        cuts.add(lo)
    res = evaluate_chunks(
        table, k, chunk=chunk, window=window, occupied=sorted(cuts), deg_tol=deg_tol
    )
    area = zone_area(table.lattice)
    weight = area / (2.0 * math.pi) ** 2 / len(k)  # [dk] per point
    E = res["E"][:, lo:hi]
    mus = np.asarray(mus, float)
    out: Dict[str, Any] = {
        "mu": mus,
        "M": [],
        "BCD": [],
        "sigma_xy_e2_h": [],
        "dos": [],
        "drude": [],
        "count": [],
    }
    base_omega = res["blocks"][lo]["omega"].sum() if lo > 0 else 0.0
    for mu in mus:
        if kT > 0:
            wfs = minus_fprime(E - mu, kT)
            occ = fermi((E - mu) / kT)
        else:
            wfs = np.zeros_like(E)
            occ = (E < mu).astype(float)
        M = np.einsum("kn,kna,knbc->abc", wfs, res["v"], res["G"]) * weight
        D = np.einsum("kn,kna,kn->a", wfs, res["v"], res["omega"]) * weight
        sea = np.sum(occ * res["omega"]) + base_omega
        out["M"].append(M)
        out["BCD"].append(D)
        # sigma_xy = -(e^2/h)(1/2 pi) sum int f Omega d^2k; d^2k per point = area / nk
        out["sigma_xy_e2_h"].append(-sea * area / len(k) / (2.0 * math.pi))
        out["dos"].append(np.sum(wfs) / len(k))
        out["drude"].append(
            np.einsum("kn,kna,knb->ab", wfs, res["v"], res["v"]) * weight
        )
        out["count"].append(lo + np.sum(occ) / len(k))
    for key in ("M", "BCD", "sigma_xy_e2_h", "dos", "drude", "count"):
        out[key] = np.asarray(out[key])
    out["manifold"] = {
        c: {
            "C": float(
                res["blocks"][c]["omega"].sum() * area / len(k) / (2.0 * math.pi)
            ),
            "C_in": float(
                res["blocks"][c]["omega_in"].sum() * area / len(k) / (2.0 * math.pi)
            ),
            "K_in": res["blocks"][c]["g_in"].sum(axis=0)
            * area
            / len(k)
            / (2.0 * math.pi),
            "psd_min_rel": float(res["blocks"][c]["psd_min_rel"].min()),
            "min_gap": float(res["blocks"][c]["gap"].min()),
        }
        for c in res["blocks"]
    }
    return out


# ----------------------------------------------------------------- tensors
def chi_from_dipole(M: np.ndarray, c_out: float, c_in: float) -> np.ndarray:
    """chi_abc / (e^3/hbar) = -[c_out M_abc + c_in (M_bac + M_cab)], from M (..., 2, 2, 2)."""
    M = np.asarray(M, float)
    # M_bac swaps the first two indices; M_cab moves the first index last.
    return -(c_out * M + c_in * (np.swapaxes(M, -3, -2) + np.moveaxis(M, -3, -1)))


def chi_all_conventions(M: np.ndarray) -> Dict[str, np.ndarray]:
    """chi in mA nm / V^2 for every published (c_out, c_in) set."""
    return {
        name: chi_from_dipole(M, c_out, c_in) * CHI_MA_NM_PER_V2_PER_A_PER_EV
        for name, (c_out, c_in) in CONVENTION_SETS.items()
    }


def trigonal(M: np.ndarray) -> Dict[str, Any]:
    """The m = 3 part of an in-plane rank-3 tensor symmetric in (b, c).

    Z = sum M_abc zeta_a zeta_b zeta_c with zeta = (1, i) gives alpha + i beta = Z/4:
    alpha = (M_xxx - M_xyy - 2 M_yxy)/4, beta = (M_yxx + 2 M_xxy - M_yyy)/4,
    A3 = |alpha + i beta|, theta0 = atan2(beta, alpha)/3 (degrees). Rotating M
    by phi about the normal multiplies alpha + i beta by e^{3 i phi}.
    """
    M = np.asarray(M, float)
    alpha = (M[..., 0, 0, 0] - M[..., 0, 1, 1] - 2.0 * M[..., 1, 0, 1]) / 4.0
    beta = (M[..., 1, 0, 0] + 2.0 * M[..., 0, 0, 1] - M[..., 1, 1, 1]) / 4.0
    A3 = np.hypot(alpha, beta)
    theta0 = np.degrees(np.arctan2(beta, alpha)) / 3.0
    return {"alpha": alpha, "beta": beta, "A3": A3, "theta0_deg": theta0}


def trigonal_tensor(alpha: float, beta: float) -> np.ndarray:
    """The fully symmetric m = 3 tensor of amplitude (alpha, beta)."""
    T = np.zeros((2, 2, 2))
    T[0, 0, 0] = alpha
    for idx in ((0, 1, 1), (1, 0, 1), (1, 1, 0)):
        T[idx] = -alpha
    for idx in ((0, 0, 1), (0, 1, 0), (1, 0, 0)):
        T[idx] = beta
    T[1, 1, 1] = -beta
    return T


def rotate_tensor(M: np.ndarray, phi: float) -> np.ndarray:
    """M_abc rotated by phi (radians) about the normal."""
    c, s = math.cos(phi), math.sin(phi)
    R = np.array([[c, -s], [s, c]])
    return np.einsum("ai,bj,ck,ijk->abc", R, R, R, np.asarray(M, float))


def c3_residual_tensor(M: np.ndarray) -> float:
    """|M - its m = 3 part| / |M| (Frobenius): zero for a C3-symmetric M."""
    M = np.asarray(M, float)
    norm = float(np.linalg.norm(M))
    if norm == 0.0:
        return 0.0
    t = trigonal(M)
    return float(
        np.linalg.norm(M - trigonal_tensor(float(t["alpha"]), float(t["beta"]))) / norm
    )


def c3_residual_matrix(K: np.ndarray) -> float:
    """|K - (tr K / 2) 1| / |K|: zero for a C3-symmetric rank-2 tensor."""
    K = np.asarray(K, float)
    norm = float(np.linalg.norm(K))
    if norm == 0.0:
        return 0.0
    return float(np.linalg.norm(K - 0.5 * np.trace(K) * np.eye(2)) / norm)


# ----------------------------------------------------------------- output
def _round_sig(value: float, digits: int) -> Optional[float]:
    if not math.isfinite(value):
        return None
    if value == 0.0:
        return 0.0
    return float(f"{value:.{digits}g}")


def compact(value: Any, digits: int = 5) -> Any:
    """JSON-ready copy with floats to ``digits`` significant digits, NaN/inf -> None."""
    if isinstance(value, dict):
        return {str(k): compact(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [compact(v, digits) for v in value]
    if isinstance(value, np.ndarray):
        return compact(value.tolist(), digits)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _round_sig(float(value), digits)
    return value


def write_json_compact(path: str, payload: Any, digits: int = 5) -> int:
    """Write ``payload`` with no indentation and ``digits`` significant digits.

    Returns the size in bytes. Significant digits, not decimal places: these
    curves span decades. Written to a temporary name and renamed.
    """
    text = json.dumps(compact(payload, digits), separators=(",", ":"), allow_nan=False)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as handle:
        handle.write(text)
    os.replace(tmp, path)
    return len(text.encode())


def decimate(curves: Any, stride: int, length: int) -> Any:
    """Every array or list whose leading axis is the mu grid (``length``), taken
    at ``stride``; everything else unchanged."""
    if isinstance(curves, dict):
        return {key: decimate(value, stride, length) for key, value in curves.items()}
    if (
        isinstance(curves, np.ndarray)
        and curves.ndim >= 1
        and curves.shape[0] == length
    ):
        return curves[::stride]
    if (
        isinstance(curves, list)
        and len(curves) == length
        and curves
        and not isinstance(curves[0], (str, dict))
    ):
        return curves[::stride]
    return curves


def units() -> Dict[str, Any]:
    """The unit block of the response file."""
    return {
        "energy": "eV, relative to the source E_F",
        "k": "1/angstrom, 2 pi included",
        "M": "angstrom/eV",
        "BCD": "angstrom",
        "chi": "mA nm / V^2",
        "chi_per_M": CHI_MA_NM_PER_V2_PER_A_PER_EV,
        "sigma_xy": "e^2/h",
        "dos": "states / eV / cell",
        "drude": "eV",
        "density": "cm^-2",
        "e2_over_h_S": E2_OVER_H_S,
        "e2_over_hbar_S": E2_OVER_HBAR_S,
    }


# ----------------------------------------------------------------- triangle rules
def _sorted_triangles(e: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vertex energies sorted ascending per triangle, X permuted alike."""
    order = np.argsort(e, axis=1, kind="stable")
    es = np.take_along_axis(e, order, axis=1)
    Xs = np.take_along_axis(X, order[:, :, None], axis=1)
    return es, Xs


def _safe(x: np.ndarray) -> np.ndarray:
    return np.where(np.abs(x) > 0.0, x, 1.0)


def _theta_one(
    es: np.ndarray, Xs: np.ndarray, area: np.ndarray, m: float
) -> np.ndarray:
    """Per-triangle int theta(m - E) X dA (nt, nX) for sorted vertex energies."""
    e1, e2, e3 = es[:, 0], es[:, 1], es[:, 2]
    X1, X2, X3 = Xs[:, 0], Xs[:, 1], Xs[:, 2]
    full = area[:, None] * (X1 + X2 + X3) / 3.0
    val = np.where((m >= e3)[:, None], full, 0.0)
    low = (m > e1) & (m <= e2)
    if low.any():
        i = np.flatnonzero(low)
        t12 = (m - e1[i]) / _safe(e2[i] - e1[i])
        t13 = (m - e1[i]) / _safe(e3[i] - e1[i])
        sub = area[i] * t12 * t13
        xm = (
            3.0 * X1[i]
            + t12[:, None] * (X2[i] - X1[i])
            + t13[:, None] * (X3[i] - X1[i])
        ) / 3.0
        val[i] = sub[:, None] * xm
    high = (m > e2) & (m < e3)
    if high.any():
        i = np.flatnonzero(high)
        s31 = (e3[i] - m) / _safe(e3[i] - e1[i])
        s32 = (e3[i] - m) / _safe(e3[i] - e2[i])
        sub = area[i] * s31 * s32
        xm = (
            3.0 * X3[i]
            + s31[:, None] * (X1[i] - X3[i])
            + s32[:, None] * (X2[i] - X3[i])
        ) / 3.0
        val[i] = full[i] - sub[:, None] * xm
    return val


def _delta_one(
    es: np.ndarray, Xs: np.ndarray, area: np.ndarray, m: float
) -> np.ndarray:
    """Per-triangle int delta(m - E) X dA (nt, nX) for sorted vertex energies."""
    e1, e2, e3 = es[:, 0], es[:, 1], es[:, 2]
    X1, X2, X3 = Xs[:, 0], Xs[:, 1], Xs[:, 2]
    val = np.zeros((len(es), Xs.shape[-1]))
    low = (m > e1) & (m <= e2) & (e3 > e1)
    if low.any():
        i = np.flatnonzero(low)
        t12 = (m - e1[i]) / _safe(e2[i] - e1[i])
        t13 = (m - e1[i]) / _safe(e3[i] - e1[i])
        dens = 2.0 * area[i] * (m - e1[i]) / _safe((e2[i] - e1[i]) * (e3[i] - e1[i]))
        mean = X1[i] + 0.5 * (
            t12[:, None] * (X2[i] - X1[i]) + t13[:, None] * (X3[i] - X1[i])
        )
        val[i] = dens[:, None] * mean
    high = (m > e2) & (m < e3)
    if high.any():
        i = np.flatnonzero(high)
        s31 = (e3[i] - m) / _safe(e3[i] - e1[i])
        s32 = (e3[i] - m) / _safe(e3[i] - e2[i])
        dens = 2.0 * area[i] * (e3[i] - m) / _safe((e3[i] - e1[i]) * (e3[i] - e2[i]))
        mean = X3[i] + 0.5 * (
            s31[:, None] * (X1[i] - X3[i]) + s32[:, None] * (X2[i] - X3[i])
        )
        val[i] = dens[:, None] * mean
    return val


def triangle_theta(
    e: np.ndarray,
    X: np.ndarray,
    area: np.ndarray,
    mu: np.ndarray,
    per_triangle: bool = False,
) -> np.ndarray:
    """sum_t int_t theta(mu - E) X dA for E and X linear on each triangle.

    e (nt, 3) vertex energies; X (nt, 3, nX); area (nt,); mu (nmu,).
    Returns (nmu, nX), or (nmu, nt, nX) with ``per_triangle``. Exact: below
    the middle vertex the region E < mu is a sub-triangle, above it the
    complement of one.
    """
    es, Xs = _sorted_triangles(np.asarray(e, float), np.asarray(X, float))
    area = np.asarray(area, float)
    vals = [_theta_one(es, Xs, area, float(m)) for m in np.asarray(mu, float).ravel()]
    if per_triangle:
        return np.stack(vals)
    return np.stack([v.sum(axis=0) for v in vals])


def triangle_delta(
    e: np.ndarray,
    X: np.ndarray,
    area: np.ndarray,
    mu: np.ndarray,
    per_triangle: bool = False,
) -> np.ndarray:
    """sum_t int_t delta(mu - E) X dA, the mu-derivative of ``triangle_theta``.

    For linear E the iso-line crosses two edges; X averages over its endpoints.
    """
    es, Xs = _sorted_triangles(np.asarray(e, float), np.asarray(X, float))
    area = np.asarray(area, float)
    vals = [_delta_one(es, Xs, area, float(m)) for m in np.asarray(mu, float).ravel()]
    if per_triangle:
        return np.stack(vals)
    return np.stack([v.sum(axis=0) for v in vals])


# Batch sizes that bound the temporaries however large the mesh grows:
# (triangle, mu) pairs evaluated at once, (piece, block) pairs of the
# cubic-piece accumulator, triangle-bands per leaf batch of the zone-wide sums,
# and vertices per batch of the smeared cross-check.
PAIRS_PER_CHUNK = 2_000_000
BLOCK_PAIRS_PER_CHUNK = 100_000
TRIANGLES_PER_CHUNK = 262_144
VERTICES_PER_CHUNK = 131_072
# Grid points per block of the cubic-piece accumulator. Its pieces are at
# least a step wide, so rounding stays near 1e-16 GRID_BLOCK^3 (2e-10) of a
# piece's own scale.
GRID_BLOCK = 128
# States per eV per cell below which a mu has no Fermi surface to cross-check.
SMEARED_DOS_FLOOR = 1e-4


def _accumulate(theta: np.ndarray, j: np.ndarray, val: np.ndarray) -> None:
    """theta[j] += val column by column (bincount is far faster than add.at)."""
    n = theta.shape[0]
    for c in range(val.shape[1]):
        theta[:, c] += np.bincount(j, weights=val[:, c], minlength=n)[:n]


def _range_pairs(first: np.ndarray, count: np.ndarray, size: int):
    """Batches (owner, offset) over the ranges [first, first + count): owner
    indexes the ranges, offset runs along each one."""
    own = np.flatnonzero(count > 0)
    if not len(own):
        return
    counts = count[own]
    starts = np.cumsum(counts) - counts
    total = int(counts.sum())
    for s in range(0, total, size):
        idx = np.arange(s, min(total, s + size))
        k = np.searchsorted(starts, idx, side="right") - 1
        yield own[k], idx - starts[k]


def _piece_coefficients(
    kind: str, low: bool, es: np.ndarray, Xs: np.ndarray, area: np.ndarray
) -> np.ndarray:
    """Each triangle's Theta (or its mu-derivative, the delta rule) on one piece
    of (e1, e3) as a cubic sum_k c_k s^k: the low piece (e1, e2] in s = mu - e1,
    the high piece (e2, e3) in s = e3 - mu. Returns c (nt, 4, nX)."""
    e1, e2, e3 = es[:, 0], es[:, 1], es[:, 2]
    X1, X2, X3 = Xs[:, 0], Xs[:, 1], Xs[:, 2]
    c = np.zeros((len(es), 4, Xs.shape[-1]))
    if low:
        d2, d3 = e2 - e1, e3 - e1
        w = (area / (d2 * d3))[:, None]
        q = (X2 - X1) / d2[:, None] + (X3 - X1) / d3[:, None]
        if kind == "theta":
            c[:, 2] = w * X1
            c[:, 3] = w * q / 3.0
        else:
            c[:, 1] = 2.0 * w * X1
            c[:, 2] = w * q
    else:
        d31, d32 = e3 - e1, e3 - e2
        w = (area / (d31 * d32))[:, None]
        r = (X1 - X3) / d31[:, None] + (X2 - X3) / d32[:, None]
        if kind == "theta":
            c[:, 0] = area[:, None] * (X1 + X2 + X3) / 3.0
            c[:, 2] = -w * X3
            c[:, 3] = -w * r / 3.0
        else:
            c[:, 1] = 2.0 * w * X3
            c[:, 2] = w * r
    return c


def _add_cubic_pieces(
    out: np.ndarray,
    coef: np.ndarray,
    anchor: np.ndarray,
    sign: float,
    first: np.ndarray,
    stop: np.ndarray,
    mu0: float,
    step: float,
) -> None:
    """out[j] += sum_k coef_k (sign (mu_j - anchor))^k over j in [first, stop) of
    each piece, mu_j = mu0 + j step.

    A piece enters each block of ``GRID_BLOCK`` points it touches once, as a
    change of the cubic's coefficients in the block's own coordinate where it
    starts and the reverse where it stops; a running sum along each block then
    holds the active cubic at every point. The cost is per block, not per grid
    point, and the short coordinate bounds the rounding (see ``GRID_BLOCK``)."""
    n, nX = out.shape
    B = GRID_BLOCK
    nblocks = -(-n // B)
    g = coef * ((sign * step) ** np.arange(4.0))[None, :, None]
    p = (np.asarray(anchor, float) - mu0) / step  # the anchor in grid units
    block0 = first // B
    span = (stop - 1) // B - block0 + 1
    acc = np.zeros(nblocks * (B + 1) * 4 * nX)
    cols = np.arange(4 * nX)
    for k, off in _range_pairs(block0, span, BLOCK_PAIRS_PER_CHUNK):
        b = block0[k] + off
        u0 = np.maximum(first[k] - b * B, 0)
        u1 = np.minimum(stop[k] - b * B, B)
        ub = (p[k] - b * B)[:, None]
        gk = g[k]
        local = np.empty_like(gk)  # sum_k g_k (u - ub)^k in powers of u
        local[:, 3] = gk[:, 3]
        local[:, 2] = gk[:, 2] - 3.0 * ub * gk[:, 3]
        local[:, 1] = gk[:, 1] - ub * (2.0 * gk[:, 2] - 3.0 * ub * gk[:, 3])
        local[:, 0] = gk[:, 0] - ub * (gk[:, 1] - ub * (gk[:, 2] - ub * gk[:, 3]))
        rows = np.concatenate([b * (B + 1) + u0, b * (B + 1) + u1])
        acc += np.bincount(
            (rows[:, None] * (4 * nX) + cols).ravel(),
            weights=np.concatenate([local, -local]).reshape(-1),
            minlength=acc.size,
        )
    active = np.cumsum(acc.reshape(nblocks, B + 1, 4, nX), axis=1)[:, :B]
    u = np.arange(B, dtype=float)[None, :, None]
    vals = active[:, :, 0] + u * (
        active[:, :, 1] + u * (active[:, :, 2] + u * active[:, :, 3])
    )
    out += vals.reshape(nblocks * B, nX)[:n]


def _grid_rule(
    kind: str,
    es: np.ndarray,
    Xs: np.ndarray,
    area: np.ndarray,
    mu0: float,
    step: float,
    n: int,
    pairs_per_chunk: int = PAIRS_PER_CHUNK,
) -> np.ndarray:
    """The theta or delta rule on mu_j = mu0 + j step for triangles with sorted
    vertex energies (``_sorted_triangles``): (n, nX)."""
    nX = Xs.shape[-1]
    out = np.zeros((n, nX))
    if n <= 0 or not len(es):
        return out
    area = np.asarray(area, float)
    pos = (es - mu0) / step
    lo = np.clip(np.floor(pos[:, 0]).astype(np.int64) + 1, 0, n)  # first mu > e1
    hi = np.clip(np.ceil(pos[:, 2]).astype(np.int64), 0, n)  # first mu >= e3
    mid = np.clip(np.floor(pos[:, 1]).astype(np.int64) + 1, lo, hi)  # first mu > e2
    if kind == "theta":
        # Wholly below mu_j: one cumulative sum.
        full = area[:, None] * Xs.sum(axis=1) / 3.0
        steps = np.zeros((n + 1, nX))
        for c in range(nX):
            steps[:, c] = np.bincount(hi, weights=full[:, c], minlength=n + 1)[: n + 1]
        out += np.cumsum(steps, axis=0)[:n]
    # Inside (e1, e3) each triangle is two cubic pieces, added per block. A
    # piece narrower than a step holds a grid point or two and is evaluated
    # there instead: its cubic's coefficients grow as 1/width^3.
    thin_t, thin_first, thin_count = [], [], []
    for low, first, stop, width in (
        (True, lo, mid, es[:, 1] - es[:, 0]),
        (False, mid, hi, es[:, 2] - es[:, 1]),
    ):
        has = stop > first
        wide = np.flatnonzero(has & (width >= step))
        if len(wide):
            _add_cubic_pieces(
                out,
                _piece_coefficients(kind, low, es[wide], Xs[wide], area[wide]),
                es[wide, 0] if low else es[wide, 2],
                1.0 if low else -1.0,
                first[wide],
                stop[wide],
                mu0,
                step,
            )
        thin = np.flatnonzero(has & (width < step))
        thin_t.append(thin)
        thin_first.append(first[thin])
        thin_count.append(stop[thin] - first[thin])
    tri = np.concatenate(thin_t)
    first = np.concatenate(thin_first)
    one = _theta_one if kind == "theta" else _delta_one
    for k, off in _range_pairs(first, np.concatenate(thin_count), pairs_per_chunk):
        t_id = tri[k]
        j = first[k] + off
        val = _pair_values(one, es[t_id], Xs[t_id], area[t_id], mu0 + j * step)
        _accumulate(out, j, val)
    return out


def _pair_values(
    one, es: np.ndarray, Xs: np.ndarray, area: np.ndarray, m: np.ndarray
) -> np.ndarray:
    """``one`` (the theta or delta kernel) with a per-row mu."""
    e1, e2, e3 = es[:, 0], es[:, 1], es[:, 2]
    X1, X2, X3 = Xs[:, 0], Xs[:, 1], Xs[:, 2]
    val = np.zeros((len(es), Xs.shape[-1]))
    low = (m > e1) & (m <= e2)
    high = (m > e2) & (m < e3)
    if one is _theta_one:
        full = area[:, None] * (X1 + X2 + X3) / 3.0
        val[m >= e3] = full[m >= e3]
        if low.any():
            q = np.flatnonzero(low)
            t12 = (m[q] - e1[q]) / _safe(e2[q] - e1[q])
            t13 = (m[q] - e1[q]) / _safe(e3[q] - e1[q])
            sub = area[q] * t12 * t13
            xm = (
                3.0 * X1[q]
                + t12[:, None] * (X2[q] - X1[q])
                + t13[:, None] * (X3[q] - X1[q])
            ) / 3.0
            val[q] = sub[:, None] * xm
        if high.any():
            q = np.flatnonzero(high)
            s31 = (e3[q] - m[q]) / _safe(e3[q] - e1[q])
            s32 = (e3[q] - m[q]) / _safe(e3[q] - e2[q])
            sub = area[q] * s31 * s32
            xm = (
                3.0 * X3[q]
                + s31[:, None] * (X1[q] - X3[q])
                + s32[:, None] * (X2[q] - X3[q])
            ) / 3.0
            val[q] = full[q] - sub[:, None] * xm
    else:
        low &= e3 > e1
        if low.any():
            q = np.flatnonzero(low)
            t12 = (m[q] - e1[q]) / _safe(e2[q] - e1[q])
            t13 = (m[q] - e1[q]) / _safe(e3[q] - e1[q])
            dens = (
                2.0
                * area[q]
                * (m[q] - e1[q])
                / _safe((e2[q] - e1[q]) * (e3[q] - e1[q]))
            )
            mean = X1[q] + 0.5 * (
                t12[:, None] * (X2[q] - X1[q]) + t13[:, None] * (X3[q] - X1[q])
            )
            val[q] = dens[:, None] * mean
        if high.any():
            q = np.flatnonzero(high)
            s31 = (e3[q] - m[q]) / _safe(e3[q] - e1[q])
            s32 = (e3[q] - m[q]) / _safe(e3[q] - e2[q])
            dens = (
                2.0
                * area[q]
                * (e3[q] - m[q])
                / _safe((e3[q] - e1[q]) * (e3[q] - e2[q]))
            )
            mean = X3[q] + 0.5 * (
                s31[:, None] * (X1[q] - X3[q]) + s32[:, None] * (X2[q] - X3[q])
            )
            val[q] = dens[:, None] * mean
    return val


def theta_on_grid(
    e: np.ndarray,
    X: np.ndarray,
    area: np.ndarray,
    mu0: float,
    step: float,
    n: int,
    pairs_per_chunk: int = PAIRS_PER_CHUNK,
) -> np.ndarray:
    """``triangle_theta`` on the uniform grid mu_j = mu0 + j step (j < n), scaling
    with the triangles rather than with n.

    A triangle wholly below mu_j enters by one cumulative sum; inside its energy
    range it is a cubic per piece, added per block of grid points. Returns (n, nX).
    """
    es, Xs = _sorted_triangles(np.asarray(e, float), np.asarray(X, float))
    return _grid_rule("theta", es, Xs, area, mu0, step, n, pairs_per_chunk)


def delta_on_grid(
    e: np.ndarray,
    X: np.ndarray,
    area: np.ndarray,
    mu0: float,
    step: float,
    n: int,
    pairs_per_chunk: int = PAIRS_PER_CHUNK,
) -> np.ndarray:
    """``triangle_delta`` on the uniform grid mu_j = mu0 + j step (j < n)."""
    es, Xs = _sorted_triangles(np.asarray(e, float), np.asarray(X, float))
    return _grid_rule("delta", es, Xs, area, mu0, step, n, pairs_per_chunk)


# ----------------------------------------------------------------- adaptive integration
# Per-band integrand columns carried by every vertex.
NX = 13
X_COUNT = 0
X_OMEGA = 1
X_VG = slice(
    2, 8
)  # v_a G_bc for (a; bc) = (x; xx), (x; xy), (x; yy), (y; xx), (y; xy), (y; yy)
X_VO = slice(8, 10)  # v_a Omega
X_VV = slice(10, 13)  # v_a v_b for xx, xy, yy
# Per-vertex mu-independent block columns.
NB = 6
B_OMEGA_C = 0  # the always-occupied bands below the window
B_OMEGA_S = 1
B_OMEGA_S_IN = 2
B_G_S = slice(3, 6)  # g_S,in xx, xy, yy
_SYM = ((0, 0), (0, 1), (1, 1))


def dipole_tensor(vg: np.ndarray) -> np.ndarray:
    """M (..., 2, 2, 2) from the six v_a G_bc columns."""
    vg = np.asarray(vg, float)
    M = np.zeros(vg.shape[:-1] + (2, 2, 2))
    col = 0
    for a in range(2):
        for b, c in _SYM:
            M[..., a, b, c] = vg[..., col]
            M[..., a, c, b] = vg[..., col]
            col += 1
    return M


def symmetric_tensor(xyz: np.ndarray) -> np.ndarray:
    """(..., 2, 2) from (xx, xy, yy) columns."""
    xyz = np.asarray(xyz, float)
    out = np.empty(xyz.shape[:-1] + (2, 2))
    out[..., 0, 0] = xyz[..., 0]
    out[..., 0, 1] = out[..., 1, 0] = xyz[..., 1]
    out[..., 1, 1] = xyz[..., 2]
    return out


def _fields(res: Dict[str, Any]) -> np.ndarray:
    v, G, om = res["v"], res["G"], res["omega"]
    X = np.empty(om.shape + (NX,))
    X[..., X_COUNT] = 1.0
    X[..., X_OMEGA] = om
    col = 2
    for a in range(2):
        for b, c in _SYM:
            X[..., col] = v[..., a] * G[..., b, c]
            col += 1
    X[..., 8] = v[..., 0] * om
    X[..., 9] = v[..., 1] * om
    X[..., 10] = v[..., 0] ** 2
    X[..., 11] = v[..., 0] * v[..., 1]
    X[..., 12] = v[..., 1] ** 2
    return X


def _blocks(res: Dict[str, Any], lo: int, n_occ: Optional[int]) -> np.ndarray:
    nk = res["E"].shape[0]
    B = np.zeros((nk, NB))
    if lo > 0:
        B[:, B_OMEGA_C] = res["blocks"][lo]["omega"]
    if n_occ:
        blk = res["blocks"][n_occ]
        B[:, B_OMEGA_S] = blk["omega"]
        B[:, B_OMEGA_S_IN] = blk["omega_in"]
        B[:, 3] = blk["g_in"][:, 0, 0]
        B[:, 4] = blk["g_in"][:, 0, 1]
        B[:, 5] = blk["g_in"][:, 1, 1]
    return B


def make_payload(
    table: Table,
    *,
    window: Tuple[int, int],
    n_occ: Optional[int] = None,
    tau_diag: Optional[np.ndarray] = None,
    deg_tol: float = 1e-6,
):
    """The per-vertex values the integrator needs, at both levels from one eigh.

    The primary level is the table's; with a full table and ``tau_diag`` the
    diagonal level rides along on the same eigensystem.
    """
    lo, hi = int(window[0]), int(window[1])
    cuts = [c for c in (lo, n_occ) if c]

    def payload(kfrac: np.ndarray) -> Dict[str, np.ndarray]:
        F = table.fourier(kfrac)
        E, U = np.linalg.eigh(F[:, 0])
        common = dict(window=(lo, hi), occupied=cuts, deg_tol=deg_tol, eig=(E, U))
        if table.level == LEVEL_FULL:
            main = geometry(
                F[:, 0], (F[:, 1], F[:, 2]), AW=(F[:, 3], F[:, 4]), OW=F[:, 5], **common
            )
        else:
            main = geometry(F[:, 0], (F[:, 1], F[:, 2]), tau=table.tau, **common)
        out = {
            "E": E[:, lo:hi],
            "E_above": E[:, min(hi, table.n - 1)],
            "v": main["v"],
            "v_pair": main["v_pair"],
            "X": _fields(main),
            "B": _blocks(main, lo, n_occ),
        }
        if n_occ:
            out["gap_S"] = main["blocks"][n_occ]["gap"]
            out["v_S"] = main["blocks"][n_occ]["v_cut"]
            out["psd_S"] = main["blocks"][n_occ]["psd_min_rel"]
        if table.level == LEVEL_FULL and tau_diag is not None:
            diag = geometry(F[:, 0], (F[:, 1], F[:, 2]), tau=tau_diag, **common)
            out["X_diag"] = _fields(diag)
            out["B_diag"] = _blocks(diag, lo, n_occ)
        return out

    return payload


def _local_extrema(grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Masks of strict local minima and maxima on a periodic 2D grid."""
    nbrs = [
        np.roll(np.roll(grid, dx, 0), dy, 1)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if dx or dy
    ]
    stack = np.stack(nbrs)
    return grid < stack.min(axis=0), grid > stack.max(axis=0)


def occupation_pass(
    table: Table,
    *,
    n: int,
    mu_range: Tuple[float, float],
    kT_max: float,
    occupied_bands: Optional[int] = None,
    insulator_min_gap: float = 1e-4,
    deg_tol: float = 1e-6,
    hot_gap: float = 0.05,
    extra_kfrac: Optional[np.ndarray] = None,
    c3_points: int = 512,
    chunk: int = 4096,
    energies_fn=None,
) -> Dict[str, Any]:
    """Energies on a Gamma-centred n x n grid (plus ``extra_kfrac``): the band
    window, the electron count, the gap, the refinement seeds and the PT flag.

    Energies relative to E_F (eV); ``mu_range`` and ``kT_max`` in eV.
    """
    energies = energies_fn or table.energies
    k = uniform_grid(n)
    E = np.concatenate([energies(k[s : s + chunk]) for s in range(0, len(k), chunk)])
    if extra_kfrac is not None and len(extra_kfrac):
        E_all = np.concatenate([E, energies(np.asarray(extra_kfrac, float))])
    else:
        E_all = E
    nb = E.shape[1]
    lo_e = mu_range[0] - 20.0 * kT_max - 0.1
    hi_e = mu_range[1] + 20.0 * kT_max + 0.1
    emin, emax = E_all.min(axis=0), E_all.max(axis=0)
    inside = np.flatnonzero((emax >= lo_e) & (emin <= hi_e))
    if len(inside):
        lo, hi = int(inside[0]), int(inside[-1]) + 1
    else:
        # mu deep in a gap: the band just below it (or the lowest) is the window;
        # the response is then the always-occupied block's.
        below = int(np.sum(emax < lo_e))
        lo = max(below - 1, 0)
        hi = lo + 1
    # Move the cuts to global gaps, so the always-occupied block never touches the window.
    while lo > 0 and emax[lo - 1] >= emin[lo] - 1e-9:
        lo -= 1
    while hi < nb and emin[hi] <= emax[hi - 1] + 1e-9:
        hi += 1
    count_ef = float(np.mean(np.sum(E < 0.0, axis=1)))
    n_occ = int(occupied_bands) if occupied_bands else int(round(count_ef))
    frac = count_ef - math.floor(count_ef)
    out: Dict[str, Any] = {
        "window": (lo, hi),
        "band_min": emin,
        "band_max": emax,
        "count_at_EF": count_ef,
        "count_fraction_ambiguous": bool(0.2 <= frac <= 0.8),
        "n_occ": n_occ,
        "grid": n,
    }
    if 0 < n_occ < nb:
        direct = E_all[:, n_occ] - E_all[:, n_occ - 1]
        out["min_direct_gap"] = float(direct.min())
        out["min_direct_gap_k"] = (
            k[int(np.argmin(direct[: len(k)]))] if len(direct) else None
        )
        out["global_gap"] = float(emin[n_occ] - emax[n_occ - 1])
        out["S_applies"] = bool(direct.min() >= insulator_min_gap)
        g0 = np.flatnonzero(np.all(np.abs(k[:, :2]) < 1e-12, axis=1))
        out["gap_at_gamma"] = float(direct[g0[0]]) if len(g0) else None
        # mu_CNP at T = 0: midway in a gap, else the energy where N(mu) = n_occ.
        if out["global_gap"] > 0:
            out["mu_cnp"] = float(0.5 * (emin[n_occ] + emax[n_occ - 1]))
        else:
            flat = np.sort(E.ravel())
            out["mu_cnp"] = (
                float(flat[int(round(n_occ * len(k))) - 1])
                if n_occ * len(k) <= len(flat)
                else 0.0
            )
    else:
        out["S_applies"] = False
    # Seeds: hot spots of small gaps and band extrema inside the extended range.
    seeds: List[np.ndarray] = []
    seed_info: List[Dict[str, Any]] = []
    mu_seeds: List[float] = []
    grid_shape = (n, n)
    pairs = [(b, b + 1) for b in range(lo, min(hi, nb - 1))]
    if out.get("S_applies") is not None and 0 < n_occ < nb:
        pairs.append((n_occ - 1, n_occ))
    for b0, b1 in sorted(set(pairs)):
        gap = (E[:, b1] - E[:, b0]).reshape(grid_shape)
        mins, _ = _local_extrema(gap)
        for idx in np.flatnonzero(
            mins.ravel() & (gap.ravel() < hot_gap) & (gap.ravel() > deg_tol)
        ):
            seeds.append(k[idx])
            seed_info.append(
                {
                    "kind": "gap",
                    "bands": [b0, b1],
                    "gap_meV": float(gap.ravel()[idx]) * 1e3,
                }
            )
    for b in range(lo, hi):
        band = E[:, b].reshape(grid_shape)
        mins, maxs = _local_extrema(band)
        for idx in np.flatnonzero((mins | maxs).ravel()):
            e = float(band.ravel()[idx])
            if lo_e <= e <= hi_e:
                seeds.append(k[idx])
                seed_info.append({"kind": "extremum", "band": b, "energy_meV": e * 1e3})
                if mu_range[0] <= e <= mu_range[1]:
                    mu_seeds.append(e)
    out["seeds"] = np.asarray(seeds, float).reshape(-1, 3)
    out["seed_info"] = seed_info
    out["mu_seeds"] = sorted(mu_seeds)
    # PT pairs: most adjacent window pairs split by less than 1 meV.
    splits = [float(np.median(E[:, b + 1] - E[:, b])) for b in range(lo, hi - 1)]
    tiny = [s for s in splits if s < 1e-3]
    out["pt_pairs"] = {
        "detected": bool(splits) and len(tiny) > 0.5 * len(splits),
        "median_split_meV": float(np.median(splits) * 1e3) if splits else None,
    }
    # The Wannier C3 floor: how far the window bands are from C3-symmetric.
    if is_hexagonal(table.lattice) and c3_points:
        rng = np.random.default_rng(12345)
        pts = np.c_[rng.uniform(-0.5, 0.5, (c3_points, 2)), np.zeros(c3_points)]
        e1 = energies(pts)[:, lo:hi]
        e2 = energies(c3_rotate_frac(pts, table.lattice))[:, lo:hi]
        out["c3_band_floor"] = float(np.abs(e1 - e2).max())
    else:
        out["c3_band_floor"] = None
    return out


# The per-vertex buffer holding each vertex's integer key (x * unit + y).
VERTEX_KEY = "_key"


class AdaptiveMesh:
    """A Gamma-centred quadtree over [-1/2, 1/2)^2 in fractional k.

    Each leaf cell is four triangles from its centre to its edges. Vertex
    coordinates are integers in units of 1/(base 2^DEPTH) and wrap
    periodically, so a vertex is evaluated once however many cells share it.
    Hanging vertices are harmless: each triangle is integrated on its own.
    """

    # Levels below a base cell. A vertex key x * unit + y stays in int64 while
    # unit = base 2^DEPTH < 2^31.5, so for a base mesh up to MAX_BASE.
    DEPTH = 20
    MAX_BASE = 2048

    def __init__(
        self,
        base: int,
        table_lattice: np.ndarray,
        payload,
        frame: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.base = int(base)
        if not 1 <= self.base <= self.MAX_BASE:
            raise ValueError(f"base mesh {self.base} outside 1..{self.MAX_BASE}")
        self.unit = self.base << self.DEPTH
        self.s0 = 1 << self.DEPTH
        lat = np.asarray(table_lattice, float).reshape(3, 3)
        frame = frame or layer_frame(lat)
        self.b = reciprocal_in_plane(lat, frame)
        self.area_bz = zone_area(lat)
        self.diag = float(
            max(
                np.linalg.norm(self.b[0] + self.b[1]),
                np.linalg.norm(self.b[0] - self.b[1]),
            )
        )
        self.payload = payload
        # Vertex id per integer point, keyed x * unit + y. The values live in
        # buffers that double when full, so a round never copies the mesh.
        self.key: Dict[int, int] = {}
        self._buf: Dict[str, np.ndarray] = {}
        self.nv = 0
        n = self.base
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        self.lvl = np.zeros(n * n, np.int64)
        self.cx = (i.ravel() * self.s0).astype(np.int64)
        self.cy = (j.ravel() * self.s0).astype(np.int64)
        self.vid = self._cell_vertices(self.lvl, self.cx, self.cy)
        self.ind = np.full(len(self.lvl), np.nan)
        # Whether the must-split rules have seen a leaf: they depend only on its
        # own vertices, so a leaf is judged once.
        self.checked = np.zeros(len(self.lvl), bool)
        self._index: Optional[Dict[Tuple[int, int, int], int]] = None
        self.vertices_by_rule = {
            "base": int(self.nv),
            "seed": 0,
            "gap": 0,
            "edge": 0,
            "error": 0,
        }

    # -- vertices ---------------------------------------------------------
    def _frac(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fractional k of integer points: integer 0 is -1/2, as in ``contains``."""
        fx = (x % self.unit) / self.unit - 0.5
        fy = (y % self.unit) / self.unit - 0.5
        return np.stack([fx, fy, np.zeros_like(fx)], axis=1)

    def ensure(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vertex ids for integer points, evaluating the new ones in one batch."""
        x = np.asarray(x, np.int64).ravel() % self.unit
        y = np.asarray(y, np.int64).ravel() % self.unit
        keys = (x * self.unit + y).tolist()
        new: Dict[int, int] = {}
        for kk in keys:
            if kk not in self.key and kk not in new:
                new[kk] = self.nv + len(new)
        if new:
            pts = np.fromiter(new.keys(), np.int64, len(new))
            vals = dict(self.payload(self._frac(pts // self.unit, pts % self.unit)))
            vals[VERTEX_KEY] = pts
            self._store(vals)
            self.key.update(new)
        return np.array([self.key[kk] for kk in keys], np.int64)

    def vertex_kfrac(self, vids: np.ndarray) -> np.ndarray:
        """Fractional k (n, 3) of existing vertices."""
        keys = self._buf[VERTEX_KEY][np.asarray(vids, np.int64)]
        return self._frac(keys // self.unit, keys % self.unit)

    def _store(self, vals: Dict[str, np.ndarray]) -> None:
        n = len(next(iter(vals.values())))
        need = self.nv + n
        for name, arr in vals.items():
            arr = np.asarray(arr)
            buf = self._buf.get(name)
            if buf is None or len(buf) < need:
                size = max(need, 2 * (len(buf) if buf is not None else 0), 1024)
                grown = np.empty((size,) + arr.shape[1:], arr.dtype)
                if buf is not None:
                    grown[: self.nv] = buf[: self.nv]
                self._buf[name] = buf = grown
            buf[self.nv : need] = arr
        self.nv = need

    @property
    def vals(self) -> Dict[str, np.ndarray]:
        """Per-vertex values: views of the filled rows."""
        return {name: buf[: self.nv] for name, buf in self._buf.items()}

    def _cell_vertices(
        self, lvl: np.ndarray, cx: np.ndarray, cy: np.ndarray
    ) -> np.ndarray:
        s = self.s0 >> lvl
        h = s >> 1
        xs = np.stack([cx, cx + s, cx + s, cx, cx + h], axis=1)
        ys = np.stack([cy, cy, cy + s, cy + s, cy + h], axis=1)
        return self.ensure(xs, ys).reshape(len(lvl), 5)

    # -- geometry ----------------------------------------------------------
    def diameter(self, lvl: np.ndarray) -> np.ndarray:
        """Cartesian diameter (1/Angstrom) of cells at these levels."""
        return self.diag * (self.s0 >> np.asarray(lvl, np.int64)) / self.unit

    def triangles(
        self, vid: np.ndarray, lvl: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """(ncell * 4, 3) vertex ids and (ncell * 4,) Cartesian areas."""
        c = vid[:, :4]
        m = vid[:, 4]
        tri = np.stack(
            [np.stack([c[:, q], c[:, (q + 1) % 4], m], axis=1) for q in range(4)],
            axis=1,
        ).reshape(-1, 3)
        side = (self.s0 >> np.asarray(lvl, np.int64)) / self.unit
        area = np.repeat(side**2 * self.area_bz / 4.0, 4)
        return tri, area

    def children(self, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lvl, cx, cy = self.lvl[idx], self.cx[idx], self.cy[idx]
        h = (self.s0 >> lvl) >> 1
        clvl = np.repeat(lvl + 1, 4)
        ccx = np.stack([cx, cx + h, cx, cx + h], axis=1).ravel()
        ccy = np.stack([cy, cy, cy + h, cy + h], axis=1).ravel()
        return clvl, ccx, ccy

    def split(self, idx: np.ndarray, rule: str) -> None:
        idx = np.unique(np.asarray(idx, np.int64))
        if not len(idx):
            return
        before = self.nv
        clvl, ccx, ccy = self.children(idx)
        cvid = self._cell_vertices(clvl, ccx, ccy)
        keep = np.ones(len(self.lvl), bool)
        keep[idx] = False
        self.lvl = np.concatenate([self.lvl[keep], clvl])
        self.cx = np.concatenate([self.cx[keep], ccx])
        self.cy = np.concatenate([self.cy[keep], ccy])
        self.vid = np.concatenate([self.vid[keep], cvid])
        self.ind = np.concatenate([self.ind[keep], np.full(len(clvl), np.nan)])
        self.checked = np.concatenate([self.checked[keep], np.zeros(len(clvl), bool)])
        self._index = None
        self.vertices_by_rule[rule] = self.vertices_by_rule.get(rule, 0) + (
            self.nv - before
        )

    def contains(self, kfrac: np.ndarray) -> np.ndarray:
        """Leaf index holding each fractional point (-1 if none)."""
        if self._index is None:
            self._index = {
                key: i
                for i, key in enumerate(
                    zip(self.lvl.tolist(), self.cx.tolist(), self.cy.tolist())
                )
            }
        x = np.floor((np.asarray(kfrac)[:, 0] + 0.5) % 1.0 * self.unit).astype(np.int64)
        y = np.floor((np.asarray(kfrac)[:, 1] + 0.5) % 1.0 * self.unit).astype(np.int64)
        top = int(self.lvl.max()) if len(self.lvl) else 0
        out = np.full(len(x), -1, np.int64)
        for i, (xi, yi) in enumerate(zip(x.tolist(), y.tolist())):
            for level in range(top + 1):
                s = self.s0 >> level
                hit = self._index.get((level, xi - xi % s, yi - yi % s))
                if hit is not None:
                    out[i] = hit
                    break
        return out


def _cell_block_integral(
    mesh: AdaptiveMesh, vid: np.ndarray, lvl: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """Per cell: sum over its 4 triangles of area x mean(B): (ncell, nB)."""
    side = (mesh.s0 >> np.asarray(lvl, np.int64)) / mesh.unit
    w = side**2 * mesh.area_bz
    corners = B[vid[:, :4]].sum(axis=1)
    return w[:, None] * (corners / 6.0 + B[vid[:, 4]] / 3.0)


def _grouped(
    kind: str,
    e: np.ndarray,
    X: np.ndarray,
    area: np.ndarray,
    mu: np.ndarray,
    group: np.ndarray,
    ngroups: int,
) -> np.ndarray:
    """theta- or delta-rule integrals summed per group: (nmu, ngroups, nX).

    As ``theta_on_grid`` on an arbitrary mu set: a triangle wholly below mu_j
    enters through one cumulative sum, and only the (triangle, mu) pairs with
    mu inside the triangle's energy range get the partial formula.
    """
    es, Xs = _sorted_triangles(np.asarray(e, float), np.asarray(X, float))
    area = np.asarray(area, float)
    group = np.asarray(group, np.int64)
    mu = np.asarray(mu, float).ravel()
    order = np.argsort(mu, kind="stable")
    ms = mu[order]
    nmu, nX = len(ms), Xs.shape[-1]
    flat = np.zeros((nmu * ngroups, nX))
    if nmu and len(es):
        lo = np.searchsorted(ms, es[:, 0], side="right")  # first mu > e1
        hi = np.searchsorted(ms, es[:, 2], side="left")  # first mu >= e3
        if kind == "theta":
            full = area[:, None] * Xs.sum(axis=1) / 3.0
            size = (nmu + 1) * ngroups
            steps = np.empty((size, nX))
            for c in range(nX):
                steps[:, c] = np.bincount(
                    hi * ngroups + group, weights=full[:, c], minlength=size
                )
            flat += np.cumsum(steps.reshape(nmu + 1, ngroups, nX), axis=0)[
                :nmu
            ].reshape(-1, nX)
        count = np.maximum(hi - lo, 0)
        tri = np.flatnonzero(count)
        if len(tri):
            starts = np.cumsum(count[tri]) - count[tri]
            total = int(count[tri].sum())
            one = _theta_one if kind == "theta" else _delta_one
            for s in range(0, total, PAIRS_PER_CHUNK):
                idx = np.arange(s, min(total, s + PAIRS_PER_CHUNK))
                owner = np.searchsorted(starts, idx, side="right") - 1
                t_id = tri[owner]
                j = lo[t_id] + (idx - starts[owner])
                val = _pair_values(one, es[t_id], Xs[t_id], area[t_id], ms[j])
                _accumulate(flat, j * ngroups + group[t_id], val)
    out = np.empty((nmu, ngroups, nX))
    out[order] = flat.reshape(nmu, ngroups, nX)
    return out


FS_COLUMNS = np.r_[
    X_COUNT, np.arange(2, 10)
]  # DOS, M (6), BCD (2): the delta-rule indicators
SEA_COLUMNS = np.array([X_COUNT, X_OMEGA])


class Integrator:
    """Adaptive zone integration of one model (see the module docstring for units).

    Phases: must-split rules (seeds, the restricted and capped gap rule, the edge
    rule), then error-driven refinement with lookahead to ``tolerance`` and on to
    ``tolerance / 2``, with a snapshot of every integral at each stage.
    """

    def __init__(
        self,
        payload,
        lattice: np.ndarray,
        *,
        window: Tuple[int, int],
        n_occ: Optional[int],
        s_applies: bool,
        mu_range: Tuple[float, float],
        mu_subset: Sequence[float],
        headline_mu: Dict[str, float],
        kT_max: float,
        base: int = 64,
        tolerance: float = 1e-3,
        max_points: int = 1_000_000,
        gap_ratio: float = 0.1,
        gap_floor: float = 5e-5,
        edge_resolution: float = 2e-4,
        seeds: Optional[np.ndarray] = None,
        seed_diameter: float = 0.0,
        seed_spread: float = 0.0,
        deadline: Optional[float] = None,
        frame: Optional[Dict[str, np.ndarray]] = None,
        log=None,
    ):
        self.mesh = AdaptiveMesh(base, lattice, payload, frame=frame)
        self.window = (int(window[0]), int(window[1]))
        self.n_occ = n_occ
        self.s_applies = bool(s_applies and n_occ)
        # The extended range of the window definition: +-20 kT and 0.1 eV beyond.
        self.ext = (
            mu_range[0] - 20.0 * kT_max - 0.1,
            mu_range[1] + 20.0 * kT_max + 0.1,
        )
        self.mu_subset = np.unique(np.asarray(list(mu_subset), float))
        self.headline_mu = dict(headline_mu)
        self.tolerance = float(tolerance)
        self.max_points = int(max_points)
        self.gap_ratio = float(gap_ratio)
        self.gap_floor = float(gap_floor)
        self.edge_resolution = float(edge_resolution)
        self.seeds = (
            np.zeros((0, 3))
            if seeds is None
            else np.asarray(seeds, float).reshape(-1, 3)
        )
        if len(self.seeds) and seed_spread > 0:
            # A seed is a grid point near the feature, not the feature: refine the
            # cells under a 3x3 stencil of one grid spacing around it, so a pocket
            # that straddles a cell boundary is refined on both sides.
            offs = (
                np.array([(i, j, 0) for i in (-1, 0, 1) for j in (-1, 0, 1)], float)
                * seed_spread
            )
            self.seeds = (self.seeds[:, None, :] + offs[None]).reshape(-1, 3)
        self.seed_diameter = float(seed_diameter)
        self.deadline = deadline
        self.log = log or (lambda message: None)
        self.history: List[Dict[str, Any]] = []
        self.gap_rule_capped = False
        self.stop = None
        self.snapshots: Dict[str, Dict[str, Any]] = {}

    # -- budget ------------------------------------------------------------
    def _out_of_budget(self) -> Optional[str]:
        if self.mesh.nv >= self.max_points:
            return "budget"
        if self.deadline is not None and time.time() > self.deadline:
            return "time"
        return None

    # -- must-split rules --------------------------------------------------
    def _rule_flags(self) -> Dict[str, np.ndarray]:
        """The must-split flags of every leaf; only leaves not judged before are
        evaluated (the rules depend on a leaf's own vertices and the seeds)."""
        mesh = self.mesh
        vals = mesh.vals
        todo = np.flatnonzero(~mesh.checked)
        n_all = len(mesh.lvl)
        flags: Dict[str, np.ndarray] = {}
        if len(self.seeds):
            seed = np.zeros(n_all, bool)
            cells = mesh.contains(self.seeds)
            cells = cells[cells >= 0]
            d_all = mesh.diameter(mesh.lvl[cells])
            seed[cells] = d_all > self.seed_diameter
            flags["seed"] = seed
        mesh.checked[todo] = True
        if not len(todo):
            return flags
        vid = mesh.vid[todo]
        d = mesh.diameter(mesh.lvl[todo])

        def widen(sub: np.ndarray) -> np.ndarray:
            full = np.zeros(n_all, bool)
            full[todo] = sub
            return full

        E = vals["E"][vid]  # (ncell, 5, nW)
        if not self.gap_rule_capped:
            above = vals["E_above"][vid]
            nxt = np.concatenate([E[:, :, 1:], above[:, :, None]], axis=2)
            gap = (nxt - E).min(axis=1)
            vmax = vals["v_pair"][vid].max(axis=1)
            near = ((E >= self.ext[0]) & (E <= self.ext[1])).any(axis=1)
            near_next = ((nxt >= self.ext[0]) & (nxt <= self.ext[1])).any(axis=1)
            split_pair = (
                d[:, None] * vmax > self.gap_ratio * np.maximum(gap, self.gap_floor)
            ) & (near | near_next)
            split_pair &= (
                gap > 1e-9
            )  # a symmetry-degenerate pair is not an anticrossing
            gap_flag = split_pair.any(axis=1)
            if self.s_applies and "gap_S" in vals:
                gs = vals["gap_S"][vid].min(axis=1)
                vs = vals["v_S"][vid].max(axis=1)
                gap_flag |= d * vs > self.gap_ratio * np.maximum(gs, self.gap_floor)
            flags["gap"] = widen(gap_flag)
        v = vals["v"][vid]  # (ncell, 5, nW, 2)
        span = E.max(axis=1) - E.min(axis=1)
        turn = (v.min(axis=1) < 0) & (v.max(axis=1) > 0)
        inner = turn.all(axis=-1)
        reach = (E.max(axis=1) >= self.ext[0]) & (E.min(axis=1) <= self.ext[1])
        flags["edge"] = widen(
            (inner & reach & (span > self.edge_resolution)).any(axis=1)
        )
        return flags

    def must_split(self, max_rounds: int = 200) -> None:
        for _ in range(max_rounds):
            reason = self._out_of_budget()
            if reason:
                self.stop = reason
                return
            flags = self._rule_flags()
            chosen: Dict[str, np.ndarray] = {}
            taken = np.zeros(len(self.mesh.lvl), bool)
            for rule in ("seed", "gap", "edge"):
                f = flags.get(rule)
                if f is None:
                    continue
                idx = np.flatnonzero(
                    f & ~taken & (self.mesh.lvl < AdaptiveMesh.DEPTH - 2)
                )
                if len(idx):
                    chosen[rule] = idx
                    taken[idx] = True
            if not chosen:
                return
            removed = np.zeros(0, np.int64)
            for rule, idx in chosen.items():
                # A split drops its leaves and appends the children, so the
                # indices taken before this round's earlier splits shift down.
                cur = idx - np.searchsorted(removed, idx)
                if (
                    rule == "gap"
                    and self.mesh.vertices_by_rule.get("gap", 0)
                    >= 0.3 * self.max_points
                ):
                    self.gap_rule_capped = True
                    # Judged again next round, by the rules still in force.
                    self.mesh.checked[cur] = False
                    continue
                self.mesh.split(cur, rule)
                removed = np.union1d(removed, idx)
                if (
                    rule == "gap"
                    and self.mesh.vertices_by_rule["gap"] >= 0.3 * self.max_points
                ):
                    self.gap_rule_capped = True
            self.history.append(
                {
                    "phase": "must_split",
                    "vertices": int(self.mesh.nv),
                    "leaves": int(len(self.mesh.lvl)),
                }
            )

    # -- error indicators --------------------------------------------------
    def _triangle_fields(
        self, vid: np.ndarray, lvl: np.ndarray, X: np.ndarray, E: np.ndarray
    ):
        tri, area = self.mesh.triangles(vid, lvl)
        nW = E.shape[1]
        e = np.moveaxis(E[tri], 2, 1).reshape(-1, 3)  # (nt * nW, 3)
        x = np.moveaxis(X[tri], 2, 1).reshape(-1, 3, X.shape[-1])
        a = np.repeat(area, nW)
        cell = np.repeat(np.arange(len(vid)), 4 * nW)
        return e, x, a, cell

    def _chunk_cells(self) -> int:
        """Leaves per batch: ``TRIANGLES_PER_CHUNK`` triangle-bands."""
        n_bands = self.mesh.vals["E"].shape[1] if self.mesh.nv else 1
        return max(64, TRIANGLES_PER_CHUNK // (4 * max(1, n_bands)))

    def _chunks(self):
        """Leaf slices: zone-wide sums never hold the whole mesh's triangles."""
        n, size = len(self.mesh.lvl), self._chunk_cells()
        return [slice(s, min(n, s + size)) for s in range(0, n, size)]

    def _scales(self) -> Dict[str, Any]:
        mesh = self.mesh
        vals = mesh.vals
        absX = np.abs(vals["X"])
        fs = np.zeros((len(self.mu_subset), len(FS_COLUMNS)))
        sea_omega = 0.0
        for part in self._chunks():
            e, x, a, _ = self._triangle_fields(
                mesh.vid[part], mesh.lvl[part], absX, vals["E"]
            )
            fs += _grouped(
                "delta",
                e,
                x[..., FS_COLUMNS],
                a,
                self.mu_subset,
                np.zeros(len(e), np.int64),
                1,
            )[:, 0]
            sea_omega += float(np.sum(a * x[:, :, X_OMEGA].mean(axis=1)))
        blocks = np.abs(
            _cell_block_integral(mesh, mesh.vid, mesh.lvl, np.abs(vals["B"]))
        ).sum(axis=0)
        sea_omega += blocks[B_OMEGA_C]
        return {
            "fs": np.maximum(fs, 1e-300),
            "sea": np.array(
                [
                    mesh.area_bz * (self.window[1] - self.window[0]),
                    max(sea_omega, 1e-300),
                ]
            ),
            "blocks": np.maximum(blocks, 1e-300),
            "fs_abs_vg": fs[:, 1:7],
        }

    def _headline_signed(self) -> Dict[float, Tuple[float, float]]:
        """Current alpha and beta (zone integrals of the signed v G) at each headline mu."""
        mesh = self.mesh
        vals = mesh.vals
        mus = np.array(sorted(set(self.headline_mu.values())), float)
        if not len(mus):
            return {}
        vg = np.zeros((len(mus), 6))
        for part in self._chunks():
            e, x, a, _ = self._triangle_fields(
                mesh.vid[part], mesh.lvl[part], vals["X"], vals["E"]
            )
            vg += _grouped(
                "delta", e, x[..., X_VG], a, mus, np.zeros(len(e), np.int64), 1
            )[:, 0]
        out = {}
        for m, row in zip(mus, vg):
            tr = trigonal(dipole_tensor(row))
            out[float(m)] = (float(tr["A3"]), float(np.abs(row).max()))
        return out

    def _corner_triangles(
        self, vid: np.ndarray, lvl: np.ndarray, X: np.ndarray, E: np.ndarray
    ):
        """Each cell as two triangles on each diagonal, corners only, weights 1/2."""
        c = vid[:, :4]
        tri = np.stack(
            [c[:, [0, 1, 2]], c[:, [0, 2, 3]], c[:, [1, 2, 3]], c[:, [1, 3, 0]]], axis=1
        ).reshape(-1, 3)
        side = (self.mesh.s0 >> np.asarray(lvl, np.int64)) / self.mesh.unit
        area = np.repeat(side**2 * self.mesh.area_bz / 4.0, 4)  # (cell/2) x weight 1/2
        nW = E.shape[1]
        e = np.moveaxis(E[tri], 2, 1).reshape(-1, 3)
        x = np.moveaxis(X[tri], 2, 1).reshape(-1, 3, X.shape[-1])
        a = np.repeat(area, nW)
        cell = np.repeat(np.arange(len(vid)), 4 * nW)
        return e, x, a, cell

    def _error_sums(
        self,
        idx: np.ndarray,
        scales: Dict[str, Any],
        head: Dict[float, Tuple[float, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """``_errors`` over cell chunks: each cell's largest normalised error and
        every quantity's summed error, never the (cells x quantities) matrix."""
        ind = np.empty(len(idx))
        total = None
        size = self._chunk_cells()
        for s in range(0, len(idx), size):
            err = self._errors(idx[s : s + size], scales, head)
            ind[s : s + size] = np.abs(err).max(axis=1)
            part = err.sum(axis=0)
            total = part if total is None else total + part
        return ind, total

    def _errors(
        self,
        idx: np.ndarray,
        scales: Dict[str, Any],
        head: Dict[float, Tuple[float, float]],
    ) -> np.ndarray:
        """Embedded error estimate per cell and per controlled quantity: its four
        centre triangles against the corner-only rule (both diagonals averaged),
        divided by 3 (the fine rule's share for an O(h^2) pair, Richardson) and
        normalised by that quantity's scale. Returns (len(idx), nK); the last
        ``self._n_signed`` columns (alpha and beta at the headline mu) are
        signed, because those are cancellation-dominated combinations whose
        global error is the signed sum, not the sum of local magnitudes.

        It needs no new vertex: for a mu-independent integrand the two rules
        differ by the area times (centre - corner mean)/3.
        """
        mesh = self.mesh
        vals = mesh.vals
        X, E, B = vals["X"], vals["E"], vals["B"]
        n = len(idx)
        vid = mesh.vid[idx]
        lvl = mesh.lvl[idx]
        parts: List[np.ndarray] = []
        side = (mesh.s0 >> lvl) / mesh.unit
        w = side**2 * mesh.area_bz
        fine_b = _cell_block_integral(mesh, vid, lvl, B)
        coarse_b = w[:, None] * B[vid[:, :4]].mean(axis=1)
        cols = [B_OMEGA_C] + ([B_OMEGA_S] if self.s_applies else [])
        for col in cols:
            parts.append(
                (
                    np.abs(fine_b[:, col] - coarse_b[:, col])
                    / (3.0 * scales["blocks"][col])
                )[:, None]
            )
        if self.s_applies:
            dg = (fine_b[:, 3] + fine_b[:, 5]) - (coarse_b[:, 3] + coarse_b[:, 5])
            parts.append(
                (
                    np.abs(dg)
                    / (3.0 * max(scales["blocks"][3] + scales["blocks"][5], 1e-300))
                )[:, None]
            )
        ef, xf, af, cf = self._triangle_fields(vid, lvl, X, E)
        ec, xc, ac, cc = self._corner_triangles(vid, lvl, X, E)
        fs_f = _grouped("delta", ef, xf[..., FS_COLUMNS], af, self.mu_subset, cf, n)
        fs_c = _grouped("delta", ec, xc[..., FS_COLUMNS], ac, self.mu_subset, cc, n)
        dfs = np.abs(fs_f - fs_c) / (3.0 * scales["fs"][:, None, :])  # (nmu, n, 9)
        parts.append(np.moveaxis(dfs, 1, 0).reshape(n, -1))
        sea_f = _grouped("theta", ef, xf[..., SEA_COLUMNS], af, self.mu_subset, cf, n)
        sea_c = _grouped("theta", ec, xc[..., SEA_COLUMNS], ac, self.mu_subset, cc, n)
        dsea = np.abs(sea_f - sea_c) / (3.0 * scales["sea"][None, None, :])
        parts.append(np.moveaxis(dsea, 1, 0).reshape(n, -1))
        signed = 0
        for j, m in enumerate(self.mu_subset):
            if float(m) not in head:
                continue
            A3, _ = head[float(m)]
            S_M = float(scales["fs_abs_vg"][j].max())
            tr = trigonal(dipole_tensor(fs_f[j][:, 1:7] - fs_c[j][:, 1:7]))
            norm = 3.0 * max(A3, 1e-3 * S_M, 1e-300)
            parts.append(np.stack([tr["alpha"], tr["beta"]], axis=1) / norm)
            signed += 2
        self._n_signed = signed
        return np.concatenate(parts, axis=1)

    def restart(self, headline_mu: Optional[Dict[str, float]] = None) -> None:
        """New scales (and headline mu) for the next stage; totals rebuilt over all leaves."""
        if headline_mu is not None:
            self.headline_mu = dict(headline_mu)
            self.mu_subset = np.unique(
                np.concatenate([self.mu_subset, list(headline_mu.values())])
            )
        self._scales_fixed = None
        self._totals = None
        self.mesh.ind[:] = np.nan

    def refine(self, target: float, batch: Optional[int] = None) -> bool:
        """Split cells until every controlled quantity's summed error is <= target.

        Each quantity's total is kept incrementally: a cell's error vector is
        added when it is created and subtracted, recomputed from the same
        vertices, when it is split. The normalising scales are taken once (after
        the must-split phase) so additions and subtractions match. A leaf's
        priority is its largest normalised error. Returns False on a stop.
        """
        mesh = self.mesh
        if getattr(self, "_scales_fixed", None) is None:
            self._scales_fixed = (self._scales(), self._headline_signed())
            self._totals = None
        scales, head = self._scales_fixed
        while True:
            reason = self._out_of_budget()
            todo = np.flatnonzero(np.isnan(mesh.ind))
            if len(todo):
                mesh.ind[todo], add = self._error_sums(todo, scales, head)
                self._totals = add if self._totals is None else self._totals + add
            worst = (
                float(np.max(np.abs(self._totals))) if self._totals is not None else 0.0
            )
            self.history.append(
                {
                    "phase": "error",
                    "target": target,
                    "worst": worst,
                    "vertices": int(mesh.nv),
                    "leaves": int(len(mesh.lvl)),
                }
            )
            if worst <= target:
                return True
            if reason:
                self.stop = reason
                return False
            order = np.argsort(mesh.ind)[::-1]
            cap = batch or max(16, len(mesh.lvl) // 8)
            floor = min(max(16, len(mesh.lvl) // 50), len(order))
            # Enough of the worst cells to cover the excess, as if one quantity held it all.
            cum = np.cumsum(mesh.ind[order])
            take = int(np.searchsorted(cum, worst - 0.75 * target)) + 1
            idx = order[: max(min(take, cap), floor)]
            idx = idx[mesh.lvl[idx] < AdaptiveMesh.DEPTH - 2]
            if not len(idx):
                self.stop = "depth"
                return False
            self._totals = self._totals - self._error_sums(idx, scales, head)[1]
            mesh.split(idx, "error")

    # -- snapshots ----------------------------------------------------------
    def snapshot(
        self, name: str, mu_out: np.ndarray, kTs: Sequence[float]
    ) -> Dict[str, Any]:
        """Every integral at this mesh: Theta and the delta-rule at the output mu
        (T = 0), Theta on a kT/20 grid per temperature, and the mu-independent
        block integrals. Each is a sum over triangles, taken in leaf chunks."""
        mesh = self.mesh
        vals = mesh.vals
        out: Dict[str, Any] = {"vertices": int(mesh.nv), "leaves": int(len(mesh.lvl))}
        mu_out = np.asarray(mu_out, float)
        uniform = _is_uniform(mu_out)
        step0 = float(mu_out[1] - mu_out[0]) if len(mu_out) > 1 else 1.0
        grids = {float(kT): thermal_grid(mu_out, kT) for kT in kTs if kT > 0}
        for level, xkey, bkey in (("main", "X", "B"), ("diag", "X_diag", "B_diag")):
            if xkey not in vals:
                continue
            nX = vals[xkey].shape[-1]
            theta0 = np.zeros((len(mu_out), nX))
            delta0 = np.zeros((len(mu_out), nX))
            thermal = {kT: np.zeros((g[2], nX)) for kT, g in grids.items()}
            for part in self._chunks():
                e, x, a, _ = self._triangle_fields(
                    mesh.vid[part], mesh.lvl[part], vals[xkey], vals["E"]
                )
                es, xs = _sorted_triangles(e, x)  # once for every grid below
                if uniform:
                    mu0 = float(mu_out[0])
                    theta0 += _grid_rule("theta", es, xs, a, mu0, step0, len(mu_out))
                    delta0 += _grid_rule("delta", es, xs, a, mu0, step0, len(mu_out))
                else:
                    theta0 += triangle_theta(e, x, a, mu_out)
                    delta0 += triangle_delta(e, x, a, mu_out)
                for kT, (mu0, step, n) in grids.items():
                    thermal[kT] += _grid_rule("theta", es, xs, a, mu0, step, n)
            out[level] = {
                "blocks": _cell_block_integral(
                    mesh, mesh.vid, mesh.lvl, vals[bkey]
                ).sum(axis=0),
                "theta0": theta0,
                "delta0": delta0,
                "thermal": {
                    kT: (grids[kT][0], grids[kT][1], thermal[kT]) for kT in grids
                },
            }
        self.snapshots[name] = out
        return out

    def vertex_weights(self) -> np.ndarray:
        """Cartesian area per vertex (a third of each adjacent triangle)."""
        w = np.zeros(self.mesh.nv)
        for part in self._chunks():
            tri, area = self.mesh.triangles(self.mesh.vid[part], self.mesh.lvl[part])
            w += np.bincount(
                tri.ravel(), weights=np.repeat(area / 3.0, 3), minlength=self.mesh.nv
            )
        return w

    def smeared_check(
        self, curves: Dict[str, np.ndarray], mu_out: np.ndarray, kT: float
    ) -> Dict[str, Any]:
        """The Fermi-surface integrals at kT straight from the vertices, (-f') X
        taken at each vertex and integrated as linear on each triangle, against
        the curves the Theta convolution gave, at each headline mu: the relative
        difference of the DOS and of the dipole M, each against a floor of 1e-3
        of its largest value over the mu range. A mu whose thermal DOS is below
        ``SMEARED_DOS_FLOOR`` has no Fermi surface to check (None)."""
        mesh = self.mesh
        vals = mesh.vals
        w = self.vertex_weights()
        mu_out = np.asarray(mu_out, float)
        two_pi2 = (2.0 * math.pi) ** 2
        floor_dos = 1e-3 * float(np.abs(curves["dos"]).max())
        floor_M = 1e-3 * float(np.abs(curves["M"]).max())
        cols = np.r_[X_COUNT, np.arange(X_VG.start, X_VG.stop)]
        per_mu: Dict[str, Optional[float]] = {}
        for label, mu in self.headline_mu.items():
            if not mu_out[0] <= mu <= mu_out[-1]:
                continue
            j = int(np.argmin(np.abs(mu_out - mu)))
            acc = np.zeros(len(cols))
            for s in range(0, mesh.nv, VERTICES_PER_CHUNK):
                part = slice(s, min(mesh.nv, s + VERTICES_PER_CHUNK))
                weight = minus_fprime(vals["E"][part] - mu_out[j], kT) * w[part, None]
                acc += np.einsum("vn,vnc->c", weight, vals["X"][part][..., cols])
            dos = acc[0] / mesh.area_bz
            M = dipole_tensor(acc[1:] / two_pi2)
            if abs(float(curves["dos"][j])) < SMEARED_DOS_FLOOR:
                per_mu[label] = None
                continue
            rel_dos = abs(dos - float(curves["dos"][j])) / max(
                abs(float(curves["dos"][j])), floor_dos, 1e-300
            )
            rel_M = float(np.abs(M - curves["M"][j]).max()) / max(
                float(np.abs(curves["M"][j]).max()), floor_M, 1e-300
            )
            per_mu[label] = float(max(rel_dos, rel_M))
        known = [v for v in per_mu.values() if v is not None]
        return {
            "kT_eV": float(kT),
            "per_mu": per_mu,
            "rel": max(known) if known else None,
        }


def gap_bin_labels() -> List[str]:
    """'<1', '1-10', '10-100', '>=100' (meV) for GAP_BIN_EDGES_MEV."""
    edges = [f"{e:g}" for e in GAP_BIN_EDGES_MEV]
    return (
        [f"<{edges[0]}"]
        + [f"{lo}-{hi}" for lo, hi in zip(edges[:-1], edges[1:])]
        + [f">={edges[-1]}"]
    )


def source_fields(
    table: Table,
    kfrac: np.ndarray,
    *,
    window: Tuple[int, int],
    deg_tol: float,
    frozen: Optional[Tuple[float, float]] = None,
) -> Dict[str, np.ndarray]:
    """The window bands' velocities and their G split by partner gap (and the
    part from partners outside ``frozen``), for ``dipole_sources``."""
    res = evaluate(
        table,
        kfrac,
        window=window,
        occupied=[],
        deg_tol=deg_tol,
        gap_bins=True,
        frozen=frozen,
    )
    out = {"v": res["v"], "G_bins": res["G_bins"]}
    if frozen is not None:
        out["G_outside"] = res["G_outside_frozen"]
    return out


def dipole_sources(
    integ: "Integrator", mu_points: Dict[str, float], fields_fn
) -> Dict[str, Any]:
    """Where the T = 0 dipole at each mu comes from.

    The final mesh's triangles that straddle mu are evaluated again with G
    split by partner gap (``source_fields`` through ``fields_fn``, which maps
    fractional k to its output). Each bin's share is its (alpha, beta)
    projected on the total's over |total|^2, so the bins sum to 1 and a
    negative share opposes the total. ``outside_frozen`` is the share from
    partners outside the frozen window; it overlaps the bins. None where mu
    crosses no band. Where A3 is below 1e-3 of the Fermi-surface integral of
    |v G| (the integration's own resolution), there is no trigonal part to
    split and the row says so instead.
    """
    mesh = integ.mesh
    E = mesh.vals["E"]
    cells: Dict[str, np.ndarray] = {}
    for name, mu in mu_points.items():
        hits = [np.zeros(0, np.int64)]
        for part in integ._chunks():
            Ec = E[mesh.vid[part]]  # (cells, 5 vertices, window bands)
            hit = ((Ec.min(axis=1) < mu) & (Ec.max(axis=1) > mu)).any(axis=1)
            hits.append(np.flatnonzero(hit) + part.start)
        cells[name] = np.concatenate(hits)
    out: Dict[str, Any] = {name: None for name in mu_points}
    found = [c for c in cells.values() if len(c)]
    if not found:
        return out
    verts = np.unique(mesh.vid[np.unique(np.concatenate(found))].ravel())
    res = fields_fn(mesh.vertex_kfrac(verts))
    v = res["v"]

    def vg(G: np.ndarray) -> np.ndarray:  # v_a G_bc in the X_VG column order
        return np.stack(
            [v[..., a] * G[..., b, c] for a in range(2) for b, c in _SYM], -1
        )

    bins = [vg(res["G_bins"][..., j]) for j in range(res["G_bins"].shape[-1])]
    parts = bins + ([vg(res["G_outside"])] if "G_outside" in res else [])
    parts.append(np.abs(sum(bins)))  # the scale A3 is resolved against
    X = np.stack(parts, axis=-1)  # (vertices, bands, 6, parts)
    local = np.full(mesh.nv, -1, np.int64)
    local[verts] = np.arange(len(verts))
    nW, n_bins = E.shape[1], len(bins)
    labels = gap_bin_labels()
    two_pi2 = (2.0 * math.pi) ** 2
    for name, mu in mu_points.items():
        c = cells[name]
        if not len(c):
            continue
        tri, area = mesh.triangles(mesh.vid[c], mesh.lvl[c])
        e = np.moveaxis(E[tri], 2, 1).reshape(-1, 3)
        x = np.moveaxis(X[local[tri]], 2, 1).reshape(-1, 3, 6 * len(parts))
        fs = triangle_delta(e, x, np.repeat(area, nW), np.array([mu]))[0]
        fs = fs.reshape(6, len(parts)).T / two_pi2  # (parts, 6)
        tr = trigonal(dipole_tensor(fs[:-1]))
        a_t, b_t = float(tr["alpha"][:n_bins].sum()), float(tr["beta"][:n_bins].sum())
        A3 = math.hypot(a_t, b_t)
        row: Dict[str, Any] = {
            "mu_meV": float(mu) * 1e3,
            "triangles": int(len(tri)),
            "A3_A_per_eV": A3,
        }
        if A3 <= 1e-3 * float(fs[-1].max()):
            row["reason"] = (
                "A3 is below the integration's resolution: no trigonal part to split"
            )
        else:
            share = (tr["alpha"] * a_t + tr["beta"] * b_t) / (A3 * A3)
            row["shares"] = {lab: float(s) for lab, s in zip(labels, share[:n_bins])}
            if len(share) > n_bins:
                row["outside_frozen"] = float(share[n_bins])
        out[name] = row
    return out


def mu_subset(
    mu_range: Tuple[float, float],
    extrema: Sequence[float],
    extra: Sequence[float] = (),
    uniform: int = 101,
) -> np.ndarray:
    """The error-controlled mu: a uniform set over the range, every band edge or
    extremum inside it at +-0.05, 0.2 and 1 meV, and ``extra`` (E_F, mu_CNP)."""
    lo, hi = mu_range
    pts = list(np.linspace(lo, hi, uniform))
    for e in extrema:
        for d in (0.0, 5e-5, 2e-4, 1e-3):
            for s in (-1.0, 1.0):
                x = e + s * d
                if lo <= x <= hi:
                    pts.append(x)
    pts += [x for x in extra if lo <= x <= hi]
    return np.unique(np.round(np.asarray(pts, float), 12))


def adaptive_response(
    table: Table,
    *,
    mu_out: np.ndarray,
    temperatures_K: Sequence[float] = (),
    occupied_bands: Optional[int] = None,
    insulator_min_gap: float = 1e-4,
    deg_tol: float = 1e-6,
    tolerance: float = 1e-3,
    base: int = 64,
    max_points: int = 1_000_000,
    gap_ratio: float = 0.1,
    gap_floor: float = 5e-5,
    edge_resolution: float = 2e-4,
    occupation_mesh: int = 241,
    tau_diag: Optional[np.ndarray] = None,
    seeds: bool = True,
    deadline: Optional[float] = None,
    frame: Optional[Dict[str, np.ndarray]] = None,
    log=None,
    parallel=None,
    extra_kfrac: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Zone integrals of one model on the adaptive mesh, at tolerance and at
    tolerance / 2, as curves over ``mu_out`` (eV, relative to E_F) for T = 0 and
    each temperature. Returns the curves per stage and level with the
    occupation pass, the integration record and the mu-independent integrals.

    ``parallel``, when given, maps the work: ``parallel.energies(table)`` and
    ``parallel.payload(serial)`` return drop-in replacements that split their
    k-points over workers. ``extra_kfrac`` (a k-path) joins the occupation pass.
    """
    log = log or (lambda message: None)
    mu_out = np.asarray(mu_out, float)
    temps = [0.0] + sorted(float(x) for x in temperatures_K if float(x) > 0)
    kTs = [K_B_EV * T for T in temps]
    kT_max = max(kTs)
    mu_range = (float(mu_out[0]), float(mu_out[-1]))
    t0 = time.time()
    occ = occupation_pass(
        table,
        n=occupation_mesh,
        mu_range=mu_range,
        kT_max=kT_max,
        occupied_bands=occupied_bands,
        insulator_min_gap=insulator_min_gap,
        deg_tol=deg_tol,
        extra_kfrac=extra_kfrac,
        energies_fn=parallel.energies(table) if parallel is not None else None,
    )
    log(
        f"occupation pass: window {occ['window']}, n_occ {occ['n_occ']}, {len(occ['seeds'])} seeds ({time.time() - t0:.1f}s)"
    )
    lo, hi = occ["window"]
    n_occ = occ["n_occ"] if occ.get("S_applies") else None
    payload = make_payload(
        table, window=(lo, hi), n_occ=n_occ, tau_diag=tau_diag, deg_tol=deg_tol
    )
    if parallel is not None:
        payload = parallel.payload(payload)
    spacing = float(
        np.max(
            np.linalg.norm(
                reciprocal_in_plane(table.lattice, frame or layer_frame(table.lattice)),
                axis=1,
            )
        )
    )
    head = {"E_F": 0.0}
    if occ.get("mu_cnp") is not None and mu_range[0] <= occ["mu_cnp"] <= mu_range[1]:
        head["CNP"] = float(occ["mu_cnp"])
    subset = mu_subset(mu_range, occ["mu_seeds"], list(head.values()))
    integ = Integrator(
        payload,
        table.lattice,
        window=(lo, hi),
        n_occ=n_occ,
        s_applies=bool(n_occ),
        mu_range=mu_range,
        mu_subset=subset,
        headline_mu={k: v for k, v in head.items() if mu_range[0] <= v <= mu_range[1]},
        kT_max=kT_max,
        base=base,
        tolerance=tolerance,
        max_points=max_points,
        gap_ratio=gap_ratio,
        gap_floor=gap_floor,
        edge_resolution=edge_resolution,
        seeds=occ["seeds"] if seeds else None,
        # Half an occupation-grid spacing: a seeded pocket then holds a vertex.
        seed_diameter=0.5 * spacing / occupation_mesh,
        seed_spread=1.0 / occupation_mesh,
        deadline=deadline,
        frame=frame,
        log=log,
    )
    integ.must_split()
    log(f"must-split: {integ.mesh.nv} vertices, by rule {integ.mesh.vertices_by_rule}")
    stages: Dict[str, Any] = {}
    for name, target in (("tau", tolerance), ("tau2", 0.5 * tolerance)):
        ok = integ.refine(target)
        snap = integ.snapshot(name, mu_out, kTs)
        stage: Dict[str, Any] = {
            "ok": bool(ok),
            "vertices": snap["vertices"],
            "leaves": snap["leaves"],
            "levels": {},
        }
        for level in ("main", "diag"):
            if level not in snap:
                continue
            per_T = [
                curves_at(snap[level], mu_out, kT, area_bz=integ.mesh.area_bz, lo=lo)
                for kT in kTs
            ]
            stage["levels"][level] = {"per_T": per_T, "blocks": snap[level]["blocks"]}
        stages[name] = stage
        log(
            f"stage {name}: {'converged' if ok else 'stopped (' + str(integ.stop) + ')'} at {snap['vertices']} vertices"
        )
        if not ok:
            break
        if name == "tau":
            # The second stage also controls the A3 maximum the first stage found.
            M0 = stage["levels"]["main"]["per_T"][0]["M"]
            j = int(np.argmax(trigonal(M0)["A3"]))
            head = dict(integ.headline_mu)
            head["max_A3"] = float(mu_out[j])
            integ.restart(head)
    smeared = None
    final = stages.get("tau2") or stages.get("tau")
    if final is not None and kT_max > 0:
        smeared = integ.smeared_check(
            final["levels"]["main"]["per_T"][-1], mu_out, kT_max
        )
    return {
        "occupation": occ,
        "window": (lo, hi),
        "n_occ": occ["n_occ"],
        "S_applies": bool(n_occ),
        "temperatures_K": temps,
        "kT": kTs,
        "mu_out": mu_out,
        "stages": stages,
        "smeared_check": smeared,
        "integrator": integ,
        "seconds": time.time() - t0,
    }


def _is_uniform(mu: np.ndarray) -> bool:
    mu = np.asarray(mu, float)
    if len(mu) < 3:
        return len(mu) >= 1
    d = np.diff(mu)
    return bool(np.allclose(d, d[0], rtol=1e-9, atol=1e-15) and d[0] > 0)


def thermal_grid(mu_out: np.ndarray, kT: float) -> Tuple[float, float, int]:
    """(mu0, step, n) of a kT/20 grid over the output range +-20 kT: the -f'
    bin weights are exact, and treating Theta as linear inside a bin costs
    O((step/kT)^2) ~ 2e-4."""
    mu_out = np.asarray(mu_out, float)
    step = kT / 20.0
    mu0 = float(mu_out[0]) - 20.0 * kT - step
    n = int(math.ceil((float(mu_out[-1]) - float(mu_out[0]) + 40.0 * kT) / step)) + 3
    return mu0, step, n


def curves_at(
    entry: Dict[str, Any], mu_out: np.ndarray, kT: float, *, area_bz: float, lo: int
) -> Dict[str, np.ndarray]:
    """Output curves at one temperature from a snapshot level.

    T = 0: Fermi-sea values are Theta at mu, Fermi-surface values the delta
    rule at mu. T > 0: the exact identity sum int f X = int dmu' (-f') Theta(mu')
    with bin weights f(mu_i - mu) - f(mu_{i+1} - mu), normalised; Fermi-surface
    values convolve the bin differences of Theta.
    """
    mu_out = np.asarray(mu_out, float)
    if kT <= 0:
        sea, fs = entry["theta0"], entry["delta0"]
    else:
        mu0, step, theta = entry["thermal"][float(kT)]
        grid = mu0 + step * np.arange(theta.shape[0])
        mid = 0.5 * (theta[1:] + theta[:-1])
        diff = (theta[1:] - theta[:-1]) / step
        sea = np.zeros((len(mu_out), theta.shape[1]))
        fs = np.zeros((len(mu_out), theta.shape[1]))
        for s in range(0, len(mu_out), 128):
            mus = mu_out[s : s + 128]
            f = fermi((grid[None, :] - mus[:, None]) / kT)
            w = f[:, :-1] - f[:, 1:]
            w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
            sea[s : s + 128] = w @ mid
            fs[s : s + 128] = w @ diff
    blocks = entry["blocks"]
    two_pi2 = (2.0 * math.pi) ** 2
    return {
        "count": lo + sea[:, X_COUNT] / area_bz,
        "sigma_xy_e2_h": -(sea[:, X_OMEGA] + blocks[B_OMEGA_C]) / (2.0 * math.pi),
        "dos": fs[:, X_COUNT] / area_bz,
        "M": dipole_tensor(fs[:, X_VG] / two_pi2),
        "BCD": fs[:, X_VO] / two_pi2,
        "drude": symmetric_tensor(fs[:, X_VV] / two_pi2),
    }


# ----------------------------------------------------------------- tb.dat
TB_FILENAME = "wannier90_tb.dat"
EIG_FILENAME = "wannier90.eig"


def tb_layout(path: str) -> Dict[str, Any]:
    """The header of a wannier90 ``_tb.dat`` and the exact size its body must have.

    Layout (wannier90 3.1 ``hamiltonian_write_tb``): a date line; three lattice
    rows (Angstrom, list-directed, so read by tokens); num_wann; nrpts; ndegen
    at 15 per line; then per R a blank line, the R line and N^2 lines
    ``j i Re Im`` of H, and then the same for r with ``j i xRe xIm yRe yIm zRe
    zIm``. A format's trailing ``1x`` writes nothing at the end of a record, so
    the H line length is measured and the r line length derived from the size.
    """
    with open(path, "rb") as handle:
        handle.readline()
        reals: List[float] = []
        while len(reals) < 9:
            line = handle.readline()
            if not line:
                raise ValueError("truncated _tb.dat header")
            reals += [float(x) for x in line.split()]
        num_wann = int(handle.readline().split()[0])
        nrpts = int(handle.readline().split()[0])
        ndegen: List[int] = []
        while len(ndegen) < nrpts:
            line = handle.readline()
            if not line:
                raise ValueError("truncated _tb.dat degeneracies")
            ndegen += [int(x) for x in line.split()]
        header = handle.tell()
        handle.readline()  # blank
        handle.readline()  # first R
        h_line = len(handle.readline())
    actual = os.path.getsize(path)
    nn = num_wann * num_wann
    body = actual - header - nrpts * 34
    r_line = body // (nrpts * nn) - h_line if body > 0 and nrpts and nn else 0
    expected = header + nrpts * (34 + (h_line + r_line) * nn)
    return {
        "lattice": np.asarray(reals[:9], float).reshape(3, 3),
        "num_wann": num_wann,
        "nrpts": nrpts,
        "ndegen": np.asarray(ndegen[:nrpts], int),
        "header_bytes": header,
        "h_line_bytes": h_line,
        "r_line_bytes": r_line,
        "expected_bytes": expected,
        "actual_bytes": actual,
        "lines_ok": h_line in (45, 46) and r_line == h_line + 64,
    }


def read_tb(path: str, layout: Dict[str, Any]) -> Dict[str, Any]:
    """Stream ``_tb.dat``: R (nR, 3), H (nR, N, N), r (nR, N, N, 2) in Cartesian
    x and y, all as written (not divided by ndegen) and indexed [j-1, i-1] =
    <0j|X|Ri>; plus sum |r_z|^2 off the (0, n, n) diagonal (r_z is not kept)."""
    n, nr = layout["num_wann"], layout["nrpts"]
    nn = n * n
    lh, lr = layout["h_line_bytes"], layout["r_line_bytes"]
    R = np.zeros((nr, 3), int)
    H = np.zeros((nr, n, n), complex)
    r = np.zeros((nr, n, n, 2), complex)
    rz2 = 0.0
    with open(path, "rb") as handle:
        handle.seek(layout["header_bytes"])
        for k in range(nr):
            handle.readline()
            R[k] = [int(x) for x in handle.readline().split()]
            v = np.array(handle.read(lh * nn).split(), dtype=float).reshape(n, n, 4)
            H[k] = (v[:, :, 2] + 1j * v[:, :, 3]).T
        for k in range(nr):
            handle.readline()
            Rk = [int(x) for x in handle.readline().split()]
            if Rk != R[k].tolist():
                raise ValueError(
                    f"the r block's R {Rk} differs from the H block's {R[k].tolist()}"
                )
            v = np.array(handle.read(lr * nn).split(), dtype=float).reshape(n, n, 8)
            r[k, :, :, 0] = (v[:, :, 2] + 1j * v[:, :, 3]).T
            r[k, :, :, 1] = (v[:, :, 4] + 1j * v[:, :, 5]).T
            rz = (v[:, :, 6] + 1j * v[:, :, 7]).T
            if not R[k].any():
                rz = rz - np.diag(np.diag(rz))
            rz2 += float(np.sum(np.abs(rz) ** 2)) / float(layout["ndegen"][k])
    return {"R": R, "H": H, "r": r, "rz2_offdiag": rz2}


def hermitize(R: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, float]:
    """M(R) <- [M(R) + M(-R)^dagger] / 2 over the leading R axis and the next two
    matrix axes (trailing axes allowed); returns it and max |anti-Hermitian part|."""
    index = {tuple(int(x) for x in row): i for i, row in enumerate(np.asarray(R, int))}
    out = np.empty_like(M)
    worst = 0.0
    for key, i in index.items():
        j = index.get(tuple(-x for x in key))
        if j is None:
            raise ValueError(f"-R of {list(key)} is missing")
        partner = np.conj(np.swapaxes(M[j], 0, 1))
        out[i] = 0.5 * (M[i] + partner)
        worst = max(worst, float(np.max(np.abs(0.5 * (M[i] - partner)))))
    return out, worst


def inplane_mismatch(a: np.ndarray, b: np.ndarray, lattice: np.ndarray) -> float:
    """Largest in-plane distance (Angstrom) between two centre sets modulo a1 and a2."""
    lat2 = np.asarray(lattice, float)[:2, :2]
    d = (np.asarray(a, float) - np.asarray(b, float))[:, :2]
    frac = d @ np.linalg.inv(lat2)
    frac -= np.round(frac)
    return float(np.max(np.linalg.norm(frac @ lat2, axis=1))) if len(d) else 0.0


def parse_eig(text: str) -> np.ndarray:
    """wannier90 ``.eig`` lines ``band k E`` -> E (nk, nbands) in eV."""
    rows = np.array(
        [line.split()[:3] for line in text.splitlines() if line.strip()], dtype=float
    )
    nb, nk = int(rows[:, 0].max()), int(rows[:, 1].max())
    E = np.full((nk, nb), np.nan)
    E[rows[:, 1].astype(int) - 1, rows[:, 0].astype(int) - 1] = rows[:, 2]
    return E


def dft_exactness(
    E_tb: np.ndarray, E_dft: np.ndarray, frozen: Tuple[float, float]
) -> Dict[str, Any]:
    """TB against DFT eigenvalues (eV, same zero) inside the frozen window at the
    DFT mesh points, where a Wannier model reproduces the frozen states exactly."""
    lo, hi = frozen
    worst = 0.0
    states = 0
    counts_match = True
    for e_tb, e_dft in zip(np.asarray(E_tb, float), np.asarray(E_dft, float)):
        a = np.sort(e_dft[(e_dft >= lo) & (e_dft <= hi)])
        b = np.sort(e_tb[(e_tb >= lo - 1e-3) & (e_tb <= hi + 1e-3)])
        if len(b) < len(a):
            counts_match = False
            continue
        # Each frozen DFT state has a TB partner; extra TB states near the edges are fine.
        dist = np.abs(a[:, None] - b[None, :]).min(axis=1) if len(a) else np.zeros(0)
        worst = max(worst, float(dist.max()) if len(dist) else 0.0)
        states += len(a)
    return {
        "k_points": int(len(E_dft)),
        "states": states,
        "counts_match": counts_match,
        "max_meV": worst * 1e3,
    }


def density_cm2(count_minus_nocc: np.ndarray, area_A2: float) -> np.ndarray:
    """Carriers per cm^2 from electrons per cell (1 Angstrom^2 = 1e-16 cm^2)."""
    return np.asarray(count_minus_nocc, float) / (area_A2 * 1e-16)


__all__: List[str] = [
    "CHI_MA_NM_PER_V2_PER_A_PER_EV",
    "CHI_SI_PER_A_PER_EV",
    "CONVENTION_SETS",
    "E2_OVER_HBAR_S",
    "E2_OVER_H_S",
    "K_B_EV",
    "LEVEL_DIAGONAL",
    "LEVEL_FULL",
    "Table",
    "a3_misalignment",
    "adaptive_response",
    "c3_residual_matrix",
    "c3_residual_tensor",
    "c3_rotate_frac",
    "cell_area",
    "chi_all_conventions",
    "chi_from_dipole",
    "compact",
    "decimate",
    "degenerate_clusters",
    "density_cm2",
    "dft_exactness",
    "dipole_sources",
    "gap_bin_labels",
    "hermitize",
    "inplane_mismatch",
    "evaluate",
    "evaluate_chunks",
    "fermi",
    "geometry",
    "is_hexagonal",
    "kcart",
    "layer_frame",
    "minus_fprime",
    "parse_eig",
    "read_tb",
    "reciprocal_in_plane",
    "rotate_tensor",
    "source_fields",
    "sum_rule_residual",
    "theta_on_grid",
    "delta_on_grid",
    "triangle_delta",
    "triangle_theta",
    "tb_layout",
    "trigonal",
    "trigonal_tensor",
    "uniform_grid",
    "uniform_response",
    "units",
    "write_json_compact",
    "zone_area",
]
