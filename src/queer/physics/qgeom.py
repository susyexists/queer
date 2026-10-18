"""Quantum-geometry arithmetic of a two-dimensional Wannier model.

The Bloch sums are wannier90's convention II on the Wigner-Seitz-corrected,
folded table, with E_F subtracted: H(k) = sum_R e^{ik.R} H(R). k is Cartesian in
rad/Å (2 pi included) and hbar = 1 inside band formulas, so v = dE/dk is in eV Å,
positions and connections in Å, g and Omega in Å², G in Å²/eV and
[dk] = d^2k/(2 pi)^2. Vector components are taken in the layer frame (e1, e2): e1
along a1 projected on the layer plane, e2 = n x e1, n along a1 x a2.

Positions: A^W_a(k) = sum_R e^{ik.R} r_a(R) with r_a,ij(R) = <0i|r_a|Rj>, and
Omega^W(k) = sum_R e^{ik.R} i(R_1 r_2(R) - R_2 r_1(R)) (Wang, Yates, Souza and
Vanderbilt, PRB 74, 195118 (2006), Eqs. 25 and 37). The diagonal level is
r_a(R) = diag(tau_a) delta_R0, the tight-binding model. In the eigenbasis
H = U E U^dag: V_a = U^dag dH_a U, Abar_a = U^dag A^W_a U, and
A^a_nm = Abar^a_nm + i V^a_nm / (E_m - E_n) between bands of different degenerate
clusters. Omega = -2 Im Q, and carriers have charge -e.

The intrinsic second-order response is chi_abc = -(e^3/hbar)[c_out M_abc +
c_in (M_bac + M_cab)] of the Berry-connection-polarizability (BCP) dipole
M_abc = sum_n int [dk] (-f') v_a G_bc; :data:`CONVENTION_SETS` holds the published
coefficient sets. The triangle rules integrate theta(mu - E) X and
delta(mu - E) X exactly for E and X linear on each triangle.

These are the arithmetic of the Generator's ``pythtb_qgeom`` kernel, operation for
operation; :mod:`queer.observables.qgeom` holds its tables, meshes and integrator.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np

from .constants import E_CHARGE, H_PLANCK, HBAR, K_BOLTZMANN_EV

K_B_EV = K_BOLTZMANN_EV
E2_OVER_HBAR_S = E_CHARGE**2 / HBAR
E2_OVER_H_S = E_CHARGE**2 / H_PLANCK
# (e^3/hbar) times a dipole of 1 Å/eV, in A m / V^2: the eV in the denominator
# cancels one e and the Ångström is 1e-10 m.
CHI_SI_PER_A_PER_EV = E2_OVER_HBAR_S * 1e-10
# The same in mA nm / V^2.
CHI_MA_NM_PER_V2_PER_A_PER_EV = CHI_SI_PER_A_PER_EV * 1e12

# The published (c_out, c_in) coefficient sets of chi_abc.
CONVENTION_SETS: Mapping[str, tuple[float, float]] = MappingProxyType(
    {
        # wave-packet dynamics: Gao, Yang and Niu, PRL 112, 166601 (2014)
        "wpd_1_m05": (1.0, -0.5),
        # Luttinger-Kohn: Kaplan, Holder and Yan, Eq. 12c
        "lka_2_m05": (2.0, -0.5),
        # quantum kinetics: Das et al., PRB 108, L201405 (2023)
        "qk_05_m1": (0.5, -1.0),
    }
)
# Partner-gap bins (meV) of the dipole sources: <1, 1-10, 10-100, >=100.
GAP_BIN_EDGES_MEV: tuple[float, ...] = (1.0, 10.0, 100.0)

LEVEL_FULL = "full"
LEVEL_DIAGONAL = "diagonal"

# Per-band integrand columns carried by every vertex: the count, Omega, v_a G_bc
# for (a; bc) = (x; xx), (x; xy), (x; yy), (y; xx), (y; xy), (y; yy), v_a Omega and
# v_a v_b for xx, xy, yy.
NX = 13
X_COUNT = 0
X_OMEGA = 1
X_VG = slice(2, 8)
X_VO = slice(8, 10)
X_VV = slice(10, 13)
# Per-vertex mu-independent block columns: Omega of the always-occupied bands below
# the window, Omega_S, Omega_S,in and g_S,in xx, xy, yy of the occupied manifold.
NB = 6
B_OMEGA_C = 0
B_OMEGA_S = 1
B_OMEGA_S_IN = 2
B_G_S = slice(3, 6)
SYMMETRIC_PAIRS: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 1))
# The Fermi-surface (delta-rule) columns: DOS, M (6) and BCD (2); and the Fermi-sea ones.
FS_COLUMNS = np.r_[X_COUNT, np.arange(2, 10)]
SEA_COLUMNS = np.array([X_COUNT, X_OMEGA])

# Batch sizes that bound the temporaries however large the mesh grows: (triangle, mu)
# pairs evaluated at once and (piece, block) pairs of the cubic-piece accumulator.
PAIRS_PER_CHUNK = 2_000_000
BLOCK_PAIRS_PER_CHUNK = 100_000
# Grid points per block of the cubic-piece accumulator. Its pieces are at least a
# step wide, so rounding stays near 1e-16 GRID_BLOCK^3 (2e-10) of a piece's own scale.
GRID_BLOCK = 128


# ------------------------------------------------------------------ the layer
def layer_frame(lattice: Sequence[Sequence[float]] | np.ndarray) -> dict[str, np.ndarray]:
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


def a3_misalignment(lattice: Sequence[Sequence[float]] | np.ndarray) -> float:
    """|a3_hat x n_hat|: zero when a3 is along the layer normal."""
    lat = np.asarray(lattice, float).reshape(3, 3)
    normal = np.cross(lat[0], lat[1])
    a3 = lat[2] / float(np.linalg.norm(lat[2]))
    return float(np.linalg.norm(np.cross(a3, normal / float(np.linalg.norm(normal)))))


def cell_area(lattice: Sequence[Sequence[float]] | np.ndarray) -> float:
    """|a1 x a2| in Å²."""
    lat = np.asarray(lattice, float).reshape(3, 3)
    return float(np.linalg.norm(np.cross(lat[0], lat[1])))


def zone_area(lattice: Sequence[Sequence[float]] | np.ndarray) -> float:
    """The in-plane Brillouin-zone area (2 pi)^2 / A_cell in Å^-2."""
    return (2.0 * math.pi) ** 2 / cell_area(lattice)


def reciprocal_in_plane(
    lattice: Sequence[Sequence[float]] | np.ndarray, frame: Mapping[str, np.ndarray]
) -> np.ndarray:
    """b1 and b2 (2 pi included) as rows of frame components (2, 2)."""
    lat = np.asarray(lattice, float).reshape(3, 3)
    recip = 2.0 * math.pi * np.linalg.inv(lat).T
    return np.stack([recip[:2] @ frame["e1"], recip[:2] @ frame["e2"]], axis=1)


def frame_k(kfrac: np.ndarray, binv: np.ndarray) -> np.ndarray:
    """Frame components of k (rad/Å) from in-plane fractional k (nk, 2) and the rows b1, b2."""
    return np.asarray(kfrac, float)[:, :2] @ binv


def is_hexagonal(lattice: Sequence[Sequence[float]] | np.ndarray, tol: float = 1e-4) -> bool:
    """|a1| = |a2| with a 60 or 120 degree angle."""
    lat = np.asarray(lattice, float).reshape(3, 3)
    a, b = float(np.linalg.norm(lat[0])), float(np.linalg.norm(lat[1]))
    if abs(a - b) > tol * a:
        return False
    cosang = float(lat[0] @ lat[1]) / (a * b)
    return abs(abs(cosang) - 0.5) < tol


def c3_rotate_frac(kfrac: np.ndarray, lattice: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Fractional k rotated by 120 degrees about the layer normal."""
    lat = np.asarray(lattice, float).reshape(3, 3)
    frame = layer_frame(lat)
    b = reciprocal_in_plane(lat, frame)
    c, s = math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3)
    rot = np.array([[c, -s], [s, c]])
    kc = np.asarray(kfrac, float)[:, :2] @ b
    out = (kc @ rot.T) @ np.linalg.inv(b)
    return np.concatenate([out, np.zeros((len(out), 1))], axis=1)


# ------------------------------------------------------------ band geometry
def degenerate_clusters(E: np.ndarray, tol: float) -> np.ndarray:
    """Cluster label per band (nk, N): consecutive bands closer than ``tol`` share one."""
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


def layer_geometry(
    H: np.ndarray,
    dH: Sequence[np.ndarray],
    *,
    AW: Sequence[np.ndarray] | None = None,
    OW: np.ndarray | None = None,
    tau: np.ndarray | None = None,
    window: tuple[int, int],
    occupied: Iterable[int] = (),
    deg_tol: float = 1e-6,
    gap_bins: bool = False,
    frozen: tuple[float, float] | None = None,
    eig: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Band and manifold geometry from Hermitian H(k) (nk, N, N) and its derivatives dH = (dH_1, dH_2).

    Positions: ``AW`` = (A^W_1, A^W_2) and ``OW`` = Omega^W at the full level, or the
    centres ``tau`` (N, 2) at the diagonal level; neither means none. The bands n in
    ``window`` = [lo, hi) get the velocity ``v`` (nk, nW, 2), the in-W metric ``g`` and
    the BCP ``G`` (nk, nW, 2, 2) and the complete curvature ``omega`` (nk, nW), each
    summed over every partner m outside n's degenerate cluster (``deg_tol``, eV);
    ``v_pair`` is |V_{n, n+1}|. Each cut c of ``occupied`` gives the bands [0, c)
    their in-W tensor Q_in and complete curvature under ``blocks[c]``. ``gap_bins``
    splits G by partner gap (:data:`GAP_BIN_EDGES_MEV`); ``frozen`` = (lo, hi) eV adds
    the share from partners outside that window. ``eig`` = (E, U) reuses an
    eigensystem of H.
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
        centres = np.zeros((n, 2)) if tau is None else np.asarray(tau, float).reshape(n, 2)
        Ab = [_rows_diag(U, centres[:, 0], rows), _rows_diag(U, centres[:, 1], rows)]
    cid = degenerate_clusters(E, deg_tol)
    out: dict[str, Any] = {"E": E, "cluster": cid}

    # The Omega-bar diagonal: [U^dag Omega^W U]_nn + 2 Im sum_m Abar_x,nm conj(Abar_y,nm);
    # exactly zero at the diagonal level, where diag(tau_x) and diag(tau_y) commute.
    if AW is not None:
        bar = 2.0 * np.imag(np.sum(Ab[0] * np.conj(Ab[1]), axis=-1))
        if omega_w is not None:
            bar = bar + omega_w
    else:
        bar = np.zeros((nk, top))

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
    nxt = np.minimum(np.arange(lo, hi) + 1, n - 1)
    out["v_pair"] = np.sqrt(
        sum(np.abs(np.take_along_axis(V[a][:, w], nxt[None, :, None], axis=2)[..., 0]) ** 2 for a in range(2))
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
            out["G_outside_frozen"] = np.where(sel[..., None, None], contrib, 0.0).sum(axis=2)

    blocks: dict[int, dict[str, np.ndarray]] = {}
    for c in cuts:
        S = slice(0, c)
        C = slice(c, n)
        dSC = E[:, None, C] - E[:, S, None]  # E_m - E_n, n in S, m above
        # A closed gap at the cut makes the block inapplicable there (the caller
        # judges ``gap``); it is never divided by.
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


def integrand_fields(geometry: Mapping[str, Any]) -> np.ndarray:
    """The per-band integrand columns (..., :data:`NX`) of a :func:`layer_geometry` result."""
    v, G, om = geometry["v"], geometry["G"], geometry["omega"]
    X = np.empty(om.shape + (NX,))
    X[..., X_COUNT] = 1.0
    X[..., X_OMEGA] = om
    col = 2
    for a in range(2):
        for b, c in SYMMETRIC_PAIRS:
            X[..., col] = v[..., a] * G[..., b, c]
            col += 1
    X[..., 8] = v[..., 0] * om
    X[..., 9] = v[..., 1] * om
    X[..., 10] = v[..., 0] ** 2
    X[..., 11] = v[..., 0] * v[..., 1]
    X[..., 12] = v[..., 1] ** 2
    return X


def manifold_columns(geometry: Mapping[str, Any], lo: int, n_occ: int | None) -> np.ndarray:
    """The mu-independent block columns (nk, :data:`NB`) of a :func:`layer_geometry` result."""
    nk = geometry["E"].shape[0]
    B = np.zeros((nk, NB))
    if lo > 0:
        B[:, B_OMEGA_C] = geometry["blocks"][lo]["omega"]
    if n_occ:
        blk = geometry["blocks"][n_occ]
        B[:, B_OMEGA_S] = blk["omega"]
        B[:, B_OMEGA_S_IN] = blk["omega_in"]
        B[:, 3] = blk["g_in"][:, 0, 0]
        B[:, 4] = blk["g_in"][:, 0, 1]
        B[:, 5] = blk["g_in"][:, 1, 1]
    return B


# ---------------------------------------------------------------- occupations
def fermi_function(x: np.ndarray) -> np.ndarray:
    """1 / (e^x + 1), overflow-safe."""
    x = np.clip(np.asarray(x, float), -700.0, 700.0)
    return 0.5 * (1.0 - np.tanh(0.5 * x))


def minus_fermi_derivative(e: np.ndarray, kT: float) -> np.ndarray:
    """-df/dE at energy ``e`` (eV, relative to mu) and ``kT`` (eV)."""
    x = np.clip(np.asarray(e, float) / kT, -700.0, 700.0)
    return 0.25 / kT / np.cosh(0.5 * x) ** 2


def density_cm2(count_minus_nocc: np.ndarray, area_A2: float) -> np.ndarray:
    """Carriers per cm^2 from electrons per cell (1 Å² = 1e-16 cm^2)."""
    return np.asarray(count_minus_nocc, float) / (area_A2 * 1e-16)


def is_uniform_grid(mu: np.ndarray) -> bool:
    """Whether ``mu`` is an increasing uniform grid (one or two points always are)."""
    mu = np.asarray(mu, float)
    if len(mu) < 3:
        return len(mu) >= 1
    d = np.diff(mu)
    return bool(np.allclose(d, d[0], rtol=1e-9, atol=1e-15) and d[0] > 0)


def thermal_grid(mu_out: np.ndarray, kT: float) -> tuple[float, float, int]:
    """(mu0, step, n) of a kT/20 grid over the output range +-20 kT.

    The -f' bin weights are exact, and treating Theta as linear inside a bin costs
    O((step/kT)^2) ~ 2e-4.
    """
    mu_out = np.asarray(mu_out, float)
    step = kT / 20.0
    mu0 = float(mu_out[0]) - 20.0 * kT - step
    n = int(math.ceil((float(mu_out[-1]) - float(mu_out[0]) + 40.0 * kT) / step)) + 3
    return mu0, step, n


def mu_subset(
    mu_range: tuple[float, float],
    extrema: Sequence[float],
    extra: Sequence[float] = (),
    uniform: int = 101,
) -> np.ndarray:
    """The error-controlled mu: a uniform set over the range, every band edge or extremum inside it at
    +-0.05, 0.2 and 1 meV, and ``extra`` (E_F, mu_CNP)."""
    lo, hi = mu_range
    pts: list[Any] = list(np.linspace(lo, hi, uniform))
    for e in extrema:
        for d in (0.0, 5e-5, 2e-4, 1e-3):
            for s in (-1.0, 1.0):
                x = e + s * d
                if lo <= x <= hi:
                    pts.append(x)
    pts += [x for x in extra if lo <= x <= hi]
    return np.unique(np.round(np.asarray(pts, float), 12))


# --------------------------------------------------------------------- tensors
def chi_from_dipole(M: np.ndarray, c_out: float, c_in: float) -> np.ndarray:
    """chi_abc / (e^3/hbar) = -[c_out M_abc + c_in (M_bac + M_cab)] from M (..., 2, 2, 2)."""
    M = np.asarray(M, float)
    # M_bac swaps the first two indices; M_cab moves the first index last.
    return -(c_out * M + c_in * (np.swapaxes(M, -3, -2) + np.moveaxis(M, -3, -1)))


def chi_all_conventions(M: np.ndarray) -> dict[str, np.ndarray]:
    """chi in mA nm / V^2 for every coefficient set of :data:`CONVENTION_SETS`."""
    return {
        name: chi_from_dipole(M, c_out, c_in) * CHI_MA_NM_PER_V2_PER_A_PER_EV
        for name, (c_out, c_in) in CONVENTION_SETS.items()
    }


def trigonal(M: np.ndarray) -> dict[str, Any]:
    """The m = 3 part of an in-plane rank-3 tensor symmetric in (b, c).

    Z = sum M_abc zeta_a zeta_b zeta_c with zeta = (1, i) gives alpha + i beta = Z/4:
    alpha = (M_xxx - M_xyy - 2 M_yxy)/4, beta = (M_yxx + 2 M_xxy - M_yyy)/4,
    A3 = |alpha + i beta| and theta0 = atan2(beta, alpha)/3 (degrees). Rotating M by
    phi about the normal multiplies alpha + i beta by e^{3 i phi}.
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
    """M_abc rotated by ``phi`` (radians) about the normal."""
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
    return float(np.linalg.norm(M - trigonal_tensor(float(t["alpha"]), float(t["beta"]))) / norm)


def c3_residual_matrix(K: np.ndarray) -> float:
    """|K - (tr K / 2) 1| / |K|: zero for a C3-symmetric rank-2 tensor."""
    K = np.asarray(K, float)
    norm = float(np.linalg.norm(K))
    if norm == 0.0:
        return 0.0
    return float(np.linalg.norm(K - 0.5 * np.trace(K) * np.eye(2)) / norm)


def dipole_tensor(vg: np.ndarray) -> np.ndarray:
    """M (..., 2, 2, 2) from the six v_a G_bc columns."""
    vg = np.asarray(vg, float)
    M = np.zeros(vg.shape[:-1] + (2, 2, 2))
    col = 0
    for a in range(2):
        for b, c in SYMMETRIC_PAIRS:
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


def gap_bin_labels() -> list[str]:
    """'<1', '1-10', '10-100' and '>=100' (meV) for :data:`GAP_BIN_EDGES_MEV`."""
    edges = [f"{e:g}" for e in GAP_BIN_EDGES_MEV]
    return [f"<{edges[0]}"] + [f"{lo}-{hi}" for lo, hi in zip(edges[:-1], edges[1:])] + [f">={edges[-1]}"]


# --------------------------------------------------------------- triangle rules
def sorted_triangles(e: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vertex energies (nt, 3) sorted ascending per triangle, X (nt, 3, nX) permuted alike."""
    order = np.argsort(e, axis=1, kind="stable")
    es = np.take_along_axis(e, order, axis=1)
    Xs = np.take_along_axis(X, order[:, :, None], axis=1)
    return es, Xs


def _safe(x: np.ndarray) -> np.ndarray:
    return np.where(np.abs(x) > 0.0, x, 1.0)


def theta_triangles(es: np.ndarray, Xs: np.ndarray, area: np.ndarray, m: float) -> np.ndarray:
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
        xm = (3.0 * X1[i] + t12[:, None] * (X2[i] - X1[i]) + t13[:, None] * (X3[i] - X1[i])) / 3.0
        val[i] = sub[:, None] * xm
    high = (m > e2) & (m < e3)
    if high.any():
        i = np.flatnonzero(high)
        s31 = (e3[i] - m) / _safe(e3[i] - e1[i])
        s32 = (e3[i] - m) / _safe(e3[i] - e2[i])
        sub = area[i] * s31 * s32
        xm = (3.0 * X3[i] + s31[:, None] * (X1[i] - X3[i]) + s32[:, None] * (X2[i] - X3[i])) / 3.0
        val[i] = full[i] - sub[:, None] * xm
    return val


def delta_triangles(es: np.ndarray, Xs: np.ndarray, area: np.ndarray, m: float) -> np.ndarray:
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
        mean = X1[i] + 0.5 * (t12[:, None] * (X2[i] - X1[i]) + t13[:, None] * (X3[i] - X1[i]))
        val[i] = dens[:, None] * mean
    high = (m > e2) & (m < e3)
    if high.any():
        i = np.flatnonzero(high)
        s31 = (e3[i] - m) / _safe(e3[i] - e1[i])
        s32 = (e3[i] - m) / _safe(e3[i] - e2[i])
        dens = 2.0 * area[i] * (e3[i] - m) / _safe((e3[i] - e1[i]) * (e3[i] - e2[i]))
        mean = X3[i] + 0.5 * (s31[:, None] * (X1[i] - X3[i]) + s32[:, None] * (X2[i] - X3[i]))
        val[i] = dens[:, None] * mean
    return val


def triangle_theta(
    e: np.ndarray, X: np.ndarray, area: np.ndarray, mu: np.ndarray, per_triangle: bool = False
) -> np.ndarray:
    """sum_t int_t theta(mu - E) X dA for E and X linear on each triangle.

    ``e`` (nt, 3) vertex energies, ``X`` (nt, 3, nX), ``area`` (nt,) and ``mu`` (nmu,).
    Returns (nmu, nX), or (nmu, nt, nX) with ``per_triangle``. Exact: below the middle
    vertex the region E < mu is a sub-triangle, above it the complement of one.
    """
    es, Xs = sorted_triangles(np.asarray(e, float), np.asarray(X, float))
    area = np.asarray(area, float)
    vals = [theta_triangles(es, Xs, area, float(m)) for m in np.asarray(mu, float).ravel()]
    if per_triangle:
        return np.stack(vals)
    return np.stack([v.sum(axis=0) for v in vals])


def triangle_delta(
    e: np.ndarray, X: np.ndarray, area: np.ndarray, mu: np.ndarray, per_triangle: bool = False
) -> np.ndarray:
    """sum_t int_t delta(mu - E) X dA, the mu-derivative of :func:`triangle_theta`.

    For linear E the iso-line crosses two edges, and X averages over its endpoints.
    """
    es, Xs = sorted_triangles(np.asarray(e, float), np.asarray(X, float))
    area = np.asarray(area, float)
    vals = [delta_triangles(es, Xs, area, float(m)) for m in np.asarray(mu, float).ravel()]
    if per_triangle:
        return np.stack(vals)
    return np.stack([v.sum(axis=0) for v in vals])


def accumulate_rows(theta: np.ndarray, j: np.ndarray, val: np.ndarray) -> None:
    """theta[j] += val, column by column."""
    n = theta.shape[0]
    for c in range(val.shape[1]):
        theta[:, c] += np.bincount(j, weights=val[:, c], minlength=n)[:n]


def range_pairs(first: np.ndarray, count: np.ndarray, size: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Batches (owner, offset) over the ranges [first, first + count): ``owner`` indexes the ranges and
    ``offset`` runs along each one."""
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


def piece_coefficients(kind: str, low: bool, es: np.ndarray, Xs: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Each triangle's Theta (or its mu-derivative, the delta rule) on one piece of (e1, e3) as a cubic
    sum_k c_k s^k: the low piece (e1, e2] in s = mu - e1, the high piece (e2, e3) in s = e3 - mu.
    Returns c (nt, 4, nX)."""
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


def add_cubic_pieces(
    out: np.ndarray,
    coef: np.ndarray,
    anchor: np.ndarray,
    sign: float,
    first: np.ndarray,
    stop: np.ndarray,
    mu0: float,
    step: float,
) -> None:
    """out[j] += sum_k coef_k (sign (mu_j - anchor))^k over j in [first, stop) of each piece, with
    mu_j = mu0 + j step.

    A piece enters each block of :data:`GRID_BLOCK` points it touches once, as a change
    of the cubic's coefficients in the block's own coordinate where it starts and the
    reverse where it stops; a running sum along each block then holds the active cubic
    at every point. The cost is per block, not per grid point, and the short
    coordinate bounds the rounding.
    """
    n, nX = out.shape
    B = GRID_BLOCK
    nblocks = -(-n // B)
    g = coef * ((sign * step) ** np.arange(4.0))[None, :, None]
    p = (np.asarray(anchor, float) - mu0) / step  # the anchor in grid units
    block0 = first // B
    span = (stop - 1) // B - block0 + 1
    acc = np.zeros(nblocks * (B + 1) * 4 * nX)
    cols = np.arange(4 * nX)
    for k, off in range_pairs(block0, span, BLOCK_PAIRS_PER_CHUNK):
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
    vals = active[:, :, 0] + u * (active[:, :, 1] + u * (active[:, :, 2] + u * active[:, :, 3]))
    out += vals.reshape(nblocks * B, nX)[:n]


def rule_at(kind: str, es: np.ndarray, Xs: np.ndarray, area: np.ndarray, m: np.ndarray) -> np.ndarray:
    """:func:`theta_triangles` (``kind`` "theta") or :func:`delta_triangles` ("delta") with a per-row mu."""
    e1, e2, e3 = es[:, 0], es[:, 1], es[:, 2]
    X1, X2, X3 = Xs[:, 0], Xs[:, 1], Xs[:, 2]
    val = np.zeros((len(es), Xs.shape[-1]))
    low = (m > e1) & (m <= e2)
    high = (m > e2) & (m < e3)
    if kind == "theta":
        full = area[:, None] * (X1 + X2 + X3) / 3.0
        val[m >= e3] = full[m >= e3]
        if low.any():
            q = np.flatnonzero(low)
            t12 = (m[q] - e1[q]) / _safe(e2[q] - e1[q])
            t13 = (m[q] - e1[q]) / _safe(e3[q] - e1[q])
            sub = area[q] * t12 * t13
            xm = (3.0 * X1[q] + t12[:, None] * (X2[q] - X1[q]) + t13[:, None] * (X3[q] - X1[q])) / 3.0
            val[q] = sub[:, None] * xm
        if high.any():
            q = np.flatnonzero(high)
            s31 = (e3[q] - m[q]) / _safe(e3[q] - e1[q])
            s32 = (e3[q] - m[q]) / _safe(e3[q] - e2[q])
            sub = area[q] * s31 * s32
            xm = (3.0 * X3[q] + s31[:, None] * (X1[q] - X3[q]) + s32[:, None] * (X2[q] - X3[q])) / 3.0
            val[q] = full[q] - sub[:, None] * xm
    else:
        low &= e3 > e1
        if low.any():
            q = np.flatnonzero(low)
            t12 = (m[q] - e1[q]) / _safe(e2[q] - e1[q])
            t13 = (m[q] - e1[q]) / _safe(e3[q] - e1[q])
            dens = 2.0 * area[q] * (m[q] - e1[q]) / _safe((e2[q] - e1[q]) * (e3[q] - e1[q]))
            mean = X1[q] + 0.5 * (t12[:, None] * (X2[q] - X1[q]) + t13[:, None] * (X3[q] - X1[q]))
            val[q] = dens[:, None] * mean
        if high.any():
            q = np.flatnonzero(high)
            s31 = (e3[q] - m[q]) / _safe(e3[q] - e1[q])
            s32 = (e3[q] - m[q]) / _safe(e3[q] - e2[q])
            dens = 2.0 * area[q] * (e3[q] - m[q]) / _safe((e3[q] - e1[q]) * (e3[q] - e2[q]))
            mean = X3[q] + 0.5 * (s31[:, None] * (X1[q] - X3[q]) + s32[:, None] * (X2[q] - X3[q]))
            val[q] = dens[:, None] * mean
    return val


def grid_rule(
    kind: str,
    es: np.ndarray,
    Xs: np.ndarray,
    area: np.ndarray,
    mu0: float,
    step: float,
    n: int,
    pairs_per_chunk: int = PAIRS_PER_CHUNK,
) -> np.ndarray:
    """The theta or delta rule on mu_j = mu0 + j step for triangles with sorted vertex energies: (n, nX).

    A triangle wholly below mu_j enters by one cumulative sum; inside (e1, e3) each
    triangle is two cubic pieces added per block of grid points. A piece narrower than
    a step holds a grid point or two and is evaluated there instead, since its cubic's
    coefficients grow as 1/width^3.
    """
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
        full = area[:, None] * Xs.sum(axis=1) / 3.0
        steps = np.zeros((n + 1, nX))
        for c in range(nX):
            steps[:, c] = np.bincount(hi, weights=full[:, c], minlength=n + 1)[: n + 1]
        out += np.cumsum(steps, axis=0)[:n]
    thin_t, thin_first, thin_count = [], [], []
    for low, first, stop, width in (
        (True, lo, mid, es[:, 1] - es[:, 0]),
        (False, mid, hi, es[:, 2] - es[:, 1]),
    ):
        has = stop > first
        wide = np.flatnonzero(has & (width >= step))
        if len(wide):
            add_cubic_pieces(
                out,
                piece_coefficients(kind, low, es[wide], Xs[wide], area[wide]),
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
    for k, off in range_pairs(first, np.concatenate(thin_count), pairs_per_chunk):
        t_id = tri[k]
        j = first[k] + off
        val = rule_at(kind, es[t_id], Xs[t_id], area[t_id], mu0 + j * step)
        accumulate_rows(out, j, val)
    return out


def theta_on_grid(
    e: np.ndarray,
    X: np.ndarray,
    area: np.ndarray,
    mu0: float,
    step: float,
    n: int,
    pairs_per_chunk: int = PAIRS_PER_CHUNK,
) -> np.ndarray:
    """:func:`triangle_theta` on the uniform grid mu_j = mu0 + j step (j < n), scaling with the triangles
    rather than with n. Returns (n, nX)."""
    es, Xs = sorted_triangles(np.asarray(e, float), np.asarray(X, float))
    return grid_rule("theta", es, Xs, area, mu0, step, n, pairs_per_chunk)


def delta_on_grid(
    e: np.ndarray,
    X: np.ndarray,
    area: np.ndarray,
    mu0: float,
    step: float,
    n: int,
    pairs_per_chunk: int = PAIRS_PER_CHUNK,
) -> np.ndarray:
    """:func:`triangle_delta` on the uniform grid mu_j = mu0 + j step (j < n)."""
    es, Xs = sorted_triangles(np.asarray(e, float), np.asarray(X, float))
    return grid_rule("delta", es, Xs, area, mu0, step, n, pairs_per_chunk)


def grouped_rule(
    kind: str,
    e: np.ndarray,
    X: np.ndarray,
    area: np.ndarray,
    mu: np.ndarray,
    group: np.ndarray,
    ngroups: int,
) -> np.ndarray:
    """theta- or delta-rule integrals summed per group: (nmu, ngroups, nX).

    As :func:`theta_on_grid` on an arbitrary mu set: a triangle wholly below mu_j
    enters through one cumulative sum, and only the (triangle, mu) pairs with mu inside
    the triangle's energy range get the partial formula.
    """
    es, Xs = sorted_triangles(np.asarray(e, float), np.asarray(X, float))
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
                steps[:, c] = np.bincount(hi * ngroups + group, weights=full[:, c], minlength=size)
            flat += np.cumsum(steps.reshape(nmu + 1, ngroups, nX), axis=0)[:nmu].reshape(-1, nX)
        count = np.maximum(hi - lo, 0)
        tri = np.flatnonzero(count)
        if len(tri):
            starts = np.cumsum(count[tri]) - count[tri]
            total = int(count[tri].sum())
            for s in range(0, total, PAIRS_PER_CHUNK):
                idx = np.arange(s, min(total, s + PAIRS_PER_CHUNK))
                owner = np.searchsorted(starts, idx, side="right") - 1
                t_id = tri[owner]
                j = lo[t_id] + (idx - starts[owner])
                val = rule_at(kind, es[t_id], Xs[t_id], area[t_id], ms[j])
                accumulate_rows(flat, j * ngroups + group[t_id], val)
    out = np.empty((nmu, ngroups, nX))
    out[order] = flat.reshape(nmu, ngroups, nX)
    return out


def curves_from_integrals(
    entry: Mapping[str, Any], mu_out: np.ndarray, kT: float, *, area_bz: float, lo: int
) -> dict[str, np.ndarray]:
    """Output curves at one temperature from the integrals of one level.

    ``entry`` holds ``theta0`` and ``delta0`` (the rules at ``mu_out``), ``thermal``
    {kT: (mu0, step, Theta on that grid)} and ``blocks``. T = 0: Fermi-sea values are
    Theta at mu and Fermi-surface values the delta rule at mu. T > 0: the exact
    identity sum int f X = int dmu' (-f') Theta(mu') with bin weights
    f(mu_i - mu) - f(mu_{i+1} - mu), normalised; Fermi-surface values convolve the bin
    differences of Theta. Returns the electron ``count``, ``sigma_xy_e2_h``, ``dos``,
    ``M`` (Å/eV), ``BCD`` (Å) and the Drude weight ``drude`` (eV).
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
            f = fermi_function((grid[None, :] - mus[:, None]) / kT)
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


# ---------------------------------------------------------- position matrices
def hermitize_translations(R: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, float]:
    """M(R) <- [M(R) + M(-R)^dagger] / 2 over the leading R axis and the next two matrix axes
    (trailing axes allowed); returns it and the largest |anti-Hermitian part|.

    A translation whose negative is missing raises ``ValueError``.
    """
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
    """Largest in-plane distance (Å) between two centre sets modulo a1 and a2."""
    lat2 = np.asarray(lattice, float)[:2, :2]
    d = (np.asarray(a, float) - np.asarray(b, float))[:, :2]
    frac = d @ np.linalg.inv(lat2)
    frac -= np.round(frac)
    return float(np.max(np.linalg.norm(frac @ lat2, axis=1))) if len(d) else 0.0


def dft_exactness(E_tb: np.ndarray, E_dft: np.ndarray, frozen: tuple[float, float]) -> dict[str, Any]:
    """Tight-binding against DFT eigenvalues (eV, same zero) inside the frozen window at the DFT mesh
    points, where a Wannier model reproduces the frozen states exactly.

    Each frozen DFT state needs a partner level within 1 meV of the window; extra
    levels near the edges are allowed. ``max_meV`` is the largest distance from a
    frozen state to its nearest partner.
    """
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
        dist = np.abs(a[:, None] - b[None, :]).min(axis=1) if len(a) else np.zeros(0)
        worst = max(worst, float(np.max(dist)) if len(dist) else 0.0)
        states += len(a)
    return {
        "k_points": int(len(E_dft)),
        "states": states,
        "counts_match": counts_match,
        "max_meV": worst * 1e3,
    }
