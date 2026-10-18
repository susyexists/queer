"""Quantum geometry and the intrinsic nonlinear Hall response of a 2D Wannier model.

:class:`QgeomTable` stacks one position level's Fourier table: H, i R_1 H and
i R_2 H and, at the full level, r_1, r_2 and i (R_1 r_2 - R_2 r_1), so one GEMM per
chunk of k gives every matrix (:mod:`queer.physics.qgeom` states the conventions).
:func:`evaluate` gives the band and manifold geometry at fractional k,
:func:`occupation_pass` the band window, electron count, gaps, refinement seeds
and PT flag on a Gamma-centred grid, and :class:`Integrator` the zone integrals on
an adaptive Gamma-centred quadtree of triangles: must-split rules (seeds, small
gaps with large interband velocity, band edges), then refinement by an embedded
error estimate to a tolerance and on to half of it, with exact triangle rules at
T = 0 and an exact -f' bin convolution at each temperature.
:func:`adaptive_response` runs the whole integration; :func:`dipole_sources` splits
the T = 0 dipole by partner gap.

The integration stops when the mesh reaches ``max_points`` vertices ("budget"),
when a ``deadline`` (``time.time()``) passes ("time"), or, for a deterministic
replay of a run that stopped on its time budget, when the mesh reaches
``stop_vertices`` vertices, reported as that run's "time" stop. Without a
deadline the vertex set and every integral depend only on the inputs.

Beside the kernel's arithmetic, and without changing the mesh or the history, QUEER
records the decisions rounding can flip (:meth:`Integrator.decision_record`): the
occupation pass's seeds (:func:`seed_ties`), each must-split round, each refinement
step and the second stage's A3 maximum, with every decision whose margin to its
threshold lies within its band (TIE_ROUNDING of the run's spectral radius or largest
velocity, TIE_INDICATOR of the compared indicator), per rule an additive digest of the
leaves split (:func:`set_digest`) and the range of vertex ids their splits created, and
the controlled quantities whose scale is rounding (they vanish identically, so their
error estimates are rounding noise).

:func:`layer_band_geometry` and :func:`nonlinear_hall_response` return
:class:`~queer.observables.result.ObservableResult` arrays with dimensions, units
and diagnostics. This is the Generator's ``pythtb_qgeom`` kernel, operation for
operation.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from ..physics.qgeom import (
    B_OMEGA_C,
    B_OMEGA_S,
    FS_COLUMNS,
    K_B_EV,
    LEVEL_DIAGONAL,
    LEVEL_FULL,
    SEA_COLUMNS,
    SYMMETRIC_PAIRS,
    X_OMEGA,
    X_VG,
    c3_rotate_frac,
    curves_from_integrals,
    dipole_tensor,
    fermi_function,
    gap_bin_labels,
    grid_rule,
    grouped_rule,
    integrand_fields,
    is_hexagonal,
    is_uniform_grid,
    layer_frame,
    layer_geometry,
    manifold_columns,
    minus_fermi_derivative,
    mu_subset,
    reciprocal_in_plane,
    sorted_triangles,
    thermal_grid,
    triangle_delta,
    triangle_theta,
    trigonal,
    zone_area,
)
from .result import ObservableResult

# Triangle-bands per leaf batch of the zone-wide sums and vertices per batch of the
# smeared cross-check.
TRIANGLES_PER_CHUNK = 262_144
VERTICES_PER_CHUNK = 131_072
# States per eV per cell below which a mu has no Fermi surface to cross-check.
SMEARED_DOS_FLOOR = 1e-4
# The per-vertex buffer holding each vertex's integer key (x * unit + y).
VERTEX_KEY = "_key"
STOP_BUDGET = "budget"
STOP_TIME = "time"
STOP_DEPTH = "depth"

# The decision record (QUEER's, not the kernel's): a comparison whose margin lies within its
# rounding band is "tied", since another BLAS kernel may decide it the other way. An energy or
# velocity band is TIE_ROUNDING times the run's own scale of that quantity (the spectral radius
# over the occupation grid, the largest band or interband velocity of the base mesh); the error
# indicators, their running sums and the A3 maximum are compared to TIE_INDICATOR of their value.
TIE_ROUNDING = 1e-12
TIE_INDICATOR = 1e-7
# Tied decisions listed per run at most; beyond it the record keeps its counts and digests only.
TIE_LIST_MAX = 20_000
# A cell decision's outcomes, as indices into this tuple (bit i of a mask: outcome i possible).
OUTCOMES = ("none", "seed", "gap", "edge", "error")
MUST_SPLIT_RULES = ("seed", "gap", "edge")
# The Fermi-surface columns of the error estimate, as :data:`FS_COLUMNS`.
FS_NAMES = (
    ("count",)
    + tuple(f"v_{'xy'[a]} G_{'xy'[b]}{'xy'[c]}" for a in range(2) for b, c in SYMMETRIC_PAIRS)
    + ("v_x omega", "v_y omega")
)
SEED_KINDS = ("gap", "extremum")
_MASK64 = (1 << 64) - 1


def key_hashes(keys: Any) -> np.ndarray:
    """splitmix64 of int64 keys, as uint64 (wrapping arithmetic)."""
    z = np.asarray(keys, np.int64).ravel().astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def set_digest(keys: Any) -> str:
    """An order-free digest of a set of int64 keys: the sum of their splitmix64 hashes modulo 2^64, in hex.

    Digests of disjoint sets add, so a known part of a set can be taken out of its digest.
    """
    total = int(np.sum(key_hashes(keys), dtype=np.uint64)) if np.size(keys) else 0
    return f"{total & _MASK64:016x}"


def _tri(margin: Any, band: Any) -> tuple[np.ndarray, np.ndarray]:
    """For a decision "margin > 0" (or ">= 0") known to within +-band: where it surely holds and where
    it possibly holds."""
    m = np.asarray(margin, float)
    return m > band, m >= -np.asarray(band, float)


def seed_id(index: int, kind: str, b0: int, b1: int) -> int:
    """A seed's integer id: its grid index, kind and band (pair)."""
    return ((int(index) * 4 + SEED_KINDS.index(kind)) * 4096 + int(b0)) * 4096 + int(b1)


Payload = Callable[[np.ndarray], Mapping[str, np.ndarray]]


# ----------------------------------------------------------------- the table
class QgeomTable:
    """One position level's stacked Fourier table.

    ``R`` (nR, 3) integer translations and ``H`` (nR, N, N) eV with E_F already
    subtracted, the cell ``lattice`` (Å rows) and the layer ``frame``. The full level
    takes ``r`` (nR, N, N, 2), the position matrices' frame components on the same R
    set as H (unfold them together); the diagonal level takes the centres ``tau``
    (N, 2) in frame components. Per R the stack holds H, i R_1 H, i R_2 H and, at the
    full level, r_1, r_2 and i (R_1 r_2 - R_2 r_1); :meth:`fourier` hermitizes each
    matrix at k.
    """

    def __init__(
        self,
        R: np.ndarray,
        H: np.ndarray,
        lattice: Sequence[Sequence[float]] | np.ndarray,
        frame: Mapping[str, np.ndarray],
        *,
        tau: np.ndarray | None = None,
        r: np.ndarray | None = None,
    ):
        self.R = np.asarray(R, int).reshape(-1, 3)
        H = np.asarray(H, complex)
        self.lattice = np.asarray(lattice, float).reshape(3, 3)
        self.frame = dict(frame)
        self.n = int(H.shape[1])
        Rc = self.R @ self.lattice
        self.R12 = np.stack([Rc @ frame["e1"], Rc @ frame["e2"]], axis=1)
        r1 = self.R12[:, 0, None, None]
        r2 = self.R12[:, 1, None, None]
        blocks = [H, 1j * r1 * H, 1j * r2 * H]
        self.level = LEVEL_DIAGONAL
        self.tau: np.ndarray | None = None
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
        self.stack = np.stack(blocks, axis=1).reshape(len(self.R), self.T * self.n * self.n)
        self.binv = reciprocal_in_plane(self.lattice, frame)

    @classmethod
    def from_model(cls, model: Any, frame: Mapping[str, np.ndarray] | None = None, *, level: str = "auto"):
        """The table of a :class:`~queer.models.wannier.WannierModel` (its translations, hoppings, E_F,
        centres and, at the full level, position matrices).

        ``level`` ``auto`` takes the full level when the model has position matrices,
        ``diagonal`` the centres. The full level reads the in-plane Cartesian
        components of r and needs the layer normal along z.
        """
        if level not in ("auto", LEVEL_FULL, LEVEL_DIAGONAL):
            raise ValueError("level must be auto, full or diagonal")
        lattice = np.asarray(model.lattice, float)
        frame = dict(frame or layer_frame(lattice))
        R = np.asarray(model.translations)
        H = np.asarray(model.hopping_matrices, complex).copy()
        home = np.flatnonzero(~R.any(axis=1))
        if not len(home):
            raise ValueError("the table has no R = 0 block")
        H[int(home[0])] -= float(model.fermi_energy) * np.eye(H.shape[1])
        position = getattr(model, "position_matrices", None)
        if level == LEVEL_FULL and position is None:
            raise ValueError("the full level needs position matrices")
        if position is not None and level != LEVEL_DIAGONAL:
            if abs(abs(frame["normal"][2]) - 1.0) > 1e-6:
                raise ValueError("the full level needs the layer normal along z")
            proj = np.stack([frame["e1"][:2], frame["e2"][:2]], axis=1)
            return cls(R, H, lattice, frame, r=np.asarray(position)[..., :2] @ proj)
        if model.centres is None:
            raise ValueError("the diagonal level needs the Wannier centres")
        tau_cart = np.asarray(model.centres, float) @ np.linalg.inv(lattice) @ lattice
        tau = np.stack([tau_cart @ frame["e1"], tau_cart @ frame["e2"]], axis=1)
        return cls(R, H, lattice, frame, tau=tau)

    @property
    def nbytes(self) -> int:
        return int(self.stack.nbytes)

    def energies(self, kfrac: np.ndarray) -> np.ndarray:
        """Eigenvalues (nk, N) at fractional k from the H block of the stack."""
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


def evaluate(table: QgeomTable, kfrac: np.ndarray, **kwargs: Any) -> dict[str, Any]:
    """:func:`~queer.physics.qgeom.layer_geometry` at fractional k (nk, 3) of a table's level."""
    F = table.fourier(kfrac)
    if table.level == LEVEL_FULL:
        return layer_geometry(F[:, 0], (F[:, 1], F[:, 2]), AW=(F[:, 3], F[:, 4]), OW=F[:, 5], **kwargs)
    return layer_geometry(F[:, 0], (F[:, 1], F[:, 2]), tau=table.tau, **kwargs)


def evaluate_chunks(table: QgeomTable, kfrac: np.ndarray, chunk: int = 256, **kwargs: Any) -> dict[str, Any]:
    """:func:`evaluate` over chunks of ``chunk`` points, concatenated."""
    parts = [evaluate(table, kfrac[s : s + chunk], **kwargs) for s in range(0, len(kfrac), chunk)]
    out: dict[str, Any] = {}
    for key in parts[0]:
        if key == "blocks":
            out["blocks"] = {
                c: {k: np.concatenate([p["blocks"][c][k] for p in parts]) for k in parts[0]["blocks"][c]}
                for c in parts[0]["blocks"]
            }
        else:
            out[key] = np.concatenate([p[key] for p in parts])
    return out


def sum_rule_residual(table: QgeomTable, kfrac: np.ndarray, deg_tol: float = 1e-6) -> float:
    """max |sum_n G^n| over all bands relative to max |G^n|: exactly zero."""
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


def uniform_response(
    table: QgeomTable,
    n: int,
    *,
    window: tuple[int, int],
    mus: Sequence[float],
    kT: float = 0.0,
    occupied: Sequence[int] = (),
    deg_tol: float = 1e-6,
    chunk: int = 256,
) -> dict[str, Any]:
    """Zone integrals on the Gamma-centred n x n grid with direct weights.

    kT = 0: Fermi-sea sums with theta(mu - E) (gapped mu only); kT > 0: -f' weights.
    Returns per mu: M (2, 2, 2) in Å/eV, BCD (2) in Å, ``sigma_xy_e2_h`` of the window
    bands plus the always-occupied bands [0, lo) (included when lo > 0), the DOS
    (states/eV/cell), the Drude weight (2, 2) in eV and the electron count; and per
    manifold cut its C, C_in and K_in.
    """
    k = uniform_grid(n)
    lo, hi = int(window[0]), int(window[1])
    cuts = set(int(c) for c in occupied)
    if lo > 0:
        cuts.add(lo)
    res = evaluate_chunks(table, k, chunk=chunk, window=window, occupied=sorted(cuts), deg_tol=deg_tol)
    area = zone_area(table.lattice)
    weight = area / (2.0 * math.pi) ** 2 / len(k)  # [dk] per point
    E = res["E"][:, lo:hi]
    mu_values = np.asarray(mus, float)
    out: dict[str, Any] = {
        "mu": mu_values,
        "M": [],
        "BCD": [],
        "sigma_xy_e2_h": [],
        "dos": [],
        "drude": [],
        "count": [],
    }
    base_omega = res["blocks"][lo]["omega"].sum() if lo > 0 else 0.0
    for mu in mu_values:
        if kT > 0:
            wfs = minus_fermi_derivative(E - mu, kT)
            occ = fermi_function((E - mu) / kT)
        else:
            wfs = np.zeros_like(E)
            occ = (E < mu).astype(float)
        M = np.einsum("kn,kna,knbc->abc", wfs, res["v"], res["G"]) * weight
        D = np.einsum("kn,kna,kn->a", wfs, res["v"], res["omega"]) * weight
        sea = np.sum(occ * res["omega"]) + base_omega
        out["M"].append(M)
        out["BCD"].append(D)
        out["sigma_xy_e2_h"].append(-sea * area / len(k) / (2.0 * math.pi))
        out["dos"].append(np.sum(wfs) / len(k))
        out["drude"].append(np.einsum("kn,kna,knb->ab", wfs, res["v"], res["v"]) * weight)
        out["count"].append(lo + np.sum(occ) / len(k))
    for key in ("M", "BCD", "sigma_xy_e2_h", "dos", "drude", "count"):
        out[key] = np.asarray(out[key])
    out["manifold"] = {
        c: {
            "C": float(res["blocks"][c]["omega"].sum() * area / len(k) / (2.0 * math.pi)),
            "C_in": float(res["blocks"][c]["omega_in"].sum() * area / len(k) / (2.0 * math.pi)),
            "K_in": res["blocks"][c]["g_in"].sum(axis=0) * area / len(k) / (2.0 * math.pi),
            "psd_min_rel": float(res["blocks"][c]["psd_min_rel"].min()),
            "min_gap": float(res["blocks"][c]["gap"].min()),
        }
        for c in res["blocks"]
    }
    return out


# ------------------------------------------------------------- vertex payload
def make_payload(
    table: QgeomTable,
    *,
    window: tuple[int, int],
    n_occ: int | None = None,
    tau_diag: np.ndarray | None = None,
    deg_tol: float = 1e-6,
) -> Payload:
    """The per-vertex values the integrator needs, at both levels from one eigensolve.

    The primary level is the table's; with a full table and ``tau_diag`` the
    diagonal level rides along on the same eigensystem.
    """
    lo, hi = int(window[0]), int(window[1])
    cuts = [c for c in (lo, n_occ) if c]

    def payload(kfrac: np.ndarray) -> dict[str, np.ndarray]:
        F = table.fourier(kfrac)
        E, U = np.linalg.eigh(F[:, 0])
        common: dict[str, Any] = dict(window=(lo, hi), occupied=cuts, deg_tol=deg_tol, eig=(E, U))
        if table.level == LEVEL_FULL:
            main = layer_geometry(F[:, 0], (F[:, 1], F[:, 2]), AW=(F[:, 3], F[:, 4]), OW=F[:, 5], **common)
        else:
            main = layer_geometry(F[:, 0], (F[:, 1], F[:, 2]), tau=table.tau, **common)
        out = {
            "E": E[:, lo:hi],
            "E_above": E[:, min(hi, table.n - 1)],
            "v": main["v"],
            "v_pair": main["v_pair"],
            "X": integrand_fields(main),
            "B": manifold_columns(main, lo, n_occ),
        }
        if n_occ:
            out["gap_S"] = main["blocks"][n_occ]["gap"]
            out["v_S"] = main["blocks"][n_occ]["v_cut"]
            out["psd_S"] = main["blocks"][n_occ]["psd_min_rel"]
        if table.level == LEVEL_FULL and tau_diag is not None:
            diag = layer_geometry(F[:, 0], (F[:, 1], F[:, 2]), tau=tau_diag, **common)
            out["X_diag"] = integrand_fields(diag)
            out["B_diag"] = manifold_columns(diag, lo, n_occ)
        return out

    return payload


def local_extrema(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Masks of strict local minima and maxima on a periodic 2D grid (eight neighbours)."""
    nbrs = [np.roll(np.roll(grid, dx, 0), dy, 1) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
    stack = np.stack(nbrs)
    return grid < stack.min(axis=0), grid > stack.max(axis=0)


def extremum_margins(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The margins of :func:`local_extrema`'s strict tests: the neighbours' minimum less the value (a
    minimum where positive) and the value less the neighbours' maximum (a maximum where positive)."""
    nbrs = [np.roll(np.roll(grid, dx, 0), dy, 1) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
    stack = np.stack(nbrs)
    return stack.min(axis=0) - grid, grid - stack.max(axis=0)


def occupation_pass(
    table: QgeomTable,
    *,
    n: int,
    mu_range: tuple[float, float],
    kT_max: float,
    occupied_bands: int | None = None,
    insulator_min_gap: float = 1e-4,
    deg_tol: float = 1e-6,
    hot_gap: float = 0.05,
    extra_kfrac: np.ndarray | None = None,
    c3_points: int = 512,
    chunk: int = 4096,
    energies_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Energies on a Gamma-centred n x n grid (plus ``extra_kfrac``): the band window, the electron
    count, the gaps, the refinement seeds, the PT flag and the Wannier C3 band floor.

    Energies are relative to E_F (eV); ``mu_range`` and ``kT_max`` are in eV. The
    window holds every band reaching the range +-20 kT_max widened by 0.1 eV, its cuts
    moved to global gaps; ``n_occ`` is ``occupied_bands`` or the rounded count below
    E_F; seeds are the local gap minima below ``hot_gap`` and the band extrema in the
    widened range.

    ``ties`` (QUEER's decision record, the returned values unchanged) receives the seed
    decisions rounding can flip: :func:`seed_ties`.
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
        # mu deep in a gap: the band just below it (or the lowest) is the window; the
        # response is then the always-occupied block's.
        below = int(np.sum(emax < lo_e))
        lo = max(below - 1, 0)
        hi = lo + 1
    while lo > 0 and emax[lo - 1] >= emin[lo] - 1e-9:
        lo -= 1
    while hi < nb and emin[hi] <= emax[hi - 1] + 1e-9:
        hi += 1
    count_ef = float(np.mean(np.sum(E < 0.0, axis=1)))
    n_occ = int(occupied_bands) if occupied_bands else int(round(count_ef))
    frac = count_ef - math.floor(count_ef)
    out: dict[str, Any] = {
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
        out["min_direct_gap_k"] = k[int(np.argmin(direct[: len(k)]))] if len(direct) else None
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
                float(flat[int(round(n_occ * len(k))) - 1]) if n_occ * len(k) <= len(flat) else 0.0
            )
    else:
        out["S_applies"] = False
    seeds: list[np.ndarray] = []
    seed_info: list[dict[str, Any]] = []
    mu_seeds: list[float] = []
    grid_shape = (n, n)
    pairs = [(b, b + 1) for b in range(lo, min(hi, nb - 1))]
    if out.get("S_applies") is not None and 0 < n_occ < nb:
        pairs.append((n_occ - 1, n_occ))
    for b0, b1 in sorted(set(pairs)):
        gap = (E[:, b1] - E[:, b0]).reshape(grid_shape)
        mins, _ = local_extrema(gap)
        for idx in np.flatnonzero(mins.ravel() & (gap.ravel() < hot_gap) & (gap.ravel() > deg_tol)):
            seeds.append(k[idx])
            seed_info.append({"kind": "gap", "bands": [b0, b1], "gap_meV": float(gap.ravel()[idx]) * 1e3})
    for b in range(lo, hi):
        band = E[:, b].reshape(grid_shape)
        mins, maxs = local_extrema(band)
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
    if ties is not None:
        ties.update(
            seed_ties(
                E,
                sorted(set(pairs)),
                (lo, hi),
                grid_shape,
                hot_gap=hot_gap,
                deg_tol=deg_tol,
                extended=(lo_e, hi_e),
                mu_range=mu_range,
                radius=float(max(np.abs(emin).max(), np.abs(emax).max())),
            )
        )
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


def seed_ties(
    E: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    window: tuple[int, int],
    grid_shape: tuple[int, int],
    *,
    hot_gap: float,
    deg_tol: float,
    extended: tuple[float, float],
    mu_range: tuple[float, float],
    radius: float,
) -> dict[str, Any]:
    """The occupation pass's seed decisions and those rounding can flip (QUEER's record).

    A seed is a strict local minimum of a pair's gap below ``hot_gap`` and above ``deg_tol``,
    or a strict local extremum of a window band inside the ``extended`` range; its energy
    joins the controlled mu inside ``mu_range``. An eigenvalue is known to within
    ``TIE_ROUNDING`` times the spectral ``radius``: a band value to that band, a gap or a
    neighbour difference of band values to twice it, a neighbour difference of gaps to four
    times it. Returns the seed count, the digest of their ids (:func:`seed_id`), the number
    of decisions judged and the tied ones as [grid index, kind, b0, b1, taken].
    """
    band = TIE_ROUNDING * radius
    ids: list[int] = []
    tied: list[list[int]] = []
    judged = 0
    mu_seeds = mu_tied = 0
    for b0, b1 in pairs:
        gap = (E[:, b1] - E[:, b0]).reshape(grid_shape)
        mins, _ = local_extrema(gap)
        g = gap.ravel()
        taken = mins.ravel() & (g < hot_gap) & (g > deg_tol)
        lower, _ = extremum_margins(gap)
        s1, p1 = _tri(lower.ravel(), 4.0 * band)
        s2, p2 = _tri(hot_gap - g, 2.0 * band)
        s3, p3 = _tri(g - deg_tol, 2.0 * band)
        sure, possible = s1 & s2 & s3, p1 & p2 & p3
        ids += [seed_id(int(i), "gap", b0, b1) for i in np.flatnonzero(taken)]
        tied += [[int(i), 0, b0, b1, int(taken[i])] for i in np.flatnonzero(possible & ~sure)]
        judged += len(g)
    for b in range(*window):
        grid = E[:, b].reshape(grid_shape)
        mins, maxs = local_extrema(grid)
        e = grid.ravel()
        taken = (mins | maxs).ravel() & (e >= extended[0]) & (e <= extended[1])
        lower, upper = extremum_margins(grid)
        smin, pmin = _tri(lower.ravel(), 2.0 * band)
        smax, pmax = _tri(upper.ravel(), 2.0 * band)
        slo, plo = _tri(e - extended[0], band)
        shi, phi = _tri(extended[1] - e, band)
        sure, possible = (smin | smax) & slo & shi, (pmin | pmax) & plo & phi
        ids += [seed_id(int(i), "extremum", b, b) for i in np.flatnonzero(taken)]
        tied += [[int(i), 1, b, b, int(taken[i])] for i in np.flatnonzero(possible & ~sure)]
        judged += len(e)
        in_mu = taken & (e >= mu_range[0]) & (e <= mu_range[1])
        m0s, m0p = _tri(e - mu_range[0], band)
        m1s, m1p = _tri(mu_range[1] - e, band)
        mu_seeds += int(np.count_nonzero(in_mu))
        mu_tied += int(np.count_nonzero((possible & m0p & m1p) & ~(sure & m0s & m1s)))
    return {
        "count": len(ids),
        "digest": set_digest(np.asarray(ids, np.int64)),
        "judged": judged,
        "band_eV": band,
        "tied_count": len(tied),
        "tied": tied,
        "mu_seeds": mu_seeds,
        "mu_seeds_tied": mu_tied,
    }


# ---------------------------------------------------------------- the quadtree
class AdaptiveMesh:
    """A Gamma-centred quadtree over [-1/2, 1/2)^2 in fractional k.

    Each leaf cell is four triangles from its centre to its edges. Vertex coordinates
    are integers in units of 1/(base 2^DEPTH) and wrap periodically, so a vertex is
    evaluated once however many cells share it. Hanging vertices are harmless: each
    triangle is integrated on its own.
    """

    # Levels below a base cell. A vertex key x * unit + y stays in int64 while
    # unit = base 2^DEPTH < 2^31.5, so for a base mesh up to MAX_BASE.
    DEPTH = 20
    MAX_BASE = 2048

    def __init__(
        self,
        base: int,
        table_lattice: np.ndarray,
        payload: Payload,
        frame: Mapping[str, np.ndarray] | None = None,
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
        self.diag = float(max(np.linalg.norm(self.b[0] + self.b[1]), np.linalg.norm(self.b[0] - self.b[1])))
        self.payload = payload
        # Vertex id per integer point, keyed x * unit + y. The values live in buffers
        # that double when full, so a round never copies the mesh.
        self.key: dict[int, int] = {}
        self._buf: dict[str, np.ndarray] = {}
        self.nv = 0
        n = self.base
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        self.lvl = np.zeros(n * n, np.int64)
        self.cx = (i.ravel() * self.s0).astype(np.int64)
        self.cy = (j.ravel() * self.s0).astype(np.int64)
        self.vid = self._cell_vertices(self.lvl, self.cx, self.cy)
        self.ind = np.full(len(self.lvl), np.nan)
        # Whether the must-split rules have seen a leaf: they depend only on its own
        # vertices, so a leaf is judged once.
        self.checked = np.zeros(len(self.lvl), bool)
        self._index: dict[tuple[int, int, int], int] | None = None
        self.vertices_by_rule = {"base": int(self.nv), "seed": 0, "gap": 0, "edge": 0, "error": 0}

    def _frac(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fractional k of integer points: integer 0 is -1/2, as in :meth:`contains`."""
        fx = (x % self.unit) / self.unit - 0.5
        fy = (y % self.unit) / self.unit - 0.5
        return np.stack([fx, fy, np.zeros_like(fx)], axis=1)

    def ensure(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vertex ids for integer points, evaluating the new ones in one batch."""
        x = np.asarray(x, np.int64).ravel() % self.unit
        y = np.asarray(y, np.int64).ravel() % self.unit
        keys = (x * self.unit + y).tolist()
        new: dict[int, int] = {}
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

    def vertex_keys(self) -> np.ndarray:
        """Every vertex's integer key x * unit + y (x, y in units of 1/unit from -1/2), in creation order."""
        if not self.nv:
            return np.zeros(0, np.int64)
        return np.asarray(self._buf[VERTEX_KEY][: self.nv], np.int64)

    def _store(self, vals: Mapping[str, np.ndarray]) -> None:
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
    def vals(self) -> dict[str, np.ndarray]:
        """Per-vertex values: views of the filled rows."""
        return {name: buf[: self.nv] for name, buf in self._buf.items()}

    def _cell_vertices(self, lvl: np.ndarray, cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
        s = self.s0 >> lvl
        h = s >> 1
        xs = np.stack([cx, cx + s, cx + s, cx, cx + h], axis=1)
        ys = np.stack([cy, cy, cy + s, cy + s, cy + h], axis=1)
        return self.ensure(xs, ys).reshape(len(lvl), 5)

    def diameter(self, lvl: np.ndarray) -> np.ndarray:
        """Cartesian diameter (1/Å) of cells at these levels."""
        return self.diag * (self.s0 >> np.asarray(lvl, np.int64)) / self.unit

    def triangles(self, vid: np.ndarray, lvl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    def children(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        self.vertices_by_rule[rule] = self.vertices_by_rule.get(rule, 0) + (self.nv - before)

    def contains(self, kfrac: np.ndarray) -> np.ndarray:
        """Leaf index holding each fractional point (-1 if none)."""
        if self._index is None:
            self._index = {
                key: i for i, key in enumerate(zip(self.lvl.tolist(), self.cx.tolist(), self.cy.tolist()))
            }
        x = np.floor((np.asarray(kfrac)[:, 0] + 0.5) % 1.0 * self.unit).astype(np.int64)
        y = np.floor((np.asarray(kfrac)[:, 1] + 0.5) % 1.0 * self.unit).astype(np.int64)
        top = int(np.max(self.lvl)) if len(self.lvl) else 0
        out = np.full(len(x), -1, np.int64)
        for i, (xi, yi) in enumerate(zip(x.tolist(), y.tolist())):
            for level in range(top + 1):
                s = self.s0 >> level
                hit = self._index.get((level, xi - xi % s, yi - yi % s))
                if hit is not None:
                    out[i] = hit
                    break
        return out


def cell_block_integral(mesh: AdaptiveMesh, vid: np.ndarray, lvl: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per cell: the sum over its 4 triangles of area x mean(B): (ncell, nB)."""
    side = (mesh.s0 >> np.asarray(lvl, np.int64)) / mesh.unit
    w = side**2 * mesh.area_bz
    corners = B[vid[:, :4]].sum(axis=1)
    return w[:, None] * (corners / 6.0 + B[vid[:, 4]] / 3.0)


# ---------------------------------------------------------------- integration
class Integrator:
    """Adaptive zone integration of one model.

    Phases: the must-split rules (seeds, the restricted and capped gap rule, the
    edge rule), then error-driven refinement with lookahead to ``tolerance`` and on
    to ``tolerance / 2``, with a snapshot of every integral at each stage. The
    integration stops with ``stop`` "budget" at ``max_points`` vertices, "time" past
    ``deadline`` or at ``stop_vertices`` vertices (a replay of a time stop), and
    "depth" when the worst cells cannot be split further.
    """

    def __init__(
        self,
        payload: Payload,
        lattice: np.ndarray,
        *,
        window: tuple[int, int],
        n_occ: int | None,
        s_applies: bool,
        mu_range: tuple[float, float],
        mu_subset: Sequence[float] | np.ndarray,
        headline_mu: Mapping[str, float],
        kT_max: float,
        base: int = 64,
        tolerance: float = 1e-3,
        max_points: int = 1_000_000,
        gap_ratio: float = 0.1,
        gap_floor: float = 5e-5,
        edge_resolution: float = 2e-4,
        seeds: np.ndarray | None = None,
        seed_diameter: float = 0.0,
        seed_spread: float = 0.0,
        deadline: float | None = None,
        stop_vertices: int | None = None,
        frame: Mapping[str, np.ndarray] | None = None,
        log: Callable[[str], None] | None = None,
    ):
        self.mesh = AdaptiveMesh(base, lattice, payload, frame=frame)
        self.window = (int(window[0]), int(window[1]))
        self.n_occ = n_occ
        self.s_applies = bool(s_applies and n_occ)
        # The extended range of the window definition: +-20 kT and 0.1 eV beyond.
        self.ext = (mu_range[0] - 20.0 * kT_max - 0.1, mu_range[1] + 20.0 * kT_max + 0.1)
        self.mu_subset = np.unique(np.asarray(list(mu_subset), float))
        self.headline_mu = dict(headline_mu)
        self.tolerance = float(tolerance)
        self.max_points = int(max_points)
        self.gap_ratio = float(gap_ratio)
        self.gap_floor = float(gap_floor)
        self.edge_resolution = float(edge_resolution)
        self.seeds = np.zeros((0, 3)) if seeds is None else np.asarray(seeds, float).reshape(-1, 3)
        if len(self.seeds) and seed_spread > 0:
            # A seed is a grid point near the feature, not the feature: the cells under a
            # 3x3 stencil of one grid spacing around it are refined, so a pocket that
            # straddles a cell boundary is refined on both sides.
            offs = np.array([(i, j, 0) for i in (-1, 0, 1) for j in (-1, 0, 1)], float) * seed_spread
            self.seeds = (self.seeds[:, None, :] + offs[None]).reshape(-1, 3)
        self.seed_diameter = float(seed_diameter)
        self.deadline = deadline
        self.stop_vertices = None if stop_vertices is None else int(stop_vertices)
        self.log = log or (lambda message: None)
        self.history: list[dict[str, Any]] = []
        self.gap_rule_capped = False
        self.stop: str | None = None
        self.stop_at_vertices: int | None = None
        self.snapshots: dict[str, dict[str, Any]] = {}
        self._scales_fixed: tuple[dict[str, Any], dict[float, tuple[float, float]]] | None = None
        self._totals: np.ndarray | None = None
        self._n_signed = 0
        # QUEER's decision record (:meth:`decision_record`), kept beside the kernel's arithmetic.
        # ``tie_scales`` may preset the energy scale (eV, the spectral radius) and the velocity
        # scale (eV Å); the base mesh's own values are the floor of each.
        self.tie_scales: dict[str, float] = {}
        self.seed_ties: dict[str, Any] | None = None
        self.decision_rounds: list[dict[str, Any]] = []
        self._bands: dict[str, float] | None = None
        self._bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._judged = np.zeros(0, np.int64)
        self._listed = 0
        self._truncated = False
        # Per stage: which error-estimate columns normalise a rounding-level quantity, and per leaf
        # whether its indicator (its largest normalised error) is one of theirs.
        self._noise = np.zeros(0, bool)
        self._noisy = np.zeros(0, bool)
        self._noise_names: dict[str, int] = {}
        self._last_noisy = np.zeros(0, bool)

    # -- budget ------------------------------------------------------------
    def _out_of_budget(self) -> str | None:
        if self.mesh.nv >= self.max_points:
            return STOP_BUDGET
        if self.stop_vertices is not None and self.mesh.nv >= self.stop_vertices:
            return STOP_TIME
        if self.deadline is not None and time.time() > self.deadline:
            return STOP_TIME
        return None

    def _stopped(self, reason: str) -> None:
        self.stop = reason
        if self.stop_at_vertices is None:
            self.stop_at_vertices = int(self.mesh.nv)

    # -- must-split rules --------------------------------------------------
    def _rule_flags(self) -> dict[str, np.ndarray]:
        """The must-split flags of every leaf; only leaves not judged before are evaluated (the rules
        depend on a leaf's own vertices and the seeds)."""
        mesh = self.mesh
        vals = mesh.vals
        todo = np.flatnonzero(~mesh.checked)
        n_all = len(mesh.lvl)
        flags: dict[str, np.ndarray] = {}
        bands = self._tie_bands()
        self._bounds = bounds = {}
        cells = np.zeros(0, np.int64)
        if len(self.seeds):
            seed = np.zeros(n_all, bool)
            cells = mesh.contains(self.seeds)
            cells = cells[cells >= 0]
            d_all = mesh.diameter(mesh.lvl[cells])
            seed[cells] = d_all > self.seed_diameter
            flags["seed"] = seed
            bounds["seed"] = self._widen_bounds(
                n_all, cells, _tri(d_all - self.seed_diameter, bands["seed_diameter_invA"])
            )
        self._judged = np.union1d(todo, cells)
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
            split_pair = (d[:, None] * vmax > self.gap_ratio * np.maximum(gap, self.gap_floor)) & (
                near | near_next
            )
            split_pair &= gap > 1e-9  # a symmetry-degenerate pair is not an anticrossing
            gap_flag = split_pair.any(axis=1)
            gs = vs = None
            if self.s_applies and "gap_S" in vals:
                gs = vals["gap_S"][vid].min(axis=1)
                vs = vals["v_S"][vid].max(axis=1)
                gap_flag |= d * vs > self.gap_ratio * np.maximum(gs, self.gap_floor)
            flags["gap"] = widen(gap_flag)
            bounds["gap"] = self._widen_bounds(n_all, todo, self._gap_bounds(E, nxt, gap, vmax, d, gs, vs))
        v = vals["v"][vid]  # (ncell, 5, nW, 2)
        span = E.max(axis=1) - E.min(axis=1)
        turn = (v.min(axis=1) < 0) & (v.max(axis=1) > 0)
        inner = turn.all(axis=-1)
        reach = (E.max(axis=1) >= self.ext[0]) & (E.min(axis=1) <= self.ext[1])
        flags["edge"] = widen((inner & reach & (span > self.edge_resolution)).any(axis=1))
        bounds["edge"] = self._widen_bounds(n_all, todo, self._edge_bounds(E, v, span))
        return flags

    # -- the decision record (QUEER's) -----------------------------------
    def _tie_bands(self) -> dict[str, float]:
        """The rounding bands of the must-split comparisons, fixed at the first judgement (the base mesh).

        Energies: TIE_ROUNDING times the spectral radius (``tie_scales["energy"]``, at least the
        base mesh's largest window energy); velocities: TIE_ROUNDING times the largest band,
        interband or occupied-block velocity of the base mesh (at least ``tie_scales["velocity"]``);
        the seed rule's diameter: TIE_ROUNDING times the seed diameter.
        """
        if self._bands is None:
            vals = self.mesh.vals
            energy = max(
                [float(self.tie_scales.get("energy", 0.0))]
                + [float(np.max(np.abs(vals[k]))) for k in ("E", "E_above") if k in vals and vals[k].size]
            )
            velocity = max(
                [float(self.tie_scales.get("velocity", 0.0))]
                + [
                    float(np.max(np.abs(vals[k])))
                    for k in ("v", "v_pair", "v_S")
                    if k in vals and vals[k].size
                ]
            )
            self._bands = {
                "rounding": TIE_ROUNDING,
                "indicator": TIE_INDICATOR,
                "energy_scale_eV": energy,
                "velocity_scale_eVA": velocity,
                "energy_eV": TIE_ROUNDING * energy,
                "velocity_eVA": TIE_ROUNDING * velocity,
                "seed_diameter_invA": TIE_ROUNDING * self.seed_diameter,
            }
        return self._bands

    @staticmethod
    def _widen_bounds(
        n_all: int, index: np.ndarray, bound: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        sure, possible = np.zeros(n_all, bool), np.zeros(n_all, bool)
        sure[index], possible[index] = bound
        return sure, possible

    def _gap_bounds(self, E, nxt, gap, vmax, d, gs, vs) -> tuple[np.ndarray, np.ndarray]:
        """Where the gap rule surely and possibly holds, each comparison within its band: d vmax against
        gap_ratio max(gap, gap_floor) to d times the velocity band plus gap_ratio times twice the energy
        band, the extended range to the energy band, the 1e-9 eV degeneracy floor to twice it."""
        b = self._tie_bands()
        bE, bV = b["energy_eV"], b["velocity_eVA"]
        lo, hi = self.ext
        ratio_s, ratio_p = _tri(
            d[:, None] * vmax - self.gap_ratio * np.maximum(gap, self.gap_floor),
            d[:, None] * bV + self.gap_ratio * 2.0 * bE,
        )
        near_s, near_p = _tri(np.minimum(E - lo, hi - E).max(axis=1), bE)
        next_s, next_p = _tri(np.minimum(nxt - lo, hi - nxt).max(axis=1), bE)
        split_s, split_p = _tri(gap - 1e-9, 2.0 * bE)
        sure = (ratio_s & (near_s | next_s) & split_s).any(axis=1)
        possible = (ratio_p & (near_p | next_p) & split_p).any(axis=1)
        if gs is not None:
            s_s, s_p = _tri(
                d * vs - self.gap_ratio * np.maximum(gs, self.gap_floor), d * bV + self.gap_ratio * 2.0 * bE
            )
            sure, possible = sure | s_s, possible | s_p
        return sure, possible

    def _edge_bounds(self, E, v, span) -> tuple[np.ndarray, np.ndarray]:
        """Where the edge rule surely and possibly holds: each velocity component's sign within the
        velocity band, the extended range within the energy band, the span within twice it."""
        b = self._tie_bands()
        bE, bV = b["energy_eV"], b["velocity_eVA"]
        neg_s, neg_p = _tri(-v.min(axis=1), bV)
        pos_s, pos_p = _tri(v.max(axis=1), bV)
        reach_lo_s, reach_lo_p = _tri(E.max(axis=1) - self.ext[0], bE)
        reach_hi_s, reach_hi_p = _tri(self.ext[1] - E.min(axis=1), bE)
        span_s, span_p = _tri(span - self.edge_resolution, 2.0 * bE)
        sure = ((neg_s & pos_s).all(axis=-1) & reach_lo_s & reach_hi_s & span_s).any(axis=1)
        possible = ((neg_p & pos_p).all(axis=-1) & reach_lo_p & reach_hi_p & span_p).any(axis=1)
        return sure, possible

    def _centre_keys(self, idx: np.ndarray) -> np.ndarray:
        """The integer key of each leaf's centre vertex, which names the cell."""
        return np.asarray(self.mesh._buf[VERTEX_KEY][self.mesh.vid[np.asarray(idx, np.int64), 4]], np.int64)

    def _list(self, rows: list[list[Any]]) -> list[list[Any]]:
        """Rows of tied decisions within the run's listing budget (:data:`TIE_LIST_MAX`)."""
        room = max(TIE_LIST_MAX - self._listed, 0)
        if len(rows) > room:
            self._truncated = True
            rows = rows[:room]
        self._listed += len(rows)
        return rows

    def _must_split_round(self, chosen: Mapping[str, np.ndarray]) -> dict[str, Any]:
        """The record of a must-split round, before its splits: each judged leaf's possible outcomes
        (the first rule that holds, in the order seed, gap, edge, below the depth limit; a capped gap
        rule defers its leaves), the tied ones and the digest of the leaves each rule splits."""
        mesh = self.mesh
        n_all = len(mesh.lvl)
        deferred = "gap" in chosen and mesh.vertices_by_rule.get("gap", 0) >= 0.3 * self.max_points
        allowed = mesh.lvl < AdaptiveMesh.DEPTH - 2
        possible = np.zeros((n_all, len(OUTCOMES)), bool)
        settled = np.zeros(n_all, bool)
        counts: dict[str, int] = {}
        judged = self._judged
        for rule in MUST_SPLIT_RULES:
            if rule not in self._bounds:
                continue
            sure, maybe = self._bounds[rule]
            reachable = ~settled & allowed
            code = OUTCOMES.index("none" if rule == "gap" and deferred else rule)
            possible[:, code] |= maybe & reachable
            counts[rule] = int(np.count_nonzero((maybe & ~sure & reachable)[judged]))
            settled |= sure & allowed
        possible[:, 0] |= ~settled
        outcome = np.zeros(n_all, np.int64)
        splits: dict[str, int] = {}
        digest: dict[str, str] = {}
        for rule, idx in chosen.items():
            if rule == "gap" and deferred:
                continue
            outcome[idx] = OUTCOMES.index(rule)
            splits[rule] = int(len(idx))
            digest[rule] = set_digest(self._centre_keys(idx))
        # The run's own outcome lies inside its possible ones unless the bounds are not the rules.
        outside = int(np.count_nonzero(~possible[judged, outcome[judged]]))
        tied = judged[np.count_nonzero(possible[judged], axis=1) > 1]
        masks = (possible[tied] * (1 << np.arange(len(OUTCOMES)))).sum(axis=1)
        rows = [[int(k), int(m), int(o)] for k, m, o in zip(self._centre_keys(tied), masks, outcome[tied])]
        return {
            "phase": "must_split",
            "history": None,
            "judged": int(len(judged)),
            "tied_count": int(len(tied)),
            "tied_by_rule": counts,
            "outside": outside,
            "tied": self._list(rows),
            "splits": splits,
            "digest": digest,
            "gap_deferred": bool(deferred),
            "leaves": [n_all, n_all],
            "vertices": [int(mesh.nv), int(mesh.nv)],
            "ranges": {},
        }

    def _error_round(self, target: float, worst: float) -> dict[str, Any]:
        """The record of a refinement step: its stop test worst <= target, tied within TIE_INDICATOR of
        the larger of the two."""
        band = TIE_INDICATOR * max(target, worst)
        return {
            "phase": "error",
            "history": len(self.history) - 1,
            "target": target,
            "stop": {
                "worst": worst,
                "margin": target - worst,
                "band": band,
                "tied": bool(abs(target - worst) <= band),
                "by": None,
            },
            "judged": 0,
            "tied_count": 0,
            "tied": [],
            "splits": {},
            "digest": {},
            "leaves": [len(self.mesh.lvl), len(self.mesh.lvl)],
            "vertices": [int(self.mesh.nv), int(self.mesh.nv)],
            "ranges": {},
        }

    def _cut_ties(
        self,
        entry: dict[str, Any],
        order: np.ndarray,
        cum: np.ndarray,
        worst: float,
        target: float,
        cap: int,
        floor: int,
    ) -> None:
        """The refinement's batch: the count taken (the cumulative indicator against worst - 0.75 target,
        within TIE_INDICATOR of worst) and the leaves whose membership rounding can flip (indicators
        within TIE_INDICATOR of the cut value of the batch's last one, twice that between two leaves)."""
        mesh = self.mesh
        threshold = worst - 0.75 * target
        band_sum = TIE_INDICATOR * max(worst, target)

        def taken(take: int) -> int:
            # The batch order[:n] holds at most every leaf.
            return min(max(min(take, cap), floor), len(order))

        n = taken(int(np.searchsorted(cum, threshold)) + 1)
        n_lo = taken(int(np.searchsorted(cum, threshold - band_sum)) + 1)
        n_hi = taken(int(np.searchsorted(cum, threshold + band_sum)) + 1)
        values = mesh.ind[order]
        cut = float(values[n - 1]) if n else 0.0
        band = TIE_INDICATOR * abs(cut)
        # Possibly in the top n_hi: within 2 bands of its n_hi-th value; surely in the top n_lo:
        # more than 2 bands above the value after it.
        possible = values >= (values[n_hi - 1] - 2.0 * band if n_hi else np.inf)
        sure = values > (values[n_lo] + 2.0 * band if n_lo < len(values) else -np.inf)
        rank = np.flatnonzero(possible & ~sure & (mesh.lvl[order] < AdaptiveMesh.DEPTH - 2))
        cells = order[rank]
        split = rank < n
        mask = (1 << OUTCOMES.index("none")) | (1 << OUTCOMES.index("error"))
        code = OUTCOMES.index("error")
        rows = [
            [int(k), mask, code if s else 0, float(value - cut)]
            for k, s, value in zip(self._centre_keys(cells), split, values[rank])
        ]
        entry.update(
            judged=int(len(order)),
            tied_count=int(len(cells)),
            tied=self._list(rows),
            take={
                "n": int(n),
                "range": [int(n_lo), int(n_hi)],
                "cap": int(cap),
                "floor": int(floor),
                "band": band_sum,
            },
            cut={"value": cut, "band": band},
        )

    def _close_round(self, entry: dict[str, Any]) -> None:
        entry["leaves"][1] = int(len(self.mesh.lvl))
        entry["vertices"][1] = int(self.mesh.nv)
        self.decision_rounds.append(entry)

    def record_restart(self, mu_out: np.ndarray, A3: np.ndarray, chosen: int, scale: float) -> None:
        """The second stage's A3 maximum: the grid mu whose A3 lies within TIE_INDICATOR of ``scale``
        (the larger of the largest A3 and the largest |M| element) of the maximum."""
        band = TIE_INDICATOR * scale
        tied = np.flatnonzero(np.asarray(A3) >= float(A3[chosen]) - 2.0 * band)
        self.decision_rounds.append(
            {
                "phase": "restart",
                "history": None,
                "chosen_mu_eV": float(mu_out[chosen]),
                "tied_mu_eV": [float(mu_out[j]) for j in tied],
                "band": band,
                "tied_count": int(len(tied) - 1),
                "mu_subset": int(len(self.mu_subset)),
                "leaves": [len(self.mesh.lvl)] * 2,
                "vertices": [int(self.mesh.nv)] * 2,
            }
        )

    def decision_record(self) -> dict[str, Any]:
        """QUEER's record of the integration's decisions within rounding of their thresholds.

        ``bands`` holds the rounding bands and the scales they come from; ``seeds`` the
        occupation pass's seed decisions (:func:`seed_ties`); ``rounds`` one entry per
        decision round in order: each must-split round (``history`` the index of the history
        entry it appended, None for the round that split nothing), each refinement step
        (``history`` the entry it appended before deciding) and the second stage's restart.
        A round lists its tied decisions as [cell centre key, mask of the possible outcomes,
        the run's outcome (indices into ``outcomes``) and, in a refinement step, the indicator
        less the cut value], counts them, and gives per rule the number of leaves split, the
        digest of their centre keys (:func:`set_digest`) and the range of the vertex ids their
        splits created. ``truncated`` is true when more than TIE_LIST_MAX decisions were tied:
        the counts and digests stay complete.
        """
        return {
            "outcomes": list(OUTCOMES),
            "bands": dict(self._tie_bands()) if self.mesh.nv else {},
            "seeds": self.seed_ties,
            "rounds": self.decision_rounds,
            "truncated": self._truncated,
        }

    def must_split(self, max_rounds: int = 200) -> None:
        for _ in range(max_rounds):
            reason = self._out_of_budget()
            if reason:
                self._stopped(reason)
                return
            flags = self._rule_flags()
            chosen: dict[str, np.ndarray] = {}
            taken = np.zeros(len(self.mesh.lvl), bool)
            for rule in ("seed", "gap", "edge"):
                f = flags.get(rule)
                if f is None:
                    continue
                idx = np.flatnonzero(f & ~taken & (self.mesh.lvl < AdaptiveMesh.DEPTH - 2))
                if len(idx):
                    chosen[rule] = idx
                    taken[idx] = True
            entry = self._must_split_round(chosen)
            if not chosen:
                self._close_round(entry)
                return
            removed = np.zeros(0, np.int64)
            for rule, idx in chosen.items():
                # A split drops its leaves and appends the children, so the indices taken
                # before this round's earlier splits shift down.
                cur = idx - np.searchsorted(removed, idx)
                if rule == "gap" and self.mesh.vertices_by_rule.get("gap", 0) >= 0.3 * self.max_points:
                    self.gap_rule_capped = True
                    # Judged again next round, by the rules still in force.
                    self.mesh.checked[cur] = False
                    continue
                start = int(self.mesh.nv)
                self.mesh.split(cur, rule)
                entry["ranges"][rule] = [start, int(self.mesh.nv)]
                removed = np.union1d(removed, idx)
                if rule == "gap" and self.mesh.vertices_by_rule["gap"] >= 0.3 * self.max_points:
                    self.gap_rule_capped = True
            self.history.append(
                {"phase": "must_split", "vertices": int(self.mesh.nv), "leaves": int(len(self.mesh.lvl))}
            )
            entry["history"] = len(self.history) - 1
            self._close_round(entry)

    # -- error indicators --------------------------------------------------
    def _triangle_fields(self, vid: np.ndarray, lvl: np.ndarray, X: np.ndarray, E: np.ndarray):
        tri, area = self.mesh.triangles(vid, lvl)
        nW = E.shape[1]
        e = np.moveaxis(E[tri], 2, 1).reshape(-1, 3)  # (nt * nW, 3)
        x = np.moveaxis(X[tri], 2, 1).reshape(-1, 3, X.shape[-1])
        a = np.repeat(area, nW)
        cell = np.repeat(np.arange(len(vid)), 4 * nW)
        return e, x, a, cell

    def _chunk_cells(self) -> int:
        """Leaves per batch: :data:`TRIANGLES_PER_CHUNK` triangle-bands."""
        n_bands = self.mesh.vals["E"].shape[1] if self.mesh.nv else 1
        return max(64, TRIANGLES_PER_CHUNK // (4 * max(1, n_bands)))

    def chunks(self) -> list[slice]:
        """Leaf slices: zone-wide sums never hold the whole mesh's triangles."""
        n, size = len(self.mesh.lvl), self._chunk_cells()
        return [slice(s, min(n, s + size)) for s in range(0, n, size)]

    def _scales(self) -> dict[str, Any]:
        mesh = self.mesh
        vals = mesh.vals
        absX = np.abs(vals["X"])
        fs = np.zeros((len(self.mu_subset), len(FS_COLUMNS)))
        sea_omega = 0.0
        for part in self.chunks():
            e, x, a, _ = self._triangle_fields(mesh.vid[part], mesh.lvl[part], absX, vals["E"])
            fs += grouped_rule(
                "delta", e, x[..., FS_COLUMNS], a, self.mu_subset, np.zeros(len(e), np.int64), 1
            )[:, 0]
            sea_omega += float(np.sum(a * x[:, :, X_OMEGA].mean(axis=1)))
        blocks = np.abs(cell_block_integral(mesh, mesh.vid, mesh.lvl, np.abs(vals["B"]))).sum(axis=0)
        sea_omega += blocks[B_OMEGA_C]
        return {
            "fs": np.maximum(fs, 1e-300),
            "sea": np.array([mesh.area_bz * (self.window[1] - self.window[0]), max(sea_omega, 1e-300)]),
            "blocks": np.maximum(blocks, 1e-300),
            "fs_abs_vg": fs[:, 1:7],
        }

    def _headline_signed(self) -> dict[float, tuple[float, float]]:
        """The current A3 and the largest |v G| integral at each headline mu (zone integrals of signed v G)."""
        mesh = self.mesh
        vals = mesh.vals
        mus = np.array(sorted(set(self.headline_mu.values())), float)
        if not len(mus):
            return {}
        vg = np.zeros((len(mus), 6))
        for part in self.chunks():
            e, x, a, _ = self._triangle_fields(mesh.vid[part], mesh.lvl[part], vals["X"], vals["E"])
            vg += grouped_rule("delta", e, x[..., X_VG], a, mus, np.zeros(len(e), np.int64), 1)[:, 0]
        out = {}
        for m, row in zip(mus, vg):
            tr = trigonal(dipole_tensor(row))
            out[float(m)] = (float(tr["A3"]), float(np.abs(row).max()))
        return out

    def _corner_triangles(self, vid: np.ndarray, lvl: np.ndarray, X: np.ndarray, E: np.ndarray):
        """Each cell as two triangles on each diagonal, corners only, weights 1/2."""
        c = vid[:, :4]
        tri = np.stack([c[:, [0, 1, 2]], c[:, [0, 2, 3]], c[:, [1, 2, 3]], c[:, [1, 3, 0]]], axis=1).reshape(
            -1, 3
        )
        side = (self.mesh.s0 >> np.asarray(lvl, np.int64)) / self.mesh.unit
        area = np.repeat(side**2 * self.mesh.area_bz / 4.0, 4)  # (cell/2) x weight 1/2
        nW = E.shape[1]
        e = np.moveaxis(E[tri], 2, 1).reshape(-1, 3)
        x = np.moveaxis(X[tri], 2, 1).reshape(-1, 3, X.shape[-1])
        a = np.repeat(area, nW)
        cell = np.repeat(np.arange(len(vid)), 4 * nW)
        return e, x, a, cell

    def _error_sums(
        self, idx: np.ndarray, scales: Mapping[str, Any], head: Mapping[float, tuple[float, float]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`_errors` over cell chunks: each cell's largest normalised error and every quantity's
        summed error, never the (cells x quantities) matrix."""
        ind = np.empty(len(idx))
        total: np.ndarray | None = None
        size = self._chunk_cells()
        noisy = np.zeros(len(idx), bool)
        for s in range(0, len(idx), size):
            err = self._errors(idx[s : s + size], scales, head)
            ind[s : s + size] = np.abs(err).max(axis=1)
            part = err.sum(axis=0)
            total = part if total is None else total + part
            if self._noise.any():
                # Leaves whose largest normalised error is a rounding-level quantity's.
                rounding = np.abs(err[:, self._noise]).max(axis=1)
                real = np.abs(err[:, ~self._noise]).max(axis=1) if (~self._noise).any() else 0.0
                noisy[s : s + size] = (rounding > 0.0) & (rounding >= real)
        assert total is not None
        self._last_noisy = noisy
        return ind, total

    def _noise_columns(self, scales: Mapping[str, Any], head: Mapping[float, tuple[float, float]]) -> None:
        """The error-estimate columns (in :meth:`_errors`' order) whose normalising scale is rounding: a
        quantity that vanishes identically (by symmetry), whose estimate is then rounding over rounding.

        A scale is rounding when it is nonzero (above the 1e-300 floor of an exactly zero integrand)
        and at most TIE_ROUNDING times its family's: for a Fermi-surface column v_a G_bc or v_a omega the
        largest of those columns at the same mu, for the sea and block columns (omega, the occupied
        block's omega and metric) the largest of the area-integrated geometric scales. The count
        columns are never rounding. A headline mu's signed alpha and beta mix its v_a G_bc columns:
        they are rounding where one of those is.
        """
        blocks, fs, sea = scales["blocks"], scales["fs"], scales["sea"]
        geometric = float(max(np.max(blocks), sea[1]))

        def rounding(scale: Any, family: Any) -> np.ndarray:
            scale = np.asarray(scale, float)
            return (scale > 1e-300) & (scale <= TIE_ROUNDING * np.asarray(family, float))

        cols = [B_OMEGA_C] + ([B_OMEGA_S] if self.s_applies else [])
        parts = [rounding(blocks[cols], geometric)]
        names = [("block omega", "block omega_S")[i] for i in range(len(cols))]
        if self.s_applies:
            parts.append(rounding([blocks[3] + blocks[5]], geometric))
            names.append("block g_S trace")
        fs_noise = rounding(fs, fs[:, 1:].max(axis=1, keepdims=True))
        fs_noise[:, 0] = False
        parts.append(fs_noise.ravel())
        sea_noise = np.zeros((len(self.mu_subset), 2), bool)
        sea_noise[:, 1] = bool(rounding(sea[1], geometric))
        parts.append(sea_noise.ravel())
        for j, m in enumerate(self.mu_subset):
            if float(m) in head:
                parts.append(np.repeat(bool(fs_noise[j, 1:7].any()), 2))
        self._noise = np.concatenate(parts) if parts else np.zeros(0, bool)
        counts: dict[str, int] = {}
        for name, flag in zip(names, parts[0].tolist() + (parts[1].tolist() if self.s_applies else [])):
            if flag:
                counts[name] = 1
        for q, name in enumerate(FS_NAMES):
            if fs_noise[:, q].any():
                counts[f"Fermi-surface {name}"] = int(fs_noise[:, q].sum())
        if sea_noise[:, 1].any():
            counts["sea omega"] = int(len(self.mu_subset))
        signed = [j for j, m in enumerate(self.mu_subset) if float(m) in head and fs_noise[j, 1:7].any()]
        if signed:
            counts["headline alpha, beta"] = len(signed)
        self._noise_names = counts

    def _errors(
        self, idx: np.ndarray, scales: Mapping[str, Any], head: Mapping[float, tuple[float, float]]
    ) -> np.ndarray:
        """The embedded error estimate per cell and per controlled quantity: its four centre triangles
        against the corner-only rule (both diagonals averaged), divided by 3 (the fine rule's share for
        an O(h^2) pair, Richardson) and normalised by that quantity's scale. Returns (len(idx), nK); the
        last ``self._n_signed`` columns (alpha and beta at the headline mu) are signed, because those
        are cancellation-dominated combinations whose global error is the signed sum.

        It needs no new vertex: for a mu-independent integrand the two rules differ by the
        area times (centre - corner mean)/3.
        """
        mesh = self.mesh
        vals = mesh.vals
        X, E, B = vals["X"], vals["E"], vals["B"]
        n = len(idx)
        vid = mesh.vid[idx]
        lvl = mesh.lvl[idx]
        parts: list[np.ndarray] = []
        side = (mesh.s0 >> lvl) / mesh.unit
        w = side**2 * mesh.area_bz
        fine_b = cell_block_integral(mesh, vid, lvl, B)
        coarse_b = w[:, None] * B[vid[:, :4]].mean(axis=1)
        cols = [B_OMEGA_C] + ([B_OMEGA_S] if self.s_applies else [])
        for col in cols:
            parts.append((np.abs(fine_b[:, col] - coarse_b[:, col]) / (3.0 * scales["blocks"][col]))[:, None])
        if self.s_applies:
            dg = (fine_b[:, 3] + fine_b[:, 5]) - (coarse_b[:, 3] + coarse_b[:, 5])
            parts.append(
                (np.abs(dg) / (3.0 * max(scales["blocks"][3] + scales["blocks"][5], 1e-300)))[:, None]
            )
        ef, xf, af, cf = self._triangle_fields(vid, lvl, X, E)
        ec, xc, ac, cc = self._corner_triangles(vid, lvl, X, E)
        fs_f = grouped_rule("delta", ef, xf[..., FS_COLUMNS], af, self.mu_subset, cf, n)
        fs_c = grouped_rule("delta", ec, xc[..., FS_COLUMNS], ac, self.mu_subset, cc, n)
        dfs = np.abs(fs_f - fs_c) / (3.0 * scales["fs"][:, None, :])  # (nmu, n, 9)
        parts.append(np.moveaxis(dfs, 1, 0).reshape(n, -1))
        sea_f = grouped_rule("theta", ef, xf[..., SEA_COLUMNS], af, self.mu_subset, cf, n)
        sea_c = grouped_rule("theta", ec, xc[..., SEA_COLUMNS], ac, self.mu_subset, cc, n)
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

    def restart(self, headline_mu: Mapping[str, float] | None = None) -> None:
        """New scales (and headline mu) for the next stage; the totals are rebuilt over all leaves."""
        if headline_mu is not None:
            self.headline_mu = dict(headline_mu)
            self.mu_subset = np.unique(np.concatenate([self.mu_subset, list(headline_mu.values())]))
        self._scales_fixed = None
        self._totals = None
        self.mesh.ind[:] = np.nan

    @property
    def scales_fixed(self) -> tuple[dict[str, Any], dict[float, tuple[float, float]]] | None:
        """The normalising scales and headline values of the current stage, once taken."""
        return self._scales_fixed

    def refine(self, target: float, batch: int | None = None) -> bool:
        """Split cells until every controlled quantity's summed error is at most ``target``.

        Each quantity's total is kept incrementally: a cell's error vector is added when it
        is created and subtracted, recomputed from the same vertices, when it is split. The
        normalising scales are taken once (after the must-split phase) so additions and
        subtractions match. A leaf's priority is its largest normalised error. Returns False
        on a stop.
        """
        mesh = self.mesh
        if self._scales_fixed is None:
            self._scales_fixed = (self._scales(), self._headline_signed())
            self._totals = None
            self._noise_columns(*self._scales_fixed)
            self._noisy = np.zeros(len(mesh.lvl), bool)
            stage_start = True
        else:
            stage_start = False
        scales, head = self._scales_fixed
        while True:
            reason = self._out_of_budget()
            todo = np.flatnonzero(np.isnan(mesh.ind))
            if len(todo):
                mesh.ind[todo], add = self._error_sums(todo, scales, head)
                self._noisy[todo] = self._last_noisy
                self._totals = add if self._totals is None else self._totals + add
            worst = float(np.max(np.abs(self._totals))) if self._totals is not None else 0.0
            self.history.append(
                {
                    "phase": "error",
                    "target": target,
                    "worst": worst,
                    "vertices": int(mesh.nv),
                    "leaves": int(len(mesh.lvl)),
                }
            )
            entry = self._error_round(target, worst)
            if stage_start:
                entry["noise_quantities"] = dict(self._noise_names)
                stage_start = False
            if self._noise.any():
                totals = np.abs(self._totals) if self._totals is not None else np.zeros(len(self._noise))
                entry["noise"] = {
                    "columns": int(self._noise.sum()),
                    "leaves": int(np.count_nonzero(self._noisy)),
                    "worst_real": float(np.max(totals[~self._noise])) if (~self._noise).any() else 0.0,
                    "worst_rounding": float(np.max(totals[self._noise])),
                }
            if worst <= target:
                entry["stop"]["by"] = "target"
                self._close_round(entry)
                return True
            if reason:
                entry["stop"]["by"] = reason
                self._close_round(entry)
                self._stopped(reason)
                return False
            order = np.argsort(mesh.ind)[::-1]
            cap = batch or max(16, len(mesh.lvl) // 8)
            floor = min(max(16, len(mesh.lvl) // 50), len(order))
            # Enough of the worst cells to cover the excess, as if one quantity held it all.
            cum = np.cumsum(mesh.ind[order])
            take = int(np.searchsorted(cum, worst - 0.75 * target)) + 1
            idx = order[: max(min(take, cap), floor)]
            self._cut_ties(entry, order, cum, worst, target, cap, floor)
            idx = idx[mesh.lvl[idx] < AdaptiveMesh.DEPTH - 2]
            if not len(idx):
                entry["stop"]["by"] = STOP_DEPTH
                self._close_round(entry)
                self._stopped(STOP_DEPTH)
                return False
            assert self._totals is not None
            self._totals = self._totals - self._error_sums(idx, scales, head)[1]
            entry["splits"] = {"error": int(len(idx))}
            entry["digest"] = {"error": set_digest(self._centre_keys(idx))}
            start = int(mesh.nv)
            keep = np.ones(len(mesh.lvl), bool)
            keep[np.unique(idx)] = False
            mesh.split(idx, "error")
            self._noisy = np.concatenate([self._noisy[keep], np.zeros(len(mesh.lvl) - int(keep.sum()), bool)])
            entry["ranges"] = {"error": [start, int(mesh.nv)]}
            self._close_round(entry)

    # -- snapshots ----------------------------------------------------------
    def snapshot(self, name: str, mu_out: np.ndarray, kTs: Sequence[float]) -> dict[str, Any]:
        """Every integral at this mesh: Theta and the delta rule at the output mu (T = 0), Theta on a
        kT/20 grid per temperature, and the mu-independent block integrals, each a sum over triangles
        taken in leaf chunks."""
        mesh = self.mesh
        vals = mesh.vals
        out: dict[str, Any] = {"vertices": int(mesh.nv), "leaves": int(len(mesh.lvl))}
        mu_out = np.asarray(mu_out, float)
        uniform = is_uniform_grid(mu_out)
        step0 = float(mu_out[1] - mu_out[0]) if len(mu_out) > 1 else 1.0
        grids = {float(kT): thermal_grid(mu_out, kT) for kT in kTs if kT > 0}
        for level, xkey, bkey in (("main", "X", "B"), ("diag", "X_diag", "B_diag")):
            if xkey not in vals:
                continue
            nX = vals[xkey].shape[-1]
            theta0 = np.zeros((len(mu_out), nX))
            delta0 = np.zeros((len(mu_out), nX))
            thermal = {kT: np.zeros((g[2], nX)) for kT, g in grids.items()}
            for part in self.chunks():
                e, x, a, _ = self._triangle_fields(mesh.vid[part], mesh.lvl[part], vals[xkey], vals["E"])
                es, xs = sorted_triangles(e, x)  # once for every grid below
                if uniform:
                    mu0 = float(mu_out[0])
                    theta0 += grid_rule("theta", es, xs, a, mu0, step0, len(mu_out))
                    delta0 += grid_rule("delta", es, xs, a, mu0, step0, len(mu_out))
                else:
                    theta0 += triangle_theta(e, x, a, mu_out)
                    delta0 += triangle_delta(e, x, a, mu_out)
                for kT, (mu0, step, n) in grids.items():
                    thermal[kT] += grid_rule("theta", es, xs, a, mu0, step, n)
            out[level] = {
                "blocks": cell_block_integral(mesh, mesh.vid, mesh.lvl, vals[bkey]).sum(axis=0),
                "theta0": theta0,
                "delta0": delta0,
                "thermal": {kT: (grids[kT][0], grids[kT][1], thermal[kT]) for kT in grids},
            }
        self.snapshots[name] = out
        return out

    def vertex_weights(self) -> np.ndarray:
        """Cartesian area per vertex (a third of each adjacent triangle)."""
        w = np.zeros(self.mesh.nv)
        for part in self.chunks():
            tri, area = self.mesh.triangles(self.mesh.vid[part], self.mesh.lvl[part])
            w += np.bincount(tri.ravel(), weights=np.repeat(area / 3.0, 3), minlength=self.mesh.nv)
        return w

    def smeared_check(
        self, curves: Mapping[str, np.ndarray], mu_out: np.ndarray, kT: float
    ) -> dict[str, Any]:
        """The Fermi-surface integrals at kT straight from the vertices against the Theta convolution.

        (-f') X is taken at each vertex and integrated as linear on each triangle; at each
        headline mu the relative difference of the DOS and of the dipole M is measured,
        each against a floor of 1e-3 of its largest value over the mu range. A mu whose
        thermal DOS is below :data:`SMEARED_DOS_FLOOR` has no Fermi surface to check (None).
        """
        mesh = self.mesh
        vals = mesh.vals
        w = self.vertex_weights()
        mu_out = np.asarray(mu_out, float)
        two_pi2 = (2.0 * math.pi) ** 2
        floor_dos = 1e-3 * float(np.abs(curves["dos"]).max())
        floor_M = 1e-3 * float(np.abs(curves["M"]).max())
        cols = np.r_[0, np.arange(X_VG.start, X_VG.stop)]
        per_mu: dict[str, float | None] = {}
        for label, mu in self.headline_mu.items():
            if not mu_out[0] <= mu <= mu_out[-1]:
                continue
            j = int(np.argmin(np.abs(mu_out - mu)))
            acc = np.zeros(len(cols))
            for s in range(0, mesh.nv, VERTICES_PER_CHUNK):
                part = slice(s, min(mesh.nv, s + VERTICES_PER_CHUNK))
                weight = minus_fermi_derivative(vals["E"][part] - mu_out[j], kT) * w[part, None]
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
        return {"kT_eV": float(kT), "per_mu": per_mu, "rel": max(known) if known else None}


# ----------------------------------------------------------- dipole sources
def source_fields(
    table: QgeomTable,
    kfrac: np.ndarray,
    *,
    window: tuple[int, int],
    deg_tol: float,
    frozen: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    """The window bands' velocities and their G split by partner gap (and the part from partners
    outside ``frozen``), for :func:`dipole_sources`."""
    res = evaluate(table, kfrac, window=window, occupied=[], deg_tol=deg_tol, gap_bins=True, frozen=frozen)
    out = {"v": res["v"], "G_bins": res["G_bins"]}
    if frozen is not None:
        out["G_outside"] = res["G_outside_frozen"]
    return out


def dipole_sources(
    integ: Integrator, mu_points: Mapping[str, float], fields_fn: Callable[[np.ndarray], Mapping[str, Any]]
) -> dict[str, Any]:
    """Where the T = 0 dipole at each mu comes from.

    The final mesh's triangles that straddle mu are evaluated again with G split by
    partner gap (:func:`source_fields` through ``fields_fn``, which maps fractional k to
    its output). Each bin's share is its (alpha, beta) projected on the total's over
    |total|^2, so the bins sum to 1 and a negative share opposes the total.
    ``outside_frozen`` is the share from partners outside the frozen window; it overlaps
    the bins. None where mu crosses no band. Where A3 is below 1e-3 of the Fermi-surface
    integral of |v G| (the integration's own resolution), there is no trigonal part to
    split and the row says so instead.
    """
    mesh = integ.mesh
    E = mesh.vals["E"]
    cells: dict[str, np.ndarray] = {}
    for name, mu in mu_points.items():
        hits = [np.zeros(0, np.int64)]
        for part in integ.chunks():
            Ec = E[mesh.vid[part]]  # (cells, 5 vertices, window bands)
            hit = ((Ec.min(axis=1) < mu) & (Ec.max(axis=1) > mu)).any(axis=1)
            hits.append(np.flatnonzero(hit) + part.start)
        cells[name] = np.concatenate(hits)
    out: dict[str, Any] = {name: None for name in mu_points}
    found = [c for c in cells.values() if len(c)]
    if not found:
        return out
    verts = np.unique(mesh.vid[np.unique(np.concatenate(found))].ravel())
    res = fields_fn(mesh.vertex_kfrac(verts))
    v = res["v"]

    def vg(G: np.ndarray) -> np.ndarray:  # v_a G_bc in the X_VG column order
        return np.stack([v[..., a] * G[..., b, c] for a in range(2) for b, c in SYMMETRIC_PAIRS], -1)

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
        row: dict[str, Any] = {"mu_meV": float(mu) * 1e3, "triangles": int(len(tri)), "A3_A_per_eV": A3}
        if A3 <= 1e-3 * float(fs[-1].max()):
            row["reason"] = "A3 is below the integration's resolution: no trigonal part to split"
        else:
            share = (tr["alpha"] * a_t + tr["beta"] * b_t) / (A3 * A3)
            row["shares"] = {lab: float(s) for lab, s in zip(labels, share[:n_bins])}
            if len(share) > n_bins:
                row["outside_frozen"] = float(share[n_bins])
        out[name] = row
    return out


# ---------------------------------------------------------- adaptive response
def adaptive_response(
    table: QgeomTable,
    *,
    mu_out: np.ndarray,
    temperatures_K: Sequence[float] = (),
    occupied_bands: int | None = None,
    insulator_min_gap: float = 1e-4,
    deg_tol: float = 1e-6,
    tolerance: float = 1e-3,
    base: int = 64,
    max_points: int = 1_000_000,
    gap_ratio: float = 0.1,
    gap_floor: float = 5e-5,
    edge_resolution: float = 2e-4,
    occupation_mesh: int = 241,
    tau_diag: np.ndarray | None = None,
    seeds: bool = True,
    deadline: float | None = None,
    stop_vertices: int | None = None,
    frame: Mapping[str, np.ndarray] | None = None,
    log: Callable[[str], None] | None = None,
    parallel: Any = None,
    extra_kfrac: np.ndarray | None = None,
) -> dict[str, Any]:
    """Zone integrals of one model on the adaptive mesh at ``tolerance`` and at ``tolerance / 2``, as
    curves over ``mu_out`` (eV, relative to E_F) for T = 0 and each temperature.

    Returns the occupation pass, the window, ``n_occ``, ``S_applies``, the temperatures
    (T = 0 first), the stages ``tau`` and ``tau2`` with each level's curves per
    temperature and block integrals, the smeared cross-check at the highest temperature,
    the :class:`Integrator` and the wall time. ``parallel``, when given, maps the work:
    ``parallel.energies(table)`` and ``parallel.payload(serial)`` return drop-in
    replacements that split their k-points over workers. ``extra_kfrac`` (a k-path) joins
    the occupation pass. ``deadline`` and ``stop_vertices`` are the integrator's time stop
    and its replay.
    """
    log = log or (lambda message: None)
    mu_out = np.asarray(mu_out, float)
    temps = [0.0] + sorted(float(x) for x in temperatures_K if float(x) > 0)
    kTs = [K_B_EV * T for T in temps]
    kT_max = max(kTs)
    mu_range = (float(mu_out[0]), float(mu_out[-1]))
    t0 = time.time()
    seed_record: dict[str, Any] = {}
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
        ties=seed_record,
    )
    log(
        f"occupation pass: window {occ['window']}, n_occ {occ['n_occ']}, {len(occ['seeds'])} seeds "
        f"({time.time() - t0:.1f}s)"
    )
    lo, hi = occ["window"]
    n_occ = occ["n_occ"] if occ.get("S_applies") else None
    payload = make_payload(table, window=(lo, hi), n_occ=n_occ, tau_diag=tau_diag, deg_tol=deg_tol)
    if parallel is not None:
        payload = parallel.payload(payload)
    spacing = float(
        np.max(
            np.linalg.norm(reciprocal_in_plane(table.lattice, frame or layer_frame(table.lattice)), axis=1)
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
        stop_vertices=stop_vertices,
        frame=frame,
        log=log,
    )
    integ.seed_ties = seed_record if seeds else None
    integ.tie_scales["energy"] = float(max(np.abs(occ["band_min"]).max(), np.abs(occ["band_max"]).max()))
    integ.must_split()
    log(f"must-split: {integ.mesh.nv} vertices, by rule {integ.mesh.vertices_by_rule}")
    stages: dict[str, Any] = {}
    for name, target in (("tau", tolerance), ("tau2", 0.5 * tolerance)):
        ok = integ.refine(target)
        snap = integ.snapshot(name, mu_out, kTs)
        stage: dict[str, Any] = {
            "ok": bool(ok),
            "vertices": snap["vertices"],
            "leaves": snap["leaves"],
            "levels": {},
        }
        for level in ("main", "diag"):
            if level not in snap:
                continue
            per_T = [
                curves_from_integrals(snap[level], mu_out, kT, area_bz=integ.mesh.area_bz, lo=lo)
                for kT in kTs
            ]
            stage["levels"][level] = {"per_T": per_T, "blocks": snap[level]["blocks"]}
        stages[name] = stage
        status = "converged" if ok else f"stopped ({integ.stop})"
        log(f"stage {name}: {status} at {snap['vertices']} vertices")
        if not ok:
            break
        if name == "tau":
            # The second stage also controls the A3 maximum the first stage found.
            M0 = stage["levels"]["main"]["per_T"][0]["M"]
            j = int(np.argmax(trigonal(M0)["A3"]))
            head = dict(integ.headline_mu)
            head["max_A3"] = float(mu_out[j])
            integ.restart(head)
            A3 = trigonal(M0)["A3"]
            integ.record_restart(mu_out, A3, j, float(max(np.max(A3), np.max(np.abs(M0)))))
    smeared = None
    final = stages.get("tau2") or stages.get("tau")
    if final is not None and kT_max > 0:
        smeared = integ.smeared_check(final["levels"]["main"]["per_T"][-1], mu_out, kT_max)
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


# ------------------------------------------------------------- observables
def layer_band_geometry(
    table: QgeomTable,
    kpoints: np.ndarray,
    *,
    window: tuple[int, int],
    occupied: Sequence[int] = (),
    deg_tol: float = 1e-6,
    chunk: int = 256,
) -> ObservableResult:
    """Band-resolved quantum geometry of the window bands at fractional ``kpoints`` (nk, 3).

    Energies (relative to E_F) of every band; per window band the velocity, the in-W
    metric g, the complete curvature Omega = -2 Im Q and the BCP G = 2 Re sum_m
    A^a_nm A^b_mn / (E_n - E_m), in frame components (e1, e2); per occupied cut c the
    in-W metric and complete curvature of the bands [0, c).
    """
    points = np.asarray(kpoints, float).reshape(-1, 3)
    if not len(points):
        raise ValueError("kpoints must be a nonempty (Nk,3) array")
    res = evaluate_chunks(table, points, chunk=chunk, window=window, occupied=list(occupied), deg_tol=deg_tol)
    cuts = sorted(res["blocks"])
    arrays: dict[str, np.ndarray] = {
        "kpoints": points,
        "energies": res["E"].T,
        "velocity": np.swapaxes(res["v"], 0, 1),
        "metric": np.swapaxes(res["g"], 0, 1),
        "curvature": res["omega"].T,
        "bcp": np.swapaxes(res["G"], 0, 1),
    }
    dims = {
        "kpoints": ("kpoint", "fractional"),
        "energies": ("band", "kpoint"),
        "velocity": ("window_band", "kpoint", "frame"),
        "metric": ("window_band", "kpoint", "frame", "frame2"),
        "curvature": ("window_band", "kpoint"),
        "bcp": ("window_band", "kpoint", "frame", "frame2"),
    }
    units = {
        "kpoints": "fractional",
        "energies": "eV",
        "velocity": "eV Angstrom",
        "metric": "Angstrom^2",
        "curvature": "Angstrom^2",
        "bcp": "Angstrom^2/eV",
    }
    if cuts:
        arrays["manifold_metric"] = np.stack([res["blocks"][c]["g_in"] for c in cuts])
        arrays["manifold_curvature"] = np.stack([res["blocks"][c]["omega"] for c in cuts])
        dims["manifold_metric"] = ("cut", "kpoint", "frame", "frame2")
        dims["manifold_curvature"] = ("cut", "kpoint")
        units["manifold_metric"] = "Angstrom^2"
        units["manifold_curvature"] = "Angstrom^2"
    return ObservableResult(
        arrays,
        dims,
        units,
        {
            "observable": "layer_band_geometry",
            "position_level": table.level,
            "window": [int(window[0]), int(window[1])],
            "cuts": [int(c) for c in cuts],
            "degeneracy_tol_eV": float(deg_tol),
            "frame": {key: [float(x) for x in value] for key, value in table.frame.items()},
            "curvature_convention": "-2 Im Q",
            "metric_completeness": "in-Wannier lower bound",
        },
    )


def nonlinear_hall_response(table: QgeomTable, mu: np.ndarray, **settings: Any) -> ObservableResult:
    """The adaptive zone integrals of :func:`adaptive_response` at the final stage, per temperature and mu.

    Arrays: the electron ``count`` per cell, ``sigma_xy`` (e^2/h), the ``dos``
    (states/eV/cell), the BCP dipole ``M`` (Å/eV), its trigonal amplitude ``A3`` and
    phase ``theta0`` (degrees), the Berry-curvature dipole ``BCD`` (Å) and the Drude
    weight ``drude`` (eV) of the table's level. Metadata hold the window, the occupied
    manifold's Chern number and in-W K when its gap is open, and the integration record.
    """
    res = adaptive_response(table, mu_out=mu, **settings)
    stages = res["stages"]
    final = stages.get("tau2") or stages["tau"]
    per_T = final["levels"]["main"]["per_T"]
    integ = res["integrator"]
    arrays = {
        "mu": np.asarray(res["mu_out"], float),
        "temperature": np.asarray(res["temperatures_K"], float),
        "count": np.stack([c["count"] for c in per_T]),
        "sigma_xy": np.stack([c["sigma_xy_e2_h"] for c in per_T]),
        "dos": np.stack([c["dos"] for c in per_T]),
        "M": np.stack([c["M"] for c in per_T]),
        "A3": np.stack([trigonal(c["M"])["A3"] for c in per_T]),
        "theta0": np.stack([trigonal(c["M"])["theta0_deg"] for c in per_T]),
        "BCD": np.stack([c["BCD"] for c in per_T]),
        "drude": np.stack([c["drude"] for c in per_T]),
    }
    dims = {
        "mu": ("mu",),
        "temperature": ("temperature",),
        "count": ("temperature", "mu"),
        "sigma_xy": ("temperature", "mu"),
        "dos": ("temperature", "mu"),
        "M": ("temperature", "mu", "frame", "frame2", "frame3"),
        "A3": ("temperature", "mu"),
        "theta0": ("temperature", "mu"),
        "BCD": ("temperature", "mu", "frame"),
        "drude": ("temperature", "mu", "frame", "frame2"),
    }
    units = {
        "mu": "eV",
        "temperature": "K",
        "count": "electrons/cell",
        "sigma_xy": "e^2/h",
        "dos": "states/eV/cell",
        "M": "Angstrom/eV",
        "A3": "Angstrom/eV",
        "theta0": "degree",
        "BCD": "Angstrom",
        "drude": "eV",
    }
    manifold: dict[str, Any] = {"applicable": bool(res["S_applies"]), "n_occ": int(res["n_occ"])}
    if res["S_applies"]:
        b = final["levels"]["main"]["blocks"]
        two_pi = 2.0 * math.pi
        manifold.update(
            chern=float(b[1] / two_pi),
            chern_in=float(b[2] / two_pi),
            K_in=(np.array([[b[3], b[4]], [b[4], b[5]]]) / two_pi).tolist(),
        )
    return ObservableResult(
        arrays,
        dims,
        units,
        {
            "observable": "nonlinear_hall_response",
            "position_level": table.level,
            "window": [int(x) for x in res["window"]],
            "occupied_manifold": manifold,
            "integration": {
                "vertices": int(final["vertices"]),
                "converged": bool(stages["tau"]["ok"]),
                "confirmed": bool(stages.get("tau2", {}).get("ok")),
                "stop": integ.stop,
                "vertices_by_rule": {k: int(v) for k, v in integ.mesh.vertices_by_rule.items()},
            },
            "conventions": {
                "omega": "-2 Im Q",
                "sigma_xy": "-(e^2/h)(1/2pi) sum int f Omega, charge -e",
                "M": "sum_n int [dk] (-f') v_a G_bc",
                "frame": {key: [float(x) for x in value] for key, value in table.frame.items()},
            },
        },
    )
