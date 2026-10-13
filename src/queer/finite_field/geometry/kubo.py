"""Sum-over-states (Kubo) band geometry in the field frame.

With X^i = U^dag (dH/dk_i) U (i in {s, t}), for a band n:

    Omega_n      = -2 Im sum_{m != n} X^s_nm X^t_mn / (eps_n - eps_m)^2          [A^2]
    g^{nm}_ij    = Re(X^i_nm X^j_mn) / (eps_n - eps_m)^2                          [A^2]

g^{cv} is the band-resolved (two-band) quantum metric of Eq. 8 (|A_x,cv|^2 = g_xx in Methods 6);
the total metric of band n is sum_{m != n} g^{nm}. Only the rows n in {v, c} of X are formed, so
the cost after `eigh` is O(n^2) per k-point.

Degeneracy guard: pairs with |eps_n - eps_m| <= deg_tol are dropped from the sums and the k-point is
flagged (deg_flag). For symmetry-protected crossings (e.g. the t_x sectors of Eq. 9 at VL = 0 on
|k| = sqrt(M/B)) the dropped matrix element is physically zero. At a gap closing it is not: within
eps_g <= deg_tol of a touching point (|k| <~ 3e-12 1/A at VLc) the guard also drops the (v, c) pair,
whose |<c|dH|v>| = hbar v is finite, so Omega loses its singular term and g^{cv} = 0 there. This is
consistent only because CrossingPolicy.touching_floor equals deg_tol (both 1e-12 eV): a crossing with
eps_g < touching_floor is a touching point (E_c = 0, P = 1) whose metric is never used. Changing one
of the two alone gives E_c = NaN ("non-positive metric") at crossings with touching_floor <= eps_g <=
deg_tol. Near Gamma, ModelGeometry serves |k| < k_patch from its precision patch instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..models.base import rotate_derivatives


@dataclass
class BandGeometry:
    """Geometry of the (v, c) pair at a set of k-points; metric components ordered [ss, st, tt]."""

    eps_v: np.ndarray
    eps_c: np.ndarray
    omega_v: np.ndarray
    omega_c: np.ndarray
    g_cv: np.ndarray  # (..., 3)
    g_v_total: np.ndarray  # (..., 3)
    g_c_total: np.ndarray  # (..., 3)
    v_cv: np.ndarray  # (..., 2) complex: <c|dH/ds|v>, <c|dH/dt|v> (gauge-dependent phase), eV*A
    deg_flag: np.ndarray  # (...,) bool
    eps: np.ndarray | None = None  # (..., n) all eigenvalues if requested
    extras: dict = field(default_factory=dict)

    @property
    def eps_g(self) -> np.ndarray:
        return self.extras.get("gap", self.eps_c - self.eps_v)

    def omega(self, mode: str = "c") -> np.ndarray:
        """Berry curvature entering Eqs. 10-11: "c", "v" or "c_minus_v"."""
        if mode == "c":
            return self.omega_c
        if mode == "v":
            return self.omega_v
        if mode == "c_minus_v":
            return self.omega_c - self.omega_v
        raise ValueError(f"unknown Omega mode {mode!r}")

    def metric(self, mode: str = "cv") -> np.ndarray:
        """Metric tensor [ss, st, tt]: "cv" (band-resolved), "v_total" or "c_total"."""
        if mode == "cv":
            return self.g_cv
        if mode == "v_total":
            return self.g_v_total
        if mode == "c_total":
            return self.g_c_total
        raise ValueError(f"unknown metric mode {mode!r}")

    def reshape(self, shape) -> "BandGeometry":
        def r(a, tail=()):
            return None if a is None else a.reshape(tuple(shape) + tail)

        n = None if self.eps is None else self.eps.shape[-1]
        extras = dict(self.extras)
        if "gap" in extras:
            extras["gap"] = np.asarray(extras["gap"]).reshape(shape)
        return BandGeometry(
            r(self.eps_v),
            r(self.eps_c),
            r(self.omega_v),
            r(self.omega_c),
            r(self.g_cv, (3,)),
            r(self.g_v_total, (3,)),
            r(self.g_c_total, (3,)),
            r(self.v_cv, (2,)),
            r(self.deg_flag),
            None if self.eps is None else r(self.eps, (n,)),
            extras,
        )


def band_gap(
    model, kx, ky, bands: tuple[int, int] | None = None, max_chunk: int = 65536, mem_bytes: float = 256e6
) -> np.ndarray:
    """eps_c - eps_v only (eigvalsh), with the broadcast shape of (kx, ky)."""
    kx, ky = np.broadcast_arrays(np.asarray(kx, dtype=float), np.asarray(ky, dtype=float))
    v, c = bands if bands is not None else model.band_pair()
    if getattr(model, "native_band_geometry", False):
        return model.gap(kx, ky, bands=(v, c), mem_bytes=mem_bytes)
    kxf, kyf = kx.ravel(), ky.ravel()
    out = np.empty(kxf.size)
    chunk = _chunk_size(model.n_orb, max_chunk, mem_bytes)
    for i0 in range(0, kxf.size, chunk):
        sl = slice(i0, min(kxf.size, i0 + chunk))
        eps = np.linalg.eigvalsh(model.hamiltonian(kxf[sl], kyf[sl]))
        out[sl] = eps[:, c] - eps[:, v]
    return out.reshape(kx.shape)


def _chunk_size(n: int, max_chunk: int, mem_bytes: float) -> int:
    # eigenvectors + two derivative matrices + temporaries, complex128
    per_point = 16.0 * n * n * 6.0
    return int(max(1, min(max_chunk, mem_bytes // per_point)))


def band_geometry(
    model,
    kx,
    ky,
    angle: float = 0.0,
    bands: tuple[int, int] | None = None,
    deg_tol: float = 1e-12,
    keep_eps: bool = False,
    max_chunk: int = 65536,
    mem_bytes: float = 256e6,
) -> BandGeometry:
    """Band geometry of the (v, c) pair at Cartesian (kx, ky) (1/A) for a field at `angle`.

    Returns arrays with the broadcast shape of (kx, ky). Wannier models in the "wysv" gauge (full
    position matrix) are routed to their covariant-velocity geometry (models/wannier.py).
    """
    if getattr(model, "native_band_geometry", False):
        return model.geometry(
            kx, ky, angle=angle, bands=bands, deg_tol=deg_tol, keep_eps=keep_eps, mem_bytes=mem_bytes
        )
    kx, ky = np.broadcast_arrays(np.asarray(kx, dtype=float), np.asarray(ky, dtype=float))
    shape = kx.shape
    kxf, kyf = kx.ravel(), ky.ravel()
    nk = kxf.size
    v, c = bands if bands is not None else model.band_pair()
    sel = np.array([v, c])
    n = model.n_orb
    chunk = _chunk_size(n, max_chunk, mem_bytes)

    out = {name: np.empty(nk) for name in ("eps_v", "eps_c", "omega_v", "omega_c")}
    g_cv = np.empty((nk, 3))
    g_tot = np.empty((nk, 2, 3))
    v_cv = np.empty((nk, 2), dtype=complex)
    deg = np.zeros(nk, dtype=bool)
    eps_all = np.empty((nk, n)) if keep_eps else None
    self_mask = np.zeros((2, n), dtype=bool)
    self_mask[0, v] = self_mask[1, c] = True

    for i0 in range(0, nk, chunk):
        sl = slice(i0, min(nk, i0 + chunk))
        if hasattr(model, "hamiltonian_and_derivatives"):  # one fused Fourier transform (Wannier models)
            H, dHx, dHy = model.hamiltonian_and_derivatives(kxf[sl], kyf[sl])
        else:
            H = model.hamiltonian(kxf[sl], kyf[sl])
            dHx, dHy = model.dhamiltonian(kxf[sl], kyf[sl])
        dHs, dHt = rotate_derivatives(dHx, dHy, angle)
        eps, U = np.linalg.eigh(H)
        UsH = np.conj(np.swapaxes(U[:, :, sel], -1, -2))  # (m, 2, n)
        Xs = (UsH @ dHs) @ U  # (m, 2, n): rows v, c
        Xt = (UsH @ dHt) @ U
        d = eps[:, sel][:, :, None] - eps[:, None, :]  # (m, 2, n)
        ok = (np.abs(d) > deg_tol) & ~self_mask
        inv_d2 = np.where(ok, 1.0 / np.where(ok, d, 1.0) ** 2, 0.0)
        cross = Xs * np.conj(Xt)  # X^s_nm X^t_mn
        omega = -2.0 * np.sum(cross.imag * inv_d2, axis=-1)  # (m, 2)
        gss = (np.abs(Xs) ** 2) * inv_d2
        gst = cross.real * inv_d2
        gtt = (np.abs(Xt) ** 2) * inv_d2
        out["eps_v"][sl] = eps[:, v]
        out["eps_c"][sl] = eps[:, c]
        out["omega_v"][sl] = omega[:, 0]
        out["omega_c"][sl] = omega[:, 1]
        g_cv[sl] = np.stack([gss[:, 0, c], gst[:, 0, c], gtt[:, 0, c]], axis=-1)
        g_tot[sl] = np.stack([gss.sum(-1), gst.sum(-1), gtt.sum(-1)], axis=-1)
        v_cv[sl] = np.stack([Xs[:, 1, v], Xt[:, 1, v]], axis=-1)
        deg[sl] = np.any((np.abs(d) <= deg_tol) & ~self_mask, axis=(-1, -2))
        if keep_eps:
            eps_all[sl] = eps

    geo = BandGeometry(
        out["eps_v"],
        out["eps_c"],
        out["omega_v"],
        out["omega_c"],
        g_cv,
        g_tot[:, 0],
        g_tot[:, 1],
        v_cv,
        deg,
        eps_all,
    )
    return geo.reshape(shape)
