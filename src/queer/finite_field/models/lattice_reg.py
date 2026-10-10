"""Lattice regularization of the four-band model (Eq. 9) as a Wannier tight-binding model.

Purpose: an end-to-end stand-in for the per-D wannier90 models while no DFT data exist. Eq. 9 is put on
a 2D Bravais lattice (lattice constant a, default 4.33 A, the in-plane constant of MnBi2Te4) by
replacing the continuum monomials with lattice functions f(k) = sum_R c_R e^{ik.R} that agree with
them near Gamma up to O((ka)^2) relative corrections.

Triangular lattice (default): a_1 = a (1, 0), a_2 = a (-1/2, sqrt3/2); nearest neighbours +-a_j with
a_3 = -a_1 - a_2, i.e. a_j at 0, 120, 240 deg (sum_j a_j = 0, sum_j a_j a_j^T = 3a^2/2):
    k_lat  = (2 / (3a^2)) sum_j a_j sin(k.a_j)        = k [1 - (ka)^2/8 + O((ka)^4)]
    k2_lat = (4 / (3a^2)) sum_j [1 - cos(k.a_j)]      = k^2 [1 - (ka)^2/16 + O((ka)^4)]
    c3_lat = -(8 / a^3) sum_j sin(k.a_j)               = k^3 cos 3theta [1 - (ka)^2/16 + O((ka)^4)]
(cos^3 and cos^5 sums over three directions 120 deg apart keep only the isotropic and cos 3theta parts,
so the O((ka)^2) corrections are isotropic; sixfold anisotropy enters at O((ka)^4).) Gamma-K is along x.

Square lattice (lattice="square", a_1 = a x, a_2 = a y; x = kx a, y = ky a):
    kx_lat = sin(x)/a,  k2_lat = (2/a^2)(2 - cos x - cos y),
    c3_lat = [-4 sin x - sin 2x + 3 sin(x + y) + 3 sin(x - y)] / a^3 = kx^3 - 3 kx ky^2 + O(k^5),
with anisotropic O((ka)^2) relative corrections.

Model:  H_lat(k) = A (kx_lat sx + ky_lat sy) t0 + (M - B k2_lat) sz t0 + Bl sz tx + VL s0 tz
                   [+ (B0 k2_lat + D3 c3_lat) s0 t0 if include_identity] [+ FourBandModel's d3_layer="tz"
                   and warping terms with c3_lat].
Basis sigma (x) tau, index 2 s + t as in models.fourband: orbitals 0, 2 are the top surface (t_z = +1,
centre z = +z_sep/2), orbitals 1, 3 the bottom one (z = -z_sep/2), all at the same in-plane position,
so the "naive" and "tba" gauges coincide; r(R) = diag(tau) at R = 0, so "wysv" coincides with them too.

Accuracy (triangular, a = 4.33 A, |k| <= 1e-2 1/A, (ka)^2 <= 1.9e-3; VL/VLc in {0, 0.5, 1.6}): the
differences of eps_g, Omega_c and g^{cv} from the continuum, normalized by their maxima over the disk,
are 4e-6 ... 3.6e-4. At small k the leading corrections are the Jacobian of k -> k_lat:
Omega_lat / Omega = 1 - (ka)^2/2, g_rr: 1 - 3(ka)^2/4, g_theta-theta: 1 - (ka)^2/4.
Over the Brillouin zone the traceless model is insulating (M - B k2_lat reaches ~ +0.3 eV at K and M)
and has the continuum's Chern numbers (band 1: -1 -> 0 at VLc; band 0: -1). The identity terms are
dropped by default: on the lattice D3 c3_lat reaches tens of eV at the zone boundary and the model turns
semimetallic, and they never enter Eqs. 7-13.
"""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from ...models.wannier import WannierModel
from .fourband import G_BL, G_VL, G_X, G_Y, G_Z, I4, WARPING_MATRICES, FourBandModel, FourBandParams

LATTICES = ("triangular", "square")


def _add_sin(table: dict, d: tuple[int, int], w: complex) -> None:
    """Add w sin(k.d) = w (e^{ik.d} - e^{-ik.d}) / (2i) to a coefficient table."""
    md = (-d[0], -d[1])
    table[d] = table.get(d, 0.0) - 0.5j * w
    table[md] = table.get(md, 0.0) + 0.5j * w


def lattice_terms(lattice: str = "triangular", a: float = 4.33) -> tuple[np.ndarray, dict[str, dict]]:
    """In-plane primitive vectors (2, 2) (rows, A) and Fourier tables {name: {(n1, n2): c_R}} of the lattice
    functions "kx", "ky", "k2", "c3" (f(k) = sum_R c_R e^{ik.R}, R = n1 a_1 + n2 a_2)."""
    if lattice == "triangular":
        a1 = np.array([a, 0.0])
        a2 = np.array([-0.5 * a, 0.5 * math.sqrt(3.0) * a])
        nn = {(1, 0): a1, (0, 1): a2, (-1, -1): -a1 - a2}  # a_j at 0, 120, 240 deg
        kx, ky, k2, c3 = {}, {}, {(0, 0): 4.0 / a**2}, {}
        for (i, j), d in nn.items():
            for s in (1, -1):
                R = (s * i, s * j)
                kx[R] = -1j * s * d[0] / (3.0 * a * a)  # k_lat = sum_d (-i d / 3a^2) e^{ik.d}
                ky[R] = -1j * s * d[1] / (3.0 * a * a)
                k2[R] = -2.0 / (3.0 * a * a)
            _add_sin(c3, (i, j), -8.0 / a**3)
        return np.stack([a1, a2]), {"kx": kx, "ky": ky, "k2": k2, "c3": c3}
    if lattice == "square":
        kx, ky, c3 = {}, {}, {}
        _add_sin(kx, (1, 0), 1.0 / a)
        _add_sin(ky, (0, 1), 1.0 / a)
        k2 = {
            (0, 0): 4.0 / a**2,
            (1, 0): -1.0 / a**2,
            (-1, 0): -1.0 / a**2,
            (0, 1): -1.0 / a**2,
            (0, -1): -1.0 / a**2,
        }
        for d, w in (((1, 0), -4.0), ((2, 0), -1.0), ((1, 1), 3.0), ((1, -1), 3.0)):
            _add_sin(c3, d, w / a**3)
        return np.array([[a, 0.0], [0.0, a]]), {"kx": kx, "ky": ky, "k2": k2, "c3": c3}
    raise ValueError(f"lattice must be one of {LATTICES}, got {lattice!r}")


def regularized_monomials(kx, ky, a: float = 4.33, lattice: str = "triangular") -> dict[str, np.ndarray]:
    """The lattice functions kx_lat, ky_lat, k2_lat, c3_lat at (kx, ky) (1/A), evaluated from their tables."""
    basis, terms = lattice_terms(lattice, a)
    kx, ky = np.broadcast_arrays(np.asarray(kx, dtype=float), np.asarray(ky, dtype=float))
    out = {}
    for name, table in terms.items():
        val = np.zeros(kx.shape, dtype=complex)
        for (n1, n2), cR in table.items():
            Rv = n1 * basis[0] + n2 * basis[1]
            val += cR * np.exp(1j * (kx * Rv[0] + ky * Rv[1]))
        out[name] = val.real
    return out


def lattice_regularized_fourband(
    params: FourBandParams | FourBandModel | None = None,
    a: float = 4.33,
    lattice: str = "triangular",
    z_sep: float = 10.0,
    *,
    include_identity: bool | None = None,
    c: float | None = None,
) -> WannierModel:
    """Eq. 9 on a lattice (see the module docstring) as a WannierModel in the "tba" gauge, n_occ = 2.

    params: FourBandParams, or a FourBandModel whose options (d3_layer, warping, include_identity) are
    mirrored. include_identity: keep B0 k2_lat + D3 c3_lat (default: False for params, the model's
    setting for a FourBandModel). c: out-of-plane cell length for the lattice matrix (default z_sep + 20 A).
    """
    if isinstance(params, FourBandModel):
        fb = params
        ident = fb.include_identity if include_identity is None else include_identity
    else:
        fb = FourBandModel(params if params is not None else FourBandParams())
        ident = bool(include_identity)
    p = fb.p
    basis, terms = lattice_terms(lattice, a)
    c = float(z_sep + 20.0 if c is None else c)
    if c <= z_sep:
        raise ValueError("the cell length c must exceed z_sep")

    coeffs = {"one": {(0, 0): 1.0}, **terms}
    mats = {
        "kx": p.A * G_X,
        "ky": p.A * G_Y,
        "one": p.M * G_Z + p.Bl * G_BL + p.VL * G_VL,
        "k2": -p.B * G_Z,
        "c3": np.zeros((4, 4), dtype=complex),
    }
    if ident:
        mats["k2"] = mats["k2"] + p.B0 * I4
        if fb.d3_layer == "t0":
            mats["c3"] = mats["c3"] + p.D3 * I4
    if fb.d3_layer == "tz":
        mats["c3"] = mats["c3"] + p.D3 * G_VL
    if fb.warping:
        mats["c3"] = mats["c3"] + fb.warping * WARPING_MATRICES[fb.warping_matrix]

    keys = sorted(
        {(0, 0)} | {R for name, table in coeffs.items() if np.any(mats[name]) for R in table},
        key=lambda R: (R != (0, 0), R),
    )  # R = 0 first
    H_R = np.zeros((len(keys), 4, 4), dtype=complex)
    for i, R in enumerate(keys):
        for name, table in coeffs.items():
            if R in table:
                H_R[i] += table[R] * mats[name]
    lat = np.zeros((3, 3))
    lat[:2, :2] = basis
    lat[2, 2] = c
    R_int = np.array([[n1, n2, 0] for n1, n2 in keys], dtype=np.int64)
    centres = np.zeros((4, 3))
    centres[:, 2] = 0.5 * z_sep * np.array([1.0, -1.0, 1.0, -1.0])  # index 2s + t: t = 0 top, t = 1 bottom
    r_R = np.zeros((len(keys), 4, 4, 3), dtype=complex)
    r_R[0, np.arange(4), np.arange(4), :] = centres  # keys[0] is R = 0
    meta = {
        "source": "lattice_regularized_fourband",
        "lattice_type": lattice,
        "a": a,
        "z_sep": z_sep,
        "c": c,
        "params": asdict(p),
        "include_identity": ident,
        "d3_layer": fb.d3_layer,
        "warping": fb.warping,
        "warping_matrix": fb.warping_matrix,
    }
    model = WannierModel(R_int, H_R, reciprocal=2 * np.pi * np.linalg.inv(lat).T)
    model.lattice = lat
    model.centres = centres
    model.n_occ = 2
    model.meta = meta
    model.layer_projector = lambda: np.diag([1.0, 0.0, 1.0, 0.0])
    return model


__all__ = ["LATTICES", "lattice_regularized_fourband", "lattice_terms", "regularized_monomials"]
