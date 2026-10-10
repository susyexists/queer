"""Geometry provider from precomputed grids (user-supplied Wannier geometry, or a sampled master grid).

NPZ schema (internal units, model frame):
  layout "cartesian": kx (nx,), ky (ny,) strictly increasing, 1/A; fields of shape (nx, ny)
  layout "polar":     r (nr,) increasing with r[0] > 0, 1/A; phi (nphi,) uniform on [0, 2 pi); fields (nr, nphi)
  eps_v, eps_c (eV); omega_c, omega_v (A^2); g_cv (..., 3) = [xx, xy, yy] band-resolved metric (A^2);
  optional g_v_total, g_c_total (..., 3); "__meta__" JSON (band indices, gauge, D, metric kind).

The interpolated quantities are smooth surrogates that are exact constants or polynomials for a
two-band Dirac gap, eps_g^2, Omega eps_g^3 and g eps_g^4 (plus the band centre), so sharply peaked
Omega and g near small gaps are reproduced from coarse grids. Bicubic splines; polar grids are
periodic in phi. Wavefunctions are not available, so the geometric (shift-vector) phase and the
TDSE validator cannot use this provider.

Queries outside the grid:
  * polar, r < r[0]: every surrogate is extrapolated linearly in r^2 from the first two radial nodes
    (at the query's phi). Near Gamma eps_g^2 = 4 (m0^2 + hbar_v^2 r^2) + O(r^4), and Omega eps_g^3 and
    the Cartesian g eps_g^4 are linear in r^2 as well, so the 1/k curvature at the critical point
    continues below r[0].
  * polar, r > r[-1], and cartesian queries outside the [kx, ky] box: clamped to the boundary and
    flagged in `BandGeometry.extras["outside_grid"]` (boolean mask; transport/lines.py reports it).
    `extras["below_r0"]` marks the extrapolated points.
An interpolated eps_g^2 <= 0 (a closing gap that the grid does not resolve) raises ValueError.

Choosing r[0] near D_c (gap at Gamma 2|m0| with |m0| <~ |m2| r[0]^2):
  * The two-node secant misses the k^4 term of eps_g^2 = 4 (m0^2 + hbar_v^2 k^2 + m2^2 k^4 + ...), so
    at D_c the extrapolated eps_g^2 turns negative (ValueError) below r ~ |m2| r[0]^2 / hbar_v. The
    transverse quadrature reaches |t| ~ quadrature.dmin / 2, hence r[0] <~ sqrt(hbar_v dmin / (2 |m2|))
    (four-band model, dmin = 1e-12 1/A: r[0] <~ 4e-8 1/A; r[0] = 1e-7 and 1e-6 fail at VL = VLc).
    Raising is deliberate: continuing eps_g^2 ~ r^2 instead runs, but the Omega eps_g^3 secant then
    errs by ~(r[0]/r)^2 (J_H at 1 V/m off by -88% for r[0] = 1e-6).
  * A relative data error err at r[0] and r[1] continues below r[0] as an Omega error of about
    2 err / (r[1]^2/r[0]^2 - 1) * (r[0]/r)^2. Float64 sums over states have such errors at D_c
    (four-band: Omega_c off by 4e-5 at k = 1e-8 and 3e-3 at 1e-9 1/A), so sample a provider that is
    accurate there (ModelGeometry.at_cartesian applies the near-Gamma patch of isotropic models) when
    the grid must serve fields below ~100 V/m at D_c.
  Measured at VL = VLc (four-band, PAPER, 240 x 48 polar grid up to 0.045 1/A, E = 1 ... 1e4 V/m):
  samples of ModelGeometry (patched) with r[0] = 1e-8, 1e-10 or 1e-12 reproduce J_H to <= 1e-6 and
  hbar_v, C_s to 1e-7 (also below r[0]); unpatched float64 samples gave J_H errors of -11%
  (r[0] = 1e-8) and +1.2% (1e-9) at 1 V/m and of -1.4% / +0.15% at 10 V/m. Away from D_c
  (|m0| >> |m2| r[0]^2) any r[0] in the Dirac regime (r[0] << hbar_v/|m2|, four-band ~4e-3 1/A) works.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import RectBivariateSpline

from .._multiplicity import validated_spin_degeneracy
from .._storage import load_npz, save_npz, stable_hash
from ..models.base import field_frame
from .kubo import BandGeometry

_TENSORS = ("g_cv", "g_v_total", "g_c_total")


class GriddedGeometry:
    model = None  # no wavefunctions
    isotropic = False

    def __init__(self, arrays: dict, angle: float = 0.0, meta: dict | None = None):
        self.angle = float(angle)
        if not math.isfinite(self.angle):
            raise ValueError("grid angle must be finite radians")
        self.meta = dict(meta or {})
        recorded = "spin_degeneracy" in self.meta
        self.spin_degeneracy = validated_spin_degeneracy(self.meta.get("spin_degeneracy", 1))
        self.meta["spin_degeneracy"] = self.spin_degeneracy
        self.meta.setdefault(
            "spin_degeneracy_source", "archive" if recorded else "legacy_default_single_pair"
        )
        self.layout = "polar" if "r" in arrays else "cartesian"
        a = {k: np.asarray(v, dtype=float) for k, v in arrays.items()}
        required = {"eps_v", "eps_c", "omega_c", "omega_v", "g_cv"}
        required.update({"r", "phi"} if self.layout == "polar" else {"kx", "ky"})
        if missing := required - set(a):
            raise ValueError(f"Geometry grid is missing fields: {sorted(missing)}")
        coordinates = ("r", "phi") if self.layout == "polar" else ("kx", "ky")
        for name in coordinates:
            if a[name].ndim != 1 or len(a[name]) < 4 or not np.isfinite(a[name]).all():
                raise ValueError(
                    f"Grid coordinate {name} must contain at least four finite one-dimensional nodes"
                )
        shape = tuple(len(a[name]) for name in coordinates)
        for name in ("eps_v", "eps_c", "omega_c", "omega_v", "gap", *_TENSORS):
            if name in a:
                expected = shape + ((3,) if name in _TENSORS else ())
                if a[name].shape != expected or not np.isfinite(a[name]).all():
                    raise ValueError(f"Grid field {name} must be finite with shape {expected}")
        if self.layout == "polar":
            self.r, self.phi = a["r"], a["phi"]
            if self.r[0] <= 0 or np.any(np.diff(self.r) <= 0):
                raise ValueError("r must be positive and increasing")
            dphi = np.diff(self.phi)
            if not np.allclose(dphi, 2 * math.pi / self.phi.size) or abs(self.phi[0]) > 1e-12:
                raise ValueError("phi must be uniform on [0, 2 pi)")
            self.bounds = (self.r[0], self.r[-1])
        else:
            self.kx, self.ky = a["kx"], a["ky"]
            if np.any(np.diff(self.kx) <= 0) or np.any(np.diff(self.ky) <= 0):
                raise ValueError("kx, ky must be strictly increasing")
            self.bounds = ((self.kx[0], self.kx[-1]), (self.ky[0], self.ky[-1]))
        difference = a["eps_c"] - a["eps_v"]
        eg = a.get("gap", difference)
        if "gap" in a:
            tolerance = 16 * np.finfo(float).eps * (np.abs(a["eps_v"]) + np.abs(a["eps_c"])) + 1e-10 * np.abs(
                eg
            )
            if np.any(np.abs(eg - difference) > tolerance):
                raise ValueError("Grid gap conflicts with its band energies")
        if np.any(eg <= 0):
            raise ValueError("eps_c - eps_v must be positive on the grid (gap closings must be avoided)")
        sur = {
            "gap2": eg * eg,
            "mid": 0.5 * (a["eps_c"] + a["eps_v"]),
            "om_c": a["omega_c"] * eg**3,
            "om_v": a["omega_v"] * eg**3,
        }
        self.tensors = [t for t in _TENSORS if t in a]
        if "g_cv" not in self.tensors:
            raise ValueError("g_cv (band-resolved metric) is required")
        for t in self.tensors:
            for i, comp in enumerate(("xx", "xy", "yy")):
                sur[f"{t}_{comp}"] = a[t][..., i] * eg**4
        self._raw = a
        self._splines = {k: self._spline(v) for k, v in sur.items()}

    # ---- construction ---------------------------------------------------------------------------
    def _spline(self, z):
        if self.layout == "cartesian":
            return RectBivariateSpline(self.kx, self.ky, z, kx=3, ky=3)
        pad = 3
        phi = np.concatenate([self.phi[-pad:] - 2 * math.pi, self.phi, self.phi[:pad] + 2 * math.pi])
        zz = np.concatenate([z[:, -pad:], z, z[:, :pad]], axis=1)
        return RectBivariateSpline(self.r, phi, zz, kx=3, ky=3)

    @classmethod
    def from_npz(cls, path, angle: float = 0.0) -> "GriddedGeometry":
        arrays, meta = load_npz(path)
        return cls(arrays, angle, meta)

    def save(self, path):
        keys = ("r", "phi") if self.layout == "polar" else ("kx", "ky")
        fields = (
            "eps_v",
            "eps_c",
            "omega_c",
            "omega_v",
            *self.tensors,
            *(("gap",) if "gap" in self._raw else ()),
        )
        return save_npz(path, {k: self._raw[k] for k in (*keys, *fields)}, self.meta)

    @classmethod
    def sample(
        cls, provider, *, r=None, phi=None, kx=None, ky=None, angle: float = 0.0, meta: dict | None = None
    ):
        """Master grid from a geometry provider (e.g. ModelGeometry of a Wannier model)."""
        if r is not None:
            R, PH = np.meshgrid(np.asarray(r, float), np.asarray(phi, float), indexing="ij")
            X, Y = R * np.cos(PH), R * np.sin(PH)
            coords = {"r": np.asarray(r, float), "phi": np.asarray(phi, float)}
        else:
            X, Y = np.meshgrid(np.asarray(kx, float), np.asarray(ky, float), indexing="ij")
            coords = {"kx": np.asarray(kx, float), "ky": np.asarray(ky, float)}
        geo = provider.at_cartesian(X, Y)
        arrays = {
            **coords,
            "gap": geo.eps_g,
            "eps_v": geo.eps_v,
            "eps_c": geo.eps_c,
            "omega_c": geo.omega_c,
            "omega_v": geo.omega_v,
            "g_cv": geo.g_cv,
            "g_v_total": geo.g_v_total,
            "g_c_total": geo.g_c_total,
        }
        m = {
            "sampled_from": provider.describe() if hasattr(provider, "describe") else str(provider),
            **(meta or {}),
        }
        multiplicity = validated_spin_degeneracy(getattr(provider, "spin_degeneracy", 1))
        if "spin_degeneracy" in m and validated_spin_degeneracy(m["spin_degeneracy"]) != multiplicity:
            raise ValueError("Sampled grid spin_degeneracy conflicts with the source provider")
        m["spin_degeneracy"] = multiplicity
        m["spin_degeneracy_source"] = "sampled_provider"
        return cls(arrays, angle, m)

    def fingerprint(self) -> str:
        """Content hash of the grid arrays, layout, field angle and metadata (cache key of harmonic maps)."""
        return stable_hash(
            {"provider": "gridded", "layout": self.layout, "angle": self.angle, "meta": self.meta},
            {k: self._raw[k] for k in sorted(self._raw)},
        )

    # ---- provider API -----------------------------------------------------------------------------
    def _eval(self, kx, ky):
        """Surrogates at Cartesian k; returns (values, outside mask, below-r[0] mask)."""
        kx, ky = np.broadcast_arrays(np.asarray(kx, float), np.asarray(ky, float))
        below = np.zeros(kx.shape, dtype=bool)
        if self.layout == "cartesian":
            x = np.clip(kx, *self.bounds[0])
            y = np.clip(ky, *self.bounds[1])
            outside = (x != kx) | (y != ky)
            vals = {k: s.ev(x.ravel(), y.ravel()).reshape(x.shape) for k, s in self._splines.items()}
            return vals, outside, below
        r = np.hypot(kx, ky)
        phi = np.mod(np.arctan2(ky, kx), 2 * math.pi)
        outside = r > self.r[-1]
        below = r < self.r[0]
        rc = np.clip(r, *self.bounds)
        vals = {k: s.ev(rc.ravel(), phi.ravel()).reshape(r.shape) for k, s in self._splines.items()}
        if np.any(below):
            r0, r1 = self.r[0], self.r[1]
            ph = phi[below]
            u = (r[below] ** 2 - r0 * r0) / (r1 * r1 - r0 * r0)  # linear in r^2 through the first two nodes
            for k, s in self._splines.items():
                f0 = s.ev(np.full(ph.shape, r0), ph)
                f1 = s.ev(np.full(ph.shape, r1), ph)
                vals[k][below] = f0 + u * (f1 - f0)
        return vals, outside, below

    def _eps_g(self, gap2, kx, ky) -> np.ndarray:
        bad = ~(gap2 > 0)
        if np.any(bad):
            i = np.flatnonzero(bad)[0]
            raise ValueError(
                f"GriddedGeometry: interpolated eps_g^2 <= 0 at {int(bad.sum())} point(s), e.g. "
                f"k = ({np.ravel(kx)[i]:.3e}, {np.ravel(ky)[i]:.3e}) 1/A: the grid does not resolve a "
                "(nearly) closing gap; refine it there, or use a polar grid around the closing point with a "
                "small enough r[0] (module docstring)"
            )
        return np.sqrt(gap2)

    def at_cartesian(self, kx, ky, angle: float = 0.0) -> BandGeometry:
        kx, ky = np.broadcast_arrays(np.asarray(kx, float), np.asarray(ky, float))
        v, outside, below = self._eval(kx, ky)
        eg = self._eps_g(v["gap2"], kx, ky)
        c, sn = math.cos(angle), math.sin(angle)

        def tensor(name):
            if f"{name}_xx" not in v:
                return np.full(kx.shape + (3,), np.nan)
            xx, xy, yy = (v[f"{name}_{comp}"] / eg**4 for comp in ("xx", "xy", "yy"))
            gss = c * c * xx + 2 * c * sn * xy + sn * sn * yy
            gtt = sn * sn * xx - 2 * c * sn * xy + c * c * yy
            gst = -c * sn * xx + (c * c - sn * sn) * xy + c * sn * yy
            return np.stack([gss, gst, gtt], axis=-1)

        return BandGeometry(
            eps_v=v["mid"] - 0.5 * eg,
            eps_c=v["mid"] + 0.5 * eg,
            omega_v=v["om_v"] / eg**3,
            omega_c=v["om_c"] / eg**3,
            g_cv=tensor("g_cv"),
            g_v_total=tensor("g_v_total"),
            g_c_total=tensor("g_c_total"),
            v_cv=np.full(kx.shape + (2,), np.nan + 0j),
            deg_flag=np.zeros(kx.shape, dtype=bool),
            extras={"outside_grid": outside, "below_r0": below, "gap": eg},
        )

    def geometry(self, s, t) -> BandGeometry:
        kx, ky = field_frame(s, t, self.angle)
        return self.at_cartesian(kx, ky, self.angle)

    __call__ = geometry

    def gap(self, s, t) -> np.ndarray:
        kx, ky = field_frame(s, t, self.angle)
        v, _, _ = self._eval(kx, ky)
        return self._eps_g(v["gap2"], kx, ky)

    def diabatic_radius(self):
        return None

    def exact_branch_points(self, t):
        return None

    def describe(self) -> dict:
        return {"provider": "gridded", "layout": self.layout, "angle": self.angle, **self.meta}
