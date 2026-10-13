"""Geometry providers: band geometry in the field frame (s along E, t perpendicular).

`ModelGeometry` wraps a band model:
  * the identity part is dropped when the model offers `traceless()` (eps_g, Omega, g are unchanged
    and the float64 floor near Gamma is lowered);
  * near a gap closing at Gamma (eps_g(Gamma) < patch_gap) points with |k| < k_patch are served by a
    precision patch: the extended-precision `RadialPatch` for isotropic four-band models, the float64
    Lowdin downfolding `LowdinPatch` (geometry/lowdin.py) for every other supported model (Wannier models
    in any gauge, polynomial models). `geometry`, `gap` and `at_cartesian` all apply it; with a Lowdin
    patch every energy is measured from its reference E_ref (the midgap at Gamma, `energy_ref`);
  * if no patch can be built, `precision_floor` records the radius below which float64 Omega errs.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import numpy as np

from .._multiplicity import validated_spin_degeneracy
from ..models.base import field_frame
from .kubo import BandGeometry, band_gap, band_geometry
from .lowdin import LowdinPatch
from .patch import RadialPatch

_EPS = np.finfo(float).eps


def _empty_geometry(shape) -> BandGeometry:
    def e(tail=(), dtype=float):
        return np.empty(tuple(shape) + tail, dtype=dtype)

    return BandGeometry(
        e(), e(), e(), e(), e((3,)), e((3,)), e((3,)), e((2,), complex), np.zeros(shape, dtype=bool)
    )


def _assign(dst: BandGeometry, mask, src: BandGeometry) -> None:
    for name in (
        "eps_v",
        "eps_c",
        "omega_v",
        "omega_c",
        "g_cv",
        "g_v_total",
        "g_c_total",
        "v_cv",
        "deg_flag",
    ):
        getattr(dst, name)[mask] = getattr(src, name)
    if "gap" not in dst.extras:
        dst.extras["gap"] = np.empty(np.shape(dst.eps_v), dtype=float)
    dst.extras["gap"][mask] = src.eps_g


class ModelGeometry:
    """Band geometry of a model in the field frame of `angle`, with the near-Gamma precision patch.

    patch: "auto" (patch when eps_g(Gamma) < patch_gap; if none can be built, record precision_floor),
        True (always; RadialPatch if isotropic, else LowdinPatch; ValueError if impossible), "radial",
        "lowdin", or False (never).
    k_patch: patch radius, 1/A. deg_tol: degeneracy guard of the float64 sum over states (kubo.py).
    Attributes: patch, energy_ref (eV subtracted from all energies), gap_gamma (eV), precision_floor
    (None or dict, see _estimate_precision_floor), patch_error (why an automatic patch failed, or None).
    """

    def __init__(
        self,
        model,
        angle: float = 0.0,
        *,
        use_traceless: bool = True,
        patch: str | bool = "auto",
        k_patch: float = 1e-5,
        patch_gap: float = 1e-6,
        deg_tol: float = 1e-12,
    ):
        if patch not in (True, False, "auto", "radial", "lowdin"):
            raise ValueError(f"patch must be True, False, 'auto', 'radial' or 'lowdin', got {patch!r}")
        self.source_model = model
        self.spin_degeneracy = validated_spin_degeneracy(getattr(model, "spin_degeneracy", 1))
        self.model = model.traceless() if (use_traceless and hasattr(model, "traceless")) else model
        self.angle = float(angle)
        self.k_patch = k_patch
        self.patch_gap = patch_gap
        self.deg_tol = deg_tol
        self.mem_bytes = 256e6
        self.patch = None
        self.patch_error = None
        self.energy_ref = 0.0
        self.precision_floor = None
        self.gap_gamma = float(band_gap(self.model, 0.0, 0.0))
        near_critical = self.gap_gamma < patch_gap
        if patch in (True, "radial", "lowdin") or (patch == "auto" and near_critical):
            kind = patch if patch in ("radial", "lowdin") else ("radial" if self._radial_ok else "lowdin")
            if kind == "radial":
                if not self._radial_ok:
                    raise ValueError("the radial precision patch requires an isotropic four-band model")
                self.patch = RadialPatch(model.p, k_max=k_patch)
            else:
                try:
                    if not getattr(self.model, "supports_lowdin_patch", True):
                        raise ValueError("This model has no certified nonorthogonal precision patch")
                    self.patch = LowdinPatch(self.model, k_max=k_patch)
                    self.energy_ref = self.patch.E_ref
                except ValueError as err:
                    if patch != "auto":
                        raise
                    self.patch_error = str(err)
        if self.patch is None and near_critical:
            self.precision_floor = self._estimate_precision_floor()
        elif self.patch is not None and not getattr(self.patch, "exact_pair", True):
            self.precision_floor = {
                "reason": f"{self.patch.P.size} bands within the patch tolerance of the v/c pair at Gamma: "
                "pair term from float64 eigenvectors of H_eff (relative errors grow toward k -> 0 "
                "at exact criticality and with the patch radius)",
                "k_floor": None,
                "n_P": int(self.patch.P.size),
            }
        if self.patch is not None and self.precision_floor is None:
            ov = self.patch_overlap()
            if not (ov is not None and ov <= 1e-3):
                self.precision_floor = {
                    "reason": f"patch overlap certificate {ov} > 1e-3 (patch and float64 disagree at "
                    "the patch radius: a band joining the v/c pair inside the patch?)",
                    "k_floor": None,
                    "patch_overlap": None if ov is None else float(ov),
                }

    @property
    def _radial_ok(self) -> bool:
        return self.isotropic and hasattr(self.source_model, "p")

    # ---- evaluation ----------------------------------------------------------------------------
    def cartesian(self, s, t):
        return field_frame(s, t, self.angle)

    def _near(self, kx, ky):
        if self.patch is None:
            return np.zeros(np.shape(kx), dtype=bool)
        return np.hypot(kx, ky) < self.k_patch

    def _evaluate(self, kx, ky, angle: float, s, t) -> BandGeometry:
        """Geometry at Cartesian (kx, ky) for a field at `angle`; (s, t) are the same points in that frame."""
        kx, ky, s, t = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (kx, ky, s, t)))
        near = self._near(kx, ky)
        if not near.any():
            geo = band_geometry(
                self.model, kx, ky, angle=angle, deg_tol=self.deg_tol, mem_bytes=self.mem_bytes
            )
            if self.energy_ref:
                geo.eps_v, geo.eps_c = geo.eps_v - self.energy_ref, geo.eps_c - self.energy_ref
            return geo
        geo = _empty_geometry(kx.shape)
        far = ~near
        if far.any():
            gf = band_geometry(
                self.model, kx[far], ky[far], angle=angle, deg_tol=self.deg_tol, mem_bytes=self.mem_bytes
            )
            gf.eps_v, gf.eps_c = gf.eps_v - self.energy_ref, gf.eps_c - self.energy_ref
            _assign(geo, far, gf)
            geo.extras.update({k: v for k, v in gf.extras.items() if np.ndim(v) == 0})
        _assign(geo, near, self.patch.evaluate(kx[near], ky[near], angle, s=s[near], t=t[near]))
        geo.extras["patched"] = near
        return geo

    def _on_line(self, t):
        """Use factorized Bloch sums when all points share one field-line offset."""
        on_line = getattr(self.model, "on_line", None)
        tf = np.ravel(t)
        if on_line is None or tf.size == 0 or not np.all(tf == tf[0]):
            return nullcontext()
        if getattr(self.model, "native_band_geometry", False):
            return on_line(self.angle, float(tf[0]), mem_bytes=self.mem_bytes)
        return on_line(self.angle, float(tf[0]))

    def geometry(self, s, t) -> BandGeometry:
        """Band geometry at field-frame coordinates (broadcast shape of s and t)."""
        s, t = np.broadcast_arrays(np.asarray(s, dtype=float), np.asarray(t, dtype=float))
        kx, ky = self.cartesian(s, t)
        with self._on_line(t):
            return self._evaluate(kx, ky, self.angle, s, t)

    __call__ = geometry

    def gap(self, s, t) -> np.ndarray:
        s, t = np.broadcast_arrays(np.asarray(s, dtype=float), np.asarray(t, dtype=float))
        kx, ky = self.cartesian(s, t)
        near = self._near(kx, ky)
        with self._on_line(t):
            if not near.any():
                return band_gap(self.model, kx, ky, mem_bytes=self.mem_bytes)
            g = np.empty(kx.shape)
            if (~near).any():
                g[~near] = band_gap(self.model, kx[~near], ky[~near], mem_bytes=self.mem_bytes)
        g[near] = self.patch.gap(kx[near], ky[near])
        return g

    def at_cartesian(self, kx, ky) -> BandGeometry:
        """Geometry at Cartesian k in the model frame with metric components in the model frame (patched)."""
        kx, ky = np.broadcast_arrays(np.asarray(kx, dtype=float), np.asarray(ky, dtype=float))
        return self._evaluate(kx, ky, 0.0, kx, ky)

    # ---- isotropic geometry ----------------------------------------------------------------------
    @property
    def isotropic(self) -> bool:
        return bool(getattr(self.source_model, "is_isotropic_geometry", False))

    def radial_extrema(self, k_max: float, n: int = 4097, prominence_rel: float = 1e-12):
        """Local extrema of eps_g(|k|) on (0, k_max) for isotropic models (cached per arguments).

        Same rules as lz.crossings.gap_minima: candidates need a prominence above
        tau = prominence_rel * max(eps_g), and adjacent candidates without a barrier above tau between
        them form one basin (one extremum), so rounding wiggles on flat bottoms are not extrema.
        Returns (radial minima, radial maxima, zero_is_min). Whether k = 0 is a minimum is inferred
        from the first resolvable extremum (the profile is flat to rounding at the origin).
        """
        if not self.isotropic:
            raise ValueError("radial extrema require isotropic geometry")
        key = (float(k_max), int(n), float(prominence_rel))
        cache = self.__dict__.setdefault("_radial_cache", {})
        if key in cache:
            return cache[key]
        from scipy.optimize import minimize_scalar
        from scipy.signal import find_peaks

        k = np.unique(
            np.concatenate([np.linspace(0.0, k_max, n), np.geomspace(1e-12 * k_max, k_max, n // 4)])
        )
        g = self.gap(k, np.zeros_like(k))
        tau = prominence_rel * float(np.max(g)) + 1e-30

        def gap1(r):
            return float(self.gap(np.array([r]), np.array([0.0]))[0])

        def extrema(sign):  # sign = +1: minima of eps_g, -1: maxima
            f = sign * g
            cand, _ = find_peaks(-f, prominence=tau)
            cand = cand[(cand > 0) & (cand < k.size - 1)]
            basins = []
            for i in cand:
                if basins:
                    j = basins[-1][-1]
                    if np.max(f[j : i + 1]) - max(f[i], f[j]) < tau:
                        basins[-1].append(i)
                        continue
                basins.append([i])
            obj = (lambda r: gap1(r) ** 2) if sign > 0 else (lambda r: -gap1(r))  # noqa: E731
            out = []
            for b in basins:
                i = b[len(b) // 2]
                lo, hi = k[b[0] - 1], k[b[-1] + 1]
                res = minimize_scalar(
                    obj, bounds=(lo, hi), method="bounded", options={"xatol": max(1e-15, 1e-12 * (hi - lo))}
                )
                out.append(float(res.x) if obj(res.x) <= obj(k[i]) else float(k[i]))
            return sorted(out)

        k_min, k_mx = extrema(+1), extrema(-1)
        first = min([(r, "min") for r in k_min] + [(r, "max") for r in k_mx], default=None)
        zero_is_min = (first[1] == "max") if first is not None else bool(g[-1] > g[0])
        out = (np.array(k_min), np.array(k_mx), zero_is_min)
        cache[key] = out
        return out

    def line_minima_radial(self, t: float, k_max: float, prominence_rel: float = 1e-12) -> list:
        """Exact minima of eps_g(sqrt(s^2 + t^2)) along s for isotropic geometry.

        s = +-sqrt(k_m^2 - t^2) for every radial minimum k_m > |t|, plus s = 0 when eps_g increases
        radially at r = |t| (the line's closest approach to Gamma). prominence_rel as in radial_extrema.
        """
        k_min, k_mx, zero_is_min = self.radial_extrema(k_max, prominence_rel=prominence_rel)
        r = abs(t)
        pts = []
        for km in k_min:
            if km > r:
                d = float(np.sqrt(km * km - r * r))
                pts += [-d, d]
        below = [(x, "min") for x in k_min if x <= r] + [(x, "max") for x in k_mx if x <= r]
        last = max(below, default=None)
        increasing = (last[1] == "min") if last is not None else zero_is_min
        if increasing:
            pts.append(0.0)
        return sorted(pts)

    # ---- model-specific hooks -------------------------------------------------------------------
    def exact_branch_points(self, t: float):
        """[(k_c, E_DDP)] on the line at t from the closed-form four-band condition, or None."""
        m = self.source_model
        p = getattr(m, "p", None)
        if p is None or not getattr(m, "is_isotropic_geometry", False) or getattr(m, "warping", 0.0):
            return None
        from ..lz.landau_zener import fourband_ddp_t

        return fourband_ddp_t(p, t, dominant_only=False)

    def diabatic_radius(self) -> float | None:
        """Radius of the uncoupled surface-band crossing circle, if the model provides it (four-band)."""
        p = getattr(self.source_model, "p", None)
        return p.k_diabatic() if p is not None and hasattr(p, "k_diabatic") else None

    def gap_at_gamma(self) -> float:
        return float(self.gap(0.0, 0.0))

    # ---- precision bookkeeping ------------------------------------------------------------------
    def _estimate_precision_floor(self, k_ref: float | None = None) -> dict:
        """Where the unpatched float64 geometry fails near Gamma (recorded when no patch exists).

        Estimate: the float64 rounding of H acts as a mass error noise = eps_mach * S (S = sum_R max|H_R|
        for Wannier models, ||H(Gamma)|| otherwise). Omega_c ~ (m0 - m2 k^2) is then wrong by more than 1%
        where |m0| + |m2| k^2 < 100 noise:
            k_floor_est = sqrt((100 noise - |m0|) / |m2|)    (None if 100 noise <= |m0|: float64 suffices),
        with m0 = eps_g(Gamma)/2 and m2 = 2 hbar_v k |Omega_c| from float64 at k_ref = 1e-4 1/A (inside the
        Dirac regime k << hbar_v/|m2| and above the float64 floor).
        Measurement (only when the estimate is not None): k_floor_plateau = the largest |k| in
        [1e-12, k_ref] where float64 k*Omega_c leaves its value at k_ref by more than 1% (two directions).
        k_floor = the larger of the two; k_floor_gap = 100 noise / (2 hbar_v), where eps_g loses 1%.
        """
        m = self.model
        if hasattr(m, "H_R"):
            scale = float(np.sum(np.max(np.abs(np.asarray(m.H_R)), axis=(1, 2))))
        else:
            scale = float(np.linalg.norm(np.asarray(m.hamiltonian(0.0, 0.0)), 2))
        noise = _EPS * scale
        m0 = 0.5 * self.gap_gamma
        k_ref = 1e-4 if k_ref is None else k_ref
        th = np.array([0.0, 0.5 * math.pi])
        geo = band_geometry(m, k_ref * np.cos(th), k_ref * np.sin(th))
        hv = float(np.mean(geo.eps_g)) / (2.0 * k_ref)
        m2 = 2.0 * hv * k_ref * float(np.mean(np.abs(geo.omega_c)))
        excess = 100.0 * noise - m0
        k_est = math.sqrt(excess / m2) if (excess > 0 and m2 > 0) else None
        k_plat = None
        if k_est is not None:
            k = np.geomspace(1e-12, k_ref, 85)
            for a in th:
                g = band_geometry(m, k * math.cos(a), k * math.sin(a))
                kom = k * g.omega_c
                bad = ~(np.abs(kom / kom[-1] - 1.0) <= 1e-2)
                if bad.any():
                    k_plat = max(k_plat or 0.0, float(k[bad].max()))
        k_floor = None if k_est is None else max(k_est, k_plat or 0.0)
        return {
            "k_floor": k_floor,
            "k_floor_est": k_est,
            "k_floor_plateau": k_plat,
            "k_floor_gap": 100.0 * noise / (2.0 * hv) if hv > 0 else None,
            "noise_eV": float(noise),
            "m0_eV": float(m0),
            "m2_est": float(m2),
            "hbar_v_est": float(hv),
            "criterion": "1% error in Omega_c",
            "reason": self.patch_error or "patch disabled",
        }

    def patch_overlap(self) -> float | None:
        """Certificate: max relative difference of the patch from float64 kubo where float64 is reliable
        ([k_patch/10, k_patch] for RadialPatch, [k_patch, 10 k_patch] for LowdinPatch); cached."""
        if self.patch is None:
            return None
        if "_overlap" not in self.__dict__:
            self._overlap = float(self.patch.overlap_error(self.model))
        return self._overlap

    def describe(self) -> dict:
        d = {
            "spin_degeneracy": self.spin_degeneracy,
            "angle": self.angle,
            "patched": self.patch is not None,
            "k_patch": self.k_patch,
            "patch_gap": self.patch_gap,
            "gap_gamma": self.gap_gamma,
            "energy_ref": self.energy_ref,
            "patch": None if self.patch is None else self.patch.describe(),
            "patch_overlap": self.patch_overlap(),
            "precision_floor": self.precision_floor,
        }
        if self.patch_error is not None:
            d["patch_error"] = self.patch_error
        if hasattr(self.source_model, "describe"):
            d["model"] = self.source_model.describe()
        return d


def field_unit_vectors(angle: float):
    s = np.array([math.cos(angle), math.sin(angle)])
    return s, np.array([-s[1], s[0]])
