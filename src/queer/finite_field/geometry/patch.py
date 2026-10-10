"""Near-Gamma extended-precision patch for isotropic geometry (four-band model).

At the critical point the sum-over-states Berry curvature loses all digits below |k| ~ 1e-9 1/A in
float64 (Omega is a small, k/v-suppressed part of interband products whose eigenvector errors are
~eps_mach ||H|| / eps_g), which flips the sign of J_H at low fields. For geometry that depends on
|k| only (g_r-theta = 0), radial tables computed with mpmath remove the floor. Non-isotropic models use
geometry.lowdin.LowdinPatch instead.

Tabulated surrogates are smooth in log k and tend to constants on both sides of the crossover
k_s = |m0|/hbar_v: eps_g / w, Omega * w and g * w^2 with w = sqrt(k^2 + k_s^2); at exact criticality
(k_s = 0) the radial metrics g_rr -> C^2 are tabulated without weight. The table starts at
k_min = min(1e-14, 1e-4 k_s), so below it every surrogate is constant to O((k / k_min)^2) and the
constant extrapolation is the exact asymptote of the four-band model.

Precision: the relative error of Omega is ~ 10^-dps ||H|| / (8 hbar_v C k^2); dps = 50 keeps it
below 1e-12 for k >= 1e-21 1/A. Off criticality Omega has a second crossover at sqrt(|m0/m2|)
(numerator m0 - m2 k^2); the cubic-spline error is O(h^4): ~1.4e-6 with the default 30 points per decade.
Splines are rebuilt lazily from the stored tables, so patches pickle (worker pools).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import CubicSpline

from .kubo import BandGeometry


def _mpmath():
    try:
        import mpmath
    except ImportError as error:
        from ...capabilities import CapabilityError

        raise CapabilityError("The radial precision patch requires queer[precision] (mpmath)") from error
    return mpmath


_FIELDS = ("omega_v", "omega_c", "omega_cv", "g_cv_rr", "g_cv_tt", "g_v_rr", "g_v_tt", "g_c_rr", "g_c_tt")


def _mp_params(p, dps: int, snap_critical: bool):
    """(A, B, M, Bl, VL) as mpf; VL is set to sqrt(M^2 - Bl^2) exactly if snap_critical."""
    mpmath = _mpmath()
    with mpmath.workdps(dps):
        A, B, M, Bl = (mpmath.mpf(x) for x in (p.A, p.B, p.M, p.Bl))
        VL = mpmath.sqrt(M * M - Bl * Bl) * (1 if p.VL >= 0 else -1) if snap_critical else mpmath.mpf(p.VL)
        return A, B, M, Bl, VL


def _mp_geometry(p, k: float, dps: int, snap_critical: bool = False):
    """Geometry of the traceless four-band model at (k, 0) with `dps` decimal digits."""
    mpmath = _mpmath()
    with mpmath.workdps(dps):
        k = mpmath.mpf(k)
        A, B, M, Bl, VL = _mp_params(p, dps, snap_critical)
        iu = mpmath.mpc(0, 1)

        # basis index 2*s + t with s, t in {0: up, 1: down}; matrices sigma (x) tau
        def kr(a, b):
            out = mpmath.matrix(4, 4)
            for i in range(2):
                for j in range(2):
                    for k1 in range(2):
                        for l1 in range(2):
                            out[2 * i + k1, 2 * j + l1] = a[i][j] * b[k1][l1]
            return out

        s0 = [[1, 0], [0, 1]]
        sx = [[0, 1], [1, 0]]
        sy = [[0, -iu], [iu, 0]]
        sz = [[1, 0], [0, -1]]
        GX, GY, GZ, GBL, GVL = kr(sx, s0), kr(sy, s0), kr(sz, s0), kr(sz, sx), kr(s0, sz)
        H = A * k * GX + (M - B * k * k) * GZ + Bl * GBL + VL * GVL
        dHx = A * GX - 2 * B * k * GZ
        dHy = A * GY
        E, Q = mpmath.eighe(H)
        order = sorted(range(4), key=lambda i: E[i])
        E = [E[i] for i in order]
        Q = mpmath.matrix([[Q[r, i] for i in order] for r in range(4)])
        QH = Q.H
        Xx = QH * dHx * Q
        Xy = QH * dHy * Q
        res = {}
        v, c = 1, 2
        for name, n in (("v", v), ("c", c)):
            om = mpmath.mpf(0)
            grr = mpmath.mpf(0)
            gtt = mpmath.mpf(0)
            for m in range(4):
                if m == n:
                    continue
                d2 = (E[n] - E[m]) ** 2
                om += -2 * mpmath.im(Xx[n, m] * Xy[m, n]) / d2
                grr += abs(Xx[n, m]) ** 2 / d2
                gtt += abs(Xy[n, m]) ** 2 / d2
            res[f"omega_{name}"] = om
            res[f"g_{name}_rr"] = grr
            res[f"g_{name}_tt"] = gtt
        d2 = (E[c] - E[v]) ** 2
        res["omega_cv"] = -2 * mpmath.im(Xx[c, v] * Xy[v, c]) / d2  # the (c, v) pair term of omega_c
        res["g_cv_rr"] = abs(Xx[c, v]) ** 2 / d2
        res["g_cv_tt"] = abs(Xy[c, v]) ** 2 / d2
        res["eps_v"] = E[v]
        res["eps_c"] = E[c]
        return {key: float(val) for key, val in res.items()}


class RadialPatch:
    """Extended-precision radial geometry tables of an isotropic four-band model for |k| <= k_max.

    If VL is within a few ulp of VLc (the float value of x = 1), the table is computed at the exact
    critical point (`exact_critical`): a float VL leaves a spurious mass m0 ~ 1e-19 eV that would
    dominate Omega below k ~ 1e-10 1/A. k_min is lowered to 1e-4 k_s when that is smaller (nearly but not
    exactly critical models), keeping n nodes per 8 decades.
    """

    def __init__(
        self,
        params,
        k_min: float = 1e-14,
        k_max: float = 1e-6,
        n: int = 241,
        dps: int = 50,
        snap_critical: bool = True,
    ):
        mpmath = _mpmath()
        self.params = params
        self.exact_critical = bool(
            snap_critical and abs(abs(params.VL) - params.vlc) <= 8 * np.finfo(float).eps * params.vlc
        )
        with mpmath.workdps(dps):
            A, B, M, Bl, VL = _mp_params(params, dps, self.exact_critical)
            R = mpmath.sqrt(Bl * Bl + VL * VL)
            self.k_s = float(abs(R - abs(M)) / (A * Bl / R))
        if self.k_s > 0:
            k_min = min(k_min, 1e-4 * self.k_s)
        self.k_min, self.k_max = k_min, k_max
        n = max(n, int(math.ceil(n * math.log10(k_max / k_min) / 8.0)))
        self._logk = np.linspace(math.log(k_min), math.log(k_max), n)
        k = np.exp(self._logk)
        rows = [_mp_geometry(params, float(ki), dps, self.exact_critical) for ki in k]
        tab = {"eps_g": np.array([r["eps_c"] - r["eps_v"] for r in rows]) / self._w("eps_g", k)}
        for f in _FIELDS:
            tab[f] = np.array([r[f] for r in rows]) * self._w(f, k)
        self._tab = tab
        self._splines = {}

    # ---- surrogates ------------------------------------------------------------------------------
    def _w(self, field: str, k):
        """Weight that makes the tabulated surrogate constant at k -> 0 (and at k >> k_s)."""
        w2 = k**2 + self.k_s**2
        if field == "eps_g" or field.startswith("omega"):
            return np.sqrt(w2)
        if self.exact_critical and field.endswith("_rr"):
            return np.ones_like(w2)  # g_rr -> C^2 at the critical point
        return w2

    def _spline(self, field: str):
        if field not in self._splines:
            self._splines[field] = CubicSpline(self._logk, self._tab[field])
        return self._splines[field]

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_splines"] = {}  # scipy >= 1.18 splines keep array-API module references (unpicklable)
        return state

    def _radial(self, k, fields=("eps_g",) + _FIELDS):
        k = np.asarray(k, dtype=float)
        lk = np.log(np.clip(k, self.k_min, self.k_max))  # constant below k_min: exact asymptote (docstring)
        out = {}
        with np.errstate(
            divide="ignore", invalid="ignore"
        ):  # Omega, g diverge at k = 0 exactly at criticality
            for f in fields:
                v = self._spline(f)(lk)
                out[f] = v * self._w(f, k) if f == "eps_g" else v / self._w(f, k)
        return out

    # ---- provider API ------------------------------------------------------------------------------
    def gap(self, kx, ky) -> np.ndarray:
        """eps_c - eps_v at Cartesian (kx, ky)."""
        kx, ky = np.broadcast_arrays(np.asarray(kx, dtype=float), np.asarray(ky, dtype=float))
        return self._radial(np.hypot(kx, ky), ("eps_g",))["eps_g"]

    def evaluate(self, kx, ky, angle: float = 0.0, s=None, t=None) -> BandGeometry:
        """Patched geometry at Cartesian (kx, ky) in the field frame of `angle` (|k| <= k_max).

        s, t: field-frame coordinates of the same points if known exactly (then k_hat.s_hat = s/|k| is
        exact, e.g. 1 on the t = 0 line); otherwise derived from (kx, ky).
        """
        kx, ky = np.broadcast_arrays(np.asarray(kx, dtype=float), np.asarray(ky, dtype=float))
        if s is not None and t is not None:
            s, t = (np.broadcast_to(np.asarray(z, dtype=float), kx.shape) for z in (s, t))
            k = np.hypot(s, t)
            safe = np.where(k > 0, k, 1.0)
            ca, sa = np.where(k > 0, s / safe, 1.0), np.where(k > 0, t / safe, 0.0)
        else:
            k = np.hypot(kx, ky)
            safe = np.where(k > 0, k, 1.0)
            rx, ry = np.where(k > 0, kx / safe, math.cos(angle)), np.where(k > 0, ky / safe, math.sin(angle))
            cs, sn = math.cos(angle), math.sin(angle)
            ca = rx * cs + ry * sn  # k_hat . s_hat
            sa = -rx * sn + ry * cs  # k_hat . t_hat
        r = self._radial(k)

        def tensor(grr, gtt):
            # g = grr k_hat k_hat + gtt theta_hat theta_hat, with theta_hat = z x k_hat
            with np.errstate(invalid="ignore"):
                gss = grr * ca**2 + gtt * sa**2
                gtt_ = grr * sa**2 + gtt * ca**2
                gst = (grr - gtt) * ca * sa
            return np.stack([gss, gst, gtt_], axis=-1)

        eps_g = r["eps_g"]
        with np.errstate(divide="ignore", invalid="ignore"):
            # pair transition in the gauge X_theta >= 0: X_r X_theta* = -i Omega_cv eps_g^2 / 2 (g_r-theta = 0)
            x_t = np.sqrt(r["g_cv_tt"]) * eps_g
            x_r = -0.5j * r["omega_cv"] * eps_g**2 / x_t
            v_cv = np.stack(
                [ca * x_r - sa * x_t, sa * x_r + ca * x_t], axis=-1
            )  # s_hat = ca k_hat - sa theta_hat
        return BandGeometry(
            eps_v=-0.5 * eps_g,
            eps_c=0.5 * eps_g,
            omega_v=r["omega_v"],
            omega_c=r["omega_c"],
            g_cv=tensor(r["g_cv_rr"], r["g_cv_tt"]),
            g_v_total=tensor(r["g_v_rr"], r["g_v_tt"]),
            g_c_total=tensor(r["g_c_rr"], r["g_c_tt"]),
            v_cv=v_cv,
            deg_flag=~(eps_g > 0),
            extras={"patched": np.ones(k.shape, dtype=bool)},
        )

    def describe(self) -> dict:
        return {
            "kind": "radial",
            "k_min": self.k_min,
            "k_max": self.k_max,
            "k_s": self.k_s,
            "exact_critical": self.exact_critical,
            "n": int(self._logk.size),
        }

    def overlap_error(self, model, k_lo: float | None = None, k_hi: float | None = None, n: int = 9) -> float:
        """Max relative difference of k*Omega_c and eps_g between the patch and float64 on [k_lo, k_hi]."""
        from .kubo import band_geometry

        k_hi = self.k_max if k_hi is None else k_hi
        k_lo = k_hi / 10.0 if k_lo is None else k_lo
        k = np.geomspace(k_lo, k_hi, n)
        ref = band_geometry(model, k, 0.0 * k)
        pat = self.evaluate(k, 0.0 * k)
        e1 = np.max(np.abs(pat.omega_c - ref.omega_c) / np.maximum(np.abs(ref.omega_c), 1e-300))
        e2 = np.max(np.abs(pat.eps_g - ref.eps_g) / ref.eps_g)
        return float(max(e1, e2))
