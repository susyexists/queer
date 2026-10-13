"""Hall current J_H(E) of Eqs. 10-11 (generalized to any number of crossings per line).

    J_H(E) = G * E * Int dt sum_j rho_cc^{(j)}(E, t) Int_seg_j exp(-|s - s_j|/lambda_1) Omega(s, t) ds

G = e^2/h (as printed) or e^2/(2 pi h); lambda_1 = e|E|T1/hbar. rho_cc^{(j)} is the c-band population
after crossing j: P_LZ (Eq. 7) for the first crossing and P_int (Eq. 12) after the second, from the
sequential density-matrix model (populations relax with T1, coherences with 1/(2T1) + 1/T2).
The path direction is charge_sign * sign(E) (paper: hbar dk/dt = +eE). E < 0 is evaluated on the
explicitly mirrored path; oddness is never assumed.

Twisted (nonreciprocal) LZ, PHYSICAL only. Motion d = +-1 along s through an isolated crossing gives
ln P_d = -E_c/|E| - d tau at low field, tau = (pi/2) R_s Im k_c (transport/lines.py); the Houston TDSE
confirms ln(P_+/P_-) -> -2 tau within 2-15% for VL/VLc <= 1. The factor multiplies the transition
amplitude of the whole line, so it is applied to the populations rho^{(j)} after the sequential
propagation, with the line-mean tau (crossing-to-crossing differences of tau enter the individual P_j):
  suppressed direction (d tau >= 0):  rho e^{-d tau};
  enhanced direction (d tau < 0):     odds form rho'/(1 - rho') = e^{a} rho/(1 - rho),
                                      a = |tau| / (1 + (|tau|/1.35)^8).
The enhancement keeps its low-field value for small |tau| (1.5% below it at |tau| = 0.8) and fades out
smoothly beyond |tau| ~ 1.3: on strongly twisted lines (flag 'twisted_lz_strong', |tau| > 1) the TDSE
enhancement falls far below e^{|tau|} already at E ~ 0.15 E_c (outer lines saturate at the sudden-limit
overlap), while the suppression follows e^{-|tau|}. The window is smooth so the t-integrand has no kinks.
Against the TDSE (four-band, 61 lines x 2 directions, VL/VLc = 0.5-2, E = 1e3-5e4 V/m, T = inf) the line
RMS is 0.056 with this term and 0.109 without it (|tau| <= 1: 0.050 vs 0.127; strong lines: suppressed
direction 0.014, enhanced 0.082 vs 0.100). Twisting each P_j of a crossing pair instead, or clipping
ln P at 0 as before, makes pair populations collapse.

Geometry stage (`prepare`): transverse extent t_max (beyond it E_c > p_cut * E_max), tangency
breakpoints where the number of crossings changes (the integrand jumps there), logistic t-nodes,
per-line solutions (lines.py). Response stage (`J`): vectorized over E.

prepare(n_workers > 1) runs the layout scan in parallel when the provider is light to ship: the probe lines of
both sides at once, then p-section rounds reaching the serial bisection bracket widths. Lines are solved one
per task. Native worker views share read-only model arrays through joblib and omit raw input records and
factorization caches. Execution budgets and parent-owned checkpoint callbacks apply to both stages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .._multiplicity import validated_spin_degeneracy
from ..config import PAPER, RunConfig
from ..geometry.provider import ModelGeometry
from ..lz.crossings import find_crossings
from ..units import hall_prefactor, kappa
from .lines import LineSolution, solve_line
from .quadrature import composite_nodes
from .tables import LambdaGrid, eval_tables

# clustering distance relative to the shortest relaxation length lambda(E_min): the end slabs of the
# logistic panels are then resolved far below the quadrature tolerance (quadrature.py)
_DMIN_PER_LAMBDA = 1e-5
_NO_CLUSTER = 0.05  # outer t-panel ends at t_max (integrand ~ exp(-p_cut)): no clustering needed
_TWIST_WINDOW = 1.35


@dataclass
class _Group:
    """Lines with the same number of crossings n, packed for the kernel."""

    n: int
    w: np.ndarray  # (L,) t-weights
    t: np.ndarray  # (L,)
    Ec: dict  # metric (or "ddp") -> (L, n), ascending-s order
    gaps: np.ndarray  # (L, n-1) gap integrals between ascending crossings, eV/A
    seg_len: np.ndarray  # (L, n-1)
    tables: dict  # (band, direction) -> (L, n, K, 3), traversal order
    F_inf: dict  # (band, direction) -> (L, n)
    M1: dict  # (band, direction) -> (L, n)
    geo_phase: dict | None = None  # direction -> (L, n-1)
    twist: np.ndarray | None = None  # (L, n) tau = (pi/2) R_s Im k_c, ascending order (PHYSICAL)


def _make_group(lines: list, weights) -> _Group:
    """Pack lines with equal crossing number n (> 0)."""
    n = lines[0].n
    L = len(lines)
    Ec = {
        m: np.array([[c.Ec[m] for c in ln.crossings.crossings] for ln in lines]).reshape(L, n)
        for m in ("cv", "v_total", "c_total")
    }
    Ec["ddp"] = np.array(
        [[c.Ec_ddp if c.Ec_ddp is not None else np.nan for c in ln.crossings.crossings] for ln in lines]
    ).reshape(L, n)
    tabs = lines[0].tables
    return _Group(
        n=n,
        w=np.asarray(weights, dtype=float).reshape(L),
        t=np.array([ln.t for ln in lines]),
        Ec=Ec,
        gaps=np.array([ln.gap_integrals for ln in lines]).reshape(L, n - 1),
        seg_len=np.array([np.diff(ln.s_cross) for ln in lines]).reshape(L, n - 1),
        tables={k: np.stack([ln.tables[k] for ln in lines]) for k in tabs} if tabs else {},
        F_inf={k: np.stack([ln.F_inf[k] for ln in lines]) for k in lines[0].F_inf} if tabs else {},
        M1={k: np.stack([ln.M1[k] for ln in lines]) for k in lines[0].M1} if tabs else {},
        geo_phase={d: np.array([ln.geo_phase[d] for ln in lines]).reshape(L, n - 1) for d in (1, -1)}
        if (n >= 2 and lines[0].geo_phase)
        else None,
        twist=np.array([ln.twist for ln in lines]).reshape(L, n) if lines[0].twist is not None else None,
    )


def twisted_probability(P: np.ndarray, d_tau: np.ndarray) -> np.ndarray:
    """Twisted-LZ transform of a population P for motion d with exponent tau (module docstring).

    d tau >= 0: P e^{-d tau};  d tau < 0: odds form with the factor e^a, a = |tau| / (1 + (|tau|/1.35)^8).
    Low-field limit P e^{-d tau} for small |tau|; the result stays in [0, 1].
    """
    P, d_tau = np.broadcast_arrays(np.asarray(P, dtype=float), np.asarray(d_tau, dtype=float))
    with np.errstate(over="ignore", invalid="ignore"):
        sup = P * np.exp(-np.maximum(d_tau, 0.0))
        m = -np.minimum(d_tau, 0.0)
        f = np.exp(m / (1.0 + (m / _TWIST_WINDOW) ** 8))  # smooth window, maximum e^{1.27}
        enh = P * f / (1.0 - P + P * f)
    return np.where(d_tau >= 0, sup, enh)


@dataclass
class TransverseLayout:
    t_nodes: np.ndarray
    t_weights: np.ndarray
    breaks: np.ndarray
    t_max: tuple  # (negative side, positive side) magnitudes
    mirrored: bool
    flags: set = field(default_factory=set)


def _propagate(P, d, theta, k1, k2, coherence: str):
    """rho_cc after each crossing for n crossings: P (n, nE), d/theta (n-1, ...), k1/k2 (nE,)."""
    n = P.shape[0]
    rho_cc = P[0].copy()
    rho_cv = -np.sqrt(P[0] * (1.0 - P[0])) + 0j
    out = [rho_cc.copy()]
    for j in range(1, n):
        decay_pop = np.exp(-d[j - 1] / k1)
        decay_coh = np.exp(-d[j - 1] / k2) * (np.exp(-d[j - 1] / (2.0 * k1)) if coherence == "paper" else 1.0)
        rho_cc = rho_cc * decay_pop
        rho_cv = rho_cv * decay_coh * np.exp(-1j * theta[j - 1])
        rho_vv = 1.0 - rho_cc
        p = P[j]
        sq = np.sqrt(p * (1.0 - p))
        new_cc = (1.0 - p) * rho_cc + p * rho_vv - 2.0 * sq * rho_cv.real
        new_cv = sq * (rho_cc - rho_vv) + (1.0 - p) * rho_cv - p * np.conj(rho_cv)
        rho_cc, rho_cv = new_cc, new_cv
        out.append(rho_cc.copy())
    return np.stack(out)


class HallCalculator:
    """J_H(E) for a band model (or a geometry provider) under a RunConfig."""

    def __init__(
        self,
        model=None,
        cfg: RunConfig = PAPER,
        *,
        provider=None,
        keep_nodes: bool = False,
        t_scan: int = 48,
        certificate=None,
        use_numba: bool = True,
    ):
        if provider is None and model is None:
            raise ValueError("model or provider required")
        self.cfg = cfg
        if provider is not None:
            pa = getattr(provider, "angle", None)
            if pa is None or abs(math.remainder(float(pa) - cfg.drive.angle, 2.0 * math.pi)) > 1e-12:
                raise ValueError(
                    f"provider field angle {pa} differs from cfg.drive.angle = {cfg.drive.angle}: build the "
                    "provider (or grid) for the field direction of the run"
                )
        self.provider = (
            provider
            if provider is not None
            else ModelGeometry(model, cfg.drive.angle, k_patch=cfg.quadrature.k_patch)
        )
        self.spin_degeneracy = validated_spin_degeneracy(getattr(self.provider, "spin_degeneracy", 1))
        self._check_supported()
        q = cfg.quadrature
        # Node clustering must resolve the shortest relaxation length lambda(E_min) = e E_min T1/hbar
        # (the panel end slabs at the injection crossings and at t = 0 scale with dmin/lambda).
        lam_min = float(kappa(cfg.drive.E_min, cfg.relaxation.T1))
        if not lam_min > 0:
            raise ValueError("drive.E_min must be > 0")
        dmin = min(q.dmin, _DMIN_PER_LAMBDA * lam_min) if math.isfinite(lam_min) else q.dmin
        if dmin != q.dmin:
            self.cfg = cfg = cfg.with_(**{"quadrature.dmin": dmin})
        self.grid = LambdaGrid.make(30.0 * dmin, 1e3 * cfg.crossings.window, q.lam_per_decade)
        self.keep_nodes = keep_nodes
        self.t_scan = t_scan
        self.certificate = certificate
        self.use_numba = False
        if use_numba:
            from . import kernels

            self.use_numba = bool(kernels.HAVE_NUMBA)
        self.layout: TransverseLayout | None = None
        self.lines: list[LineSolution] | None = None
        self._groups: dict[int, _Group] | None = None
        self.response_flags: set = set()

    # ------------------------------------------------------------------ configuration checks
    def _check_supported(self):
        from ...capabilities import CapabilityError

        c = self.cfg
        if hasattr(self.provider, "tensors"):
            metric = "g_" + c.conventions.metric
            if metric not in self.provider.tensors:
                raise CapabilityError(f"The geometry grid does not contain requested metric {metric}")
        model = getattr(self.provider, "source_model", getattr(self.provider, "model", None))
        if c.conventions.ec_method == "ddp" or c.crossings.definition == "branch_point":
            if model is not None and not getattr(model, "complex_k", False):
                raise CapabilityError(
                    "Branch-point/DDP response requires a raw analytic complex-k Hamiltonian; "
                    "select an explicit Eq8/gap-minima profile for this model"
                )
        if c.phase.geometric or c.phase.twisted_lz:
            if model is None or not getattr(model, "supports_wavefunction_phases", True):
                raise CapabilityError(
                    "Geometric and twisted-LZ phases require certified physical wavefunction links"
                )
        if c.crossings.mode != "per_line":
            raise CapabilityError(
                "Global crossing substitution has no validated native formulation; use per_line"
            )
        if c.crossings.merge != "none":
            raise CapabilityError(
                "Crossing merging has no validated formulation; use merge=none and inspect not_separated flags"
            )

    @property
    def _physical_extras(self) -> bool:
        ph = self.cfg.phase
        return bool(ph.geometric or ph.stokes or ph.twisted_lz)

    @property
    def _wavefunction_phases(self) -> bool:
        """Phases that depend on wavefunctions, not only on eps_g, Omega and g (symmetry.py)."""
        ph = self.cfg.phase
        return bool(ph.geometric or ph.twisted_lz)

    # ------------------------------------------------------------------ symmetry
    def selection_rules(self) -> dict:
        """Certified shortcuts for this run (all False unless use_symmetry_shortcuts)."""
        if not self.cfg.use_symmetry_shortcuts:
            return {"e_odd": False, "t_mirror": False, "isotropic": False, "c3": False}
        from ..symmetry import certify, selection_rules

        angle = self.cfg.drive.angle
        wf = self._wavefunction_phases
        if self.certificate is None or (wf and self.certificate.wavefunction_angle is None):
            shift_fn = None
            model = getattr(self.provider, "model", None)
            if wf and model is not None:
                from ..geometry.wilson import shift_field

                def shift_fn(kx, ky):
                    return shift_field(model, kx, ky, angle)

            self.certificate = certify(
                self.provider.at_cartesian, field_angles=(angle,), shift_fn=shift_fn, shift_angle=angle
            )
        return selection_rules(self.certificate, angle, wavefunction_phases=wf)

    # ------------------------------------------------------------------ transverse layout
    def _line_summary(self, t: float, structure: bool = True):
        """(structure, smallest exponent): structure changes mark the transverse breakpoints. It is the
        number of crossings and, for branch-point crossings (whose number rarely changes), also the number
        of gap minima: the line geometry changes at the ring tangency either way. structure=False skips
        the gap-minimum count (the t_max search needs only the exponent)."""
        cfg = self.cfg
        ddp = cfg.conventions.ec_method == "ddp"
        lc = find_crossings(
            self.provider, t, cfg.crossings, want_ddp=ddp, e_cut=cfg.quadrature.p_cut * cfg.drive.E_max
        )
        ec = lc.Ec(cfg.conventions.metric, "ddp" if ddp else "eq8")
        key = lc.n
        if structure and cfg.crossings.definition == "branch_point":
            key = (lc.n, len(self._minima_cached(t)))
        return key, (float(np.nanmin(ec)) if ec.size else math.inf)

    def _minima_cached(self, t: float):
        """Gap minima of the line at t, memoized for the layout scan (the generic branch-point search runs
        the same scan; with isotropic models it is the cheap radial path)."""
        cache = self.__dict__.setdefault("_minima_memo", {})
        key = float(t)
        if key not in cache:
            from ..lz.crossings import _minima

            cache[key] = _minima(self.provider, key, self.cfg.crossings)[0]
        return cache[key]

    def _scan_side(self, sign: int):
        pol, q, drive = self.cfg.crossings, self.cfg.quadrature, self.cfg.drive
        W = pol.window
        thresh = q.p_cut * drive.E_max
        mags = np.geomspace(1e-7 * W, W, self.t_scan)
        summ = [self._line_summary(sign * m) for m in mags]
        n = [a for a, _ in summ]
        ec = np.array([b for _, b in summ])
        flags = set()
        below = np.where(ec <= thresh)[0]
        if q.t_max is not None:
            t_max = float(q.t_max)
        elif below.size == 0:
            # every scanned line has E_c > p_cut * E_max (or no crossing): J <= exp(-p_cut) of its scale, typically
            # exactly 0 after underflow; keep the strip |t| < 1e-7 W around Gamma and say so
            t_max = float(mags[0])
            flags.add("no_tunnelling")
        elif below[-1] == mags.size - 1:
            t_max = W
            flags.add("t_window_limited")
        else:
            lo, hi = mags[below[-1]], mags[below[-1] + 1]
            for _ in range(40):
                mid = math.sqrt(lo * hi)
                if self._line_summary(sign * mid, structure=False)[1] <= thresh:
                    lo = mid
                else:
                    hi = mid
            t_max = float(hi)
        tangency = []
        for i in range(mags.size - 1):
            if mags[i] >= t_max:
                break
            if n[i] != n[i + 1]:
                lo, hi = mags[i], min(mags[i + 1], t_max)
                n_lo = n[i]
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if self._line_summary(sign * mid)[0] == n_lo:
                        lo = mid
                    else:
                        hi = mid
                    if hi - lo < 1e-13 * W:
                        break
                tb = 0.5 * (lo + hi)
                if tb < t_max * (1.0 - 1e-6):  # crossings cut by e_cut vanish at t_max itself: no extra panel
                    tangency.append(tb)
        return t_max, tangency, flags

    def _scan_sides_parallel(self, signs, session, n_workers: int, provider=None) -> dict:
        """_scan_side for every sign at once on a process pool: the probe lines in one round, then p-section rounds
        for t_max (geometric, to the bracket ratio of the 40 serial bisections) and for each tangency (linear, to
        1e-13 W or the width of the 60 serial bisections)."""
        pol, q, drive = self.cfg.crossings, self.cfg.quadrature, self.cfg.drive
        W = pol.window
        thresh = q.p_cut * drive.E_max
        mags = np.geomspace(1e-7 * W, W, self.t_scan)

        def run(points):  # [(sign, |t|, structure)] -> [(structure key, smallest exponent)]
            from joblib import delayed

            worker_provider = self.provider if provider is None else provider
            return list(
                session.parallel()(
                    delayed(_summary_task)((worker_provider, self.cfg, sg * m, st)) for sg, m, st in points
                )
            )

        res = run([(sg, m, True) for sg in signs for m in mags])
        out, tmax_br = {}, {}
        for i, sg in enumerate(signs):
            part = res[i * mags.size : (i + 1) * mags.size]
            n, ec = [a for a, _ in part], np.array([b for _, b in part])
            flags, t_max = set(), None
            below = np.where(ec <= thresh)[0]
            if q.t_max is not None:
                t_max = float(q.t_max)
            elif below.size == 0:
                t_max = float(mags[0])
                flags.add("no_tunnelling")
            elif below[-1] == mags.size - 1:
                t_max = W
                flags.add("t_window_limited")
            else:
                lo, hi = mags[below[-1]], mags[below[-1] + 1]
                tmax_br[sg] = [lo, hi, math.log(hi / lo) / 2.0**40]
            out[sg] = {"n": n, "t_max": t_max, "flags": flags, "tangency": []}
        while tmax_br:  # p-section of the threshold crossing in log t
            p = max(2, n_workers // len(tmax_br))
            pts, owners = [], []
            for sg, (lo, hi, _) in tmax_br.items():
                xs = lo * (hi / lo) ** (np.arange(1, p + 1) / (p + 1))
                owners.append((sg, xs, len(pts)))
                pts += [(sg, x, False) for x in xs]
            res = run(pts)
            for sg, xs, i0 in owners:
                lo, hi, target = tmax_br[sg]
                ok = np.where(np.array([r[1] for r in res[i0 : i0 + xs.size]]) <= thresh)[0]
                if ok.size == 0:
                    hi = xs[0]
                elif ok[-1] == xs.size - 1:
                    lo = xs[-1]
                else:
                    lo, hi = xs[ok[-1]], xs[ok[-1] + 1]
                if math.log(hi / lo) <= target:
                    out[sg]["t_max"] = float(hi)
                    del tmax_br[sg]
                else:
                    tmax_br[sg] = [lo, hi, target]
        tang = []  # [sign, lo, hi, key below, stop width]
        for sg in signs:
            n, t_max = out[sg]["n"], out[sg]["t_max"]
            for i in range(mags.size - 1):
                if mags[i] >= t_max:
                    break
                if n[i] != n[i + 1]:
                    lo, hi = mags[i], min(mags[i + 1], t_max)
                    tang.append([sg, lo, hi, n[i], max(1e-13 * W, (hi - lo) / 2.0**60)])
        while (
            tang
        ):  # p-section of the structure change (first point whose structure differs from the low end)
            p = max(2, n_workers // len(tang))
            pts, owners = [], []
            for b in tang:
                xs = b[1] + (b[2] - b[1]) * np.arange(1, p + 1) / (p + 1)
                owners.append((b, xs, len(pts)))
                pts += [(b[0], x, True) for x in xs]
            res = run(pts)
            left = []
            for b, xs, i0 in owners:
                keys = [r[0] for r in res[i0 : i0 + xs.size]]
                diff = [j for j, k in enumerate(keys) if k != b[3]]
                if diff:
                    j = diff[0]
                    b[1], b[2] = (xs[j - 1] if j > 0 else b[1]), xs[j]
                else:
                    b[1] = xs[-1]
                if b[2] - b[1] < b[4]:
                    tb = 0.5 * (b[1] + b[2])
                    if tb < out[b[0]]["t_max"] * (1.0 - 1e-6):  # as _scan_side
                        out[b[0]]["tangency"].append(tb)
                else:
                    left.append(b)
            tang = left
        return {sg: (out[sg]["t_max"], sorted(out[sg]["tangency"]), out[sg]["flags"]) for sg in signs}

    def _t_mirror_certified(self) -> bool:
        return bool(self.selection_rules()["t_mirror"])

    def transverse_layout(self, session=None, n_workers: int = 1, worker_provider=None) -> TransverseLayout:
        q = self.cfg.quadrature
        mirrored = self._t_mirror_certified()
        if session is None:
            tp, tang_p, flags = self._scan_side(+1)
            negative = None
        else:
            sides = self._scan_sides_parallel(
                (+1,) if mirrored else (+1, -1), session, n_workers, worker_provider
            )
            tp, tang_p, flags = sides[+1]
            negative = None if mirrored else sides[-1]
        if mirrored:
            breaks = np.array([0.0, *tang_p, tp])
            # no clustering at t_max: the integrand is ~exp(-p_cut) there
            x, w, _ = composite_nodes(
                breaks, q.du, q.dmin, d_outer=(q.dmin, _NO_CLUSTER * (breaks[-1] - breaks[-2]))
            )
            return TransverseLayout(x, 2.0 * w, breaks, (tp, tp), True, flags)
        tn, tang_n, flags_n = self._scan_side(-1) if negative is None else negative
        breaks = np.array(sorted({-tn, *[-x for x in tang_n], 0.0, *tang_p, tp}))
        x, w, _ = composite_nodes(
            breaks,
            q.du,
            q.dmin,
            d_outer=(_NO_CLUSTER * (breaks[1] - breaks[0]), _NO_CLUSTER * (breaks[-1] - breaks[-2])),
        )
        return TransverseLayout(x, w, breaks, (tn, tp), False, flags | flags_n)

    # ------------------------------------------------------------------ geometry stage
    def prepare(
        self,
        n_workers: int = 1,
        *,
        execution=None,
        load_line=None,
        save_line=None,
        save_layout=None,
        load_layout=None,
    ) -> "HallCalculator":
        from joblib import delayed
        from threadpoolctl import threadpool_limits

        from ...execution import ExecutionOptions, execution_session

        options = (execution or ExecutionOptions(workers=n_workers)).resolved()
        if options.memory_mb is not None and isinstance(self.provider, ModelGeometry):
            self.provider.mem_bytes = min(256e6, options.memory_mb * 1024**2 * 0.8 / options.workers)
            if hasattr(self.provider.patch, "mem_bytes"):
                self.provider.patch.mem_bytes = self.provider.mem_bytes
        worker_provider = _worker_provider(self.provider) if options.workers > 1 else self.provider
        light = options.workers > 1 and (
            worker_provider is not self.provider or _light_to_ship(worker_provider)
        )
        with execution_session(options) as session, threadpool_limits(limits=options.native_threads):
            self.layout = load_layout() if load_layout is not None else None
            if self.layout is None:
                self.layout = self.transverse_layout(
                    session=session if light else None,
                    n_workers=options.workers,
                    worker_provider=worker_provider,
                )
            if save_layout is not None:
                save_layout(self.layout)
            ts = [float(t) for t in self.layout.t_nodes]
            self.lines = [load_line(i, t) if load_line is not None else None for i, t in enumerate(ts)]
            missing = [(i, t) for i, t in enumerate(ts) if self.lines[i] is None]
            batch = options.bounded_batch_size(
                max(16 * self.provider.model.n_orb**2 * 12, 1024)
                if getattr(self.provider, "model", None) is not None
                else 1024
            )
            if options.workers > 1 and missing:
                batch = 1 if light else min(batch, max(1, math.ceil(len(missing) / (4 * options.workers))))
                tasks = [missing[i : i + batch] for i in range(0, len(missing), batch)]
                if light:
                    results = session.parallel()(
                        delayed(_solve_line_task)(
                            (worker_provider, items[0][1], self.cfg, self.grid, self.keep_nodes)
                        )
                        for items in tasks
                    )
                    results = ([line] for line in results)
                else:
                    results = session.parallel()(
                        delayed(_solve_lines_task)(
                            (worker_provider, [t for _, t in items], self.cfg, self.grid, self.keep_nodes)
                        )
                        for items in tasks
                    )
                for items, solved in zip(tasks, results):
                    for (index, t), line in zip(items, solved):
                        self.lines[index] = line
                        if save_line is not None:
                            save_line(index, t, line)
            else:
                for index, t in missing:
                    line = solve_line(self.provider, t, self.cfg, self.grid, self.keep_nodes)
                    self.lines[index] = line
                    if save_line is not None:
                        save_line(index, t, line)
        self._pack()
        return self

    def _pack(self):
        key = self._ec_key
        if key == "ddp":
            unsupported = [
                (line.t, crossing.ddp_source)
                for line in self.lines
                for crossing in line.crossings.crossings
                if not crossing.touching and crossing.ddp_source not in {"exact", "model", "fit", "eq8"}
            ]
            if unsupported:
                raise ValueError(f"unknown DDP source {unsupported[0][1]!r} at t = {unsupported[0][0]:.6g}")
        bad = [
            ln.t
            for ln in self.lines
            for c in ln.crossings.crossings
            if c.Ec_ddp is None and key == "ddp" or (key != "ddp" and not math.isfinite(c.Ec[key]))
        ]
        if bad:
            raise ValueError(
                f"critical field undefined ({key}) on {len(bad)} line(s), e.g. t = {bad[0]:.3e}: "
                "non-positive metric at a crossing (grid too coarse?)"
            )
        neg = [
            ln.t
            for ln in self.lines
            for c in ln.crossings.crossings
            if (c.Ec_ddp if key == "ddp" else c.Ec[key]) < 0
        ]
        if neg:
            raise ValueError(f"negative critical field ({key}) on {len(neg)} line(s), e.g. t = {neg[0]:.3e}")
        by_n = {}
        for line, w in zip(self.lines, self.layout.t_weights):
            if line.n == 0:
                continue
            by_n.setdefault(line.n, []).append((line, w))
        self._groups = {
            n: _make_group([it[0] for it in items], [it[1] for it in items]) for n, items in by_n.items()
        }

    # ------------------------------------------------------------------ response stage
    def _selected(self, g: _Group, direction: int):
        mode = self.cfg.conventions.berry
        if mode == "c":
            return g.tables[("c", direction)], g.F_inf[("c", direction)], g.M1[("c", direction)]
        if mode == "v":
            return g.tables[("v", direction)], g.F_inf[("v", direction)], g.M1[("v", direction)]
        return (
            g.tables[("c", direction)] - g.tables[("v", direction)],
            g.F_inf[("c", direction)] - g.F_inf[("v", direction)],
            g.M1[("c", direction)] - g.M1[("v", direction)],
        )

    def _rho(
        self, g: _Group, direction: int, Ea: np.ndarray, lam1: np.ndarray, lam2: np.ndarray
    ) -> np.ndarray:
        """Populations rho_cc^{(j)} in traversal order: (L, n, nE)."""
        Ec = g.Ec[self._ec_key]
        if direction < 0:
            Ec = Ec[:, ::-1]
        with np.errstate(divide="ignore", over="ignore"):
            P = np.exp(-Ec[:, :, None] / Ea[None, None, :])
        if not (self.cfg.phase.twisted_lz and g.twist is not None):
            return self._rho_untwisted(g, direction, P, Ea, lam1, lam2)
        tw = (g.twist if direction > 0 else g.twist[:, ::-1])[:, :, None]
        tbar = tw.mean(axis=1, keepdims=True)
        if g.n > 1:
            P = twisted_probability(P, direction * (tw - tbar))  # differences between the crossings
        return twisted_probability(self._rho_untwisted(g, direction, P, Ea, lam1, lam2), direction * tbar)

    def _rho_untwisted(self, g: _Group, direction: int, P, Ea, lam1, lam2) -> np.ndarray:
        if g.n == 1:
            return P
        if self._physical_extras:
            return self._rho_physical(g, direction, P, Ea, lam1, lam2)
        gaps = g.gaps if direction > 0 else g.gaps[:, ::-1]
        dseg = g.seg_len if direction > 0 else g.seg_len[:, ::-1]
        theta = gaps[:, :, None] * 1e10 / Ea[None, None, :]  # dynamical phase Phi = Int eps_g ds / (e|E|)
        coherence = self.cfg.relaxation.coherence
        if g.n == 2:
            P1, P2 = P[:, 0], P[:, 1]
            d = dseg[:, 0, None]
            pop = np.exp(-d / lam1[None, :])
            coh = np.exp(-d / lam2[None, :]) * (
                np.exp(-d / (2.0 * lam1[None, :])) if coherence == "paper" else 1.0
            )
            rho2 = (
                P2
                + P1 * (1.0 - 2.0 * P2) * pop
                + 2.0 * np.sqrt(P1 * P2 * (1.0 - P1) * (1.0 - P2)) * coh * np.cos(theta[:, 0])
            )
            return np.stack([P1, rho2], axis=1)
        out = np.empty_like(P)
        for i in range(P.shape[0]):
            out[i] = _propagate(P[i], dseg[i], theta[i], lam1, lam2, coherence)
        return out

    def _group_numpy(self, g: _Group, direction: int, Ea, lam1, lam2, chunk: int = 4096) -> np.ndarray:
        tab, finf, m1 = self._selected(g, direction)
        acc = np.zeros(Ea.size)
        for i0 in range(0, Ea.size, chunk):
            sl = slice(i0, i0 + chunk)
            L = eval_tables(tab, finf, m1, self.grid, lam1[sl])  # (L, n, nE)
            rho = self._rho(g, direction, Ea[sl], lam1[sl], lam2[sl])
            acc[sl] = g.w @ np.sum(rho * L, axis=1)
        return acc

    @property
    def _ec_key(self) -> str:
        return "ddp" if self.cfg.conventions.ec_method == "ddp" else self.cfg.conventions.metric

    def _rho_physical(self, g: _Group, direction: int, P, Ea, lam1, lam2):
        """Sequential density-matrix populations with geometric and Stokes phases (PHYSICAL)."""
        from ..lz.propagation import propagate_crossings, stokes_phase_from_P

        gaps = g.gaps if direction > 0 else g.gaps[:, ::-1]
        dseg = g.seg_len if direction > 0 else g.seg_len[:, ::-1]
        theta = gaps[:, :, None] * 1e10 / Ea[None, None, :]
        if self.cfg.phase.geometric and g.geo_phase is not None:
            theta = theta + g.geo_phase[direction][:, :, None]
        phiS = stokes_phase_from_P(P) if self.cfg.phase.stokes else None
        tr = lambda a: np.moveaxis(a, 1, 0)  # noqa: E731  (L, n, ...) -> (n, L, ...)
        rho = propagate_crossings(
            tr(P),
            tr(dseg)[:, :, None],
            tr(theta),
            lam1,
            lam2,
            None if phiS is None else tr(phiS),
            self.cfg.relaxation.coherence,
        )
        return np.moveaxis(rho, 0, 1)

    def _check_fields(self, E: np.ndarray):
        E_max = self.cfg.drive.E_max
        num = E[~np.isnan(E)]
        if num.size and np.max(np.abs(num)) > E_max * (1.0 + 1e-9):
            raise ValueError(
                f"|E| = {np.max(np.abs(num)):.12g} V/m exceeds drive.E_max = {E_max:.12g}: the transverse "
                "integration was truncated where E_c > p_cut * E_max (raise drive.E_max)"
            )

    def _check_interference(self, Ea: np.ndarray):
        """Flag fields at which the t-nodes under-resolve the interference term cos Phi(t).

        The node density does not depend on E, while Phi = Int eps_g ds / (e|E|) grows as 1/|E|. For adjacent
        lines (same group, same side of t = 0) whose phases differ by more than pi/2, the interference
        contribution w * 2 sqrt(P1 P2 (1-P1)(1-P2)) e^{-Gamma d} |F| (F: the line's Berry-curvature integral
        after the second crossing) is summed and compared with the total sum_lines w rho |F|; above 1e-3
        the field sets 'interference_under_resolved'.
        """
        rel = self.cfg.relaxation
        key = self._ec_key
        if not any(g.n >= 2 for g in self._groups.values()):
            return
        k1, k2 = kappa(Ea, rel.T1), kappa(Ea, rel.T2)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            total = np.zeros(Ea.size)
            for g in self._groups.values():
                F = np.abs(g.F_inf[("c", 1)]) + np.abs(
                    g.F_inf[("v", 1)]
                )  # (L, n): scale of the line integrals
                P = np.exp(-g.Ec[key][:, :, None] / Ea[None, None, :])
                total += np.sum(g.w[:, None, None] * P * F[:, :, None], axis=(0, 1))
            aliased = np.zeros(Ea.size)
            for g in self._groups.values():
                if g.n < 2 or g.t.size < 2:
                    continue
                order = np.argsort(g.t)
                t, gap, d = g.t[order], g.gaps[order, 0], g.seg_len[order, 0]
                same = np.sign(t[1:]) == np.sign(t[:-1])
                dphi = np.abs(np.diff(gap))[same, None] * 1e10 / Ea[None, :]
                if not np.any(dphi > 0.5 * math.pi):
                    continue
                Ec = g.Ec[key][order]
                P1, P2 = (np.exp(-Ec[:, k, None] / Ea[None, :]) for k in (0, -1))
                rate = d[:, None] / k2[None, :] + (
                    0.5 * d[:, None] / k1[None, :] if rel.coherence == "paper" else 0.0
                )
                amp = 2.0 * np.sqrt(P1 * P2 * (1 - P1) * (1 - P2)) * np.exp(-rate)
                F = (np.abs(g.F_inf[("c", 1)]) + np.abs(g.F_inf[("v", 1)]))[order, -1]
                contrib = (g.w[order] * F)[:, None] * amp
                pair = np.maximum(contrib[1:], contrib[:-1])[same]
                aliased += np.sum(np.where(dphi > 0.5 * math.pi, pair, 0.0), axis=0)
            if np.any(aliased > 1e-3 * total):
                self.response_flags.add("interference_under_resolved")

    def J(self, E, chunk: int = 4096) -> np.ndarray:
        """J_H in A/m (times conventions.figure_scale) at fields E (V/m, any sign; NaN propagates).

        |E| must not exceed drive.E_max. Fields below drive.E_min are evaluated, but the node clustering
        was chosen for E_min (relative error ~ (E_min/|E|) * 3e-8). Under-resolved interference is
        recorded in response_flags.
        """
        if self._groups is None:
            self.prepare()
        E = np.asarray(E, dtype=float)
        self._check_fields(E)
        Ef = np.abs(E[np.isfinite(E) & (E != 0)])
        if Ef.size and "interference_under_resolved" not in self.response_flags:
            lo, hi = float(Ef.min()), float(Ef.max())
            self._check_interference(np.geomspace(lo, hi, 64) if hi > lo else np.array([lo]))
        flat = E.ravel()
        out = np.zeros_like(flat)
        out[np.isnan(flat)] = np.nan
        conv, rel = self.cfg.conventions, self.cfg.relaxation
        G = hall_prefactor(conv.prefactor) * conv.figure_scale * self.spin_degeneracy
        for sgn in (+1, -1):
            idx = np.where(np.sign(flat) == sgn)[0]
            if idx.size == 0:
                continue
            direction = conv.charge_sign * sgn
            if self.use_numba and not self._physical_extras:
                from .kernels import run_group

                Ea = np.abs(flat[idx])
                lam1, lam2 = kappa(Ea, rel.T1), kappa(Ea, rel.T2)
                acc = np.zeros(idx.size)
                for g in self._groups.values():
                    if g.n <= 2:
                        tab, finf, m1 = self._selected(g, direction)
                        Ec = g.Ec[self._ec_key] if direction > 0 else g.Ec[self._ec_key][:, ::-1]
                        gaps = g.gaps if direction > 0 else g.gaps[:, ::-1]
                        dseg = g.seg_len if direction > 0 else g.seg_len[:, ::-1]
                        acc += run_group(
                            g.n, Ea, lam1, lam2, g.w, Ec, gaps, dseg, tab, finf, m1, self.grid, rel.coherence
                        )
                    else:
                        acc += self._group_numpy(g, direction, Ea, lam1, lam2)
                out[idx] = G * flat[idx] * acc
                continue
            for i0 in range(0, idx.size, chunk):
                ii = idx[i0 : i0 + chunk]
                Ea = np.abs(flat[ii])
                lam1, lam2 = kappa(Ea, rel.T1), kappa(Ea, rel.T2)
                acc = np.zeros(ii.size)
                for g in self._groups.values():
                    acc += self._group_numpy(g, direction, Ea, lam1, lam2)
                out[ii] = G * flat[ii] * acc
        return out.reshape(E.shape)

    def line_populations(self, line: LineSolution, E, direction: int) -> np.ndarray:
        """rho_cc after each crossing of one line (traversal order), shape (n, nE); E > 0 in V/m."""
        Ea = np.atleast_1d(np.asarray(E, dtype=float))
        rel = self.cfg.relaxation
        g = _make_group([line], [1.0])
        return self._rho(g, direction, Ea, kappa(Ea, rel.T1), kappa(Ea, rel.T2))[0]

    def J_direct(self, E) -> np.ndarray:
        """Reference J_H by direct node sums (requires keep_nodes=True); no lambda-tables."""
        if self.lines is None:
            self.prepare()
        if not self.keep_nodes:
            raise RuntimeError("J_direct needs HallCalculator(..., keep_nodes=True)")
        E = np.atleast_1d(np.asarray(E, dtype=float))
        conv, rel = self.cfg.conventions, self.cfg.relaxation
        G = hall_prefactor(conv.prefactor) * conv.figure_scale * self.spin_degeneracy
        out = np.zeros_like(E)
        for k, Ek in enumerate(E):
            if Ek == 0:
                continue
            direction = conv.charge_sign * int(np.sign(Ek))
            lam1 = kappa(np.array([abs(Ek)]), rel.T1)
            acc = 0.0
            for line, w in zip(self.lines, self.layout.t_weights):
                if line.n == 0:
                    continue
                rho = self.line_populations(line, abs(Ek), direction)[:, 0]
                segs = line.segments[direction]
                mode = conv.berry
                for j, sg in enumerate(segs):
                    om = {"c": sg.omega_c, "v": sg.omega_v, "c_minus_v": sg.omega_c - sg.omega_v}[mode]
                    acc += w * rho[j] * np.sum(sg.w * om * np.exp(-sg.r / lam1[0]))
            out[k] = G * Ek * acc
        return out

    def diagnostics(self) -> dict:
        n_by = {}
        flags = {}
        for line in self.lines or []:
            n_by[line.n] = n_by.get(line.n, 0) + 1
            for f in line.flags:
                flags[f] = flags.get(f, 0) + 1
        lay = self.layout
        model_names = {
            "not_separated",
            "twisted_lz_strong",
            "ddp_invalid",
            "branch_point_missing",
            "branch_point_unvalidated",
            "ddp_fallback_eq8",
            "diabatic_unavailable",
            "diabatic_ec_off_stationary",
            "too_many_crossings",
            "near_degenerate",
        }
        numerical_names = {
            "phase_under_resolved",
            "shift_under_resolved",
            "outside_grid",
            "window_edge_min",
            "branch_point_fit_poor",
        }
        model_flags = {name: count for name, count in flags.items() if name in model_names}
        numerical_flags = {name: count for name, count in flags.items() if name in numerical_names}
        refinements = []
        for line in self.lines or []:
            info = line.phase_refinement
            refinements.extend(item for items in info.get("geometric", {}).values() for item in items)
            refinements.extend(info.get("shift", []))
        phase_diagnostics = {
            "checks": len(refinements),
            "unresolved": sum(not info["converged"] for info in refinements),
            "unchecked": sum(info["error"] is None for info in refinements),
            "maximum_error": max(
                (info["error"] for info in refinements if info["error"] is not None), default=None
            ),
        }
        prov = (
            self.provider.describe()
            if hasattr(self.provider, "describe")
            else {"type": type(self.provider).__name__}
        )
        return {
            "spin_degeneracy": self.spin_degeneracy,
            "n_lines": len(self.lines or []),
            "crossings_per_line": n_by,
            "line_flags": flags,
            "t_max": None if lay is None else lay.t_max,
            "mirrored": None if lay is None else lay.mirrored,
            "t_breaks": None if lay is None else lay.breaks.tolist(),
            "layout_flags": None if lay is None else sorted(lay.flags),
            "response_flags": sorted(self.response_flags),
            "model_validity_flags": model_flags,
            "numerical_convergence_flags": {
                "lines": numerical_flags,
                "response": sorted(self.response_flags),
                "window": [] if lay is None else sorted(lay.flags),
            },
            "phase_refinement": phase_diagnostics,
            "selection_rules": self.selection_rules() if self.cfg.use_symmetry_shortcuts else None,
            "dmin": self.cfg.quadrature.dmin,
            "patched": getattr(self.provider, "patch", None) is not None,
            "provider": prov,
        }


def _light_to_ship(provider, limit: float = 64e6) -> bool:
    """Whether the provider pickles small enough to send with every task."""
    import pickle

    return len(pickle.dumps(provider, protocol=pickle.HIGHEST_PROTOCOL)) <= limit


def _worker_provider(provider):
    """Strip native input records and caches while retaining the provider's physical operators."""
    import copy

    models = {}
    worker = provider
    for name in ("source_model", "model"):
        model = getattr(provider, name, None)
        if model is None or not hasattr(model, "light_worker_view"):
            continue
        if worker is provider:
            worker = copy.copy(provider)
        if id(model) not in models:
            models[id(model)] = model.light_worker_view()
        setattr(worker, name, models[id(model)])
    patch = getattr(provider, "patch", None)
    if worker is not provider and patch is not None:
        worker.patch = copy.copy(patch)
        model = getattr(patch, "_model", None)
        if id(model) in models:
            worker.patch._model = models[id(model)]
        if hasattr(worker.patch, "_exp"):
            worker.patch._exp = None
    return worker


def _solve_line_task(args):
    provider, t, cfg, grid, keep = args
    return solve_line(provider, t, cfg, grid, keep)


def _summary_task(args):
    """(structure key, smallest exponent) of one line: HallCalculator._line_summary for a worker process."""
    provider, cfg, t, structure = args
    ddp = cfg.conventions.ec_method == "ddp"
    lc = find_crossings(
        provider, t, cfg.crossings, want_ddp=ddp, e_cut=cfg.quadrature.p_cut * cfg.drive.E_max
    )
    ec = lc.Ec(cfg.conventions.metric, "ddp" if ddp else "eq8")
    key = lc.n
    if structure and cfg.crossings.definition == "branch_point":
        from ..lz.crossings import _minima

        key = (lc.n, len(_minima(provider, t, cfg.crossings)[0]))
    return key, (float(np.nanmin(ec)) if ec.size else math.inf)


def _solve_lines_task(args):
    provider, ts, cfg, grid, keep = args
    return [solve_line(provider, t, cfg, grid, keep) for t in ts]
