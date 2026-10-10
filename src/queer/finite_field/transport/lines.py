"""Per-line geometry stage of Eqs. 10-11: crossings, segments, phase integrals and lambda-tables.

On the line k = s e + t (z x e) the electron moves toward +s for direction +1 (paper: E > 0) and
toward -s for direction -1. With crossings s_1 < ... < s_n met in traversal order, segment j runs from
crossing j to the next crossing (or to the window edge) and carries the c-band population
rho_cc^{(j)} injected up to crossing j, decaying as exp(-|s - s_j|/lambda):

    F_line(E) = sum_j rho_cc^{(j)}(E) * Int_seg_j exp(-|s - s_j|/lambda) Omega(s) ds.

This reduces to Eq. 10 (n = 1) and Eq. 11 (n = 2, rho^{(1)} = P_LZ, rho^{(2)} = P_int). Everything
here is independent of E, T1 and T2.

PHYSICAL extras: gamma_R between consecutive crossings (geometry/wilson.py) and, for twisted LZ, the
nonreciprocity exponent of every crossing, tau = (pi/2) R_s Im k_c (R_s: shift vector for motion along
+s; Im k_c: the DDP branch point, or the hyperbolic 1/(2 sqrt g_ss) without one; 0 at touching points).
Houston TDSE on single-crossing lines gives ln(P_+/P_-) -> -2 tau at low field within 2-11%
(VL/VLc = 0.5-1); the hyperbolic Im k_c is 4-30% worse. Lines with |tau| > 1 are flagged
'twisted_lz_strong': there only the suppressed direction and the low-field limit are reliable
(transport/hall.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...capabilities import CapabilityError
from ..lz.crossings import LineCrossings, find_crossings
from .quadrature import composite_nodes, merge_breaks
from .tables import LambdaGrid, build_table


@dataclass
class Segment:
    start: float  # crossing where the population is injected
    end: float
    r: np.ndarray  # |s - start| at the nodes
    w: np.ndarray  # quadrature weights
    omega_c: np.ndarray
    omega_v: np.ndarray


@dataclass
class LineSolution:
    t: float
    crossings: LineCrossings
    s_cross: np.ndarray  # ascending
    gap_integrals: np.ndarray  # Int eps_g ds between consecutive (ascending) crossings, eV/A
    segments: dict  # direction (+1/-1) -> list[Segment] in traversal order
    tables: dict  # (band, direction) -> (n, K, 3)
    F_inf: dict  # (band, direction) -> (n,)
    M1: dict  # (band, direction) -> (n,)
    flags: set = field(default_factory=set)
    geo_phase: dict = field(default_factory=dict)  # direction -> (n-1,) gamma_R in traversal order (PHYSICAL)
    shift: np.ndarray | None = None  # (n,) shift vector R_s at the (ascending) crossings, A (PHYSICAL)
    g_cross: np.ndarray | None = None  # (n,) g^{cv}_ss at the (ascending) crossings, A^2
    twist: np.ndarray | None = None  # (n,) twisted-LZ exponent tau at the (ascending) crossings (PHYSICAL)
    phase_refinement: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.s_cross.size


def _segments(s, w, oc, ov, s_cross, window, direction):
    order = s_cross if direction > 0 else s_cross[::-1]
    segs = []
    for j, start in enumerate(order):
        if direction > 0:
            end = order[j + 1] if j + 1 < len(order) else window
            m = (s > start) & (s < end)
            r = s[m] - start
        else:
            end = order[j + 1] if j + 1 < len(order) else -window
            m = (s < start) & (s > end)
            r = start - s[m]
        segs.append(Segment(float(start), float(end), r, w[m], oc[m], ov[m]))
    return segs


def solve_line(provider, t: float, cfg, grid: LambdaGrid, keep_nodes: bool = False) -> LineSolution:
    """Geometry stage for one line (config: cfg.crossings, cfg.quadrature)."""
    pol, quad = cfg.crossings, cfg.quadrature
    W = pol.window
    lc = find_crossings(
        provider, t, pol, want_ddp=cfg.conventions.ec_method == "ddp", e_cut=quad.p_cut * cfg.drive.E_max
    )
    for crossing in lc.crossings:
        selected_metric = crossing.g_ss[cfg.conventions.metric]
        if not crossing.touching and (not np.isfinite(selected_metric) or selected_metric <= 0):
            raise ValueError(
                "Requested quantum metric is undefined or non-positive at a non-touching crossing; "
                "refine the grid or provide the missing physical operators"
            )
    lc.crossings.sort(key=lambda c: c.s)
    s_cross = np.sort(lc.s) if lc.n else np.empty(0)
    flags = set(lc.flags)
    pts = [-W, W, *s_cross]
    if not np.any(np.abs(s_cross) < 10 * quad.dmin):
        pts.append(0.0)  # closest approach to Gamma (peak of |Omega| for small t)
    breaks = merge_breaks(pts, 10 * quad.dmin)
    s, w, _ = composite_nodes(breaks, quad.du, quad.dmin, d_outer=1e-3 * W)
    geo = provider.geometry(s, np.full_like(s, t))
    oc, ov, eg = geo.omega_c, geo.omega_v, geo.eps_g
    if not np.isfinite(geo.omega(cfg.conventions.berry)).all() or not np.isfinite(eg).all():
        raise ValueError(
            "Selected geometry is unresolved on integration nodes; refine the band manifold or grid"
        )
    if np.any(geo.deg_flag):
        flags.add("near_degenerate")
    if np.any(geo.extras.get("outside_grid", False)):
        flags.add("outside_grid")  # gridded provider clamped (or extrapolated) values on this line
    gaps = np.array([np.sum((w * eg)[(s > a) & (s < b)]) for a, b in zip(s_cross[:-1], s_cross[1:])])
    segments, tables, F_inf, M1 = {}, {}, {}, {}
    for d in (+1, -1):
        segs = _segments(s, w, oc, ov, s_cross, W, d)
        segments[d] = segs
        for band in ("c", "v"):
            tabs, finf, m1 = [], [], []
            for sg in segs:
                tab, fi, mm = build_table(sg.r, sg.w * (sg.omega_c if band == "c" else sg.omega_v), grid)
                tabs.append(tab)
                finf.append(fi)
                m1.append(mm)
            tables[(band, d)] = np.array(tabs).reshape((len(segs), grid.K, 3))
            F_inf[(band, d)] = np.array(finf)
            M1[(band, d)] = np.array(m1)
    if not keep_nodes:
        for d in segments:
            segments[d] = [
                Segment(sg.start, sg.end, np.empty(0), np.empty(0), np.empty(0), np.empty(0))
                for sg in segments[d]
            ]
    if lc.n == 0:
        flags.add("no_crossing")
    sol = LineSolution(float(t), lc, s_cross, gaps, segments, tables, F_inf, M1, flags)
    sol.g_cross = np.array([c.g_ss["cv"] for c in lc.crossings])
    model = getattr(provider, "model", None)
    if (cfg.phase.geometric or cfg.phase.twisted_lz) and model is None:
        raise CapabilityError(
            "geometric/twisted-LZ phases need wavefunctions; this provider supplies only "
            "precomputed fields (use a model-based provider or switch the phases off)"
        )
    if cfg.phase.geometric and lc.n >= 2:
        from ..geometry.wilson import geometric_phase_refined

        sol.phase_refinement["geometric"] = {}
        for d in (+1, -1):
            order = s_cross if d > 0 else s_cross[::-1]
            refined = [
                geometric_phase_refined(
                    model,
                    a,
                    b,
                    t,
                    provider.angle,
                    n=cfg.phase.phase_nodes,
                    n_max=cfg.phase.phase_max_nodes,
                    tolerance=cfg.phase.phase_tolerance,
                )
                for a, b in zip(order[:-1], order[1:])
            ]
            sol.geo_phase[d] = np.asarray([value for value, _ in refined])
            sol.phase_refinement["geometric"][str(d)] = [info for _, info in refined]
            if any(not info["converged"] for _, info in refined):
                sol.flags.add("phase_under_resolved")
    if cfg.phase.twisted_lz and lc.n:
        from ..geometry.wilson import shift_vector_refined

        refined = [
            shift_vector_refined(
                model,
                x,
                t,
                provider.angle,
                h=1e-3 * max(abs(x), abs(t), 1e-8),
                tolerance=cfg.phase.phase_tolerance,
            )
            for x in s_cross
        ]
        sol.shift = np.asarray([value for value, _ in refined])
        sol.phase_refinement["shift"] = [info for _, info in refined]
        if any(not info["converged"] for _, info in refined):
            sol.flags.add("shift_under_resolved")
        im_kc = np.array(
            [
                0.0
                if c.touching
                else (
                    c.branch_point.imag
                    if (c.branch_point is not None and c.branch_point.imag > 0)
                    else 0.5 / np.sqrt(c.g_ss["cv"])
                )
                for c in lc.crossings
            ]
        )
        sol.twist = 0.5 * np.pi * sol.shift * im_kc
        if np.max(np.abs(sol.twist)) > 1.0:
            sol.flags.add("twisted_lz_strong")
    return sol
