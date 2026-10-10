"""Avoided crossings along a field line k = s e + t (z x e) and their Eq. 8 critical fields.

Definitions (CrossingPolicy.definition):
  gap_minima  - interior local minima of eps_g(s) (Methods 6 assumes d eps_g/ds = 0 at the crossing);
                used for E_c in Ext. Data Fig. 7e.
  diabatic    - crossings of the uncoupled surface bands; used for Lambda in Ext. Data Fig. 7e and E_c in 9f.
                Four-band (isotropic): the circle |k| = k_d, A^2 k^2 + (M - B k^2)^2 = VL^2 (closed form).
                Other models with model.layer_projector() P: crossings at the band-pair level of the
                layer-decoupled Hamiltonian P H P + (1 - P) H (1 - P), refined by Brent's method
                (`layer_diabatic_points`; for Eq. 9 this drops exactly Bl sz tx and reproduces k_d).
                Lines that miss the diabatic curve keep their gap minimum (kind 'gap_min'); providers
                with neither route fall back to gap minima on every line ('diabatic_unavailable').
                Eq. 8 is then evaluated where d eps_g/ds != 0 (flagged).
  branch_point- real parts of the complex-k degeneracies (branch points) of eps_c - eps_v nearest the
                real line (PHYSICAL preset); in Eq. 9 they start at |M|/VLc like the diabatic ones.
                Exact for the four-band model; otherwise found on a Chebyshev interpolant of eps_g^2 and
                validated on the complex-k Hamiltonian (landau_zener.line_branch_points). Gap minima
                left without a branch point keep a crossing with the Eq. 8 exponent.

With want_ddp, every crossing also carries the Dykhne-Davis-Pechukas exponent of its branch point
(Ec_ddp, landau_zener.py), the exact adiabatic replacement of Eq. 8. E_c and E_DDP are always >= 0.

For each crossing the critical field (Eq. 8) is stored for every metric choice:
  E_c = pi eps_g / (4 sqrt(g_ss)),  g_ss = band-resolved g^{cv} ("cv") or the total metric of v or c.

LineCrossings.definition is the definition in effect on the line: "gap_minima", "diabatic" (closed
form), "diabatic_layer", "branch_point" (exact or validated) or "branch_point_unvalidated" (fit only).
Line flags:
  window_edge_min, too_many_crossings, touching - scan and count bookkeeping;
  diabatic_ec_off_stationary, branch_point_ec_off_stationary - Eq. 8 is evaluated where d eps_g/ds != 0;
  diabatic_unavailable        - definition 'diabatic' without k_d or a layer projector (or, on this line,
                                 a layer split that moves a remote level across the gap): gap minima used;
  branch_point_unvalidated     - branch points from the continued fit only (no complex-k model);
  branch_point_fit_poor        - unresolved interpolant (unvalidated route): no branch points from it;
  branch_point_missing         - a gap minimum with Eq. 8 E_c < e_cut has no branch point; its crossing
                                 is kept at the minimum with the Eq. 8 exponent;
  branch_point_path_ambiguous  - a root was dropped: the (v, c) pair could not be followed up its DDP path;
  ddp_invalid                  - a dominant branch point with non-finite or non-positive E_DDP was dropped;
  ddp_fallback_eq8             - a crossing carries the Eq. 8 exponent as Ec_ddp;
  not_separated                - two consecutive crossings are closer than their transition width
                                 (branch points: |Re k_1 - Re k_2| < (Im k_1 + Im k_2)/2; otherwise
                                 |s_1 - s_2| below the mean hyperbolic width 1/(2 sqrt g_ss)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import find_peaks

from ..units import critical_field

_METRICS = ("cv", "v_total", "c_total")


@dataclass
class Crossing:
    s: float
    eps_g: float
    g_ss: dict  # metric mode -> g_ss (A^2)
    Ec: dict  # metric mode -> E_c (V/m)
    touching: bool = False
    kind: str = "gap_min"  # "gap_min" | "diabatic" | "branch"
    Ec_ddp: float | None = None  # V/m, DDP exponent of the associated branch point
    branch_point: complex | None = None
    ddp_source: str | None = (
        None  # origin of Ec_ddp: "exact" | "model" (validated) | "fit" | "eq8" (fallback)
    )


@dataclass
class LineCrossings:
    t: float
    crossings: list = field(default_factory=list)
    flags: set = field(default_factory=set)
    definition: str = ""  # definition in effect on this line (module docstring)

    @property
    def n(self) -> int:
        return len(self.crossings)

    @property
    def s(self) -> np.ndarray:
        return np.array([c.s for c in self.crossings])

    def Ec(self, metric: str = "cv", method: str = "eq8") -> np.ndarray:
        if method == "ddp":
            return np.array([c.Ec_ddp if c.Ec_ddp is not None else np.nan for c in self.crossings])
        return np.array([c.Ec[metric] for c in self.crossings])


def _scan_grid(window: float, n: int, d0: float) -> np.ndarray:
    """Uniform grid on [-W, W] merged with a geometric cluster around s = 0 (down to d0)."""
    uni = np.linspace(-window, window, n)
    geo = np.geomspace(d0, window, max(n // 8, 64))
    s = np.concatenate([uni, geo, -geo, [0.0]])
    s = np.unique(s)
    keep = np.concatenate([[True], np.diff(s) > 1e-13 * window])
    return s[keep]


def _crossing(provider, s: float, t: float, kind: str, touching_floor: float) -> Crossing:
    geo = provider.geometry(np.array([s]), np.array([t]))
    eg = float(geo.eps_g[0])
    g = {m: float(geo.metric(m)[0, 0]) for m in _METRICS}
    touching = eg < touching_floor
    # a non-positive (e.g. coarsely interpolated) metric gives NaN for that metric choice only
    Ec = {
        m: (0.0 if touching else (float(critical_field(eg, g[m])) if g[m] > 0 else math.nan))
        for m in _METRICS
    }
    return Crossing(s=float(s), eps_g=eg, g_ss=g, Ec=Ec, touching=touching, kind=kind)


def gap_minima(
    provider,
    t: float,
    window: float,
    scan_points: int = 2049,
    prominence_rel: float = 1e-9,
    d0: float = 1e-12,
) -> tuple[list, set]:
    """Positions of interior local minima of eps_g along the line (refined by bounded Brent on eps_g^2)."""
    s = _scan_grid(window, scan_points, d0)
    eg = provider.gap(s, np.full_like(s, t))
    flags = set()
    tau = prominence_rel * float(np.max(eg)) + 1e-30
    # interior minima whose prominence (rise to the higher of the two bounding maxima) exceeds tau;
    # find_peaks also handles flat plateaus on the clustered grid
    cand, _ = find_peaks(-eg, prominence=tau)
    cand = cand[(cand > 0) & (cand < s.size - 1)]
    if eg[0] < eg[1]:
        flags.add("window_edge_min")
    if eg[-1] < eg[-2]:
        flags.add("window_edge_min")
    # One basin per group of candidates without a barrier above tau between them: on flat bottoms
    # eps_g is constant to the last bit and every rounding wiggle is a discrete "minimum".
    basins = []
    for i in cand:
        if basins:
            j = basins[-1][-1]
            if np.max(eg[j : i + 1]) - max(eg[i], eg[j]) < tau:
                basins[-1].append(i)
                continue
        basins.append([i])

    def f(x):
        return float(provider.gap(np.array([x]), np.array([t]))[0]) ** 2

    out = []
    for b in basins:
        i = b[len(b) // 2]
        lo, hi = s[b[0] - 1], s[b[-1] + 1]
        res = minimize_scalar(
            f, bounds=(lo, hi), method="bounded", options={"xatol": max(1e-15, 1e-12 * (hi - lo))}
        )
        out.append(float(res.x) if f(res.x) <= eg[i] ** 2 else float(s[i]))
    return sorted(out), flags


def _minima(provider, t: float, policy) -> tuple[list, set]:
    if policy.use_radial and getattr(
        provider, "isotropic", False
    ):  # same prominence threshold as the scan path
        return provider.line_minima_radial(t, policy.window, prominence_rel=policy.prominence_rel), set()
    return gap_minima(provider, t, policy.window, policy.scan_points, policy.prominence_rel)


# ---- diabatic crossings ---------------------------------------------------------------------------------


def _layer_split(model):
    """(V, top) with V^dag P V = diag(top) for P = model.layer_projector() (V None if P is diagonal), or None."""
    if model is None or not hasattr(model, "layer_projector"):
        return None
    try:
        P = np.asarray(model.layer_projector(), dtype=complex)
    except (ValueError, TypeError):
        return None
    n = getattr(model, "n_orb", P.shape[0] if P.ndim == 2 else 0)
    if P.shape != (n, n) or not (
        np.allclose(P, P.conj().T, atol=1e-10) and np.allclose(P @ P, P, atol=1e-10)
    ):
        return None
    if np.allclose(P, np.diag(np.diag(P)), atol=1e-12):
        V, top = None, np.real(np.diag(P)) > 0.5
    else:
        w, V = np.linalg.eigh(P)
        top = w > 0.5
    if top.all() or not top.any():
        return None
    return V, top


def _brent(fun, a, b, ya, yb, xtol):
    """brentq on [a, b]; if a pointwise re-evaluation does not confirm the sign change of the scan
    (rounding-level values, e.g. batched vs single-point GEMM), the endpoint with the smaller |y|."""
    try:
        return brentq(fun, a, b, xtol=xtol)
    except ValueError:
        return float(a if abs(ya) <= abs(yb) else b)


def _zeros_on_scan(fun, s, y, xtol: float = 1e-16) -> list:
    """Zeros of a smooth fun sampled as y on the grid s: sign changes, plus pairs of zeros hidden between
    two grid points near an extremum (refined by bounded Brent), all refined by Brent's method."""
    out = [float(x) for x in s[y == 0]]
    sg = np.sign(y)
    for i in np.nonzero(sg[:-1] * sg[1:] < 0)[0]:
        out.append(_brent(fun, s[i], s[i + 1], y[i], y[i + 1], xtol))
    ay = np.abs(y)
    ext = (
        np.nonzero(
            (ay[1:-1] < ay[:-2])
            & (ay[1:-1] <= ay[2:])
            & (sg[:-2] == sg[1:-1])
            & (sg[1:-1] == sg[2:])
            & (sg[1:-1] != 0)
        )[0]
        + 1
    )
    for i in ext:
        x0, x1, x2 = s[i - 1], s[i], s[i + 1]
        y1 = y[i]
        # parabola y1 + p1 u + p2 u^2 (u = x - x1) through the three samples: refine only if its vertex
        # dips to (or across) zero
        d0, d2 = (y[i - 1] - y1) / (x0 - x1), (y[i + 1] - y1) / (x2 - x1)
        p2 = (d2 - d0) / (x2 - x0)
        if p2 == 0:
            continue
        p1 = d0 - p2 * (x0 - x1)
        yv = y1 - p1 * p1 / (4.0 * p2)
        if np.sign(yv) == sg[i] and abs(yv) > 0.5 * abs(y1):
            continue
        sign = sg[i]
        res = minimize_scalar(
            lambda x: sign * fun(x),
            bounds=(x0, x2),
            method="bounded",
            options={"xatol": max(xtol, 1e-14 * (x2 - x0))},
        )
        fm = fun(res.x)
        if sign * fm < 0:
            out += [_brent(fun, x0, res.x, y[i - 1], fm, xtol), _brent(fun, res.x, x2, fm, y[i + 1], xtol)]
    return sorted(out)


def layer_diabatic_points(
    model, t: float, angle: float = 0.0, window: float = 0.03, scan_points: int = 2049, split=None
) -> list:
    """Diabatic crossings on the line: band-pair-level crossings of P H P + (1 - P) H (1 - P), P the layer
    projector (`model.layer_projector()`; `split` from _layer_split).

    With n_occ diabatic levels below the gap, F_j(s) = e_T[j-1](s) - e_B[n_occ-j](s) (j-th top level minus
    the (n_occ-j+1)-th bottom level) vanishes exactly where a top and a bottom diabat cross with n_occ - 1
    levels below them. j runs over the values adjacent to the number of top levels among the lowest
    n_occ on the scan. n_occ is counted below the adiabatic gap centre at both line ends (the truncation can
    move remote levels across the gap; diabatic crossings exchange a top and a bottom level and conserve the
    count). Returns the sorted crossing positions s (1/A), or None if the two ends disagree (a truncated
    remote level crosses the gap: the layer-diabatic picture does not apply on this line).
    """
    split = _layer_split(model) if split is None else split
    if split is None:
        raise ValueError("model has no usable layer projector")
    V, top = split
    v, c = model.band_pair()
    cs, sn = math.cos(angle), math.sin(angle)

    def hk(s):
        s = np.atleast_1d(np.asarray(s, dtype=float))
        return model.hamiltonian(s * cs - t * sn, s * sn + t * cs)

    def levels(s):
        H = hk(s)
        if V is not None:
            H = np.conj(V.T) @ H @ V
        eT = np.linalg.eigvalsh(H[:, top][:, :, top])
        eB = np.linalg.eigvalsh(H[:, ~top][:, :, ~top])
        return eT, eB

    ends = np.array([-0.98 * window, 0.98 * window])
    ev = np.linalg.eigvalsh(hk(ends))
    mu = 0.5 * (ev[:, v] + ev[:, c])
    eT, eB = levels(ends)
    counts = {int(np.sum(eT[i] < mu[i]) + np.sum(eB[i] < mu[i])) for i in range(2)}
    if len(counts) != 1:
        return None
    n_occ = counts.pop()
    s = _scan_grid(window, scan_points, 1e-12)
    eT, eB = levels(s)
    nT, nB = eT.shape[1], eB.shape[1]
    lab = np.concatenate([np.ones(nT, dtype=int), np.zeros(nB, dtype=int)])
    order = np.argsort(np.concatenate([eT, eB], axis=1), axis=1, kind="stable")
    NT = lab[order[:, :n_occ]].sum(axis=1)
    js = sorted(
        {j for j0 in np.unique(NT) for j in (j0, j0 + 1) if 1 <= j <= nT and 0 <= n_occ - j <= nB - 1}
    )
    out = []
    for j in js:

        def F(x, j=j):
            a, b = levels(x)
            return float(a[0, j - 1] - b[0, n_occ - j])

        out += _zeros_on_scan(F, s, eT[:, j - 1] - eB[:, n_occ - j])
    out = sorted(out)
    return [x for i, x in enumerate(out) if i == 0 or x - out[i - 1] > 1e-12 * window]


def layer_polarization_points(
    model, t: float, angle: float = 0.0, window: float = 0.03, scan_points: int = 2049
) -> list:
    """Zeros along the line of <c|P|c> - <v|P|v> (P = model.layer_projector()), refined by Brent's method.

    Diagnostic only: in Eq. 9 the layer character of v and c swaps at every |k| beyond the k = 0 band
    inversion (x > 1), so these zeros exist below the diabatic onset and lie well outside k_d near it.
    """
    P = np.asarray(model.layer_projector(), dtype=complex)
    v, c = model.band_pair()
    cs, sn = math.cos(angle), math.sin(angle)

    def D(s):
        s = np.atleast_1d(np.asarray(s, dtype=float))
        _, U = np.linalg.eigh(model.hamiltonian(s * cs - t * sn, s * sn + t * cs))
        uv, uc = U[..., v], U[..., c]
        return (np.einsum("ki,ij,kj->k", uc.conj(), P, uc) - np.einsum("ki,ij,kj->k", uv.conj(), P, uv)).real

    s = _scan_grid(window, scan_points, 1e-12)
    return _zeros_on_scan(lambda x: float(D(x)[0]), s, D(s))


def _diabatic(provider, t: float, policy=None):
    """(points, effective definition), or (None, None) if the provider supports no diabatic route."""
    if getattr(provider, "isotropic", False) and hasattr(provider, "diabatic_radius"):
        kd = provider.diabatic_radius()
        if kd is not None:
            if abs(t) >= kd:
                return [], "diabatic"
            s = float(np.sqrt(kd * kd - t * t))
            return ([-s, s] if s > 0 else [0.0]), "diabatic"
    model = getattr(provider, "model", None)
    split = _layer_split(model)
    if split is None:
        return None, None
    window = policy.window if policy is not None else 0.03
    n = (
        max(513, policy.scan_points // 4) if policy is not None else 513
    )  # zeros between points: extremum check
    pts = layer_diabatic_points(model, t, getattr(provider, "angle", 0.0), window, n, split)
    return (None, None) if pts is None else (pts, "diabatic_layer")


def diabatic_points(provider, t: float, policy=None) -> list:
    """Crossings of the uncoupled surface bands on the line: the four-band circle of radius k_d, else the
    layer-decoupled Hamiltonian ([] if there are none or the provider supports neither)."""
    return _diabatic(provider, t, policy)[0] or []


# ---- branch points --------------------------------------------------------------------------------------


@dataclass
class _BranchSearch:
    roots: list  # [(k_c, E_DDP, source)]: dominant, 0 < E_DDP finite, sorted by Re (no e_cut)
    flags: set
    orphans: list  # gap minima (Eq. 8 E_c < e_cut) without an associated branch point
    definition: str


def _doubling_distance(provider, t: float, s_m: float, eg_m: float, guess: float, window: float) -> float:
    """Distance from the gap minimum s_m at which eps_g^2 doubles (the nearer side: the local scale; on
    the inner side of a ring minimum the gap may never double before the next minimum).

    Equal to Im k_c for a hyperbolic minimum and to (Im k_c)/1.55 for a quartic one: the window scale.
    """
    target = math.sqrt(2.0) * eg_m
    out = []
    for sign in (-1.0, 1.0):
        lim = window - sign * s_m
        if lim <= 0:
            continue
        d = guess * 2.0 ** np.arange(-6, 14)
        d = np.append(d[d < lim], lim)
        gv = provider.gap(s_m + sign * d, np.full(d.size, t))
        above = np.nonzero(gv >= target)[0]
        if above.size == 0:
            out.append(lim)
        elif above[0] == 0:
            out.append(float(d[0]))
        else:
            i = above[0]

            def f(x, sign=sign):
                return float(provider.gap(np.array([s_m + sign * x]), np.array([t]))[0]) - target

            out.append(brentq(f, d[i - 1], d[i], xtol=1e-3 * d[i - 1]))
    return min(out) if out else window


def _generic_branch_points(provider, t: float, policy):
    """Branch points from interpolants of eps_g^2 around the gap minima (validated when possible).

    Returns (roots [(k_c, E, source)], flags, minima [(s_m, Eq. 8 E_c, d_m)], definition).
    """
    from .landau_zener import line_branch_points

    flags = set()
    mins, _ = _minima(provider, t, policy)
    if not mins:
        return [], flags, [], "branch_point"
    mins = np.asarray(mins, dtype=float)
    geo = provider.geometry(mins, np.full(mins.size, t))
    W = policy.window
    info, intervals = [], []
    for s_m, e_m, g_m in zip(mins, geo.eps_g, geo.metric("cv")[:, 0]):
        if e_m < policy.touching_floor:
            continue  # touching point: real degeneracy, P = 1 without a branch point
        guess = 0.5 / math.sqrt(g_m) if g_m > 0 else 1e-3 * W
        d = _doubling_distance(provider, t, float(s_m), float(e_m), guess, W)
        info.append((float(s_m), float(critical_field(e_m, g_m)) if g_m > 0 else math.nan, d))
        intervals.append([max(-W, s_m - 3.0 * d), min(W, s_m + 3.0 * d), d])
    merged = []
    for lo, hi, d in sorted(intervals):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
            merged[-1][2] = max(merged[-1][2], d)
        else:
            merged.append([lo, hi, d])
    roots, validated = [], True
    for lo, hi, d in merged:
        res = line_branch_points(provider, t, lo, hi, im_max=2.0 * d)
        flags |= res["flags"]
        validated &= res["validated"]
        roots += [(complex(z), float(e), res["source"]) for z, e in zip(res["branch_points"], res["E_ddp"])]
    return roots, flags, info, ("branch_point" if validated else "branch_point_unvalidated")


def _branch_search(provider, t: float, policy, e_cut: float = np.inf) -> _BranchSearch:
    from .landau_zener import dominant_roots

    exact = provider.exact_branch_points(t) if hasattr(provider, "exact_branch_points") else None
    if exact is not None:
        raw, flags, info, definition = (
            [(complex(z), float(e), "exact") for z, e in exact],
            set(),
            [],
            "branch_point",
        )
    else:
        raw, flags, info, definition = _generic_branch_points(provider, t, policy)
    # dominance (the nearest root per vertical line and its cone) first, then the validity of E_DDP
    roots = dominant_roots(raw, key=lambda r: r[0])
    good = [r for r in roots if math.isfinite(r[1]) and r[1] > 0]
    if len(good) < len(roots):
        flags.add("ddp_invalid")
    good.sort(key=lambda r: r[0].real)
    orphans = [
        s_m
        for s_m, ec8, d in info
        if ec8 < e_cut and not any(abs(z.real - s_m) <= 2.0 * d + z.imag for z, _, _ in good)
    ]
    if orphans:
        flags.add("branch_point_missing")
    return _BranchSearch(good, flags, orphans, definition)


def branch_points(provider, t: float, policy, e_cut: float = np.inf, flags: set | None = None) -> list:
    """Dominant branch points [(k_c, E_DDP)] on the line at t with 0 < E_DDP < e_cut, sorted by Re k_c.

    Exact for the four-band model; otherwise eps_g^2 is interpolated around the gap minima, continued
    and validated on the complex-k Hamiltonian (landau_zener.line_branch_points). Dominance (a lower root
    in the downward cone, e.g. on the same vertical line) is applied first, then roots with non-finite or
    non-positive E_DDP are dropped ('ddp_invalid'), then e_cut. Line flags are added to `flags` if given.
    """
    bs = _branch_search(provider, t, policy, e_cut)
    if flags is not None:
        flags |= bs.flags
    return [(z, e) for z, e, _ in bs.roots if e < e_cut]


# ---- crossings on a line --------------------------------------------------------------------------------


def _not_separated(crossings, branch: bool) -> bool:
    cr = sorted(crossings, key=lambda c: c.s)
    for a, b in zip(cr[:-1], cr[1:]):
        if branch and a.branch_point is not None and b.branch_point is not None:
            width = 0.5 * (a.branch_point.imag + b.branch_point.imag)
            if abs(b.branch_point.real - a.branch_point.real) < width:
                return True
            continue
        w = [0.5 / math.sqrt(c.g_ss["cv"]) if (c.g_ss["cv"] > 0 and not c.touching) else 0.0 for c in (a, b)]
        if abs(b.s - a.s) < 0.5 * (w[0] + w[1]):
            return True
    return False


def find_crossings(
    provider, t: float, policy, want_ddp: bool = False, e_cut: float = np.inf
) -> LineCrossings:
    """Crossings on the line at transverse coordinate t according to `policy` (config.CrossingPolicy).

    want_ddp attaches the DDP exponent of the nearest associated branch point (|Re k_c - s| <= 3 Im k_c)
    to every crossing, else the Eq. 8 value ('ddp_fallback_eq8'); e_cut (V/m) discards branch points
    whose exponent is irrelevant (P < exp(-e_cut/E_max)) as crossings of the branch_point definition.
    """
    flags = set()
    attached = None
    if policy.definition == "gap_minima":
        pts, fl = _minima(provider, t, policy)
        flags |= fl
        kinds, definition = ["gap_min"] * len(pts), "gap_minima"
    elif policy.definition == "diabatic":
        pts, definition = _diabatic(provider, t, policy)
        if definition is None:
            flags.add("diabatic_unavailable")
        if pts:
            kinds = ["diabatic"] * len(pts)
            flags.add("diabatic_ec_off_stationary")
        else:
            # outside the diabatic curve the line has a single avoided crossing at the gap minimum
            pts, fl = _minima(provider, t, policy)
            flags |= fl
            kinds, definition = ["gap_min"] * len(pts), "gap_minima"
    elif policy.definition == "branch_point":
        bs = _branch_search(provider, t, policy, e_cut)
        flags |= bs.flags | {"branch_point_ec_off_stationary"}
        items = sorted(
            [(r[0].real, "branch", r) for r in bs.roots if r[1] < e_cut]
            + [(s_m, "gap_min", None) for s_m in bs.orphans],
            key=lambda it: it[0],
        )
        pts, kinds, attached = [it[0] for it in items], [it[1] for it in items], [it[2] for it in items]
        definition = bs.definition
    else:
        raise ValueError(policy.definition)
    if len(pts) > policy.max_crossings:
        flags.add("too_many_crossings")
        pts, kinds = pts[: policy.max_crossings], kinds[: policy.max_crossings]
        attached = None if attached is None else attached[: policy.max_crossings]
    lc = LineCrossings(t=float(t), flags=set(flags), definition=definition)
    for x, kind in zip(pts, kinds):
        lc.crossings.append(_crossing(provider, x, t, kind, policy.touching_floor))
    if attached is not None:
        for c, r in zip(lc.crossings, attached):
            if r is None:  # gap minimum without a branch point: Eq. 8 exponent
                c.Ec_ddp, c.ddp_source = (0.0 if c.touching else c.Ec["cv"]), "eq8"
                lc.flags.add("ddp_fallback_eq8")
            else:
                c.Ec_ddp, c.branch_point, c.ddp_source = (0.0 if c.touching else r[1]), r[0], r[2]
    elif want_ddp:
        bs = _branch_search(provider, t, policy, e_cut)
        lc.flags |= bs.flags - {"branch_point_missing"}
        for c in lc.crossings:
            near = [r for r in bs.roots if abs(r[0].real - c.s) <= 3.0 * r[0].imag]
            if near:
                z, e, src = min(near, key=lambda r: abs(r[0].real - c.s))
                c.Ec_ddp, c.branch_point, c.ddp_source = (0.0 if c.touching else e), z, src
            else:  # no branch point for this crossing: fall back to Eq. 8
                c.Ec_ddp, c.ddp_source = c.Ec["cv"], "eq8"
                lc.flags.add("ddp_fallback_eq8")
    if _not_separated(lc.crossings, policy.definition == "branch_point"):
        lc.flags.add("not_separated")
    if any(c.touching for c in lc.crossings):
        lc.flags.add("touching")
    return lc


def k_lz(Ec: float, g_ss: float, E: float) -> float:
    """Momentum width of the LZ transition (Methods 6): (1/(4 sqrt g)) max(sqrt(2 pi |E|/E_c), 1)."""
    if Ec <= 0:
        return np.inf
    return float(max(np.sqrt(2.0 * np.pi * abs(E) / Ec), 1.0) / (4.0 * np.sqrt(g_ss)))
