"""Beyond Eq. 8: complex branch points and the Dykhne-Davis-Pechukas (DDP) tunnelling exponent.

Eq. 8, E_c = pi eps_g / (4 sqrt(g)), is exact only for a hyperbolic gap. In general the adiabatic
exponent is set by the complex degeneracy k_c (eps_c = eps_v) nearest the real line:

    P = exp(-E_DDP/|E|),   E_DDP = (2/e) Im Int_{Re k_c}^{k_c} eps_g(k) dk      (V/m with 1e10).

For a hyperbolic gap eps_g = 2 sqrt(m^2 + v^2 s^2), k_c = i m/v and E_DDP = pi m^2/(e v) = Eq. 8.
Branch points also define the "branch_point" crossings (PHYSICAL preset): their real parts locate the
interfering LZ events (they start at |M|/VLc = 1.118 in Eq. 9, like the diabatic crossings).

Four-band model: exact route, S(k^2)^2 = 4 P(k^2) (`fourband_ddp_t`).

General models (`line_branch_points`, one window of the line):
  1. candidates: roots of a Chebyshev interpolant of eps_g^2 sampled on the real axis, chopped at its
     noise plateau (`standard_chop`). Beyond the plateau the series fits rounding noise and its roots
     ring the window, which is what produced spurious "branch points" before.
  2. validation (models whose `hamiltonian` accepts complex k, `complex_k = True`): H(k) is entire, so
     each candidate is refined by secant iteration on f(s) = (lambda_c - lambda_v)^2, the (v, c) pair
     of the non-Hermitian H(s e + t z x e) followed continuously from the real axis
     (`PairContinuation`). A root counts only if the pair continued up the vertical path from Re k_c
     coalesces at k_c; E_DDP is then integrated over the exact eigenvalues on that path.
     Candidates that fail are fit artefacts and are dropped.
  3. dominance (`dominant_roots`): a root is dropped when a lower one lies in its downward 45-degree
     cone, |Re k_b - Re k_a| <= Im k_b - Im k_a, e.g. a root above another one on the same vertical line,
     whose DDP path would run through the nearer branch point.
Providers without a complex-k model (geometry grids) keep the analytically continued interpolant,
restricted to resolved fits and to roots that are stable under truncation (unvalidated).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.fft import dct
from scipy.special import roots_legendre

from ..units import EV_PER_A_TO_V_PER_M


def _sqrt_continuous(z):
    """sqrt along a path with the branch chosen continuously from a positive value at z[0]."""
    r = np.sqrt(z.astype(complex))
    if r[0].real < 0:
        r[0] = -r[0]
    for i in range(1, r.size):
        if (r[i] * np.conj(r[i - 1])).real < 0:
            r[i] = -r[i]
    return r


def _gl01(n: int):
    """Gauss-Legendre nodes and weights on (0, 1)."""
    w, wt = roots_legendre(n)
    return 0.5 * (w + 1.0), 0.5 * wt


def ddp_exponent_from_poly(coef, domain, k_c: complex, n: int = 64) -> float:
    """E_DDP (V/m) for eps_g^2 given as a Chebyshev series on `domain`, branch point k_c (Im > 0).

    Vertical path k = Re k_c + i q, q in [0, Im k_c]; substitution q = Im k_c (1 - w^2) removes the
    square-root endpoint singularity.
    """
    a, b = domain
    x0 = k_c.real
    y = k_c.imag
    w, wt = _gl01(n)
    q = y * (1.0 - w * w)
    k = x0 + 1j * q
    xi = (2.0 * k - (a + b)) / (b - a)
    order = np.argsort(q)  # start at the real axis
    eg2 = C.chebval(xi, coef)
    eg = np.empty(n, dtype=complex)
    eg[order] = _sqrt_continuous(eg2[order])
    # Im Int eps_g dk with dk = i dq = i y 2 w dw  ->  Int Re(eps_g) y 2 w dw
    val = np.sum(wt * np.real(eg) * y * 2.0 * w)
    return 2.0 * val * EV_PER_A_TO_V_PER_M


def fit_gap_squared(gap_fn, a: float, b: float, deg: int = 16, n: int = 96):
    """Chebyshev least-squares fit of eps_g^2 on [a, b] from real-axis samples gap_fn(s)."""
    xi = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    s = 0.5 * (a + b) + 0.5 * (b - a) * xi
    eg = gap_fn(s)
    coef = C.chebfit(xi, eg * eg, deg)
    resid = float(np.max(np.abs(C.chebval(xi, coef) - eg * eg)) / np.max(eg * eg))
    return coef, (a, b), resid


def branch_points_from_fit(coef, domain, im_max: float) -> list:
    """Roots of the fitted eps_g^2 with 0 < Im < im_max and Re inside the fit domain, sorted by Re."""
    a, b = domain
    roots = C.chebroots(coef)
    k = 0.5 * (a + b) + 0.5 * (b - a) * roots
    keep = [z for z in k if 0 < z.imag < im_max and a < z.real < b]
    return sorted(keep, key=lambda z: z.real)


# ---- adaptive Chebyshev interpolation -----------------------------------------------------------------


def standard_chop(coeffs, tol: float = 1e-14) -> int:
    """Number of leading Chebyshev coefficients to keep: the series is cut where its envelope reaches
    the noise plateau (Aurentz & Trefethen, ACM TOMS 43, 33 (2017), "standardChop"). Returns
    len(coeffs) when no plateau is found (unresolved)."""
    n = len(coeffs)
    if n < 17:
        return n
    b = np.abs(np.asarray(coeffs, dtype=float))
    env = np.maximum.accumulate(b[::-1])[::-1]  # env[j] = max |c_k|, k >= j
    if env[0] == 0.0:
        return 1
    env = env / env[0]
    plateau = None
    for j in range(2, n + 1):  # 1-based indices as in the reference implementation
        j2 = int(math.floor(1.25 * j + 5.5))
        if j2 > n:
            return n
        e1, e2 = env[j - 1], env[j2 - 1]
        if e1 == 0.0 or e2 / e1 > 3.0 * (1.0 - math.log(e1) / math.log(tol)):
            plateau = j - 1
            break
    if env[plateau - 1] == 0.0:
        return plateau
    j3 = int(np.sum(env >= tol ** (7.0 / 6.0)))
    if j3 < j2:
        j2 = j3 + 1
        env[j2 - 1] = tol ** (7.0 / 6.0)
    cc = np.log10(env[:j2]) + np.linspace(0.0, (-1.0 / 3.0) * math.log10(tol), j2)
    return max(int(np.argmin(cc)), 1)


def chebyshev_interpolant(
    fun, a: float, b: float, n_min: int = 33, n_max: int = 257, tol: float = 1e-14
) -> dict:
    """Chebyshev interpolant of fun on [a, b] at 2^k + 1 Chebyshev-Lobatto points (nested, doubled up to
    n_max until the coefficients reach a plateau), chopped by `standard_chop`.

    Returns {"coef": chopped coefficients, "full": all coefficients, "domain": (a, b), "n": samples,
    "resolved": plateau found, "residual": max |chopped - samples| / max |samples|}.
    """
    n = max(int(n_min), 17)
    n = (1 << int(math.ceil(math.log2(n - 1)))) + 1
    x = np.cos(np.pi * np.arange(n) / (n - 1))
    y = np.asarray(fun(0.5 * (a + b) + 0.5 * (b - a) * x), dtype=float)
    while True:
        c = dct(y, type=1) / (n - 1)
        c[0] *= 0.5
        c[-1] *= 0.5
        L = standard_chop(c, tol)
        if L < n or n >= n_max:
            break
        n2 = 2 * (n - 1) + 1
        xn = np.cos(np.pi * np.arange(1, n2, 2) / (n2 - 1))  # the new (odd) points
        yn = np.asarray(fun(0.5 * (a + b) + 0.5 * (b - a) * xn), dtype=float)
        y2 = np.empty(n2)
        y2[0::2], y2[1::2] = y, yn
        x = np.cos(np.pi * np.arange(n2) / (n2 - 1))
        n, y = n2, y2
    coef = c[:L]
    scale = float(np.max(np.abs(y))) or 1.0
    resid = float(np.max(np.abs(C.chebval(x, coef) - y)) / scale)
    return {
        "coef": coef,
        "full": c,
        "domain": (float(a), float(b)),
        "n": n,
        "resolved": L < n,
        "residual": resid,
    }


def _series_roots(coef, domain) -> np.ndarray:
    a, b = domain
    if len(coef) < 2:
        return np.empty(0, dtype=complex)
    return 0.5 * (a + b) + 0.5 * (b - a) * C.chebroots(coef).astype(complex)


# ---- complex-k continuation of the (v, c) pair --------------------------------------------------------


def supports_complex_k(model) -> bool:
    """True if model.hamiltonian(kx, ky) accepts complex k (declared by `complex_k = True`)."""
    return bool(getattr(model, "complex_k", False))


class PairContinuation:
    """The (v, c) eigenvalue pair of a band model continued to complex s on the line k = s e + t (z x e).

    model.hamiltonian must accept complex k. Eigenvalues of the non-Hermitian H(k) are followed from
    the real axis: the two eigenvalues nearest a linear prediction are taken as the pair if each of them is
    at least twice as close to the previous pair as to any other previous eigenvalue, and every other
    eigenvalue at least twice as close to the other previous eigenvalues; otherwise the step is halved
    (up to max_split times). The order within the pair is irrelevant: f is symmetric.
    """

    def __init__(
        self,
        model,
        t: float,
        angle: float = 0.0,
        bands: tuple[int, int] | None = None,
        max_split: int = 16,
        max_eig: int = 4000,
    ):
        self.model = model
        self.t = float(t)
        self.cos, self.sin = math.cos(angle), math.sin(angle)
        self.v, self.c = bands if bands is not None else model.band_pair()
        self.max_split = int(max_split)
        self.max_eig = int(max_eig)  # budget of eigen-decompositions per track() call
        self.n_eig = 0  # eigen-decompositions performed (cost counter)

    def _k(self, s):
        return s * self.cos - self.t * self.sin, s * self.sin + self.t * self.cos

    def eigvals(self, s) -> np.ndarray:
        """All eigenvalues of H at complex s (shape (m, n), unordered)."""
        s = np.atleast_1d(np.asarray(s, dtype=complex))
        kx, ky = self._k(s)
        self.n_eig += s.size
        return np.linalg.eigvals(self.model.hamiltonian(kx, ky))

    def real_pair(self, s0: float) -> np.ndarray:
        """(eps_v, eps_c) at real s0."""
        return self.real_state(s0)[1]

    def real_state(self, s0: float):
        """Tracking state (s, pair, other eigenvalues) at real s0."""
        kx, ky = self._k(np.array([float(s0)]))
        self.n_eig += 1
        ev = np.linalg.eigvalsh(self.model.hamiltonian(kx, ky))[0]
        pair = np.array([ev[self.v], ev[self.c]], dtype=complex)
        return complex(s0), pair, np.delete(ev, [self.v, self.c]).astype(complex)

    @staticmethod
    def _split_pair(lam, pred):
        """Indices (a, b) of two distinct eigenvalues closest to the predicted pair."""
        d = np.abs(lam[None, :] - pred[:, None])  # (2, n)
        best = None
        for a in np.argsort(d[0])[:3]:
            for b in np.argsort(d[1])[:3]:
                if a != b and (best is None or d[0, a] + d[1, b] < best[0]):
                    best = (d[0, a] + d[1, b], a, b)
        return best[1], best[2]

    def track(self, path, start, prev=None):
        """Pairs (m, 2) along the complex points `path`, or None if the pair cannot be told apart from
        the other eigenvalues. start: state at path[0] (from real_state or `last`); prev: optional
        earlier (s, pair) for the first prediction. The final state is kept in self.last."""
        path = np.asarray(path, dtype=complex)
        out = np.empty((path.size, 2), dtype=complex)
        out[0] = start[1]
        self.last = start
        if path.size == 1:
            return out
        self._budget = self.n_eig + self.max_eig
        lams = self.eigvals(path[1:])
        hist = [prev, start]
        for i in range(1, path.size):
            st = self._step(hist, path[i], lams[i - 1], 0)
            if st is None:
                return None
            out[i] = st[1]
        self.last = hist[-1]
        return out

    def _step(self, hist, s_new, lam_new, depth):
        s0, p0, rest0 = hist[-1]
        if hist[-2] is not None and hist[-2][0] != s0:
            s1, p1 = hist[-2][0], hist[-2][1]
            pred = p0 + (p0 - p1) * (s_new - s0) / (s0 - s1)
        else:
            pred = p0
        a, b = self._split_pair(lam_new, pred)
        pair = lam_new[[a, b]]
        rest = np.delete(lam_new, [a, b])
        if rest0.size == 0:
            ok = True
        else:  # continuity: pair members come from the old pair, all other eigenvalues from the others
            dp_pair = np.min(np.abs(pair[:, None] - p0[None, :]), axis=1)
            dr_pair = np.min(np.abs(pair[:, None] - rest0[None, :]), axis=1)
            dp_rest = np.min(np.abs(rest[:, None] - p0[None, :]), axis=1)
            dr_rest = np.min(np.abs(rest[:, None] - rest0[None, :]), axis=1)
            ok = bool(np.all(dp_pair < 0.5 * dr_pair) and np.all(dr_rest < 0.5 * dp_rest))
        if ok:
            st = (complex(s_new), pair, rest)
            hist.append(st)
            return st
        if depth >= self.max_split or self.n_eig >= self._budget:
            return None
        s_mid = 0.5 * (s0 + s_new)
        if self._step(hist, s_mid, self.eigvals([s_mid])[0], depth + 1) is None:
            return None
        return self._step(hist, s_new, lam_new, depth + 1)

    @staticmethod
    def f(pair) -> complex:
        """(lambda_c - lambda_v)^2: analytic at the branch point, which is a simple zero."""
        return complex((pair[1] - pair[0]) ** 2)

    def near_degeneracy(self, z0: complex, pair0=None) -> float:
        """Smallest separation at z0 of two eigenvalues whose midpoint lies within eps_g/2 of the gap centre
        at Re z0, in units of eps_g(Re z0) (cheap screening of candidates; inf if there is none)."""
        p = self.real_pair(z0.real) if pair0 is None else pair0
        eg = float(np.real(p[1] - p[0]))
        if not eg > 0:
            return math.inf
        lam = self.eigvals([z0])[0]
        mid = 0.5 * (p[0] + p[1])
        near = lam[np.argsort(np.abs(lam - mid))[:6]]
        sep = np.abs(near[:, None] - near[None, :])
        cen = np.abs(0.5 * (near[:, None] + near[None, :]) - mid)
        ok = (cen < 0.5 * eg) & ~np.eye(near.size, dtype=bool)
        return float(np.min(sep[ok])) / eg if ok.any() else math.inf

    def refine(self, z0: complex, deflate=(), n_up: int = 12, maxit: int = 40, xtol: float = 1e-13):
        """Secant iteration from the candidate z0 (Im > 0) on f(s) / prod(s - r), r in `deflate` (roots
        already found, so that a second candidate of a close pair converges to the partner).
        Returns (k_c, converged)."""
        y0 = z0.imag
        if not y0 > 0:
            return z0, False
        defl = np.asarray(deflate, dtype=complex)

        def g(s, st):
            return self.f(st[1]) / np.prod(s - defl) if defl.size else self.f(st[1])

        path = z0.real + 1j * y0 * np.linspace(0.0, 1.0, n_up + 1)
        pairs = self.track(path, self.real_state(z0.real))
        if pairs is None:
            return z0, False
        sa, sta = complex(z0), self.last
        fa = g(sa, sta)
        sb = sa + 1e-3 * y0 * (1.0 + 1.0j)  # off-axis offset: no stagnation on symmetry lines
        if self.track([sa, sb], sta, prev=(path[-2], pairs[-2])) is None:
            return z0, False
        stb = self.last
        fb = g(sb, stb)
        step = math.inf
        for _ in range(maxit):
            if fb == 0:
                return sb, True
            if fb == fa:
                break
            ds = -fb * (sb - sa) / (fb - fa)
            lim = 0.25 * sb.imag  # trust region: stay in the upper half plane
            if abs(ds) > lim:
                ds *= lim / abs(ds)
            sc = sb + ds
            if not sc.imag > 0:
                return sc, False
            if self.track([sb, sc], stb, prev=(sa, sta[1])) is None:
                return sc, False
            sa, sta, fa = sb, stb, fb
            sb, stb = sc, self.last
            fb = g(sb, stb)
            if abs(ds) <= xtol * abs(sb):
                return sb, True
            if abs(ds) <= 1e-9 * sb.imag and abs(ds) > 0.5 * step:
                return sb, True  # stalled at the rounding level of f (noise-limited convergence)
            step = abs(ds)
        self.last = stb
        return sb, step <= 1e-8 * sb.imag

    def partner_estimate(self, k_c: complex):
        """Zero of the linear model of f(s) / (s - k_c) around k_c, called right after refine() converged
        at k_c: close to k_c only if a second root is (a near-double root, e.g. an almost merged pair
        that one fit candidate may represent); about 2 Im k_c away for an isolated hyperbolic root."""
        h = 0.01 * k_c.imag
        st0 = self.last
        g = []
        for d in (-h, h):
            if self.track([k_c, k_c + d], st0) is None:
                return None
            g.append(self.f(self.last[1]) / d)
        self.last = st0
        if g[1] == g[0]:
            return None
        return complex(k_c + h - g[1] * 2.0 * h / (g[1] - g[0]))

    def vertical_ddp(self, k_c: complex, n: int = 32, coalesce_tol: float = 1e-6):
        """(E_DDP in V/m, pair coalesces at k_c) by Gauss-Legendre on the vertical path from Re k_c to k_c
        (q = Im k_c (1 - w^2)) over the continued exact eigenvalues."""
        y = k_c.imag
        w, wt = _gl01(n)
        q = y * (1.0 - w * w)
        order = np.argsort(q)
        path = np.concatenate([[k_c.real + 0j], k_c.real + 1j * q[order], [k_c]])
        pairs = self.track(path, self.real_state(k_c.real))
        if pairs is None:
            return math.nan, False
        f = (pairs[:, 1] - pairs[:, 0]) ** 2
        ok = bool(abs(f[-1]) <= coalesce_tol * abs(f[0]))
        eg = np.empty(n, dtype=complex)
        eg[order] = _sqrt_continuous(f[:-1])[1:]
        val = np.sum(wt * np.real(eg) * y * 2.0 * w)
        return float(2.0 * val * EV_PER_A_TO_V_PER_M), ok


# ---- dominance ------------------------------------------------------------------------------------------


def dominant_roots(roots, key=lambda r: r) -> list:
    """Drop roots that have a lower root in their downward 45-degree cone (|dRe| <= dIm) and duplicates.

    The DDP path of a dominated root runs through (or next to) the nearer branch point, so its exponent
    is not that of an independent LZ event (it can even come out negative). Equal-height pairs (e.g. the
    two roots of an unresolved crossing pair) are kept. `key` maps an item to its complex root.
    """
    items = sorted(roots, key=lambda r: (key(r).imag, key(r).real))
    kept = []
    for r in items:
        z = key(r)
        dup = any(abs(z - key(o)) <= 1e-9 * max(abs(z), abs(key(o))) for o in kept)
        shadowed = any(
            key(o).imag < z.imag * (1.0 - 1e-9) and abs(z.real - key(o).real) <= z.imag - key(o).imag
            for o in kept
        )
        if not (dup or shadowed):
            kept.append(r)
    return kept


# ---- generic route: one window of a line ----------------------------------------------------------------


def line_branch_points(
    provider,
    t: float,
    s_lo: float,
    s_hi: float,
    im_max: float | None = None,
    *,
    validate: bool = True,
    n_max: int = 257,
    n_ddp: int = 32,
    screen: float = 0.8,
    fit_tol: float | None = None,
    max_residual: float = 1e-6,
) -> dict:
    """Dominant branch points of eps_g on the line at t with 0 < Im < im_max and Re in (s_lo, s_hi).

    Candidates come from the chopped Chebyshev interpolant of eps_g^2 (`chebyshev_interpolant`, plateau
    tolerance fit_tol: 1e-14 for models, 1e-10 for interpolated grids). With a complex-k model
    (provider.model, `complex_k`) and validate=True every candidate is screened (two eigenvalues centred
    within eps_g/2 of the gap centre must be closer than `screen` x eps_g(Re z)), refined (with deflation
    by the roots already found, plus a search for the partner of near-double roots) and verified on the
    exact H(k), and E_DDP is integrated over the exact eigenvalues. Otherwise (unvalidated) the continued
    interpolant is used for roots that move by < 2e-2 Im under truncation to 80% of its length (genuine
    roots of interpolated grids move by ~1e-2 Im at most, fit artefacts by ~5e-2 Im or more), and only if it
    is resolved with a relative residual <= max_residual ('branch_point_fit_poor' otherwise).

    Returns {"branch_points": [k_c], "E_ddp": [V/m], "source": "model" | "fit", "validated": bool,
    "fit_residual", "resolved", "n_candidates", "flags": set}.
    """
    t = float(t)
    im_max = 0.5 * (s_hi - s_lo) if im_max is None else im_max
    model = getattr(provider, "model", None)
    use_model = validate and model is not None and supports_complex_k(model)
    tol = (1e-14 if use_model else 1e-10) if fit_tol is None else fit_tol
    fit = chebyshev_interpolant(
        lambda s: provider.gap(s, np.full_like(s, t)) ** 2, s_lo, s_hi, n_max=n_max, tol=tol
    )
    flags = set()
    mid, half = 0.5 * (s_lo + s_hi), 0.5 * (s_hi - s_lo)
    cand = [
        z
        for z in _series_roots(fit["coef"], fit["domain"])
        if 0 < z.imag < im_max and abs(z.real - mid) < half
    ]
    cand.sort(key=lambda z: z.imag)
    out = {
        "fit_residual": fit["residual"],
        "resolved": fit["resolved"],
        "n_candidates": len(cand),
        "flags": flags,
    }
    if use_model:
        pc = PairContinuation(model, t, getattr(provider, "angle", 0.0))
        roots = []
        for z0 in cand:
            if any(o.imag < z0.imag / 1.5 and abs(z0.real - o.real) <= z0.imag - o.imag for o in roots):
                continue  # well inside the cone of a validated root: irrelevant
            if pc.near_degeneracy(z0) > screen:
                continue
            seeds = [(complex(z0), True)]
            while seeds:
                z, primary = seeds.pop()
                k_c, ok = pc.refine(z, deflate=[o for o in roots if abs(o - z) < 2.0 * z.imag])
                if (
                    ok
                    and k_c.imag > 0
                    and abs(k_c - z) <= 0.5 * z.imag
                    and abs(k_c.real - mid) < half
                    and all(abs(k_c - o) > 1e-9 * abs(o) for o in roots)
                ):
                    roots.append(k_c)
                    zp = pc.partner_estimate(k_c) if primary else None
                    if zp is not None and abs(zp - k_c) < 0.5 * k_c.imag and zp.imag > 0:
                        seeds.append((zp, False))  # near-double root: the fit may have merged the pair
        bps, E = [], []
        for k_c in dominant_roots(roots):
            e, ok = pc.vertical_ddp(k_c, n_ddp)
            if ok:
                bps.append(k_c)
                E.append(e)
            else:
                flags.add("branch_point_path_ambiguous")
        out.update(branch_points=bps, E_ddp=E, source="model", validated=True, n_eig=pc.n_eig)
        return out
    # unvalidated: continued interpolant, resolved fits only, roots stable under truncation
    flags.add("branch_point_unvalidated")
    bps, E = [], []
    if not fit["resolved"] or fit["residual"] > max_residual:
        flags.add("branch_point_fit_poor")
    else:
        coef = fit["coef"]
        short = _series_roots(coef[: max(2, int(0.8 * len(coef)))], fit["domain"])
        stable = [z for z in cand if short.size and np.min(np.abs(short - z)) <= 2e-2 * z.imag]
        for z in dominant_roots(stable):
            bps.append(z)
            E.append(ddp_exponent_from_poly(coef, fit["domain"], z))
    out.update(branch_points=bps, E_ddp=E, source="fit", validated=False)
    return out


# ---- four-band model: exact branch points -------------------------------------------------------


def _fourband_discriminant_coeffs(params):
    """S(q)^2 - 4 P(q) (q = k^2) in the cancellation-free form (d0 + a1 q + a2 q^2)^2 + 4 A^2 Bl^2 q.

    S = A^2 q + m^2 + Bl^2 + VL^2, P = m^2 Bl^2 + VL^2 (A^2 q + m^2), m = M - B q; d0 = M^2 - Bl^2 - VL^2
    is formed as (|M| - R)(|M| + R), so the critical point (d0 -> 0) keeps full relative precision.
    Returns (d0, a1, a2, c) with c the ascending power-series coefficients.
    """
    A, B, M, Bl, VL = params.A, params.B, params.M, params.Bl, params.VL
    R = math.hypot(Bl, VL)
    d0 = (abs(M) - R) * (abs(M) + R)
    a1, a2 = A * A - 2.0 * M * B, B * B
    c = np.array(
        [d0 * d0, 2.0 * d0 * a1 + 4.0 * A * A * Bl * Bl, a1 * a1 + 2.0 * d0 * a2, 2.0 * a1 * a2, a2 * a2]
    )
    return d0, a1, a2, c


def fourband_branch_points_k2(params) -> np.ndarray:
    """Complex k^2 where the traceless four-band gap closes: S(k^2)^2 - 4 P(k^2) = 0 (all roots)."""
    from numpy.polynomial import polynomial as Pl

    return Pl.polyroots(_fourband_discriminant_coeffs(params)[3])


def fourband_ddp_t(params, t: float, s_r: float = 0.0, n: int = 96) -> list:
    """Exact dominant branch points on the line at t (Im s > 0) and their E_DDP (V/m), sorted by (Im, Re).

    Roots in the downward cone of a nearer one (`dominant_roots`), e.g. further up the same vertical
    line, are not returned: their vertical DDP path would cross the nearer branch point.
    """
    out = []
    for q in fourband_branch_points_k2(params):
        s = np.sqrt(q - t * t + 0j)
        for z in (s, -s):
            if z.imag > 0:
                out.append(complex(z))
    out = sorted(dominant_roots(out), key=lambda z: (abs(z.imag), z.real))
    return [(z, _fourband_ddp_exact(params, t, z, n)) for z in out]


def _fourband_ddp_exact(params, t, k_c, n=96):
    w, wt = _gl01(n)
    y = k_c.imag
    q = y * (1.0 - w * w)
    s = k_c.real + 1j * q
    k2 = s * s + t * t
    A, B, M, Bl, VL = params.A, params.B, params.M, params.Bl, params.VL
    m = M - B * k2
    a = A * A * k2 + m * m
    S = a + Bl * Bl + VL * VL
    P = m * m * Bl * Bl + VL * VL * a
    order = np.argsort(q)
    rootP = np.empty(n, dtype=complex)
    rootP[order] = _sqrt_continuous(P[order])
    # S - 2 sqrt(P) = (S^2 - 4P) / (S + 2 sqrt(P)) without the cancellation near the gap closing
    d0, a1, a2, _ = _fourband_discriminant_coeffs(params)
    lin = d0 + a1 * k2 + a2 * k2 * k2
    inner = (lin * lin + 4.0 * A * A * Bl * Bl * k2) / (S + 2.0 * rootP)
    eg = np.empty(n, dtype=complex)
    eg[order] = 2.0 * _sqrt_continuous(inner[order])
    val = np.sum(wt * np.real(eg) * y * 2.0 * w)
    return 2.0 * val * EV_PER_A_TO_V_PER_M
