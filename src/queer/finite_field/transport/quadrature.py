"""Single-exponential (logistic) quadrature clustered at panel ends.

On a panel [a, b]: x = a + (b - a) sigma(u), w = du (b - a) sigma (1 - sigma), sigma(u) = 1/(1 + e^-u),
u in [-U_a, U_b] with U = ln((b - a)/d). Near each end the node spacing is geometric (~1/du nodes per
e-fold) down to the clustering distance d, nodes never coincide with panel ends, and the trapezoid
rule in u converges exponentially for integrands that are analytic inside the panel, including
integrable end singularities such as log|x - a| or 1/sqrt(...) kinks.

The truncated trapezoid sum covers u in [-(n_a + 1/2) du, (n_b + 1/2) du]; the two end slabs it leaves
(widths ~d_a and ~d_b in x) get one midpoint node each. The truncation error is then O(d^2 f') for
smooth f instead of O(d f) (e.g. the relaxation weight exp(-|x - a|/lambda) with d << lambda).
"""

from __future__ import annotations

import math

import numpy as np


def logistic_nodes(a: float, b: float, du: float = 0.3, d_a: float = 1e-12, d_b: float = 1e-12):
    """Nodes and weights on [a, b] clustered down to distance d_a at a and d_b at b."""
    L = b - a
    if L <= 0:
        return np.empty(0), np.empty(0)
    Ua = math.log(max(L / max(d_a, 1e-300), 1.0 + 1e-12))
    Ub = math.log(max(L / max(d_b, 1e-300), 1.0 + 1e-12))
    n_a = int(math.ceil(Ua / du))
    n_b = int(math.ceil(Ub / du))
    u = du * np.arange(-n_a, n_b + 1)
    sig = 1.0 / (1.0 + np.exp(-u))  # logistic; both factors keep full relative precision
    one_minus = 1.0 / (1.0 + np.exp(u))
    x = np.where(u <= 0, a + L * sig, b - L * one_minus)  # measured from the nearer end
    w = du * L * sig * one_minus
    # end slabs [a, a + ta] and [b - tb, b] not covered by the sum: one midpoint node each (skipped
    # when the slab is below the floating-point resolution of the end point)
    ta = L / (1.0 + math.exp((n_a + 0.5) * du))
    tb = L / (1.0 + math.exp((n_b + 0.5) * du))
    xa, xb = a + 0.5 * ta, b - 0.5 * tb
    head = [(xa, ta)] if (ta > 0 and a < xa < x[0]) else []
    tail = [(xb, tb)] if (tb > 0 and x[-1] < xb < b) else []
    if head or tail:
        x = np.concatenate([[p for p, _ in head], x, [p for p, _ in tail]])
        w = np.concatenate([[q for _, q in head], w, [q for _, q in tail]])
    return x, w


def composite_nodes(breaks, du: float = 0.3, dmin: float = 1e-12, d_outer=None):
    """Concatenated logistic panels between sorted breakpoints.

    Every interior breakpoint is approached to within dmin from both sides; the two outermost ends use
    d_outer: None (dmin), a number, or a pair (lower end, upper end). Returns (x, w, panel_index).
    """
    br = np.unique(np.asarray(breaks, dtype=float))
    d_outer = dmin if d_outer is None else d_outer
    d_lo, d_hi = (d_outer, d_outer) if np.isscalar(d_outer) else (float(d_outer[0]), float(d_outer[1]))
    xs, ws, idx = [], [], []
    for i, (a, b) in enumerate(zip(br[:-1], br[1:])):
        da = d_lo if i == 0 else dmin
        db = d_hi if i == len(br) - 2 else dmin
        x, w = logistic_nodes(a, b, du, da, db)
        xs.append(x)
        ws.append(w)
        idx.append(np.full(x.size, i))
    if not xs:
        return np.empty(0), np.empty(0), np.empty(0, dtype=int)
    return np.concatenate(xs), np.concatenate(ws), np.concatenate(idx)


def merge_breaks(points, tol: float) -> np.ndarray:
    """Sorted unique breakpoints with points closer than tol merged (first kept)."""
    pts = np.sort(np.asarray(points, dtype=float))
    out = [pts[0]] if pts.size else []
    for p in pts[1:]:
        if p - out[-1] > tol:
            out.append(p)
    return np.array(out)
