import math

import numpy as np
import pytest

from queer.finite_field.transport.quadrature import composite_nodes, logistic_nodes, merge_breaks
from queer.finite_field.transport.tables import LambdaGrid, build_table, eval_tables


def test_logistic_nodes_integrate_smooth_and_endpoint_singular():
    x, w = logistic_nodes(0.0, 1.0, du=0.2, d_a=1e-14, d_b=1e-14)
    assert np.all((x > 0) & (x < 1))
    assert np.sum(w * np.cos(x)) == pytest.approx(math.sin(1.0), rel=1e-10)
    assert np.sum(w * np.log(x)) == pytest.approx(-1.0, rel=1e-9)  # log endpoint singularity
    assert np.sum(w / np.sqrt(x)) == pytest.approx(2.0, rel=1e-6)


def test_composite_nodes_respect_breaks():
    x, w, idx = composite_nodes([-1.0, 0.0, 0.3, 1.0], du=0.25, dmin=1e-12)
    for b in (-1.0, 0.0, 0.3, 1.0):
        assert np.min(np.abs(x - b)) > 0
    f = np.abs(x - 0.3) ** 0.5 * np.exp(-np.abs(x))  # kink at a break
    ref = 0.0
    from scipy.integrate import quad

    for a, b in ((-1, 0), (0, 0.3), (0.3, 1)):
        ref += quad(lambda y: abs(y - 0.3) ** 0.5 * math.exp(-abs(y)), a, b, epsabs=1e-14)[0]
    assert np.sum(w * f) == pytest.approx(ref, rel=1e-9)
    assert set(np.unique(idx)) == {0, 1, 2}


def test_merge_breaks():
    assert np.allclose(merge_breaks([0.0, 1e-13, 0.5, 0.5 + 1e-14, -1.0], 1e-12), [-1.0, 0.0, 0.5])


def test_tables_match_direct_sums(rng):
    grid = LambdaGrid.make(3e-11, 30.0, per_decade=8)
    r = np.sort(rng.uniform(0, 0.03, 400)) ** 1.0
    r = np.concatenate([np.geomspace(1e-12, 1e-3, 200), r])
    c = rng.normal(size=r.size) * np.exp(-r / 0.01)
    tab, finf, m1 = build_table(r, c, grid)
    lam = np.geomspace(1e-11, 1e3, 400)
    got = eval_tables(tab, finf, m1, grid, lam)
    ref = np.array([np.sum(c * np.exp(-r / length)) for length in lam])
    inside = (lam > grid.lam_lo) & (lam < grid.lam_hi)
    assert np.max(np.abs(got[inside] - ref[inside])) < 1e-7 * np.max(np.abs(ref))
    above = lam > grid.lam_hi
    assert np.max(np.abs(got[above] - ref[above])) < 1e-6 * np.max(np.abs(ref))
    assert eval_tables(tab, finf, m1, grid, np.array([np.inf]))[0] == pytest.approx(c.sum())


def test_tables_batched_shapes():
    grid = LambdaGrid.make(1e-10, 1.0, per_decade=6)
    r = np.linspace(1e-6, 0.01, 50)
    tabs = []
    for k in range(6):
        tab, _, _ = build_table(r, np.full_like(r, k + 1.0), grid)
        tabs.append(tab)
    tabs = np.array(tabs).reshape(2, 3, grid.K, 3)
    out = eval_tables(tabs, np.ones((2, 3)), np.zeros((2, 3)), grid, np.array([1e-3, np.inf, 1e-12]))
    assert out.shape == (2, 3, 3)
    assert out[0, 0, 1] == 1.0 and out[1, 2, 1] == 1.0


def test_end_slabs_are_integrated():
    # relaxation weight exp(-x/lambda) with the clustering distance only 30x below lambda: the end slab
    # [0, ~d] gets a midpoint node, so the error is O(d du^2 / lambda) instead of O(d / lambda) = 3%
    lam = 3e-12
    for d, tol in ((lam / 30, 2e-4), (1e-5 * lam, 1e-7)):
        x, w = logistic_nodes(0.0, 1e-2, 0.3, d, 1e-5)
        assert np.all((x > 0) & (x < 1e-2))
        assert np.sum(w * np.exp(-x / lam)) / lam - 1 == pytest.approx(0.0, abs=tol)
    # a smooth integrand that is not small at a coarsely clustered end
    x, w = logistic_nodes(0.0, 2.0, 0.3, 1e-12, 0.1)
    assert np.sum(w * x * x) == pytest.approx(8.0 / 3.0, rel=2e-3)
    x, w, _ = composite_nodes([-1.0, 0.0, 2.0], 0.3, 1e-12, d_outer=(1e-12, 0.1))
    assert np.sum(w * x * x) == pytest.approx(3.0, rel=2e-3)
