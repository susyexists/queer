"""QUEER's quantum-geometry port reproduces the Generator's ``pythtb_qgeom`` kernel.

Each test feeds identical inputs to a QUEER function and to the vendored original
(``tests/oracles/generator_pythtb_qgeom.py``, the ``generator_qgeom`` fixture) on
randomized and structured inputs; every result must agree to 1e-12 of its scale.
The analytic acceptance of the same code lives in ``test_qgeom_physics.py``.
"""

import math
import time

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from qgeom_models import HEXAGONAL, OBLIQUE, layer, position_table
from tight_binding_models import haldane
from wannier90_tb import write_tb

from queer.io.wannier import read_eig, read_tb_layout, read_tb_records
from queer.observables import qgeom as obs
from queer.physics import qgeom as phys

TOL = 1e-12


def close(port, original, tol=TOL):
    """Equal shapes and |port - original| <= tol max(1, |original|) everywhere (NaN where both are NaN)."""
    a, b = np.asarray(port), np.asarray(original)
    assert a.shape == b.shape, (a.shape, b.shape)
    if a.dtype.kind in "biu" or b.dtype.kind in "biu" or a.dtype == object:
        assert_array_equal(a, b)
        return
    identical = (a == b) | (np.isnan(a) & np.isnan(b))
    finite = np.abs(b[np.isfinite(b)])
    scale = max(1.0, float(finite.max()) if finite.size else 1.0)
    with np.errstate(invalid="ignore"):
        assert np.all(identical | (np.abs(a - b) <= tol * scale)), float(np.nanmax(np.abs(a - b)))


def same_tree(port, original, tol=TOL, path="root"):
    """Nested dicts, lists and arrays equal key for key, numbers to ``tol`` of their scale."""
    if isinstance(original, dict):
        assert isinstance(port, dict), path
        assert set(map(str, port)) == set(map(str, original)), (
            path,
            sorted(map(str, port)),
            sorted(original),
        )
        lookup = {str(key): value for key, value in port.items()}
        for key, value in original.items():
            same_tree(lookup[str(key)], value, tol, f"{path}.{key}")
    elif isinstance(original, (list, tuple)) and not (original and isinstance(original[0], (int, float))):
        assert len(port) == len(original), path
        for index, (a, b) in enumerate(zip(port, original)):
            same_tree(a, b, tol, f"{path}[{index}]")
    elif original is None or isinstance(original, (str, bool)):
        assert port == original, (path, port, original)
    else:
        close(port, original, tol)


def tables(core, rng, *, bands=3, lattice=OBLIQUE, full=False, fermi=0.0):
    """The same layer as the oracle's and QUEER's tables, diagonal level or full."""
    R, H, lattice, centres = layer(rng, bands=bands, lattice=lattice)
    H = H.copy()
    H[np.flatnonzero(~R.any(axis=1))[0]] -= fermi * np.eye(bands)
    frame = core.layer_frame(lattice)
    tau = np.stack([centres @ frame["e1"], centres @ frame["e2"]], axis=1)
    if full:
        r = position_table(rng, R, centres)[..., :2] @ np.stack([frame["e1"][:2], frame["e2"][:2]], axis=1)
        return core.Table(R, H, lattice, frame, r=r), obs.QgeomTable(R, H, lattice, frame, r=r), tau
    return core.Table(R, H, lattice, frame, tau=tau), obs.QgeomTable(R, H, lattice, frame, tau=tau), tau


def kpoints(rng, n):
    return np.c_[rng.uniform(-0.5, 0.5, size=(n, 2)), np.zeros(n)]


def test_lattice_frame_and_constants_equal_the_kernel(generator_qgeom):
    core = generator_qgeom
    rng = np.random.default_rng(1)
    for name in (
        "CHI_MA_NM_PER_V2_PER_A_PER_EV",
        "CHI_SI_PER_A_PER_EV",
        "E2_OVER_HBAR_S",
        "E2_OVER_H_S",
        "K_B_EV",
        "GAP_BIN_EDGES_MEV",
        "NX",
        "NB",
        "LEVEL_FULL",
        "LEVEL_DIAGONAL",
    ):
        assert getattr(phys, name) == getattr(core, name), name
    assert dict(phys.CONVENTION_SETS) == core.CONVENTION_SETS
    assert_array_equal(phys.FS_COLUMNS, core.FS_COLUMNS)
    assert_array_equal(phys.SEA_COLUMNS, core.SEA_COLUMNS)
    cells = [OBLIQUE, HEXAGONAL, haldane().lattice]
    cells += [rng.normal(size=(3, 3)) + np.diag([3.0, 3.0, 9.0]) for _ in range(4)]
    for lattice in cells:
        same_tree(phys.layer_frame(lattice), core.layer_frame(lattice))
        frame = core.layer_frame(lattice)
        for name in ("a3_misalignment", "cell_area", "zone_area", "is_hexagonal"):
            assert getattr(phys, name)(lattice) == getattr(core, name)(lattice), name
        close(phys.reciprocal_in_plane(lattice, frame), core.reciprocal_in_plane(lattice, frame))
        k = kpoints(rng, 17)
        binv = core.reciprocal_in_plane(lattice, frame)
        close(phys.frame_k(k, binv), core.kcart(k, binv))
        close(phys.c3_rotate_frac(k, lattice), core.c3_rotate_frac(k, lattice))
    assert phys.is_hexagonal(HEXAGONAL) and not phys.is_hexagonal(OBLIQUE)
    with pytest.raises(ValueError, match="parallel"):
        phys.layer_frame([[1, 0, 0], [2, 0, 0], [0, 0, 1]])


@pytest.mark.parametrize("full", [False, True])
def test_the_stacked_table_and_its_fourier_sums_equal_the_kernel(generator_qgeom, full):
    core = generator_qgeom
    rng = np.random.default_rng(2 + full)
    original, port, _ = tables(core, rng, bands=4, full=full, fermi=0.3)
    assert port.level == original.level and port.T == original.T and port.n == original.n
    close(port.stack, original.stack)
    close(port.R12, original.R12)
    close(port.binv, original.binv)
    assert port.nbytes == original.nbytes
    k = kpoints(rng, 23)
    close(port.energies(k), original.energies(k))
    close(port.fourier(k), original.fourier(k))
    with pytest.raises(ValueError, match="shape"):
        obs.QgeomTable(port.R, np.zeros((len(port.R), 4, 4)), OBLIQUE, port.frame, r=np.zeros((2, 4, 4, 2)))
    with pytest.raises(ValueError, match="tau"):
        obs.QgeomTable(port.R, np.zeros((len(port.R), 4, 4)), OBLIQUE, port.frame)


def test_degenerate_clusters_equal_the_kernel(generator_qgeom):
    rng = np.random.default_rng(4)
    E = np.sort(rng.normal(size=(9, 7)), axis=1)
    E[:, 3] = E[:, 2] + rng.uniform(0, 2e-6, size=9)
    for tol in (0.0, 1e-6, 1e-3, 0.5):
        assert_array_equal(phys.degenerate_clusters(E, tol), generator_qgeom.degenerate_clusters(E, tol))


@pytest.mark.parametrize("full", [False, True])
def test_layer_geometry_equals_the_kernel_geometry(generator_qgeom, full):
    core = generator_qgeom
    rng = np.random.default_rng(10 + full)
    original, port, tau = tables(core, rng, bands=5, full=full)
    k = kpoints(rng, 19)
    F = original.fourier(k)
    positions = {"AW": (F[:, 3], F[:, 4]), "OW": F[:, 5]} if full else {"tau": tau}
    for window, occupied, extra in (
        ((0, 5), (), {}),
        ((1, 4), (1, 2, 3), {}),
        ((2, 3), (2, 7, 0), {"gap_bins": True}),
        ((0, 2), (3,), {"gap_bins": True, "frozen": (-0.5, 0.4), "deg_tol": 1e-3}),
    ):
        kwargs = dict(window=window, occupied=occupied, **extra)
        expected = core.geometry(F[:, 0], (F[:, 1], F[:, 2]), **positions, **kwargs)
        same_tree(phys.layer_geometry(F[:, 0], (F[:, 1], F[:, 2]), **positions, **kwargs), expected)
        same_tree(obs.evaluate(port, k, **kwargs), core.evaluate(original, k, **kwargs))
        eig = np.linalg.eigh(F[:, 0])
        same_tree(phys.layer_geometry(F[:, 0], (F[:, 1], F[:, 2]), **positions, eig=eig, **kwargs), expected)
    same_tree(
        obs.evaluate_chunks(port, k, chunk=4, window=(1, 4), occupied=[2]),
        core.evaluate_chunks(original, k, chunk=4, window=(1, 4), occupied=[2]),
    )
    assert obs.sum_rule_residual(port, k) == core.sum_rule_residual(original, k)
    assert obs.sum_rule_residual(port, k) < 1e-10
    without = core.geometry(F[:, 0], (F[:, 1], F[:, 2]), window=(0, 2))
    same_tree(phys.layer_geometry(F[:, 0], (F[:, 1], F[:, 2]), window=(0, 2)), without)


def test_tensors_and_occupations_equal_the_kernel(generator_qgeom):
    core = generator_qgeom
    rng = np.random.default_rng(20)
    M = rng.normal(size=(6, 2, 2, 2))
    M = 0.5 * (M + np.swapaxes(M, -1, -2))
    for c_out, c_in in ((1.0, -0.5), (2.0, -0.5), (0.3, 0.7)):
        close(phys.chi_from_dipole(M, c_out, c_in), core.chi_from_dipole(M, c_out, c_in))
    same_tree(phys.chi_all_conventions(M), core.chi_all_conventions(M))
    same_tree(phys.trigonal(M), core.trigonal(M))
    for alpha, beta in ((0.3, -1.2), (0.0, 0.0), (-2.0, 0.5)):
        close(phys.trigonal_tensor(alpha, beta), core.trigonal_tensor(alpha, beta))
    for phi in (0.0, 0.4, -2.1):
        close(phys.rotate_tensor(M[0], phi), core.rotate_tensor(M[0], phi))
    for tensor in (M[1], np.zeros((2, 2, 2)), core.trigonal_tensor(0.2, 0.1)):
        assert phys.c3_residual_tensor(tensor) == core.c3_residual_tensor(tensor)
    for K in (rng.normal(size=(2, 2)), np.eye(2), np.zeros((2, 2))):
        assert phys.c3_residual_matrix(K) == core.c3_residual_matrix(K)
    vg = rng.normal(size=(5, 4, 6))
    close(phys.dipole_tensor(vg), core.dipole_tensor(vg))
    close(phys.symmetric_tensor(vg[..., :3]), core.symmetric_tensor(vg[..., :3]))
    x = rng.normal(scale=300, size=50)
    close(phys.fermi_function(x), core.fermi(x))
    for kT in (1e-4, 2.6e-3):
        close(phys.minus_fermi_derivative(x * 1e-3, kT), core.minus_fprime(x * 1e-3, kT))
    close(phys.density_cm2(x, 12.3), core.density_cm2(x, 12.3))
    assert phys.gap_bin_labels() == core.gap_bin_labels() == ["<1", "1-10", "10-100", ">=100"]
    for grid in (np.linspace(-0.1, 0.1, 2001), np.array([0.0]), np.array([0.0, 1.0]), np.r_[0, 1, 3.0]):
        assert phys.is_uniform_grid(grid) == core._is_uniform(grid)
        assert phys.thermal_grid(grid, 1.7e-4) == core.thermal_grid(grid, 1.7e-4)
    extrema = list(rng.uniform(-0.12, 0.12, size=9))
    for args in (((-0.1, 0.1), extrema, [0.0, 0.031]), ((-0.05, 0.02), [], [])):
        close(phys.mu_subset(*args), core.mu_subset(*args))
        close(phys.mu_subset(*args, uniform=7), core.mu_subset(*args, uniform=7))


def triangle_inputs(rng, nt=400, nX=5):
    e = rng.normal(scale=0.1, size=(nt, 3))
    e[:20, 1] = e[:20, 0]  # flat edges
    e[20:30] = e[20:30, :1]  # flat triangles
    X = rng.normal(size=(nt, 3, nX))
    area = rng.uniform(0.5, 2.0, size=nt) * 1e-3
    return e, X, area


def test_triangle_rules_equal_the_kernel(generator_qgeom):
    core = generator_qgeom
    rng = np.random.default_rng(30)
    e, X, area = triangle_inputs(rng)
    mu = np.r_[np.linspace(-0.4, 0.4, 41), e[3, 1], e[40, 0], e[50, 2]]
    es, Xs = core._sorted_triangles(e, X)
    port_sorted = phys.sorted_triangles(e, X)
    close(port_sorted[0], es)
    close(port_sorted[1], Xs)
    for m in mu[::5]:
        close(phys.theta_triangles(es, Xs, area, float(m)), core._theta_one(es, Xs, area, float(m)))
        close(phys.delta_triangles(es, Xs, area, float(m)), core._delta_one(es, Xs, area, float(m)))
    for per in (False, True):
        close(phys.triangle_theta(e, X, area, mu, per), core.triangle_theta(e, X, area, mu, per))
        close(phys.triangle_delta(e, X, area, mu, per), core.triangle_delta(e, X, area, mu, per))
    m_rows = rng.uniform(-0.3, 0.3, size=len(e))
    for kind, one in (("theta", core._theta_one), ("delta", core._delta_one)):
        close(phys.rule_at(kind, es, Xs, area, m_rows), core._pair_values(one, es, Xs, area, m_rows))
        for low in (True, False):
            wide = np.flatnonzero((es[:, 2] - es[:, 0]) > 1e-3)
            close(
                phys.piece_coefficients(kind, low, es[wide], Xs[wide], area[wide]),
                core._piece_coefficients(kind, low, es[wide], Xs[wide], area[wide]),
            )
        for mu0, step, n, chunk in ((-0.4, 0.01, 81, 2_000_000), (-0.2, 0.0007, 600, 97), (0.5, 0.1, 3, 10)):
            close(
                phys.grid_rule(kind, es, Xs, area, mu0, step, n, chunk),
                core._grid_rule(kind, es, Xs, area, mu0, step, n, chunk),
            )
        group = rng.integers(0, 7, size=len(e))
        close(
            phys.grouped_rule(kind, e, X, area, mu, group, 7),
            core._grouped(kind, e, X, area, mu, group, 7),
        )
    close(phys.theta_on_grid(e, X, area, -0.4, 0.002, 401), core.theta_on_grid(e, X, area, -0.4, 0.002, 401))
    close(phys.delta_on_grid(e, X, area, -0.4, 0.002, 401), core.delta_on_grid(e, X, area, -0.4, 0.002, 401))
    first, count = rng.integers(0, 50, size=30), rng.integers(0, 9, size=30)
    port_pairs = list(phys.range_pairs(first, count, 17))
    original_pairs = list(core._range_pairs(first, count, 17))
    assert len(port_pairs) == len(original_pairs)
    for (a, b), (c, d) in zip(port_pairs, original_pairs):
        assert_array_equal(a, c)
        assert_array_equal(b, d)
    target_a, target_b = np.zeros((60, 3)), np.zeros((60, 3))
    j, val = rng.integers(0, 60, size=200), rng.normal(size=(200, 3))
    phys.accumulate_rows(target_a, j, val)
    core._accumulate(target_b, j, val)
    close(target_a, target_b)
    coef = rng.normal(size=(12, 4, 2))
    anchor = rng.uniform(-0.1, 0.1, size=12)
    lo = rng.integers(0, 200, size=12)
    hi = lo + rng.integers(1, 300, size=12)
    out_a, out_b = np.zeros((500, 2)), np.zeros((500, 2))
    phys.add_cubic_pieces(out_a, coef, anchor, -1.0, lo, np.minimum(hi, 500), -0.2, 0.001)
    core._add_cubic_pieces(out_b, coef, anchor, -1.0, lo, np.minimum(hi, 500), -0.2, 0.001)
    close(out_a, out_b)


def test_fields_payload_and_uniform_response_equal_the_kernel(generator_qgeom):
    core = generator_qgeom
    rng = np.random.default_rng(40)
    original, port, tau = tables(core, rng, bands=4, full=True)
    k = kpoints(rng, 13)
    res = core.evaluate(original, k, window=(1, 3), occupied=[1, 2])
    close(phys.integrand_fields(res), core._fields(res))
    whole = core.evaluate(original, k, window=(0, 3), occupied=[3])
    for geometry, lo, n_occ in ((res, 1, 2), (res, 0, None), (res, 1, None), (whole, 0, 3)):
        close(phys.manifold_columns(geometry, lo, n_occ), core._blocks(geometry, lo, n_occ))
    for kwargs in (
        dict(window=(1, 3), n_occ=2, tau_diag=tau),
        dict(window=(0, 4)),
        dict(window=(1, 2), n_occ=None),
    ):
        same_tree(obs.make_payload(port, **kwargs)(k), core.make_payload(original, **kwargs)(k))
    close(obs.uniform_grid(7), core.uniform_grid(7))
    close(obs.uniform_grid(8), core.uniform_grid(8))
    for kT in (0.0, 2e-3):
        kwargs = dict(window=(1, 4), mus=[-0.2, 0.0, 0.1], kT=kT, occupied=[2], chunk=17)
        same_tree(obs.uniform_response(port, 9, **kwargs), core.uniform_response(original, 9, **kwargs))


def test_occupation_pass_equals_the_kernel(generator_qgeom):
    core = generator_qgeom
    rng = np.random.default_rng(50)
    cases = [tables(core, rng, bands=4, lattice=HEXAGONAL), tables(core, rng, bands=3, fermi=0.4)]
    model = haldane(staggered=0.2)
    H = np.asarray(model.hopping_matrices).copy()
    frame = core.layer_frame(model.lattice)
    tau = np.stack([model.centres @ frame["e1"], model.centres @ frame["e2"]], axis=1)
    cases.append(
        (
            core.Table(model.translations, H, model.lattice, frame, tau=tau),
            obs.QgeomTable(model.translations, H, model.lattice, frame, tau=tau),
            tau,
        )
    )
    path = np.c_[np.linspace(0, 0.5, 9), np.linspace(0, 0.3, 9), np.zeros(9)]
    for original, port, _ in cases:
        for kwargs in (
            dict(n=21, mu_range=(-0.3, 0.3), kT_max=1e-3),
            dict(n=16, mu_range=(-0.05, 0.05), kT_max=0.0, occupied_bands=1, extra_kfrac=path, chunk=50),
            dict(n=25, mu_range=(3.0, 3.1), kT_max=2e-3, insulator_min_gap=0.5, hot_gap=0.4, c3_points=0),
        ):
            same_tree(obs.occupation_pass(port, **kwargs), core.occupation_pass(original, **kwargs))
        grid = rng.normal(size=(7, 9))
        for a, b in zip(obs.local_extrema(grid), core._local_extrema(grid)):
            assert_array_equal(a, b)


def test_the_adaptive_mesh_equals_the_kernel(generator_qgeom):
    core = generator_qgeom
    rng = np.random.default_rng(60)
    original, port, _ = tables(core, rng, bands=3)
    payload_a = obs.make_payload(port, window=(0, 3))
    payload_b = core.make_payload(original, window=(0, 3))
    a = obs.AdaptiveMesh(4, port.lattice, payload_a, frame=port.frame)
    b = core.AdaptiveMesh(4, original.lattice, payload_b, frame=original.frame)
    for step in range(3):
        idx = np.unique(rng.integers(0, len(b.lvl), size=5))
        a.split(idx, "error")
        b.split(idx, "error")
    for name in ("lvl", "cx", "cy", "vid", "checked"):
        assert_array_equal(getattr(a, name), getattr(b, name))
    assert a.nv == b.nv and a.vertices_by_rule == b.vertices_by_rule and a.key == b.key
    same_tree(a.vals, b.vals)
    assert_array_equal(a.vertex_keys(), b._buf["_key"][: b.nv])
    vids = np.arange(0, a.nv, 3)
    close(a.vertex_kfrac(vids), b.vertex_kfrac(vids))
    close(a.diameter(a.lvl), b.diameter(b.lvl))
    for x, y in zip(a.triangles(a.vid, a.lvl), b.triangles(b.vid, b.lvl)):
        close(x, y)
    probe = kpoints(rng, 40)
    assert_array_equal(a.contains(probe), b.contains(probe))
    B = rng.normal(size=(a.nv, 6))
    close(obs.cell_block_integral(a, a.vid, a.lvl, B), core._cell_block_integral(b, b.vid, b.lvl, B))
    with pytest.raises(ValueError, match="base mesh"):
        obs.AdaptiveMesh(0, port.lattice, payload_a)


def integration_settings():
    return dict(
        mu_out=np.linspace(-0.25, 0.25, 51),
        temperatures_K=[30.0, 10.0],
        base=8,
        max_points=4000,
        tolerance=2e-2,
        occupation_mesh=31,
        gap_ratio=0.2,
        edge_resolution=2e-3,
    )


def assert_same_integration(port, original):
    """Two adaptive_response results: every curve, record and vertex equal."""
    integ_a, integ_b = port.pop("integrator"), original.pop("integrator")
    port.pop("seconds"), original.pop("seconds")
    same_tree(port, original)
    assert_array_equal(integ_a.mesh.vertex_keys(), integ_b.mesh._buf["_key"][: integ_b.mesh.nv])
    assert integ_a.stop == integ_b.stop
    assert integ_a.gap_rule_capped == integ_b.gap_rule_capped
    assert integ_a.mesh.vertices_by_rule == integ_b.mesh.vertices_by_rule
    same_tree(integ_a.history, integ_b.history)
    close(integ_a.mu_subset, integ_b.mu_subset)
    assert integ_a.headline_mu == integ_b.headline_mu
    close(integ_a.vertex_weights(), integ_b.vertex_weights())
    return integ_a, integ_b


@pytest.mark.parametrize("case", ["metal_full", "insulator_diagonal", "hexagonal_budget"])
def test_adaptive_response_equals_the_kernel(generator_qgeom, case):
    core = generator_qgeom
    rng = np.random.default_rng({"metal_full": 70, "insulator_diagonal": 71, "hexagonal_budget": 72}[case])
    settings = integration_settings()
    if case == "metal_full":
        original, port, tau = tables(core, rng, bands=3, full=True, fermi=0.5)
        settings["tau_diag"] = tau
    elif case == "insulator_diagonal":
        original, port, _ = tables(core, rng, bands=2, fermi=0.0)
        settings.update(occupied_bands=1, insulator_min_gap=1e-3)
    else:
        original, port, _ = tables(core, rng, bands=3, lattice=HEXAGONAL, fermi=0.6)
        settings.update(max_points=900, tolerance=1e-4)
    port_result = obs.adaptive_response(port, **settings)
    original_result = core.adaptive_response(original, **settings)
    integ_a, integ_b = assert_same_integration(port_result, original_result)
    if case == "hexagonal_budget":
        assert integ_a.stop == "budget"
    mu_points = dict(integ_b.headline_mu)
    window = tuple(port_result["window"])
    frozen = (-0.3, 0.2)

    def fields_a(k):
        return obs.source_fields(port, k, window=window, deg_tol=1e-6, frozen=frozen)

    def fields_b(k):
        return core.source_fields(original, k, window=window, deg_tol=1e-6, frozen=frozen)

    k = kpoints(rng, 5)
    same_tree(fields_a(k), fields_b(k))
    same_tree(
        obs.dipole_sources(integ_a, mu_points, fields_a), core.dipole_sources(integ_b, mu_points, fields_b)
    )
    same_tree(
        obs.dipole_sources(integ_a, {"far": 50.0}, fields_a),
        core.dipole_sources(integ_b, {"far": 50.0}, fields_b),
    )


def test_a_replayed_time_stop_reproduces_the_kernel_stopped_by_its_clock(generator_qgeom, monkeypatch):
    """The kernel stopped by its deadline at V vertices and QUEER's stop at V, without a clock, agree."""
    core = generator_qgeom
    rng = np.random.default_rng(80)
    original, port, _ = tables(core, rng, bands=3, fermi=0.4)
    settings = integration_settings()
    settings["tolerance"] = 1e-3
    free = core.adaptive_response(original, **settings)
    history = [entry["vertices"] for entry in free["integrator"].history]
    assert len(set(history)) > 3
    threshold = sorted(set(history))[len(set(history)) // 2]
    counter = {"nv": 0}
    real_ensure = core.AdaptiveMesh.ensure

    def counting_ensure(self, x, y):
        out = real_ensure(self, x, y)
        counter["nv"] = self.nv
        return out

    class Clock:
        @staticmethod
        def time():
            return 1000.0 if counter["nv"] >= threshold else 0.0

    monkeypatch.setattr(core.AdaptiveMesh, "ensure", counting_ensure)
    monkeypatch.setattr(core, "time", Clock)
    stopped = core.adaptive_response(original, deadline=500.0, **settings)
    monkeypatch.undo()
    assert stopped["integrator"].stop == "time"
    final = stopped["stages"].get("tau2") or stopped["stages"]["tau"]
    replay = obs.adaptive_response(port, stop_vertices=final["vertices"], **settings)
    integ_a, _ = assert_same_integration(replay, stopped)
    assert integ_a.stop == "time" and integ_a.stop_at_vertices == final["vertices"]
    unstopped = obs.adaptive_response(port, deadline=time.time() + 3600.0, **settings)
    assert unstopped["integrator"].stop != "time"


def test_tb_file_readers_equal_the_kernel(generator_qgeom, tmp_path):
    core = generator_qgeom
    rng = np.random.default_rng(90)
    R, H, lattice, centres = layer(rng, bands=3)
    R = R[np.lexsort(R.T[::-1])][:7]
    keep = {tuple(r) for r in R}
    R = np.array([r for r in R if tuple(-r) in keep] or [[0, 0, 0]])
    ndegen = rng.integers(1, 4, size=len(R))
    path = tmp_path / "wannier90_tb.dat"
    write_tb(path, lattice, R, ndegen, rng.normal(size=(len(R), 3, 3)), rng.normal(size=(len(R), 3, 3, 3)))
    port, original = read_tb_layout(path), core.tb_layout(str(path))
    for key, value in original.items():
        close(getattr(port, key), value)
    records, expected = read_tb_records(path, port), core.read_tb(str(path), original)
    for key in ("R", "H", "r"):
        close(getattr(records, key), expected[key])
    assert records.rz2_offdiag == expected["rz2_offdiag"]
    truncated = tmp_path / "short_tb.dat"
    truncated.write_bytes(path.read_bytes()[:-40])
    assert read_tb_layout(truncated).expected_bytes != read_tb_layout(truncated).actual_bytes
    with pytest.raises(ValueError):
        (tmp_path / "header_tb.dat").write_text("x\n1 0 0\n")
        read_tb_layout(tmp_path / "header_tb.dat")
    M = rng.normal(size=(len(R), 3, 3, 2)) + 1j * rng.normal(size=(len(R), 3, 3, 2))
    for a, b in zip(phys.hermitize_translations(R, M), core.hermitize(R, M)):
        close(a, b)
    with pytest.raises(ValueError, match="missing"):
        phys.hermitize_translations(np.array([[1, 0, 0]]), M[:1])
    shifted = centres + lattice[0] - 2 * lattice[1]
    for a, b in ((centres, shifted), (centres, centres + 1e-3), (centres[:0], centres[:0])):
        assert phys.inplane_mismatch(a, b, lattice) == core.inplane_mismatch(a, b, lattice)


def test_eig_and_dft_exactness_equal_the_kernel(generator_qgeom, tmp_path):
    core = generator_qgeom
    rng = np.random.default_rng(100)
    energies = np.sort(rng.normal(scale=3, size=(6, 8)), axis=1)
    lines = [f"{b + 1:5d}{k + 1:5d}{energies[k, b]:18.12f}" for k in range(6) for b in range(8)]
    text = "\n".join(lines) + "\n"
    path = tmp_path / "wannier90.eig"
    path.write_text(text)
    close(read_eig(path), core.parse_eig(text))
    tb = energies + rng.normal(scale=1e-5, size=energies.shape)
    tb_short = tb[:, :5]
    for frozen in ((-1.0, 1.0), (-10.0, 10.0), (5.0, 6.0)):
        for e_tb in (tb, tb_short):
            assert phys.dft_exactness(e_tb, energies, frozen) == core.dft_exactness(e_tb, energies, frozen)


def test_kernel_constants_of_the_integrator_equal_the_port(generator_qgeom):
    core = generator_qgeom
    assert obs.TRIANGLES_PER_CHUNK == core.TRIANGLES_PER_CHUNK
    assert obs.VERTICES_PER_CHUNK == core.VERTICES_PER_CHUNK
    assert obs.SMEARED_DOS_FLOOR == core.SMEARED_DOS_FLOOR
    assert obs.VERTEX_KEY == core.VERTEX_KEY
    assert phys.PAIRS_PER_CHUNK == core.PAIRS_PER_CHUNK
    assert phys.BLOCK_PAIRS_PER_CHUNK == core.BLOCK_PAIRS_PER_CHUNK
    assert phys.GRID_BLOCK == core.GRID_BLOCK
    assert (obs.AdaptiveMesh.DEPTH, obs.AdaptiveMesh.MAX_BASE) == (
        core.AdaptiveMesh.DEPTH,
        core.AdaptiveMesh.MAX_BASE,
    )
    assert math.isclose(phys.CHI_MA_NM_PER_V2_PER_A_PER_EV, 2.434135e-2, rel_tol=1e-6)


def test_the_drivers_summary_helpers_equal_the_ports(generator_core, generator_qgeom, tmp_path):
    """The driver's own QGEOM helpers in ``pythtb_core`` and the kernel's output helpers."""
    from queer.compat.generator import qgeom_step as step

    core, kernel = generator_core, generator_qgeom
    rng = np.random.default_rng(110)
    mu = np.linspace(-0.1, 0.1, 21)
    for count in (np.linspace(1.5, 2.5, 21), np.r_[np.full(10, 2.0), np.linspace(2.0, 3.0, 11)], np.ones(21)):
        for n_occ in (1, 2, 3):
            assert step.mu_cnp(count, mu, n_occ) == core._qgeom_mu_cnp(count, mu, n_occ)
    for M in (rng.normal(size=(2, 2, 2)), kernel.trigonal_tensor(0.3, 0.1), np.zeros((2, 2, 2))):
        assert step.chi_amplitudes(M) == core._chi_amplitudes(M, kernel)
    for gap in (None, 1e-3, 5e-5, -5e-5, -1e-2):
        occ = {"global_gap": gap}
        assert step.state_at_cnp(occ, 1e-4) == core._state_at_cnp(occ, 1e-4)
    original, port, _ = tables(kernel, rng, bands=3, fermi=0.4)
    settings = integration_settings()
    a = obs.adaptive_response(port, **settings)["integrator"]
    b = kernel.adaptive_response(original, **settings)["integrator"]
    bcd = rng.normal(size=2)
    for m in (0.0, float(a.mu_subset[3]), 0.0123):
        assert step.bcd_residual(a, bcd, m) == core._bcd_residual(b, bcd, m)
    payload = {"x": np.linspace(0, 1, 7), "y": [1.23456789, float("nan"), 3], "z": {"t": np.float32(2.5)}}
    assert step.compact(payload) == kernel.compact(payload)
    assert step.compact(payload, 3) == kernel.compact(payload, 3)
    size_a = step.write_json_compact(tmp_path / "a.json", payload)
    size_b = kernel.write_json_compact(str(tmp_path / "b.json"), payload)
    assert size_a == size_b and (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    curves = {"mu": np.arange(9.0), "list": list(range(9)), "short": [1, 2], "nested": {"c": np.ones((9, 2))}}
    for stride in (1, 2, 4):
        same_tree(step.decimate(curves, stride, 9), kernel.decimate(curves, stride, 9))
    assert step.units() == kernel.units()
    assert step.QGEOM_OUT_BUDGET_BYTES == core.QGEOM_OUT_BUDGET_BYTES
    assert (
        step.QGEOM_SMEARED_WARN == core.QGEOM_SMEARED_WARN
        and step.ANTI_HERMITIAN_WARN == core.ANTI_HERMITIAN_WARN
    )
    assert step.FULL_POSITION_NOTE == core.FULL_POSITION_NOTE


def test_the_display_path_equals_the_drivers(generator_core, generator_qgeom):
    """``path_points``: the band path plus log-spaced points around each tick, as ``_qgeom_path_points``."""
    from types import SimpleNamespace

    from queer.compat.generator import qgeom_step as step
    from queer.compat.generator.pythtb_params import translate_pythtb

    core, kernel = generator_core, generator_qgeom
    lattice = HEXAGONAL
    frame = kernel.layer_frame(lattice)
    kpoints = np.r_[
        np.c_[np.linspace(0, 0.5, 11), np.zeros(11)],
        np.c_[np.linspace(0.5, 1 / 3, 8), np.linspace(0, 1 / 3, 8)],
    ]
    path = {
        "kpoints": np.c_[kpoints, np.zeros(len(kpoints))],
        "tick_indices": [0, 10, 18],
        "tick_labels": ["G", "M", "K"],
        "break_indices": [11],
    }
    for settings in (
        {"path_log_points": 5},
        {"path_log_points": 0},
        {"path_kpoints_frac": [[0, 0, 0], [0.5, 0, 0]]},
    ):
        params = translate_pythtb({"step_type": "PYTHTB_QGEOM", **settings})
        ours = SimpleNamespace(params=params, lattice=lattice, band_path=lambda: dict(path), record=None)
        theirs = SimpleNamespace(
            params=dict(params.values), lattice=lattice, band_path=lambda: dict(path), warnings=[]
        )
        same_tree(step.path_points(ours, frame), core._qgeom_path_points(theirs, kernel, frame))


def test_the_full_position_table_and_its_checks_equal_the_drivers(
    generator_core, generator_qgeom, tmp_path, monkeypatch
):
    """``full_table`` against ``_qgeom_full_table`` on a staged WANNIER job (no Wigner-Seitz correction)."""
    import json
    from types import SimpleNamespace

    from generator_golden_cases import write_case

    from queer.compat.generator import qgeom_step as step
    from queer.compat.generator.pythtb_params import translate_pythtb
    from queer.compat.generator.wannier_steps import WannierStepContext

    core, kernel = generator_core, generator_qgeom
    params = write_case("qgeom_full_insulator", tmp_path)
    context = WannierStepContext(
        translate_pythtb(json.loads(params.read_text())), tmp_path, log=lambda line: None
    )
    context.build_model()
    assert context.ws_report["mode"] == "off"
    frame = kernel.layer_frame(context.lattice)
    H = np.asarray(context.table_H, complex).copy()
    H[np.flatnonzero(~context.table_R.any(axis=1))[0]] -= context.fermi * np.eye(context.n_orb)
    table, report = step.full_table(context, frame, H)
    driver = SimpleNamespace(
        n_orb=context.n_orb,
        lattice=context.lattice,
        centres=context.centres,
        hr_degeneracy=context.hr_degeneracy,
        table_R=context.table_R,
        table_H=context.table_H,
        params=dict(context.params.values),
        warnings=[],
        log=lambda line: None,
        _record_input=lambda name: None,
        ws_apply=lambda R, stack: core.fold_axes(R, stack, context.folded_axes),
    )
    monkeypatch.chdir(tmp_path)
    expected_table, expected_report = core._qgeom_full_table(driver, kernel, frame, H)
    close(table.stack, expected_table.stack)
    assert table.level == expected_table.level == "full"
    for value in (report, expected_report):
        value["tb"].pop("read_seconds")
    same_tree(report, expected_report)
