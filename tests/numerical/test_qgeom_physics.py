"""Analytic acceptance of QUEER's quantum-geometry integrator.

The triangle rules are checked against exact polygon clipping of linear fields,
the band geometry against QUEER's own Kubo curvature and finite-basis metric and
against its sum rules, and the adaptive zone integrals against the Haldane model's
Chern number, Hall plateau and a uniform-grid thermal integral.
"""

import math

import numpy as np
import pytest
from qgeom_models import HEXAGONAL, OBLIQUE, layer, position_table
from tight_binding_models import haldane

from queer.observables import qgeom as obs
from queer.observables.geometry import band_pair_geometry, berry_curvature
from queer.observables.result import ObservableResult
from queer.physics import qgeom as phys


def clipped_integral(e, X, area, mu):
    """int theta(mu - E) X dA over one triangle: the triangle clipped to E < mu, and the linear X
    integrated over the polygon left (each fan piece's area times X at its centroid)."""
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    polygon = []
    for a in range(3):
        b = (a + 1) % 3
        if e[a] < mu:
            polygon.append((corners[a], X[a]))
        if (e[a] < mu) != (e[b] < mu):
            t = (mu - e[a]) / (e[b] - e[a])
            polygon.append((corners[a] + t * (corners[b] - corners[a]), X[a] + t * (X[b] - X[a])))
    weighted = np.zeros(X.shape[1])
    if len(polygon) < 3:
        return weighted
    p0, x0 = polygon[0]
    for (p1, x1), (p2, x2) in zip(polygon[1:-1], polygon[2:]):
        piece = 0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))
        weighted += piece * (x0 + x1 + x2) / 3.0
    return area * weighted / 0.5


def test_the_triangle_rules_integrate_linear_fields_exactly():
    rng = np.random.default_rng(1)
    for _ in range(6):
        e = rng.uniform(-0.1, 0.1, size=3)
        X = rng.normal(size=(3, 2))
        area = 0.7
        for mu in (-0.2, *np.sort(e) + 1e-4, float(np.mean(e)), 0.2):
            theta = phys.triangle_theta(e[None], X[None], np.array([area]), np.array([mu]))[0]
            np.testing.assert_allclose(theta, clipped_integral(e, X, area, mu), atol=1e-13)
    e = np.array([-0.05, 0.01, 0.08])
    X = np.array([[1.0, 0.5], [2.0, -1.0], [-0.5, 3.0]])
    for mu in (-0.03, 0.0, 0.05):
        delta = phys.triangle_delta(e[None], X[None], np.array([1.0]), np.array([mu]))[0]
        # The delta rule is the mu derivative of the theta rule.
        h = 1e-6
        up, down = clipped_integral(e, X, 1.0, mu + h), clipped_integral(e, X, 1.0, mu - h)
        np.testing.assert_allclose(delta, (up - down) / (2 * h), rtol=1e-6)


def test_grid_rules_equal_the_pointwise_rules_on_every_grid_point():
    rng = np.random.default_rng(2)
    e = rng.normal(scale=0.05, size=(300, 3))
    X = rng.normal(size=(300, 3, 4))
    area = rng.uniform(0.5, 1.5, size=300)
    mu = -0.2 + 0.001 * np.arange(401)
    np.testing.assert_allclose(
        phys.theta_on_grid(e, X, area, -0.2, 0.001, 401), phys.triangle_theta(e, X, area, mu), atol=1e-10
    )
    np.testing.assert_allclose(
        phys.delta_on_grid(e, X, area, -0.2, 0.001, 401), phys.triangle_delta(e, X, area, mu), atol=1e-8
    )
    group = rng.integers(0, 3, size=300)
    grouped = phys.grouped_rule("theta", e, X, area, mu[::7], group, 3)
    np.testing.assert_allclose(grouped.sum(axis=1), phys.triangle_theta(e, X, area, mu[::7]), atol=1e-10)


def test_the_trigonal_amplitude_rotates_with_three_times_the_angle():
    rng = np.random.default_rng(3)
    T = phys.trigonal_tensor(0.3, -0.7)
    assert phys.c3_residual_tensor(T) < 1e-15
    np.testing.assert_allclose(phys.rotate_tensor(T, 2 * math.pi / 3), T, atol=1e-14)
    M = rng.normal(size=(2, 2, 2))
    M = 0.5 * (M + np.swapaxes(M, 1, 2))
    base = phys.trigonal(M)
    for phi in (0.3, -1.1, 2.0):
        rotated = phys.trigonal(phys.rotate_tensor(M, phi))
        z = (base["alpha"] + 1j * base["beta"]) * np.exp(3j * phi)
        np.testing.assert_allclose([rotated["alpha"], rotated["beta"]], [z.real, z.imag], atol=1e-13)
        assert rotated["A3"] == pytest.approx(base["A3"], rel=1e-13)
    chi = phys.chi_all_conventions(T)
    # A fully symmetric trigonal M gives chi = -(c_out + 2 c_in) M: 0, -1 and +1.5 times M.
    scale = phys.CHI_MA_NM_PER_V2_PER_A_PER_EV
    np.testing.assert_allclose(chi["wpd_1_m05"], 0.0, atol=1e-18)
    np.testing.assert_allclose(chi["lka_2_m05"], -1.0 * scale * T, atol=1e-16)
    np.testing.assert_allclose(chi["qk_05_m1"], 1.5 * scale * T, atol=1e-16)


def diagonal_table(model, fermi=0.0):
    H = np.asarray(model.hopping_matrices, complex).copy()
    H[np.flatnonzero(~np.asarray(model.translations).any(axis=1))[0]] -= fermi * np.eye(model.nbnd)
    frame = phys.layer_frame(model.lattice)
    tau = np.stack([model.centres @ frame["e1"], model.centres @ frame["e2"]], axis=1)
    return obs.QgeomTable(model.translations, H, model.lattice, frame, tau=tau)


def test_the_diagonal_geometry_is_queers_kubo_curvature_and_finite_basis_metric():
    rng = np.random.default_rng(4)
    from queer.models.wannier import WannierModel

    R, H, lattice, centres = layer(rng, bands=4, lattice=OBLIQUE)
    model = WannierModel(R, H, lattice=lattice, centres=centres)
    table = obs.QgeomTable.from_model(model)
    k = np.c_[rng.uniform(-0.5, 0.5, size=(9, 2)), np.zeros(9)]
    frame = table.frame
    P = np.stack([frame["e1"], frame["e2"]])
    res = obs.evaluate(table, k, window=(0, 4), occupied=[2])
    for band in range(4):
        partner = band + 1 if band < 3 else band - 1
        pair = (band, partner) if band < partner else (partner, band)
        geometry = band_pair_geometry(model, k, bands=pair)
        metric = geometry["total_metric"][pair.index(band)]
        np.testing.assert_allclose(res["g"][:, band], np.einsum("ia,kab,jb->kij", P, metric, P), atol=1e-10)
        curvature = berry_curvature(model, k, bands=[band], engine="kubo")["curvature"][0]
        omega = np.einsum("a,kab,b->k", frame["e1"], curvature, frame["e2"])
        np.testing.assert_allclose(res["omega"][:, band], omega, atol=1e-10)
    total = berry_curvature(model, k, occupied_bands=2, engine="kubo")["curvature"]
    np.testing.assert_allclose(
        res["blocks"][2]["omega"], np.einsum("a,kab,b->k", frame["e1"], total, frame["e2"]), atol=1e-10
    )
    assert obs.sum_rule_residual(table, k) < 1e-12


def test_the_full_level_with_diagonal_positions_is_the_diagonal_level():
    rng = np.random.default_rng(5)
    R, H, lattice, centres = layer(rng, bands=3)
    frame = phys.layer_frame(lattice)
    tau = np.stack([centres @ frame["e1"], centres @ frame["e2"]], axis=1)
    r = np.zeros(H.shape + (2,), complex)
    home = np.flatnonzero(~R.any(axis=1))[0]
    r[home, np.arange(3), np.arange(3)] = tau
    full = obs.QgeomTable(R, H, lattice, frame, r=r)
    diagonal = obs.QgeomTable(R, H, lattice, frame, tau=tau)
    k = np.c_[rng.uniform(-0.5, 0.5, size=(11, 2)), np.zeros(11)]
    a = obs.evaluate(full, k, window=(0, 3), occupied=[1, 2])
    b = obs.evaluate(diagonal, k, window=(0, 3), occupied=[1, 2])
    for key in ("v", "g", "omega", "G"):
        np.testing.assert_allclose(a[key], b[key], atol=1e-10)
    np.testing.assert_allclose(a["omega_bar"], 0.0, atol=1e-12)
    for c in (1, 2):
        for key in ("g_in", "omega"):
            np.testing.assert_allclose(a["blocks"][c][key], b["blocks"][c][key], atol=1e-10)
    # Real position matrices add an Omega-bar term but keep the manifold PSD and K >= |C| per point.
    r_full = position_table(rng, R, centres)[..., :2] @ np.stack([frame["e1"][:2], frame["e2"][:2]], axis=1)
    c = obs.evaluate(obs.QgeomTable(R, H, lattice, frame, r=r_full), k, window=(0, 3), occupied=[1])
    assert np.all(c["blocks"][1]["psd_min_rel"] > -1e-12)
    assert np.all(
        np.trace(c["blocks"][1]["g_in"], axis1=1, axis2=2) >= np.abs(c["blocks"][1]["omega_in"]) - 1e-12
    )


@pytest.mark.parametrize(("staggered", "chern"), [(0.2, -1), (1.2, 0)])
def test_the_adaptive_integral_gives_the_haldane_chern_number_and_hall_plateau(staggered, chern):
    model = haldane(staggered=staggered)
    table = diagonal_table(model)
    mu = np.linspace(-0.1, 0.1, 21)
    result = obs.nonlinear_hall_response(
        table, mu, temperatures_K=[30.0], base=16, max_points=20000, tolerance=1e-3, occupation_mesh=41
    )
    assert isinstance(result, ObservableResult)
    manifold = result.metadata["occupied_manifold"]
    assert manifold["applicable"] and manifold["n_occ"] == 1
    assert manifold["chern"] == pytest.approx(chern, abs=2e-3)
    assert result.dims["M"] == ("temperature", "mu", "frame", "frame2", "frame3")
    # sigma_xy = -C e^2/h in the gap at T = 0, no Fermi surface, the count is the filling.
    np.testing.assert_allclose(result["sigma_xy"][0], -manifold["chern"], atol=1e-12)
    np.testing.assert_allclose(result["dos"][0], 0.0, atol=1e-12)
    np.testing.assert_allclose(result["M"][0], 0.0, atol=1e-12)
    np.testing.assert_allclose(result["count"][0], 1.0, atol=1e-12)
    K = np.array(manifold["K_in"])
    assert np.trace(K) >= abs(manifold["chern_in"]) - 1e-6


def test_the_adaptive_thermal_integrals_meet_a_fine_uniform_grid():
    model = haldane(staggered=0.2)
    table = diagonal_table(model, fermi=0.9)
    mu = np.linspace(-0.05, 0.05, 11)
    kT = phys.K_B_EV * 300.0
    adaptive = obs.nonlinear_hall_response(
        table, mu, temperatures_K=[300.0], base=32, max_points=200000, tolerance=1e-4, occupation_mesh=61
    )
    uniform = obs.uniform_response(table, 360, window=(0, 2), mus=mu, kT=kT, chunk=4096)
    np.testing.assert_allclose(adaptive["dos"][1], uniform["dos"], rtol=5e-3)
    np.testing.assert_allclose(adaptive["sigma_xy"][1], uniform["sigma_xy_e2_h"], atol=2e-3)
    scale = np.abs(uniform["M"]).max()
    np.testing.assert_allclose(adaptive["M"][1], uniform["M"], atol=2e-2 * scale)
    # The dipole of a C3-symmetric model is trigonal.
    assert phys.c3_residual_tensor(adaptive["M"][1][5]) < 2e-2


def test_layer_band_geometry_is_an_observable_with_units():
    rng = np.random.default_rng(6)
    R, H, lattice, centres = layer(rng, bands=3, lattice=HEXAGONAL)
    frame = phys.layer_frame(lattice)
    tau = np.stack([centres @ frame["e1"], centres @ frame["e2"]], axis=1)
    table = obs.QgeomTable(R, H, lattice, frame, tau=tau)
    k = np.c_[rng.uniform(-0.5, 0.5, size=(5, 2)), np.zeros(5)]
    result = obs.layer_band_geometry(table, k, window=(0, 2), occupied=[1])
    assert result.units["bcp"] == "Angstrom^2/eV" and result.dims["metric"] == (
        "window_band",
        "kpoint",
        "frame",
        "frame2",
    )
    assert result["energies"].shape == (3, 5) and result["manifold_curvature"].shape == (1, 5)
    assert result.metadata["position_level"] == "diagonal"
    with pytest.raises(ValueError):
        obs.layer_band_geometry(table, np.zeros((0, 3)), window=(0, 2))


def test_a_wannier_model_gives_its_table_at_both_levels():
    from queer.models.wannier import WannierModel

    rng = np.random.default_rng(7)
    R, H, lattice, centres = layer(rng, bands=3)
    r = position_table(rng, R, centres)
    model = WannierModel(R, H, lattice=lattice, centres=centres, position=r, fermi_energy=0.4)
    frame = phys.layer_frame(lattice)
    shifted = H.copy()
    shifted[np.flatnonzero(~R.any(axis=1))[0]] -= 0.4 * np.eye(3)
    proj = np.stack([frame["e1"][:2], frame["e2"][:2]], axis=1)
    full = obs.QgeomTable.from_model(model)
    expected = obs.QgeomTable(R, shifted, lattice, frame, r=r[..., :2] @ proj)
    assert full.level == "full"
    np.testing.assert_array_equal(full.stack, expected.stack)
    diagonal = obs.QgeomTable.from_model(model, level="diagonal")
    tau = np.stack([centres @ frame["e1"], centres @ frame["e2"]], axis=1)
    assert diagonal.level == "diagonal"
    np.testing.assert_allclose(diagonal.tau, tau, atol=1e-13)
    tilted = lattice.copy()
    tilted[1, 2] = 0.5
    with pytest.raises(ValueError, match="normal along z"):
        obs.QgeomTable.from_model(WannierModel(R, H, lattice=tilted, centres=centres, position=r))
    with pytest.raises(ValueError, match="position matrices"):
        obs.QgeomTable.from_model(WannierModel(R, H, lattice=lattice, centres=centres), level="full")
