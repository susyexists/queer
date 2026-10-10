import math

import numpy as np
import pytest

from queer.finite_field.config import PAPER, PAPER_FIGURES, PHYSICAL
from queer.finite_field.harmonics import C_F, local_exponent
from queer.finite_field.models import FourBandModel, FourBandParams
from queer.finite_field.transport import critical as cr
from queer.finite_field.transport.hall import HallCalculator

pytest.importorskip("mpmath", reason="extended-precision reference extra")
pytestmark = pytest.mark.precision

P = FourBandParams()


@pytest.fixture(scope="module")
def crit_inf():
    cfg = PAPER.with_(
        **{"relaxation.T1": math.inf, "relaxation.T2": math.inf, "drive.E_max": 1e3, "drive.E_min": 1e-5},
        use_symmetry_shortcuts=True,
    )
    return HallCalculator(FourBandModel(P.at(1.0)), cfg).prepare()


def test_constants(crit_inf):
    cc = cr.critical_constants(crit_inf.provider)
    assert cc["hbar_v"] == pytest.approx(P.at(1.0).hbar_v, rel=1e-6)
    assert cc["C_s"] == pytest.approx(-P.at(1.0).C_critical, rel=1e-5)
    assert cc["C_spread"] < 1e-4  # isotropic, converged in k
    assert crit_inf.diagnostics()["patched"]


def test_log_corrected_asymptote_t1_infinite(crit_inf):
    cc = cr.critical_constants(crit_inf.provider)
    kc = cr.critical_kc(crit_inf.provider, cc["C_s"])
    assert kc == pytest.approx(1.146e-3, rel=1e-2)
    E = np.geomspace(1e-5, 1.0, 11)
    J = crit_inf.J(E)
    assert np.allclose(J / cr.j_critical_log(E, cc["C_s"], cc["hbar_v"], kc), 1.0, atol=5e-4)
    # main-text law misses the log factor ln(2 k_c/delta): ratio ~12 at 1e-5 V/m
    assert J[0] / cr.j_critical_analytic(E[:1], cc["C_s"], cc["hbar_v"])[0] == pytest.approx(11.79, rel=1e-2)


def test_ext9b_like_local_exponent(crit_inf):
    E = np.geomspace(1e-4, 1e3, 36)
    p = local_exponent(E, crit_inf.J(E))
    at = {1e-2: 1.440, 1.0: 1.417, 1e2: 1.363}
    for e, ref in at.items():
        assert p[np.argmin(np.abs(E - e))] == pytest.approx(ref, abs=4e-3)


def test_relaxation_crossover_exponent():
    cfg = PAPER.with_(**{"drive.E_max": 1e3, "drive.E_min": 1e-5}, use_symmetry_shortcuts=True)
    hc = HallCalculator(FourBandModel(P.at(1.0)), cfg).prepare()
    E = np.geomspace(1e-4, 1e3, 36)
    p = local_exponent(E, hc.J(E))
    for e, ref in {1e-3: 1.940, 1e-1: 1.917, 1.0: 1.897, 1e2: 1.822}.items():
        assert p[np.argmin(np.abs(E - e))] == pytest.approx(ref, abs=5e-3)
    assert np.all(p[1:-1] < 2.0)


def test_c_inversion_roundtrip():
    hv, C = 0.147213, -140.45
    E0 = np.array([1e2, 1e3])
    N = np.array([3, 5, 9, 21])
    b = cr.critical_harmonics_analytic(E0, N, C, hv)
    assert np.allclose(cr.c_from_harmonic(N[None, :], b, E0[:, None], hv), C, rtol=1e-12)
    approx = cr.c_from_harmonic(N[None, :], b, E0[:, None], hv, finite_N=False)
    assert approx[0, 0] / C == pytest.approx(1.3245, rel=1e-3)  # c_F underestimates N^{5/2}|b_3| by 32%
    assert approx[0, -1] / C == pytest.approx(1.0, abs=5e-3)


def test_c_from_device_consistency():
    # synthesize a device signal from the critical law and invert it
    hv, C = 0.147439, 207.1  # paper: v = 2.24e4 m/s, C_EXP = 207.1 A
    L, W, R = 10e-6, 5e-6, 2e3
    N = np.array([3, 9, 17])
    current = np.array([1e-7, 2e-7, 4e-7])
    E0 = current * R / L
    bN = cr.critical_harmonics_analytic(E0, N, C, hv)[np.arange(3), np.arange(3)]
    V_N = bN * R * W**2 / L
    assert np.allclose(cr.c_from_device(N, V_N, current, L, W, R, hv, finite_N=True), C, rtol=1e-10)
    # the paper's asymptotic c_F overestimates C from low harmonics (by 32% at N = 3)
    assert cr.c_from_device(N, V_N, current, L, W, R, hv)[0] / C == pytest.approx(1.3245, rel=1e-3)


def test_c_inversion_signs_normalization_and_orders():
    hv, C = 0.147213, -140.45
    E0 = np.array([1e2, 1e3])
    N = np.array([1, 3, 5, 7])
    b = cr.critical_harmonics_analytic(E0, N, C, hv)
    approx = cr.c_from_harmonic(N[None, :], b, E0[:, None], hv, finite_N=False)
    assert np.all(
        np.sign(approx) == np.sign(C)
    )  # the asymptotic factor keeps the sign of beta_N, also at N = 1
    assert approx[0, 0] / C == pytest.approx(0.9153117 / C_F, rel=1e-6)  # 24% low at N = 1
    bf = b * PAPER_FIGURES.conventions.figure_scale  # harmonics of PAPER_FIGURES-based presets
    assert np.allclose(cr.c_from_harmonic(N[None, :], bf, E0[:, None], hv, cfg=PAPER_FIGURES), C, rtol=1e-12)
    assert np.allclose(
        cr.c_from_harmonic(N[None, :], bf, E0[:, None], hv, figure_scale=1 / (4 * math.pi)), C, rtol=1e-12
    )
    bp = cr.critical_harmonics_analytic(E0, N, C, hv, prefactor="e2_2pi_h")
    assert np.allclose(cr.c_from_harmonic(N[None, :], bp, E0[:, None], hv, cfg=PHYSICAL), C, rtol=1e-12)
    with pytest.raises(ValueError, match="odd"):
        cr.c_from_harmonic([2], [1.0], [1e3], hv)


def test_c_from_device_with_sheet_resistance():
    hv, C = 0.147439, 207.1
    L, W, R = 10e-6, 5e-6, 2e3  # R = V_xx / current (longitudinal); sheet resistance rho = R W / L
    N = np.array([3, 9, 17])
    current = np.array([1e-7, 2e-7, 4e-7])
    bN = cr.critical_harmonics_analytic(current * R / L, N, C, hv)[np.arange(3), np.arange(3)]
    V_N = bN * R * W**2 / L
    rho = R * W / L
    assert np.allclose(
        cr.c_from_device(N, V_N, current, L, W, None, hv, finite_N=True, rho_xx=rho), C, rtol=1e-10
    )
    with pytest.raises(ValueError, match="not both"):
        cr.c_from_device(N, V_N, current, L, W, R, hv, rho_xx=rho)
    with pytest.raises(ValueError, match="odd"):
        cr.c_from_device([2], [1e-6], [1e-7], L, W, R, hv)
