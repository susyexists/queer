import math

import numpy as np
import pytest

from queer.finite_field.config import PAPER, PHYSICAL
from queer.finite_field.geometry.provider import ModelGeometry
from queer.finite_field.lz.crossings import branch_points, find_crossings, k_lz
from queer.finite_field.lz.landau_zener import fourband_ddp_t
from queer.finite_field.models import FourBandModel, FourBandParams
from queer.finite_field.transport.lines import solve_line
from queer.finite_field.transport.tables import LambdaGrid
from queer.finite_field.units import interference_frequency

P = FourBandParams()
GAP_MIN = PAPER.crossings
DIABATIC = PAPER.with_(**{"crossings.definition": "diabatic"}).crossings


def _prov(x):
    if abs(x - 1.0) < 1e-4:
        pytest.importorskip("mpmath", reason="critical radial precision extra")
    return ModelGeometry(FourBandModel(P.at(x)))


@pytest.mark.parametrize("x", [0.0, 0.5, 0.9, 1.1, 1.2])
def test_single_crossing_at_gamma_matches_closed_form(x):
    lc = find_crossings(_prov(x), 0.0, GAP_MIN)
    assert lc.n == 1 and abs(lc.s[0]) < 1e-9
    assert lc.Ec()[0] == pytest.approx(P.at(x).ec_k0, rel=1e-8)


# E_c on the t = 0 line (Eq. 8 at the gap minimum), kV/m; Ext. Data Fig. 7e agrees within 0.3%
EC_TABLE = {0.2: 28.834, 0.7: 9.114, 1.3: 10.437, 1.5: 13.676, 1.7: 15.463, 2.0: 17.023}


@pytest.mark.parametrize("x", sorted(EC_TABLE))
def test_ec_table_ext7e(x):
    lc = find_crossings(_prov(x), 0.0, GAP_MIN)
    assert lc.Ec().min() / 1e3 == pytest.approx(EC_TABLE[x], rel=2e-4)


def test_ring_onset_and_positions():
    xs = P.ring_threshold_x()
    assert find_crossings(_prov(xs - 2e-3), 0.0, GAP_MIN).n == 1
    assert find_crossings(_prov(xs + 2e-3), 0.0, GAP_MIN).n == 2
    lc = find_crossings(_prov(1.4), 0.0, GAP_MIN)
    k0 = lc.s[1]
    # minimizer-limited symmetry of the two positions (~sqrt(rounding/curvature)); 3e-11 1/A here
    assert lc.s[0] == pytest.approx(-k0, rel=1e-6) and k0 == pytest.approx(2.0642e-3, rel=1e-4)
    for t in (0.5e-3, 1.5e-3, 0.99 * k0):
        lt = find_crossings(_prov(1.4), t, GAP_MIN)
        assert lt.n == 2
        assert lt.s[1] == pytest.approx(math.sqrt(k0**2 - t**2), rel=1e-5)
    for t in (1.001 * k0, 3e-3):
        assert find_crossings(_prov(1.4), t, GAP_MIN).n == 1


@pytest.mark.precision
def test_critical_point_ec_quadratic_in_t():
    pytest.importorskip("mpmath", reason="critical radial precision extra")
    p = P.at(1.0)
    for t in (1e-5, 1e-6):
        lc = find_crossings(_prov(1.0), t, GAP_MIN)
        assert lc.n == 1
        assert lc.Ec()[0] == pytest.approx(math.pi * p.hbar_v * t**2 * 1e10, rel=2e-3 * t / 1e-5 + 1e-6)


def test_diabatic_crossings_and_lambda():
    p2 = P.at(2.0)
    kd = p2.k_diabatic()
    lc = find_crossings(_prov(2.0), 0.0, DIABATIC)
    assert lc.n == 2 and np.allclose(lc.s, [-kd, kd], rtol=1e-12)
    assert "diabatic_ec_off_stationary" in lc.flags
    t = 0.5 * kd
    lt = find_crossings(_prov(2.0), t, DIABATIC)
    assert np.allclose(lt.s, [-math.sqrt(kd**2 - t**2), math.sqrt(kd**2 - t**2)], rtol=1e-12)
    # below the diabatic onset the definition falls back to the gap minimum
    assert find_crossings(_prov(1.1), 0.0, DIABATIC).n == 1


@pytest.mark.parametrize(
    "definition, x, lam_kVm, tol",
    [("gap_minima", 2.0, 18.697, 2e-4), ("gap_minima", 1.8, 13.145, 2e-4), ("diabatic", 2.0, 18.872, 5e-4)],
)
def test_interference_frequency_on_t0_line(definition, x, lam_kVm, tol):
    cfg = PAPER.with_(**{"crossings.definition": definition})
    grid = LambdaGrid.make(3e-11, 30.0, 8)
    line = solve_line(_prov(x), 0.0, cfg, grid)
    assert line.n == 2
    assert interference_frequency(line.gap_integrals[0]) / 1e3 == pytest.approx(lam_kVm, rel=tol)


def test_k_lz():
    assert k_lz(1e4, 1e4, 1e3) == pytest.approx(1 / (4 * 100.0))
    assert k_lz(1e4, 1e4, 1e5) == pytest.approx(math.sqrt(2 * math.pi * 10) / 400.0)
    assert k_lz(0.0, 1.0, 1.0) == math.inf


# ---- branch points: dominance before e_cut, E_DDP > 0 (review: e_cut ordering) ----------------------------

PHYS = PHYSICAL.crossings


@pytest.mark.parametrize(
    "x, t",
    [
        (0.9, 0.016),
        (0.5, 0.0155),
        (0.5, 0.016),
        pytest.param(1.0, 0.016, marks=pytest.mark.precision),
        (1.1, 0.016),
    ],
)
def test_far_roots_never_replace_a_cut_near_root(x, t):
    """The roots on the same vertical line beyond the nearest one (0.325i, 0.328i) used to get E_DDP < 0 and
    become crossings with ln P > 0 once the near root (E_DDP > e_cut = 2e6 V/m) was cut first."""
    roots = fourband_ddp_t(P.at(x), t)
    assert len(roots) == 1 and roots[0][0].imag < 0.02 and 2e6 < roots[0][1] < 2.5e6
    assert find_crossings(_prov(x), t, PHYS, want_ddp=True, e_cut=2e6).n == 0
    lc = find_crossings(_prov(x), t, PHYS, want_ddp=True, e_cut=np.inf)
    assert lc.n == 1 and lc.crossings[0].Ec_ddp == pytest.approx(roots[0][1])


class _Stub(ModelGeometry):
    def __init__(self, model, roots):
        super().__init__(model)
        self.roots = roots

    def exact_branch_points(self, t):
        return self.roots


def test_nonpositive_or_nonfinite_ddp_is_dropped():
    good = (2e-3 + 1e-3j, 1.5e4)
    stub = _Stub(FourBandModel(P.at(1.5)), [(-2e-3 + 1e-3j, -5.0e3), (5e-3 + 1e-3j, math.nan), good])
    fl = set()
    assert branch_points(stub, 0.0, PHYS, 2e6, flags=fl) == [good] and "ddp_invalid" in fl
    lc = find_crossings(stub, 0.0, PHYS, want_ddp=True, e_cut=2e6)
    assert lc.n == 1 and "ddp_invalid" in lc.flags and lc.crossings[0].Ec_ddp == good[1]
    # dominance comes first: a root shadowed by a lower one cannot resurface when that one is invalid
    stub.roots = [(2e-3 + 1e-3j, -1.0), (2.5e-3 + 3e-3j, 4e4)]
    assert branch_points(stub, 0.0, PHYS, 2e6) == []


def test_exponents_are_never_negative():
    for x in (0.5, 0.9, 1.15, 1.5):
        for t in (0.0, 2e-3, 8e-3, 0.016):
            lc = find_crossings(_prov(x), t, PHYS, want_ddp=True, e_cut=2e6)
            assert np.all(lc.Ec(method="ddp") >= 0) and np.all(lc.Ec() >= 0)


# ---- unresolved crossing pairs (review: not_separated) ------------------------------------------------------


def test_not_separated_flag():
    # branch points: |Re k_1 - Re k_2| < (Im k_1 + Im k_2)/2
    near_onset = find_crossings(_prov(1.125), 0.0, PHYS, want_ddp=True)
    assert near_onset.n == 2 and "not_separated" in near_onset.flags  # +-2.87e-4 + 1.417e-3 i
    assert "not_separated" not in find_crossings(_prov(2.0), 0.0, PHYS, want_ddp=True).flags
    # diabatic pair near tangency vs well inside the circle (hyperbolic width 1/(2 sqrt g))
    kd = P.at(1.5).k_diabatic()
    assert "not_separated" in find_crossings(_prov(1.5), 0.999 * kd, DIABATIC).flags
    assert "not_separated" not in find_crossings(_prov(1.5), 0.0, DIABATIC).flags
    # gap minima just above the ring onset
    xs = P.ring_threshold_x()
    lc = find_crossings(_prov(xs + 2e-3), 0.0, GAP_MIN)
    assert lc.n == 2 and "not_separated" in lc.flags
    assert "not_separated" not in find_crossings(_prov(1.4), 0.0, GAP_MIN).flags


def test_effective_definition_is_recorded():
    assert find_crossings(_prov(1.5), 0.0, GAP_MIN).definition == "gap_minima"
    assert find_crossings(_prov(1.5), 0.0, DIABATIC).definition == "diabatic"
    assert find_crossings(_prov(1.5), 0.01, DIABATIC).definition == "gap_minima"  # outside the circle
    assert find_crossings(_prov(0.5), 0.0, DIABATIC).definition == "gap_minima"  # below the onset
    lc = find_crossings(_prov(1.5), 0.0, PHYS)
    assert lc.definition == "branch_point" and all(c.ddp_source == "exact" for c in lc.crossings)


# ---- t = 0 line scan (review: lines_scan ignored ec_method) -------------------------------------------------


def test_line_scan_reports_the_exponent_jh_uses():
    from queer.finite_field import crossing_scan

    x = np.array([0.5, 0.9, 1.15])
    physical = [crossing_scan(FourBandModel(P.at(value)), [0.0], options=PHYSICAL) for value in x]
    paper = [crossing_scan(FourBandModel(P.at(value)), [0.0], options=PAPER) for value in x]
    assert np.allclose([row["selected_Ec"][0] for row in physical], [16433.79, 1153.166, 6134.3], rtol=2e-5)
    assert np.allclose([row["Ec_eq8"][0] for row in physical], [19034.42, 1284.6, 4419.0], rtol=2e-4)
    assert all(np.array_equal(row["selected_Ec"], row["Ec_ddp"]) for row in physical)
    assert all(np.array_equal(row["selected_Ec"], row["Ec_eq8"]) for row in paper)
    assert all(np.isnan(row["Ec_ddp"]).all() for row in paper)
