"""Critical and generic branch-point regressions from the current HHH reference."""

import math

import numpy as np
import pytest

from queer.finite_field import FourBandModel, FourBandParams, NativePlaneModel, crossing_scan
from queer.finite_field.config import PAPER, PHYSICAL
from queer.finite_field.geometry.gridded import GriddedGeometry
from queer.finite_field.geometry.provider import ModelGeometry
from queer.finite_field.lz.crossings import find_crossings
from queer.finite_field.lz.landau_zener import fourband_branch_points_k2, fourband_ddp_t
from queer.finite_field.models.lattice_reg import lattice_regularized_fourband
from queer.finite_field.transport.hall import HallCalculator

P = FourBandParams()


def _lattice(x, *, identity=False):
    reference = FourBandModel(P.at(x)) if identity else P.at(x)
    return NativePlaneModel(lattice_regularized_fourband(reference, 4.33), band_pair=(1, 2))


def test_critical_ddp_exponent_is_the_dirac_value():
    p = P.at(1.0)
    for t in (1e-13, 1e-10, 1e-8, 1e-6):
        ((z, exponent),) = fourband_ddp_t(p, t)
        assert z.imag == pytest.approx(t, rel=1e-9)
        assert exponent == pytest.approx(math.pi * p.hbar_v * t**2 * 1e10, rel=1e-6)
    assert fourband_ddp_t(p, 0.0) == [(0j, 0.0)]


def test_near_critical_discriminant_root_keeps_relative_precision():
    p = P.at(1.0 + 1e-10)
    root = min(fourband_branch_points_k2(p), key=abs)
    d0 = (abs(p.M) - p.R) * (abs(p.M) + p.R)
    assert root.real == pytest.approx(-(d0**2) / (4 * p.A**2 * p.Bl**2), rel=1e-8)
    assert root.imag == 0


@pytest.mark.precision
def test_critical_crossing_scan_keeps_touching_root():
    pytest.importorskip("mpmath")
    result = crossing_scan(FourBandModel(P.at(1.0)), [0.0], options=PHYSICAL)
    assert result["position"].size == 1
    assert result["selected_Ec"][0] == 0
    assert "touching" in result.metadata["line_flags"][0]


@pytest.mark.parametrize("x,t", [(3.0, 0.012), (4.0, 0.0), (2.5, 0.015), (1.5, 0.004), (1.2, 0.0)])
def test_generic_branch_points_follow_dispersing_band_centres(x, t):
    lc = find_crossings(
        ModelGeometry(_lattice(x, identity=True)), t, PHYSICAL.crossings, want_ddp=True, e_cut=2e6
    )
    exact = fourband_ddp_t(P.at(x), t)
    assert lc.n == len(exact)
    assert "branch_point_missing" not in lc.flags
    for crossing, (root, exponent) in zip(
        sorted(lc.crossings, key=lambda c: c.s), sorted(exact, key=lambda r: r[0].real)
    ):
        assert crossing.ddp_source == "model"
        assert abs(crossing.branch_point - root) < 2e-3 * abs(root)
        assert crossing.Ec_ddp == pytest.approx(exponent, rel=3e-3)


def test_generic_route_keeps_touching_minima():
    provider = ModelGeometry(_lattice(1.0))
    for t in (0.0, 1e-13):
        lc = find_crossings(provider, t, PHYSICAL.crossings, want_ddp=True, e_cut=2e6)
        assert lc.n == 1
        assert lc.crossings[0].touching
        assert lc.crossings[0].Ec_ddp == 0


def test_coalescing_vertical_partner_is_flagged():
    flags = {
        x: find_crossings(
            ModelGeometry(FourBandModel(P.at(x))), 0.0, PHYSICAL.crossings, want_ddp=True, e_cut=2e6
        ).flags
        for x in (0.9, 1.10, 1.117)
    }
    assert "not_separated" in flags[1.117]
    assert "not_separated" not in flags[1.10]
    assert "not_separated" not in flags[0.9]


def test_grid_branch_points_survive_stability_check_and_report_fit():
    model = FourBandModel(P.at(1.5))
    grid = GriddedGeometry.sample(
        ModelGeometry(model),
        r=np.geomspace(1e-6, 0.03, 160),
        phi=np.linspace(0, 2 * np.pi, 64, endpoint=False),
    )
    cfg = PAPER.with_(**{"conventions.ec_method": "ddp", "crossings.definition": "branch_point"})
    for t in (0.0, 0.004, 0.008):
        result = crossing_scan(grid, [t], options=cfg)
        assert "branch_point_missing" not in result.metadata["line_flags"][0]
        assert "branch_point_unvalidated" in result.metadata["line_flags"][0]
        assert set(result.metadata["ddp_source"]) == {"fit"}
        exact = fourband_ddp_t(P.at(1.5), t)[0][1]
        assert np.all(np.abs(result["selected_Ec"] / exact - 1) < 1e-2)


def test_branch_layout_breaks_at_ring_without_cutoff_panel():
    cfg = PAPER.with_(**{"conventions.ec_method": "ddp", "crossings.definition": "branch_point"})
    layout = HallCalculator(FourBandModel(P.at(2.0)), cfg, use_numba=False).transverse_layout()
    assert np.any(np.abs(np.abs(layout.breaks) - 4.578e-3) < 1e-6)
    layout = HallCalculator(FourBandModel(P.at(0.9)), PHYSICAL, use_numba=False).transverse_layout()
    assert layout.breaks.size == 3


def test_missing_diabatic_projector_uses_flagged_gap_minima():
    grid = GriddedGeometry.sample(
        ModelGeometry(FourBandModel(P.at(0.5))),
        r=np.geomspace(1e-6, 0.03, 80),
        phi=np.linspace(0, 2 * np.pi, 24, endpoint=False),
    )
    result = crossing_scan(grid, [0.0], options="ext7a")
    assert result["position"].size == 1
    assert "diabatic_unavailable" in result.metadata["line_flags"][0]
