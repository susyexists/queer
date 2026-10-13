"""Gamma interpolation and precision certificates from the current finite-field reference."""

from contextlib import contextmanager

import numpy as np
import pytest

from queer.finite_field.config import PAPER
from queer.finite_field.geometry.gridded import GriddedGeometry
from queer.finite_field.geometry.lowdin import LowdinPatch
from queer.finite_field.geometry.provider import ModelGeometry
from queer.finite_field.models import FourBandModel, FourBandParams
from queer.finite_field.models.lattice_reg import lattice_regularized_fourband
from queer.finite_field.native import NativePlaneModel
from queer.finite_field.transport.hall import HallCalculator
from queer.models.wannier import WannierModel


@pytest.mark.parametrize("x", [1.0, 1 + 1e-7, 1 - 1e-7])
def test_gamma_anchor_resolves_signed_small_field_hall_core(x, tmp_path):
    pytest.importorskip("mpmath")
    provider = ModelGeometry(FourBandModel(FourBandParams().at(x)))
    r = np.r_[np.geomspace(1e-6, 2e-3, 120), np.linspace(2.1e-3, 0.045, 120)]
    grid = GriddedGeometry.sample(provider, r=r, phi=np.linspace(0, 2 * np.pi, 48, endpoint=False))
    assert grid.extrapolation_reliable and grid.describe()["gamma_anchor"]
    fields = np.array([-1e4, -1e2, -1.0, 0.0, 1.0, 1e2, 1e4])
    reference = HallCalculator(provider=provider, cfg=PAPER).prepare().J(fields)
    actual = HallCalculator(provider=grid, cfg=PAPER).prepare()
    np.testing.assert_allclose(actual.J(fields), reference, rtol=2e-6, atol=0)
    assert "outside_grid" not in actual.diagnostics()["line_flags"]
    path = grid.save(tmp_path / "anchored.npz")
    restored = GriddedGeometry.from_npz(path)
    assert restored.fingerprint() == grid.fingerprint()
    np.testing.assert_array_equal(restored.gap([1e-10, 1e-7], 0), grid.gap([1e-10, 1e-7], 0))


@pytest.mark.parametrize("x,r0,reliable", [(1 + 1e-7, 1e-5, False), (1 - 1e-6, 1e-5, True)])
def test_unanchored_quadratic_core_is_certified_against_cubic(x, r0, reliable):
    pytest.importorskip("mpmath")
    provider = ModelGeometry(FourBandModel(FourBandParams().at(x)))
    r = np.r_[np.geomspace(r0, 2e-3, 90), np.linspace(2.1e-3, 0.045, 120)]
    anchored = GriddedGeometry.sample(provider, r=r, phi=np.linspace(0, 2 * np.pi, 48, endpoint=False))
    bare = GriddedGeometry(
        {key: value for key, value in anchored._raw.items() if not key.startswith("gamma_")}
    )
    assert bare.extrapolation_reliable == reliable
    geometry = bare.geometry([r0 / 10, r0 * 2], 0)
    assert geometry.extras["outside_grid"].tolist() == [not reliable, False]


def test_gamma_validation_preserves_independent_gap_and_spin(tmp_path):
    model = FourBandModel(FourBandParams().at(0.5))
    provider = ModelGeometry(model)
    provider.spin_degeneracy = 2
    grid = GriddedGeometry.sample(provider, r=np.geomspace(1e-6, 0.03, 20), phi=np.arange(8) * np.pi / 4)
    raw = dict(grid._raw)
    raw["gamma_gap"] = raw["gamma_gap"] * 2
    with pytest.raises(ValueError, match="Gamma gap"):
        GriddedGeometry(raw)
    raw = dict(grid._raw)
    del raw["gamma_g_cv"]
    with pytest.raises(ValueError, match="Gamma geometry is missing"):
        GriddedGeometry(raw)
    restored = GriddedGeometry.from_npz(grid.save(tmp_path / "spin.npz"))
    assert restored.spin_degeneracy == 2
    assert "gamma_gap" in restored._raw


def _native_with_weak_partner():
    source = lattice_regularized_fourband(FourBandParams().at(1 + 1e-6))
    hopping = np.zeros((len(source.translations), 5, 5), complex)
    hopping[:, :4, :4] = source.hopping_matrices
    origin = np.flatnonzero(np.all(source.translations == 0, axis=1))[0]
    hopping[origin, 4, 4] = 2e-5
    model = WannierModel(
        source.translations, hopping, lattice=source.lattice, centres=np.vstack([source.centres, [[0, 0, 0]]])
    )
    return NativePlaneModel(model, band_pair=(1, 2))


def test_lowdin_per_band_selection_keeps_weak_partner_remote_and_zero_tolerance_pair():
    view = _native_with_weak_partner()
    patch = LowdinPatch(view, k_max=1e-5)
    assert patch.P.tolist() == [1, 2] and patch.exact_pair
    explicit = LowdinPatch(view, k_max=1e-5, p_tol=0)
    assert explicit.P.tolist() == [1, 2]
    k = np.geomspace(1e-12, 1e-6, 4)
    np.testing.assert_allclose(explicit.evaluate(k, k).eps_g, patch.evaluate(k, k).eps_g, rtol=1e-13)


def test_provider_reports_enlarged_lowdin_subspace_and_bad_overlap(monkeypatch):
    view = _native_with_weak_partner()
    enlarged = ModelGeometry(view, patch=True, k_patch=1e-4)
    assert not enlarged.patch.exact_pair
    assert enlarged.precision_floor["n_P"] == 3
    assert "float64 eigenvectors" in enlarged.precision_floor["reason"]
    monkeypatch.setattr(LowdinPatch, "overlap_error", lambda *args, **kwargs: 0.02)
    disagrees = ModelGeometry(view, patch=True, k_patch=1e-6)
    assert disagrees.precision_floor["patch_overlap"] == 0.02


def test_provider_line_context_only_for_constant_transverse_offset():
    model = FourBandModel(FourBandParams().at(0.5))
    calls = []

    @contextmanager
    def on_line(angle, offset):
        calls.append((angle, offset))
        yield

    model.on_line = on_line
    provider = ModelGeometry(model, angle=0.3, use_traceless=False, patch=False)
    provider.geometry([1e-3, 2e-3], [0.002, 0.002])
    provider.gap([1e-3, 2e-3], 0.002)
    provider.gap([1e-3, 2e-3], [0.002, 0.003])
    assert calls == [(0.3, 0.002), (0.3, 0.002)]
