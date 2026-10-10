"""Implicit OpenMX spin copies scale currents while preserving pair physics."""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from queer import OpenMXModel
from queer.finite_field import (
    PAPER,
    FourBandModel,
    FourBandParams,
    GriddedGeometry,
    ModelGeometry,
    NativePlaneModel,
    PreparedHallResponse,
    critical_hall,
    resolve_options,
)
from queer.finite_field.transport.reference import naive_jh


@pytest.fixture(scope="module")
def spin_responses():
    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.diag([1, -1])
    translations = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    hopping = 1e-3 * np.array([0.3 * sz, sx / (2j), -sx / (2j), sy / (2j), -sy / (2j)])
    overlap = np.zeros_like(hopping)
    overlap[0] = np.eye(2)
    models = [
        OpenMXModel(
            translations,
            hopping,
            overlap,
            lattice=np.diag([2, 3, 4]),
            position=np.zeros((3, *hopping.shape)),
            **({"spin_degeneracy": 1} if explicit else {}),
        )
        for explicit in (True, False)
    ]
    options = resolve_options(
        {
            "preset": "physical_eq8",
            "drive": {"E_min": 1e5, "E_max": 1e7},
            "crossings": {"window": 0.2, "scan_points": 65},
            "quadrature": {"t_max": 0.02, "du": 1.0, "dmin": 1e-4},
            "fourier": {"n_theta": 32, "n_cap": 128, "tol": 1e-6},
        }
    )
    views = [NativePlaneModel(model, band_pair=(0, 1)) for model in models]
    first = PreparedHallResponse(views[0], options, keep_nodes=True).prepare()
    # The geometric tables are per pair and therefore identical for both spins.
    second = PreparedHallResponse(views[1], options, keep_nodes=True).prepare(
        load_layout=lambda: first.layout, load_line=lambda index, offset: first.lines[index]
    )
    return first, second


def test_native_implicit_spin_copies_scale_current_and_direct_reference(spin_responses):
    first, second = spin_responses
    fields = np.array([-4e6, 0.0, 4e6])
    single = first.hall(fields)
    doubled = second.hall(fields)
    assert abs(single["current"][-1]) > 1e-8
    assert_allclose(doubled["current"], 2 * single["current"], atol=0, rtol=1e-14)
    assert_allclose(
        second.calculator.J_direct(fields), 2 * first.calculator.J_direct(fields), atol=0, rtol=1e-14
    )
    assert single.metadata["spin_degeneracy"] == 1
    assert doubled.metadata["spin_degeneracy"] == 2
    assert first.provider.describe()["spin_degeneracy"] == 1
    assert second.provider.describe()["spin_degeneracy"] == 2
    assert first.signature() != second.signature()
    a, b = [response.provider.geometry([0.01], [0.005]) for response in spin_responses]
    for name in ("eps_g", "omega_c", "omega_v", "g_cv"):
        assert_allclose(getattr(a, name), getattr(b, name), atol=0, rtol=1e-14)
    line = next(line for line in first.lines if line.n)
    assert_allclose(
        first.calculator.line_populations(line, [4e6], 1),
        second.calculator.line_populations(line, [4e6], 1),
        atol=0,
        rtol=0,
    )


def test_native_implicit_spin_copies_scale_all_harmonic_components(spin_responses):
    single, doubled = [response.harmonics([4e6], n_max=3) for response in spin_responses]
    assert np.max(np.abs(single["b"])) > 1e-8
    for key in ("a", "b", "c", "c0"):
        assert_allclose(doubled[key], 2 * single[key], atol=0, rtol=1e-14)
    assert np.array_equal(single["converged"], doubled["converged"])


def test_optional_compiled_current_keeps_implicit_spin_normalization(spin_responses):
    pytest.importorskip("numba")
    _, doubled = spin_responses
    accelerated = PreparedHallResponse(doubled.model, doubled.options, keep_nodes=True, acceleration=True)
    accelerated.prepare(
        load_layout=lambda: doubled.layout, load_line=lambda index, offset: doubled.lines[index]
    )
    fields = [-4e6, 0.0, 4e6]
    assert accelerated.describe()["spin_degeneracy"] == 2
    assert_allclose(accelerated.J(fields), doubled.J(fields), atol=1e-13, rtol=1e-12)


def test_sampled_grid_archives_preserve_spin_current_normalization(spin_responses, tmp_path):
    _, doubled = spin_responses
    axes = np.linspace(-0.2, 0.2, 17)
    sampled = GriddedGeometry.sample(doubled.provider, kx=axes, ky=axes)
    archive = sampled.save(tmp_path / "spin_grid.npz")
    restored = GriddedGeometry.from_npz(archive)
    assert restored.spin_degeneracy == 2
    assert restored.describe()["spin_degeneracy"] == 2
    with np.load(archive, allow_pickle=False) as arrays:
        assert json.loads(str(arrays["__meta__"]))["spin_degeneracy"] == 2
    # Arrays without a multiplicity declaration retain the historical single-pair meaning.
    legacy = GriddedGeometry(sampled._raw)
    assert legacy.spin_degeneracy == 1
    assert legacy.meta["spin_degeneracy_source"] == "legacy_default_single_pair"
    one = PreparedHallResponse(legacy, doubled.options, keep_nodes=True).prepare()
    two = PreparedHallResponse(restored, doubled.options, keep_nodes=True).prepare(
        load_layout=lambda: one.layout, load_line=lambda index, offset: one.lines[index]
    )
    fields = [-4e6, 0.0, 4e6]
    assert np.max(np.abs(one.J(fields))) > 1e-8
    assert_allclose(two.J(fields), 2 * one.J(fields), atol=0, rtol=1e-14)
    assert one.signature() != two.signature()
    assert_allclose(
        two.harmonics([4e6], n_max=3)["b"], 2 * one.harmonics([4e6], n_max=3)["b"], atol=0, rtol=1e-14
    )
    with pytest.raises(ValueError, match="conflicts"):
        GriddedGeometry.sample(doubled.provider, kx=axes, ky=axes, meta={"spin_degeneracy": 1})


@pytest.mark.parametrize("factor", [0, 3, True, 1.5, float("nan"), None])
def test_geometry_archive_rejects_invalid_spin_normalization(factor):
    # Metadata is checked before interpolation; malformed counts cannot silently normalize currents.
    with pytest.raises(ValueError, match="spin_degeneracy"):
        GriddedGeometry({}, meta={"spin_degeneracy": factor})


def test_nested_independent_reference_preserves_spin_count_through_traceless_copy():
    model = FourBandModel(FourBandParams().at(1.4))
    single = naive_jh(model, 1e4, PAPER)
    model.spin_degeneracy = 2
    assert not hasattr(model.traceless(), "spin_degeneracy")
    doubled = naive_jh(model, 1e4, PAPER)
    assert abs(single) > 1e-8
    assert doubled == pytest.approx(2 * single, rel=1e-13)
    assert ModelGeometry(model).spin_degeneracy == 2


@pytest.mark.precision
def test_critical_currents_scale_spin_copies_without_changing_pair_constants():
    pytest.importorskip("mpmath")
    model = FourBandModel(FourBandParams().at(1.0))
    single = critical_hall(model, [1.0, 100.0], options=PAPER)
    model.spin_degeneracy = 2
    doubled = critical_hall(model, [1.0, 100.0], options=PAPER)
    for key in ("current", "log_corrected_current"):
        assert_allclose(doubled[key], 2 * single[key], atol=0, rtol=1e-13)
    assert doubled.metadata["constants"] == single.metadata["constants"]
    assert doubled.metadata["spin_degeneracy"] == 2
