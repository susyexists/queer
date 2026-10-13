"""Hall quadrature, field limits, and parallel preparation acceptance."""

import math
from dataclasses import replace

import numpy as np
import pytest

from queer.finite_field import FourBandModel, FourBandParams, NativePlaneModel
from queer.finite_field.config import PAPER, PHYSICAL
from queer.finite_field.models.lattice_reg import lattice_regularized_fourband
from queer.finite_field.transport.hall import HallCalculator, _worker_provider, twisted_probability

P = FourBandParams()


def test_twisted_probability_keeps_limits_and_fades_strong_enhancement():
    population = np.array([0, 1e-12, 0.2, 0.7, 1.0])
    for exponent in (-3.0, -1.5, -0.5, -0.01, 0.5, 3.0):
        actual = twisted_probability(population, exponent)
        assert np.all((actual >= 0) & (actual <= 1))
        magnitude = max(-exponent, 0)
        low_field = -exponent if exponent >= 0 else magnitude / (1 + (magnitude / 1.35) ** 8)
        assert actual[1] == pytest.approx(1e-12 * math.exp(low_field), rel=1e-9)
    assert twisted_probability(1e-12, -0.5) / 1e-12 == pytest.approx(math.exp(0.5), rel=2e-3)
    assert twisted_probability(1e-12, -3) / 1e-12 < 1.01
    np.testing.assert_allclose(twisted_probability(population, 0), population)
    np.testing.assert_allclose(twisted_probability(population, 0.5), population * math.exp(-0.5))


@pytest.mark.parametrize("cfg", [PAPER, PHYSICAL])
def test_no_tunnelling_is_flagged_and_returns_zero(cfg):
    model = FourBandModel(replace(P.at(0.5), M=-0.02))
    calculator = HallCalculator(model, cfg, use_numba=False).prepare()
    assert "no_tunnelling" in calculator.diagnostics()["layout_flags"]
    assert calculator.layout.t_max[1] < 1e-6
    np.testing.assert_array_equal(calculator.J([-1e4, 0, 1e4]), 0)


def test_fields_above_prepared_extent_including_infinity_raise():
    calculator = HallCalculator(FourBandModel(P.at(0.5)), PAPER, use_numba=False).prepare()
    for field in (50001, -50001, np.inf, -np.inf):
        with pytest.raises(ValueError, match="E_max"):
            calculator.J([field])
    response = calculator.J([np.nan, 0, 1e4])
    assert np.isnan(response[0]) and response[1] == 0 and np.isfinite(response[2])


def test_interference_flag_only_where_weighted_aliasing_matters():
    model = FourBandModel(P.at(1.8))
    base = PAPER.with_(use_symmetry_shortcuts=True)
    cfg = base.with_(**{"relaxation.T1": math.inf, "relaxation.T2": math.inf})
    coherent = HallCalculator(model, cfg, use_numba=False).prepare()
    coherent.J([3e4])
    assert "interference_under_resolved" not in coherent.response_flags
    coherent.J([1e3])
    assert "interference_under_resolved" in coherent.response_flags
    damped = HallCalculator(model, base, use_numba=False).prepare()
    damped.J(np.linspace(1e3, 5e4, 50))
    assert "interference_under_resolved" not in damped.response_flags


@pytest.mark.parametrize("native", [False, True])
def test_parallel_layout_and_lines_match_serial_with_parent_checkpoints(native, monkeypatch):
    from queer.finite_field.transport import hall

    def reject_batch(args):
        pytest.fail("Light models must dispatch one line per task")

    monkeypatch.setattr(hall, "_solve_lines_task", reject_batch)
    model = FourBandModel(P.at(1.8))
    cfg = PAPER
    if native:
        model = NativePlaneModel(lattice_regularized_fourband(P.at(1.8), 4.33), band_pair=(1, 2))
        cfg = cfg.with_(
            **{
                "crossings.scan_points": 129,
                "quadrature.du": 2.0,
                "quadrature.dmin": 1e-7,
                "quadrature.t_max": 0.0002,
            }
        )
    serial = HallCalculator(model, cfg, t_scan=16, use_numba=False).prepare(1)

    class ParallelLayout(HallCalculator):
        def _scan_sides_parallel(self, *args, **kwargs):
            self.parallel_layout_called = True
            return super()._scan_sides_parallel(*args, **kwargs)

    saved = {}
    parallel = ParallelLayout(model, cfg, t_scan=16, use_numba=False).prepare(
        3,
        save_line=lambda i, t, line: saved.setdefault(i, (t, line)),
    )
    from queer.execution import cpu_budget

    if cpu_budget() > 1:
        assert parallel.parallel_layout_called
    assert saved.keys() == set(range(len(parallel.lines)))
    assert serial.layout.mirrored == parallel.layout.mirrored
    assert serial.layout.flags == parallel.layout.flags
    np.testing.assert_allclose(serial.layout.t_max, parallel.layout.t_max, rtol=1e-9, atol=0)
    np.testing.assert_allclose(
        serial.layout.breaks, parallel.layout.breaks, rtol=0, atol=1e-9 * cfg.crossings.window
    )
    assert serial.layout.t_nodes.size == parallel.layout.t_nodes.size
    fields = [-2e4, 1e3, 5e3, 2e4, 4e4]
    a, b = serial.J(fields), parallel.J(fields)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-8 * np.max(np.abs(a)))


def test_native_worker_view_keeps_operator_arrays_and_excludes_raw_records():
    from queer.finite_field.geometry.provider import ModelGeometry

    model = lattice_regularized_fourband(P.at(1.0), 4.33)
    model.raw_data = {"unused": np.zeros(1000)}
    native = NativePlaneModel(model, band_pair=(1, 2))
    provider = ModelGeometry(native)
    worker = _worker_provider(provider)
    assert worker is not provider
    assert worker.model is worker.source_model
    assert worker.model.source_model is not model
    assert worker.model.source_model.raw_data is None
    assert model.raw_data is not None
    assert worker.model.source_model.hopping_matrices is model.hopping_matrices
    assert worker.patch._model is worker.model
    assert provider.patch._model is native
    points = np.array([1e-7, 1e-5, 1e-3])
    np.testing.assert_array_equal(worker.gap(points, points * 0), provider.gap(points, points * 0))
