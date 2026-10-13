"""Numerical figure identities and rendering that cannot load band models."""

import numpy as np
import pytest

from queer.finite_field.figures import compute_figure


def test_paper_single_and_double_passage_figure_identities():
    fields = [1000.0, 1e4, 2e4]
    single = compute_figure("fig4b", fields=fields)
    assert single.metadata["critical_field"] == pytest.approx(1284.5765, rel=1e-7)
    assert np.array_equal(
        single["probability"], np.exp(-single.metadata["critical_field"] / single["fields"])
    )
    double = compute_figure("fig4d", fields=fields)
    ec, frequency = double.metadata["critical_field"], double.metadata["interference_frequency"]
    assert ec == pytest.approx(15757.27729246, rel=1e-9)
    assert frequency == pytest.approx(12138.99112985, rel=1e-9)
    assert double.metadata["options"]["crossings"]["definition"] == "diabatic"
    # The first three fringe zeros pin the recovered displacement/crossing recipe.
    assert np.allclose(2 * frequency / np.array([1, 3, 5]), [24278, 8093, 4856], atol=2)
    probability = np.exp(-ec / double["fields"])
    assert np.allclose(
        double["probability"],
        2 * probability * (1 - probability) * (1 + np.cos(2 * np.pi * frequency / double["fields"])),
    )
    assert np.all(double["probability"] >= 0)
    assert np.all(double["probability"] <= double["envelope"])


def test_critical_geometry_figure_and_artifact_only_render(tmp_path, monkeypatch):
    pytest.importorskip("mpmath")
    result = compute_figure("fig6bd", ky=[1e-7, 1e-6, 1e-5], xs=[0.9, 1.1])
    assert np.allclose(result["curvature"][:2], result["critical_reference"][:2], rtol=1e-4)
    monkeypatch.setattr("queer.finite_field.figures._model", lambda *args: pytest.fail("Model loaded"))
    from queer.plotting.paper_figures import render_paper_figure

    assert render_paper_figure(result.arrays, result.metadata, tmp_path) == ["fig6bd.png"]
    assert (tmp_path / "fig6bd.png").is_file()


def test_band_geometry_figure_preserves_absolute_identity_dispersion():
    pytest.importorskip("mpmath")
    result = compute_figure("fig5ab", xs=[0.5, 1.0], k=[-0.001, 0, 0.001], kmap=[-0.001, 0, 0.001])
    assert result["energies"].shape == (2, 3, 4)
    assert result["pair_metric"].shape == (2, 3)
    assert result["valid"].shape == (2, 3, 3)
    assert not result["valid"][1, 1, 1]
    assert not np.allclose(result["energies"].sum(axis=-1), 0)
