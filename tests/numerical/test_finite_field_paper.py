"""Pinned digitized paper curves with independent native reproduction settings."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from queer.finite_field.analysis.paper import compare_maps
from queer.finite_field.figures import compute_figure

DATA = Path(__file__).resolve().parents[1] / "fixtures/finite_field/paper_targets.json"
pytestmark = pytest.mark.paper


def test_digitized_reference_fingerprint_and_central_line_curves():
    targets = json.loads(DATA.read_text())
    source = json.loads((DATA.parent / "reference_sources.json").read_text())
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == source["paper_targets_sha256"]
    for panel, key, tolerance in (
        ("ext7e_left", "Ec_gap_minimum", 0.08),
        ("ext7e_right", "Lambda_diabatic", 0.05),
    ):
        curve = targets[panel]
        x, y = np.asarray(curve["x"])[::19], np.asarray(curve["y"])[::19]
        mask = np.abs(x - 1) > 0.07
        result = compute_figure("ext7e", xs=x[mask])
        difference = abs(result[key] / 1e3 - y[mask])
        assert np.all(difference < np.maximum(tolerance, curve.get("tolerance_rel", 0) * y[mask]))


def test_critical_paper_harmonic_normalization():
    pytest.importorskip("mpmath")
    targets = json.loads(DATA.read_text())
    fields = np.array([0.01, 1.0, 100.0])
    result = compute_figure("ext9b", scaled_amplitudes=fields, orders=[3, 11])
    for index, order in enumerate(result["orders"]):
        curve = next(c for c in targets["ext9b"]["curves"] if c["N"] == order)
        expected = np.interp(np.log10(fields), np.log10(curve["x"]), np.log10(curve["y"]))
        assert (
            np.max(abs(np.log10(abs(result["scaled_sine"][index])) - expected))
            < targets["ext9b"]["tolerance_log10"]
        )


def test_digitized_map_comparison_keeps_missing_diagnostics_null():
    paper = {"x": [1.1, 1.2], "N": [3], "maps": {3: np.array([[1.0, -2.0]])}, "clim": {3: 2.0}}
    result = compare_maps({3: np.array([[2.0, np.nan]])}, paper)
    window = result["harmonics"]["3/window"]
    assert window["pixels"] == 0 and window["invalid_model_pixels"] == 1
    assert window["sign_agreement"] is None and window["median_ratio"] is None
