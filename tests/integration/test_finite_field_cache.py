"""Reusable preparation identity, integrity and immutable checkpoint acceptance."""

import numpy as np
import pytest

from queer.finite_field import FourBandModel, FourBandParams, PreparedHallResponse, resolve_options


def settings():
    return resolve_options(
        {
            "preset": "physical_eq8",
            "drive": {"E_min": 1e3, "E_max": 3e4},
            "crossings": {"window": 0.002, "scan_points": 65},
            "quadrature": {"t_max": 0.0002, "du": 2.0, "dmin": 1e-7, "lam_per_decade": 4},
        }
    )


def test_reusable_cache_survives_fourier_changes_and_rejects_corruption(tmp_path, monkeypatch):
    model = FourBandModel(FourBandParams().at(0.5))
    prepared = PreparedHallResponse(model, settings(), cache_dir=tmp_path).prepare()
    expected = prepared.J([-1e4, 0, 2e4])
    paths = sorted(tmp_path.rglob("*.npz"))
    saved = {path: path.read_bytes() for path in paths}
    different_fourier = settings().with_(**{"fourier.n_max": 9})
    restored = PreparedHallResponse(model, different_fourier, cache_dir=tmp_path)
    assert restored.signature() == prepared.signature()
    monkeypatch.setattr(
        "queer.finite_field.transport.hall.solve_line",
        lambda *args, **kwargs: pytest.fail("Cached line recomputed"),
    )
    monkeypatch.setattr(
        "queer.finite_field.transport.hall.HallCalculator.transverse_layout",
        lambda *args, **kwargs: pytest.fail("Cached layout recomputed"),
    )
    assert np.array_equal(restored.J([-1e4, 0, 2e4]), expected)
    assert saved == {path: path.read_bytes() for path in paths}
    changed_relaxation = PreparedHallResponse(model, settings().with_(**{"relaxation.T1": 3e-12}))
    assert changed_relaxation.signature() != prepared.signature()
    paths[0].write_bytes(paths[0].read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum|integrity|corrupt"):
        PreparedHallResponse(model, settings(), cache_dir=tmp_path).prepare()
