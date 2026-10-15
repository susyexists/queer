"""Renderer dispatch is artifact-only and returns concrete paths."""

from pathlib import Path

import numpy as np
import pytest

from queer.workflows.catalog import renderer_for, renderer_reference


@pytest.mark.parametrize(
    "name,case,arrays,module",
    [
        (
            "bands",
            "bands",
            {"energies": [[-1, 0, 1]], "symmetry": [0, 2], "labels": ["G", "X"]},
            "legacy_artifacts",
        ),
        ("dos", "result", {"omega": [-1, 0, 1], "dos": [0, 2, 0]}, "equilibrium_artifacts"),
        ("nvt", "result", {"time_ps": [0, 1], "etot_eV": [-1, -1]}, "service_artifacts"),
        (
            "landau_zener_hall",
            "result",
            {"fields": [-1000, 0, 1000], "current": [-1, 0, 1]},
            "finite_field_artifacts",
        ),
    ],
)
def test_renderer_catalog_returns_paths_without_loading_models(
    tmp_path, monkeypatch, name, case, arrays, module
):
    import matplotlib

    from queer.models.wannier import WannierModel

    matplotlib.use("Agg")

    def prohibited(*args, **kwargs):
        raise AssertionError("Artifact renderers must not construct models or workflow configs")

    monkeypatch.setattr(WannierModel, "from_directory", prohibited)
    monkeypatch.setattr("queer.config.config_from_mapping", prohibited)
    np.savez(tmp_path / "result.npz", **arrays)
    output = tmp_path / "figures"
    output.mkdir()
    manifest = {
        "workflow": name,
        "cases": {case: {"path": "result.npz", "metadata": {}}},
    }
    assert renderer_reference(name) == f"queer.plotting.{module}:render"
    paths = renderer_for(name)(tmp_path, manifest, output)
    assert paths == [output / f"{name}.png"]
    assert all(isinstance(path, Path) and path.is_file() for path in paths)
