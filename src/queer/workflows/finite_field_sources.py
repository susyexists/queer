"""Resolve the transitive inputs and models of finite-field calculations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .finite_field_specs import _options

if TYPE_CHECKING:
    from ..config import WorkflowConfig


def _path(directory, value):
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else directory / candidate).resolve()


def finite_field_input_files(config: WorkflowConfig) -> list[Path]:
    """Hash every consumed file, including the leaves of a family manifest."""
    inputs, params = config.inputs, config.parameters
    if "family_manifest" in inputs:
        family_path = Path(inputs["family_manifest"])
        data = json.loads(family_path.read_text())
        members = data if isinstance(data, list) else data.get("members", data.get("models", []))
        files = [family_path]
        for member in members:
            leaves = member.get("files", {})
            kind = member.get("model", "openmx" if "scfout" in leaves else "wannier")
            resolved = {key: _path(family_path.parent, value) for key, value in leaves.items() if value}
            files.extend(resolved.values())
            model_files = {key: str(path) for key, path in resolved.items() if key != "grid"}
            if model_files:
                directory = Path(next(iter(model_files.values()))).parent
                family_params = {
                    **params,
                    "interpolation": "ws_distance" if "wsvec" in model_files else "standard",
                }
                family_inputs: dict = {"material_dir": directory, "model": kind, **model_files}
                if kind == "wannier" and "tb" in model_files and "hr" not in model_files:
                    # Family loaders select TB-only inputs explicitly, even when
                    # an unrelated default HR file exists beside the member.
                    family_inputs["hr"] = None
                files.extend(_model_files(family_inputs, family_params))
        return sorted(set(files))
    for key in ("geometry_grid", "response_artifact"):
        if key in inputs:
            source = Path(inputs[key])
            if source.is_dir():
                from ..io.artifacts import load_manifest

                manifest = load_manifest(source)
                return [
                    source / "manifest.json",
                    *[source / item["path"] for item in manifest["cases"].values()],
                ]
            return [source]
    return _model_files(inputs, params)


def _model_files(inputs, params):
    if inputs.get("model") == "fourband":
        return []
    from .sources import model_input_files

    return model_input_files(inputs, params)


def load_finite_field_source(config: WorkflowConfig):
    inputs, params = config.inputs, config.parameters
    if config.workflow == "fit_hall":
        return None
    if "geometry_grid" in inputs:
        from ..finite_field.geometry.gridded import GriddedGeometry

        path = Path(inputs["geometry_grid"])
        if path.is_dir():
            from ..io.artifacts import load_manifest

            manifest = load_manifest(path)
            path = path / manifest["cases"]["result"]["path"]
        from ..finite_field.config import resolve_options

        return GriddedGeometry.from_npz(path, angle=resolve_options(_options(params)).drive.angle)
    if "family_manifest" in inputs:
        from ..finite_field.family import ModelFamily

        return ModelFamily.from_manifest(
            inputs["family_manifest"],
            n_occ=params.get("occupied_bands"),
            plane=params.get("plane"),
            band_pair=params.get("band_pair"),
        )
    if inputs["model"] == "fourband":
        from ..finite_field.models.fourband import FourBandModel, FourBandParams

        values = dict(params.get("model", {}))
        ratio = values.pop("x", None)
        model_options = {
            key: values.pop(key)
            for key in (
                "warping",
                "warping_matrix",
                "d3_layer",
                "include_identity",
            )
            if key in values
        }
        parameters = FourBandParams(**values)
        if ratio is not None:
            parameters = parameters.at(float(ratio))
        return FourBandModel(parameters, **model_options)
    from .sources import load_electronic_model

    model = load_electronic_model(config, default_fermi=True)
    if model.g_vec is None and "lattice" in params:
        from ..geometry.lattice import reciprocal_lattice

        model.lattice = np.asarray(params["lattice"], float)
        model.g_vec = reciprocal_lattice(model.lattice)
    return model
