"""Checked family scans and isolated-member submission/aggregation."""

import json
from pathlib import Path

import numpy as np

from ..finite_field.assembly import stack_member_results
from ..io.artifacts import RunStore, owned_run
from ..io.results import read_arrays
from ..observables.result import ObservableResult
from .finite_field_specs import _options


def _harmonic_map(config, source, store):
    from .finite_field import _commit_result, _prepare, _read, _restore_result

    params = config.parameters
    if "family_manifest" in config.inputs:
        from ..finite_field.api import resolve_options

        options = resolve_options(_options(params))
        coordinates, labels = np.asarray(source.D), source.labels
        count = len(source.members)
    else:
        if config.inputs.get("model") != "fourband":
            raise ValueError("model_values scans currently require an analytical fourband model")
        coordinates = np.asarray(params["model_values"], float)
        parameter = params.get("model_parameter", "x")
        labels = [str(value) for value in coordinates]
        count = len(coordinates)
    results = []
    for index in range(count):
        case = f"members/{index:06d}/Fourier"
        if store.completed(case):
            arrays, metadata = _read(store, case)
            result = _restore_result(arrays, metadata)
        else:
            item = (
                source.source(index, angle=options.drive.angle)
                if "family_manifest" in config.inputs
                else _analytical_member(config, parameter, coordinates[index])
            )
            model = (
                item()
                if callable(item) and not hasattr(item, "hamiltonian") and not hasattr(item, "geometry")
                else item
            )
            prepared = _prepare(config, model, store, namespace=f"members/{index:06d}")
            result = prepared.harmonics(params["amplitudes"], n_max=params.get("n_max"))
            _commit_result(store, case, result)
        results.append(result)
    first = results[0]
    arrays, dims, units = stack_member_results(results, shared={"amplitudes", "harmonics", "orders"})
    arrays["parameter"] = coordinates
    arrays["labels"] = np.asarray(labels)
    dims.update(parameter=("member",), labels=("member",))
    units.update(parameter="1", labels="1")
    normalization = params.get("normalize_by_Dc", False)
    critical = None
    if normalization is not False:
        if "family_manifest" not in config.inputs and normalization is True:
            raise ValueError("normalize_by_Dc=true requires a family; supply a numerical critical field")
        critical = source.critical_D() if normalization is True else float(normalization)
        if critical is None or not np.isfinite(critical) or critical == 0:
            raise ValueError("A unique finite nonzero critical field is required for D/Dc normalization")
        arrays["parameter"] = coordinates / critical
    return ObservableResult(
        arrays,
        dims,
        units,
        {
            **first.metadata,
            "observable": "harmonic_map",
            "labels": list(labels),
            "normalization": normalization,
            "critical_field": critical,
            "critical_field_source": "family_tracking"
            if normalization is True
            else "explicit"
            if normalization is not False
            else None,
            "member_count": len(results),
            "members": [result.metadata for result in results],
        },
    )


def _analytical_member(config, parameter, value):
    from dataclasses import replace

    from .finite_field_sources import load_finite_field_source

    model = dict(config.parameters.get("model", {}))
    if parameter != "x":
        model.pop("x", None)
    model[parameter] = float(value)
    member = replace(config, parameters={**config.parameters, "model": model})
    return load_finite_field_source(member)


def _crossing_map(config, source, store):
    from ..finite_field import crossing_scan, resolve_options
    from .finite_field import _commit_result, _kwargs, _read, _restore_result

    params = config.parameters
    family = "family_manifest" in config.inputs
    coordinates = np.asarray(source.D if family else params["model_values"], float)
    labels = source.labels if family else [str(value) for value in coordinates]
    angle = resolve_options(_options(params)).drive.angle
    rows = []
    for index, value in enumerate(coordinates):
        case = f"members/{index:06d}/crossings"
        if store.completed(case):
            arrays, metadata = _read(store, case)
            row = _restore_result(arrays, metadata)
        else:
            item = (
                source.source(index, angle=angle)
                if family
                else _analytical_member(config, params.get("model_parameter", "x"), value)
            )
            row = crossing_scan(item, params.get("offsets", [0.0]), **_kwargs(config, item))
            _commit_result(store, case, row)
        rows.append(row)
    arrays = {key: np.concatenate([row[key] for row in rows]) for key in rows[0].arrays}
    line_base = np.cumsum([0, *[len(row["offsets"]) for row in rows]])
    for key in ("line_index", "segment_line_index"):
        arrays[key] = np.concatenate([row[key] + line_base[index] for index, row in enumerate(rows)])
    arrays["parameter"] = coordinates
    arrays["labels"] = np.asarray(labels)
    arrays["member_index"] = np.concatenate(
        [np.full(len(row["position"]), index, int) for index, row in enumerate(rows)]
    )
    arrays["line_member_index"] = np.concatenate(
        [np.full(len(row["offsets"]), index, int) for index, row in enumerate(rows)]
    )
    arrays["segment_member_index"] = np.concatenate(
        [np.full(len(row["segment_start"]), index, int) for index, row in enumerate(rows)]
    )
    dims = {**rows[0].dims, "parameter": ("member",), "labels": ("member",), "member_index": ("crossing",)}
    units = {**rows[0].units, "parameter": "D" if family else "1", "labels": "1", "member_index": "index"}
    dims.update(line_member_index=("line",), segment_member_index=("segment",))
    units.update(line_member_index="index", segment_member_index="index")
    return ObservableResult(
        arrays, dims, units, {"observable": "crossing_map", "members": [row.metadata for row in rows]}
    )


def split_harmonic_map(config):
    """Resolve isolated child workflows without retaining loaded family models."""
    from ..config import config_from_mapping
    from .finite_field_sources import load_finite_field_source

    if config.workflow != "harmonic_map":
        raise ValueError("--split-members requires a harmonic_map configuration")
    params = dict(config.parameters)
    params.pop("model_values", None)
    parameter = params.pop("model_parameter", "x")
    normalization = params.pop("normalize_by_Dc", False)
    family = load_finite_field_source(config) if "family_manifest" in config.inputs else None
    coordinates = np.asarray(family.D if family is not None else config.parameters["model_values"], float)
    labels = family.labels if family is not None else [str(value) for value in coordinates]
    critical = (
        family.critical_D()
        if normalization is True and family is not None
        else float(normalization)
        if normalization is not False
        else None
    )
    if normalization is True and family is None:
        raise ValueError("normalize_by_Dc=true requires validated family tracking")
    if critical is not None and (not np.isfinite(critical) or critical == 0):
        raise ValueError("Critical-field normalization requires a finite nonzero value")
    children = []
    for index, value in enumerate(coordinates):
        child = config.as_dict()
        child["workflow"] = "hall_harmonics"
        child_params = dict(params)
        if family is not None:
            member = family.members[index]
            if "grid" in member.files:
                child_inputs = {"geometry_grid": member.files["grid"]}
            else:
                loader = member.model_loader or member.loader
                for key in ("ef", "band_pair", "occupied_bands", "lattice"):
                    child_params.pop(key, None)
                child_inputs = {
                    "model": "openmx" if "scfout" in member.files else "wannier",
                    "material_dir": str(Path(next(iter(member.files.values()))).parent),
                    **loader.files,
                }
                if child_inputs["model"] == "wannier":
                    if "tb" in child_inputs and "hr" not in child_inputs:
                        child_inputs["hr"] = child_inputs["tb"]
                    child_params["interpolation"] = "ws_distance" if "wsvec" in loader.files else "standard"
                child_params["plane"] = {
                    key: getattr(loader.plane, key) for key in ("origin", "direction", "transverse")
                }
                if loader.band_pair is not None:
                    child_params["band_pair"] = list(loader.band_pair)
                elif loader.n_occ is not None:
                    child_params["occupied_bands"] = loader.n_occ
                if getattr(loader, "lattice", None) is not None:
                    child_params["lattice"] = np.asarray(loader.lattice).tolist()
        else:
            child_inputs = dict(config.inputs)
            model = dict(params.get("model", {}))
            if parameter != "x":
                model.pop("x", None)
            model[parameter] = float(value)
            child_params["model"] = model
        child["inputs"], child["parameters"] = child_inputs, child_params
        children.append(config_from_mapping(child))
    return children, {
        "config": config.as_dict(),
        "parameters": coordinates.tolist(),
        "labels": list(labels),
        "normalization": normalization,
        "critical_field": critical,
    }


def aggregate_member_runs(dispatch_path, *, run_dir=None):
    """Aggregate completed isolated members after checking inputs, schemas, and checksums."""
    from ..config import config_from_mapping
    from ..io.artifacts import canonical_json, digest_file, load_manifest
    from .finite_field import _commit_result, _restore_result
    from .finite_field_sources import finite_field_input_files

    path = Path(dispatch_path).expanduser().resolve()
    dispatch = json.loads(path.read_text())
    if "aggregation" not in dispatch:
        raise ValueError("Dispatch has no split-member harmonic-map aggregation contract")
    specification = dispatch["aggregation"]
    config = config_from_mapping(specification["config"])
    coordinates, labels = specification["parameters"], specification["labels"]
    if len(dispatch["cases"]) != len(coordinates) or len(labels) != len(coordinates):
        raise ValueError("Member dispatch count does not match the parent map")
    files = [path, *finite_field_input_files(config)]
    results: list[ObservableResult] = []
    for index, case in enumerate(dispatch["cases"]):
        if digest_file(Path(case["config"])) != case["config_sha256"]:
            raise ValueError("Submitted member configuration changed")
        for item in case.get("inputs", []):
            if digest_file(Path(item["path"])) != item["sha256"]:
                raise ValueError("Submitted scientific input changed")
        root = Path(dispatch["run_root"]) / f"task-{index:04d}"
        manifest = load_manifest(root)
        expected = case["resolved_config"]
        actual = manifest["config"]
        if (
            actual["workflow"] != "hall_harmonics"
            or canonical_json(actual["parameters"]) != canonical_json(expected["parameters"])
            or canonical_json(actual["inputs"]) != canonical_json(expected["inputs"])
        ):
            raise ValueError("Member run scientific configuration differs from dispatch")
        record = manifest["cases"]["result"]
        product = root / record["path"]
        files.extend([root / "manifest.json", product])
        arrays = read_arrays(product)
        metadata = record["metadata"]
        result = _restore_result(arrays, metadata)
        results.append(result)
    first = results[0]
    arrays, dims, units = stack_member_results(results, shared={"amplitudes", "orders"})
    critical = specification.get("critical_field")
    arrays["parameter"] = (
        np.asarray(coordinates, float) / critical if critical is not None else np.asarray(coordinates, float)
    )
    arrays["labels"] = np.asarray(labels)
    dims.update(parameter=("member",), labels=("member",))
    result = ObservableResult(
        arrays,
        dims,
        {**units, "parameter": "1", "labels": "1"},
        {
            **first.metadata,
            "observable": "harmonic_map",
            "aggregation": "isolated checked member runs",
            "source_sha256": first.metadata.get("source_sha256"),
            "members": [row.metadata for row in results],
            "critical_field": critical,
            "critical_field_source": "family_tracking"
            if specification["normalization"] is True
            else "explicit"
            if critical is not None
            else None,
            "normalization": specification["normalization"],
        },
    )
    target = Path(run_dir).expanduser().resolve() if run_dir else Path(dispatch["run_root"]) / "map"
    store = RunStore(target, config.as_dict(), files, resume=(target / "manifest.json").exists())
    with owned_run(store):
        if not store.completed("result"):
            _commit_result(store, "result", result)
    return target
