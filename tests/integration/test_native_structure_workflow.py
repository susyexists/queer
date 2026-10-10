"""Native structure preparation, durable records and Generator export contracts."""

from __future__ import annotations

import builtins
import json

import numpy as np
import pytest

from queer.compat.generator import export_run
from queer.config import config_from_mapping
from queer.io.artifacts import digest_file
from queer.io.atomistic import parse_poscar
from queer.io.records import structure_from_arrays
from queer.workflows import run_workflow

POSCAR = """Synthetic O H fixture
1
4 0 0
0 4 0
0 0 5
O H
1 1
Selective dynamics
Direct
0 0 0 F T T
0.25 0.25 0.2 T F T
"""


def structure_config(tmp_path, *, transforms=None, symmetry=False):
    material = tmp_path / "material"
    material.mkdir()
    (material / "POSCAR").write_text(POSCAR)
    return config_from_mapping(
        {
            "workflow": "structure",
            "inputs": {"material_dir": material, "structure": "POSCAR"},
            "parameters": {"transforms": transforms or [], "symmetry": symmetry},
            "execution": {"native_threads": 1},
            "output": {"results_dir": tmp_path / "results"},
        }
    )


def load_result(run):
    manifest = json.loads((run / "manifest.json").read_text())
    record = manifest["cases"]["result"]
    with np.load(run / record["path"], allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return manifest, record, arrays


def test_native_structure_transform_export_and_resume(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def primitive_only_import(name, *args, **kwargs):
        if name.split(".")[0] in {"ase", "seekpath", "Dans_Diffraction"}:
            raise AssertionError(f"Native workflow attempted atomistic framework import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", primitive_only_import)
    config = structure_config(
        tmp_path,
        transforms=[
            {"type": "strain", "params": {"mode": "uniaxial", "axis": "x", "percent": 10}},
            {"type": "layer", "params": {"vacuumAng": 2}},
            {"type": "supercell", "params": {"a": 2}},
            {"type": "vacancy", "params": {"fraction": 0.25}},
        ],
    )
    run = run_workflow(config)
    manifest, record, arrays = load_result(run)
    assert manifest["status"] == "completed"
    assert manifest["scientific_model"]["input_kind"] == "structure"
    assert [item["sha256"] for item in manifest["inputs"]] == [digest_file(config.material_dir / "POSCAR")]
    prepared = structure_from_arrays(arrays)
    np.testing.assert_allclose(prepared.cell, np.diag([8.8, 4, 7]))
    np.testing.assert_allclose(prepared.positions, [[0, 0, 1], [1.1, 1, 2], [4.4, 0, 1]])
    assert prepared.symbols == ["O", "H", "O"]
    assert prepared.metadata["source_indices"] == [0, 1, 0]
    assert prepared.metadata["constraint_basis"] == "fractional"
    np.testing.assert_array_equal(
        prepared.fixed_coordinates, [[True, False, False], [False, True, False], [True, False, False]]
    )
    assert record["metadata"]["units"]["positions"] == "Å"
    assert record["metadata"]["n_variants"] == 1

    exported = export_run(run)
    payload = json.loads((exported / "structure.json").read_text())
    assert payload["formula"] == "HO2"
    assert payload["natoms"] == 3
    assert payload["species"] == prepared.symbols
    assert payload["poscar_atom_order"] == [1, 0, 2]
    ordered = parse_poscar(payload["poscar"])
    np.testing.assert_allclose(ordered.positions, prepared.positions[[1, 0, 2]])
    np.testing.assert_array_equal(ordered.fixed_coordinates, prepared.fixed_coordinates[[1, 0, 2]])
    assert payload["latticeParams"]["c"] == 7
    assert payload["queer_provenance"]["source_fingerprint"] == manifest["fingerprint"]
    export_manifest = json.loads((exported / "manifest.json").read_text())
    products = {item["path"]: item["sha256"] for item in export_manifest["products"]}
    assert {"structure.json", "POSCAR"} <= products.keys()
    for filename, checksum in products.items():
        assert checksum == digest_file(exported / filename)
    np.testing.assert_allclose(parse_poscar((exported / "POSCAR").read_text()).positions, ordered.positions)

    artifact = run / record["path"]
    checksum, timestamp = digest_file(artifact), artifact.stat().st_mtime_ns

    def no_recalculation(*args, **kwargs):
        raise AssertionError("Completed structure workflow must resume without parsing")

    monkeypatch.setattr("queer.io.atomistic.read_structure", no_recalculation)
    run_workflow(config, run_dir=run, resume=True)
    assert digest_file(artifact) == checksum
    assert artifact.stat().st_mtime_ns == timestamp
    assert export_run(run, output_dir=tmp_path / "reexport").is_dir()


def test_structure_variants_are_distinct_durable_cases(tmp_path):
    config = structure_config(
        tmp_path,
        transforms=[
            {"type": "strain", "params": {"mode": "uniaxial", "axis": "y", "min": 0, "max": 10, "steps": 3}}
        ],
    )
    run = run_workflow(config)
    _, record, arrays = load_result(run)
    assert record["metadata"]["n_variants"] == 3
    for i, length in enumerate([4, 4.2, 4.4]):
        variant = structure_from_arrays(arrays, prefix=f"variant_{i:06d}_")
        assert variant.cell[1, 1] == pytest.approx(length)
        assert variant.positions[1, 1] == pytest.approx(length / 4)
        np.testing.assert_array_equal(variant.fixed_coordinates, [[True, False, False], [False, True, False]])


def test_structure_resume_rejects_changed_source_and_result(tmp_path):
    config = structure_config(tmp_path)
    run = run_workflow(config)
    _, record, _ = load_result(run)
    source = config.material_dir / "POSCAR"
    source.write_text(POSCAR.replace("0.25 0.25", "0.3 0.25"))
    with pytest.raises(ValueError, match="Resume rejected"):
        run_workflow(config, run_dir=run, resume=True)
    source.write_text(POSCAR)
    artifact = run / record["path"]
    artifact.write_bytes(artifact.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="changed|removed|checksum"):
        export_run(run)


def test_structure_workflow_native_symmetry_path(tmp_path):
    pytest.importorskip("spglib")
    config = structure_config(tmp_path, symmetry=True)
    run = run_workflow(config)
    _, record, _ = load_result(run)
    path = record["metadata"]["band_path"]
    assert path["point_coords"]["GAMMA"] == [0, 0, 0]
    assert path["segments"]
    assert path["coordinate_basis"] == "original"
    payload = json.loads((export_run(run) / "structure.json").read_text())
    assert payload["queer_provenance"]["scientific_metadata"]["band_path"] == path
