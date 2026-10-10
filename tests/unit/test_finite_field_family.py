"""Family provenance, lazy loading, and physical critical-point tracking."""

import json

import numpy as np
import pytest

from queer.finite_field.family import ModelFamily


def manifest(tmp_path, values=(-1.0, -0.5, 0.5, 1.0)):
    entries = []
    for i, value in enumerate(values):
        path = tmp_path / f"m{i}_hr.dat"
        rows = [
            f"0 0 0 {a + 1} {b + 1} {(-value if a == 0 else value) if a == b else 0} 0"
            for a in range(2)
            for b in range(2)
        ]
        path.write_text("synthetic\n2\n1\n1\n" + "\n".join(rows) + "\n")
        entries.append({"label": str(i), "D": value, "files": {"hr": path.name}})
    file = tmp_path / "manifest.json"
    file.write_text(json.dumps({"n_occ": 1, "lattice": np.eye(3).tolist(), "members": entries}))
    return file


def test_family_tracks_band_character_and_retains_root_evidence(tmp_path):
    family = ModelFamily.from_manifest(manifest(tmp_path))
    assert len(family._models) == 0
    report = family.critical_D_report()
    assert report["D_c"] == pytest.approx(0)
    assert "characters swap without mixing" in report["roots"][0]["reason"]
    assert report["roots"][0]["mixing"] == 0
    assert report["bands"] == [[0, 1], [0, 1], [1, 0], [1, 0]]
    json.dumps(report, allow_nan=False)
    assert family.consistency()["ok"]
    assert len(family._models) <= 2


def test_family_reloads_when_file_contents_change(tmp_path):
    family = ModelFamily.from_manifest(manifest(tmp_path))
    first = family.model(0)
    path = tmp_path / "m0_hr.dat"
    path.write_text(path.read_text().replace("1.0 0", "2.0 0"))
    assert family.model(0) is not first


def test_family_transitive_inputs_include_manifest(tmp_path):
    path = manifest(tmp_path)
    family = ModelFamily.from_manifest(path)
    assert len(family.input_files()) == 5
    assert path in family.input_files()


def test_duplicate_parameters_rejected(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        ModelFamily.from_manifest(manifest(tmp_path, values=(1.0, 1.0)))
