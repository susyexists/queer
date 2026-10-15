"""Independent persistence, provenance, and scientific reduction contracts."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from queer.config import config_from_mapping, load_config
from queer.finite_field.assembly import stack_member_results
from queer.io.artifacts import RunStore, atomic_json, owned_run
from queer.models.wannier import WannierModel
from queer.observables.integration import integrate_unit_box
from queer.observables.result import ObservableResult
from queer.workflows import render_run, run_workflow
from queer.workflows.catalog import REGISTRY
from queer.workflows.reduction import reduce_cases
from queer.workflows.runner import _input_files

ROOT = Path(__file__).resolve().parents[2]


class MemoryCheckpoint:
    def __init__(self):
        self.records = {}

    def load(self, case):
        return self.records.get(case)

    def save(self, case, arrays, metadata):
        self.records[case] = ({key: np.array(value) for key, value in arrays.items()}, metadata)


def test_local_quadrature_resumes_with_an_injected_memory_checkpoint():
    checkpoint = MemoryCheckpoint()
    controls = {"method": "cubature", "absolute_tolerance": 1e-3, "relative_tolerance": 0.0}
    result = integrate_unit_box(lambda points: points[:, 0] ** 2, (2, 1, 1), controls, checkpoint=checkpoint)
    assert checkpoint.records
    np.testing.assert_allclose(result.value, 1 / 3, rtol=0, atol=1e-15)

    def prohibited(points):
        raise AssertionError("Completed quadrature samples must be reused")

    resumed = integrate_unit_box(prohibited, (2, 1, 1), controls, checkpoint=checkpoint)
    np.testing.assert_array_equal(resumed.value, result.value)


def test_source_inventory_discovers_sidecars_without_constructing_a_model(tmp_path, monkeypatch):
    source = tmp_path / "material"
    shutil.copytree(ROOT / "tests/fixtures/wannier/chain", source)
    centres = source / "wannier90_centres.xyz"
    centres.write_text("1\nWannier centre\nX 0 0 0\n")
    config = load_config(ROOT / "configs/examples/bands.toml", overrides={"material_dir": source})

    def prohibited(*args, **kwargs):
        raise AssertionError("Input inventory must not parse a model")

    monkeypatch.setattr(WannierModel, "from_directory", prohibited)
    assert centres in _input_files(config)
    finite = config_from_mapping(
        {
            "workflow": "crossing_scan",
            "inputs": {"model": "wannier", "material_dir": source},
            "parameters": {"plane": {"direction": [1, 0, 0], "transverse": [0, 1, 0]}},
        }
    )
    assert centres in _input_files(finite)


def test_changed_implicit_sidecar_rejects_resume_but_old_completed_run_renders(tmp_path, monkeypatch):
    source = tmp_path / "material"
    shutil.copytree(ROOT / "tests/fixtures/wannier/chain", source)
    centres = source / "wannier90_centres.xyz"
    centres.write_text("1\nWannier centre\nX 0 0 0\n")
    config = load_config(ROOT / "configs/examples/bands.toml", overrides={"material_dir": source})
    run = run_workflow(config, run_dir=tmp_path / "run")
    manifest = json.loads((run / "manifest.json").read_text())
    assert str(centres) in {record["path"] for record in manifest["inputs"]}
    centres.write_text("1\nChanged Wannier centre\nX 0.1 0 0\n")
    with pytest.raises(ValueError, match="Resume rejected"):
        run_workflow(config, run_dir=run, resume=True)
    # Completed readers use the recorded schema and arrays, not today's algorithm.
    manifest["algorithm_version"] = "1.2.1-native-finite-field-v1"
    atomic_json(run / "manifest.json", manifest)

    def prohibited(*args, **kwargs):
        raise AssertionError("Artifact rendering must not load a model")

    monkeypatch.setattr(WannierModel, "from_directory", prohibited)
    assert (render_run(run) / "bands.png").is_file()


def tile_metadata(weight, *, dimension=("energy",)):
    return {
        "dims": {"omega": ("energy",), "dos": dimension},
        "units": {"omega": "eV", "dos": "states/eV/cell"},
        "quadrature_weight": weight,
        "sample_count": 1,
    }


def store_config():
    return {
        "workflow": "dos",
        "parameters": {},
        "inputs": {},
        "execution": {"workers": 1, "batch_size": 256, "native_threads": 1, "memory_mb": None},
    }


@pytest.mark.parametrize("defect", ("shape", "dtype", "coordinate", "undeclared", "weight"))
def test_reduction_rejects_invalid_scientific_partials(tmp_path, defect):
    with owned_run(RunStore(tmp_path / "run", store_config(), [])) as store:
        store.commit("first", {"omega": [0.0, 1.0], "dos": [2.0, 3.0]}, metadata=tile_metadata(0.5))
        arrays = {"omega": np.array([0.0, 1.0]), "dos": np.array([4.0, 5.0])}
        metadata = tile_metadata(0.5)
        if defect == "shape":
            arrays["dos"] = np.array([4.0])
        elif defect == "dtype":
            arrays["dos"] = np.array([4, 5])
        elif defect == "coordinate":
            arrays["omega"] = np.array([0.0, 2.0])
        elif defect == "undeclared":
            arrays["unclassified"] = np.array([1.0])
        else:
            metadata["quadrature_weight"] = 0.3
        store.commit("second", arrays, metadata=metadata)
        with pytest.raises(ValueError):
            reduce_cases(store, ["first", "second"], "dos", REGISTRY["dos"].reduction)


def test_reduction_reads_json_roundtrip_dimensions_and_uses_global_weights(tmp_path):
    root = tmp_path / "run"
    with owned_run(RunStore(root, store_config(), [])) as store:
        store.commit("first", {"omega": [0.0, 1.0], "dos": [2.0, 4.0]}, metadata=tile_metadata(0.25))
    with owned_run(RunStore(root, store_config(), [], resume=True)) as store:
        store.commit("second", {"omega": [0.0, 1.0], "dos": [6.0, 8.0]}, metadata=tile_metadata(0.75))
        arrays, _ = reduce_cases(store, ["first", "second"], "dos", REGISTRY["dos"].reduction)
        np.testing.assert_array_equal(arrays["dos"], [5.0, 7.0])


def member_result(*, coordinate=1.0, value=2.0, revision="same", unit="A/m"):
    return ObservableResult(
        {"amplitudes": np.array([coordinate]), "current": np.array([value])},
        {"amplitudes": ("amplitude",), "current": ("amplitude",)},
        {"amplitudes": "V/m", "current": unit},
        {"source_sha256": revision},
    )


@pytest.mark.parametrize("difference", ({"coordinate": 2.0}, {"revision": "changed"}, {"unit": "S"}))
def test_member_assembly_rejects_incompatible_grids_revisions_and_units(difference):
    with pytest.raises(ValueError):
        stack_member_results([member_result(), member_result(**difference)], shared={"amplitudes"})


def test_member_assembly_keeps_member_order_and_shared_coordinates():
    arrays, dims, units = stack_member_results(
        [member_result(value=7.0), member_result(value=3.0)],
        shared={"amplitudes"},
    )
    np.testing.assert_array_equal(arrays["current"], [[7.0], [3.0]])
    np.testing.assert_array_equal(arrays["amplitudes"], [1.0])
    assert dims["current"] == ("member", "amplitude") and units["current"] == "A/m"


def tb_only_family(tmp_path):
    tb = tmp_path / "member_tb.dat"
    tb.write_text(
        "synthetic TB\n1 0 0\n0 1 0\n0 0 1\n2\n1\n1\n0 0 0\n1 1 -1 0\n1 2 .2 0\n2 1 .2 0\n2 2 1 0\n"
    )
    # A matching Hamiltonian allows the historical standalone HR default to
    # remain valid, while distinct centres expose accidental seed selection.
    hr = tmp_path / "wannier90_hr.dat"
    hr.write_text("synthetic HR\n2\n1\n1\n0 0 0 1 1 -1 0\n0 0 0 1 2 .2 0\n0 0 0 2 1 .2 0\n0 0 0 2 2 1 0\n")
    member_centres = tmp_path / "member_centres.xyz"
    member_centres.write_text("2\nmember centres\nX .2 0 0\nX .7 .1 0\n")
    canonical_centres = tmp_path / "wannier90_centres.xyz"
    canonical_centres.write_text("2\ncanonical centres\nX 8 0 0\nX 9 0 0\n")
    path = tmp_path / "family.json"
    path.write_text(
        json.dumps(
            {
                "n_occ": 1,
                "members": [
                    {
                        "label": "member",
                        "D": 0.1,
                        "files": {"tb": tb.name},
                    }
                ],
            }
        )
    )
    config = config_from_mapping(
        {
            "workflow": "harmonic_map",
            "inputs": {"family_manifest": path},
            "parameters": {"amplitudes": [1e4]},
        }
    )
    return config, tb, hr, member_centres, canonical_centres


def test_tb_only_family_inventory_and_split_model_preserve_member_seed(tmp_path):
    from queer.workflows.finite_field import load_finite_field_source, split_harmonic_map

    config, tb, hr, member_centres, canonical_centres = tb_only_family(tmp_path)
    family = load_finite_field_source(config)
    loader = family.members[0].model_loader
    expected = {tb, member_centres}
    assert set(loader.effective_files().values()) == expected
    assert set(_input_files(config)) == {Path(config.inputs["family_manifest"]), *expected}
    children, _ = split_harmonic_map(config)
    child = children[0]
    assert child.inputs["hr"] == child.inputs["tb"] == str(tb)
    assert set(_input_files(child)) == expected
    family_model = family.model(0).source_model
    child_model = load_finite_field_source(child)
    np.testing.assert_array_equal(child_model.centres, family_model.centres)
    for point in ([0, 0, 0], [0.2, 0.3, 0.1]):
        np.testing.assert_array_equal(child_model.hamiltonian(point), family_model.hamiltonian(point))
        np.testing.assert_array_equal(child_model.position(point), family_model.position(point))

    # Standalone explicit-TB workflows retain their established implicit HR.
    standalone = config_from_mapping(
        {
            "workflow": "hall_harmonics",
            "inputs": {"model": "wannier", "material_dir": tmp_path, "tb": tb.name},
            "parameters": {"amplitudes": [1e4]},
        }
    )
    assert set(_input_files(standalone)) == {tb, hr, canonical_centres}
    standalone_model = load_finite_field_source(standalone)
    np.testing.assert_array_equal(standalone_model.centres, [[8, 0, 0], [9, 0, 0]])


def test_tb_only_family_implicit_member_centres_are_part_of_resume_identity(tmp_path):
    config, _, _, member_centres, _ = tb_only_family(tmp_path)
    inputs = _input_files(config)
    root = tmp_path / "run"
    with owned_run(RunStore(root, config.as_dict(), inputs)) as store:
        store.commit("checkpoint", {"value": [1.0]})
    member_centres.write_text("2\nchanged member centres\nX .3 0 0\nX .7 .1 0\n")
    with pytest.raises(ValueError, match="Resume rejected"):
        RunStore(root, config.as_dict(), _input_files(config), resume=True)
