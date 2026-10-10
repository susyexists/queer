"""Downloaded Generator outputs can be compared using native operators offline."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from queer import ExecutionOptions, WannierModel
from queer.io.artifacts import digest_file

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "generator_replay", ROOT / "scripts/validation/generator_replay.py"
)
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


def downloaded_fixture(tmp_path):
    source, producer = tmp_path / "job-source", tmp_path / "job-producer"
    source.mkdir()
    producer.mkdir()
    translations = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
    positive = np.array([[-0.1, 0.01 + 0.02j], [0, 0.05]])
    hopping = np.array([positive.conj().T, [[-0.25, 0.3], [0.3, 0.45]], positive])
    rows = ["analytical replay fixture", "2", "3", "1 1 1"]
    for translation, matrix in zip(translations, hopping, strict=True):
        for i in range(2):
            for j in range(2):
                real, imag = matrix[i, j].real, matrix[i, j].imag
                rows.append(f"{translation[0]} 0 0 {i + 1} {j + 1} {real:.16g} {imag:.16g}")
    (source / "wannier90_hr.dat").write_text("\n".join(rows) + "\n")
    (source / "wannier90_centres.xyz").write_text("2\ncentres\nX 0 0 0\nX 0.2 0.1 0\n")
    (source / "wannier90.win").write_text(
        "begin unit_cell_cart\nang\n1 0 0\n0 1 0\n0 0 2\nend unit_cell_cart\nmp_grid = 4 4 1\n"
    )
    points = np.array([[0, 0, 0], [0.13, 0.2, 0], [0.5, 0.1, 0]])
    theta = 2 * np.pi * points[:, 0]
    a, d = -0.25 - 0.2 * np.cos(theta), 0.45 + 0.1 * np.cos(theta)
    b = 0.3 + (0.01 + 0.02j) * np.exp(1j * theta)
    radius = np.sqrt((a - d) ** 2 / 4 + np.abs(b) ** 2)
    reference = np.stack(((a + d) / 2 - radius, (a + d) / 2 + radius), axis=1)
    result = {
        "schema_version": 1,
        "status": "ok",
        "step_type": "PYTHTB_BANDS",
        "model": {
            "num_wann": 2,
            "n_R": 3,
            "n_R_after_ws": 3,
            "n_R_model": 3,
            "lattice": np.diag([1, 1, 2]).tolist(),
            "periodic_dirs": [0, 1],
            "folded_axes": [2],
            "fermi_energy": 0.11,
            "ws_correction": {"mode": "off"},
            "cutoffs": {"min_hopping_norm": None, "max_distance": None},
            "source_wannier_job_id": "job-source",
        },
        "bands": {
            "kpoints_frac": points.tolist(),
            "eigenvalues_eV": reference.tolist(),
            "fermi_energy": 0.11,
        },
    }
    (producer / "pythtb_results.json").write_text(json.dumps(result))
    (producer / "pythtb_params.json").write_text("{}")
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"id": "job-producer", "workflowId": "workflow-original"}))
    return source, producer, job, points, reference


def test_offline_replay_uses_exact_points_and_preserves_absolute_zero(tmp_path, monkeypatch):
    source, producer, job, points, reference = downloaded_fixture(tmp_path)
    monkeypatch.setitem(__import__("sys").modules, "pythtb", None)
    output = tmp_path / "comparison"
    assert (
        REPLAY.main(
            [
                "--producer-dir",
                str(producer),
                "--wannier-dir",
                str(source),
                "--job-metadata",
                str(job),
                "--output-dir",
                str(output),
                "--batch-size",
                "1",
            ]
        )
        == 0
    )
    report = json.loads((output / "comparison.json").read_text())
    assert report["original"]["workflow_id"] == "workflow-original"
    assert report["original"]["source_job_id"] == "job-source"
    assert report["comparisons"]["bands"]["coverage"] == 1
    assert report["comparisons"]["bands"]["max_absolute_error"] < 1e-14
    assert report["inputs"]["hr"]["sha256"] == digest_file(source / "wannier90_hr.dat")
    assert report["artifacts"]["bands"]["sha256"] == digest_file(output / "bands.npz")
    with np.load(output / "bands.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["kpoints_fractional"], points)
        np.testing.assert_allclose(arrays["native_absolute_eV"], reference, atol=1e-14)
    assert "no fitted shift" in (output / "comparison.md").read_text()


def test_missing_comparison_samples_cannot_pass():
    metrics = REPLAY.error_metrics([1, np.nan, 3], [1, 2, 3])
    assert metrics["coverage"] == pytest.approx(2 / 3)
    assert metrics["max_absolute_error"] == 0
    assert not metrics["passed"]
    with pytest.raises(ValueError, match="shapes"):
        REPLAY.error_metrics([1, 2], [1])


def test_historical_cutoff_is_rejected_before_an_inexact_replay(tmp_path):
    source, producer, _, _, _ = downloaded_fixture(tmp_path)
    result = json.loads((producer / "pythtb_results.json").read_text())
    result["model"]["cutoffs"]["max_distance"] = 2
    with pytest.raises(ValueError, match="cutoffs"):
        REPLAY.build_native_model(source, result["model"])


def test_producer_parameter_and_core_bytes_must_match_recorded_provenance(tmp_path):
    params, core = tmp_path / "params.json", tmp_path / "_pythtb_core.py"
    params.write_text('{"historical": true}')
    core.write_text("raise RuntimeError('the producer must never be imported')\n")
    provenance = {"params_sha256": digest_file(params), "core_sha256": digest_file(core)}
    integrity = REPLAY.validate_producer_fingerprints(params, provenance, core)
    assert integrity["params"]["verified"]
    assert integrity["core"]["verified"]
    assert not REPLAY.validate_producer_fingerprints(params, provenance)["core"]["verified"]
    params.write_text('{"historical": false}')
    with pytest.raises(ValueError, match="params SHA256"):
        REPLAY.validate_producer_fingerprints(params, provenance, core)


def test_total_dos_samples_only_periodic_axes_and_counts_spinor_states_once():
    model = WannierModel([[0, 0, 0]], [np.diag([-0.4, 0.4])], lattice=np.eye(3))
    original = {"energy_eV": np.linspace(-1, 1, 101), "dos_per_eV": np.zeros(101)}
    result = {
        "model": {"periodic_dirs": [0, 1]},
        "dos": {"mesh": [4, 3, 99], "k_points": 12, "sigma_eV": 0.03},
    }
    progress = []
    arrays, metadata = REPLAY.replay_total_dos(
        model,
        result,
        {"dos_step_eV": 0.02},
        original,
        ExecutionOptions(batch_size=5),
        progress=lambda completed, total: progress.append((completed, total)),
    )
    assert metadata["k_points"] == 12
    assert metadata["sampled_eigenvalues"] == 24
    assert metadata["histogram_coverage"] == 1
    assert metadata["native_integral_states"] == pytest.approx(2, abs=1e-14)
    assert np.all(arrays["kpoints_fractional"][:, 2] == 0)
    assert np.count_nonzero(arrays["histogram_counts"]) == 2
    assert progress == [(5, 12), (10, 12), (12, 12)]


def test_dos_rejects_metadata_that_claims_an_unsampled_mesh():
    model = WannierModel([[0, 0, 0]], [np.diag([-0.4, 0.4])])
    result = {"model": {"periodic_dirs": [0, 1]}, "dos": {"mesh": [4, 3], "k_points": 13, "sigma_eV": 0.03}}
    original = {"energy_eV": np.linspace(-1, 1, 101), "dos_per_eV": np.zeros(101)}
    with pytest.raises(ValueError, match="point count"):
        REPLAY.replay_total_dos(model, result, {"dos_step_eV": 0.02}, original, ExecutionOptions())


def test_historical_fermi_scalar_keeps_ceiling_nearest_and_interpolated_definitions_distinct():
    arrays = {
        "energy_relative_eV": np.array([-0.002, 0.003, 0.008]),
        "native_dos_per_eV": np.array([10, 30, 50]),
        "reference_dos_per_eV": np.array([10, 30, 50]),
    }
    report = {"dos": {}, "comparisons": {}}
    REPLAY.add_fermi_dos_comparison(
        report,
        arrays,
        {"electronic": {"dos_at_fermi_per_eV": 30}},
        policy="first_nonnegative_bin",
        atol=1e-8,
        rtol=1e-10,
    )
    diagnostics = report["dos"]["at_fermi"]
    assert diagnostics["historical_policy"] == "first_nonnegative_bin"
    assert diagnostics["recorded_per_eV"] == 30
    assert diagnostics["estimators"]["first_nonnegative_bin"]["bin_index"] == 1
    assert diagnostics["estimators"]["nearest_bin"]["native_per_eV"] == 10
    assert diagnostics["estimators"]["linear_interpolation"]["native_per_eV"] == pytest.approx(18)
    assert report["comparisons"]["dos_at_fermi"]["passed"]
    assert report["comparisons"]["dos_at_fermi_artifact_consistency"]["passed"]


def test_historical_scalar_definition_is_not_selected_to_fit_a_discrepant_value():
    report = {"dos": {}, "comparisons": {}}
    arrays = {
        "energy_relative_eV": [-0.002, 0.003, 0.008],
        "native_dos_per_eV": [10, 30, 50],
        "reference_dos_per_eV": [10, 30, 50],
    }
    REPLAY.add_fermi_dos_comparison(
        report,
        arrays,
        {"electronic": {"dos_at_fermi_per_eV": 18}},
        policy="first_nonnegative_bin",
        atol=1e-8,
        rtol=1e-10,
    )
    assert not report["comparisons"]["dos_at_fermi"]["passed"]
    assert not report["comparisons"]["dos_at_fermi_artifact_consistency"]["passed"]
