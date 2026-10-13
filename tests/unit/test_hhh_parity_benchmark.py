"""Parity acceptance treats resolved signs and absent numerical values explicitly."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "hhh_parity_benchmark", Path(__file__).resolve().parents[2] / "benchmarks/compare_hhh.py"
)
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def test_resolved_harmonic_sign_is_required_and_zero_noise_is_ignored():
    actual, reference = np.array([2e-5, -1e-4]), np.array([-2e-5, -1e-4])
    result = benchmark._compare(np, actual, reference, 1e-4)
    assert not result["passed"] and result["sign_mismatches"] == 1
    result = benchmark._compare(np, [-1e-17, 1e-4], [1e-17, 1e-4], 1e-4)
    assert result["passed"] and result["sign_components"]["real"]["meaningful_points"] == 1


def test_complex_methods_projection_uses_resolved_imaginary_component():
    result = benchmark._compare(np, [-1e-17 - 2e-5j], [1e-17 - 2e-5j], 1e-4)
    assert result["passed"] and result["sign_components"]["real"]["meaningful_points"] == 0
    assert result["sign_components"]["imag"]["meaningful_points"] == 1
    result = benchmark._compare(np, [2e-5j], [-2e-5j], 1e-4)
    assert not result["passed"] and result["sign_components"]["imag"]["mismatches"] == 1


def test_missing_crossings_and_nonfinite_values_cannot_pass_as_zero_error():
    assert not benchmark._compare(np, [], [0.0], 1e-7)["passed"]
    assert not benchmark._compare(np, [np.nan], [0.0], 1e-7)["passed"]
    assert not benchmark._compare(np, [complex(0, np.inf)], [complex(0, -np.inf)], 1e-7)["passed"]
    assert benchmark._compare(np, [np.nan], [np.nan], 1e-7)["passed"]


def test_strict_geometry_checks_each_small_metric_component():
    actual, reference = [1e5, 0.00101], [1e5, 0.001]
    assert benchmark._compare(np, actual, reference, 1e-8, floor=1e-12)["passed"]
    strict = benchmark._compare(np, actual, reference, 1e-8, floor=1e-12, pointwise=True)
    assert not strict["passed"] and not strict["pointwise_passed"]
    assert strict["max_error_over_pointwise_limit"] > 1


def test_native_validation_appendix_requires_ast_identity_outside_guarded_branch(tmp_path):
    before, after = tmp_path / "before.py", tmp_path / "after.py"
    original = (
        "class NativePlaneModel:\n"
        "    def _wannier_geometry(self, model):\n"
        "        if model.position_matrices is None:\n"
        "            return 1\n"
        "        else:\n"
        "            return 2\n"
    )
    before.write_text(original)
    after.write_text(original.replace("return 2", "validate(model)\n            return 2"))
    assert benchmark._native_position_validation_delta(before, after)["passed"]
    after.write_text(original.replace("return 1", "return 3"))
    assert not benchmark._native_position_validation_delta(before, after)["passed"]
    after.write_text(original.replace("is None", "is not None"))
    assert not benchmark._native_position_validation_delta(before, after)["passed"]


def test_scoped_position_proof_cannot_accept_other_source_or_exercised_branch_changes():
    module = "finite_field/native.py"
    report = {
        "sources": {"queer": {"files": {module: "before"}}},
        "sources_final": {"queer": {"files": {module: "after"}}},
        "source_changes_during_run": {"queer": {"files": [module]}, "hhh": {"files": []}},
        "cases": [{"name": "real/material"}],
    }
    static = {"before_sha256": "before", "after_sha256": "after", "passed": True}
    numeric = {
        "before_native_sha256": "before",
        "after_native_sha256": "after",
        "cases": {
            "real/material": {
                "position_matrices_is_none": True,
                "input_hashes_match_live": True,
                "all_products_bitwise_equal": True,
            }
        },
    }
    assert benchmark._scoped_position_validation_acceptance(report, static, numeric)["passed"]
    report["source_changes_during_run"]["queer"]["files"].append("finite_field/lz/crossings.py")
    assert not benchmark._scoped_position_validation_acceptance(report, static, numeric)["passed"]
    report["source_changes_during_run"]["queer"]["files"].pop()
    numeric["cases"]["real/material"]["position_matrices_is_none"] = False
    assert not benchmark._scoped_position_validation_acceptance(report, static, numeric)["passed"]


def test_tba_replay_rejects_a_discovered_tb_geometry_gauge(tmp_path):
    hpc = tmp_path / "hpc"
    hpc.mkdir()
    for path in (
        hpc / "wannier90_hr.dat",
        hpc / "wannier90_wsvec.dat",
        tmp_path / "wannier90_centres.xyz",
        tmp_path / "wannier90.wout",
    ):
        path.touch()
    _, files = benchmark._inputs(tmp_path)
    assert files["hr"] == hpc / "wannier90_hr.dat"
    (hpc / "wannier90_tb.dat").touch()
    with pytest.raises(ValueError, match="different geometry gauge"):
        benchmark._inputs(tmp_path)


def test_provenance_retains_initial_hashes_and_rejects_changed_numerics(tmp_path):
    (tmp_path / "finite_field").mkdir()
    (tmp_path / "workflows").mkdir()
    core, workflow = tmp_path / "finite_field/core.py", tmp_path / "workflows/finite_field.py"
    core.write_text("mass = 0.2\n")
    workflow.write_text("children = 1\n")
    initial = benchmark._source(tmp_path)
    report = {"sources": {"queer": initial}, "driver_sha256": "initial-driver-hash"}
    workflow.write_text("children = 2\n")
    assert benchmark._finalize_sources(report)
    assert report["sources"]["queer"] == initial
    assert report["source_changes_during_run"]["queer"] == {
        "files": ["workflows/finite_field.py"],
        "classification": "workflow_only",
    }
    assert report["numerical_source_fingerprints"]["queer"]["unchanged"]
    assert report["driver_sha256"] == "initial-driver-hash" and report["driver_changed_during_run"]
    core.write_text("mass = 0.3\n")
    assert not benchmark._finalize_sources(report)
    assert report["source_changes_during_run"]["queer"]["files"] == [
        "finite_field/core.py",
        "workflows/finite_field.py",
    ]
    assert not report["numerical_source_fingerprints"]["queer"]["unchanged"]


def test_shared_response_warning_mismatch_invalidates_diagnostic_parity():
    normal = {"line_flags": {}, "layout_flags": [], "response_flags": []}
    flagged = {**normal, "response_flags": ["interference_under_resolved"]}
    result = benchmark._structural_diagnostics({"queer": normal, "hhh": flagged})
    assert not result["passed"]
    assert result["hhh"]["response_flags"] == ["interference_under_resolved"]
    assert benchmark._structural_diagnostics({"queer": flagged, "hhh": flagged})["passed"]


def test_matched_minima_geometry_uses_shared_points_and_strict_geometry_tolerance():
    class Provider:
        energy_ref = 4.5
        angle = 0.0

        def __init__(self, metric_error=0):
            self.metric_error = metric_error
            self.points = None

        def geometry(self, s, t):
            self.points = s.copy()
            # A steep physical variation makes tiny minimum-location shifts visible,
            # while identical coordinates isolate an actual operator discrepancy.
            metric = 1 + 100 * s
            return SimpleNamespace(
                eps_g=np.ones_like(s) * 0.003,
                g_cv=metric + self.metric_error,
                g_v_total=metric + 1,
                g_c_total=metric + 2,
                omega_v=metric + 3,
                omega_c=-metric - 4,
            )

    qprovider, hprovider = Provider(), Provider()
    case = {}
    benchmark._matched_crossing_geometry(np, case, qprovider, hprovider, [0.001], [0.001000001], 1e-8)
    np.testing.assert_array_equal(qprovider.points, [0.001, 0.001000001])
    np.testing.assert_array_equal(hprovider.points, qprovider.points)
    assert case["matched_crossing_geometry"]["passed"]
    effects = case["matched_crossing_geometry"]["location_effects"]["g_cv"]
    np.testing.assert_allclose(effects["queer_geometry_at_queer_minus_hhh_minima"], [-1e-7])
    np.testing.assert_array_equal(effects["same_coordinate_geometry_difference_at_hhh_minima"], [0])
    qprovider.metric_error = 3e-8
    benchmark._matched_crossing_geometry(np, case, qprovider, hprovider, [0.001], [0.001000001], 1e-8)
    assert not case["matched_crossing_geometry"]["passed"]
    assert not case["products"]["matched_crossing_geometry_g_cv"]["passed"]
