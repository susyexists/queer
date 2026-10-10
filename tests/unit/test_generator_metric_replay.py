"""Reciprocal-direction metric comparison and recovered recipe validation."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[2] / "scripts/validation/generator_metric_replay.py"
SPEC = importlib.util.spec_from_file_location("generator_metric_replay", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_cartesian_metric_projection_respects_nonorthogonal_reciprocal_axes():
    metric = np.array([[2.0, 0.3, 0], [0.3, 1.0, 0], [0, 0, 4.0]])
    reciprocal = np.array([[np.sqrt(3), 1, 0], [0, 2, 0], [0, 0, 0.4]])
    projected, directions = MODULE.directional_components(metric, reciprocal, 1)
    expected = [1.75 + 0.3 * np.sqrt(3) / 2, 1, 4, 0.5 + 0.3 * np.sqrt(3) / 2, 0, 0]
    np.testing.assert_allclose(projected, expected, atol=1e-15)
    assert directions[0] @ directions[1] == pytest.approx(0.5)
    assert projected[3] != metric[0, 1]
    in_plane, _ = MODULE.directional_components(metric, reciprocal, 0)
    np.testing.assert_array_equal(in_plane[[2, 4, 5]], [0, 0, 0])


def test_metric_tables_require_declared_axes_units_and_consistent_trace(tmp_path):
    average = tmp_path / "quantum_metric.dat"
    integrated = tmp_path / "quantum_metric_integrated.dat"
    average.write_text(
        "# k-averaged g_ab in Angstrom^2\n# E(eV) g_xx g_yy g_zz g_xy g_yz g_xz tr_g\n0 3 4 5 1 2 3 12\n"
    )
    integrated.write_text(
        "# integrated g_ab in Angstrom^-1\n# E(eV) G_xx G_yy G_zz G_xy G_yz G_xz\n0 .3 .4 .5 .1 .2 .3\n"
    )
    a, b = MODULE.read_metric_tables(average, integrated)
    assert a.shape == (1, 8) and b.shape == (1, 7)
    average.write_text(average.read_text().replace("g_xx g_yy", "g_yy g_xx"))
    with pytest.raises(ValueError, match="order"):
        MODULE.read_metric_tables(average, integrated)


def test_tutorial_recipe_cannot_be_bound_to_different_saved_metric_rows(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(
        json.dumps(
            {
                "QUANTUM_METRIC": {
                    "engine": "fhs",
                    "fhs_grid": [100, 100, 1],
                    "fhs_dimension": 0,
                    "fhs_energy_window": [-0.5, 0.5],
                    "fhs_energy_points": 101,
                }
            }
        )
    )
    with pytest.raises(ValueError, match="energy recipe"):
        MODULE.validate_runtime_recipe(path, [100, 100, 1], 0, np.array([0.0]))
    with pytest.raises(ValueError, match="runtime"):
        MODULE.validate_runtime_recipe(path, [8, 8, 8], 1, np.array([0.0]))


def test_metric_replay_stops_on_unverified_source_hash(tmp_path, monkeypatch):
    source = tmp_path / "different.scfout"
    source.write_bytes(b"not the historical input")
    args = type("Args", (), {"scfout": source, "expected_scfout_sha256": "0" * 64})()
    monkeypatch.setattr(MODULE.OpenMXModel, "from_scfout", lambda *a, **k: pytest.fail("source loaded"))
    with pytest.raises(ValueError, match="SHA256"):
        MODULE.replay(args)


def test_runtime_log_recovers_actual_mesh_without_claiming_hamiltonian_binding(tmp_path):
    log = tmp_path / "running.log"
    log.write_text(
        "Parameter setting (FHS quantum metric engine):\n"
        " >> fhs_grid : 6 6 6\n"
        " >> fhs_energy_window (eV) : 0.0000 0.0000\n"
        " >> fhs_energy_points : 1\n"
        " >> fhs_dimension : 1\n"
        " >> fermi_energy (eV) : -5.209804\n"
        "FHS quantum metric: peak tr(g) = 1.554422e+01 Ang^2 at E = 0.0000 eV\n"
    )
    recovered = MODULE.read_runtime_log(log, [6, 6, 6], 1, np.array([0.0]))
    assert recovered["QUANTUM_METRIC"]["fhs_grid"] == [6, 6, 6]
    assert recovered["fermi_energy_eV_printed"] == -5.209804
    assert "Hamiltonian bytes are not established" in recovered["provenance_kind"]
    with pytest.raises(ValueError, match="mesh/dimension"):
        MODULE.read_runtime_log(log, [8, 8, 8], 1, np.array([0.0]))


def test_projector_distance_known_two_level_rotation_and_full_space_polar():
    angle = 0.43
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    # Positive nonuniform singular values are removed from the full link;
    # the occupied compression must retain the actual interband rotation.
    link = np.diag([0.3, 2.7]) @ rotation
    assert MODULE.projector_distance(link, 1) == pytest.approx(np.sin(angle) ** 2, abs=1e-15)
    assert MODULE.projector_distance(link, 0) == 0
    assert MODULE.projector_distance(link, 2) == 0
    for rank in (True, 1.2, -1, 3):
        with pytest.raises(ValueError, match="rank"):
            MODULE.projector_distance(link, rank)


def test_projector_distance_invariant_under_independent_occupied_gauges():
    rng = np.random.default_rng(49)

    def unitary(size):
        value = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
        return np.linalg.qr(value)[0]

    link = unitary(4) @ np.diag([0.4, 0.8, 1.2, 2.0]) @ unitary(4)
    left = np.zeros((4, 4), complex)
    right = left.copy()
    for gauge in (left, right):
        gauge[:2, :2], gauge[2:, 2:] = unitary(2), unitary(2)
    expected = MODULE.projector_distance(link, 2)
    assert 0 < expected < 1
    assert MODULE.projector_distance(left.conj().T @ link @ right, 2) == pytest.approx(expected, abs=2e-15)


class RotatingTwoLevel:
    """Constant-gap spinor with an independently known projector distance."""

    nbnd = 2
    g_vec = 2 * np.pi * np.eye(3)
    slope = np.array([0.8, -0.3, 0.5])

    def __init__(self, transformed=False):
        self.transformed = transformed

    def gauge(self, point):
        if not self.transformed:
            return np.eye(2)
        return np.diag(np.exp(1j * np.array([1.1, -0.7]) * np.asarray(point).sum()))

    def hamiltonian(self, point):
        angle = np.asarray(point) @ self.slope
        matrix = np.array([[np.cos(angle), np.sin(angle)], [np.sin(angle), -np.cos(angle)]])
        gauge = self.gauge(point)
        return gauge.conj().T @ matrix @ gauge

    def basis_overlap(self, bra, ket):
        return self.gauge(bra).conj().T @ self.gauge(ket)


class PositionedTwoLevel(RotatingTwoLevel):
    """A fixed orthogonal orbital basis with explicit Hermitian positions."""

    def __init__(self, origin=None, residual=True):
        super().__init__()
        origin = np.zeros(3) if origin is None else np.asarray(origin)
        centres = np.array([[0.1, 0.2, -0.3], [1.4, -0.2, 2.7]])
        matrices = [np.diag(centres[:, axis] + origin[axis]).astype(complex) for axis in range(3)]
        if residual:
            matrices[0][0, 1] = matrices[0][1, 0] = 0.17
            matrices[2][0, 1], matrices[2][1, 0] = 0.11j, -0.11j
        self.positions = np.asarray(matrices)
        self.position_blocks = tuple({(0, 0, 0): matrix} for matrix in matrices)

    def overlap(self, point):
        return np.eye(2)

    def position(self, point):
        return self.positions

    def basis_overlap(self, bra, ket):
        delta = (np.asarray(ket) - np.asarray(bra)) @ self.g_vec
        return np.eye(2) - 1j * np.einsum("a,aij->ij", delta, self.positions)


def test_centre_resummed_link_has_native_first_order_tangent_and_origin_covariance():
    model = PositionedTwoLevel()
    origin = np.array([3.8, -2.4, 6.1])
    shifted = PositionedTwoLevel(origin)
    bra, ket = np.array([0.27, 0.13, 0.31]), np.array([0.34, 0.21, 0.4])
    link = MODULE.centre_resummed_basis_overlap(model, bra, ket)
    moved = MODULE.centre_resummed_basis_overlap(shifted, bra, ket)
    phase = np.exp(-1j * ((ket - bra) @ model.g_vec) @ origin)
    np.testing.assert_allclose(moved, phase * link, atol=2e-15)
    small = bra + 1e-5 * (ket - bra)
    fine = MODULE.centre_resummed_basis_overlap(model, bra, small)
    tangent = model.basis_overlap(bra, small)
    assert np.max(np.abs(fine - tangent)) < 3e-10
    finer = bra + 5e-6 * (ket - bra)
    half_error = np.max(
        np.abs(MODULE.centre_resummed_basis_overlap(model, bra, finer) - model.basis_overlap(bra, finer))
    )
    assert half_error == pytest.approx(np.max(np.abs(fine - tangent)) / 4, rel=1e-4)
    native = MODULE.finite_step_point(model, bra, [0.0], [6, 6, 6], 1, "centre_resummed")
    moved = MODULE.finite_step_point(shifted, bra, [0.0], [6, 6, 6], 1, "centre_resummed")
    np.testing.assert_allclose(moved, native, atol=2e-14)


def test_reference_centres_are_position_diagonals_and_half_cell_ties_are_explicit():
    model = PositionedTwoLevel(residual=False)
    expected = np.stack([np.diag(matrix).real for matrix in model.positions], axis=1)
    np.testing.assert_array_equal(MODULE.reference_centres(model), expected)
    # Integer reciprocal translation gives zero displacement in this estimator.
    np.testing.assert_array_equal(
        MODULE.centre_resummed_basis_overlap(model, [0, 0, 0], [1, 0, 0]), np.eye(2)
    )
    for step in (0.5, -0.5):
        link = MODULE.centre_resummed_basis_overlap(model, [0, 0, 0], [step, 0, 0])
        np.testing.assert_allclose(link, np.diag(np.exp(2j * np.pi * step * expected[:, 0])), atol=1e-15)
    broken = type("Model", (), {"nbnd": 2, "position_blocks": ({}, {}, {})})()
    with pytest.raises(ValueError, match="zero-cell"):
        MODULE.reference_centres(broken)


def test_centre_resummed_estimator_matches_independent_embedded_spinor_fidelity():
    model = PositionedTwoLevel(residual=False)
    point = np.array([0.27, -0.08, 0.14])
    grid = np.array([13, 17, 19])
    centres = MODULE.reference_centres(model)
    steps = np.eye(3) / grid[:, None]
    angle = point @ model.slope
    bra = np.array([-np.sin(angle / 2), np.cos(angle / 2)])

    def distance(delta):
        angle = (point + delta) @ model.slope
        ket = np.array([-np.sin(angle / 2), np.cos(angle / 2)])
        phase = np.exp(-1j * (centres @ (delta @ model.g_vec)))
        return 1 - abs(np.vdot(bra, phase * ket)) ** 2

    singles = np.asarray([distance(delta) for delta in steps])
    lengths = 2 * np.pi / grid
    expected = list(singles / lengths**2)
    for i, j in ((0, 1), (1, 2), (0, 2)):
        expected.append(
            (distance(steps[i] + steps[j]) - singles[i] - singles[j]) / (2 * lengths[i] * lengths[j])
        )
    actual = MODULE.finite_step_point(model, point, [0.0], grid, 1, "centre_resummed")[0]
    np.testing.assert_allclose(actual, expected, atol=8e-15, rtol=1e-12)


def test_native_finite_steps_analytical_tensor_and_basis_gauge_covariance():
    model = RotatingTwoLevel()
    point = np.array([0.27, -0.08, 0.14])
    grid = np.array([13, 17, 19])
    steps = np.eye(3) / grid[:, None]
    distance = np.sin((steps @ model.slope) / 2) ** 2
    lengths = 2 * np.pi / grid
    expected = list(distance / lengths**2)
    for i, j in ((0, 1), (1, 2), (0, 2)):
        combined = np.sin((steps[i] + steps[j]) @ model.slope / 2) ** 2
        expected.append((combined - distance[i] - distance[j]) / (2 * lengths[i] * lengths[j]))
    native = MODULE.finite_step_point(model, point, np.array([0.0]), grid, 1)[0]
    transformed = MODULE.finite_step_point(RotatingTwoLevel(True), point, [0.0], grid, 1)[0]
    np.testing.assert_allclose(native, expected, atol=8e-15, rtol=1e-11)
    np.testing.assert_allclose(transformed, native, atol=8e-15, rtol=1e-11)
    # Shrinking finite steps recovers the spinor's differential tensor.
    fine = MODULE.finite_step_point(model, point, [0.0], grid * 10, 1)[0]
    continuum = np.outer(model.slope, model.slope) / (16 * np.pi**2)
    reference = np.array([continuum[i, j] for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))])
    assert np.max(np.abs(fine - reference)) < np.max(np.abs(native - reference)) / 90


def test_finite_step_checkpoints_resume_without_recomputing_completed_tiles(tmp_path, monkeypatch):
    source = tmp_path / "source.scfout"
    source.write_bytes(b"analytical model source fingerprint")
    run = tmp_path / "finite"
    model = RotatingTwoLevel()
    options = MODULE.ExecutionOptions(workers=1, batch_size=1, native_threads=1, memory_mb=32)
    original = MODULE.finite_step_point
    calls = []

    def interrupted(*args):
        calls.append(np.asarray(args[1]).copy())
        if len(calls) == 3:
            raise RuntimeError("simulated interruption")
        return original(*args)

    monkeypatch.setattr(MODULE, "finite_step_point", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        MODULE.finite_step_grid(model, [0.0], [2, 2, 1], 0, run, options, source, False)
    failed = json.loads((run / "manifest.json").read_text())
    assert failed["status"] == "failed"
    assert len(failed["cases"]) == 2
    before = {key: value.copy() for key, value in failed["cases"].items()}
    calls.clear()

    def resumed(*args):
        calls.append(np.asarray(args[1]).copy())
        return original(*args)

    monkeypatch.setattr(MODULE, "finite_step_point", resumed)
    result = MODULE.finite_step_grid(model, [0.0], [2, 2, 1], 0, run, options, source, True)
    assert len(calls) == 2
    completed = MODULE.load_manifest(run)
    for key, value in before.items():
        assert completed["cases"][key] == value
    # A fully completed resume needs neither model evaluation nor a new result.
    result_checkpoint = completed["cases"]["result"].copy()
    monkeypatch.setattr(MODULE, "map_points", lambda *a, **k: pytest.fail("recomputed completed result"))
    repeated = MODULE.finite_step_grid(model, [0.0], [2, 2, 1], 0, run, options, source, True)
    np.testing.assert_array_equal(repeated, result)
    assert MODULE.load_manifest(run)["cases"]["result"] == result_checkpoint
    with pytest.raises(ValueError, match="scientific configuration"):
        MODULE.finite_step_grid(model, [0.0], [2, 2, 1], 0, run, options, source, True, "centre_resummed")


@pytest.mark.parametrize("passed, exit_status", [(False, 1), (True, 0), (None, 0)])
def test_cli_exposes_failed_finite_comparison_exit_status(monkeypatch, capsys, passed, exit_status):
    report = {"status": "completed", "comparison": {"same_estimator_acceptance": None}}
    if passed is not None:
        report["finite_step_comparison"] = {"passed": passed, "max_absolute_difference_A2": 48.0}
    monkeypatch.setattr(MODULE, "replay", lambda args: report)
    status = MODULE.main(
        [
            "--scfout",
            "model.scfout",
            "--expected-scfout-sha256",
            "0" * 64,
            "--average",
            "average.dat",
            "--integrated",
            "integrated.dat",
            "--grid",
            "6",
            "6",
            "6",
            "--dimension",
            "1",
            "--binding",
            "inferred",
            "--output-dir",
            "result",
        ]
    )
    assert status == exit_status
    printed = json.loads(capsys.readouterr().out)
    if passed is not None:
        assert printed["finite_step_comparison"]["passed"] is passed


def test_cli_does_not_silently_ignore_requested_finite_basis_link(monkeypatch, capsys):
    monkeypatch.setattr(MODULE, "replay", lambda args: pytest.fail("ignored requested finite link"))
    with pytest.raises(SystemExit) as error:
        MODULE.main(
            [
                "--scfout",
                "model.scfout",
                "--expected-scfout-sha256",
                "0" * 64,
                "--average",
                "average.dat",
                "--integrated",
                "integrated.dat",
                "--grid",
                "6",
                "6",
                "6",
                "--dimension",
                "1",
                "--binding",
                "inferred",
                "--output-dir",
                "result",
                "--finite-basis-link",
                "centre_resummed",
            ]
        )
    assert error.value.code == 2
    assert "requires --finite-step" in capsys.readouterr().err
