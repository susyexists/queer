from pathlib import Path

import numpy as np
import pytest

from queer.execution import ExecutionOptions
from queer.io.wannier import read_hr, read_wsvec
from queer.model import model as LegacyModel
from queer.models.wannier import NumericalError, WannierModel

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "wannier"


def test_complex_chain_phase_degeneracy_and_fermi_reference():
    model = WannierModel.from_directory(FIXTURES / "chain", poscar="POSCAR", ef=0.1)
    points = np.array([[0.125, 0, 0], [0, 0, 0], [0.25, 0.5, 0.1], [-0.125, 0, 0]])
    phi = 2 * np.pi * points[:, 0]
    expected = 0.3 * np.cos(phi) - 0.2 * np.sin(phi)
    np.testing.assert_allclose(model.energies(points)[0], expected - 0.1, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(model.hamiltonian(points[0]), [[0.07071067811865475]], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        model.energies(points + [2, -3, 1]), model.energies(points), atol=1e-12, rtol=1e-12
    )
    assert model.g_vec.shape == (3, 3)
    assert not model.hopping_matrices.flags.writeable


def test_shuffled_orbital_rows_preserve_complex_matrix_order():
    model = WannierModel.from_directory(FIXTURES / "two_band")
    onsite = np.array([[0.5, 0.15 + 0.25j], [0.15 - 0.25j, -0.5]])
    hopping = np.array([[0.2 + 0.1j, 0.3 - 0.2j], [-0.1 + 0.4j, 0.05 - 0.07j]])
    points = np.array([[0.137, 0.2, 0.4], [-0.137, 0, 0], [0, 0, 0]])
    for point in points:
        phase = np.exp(2j * np.pi * point[0])
        expected = onsite + (phase * hopping + phase.conjugate() * hopping.conj().T) / 2
        np.testing.assert_allclose(model.hamiltonian(point), expected, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(model.solver(point), np.linalg.eigvalsh(expected), atol=1e-12, rtol=1e-12)
    values, vectors = model.eig(points)
    for point, eigenvalues, eigenvectors in zip(points, values, vectors):
        np.testing.assert_allclose(
            model.hamiltonian(point) @ eigenvectors, eigenvectors * eigenvalues, atol=1e-12, rtol=1e-12
        )
        np.testing.assert_allclose(eigenvectors.conj().T @ eigenvectors, np.eye(2), atol=1e-12, rtol=1e-12)


def test_wsvec_averages_equivalent_shifts():
    model = WannierModel.from_directory(FIXTURES / "chain", interpolation="ws_distance")
    points = np.array([[0.07, 0, 0], [0.125, 0, 0], [0.31, 0, 0]])
    phi = 2 * np.pi * points[:, 0]
    expected = 0.15 * (np.cos(phi) + np.cos(3 * phi)) - 0.1 * (np.sin(phi) + np.sin(3 * phi))
    np.testing.assert_allclose(model.energies(points)[0], expected, atol=1e-12, rtol=1e-12)
    auto = WannierModel.from_directory(FIXTURES / "chain", interpolation="auto")
    assert auto.interpolation == "ws_distance"
    np.testing.assert_allclose(auto.energies(points), model.energies(points), atol=1e-12, rtol=1e-12)


def test_expanded_export_does_not_apply_wsvec_twice(tmp_path):
    (tmp_path / "wannier90_hr.dat").write_text(
        "preweighted\n1\n2\n1 1\n-2 0 0 1 1 .15 -.1\n2 0 0 1 1 .15 .1\n"
    )
    # Expanded HR R-list intentionally differs from the informational wsvec list.
    (tmp_path / "wannier90_wsvec.dat").write_text(
        "## use_ws_distance=.true. write_ndegen_applied=.true.\n1 0 0 1 1\n1\n1 0 0\n"
    )
    expanded = WannierModel.from_directory(tmp_path, interpolation="auto")
    assert expanded.interpolation == "ndegen_applied"
    points = np.array([[0.125, 0, 0], [0.2, 0, 0]])
    phi = 4 * np.pi * points[:, 0]
    np.testing.assert_allclose(
        expanded.energies(points)[0], 0.3 * np.cos(phi) - 0.2 * np.sin(phi), atol=1e-12, rtol=1e-12
    )
    with pytest.raises(ValueError, match="already been applied"):
        WannierModel.from_directory(tmp_path, interpolation="ws_distance")
    with pytest.raises(ValueError, match="unit degeneracies"):
        WannierModel.from_directory(FIXTURES / "chain", interpolation="ndegen_applied")


def test_explicit_missing_wsvec_and_unknown_provenance_fail():
    with pytest.raises(ValueError, match="missing interpolation metadata"):
        WannierModel.from_directory(FIXTURES / "fcc", interpolation="auto")
    with pytest.raises(ValueError, match="missing interpolation metadata"):
        WannierModel.from_directory(FIXTURES / "fcc", interpolation="ws_distance")
    with pytest.raises(ValueError, match="Unknown interpolation"):
        WannierModel.from_directory(FIXTURES / "chain", interpolation="guess")


@pytest.mark.parametrize(
    "text, message",
    [
        ("header\n", "incomplete HR header"),
        ("header\n1\n1\n0\n0 0 0 1 1 1 0\n", "positive degeneracies"),
        ("header\n1\n1\n1\n0 0 0 1 1 nan 0\n", "nonfinite"),
        ("header\n1\n1\n1\n0 0 0 2 1 1 0\n", "orbital index"),
        ("header\n1\n1\n1\n0 0 0 1 1 1 0\n0 0 0 1 1 1 0\n", "duplicate"),
        ("header\n2\n1\n1\n0 0 0 1 1 1 0\n", "incomplete HR matrix"),
    ],
)
def test_malformed_hr_is_rejected(tmp_path, text, message):
    source = tmp_path / "bad_hr.dat"
    source.write_text(text)
    with pytest.raises(ValueError, match=message):
        read_hr(source)


def test_wsvec_mismatch_and_truncation_rejected(tmp_path):
    source = tmp_path / "bad_wsvec.dat"
    source.write_text("## use_ws_distance=.true.\n0 0 0 1 1\n2\n0 0 0\n")
    with pytest.raises(ValueError, match="truncated"):
        read_wsvec(source)
    source.write_text("## use_ws_distance=.true.\n0 0 0 1 1\n1\n0 0 0\n")
    with pytest.raises(ValueError, match="do not match"):
        WannierModel.from_directory(FIXTURES / "chain", interpolation="ws_distance", wsvec=source)


def test_shape_contract_first_band_empty_and_validation():
    model = WannierModel.from_directory(FIXTURES / "two_band")
    points = np.array([[0.1, 0, 0], [0.2, 0, 0]])
    all_bands = model.energies(points)
    np.testing.assert_allclose(model.energies(points, bands=0), all_bands[:1], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        model.energies(points, bands=[1, 0]), all_bands[[1, 0]], atol=1e-12, rtol=1e-12
    )
    assert model.energies(np.empty((0, 3))).shape == (2, 0)
    assert model.energies(np.empty((0, 3)), bands=0).shape == (1, 0)
    assert model.energies(points, bands=[]).shape == (0, 2)
    assert model.calculate_energy(points, band_index=0).shape == (2,)
    with pytest.raises(ValueError, match="shape"):
        model.energies([0, 0, 0])
    with pytest.raises(ValueError, match="finite"):
        model.energies([[np.nan, 0, 0]])
    for indices in (True, [-1], [2], [0.1], [[0]]):
        with pytest.raises(ValueError):
            model.energies(points, bands=indices)


def test_serial_batches_and_bounded_parallel_agree():
    model = WannierModel.from_directory(FIXTURES / "two_band")
    points = np.random.default_rng(3).random((27, 3))
    expected = model.energies(points, execution=ExecutionOptions(batch_size=27))
    for options in (
        ExecutionOptions(batch_size=2),
        ExecutionOptions(workers=2, batch_size=3),
        ExecutionOptions(batch_size=256, memory_mb=0.02),
    ):
        np.testing.assert_allclose(
            model.energies(points, execution=options), expected, atol=1e-10, rtol=1e-10
        )
    with pytest.raises(ValueError, match="too small"):
        model.energies(points, execution=ExecutionOptions(memory_mb=1e-9))


def test_nonhermitian_input_fails_and_roundoff_correction_is_recorded():
    invalid = WannierModel([[0, 0, 0]], [[[1, 1], [0, 2]]])
    with pytest.raises(NumericalError, match="Non-Hermitian.*k="):
        invalid.hamiltonian([0, 0, 0])
    rounded = WannierModel([[0, 0, 0]], [[[1, 1 + 1e-12j], [1, 2]]])
    matrix = rounded.hamiltonian([0, 0, 0])
    np.testing.assert_allclose(matrix, matrix.conj().T, atol=0, rtol=0)
    assert 0 < rounded.max_hermiticity_residual < 1e-10


def test_eigensolver_failure_has_context(monkeypatch):
    model = WannierModel.from_directory(FIXTURES / "chain")

    def fail(_):
        raise np.linalg.LinAlgError("failed")

    monkeypatch.setattr(np.linalg, "eigvalsh", fail)
    with pytest.raises(NumericalError, match="k-points starting at"):
        model.energies([[0.125, 0, 0]])


def test_legacy_model_uses_validated_kernels_without_first_band_bug():
    model = LegacyModel(path=FIXTURES / "two_band", num_core=1)
    assert model.calculate_energy([[0.125, 0, 0]], band_index=0).shape == (1,)
    np.testing.assert_allclose(
        model.fourier([0.125, 0, 0]), model.hamiltonian([0.125, 0, 0]), atol=1e-12, rtol=1e-12
    )
    assert model.calculate_energy(np.empty((0, 3))).shape == (2, 0)
    assert model.x.shape == (3, 3, 4)


def test_scheduler_budget_caps_processes_and_native_threads(monkeypatch):
    import queer.execution as execution

    monkeypatch.setattr(execution.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(execution.os, "sched_getaffinity", lambda _: set(range(8)))
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    assert execution.cpu_budget() == 4
    assert ExecutionOptions(workers=16, native_threads=2).resolved().workers == 2
    with pytest.raises(ValueError, match="exceeds CPU allocation"):
        ExecutionOptions(native_threads=5).resolved()
    for kwargs in ({"workers": 0}, {"batch_size": False}, {"native_threads": -1}, {"memory_mb": np.nan}):
        with pytest.raises(ValueError):
            ExecutionOptions(**kwargs)
