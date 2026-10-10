"""Independent native Wannier operator, sidecar and embedding acceptance cases."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from queer.execution import ExecutionOptions
from queer.io.wannier import (
    BOHR_ANGSTROM,
    normalized_operators,
    read_centres_xyz,
    read_tb,
    read_wout_lattice,
    read_wsvec,
)
from queer.models.wannier import WannierModel
from queer.observables.geometry import band_pair_geometry


def write_tb(path, translations, hopping, position, degeneracies, lattice=np.eye(3)):
    n = hopping.shape[1]
    lines = ["independent synthetic TB"]
    lines.extend(" ".join(map(str, row)) for row in lattice)
    lines.extend([str(n), str(len(translations)), " ".join(map(str, degeneracies))])
    for index, translation in enumerate(translations):
        lines.append(" ".join(map(str, translation)))
        for m, j in reversed(list(np.ndindex(n, n))):
            value = hopping[index, m, j]
            lines.append(f"{m + 1} {j + 1} {value.real:.17e} {value.imag:.17e}")
    if position is not None:
        for index in reversed(range(len(translations))):
            lines.append(" ".join(map(str, translations[index])))
            for m, j in reversed(list(np.ndindex(n, n))):
                values = position[index, m, j]
                parts = [str(m + 1), str(j + 1)]
                parts.extend(f"{part:.17e}" for value in values for part in (value.real, value.imag))
                lines.append(" ".join(parts))
    path.write_text("\n".join(lines) + "\n")


def test_tb_shuffled_rows_positions_and_tb_only_model(tmp_path):
    translations = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    a = np.array([[0.1 + 0.02j, 0.3j], [0.02, -0.2 + 0.04j]])
    hopping = np.array([np.diag([-1, 1]), a, a.conj().T])
    position = np.zeros((3, 2, 2, 3), complex)
    position[0, 0, 0], position[0, 1, 1] = [0.2, 0.1, 0], [0.7, -0.3, 0]
    position[1, :, :, 0] = a / 3
    position[2, :, :, 0] = a.conj().T / 3
    write_tb(tmp_path / "wannier90_tb.dat", translations, hopping, position, [1, 2, 2])
    raw = read_tb(tmp_path / "wannier90_tb.dat")
    assert_allclose(raw.position, position)
    assert_allclose(raw.hopping, hopping)
    model = WannierModel.from_directory(tmp_path)
    explicit_tb_only = WannierModel.from_directory(tmp_path, hr=None, tb="wannier90_tb.dat")
    point = [0.17, 0.1, 0.3]
    phase = np.exp(2j * np.pi * 0.17)
    assert_allclose(model.hamiltonian(point), hopping[0] + (phase * a + a.conj().T / phase) / 2)
    assert_allclose(explicit_tb_only.hamiltonian(point), model.hamiltonian(point))
    assert_allclose(model.position(point)[0], np.diag([0.2, 0.7]) + (phase * a + a.conj().T / phase) / 6)
    assert model.metadata["embedding"] == "full_position"
    assert "tb" in model.source_files
    assert not raw.position.flags.writeable


def test_ws_weights_are_identical_for_hamiltonian_and_positions(tmp_path):
    translations = np.array([[1, 0, 0], [-1, 0, 0]])
    hopping = np.array([[[0.3 + 0.2j]], [[0.3 - 0.2j]]])
    position = hopping[..., None] * np.array([2.0, 3.0, 4.0])
    write_tb(tmp_path / "seed_tb.dat", translations, hopping, position, [2, 2])
    (tmp_path / "seed_wsvec.dat").write_text(
        "## use_ws_distance=.true.\n1 0 0 1 1\n2\n0 0 0\n2 0 0\n-1 0 0 1 1\n2\n0 0 0\n-2 0 0\n"
    )
    raw = read_tb(tmp_path / "seed_tb.dat")
    mapping = read_wsvec(tmp_path / "seed_wsvec.dat")
    expanded, h, r = normalized_operators(raw, "ws_distance", mapping)
    assert expanded[:, 0].tolist() == [-3, -1, 1, 3]
    assert_allclose(r, h[..., None] * [2, 3, 4])
    model = WannierModel.from_directory(tmp_path, hr="seed_tb.dat", interpolation="auto")
    assert_allclose(
        model.position([0.125, 0, 0])[:, 0, 0], model.hamiltonian([0.125, 0, 0])[0, 0] * np.array([2, 3, 4])
    )


def test_centres_wout_and_exact_diagonal_links(tmp_path):
    (tmp_path / "seed_hr.dat").write_text(
        "synthetic\n2\n1\n1\n0 0 0 1 1 -1 0\n0 0 0 1 2 .3 .1\n0 0 0 2 1 .3 -.1\n0 0 0 2 2 1 0\n"
    )
    (tmp_path / "seed_centres.xyz").write_text("3\ncentres and atoms\nX .1 .2 .3\nX .6 .4 .7\nC 0 0 0\n")
    (tmp_path / "seed.wout").write_text("Lattice Vectors (Bohr)\na_1 2 0 0\na_2 0 3 0\na_3 0 0 4\n")
    centres, atoms = read_centres_xyz(tmp_path / "seed_centres.xyz")
    assert len(atoms) == 1
    assert_allclose(read_wout_lattice(tmp_path / "seed.wout"), np.diag([2, 3, 4]) * BOHR_ANGSTROM)
    model = WannierModel.from_directory(tmp_path, hr="seed_hr.dat", ef=0.5)
    bare = WannierModel(model.translations, model.hopping_matrices, lattice=model.lattice)
    assert_allclose(model.hamiltonian([0.1, 0.2, 0.3]), bare.hamiltonian([0.1, 0.2, 0.3]), atol=0)
    assert_allclose(model.position([0.1, 0, 0])[:, 0, 0], centres[0])
    delta = np.array([0.1, -0.2, 0.3])
    expected = np.diag(np.exp(-1j * (centres @ (delta @ model.g_vec))))
    assert_allclose(model.basis_overlap([0, 0, 0], delta), expected)
    assert set(model.source_files) == {"hr", "centres", "wout"}
    points = np.array([[0.1, 0, 0], [0.3, 0.2, 0.1]])
    serial = model.eigenstates(points, execution=ExecutionOptions(batch_size=1))
    parallel = model.eigenstates(points, execution=ExecutionOptions(workers=2, batch_size=1))
    assert_allclose(serial.energies, parallel.energies)
    assert_allclose(serial.energies - 0.5, model.energies(points))
    assert model.eigenstates(np.empty((0, 3)), bands=0).coefficients.shape == (0, 2, 1)


def test_cartesian_operator_derivatives_and_complex_continuation():
    translations = [[0, 0, 0], [1, 0, 0], [-1, 0, 0]]
    a = np.array([[0.1j, 0.3], [-0.2j, -0.1]])
    h = np.array([np.diag([-1, 1]), a, a.conj().T])
    r = h[..., None] * np.array([0.2, 0.3, 0.4])
    lattice = np.array([[2, 0.3, 0], [0.1, 1.3, 0], [0, 0, 5]])
    model = WannierModel(translations, h, position=r, lattice=lattice)
    point = np.array([0.17, 0.12, 0.3])
    eps = 1e-5
    for axis in range(3):
        step = np.eye(3)[axis] @ np.linalg.inv(model.g_vec) * eps
        derivative = model.derivatives(point)
        assert_allclose(
            derivative["H"][axis],
            (model.hamiltonian(point + step) - model.hamiltonian(point - step)) / (2 * eps),
            atol=2e-9,
        )
        assert_allclose(
            derivative["r"][axis],
            (model.position(point + step) - model.position(point - step)) / (2 * eps),
            atol=2e-9,
        )
    complex_point = point.astype(complex) + [0.08j, 0, 0]
    expected = sum(
        np.exp(2j * np.pi * np.dot(complex_point, translation)) * matrix
        for translation, matrix in zip(translations, h, strict=True)
    )
    actual = model.analytic_hamiltonian(complex_point)
    assert_allclose(actual, expected)
    assert np.linalg.norm(actual - actual.conj().T) > 0.1
    with pytest.raises(ValueError, match="complex"):
        model.hamiltonian(complex_point)
    with pytest.raises(ValueError, match="complex"):
        model.energies([complex_point])
    with pytest.raises(ValueError, match="complex"):
        model.eigenstates([complex_point])
    assert_allclose(model.analytic_hamiltonian(np.array([point, complex_point]))[1], actual)


def test_home_cell_embedding_geometry_is_invariant():
    translations = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    a = np.array([[0.1 + 0.02j, 0.3], [0.2j, -0.1]])
    hopping = np.array([np.diag([-1, 1]), a, a.conj().T])
    centres = np.array([[0.2, 0.3, 0], [0.7, -0.2, 0]])
    shifts = np.array([[1, 0, 0], [0, -1, 0]])
    model = WannierModel(translations, hopping, centres=centres, lattice=np.eye(3))
    expanded = {}
    for translation, matrix in zip(translations, hopping, strict=True):
        for m, n in np.ndindex(2, 2):
            key = tuple(translation + shifts[m] - shifts[n])
            expanded.setdefault(key, np.zeros((2, 2), complex))[m, n] += matrix[m, n]
    keys = sorted(expanded)
    changed = WannierModel(keys, [expanded[key] for key in keys], centres=centres + shifts, lattice=np.eye(3))
    points = [[0.13, 0.27, 0], [0.22, -0.14, 0]]
    first = band_pair_geometry(model, points, bands=(0, 1))
    second = band_pair_geometry(changed, points, bands=(0, 1))
    for key in ("energies", "gap", "pair_metric", "total_metric", "curvature"):
        assert_allclose(first[key], second[key], atol=2e-12)


@pytest.mark.parametrize("section", ["matrix", "position"])
def test_tb_incomplete_sections_are_rejected(tmp_path, section):
    path = tmp_path / "bad_tb.dat"
    write_tb(path, [[0, 0, 0]], np.array([[[-1, 0], [0, 1]]], complex), np.zeros((1, 2, 2, 3)), [1])
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:10] if section == "matrix" else lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        read_tb(path)


def test_xyz_truncated_nonfinite_and_extra_records(tmp_path):
    path = tmp_path / "bad.xyz"
    for text in ("2\ncomment\nX 0 0 0\n", "1\ncomment\nX nan 0 0\n", "1\ncomment\nX 0 0 0\nX 1 0 0\n"):
        path.write_text(text)
        with pytest.raises(ValueError):
            read_centres_xyz(path)
