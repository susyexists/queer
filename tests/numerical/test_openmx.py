"""Independent binary and analytical references for native OpenMX machinery."""

import struct
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import cholesky, solve_triangular

from queer import ExecutionOptions, OpenMXModel
from queer.io.openmx import BOHR_ANGSTROM, HARTREE_EV, read_scfout
from queer.models.wannier import NumericalError, WannierModel
from queer.observables.response import spectral_from_eig, spectral_from_green


def write_scfout(path, *, endian="<", spin=0, center=0.3, version=3):
    """A one-atom chain written from the binary specification with struct.pack."""
    records = []

    def ints(values):
        records.append(struct.pack(endian + str(len(values)) + "i", *values))

    def doubles(values):
        records.append(struct.pack(endian + str(len(values)) + "d", *values))

    ints([1, 4 * version + spin, 0, 0, 0, 2])
    ints([1])
    translations = [0, 1, -1]
    doubles([x for r in translations for x in (0, r / BOHR_ANGSTROM, 0, 0)])
    ints([x for r in translations for x in (0, r, 0, 0)])
    ints([1])
    ints([2])
    ints([1, 1, 1])
    ints([0, 1, 2])
    doubles([x for row in np.eye(3) / BOHR_ANGSTROM for x in (0, *row)])
    doubles([x for row in np.eye(3) * (2 * np.pi * BOHR_ANGSTROM) for x in (0, *row)])
    doubles([0, center / BOHR_ANGSTROM, 0, 0])
    for channel in range(spin + 1):
        onsite = (1, 2)[channel] if spin == 1 else (1, 2, 0.3, 0.4)[channel]
        doubles(
            [
                onsite / HARTREE_EV,
                0.2 / HARTREE_EV if channel < 2 else 0,
                0.2 / HARTREE_EV if channel < 2 else 0,
            ]
        )
    if spin == 3:
        for _ in range(3):
            doubles([0, 0, 0])
    doubles([1, 0.1, 0.1])
    for axis in range(3):
        doubles([0, 0.05 / BOHR_ANGSTROM if axis == 0 else 0, -0.05 / BOHR_ANGSTROM if axis == 0 else 0])
    for _ in range(3):
        doubles([0, 0, 0])
    for _ in range(spin + 1):
        doubles([0.5, 0, 0])
    for _ in range(2):
        doubles([0, 0, 0])
    ints([3])
    doubles([0.2 / HARTREE_EV, 300, 0, 0, 0, 0, 0, 0, 1, 0])
    lines = [
        "<Definition.of.Atomic.Species",
        "X X5.0-s1 X_PBE19",
        "Definition.of.Atomic.Species>",
        "<Atoms.SpeciesAndCoordinates",
        "1 X 0 0 0",
        "Atoms.SpeciesAndCoordinates>",
    ]
    ints([len(lines)])
    for line in lines:
        # Deliberate stale footer padding must not become a second input line.
        records.append((line.encode() + b"\nSTALE").ljust(256, b"\0"))
    path.write_bytes(b"".join(records))
    return path


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("spin", [0, 1, 3])
def test_binary_modes_and_independent_energies(tmp_path, endian, spin):
    source = write_scfout(tmp_path / "chain.scfout", endian=endian, spin=spin)
    raw = read_scfout(source)
    model = OpenMXModel.from_scfout(source)
    kpoints = [[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0]]
    numerator = np.array([1.4, 1, 0.6])
    metric = np.array([1.2, 1, 0.8])
    expected = numerator / metric
    if spin == 1:
        expected = np.array([expected, (numerator + 1) / metric])
        assert_allclose(model.energies(kpoints, spin="down"), expected[1:] - 0.2, atol=1e-12)
        assert model.band_channels.tolist() == [0, 1]
    elif spin == 3:
        splitting = np.sqrt(0.5)
        expected = np.array([(numerator + 0.5 - splitting) / metric, (numerator + 0.5 + splitting) / metric])
    else:
        expected = expected[None]
    assert_allclose(model.eigenstates(kpoints).energies, expected, atol=1e-12)
    assert raw.metadata["orbital_labels_complete"]
    assert all("STALE" not in line for line in raw.input_lines)
    assert len(raw.position_moments) == len(raw.momentum_components) == 3
    assert len(raw.density_components) == spin + 1
    assert len(raw.imaginary_density_components) == 2
    assert_allclose(model.lattice, np.eye(3), atol=1e-12)
    assert model.spin_degeneracy == (2 if spin == 0 else 1)
    assert_allclose(raw.position[0][(1, 0, 0)].toarray()[0, 0], 0.05 + 0.03, atol=1e-12)


def test_scalar_derivatives_and_position_covariance(tmp_path):
    model = OpenMXModel.from_scfout(write_scfout(tmp_path / "chain.scfout"))
    k = np.array([0.25, 0, 0])
    h, s = model.hamiltonian(k)[0, 0].real, model.overlap(k)[0, 0].real
    derivative = model.derivatives(k)
    energy_derivative = (derivative["H"][0, 0, 0] * s - h * derivative["S"][0, 0, 0]) / s**2
    assert_allclose(energy_derivative, -0.2, atol=1e-12)
    r = model.position(k)
    assert_allclose(r - r.conj().transpose(0, 2, 1), -1j * derivative["S"], atol=1e-12)
    delta = 1e-5
    for key, operator in [("H", model.hamiltonian), ("S", model.overlap), ("r", model.position)]:
        finite = (operator(k + [delta / (2 * np.pi), 0, 0]) - operator(k - [delta / (2 * np.pi), 0, 0])) / (
            2 * delta
        )
        assert_allclose(derivative[key][0], finite, atol=1e-10)


def test_complex_basis_covariance_and_independent_cholesky():
    h = np.array([[1, 0.2 + 0.3j], [0.2 - 0.3j, 2]])
    s = np.array([[1, 0.1j], [-0.1j, 1.4]])
    change = np.array([[1.2, 0.2j], [0.3, 0.8]])
    a = OpenMXModel([[0, 0, 0]], [h], [s], spin_degeneracy=1)
    b = OpenMXModel(
        [[0, 0, 0]], [change.conj().T @ h @ change], [change.conj().T @ s @ change], spin_degeneracy=1
    )
    solution = a.eigenstates([[0, 0, 0]])
    factor = cholesky(s, lower=True)
    left = solve_triangular(factor, h, lower=True)
    transformed = solve_triangular(factor, left.conj().T, lower=True).conj().T
    assert_allclose(solution.energies[:, 0], np.linalg.eigvalsh(transformed), atol=1e-12)
    assert_allclose(a.energies([[0, 0, 0]]), b.energies([[0, 0, 0]]), atol=1e-12)
    c = solution.coefficients[0]
    assert_allclose(c.conj().T @ s @ c, np.eye(2), atol=1e-12)
    assert_allclose(h @ c, s @ c * solution.energies[:, 0], atol=1e-12)


def test_orthogonal_and_parallel_eigenstate_parity(tmp_path):
    translations = [[0, 0, 0], [1, 0, 0], [-1, 0, 0]]
    h = np.array([[[0, 0.2], [0.2, 1]], [[0.1, 0], [0, 0.3]], [[0.1, 0], [0, 0.3]]])
    native = OpenMXModel(translations, h, spin_degeneracy=1)
    wannier = WannierModel(translations, h)
    points = np.column_stack((np.linspace(0, 1, 11), np.zeros((11, 2))))
    assert_allclose(native.energies(points), wannier.energies(points), atol=1e-12)
    serial = native.eigenstates(points, execution=ExecutionOptions(batch_size=2))
    parallel = native.eigenstates(points, execution=ExecutionOptions(workers=2, batch_size=3))
    assert_allclose(serial.energies, parallel.energies, atol=1e-12)
    assert_allclose(serial.coefficients, parallel.coefficients, atol=1e-12)


@pytest.mark.parametrize("metric", [[[0]], [[-1]]])
def test_bad_metric_rejected(metric):
    model = OpenMXModel([[0, 0, 0]], [[[1]]], [metric])
    with pytest.raises(NumericalError, match="Positive-definite"):
        model.energies([[0, 0, 0]])


def test_spin_product_basis_metric_and_occupation_are_explicit():
    with pytest.raises(ValueError, match="duplicated spin-independent"):
        OpenMXModel([[0, 0, 0]], [np.eye(2)], [np.diag([1, 2])], spin_mode="soc")
    with pytest.raises(ValueError, match="degeneracy one"):
        OpenMXModel([[0, 0, 0]], [np.eye(2)], spin_mode="soc", spin_degeneracy=2)


def test_malformed_binary_rejected(tmp_path):
    source = write_scfout(tmp_path / "bad.scfout", version=2)
    with pytest.raises(ValueError, match="version 2"):
        read_scfout(source)
    source = write_scfout(source)
    source.write_bytes(source.read_bytes()[:-200])
    with pytest.raises(ValueError, match="truncated"):
        read_scfout(source)


def test_corrupt_orbital_counts_rejected_before_quadratic_layout(tmp_path, monkeypatch):
    import queer.io.openmx as reader

    source = write_scfout(tmp_path / "bad_counts.scfout")
    payload = bytearray(source.read_bytes())
    struct.pack_into("<i", payload, 24 + 4 + 3 * 32 + 3 * 16, 100)
    source.write_bytes(payload)

    def prohibited_layout(*args, **kwargs):
        raise AssertionError("Corrupt declared matrices must fail before allocating quadratic indices")

    monkeypatch.setattr(reader.np, "meshgrid", prohibited_layout)
    with pytest.raises(ValueError, match="truncated declared matrix"):
        reader.read_scfout(source)


def test_sidecar_orbital_metadata_and_supplementary_structure_validation(tmp_path):
    source = write_scfout(tmp_path / "chain.scfout")
    sidecar = tmp_path / "chain.dat"
    raw = read_scfout(source)
    sidecar.write_text("\n".join(raw.input_lines).replace("1 X 0 0 0", "1 X 0.3 0 0"))
    assert read_scfout(source, openmx_input=sidecar).metadata["orbital_labels_complete"]
    sidecar.write_text(sidecar.read_text().replace("1 X 0.3 0 0", "2 X 0.3 0 0"))
    with pytest.raises(ValueError, match="counts"):
        read_scfout(source, openmx_input=sidecar)
    sidecar.write_text("\n".join(raw.input_lines).replace("1 X 0 0 0", "1 X 0.4 0 0"))
    with pytest.raises(ValueError, match="positions/order"):
        read_scfout(source, openmx_input=sidecar)
    sidecar.write_text(
        "Atoms.SpeciesAndCoordinates.Unit AU\n"
        + "\n".join(raw.input_lines).replace("1 X 0 0 0", f"1 X {0.3 / BOHR_ANGSTROM:.17g} 0 0")
    )
    read_scfout(source, openmx_input=sidecar)
    model = OpenMXModel.from_scfout(source)
    poscar = tmp_path / "POSCAR"
    poscar.write_text("X\n1\n1 0 0\n0 1 0\n0 0 1\nX\n1\nDirect\n0.4 0 0\n")
    with pytest.raises(ValueError, match="positions/order"):
        model.kpath("G-X", 4, poscar=poscar)
    poscar.write_text(poscar.read_text().replace("0.4 0 0", "0.3 0 0"))
    assert model.kpath("G-X", 4, poscar=poscar).path.shape == (5, 3)


def test_nonorthogonal_green_sum_and_reference_shift():
    h = np.array([[1, 0.3j], [-0.3j, 2]])
    s = np.array([[1, 0.1], [0.1, 1.2]])
    model = OpenMXModel([[0, 0, 0]], [h], [s], fermi_energy=0.7)
    frequencies = np.linspace(-5, 5, 10001)
    expected = spectral_from_eig(model.energies([[0, 0, 0]])[:, 0], frequencies, 0.03)
    actual = spectral_from_green(h, frequencies, 0.03, s, fermi_energy=0.7)
    assert_allclose(actual, expected, atol=1e-11)
    assert_allclose(np.trapezoid(actual, frequencies), 2, atol=0.009)


@pytest.mark.parametrize("name,dimension", [("FeCl2", 154), ("GaAs_soc", 104)])
def test_material_bands_against_independent_cholesky(name, dimension):
    path = Path(__file__).resolve().parents[2] / "data/references/openmx" / f"{name}.scfout"
    if not path.is_file():
        pytest.skip("Fetch pinned OpenMX references with scripts/runs/openmx_benchmark.py --fetch")
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        model = OpenMXModel.from_scfout(path)
        assert model.nbnd == dimension
        points = [[0, 0, 0], [0.17, 0.23, 0.11]]
        solution = model.eigenstates(points)
        for k, point in enumerate(points):
            h, s = model.hamiltonian(point), model.overlap(point)
            factor = cholesky(s, lower=True)
            left = solve_triangular(factor, h, lower=True)
            transformed = solve_triangular(factor, left.conj().T, lower=True).conj().T
            assert_allclose(solution.energies[:, k], np.linalg.eigvalsh(transformed), atol=1e-9)
            c = solution.coefficients[k]
            assert_allclose(c.conj().T @ s @ c, np.eye(dimension), atol=1e-10)
