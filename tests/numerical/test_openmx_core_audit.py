"""Additional independent native format checks found during observable review."""

import struct

import numpy as np
import pytest

from queer.io.openmx import BOHR_ANGSTROM, HARTREE_EV, read_scfout
from queer.models.openmx import OpenMXModel


def matrix_fixture(path, endian="<"):
    records = []

    def write(kind, values):
        values = np.asarray(values).ravel()
        records.append(struct.pack(endian + str(len(values)) + kind, *values))

    uu = np.array([[-1, 0.13], [0.13, 0.5]])
    dd = np.array([[1, -0.2], [-0.2, 2]])
    ud = np.array([[0.31, 0.21], [0.11, 0.41]])
    imaginary_ud_real_component = np.array([[0.07, 0.03], [0.19, -0.17]])
    imaginary_uu = np.array([[0, 0.08], [-0.08, 0]])
    imaginary_dd = np.array([[0, -0.12], [0.12, 0]])
    imaginary_ud = np.array([[0.09, -0.05], [0.15, 0.23]])
    upper = ud + 1j * (imaginary_ud_real_component + imaginary_ud)
    expected = np.block([[uu + 1j * imaginary_uu, upper], [upper.conj().T, dd + 1j * imaginary_dd]])
    write("i", [1, 15, 0, 0, 0, 0])
    write("i", [2])
    write("d", [0, 0, 0, 0])
    write("i", [0, 0, 0, 0])
    for value in (2, 0, 1, 0):
        write("i", [value])
    for matrix in (np.eye(3) / BOHR_ANGSTROM, np.eye(3) * 2 * np.pi * BOHR_ANGSTROM):
        write("d", [[0, *row] for row in matrix])
    write("d", [0, 0, 0, 0])
    for matrix in (uu, dd, ud, imaginary_ud_real_component, imaginary_uu, imaginary_dd, imaginary_ud):
        write("d", matrix / HARTREE_EV)
    write("d", np.eye(2))
    for axis in range(3):
        write("d", np.zeros((2, 2)))
        write("d", (axis + 1) * np.eye(2) / BOHR_ANGSTROM**2)
    for _ in range(3 + 4 + 2):
        write("d", np.zeros((2, 2)))
    write("i", [3])
    write("d", [0, 300, 0, 0, 0, 0, 0, 0, 2, 0])
    lines = [
        "<Definition.of.Atomic.Species",
        "X X2.0-s2 X_PBE19",
        "Definition.of.Atomic.Species>",
        "<Atoms.SpeciesAndCoordinates",
        "1 X 0 0 0",
        "Atoms.SpeciesAndCoordinates>",
    ]
    write("i", [len(lines)])
    records.extend((line.encode() + b"\n").ljust(256, b"\0") for line in lines)
    path.write_bytes(b"".join(records))
    return expected


@pytest.mark.parametrize("endian", ["<", ">"])
def test_full_soc_imaginary_channels_and_second_position_moments(tmp_path, endian):
    source = tmp_path / "soc.scfout"
    expected = matrix_fixture(source, endian)
    raw = read_scfout(source)
    np.testing.assert_allclose(raw.hamiltonian[(0, 0, 0)].toarray(), expected, atol=1e-14)
    model = OpenMXModel.from_scfout(source)
    np.testing.assert_allclose(model.hamiltonian([0, 0, 0]), expected, atol=1e-14)
    np.testing.assert_allclose(
        model.eigenstates([[0, 0, 0]]).energies[:, 0], np.linalg.eigvalsh(expected), atol=1e-13
    )
    for axis in range(3):
        np.testing.assert_allclose(raw.position_moments[axis][1][(0, 0, 0)].toarray(), (axis + 1) * np.eye(2))
    assert raw.metadata["position_order"] == 2
    assert [(row["zeta"], row["spin"]) for row in raw.orbital_metadata] == [
        (0, "up"),
        (1, "up"),
        (0, "down"),
        (1, "down"),
    ]


@pytest.mark.parametrize("unit", ["Ang", "AU", "Frac"])
def test_matching_sidecar_coordinates_convert_units_and_check_lattice(tmp_path, unit):
    source = tmp_path / "soc.scfout"
    matrix_fixture(source)
    binary = bytearray(source.read_bytes())
    # The one-atom x coordinate follows header, cells, lists, and two cell matrices.
    struct.pack_into("<d", binary, 292, 0.3 / BOHR_ANGSTROM)
    source.write_bytes(binary)
    raw = read_scfout(source)
    coordinate = 0.3 / BOHR_ANGSTROM if unit == "AU" else 0.3
    lines = list(raw.input_lines)
    lines[4] = f"1 X {coordinate:.17g} 0 0"
    lines.extend(
        [
            f"Atoms.SpeciesAndCoordinates.Unit {unit}",
            "Atoms.UnitVectors.Unit AU",
            "<Atoms.UnitVectors",
            f"{1 / BOHR_ANGSTROM:.17g} 0 0",
            f"0 {1 / BOHR_ANGSTROM:.17g} 0",
            f"0 0 {1 / BOHR_ANGSTROM:.17g}",
            "Atoms.UnitVectors>",
        ]
    )
    sidecar = tmp_path / "soc.dat"
    sidecar.write_text("\n".join(lines))
    model = OpenMXModel.from_scfout(source, openmx_input=sidecar)
    np.testing.assert_allclose(model.atom_positions, [[0.3, 0, 0]], atol=1e-13)
    lines[4] = f"1 X {coordinate + 0.1:.17g} 0 0"
    sidecar.write_text("\n".join(lines))
    with pytest.raises(ValueError, match="positions/order"):
        read_scfout(source, openmx_input=sidecar)
    lines[4] = f"1 X {coordinate:.17g} 0 0"
    lines[-4] = f"{2 / BOHR_ANGSTROM:.17g} 0 0"
    sidecar.write_text("\n".join(lines))
    with pytest.raises(ValueError, match="lattice"):
        read_scfout(source, openmx_input=sidecar)


def test_malformed_matrix_counts_fail_before_quadratic_layout_allocation(tmp_path, monkeypatch):
    import queer.io.openmx as native

    source = tmp_path / "soc.scfout"
    matrix_fixture(source)
    binary = bytearray(source.read_bytes())
    struct.pack_into("<i", binary, 76, 100)
    source.write_bytes(binary)

    def prohibited(*args, **kwargs):
        raise AssertionError("malformed element counts reached quadratic layout allocation")

    monkeypatch.setattr(native.np, "meshgrid", prohibited)
    with pytest.raises(ValueError, match="truncated declared matrix"):
        read_scfout(source)
