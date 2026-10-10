from pathlib import Path

import numpy as np
import pytest

from queer.functions import fd
from queer.geometry.lattice import cartesian_to_fractional, fractional_to_cartesian, reciprocal_lattice
from queer.geometry.mesh import cartesian_mesh
from queer.geometry.paths import interpolate_path, named_k_path, seekpath_points
from queer.geometry.rotations import rotation_matrix
from queer.io.structure import read_fermi_energy, read_poscar, read_qe_reciprocal
from queer.mesh import angstrom2reciprocal, mesh_crystal, reciprocal2angstrom

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "wannier"


def test_nonorthogonal_coordinate_convention_and_legacy_modern_pair():
    direct = np.array([[2.0, 0, 0], [1.0, 3.0, 0], [0.5, 0.25, 4.0]])
    reciprocal = reciprocal_lattice(direct)
    np.testing.assert_allclose(direct @ reciprocal.T, 2 * np.pi * np.eye(3), atol=1e-12, rtol=1e-12)
    points = np.array([[0.2, 0.3, 0.4], [-0.3, 0.1, 0.25]])
    expected = np.array(
        [
            [0.6283185307179586, 0.418879020478639, 0.5235987755982989],
            [-0.9424777960769379, 0.5235987755982988, 0.4777838827334477],
        ]
    )
    np.testing.assert_allclose(fractional_to_cartesian(points, reciprocal), expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(cartesian_to_fractional(expected, reciprocal), points, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        angstrom2reciprocal(reciprocal2angstrom(points, reciprocal), reciprocal),
        points,
        atol=1e-12,
        rtol=1e-12,
    )


def test_mesh_has_explicit_shape_order_and_planar_embedding():
    mesh = cartesian_mesh([2, 3, 1], factor=2, center=False)
    assert mesh.shape == (6, 3)
    np.testing.assert_allclose(mesh[:3], [[0, 0, 0], [0, 2 / 3, 0], [0, 4 / 3, 0]], atol=1e-12, rtol=1e-12)
    centered = cartesian_mesh([2, 3, 1])
    np.testing.assert_allclose(centered.mean(axis=0), 0, atol=1e-12, rtol=1e-12)
    reciprocal = reciprocal_lattice(read_poscar(FIXTURES / "chain" / "POSCAR")[0])
    planar = mesh_crystal(4, reciprocal, dimension=2)
    assert planar.shape == (3, 16)
    np.testing.assert_allclose(cartesian_to_fractional(planar.T, reciprocal)[:, 2], 0, atol=1e-12, rtol=1e-12)
    for counts in (0, [1, 2], [1, 0, 2], True):
        with pytest.raises(ValueError):
            cartesian_mesh(counts)


def test_path_integer_corners_exact_count_and_endpoints():
    corners = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [2, 1, 0]]
    sym, points = interpolate_path(7, corners)
    assert points.shape == (8, 3)
    np.testing.assert_allclose(points[sym], corners, atol=1e-12, rtol=1e-12)
    assert sym[-1] == 7
    with pytest.raises(ValueError, match="nonzero"):
        interpolate_path(3, [[0, 0, 0], [0, 0, 0]])
    with pytest.raises(ValueError, match="one interval"):
        interpolate_path(1, [[0, 0, 0], [1, 0, 0], [1, 1, 0]])


def test_named_path_uses_original_basis_for_rotated_cell(tmp_path):
    seekpath = pytest.importorskip("seekpath")
    cell, positions, types = read_poscar(FIXTURES / "fcc" / "POSCAR")
    rotation = rotation_matrix([0, 0, 1], [1, 1, 1])
    rotated = cell @ rotation.T
    source = tmp_path / "POSCAR"
    source.write_text(
        "rotated fcc\n1\n"
        + "\n".join(" ".join(map(str, row)) for row in rotated)
        + "\nAg\n1\nDirect\n0 0 0\n"
    )
    expected = seekpath.get_path_orig_cell((rotated, positions, types))["point_coords"]
    points = seekpath_points(source)
    for label in ("GAMMA", "X", "W", "K", "L"):
        np.testing.assert_allclose(points[label], expected[label], atol=1e-12, rtol=1e-12)
    path = named_k_path("G-X-W-K-GAMMA-L", 16, source)
    assert path.path.shape == (17, 3)
    assert path.labels == ["Γ", "X", "W", "K", "Γ", "L"]
    assert path.distances[-1] > 0
    np.testing.assert_allclose(path.path[path.sym[1]], points["X"], atol=1e-12, rtol=1e-12)


def test_poscar_cartesian_negative_volume_and_selective_dynamics(tmp_path):
    source = tmp_path / "POSCAR"
    source.write_text(
        "test\n-8\n1 0 0\n0 1 0\n0 0 1\nH\n1\nSelective dynamics\nCartesian\n.25 .5 .75 T T T\n"
    )
    cell, positions, types = read_poscar(source)
    np.testing.assert_allclose(cell, 2 * np.eye(3), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(positions, [[0.25, 0.5, 0.75]], atol=1e-12, rtol=1e-12)
    assert types == [1]
    source.write_text("test\n2 3 4\n1 0 0\n0 1 0\n0 0 1\n1\nDirect\n.25 .5 .75\n")
    cell, positions, _ = read_poscar(source)
    np.testing.assert_allclose(cell, np.diag([2, 3, 4]), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(positions, [[0.25, 0.5, 0.75]], atol=1e-12, rtol=1e-12)


def test_qe_reader_reports_missing_records(tmp_path):
    source = tmp_path / "nscf.out"
    source.write_text(
        "the Fermi energy is 1.25 ev\nthe Fermi energy is 2.5 ev\n"
        "lattice parameter (alat) = 4.0 a.u.\n"
        "b(1) = ( 1 0 0 )\nb(2) = ( 0 1 0 )\nb(3) = ( 0 0 1 )\n"
    )
    assert read_fermi_energy(source) == 2.5
    np.testing.assert_allclose(
        read_qe_reciprocal(source), np.eye(3) * 2 * np.pi / (4 * 0.529177210903), atol=1e-12, rtol=1e-12
    )
    source.write_text("empty output\n")
    with pytest.raises(ValueError, match="Fermi-energy"):
        read_fermi_energy(source)
    with pytest.raises(ValueError, match="incomplete QE"):
        read_qe_reciprocal(source)


def test_rotation_is_proper_for_parallel_opposite_and_general_vectors():
    for source, target in (([0, 0, 1], [1, 1, 1]), ([1, 0, 0], [-1, 0, 0]), ([1, 0, 0], [1, 0, 0])):
        matrix = rotation_matrix(source, target)
        np.testing.assert_allclose(
            matrix @ (np.asarray(source) / np.linalg.norm(source)),
            np.asarray(target) / np.linalg.norm(target),
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(np.linalg.det(matrix), 1, atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError, match="nonzero"):
        rotation_matrix([0, 0, 0], [1, 0, 0])


def test_occupation_handles_extremes_and_zero_temperature():
    np.testing.assert_allclose(fd([-1e6, 0, 1e6], T=0.01), [1, 0.5, 0], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(fd([-1, 0, 1], T=0), [1, 0.5, 0], atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError):
        fd([0], T=-1)
