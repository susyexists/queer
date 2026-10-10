"""Analytical geometry, format interoperability and magnetic expansion references."""

from __future__ import annotations

import builtins
import io
from itertools import product

import numpy as np
import pytest

from queer.geometry.structures import (
    add_vacuum,
    apply_transforms,
    compute_bonds,
    expand_variants,
    make_supercell,
    minimum_distance,
    minimum_image_vectors,
    neighbor_list,
    remove_atoms,
)
from queer.geometry.symmetry import brillouin_zone_faces, compute_band_path, primitive_structure
from queer.io.atomistic import (
    parse_cif,
    parse_poscar,
    poscar_text,
    read_extxyz,
    read_structure,
    read_trajectory,
    write_cif,
    write_extxyz,
    write_structure,
    write_trajectory,
)
from queer.structures import Evaluation, Frame, Structure


def decorated():
    return Structure(
        [[3, 0, 0], [1, 4, 0], [0.2, 0.4, 5]],
        [[0, 0, 0], [1, 1, 1]],
        [8, 1],
        magnetic_moments=[[1, 2, 3], [-1, 0, 0]],
        fixed_coordinates=[[True, False, False], [False] * 3],
        arrays={"REF_forces": [[1, 2, 3], [4, 5, 6]]},
        metadata={"REF_energy": -2.0, "REF_stress": np.diag([1, 2, 3]), "constraint_basis": "fractional"},
    )


def test_structure_validation_and_independent_copy():
    atoms = decorated()
    copy = atoms.copy()
    copy.positions[0, 0] = 100
    copy.arrays["REF_forces"][0, 0] = 100
    assert atoms.positions[0, 0] == 0
    assert atoms.arrays["REF_forces"][0, 0] == 1
    assert atoms.masses[1] == pytest.approx(1.008)
    with pytest.raises(ValueError, match="nonsingular"):
        Structure(np.zeros((3, 3)), [[0, 0, 0]], [1])
    with pytest.raises(ValueError, match="positive"):
        atoms.copy(masses=[0, 1])


def test_poscar_scaling_selective_and_stable_permutation(tmp_path):
    atoms = decorated()
    text, order = poscar_text(atoms, sort=True)
    assert order.tolist() == [1, 0]
    parsed = parse_poscar(text)
    np.testing.assert_allclose(parsed.positions, atoms.positions[order])
    np.testing.assert_array_equal(parsed.fixed_coordinates, atoms.fixed_coordinates[order])
    assert parsed.symbols == ["H", "O"]
    target = tmp_path / "POSCAR"
    write_structure(target, atoms)
    np.testing.assert_allclose(read_structure(target).positions, atoms.positions)
    parsed = parse_poscar("H\n-8\n1 0 0\n0 1 0\n0 0 1\nH\n1\nCartesian\n.25 .5 .75\n")
    np.testing.assert_allclose(parsed.cell, 2 * np.eye(3))
    np.testing.assert_allclose(parsed.positions, [[0.5, 1, 1.5]])
    v4 = parse_poscar("H\n1\n1 0 0\n0 1 0\n0 0 1\n1\nDirect\n0 0 0\n")
    assert v4.symbols == ["H"]


def test_poscar_constraint_basis_conversion_and_unrepresentable_subspace():
    atoms = Structure(
        [[0, 3, 0], [4, 0, 0], [0, 0, 5]],
        [[0, 0, 0]],
        [1],
        fixed_coordinates=[[True, False, False]],
    )
    parsed = parse_poscar(write_structure(atoms))
    np.testing.assert_array_equal(parsed.fixed_coordinates, [[False, True, False]])
    skew = decorated().copy(metadata={"constraint_basis": "cartesian"})
    with pytest.raises(ValueError, match="incompatible with POSCAR"):
        write_structure(skew)


def test_extxyz_multiframe_ref_tensor_and_native_fields():
    atoms = decorated()
    frame = Frame(
        atoms,
        np.full((2, 3), 0.25),
        step=20,
        time_fs=10,
        evaluation=Evaluation(-3, np.zeros((2, 3)), np.diag([0.1, 0.2, 0.3])),
    )
    text = write_extxyz([frame, frame])
    frames = read_extxyz(text)
    assert len(frames) == 2
    for parsed in frames:
        np.testing.assert_allclose(parsed.structure.cell, atoms.cell)
        np.testing.assert_allclose(parsed.structure.metadata["REF_stress"], atoms.metadata["REF_stress"])
        np.testing.assert_allclose(parsed.evaluation.stress_eV_A3, frame.evaluation.stress_eV_A3)
        np.testing.assert_allclose(parsed.structure.arrays["REF_forces"], atoms.arrays["REF_forces"])
        np.testing.assert_allclose(parsed.velocities, frame.velocities)
        assert parsed.step == 20
        assert parsed.evaluation.energy_eV == -3


def test_extxyz_optional_ase_bidirectional_tensor_interoperability():
    ase = pytest.importorskip("ase")
    from ase.io import read, write

    atoms = decorated().copy(metadata={"REF_energy": -2.0, "REF_stress": np.arange(9).reshape(3, 3)})
    frame = Frame(atoms, evaluation=Evaluation(-3, np.ones((2, 3)), np.diag([0.1, 0.2, 0.3])))
    external = read(io.StringIO(write_extxyz(frame)), format="extxyz")
    assert isinstance(external, ase.Atoms)
    np.testing.assert_allclose(external.info["REF_stress"], atoms.metadata["REF_stress"])
    np.testing.assert_allclose(external.arrays["REF_forces"], atoms.arrays["REF_forces"])
    np.testing.assert_allclose(external.get_stress(voigt=False), frame.evaluation.stress_eV_A3)
    buffer = io.StringIO()
    write(buffer, external, format="extxyz")
    native = read_extxyz(buffer.getvalue())[0]
    np.testing.assert_allclose(native.structure.cell, atoms.cell)
    np.testing.assert_allclose(native.structure.metadata["REF_stress"], atoms.metadata["REF_stress"])
    np.testing.assert_allclose(native.evaluation.stress_eV_A3, frame.evaluation.stress_eV_A3)
    np.testing.assert_allclose(native.structure.arrays["REF_forces"], atoms.arrays["REF_forces"])


def test_ulm_roundtrip_and_optional_ase_interoperability(tmp_path):
    atoms = decorated()
    frame = Frame(
        atoms,
        np.full((2, 3), 0.25),
        step=20,
        time_fs=10,
        evaluation=Evaluation(-3, np.zeros((2, 3)), np.diag([0.1, 0.2, 0.3])),
    )
    data = write_trajectory([frame, frame])
    parsed = read_trajectory(data)
    assert len(parsed) == 2
    np.testing.assert_allclose(parsed[0].velocities, frame.velocities)
    np.testing.assert_allclose(parsed[0].evaluation.stress_eV_A3, frame.evaluation.stress_eV_A3)
    np.testing.assert_allclose(parsed[0].structure.magnetic_moments, atoms.magnetic_moments)
    np.testing.assert_array_equal(parsed[0].structure.arrays["REF_forces"], atoms.arrays["REF_forces"])
    target = tmp_path / "sample.traj"
    write_trajectory(target, [frame])
    assert read_trajectory(target)[0].step == 20
    ulm_path = tmp_path / "sample.ulm"
    ulm_path.write_bytes(write_trajectory([frame]))
    np.testing.assert_allclose(read_structure(ulm_path).positions, atoms.positions)
    raw_path = tmp_path / "binary.frames"
    raw_path.write_bytes(write_trajectory([frame]))
    np.testing.assert_allclose(read_structure(raw_path, format="ulm").positions, atoms.positions)
    asymmetric = Frame(atoms, evaluation=Evaluation(-3, np.zeros((2, 3)), np.arange(9).reshape(3, 3)))
    with pytest.raises(ValueError, match="symmetric"):
        write_trajectory([asymmetric])
    with pytest.raises(ValueError, match="Truncated|Invalid"):
        read_trajectory(data[:-30])
    ase = pytest.importorskip("ase")
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io.trajectory import TrajectoryReader, TrajectoryWriter

    reader = TrajectoryReader(io.BytesIO(data))
    assert reader.ase_version is None
    assert reader.backend.queer_producer["name"] == "queer"
    from queer import __version__

    assert reader.backend.queer_producer["version"] == __version__
    external = reader[0]
    np.testing.assert_allclose(external.positions, atoms.positions)
    assert external.info["step"] == 20
    assert external.get_potential_energy() == -3
    np.testing.assert_allclose(external.get_velocities() * 0.09822694788464063, frame.velocities)
    external.calc = SinglePointCalculator(
        external, energy=-4, forces=np.ones((2, 3)), stress=np.arange(6, dtype=float)
    )
    external_path = tmp_path / "external.traj"
    with TrajectoryWriter(external_path, "w") as writer:
        writer.write(external)
        writer.write(external)
    native = read_trajectory(external_path)
    assert len(native) == 2
    assert native[1].evaluation.energy_eV == -4
    np.testing.assert_allclose(native[1].evaluation.forces_eV_A, np.ones((2, 3)))
    np.testing.assert_array_equal(native[1].structure.arrays["REF_forces"], atoms.arrays["REF_forces"])
    assert isinstance(external, ase.Atoms)


def test_ulm_native_array_extension_and_reserved_metadata_validation():
    atoms = decorated().copy(
        arrays={"labels": np.array(["μ", "H"]), "vectors": np.ones((2, 3), complex) * 1j}
    )
    restored = read_trajectory(write_trajectory([Frame(atoms)]))[0].structure
    for name, values in atoms.arrays.items():
        np.testing.assert_array_equal(restored.arrays[name], values)
        assert restored.arrays[name].dtype == values.dtype
    assert "_queer_atom_arrays" not in restored.metadata
    with pytest.raises(ValueError, match="reserved"):
        write_trajectory([Frame(atoms.copy(metadata={"_queer_atom_arrays": {}}))])
    with pytest.raises(ValueError, match="non-object"):
        write_trajectory([Frame(atoms.copy(arrays={"unsafe": np.array([{}, {}], object)}))])


def test_exact_triclinic_neighbors_minimum_images_and_self_images():
    cell = np.array([[1, 0, 0], [0.99, 0.1, 0], [0, 0, 2]])
    atoms = Structure(cell, [[0, 0, 0], [0.6, 0.05, 0]], [1, 1])
    vector = np.array([0.6, 0.05, 0])
    expected = min(
        (vector + np.array(image) @ cell for image in product(range(-8, 9), repeat=3)), key=np.linalg.norm
    )
    np.testing.assert_allclose(minimum_image_vectors(vector, cell), expected)
    i, j, offset, distance, vectors = neighbor_list(atoms, 0.25)
    np.testing.assert_allclose(vectors, atoms.positions[j] + offset @ cell - atoms.positions[i])
    assert np.any(i == j)
    assert np.all(distance <= 0.25 + 1e-12)
    singleton = Structure(cell, [[0, 0, 0]], [1])
    assert minimum_distance(singleton) == pytest.approx(np.sqrt(0.01**2 + 0.1**2))
    assert len(compute_bonds(singleton, max_distance=0.2)) >= 1


def test_general_supercell_and_transforms_preserve_atom_fields():
    atoms = decorated()
    matrix = [[2, 1, 0], [0, 1, 0], [0, 0, 1]]
    enlarged = make_supercell(atoms, matrix)
    assert enlarged.natoms == 4
    np.testing.assert_allclose(enlarged.cell, np.asarray(matrix) @ atoms.cell)
    np.testing.assert_allclose(enlarged.arrays["REF_forces"], np.tile(atoms.arrays["REF_forces"], (2, 1)))
    np.testing.assert_allclose(enlarged.magnetic_moments, np.tile(atoms.magnetic_moments, (2, 1)))
    reduced = remove_atoms(enlarged, [0, 2])
    assert reduced.symbols == ["H", "H"]
    np.testing.assert_allclose(reduced.magnetic_moments, [[-1, 0, 0], [-1, 0, 0]])
    repeated = make_supercell(reduced, np.diag([2, 1, 1]))
    assert repeated.metadata["source_indices"] == [1, 1, 1, 1]
    vacuum = add_vacuum(atoms.copy(fixed_coordinates=None), 3)
    np.testing.assert_allclose(
        vacuum.positions[1] - vacuum.positions[0], atoms.positions[1] - atoms.positions[0]
    )
    recipe = [
        {"type": "gaussianDistortion", "params": {"count": 3, "sigmaMin": 0.02, "sigmaMax": 0.03, "seed": 12}}
    ]
    variants = expand_variants(recipe)
    assert len(variants) == 3
    np.testing.assert_allclose(apply_transforms(atoms, variants[0]).positions, atoms.positions)
    np.testing.assert_allclose(
        apply_transforms(atoms, variants[1]).positions, apply_transforms(atoms, variants[1]).positions
    )
    with pytest.raises(ValueError, match="Unknown"):
        apply_transforms(atoms, [{"type": "imaginary"}])


def test_cell_basis_changes_preserve_fractional_constraint_subspaces():
    atoms = Structure(
        [[3, 0, 0], [1, 4, 0], [0.2, 0.4, 5]],
        [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
        [1, 1, 1],
        fixed_coordinates=[[True, False, False], [False] * 3, [True] * 3],
        metadata={"constraint_basis": "fractional"},
    )
    swapped = make_supercell(atoms, [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_array_equal(swapped.fixed_coordinates, [[False, True, False], [False] * 3, [True] * 3])
    np.testing.assert_allclose(
        swapped.cell[~swapped.fixed_coordinates[0]], atoms.cell[~atoms.fixed_coordinates[0]]
    )
    with pytest.raises(ValueError, match="constraint subspace"):
        make_supercell(atoms, [[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="constraint subspace"):
        add_vacuum(atoms, 3)
    slab = atoms.copy(fixed_coordinates=[[False, False, True], [False] * 3, [True] * 3])
    np.testing.assert_array_equal(add_vacuum(slab, 3).fixed_coordinates, slab.fixed_coordinates)
    unconstrained = atoms.copy(fixed_coordinates=[[False] * 3, [True] * 3, [False] * 3])
    mixed = make_supercell(unconstrained, [[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    np.testing.assert_array_equal(mixed.fixed_coordinates, unconstrained.fixed_coordinates)


MAGNETIC_CIF = """data_magnet
_cell_length_a 3
_cell_length_b 3
_cell_length_c 5
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
loop_
_space_group_symop_magn_operation.xyz
'x,y,z,+1'
'x,y,z+1/2,-1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Fe1 Fe 0 0 0
loop_
_atom_site_moment.label
_atom_site_moment.crystalaxis_x
_atom_site_moment.crystalaxis_y
_atom_site_moment.crystalaxis_z
Fe1 1 2 0
"""


def test_magnetic_symmetry_time_reversal_and_crystalaxis_roundtrip():
    atoms = parse_cif(MAGNETIC_CIF)
    assert atoms.natoms == 2
    np.testing.assert_allclose(atoms.magnetic_moments[0], [0, np.sqrt(3), 0], atol=1e-12)
    np.testing.assert_allclose(atoms.magnetic_moments[1], -atoms.magnetic_moments[0], atol=1e-12)
    result = parse_cif(write_cif(atoms))
    np.testing.assert_allclose(result.positions, atoms.positions, atol=1e-12)
    np.testing.assert_allclose(result.magnetic_moments, atoms.magnetic_moments, atol=1e-12)


def test_cif_axial_vector_rotation_and_untrusted_operation():
    text = MAGNETIC_CIF.replace("'x,y,z+1/2,-1'", "'-x,y,z+1/2,+1'").replace(
        "_cell_angle_gamma 120", "_cell_angle_gamma 90"
    )
    atoms = parse_cif(text)
    np.testing.assert_allclose(atoms.magnetic_moments[1], [1, -2, 0], atol=1e-12)
    with pytest.raises(ValueError, match="Unsupported"):
        parse_cif(MAGNETIC_CIF.replace("'x,y,z,+1'", "'__import__(x),y,z,+1'"))
    with pytest.raises(ValueError, match="non-affine"):
        parse_cif(MAGNETIC_CIF.replace("'x,y,z,+1'", "'x*x,y,z,+1'"))
    with pytest.raises(ValueError, match="time-reversal"):
        parse_cif(MAGNETIC_CIF.replace("'x,y,z,+1'", "'x,y,z,1.5'"))


def test_cif_reads_one_data_block_without_combining_structures():
    text = "data_description\n_description\n;\nA document header.\n;\n"
    text += MAGNETIC_CIF
    text += MAGNETIC_CIF.replace("data_magnet", "data_second").replace("Fe1 Fe 0 0 0", "Fe1 Fe 0.25 0 0")
    atoms = parse_cif(text)
    assert atoms.natoms == 2
    np.testing.assert_allclose(atoms.positions[:, 0], 0, atol=1e-12)


def test_cif_cartesian_atom_site_coordinates():
    text = MAGNETIC_CIF.replace("_atom_site_fract_", "_atom_site_cartn_").replace(
        "Fe1 Fe 0 0 0", "Fe1 Fe 1 0 1"
    )
    atoms = parse_cif(text)
    np.testing.assert_allclose(atoms.positions, [[1, 0, 1], [1, 0, 3.5]], atol=1e-12)


def test_native_symmetry_rotated_supercell_and_decorated_primitive():
    pytest.importorskip("spglib")
    cell = np.array([[0, 2, 2], [2, 0, 2], [2, 2, 0]])
    atoms = Structure(cell, [[0, 0, 0]], [47])
    base = compute_band_path(atoms)
    assert base["spacegroup_number"] == 225
    assert base["bravais_lattice_extended"] == "cF2"
    np.testing.assert_allclose(base["point_coords"]["X"], [0.5, 0, 0.5], atol=1e-12)
    angle = 0.4
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    rotated = atoms.copy(cell=cell @ rotation.T)
    transformed = compute_band_path(rotated)
    np.testing.assert_allclose(transformed["point_coords"]["X"], base["point_coords"]["X"], atol=1e-12)
    enlarged = make_supercell(atoms, np.diag([2, 1, 1]))
    with pytest.warns(RuntimeWarning, match="nonprimitive"):
        full = compute_band_path(enlarged)
    from queer.geometry.lattice import reciprocal_lattice

    np.testing.assert_allclose(
        np.asarray(full["point_coords"]["X"]) @ reciprocal_lattice(enlarged.cell),
        np.asarray(base["point_coords"]["X"]) @ reciprocal_lattice(atoms.cell),
        atol=1e-12,
    )
    reduced = primitive_structure(enlarged)
    assert reduced.natoms == 1
    antiferro = enlarged.copy(magnetic_moments=[[0, 0, 1], [0, 0, -1]])
    assert primitive_structure(antiferro).natoms == 2


def test_primitive_constraint_subspaces_preserved_or_explicitly_rejected():
    pytest.importorskip("spglib")
    atoms = Structure(
        np.eye(3) * 4,
        [[0, 0, 0], [0, 2, 2], [2, 0, 2], [2, 2, 0]],
        [47] * 4,
        metadata={"constraint_basis": "fractional", "source_indices": [10, 11, 12, 13]},
    )
    for mask in ([False] * 3, [True] * 3):
        reduced = primitive_structure(atoms.copy(fixed_coordinates=[mask] * 4))
        assert reduced.natoms == 1
        np.testing.assert_array_equal(reduced.fixed_coordinates, [mask])
        assert reduced.metadata["source_indices"][0] in {10, 11, 12, 13}
    with pytest.raises(ValueError, match="constraint subspace"):
        primitive_structure(atoms.copy(fixed_coordinates=[[True, False, False]] * 4))


def test_native_bz_cube_and_skew_volume():
    from scipy.spatial import ConvexHull

    for basis in (np.eye(3) * 2, [[2, 0, 0], [1.8, 0.4, 0], [0.1, 0.2, 3]]):
        faces = brillouin_zone_faces(basis)
        vertices = np.vstack([face[0] for face in faces])
        assert ConvexHull(vertices).volume == pytest.approx(abs(np.linalg.det(basis)), rel=1e-9)
        for points, normal in faces:
            assert np.dot(points[0], normal) > 0
            np.testing.assert_allclose(points @ normal, np.dot(points[0], normal), atol=1e-8)


def test_native_formats_do_not_import_atomistic_engines(monkeypatch):
    original = builtins.__import__

    def deny(name, *args, **kwargs):
        if name.split(".")[0] in {"ase", "seekpath", "Dans_Diffraction"}:
            raise AssertionError("Native operation imported " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny)
    atoms = parse_cif(MAGNETIC_CIF)
    parse_poscar(poscar_text(atoms)[0])
    read_extxyz(write_extxyz(atoms))
    read_trajectory(write_trajectory([Frame(atoms)]))
    compute_bonds(atoms)
