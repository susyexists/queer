"""Independent boundaries for shared structure parsing and atomistic kernels."""

import numpy as np
import pytest

from queer.atomistic import plan_displacements, run_nvt, select_min_energy_frame
from queer.atomistic._common import evaluate
from queer.geometry.structures import expand_diagonal, make_supercell
from queer.io.atomistic import parse_poscar
from queer.io.structure import read_poscar
from queer.structures import Evaluation, Frame, Structure


@pytest.mark.parametrize("scale", ["-24", "2 3 4"])
@pytest.mark.parametrize("mode", ["Direct", "Cartesian"])
def test_poscar_adapters_share_geometry_with_distinct_species_contracts(tmp_path, scale, mode):
    text = (
        f"unknown chemical labels\n{scale}\n1 0 0\n.2 1 0\n.1 .3 1\n2 1\n"
        f"Selective dynamics\n{mode}\n.2 .3 .4 T F T\n.4 .1 .2 F T T\n.1 .2 .3 T T F\n"
    )
    source = tmp_path / "POSCAR"
    source.write_text(text)
    cell, fractional, groups = read_poscar(source)
    atoms = parse_poscar(text, species=["O", "H"])
    np.testing.assert_array_equal(cell, atoms.cell)
    np.testing.assert_allclose(fractional @ cell, atoms.positions, rtol=1e-14, atol=1e-14)
    assert groups == [1, 1, 2]
    assert atoms.atomic_numbers.tolist() == [8, 8, 1]
    assert atoms.metadata["constraint_basis"] == "fractional"
    np.testing.assert_array_equal(
        atoms.fixed_coordinates, [[False, True, False], [True, False, False], [False, False, True]]
    )
    with pytest.raises(ValueError, match="VASP 4 POSCAR requires"):
        parse_poscar(text)


def test_vasp5_group_labels_preserve_species_groups_in_tuple_adapter(tmp_path):
    text = "groups\n1\n2 0 0\n0 3 0\n0 0 4\nO H O\n1 1 1\nDirect\n0 0 0\n.1 0 0\n.2 0 0\n"
    source = tmp_path / "POSCAR"
    source.write_text(text)
    assert read_poscar(source)[2] == [1, 2, 3]
    assert parse_poscar(text).atomic_numbers.tolist() == [8, 1, 8]


def test_ordered_expansion_keeps_unwrapped_coordinates_fields_and_plan_order():
    cell = np.array([[2.0, 0, 0], [0.3, 3, 0], [0.2, 0.1, 4]])
    fractional = np.array([[-0.2, 0.1, 1.3], [0.4, -0.1, 0.2]])
    structure = Structure(
        cell,
        fractional @ cell,
        [8, 1],
        pbc=(True, True, False),
        masses=[15, 2],
        magnetic_moments=[[1, 2, 3], [4, 5, 6]],
        fixed_coordinates=[[True, False, False], [False, True, False]],
        arrays={"producer_id": [17, 23]},
        metadata={"constraint_basis": "fractional", "source_indices": [5, 9]},
    )
    expanded, images = expand_diagonal(structure, (2, 2, 1))
    expected_images = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]
    np.testing.assert_array_equal(images, expected_images)
    expected_positions = np.concatenate(
        [structure.positions + np.asarray(image) @ cell for image in expected_images]
    )
    np.testing.assert_array_equal(expanded.positions, expected_positions)
    for name in ("atomic_numbers", "masses", "magnetic_moments", "fixed_coordinates"):
        np.testing.assert_array_equal(getattr(expanded, name), np.concatenate([getattr(structure, name)] * 4))
    np.testing.assert_array_equal(expanded.arrays["producer_id"], [17, 23] * 4)
    assert expanded.metadata == structure.metadata
    repeated, _ = expand_diagonal(expanded, (2, 1, 1))
    assert repeated.metadata == structure.metadata
    np.testing.assert_array_equal(repeated.positions[: expanded.natoms], expanded.positions)
    wrapped = make_supercell(structure, np.diag([2, 2, 1]))
    np.testing.assert_allclose(wrapped.fractional_positions[:, :2], expanded.fractional_positions[:, :2] % 1)
    np.testing.assert_allclose(wrapped.positions[:, 2], expanded.positions[:, 2])
    unconstrained = structure.copy(fixed_coordinates=None)
    plan = plan_displacements(unconstrained, supercell=(2, 2, 1), delta_A=0.02)
    np.testing.assert_array_equal(plan.cell_indices, images)
    for configuration, (atom, axis, sign) in zip(plan.configurations, plan.entries):
        displacement = configuration.positions - plan.reference.positions
        expected = np.zeros_like(displacement)
        expected[atom, axis] = sign * 0.02
        np.testing.assert_allclose(displacement, expected, atol=2e-16)


def test_force_producer_isolation_and_revalidation():
    structure = Structure(np.eye(3), [[0.2, 0.3, 0.4]], [1])
    original = structure.positions.copy()

    def producer(argument):
        argument.positions[:] = 99
        return Evaluation(1, [[0, 0, 0]])

    assert evaluate(producer, structure).energy_eV == 1
    np.testing.assert_array_equal(structure.positions, original)
    invalid = Evaluation(1, [[0, 0, 0]])
    invalid.forces_eV_A[:] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluate(lambda argument: invalid, structure)
    with pytest.raises(ValueError, match="force shape"):
        evaluate(lambda argument: Evaluation(1, np.zeros((2, 3))), structure)


def test_selection_total_energy_matches_dynamics_sample_for_recorded_frame():
    initial = Structure(np.eye(3) * 5, [[0, 0, 0]], [1], masses=[2])
    result = run_nvt(
        initial,
        lambda structure: Evaluation(-0.2, [[0, 0, 0]]),
        n_steps=0,
        initial_velocities=[[0.1, 0.2, 0.3]],
        remove_com=False,
        remove_rotation=False,
    )
    selected = select_min_energy_frame([result.final_frame], energy_key="etot_eV")
    assert selected.summary["selected_energy_eV"] == result.observable["etot_eV"][0]
    potential = Frame(initial, evaluation=Evaluation(-0.3, [[0, 0, 0]]))
    assert select_min_energy_frame([result.final_frame, potential]).frame is potential
