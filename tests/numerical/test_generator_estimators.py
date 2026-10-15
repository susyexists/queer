"""Independent references for the Generator estimators kept in queer.compat.generator."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from tight_binding_models import qwz

from queer.compat.generator.estimators import (
    BandEdges,
    band_deviation,
    classify_gap,
    comparison_window,
    dos_at_fermi,
    dos_energy_grid,
    eigenvalue_histogram,
    gaussian_dos,
    hopping_decay,
    k_distance,
    orbital_species,
    project_path,
    sample_path,
    surface_path,
    uniform_mesh,
    unwrap_centres,
)

ROOT = Path(__file__).resolve().parents[2]


def replay_script():
    spec = importlib.util.spec_from_file_location(
        "generator_replay", ROOT / "scripts/validation/generator_replay.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_histogram_dos_conserves_states_and_smooths_one_level_into_a_discrete_gaussian():
    model = qwz(-1)
    mesh = uniform_mesh([12, 12, 12], [0, 1])
    assert mesh.shape == (144, 3) and np.all(mesh[:, 2] == 0)
    energies = model.energies(mesh).T
    step, sigma = 0.005, 0.03
    grid = dos_energy_grid(energies, step)
    counts, outside = eigenvalue_histogram(energies, grid)
    assert outside == 0 and counts.sum() == energies.size
    dos = gaussian_dos(counts / len(mesh), step, sigma)
    # Two Wannier bands, no spin factor: the smoothed DOS integrates to two states per cell.
    assert_allclose(dos.sum() * step, 2.0, rtol=1e-12)
    single = np.zeros(401)
    single[200] = 1.0
    kernel = gaussian_dos(single, step, sigma)
    offsets = (np.arange(401) - 200) * step
    half = int(np.ceil(5 * sigma / step))
    expected = np.where(np.abs(offsets) <= half * step + 1e-12, np.exp(-0.5 * (offsets / sigma) ** 2), 0.0)
    assert_allclose(kernel, expected / expected.sum() / step, rtol=1e-12, atol=1e-15)
    assert_allclose(gaussian_dos(single, step, 0.0), single / step)
    weights = np.full(energies.shape, 0.25)
    weighted, _ = eigenvalue_histogram(energies, grid, weights)
    assert_allclose(weighted, 0.25 * counts)
    top, outside = eigenvalue_histogram([grid[-1], grid[0] - 1.0], grid)
    assert top[-1] == 1 and outside == 2


def test_dos_at_fermi_rules_match_the_replay_policies():
    centres = np.array([-0.0075, -0.0025, 0.0025, 0.0075])
    dos = np.array([1.0, 2.0, 4.0, 8.0])
    assert dos_at_fermi(centres, dos) == 3.0
    assert dos_at_fermi(centres, dos, "first_nonnegative_bin") == 4.0
    with pytest.raises(ValueError, match="rule"):
        dos_at_fermi(centres, dos, "nearest_bin")
    replay = replay_script()
    rng = np.random.default_rng(81)
    axis = np.arange(-1.0, 1.0, 0.005) + 0.0012
    density = rng.uniform(0, 3, size=axis.shape)
    report = replay.fermi_dos_diagnostics(axis, density, density, 1.0)["estimators"]
    assert (
        dos_at_fermi(axis, density, "first_nonnegative_bin")
        == report["first_nonnegative_bin"]["native_per_eV"]
    )
    assert dos_at_fermi(axis, density) == report["linear_interpolation"]["native_per_eV"]


def test_gap_classification_of_a_gapped_qwz_mesh_a_metal_and_a_level_outside_the_model():
    # At m = -1.5 the bands are -+|d| with min |d| = 0.5 reached only at Gamma.
    model = qwz(-1.5)
    mesh = uniform_mesh([16, 16, 1], [0, 1])
    energies = model.energies(mesh).T
    edges = BandEdges(2)
    edges.add(mesh[:100], energies[:100])
    rest = BandEdges(2)
    rest.add(mesh[100:], energies[100:])
    edges.merge(rest)
    gap = classify_gap(edges, 0.02)
    assert gap["n_occ"] == 1 and gap["n_occ_source"] == "auto" and not gap["metal"]
    measured = [gap["vbm_eV"], gap["cbm_eV"], gap["gap_eV"], gap["direct_gap_eV"]]
    assert_allclose(measured, [-0.5, 0.5, 1.0, 1.0], atol=1e-12)
    assert gap["vbm_k"] == gap["cbm_k"] == [0.0, 0.0, 0.0] and gap["is_direct"]
    assert edges.n_k == 256 and edges.count_min == edges.count_max == 1
    shifted = BandEdges(2)
    shifted.add(mesh, energies - 2.0)
    metal = classify_gap(shifted, 0.02)
    assert metal["metal"] and metal["n_occ"] is None and metal["bands_crossing_fermi"] == [1]
    assert not metal["fermi_level_outside_model"]
    above = BandEdges(2)
    above.add(mesh, energies - 5.0)
    assert classify_gap(above, 0.02)["fermi_level_outside_model"]
    edge_case = BandEdges(2)
    edge_case.add(mesh, energies + 0.51)
    # E_F 0.01 eV below the valence maximum still selects that gap within the tolerance.
    assert classify_gap(edge_case, 0.02)["n_occ"] == 1 and classify_gap(edge_case, 0.0)["n_occ"] is None
    assert classify_gap(shifted, 0.02, occupied_bands=1)["n_occ_source"] == "user"


def test_band_dat_window_rules_and_deviation_metrics():
    assert comparison_window({"dis_froz_min": -1.0, "dis_froz_max": 2.0, "dis_win_min": -5.0}, 1.0) == (
        -2.0,
        1.0,
    )
    assert comparison_window({"dis_froz_max": 2.0, "dis_win_min": -5.0}, 1.0) == (-6.0, 1.0)
    assert comparison_window({"dis_froz_max": 2.0}, 1.0) == (-2.0, 2.0)
    assert comparison_window({"dis_win_min": -5.0, "dis_win_max": 4.0}, 1.0) == (-6.0, 3.0)
    reference = np.array([[-1.0, 0.5], [0.2, 3.0]])
    model = reference + np.array([[0.004, -0.001], [0.0, 0.1]])
    result = band_deviation(model, reference, (-1.5, 1.0))
    assert result["max_meV"] == pytest.approx(100.0) and result["in_window_max_meV"] == pytest.approx(4.0)
    assert result["in_window_points"] == 3 and "window_empty" not in result
    empty = band_deviation(model, reference, (10.0, 11.0))
    assert empty["window_empty"] and empty["in_window_points"] == 4
    assert empty["in_window_rms_meV"] == empty["rms_meV"]


def test_wannier90_path_sampling_and_frozen_distance_across_jumps():
    lattice = np.eye(3) * 2.0
    segments = [
        ["G", [0, 0, 0], "X", [0.5, 0, 0]],
        ["X", [0.5, 0, 0], "M", [0.5, 0.5, 0]],
        ["G", [0, 0, 0], "M", [0.5, 0.5, 0]],
    ]
    path = sample_path(segments, lattice, 100)
    total = 1.0 + np.sqrt(2) / 2
    counts = [max(2, round(100 * length / total)) for length in (0.5, 0.5, np.sqrt(2) / 2)]
    assert len(path["kpoints"]) == (counts[0] + 1) + counts[1] + (counts[2] + 1)
    assert path["tick_labels"] == ["G", "X", "M", "G", "M"]
    jump = counts[0] + counts[1] + 1
    assert path["break_indices"] == [jump] and path["tick_indices"][3] == jump
    assert_array_equal(path["kpoints"][counts[0]], [0.5, 0, 0])
    distance = k_distance(path["kpoints"], lattice, path["break_indices"])
    assert distance[jump] == distance[jump - 1]
    assert_allclose(distance[-1], np.pi * total, rtol=1e-12)
    flat = project_path(
        [["G", [0, 0, 0], "A", [0, 0, 0.5]], ["G", [0, 0, 0.2], "K", [1 / 3, 1 / 3, 0.5]]], [0, 1]
    )
    assert flat == [["G", [0.0, 0.0, 0.0], "K", [1 / 3, 1 / 3, 0.0]]]


@pytest.mark.parametrize(
    "lattice,labels",
    [
        (
            np.array([[3.0, 0, 0], [-1.5, 1.5 * np.sqrt(3), 0], [0, 0, 20.0]]),
            ["G", "M", "K", "G", "M'", "M''", "G"],
        ),
        (np.array([[3.0, 0, 0], [0, 4.0, 0], [0, 0, 20.0]]), ["G", "X", "M", "Y", "G"]),
    ],
)
def test_surface_path_passes_every_surface_trim(lattice, labels):
    path = surface_path(lattice, 0, 1, 90)
    assert path["tick_labels"] == labels
    points = {tuple(np.round(point[:2], 12)) for point in path["kpoints"]}
    assert {(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)} <= points


def test_orbital_species_uses_the_nearest_periodic_image():
    lattice = np.diag([4.0, 4.0, 10.0])
    structure = {"species": ["Bi", "Te", "Bi"], "cart": [[0.1, 0.1, 5.0], [2.0, 2.0, 5.0], [3.0, 0.5, 5.0]]}
    centres = [[3.95, 3.9, 5.0], [2.2, 1.9, 5.1], [3.1, 0.6, 4.9]]
    names, index = orbital_species(centres, structure, lattice)
    assert names == ["Bi", "Te"] and index.tolist() == [0, 1, 0]
    assert orbital_species(centres, None, lattice)[0] == ["all"]


def test_unwrap_centres_rejoins_a_slab_cut_by_the_wrap_line():
    lattice = np.diag([4.0, 4.0, 80.0])
    # A film from z = 0.92 to 1.12 of the cell, written by Wannier90 across the wrap line.
    fractional = np.array(
        [[0, 0, 0.92], [0.5, 0, 0.97], [0, 0.5, 0.03], [0.5, 0.5, 0.08], [0.25, 0.25, 0.12]]
    )
    unwrapped = unwrap_centres(fractional @ lattice, lattice, [2]) @ np.linalg.inv(lattice)
    assert_allclose(unwrapped[:, 2], [-0.08, -0.03, 0.03, 0.08, 0.12], atol=1e-12)
    assert_allclose(unwrapped[:, :2], fractional[:, :2], atol=1e-12)
    assert_array_equal(unwrap_centres(fractional @ lattice, lattice, []), fractional @ lattice)


def test_hopping_decay_recovers_an_exponential_length():
    spacing, length = 1.5, 0.8
    translations = np.array([[r, 0, 0] for r in range(-8, 9)])
    hopping = np.exp(-np.abs(translations[:, 0]) * spacing / length)[:, None, None] * np.ones((1, 1, 1))
    result = hopping_decay(translations, hopping, np.zeros((1, 3)), np.diag([spacing, 10.0, 10.0]))
    assert_allclose(result["decay_length_A"], length, rtol=1e-10)
    assert result["max_distance_A"] == pytest.approx(8 * spacing)
    assert_allclose(result["bin_centres"], spacing * np.arange(9), atol=1e-12)
    growing = hopping_decay(translations, 1 / hopping, np.zeros((1, 3)), np.diag([spacing, 10.0, 10.0]))
    assert growing["decay_length_A"] is None
