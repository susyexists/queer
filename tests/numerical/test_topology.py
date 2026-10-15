"""Analytic references for Wilson-loop Chern numbers, WCC flows and Z2 indices."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from tight_binding_models import kane_mele, qwz, rice_mele

from queer.models.evaluation import states
from queer.observables.geometry import chern_number
from queer.observables.topology import (
    plane_kpoints,
    plane_loops,
    topology_planes,
    trim_points,
    wilson_plane,
)
from queer.physics.topology import (
    chern_from_phases,
    kramers_degenerate,
    plane_axes,
    wrap_phase,
    z2_from_wcc,
    z2_indices,
)


def cell_periodic_wcc(model, loop, axis):
    """Single-band WCC from PythTB's convention-I eigenvectors, closed with exp(-2 pi i tau_a)."""
    tau = model.centres @ np.linalg.inv(model.lattice)
    vectors = []
    for k in loop:
        phase = np.exp(2j * np.pi * (model.translations @ k))
        offsets = np.exp(2j * np.pi * (tau @ k))
        hamiltonian = np.einsum("r,rij->ij", phase, model.hopping_matrices) * offsets[None] / offsets[:, None]
        vectors.append(np.linalg.eigh(hamiltonian)[1][:, 0])
    vectors.append(vectors[0] * np.exp(-2j * np.pi * tau[:, axis]))
    product = np.prod([vectors[j].conj() @ vectors[j + 1] for j in range(len(loop))])
    return np.mod(-np.angle(product) / (2 * np.pi), 1.0)


def circular_distance(first, second):
    gap = np.abs(np.mod(np.asarray(first) - np.asarray(second), 1.0))
    return np.minimum(gap, 1.0 - gap)


@pytest.mark.parametrize("mass,expected", [(-3, 0), (-1, 1), (1, -1), (3, 0)])
def test_qwz_phase_diagram_from_wilson_loop_winding(mass, expected):
    model = qwz(mass)
    result = wilson_plane(model, 2, 0.0, occupied_bands=1, loop_points=31, transverse_points=24)
    assert int(result["chern"]) == expected and bool(result["converged"])
    assert abs(float(result["chern_raw"]) - expected) < 1e-10
    plaquette = chern_number(model, grid=(12, 12, 1))["chern_number"]
    assert_allclose(plaquette, expected, atol=1e-10)
    assert result.metadata["loop_axis"] == 0 and result.metadata["transverse_axis"] == 1
    assert result["wcc"].shape == (24, 1) and result["berry_phase"].shape == (24,)
    assert "z2" not in result.arrays and result.metadata["z2_computed"] is False


def test_refinement_doubles_until_resolved_and_reuses_the_coarser_loops():
    model = qwz(-1)
    refined = wilson_plane(
        model, 2, 0.0, occupied_bands=1, loop_points=21, transverse_points=2, max_refinements=4
    )
    final = refined.metadata["transverse_points"]
    assert refined.metadata["attempts"] > 1 and final == 2 ** refined.metadata["attempts"]
    assert bool(refined["converged"]) and int(refined["chern"]) == 1
    assert float(refined["max_phase_step"]) < np.pi / 2
    direct = wilson_plane(
        model, 2, 0.0, occupied_bands=1, loop_points=21, transverse_points=final, max_refinements=0
    )
    for name in ("transverse", "berry_phase", "wcc", "chern_raw"):
        assert_array_equal(refined[name], direct[name])
    capped = wilson_plane(
        model, 2, 0.0, occupied_bands=1, loop_points=21, transverse_points=2, max_refinements=0
    )
    assert not bool(capped["converged"]) and capped.metadata["transverse_points"] == 2
    odd = wilson_plane(
        model, 2, 0.0, occupied_bands=1, loop_points=21, transverse_points=9, max_refinements=0
    )
    assert odd.metadata["transverse_points"] == 10 and odd["transverse"][5] == 0.5


@pytest.mark.parametrize("v,w,centre", [(0.4, 1.0, 0.75), (1.0, 0.4, 0.25)])
def test_tau_half_chain_has_unitary_links_where_the_misapplied_closing_phase_is_singular(v, w, centre):
    model = rice_mele(v, w)
    loop_count = 81
    result = wilson_plane(model, 2, 0.0, occupied_bands=1, loop_points=loop_count, transverse_points=4)
    # Inversion about the bond centres pins the hybrid centre at 1/4 or 3/4 of a1.
    assert np.max(circular_distance(result["wcc"], centre)) < 1e-12
    assert_allclose(wrap_phase(result["berry_phase"] - 2 * np.pi * centre), 0, atol=1e-12)
    loop = plane_loops(2, 0.0, [0.0], loop_count)[0]
    solution = states(model, loop)
    first, last = solution.coefficients[0][:, :1], solution.coefficients[-1][:, :1]
    boundary = np.diag(np.exp(-2j * np.pi * np.array([0.0, 0.5])))
    # Periodic-gauge vectors have equal weights on A and B, so P exp(-2 pi i tau_a) P vanishes.
    assert abs((first.conj().T @ boundary @ first).item()) < 1e-15
    misapplied = abs((last.conj().T @ boundary @ first).item())
    lifted = loop[0] + np.array([1.0, 0.0, 0.0])
    closing = abs((last.conj().T @ model.basis_overlap(loop[-1], lifted) @ first).item())
    assert misapplied < 0.03 and closing > 0.99
    links = [
        abs(
            (
                solution.coefficients[j][:, :1].conj().T
                @ model.basis_overlap(loop[j], loop[j + 1])
                @ solution.coefficients[j + 1][:, :1]
            ).item()
        )
        for j in range(loop_count - 1)
    ]
    assert min(links) > 0.99


def test_rice_mele_wcc_equals_an_independent_cell_periodic_loop():
    model = rice_mele(0.4, 1.0, staggered=0.3)
    result = wilson_plane(model, 2, 0.0, occupied_bands=1, loop_points=81, transverse_points=2)
    loop = plane_loops(2, 0.0, [0.0], 81)[0]
    reference = cell_periodic_wcc(model, loop, 0)
    assert circular_distance(result["wcc"][0, 0], reference) < 1e-12
    # Misplacing the closing phase on periodic-gauge vectors shifts the centre far off.
    solution = states(model, loop)
    vectors = [solution.coefficients[j][:, 0] for j in range(len(loop))]
    vectors.append(vectors[0] * np.exp(-2j * np.pi * np.array([0.0, 0.5])))
    product = np.prod([vectors[j].conj() @ vectors[j + 1] for j in range(len(loop))])
    misplaced = np.mod(-np.angle(product) / (2 * np.pi), 1.0)
    assert circular_distance(misplaced, reference) > 0.1


@pytest.mark.parametrize("onsite,expected", [(1.0, 1), (2.5, 0)])
def test_kane_mele_z2_from_the_largest_gap_flow(onsite, expected):
    model = kane_mele(onsite)
    paired, split = kramers_degenerate(model.energies(trim_points([0, 1])).T, 1e-6)
    assert paired and split < 1e-12
    result = wilson_plane(model, 2, 0.0, occupied_bands=2, loop_points=31, transverse_points=12, z2=True)
    assert int(result["z2"]) == expected and not bool(result["z2_ambiguous"])
    assert int(result["z2_crossings"]) % 2 == expected
    assert int(result["chern"]) == 0 and bool(result["converged"])
    assert result["gap_midpoints"].shape == (7,) and result.metadata["z2_computed"] is True
    off_trim = wilson_plane(model, 2, 0.25, occupied_bands=2, loop_points=21, transverse_points=4, z2=True)
    assert "z2" not in off_trim.arrays and off_trim.metadata["z2_computed"] is False


def test_stacked_kane_mele_is_a_weak_topological_insulator():
    model = kane_mele(1.0, interlayer=0.1, height=3.0)
    assert kramers_degenerate(model.energies(trim_points([0, 1, 2])).T, 1e-6)[0]
    planes = topology_planes([0, 1, 2])
    assert planes == [(2, 0.0), (2, 0.5), (0, 0.0), (0, 0.5), (1, 0.0), (1, 0.5)]
    flows = {
        plane: wilson_plane(model, *plane, occupied_bands=2, loop_points=21, transverse_points=10, z2=True)
        for plane in planes
    }
    assert not any(bool(result["z2_ambiguous"]) for result in flows.values())
    indices = z2_indices({plane: int(result["z2"]) for plane, result in flows.items()})
    assert indices == {"nu1": 0, "nu2": 0, "nu3": 1, "nu0": 0, "consistent": True}


def test_z2_indices_report_unknown_and_inconsistent_planes():
    strong = {(axis, value): 1 if value == 0.0 else 0 for axis in range(3) for value in (0.0, 0.5)}
    assert z2_indices(strong) == {"nu1": 0, "nu2": 0, "nu3": 0, "nu0": 1, "consistent": True}
    clash = {**strong, (1, 0.0): 0}
    assert z2_indices(clash)["nu0"] is None and z2_indices(clash)["consistent"] is False
    partial = z2_indices({(2, 0.0): 1, (2, 0.5): None})
    assert partial == {"nu1": None, "nu2": None, "nu3": None, "nu0": None, "consistent": True}


def test_largest_gap_rule_counts_jumps_and_flags_centres_at_a_midpoint():
    # One Kramers pair separates and re-pairs across the cell boundary: a single jump.
    switching = [[0.5, 0.5], [0.35, 0.65], [0.2, 0.8], [0.05, 0.95]]
    flow = z2_from_wcc(switching, pos_tol=0.01)
    assert (flow.z2, flow.ambiguous) == (1, False)
    trivial = z2_from_wcc([[0.375, 0.625], [0.3125, 0.6875], [0.375, 0.625]])
    assert (trivial.z2, trivial.crossings, trivial.ambiguous) == (0, 0, False)
    assert_array_equal(trivial.gap_midpoints, [0.0, 0.0, 0.0])
    # The next point's 0.81 sits 0.0025 from the previous largest-gap midpoint 0.8125.
    assert z2_from_wcc([[0.25, 0.375], [0.25, 0.81]], pos_tol=0.01).ambiguous
    assert not z2_from_wcc([[0.25, 0.375], [0.25, 0.81]], pos_tol=0.002).ambiguous
    assert z2_from_wcc([[]]).min_largest_gap == 1.0 and z2_from_wcc([]).min_largest_gap is None


def test_phase_winding_counts_wrapped_steps():
    phases = -2 * np.pi * np.arange(8) / 8
    assert chern_from_phases(phases) == pytest.approx((1.0, np.pi / 4))
    assert_allclose(wrap_phase([np.pi, -np.pi, 3 * np.pi / 2]), [-np.pi, -np.pi, -np.pi / 2])
    with pytest.raises(ValueError, match="nonempty"):
        chern_from_phases([])
    assert kramers_degenerate([[0.0, 1e-9, 2.0, 2.0]], 1e-6) == (True, 1e-9)
    assert kramers_degenerate([0.0, 1.0, 2.0], 1e-6) == (False, float("inf"))


def test_plane_helpers_follow_the_cyclic_frame():
    assert [plane_axes(c) for c in range(3)] == [(1, 2), (2, 0), (0, 1)]
    with pytest.raises(ValueError, match="fixed_axis"):
        plane_axes(3)
    assert topology_planes([1, 0]) == [(2, 0.0)] and topology_planes([0, 2]) == [(1, 0.0)]
    with pytest.raises(ValueError, match="two or three"):
        topology_planes([0])
    assert_array_equal(trim_points([0, 1]), [[0, 0, 0], [0, 0.5, 0], [0.5, 0, 0], [0.5, 0.5, 0]])
    points = plane_kpoints(0, 0.5, 3)
    assert_array_equal(points[:4], [[0.5, 0, 0], [0.5, 0, 1 / 3], [0.5, 0, 2 / 3], [0.5, 1 / 3, 0]])
    assert_allclose(
        plane_kpoints(2, 0.0, 2, centred=True)[:, :2], [[-0.5, -0.5], [-0.5, 0], [0, -0.5], [0, 0]]
    )
    loops = plane_loops(1, 0.5, [0.0, 0.25], 4)
    assert loops.shape == (2, 4, 3)
    assert_array_equal(loops[1, :, 2], [0, 0.25, 0.5, 0.75])
    assert_array_equal(loops[1, :, 0], 0.25)
    assert_array_equal(loops[1, :, 1], 0.5)
