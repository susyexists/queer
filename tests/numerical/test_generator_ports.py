"""Ported Generator PythTB functions reproduce the vendored originals.

Each test feeds identical inputs to a QUEER port and to the original function in
``tests/oracles`` (loaded by the ``generator_core`` fixture). Analytic acceptance
of the same code lives in the topology, Berry-plane, slab and estimator tests.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from tight_binding_models import cell_periodic_operators, haldane, kane_mele, qwz

from queer.models.evaluation import states
from queer.models.wannier import WannierModel
from queer.observables.geometry import berry_plane, wilson_loop
from queer.observables.topology import plane_kpoints, plane_loops, wilson_plane
from queer.physics import geometry, topology

PLANE_2D = np.array([[a, b, 0] for a in (-1, 0, 1) for b in (-1, 0, 1)])


def hermitian_table(rng, translations, onsite, scale=0.25):
    """Random H(R) with H(-R) = H(R)^dagger and the given on-site diagonal."""
    n = len(onsite)
    index = {tuple(r): i for i, r in enumerate(translations)}
    table = np.zeros((len(translations), n, n), complex)
    for i, r in enumerate(translations):
        j = index[tuple(-r)]
        if j < i:
            continue
        block = scale * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
        if i == j:
            block = (block + block.conj().T) / 2 + np.diag(onsite)
        table[i], table[j] = block, block.conj().T
    return table


def random_layer(rng, bands, occupied):
    """A gapped random 2D model on an oblique cell with off-centre Wannier centres."""
    onsite = np.r_[-2.5 * np.ones(occupied), 2.5 * np.ones(bands - occupied)]
    lattice = np.array([[3.1, 0.0, 0.0], [1.2, 2.8, 0.0], [0.1, 0.2, 15.0]])
    centres = rng.uniform(-0.4, 1.4, size=(bands, 3)) @ lattice
    return WannierModel(PLANE_2D, hermitian_table(rng, PLANE_2D, onsite), lattice=lattice, centres=centres)


def circular_mismatch(first, second):
    """Largest circular distance between two sorted centre sets under the best cyclic pairing."""
    first, second = np.sort(np.mod(first, 1.0)), np.sort(np.mod(second, 1.0))
    best = np.inf
    for shift in range(max(len(second), 1)):
        gap = np.abs(first - np.roll(second, shift))
        best = min(best, float(np.max(np.minimum(gap, 1.0 - gap), initial=0.0)))
    return best


def generator_loop(core, model, loop, occupied, loop_axis):
    """The original wilson_loop on QUEER's eigenvectors turned cell-periodic (PythTB convention I)."""
    tau = model.centres @ np.linalg.inv(model.lattice)
    solution = states(model, loop)
    rows = [
        (np.exp(-2j * np.pi * (tau @ k))[:, None] * solution.coefficients[j][:, :occupied]).T
        for j, k in enumerate(loop)
    ]
    return core.wilson_loop(rows, np.exp(-2j * np.pi * tau[:, loop_axis]))


def generator_plane(core, model, fixed, value, occupied, loop_points, transverse_points, refinements, z2):
    """The transverse sweep, refinement and Z2 decisions of the generator's run_topology."""
    a, b = core.plane_axes(fixed)
    n_t = transverse_points + transverse_points % 2
    for attempt in range(refinements + 1):
        transverse = np.arange(n_t) / n_t
        results = [
            generator_loop(core, model, plane_loops(fixed, value, [t], loop_points)[0], occupied, a)
            for t in transverse
        ]
        phases = np.array([phase for phase, _ in results])
        centres = np.array([wcc for _, wcc in results])
        chern, step = core.chern_from_phases(phases)
        flow = None
        if z2 and (abs(value) < 1e-9 or abs(value - 0.5) < 1e-9):
            flow = core.z2_from_wcc(centres[: n_t // 2 + 1], 0.01)
        converged = step < np.pi / 2 and abs(chern - round(chern)) < 0.05
        if converged and not (flow and flow["ambiguous"]):
            break
        if attempt < refinements:
            n_t *= 2
    return {"n_t": n_t, "phases": phases, "centres": centres, "chern": chern, "step": step}, converged, flow


def test_phase_winding_wrap_and_kramers_ports_equal_the_generator(generator_core):
    rng = np.random.default_rng(31)
    for count in (1, 2, 7, 60):
        phases = rng.uniform(-3 * np.pi, 3 * np.pi, size=count)
        assert_array_equal(topology.wrap_phase(phases), generator_core.wrap_phase(phases))
        assert topology.chern_from_phases(phases) == generator_core.chern_from_phases(phases)
    winding = -2 * np.pi * np.arange(40) / 40 + rng.normal(scale=0.05, size=40)
    assert topology.chern_from_phases(winding) == generator_core.chern_from_phases(winding)
    cases = [rng.normal(size=(5, 6)), np.repeat(rng.normal(size=(4, 3)), 2, axis=1), rng.normal(size=7)]
    cases += [
        np.zeros((0,)),
        np.repeat(rng.normal(size=(2, 2)), 2, axis=1) + rng.normal(scale=1e-7, size=(2, 4)),
    ]
    for energies in cases:
        for tolerance in (1e-9, 1e-6, 1e-3):
            assert topology.kramers_degenerate(energies, tolerance) == generator_core.kramers_degenerate(
                energies, tolerance
            )


def test_largest_gap_z2_port_equals_the_generator(generator_core):
    rng = np.random.default_rng(32)
    for steps in (1, 2, 5, 31):
        for occupied in (0, 1, 2, 4, 6):
            flows = [rng.uniform(-1.0, 2.0, size=(steps, occupied))]
            paired = rng.uniform(0, 1, size=(steps, occupied // 2))
            flows.append(
                np.sort(np.concatenate([paired, paired + rng.normal(scale=0.03, size=paired.shape)], 1))
            )
            for flow in flows:
                for tolerance in (0.0, 0.01, 0.05):
                    port = topology.z2_from_wcc(flow, tolerance)
                    original = generator_core.z2_from_wcc(flow, tolerance)
                    assert (port.z2, port.crossings, port.ambiguous) == (
                        original["z2"],
                        original["crossings"],
                        original["ambiguous"],
                    )
                    assert_array_equal(port.gap_midpoints, original["gap_midpoints"])
                    assert port.min_largest_gap == original["min_largest_gap"]
                for row in flow:
                    assert topology.largest_gap_midpoint(row) == generator_core._largest_gap_midpoint(row)
    assert [topology.plane_axes(c) for c in range(3)] == [generator_core.plane_axes(c) for c in range(3)]


def test_queer_wilson_links_equal_the_generator_loop_on_cell_periodic_vectors(generator_core):
    rng = np.random.default_rng(33)
    worst = 0.0
    for _ in range(9):
        bands = int(rng.integers(2, 5))
        occupied = int(rng.integers(1, bands))
        model = random_layer(rng, bands, occupied)
        for fixed in (2, 0, 1):
            loop_axis = topology.plane_axes(fixed)[0]
            for t in (0.0, 0.3, 0.5):
                loop = plane_loops(fixed, 0.0, [t], 17)[0]
                port = wilson_loop(model, loop[None], occupied_bands=occupied)
                phase, centres = generator_loop(generator_core, model, loop, occupied, loop_axis)
                worst = max(worst, circular_mismatch(port["phases"][0], centres))
                worst = max(worst, abs(float(topology.wrap_phase(port["berry_phase"][0] - phase))))
    assert worst < 1e-10


@pytest.mark.parametrize("case", ["qwz", "kane_mele"])
def test_wilson_plane_refinement_and_z2_follow_the_generator(generator_core, case):
    if case == "qwz":
        lattice = np.array([[1.0, 0.0, 0.0], [0.3, 1.1, 0.0], [0.0, 0.0, 10.0]])
        centres = np.array([[0.2, 0.1, 5.0], [0.9, 0.7, 5.3]])
        model, occupied, transverse, z2 = qwz(-1, lattice, centres), 1, 2, False
    else:
        model, occupied, transverse, z2 = kane_mele(1.0), 2, 6, True
    port = wilson_plane(
        model,
        2,
        0.0,
        occupied_bands=occupied,
        loop_points=21,
        transverse_points=transverse,
        z2=z2,
        max_refinements=4,
    )
    original, converged, flow = generator_plane(
        generator_core, model, 2, 0.0, occupied, 21, transverse, 4, z2
    )
    assert port.metadata["transverse_points"] == original["n_t"]
    assert bool(port["converged"]) == converged and int(port["chern"]) == int(round(original["chern"]))
    assert_allclose(port["chern_raw"], original["chern"], atol=1e-10)
    assert_allclose(port["max_phase_step"], original["step"], atol=1e-10)
    assert_allclose(topology.wrap_phase(port["berry_phase"] - original["phases"]), 0, atol=1e-10)
    assert max(circular_mismatch(a, b) for a, b in zip(port["wcc"], original["centres"])) < 1e-10
    if flow is None:
        assert "z2" not in port.arrays
    else:
        assert (int(port["z2"]), int(port["z2_crossings"]), bool(port["z2_ambiguous"])) == (
            flow["z2"],
            flow["crossings"],
            flow["ambiguous"],
        )
        assert_allclose(port["gap_midpoints"], flow["gap_midpoints"], atol=1e-10)


def random_complex(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def test_fermi_sea_curvature_and_velocity_rotation_equal_the_generator(generator_core):
    rng = np.random.default_rng(41)
    for leading in ((), (3,), (2, 5)):
        for bands in (1, 2, 4, 7):
            energies = rng.normal(scale=0.5, size=leading + (bands,))
            if bands >= 4:
                # Pairs straddling E_F just inside and just outside the skip threshold, and a zero level.
                energies[..., 0], energies[..., 1] = -0.4e-6, 0.4e-6
                energies[..., 2], energies[..., 3] = -1.5e-6, 0.0
            v_a = random_complex(rng, leading + (bands, bands))
            v_b = random_complex(rng, leading + (bands, bands))
            for gap in (0.0, 1e-6, 1e-3):
                port = geometry.fermi_sea_curvature(v_a, v_b, energies, min_gap=gap)
                original = generator_core.fermi_sea_curvature(v_a, v_b, energies, min_gap=gap)
                assert_allclose(
                    port, original, rtol=1e-12, atol=1e-12 * np.max(np.abs(original), initial=1.0)
                )
    velocity = random_complex(rng, (2, 4, 3, 3))
    rows = np.linalg.qr(random_complex(rng, (4, 3, 3)))[0]
    assert_array_equal(
        geometry.rotate_velocity(velocity, rows), generator_core.rotate_velocity(velocity, rows)
    )


@pytest.mark.parametrize("case", ["haldane", "random"])
def test_berry_plane_equals_the_generator_maps_on_pythtb_velocities(generator_core, case):
    rng = np.random.default_rng(42)
    model = haldane() if case == "haldane" else random_layer(rng, 3, 1)
    occupied = 1
    points = plane_kpoints(2, 0.0, 5)
    sea = berry_plane(model, points, plane=(0, 1), fermi_energy=0.0)["omega_reduced"]
    bands = berry_plane(model, points, plane=(0, 1), occupied_bands=occupied)["omega_reduced"]
    original_sea, original_bands = [], []
    for k in points:
        hamiltonian, velocity = cell_periodic_operators(model, k)
        energies, vectors = np.linalg.eigh(hamiltonian)
        rotated = generator_core.rotate_velocity(velocity[:2, None], vectors.T[None])[:, 0]
        original_sea.append(generator_core.fermi_sea_curvature(rotated[0], rotated[1], energies))
        # PythTB's band-set Kubo curvature: the same sum with the zero placed inside the gap.
        middle = 0.5 * (energies[occupied - 1] + energies[occupied])
        original_bands.append(generator_core.fermi_sea_curvature(rotated[0], rotated[1], energies - middle))
    scale = np.abs(original_bands).max()
    assert_allclose(sea, original_sea, rtol=0, atol=1e-10 * scale)
    assert_allclose(bands, original_bands, rtol=0, atol=1e-10 * scale)
