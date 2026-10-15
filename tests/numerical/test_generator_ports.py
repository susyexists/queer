"""Ported Generator PythTB functions reproduce the vendored originals.

Each test feeds identical inputs to a QUEER port and to the original function in
``tests/oracles`` (loaded by the ``generator_core`` fixture). Analytic acceptance
of the same code lives in the topology, Berry-plane, slab and estimator tests.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import eigh
from tight_binding_models import cell_periodic_operators, haldane, kane_mele, qwz

from queer.compat.generator import estimators
from queer.models.evaluation import states
from queer.models.wannier import WannierModel
from queer.observables.geometry import berry_plane, wilson_loop
from queer.observables.surface import slab_edge_spectrum, slab_eigensystem
from queer.observables.topology import plane_kpoints, plane_loops, wilson_plane
from queer.physics import geometry, slab, topology

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


def test_layer_blocks_and_slab_hamiltonian_equal_the_generator(generator_core):
    rng = np.random.default_rng(61)
    cube = np.array([[a, b, c] for a in (-2, -1, 0, 1) for b in (-1, 0, 1, 2) for c in (-1, 0, 1)])
    for _ in range(8):
        bands = int(rng.integers(1, 4))
        translations = cube[rng.permutation(len(cube))[: int(rng.integers(5, len(cube)))]]
        hopping = random_complex(rng, (len(translations), bands, bands))
        finite = int(rng.integers(0, 3))
        others = [axis for axis in range(3) if axis != finite]
        periodic = others[: int(rng.integers(0, 3))]
        k = rng.uniform(-0.5, 0.5, size=3)
        port = slab.layer_blocks(translations, hopping, finite, periodic, k)
        original = generator_core.layer_blocks(translations, hopping, finite, periodic, k)
        assert list(port) == list(original)
        for d in port:
            assert_array_equal(port[d], original[d])
        for cells in (1, 2, 5):
            assert_array_equal(
                slab.slab_hamiltonian(port, cells), generator_core.slab_hamiltonian(original, cells)
            )


def test_rotated_edge_weights_and_crossings_equal_the_generator(generator_core):
    rng = np.random.default_rng(62)
    for cells, orbitals in ((4, 2), (7, 3), (2, 1), (5, 2)):
        size = cells * orbitals
        vectors = np.linalg.qr(random_complex(rng, (size, size)))[0][:, : max(1, size - 2)]
        energies = np.sort(rng.normal(scale=0.2, size=vectors.shape[1]))
        if len(energies) >= 5:
            energies[1] = energies[0]
            energies[3], energies[4] = energies[2] + 4e-6, energies[2] + 8e-6
        for edge in (1, 2, 5):
            for levels in (None, energies, energies[::-1].copy()):
                port = estimators.edge_weights(vectors, cells, orbitals, edge, energies=levels)
                original = generator_core.edge_weights(vectors, cells, orbitals, edge, energies=levels)
                for a, b in zip(port, original):
                    assert_array_equal(a, b)
    for _ in range(20):
        count = int(rng.integers(1, 12))
        levels = [rng.normal(scale=0.3, size=int(rng.integers(0, 6))) for _ in range(count)]
        left = [rng.uniform(0, 1, size=len(e)) for e in levels]
        right = [rng.uniform(0, 1, size=len(e)) for e in levels]
        for periodic in (False, True):
            for threshold in (0.3, 0.5):
                assert estimators.edge_crossings(
                    levels, left, right, periodic=periodic, threshold=threshold
                ) == generator_core.edge_crossings(
                    levels, left, right, periodic=periodic, threshold=threshold
                )


@pytest.mark.parametrize("case", ["haldane", "pt_kane_mele"])
def test_slab_levels_and_cluster_weights_equal_the_generator_windowed_solve(generator_core, case):
    model = haldane() if case == "haldane" else kane_mele(0.0, rashba=0.0)
    model.fermi_energy = 0.05
    points = np.zeros((7, 3))
    points[:, 0] = -0.5 + (np.arange(7) + 0.5) / 7
    cells, window = 8, 1.0
    spectrum = slab_edge_spectrum(model, points, finite_axis=1, cells=cells, window=window)
    table = model.hopping_matrices.copy()
    home = np.flatnonzero(np.all(model.translations == 0, axis=1))[0]
    table[home] = table[home] - model.fermi_energy * np.eye(model.nbnd)
    clustered = 0
    for index, k in enumerate(points):
        blocks = generator_core.layer_blocks(model.translations, table, 1, [0], k)
        levels, vectors = eigh(
            generator_core.slab_hamiltonian(blocks, cells), subset_by_value=(-window, window), driver="evr"
        )
        left, right = generator_core.edge_weights(vectors, cells, model.nbnd, 2, energies=levels)
        count = spectrum["state_count"][index]
        assert_array_equal(spectrum["energies"][index, :count], levels)
        assert_array_equal(slab_eigensystem(model, k, finite_axis=1, cells=cells, window=window)[0], levels)
        labels = spectrum["cluster"][index, :count]
        assert_array_equal(labels, slab.degenerate_clusters(levels, 1e-5))
        assert_allclose(spectrum["left_weight"][index, :count], slab.cluster_totals(left, labels), atol=1e-12)
        assert_allclose(
            spectrum["right_weight"][index, :count], slab.cluster_totals(right, labels), atol=1e-12
        )
        clustered += int(np.sum(spectrum["cluster_size"][index, :count] > 1))
    assert clustered > 0 if case == "pt_kane_mele" else clustered == 0


def random_lattice(rng):
    return np.diag(rng.uniform(2.5, 6.0, size=3)) + rng.normal(scale=0.5, size=(3, 3))


def hexagonal_lattice(angle):
    second = [np.cos(np.radians(angle)), np.sin(np.radians(angle)), 0.0]
    return np.array([[3.0, 0.0, 0.0], [3.0 * second[0], 3.0 * second[1], 0.0], [0.0, 0.0, 20.0]])


def random_segments(rng, count):
    segments, previous = [], rng.uniform(-0.5, 0.5, size=3)
    for index in range(count):
        start = previous if rng.uniform() < 0.6 else rng.uniform(-0.5, 0.5, size=3)
        end = rng.uniform(-0.5, 0.5, size=3)
        if index == 2:
            end = start.copy()
            end[2] = start[2] + 0.25
        segments.append([f"P{index}", start.tolist(), f"Q{index}", end.tolist()])
        previous = end
    return segments


def test_k_space_sampling_ports_equal_the_generator(generator_core):
    rng = np.random.default_rng(71)
    lattices = [random_lattice(rng) for _ in range(4)] + [hexagonal_lattice(120), hexagonal_lattice(60)]
    for lattice in lattices:
        divisions = rng.integers(1, 7, size=3)
        for periodic in ([0, 1], [0, 1, 2], [1, 2], [0]):
            for fixed in (None, {2: 0.5}, {0: 0.25, 1: -0.5}):
                assert_array_equal(
                    estimators.uniform_mesh(divisions, periodic, fixed),
                    generator_core.uniform_mesh(divisions, periodic, fixed),
                )
            segments = random_segments(rng, 5)
            assert estimators.project_path(segments, periodic) == generator_core.project_path(
                segments, periodic
            )
        for points in (1, 7, 60, 400):
            segments = random_segments(rng, int(rng.integers(1, 6)))
            port = estimators.sample_path(segments, lattice, points)
            original = generator_core.sample_path(segments, lattice, points)
            assert_array_equal(port["kpoints"], original["kpoints"])
            for key in ("tick_indices", "tick_labels", "break_indices"):
                assert port[key] == original[key]
            assert_array_equal(
                estimators.k_distance(port["kpoints"], lattice, port["break_indices"]),
                generator_core.k_distance(original["kpoints"], lattice, original["break_indices"]),
            )
        for a, b in ((0, 1), (1, 2), (2, 0)):
            for port, original in zip(
                estimators.plane_frame(lattice, a, b), generator_core.plane_frame(lattice, a, b)
            ):
                assert_array_equal(port, original)
            port = estimators.surface_path(lattice, a, b, 120)
            original = generator_core.surface_path(lattice, a, b, 120)
            assert_array_equal(port["kpoints"], original["kpoints"])
            assert (
                port["tick_labels"] == original["tick_labels"]
                and port["break_indices"] == original["break_indices"]
            )
    with pytest.raises(ValueError, match="empty"):
        estimators.sample_path([], np.eye(3), 10)


def test_band_window_deviation_and_gap_ports_equal_the_generator(generator_core):
    rng = np.random.default_rng(72)
    windows = [
        {},
        {"dis_froz_max": 1.0},
        {"dis_froz_max": 1.0, "dis_win_min": -3.0},
        {"dis_froz_min": -1.0, "dis_froz_max": 2.0},
        {"dis_win_min": -4.0, "dis_win_max": 5.0},
        {"dis_froz_min": -1.0, "dis_win_min": -2.0, "dis_win_max": 3.0},
    ]
    for win in windows:
        for fermi in (0.0, 1.3):
            assert estimators.comparison_window(win, fermi) == generator_core.comparison_window(win, fermi)
    model = rng.normal(size=(40, 6))
    reference = model + rng.normal(scale=1e-3, size=model.shape)
    for window in (None, (-0.5, 0.5), (10.0, 11.0), (-1.0, 1.0)):
        assert estimators.band_deviation(model, reference, window) == generator_core.band_deviation(
            model, reference, window
        )
    assert estimators.band_deviation(model[:0], reference[:0], None) == generator_core.band_deviation(
        model[:0], reference[:0], None
    )
    for bands, offset in ((1, 0.0), (4, 0.0), (4, 0.6), (6, -0.3), (5, 7.0)):
        kpoints = rng.uniform(-0.5, 0.5, size=(30, 3))
        centres = np.linspace(-2.0, 2.0, bands) + offset
        energies = np.sort(centres + rng.normal(scale=0.25, size=(30, bands)), axis=1)
        port, original = estimators.BandEdges(bands), generator_core.BandEdges(bands)
        for start in (0, 11, 23):
            chunk = slice(start, start + 12)
            port_part, original_part = estimators.BandEdges(bands), generator_core.BandEdges(bands)
            port_part.add(kpoints[chunk], energies[chunk])
            original_part.add(kpoints[chunk], energies[chunk])
            port.merge(port_part)
            original.merge(original_part)
        for name in (
            "minimum",
            "maximum",
            "argmin",
            "argmax",
            "direct",
            "direct_at",
            "count_min",
            "count_max",
            "n_k",
        ):
            assert_array_equal(getattr(port, name), getattr(original, name))
        for tolerance in (0.0, 0.02, 0.5):
            for occupied in (None, *range(1, bands)):
                assert estimators.classify_gap(port, tolerance, occupied) == generator_core.classify_gap(
                    original, tolerance, occupied
                )
        for invalid in (0, bands):
            with pytest.raises(ValueError, match="occupied_bands"):
                estimators.classify_gap(port, 0.02, invalid)
            with pytest.raises(generator_core.PythtbError):
                generator_core.classify_gap(original, 0.02, invalid)


def test_histogram_dos_kernel_equals_the_generator(generator_core):
    rng = np.random.default_rng(73)
    for sigma in (0.0, -1.0, 0.005, 0.03, 0.2):
        for shape in ((400,), (3, 400), (37,)):
            histogram = rng.uniform(0, 3, size=shape)
            assert_array_equal(
                estimators.gaussian_dos(histogram, 0.005, sigma),
                generator_core.gaussian_dos(histogram, 0.005, sigma),
            )


def test_species_and_localization_ports_equal_the_generator(generator_core):
    rng = np.random.default_rng(74)
    for _ in range(5):
        lattice = random_lattice(rng)
        atoms = int(rng.integers(1, 6))
        structure = {
            "species": [str(rng.choice(["Mn", "Bi", "Te"])) for _ in range(atoms)],
            "cart": rng.uniform(0, 1, size=(atoms, 3)) @ lattice,
        }
        centres = rng.uniform(-0.6, 1.6, size=(7, 3)) @ lattice
        port = estimators.orbital_species(centres, structure, lattice)
        original = generator_core.orbital_species(centres, structure, lattice)
        assert port[0] == original[0]
        assert_array_equal(port[1], original[1])
        for axes in ([], [2], [0, 2], [0, 1, 2]):
            assert_array_equal(
                estimators.unwrap_centres(centres, lattice, axes),
                generator_core.unwrap_centres(centres, lattice, axes),
            )
        translations = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-2, 0, 2) for c in (0, 1)])
        hopping = random_complex(rng, (len(translations), 7, 7)) * np.exp(
            -rng.uniform(0, 12, size=(len(translations), 7, 7))
        )
        hopping[rng.uniform(size=hopping.shape) < 0.2] = 0
        port = estimators.hopping_decay(translations, hopping, centres, lattice)
        original = generator_core.hopping_decay(translations, hopping, centres, lattice)
        assert port.keys() == original.keys()
        for key in port:
            assert_array_equal(port[key], original[key])
    for broken in (None, {}, {"species": ["A"], "cart": [[0, 0, 0], [1, 1, 1]]}, {"species": [], "cart": []}):
        port = estimators.orbital_species(np.zeros((3, 3)), broken, np.eye(3))
        original = generator_core.orbital_species(np.zeros((3, 3)), broken, np.eye(3))
        assert port[0] == original[0] == ["all"]
        assert_array_equal(port[1], original[1])
    zero = np.zeros((1, 2, 2))
    assert estimators.hopping_decay([[0, 0, 0]], zero, np.zeros((2, 3)), np.eye(3)) is None
    assert generator_core.hopping_decay([[0, 0, 0]], zero, np.zeros((2, 3)), np.eye(3)) is None
