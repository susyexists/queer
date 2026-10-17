"""Open slabs and ribbons: block-Toeplitz levels, chiral edge modes and cluster weights."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from tight_binding_models import haldane, kane_mele, qwz

from queer.compat.generator.estimators import bulk_at_fermi, edge_crossings, half_ribbon_weights
from queer.execution import ExecutionOptions
from queer.models.wannier import WannierModel
from queer.observables.geometry import chern_number
from queer.observables.surface import slab_edge_spectrum, slab_eigensystem, surface_state
from queer.physics.slab import (
    cluster_totals,
    degenerate_clusters,
    edge_probabilities,
    layer_blocks,
    slab_hamiltonian,
)


def ribbon_kpoints(count):
    """The ribbon's 1D zone along a1, half a step off the high-symmetry points."""
    points = np.zeros((count, 3))
    points[:, 0] = -0.5 + (np.arange(count) + 0.5) / count
    return points


def ribbon_flow(model, points, cells, window=0.5):
    """QUEER_SLAB's weighted flow on a ribbon cut along a2: half-ribbon weights per level and the bulk check."""
    energies, half, middle = [], [], []
    for k in points:
        levels, vectors = slab_eigensystem(model, k, finite_axis=1, cells=cells, window=window)
        first, inner = half_ribbon_weights(vectors, cells, model.nbnd, energies=levels)
        energies.append(levels)
        half.append(first)
        middle.append(inner)
    hopping = np.array(model.hopping_matrices, dtype=complex)
    home = np.flatnonzero(np.all(model.translations == 0, axis=1))[0]
    hopping[home] -= model.fermi_energy * np.eye(model.nbnd)
    bulk = bulk_at_fermi(model.translations, hopping, 1, points, 2 * cells)
    return edge_crossings(energies, half, middle, periodic=True, window=window, bulk_metal=bulk["metal"])


def test_open_chain_levels_and_ribbon_levels_equal_independent_constructions():
    cells, hop = 12, -0.7
    chain = WannierModel([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], [[[0.0]], [[hop]], [[hop]]], lattice=np.eye(3))
    levels, vectors = slab_eigensystem(chain, [0, 0, 0], finite_axis=0, cells=cells, window=5.0)
    expected = 2 * hop * np.cos(np.pi * np.arange(1, cells + 1) / (cells + 1))
    assert_allclose(levels, np.sort(expected), atol=1e-13)
    assert vectors.shape == (cells, cells)
    model = qwz(-1)
    model.fermi_energy = 0.2
    points = ribbon_kpoints(5)
    spectrum = slab_edge_spectrum(model, points, finite_axis=1, cells=8, window=10.0)
    reference = surface_state(model, points, surface_direction=1, method="bands", slab_layers=8)["energies"]
    assert_array_equal(spectrum["state_count"], 16)
    assert_allclose(spectrum["energies"], np.sort(reference, axis=0).T, atol=1e-12)
    assert spectrum.metadata["periodic_axes"] == [0] and spectrum.units["energies"] == "eV"


@pytest.mark.parametrize("phase", [np.pi / 2, -np.pi / 2])
def test_haldane_ribbon_carries_one_chiral_mode_on_each_edge(phase):
    model = haldane(phase=phase)
    chern = round(float(chern_number(model, grid=(12, 12, 1))["chern_number"]))
    assert abs(chern) == 1
    tally = ribbon_flow(model, ribbon_kpoints(60), 20)
    assert tally["left"]["net"] == -tally["right"]["net"] and abs(tally["left"]["net"]) == 1
    assert tally["left"]["up"] + tally["left"]["down"] == 1
    # The same chiral pair at every energy across the gap the ribbon's window shows.
    assert tally["status"] == "resolved", tally["reason"]
    for row in tally["per_energy"]:
        assert row["left_flow"] == pytest.approx(tally["left"]["net"], abs=0.1)
        assert row["right_flow"] == pytest.approx(-row["left_flow"], abs=1e-12)
    # Reversing the Chern number reverses the chirality on each edge.
    assert tally["left"]["net"] == (1 if chern == -1 else -1)


def test_trivial_haldane_ribbon_has_no_edge_crossings_in_serial_or_parallel_runs():
    model = haldane(staggered=1.0)
    assert abs(float(chern_number(model, grid=(12, 12, 1))["chern_number"])) < 1e-10
    tally = ribbon_flow(model, ribbon_kpoints(40), 20)
    # Nothing crosses, and nothing lies in the gap: a count of 0 is not certified from absence.
    assert tally["left"]["net"] == tally["right"]["net"] == 0 and tally["status"] == "unresolved"
    assert "rest on absence" in tally["reason"]
    assert all(row["left_flow"] == 0.0 for row in tally["per_energy"])
    spectrum = slab_edge_spectrum(model, ribbon_kpoints(40), finite_axis=1, cells=20, window=0.5)
    options = ExecutionOptions(workers=2, batch_size=4)
    parallel = slab_edge_spectrum(
        model, ribbon_kpoints(40), finite_axis=1, cells=20, window=0.5, execution=options
    )
    for name in ("energies", "left_weight", "right_weight", "cluster", "state_count"):
        assert_array_equal(parallel[name], spectrum[name])


def test_cluster_summed_weights_are_invariant_inside_degenerate_clusters():
    model = kane_mele(0.0, rashba=0.0)
    cells, k = 10, [0.45, 0.0, 0.0]
    levels, vectors = slab_eigensystem(model, k, finite_axis=1, cells=cells, window=1.0)
    labels = degenerate_clusters(levels, 1e-5)
    assert len(levels) and np.all(np.bincount(labels) == 2)
    rng = np.random.default_rng(51)
    rotated = vectors.copy()
    for label in range(labels.max() + 1):
        members = np.flatnonzero(labels == label)
        unitary = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
        rotated[:, members] = vectors[:, members] @ unitary
    before = edge_probabilities(vectors, cells, model.nbnd, 2)
    after = edge_probabilities(rotated, cells, model.nbnd, 2)
    # Per-state weights follow the arbitrary basis of each PT pair; cluster sums do not.
    assert max(np.abs(b - a).max() for b, a in zip(before, after)) > 0.1
    for b, a in zip(before, after):
        assert_allclose(cluster_totals(a, labels), cluster_totals(b, labels), atol=1e-12)
    spectrum = slab_edge_spectrum(model, [k], finite_axis=1, cells=cells, window=1.0)
    count = spectrum["state_count"][0]
    assert count == len(levels)
    assert_array_equal(spectrum["cluster_size"][0, :count], 2)
    assert_allclose(spectrum["left_weight"][0, :count], cluster_totals(before[0], labels), atol=1e-12)
    assert_allclose(spectrum["right_weight"][0, :count], cluster_totals(before[1], labels), atol=1e-12)


def test_layer_blocks_keep_finite_axis_offsets_and_drop_vacuum_hops():
    rng = np.random.default_rng(52)
    translations = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, -1, 0], [0, 0, 1], [0, 2, 0]])
    hopping = rng.normal(size=(6, 2, 2)) + 1j * rng.normal(size=(6, 2, 2))
    k = np.array([0.21, 0.0, 0.4])
    blocks = layer_blocks(translations, hopping, 1, [0], k)
    assert sorted(blocks) == [-1, 0, 1, 2]
    assert_allclose(blocks[0], hopping[0] + np.exp(2j * np.pi * 0.21) * hopping[1])
    assert_allclose(blocks[-1], np.exp(2j * np.pi * 0.21) * hopping[3])
    big = slab_hamiltonian(blocks, 3)
    assert_allclose(big[2:4, 0:2], blocks[-1])
    assert_allclose(big[0:2, 4:6], blocks[2])
    assert_allclose(big[4:6, 0:2], 0)
