"""Independent acceptance cases for native orbital-dependent WS images."""

from itertools import product

import numpy as np
import pytest

from queer.io.wannier import WannierData, normalized_operators, recompute_wsvec
from queer.io.wannier_ws import ComputedWignerSeitzShifts
from queer.models.wannier import WannierModel


def source(translations, hopping, degeneracies=None, *, position=None, lattice=None):
    hopping = np.asarray(hopping, complex)
    if degeneracies is None:
        degeneracies = np.ones(len(hopping), dtype=int)
    return WannierData(
        np.asarray(translations, dtype=int), hopping, np.asarray(degeneracies), position, lattice
    )


def test_half_supercell_images_split_hr_and_position_weights_once():
    raw = source([[1, 0, 0]], [[[6]]], [2], position=np.array([[[[2, 4, 6]]]], complex))
    mapping = recompute_wsvec(raw, [[0, 0, 0]], np.eye(3), [2, 2, 2])
    np.testing.assert_array_equal(mapping.shifts[(1, 0, 0, 1, 1)], [[-2, 0, 0], [0, 0, 0]])
    translations, h, r = normalized_operators(raw, "ws_distance", mapping)
    np.testing.assert_array_equal(translations, [[-1, 0, 0], [1, 0, 0]])
    np.testing.assert_allclose(h[:, 0, 0], [1.5, 1.5])
    np.testing.assert_allclose(r[:, 0, 0], [[0.5, 1, 1.5], [0.5, 1, 1.5]])


def test_skew_lattice_uses_cartesian_distance_not_rounded_fractional_distance():
    lattice = np.array([[1, 0, 0], [0.9, 0.3, 0], [0, 0, 2]])
    centres = np.array([[0, 0, 0], [0.51, 0.09, 0]])
    raw = source([[0, 0, 0]], [[[0, 1], [1, 0]]])
    mapping = recompute_wsvec(raw, centres, lattice, [1, 1, 1])
    shift = mapping.shifts[(0, 0, 0, 1, 2)]
    np.testing.assert_array_equal(shift, [[0, -1, 0]])
    fractional = (centres[1] - centres[0]) @ np.linalg.inv(lattice)
    np.testing.assert_array_equal(np.round(fractional), [0, 0, 0])
    # An independently larger integer cube confirms the reported nearest image.
    candidates = np.array(list(product(range(-6, 7), repeat=3)))
    distance = np.linalg.norm(candidates @ lattice + centres[1] - centres[0], axis=1)
    np.testing.assert_array_equal(candidates[np.argmin(distance)], shift[0])
    translations, hopping, _ = normalized_operators(raw, "ws_distance", mapping)
    model = WannierModel(translations, hopping, lattice=lattice, centres=centres)
    for point in [[0, 0, 0], [0.23, -0.17, 0.11]]:
        np.testing.assert_allclose(model.energies([point])[:, 0], [-1, 1], atol=1e-14)


def test_tied_images_in_a_skew_cell_are_all_retained():
    lattice = np.array([[1, 0, 0], [0.5, np.sqrt(3) / 2, 0], [0, 0, 2]])
    raw = source([[0, 0, 0]], [[[1, 2], [2, 3]]])
    mapping = recompute_wsvec(raw, [[0, 0, 0], [0.5, 0, 0]], lattice, [1, 1, 1])
    np.testing.assert_array_equal(mapping.shifts[(0, 0, 0, 1, 2)], [[-1, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(mapping.shifts[(0, 0, 0, 2, 1)], [[0, 0, 0], [1, 0, 0]])


def test_original_unwrapped_wannier_centres_determine_images():
    raw = source([[0, 0, 0]], [[[0, 1], [1, 0]]])
    mapping = recompute_wsvec(raw, [[0, 0, 0], [3, 0, 0]], np.eye(3), [2, 2, 2], search_size=3)
    np.testing.assert_array_equal(mapping.shifts[(0, 0, 0, 1, 2)], [[-4, 0, 0], [-2, 0, 0]])
    with pytest.raises(ValueError, match="increase search_size"):
        recompute_wsvec(raw, [[0, 0, 0], [30, 0, 0]], np.eye(3), [2, 2, 2])


def test_ws_reconstruction_preserves_the_original_monkhorst_pack_mesh():
    lattice = np.array([[1.7, 0, 0], [0.4, 1.2, 0], [0.1, 0.2, 2.1]])
    centres = np.array([[0, 0, 0], [0.6, 0.3, 0.2]])
    forward = np.array([[0.2, 0.7 + 0.1j], [-0.4j, 0.3]])
    raw = source(
        [[-1, 0, 0], [0, 0, 0], [1, 0, 0]],
        [forward.conj().T, [[-0.2, 0.4j], [-0.4j, 0.5]], forward],
        [2, 1, 2],
    )
    mapping = recompute_wsvec(raw, centres, lattice, [2, 2, 1])
    r, h, _ = normalized_operators(raw, "ws_distance", mapping)
    original = WannierModel(raw.translations, raw.hopping / raw.degeneracies[:, None, None], lattice=lattice)
    corrected = WannierModel(r, h, lattice=lattice)
    for point in product([0, 0.5], [0, 0.5], [0]):
        np.testing.assert_allclose(corrected.hamiltonian(point), original.hamiltonian(point), atol=1e-14)


def test_generic_k_hamiltonian_against_independent_exhaustive_image_sum():
    lattice = np.array([[1.7, 0, 0], [0.4, 1.2, 0], [0.1, 0.2, 2.1]])
    centres = np.array([[0, 0, 0], [0.6, 0.3, 0.2]])
    forward = np.array([[0.2, 0.7 + 0.1j], [-0.4j, 0.3]])
    raw = source(
        [[-1, 0, 0], [0, 0, 0], [1, 0, 0]],
        [forward.conj().T, [[-0.2, 0.4j], [-0.4j, 0.5]], forward],
        [2, 1, 2],
    )
    grid = np.array([2, 2, 1])
    mapping = recompute_wsvec(raw, centres, lattice, grid)
    r, h, _ = normalized_operators(raw, "ws_distance", mapping)
    model = WannierModel(r, h, lattice=lattice, centres=centres)
    # Enumerate a substantially larger cube, independently form the complex
    # phase average for each pair, and avoid the compact representation entirely.
    candidates = np.array(list(product(range(-5, 6), repeat=3))) * grid
    for point in [[0.23, -0.17, 0.31], [0.43, 0.37, -0.22]]:
        expected = np.zeros((2, 2), complex)
        for translation, block, degeneracy in zip(
            raw.translations, raw.hopping, raw.degeneracies, strict=True
        ):
            for i, j in product(range(2), repeat=2):
                distances = np.linalg.norm(
                    (translation + candidates) @ lattice + centres[j] - centres[i], axis=1
                )
                nearest_images = candidates[abs(distances - distances.min()) < 1e-5]
                phase_average = np.exp(2j * np.pi * ((translation + nearest_images) @ point)).mean()
                expected[i, j] += block[i, j] / degeneracy * phase_average
        np.testing.assert_allclose(model.hamiltonian(point), expected, rtol=1e-14, atol=1e-14)


def test_compact_mapping_and_normalization_are_chunk_and_translation_order_independent():
    raw = source([[-1, 0, 0], [0, 0, 0], [1, 0, 0]], np.arange(12).reshape(3, 2, 2))
    args = (raw, [[0, 0, 0], [0.5, 0, 0]], np.eye(3), [2, 2, 2])
    small = recompute_wsvec(*args, chunk_bytes=20_000)
    large = recompute_wsvec(*args, chunk_bytes=1_000_000)
    assert isinstance(small.shifts, ComputedWignerSeitzShifts)
    assert len(small.shifts) == 12
    assert small.shifts.pair_chunk == 1
    assert not small.shifts.offsets.flags.writeable
    assert not small.shifts.image_ids.flags.writeable
    for key in small.shifts:
        np.testing.assert_array_equal(small.shifts[key], large.shifts[key])
    a = normalized_operators(raw, "ws_distance", small)
    b = normalized_operators(raw, "ws_distance", large)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1])
    reordered = source(raw.translations[::-1], raw.hopping[::-1], raw.degeneracies[::-1])
    c = normalized_operators(reordered, "ws_distance", small)
    np.testing.assert_array_equal(a[0], c[0])
    np.testing.assert_allclose(a[1], c[1])
    with pytest.raises(KeyError):
        small.shifts[(0, 0, 0, 3, 1)]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"mp_grid": [1.0, 1, 1]}, "mp_grid"),
        ({"mp_grid": [1, 0, 1]}, "mp_grid"),
        ({"mp_grid": [1, 1]}, "mp_grid"),
        ({"search_size": 0}, "search_size"),
        ({"search_size": True}, "search_size"),
        ({"distance_tolerance": 0}, "distance_tolerance"),
        ({"distance_tolerance": float("nan")}, "distance_tolerance"),
        ({"chunk_bytes": 100}, "cannot fit one"),
        ({"centres": [[0, 0, 0], [1, 0, 0]]}, "centres"),
    ],
)
def test_invalid_reconstruction_inputs_raise(kwargs, match):
    raw = source([[0, 0, 0]], [[[1]]])
    settings = dict(centres=[[0, 0, 0]], lattice=np.eye(3), mp_grid=[1, 1, 1])
    settings.update(kwargs)
    with pytest.raises(ValueError, match=match):
        recompute_wsvec(raw, **settings)


def test_wannier_metadata_mismatch_is_rejected_before_normalizing():
    raw = source([[0, 0, 0]], [[[1]]], lattice=np.eye(3))
    with pytest.raises(ValueError, match="lattice does not match"):
        recompute_wsvec(raw, [[0, 0, 0]], np.eye(3) * 2, [1, 1, 1])
    mapping = recompute_wsvec(raw, [[0, 0, 0]], np.eye(3), [1, 1, 1])
    changed_translation = source([[1, 0, 0]], [[[1]]])
    changed_orbitals = source([[0, 0, 0]], [np.eye(2)])
    for changed in [changed_translation, changed_orbitals]:
        with pytest.raises(ValueError, match="keys do not match"):
            normalized_operators(changed, "ws_distance", mapping)


def test_search_certification_detects_unresolved_skew_lattice_images():
    raw = source([[0, 0, 0]], [[[0, 1], [1, 0]]])
    # Highly oblique primitive vectors admit short images far outside a small
    # coordinate cube; a local minimum alone does not certify the global image.
    lattice = [[1, 0, 0], [10.01, 0.01, 0], [0, 0, 1]]
    with pytest.raises(ValueError, match="increase search_size"):
        recompute_wsvec(raw, [[0, 0, 0], [0.4, 0.004, 0]], lattice, [1, 1, 1])
