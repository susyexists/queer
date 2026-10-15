"""Orbital-resolved WS images, vacuum folding and wsvec file images on independent references."""

from itertools import product

import numpy as np
import pytest

from queer.io.wannier import (
    WannierData,
    WignerSeitzData,
    fold_vacuum_axes,
    normalized_hopping,
    normalized_operators,
    periodic_dirs_from_table,
    read_wsvec,
    recompute_wsvec,
    unfold_from_wsvec,
)


def hermitian_table(seed, translations, nbnd):
    """Random complex table with H(-R) = H(R)^dagger and pairwise equal degeneracies."""
    rng = np.random.default_rng(seed)
    translations = np.asarray(translations)
    index = {tuple(row): i for i, row in enumerate(translations)}
    hopping = np.zeros((len(translations), nbnd, nbnd), complex)
    degeneracies = np.ones(len(translations), int)
    for i, row in enumerate(translations):
        j = index[tuple(-row)]
        if j < i:
            continue
        block = rng.normal(size=(nbnd, nbnd)) + 1j * rng.normal(size=(nbnd, nbnd))
        if i == j:
            block = (block + block.conj().T) / 2
        hopping[i], hopping[j] = block, block.conj().T
        degeneracies[i] = degeneracies[j] = rng.integers(1, 4)
    return WannierData(translations, hopping, degeneracies)


def independent_images(data, centres, lattice, mp_grid, *, search=4, tolerance=1e-5):
    """Exhaustive nearest-image search on a cube twice as wide as the production search."""
    candidates = np.array(list(product(range(-search, search + 1), repeat=3))) * np.asarray(mp_grid)
    table = {}
    for translation, block, degeneracy in zip(data.translations, data.hopping, data.degeneracies):
        for i, j in product(range(data.nbnd), repeat=2):
            vectors = (translation + candidates) @ lattice + centres[j] - centres[i]
            distance = np.sqrt(np.einsum("ta,ta->t", vectors, vectors))
            chosen = candidates[distance - distance.min() < tolerance]
            for shift in chosen:
                key = tuple(int(value) for value in translation + shift)
                table.setdefault(key, np.zeros((data.nbnd, data.nbnd), complex))
                table[key][i, j] += block[i, j] / degeneracy / len(chosen)
    keys = sorted(table)
    return np.array(keys), np.array([table[key] for key in keys])


def fourier(translations, hopping, point):
    return np.einsum("r,rij->ij", np.exp(2j * np.pi * (translations @ point)), hopping)


def test_ws_images_match_independent_search_for_oblique_cell_with_off_centre_orbitals():
    lattice = np.array([[3.0, 0, 0], [1.3, 2.7, 0], [0, 0, 18.0]])
    centres = np.array([[0.4, 0.3, 9.0], [2.95, 1.85, 9.3]])
    translations = [[a, b, 0] for a in range(-2, 3) for b in range(-2, 3)]
    data = hermitian_table(11, translations, 2)
    mapping = recompute_wsvec(data, centres, lattice, [4, 3, 1])
    tied = [key for key in mapping.shifts if len(mapping.shifts[key]) > 1]
    assert tied, "the case must exercise shared hops"
    keys, hopping = normalized_hopping(data, "ws_distance", mapping)
    expected_keys, expected = independent_images(data, centres, lattice, [4, 3, 1])
    np.testing.assert_array_equal(keys, expected_keys)
    np.testing.assert_allclose(hopping, expected, rtol=0, atol=1e-15)
    # Image sharing preserves Hermiticity and every crystal-momentum sum on the MP mesh.
    lookup = {tuple(row): index for index, row in enumerate(keys)}
    for index, row in enumerate(keys):
        np.testing.assert_allclose(hopping[lookup[tuple(-row)]], hopping[index].conj().T, atol=1e-15)
    plain = data.hopping / data.degeneracies[:, None, None]
    for point in product(np.arange(4) / 4, np.arange(3) / 3, [0.0]):
        np.testing.assert_allclose(
            fourier(keys, hopping, np.array(point)),
            fourier(data.translations, plain, np.array(point)),
            atol=1e-13,
        )


def test_thick_slab_ws_routes_hops_through_vacuum_and_folding_restores_the_layer_model():
    lattice = np.array([[4.0, 0, 0], [-2.0, 2 * np.sqrt(3), 0], [0, 0, 30.0]])
    # The slab spans 16.5 A of a 30 A cell: centres more than half a cell apart.
    centres = np.array([[0.1, 0.2, 2.0], [2.1, 1.3, 18.5], [1.0, 2.9, 10.0]])
    translations = [[a, b, 0] for a in range(-2, 3) for b in range(-2, 3)]
    data = hermitian_table(5, translations, 3)
    assert periodic_dirs_from_table(data.translations) == [0, 1]
    mapping = recompute_wsvec(data, centres, lattice, [3, 3, 1])
    keys, hopping = normalized_hopping(data, "ws_distance", mapping)
    expected_keys, expected = independent_images(data, centres, lattice, [3, 3, 1])
    np.testing.assert_array_equal(keys, expected_keys)
    np.testing.assert_allclose(hopping, expected, rtol=0, atol=1e-15)
    # The nearest image of the 16.5 A interlayer hop crosses the 13.5 A vacuum.
    assert np.any(keys[:, 2] != 0)
    assert periodic_dirs_from_table(keys) == [0, 1, 2]
    folded_keys, folded = fold_vacuum_axes(keys, hopping, [0, 1])
    assert not np.any(folded_keys[:, 2])
    assert len(folded_keys) == len(np.unique(keys[:, :2], axis=0))
    rng = np.random.default_rng(3)
    for point in np.column_stack((rng.random((6, 2)), np.zeros(6))):
        np.testing.assert_allclose(
            fourier(folded_keys, folded, point), fourier(keys, hopping, point), rtol=0, atol=1e-13
        )
    # Unfolded, the vacuum images make the slab disperse along kz.
    away = np.array([0.2, 0.1, 0.0]), np.array([0.2, 0.1, 0.5])
    assert not np.allclose(fourier(keys, hopping, away[0]), fourier(keys, hopping, away[1]))
    np.testing.assert_allclose(fourier(folded_keys, folded, away[0]), fourier(folded_keys, folded, away[1]))


def test_fold_keeps_unfolded_tables_and_trailing_operator_axes():
    translations = np.array([[0, 0, 1], [0, 0, 0], [1, 0, -1], [1, 0, 2]])
    stack = np.arange(4 * 2 * 2 * 3, dtype=float).reshape(4, 2, 2, 3) + 0.5j
    keys, folded = fold_vacuum_axes(translations, stack, [0, 1])
    np.testing.assert_array_equal(keys, [[0, 0, 0], [1, 0, 0]])
    np.testing.assert_allclose(folded[0], stack[0] + stack[1])
    np.testing.assert_allclose(folded[1], stack[2] + stack[3])
    same_keys, same = fold_vacuum_axes(translations, stack, [0, 1, 2])
    assert same is stack and np.array_equal(same_keys, translations)
    planar = translations.copy()
    planar[:, 2] = 0
    planar[3, 1] = 2
    unchanged_keys, unchanged = fold_vacuum_axes(planar, stack, [0, 1])
    assert unchanged is stack and np.array_equal(unchanged_keys, planar)
    with pytest.raises(ValueError, match="periodic_dirs"):
        fold_vacuum_axes(translations, stack, [0, 3])
    with pytest.raises(ValueError, match="integer"):
        periodic_dirs_from_table([[0.5, 0, 0]])


def test_wsvec_file_images_match_strict_mode_for_complete_mappings(tmp_path):
    lattice = np.array([[3.0, 0, 0], [1.3, 2.7, 0], [0, 0, 18.0]])
    centres = np.array([[0.4, 0.3, 9.0], [2.95, 1.85, 9.3]])
    translations = [[a, b, 0] for a in range(-2, 3) for b in range(-2, 3)]
    data = hermitian_table(17, translations, 2)
    computed = recompute_wsvec(data, centres, lattice, [4, 3, 1])
    strict_keys, strict = normalized_hopping(data, "ws_distance", computed)
    keys, values = unfold_from_wsvec(data, computed)
    np.testing.assert_array_equal(keys, strict_keys)
    np.testing.assert_allclose(values, strict, rtol=0, atol=1e-15)
    lines = ["## written with use_ws_distance=.true."]
    for key in computed.shifts:
        images = computed.shifts[key]
        lines.append(" ".join(map(str, key)))
        lines.append(str(len(images)))
        lines.extend(" ".join(map(str, image)) for image in images)
    (tmp_path / "seed_wsvec.dat").write_text("\n".join(lines) + "\n")
    packed = read_wsvec(tmp_path / "seed_wsvec.dat")
    file_keys, from_file = unfold_from_wsvec(data, packed)
    np.testing.assert_array_equal(file_keys, strict_keys)
    np.testing.assert_allclose(from_file, strict, rtol=0, atol=1e-15)
    # Stacked operators share each hop's images exactly.
    position = data.hopping[..., None] * np.array([1.0, -2.0, 0.5])
    operator_keys, operators = unfold_from_wsvec(data, packed, position)
    np.testing.assert_array_equal(operator_keys, keys)
    np.testing.assert_allclose(operators, values[..., None] * np.array([1.0, -2.0, 0.5]), atol=1e-15)
    with_position = WannierData(data.translations, data.hopping, data.degeneracies, position)
    _, _, strict_position = normalized_operators(with_position, "ws_distance", computed)
    np.testing.assert_allclose(operators, strict_position, rtol=0, atol=1e-15)


def test_wsvec_file_images_keep_absent_pairs_and_ignore_foreign_entries():
    translations = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
    hopping = np.array([[[1, 2], [3, 4]], [[5, 6], [6, 7]], [[1, 3], [2, 4]]], complex)
    data = WannierData(translations, hopping, np.array([2, 1, 2]))
    mapping = {
        (1, 0, 0, 1, 1): np.array([[0, 0, 0], [-2, 0, 0]]),
        (1, 0, 0, 1, 2): np.empty((0, 3), int),
        (5, 5, 5, 1, 1): np.array([[1, 0, 0]]),
        (0, 0, 0, 3, 1): np.array([[1, 0, 0]]),
    }
    keys, values = unfold_from_wsvec(data, mapping)
    expected = {
        (-1, 0, 0): np.array([[1, 2], [3, 4]]) / 2 + np.array([[0.25, 0], [0, 0]]),
        (0, 0, 0): np.array([[5, 6], [6, 7]]),
        (1, 0, 0): np.array([[0.25, 1.5], [1, 2]]),
    }
    np.testing.assert_array_equal(keys, sorted(expected))
    np.testing.assert_allclose(values, [expected[key] for key in sorted(expected)])
    # Wannier90 lists only zero shifts when use_ws_distance is false.
    zeros = {key: np.zeros((1, 3), int) for key in product(*[[-1, 0, 1], [0], [0]], [1, 2], [1, 2])}
    disabled = WignerSeitzData(zeros, use_ws_distance=False, ndegen_applied=False)
    plain_keys, plain = unfold_from_wsvec(data, disabled)
    standard_keys, standard = normalized_hopping(data, "standard")
    np.testing.assert_array_equal(plain_keys, standard_keys)
    np.testing.assert_allclose(plain, standard)
    with pytest.raises(ValueError, match="already applied"):
        unfold_from_wsvec(data, WignerSeitzData(zeros, use_ws_distance=True, ndegen_applied=True))
    with pytest.raises(ValueError, match="shape"):
        unfold_from_wsvec(data, zeros, np.zeros((3, 2, 3)))
