"""Wannier models from normalized tables: Hermitian policy, diagnostics, cutoffs and positions."""

from itertools import product

import numpy as np
import pytest

from queer.models.wannier import NumericalError, WannierModel

LATTICE = np.array([[2.5, 0.0, 0.0], [0.8, 2.2, 0.0], [0.0, 0.0, 12.0]])
CENTRES = np.array([[0.3, 0.2, 6.0], [1.9, 1.4, 6.4]])


def hermitian_table(seed, translations, nbnd=2):
    rng = np.random.default_rng(seed)
    translations = np.asarray(translations)
    index = {tuple(row): i for i, row in enumerate(translations)}
    hopping = np.zeros((len(translations), nbnd, nbnd), complex)
    for i, row in enumerate(translations):
        j = index[tuple(-row)]
        if j < i:
            continue
        block = rng.normal(size=(nbnd, nbnd)) + 1j * rng.normal(size=(nbnd, nbnd))
        if i == j:
            block = (block + block.conj().T) / 2
        hopping[i], hopping[j] = block, block.conj().T
    return translations, hopping


def planar_translations(radius=1):
    return np.array([[a, b, 0] for a in range(-radius, radius + 1) for b in range(-radius, radius + 1)])


def is_positive(key):
    return next((value > 0 for value in key if value), False)


def one_of_each_pair(translations, hopping):
    """PythTB's construction: positive half and upper R = 0 triangle with conjugates, real diagonal."""
    table = {}
    for key, block in zip(map(tuple, translations), hopping):
        if not any(key):
            upper = np.triu(block, 1)
            table[key] = upper + upper.conj().T + np.diag(block.diagonal().real)
        elif is_positive(key):
            table[key] = block
            table[tuple(-value for value in key)] = block.conj().T
    keys = sorted(table)
    return np.array(keys), np.array([table[key] for key in keys])


def eigenvalues(translations, hopping, points):
    phases = np.exp(2j * np.pi * points @ np.asarray(translations).T)
    matrices = np.einsum("kr,rij->kij", phases, hopping)
    return np.linalg.eigvalsh((matrices + matrices.conj().transpose(0, 2, 1)) / 2)


def generator_hermiticity_error(translations, hopping):
    index = {tuple(row): i for i, row in enumerate(translations)}
    worst = 0.0
    for key, i in index.items():
        j = index.get(tuple(-value for value in key))
        target = hopping[i] if j is None else hopping[j] - hopping[i].conj().T
        worst = max(worst, float(np.abs(target).max()))
    return worst


def test_averaging_hermitizes_a_non_hermitian_table():
    translations, hopping = hermitian_table(1, planar_translations())
    rng = np.random.default_rng(2)
    noisy = hopping.copy()
    for index in (0, 3, 4):  # a negative-half block, another, and R = 0
        noisy[index] += 1e-6 * (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    noisy[4] -= 1j * np.diag(noisy[4].diagonal().imag)
    raw = WannierModel.from_table(translations, noisy, lattice=LATTICE, centres=CENTRES)
    with pytest.raises(NumericalError, match="Non-Hermitian"):
        raw.energies([[0.1, 0.2, 0.0]])
    model = WannierModel.from_table(
        translations, noisy, lattice=LATTICE, centres=CENTRES, hermitize="average"
    )
    lookup = {tuple(row): i for i, row in enumerate(model.translations)}
    for key, i in lookup.items():
        partner = model.hopping_matrices[lookup[tuple(-value for value in key)]]
        np.testing.assert_array_equal(partner, model.hopping_matrices[i].conj().T)
        original = noisy[i] + noisy[lookup[tuple(-value for value in key)]].conj().T
        np.testing.assert_allclose(model.hopping_matrices[i], original / 2, rtol=0, atol=1e-15)
    model.energies([[0.1, 0.2, 0.0], [0.4, -0.3, 0.0]])
    assert model.metadata["hermitization"] == "average"
    assert model.metadata["hermitization_added_partners"] == []
    expected_error = generator_hermiticity_error(translations, noisy)
    assert 1e-7 < expected_error < 1e-5
    assert model.metadata["hermiticity_error"] == pytest.approx(expected_error, rel=1e-12)
    assert raw.metadata["hermitization"] == "none"
    exact = WannierModel.from_table(translations, hopping, lattice=LATTICE, hermitize="average")
    np.testing.assert_array_equal(exact.hopping_matrices, hopping)
    assert exact.metadata["hermiticity_error"] == 0.0 == exact.metadata["hermiticity_weyl_bound"]


def test_weyl_bound_covers_averaged_against_one_of_each_pair_eigenvalues():
    translations, hopping = hermitian_table(5, planar_translations(2), nbnd=3)
    rng = np.random.default_rng(9)
    noisy = hopping + 2e-3 * (rng.normal(size=hopping.shape) + 1j * rng.normal(size=hopping.shape))
    home = int(np.flatnonzero(~translations.any(axis=1))[0])
    noisy[home] -= 1j * np.diag(noisy[home].diagonal().imag)
    # One block keeps only its positive member and one only its negative member.
    lone = [i for i, key in enumerate(map(tuple, translations)) if key in {(2, 1, 0), (-1, -2, 0)}]
    keep = np.setdiff1d(
        np.arange(len(translations)), [np.flatnonzero((translations == [-2, -1, 0]).all(1))[0]]
    )
    keep = np.setdiff1d(keep, [np.flatnonzero((translations == [1, 2, 0]).all(1))[0]])
    translations, noisy = translations[keep], noisy[keep]
    assert len(lone) == 2
    model = WannierModel.from_table(translations, noisy, lattice=LATTICE, hermitize="average", centres=None)
    assert sorted(model.metadata["hermitization_added_partners"]) == [[-2, -1, 0], [1, 2, 0]]
    pair_keys, pair_table = one_of_each_pair(translations, noisy)
    points = np.column_stack((rng.random((200, 2)) - 0.5, np.zeros(200)))
    shift = np.abs(model.energies(points).T - eigenvalues(pair_keys, pair_table, points)).max()
    bound = model.metadata["hermiticity_weyl_bound"]
    assert 0 < shift <= bound
    lookup = {tuple(row): i for i, row in enumerate(translations)}
    independent = 0.0
    for key, i in lookup.items():
        j = lookup.get(tuple(-value for value in key))
        if j is None:
            independent += np.linalg.norm(noisy[i]) * (1 if is_positive(key) else 2)
        elif not any(key) or is_positive(key):
            independent += np.linalg.norm(noisy[j].conj().T - noisy[i])
    assert bound == pytest.approx(independent, rel=1e-12)


def test_imaginary_onsite_energies_above_tolerance_are_refused():
    translations, hopping = hermitian_table(3, planar_translations())
    home = int(np.flatnonzero(~translations.any(axis=1))[0])
    tilted = hopping.copy()
    tilted[home, 1, 1] += 2e-6j
    with pytest.raises(ValueError, match="imaginary parts up to 2e-06"):
        WannierModel.from_table(translations, tilted, lattice=LATTICE, hermitize="average")
    tolerated = WannierModel.from_table(
        translations, tilted, lattice=LATTICE, hermitize="average", imag_onsite_tol=1e-5
    )
    assert tolerated.metadata["onsite_imaginary_max"] == pytest.approx(2e-6)
    assert tolerated.hopping_matrices[home, 1, 1].imag == 0
    WannierModel.from_table(translations, tilted, lattice=LATTICE, hermitize="average", imag_onsite_tol=None)
    with pytest.raises(ValueError, match="imag_onsite_tol"):
        WannierModel.from_table(translations, hopping, lattice=LATTICE, imag_onsite_tol=-1)


def test_hopping_cutoffs_follow_pythtb_rules_and_report_counts():
    lattice = np.diag([2.0, 5.0, 5.0])
    centres = np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]])
    translations = np.array([[r, 0, 0] for r in range(-3, 4)])
    onsite = np.array([[0.01, 0.2 + 0.02j], [0.2 - 0.02j, -0.03]])
    first = np.array([[0.3, 0.04j], [0.5, 0.02]])
    second = np.array([[0.06, 0.001], [0.2, 0.07]])
    third = np.full((2, 2), 0.1)
    blocks = [first, second, third]
    hopping = np.array([block.conj().T for block in blocks[::-1]] + [onsite] + blocks)
    norm, distance = 0.05, 4.5
    model = WannierModel.from_table(
        translations, hopping, lattice=lattice, centres=centres, min_hopping_norm=norm, max_distance=distance
    )
    expected = {key: np.zeros((2, 2), complex) for key in map(tuple, translations)}
    counts = {"hops": 0, "dropped_by_distance": 0, "dropped_by_norm": 0}
    for key, block in zip(map(tuple, translations), hopping):
        if not any(key):
            expected[key] += np.diag(block.diagonal())
            pairs = [(0, 1)]
        elif is_positive(key):
            pairs = list(product(range(2), repeat=2))
        else:
            continue
        for i, j in pairs:
            counts["hops"] += 1
            separation = np.linalg.norm(np.asarray(key) @ lattice + centres[j] - centres[i])
            if separation > distance:
                counts["dropped_by_distance"] += 1
            elif abs(block[i, j]) < norm:
                counts["dropped_by_norm"] += 1
            else:
                expected[key][i, j] += block[i, j]
                expected[tuple(-value for value in key)][j, i] += np.conj(block[i, j])
    counts["kept_hops"] = counts["hops"] - counts["dropped_by_distance"] - counts["dropped_by_norm"]
    # A far, weak hop counts as dropped by distance; small on-site energies remain.
    assert counts == {"hops": 13, "dropped_by_distance": 5, "dropped_by_norm": 2, "kept_hops": 6}
    assert model.metadata["cutoffs"] == {"min_hopping_norm": norm, "max_distance": distance, **counts}
    keys = sorted(expected)
    for point in [[0.0, 0, 0], [0.17, 0.3, -0.2], [0.5, 0, 0]]:
        reference = np.einsum(
            "r,rij->ij", np.exp(2j * np.pi * (np.array(keys) @ point)), np.array([expected[k] for k in keys])
        )
        np.testing.assert_allclose(model.hamiltonian(point), reference, rtol=0, atol=1e-14)
    # Translations whose every hop is cut are removed; the on-site block remains.
    assert sorted(map(tuple, model.translations)) == [(r, 0, 0) for r in range(-2, 3)]
    uncut = WannierModel.from_table(translations, hopping, lattice=lattice, centres=centres)
    assert uncut.metadata["cutoffs"]["kept_hops"] == 13 and uncut.metadata["cutoffs"]["max_distance"] is None
    with pytest.raises(ValueError, match="centres"):
        WannierModel.from_table(translations, hopping, lattice=lattice, max_distance=1.0)
    with pytest.raises(ValueError, match="position"):
        WannierModel.from_table(
            translations,
            hopping,
            lattice=lattice,
            centres=centres,
            position=np.zeros(hopping.shape + (3,)),
            min_hopping_norm=0.1,
        )


def test_table_models_use_exact_diagonal_links_with_centres_as_written():
    translations, hopping = hermitian_table(4, planar_translations())
    unwrapped = CENTRES + [[0.0, 0.0, 0.0], [2 * 2.5, 0.0, 0.0]]
    model = WannierModel.from_table(
        translations, hopping, lattice=LATTICE, centres=unwrapped, fermi_energy=0.3
    )
    assert model.metadata["position_link"] == "exact_diagonal"
    assert model.metadata["embedding"] == "centres"
    first, second = np.array([0.1, -0.2, 0.0]), np.array([0.35, 0.05, 0.0])
    delta = (second - first) @ model.g_vec
    np.testing.assert_allclose(
        model.basis_overlap(first, second), np.diag(np.exp(-1j * unwrapped @ delta)), rtol=0, atol=1e-15
    )
    plain = WannierModel(translations, hopping, lattice=LATTICE, centres=unwrapped, fermi_energy=0.3)
    np.testing.assert_allclose(model.energies([first, second]), plain.energies([first, second]), atol=1e-15)
    position = np.zeros(hopping.shape + (3,), complex)
    position[np.flatnonzero(~translations.any(axis=1))[0], [0, 1], [0, 1]] = unwrapped
    full = WannierModel.from_table(
        translations, hopping, lattice=LATTICE, position=position, hermitize="average"
    )
    assert full.metadata["position_link"] == "midpoint_first_order"
    with pytest.raises(ValueError, match="hermitize"):
        WannierModel.from_table(translations, hopping, lattice=LATTICE, hermitize="lower")
    with pytest.raises(ValueError, match="distinct"):
        WannierModel.from_table(
            np.vstack((translations, translations[:1])), np.vstack((hopping, hopping[:1])), lattice=LATTICE
        )


def write_hr(path, translations, hopping, degeneracies):
    n = hopping.shape[1]
    lines = ["synthetic", str(n), str(len(translations)), " ".join(map(str, degeneracies))]
    for translation, block in zip(translations, hopping):
        for j, i in product(range(n), repeat=2):
            value = block[i, j]
            lines.append(
                f"{translation[0]} {translation[1]} {translation[2]} {i + 1} {j + 1} {value.real:.17e} {value.imag:.17e}"
            )
    path.write_text("\n".join(lines) + "\n")


def write_tb(path, lattice, translations, hopping, degeneracies, position):
    n = hopping.shape[1]
    lines = ["synthetic"]
    lines.extend(" ".join(f"{value:.17e}" for value in row) for row in lattice)
    lines.extend([str(n), str(len(translations)), " ".join(map(str, degeneracies))])
    for translation, block in zip(translations, hopping):
        lines.append(" ".join(map(str, translation)))
        lines.extend(
            f"{i + 1} {j + 1} {block[i, j].real:.17e} {block[i, j].imag:.17e}"
            for j, i in product(range(n), repeat=2)
        )
    for translation, block in zip(translations, position):
        lines.append(" ".join(map(str, translation)))
        for j, i in product(range(n), repeat=2):
            values = " ".join(f"{part.real:.17e} {part.imag:.17e}" for part in block[i, j])
            lines.append(f"{i + 1} {j + 1} {values}")
    path.write_text("\n".join(lines) + "\n")


def test_directory_position_modes(tmp_path):
    translations, hopping = hermitian_table(8, planar_translations())
    degeneracies = np.ones(len(translations), int)
    position = np.zeros(hopping.shape + (3,), complex)
    home = int(np.flatnonzero(~translations.any(axis=1))[0])
    position[home, [0, 1], [0, 1]] = CENTRES
    position[:, 0, 1, 0] = 0.05
    position[:, 1, 0, 0] = 0.05
    write_hr(tmp_path / "wannier90_hr.dat", translations, hopping, degeneracies)
    write_tb(tmp_path / "wannier90_tb.dat", LATTICE, translations, hopping, degeneracies, position)
    (tmp_path / "wannier90_centres.xyz").write_text(
        "2\ncentres\n" + "\n".join(f"X {x} {y} {z}" for x, y, z in CENTRES) + "\n"
    )
    auto = WannierModel.from_directory(tmp_path)
    full = WannierModel.from_directory(tmp_path, position="full")
    diagonal = WannierModel.from_directory(tmp_path, position="diagonal", wout=None)
    assert auto.metadata["embedding"] == full.metadata["embedding"] == "full_position"
    assert "tb" in auto.source_files and "tb" in full.source_files
    assert diagonal.metadata["position_link"] == "exact_diagonal" and diagonal.position_matrices is None
    assert "tb" not in diagonal.source_files and diagonal.lattice is None
    np.testing.assert_allclose(diagonal.centres, CENTRES)
    point = [0.21, -0.13, 0.0]
    np.testing.assert_allclose(diagonal.hamiltonian(point), auto.hamiltonian(point), atol=1e-14)
    with pytest.raises(ValueError, match="omit tb"):
        WannierModel.from_directory(tmp_path, position="diagonal", tb="wannier90_tb.dat")
    tb_only = WannierModel.from_directory(tmp_path, hr=None, tb="wannier90_tb.dat", position="diagonal")
    assert tb_only.position_matrices is None and "tb" in tb_only.source_files
    np.testing.assert_allclose(tb_only.lattice, LATTICE)
    (tmp_path / "wannier90_tb.dat").unlink()
    with pytest.raises(ValueError, match="position='full'"):
        WannierModel.from_directory(tmp_path, position="full")
    with pytest.raises(ValueError, match="position must be"):
        WannierModel.from_directory(tmp_path, position="exact")
