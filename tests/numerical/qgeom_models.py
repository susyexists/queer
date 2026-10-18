"""Small two-dimensional Wannier tables for the quantum-geometry tests.

Every table is Hermitian, H(-R) = H(R)^dagger, on a cell whose a3 lies along the
layer normal; position tables hold the centres on the R = 0 diagonal and small
off-diagonal elements with an anti-Hermitian part, as a finite-difference
wannier90 position matrix has.
"""

import numpy as np

LAYER_TRANSLATIONS = np.array([[a, b, 0] for a in (-2, -1, 0, 1, 2) for b in (-2, -1, 0, 1, 2)])
OBLIQUE = np.array([[3.1, 0.0, 0.0], [1.2, 2.8, 0.0], [0.0, 0.0, 15.0]])
HEXAGONAL = np.array([[2.5, 0.0, 0.0], [-1.25, 2.5 * np.sqrt(3) / 2, 0.0], [0.0, 0.0, 20.0]])


def hermitian_table(rng, translations, onsite, scale=0.25, decay=1.0):
    """Random H(R) with H(-R) = H(R)^dagger, hops shrinking as exp(-decay |R|), and the given on-site diagonal."""
    n = len(onsite)
    index = {tuple(r): i for i, r in enumerate(translations)}
    table = np.zeros((len(translations), n, n), complex)
    for i, r in enumerate(translations):
        j = index[tuple(-np.asarray(r))]
        if j < i:
            continue
        size = scale * np.exp(-decay * np.linalg.norm(r))
        block = size * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
        if i == j:
            block = (block + block.conj().T) / 2 + np.diag(onsite)
        table[i], table[j] = block, block.conj().T
    return table


def position_table(rng, translations, centres, scale=0.05, antihermitian=0.01):
    """Cartesian r(R) (nR, N, N, 3): the centres on the R = 0 diagonal, random Hermitian-paired
    off-diagonal elements plus an anti-Hermitian part."""
    n = len(centres)
    index = {tuple(r): i for i, r in enumerate(translations)}
    r = np.zeros((len(translations), n, n, 3), complex)
    for i, t in enumerate(translations):
        j = index[tuple(-np.asarray(t))]
        if j < i:
            continue
        block = scale * (rng.normal(size=(n, n, 3)) + 1j * rng.normal(size=(n, n, 3)))
        if i == j:
            block = (block + np.conj(np.swapaxes(block, 0, 1))) / 2
            block[np.arange(n), np.arange(n)] = centres
        r[i], r[j] = block, np.conj(np.swapaxes(block, 0, 1))
    noise = antihermitian * (rng.normal(size=r.shape) + 1j * rng.normal(size=r.shape))
    return r + noise


def layer(rng, bands=3, lattice=OBLIQUE, spread=1.2, decay=1.0):
    """(R, H, lattice, centres) of a random layer with on-site energies spread around zero."""
    onsite = np.linspace(-spread, spread, bands) + rng.normal(scale=0.05, size=bands)
    centres = np.c_[rng.uniform(0, 1, size=(bands, 2)), rng.uniform(-0.1, 0.1, size=bands)] @ lattice
    H = hermitian_table(rng, LAYER_TRANSLATIONS, onsite, decay=decay)
    return LAYER_TRANSLATIONS.copy(), H, np.asarray(lattice, float), centres
