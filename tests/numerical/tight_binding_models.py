"""Small analytic tight-binding models shared by the topology, Berry and slab tests."""

import numpy as np

from queer.models.wannier import WannierModel

PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
SIGMA = np.concatenate([np.eye(2)[None], PAULI])


def qwz(mass, lattice=None, centres=None):
    """Qi-Wu-Zhang model: C = 1 for -2 < m < 0, -1 for 0 < m < 2, otherwise 0."""
    x, y, z = PAULI
    translations = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
    hopping = [mass * z, (z - 1j * x) / 2, (z + 1j * x) / 2, (z - 1j * y) / 2, (z + 1j * y) / 2]
    lattice = np.diag([1.0, 1.0, 10.0]) if lattice is None else lattice
    return WannierModel(translations, hopping, lattice=lattice, centres=centres)


def rice_mele(v, w, staggered=0.0, stacking=0.3):
    """Chains along a1 with A at x = 0 and B at x = a1/2, stacked along a2."""
    translations = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    hopping = np.zeros((5, 2, 2), complex)
    hopping[0] = [[staggered, v], [v, -staggered]]
    hopping[1][1, 0] = hopping[2][0, 1] = w
    hopping[3] = hopping[4] = stacking * np.eye(2)
    lattice = np.diag([2.0, 3.0, 12.0])
    centres = np.array([[0.0, 0.0, 6.0], [1.0, 0.0, 6.0]])
    return WannierModel(translations, hopping, lattice=lattice, centres=centres)


def kane_mele(onsite, interlayer=None, height=10.0):
    """Kane-Mele model with Rashba coupling (PythTB's example): Z2 = 1 at onsite 1.0, 0 at 2.5.

    ``interlayer`` stacks the layers along a3 with a spin-independent hop.
    """
    lattice = np.array([[1.0, 0, 0], [0.5, np.sqrt(3) / 2, 0], [0, 0, height]])
    table = {(0, 0, 0): np.kron(np.diag([onsite, -onsite]), np.eye(2)).astype(complex)}

    def hop(amplitude, i, j, translation):
        block = np.einsum("s,sij->ij", np.asarray(amplitude, complex), SIGMA)
        reverse = tuple(-value for value in translation)
        for key, a, b, value in ((tuple(translation), i, j, block), (reverse, j, i, block.conj().T)):
            table.setdefault(key, np.zeros((4, 4), complex))[2 * a : 2 * a + 2, 2 * b : 2 * b + 2] += value

    root = np.sqrt(3) / 2
    for translation in ([0, 0, 0], [-1, 0, 0], [0, -1, 0]):
        hop([1.0, 0, 0, 0], 0, 1, translation)
    hop(1j * 0.25 * np.array([0, 0.5, -root, 0]), 0, 1, [0, 0, 0])
    hop(1j * 0.25 * np.array([0, 0.5, root, 0]), 0, 1, [-1, 0, 0])
    hop(1j * 0.25 * np.array([0, -1.0, 0, 0]), 0, 1, [0, -1, 0])
    for translation in ([1, 0, 0], [-1, 1, 0], [0, -1, 0]):
        hop([0, 0, 0, 0.3j], 0, 0, translation)
        hop([0, 0, 0, 0.3j], 1, 1, [-value for value in translation])
    if interlayer is not None:
        for orbital in (0, 1):
            hop([interlayer, 0, 0, 0], orbital, orbital, [0, 0, 1])
    keys = sorted(table)
    reduced = np.repeat([[1 / 3, 1 / 3, 0], [2 / 3, 2 / 3, 0]], 2, axis=0)
    return WannierModel(keys, [table[key] for key in keys], lattice=lattice, centres=reduced @ lattice)


def haldane(staggered=0.2, t=-1.0, t2=0.15, phase=np.pi / 2, height=10.0):
    """Haldane model with PythTB's example hops; centres on the honeycomb A and B sites."""
    lattice = np.array([[1.0, 0, 0], [0.5, np.sqrt(3) / 2, 0], [0, 0, height]])
    second = t2 * np.exp(1j * phase)
    table = {(0, 0, 0): np.diag([-staggered, staggered]).astype(complex)}

    def hop(amplitude, i, j, translation):
        reverse = tuple(-value for value in translation)
        for key, a, b, value in ((tuple(translation), i, j, amplitude), (reverse, j, i, np.conj(amplitude))):
            table.setdefault(key, np.zeros((2, 2), complex))[a, b] += value

    hop(t, 0, 1, [0, 0, 0])
    hop(t, 1, 0, [1, 0, 0])
    hop(t, 1, 0, [0, 1, 0])
    for translation, conjugate in (([1, 0, 0], False), ([1, -1, 0], True), ([0, 1, 0], True)):
        hop(np.conj(second) if conjugate else second, 0, 0, translation)
    for translation, conjugate in (([1, -1, 0], False), ([0, 1, 0], False), ([1, 0, 0], True)):
        hop(np.conj(second) if conjugate else second, 1, 1, translation)
    keys = sorted(table)
    reduced = np.array([[1 / 3, 1 / 3, 0], [2 / 3, 2 / 3, 0]])
    return WannierModel(keys, [table[key] for key in keys], lattice=lattice, centres=reduced @ lattice)


def cell_periodic_operators(model, k):
    """PythTB's convention-I H(k) and dH/dk_red at fractional k.

    H_ij(k) = sum_R H_ij(R) exp(2 pi i k.(R + tau_j - tau_i)); the derivative along
    each reduced axis is what pythtb's ``velocity(cartesian=False)`` returns.
    """
    tau = model.centres @ np.linalg.inv(model.lattice)
    span = model.translations[:, None, None, :] + tau[None, None, :, :] - tau[None, :, None, :]
    terms = np.exp(2j * np.pi * (span @ np.asarray(k, float))) * model.hopping_matrices
    return terms.sum(axis=0), np.einsum("rija,rij->aij", 2j * np.pi * span, terms)
