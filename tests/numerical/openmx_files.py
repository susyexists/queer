"""OpenMX v3 scfout files of analytic one-atom models, written from the binary record layout.

``write_scfout_model`` puts every orbital on one atom at the origin and makes
every listed cell translation a neighbour. Matrices are given per translation in
eV and Å; the writer stores Hartree and Bohr as OpenMX does. ``qwz_spinor_scfout``
writes the Qi-Wu-Zhang Chern insulator as the spinor of one spatial orbital
(SpinP_switch 3): H(k) = lambda (sin kx sx + sin ky sy) + (m + t (cos kx + cos ky)) sz,
so its lower band has |C| = 1 for 0 < |m| < 2|t| and C = 0 beyond.
"""

import struct

import numpy as np

from queer.io.openmx import BOHR_ANGSTROM, HARTREE_EV

SQUARE_TRANSLATIONS = ((0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))


def write_scfout_model(
    path,
    *,
    lattice,
    translations,
    hamiltonian,
    spin=0,
    imaginary=None,
    overlap=None,
    chemp_eV=0.0,
    electrons=1.0,
):
    """Write a one-atom scfout and return its path.

    ``hamiltonian`` holds ``spin + 1`` real components (OpenMX's ``Hks``), each of
    shape (NR, n, n) in eV; for ``spin == 3`` ``imaginary`` holds the three
    ``iHks`` components. ``overlap`` (NR, n, n) defaults to the identity at R = 0.
    Position moments are zero (orbitals centred on the atom).
    """
    lattice = np.asarray(lattice, float)
    translations = np.asarray(translations, int)
    components = np.asarray(hamiltonian, float)
    count = len(translations)
    n = components.shape[-1]
    if overlap is None:
        overlap = np.zeros((count, n, n))
        overlap[0] = np.eye(n)
    imaginary = np.zeros((3, count, n, n)) if imaginary is None else np.asarray(imaginary, float)
    records = []

    def ints(values):
        records.append(struct.pack("<" + str(len(values)) + "i", *[int(v) for v in values]))

    def doubles(values):
        values = [float(v) for v in np.ravel(values)]
        records.append(struct.pack("<" + str(len(values)) + "d", *values))

    def matrices(stack):
        for block in np.asarray(stack, float):
            doubles(block.ravel())

    bohr = lattice / BOHR_ANGSTROM
    ints([1, 4 * 3 + spin, 0, 0, 0, count - 1])
    ints([1])
    doubles([x for cell in translations for x in (0, *(cell @ bohr))])
    ints([x for cell in translations for x in (0, *cell)])
    ints([n])
    ints([count - 1])
    ints([1] * count)
    ints(list(range(count)))
    doubles([x for row in bohr for x in (0, *row)])
    doubles([x for row in 2 * np.pi * np.linalg.inv(bohr).T for x in (0, *row)])
    doubles([0, 0, 0, 0])
    for component in components:
        matrices(component / HARTREE_EV)
    if spin == 3:
        for component in imaginary:
            matrices(component / HARTREE_EV)
    matrices(overlap)
    for _ in range(3):
        matrices(np.zeros((count, n, n)))
    for _ in range(3):
        matrices(np.zeros((count, n, n)))
    for _ in range(spin + 1):
        matrices(np.zeros((count, n, n)))
    for _ in range(2):
        matrices(np.zeros((count, n, n)))
    ints([3])
    doubles([chemp_eV / HARTREE_EV, 300, 0, 0, 0, 0, 0, 0, electrons, 0])
    lines = [
        "<Definition.of.Atomic.Species",
        f"X X5.0-s{n} X_PBE19",
        "Definition.of.Atomic.Species>",
        "<Atoms.SpeciesAndCoordinates",
        "1 X 0 0 0",
        "Atoms.SpeciesAndCoordinates>",
    ]
    ints([len(lines)])
    for line in lines:
        records.append(line.encode().ljust(256, b"\0"))
    path.write_bytes(b"".join(records))
    return path


def qwz_spinor_scfout(
    path, *, mass=1.0, hopping=1.0, coupling=1.0, chemp_eV=0.0, a=2.0, c=10.0, lattice=None
):
    """The QWZ model on a square lattice (a, c in Å) as one spinor orbital; returns the path.

    ``lattice`` (rows in Å) replaces ``diag(a, a, c)``: the hoppings stay along a1 and a2.
    """
    lattice = np.diag([a, a, c]) if lattice is None else np.asarray(lattice, float)
    uu = np.array([mass, hopping / 2, hopping / 2, hopping / 2, hopping / 2])
    up_down_real = np.array([0.0, 0.0, 0.0, -coupling / 2, coupling / 2])
    up_down_imaginary = np.array([0.0, -coupling / 2, coupling / 2, 0.0, 0.0])
    hamiltonian = np.stack((uu, -uu, up_down_real, up_down_imaginary))[:, :, None, None]
    return write_scfout_model(
        path,
        lattice=lattice,
        translations=SQUARE_TRANSLATIONS,
        hamiltonian=hamiltonian,
        spin=3,
        chemp_eV=chemp_eV,
    )


def collinear_scfout(path, *, chemp_eV=0.0):
    """A two-orbital square-lattice model, its spin channels split by 0.5 eV (real, so no curvature)."""
    lattice = np.diag([2.0, 2.0, 10.0])
    spatial = np.zeros((5, 2, 2))
    spatial[0] = [[-1.0, 0.3], [0.3, 1.0]]
    spatial[1:] = [[0.25, 0.1], [0.1, -0.25]]
    shift = np.zeros((5, 2, 2))
    shift[0] = 0.5 * np.eye(2)
    return write_scfout_model(
        path,
        lattice=lattice,
        translations=SQUARE_TRANSLATIONS,
        hamiltonian=np.stack((spatial, spatial + shift)),
        spin=1,
        chemp_eV=chemp_eV,
        electrons=2.0,
    )
