"""OpenMX v3 scfout files of analytic one-atom models, written from the binary record layout.

``write_scfout_model`` puts every orbital on one atom at the origin and makes
every listed cell translation a neighbour. Matrices are given per translation in
eV and Å; the writer stores Hartree and Bohr as OpenMX does. ``shift_spinor_scfout``
and ``shift_collinear_scfout`` write gapped models without inversion for the shift
current. ``qwz_spinor_scfout``
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


CUBIC_TRANSLATIONS = ((0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
PAULI = (
    np.eye(2),
    np.array([[0, 1], [1, 0]], complex),
    np.array([[0, -1j], [1j, 0]]),
    np.diag([1.0, -1.0]).astype(complex),
)


def shift_spinor_scfout(
    path,
    *,
    levels=(-1.5, 1.5),
    hopping=(-0.15, 0.15),
    mixing_even=(0.10, 0.16, 0.22),
    mixing_odd=(0.12, 0.09, 0.07),
    onsite_mixing=0.05,
    soc=0.15,
    soc_z=0.06,
    exchange=0.2,
    lattice=None,
    chemp_eV=0.0,
):
    """A gapped 3D two-orbital spinor model without inversion or time reversal (SpinP_switch 3).

    Two orbitals (``levels``) on one atom of an orthorhombic cell (3.0, 3.2, 3.5 Å)
    hop to the six nearest cells (``hopping``); their mixing ``v e^{ik} + w e^{-ik}``
    along each axis has an even part (``mixing_even``, cos k) and an odd part
    (``mixing_odd``, sin k), so no orbital parity is a symmetry; the spin terms
    ``soc (sin kx sx + sin ky sy) + soc_z sin kz sz`` (half on the second orbital)
    and an exchange field ``exchange sz`` split every level, so with the defaults no
    two levels come closer than 0.28 eV anywhere and the two lower bands lie 1.06 eV
    or more below the two upper ones.
    """
    n = 2
    lattice = np.diag([3.0, 3.2, 3.5]) if lattice is None else np.asarray(lattice, float)
    s0, sx, sy, sz = PAULI
    onsite = np.kron(s0, np.diag(levels)) + exchange * np.kron(sz, np.eye(n))
    onsite = onsite + onsite_mixing * np.kron(s0, np.array([[0.0, 1.0], [1.0, 0.0]]))
    spins = (soc * sx, soc * sy, soc_z * sz)
    blocks = np.zeros((len(CUBIC_TRANSLATIONS), 2 * n, 2 * n), complex)
    blocks[0] = onsite
    for axis in range(3):
        forward = (mixing_even[axis] + mixing_odd[axis]) / 2
        backward = (mixing_even[axis] - mixing_odd[axis]) / 2
        spatial = np.array([[hopping[0], forward], [backward, hopping[1]]], complex)
        block = np.kron(s0, spatial) + np.kron(spins[axis] / 2j, np.diag([1.0, 0.5]))
        blocks[1 + 2 * axis] = block
        blocks[2 + 2 * axis] = block.conj().T
    up, down, mixed = blocks[:, :n, :n], blocks[:, n:, n:], blocks[:, :n, n:]
    return write_scfout_model(
        path,
        lattice=lattice,
        translations=CUBIC_TRANSLATIONS,
        hamiltonian=np.stack((up.real, down.real, mixed.real, mixed.imag)),
        imaginary=np.stack((up.imag, down.imag, np.zeros_like(mixed.real))),
        spin=3,
        chemp_eV=chemp_eV,
        electrons=2.0,
    )


def shift_collinear_scfout(
    path,
    *,
    levels=(-1.2, 1.2),
    hopping=(-0.2, 0.18),
    forward=(0.25, 0.18),
    backward=(0.05, -0.08),
    split=0.25,
    spin=1,
    chemp_eV=0.0,
):
    """A gapped 2D two-orbital model without inversion, collinear (``spin`` 1) or non-spin-polarized (0).

    The real mixing ``forward e^{ik} + backward e^{-ik}`` has both a cos and a sin
    part along each axis of a rectangular cell (3.0, 3.3, 12 Å). With ``spin`` 1 the
    two channels' orbital levels move by ``+split`` and ``-split`` in opposite
    directions, so their shift currents differ.
    """
    lattice = np.diag([3.0, 3.3, 12.0])

    def channel(sign):
        blocks = np.zeros((len(SQUARE_TRANSLATIONS), 2, 2))
        blocks[0] = np.diag([levels[0] + sign * split, levels[1] - sign * split]) + 0.04 * np.array(
            [[0.0, 1.0], [1.0, 0.0]]
        )
        for axis in range(2):
            hop = np.array([[hopping[0], forward[axis]], [backward[axis], hopping[1]]])
            blocks[1 + 2 * axis] = hop
            blocks[2 + 2 * axis] = hop.T
        return blocks

    components = (channel(1), channel(-1)) if spin == 1 else (channel(0),)
    return write_scfout_model(
        path,
        lattice=lattice,
        translations=SQUARE_TRANSLATIONS,
        hamiltonian=np.stack(components),
        spin=spin,
        chemp_eV=chemp_eV,
        electrons=2.0,
    )
