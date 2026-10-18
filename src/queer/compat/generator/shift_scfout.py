"""The shift-current recomputation's own parse of an OpenMX v3 scfout, and the two sets of real-space operators it
builds from it: QUEER's physical definition and what PyATB 60fc6b2a reads.

The comparison (:mod:`~queer.compat.generator.parity_shift`) judges QUEER's reported spectrum, which QUEER's
runner computes from QUEER's own scfout reader (``queer.io.openmx`` and the reconciliation in
``queer.models.openmx``). A recomputation that read the scfout through that reader would share its defects: a
wrong Hartree constant, a wrong reconciliation of the translation identities or a spin-down channel parsed from
the wrong records would move QUEER's value and the recomputation alike. This module reads the binary records itself
(the layout of OpenMX's ``read_scfout.c``, version 3, consulted through QUEER's reader for the layout only, and
PyATB's ``openmx_read_scfout.py`` for PyATB's semantics; read only, written here anew) and converts them with
constants of its own:

- **records** (:func:`read_records`): the header (atoms, ``4 version + SpinP_switch``, cells), the position
  expansion order, the cell translations, each atom's orbital and neighbour counts, its neighbours and their cells,
  the lattice and the atomic positions (Bohr), then the matrix sets in their order: ``Hks`` (SpinP_switch + 1
  components, Hartree), ``iHks`` (3, with spin-orbit coupling), ``OLP``, ``OLPpo`` (3 axes x order, the
  bra-centred moments, Bohr), ``OLPmo`` (3), ``DM`` (SpinP_switch + 1) and ``iDM`` (2), each a row-major
  ``orbitals(a) x orbitals(b)`` block per atom ``a`` and neighbour ``b``; the solver, ten scalars and the embedded
  input lines (256 bytes each) end the file, whose length must be exactly that. Each set is held dense per
  translation over the translations recorded and their negatives.
- **QUEER's physical operators** (:func:`physical_channels`): CODATA's Hartree and Bohr (``scipy.constants``, the
  values QUEER's units are defined by); the spin-up and, for a collinear scfout, the spin-down channel from their
  own ``Hks`` components; with spin-orbit coupling the spinor ``[[uu, ud], [ud(-R)^dagger, dd]]`` with ``uu = Hks0
  + i iHks0``, ``dd = Hks1 + i iHks1``, ``ud = Hks2 + i (Hks3 + iHks2)``; the overlap (spin-diagonal for spinors);
  the absolute position operator ``r_a(R) = OLPpo_a(R) + tau_a S(R)`` (``tau`` the bra orbital's atom); then the
  definition's translation identities, each record replaced by its projection onto them: ``H(R) = (H(R) +
  H(-R)^dagger) / 2``, the same for ``S`` and ``r_a(R) = (r_a(R) + r_a(-R)^dagger + R_a S(R)) / 2`` (``R_a``
  Cartesian), which holds ``r_a(R) - r_a(-R)^dagger = R_a S(R)`` exactly.
- **PyATB's operators** (:func:`literal_channel`): its Hartree 27.211324570273 eV and Bohr 0.52917721 Angstrom;
  ``nspin 1`` reads the spin-up ``Hks`` only and replaces ``H`` and ``S`` by ``(X(R) + X(-R)^dagger) / 2`` over
  the translations and their negatives; ``nspin 4`` reads the spinor as above (its ``H21`` block from ``H12(-R)^
  dagger``) and the overlap as recorded; its position operator is the real part of ``OLPpo_a(R) + tau_a S(R)`` with
  the recorded overlap, never symmetrized. PyATB then sums the upper triangles of these
  (:func:`~queer.compat.generator.pyatb_shift_integrand.bloch_operators`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# PyATB 60fc6b2a's reader: AU2EV and BOHR_TO_ANG.
PYATB_HARTREE = 27.211324570273
PYATB_BOHR = 0.52917721
SCFOUT_VERSION = 3
SPIN_MODES = {0: "nonmagnetic", 1: "collinear", 3: "soc"}
INPUT_LINE_BYTES = 256


def codata_hartree() -> float:
    """CODATA's Hartree energy in eV (``scipy.constants``): QUEER's eV per Hartree."""
    from scipy.constants import physical_constants

    return float(physical_constants["Hartree energy in eV"][0])


def codata_bohr() -> float:
    """CODATA's Bohr radius in Angstrom (``scipy.constants``): QUEER's Angstrom per Bohr."""
    from scipy.constants import physical_constants

    return float(physical_constants["Bohr radius"][0]) * 1e10


@dataclass(frozen=True)
class Blocks:
    """One channel's real-space operators as dense stacks over one list of translations (eV and Angstrom;
    ``position`` ``(3, NR, n, n)``, None without position records)."""

    translations: np.ndarray
    cartesian: np.ndarray
    hamiltonian: np.ndarray
    overlap: np.ndarray
    position: np.ndarray | None

    @property
    def size(self) -> int:
        return int(self.hamiltonian.shape[-1])


@dataclass(frozen=True)
class Records:
    """An scfout's records as written: ``hamiltonian`` ``(SpinP_switch + 1, NR, n, n)`` and ``imaginary`` ``(3, NR,
    n, n)`` (Hartree; None without spin-orbit coupling), ``overlap`` ``(NR, n, n)``, ``position`` ``(3, NR, n, n)``
    (the first bra-centred moments, Bohr; None when the scfout has none), over ``translations`` ``(NR, 3)`` (those
    recorded and their negatives, sorted); the lattice rows and each orbital's atom ``centres`` ``(n, 3)`` in Bohr."""

    source: Path
    spin_flag: int
    lattice: np.ndarray
    translations: np.ndarray
    centres: np.ndarray
    hamiltonian: np.ndarray
    imaginary: np.ndarray | None
    overlap: np.ndarray
    position: np.ndarray | None

    @property
    def spin_mode(self) -> str:
        return SPIN_MODES[self.spin_flag]

    @property
    def nspin(self) -> int:
        """PyATB's ``nspin`` for this layout: 4 with spin-orbit coupling, else 1."""
        return 4 if self.spin_flag == 3 else 1

    @property
    def orbitals(self) -> int:
        return int(self.overlap.shape[-1])

    @property
    def bands(self) -> int:
        """The levels of one channel: the spinor bands with spin-orbit coupling, else the orbitals."""
        return 2 * self.orbitals if self.spin_flag == 3 else self.orbitals

    @property
    def channels(self) -> int:
        return 2 if self.spin_flag == 1 else 1

    @property
    def handedness(self) -> int:
        """The sign of the lattice's determinant."""
        return 1 if float(np.linalg.det(self.lattice)) > 0 else -1


class _Cursor:
    def __init__(self, data: bytes, endian: str, source: Path):
        self.data, self.endian, self.source, self.offset = data, endian, source, 0

    def take(self, dtype: str, count: int, what: str) -> np.ndarray:
        size = np.dtype(dtype).itemsize
        if count < 0 or self.offset + size * count > len(self.data):
            raise ValueError(f"{self.source}: the scfout ends inside {what} (byte {self.offset})")
        values = np.frombuffer(self.data, self.endian + dtype, count, self.offset).astype(dtype)
        self.offset += size * count
        if dtype == "f8" and not np.isfinite(values).all():
            raise ValueError(f"{self.source}: {what} holds a nonfinite value")
        return values

    def skip(self, size: int, what: str) -> None:
        if self.offset + size > len(self.data):
            raise ValueError(f"{self.source}: the scfout ends inside {what} (byte {self.offset})")
        self.offset += size


def _endianness(data: bytes, source: Path) -> str:
    if len(data) < 24:
        raise ValueError(f"{source}: no scfout header")
    found = []
    for endian in ("<", ">"):
        head = np.frombuffer(data, endian + "i4", 6)
        version, spin = divmod(int(head[1]), 4)
        if head[0] > 0 and head[1] >= 0 and version == SCFOUT_VERSION and spin in SPIN_MODES and head[5] >= 0:
            found.append(endian)
    if len(found) != 1:
        raise ValueError(f"{source}: not a version-{SCFOUT_VERSION} scfout of a known spin layout")
    return found[0]


def read_records(path: str | Path) -> Records:
    """Parse ``path`` (the module docstring's layout)."""
    source = Path(path).expanduser().resolve()
    data = source.read_bytes()
    cursor = _Cursor(data, _endianness(data, source), source)
    head = cursor.take("i4", 6, "the header")
    atoms, spin, cells = int(head[0]), int(head[1]) % 4, int(head[5]) + 1
    order = int(cursor.take("i4", 1, "the position expansion order")[0])
    if order < 0:
        raise ValueError(f"{source}: negative position expansion order")
    cursor.skip(8 * 4 * cells, "the Cartesian cell translations")
    integer_cells = cursor.take("i4", 4 * cells, "the cell translations").reshape(cells, 4)[:, 1:]
    orbitals = cursor.take("i4", atoms, "the orbital counts")
    neighbours = cursor.take("i4", atoms, "the neighbour counts")
    if np.any(orbitals < 1) or np.any(neighbours < 0):
        raise ValueError(f"{source}: invalid orbital or neighbour counts")
    partner = [
        cursor.take("i4", int(count) + 1, f"atom {a}'s neighbours") for a, count in enumerate(neighbours)
    ]
    cell = [
        cursor.take("i4", int(count) + 1, f"atom {a}'s neighbour cells") for a, count in enumerate(neighbours)
    ]
    lattice = cursor.take("f8", 12, "the lattice").reshape(3, 4)[:, 1:]
    cursor.skip(8 * 12, "the reciprocal lattice")
    positions = cursor.take("f8", 4 * atoms, "the atomic positions").reshape(atoms, 4)[:, 1:]
    if abs(float(np.linalg.det(lattice))) < 1e-12:
        raise ValueError(f"{source}: singular lattice")
    offsets = np.concatenate(([0], np.cumsum(orbitals))).astype(int)
    size = int(offsets[-1])
    # Each (atom, neighbour) block: its translation, rows, columns and place in a matrix set.
    pairs = []
    start = 0
    for a in range(atoms):
        for b, c in zip(partner[a], cell[a], strict=True):
            b, c = int(b) - 1, int(c)
            if not (0 <= b < atoms and 0 <= c < cells):
                raise ValueError(f"{source}: atom {a} lists an invalid neighbour")
            rows, cols = int(orbitals[a]), int(orbitals[b])
            pairs.append((tuple(int(x) for x in integer_cells[c]), offsets[a], offsets[b], rows, cols, start))
            start += rows * cols
    elements = start
    keys = {pair[0] for pair in pairs}
    keys |= {(-x, -y, -z) for x, y, z in keys}
    translations = np.array(sorted(keys), dtype=int).reshape(-1, 3)
    index = {tuple(int(v) for v in key): i for i, key in enumerate(translations)}

    def matrix_set(what: str) -> np.ndarray:
        flat = cursor.take("f8", elements, what)
        dense = np.zeros((len(translations), size, size))
        for key, row, col, rows, cols, at in pairs:
            dense[index[key], row : row + rows, col : col + cols] += flat[at : at + rows * cols].reshape(
                rows, cols
            )
        return dense

    def skip_sets(count: int, what: str) -> None:
        cursor.skip(8 * elements * count, what)

    hamiltonian = np.stack([matrix_set(f"Hks[{s}]") for s in range(spin + 1)])
    imaginary = np.stack([matrix_set(f"iHks[{s}]") for s in range(3)]) if spin == 3 else None
    overlap = matrix_set("OLP")
    position = None
    if order:
        moments = []
        for axis in range(3):
            moments.append(matrix_set(f"OLPpo[{axis}][0]"))
            skip_sets(order - 1, f"OLPpo[{axis}]")
        position = np.stack(moments)
    skip_sets(3, "OLPmo")
    skip_sets(spin + 1, "DM")
    skip_sets(2, "iDM")
    cursor.skip(4, "the solver")
    cursor.take("f8", 10, "the SCF scalars")
    remaining = len(data) - cursor.offset
    if remaining:
        lines = int(cursor.take("i4", 1, "the input line count")[0])
        if lines < 0 or remaining != 4 + INPUT_LINE_BYTES * lines:
            raise ValueError(
                f"{source}: {remaining} bytes after the records are not the embedded input lines"
            )
    return Records(
        source=source,
        spin_flag=spin,
        lattice=lattice,
        translations=translations,
        centres=np.repeat(positions, orbitals, axis=0),
        hamiltonian=hamiltonian,
        imaginary=imaginary,
        overlap=overlap,
        position=position,
    )


# --------------------------------------------------------------------------- assembly
def _partners(translations: np.ndarray) -> np.ndarray:
    """Each translation's negative's index (the translations hold their negatives)."""
    index = {tuple(int(v) for v in key): i for i, key in enumerate(translations)}
    return np.array([index[tuple(-int(v) for v in key)] for key in translations], dtype=int)


def _adjoint(stack: np.ndarray) -> np.ndarray:
    return np.swapaxes(stack.conj(), -1, -2)


def _mirrored(stack: np.ndarray, partners: np.ndarray) -> np.ndarray:
    """``X(-R)^dagger`` at every ``R``."""
    return _adjoint(stack[..., partners, :, :])


def _spinor(records: Records) -> np.ndarray:
    """The spinor Hamiltonian ``[[uu, ud], [ud(-R)^dagger, dd]]`` (Hartree)."""
    real, imaginary = records.hamiltonian, records.imaginary
    assert imaginary is not None
    up = real[0] + 1j * imaginary[0]
    down = real[1] + 1j * imaginary[1]
    mixed = real[2] + 1j * (real[3] + imaginary[2])
    n = records.orbitals
    out = np.zeros((len(records.translations), 2 * n, 2 * n), complex)
    out[:, :n, :n], out[:, n:, n:], out[:, :n, n:] = up, down, mixed
    out[:, n:, :n] = _mirrored(mixed, _partners(records.translations))
    return out


def _spin_diagonal(stack: np.ndarray) -> np.ndarray:
    n = stack.shape[-1]
    out = np.zeros(stack.shape[:-2] + (2 * n, 2 * n), complex)
    out[..., :n, :n] = out[..., n:, n:] = stack
    return out


def _absolute_position(records: Records, bohr: float) -> np.ndarray | None:
    """``(OLPpo_a(R) + tau_a S(R)) bohr`` ``(3, NR, n, n)`` with the recorded overlap; None without moments."""
    if records.position is None:
        return None
    centred = records.centres.T[:, None, :, None] * records.overlap[None]
    return (records.position + centred) * bohr


def _hamiltonians(records: Records) -> list[np.ndarray]:
    """Each channel's recorded Hamiltonian (Hartree): the spinor, the one channel, or spin-up then spin-down."""
    if records.spin_flag == 3:
        return [_spinor(records)]
    return [records.hamiltonian[channel].astype(complex) for channel in range(records.channels)]


def hermitian_residual(stack: np.ndarray, translations: np.ndarray) -> float:
    """The largest ``|X(R) - X(-R)^dagger|`` (rows' absolute sums) over the translations."""
    difference = stack - _mirrored(stack, _partners(translations))
    return float(np.max(np.abs(difference).sum(axis=-1), initial=0.0))


def exactness_bound(records: Records) -> float:
    """``max(1e-9 eV, 10 x)``, ``x`` the largest ``|H(R) - H(-R)^dagger|`` (rows' absolute sums, CODATA eV) of the
    recorded Hamiltonians: the spacing below which two levels are one degenerate level of the data."""
    hartree = codata_hartree()
    largest = max(hermitian_residual(h * hartree, records.translations) for h in _hamiltonians(records))
    return max(1e-9, 10.0 * largest)


def physical_channels(records: Records) -> list[Blocks]:
    """QUEER's physical operators of each channel (the module docstring): CODATA units, each record projected onto
    the translation identities."""
    hartree, bohr = codata_hartree(), codata_bohr()
    translations = records.translations
    partners = _partners(translations)
    cartesian = translations @ (records.lattice * bohr)
    overlap = records.overlap.astype(complex)
    overlap = (overlap + _mirrored(overlap, partners)) / 2
    position = _absolute_position(records, bohr)
    if position is not None:
        moved = cartesian.T[:, :, None, None] * overlap[None]
        position = (position + _mirrored(position.astype(complex), partners) + moved) / 2
    if records.spin_flag == 3:
        overlap = _spin_diagonal(overlap)
        position = None if position is None else _spin_diagonal(position)
    out = []
    for hamiltonian in _hamiltonians(records):
        hamiltonian = hamiltonian * hartree
        hamiltonian = (hamiltonian + _mirrored(hamiltonian, partners)) / 2
        out.append(Blocks(translations, cartesian, hamiltonian, overlap, position))
    return out


def literal_channel(records: Records) -> Blocks:
    """The operators PyATB 60fc6b2a reads (the module docstring): its Hartree and Bohr, the spin-up channel of an
    ``nspin 1`` scfout with ``H`` and ``S`` Hermitized in real space, the spinor and recorded overlap of an ``nspin
    4`` one, the real part of the recorded position operator."""
    translations = records.translations
    partners = _partners(translations)
    cartesian = translations @ (records.lattice * PYATB_BOHR)
    position = _absolute_position(records, PYATB_BOHR)
    if position is not None:
        position = position.real.astype(complex)
    overlap = records.overlap.astype(complex)
    if records.spin_flag == 3:
        hamiltonian = _spinor(records)
        overlap = _spin_diagonal(overlap)
        position = None if position is None else _spin_diagonal(position)
    else:
        hamiltonian = records.hamiltonian[0].astype(complex)
        hamiltonian = (hamiltonian + _mirrored(hamiltonian, partners)) / 2
        overlap = (overlap + _mirrored(overlap, partners)) / 2
    return Blocks(translations, cartesian, hamiltonian * PYATB_HARTREE, overlap, position)


def residuals(records: Records) -> dict[str, float]:
    """The records' distances from the physical translation identities (eV for ``H``; facts, not gates)."""
    hartree = codata_hartree()
    out = {
        "hamiltonian_eV": max(
            hermitian_residual(h * hartree, records.translations) for h in _hamiltonians(records)
        ),
        "overlap": hermitian_residual(records.overlap.astype(complex), records.translations),
    }
    position = _absolute_position(records, codata_bohr())
    if position is not None:
        partners = _partners(records.translations)
        cartesian = records.translations @ (records.lattice * codata_bohr())
        overlap = records.overlap.astype(complex)
        moved = cartesian.T[:, :, None, None] * overlap[None]
        gap = position - _mirrored(position.astype(complex), partners) - moved
        out["position_Angstrom"] = float(np.max(np.abs(gap).sum(axis=-1), initial=0.0))
    return out


def volume(records: Records, bohr: float) -> float:
    """The cell volume ``|det A|`` in Angstrom^3 for ``bohr`` Angstrom per Bohr."""
    return abs(float(np.linalg.det(records.lattice))) * bohr**3


__all__ = [
    "PYATB_BOHR",
    "PYATB_HARTREE",
    "Blocks",
    "Records",
    "codata_bohr",
    "codata_hartree",
    "exactness_bound",
    "literal_channel",
    "physical_channels",
    "read_records",
    "residuals",
    "volume",
]
