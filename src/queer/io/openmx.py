"""Native sparse reader for the documented OpenMX v3 SCFOUT interface.

The binary layout follows OpenMX's developer interface. Parsing and assembly
are independent implementations; no OpenMX or PyATB runtime is required.
All public operators use eV and Angstrom, except explicitly named raw records.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from ..physics.constants import BOHR_ANGSTROM as BOHR_ANGSTROM
from ..physics.constants import HARTREE_EV as HARTREE_EV

Blocks = dict[tuple[int, int, int], csr_matrix]
_HARMONICS = {
    0: ("s",),
    1: ("px", "py", "pz"),
    2: ("dz2", "dx2-y2", "dxy", "dxz", "dyz"),
    3: ("fz3", "fxz2", "fyz2", "fz(x2-y2)", "fxyz", "fx(x2-3y2)", "fy(3x2-y2)"),
}


@dataclass
class OpenMXData:
    """Sparse real-space operators and retained SCF metadata.

    Position moments are bra-centered raw powers, converted to Angstrom**order.
    Momentum and density components preserve SCF component ordering; momentum
    records retain their original atomic-unit derivative representation.
    """

    hamiltonian: Blocks
    overlap: Blocks
    position: tuple[Blocks, Blocks, Blocks]
    lattice: np.ndarray
    atom_positions: np.ndarray
    atom_orbitals: np.ndarray
    spin_mode: str
    fermi_energy: float
    electron_count: float
    temperature_kelvin: float
    position_moments: tuple[tuple[Blocks, ...], ...] = ()
    momentum_components: tuple[Blocks, ...] = ()
    density_components: tuple[Blocks, ...] = ()
    imaginary_density_components: tuple[Blocks, ...] = ()
    input_lines: tuple[str, ...] = ()
    orbital_metadata: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    source: Path | None = None


class _BinaryReader:
    def __init__(self, stream: BinaryIO, source: Path, endian: str, size: int):
        self.stream, self.source, self.endian, self.size = stream, source, endian, size

    def array(self, dtype: str, count: int, context: str) -> np.ndarray:
        if count < 0:
            raise ValueError(f"{self.source}: negative count for {context}")
        itemsize = np.dtype(dtype).itemsize
        offset = self.stream.tell()
        if count > (self.size - offset) // itemsize:
            raise ValueError(f"{self.source}: truncated {context} at byte {offset}")
        raw = self.stream.read(count * itemsize)
        result = np.frombuffer(raw, dtype=self.endian + dtype).astype(dtype, copy=True)
        if dtype == "f8" and not np.isfinite(result).all():
            raise ValueError(f"{self.source}: nonfinite {context} at byte {offset}")
        return result

    def ints(self, count: int, context: str):
        return self.array("i4", count, context)

    def doubles(self, count: int, context: str):
        return self.array("f8", count, context)


def _combine(*components: tuple[Blocks, complex]) -> Blocks:
    keys = set().union(*(blocks for blocks, _ in components))
    out = {}
    for key in keys:
        shape = next(blocks[key].shape for blocks, _ in components if key in blocks)
        value = csr_matrix(shape, dtype=complex)
        for blocks, coefficient in components:
            if key in blocks:
                value = value + coefficient * blocks[key]
        value.eliminate_zeros()
        out[key] = value
    return out


def _spin_diagonal(blocks: Blocks) -> Blocks:
    from scipy.sparse import block_diag

    return {key: block_diag((value, value), format="csr") for key, value in blocks.items()}


def _spin_hamiltonian(real: tuple[Blocks, ...], imaginary: tuple[Blocks, ...], mode: str) -> Blocks:
    from scipy.sparse import bmat

    if mode == "nonmagnetic":
        return _combine((real[0], HARTREE_EV))
    if mode == "collinear":
        return {
            key: bmat([[real[0][key], None], [None, real[1][key]]], format="csr") * HARTREE_EV
            for key in real[0]
        }
    uu = _combine((real[0], 1), (imaginary[0], 1j))
    dd = _combine((real[1], 1), (imaginary[1], 1j))
    ud = _combine((real[2], 1), (real[3], 1j), (imaginary[2], 1j))
    keys = set(uu) | set(dd) | set(ud)
    keys |= {(-key[0], -key[1], -key[2]) for key in keys}
    n = next(iter(uu.values())).shape[0]
    zero = csr_matrix((n, n), dtype=complex)
    output = {}
    for key in keys:
        reverse = (-key[0], -key[1], -key[2])
        lower = ud[reverse] if reverse in ud else zero
        output[key] = (
            bmat([[uu.get(key, zero), ud.get(key, zero)], [lower.getH(), dd.get(key, zero)]], format="csr")
            * HARTREE_EV
        )
    return output


def parse_input_metadata(lines, atom_orbitals, spin_mode="nonmagnetic") -> tuple[list[dict], dict]:
    """Interpret SCF input labels without guessing ionic pseudopotential charges."""
    species, atoms = {}, {}
    duplicate_atoms = False
    section = None
    for original in lines:
        line = original.split("#", 1)[0].strip()
        lower = line.lower()
        if lower.startswith("<definition.of.atomic.species"):
            section = "species"
        elif lower.startswith("<atoms.speciesandcoordinates"):
            section = "atoms"
        elif lower.endswith(">"):
            section = None
        elif line and section == "species":
            parts = line.split()
            if len(parts) >= 3:
                species[parts[0]] = {"basis": parts[1], "pseudopotential": parts[2]}
        elif line and section == "atoms":
            parts = line.split()
            if len(parts) >= 5 and parts[0].isdigit():
                duplicate_atoms |= int(parts[0]) - 1 in atoms
                atoms[int(parts[0]) - 1] = parts[1]
    labels = []
    valid = bool(species and set(atoms) == set(range(len(atom_orbitals))) and not duplicate_atoms)
    for atom, count in enumerate(atom_orbitals):
        alias = atoms.get(atom)
        spec = species.get(alias, {})
        token = spec.get("basis", "")
        functions = []
        suffix = token.split("-", 1)[-1]
        for angular, number, contracted in re.findall(r"([spdfgh])(\d+)(?:>(\d+))?", suffix):
            angular_momentum = "spdfgh".index(angular)
            multiplicity = int(contracted or number)
            for zeta in range(multiplicity):
                for m in range(2 * angular_momentum + 1):
                    functions.append(
                        {
                            "atom_index": atom,
                            "species": alias,
                            "l": angular_momentum,
                            "zeta": zeta,
                            "m": m,
                            "harmonic": _HARMONICS.get(
                                angular_momentum, tuple(map(str, range(2 * angular_momentum + 1)))
                            )[m],
                            "pao_key": token.split("-", 1)[0],
                            "contracted": bool(contracted),
                        }
                    )
        if len(functions) != int(count):
            valid = False
            functions = [
                {"atom_index": atom, "species": alias, "orbital_index": i} for i in range(int(count))
            ]
        labels.extend(functions)
    if spin_mode != "nonmagnetic":
        labels = [{**row, "spin": spin} for spin in ("up", "down") for row in labels]
    return labels, {
        "species": species,
        "atom_species": [atoms.get(atom) for atom in range(len(atom_orbitals))],
        "orbital_labels_complete": valid,
    }


def _validate_sidecar_structure(lines, lattice, positions, source):
    """Check supplementary labels against the authoritative binary structure."""
    units = {"atoms.speciesandcoordinates.unit": "ang", "atoms.unitvectors.unit": "ang"}
    section, coordinates, vectors = None, {}, []
    for original in lines:
        line = original.split("#", 1)[0].strip()
        parts = line.split()
        if not parts:
            continue
        lower = parts[0].lower()
        if lower in units and len(parts) > 1:
            units[lower] = parts[1].lower()
        elif lower == "<atoms.speciesandcoordinates":
            section = "coordinates"
        elif lower == "<atoms.unitvectors":
            section = "cell"
        elif lower.endswith(">"):
            section = None
        elif section == "coordinates":
            try:
                index = int(parts[0]) - 1
                if index in coordinates or len(parts) < 5:
                    raise ValueError("duplicate or incomplete atomic coordinate")
                coordinates[index] = [float(x) for x in parts[2:5]]
            except ValueError as error:
                raise ValueError(f"{source}: invalid supplemental coordinates ({error})") from error
        elif section == "cell":
            try:
                vectors.append([float(x) for x in parts[:3]])
            except ValueError as error:
                raise ValueError(f"{source}: invalid supplemental lattice") from error
    if set(coordinates) != set(range(len(positions))):
        raise ValueError(f"{source}: supplemental atom indices/count do not match SCFOUT")
    unit = units["atoms.speciesandcoordinates.unit"]
    supplied = np.asarray([coordinates[i] for i in range(len(positions))])
    if unit == "au":
        supplied *= BOHR_ANGSTROM
    elif unit == "frac":
        supplied = supplied @ lattice
    elif unit != "ang":
        raise ValueError(f"{source}: unsupported supplemental coordinate unit {unit}")
    if not np.isfinite(supplied).all():
        raise ValueError(f"{source}: nonfinite supplemental coordinates")
    difference = (supplied - positions) @ np.linalg.inv(lattice)
    difference -= np.rint(difference)
    if np.max(np.linalg.norm(difference @ lattice, axis=1), initial=0) > 1e-5:
        raise ValueError(f"{source}: supplemental positions/order do not match SCFOUT")
    if vectors:
        supplied_cell = np.asarray(vectors)
        unit = units["atoms.unitvectors.unit"]
        if unit == "au":
            supplied_cell *= BOHR_ANGSTROM
        elif unit != "ang":
            raise ValueError(f"{source}: unsupported supplemental cell unit {unit}")
        if supplied_cell.shape != (3, 3) or not np.allclose(supplied_cell, lattice, atol=1e-6, rtol=1e-8):
            raise ValueError(f"{source}: supplemental lattice does not match SCFOUT")


def read_scfout(path: str | Path, *, openmx_input: str | Path | None = None) -> OpenMXData:
    """Read v3 SCFOUT directly; exact reads validate counts before allocation."""
    source = Path(path).expanduser().resolve()
    size = source.stat().st_size
    with source.open("rb") as stream:
        head = stream.read(24)
        if len(head) != 24:
            raise ValueError(f"{source}: truncated SCFOUT header")
        candidates = []
        for endian in ("<", ">"):
            header = struct.unpack(endian + "6i", head)
            if 0 < header[0] <= size // 4 and 0 <= header[1] < 64 and 0 <= header[5] <= size // 16:
                candidates.append((endian, header))
        if len(candidates) != 1:
            raise ValueError(f"{source}: cannot uniquely determine SCFOUT endianness")
        endian, header = candidates[0]
        version, spin = divmod(header[1], 4)
        if version != 3:
            raise ValueError(f"{source}: unsupported SCFOUT version {version}; expected v3")
        if spin not in (0, 1, 3):
            raise ValueError(f"{source}: unsupported spin flag {spin}")
        mode = {0: "nonmagnetic", 1: "collinear", 3: "soc"}[spin]
        reader = _BinaryReader(stream, source, endian, size)
        atomnum, cells = header[0], header[5] + 1
        order = int(reader.ints(1, "position expansion order")[0])
        if not 0 <= order <= 64:
            raise ValueError(f"{source}: invalid position expansion order {order}")
        translations_cart = reader.doubles(4 * cells, "cell translations").reshape(cells, 4)[:, 1:]
        translations = reader.ints(4 * cells, "integer cell translations").reshape(cells, 4)[:, 1:]
        counts = reader.ints(atomnum, "orbital counts")
        neighbors = reader.ints(atomnum, "neighbor counts")
        if np.any(counts < 1) or np.any(neighbors < 0) or int(counts.sum()) > size // 8:
            raise ValueError(f"{source}: invalid atom/orbital/neighbor counts")
        atoms = [reader.ints(int(count) + 1, f"neighbors of atom {i}") for i, count in enumerate(neighbors)]
        cells_by_atom = [
            reader.ints(int(count) + 1, f"neighbor cells of atom {i}") for i, count in enumerate(neighbors)
        ]
        lattice = reader.doubles(12, "lattice vectors").reshape(3, 4)[:, 1:]
        reciprocal_raw = reader.doubles(12, "reciprocal vectors").reshape(3, 4)[:, 1:]
        positions = reader.doubles(4 * atomnum, "atomic positions").reshape(atomnum, 4)[:, 1:]
        if abs(np.linalg.det(lattice)) < 1e-12:
            raise ValueError(f"{source}: singular lattice")
        if not np.allclose(translations @ lattice, translations_cart, atol=1e-8, rtol=1e-8):
            raise ValueError(f"{source}: Cartesian/integer translations disagree with lattice")
        if not np.allclose(lattice @ reciprocal_raw.T, 2 * np.pi * np.eye(3), atol=1e-8, rtol=1e-8):
            raise ValueError(f"{source}: reciprocal vectors disagree with lattice")
        elements = 0
        for a in range(atomnum):
            for neighbor, cell in zip(atoms[a], cells_by_atom[a], strict=True):
                b = int(neighbor) - 1
                if not 0 <= b < atomnum or not 0 <= cell < cells:
                    raise ValueError(f"{source}: invalid neighbor index for atom {a}")
                elements += int(counts[a]) * int(counts[b])
        components = 2 * (spin + 1) + (3 if spin == 3 else 0) + 1 + 3 * order + 3 + 2
        if 8 * components * elements + 88 > size - stream.tell():
            raise ValueError(f"{source}: truncated declared matrix records at byte {stream.tell()}")
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(int)
        n = int(offsets[-1])
        layout = []
        for a in range(atomnum):
            for neighbor, cell in zip(atoms[a], cells_by_atom[a], strict=True):
                b = int(neighbor) - 1
                if not 0 <= b < atomnum or not 0 <= cell < cells:
                    raise ValueError(f"{source}: invalid neighbor index for atom {a}")
                row, col = np.meshgrid(
                    np.arange(offsets[a], offsets[a + 1]),
                    np.arange(offsets[b], offsets[b + 1]),
                    indexing="ij",
                )
                key = (int(translations[cell, 0]), int(translations[cell, 1]), int(translations[cell, 2]))
                layout.append((key, row.ravel(), col.ravel()))

        def matrices(number: int, name: str) -> tuple[Blocks, ...]:
            output = []
            for component in range(number):
                entries: dict[tuple[int, int, int], list] = {}
                for key, row, col in layout:
                    value = reader.doubles(len(row), f"{name}[{component}] matrix")
                    entries.setdefault(key, []).append((row, col, value))
                blocks = {}
                for key, parts in entries.items():
                    rows, cols, values = (np.concatenate([part[i] for part in parts]) for i in range(3))
                    block = coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
                    block.eliminate_zeros()
                    blocks[key] = block
                output.append(blocks)
            return tuple(output)

        real_h = matrices(spin + 1, "Hamiltonian")
        imag_h = matrices(3, "imaginary Hamiltonian") if spin == 3 else ()
        spatial_s = matrices(1, "overlap")[0]
        moments = tuple(
            tuple(
                _combine((matrices(1, f"position axis {axis} order {m + 1}")[0], BOHR_ANGSTROM ** (m + 1)))
                for m in range(order)
            )
            for axis in range(3)
        )
        momentum = matrices(3, "momentum derivative")
        density = matrices(spin + 1, "density")
        imaginary_density = matrices(2, "imaginary density")
        solver = int(reader.ints(1, "solver identifier")[0])
        scalars = reader.doubles(10, "SCF scalar metadata")
        lines = []
        if stream.tell() <= size:
            line_count = int(reader.ints(1, "embedded input line count")[0])
            if line_count < 0 or line_count > (size - stream.tell()) // 256:
                raise ValueError(f"{source}: truncated embedded input records")
            for _ in range(line_count):
                record = stream.read(256).split(b"\0", 1)[0].split(b"\n", 1)[0]
                lines.append(record.decode("utf-8", errors="replace"))
            if stream.tell() != size:
                raise ValueError(f"{source}: unexpected trailing bytes after embedded input")
        if openmx_input is not None:
            external = Path(openmx_input).expanduser().resolve()
            supplemental = external.read_text().splitlines()
            _, existing = parse_input_metadata(lines, counts, mode)
            _, added = parse_input_metadata(supplemental, counts, mode)
            if existing["species"] and added["species"] != existing["species"]:
                raise ValueError(f"{external}: species/basis metadata differs from SCFOUT")
            if not added["orbital_labels_complete"]:
                raise ValueError(f"{external}: atom/species orbital counts do not match SCFOUT")
            if all(existing["atom_species"]) and existing["atom_species"] != added["atom_species"]:
                raise ValueError(f"{external}: atom species/order differs from SCFOUT")
            _validate_sidecar_structure(
                supplemental, lattice * BOHR_ANGSTROM, positions * BOHR_ANGSTROM, external
            )
            lines = supplemental
        labels, input_metadata = parse_input_metadata(lines, counts, mode)
        centers = np.repeat(positions * BOHR_ANGSTROM, counts, axis=0)
        if mode != "nonmagnetic":
            centers = np.tile(centers, (2, 1))
        for label, center in zip(labels, centers, strict=True):
            label["center"] = center.tolist()
        overlap = spatial_s if mode == "nonmagnetic" else _spin_diagonal(spatial_s)
        absolute_position = []
        for axis in range(3):
            if order:
                relative = moments[axis][0]
                centered = {
                    key: block.multiply(np.repeat(positions[:, axis] * BOHR_ANGSTROM, counts)[:, None])
                    for key, block in spatial_s.items()
                }
                operator = _combine((relative, 1), (centered, 1))
                absolute_position.append(operator if mode == "nonmagnetic" else _spin_diagonal(operator))
            else:
                absolute_position.append({})
        return OpenMXData(
            _spin_hamiltonian(real_h, imag_h, mode),
            overlap,
            (absolute_position[0], absolute_position[1], absolute_position[2]),
            lattice * BOHR_ANGSTROM,
            positions * BOHR_ANGSTROM,
            counts,
            mode,
            float(scalars[0] * HARTREE_EV),
            float(scalars[8]),
            float(scalars[1]),
            moments,
            momentum,
            density,
            imaginary_density,
            tuple(lines),
            labels,
            {
                "scfout_version": version,
                "endianness": "little" if endian == "<" else "big",
                "position_order": order,
                "solver": solver,
                "spin_flag": spin,
                "neighbor_atoms": [(row - 1).tolist() for row in atoms],
                "neighbor_translations": [translations[row].tolist() for row in cells_by_atom],
                "raw_reciprocal_bohr": reciprocal_raw.tolist(),
                "raw_cell_translations_bohr": translations_cart.tolist(),
                "hartree_to_ev": HARTREE_EV,
                "bohr_to_angstrom": BOHR_ANGSTROM,
                "pyatb_legacy_hartree_to_ev": 27.211324570273,
                "pyatb_legacy_bohr_to_angstrom": 0.52917721,
                **input_metadata,
            },
            source,
        )
