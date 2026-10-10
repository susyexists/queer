"""Native localized-Hamiltonian graphs with explicit basis and edge identity.

Scientific archives use eV and Angstrom, contain only numeric/string arrays,
and never require Torch, PyG, HamGNN, or an external OpenMX decoder.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .artifacts import atomic_npz, canonical_json
from .openmx import OpenMXData, read_scfout


@dataclass
class HamiltonianGraph:
    atomic_numbers: np.ndarray
    cell: np.ndarray
    positions: np.ndarray
    orbital_counts: np.ndarray
    orbital_slots: np.ndarray
    edge_index: np.ndarray
    cell_shift: np.ndarray
    inverse_edges: np.ndarray
    hamiltonian: np.ndarray
    overlap: np.ndarray
    h0: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        integer_fields = (
            "atomic_numbers",
            "orbital_counts",
            "orbital_slots",
            "edge_index",
            "cell_shift",
            "inverse_edges",
        )
        for key in integer_fields:
            value = np.asarray(getattr(self, key))
            if value.dtype.kind not in "iu" and not np.equal(value, np.round(value)).all():
                raise ValueError(f"{key} must contain integers")
            setattr(self, key, value.astype(np.int64))
        for key in ("cell", "positions", "hamiltonian", "overlap"):
            value = np.asarray(getattr(self, key))
            if value.dtype.kind not in "fci" or not np.isfinite(value).all():
                raise ValueError(f"{key} must contain finite numerical values")
            setattr(self, key, value.astype(complex if np.iscomplexobj(value) else float))
        n, e = len(self.atomic_numbers), self.edge_index.shape[-1]
        if (
            n == 0
            or self.atomic_numbers.shape != (n,)
            or np.any((self.atomic_numbers < 1) | (self.atomic_numbers > 118))
        ):
            raise ValueError("atomic_numbers must identify nonempty real elements")
        if self.cell.shape != (3, 3) or abs(np.linalg.det(self.cell)) < 1e-12:
            raise ValueError("graph cell must be nonsingular (3,3)")
        if self.positions.shape != (n, 3) or self.orbital_counts.shape != (n,):
            raise ValueError("graph positions/orbital counts do not match atoms")
        if self.orbital_slots.ndim != 2 or self.orbital_slots.shape[0] != n:
            raise ValueError("orbital_slots must have one row per atom")
        width = self.orbital_slots.shape[1]
        if np.any((self.orbital_counts < 1) | (self.orbital_counts > width)):
            raise ValueError("invalid orbital counts")
        for i, count in enumerate(self.orbital_counts):
            slots = self.orbital_slots[i, :count]
            if np.any((slots < 0) | (slots >= width)) or len(set(slots)) != count:
                raise ValueError("orbital slot placement is invalid or duplicated")
            if np.any(self.orbital_slots[i, count:] != -1):
                raise ValueError("unused orbital slot entries must be -1")
        if self.edge_index.shape != (2, e) or np.any((self.edge_index < 0) | (self.edge_index >= n)):
            raise ValueError("edge_index must have shape (2,E) with valid atoms")
        if self.cell_shift.shape != (e, 3) or self.inverse_edges.shape != (e,):
            raise ValueError("edge shifts/inverse maps do not match edges")
        if np.any((self.inverse_edges < 0) | (self.inverse_edges >= e)):
            raise ValueError("invalid inverse edge index")
        keys = [
            (int(a), int(b), *map(int, r))
            for (a, b), r in zip(self.edge_index.T, self.cell_shift, strict=True)
        ]
        if len(set(keys)) != e:
            raise ValueError("graph contains duplicate edges")
        if any(a == b and r == (0, 0, 0) for a, b, *shift in keys for r in [tuple(shift)]):
            raise ValueError("Zero-translation self edges duplicate the implicit onsite rows")
        for i, inverse in enumerate(self.inverse_edges):
            if (
                self.inverse_edges[inverse] != i
                or not np.array_equal(self.edge_index[:, inverse], self.edge_index[::-1, i])
                or not np.array_equal(self.cell_shift[inverse], -self.cell_shift[i])
            ):
                raise ValueError("inverse edge map does not preserve reversed atoms/translations")
        shape = (n + e, width, width)
        if self.hamiltonian.shape != shape or self.overlap.shape != shape:
            raise ValueError(f"padded H and S must have shape {shape}")
        if self.h0 is not None:
            self.h0 = np.asarray(self.h0)
            if self.h0.shape != shape or not np.isfinite(self.h0).all():
                raise ValueError("H0 must match finite padded Hamiltonian blocks")
        if np.any(np.abs(self.hamiltonian[~self.orbital_mask]) > 0) or np.any(
            np.abs(self.overlap[~self.orbital_mask]) > 0
        ):
            raise ValueError("nonzero entries outside the declared orbital basis")
        self.metadata = dict(self.metadata)
        canonical_json(self.metadata)

    @property
    def natoms(self):
        return len(self.atomic_numbers)

    @property
    def nao_max(self):
        return self.orbital_slots.shape[1]

    @property
    def row_atoms(self):
        onsite = np.arange(self.natoms)
        return np.column_stack((np.r_[onsite, self.edge_index[0]], np.r_[onsite, self.edge_index[1]]))

    @property
    def orbital_mask(self):
        mask = np.zeros(self.hamiltonian.shape, bool)
        for row, (a, b) in enumerate(self.row_atoms):
            left = self.orbital_slots[a, : self.orbital_counts[a]]
            right = self.orbital_slots[b, : self.orbital_counts[b]]
            mask[row][np.ix_(left, right)] = True
        return mask

    def arrays(self):
        keys = (
            "atomic_numbers",
            "cell",
            "positions",
            "orbital_counts",
            "orbital_slots",
            "edge_index",
            "cell_shift",
            "inverse_edges",
            "hamiltonian",
            "overlap",
        )
        result = {key: np.asarray(getattr(self, key)) for key in keys}
        if self.h0 is not None:
            result["h0"] = self.h0
        return result

    def with_predictions(self, predictions, *, unit="eV"):
        from .openmx import HARTREE_EV

        value = np.asarray(predictions)
        if value.shape == (len(self.hamiltonian), self.nao_max**2):
            value = value.reshape(self.hamiltonian.shape)
        if value.shape != self.hamiltonian.shape or not np.isfinite(value).all():
            raise ValueError("predictions do not match graph rows/basis")
        if unit not in {"eV", "Ha"}:
            raise ValueError("prediction unit must be eV or Ha")
        values = dict(self.arrays())
        values["hamiltonian"] = np.where(self.orbital_mask, value, 0) * (HARTREE_EV if unit == "Ha" else 1)
        return HamiltonianGraph(
            **values,
            metadata={
                **self.metadata,
                "hamiltonian_source": "prediction",
                "source_has_scf_reference": bool(self.metadata.get("has_scf_reference")),
                "has_scf_reference": False,
            },
        )


def _slots_from_labels(data, width):
    # The declared checkpoint representation reserves s3, p2, d1/d2, f1.
    # Generate placement from orbital quantum labels, never from element guesses.
    offsets, capacities = {0: 0, 1: 3, 2: 9, 3: 19}, {0: 3, 1: 2, 2: 1 if width == 14 else 2, 3: 1}
    output = []
    start = 0
    for count in data.atom_orbitals:
        labels = data.orbital_metadata[start : start + count]
        values = []
        for label in labels:
            if not {"l", "zeta", "m"}.issubset(label):
                raise ValueError("Incomplete orbital labels; supply basis_layout explicitly")
            angular, zeta, m = (int(label[key]) for key in ("l", "zeta", "m"))
            if (
                angular not in offsets
                or zeta >= capacities[angular]
                or zeta < 0
                or m < 0
                or m >= 2 * angular + 1
            ):
                raise ValueError("basis exceeds padded shell capacity; supply compatible basis_layout")
            slot = offsets[angular] + zeta * (2 * angular + 1) + m
            if slot >= width:
                raise ValueError("orbital shell does not fit nao_max")
            values.append(slot)
        output.append(values)
        start += count
    return output


def pack_openmx(
    source: OpenMXData | str | Path,
    *,
    nao_max=14,
    basis_layout=None,
    atomic_numbers=None,
    h0=None,
    overlap_only=False,
    openmx_input=None,
    spin=None,
) -> HamiltonianGraph:
    """Pack original OpenMX neighbor order; checkpoint basis placement is explicit."""
    data = source if isinstance(source, OpenMXData) else read_scfout(source, openmx_input=openmx_input)
    if data.spin_mode == "soc":
        raise ValueError("Generator padded graphs require a non-SOC Hamiltonian")
    if data.spin_mode == "collinear" and spin not in {"up", "down"}:
        raise ValueError("Collinear graph packing requires explicit spin=up/down")
    if data.spin_mode == "nonmagnetic" and spin is not None:
        raise ValueError("A nonmagnetic graph has no explicit spin channel")
    if nao_max not in (14, 19, 26):
        raise ValueError("nao_max must be 14, 19, or 26")
    n = len(data.atom_orbitals)
    if atomic_numbers is None:
        from ..elements import atomic_number

        aliases = data.metadata.get("atom_species", [])
        if len(aliases) != n or not all(aliases):
            raise ValueError("Element metadata absent; supply atomic_numbers")
        atomic_numbers = []
        for name in aliases:
            match = re.match(r"[A-Z][a-z]?", name)
            if match is None:
                raise ValueError(f"Unknown element alias {name!r}; supply atomic_numbers")
            atomic_numbers.append(atomic_number(match.group()))
    counts = np.asarray(data.atom_orbitals, int)
    placements = _slots_from_labels(data, nao_max) if basis_layout is None else basis_layout
    if isinstance(placements, Mapping):
        placements = [
            placements[int(z)] if int(z) in placements else placements[str(int(z))] for z in atomic_numbers
        ]
    if len(placements) != n:
        raise ValueError("basis_layout must have one placement per atom")
    slots = np.full((n, nao_max), -1, int)
    for i, (indices, count) in enumerate(zip(placements, counts, strict=True)):
        if len(indices) != count:
            raise ValueError("basis_layout length differs from SCF orbital count")
        slots[i, :count] = indices
    neighbors = data.metadata.get("neighbor_atoms")
    translations = data.metadata.get("neighbor_translations")
    if neighbors is None or translations is None:
        raise ValueError("Packing requires original neighbor order from native read_scfout")
    edges, shifts = [], []
    for a, (adjacent, cells) in enumerate(zip(neighbors, translations, strict=True)):
        if not adjacent or adjacent[0] != a or list(cells[0]) != [0, 0, 0]:
            raise ValueError("SCF neighbor list must begin with zero-translation onsite atom")
        for b, r in zip(adjacent[1:], cells[1:], strict=True):
            edges.append((a, b))
            shifts.append(r)
    edge_index = np.asarray(edges, int).reshape(-1, 2).T
    cell_shift = np.asarray(shifts, int).reshape(-1, 3)
    lookup = {
        (int(a), int(b), *map(int, r)): i for i, ((a, b), r) in enumerate(zip(edges, cell_shift, strict=True))
    }
    try:
        inverse = np.array(
            [lookup[(b, a, *map(int, -r))] for (a, b), r in zip(edges, cell_shift, strict=True)], int
        )
    except KeyError as error:
        raise ValueError("SCF neighbor list is missing a reverse edge") from error
    offsets = np.r_[0, counts.cumsum()]
    channel_offset = int(offsets[-1]) if spin == "down" else 0

    def padded(blocks):
        result = np.zeros((n + len(edges), nao_max, nao_max), dtype=complex)
        rows = [(a, a, (0, 0, 0)) for a in range(n)] + [
            (a, b, tuple(r)) for (a, b), r in zip(edges, cell_shift, strict=True)
        ]
        for i, (a, b, r) in enumerate(rows):
            if r not in blocks:
                raise ValueError(f"SCF operator missing translation {r}")
            block = blocks[r][
                channel_offset + offsets[a] : channel_offset + offsets[a + 1],
                channel_offset + offsets[b] : channel_offset + offsets[b + 1],
            ].toarray()
            result[i][np.ix_(slots[a, : counts[a]], slots[b, : counts[b]])] = block
        return np.real_if_close(result)

    h0_data = None if h0 is None else (h0 if isinstance(h0, OpenMXData) else read_scfout(h0))
    if h0_data is not None:
        if (
            not np.array_equal(h0_data.atom_orbitals, counts)
            or h0_data.spin_mode != data.spin_mode
            or not np.allclose(h0_data.lattice, data.lattice)
            or not np.allclose(h0_data.atom_positions, data.atom_positions)
            or h0_data.metadata.get("neighbor_atoms") != neighbors
            or h0_data.metadata.get("neighbor_translations") != translations
        ):
            raise ValueError("H0 structure, basis, or neighbor order differs from SCF")
    hamiltonian = padded(data.hamiltonian)
    return HamiltonianGraph(
        np.asarray(atomic_numbers),
        data.lattice,
        data.atom_positions,
        counts,
        slots,
        edge_index,
        cell_shift,
        inverse,
        hamiltonian,
        padded(data.overlap),
        padded(h0_data.hamiltonian) if h0_data is not None else hamiltonian.copy(),
        {
            "schema_version": 1,
            "energy_unit": "eV",
            "length_unit": "Angstrom",
            "fermi_energy_eV": None if overlap_only else data.fermi_energy,
            "reported_file_fermi_energy_eV": data.fermi_energy,
            "electron_count": data.electron_count if spin is None else None,
            "total_electron_count": data.electron_count,
            "spin_channel": spin,
            "spin_degeneracy": 2 if spin is None else 1,
            "has_scf_reference": not overlap_only,
            "h0_aliases_h": h0 is None,
            "hamiltonian_source": "h0" if overlap_only else "scf",
            "source": str(data.source) if data.source is not None else None,
        },
    )


def write_graphs(path: str | Path, graphs: Mapping[str, HamiltonianGraph]) -> Path:
    """Write numeric-only multi-structure graphs with preserved insertion order."""
    if not graphs:
        raise ValueError("At least one graph is required")
    arrays, descriptions = {}, []
    for i, (key, graph) in enumerate(graphs.items()):
        prefix = f"g{i:06d}_"
        arrays.update({prefix + name: value for name, value in graph.arrays().items()})
        descriptions.append({"id": str(key), "prefix": prefix, "metadata": graph.metadata})
    arrays["metadata_json"] = np.asarray(canonical_json({"schema_version": 1, "graphs": descriptions}))
    target = Path(path)
    atomic_npz(target, arrays)
    return target


def read_graphs(path: str | Path) -> dict[str, HamiltonianGraph]:
    with np.load(path, allow_pickle=False) as archive:
        description = json.loads(str(archive["metadata_json"]))
        if description.get("schema_version") != 1:
            raise ValueError("Unsupported Hamiltonian graph archive")
        result = {}
        for row in description["graphs"]:
            prefix = row["prefix"]
            arrays = {name[len(prefix) :]: archive[name] for name in archive.files if name.startswith(prefix)}
            if row["id"] in result:
                raise ValueError("Duplicate graph identifiers")
            result[row["id"]] = HamiltonianGraph(**arrays, metadata=row["metadata"])
    return result


def masked_hamiltonian_errors(graphs, predictions, *, unit="eV", splits=None):
    """Errors over actual orbitals, never padded zeros or H0-only pseudo-targets."""
    from .openmx import HARTREE_EV

    values = list(graphs.values()) if isinstance(graphs, Mapping) else list(graphs)
    if not values or any(not g.metadata.get("has_scf_reference", False) for g in values):
        raise ValueError("Hamiltonian errors require a real SCF reference for every graph")
    if unit not in {"eV", "Ha"}:
        raise ValueError("prediction unit must be eV or Ha")
    prediction = np.asarray(predictions) * (HARTREE_EV if unit == "Ha" else 1)
    target = np.concatenate([g.hamiltonian.reshape(len(g.hamiltonian), -1) for g in values])
    mask = np.concatenate([g.orbital_mask.reshape(len(g.hamiltonian), -1) for g in values])
    if prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise ValueError("Prediction shapes/order do not match graph rows")
    bounds = np.r_[0, np.cumsum([len(g.hamiltonian) for g in values])]
    result = {}
    for key, indices in (splits or {"all": range(len(values))}).items():
        rows = np.zeros(len(target), bool)
        for index in indices:
            if not 0 <= index < len(values):
                raise ValueError("Invalid structure split index")
            rows[bounds[index] : bounds[index + 1]] = True
        errors = np.abs(prediction - target)[mask & rows[:, None]]
        if len(errors):
            result[key] = {
                "mae_eV": float(errors.mean()),
                "rmse_eV": float(np.sqrt((errors**2).mean())),
                "orbital_entries": len(errors),
                "structures": int(len(indices)),
            }
    return result
