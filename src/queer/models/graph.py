"""Native generalized-eigenproblem reconstruction from padded graph blocks."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from ..capabilities import CapabilityError
from ..io.hamiltonian import HamiltonianGraph
from .openmx import OpenMXModel


class GraphHamiltonianModel(OpenMXModel):
    """Only operators actually supplied by the graph are advertised."""

    def __init__(
        self, graph: HamiltonianGraph, *, fermi_energy=None, hermitian_policy="validate", execution=None
    ):
        if hermitian_policy not in {"validate", "average"}:
            raise ValueError("hermitian_policy must be validate or average")
        counts, n = graph.orbital_counts, graph.natoms
        offsets = np.r_[0, counts.cumsum()]
        shifts = np.vstack((np.zeros((n, 3), int), graph.cell_shift))

        def matrices(values):
            blocks: dict[tuple[int, ...], np.ndarray] = {}
            for row, ((a, b), r) in enumerate(zip(graph.row_atoms, shifts, strict=True)):
                key = tuple(map(int, r))
                block = blocks.setdefault(key, np.zeros((offsets[-1], offsets[-1]), complex))
                left = graph.orbital_slots[a, : counts[a]]
                right = graph.orbital_slots[b, : counts[b]]
                block[offsets[a] : offsets[a + 1], offsets[b] : offsets[b + 1]] = values[row][
                    np.ix_(left, right)
                ]
            keys = set(blocks) | {tuple(-np.array(r)) for r in blocks}
            zero = np.zeros((offsets[-1], offsets[-1]), complex)
            output = {}
            for key in keys:
                reverse = tuple(-np.array(key))
                value, partner = blocks.get(key, zero), blocks.get(reverse, zero).conj().T
                if hermitian_policy == "validate" and not np.allclose(value, partner, atol=1e-7, rtol=1e-7):
                    raise ValueError("Graph operators violate translation Hermiticity; opt in to averaging")
                output[key] = csr_matrix((value + partner) / 2 if hermitian_policy == "average" else value)
            return output

        hopping, overlap = matrices(graph.hamiltonian), matrices(graph.overlap)
        labels = [
            {"atom_index": atom, "orbital_index": orbital, "center": graph.positions[atom].tolist()}
            for atom, count in enumerate(counts)
            for orbital in range(count)
        ]
        ef = graph.metadata.get("fermi_energy_eV") if fermi_energy is None else fermi_energy
        if ef is None:
            raise CapabilityError(
                "Graph Hamiltonians require a declared Fermi reference; pass fermi_energy explicitly"
            )
        super().__init__(
            np.array(sorted(set(hopping) | set(overlap))),
            hopping,
            overlap,
            lattice=graph.cell,
            fermi_energy=ef,
            spin_degeneracy=graph.metadata.get("spin_degeneracy", 2),
            atom_positions=graph.positions,
            orbital_metadata=labels,
            execution=execution,
        )
        self.graph = graph
        self.electron_count = graph.metadata.get("electron_count")
        self.metadata.update(
            {
                **graph.metadata,
                "hermitian_policy": hermitian_policy,
                "representation": "native graph Hamiltonian",
            }
        )
