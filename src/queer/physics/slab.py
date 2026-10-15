"""Open slabs and ribbons cut from a periodic hopping table, and their edge weights.

A slab keeps ``n_cells`` whole unit cells along its finite axis with no
termination change, so H_slab[w, w + d] = B_d(k_par) for cells w and w + d.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def layer_blocks(translations, hopping, finite_axis: int, periodic_axes: Sequence[int], k_par):
    """Inter-cell blocks B_d(k_par) = sum_(R: R_f = d) H(R) exp(2 pi i k_par . R_par).

    ``k_par`` is fractional (3,); only its ``periodic_axes`` components enter.
    Hops along an axis that is neither finite nor periodic (a layer's vacuum)
    are dropped; such a model has none once its vacuum axes are folded.
    """
    translations = np.asarray(translations, int)
    others = [axis for axis in range(3) if axis != finite_axis and axis not in periodic_axes]
    blocks: dict[int, np.ndarray] = {}
    k_par = np.asarray(k_par, float)
    for r, h in zip(translations, hopping):
        if any(r[axis] != 0 for axis in others):
            continue
        phase = np.exp(2j * np.pi * sum(k_par[axis] * r[axis] for axis in periodic_axes))
        d = int(r[finite_axis])
        if d in blocks:
            blocks[d] = blocks[d] + phase * h
        else:
            blocks[d] = phase * h
    return blocks


def slab_hamiltonian(blocks, n_cells: int) -> np.ndarray:
    """Block-Toeplitz Hamiltonian of ``n_cells`` cells: H[w, w + d] = B_d, cell-major."""
    n = next(iter(blocks.values())).shape[0]
    big = np.zeros((n_cells * n, n_cells * n), dtype=complex)
    for d, block in blocks.items():
        for w in range(n_cells):
            w2 = w + d
            if 0 <= w2 < n_cells:
                big[w * n : (w + 1) * n, w2 * n : (w2 + 1) * n] += block
    return big


def edge_cells_used(edge_cells: int, n_cells: int) -> int:
    """Cells counted per edge: ``edge_cells`` clipped to [1, n_cells // 2]."""
    return max(1, min(int(edge_cells), n_cells // 2))


def edge_probabilities(vectors, n_cells: int, n_orb: int, edge_cells: int):
    """Probability of each eigenvector (columns) on the first and the last edge cells."""
    edge = edge_cells_used(edge_cells, n_cells)
    cells = (np.abs(np.asarray(vectors)) ** 2).reshape(n_cells, n_orb, -1).sum(axis=1)
    return cells[:edge].sum(axis=0), cells[-edge:].sum(axis=0)


def degenerate_clusters(energies, tol: float) -> np.ndarray:
    """Cluster label of every level: energy-sorted neighbours closer than ``tol`` share one.

    Labels count clusters in increasing energy and are returned in input order.
    """
    e = np.asarray(energies, float).reshape(-1)
    order = np.argsort(e, kind="stable")
    labels = np.empty(len(e), dtype=np.int64)
    if len(e):
        starts = np.r_[True, np.diff(e[order]) >= tol]
        labels[order] = np.cumsum(starts) - 1
    return labels


def cluster_totals(values, labels) -> np.ndarray:
    """Per-level sum of ``values`` over its whole cluster, invariant to rotations within it."""
    values = np.asarray(values, float)
    labels = np.asarray(labels, np.int64)
    totals = np.zeros(int(labels.max()) + 1 if len(labels) else 0)
    np.add.at(totals, labels, values)
    return totals[labels]
