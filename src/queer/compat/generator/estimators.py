"""Estimators and conventions of the Generator's PythTB engine, kept exactly.

These reproduce the numerical choices of the Generator's ``pythtb_core`` driver,
so a QUEER run computes what a PYTHTB step computed from the same inputs. Where
a choice is basis- or sampling-dependent the docstring says so; the physical
counterparts live in the library. Energies are relative to the Fermi level and
k-points are fractional unless stated otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ...physics.slab import edge_cells_used

# ---------------------------------------------------------------------------
# Slab and ribbon edge states
# ---------------------------------------------------------------------------


def edge_weights(
    eigvecs,
    n_cells: int,
    n_orb: int,
    edge_cells: int,
    energies=None,
    degenerate_tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-state weight of each eigenvector (columns) on the first and the last edge cells.

    With ``energies``, each degenerate cluster (energy-sorted neighbours closer
    than ``degenerate_tol`` eV) is first rotated to diagonalize the first-edge
    projector, which separates the two edges where the subspace allows. The
    per-state values remain basis-dependent inside such a cluster; their cluster
    sums (:func:`queer.physics.slab.cluster_totals`) are the physical weights.
    """
    edge = edge_cells_used(edge_cells, n_cells)
    vecs = np.array(eigvecs, dtype=complex, copy=True)
    if energies is not None and len(energies) > 1:
        mask = np.zeros(n_cells * n_orb)
        mask[: edge * n_orb] = 1.0
        order = np.argsort(energies)
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and energies[order[stop]] - energies[order[stop - 1]] < degenerate_tol:
                stop += 1
            if stop - start > 1:
                index = order[start:stop]
                sub = vecs[:, index]
                projector = sub.conj().T @ (mask[:, None] * sub)
                _, rotation = np.linalg.eigh(projector)
                vecs[:, index] = sub @ rotation
            start = stop
    probability = np.abs(vecs) ** 2
    cells = probability.reshape(n_cells, n_orb, -1).sum(axis=1)
    return cells[:edge].sum(axis=0), cells[-edge:].sum(axis=0)


def edge_crossings(
    energies: Sequence,
    left: Sequence,
    right: Sequence,
    *,
    periodic: bool = False,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Count edge-localized states crossing E_F = 0 between neighbouring k-points.

    A state whose weight on an edge is at least ``threshold`` at k_j is matched
    to the nearest-energy state localized on the same edge at k_(j+1), so two
    chiral modes crossing at E_F (one per edge) are not swapped. A sign change
    of its energy is a crossing, up or down by the sign of the step; ``net`` is
    up minus down. ``periodic`` also pairs the last k-point with the first (a
    ribbon's 1D zone). The counts depend on per-state weights and are
    informational; the bulk invariant is the authority.
    """
    tally: dict[str, Any] = {"left": {"up": 0, "down": 0}, "right": {"up": 0, "down": 0}}
    count = len(energies)
    pairs = [(j, j + 1) for j in range(count - 1)]
    if periodic and count > 1:
        pairs.append((count - 1, 0))
    for side, weights in (("left", left), ("right", right)):
        for j, j1 in pairs:
            on0 = np.asarray(weights[j]) >= threshold
            on1 = np.asarray(weights[j1]) >= threshold
            if not on0.any() or not on1.any():
                continue
            here = np.asarray(energies[j])[on0]
            there = np.asarray(energies[j1])[on1]
            for e in here:
                match = there[int(np.argmin(np.abs(there - e)))]
                if (e < 0.0) != (match < 0.0):
                    tally[side]["up" if match > e else "down"] += 1
    for side in ("left", "right"):
        tally[side]["net"] = tally[side]["up"] - tally[side]["down"]
    return tally
