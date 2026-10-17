"""PyATB's own schemes evaluated with QUEER's per-k functions: diagnostics for a comparison.

QUEER's OpenMX steps report QUEER's physical estimators. A parity comparison with
PyATB sometimes needs what PyATB's own scheme gives on the same model; these
functions evaluate that scheme with QUEER's model and per-k functions, as the
Wannier comparator recomputes an old producer's DOS rule:

- :func:`refined_sum` is PyATB's refined Grid sum (``integrate_mode Grid`` with a
  threshold below 1e30): the Gamma-origin nodes, a node whose judged value
  reaches the threshold replaced by the ``adaptive_grid`` sub-grid that starts
  half a coarse cell below it. On the grid and its double it gives PyATB's own
  discretisation error ``e_P``, which PyATB does not write for a Grid.
- :func:`pyatb_rule_rows` is the occupied curvature at a flagged k-point under
  PyATB's rules: its direct engine zeroes the coefficient derivative of every pair
  closer than 1e-4 eV, and a level within the Fermi-energy tolerance may be
  occupied in one code and empty in the other.

Their values are recorded only in the ``pyatb_emulation`` blocks and the flagged
sets of ``queer_results.json``, never as a result (``sigma_S_cm``,
``chern.value``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ...models.evaluation import band_geometry, derivatives
from ...physics.geometry import band_curvatures, energy_blocks, projected_basis_curvature
from ...physics.statistics import occupations
from .pyatb_conventions import PYATB_DEGENERATE_EV

# A node within this relative distance of the threshold may be judged the other way by PyATB.
TIE_RELATIVE = 1e-8


def gamma_nodes(grid: Any, origin: Any, vectors: Any) -> np.ndarray:
    """``origin + sum_i v_i n_i / N_i`` for ``n_i = 0 .. N_i - 1``, the first axis slowest: PyATB's Grid nodes."""
    mesh = np.asarray(grid, int)
    index = np.indices(tuple(mesh)).reshape(len(mesh), -1).T
    return np.asarray(origin, float) + (index / mesh) @ np.asarray(vectors, float)


def refined_sum(
    nodes: np.ndarray,
    rows: np.ndarray,
    rows_at: Callable[[np.ndarray], np.ndarray],
    *,
    grid: Any,
    subgrid: Any,
    threshold: float,
    vectors: Any,
    project: Callable[[np.ndarray], np.ndarray],
    magnitude: Callable[[np.ndarray], np.ndarray],
    regularise: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """PyATB's refined Grid mean over ``nodes`` (its Gamma-origin grid) with their per-k ``rows``.

    ``regularise(points, rows)`` gives the rows PyATB sees (its own rule at the
    flagged points) and each point's occupation window. A node whose
    ``magnitude`` reaches ``threshold`` is then replaced by the ``subgrid`` nodes
    ``node - (sum_i v_i / N_i) / 2 + sum_i (v_i / N_i) m_i / n_i`` (``rows_at``
    gives their rows), each weighted ``1 / (N n)``; every other node counts once
    with weight ``1 / N``. ``project`` maps rows to the summed values.

    Returns the ``mean``, the ``flagged`` count, the nodes within a relative 1e-8
    of the threshold (``ties``) with ``tie_allowance``, the change a different
    decision at them would make, and ``flag_allowance``, the weighted occupation
    windows of the points the mean uses.
    """
    mesh = np.asarray(grid, int)
    weight = 1.0 / float(np.prod(mesh))
    nodes = np.asarray(nodes, float)
    rows, windows = regularise(nodes, rows)
    values = np.asarray(project(rows), float)
    size = np.asarray(magnitude(rows), float)
    flagged = size >= threshold
    near = np.abs(size - threshold) <= TIE_RELATIVE * abs(threshold)
    step = np.asarray(vectors, float) / mesh[:, None]
    offsets = gamma_nodes(subgrid, np.zeros(3), step)
    chosen = np.flatnonzero(flagged | near)
    corner = nodes[chosen] - 0.5 * step.sum(axis=0)
    points = (corner[:, None, :] + offsets[None]).reshape(-1, 3)
    if len(points):
        sub_rows, sub_windows = regularise(points, np.asarray(rows_at(points)))
    else:
        sub_rows, sub_windows = np.empty((0, rows.shape[1])), np.empty((0, values.shape[1]))
    sub_values = np.asarray(project(sub_rows), float).reshape(len(chosen), len(offsets), values.shape[1])
    means = sub_values.mean(axis=1)
    refined = flagged[chosen]
    total = values[~flagged].sum(axis=0) + means[refined].sum(axis=0)
    ties = near[chosen]
    tie_allowance = np.abs(means[ties] - values[chosen][ties]).sum(axis=0) * weight
    used = np.repeat(refined, len(offsets))
    flag_allowance = windows[~flagged].sum(axis=0) * weight
    if used.any():
        flag_allowance = flag_allowance + sub_windows[used].sum(axis=0) * weight / len(offsets)
    return {
        "mean": total * weight,
        "flagged": int(flagged.sum()),
        "ties": int(near.sum()),
        "tie_allowance": tie_allowance,
        "flag_allowance": flag_allowance,
        "sub_points": int(len(points)),
    }


def _pair_cut(data: dict[str, Any]) -> dict[str, Any]:
    """``data`` with the coefficient derivative of every pair closer than 1e-4 eV set to zero, as PyATB's direct
    engine sets it, and the connections rebuilt from it."""
    energies = np.asarray(data["energies"], float)
    close = np.abs(energies[None, :] - energies[:, None]) <= PYATB_DEGENERATE_EV
    np.fill_diagonal(close, False)
    d = np.array(data["coefficient_derivatives"], copy=True)
    d[:, close] = 0.0
    r = data["projected_derivatives"]["r"]
    connections = 1j * d + r.conj().transpose(0, 2, 1)
    connections = (connections + connections.conj().transpose(0, 2, 1)) * 0.5
    return {**data, "coefficient_derivatives": d, "connections": connections}


def pyatb_rule_rows(
    model: Any,
    mu: float,
    engine: str,
    occupied_bands: int | None,
    delta_ef: float,
    projection: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Per k-point: PyATB's occupied curvature tensor and how far its occupation may move it.

    Columns 0-8 hold the occupied curvature tensor (Å², row-major) under PyATB's
    direct-engine rule: the coefficient derivative of every pair closer than 1e-4
    eV is zero (its Kubo engine has no cut, so the tensor is QUEER's). The next
    ``m`` columns hold ``sum_B |P Omega_B|`` over the degenerate blocks with a
    level within ``delta_ef`` of ``mu``, which either code may count as occupied
    (none for a band-count occupation); ``projection`` (m x 9) maps a tensor to
    the summed components.
    """
    projection = np.asarray(projection, float)
    rows = np.zeros((len(points), 9 + projection.shape[0]))
    mus = np.array([float(mu)])
    for index, point in enumerate(np.asarray(points, float)):
        data = band_geometry(model, point)
        energies = np.asarray(data["energies"], float)
        if occupied_bands is None:
            weights = occupations(energies[:, None], mus[None], 0.0)[:, 0]
        else:
            weights = np.zeros(len(energies))
            weights[np.argsort(energies, kind="stable")[: int(occupied_bands)]] = 1.0
        if engine == "direct":
            cut = _pair_cut(data)
            curvature = band_curvatures(
                cut, "direct", projected_basis_curvature(cut, derivatives(model, point)["r"])
            )
        else:
            curvature = band_curvatures(data, engine, None)
        rows[index, :9] = np.einsum("n,nab->ab", weights, curvature).ravel()
        if occupied_bands is None:
            for block in energy_blocks(data):
                if np.min(np.abs(energies[block] - mu)) < delta_ef:
                    rows[index, 9:] += np.abs(projection @ curvature[block].sum(axis=0).ravel())
    return rows


def vector_projection() -> np.ndarray:
    """The 3 x 9 map of a row-major tensor to its axial vector ``(T_yz, T_zx, T_xy)``."""
    projection = np.zeros((3, 9))
    for axis, (a, b) in enumerate(((1, 2), (2, 0), (0, 1))):
        projection[axis, 3 * a + b] = 1.0
    return projection


def plane_projection(first: Any, second: Any) -> np.ndarray:
    """The 1 x 9 map ``T -> sum_ab T_ab u_a w_b`` of a plane spanned by Cartesian ``u`` and ``w``."""
    return np.outer(np.asarray(first, float), np.asarray(second, float)).reshape(1, 9)
