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
- :func:`fhs_plaquette_phases` is PyATB's FHS plaquette scan: its link from k to
  k + dk is ``W(k)^+ M W(k + dk)`` with ``M = sum_R exp(2 pi i k.R) (S_R + i dk.r_R)``,
  the first-order expansion of ``exp(i dk.r)`` at the link's start, and a plaquette
  phase is the argument of the product of the leading minors of its four links at
  each vertex's occupied count, averaged over the vertices when the counts
  differ. Beside it, QUEER's own midpoint links on the same eigenvectors, so both
  link schemes are measured at once.

Their values are recorded only in the ``pyatb_emulation`` blocks and the flagged
sets of ``queer_results.json``, never as a result (``sigma_S_cm``,
``chern.value``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from ...execution import ExecutionOptions
from ...models.evaluation import band_geometry, derivatives
from ...observables.integration import SUBGRID_BATCH_POINTS
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
    batch_points: int = SUBGRID_BATCH_POINTS,
) -> dict[str, Any]:
    """PyATB's refined Grid mean over ``nodes`` (its Gamma-origin grid) with their per-k ``rows``.

    ``regularise(points, rows)`` gives the rows PyATB sees (its own rule at the
    flagged points) and each point's occupation window. A node whose
    ``magnitude`` reaches ``threshold`` is then replaced by the ``subgrid`` nodes
    ``node - (sum_i v_i / N_i) / 2 + sum_i (v_i / N_i) m_i / n_i`` (``rows_at``
    gives their rows), each weighted ``1 / (N n)``; every other node counts once
    with weight ``1 / N``. ``project`` maps rows to the summed values. The
    sub-grids are evaluated in batches of whole nodes, at most ``batch_points``
    points per ``rows_at`` call (or one node's sub-grid), and only each node's mean
    and window sum are kept.

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
    width = values.shape[1]
    means = np.empty((len(chosen), width))
    window_sums = np.empty((len(chosen), width))
    batch = max(1, int(batch_points) // len(offsets))
    for start in range(0, len(chosen), batch):
        stop = min(start + batch, len(chosen))
        points = (corner[start:stop, None, :] + offsets[None]).reshape(-1, 3)
        sub_rows, sub_windows = regularise(points, np.asarray(rows_at(points)))
        shape = (stop - start, len(offsets), width)
        means[start:stop] = np.asarray(project(sub_rows), float).reshape(shape).mean(axis=1)
        window_sums[start:stop] = np.asarray(sub_windows, float).reshape(shape).sum(axis=1)
    refined = flagged[chosen]
    total = values[~flagged].sum(axis=0) + means[refined].sum(axis=0)
    ties = near[chosen]
    tie_allowance = np.abs(means[ties] - values[chosen][ties]).sum(axis=0) * weight
    flag_allowance = windows[~flagged].sum(axis=0) * weight
    if refined.any():
        flag_allowance = flag_allowance + window_sums[refined].sum(axis=0) * weight / len(offsets)
    return {
        "mean": total * weight,
        "flagged": int(flagged.sum()),
        "ties": int(near.sum()),
        "tie_allowance": tie_allowance,
        "flag_allowance": flag_allowance,
        "sub_points": int(len(chosen) * len(offsets)),
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


# ---------------------------------------------------------------------------
# FHS plaquettes
# ---------------------------------------------------------------------------

# Plaquettes whose link matrices a worker holds at once.
FHS_BATCH = 32
_SERIAL = ExecutionOptions(workers=1, native_threads=1)


class BlochTable:
    """Real-space blocks ``R -> X_R`` stacked for the Bloch sums ``sum_R exp(2 pi i k.R) X_R`` at many k."""

    def __init__(self, blocks: Any, size: int):
        keys = sorted(blocks)
        self.size = int(size)
        self.translations = np.asarray(keys, float).reshape(-1, 3)
        rows, columns, values = [np.zeros(0, int)], [np.zeros(0, int)], [np.zeros(0, complex)]
        for index, key in enumerate(keys):
            block = coo_matrix(blocks[key])
            rows.append(np.full(block.nnz, index))
            columns.append(block.row * self.size + block.col)
            values.append(np.asarray(block.data, complex))
        self.matrix = csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(len(keys), self.size**2),
        )

    def at(self, points: np.ndarray) -> np.ndarray:
        """The Bloch sums at fractional ``points``: (n, N, N)."""
        points = np.asarray(points, float).reshape(-1, 3)
        phases = np.exp(2j * np.pi * (points @ self.translations.T))
        return np.asarray(self.matrix.T @ phases.T).T.reshape(len(points), self.size, self.size)


def leading_minors(matrices: np.ndarray) -> np.ndarray:
    """``det(A[:n, :n])`` for n = 1..N of each matrix, by elimination without pivoting.

    Each minor is the product of the first n pivots; a zero pivot makes it and every
    later minor zero. This is the determinant rule of PyATB's FHS engine.
    """
    work = np.array(matrices, dtype=complex, copy=True)
    count, size = work.shape[0], work.shape[-1]
    minors = np.empty((count, size), complex)
    running = np.ones(count, complex)
    for i in range(size):
        pivot = work[:, i, i].copy()
        running = running * pivot
        minors[:, i] = running
        if i + 1 < size:
            safe = np.where(pivot == 0, 1.0, pivot)
            factors = work[:, i + 1 :, i] / safe[:, None]
            work[:, i + 1 :, i + 1 :] -= factors[:, :, None] * work[:, i, None, i + 1 :]
    return minors


def _vertex_average(phase_of: Callable[[int], float], counts: np.ndarray) -> np.ndarray:
    """A plaquette's phase per energy: the phase at the vertices' common occupied count, else the mean of
    the four vertices' phases (zero at a vertex without occupied levels)."""
    values = np.array([[phase_of(int(n)) if n > 0 else 0.0 for n in row] for row in counts])
    uniform = np.all(counts == counts[0], axis=0) & (counts[0] > 0)
    return np.where(uniform, values[0], 0.25 * values.sum(axis=0))


def fhs_plaquette_phases(
    model: Any,
    overlap: BlochTable,
    position: Sequence[BlochTable],
    connection: Sequence[BlochTable],
    mus: np.ndarray,
    steps: np.ndarray,
    origins: np.ndarray,
) -> np.ndarray:
    """Sums over the plaquettes at ``origins`` of their phase over 2 pi, per energy, in two link schemes.

    A plaquette has the corners ``k, k + du, k + du + dv, k + dv`` (``steps = (du, dv)``,
    fractional) and is walked in that order. Row 0 is PyATB's scheme: the link
    starting at ``k`` with step ``dk`` is ``W(k)^+ [S(k) + i dk.r(k)] W(k + dk)`` with the
    raw position operator ``position``, its determinants are the leading minors
    without pivoting, and a vertex count n takes the phase of the product of the four
    links' n-th minors. Row 1 is QUEER's scheme on the same eigenvectors:
    ``W^+ [S(m) - i dk.A(m)] W`` at the link's midpoint ``m``, with ``A`` the hermitian part
    of ``connection``, and pivoted determinants. The occupied count of a vertex is
    its number of levels below each of ``mus``.
    """
    mus = np.asarray(mus, float)
    du, dv = np.asarray(steps, float)
    offsets = np.array([np.zeros(3), du, du + dv, dv])
    moves = np.array([du, dv, -du, -dv])
    cartesian = moves @ reciprocal_rows(model)
    total = np.zeros((2, len(mus)))
    for start in range(0, len(origins), FHS_BATCH):
        batch = np.asarray(origins[start : start + FHS_BATCH], float)
        corners = (batch[:, None, :] + offsets[None]).reshape(-1, 3)
        solution = model.eigenstates(corners, execution=_SERIAL)
        energies = np.asarray(solution.energies, float).T.reshape(len(batch), 4, -1)
        vectors = np.asarray(solution.coefficients).reshape(len(batch), 4, model.nbnd, -1)
        s, r = overlap.at(corners), np.stack([axis.at(corners) for axis in position], axis=1)
        middle = (corners.reshape(len(batch), 4, 3) + 0.5 * moves[None]).reshape(-1, 3)
        s_mid = overlap.at(middle)
        a_mid = np.stack([axis.at(middle) for axis in connection], axis=1)
        a_mid = 0.5 * (a_mid + a_mid.conj().swapaxes(-1, -2))
        s, r = s.reshape(len(batch), 4, *s.shape[1:]), r.reshape(len(batch), 4, *r.shape[1:])
        s_mid = s_mid.reshape(len(batch), 4, *s_mid.shape[1:])
        a_mid = a_mid.reshape(len(batch), 4, *a_mid.shape[1:])
        for p in range(len(batch)):
            ends = [(vectors[p, a], vectors[p, (a + 1) % 4]) for a in range(4)]
            pyatb = np.array(
                [
                    first.conj().T @ (s[p, a] + 1j * np.einsum("x,xij->ij", cartesian[a], r[p, a])) @ second
                    for a, (first, second) in enumerate(ends)
                ]
            )
            queer = np.array(
                [
                    first.conj().T
                    @ (s_mid[p, a] - 1j * np.einsum("x,xij->ij", cartesian[a], a_mid[p, a]))
                    @ second
                    for a, (first, second) in enumerate(ends)
                ]
            )
            counts = np.array([np.searchsorted(energies[p, v], mus, side="left") for v in range(4)])
            minors = leading_minors(pyatb).prod(axis=0)
            pyatb_phase = np.angle(minors)
            queer_phase: dict[int, float] = {}

            def queer_of(n: int, links: np.ndarray = queer, cache: dict[int, float] = queer_phase) -> float:
                if n not in cache:
                    signs, _ = np.linalg.slogdet(links[:, :n, :n])
                    cache[n] = float(np.angle(np.prod(signs)))
                return cache[n]

            total[0] += _vertex_average(lambda n: float(pyatb_phase[n - 1]), counts) / (2 * np.pi)
            total[1] += _vertex_average(queer_of, counts) / (2 * np.pi)
    return total


def reciprocal_rows(model: Any) -> np.ndarray:
    """The reciprocal lattice rows with the 2 pi (1/Å): a fractional step's Cartesian length."""
    return 2 * np.pi * np.linalg.inv(np.asarray(model.lattice, float)).T
