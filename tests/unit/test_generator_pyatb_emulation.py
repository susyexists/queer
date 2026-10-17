"""PyATB's refined Grid sum as QUEER emulates it, against a direct transcription of its rule."""

import itertools

import numpy as np
import pytest

from queer.compat.generator.pyatb_emulation import (
    gamma_nodes,
    leading_minors,
    plane_projection,
    refined_sum,
    vector_projection,
)


def field(points):
    """Two smooth periodic components with sharp peaks, the rows a per-k function returns."""
    k = np.asarray(points, float)
    first = 3 * np.exp(4 * np.cos(2 * np.pi * k[:, 0]) + 2 * np.cos(2 * np.pi * k[:, 1])) / np.exp(6)
    second = np.sin(2 * np.pi * k[:, 0]) * np.cos(2 * np.pi * (k[:, 1] - k[:, 2]))
    return np.column_stack([first, second])


def brute_force(grid, subgrid, threshold, origin, vectors):
    """PyATB's grid_integrate_3D rule, node by node: (mean, number of refined nodes)."""
    vectors = np.asarray(vectors, float)
    total, refined = np.zeros(2), 0
    for n in itertools.product(*(range(size) for size in grid)):
        node = origin + sum(vectors[axis] * n[axis] / grid[axis] for axis in range(3))
        value = field(node[None])[0]
        if np.linalg.norm(value) < threshold:
            total += value / np.prod(grid)
            continue
        refined += 1
        start = node - sum(vectors[axis] / grid[axis] for axis in range(3)) * 0.5
        for m in itertools.product(*(range(size) for size in subgrid)):
            point = start + sum(vectors[axis] / grid[axis] * m[axis] / subgrid[axis] for axis in range(3))
            total += field(point[None])[0] / np.prod(grid) / np.prod(subgrid)
    return total, refined


def plain(points, rows):
    return rows, np.zeros_like(rows)


@pytest.mark.parametrize(
    ("grid", "subgrid", "origin", "vectors"),
    [
        ([8, 6, 1], [2, 3, 1], np.zeros(3), np.eye(3)),
        ([5, 5, 2], [2, 2, 2], np.zeros(3), np.eye(3)),
        (
            [6, 6, 1],
            [3, 3, 1],
            np.array([0.1, 0.0, 0.2]),
            [[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        ),
    ],
)
def test_the_refined_sum_is_pyatbs_rule_node_by_node(grid, subgrid, origin, vectors):
    nodes = gamma_nodes(grid, origin, vectors)
    assert len(nodes) == np.prod(grid)
    expected, refined = brute_force(grid, subgrid, 0.6, origin, vectors)
    result = refined_sum(
        nodes,
        field(nodes),
        field,
        grid=grid,
        subgrid=subgrid,
        threshold=0.6,
        vectors=vectors,
        project=lambda rows: rows,
        magnitude=lambda rows: np.linalg.norm(rows, axis=1),
        regularise=plain,
    )
    assert 0 < refined < np.prod(grid) and result["flagged"] == refined
    np.testing.assert_allclose(result["mean"], expected, rtol=1e-13, atol=1e-15)
    assert result["sub_points"] == refined * np.prod(subgrid) and result["ties"] == 0


def test_the_refined_sum_evaluates_its_sub_grids_in_batches_of_whole_nodes():
    grid, subgrid = [8, 6, 1], [2, 3, 1]
    nodes = gamma_nodes(grid, np.zeros(3), np.eye(3))
    calls = []

    def rows_at(points):
        calls.append(len(points))
        return field(points)

    def windows(points, rows):
        return rows, np.abs(rows)

    settings = {
        "grid": grid,
        "subgrid": subgrid,
        "threshold": 0.6,
        "vectors": np.eye(3),
        "project": lambda rows: rows,
        "magnitude": lambda rows: np.linalg.norm(rows, axis=1),
        "regularise": windows,
    }
    whole = refined_sum(nodes, field(nodes), field, **settings)
    batched = refined_sum(nodes, field(nodes), rows_at, batch_points=13, **settings)
    # 13 points hold two whole 6-point sub-grids; the last batch holds what is left.
    assert sum(calls) == batched["sub_points"] == whole["sub_points"] and max(calls) == 12
    assert all(count % 6 == 0 for count in calls) and len(calls) == -(-whole["sub_points"] // 12)
    for key in ("mean", "tie_allowance", "flag_allowance"):
        np.testing.assert_allclose(batched[key], whole[key], rtol=1e-14, atol=1e-16)
    assert batched["flagged"] == whole["flagged"] and batched["ties"] == whole["ties"]


def test_a_1x1x1_subgrid_still_moves_a_flagged_node_half_a_cell():
    nodes = gamma_nodes([4, 4, 1], np.zeros(3), np.eye(3))
    result = refined_sum(
        nodes,
        field(nodes),
        field,
        grid=[4, 4, 1],
        subgrid=[1, 1, 1],
        threshold=0.0,
        vectors=np.eye(3),
        project=lambda rows: rows,
        magnitude=lambda rows: np.linalg.norm(rows, axis=1),
        regularise=plain,
    )
    # Every node is replaced by the node half a cell below it, the third axis included.
    shifted = nodes - np.array([0.125, 0.125, 0.5])
    np.testing.assert_allclose(result["mean"], field(shifted).mean(axis=0), rtol=1e-14)


def test_ties_and_occupation_windows_are_reported():
    nodes = gamma_nodes([4, 4, 1], np.zeros(3), np.eye(3))
    rows = field(nodes)
    sizes = np.linalg.norm(rows, axis=1)
    order = np.argsort(sizes, kind="stable")
    # A node exactly at the threshold is a tie; the smallest node is never refined.
    threshold, quiet = float(sizes[order[8]]), order[0]
    windows = np.zeros_like(rows)
    windows[quiet] = [1.0, 2.0]

    def regularise(points, given):
        return given, (windows if len(points) == len(nodes) else np.zeros_like(given))

    result = refined_sum(
        nodes,
        rows,
        field,
        grid=[4, 4, 1],
        subgrid=[2, 2, 1],
        threshold=threshold,
        vectors=np.eye(3),
        project=lambda r: r,
        magnitude=lambda r: np.linalg.norm(r, axis=1),
        regularise=regularise,
    )
    assert result["ties"] >= 1 and np.any(result["tie_allowance"] > 0)
    np.testing.assert_allclose(result["flag_allowance"], [1 / 16, 2 / 16])


def test_projections_map_the_tensor_to_its_vector_and_plane_flux():
    tensor = np.array([[0.0, 3.0, -2.0], [-3.0, 0.0, 1.0], [2.0, -1.0, 0.0]])
    np.testing.assert_array_equal(vector_projection() @ tensor.ravel(), [1.0, 2.0, 3.0])
    u, w = np.array([1.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0])
    assert (plane_projection(u, w) @ tensor.ravel())[0] == pytest.approx(np.cross(u, w) @ [1.0, 2.0, 3.0])


def test_leading_minors_are_the_unpivoted_determinants():
    rng = np.random.default_rng(7)
    matrices = rng.normal(size=(3, 5, 5)) + 1j * rng.normal(size=(3, 5, 5)) + 4 * np.eye(5)
    minors = leading_minors(matrices)
    for row, matrix in zip(minors, matrices, strict=True):
        np.testing.assert_allclose(row, [np.linalg.det(matrix[:n, :n]) for n in range(1, 6)], rtol=1e-12)
    # A zero pivot zeroes every later minor, as PyATB's elimination does, even where the
    # pivoted determinant is not zero.
    np.testing.assert_array_equal(leading_minors(np.array([[[0.0, 1.0], [1.0, 0.0]]])), [[0.0, 0.0]])


def test_bloch_tables_sum_their_sparse_blocks():
    from scipy.sparse import csr_matrix

    from queer.compat.generator.pyatb_emulation import BlochTable

    rng = np.random.default_rng(3)
    blocks = {
        key: csr_matrix(rng.normal(size=(3, 3)) * (rng.random((3, 3)) > 0.4))
        for key in [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, -1, 2)]
    }
    points = np.array([[0.1, -0.3, 0.2], [0.5, 0.25, 0.0], [0.0, 0.0, 0.0]])
    table = BlochTable(blocks, 3)
    for point, value in zip(points, table.at(points), strict=True):
        expected = sum(
            np.exp(2j * np.pi * np.dot(point, key)) * block.toarray() for key, block in blocks.items()
        )
        np.testing.assert_allclose(value, expected, atol=1e-13)
