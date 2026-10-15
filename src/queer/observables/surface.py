"""Nonorthogonal slab and semi-infinite surface Green functions."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, solve

from ..execution import ExecutionOptions
from ..geometry.lattice import validated_points
from ..physics.broadening import lorentzian as lorentzian_shape
from ..physics.slab import (
    cluster_totals,
    degenerate_clusters,
    edge_cells_used,
    edge_probabilities,
    layer_blocks,
    slab_hamiltonian,
)
from ..physics.validation import NumericalError
from ._common import positive
from ._sampling import map_points
from ._tb import degeneracy_tolerance
from .electronic import _mean_manifolds, _projection_manifolds
from .result import ObservableResult


def _blocks(model):
    if hasattr(model, "hamiltonian_blocks"):
        return model.hamiltonian_blocks, model.overlap_blocks
    hamiltonian = {
        tuple(map(int, r)): h for r, h in zip(model.translations, model.hopping_matrices, strict=True)
    }
    return hamiltonian, {(0, 0, 0): np.eye(model.nbnd)}


def _dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _principal_layers(model, point, direction, layers):
    hamiltonian, overlap = _blocks(model)
    dimension = model.nbnd
    size = layers * dimension
    arrays = [np.zeros((size, size), dtype=complex) for _ in range(4)]
    for mapping, onsite, forward in ((hamiltonian, arrays[0], arrays[1]), (overlap, arrays[2], arrays[3])):
        for translation, block in mapping.items():
            parallel = np.array(translation, dtype=float)
            parallel[direction] = 0
            value = _dense(block) * np.exp(2j * np.pi * (point @ parallel))
            for source in range(layers):
                destination = source + translation[direction]
                shift, destination = divmod(destination, layers)
                if shift == 0:
                    onsite[
                        source * dimension : (source + 1) * dimension,
                        destination * dimension : (destination + 1) * dimension,
                    ] += value
                elif shift == 1:
                    forward[
                        source * dimension : (source + 1) * dimension,
                        destination * dimension : (destination + 1) * dimension,
                    ] += value
                elif shift not in (-1, 0, 1):
                    raise ValueError("coupling_layers is too small to include all interlayer couplings")
    for label, matrix in (("Hamiltonian", arrays[0]), ("overlap", arrays[2])):
        residual = np.linalg.norm(matrix - matrix.conj().T, ord=np.inf)
        if residual > 1e-10 * (1 + np.linalg.norm(matrix, ord=np.inf)):
            raise NumericalError(f"non-Hermitian surface {label} at k={point.tolist()}")
    return arrays


def _decimate(z, h0, h1, s0, s1, max_iterations, tolerance):
    dimension = len(h0)
    identity = np.eye(dimension)
    bulk = z * s0 - h0
    top, bottom = bulk.copy(), bulk.copy()
    forward, backward = z * s1 - h1, z * s1.conj().T - h1.conj().T
    for iteration in range(max_iterations):
        inverse = solve(bulk, identity, check_finite=False)
        upper, lower = forward @ inverse @ backward, backward @ inverse @ forward
        top -= upper
        bottom -= lower
        bulk -= upper + lower
        forward, backward = -forward @ inverse @ forward, -backward @ inverse @ backward
        residual = max(np.linalg.norm(forward, ord=np.inf), np.linalg.norm(backward, ord=np.inf))
        if residual <= tolerance * (1 + np.linalg.norm(bulk, ord=np.inf)):
            top_green = solve(top, identity, check_finite=False)
            bottom_green = solve(bottom, identity, check_finite=False)
            bulk_green = solve(bulk, identity, check_finite=False)
            return top_green, bottom_green, bulk_green, iteration + 1
    raise NumericalError(f"surface decimation failed to converge within {max_iterations} iterations")


def _slab(h0, h1, s0, s1, layers):
    size = len(h0)
    hamiltonian = np.kron(np.eye(layers), h0)
    overlap = np.kron(np.eye(layers), s0)
    for layer in range(layers - 1):
        first, second = slice(layer * size, (layer + 1) * size), slice((layer + 1) * size, (layer + 2) * size)
        hamiltonian[first, second], hamiltonian[second, first] = h1, h1.conj().T
        overlap[first, second], overlap[second, first] = s1, s1.conj().T
    return hamiltonian, overlap


def _slab_eigenstates(model, hamiltonian, overlap):
    if getattr(model, "spin_mode", None) != "collinear":
        values, vectors = eigh(hamiltonian, overlap, check_finite=True)
        return values, vectors, np.zeros(len(values), dtype=int)
    # The slab groups cells first, while each cell is all-up then all-down.
    block = np.arange(len(hamiltonian)).reshape(-1, model.nbnd)
    up, down = block[:, : model.n_spatial].ravel(), block[:, model.n_spatial :].ravel()
    values, vectors = [], np.zeros_like(hamiltonian)
    offset = 0
    for indices in (up, down):
        energy, coeff = eigh(
            hamiltonian[np.ix_(indices, indices)], overlap[np.ix_(indices, indices)], check_finite=True
        )
        values.extend(energy)
        vectors[np.ix_(indices, np.arange(offset, offset + len(indices)))] = coeff
        offset += len(indices)
    return np.asarray(values), vectors, np.repeat([0, 1], len(up))


def surface_state(
    model,
    kpoints,
    omega=None,
    surface_direction="c",
    coupling_layers=None,
    method="green",
    slab_layers=20,
    eta=0.01,
    max_iterations=200,
    tolerance=1e-12,
    execution=None,
):
    """Return layer Mulliken spectra, with energies relative to the model EF.

    Principal layers automatically include the longest perpendicular hopping.
    ``green`` computes semi-infinite top/bottom and bulk responses; ``slab``
    evaluates a finite slab by generalized diagonalization; ``direct_green``
    inverts that finite slab's nonorthogonal Green operator directly. ``bands``
    returns finite-slab eigenenergies, layer projections and physical spin.
    """
    points = validated_points(kpoints)
    aliases = {"green_fun": "green", "direct_diag": "slab"}
    method = aliases.get(method, method)
    if method not in {"green", "slab", "direct_green", "bands"}:
        raise ValueError("method must be green, slab, direct_green or bands")
    if method != "bands":
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        if omega.ndim != 1 or not np.isfinite(omega).all():
            raise ValueError("spectral surface methods require a finite one-dimensional omega array")
    else:
        omega = np.array([], dtype=float)
    positive(eta, "eta")
    positive(tolerance, "tolerance")
    directions = {"a": 0, "b": 1, "c": 2, "x": 0, "y": 1, "z": 2}
    direction = directions.get(surface_direction, surface_direction)
    if direction not in (0, 1, 2):
        raise ValueError("surface_direction must be a, b, c or an axis index")
    hamiltonian, overlap = _blocks(model)
    minimum = max(1, max(abs(r[direction]) for r in set(hamiltonian) | set(overlap)))
    layers = minimum if coupling_layers is None else coupling_layers
    if isinstance(layers, bool) or int(layers) != layers or layers < minimum:
        raise ValueError(f"coupling_layers must be an integer at least {minimum}")
    if isinstance(slab_layers, bool) or int(slab_layers) != slab_layers or slab_layers < 2:
        raise ValueError("slab_layers must be an integer at least two")
    if isinstance(max_iterations, bool) or int(max_iterations) != max_iterations or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    dimension = int(layers) * model.nbnd
    working_dimension = dimension if method == "green" else dimension * int(slab_layers)
    options = (execution or getattr(model, "execution", ExecutionOptions())).resolved()
    # At least one dense surface/slab solve must fit the numerical memory budget.
    options.bounded_batch_size(256 * working_dimension**2)
    top, bottom, bulk = (np.empty((len(omega), len(points))) for _ in range(3))
    iteration_counts = np.zeros((len(omega), len(points)), dtype=int)
    identity = np.eye(working_dimension)
    if method == "bands":
        eigenenergies = np.empty((working_dimension, len(points)))
        layer_weights = np.empty((int(slab_layers), working_dimension, len(points)))
        spin = np.zeros((working_dimension, len(points), 3))
        manifold_sizes = np.ones_like(eigenenergies, dtype=int)
        channels = np.zeros(working_dimension, dtype=int)
    for k, point in enumerate(points):
        h0, h1, s0, s1 = _principal_layers(model, point, int(direction), int(layers))
        if method != "green":
            slab_h, slab_s = _slab(h0, h1, s0, s1, int(slab_layers))
        if method == "bands":
            values, vectors, channels = _slab_eigenstates(model, slab_h, slab_s)
            blocks, sizes = _projection_manifolds(model, values, channels, np.arange(working_dimension))
            manifold_sizes[:, k] = sizes
            eigenenergies[:, k] = values - model.fermi_energy
            mulliken = (vectors.conj() * (slab_s @ vectors)).real
            layer_weights[:, :, k] = mulliken.reshape(int(slab_layers), dimension, working_dimension).sum(
                axis=1
            )
            if getattr(model, "spin_mode", "nonmagnetic") != "nonmagnetic":
                cells = np.arange(working_dimension).reshape(-1, model.nbnd)
                up, down = cells[:, : model.n_spatial].ravel(), cells[:, model.n_spatial :].ravel()
                operators = np.zeros((3, working_dimension, working_dimension), complex)
                operators[0, up, down] = operators[0, down, up] = 1
                operators[1, up, down], operators[1, down, up] = -1j, 1j
                operators[2, up, up], operators[2, down, down] = 1, -1
                spin[:, k] = (
                    0.5 * np.einsum("in,aij,jn->na", vectors.conj(), slab_s @ operators, vectors).real
                )
            _mean_manifolds(layer_weights[:, :, k], blocks, 1)
            _mean_manifolds(spin[:, k], blocks, 0)
            continue
        if method == "slab":
            values, vectors = eigh(slab_h, slab_s, check_finite=True)
            mulliken = (vectors.conj() * (slab_s @ vectors)).real
            weights_top = mulliken[:dimension].sum(axis=0)
            weights_bottom = mulliken[-dimension:].sum(axis=0)
            for w, frequency in enumerate(omega):
                lorentzian = lorentzian_shape(frequency + model.fermi_energy - values, eta)
                top[w, k] = weights_top @ lorentzian
                bottom[w, k] = weights_bottom @ lorentzian
                bulk[w, k] = lorentzian.sum() / int(slab_layers)
            continue
        for w, frequency in enumerate(omega):
            z = frequency + model.fermi_energy + 1j * eta
            if method == "direct_green":
                green = solve(z * slab_s - slab_h, identity, check_finite=False)
                metric_green = slab_s @ green
                top[w, k] = -np.trace(metric_green[:dimension, :dimension]).imag / np.pi
                bottom[w, k] = -np.trace(metric_green[-dimension:, -dimension:]).imag / np.pi
                bulk[w, k] = -np.trace(metric_green).imag / (np.pi * int(slab_layers))
            else:
                gt, gb, gm, iterations = _decimate(z, h0, h1, s0, s1, int(max_iterations), tolerance)
                # Include neighboring overlap blocks in layer Mulliken density.
                gt10 = -gt @ (z * s1.conj().T - h1.conj().T) @ gt
                gb01 = -gb @ (z * s1 - h1) @ gb
                gm10 = -gt @ (z * s1.conj().T - h1.conj().T) @ gm
                gm01 = -gb @ (z * s1 - h1) @ gm
                top[w, k] = -np.trace(s0 @ gt + s1 @ gt10).imag / np.pi
                bottom[w, k] = -np.trace(s0 @ gb + s1.conj().T @ gb01).imag / np.pi
                bulk[w, k] = -np.trace(s0 @ gm + s1 @ gm10 + s1.conj().T @ gm01).imag / np.pi
                iteration_counts[w, k] = iterations
    degeneracy = float(getattr(model, "spin_degeneracy", 1))
    if method == "bands":
        mode = getattr(model, "spin_mode", "nonmagnetic")
        labels = (
            np.where(channels == 0, "up", "down") if mode == "collinear" else np.full(len(channels), mode)
        )
        return ObservableResult(
            {
                "kpoints": points,
                "energies": eigenenergies,
                "layer_weights": layer_weights,
                "surface_weights": layer_weights[[0, -1]],
                "spin": spin,
                "band_channels": channels,
                "channel_labels": labels,
                "band_indices": np.arange(working_dimension),
                "manifold_sizes": manifold_sizes,
            },
            {
                "kpoints": ("k", "cartesian"),
                "energies": ("band", "k"),
                "layer_weights": ("layer", "band", "k"),
                "surface_weights": ("surface", "band", "k"),
                "spin": ("band", "k", "cartesian"),
                "band_channels": ("band",),
                "channel_labels": ("band",),
                "band_indices": ("band",),
                "manifold_sizes": ("band", "k"),
            },
            {
                "kpoints": "fractional_reciprocal",
                "energies": "eV",
                "layer_weights": "1",
                "surface_weights": "1",
                "spin": "hbar",
                "band_channels": "1",
                "channel_labels": "1",
                "band_indices": "1",
                "manifold_sizes": "1",
            },
            {
                "method": "bands",
                "coupling_layers": int(layers),
                "slab_layers": int(slab_layers),
                "projection": "layer_mulliken",
                "surface_order": ["top", "bottom"],
                "spin_degeneracy": degeneracy,
                "fermi_reference": float(model.fermi_energy),
                "degenerate_state_interpretation": "complete_subspace_mean_per_state",
                "degeneracy_tolerance_eV": degeneracy_tolerance(model),
            },
        )
    return ObservableResult(
        {
            "kpoints": points,
            "omega": omega,
            "spectrum": top * degeneracy,
            "bottom_spectrum": bottom * degeneracy,
            "bulk_spectrum": bulk * degeneracy,
            "iterations": iteration_counts,
        },
        {
            "kpoints": ("k", "cartesian"),
            "omega": ("energy",),
            "spectrum": ("energy", "k"),
            "bottom_spectrum": ("energy", "k"),
            "bulk_spectrum": ("energy", "k"),
            "iterations": ("energy", "k"),
        },
        {
            "kpoints": "fractional_reciprocal",
            "omega": "eV",
            "spectrum": "states/eV/principal_layer",
            "bottom_spectrum": "states/eV/principal_layer",
            "bulk_spectrum": "states/eV/principal_layer",
            "iterations": "1",
        },
        {
            "method": method,
            "coupling_layers": int(layers),
            "slab_layers": int(slab_layers),
            "projection": "layer_mulliken_including_neighbor_overlap",
            "converged": True,
        },
    )


def _slab_axes(model, finite_axis, periodic_axes):
    if isinstance(finite_axis, bool) or finite_axis not in (0, 1, 2):
        raise ValueError("finite_axis must be a lattice axis index 0, 1 or 2")
    if periodic_axes is None:
        translations = np.asarray(model.translations)
        periodic_axes = [axis for axis in range(3) if axis != finite_axis and np.any(translations[:, axis])]
    periodic = sorted({int(axis) for axis in periodic_axes})
    if any(axis not in (0, 1, 2) for axis in periodic) or int(finite_axis) in periodic:
        raise ValueError("periodic_axes must be lattice axes other than finite_axis")
    return int(finite_axis), periodic


def _home_block(model):
    """Index and E_F-shifted copy of the home-cell block, or None without one."""
    home = np.flatnonzero(~np.asarray(model.translations).any(axis=1))
    if not len(home):
        return None
    return int(home[0]), model.hopping_matrices[home[0]] - model.fermi_energy * np.eye(model.nbnd)


def _windowed_slab(model, home, finite_axis, periodic, cells, window, k_par):
    # Views of the model's blocks; only the home block is replaced by its shifted copy.
    blocks = list(model.hopping_matrices)
    if home is not None:
        blocks[home[0]] = home[1]
    big = slab_hamiltonian(layer_blocks(model.translations, blocks, finite_axis, periodic, k_par), cells)
    if home is None:
        big -= model.fermi_energy * np.eye(len(big))
    return eigh(big, subset_by_value=(-window, window), driver="evr")


def _slab_options(model, cells, window):
    if isinstance(cells, bool) or int(cells) != cells or cells < 2:
        raise ValueError("cells must be an integer of at least two")
    positive(window, "window")
    return int(cells), float(window)


def slab_eigensystem(model, k_par, *, finite_axis, cells, window=0.5, periodic_axes=None):
    """Levels of an open slab with E - E_F in (-window, window] at one in-plane k, and their vectors.

    The slab keeps ``cells`` whole unit cells along ``finite_axis`` with no change
    of termination: H[w, w + d] = B_d(k_par), the blocks of
    :func:`queer.physics.slab.layer_blocks`, cell-major with the model's orbital
    order in each cell. ``periodic_axes`` defaults to the other axes that the
    hopping table crosses; hops along any remaining axis are dropped. E_F is
    taken off the home-cell block before assembly, so energies are relative to
    it, and the eigenvectors are columns.
    """
    finite, periodic = _slab_axes(model, finite_axis, periodic_axes)
    count, width = _slab_options(model, cells, window)
    point = np.asarray(k_par, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("k_par must be a finite fractional (3,) point")
    return _windowed_slab(model, _home_block(model), finite, periodic, count, width, point)


def slab_edge_spectrum(
    model,
    kpoints,
    *,
    finite_axis,
    cells,
    window=0.5,
    edge_cells=2,
    periodic_axes=None,
    degenerate_tol=1e-5,
    execution=None,
):
    """Open-slab levels near E_F with each degenerate cluster's weight on the two edges.

    Levels are those of :func:`slab_eigensystem`. A cluster is a run of
    energy-sorted levels closer than ``degenerate_tol`` eV; every level carries its
    cluster's total probability on the first (``left``) and last (``right``)
    ``edge_cells`` cells, which does not depend on the basis chosen inside the
    cluster. Arrays are padded per k-point to the largest level count: NaN
    energies and weights, cluster -1 and size 0, with ``state_count`` per k.
    """
    points = validated_points(kpoints)
    finite, periodic = _slab_axes(model, finite_axis, periodic_axes)
    count, width = _slab_options(model, cells, window)
    positive(degenerate_tol, "degenerate_tol")
    edge = edge_cells_used(edge_cells, count)
    home = _home_block(model)
    dimension = count * model.nbnd
    options = (execution or getattr(model, "execution", None) or ExecutionOptions()).resolved()
    # One dense slab matrix, its eigenvectors and the solver workspace must fit the budget.
    options.bounded_batch_size(256 * dimension**2)

    def at(point):
        energies, vectors = _windowed_slab(model, home, finite, periodic, count, width, point)
        left, right = edge_probabilities(vectors, count, model.nbnd, edge)
        labels = degenerate_clusters(energies, degenerate_tol)
        sizes = np.bincount(labels)[labels] if len(labels) else np.zeros(0, dtype=np.int64)
        return energies, cluster_totals(left, labels), cluster_totals(right, labels), labels, sizes

    rows = map_points(model, points, at, options)
    states = np.asarray([len(row[0]) for row in rows], dtype=np.int64)
    width_out = int(np.max(states, initial=0))
    energies, left, right = (np.full((len(points), width_out), np.nan) for _ in range(3))
    clusters = np.full((len(points), width_out), -1, dtype=np.int64)
    sizes = np.zeros((len(points), width_out), dtype=np.int64)
    for index, (level, first, last, label, size) in enumerate(rows):
        span = slice(0, len(level))
        energies[index, span], left[index, span], right[index, span] = level, first, last
        clusters[index, span], sizes[index, span] = label, size
    pair = ("kpoint", "state")
    return ObservableResult(
        {
            "kpoints": points,
            "energies": energies,
            "left_weight": left,
            "right_weight": right,
            "cluster": clusters,
            "cluster_size": sizes,
            "state_count": states,
        },
        {
            "kpoints": ("kpoint", "fractional"),
            "energies": pair,
            "left_weight": pair,
            "right_weight": pair,
            "cluster": pair,
            "cluster_size": pair,
            "state_count": ("kpoint",),
        },
        {
            "kpoints": "fractional",
            "energies": "eV",
            "left_weight": "1",
            "right_weight": "1",
            "cluster": "1",
            "cluster_size": "1",
            "state_count": "1",
        },
        {
            "observable": "slab_edge_spectrum",
            "finite_axis": finite,
            "periodic_axes": periodic,
            "cells": count,
            "states": dimension,
            "window_eV": width,
            "edge_cells": edge,
            "degenerate_tolerance_eV": float(degenerate_tol),
            "energy_reference": "relative to the model Fermi energy",
            "weight_definition": "probability on the edge cells summed over each degenerate cluster",
            "layout": "cell-major, model orbital order within each cell",
        },
    )
