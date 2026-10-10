"""Nonorthogonal slab and semi-infinite surface Green functions."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, solve

from ..execution import ExecutionOptions
from ..geometry.lattice import validated_points
from ..models.wannier import NumericalError
from ._common import positive
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
                lorentzian = eta / np.pi / ((frequency + model.fermi_energy - values) ** 2 + eta**2)
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
