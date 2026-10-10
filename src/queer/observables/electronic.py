"""Electronic bands, Mulliken projections, densities, and node diagnostics.

Energies returned by these routines are relative to the model Fermi reference.
Temperatures are in kelvin; broadenings and absolute chemical potentials are in eV.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize

from ..execution import ExecutionOptions
from ..geometry.lattice import validated_points
from ._common import positive
from ._tb import degeneracy_tolerance, kmesh, normalized_weights, occupations, states
from .result import ObservableResult


def _result(arrays, dims, units, **metadata):
    return ObservableResult(arrays, dims, units, metadata)


def _axis(values, name):
    array = np.atleast_1d(np.asarray(values, dtype=float))
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return array


def _degeneracy(model):
    return float(getattr(model, "spin_degeneracy", 1))


def _batches(model, points, execution):
    options = (execution or getattr(model, "execution", ExecutionOptions())).resolved()
    batch_size = options.bounded_batch_size(256 * model.nbnd**2 + 64 * model.nbnd)
    for start in range(0, len(points), batch_size):
        yield start, points[start : start + batch_size], options


def _selection(model, bands):
    if hasattr(model, "_band_indices"):
        return model._band_indices(bands)
    selection = np.arange(model.nbnd) if bands is None else np.atleast_1d(bands)
    if selection.dtype.kind not in "iu" or np.any(selection < 0) or np.any(selection >= model.nbnd):
        raise ValueError("bands must be integer model indices")
    return selection.astype(int)


def _group_indices(model, dimension, groups):
    if groups is None:
        metadata = getattr(model, "orbital_metadata", [])
        labels = [str(i) for i in range(dimension)]
        if len(metadata) == dimension:
            labels = [
                f"{row.get('species', '')}{row.get('atom_index', i)}:"
                f"{row.get('zeta', 0)}{row.get('harmonic', row.get('m', i))}:"
                f"{row.get('spin', '')}"
                for i, row in enumerate(metadata)
            ]
        return labels, [np.array([i]) for i in range(dimension)]
    if not isinstance(groups, dict) or not groups:
        raise ValueError("groups must be a nonempty mapping of labels to orbital indices")
    labels, indices = [], []
    for label, selection in groups.items():
        array = np.asarray(selection)
        if array.ndim != 1 or not len(array) or array.dtype.kind not in "iu":
            raise ValueError("each projection group must contain integer orbital indices")
        if np.any(array < 0) or np.any(array >= dimension) or len(np.unique(array)) != len(array):
            raise ValueError("projection orbital indices must be distinct and in range")
        labels.append(str(label))
        indices.append(array.astype(int))
    return labels, indices


def _mulliken(solution):
    coefficients = solution.coefficients
    weighted = np.einsum("kij,kjb->kib", solution.overlaps, coefficients, optimize=True)
    return (coefficients.conj() * weighted).real.transpose(1, 2, 0)


def _projection_manifolds(model, energies, channels, selected):
    """Validate complete selection and return same-channel degenerate blocks."""
    tolerance = degeneracy_tolerance(model)
    included = set(map(int, selected))
    blocks = []
    sizes = np.ones(len(energies), dtype=int)
    for channel in np.unique(channels):
        indices = np.flatnonzero(channels == channel)
        indices = indices[np.argsort(energies[indices], kind="stable")]
        groups = np.split(indices, np.flatnonzero(np.diff(energies[indices]) > tolerance) + 1)
        for group in groups:
            if not len(group):
                continue
            block = set(map(int, group))
            if included & block and not block <= included:
                raise ValueError(
                    "band selection cuts a degenerate subspace; select every band in the manifold"
                )
            sizes[group] = len(group)
            blocks.append(group)
    return blocks, sizes


def _mean_manifolds(values, blocks, band_axis):
    """Replace individual projections by the invariant trace per state."""
    array = np.moveaxis(values, band_axis, 0)
    for block in blocks:
        if len(block) > 1:
            array[block] = array[block].mean(axis=0, keepdims=True)
    return values


def _line_shape(difference, eta, smearing):
    eta = positive(eta, "eta")
    if smearing == "gaussian":
        return np.exp(-0.5 * (difference / eta) ** 2) / (eta * np.sqrt(2 * np.pi))
    if smearing == "lorentzian":
        return eta / np.pi / (difference**2 + eta**2)
    raise ValueError("smearing must be gaussian or lorentzian")


def band_structure(model, kpoints, bands=None, scissors=0.0, n_occ=None, spin=None, execution=None):
    points = validated_points(kpoints)
    if not np.isfinite(scissors):
        raise ValueError("scissors must be finite")
    if scissors and (n_occ is None or isinstance(n_occ, bool) or int(n_occ) != n_occ or n_occ < 0):
        raise ValueError("a scissors correction requires nonnegative integer n_occ")
    # Apply scissors before selection so the occupied threshold remains global.
    energies = model.energies(points, execution=execution)
    if scissors:
        channels = np.asarray(getattr(model, "band_channels", np.zeros(len(energies), dtype=int)))
        for channel in np.unique(channels):
            indices = np.flatnonzero(channels == channel)
            if n_occ > len(indices):
                raise ValueError("n_occ exceeds the per-channel band count")
            if 0 < n_occ < len(indices):
                gap = energies[indices[int(n_occ)]] - energies[indices[int(n_occ) - 1]]
                if np.any(np.abs(gap) <= degeneracy_tolerance(model)):
                    raise ValueError("scissors occupied boundary cuts a degenerate subspace")
            energies[indices[int(n_occ) :]] += scissors
    if hasattr(model, "_band_indices"):
        selection = model._band_indices(bands, spin)
    else:
        if spin is not None:
            raise ValueError("spin selection requires a collinear model")
        selection = _selection(model, bands)
    channels = np.asarray(getattr(model, "band_channels", np.zeros(len(energies), dtype=int)))[selection]
    if getattr(model, "spin_mode", None) == "collinear":
        labels = np.where(channels == 0, "up", "down")
    else:
        labels = np.full(len(selection), getattr(model, "spin_mode", "unspecified"))
    return _result(
        {
            "kpoints": points,
            "energies": energies[selection],
            "band_indices": selection,
            "band_channels": channels,
            "channel_labels": labels,
        },
        {
            "kpoints": ("k", "cartesian"),
            "energies": ("band", "k"),
            "band_indices": ("band",),
            "band_channels": ("band",),
            "channel_labels": ("band",),
        },
        {
            "kpoints": "fractional_reciprocal",
            "energies": "eV",
            "band_indices": "1",
            "band_channels": "1",
            "channel_labels": "1",
        },
        fermi_reference=float(model.fermi_energy),
        scissors=float(scissors),
    )


def fat_band(model, kpoints, bands=None, groups=None, execution=None):
    points = validated_points(kpoints)
    selected = _selection(model, bands)
    labels, selections = _group_indices(model, model.nbnd, groups)
    weights = np.empty((len(labels), len(selected), len(points)))
    energies = np.empty((len(selected), len(points)))
    manifold_sizes = np.ones_like(energies, dtype=int)
    channels = np.asarray(getattr(model, "band_channels", np.zeros(model.nbnd, dtype=int)))[selected]
    for start, batch, options in _batches(model, points, execution):
        solution = states(model, batch, execution=options)
        orbital_weights = _mulliken(solution)
        for local in range(len(batch)):
            blocks, sizes = _projection_manifolds(
                model, solution.energies[:, local], solution.band_channels, selected
            )
            _mean_manifolds(orbital_weights[:, :, local], blocks, 1)
            manifold_sizes[:, start + local] = sizes[selected]
        end = start + len(batch)
        energies[:, start:end] = solution.energies[selected] - model.fermi_energy
        weights[:, :, start:end] = np.asarray(
            [orbital_weights[index].sum(axis=0)[selected] for index in selections]
        )
    return _result(
        {
            "kpoints": points,
            "energies": energies,
            "weights": weights,
            "groups": np.asarray(labels),
            "band_channels": channels,
            "manifold_sizes": manifold_sizes,
        },
        {
            "kpoints": ("k", "cartesian"),
            "energies": ("band", "k"),
            "weights": ("group", "band", "k"),
            "groups": ("group",),
            "band_channels": ("band",),
            "manifold_sizes": ("band", "k"),
        },
        {
            "kpoints": "fractional_reciprocal",
            "energies": "eV",
            "weights": "1",
            "groups": "1",
            "band_channels": "1",
            "manifold_sizes": "1",
        },
        projection="mulliken",
        signed_weights=True,
        degenerate_state_interpretation="complete_subspace_mean_per_state",
        degeneracy_tolerance_eV=degeneracy_tolerance(model),
    )


def dos(model, kpoints, omega, eta=0.05, weights=None, smearing="gaussian", execution=None):
    points, omega = validated_points(kpoints), _axis(omega, "omega")
    integration_weights = normalized_weights(points, weights)
    density = np.zeros(len(omega))
    for start, batch, options in _batches(model, points, execution):
        energies = model.energies(batch, execution=options)
        for eigenvalues, weight in zip(
            energies.T, integration_weights[start : start + len(batch)], strict=True
        ):
            for energy in eigenvalues:
                density += weight * _line_shape(omega - energy, eta, smearing)
    density *= _degeneracy(model)
    return _result(
        {"omega": omega, "dos": density},
        {"omega": ("energy",), "dos": ("energy",)},
        {"omega": "eV", "dos": "states/eV/cell"},
        smearing=smearing,
        spin_degeneracy=_degeneracy(model),
    )


def pdos(model, kpoints, omega, eta=0.05, weights=None, groups=None, smearing="gaussian", execution=None):
    points, omega = validated_points(kpoints), _axis(omega, "omega")
    integration_weights = normalized_weights(points, weights)
    labels, selections = _group_indices(model, model.nbnd, groups)
    density = np.zeros((len(labels), len(omega)))
    total = np.zeros(len(omega))
    for start, batch, options in _batches(model, points, execution):
        solution = states(model, batch, execution=options)
        orbital_weights = _mulliken(solution)
        projected = np.asarray([orbital_weights[index].sum(axis=0) for index in selections])
        for k, weight in enumerate(integration_weights[start : start + len(batch)]):
            for band, energy in enumerate(solution.energies[:, k] - model.fermi_energy):
                shape = weight * _degeneracy(model) * _line_shape(omega - energy, eta, smearing)
                density += projected[:, band, k, None] * shape
                total += shape
    return _result(
        {"omega": omega, "pdos": density, "dos": total, "groups": np.asarray(labels)},
        {"omega": ("energy",), "pdos": ("group", "energy"), "dos": ("energy",), "groups": ("group",)},
        {"omega": "eV", "pdos": "states/eV/cell", "dos": "states/eV/cell", "groups": "1"},
        projection="mulliken",
        signed_weights=True,
        smearing=smearing,
    )


def jdos(
    model,
    kpoints,
    omega,
    eta=0.05,
    weights=None,
    temperature_kelvin=0.0,
    chemical_potential=None,
    occupied=None,
    smearing="gaussian",
    execution=None,
):
    points, omega = validated_points(kpoints), _axis(omega, "omega")
    integration_weights = normalized_weights(points, weights)
    chemical_potential = model.fermi_energy if chemical_potential is None else chemical_potential
    if occupied is not None:
        if isinstance(occupied, bool) or int(occupied) != occupied or not 0 <= occupied <= model.nbnd:
            raise ValueError("occupied must be an integer band count")
    density = np.zeros(len(omega))
    channels = np.asarray(getattr(model, "band_channels", np.zeros(model.nbnd, dtype=int)))
    for start, batch, options in _batches(model, points, execution):
        energies = model.energies(batch, execution=options) + model.fermi_energy
        occupation = occupations(energies, chemical_potential, temperature_kelvin)
        if occupied is not None:
            occupation[:] = 0
            for k in range(len(batch)):
                occupation[np.argsort(energies[:, k])[: int(occupied)], k] = 1
        for k, weight in enumerate(integration_weights[start : start + len(batch)]):
            for lower in range(len(energies)):
                for upper in range(len(energies)):
                    if channels[lower] != channels[upper]:
                        continue
                    transition = energies[upper, k] - energies[lower, k]
                    population = occupation[lower, k] - occupation[upper, k]
                    if transition > 0 and population > 0:
                        density += weight * population * _line_shape(omega - transition, eta, smearing)
    density *= _degeneracy(model)
    return _result(
        {"omega": omega, "jdos": density},
        {"omega": ("energy",), "jdos": ("energy",)},
        {"omega": "eV", "jdos": "transitions/eV/cell"},
        temperature_kelvin=float(temperature_kelvin),
        chemical_potential_eV=float(chemical_potential),
    )


def fermi_energy(
    model, kpoints, electron_num, temperature_kelvin=0.0, weights=None, tolerance=1e-10, execution=None
):
    points = validated_points(kpoints)
    integration_weights = normalized_weights(points, weights)
    energies = model.energies(points, execution=execution) + model.fermi_energy
    degeneracy = _degeneracy(model)
    capacity = degeneracy * len(energies)
    if not np.isfinite(electron_num) or not 0 <= electron_num <= capacity:
        raise ValueError(f"electron_num must lie within 0..{capacity:g}")
    positive(tolerance, "tolerance")
    if not len(points):
        raise ValueError("Fermi energy requires nonempty kpoints")
    if not np.isfinite(temperature_kelvin) or temperature_kelvin < 0:
        raise ValueError("temperature must be nonnegative kelvin")
    low, high = float(energies.min()), float(energies.max())
    if temperature_kelvin == 0:
        order = np.argsort(energies.ravel(), kind="stable")
        levels = energies.ravel()[order]
        counts = np.tile(integration_weights, len(energies))[order] * degeneracy
        cumulative = np.cumsum(counts)
        if electron_num == 0:
            mu = low - 1.0
        elif electron_num == capacity:
            mu = high + 1.0
        else:
            index = min(int(np.searchsorted(cumulative, electron_num - tolerance)), len(levels) - 1)
            mu = float(levels[index])
            if abs(cumulative[index] - electron_num) <= tolerance and index + 1 < len(levels):
                mu = float((levels[index] + levels[index + 1]) / 2)
    else:

        def count(mu):
            return degeneracy * np.sum(occupations(energies, mu, temperature_kelvin) * integration_weights)

        margin = max(1.0, temperature_kelvin * 8.617333262145e-5 * 50)
        if electron_num in (0, capacity):
            mu = low - margin if electron_num == 0 else high + margin
        else:
            mu = brentq(lambda x: count(x) - electron_num, low - margin, high + margin, xtol=tolerance)
    return _result(
        {"fermi_energy": np.asarray(mu), "chemical_potential": np.asarray(mu)},
        {"fermi_energy": (), "chemical_potential": ()},
        {"fermi_energy": "eV", "chemical_potential": "eV"},
        electron_num=float(electron_num),
        temperature_kelvin=float(temperature_kelvin),
        zero_temperature_partial_occupation=bool(temperature_kelvin == 0),
        chemical_potential_reference="absolute",
    )


def fermi_surface(model, kpoints, chemical_potential=None, width=0.01, bands=None, execution=None):
    positive(width, "width")
    points = validated_points(kpoints)
    energies = model.energies(points, bands=bands, execution=execution)
    chemical_potential = model.fermi_energy if chemical_potential is None else chemical_potential
    if not np.isfinite(chemical_potential):
        raise ValueError("chemical_potential must be finite")
    distance = energies + model.fermi_energy - chemical_potential
    mask = np.abs(distance) <= width
    return _result(
        {
            "kpoints": points,
            "energies": energies,
            "mask": mask,
            "weights": np.exp(-0.5 * (distance / width) ** 2),
        },
        {
            "kpoints": ("k", "cartesian"),
            "energies": ("band", "k"),
            "mask": ("band", "k"),
            "weights": ("band", "k"),
        },
        {"kpoints": "fractional_reciprocal", "energies": "eV", "mask": "1", "weights": "1"},
        chemical_potential=float(chemical_potential),
        chemical_potential_reference="absolute",
        width=float(width),
        representation="sampled_surface",
    )


def spin_texture(model, kpoints, bands=None, execution=None):
    points = validated_points(kpoints)
    selected = _selection(model, bands)
    energies = np.empty((len(selected), len(points)))
    spin = np.zeros((*energies.shape, 3))
    manifold_sizes = np.ones_like(energies, dtype=int)
    channels = np.asarray(getattr(model, "band_channels", np.zeros(model.nbnd, dtype=int)))[selected]
    for start, batch, options in _batches(model, points, execution):
        solution = states(model, batch, execution=options)
        energies[:, start : start + len(batch)] = solution.energies[selected] - model.fermi_energy
        for local, point in enumerate(batch):
            k = start + local
            blocks, sizes = _projection_manifolds(
                model, solution.energies[:, local], solution.band_channels, selected
            )
            projected_spin = np.zeros((model.nbnd, 3))
            if hasattr(model, "spin_operators"):
                operators = model.spin_operators(point)
                vectors = solution.coefficients[local]
                projected_spin = 0.5 * np.einsum("in,aij,jn->na", vectors.conj(), operators, vectors).real
            elif getattr(model, "spin_mode", "nonmagnetic") == "collinear":
                projected_spin[:, 2] = np.where(solution.band_channels == 0, 0.5, -0.5)
            spin[:, k] = _mean_manifolds(projected_spin, blocks, 0)[selected]
            manifold_sizes[:, k] = sizes[selected]
    return _result(
        {
            "kpoints": points,
            "energies": energies,
            "spin": spin,
            "band_channels": channels,
            "manifold_sizes": manifold_sizes,
        },
        {
            "kpoints": ("k", "cartesian"),
            "energies": ("band", "k"),
            "spin": ("band", "k", "cartesian"),
            "band_channels": ("band",),
            "manifold_sizes": ("band", "k"),
        },
        {
            "kpoints": "fractional_reciprocal",
            "energies": "eV",
            "spin": "hbar",
            "band_channels": "1",
            "manifold_sizes": "1",
        },
        spin_order="all_up_then_all_down",
        degenerate_state_interpretation="complete_subspace_mean_per_state",
        degeneracy_tolerance_eV=degeneracy_tolerance(model),
    )


def reduce_basis(model, kpoints, energy_range=None, threshold=0.02, bands=None, execution=None):
    positive(threshold, "threshold")
    points = validated_points(kpoints)
    if energy_range is not None:
        bounds = _axis(energy_range, "energy_range")
        if len(bounds) != 2 or bounds[0] > bounds[1]:
            raise ValueError("energy_range requires ordered lower and upper limits")
    scores = np.zeros(model.nbnd)
    selected = _selection(model, bands)
    for _, batch, options in _batches(model, points, execution):
        solution = states(model, batch, execution=options)
        orbital_weights = _mulliken(solution)
        for local in range(len(batch)):
            blocks, _ = _projection_manifolds(
                model, solution.energies[:, local], solution.band_channels, selected
            )
            _mean_manifolds(orbital_weights[:, :, local], blocks, 1)
        values = solution.energies[selected] - model.fermi_energy
        mask = (
            np.ones(values.shape, dtype=bool)
            if energy_range is None
            else ((values >= bounds[0]) & (values <= bounds[1]))
        )
        scores = np.maximum(
            scores,
            np.max(np.where(mask[None], np.abs(orbital_weights[:, selected]), 0), axis=(1, 2), initial=0),
        )
    keep = scores >= threshold
    return _result(
        {
            "scores": scores,
            "keep_mask": keep,
            "keep_indices": np.flatnonzero(keep),
            "remove_indices": np.flatnonzero(~keep),
        },
        {
            "scores": ("orbital",),
            "keep_mask": ("orbital",),
            "keep_indices": ("retained_orbital",),
            "remove_indices": ("removed_orbital",),
        },
        {"scores": "1", "keep_mask": "1", "keep_indices": "1", "remove_indices": "1"},
        diagnostic="maximum_absolute_mulliken_weight",
        threshold=float(threshold),
        degenerate_state_interpretation="complete_subspace_mean_per_state",
        degeneracy_tolerance_eV=degeneracy_tolerance(model),
    )


def find_nodes(
    model,
    bands=None,
    initial_grid=(8, 8, 8),
    threshold=1e-6,
    initial_threshold=0.2,
    energy_range=None,
    max_iterations=300,
    execution=None,
):
    positive(threshold, "threshold")
    positive(initial_threshold, "initial_threshold")
    if isinstance(max_iterations, bool) or int(max_iterations) != max_iterations or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if energy_range is not None:
        energy_range = _axis(energy_range, "energy_range")
        if len(energy_range) != 2 or energy_range[0] > energy_range[1]:
            raise ValueError("energy_range requires ordered lower and upper limits")
    points = kmesh(initial_grid)
    energies = model.energies(points, execution=execution)
    pairs: list[tuple[int, int]] = []
    if bands is None:
        channels = np.asarray(getattr(model, "band_channels", np.zeros(model.nbnd, dtype=int)))
        for channel in np.unique(channels):
            indices = np.flatnonzero(channels == channel)
            pairs.extend(zip(indices[:-1], indices[1:], strict=True))
    else:
        pair = np.asarray(bands)
        if pair.shape != (2,) or pair.dtype.kind not in "iu":
            raise ValueError("bands must specify an ordered integer pair of model indices")
        pairs = [(int(pair[0]), int(pair[1]))]
    if any(len(pair) != 2 or not 0 <= pair[0] < pair[1] < len(energies) for pair in pairs):
        raise ValueError("bands must specify an ordered pair of model indices")
    nodes: list[np.ndarray] = []
    gaps: list[float] = []
    node_pairs: list[tuple[int, int]] = []
    for lower, upper in pairs:
        candidates = np.flatnonzero(np.abs(energies[upper] - energies[lower]) <= initial_threshold)
        for candidate in candidates:

            def objective(point):
                values = model.energies(np.mod(point, 1)[None], execution=execution)[:, 0]
                return float((values[upper] - values[lower]) ** 2)

            solution = minimize(
                objective,
                points[candidate],
                method="Nelder-Mead",
                options={"maxiter": max_iterations, "xatol": threshold * 0.01, "fatol": threshold**2 * 0.01},
            )
            point = np.mod(solution.x, 1)
            values = model.energies(point[None], execution=execution)[:, 0]
            gap = float(abs(values[upper] - values[lower]))
            middle = (values[upper] + values[lower]) / 2
            if (
                gap > threshold
                or energy_range is not None
                and not energy_range[0] <= middle <= energy_range[1]
            ):
                continue
            if any(
                np.linalg.norm((point - old + 0.5) % 1 - 0.5) < 1e-4 and pair == (lower, upper)
                for old, pair in zip(nodes, node_pairs, strict=True)
            ):
                continue
            nodes.append(point)
            gaps.append(gap)
            node_pairs.append((lower, upper))
    return _result(
        {
            "kpoints": np.asarray(nodes).reshape(-1, 3),
            "gaps": np.asarray(gaps),
            "band_pairs": np.asarray(node_pairs, dtype=int).reshape(-1, 2),
        },
        {"kpoints": ("node", "cartesian"), "gaps": ("node",), "band_pairs": ("node", "pair")},
        {"kpoints": "fractional_reciprocal", "gaps": "eV", "band_pairs": "1"},
        threshold=float(threshold),
        search="periodic_gap_minimization",
        exhaustive=False,
    )
