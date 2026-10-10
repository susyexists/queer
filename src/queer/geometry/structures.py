"""Native periodic neighbors, structure transforms and variant recipes."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

from ..elements import COVALENT_RADII, atomic_number
from ..structures import Structure


def neighbor_list(structure: Structure, cutoff, *, self_interaction=False):
    """Directed neighbors (i,j,image,distance,vector), including periodic self images.

    ``vector = positions[j] + image @ cell - positions[i]``. A scalar cutoff
    is a pair distance; a length-N array supplies additive per-atom radii.
    All periodic images inside the cutoff are returned, even in skew cells.
    """
    n = structure.natoms
    radii = np.asarray(cutoff, float)
    pair_cutoff: float | np.ndarray
    if radii.ndim == 0:
        maximum = float(radii)
        pair_cutoff = maximum
    elif radii.shape == (n,):
        pair_cutoff = radii[:, None] + radii[None, :]
        maximum = 2 * float(radii.max(initial=0))
    else:
        raise ValueError("cutoff must be scalar or one nonnegative radius per atom")
    if not np.isfinite(radii).all() or np.any(radii < 0):
        raise ValueError("cutoffs must be finite and nonnegative")
    if n == 0 or maximum == 0:
        return np.empty(0, int), np.empty(0, int), np.empty((0, 3), int), np.empty(0), np.empty((0, 3))
    positions = structure.positions.copy()
    wraps = np.zeros((n, 3), int)
    bounds = np.zeros(3, int)
    if any(structure.pbc):
        inverse = np.linalg.inv(structure.cell)
        fractional = positions @ inverse
        for axis, periodic in enumerate(structure.pbc):
            if periodic:
                wraps[:, axis] = np.floor(fractional[:, axis]).astype(int)
                bounds[axis] = int(np.ceil(maximum * np.linalg.norm(inverse[:, axis]))) + 1
        positions -= wraps @ structure.cell
    delta = positions[None, :, :] - positions[:, None, :]
    chunks = []
    for image in product(*(range(-bound, bound + 1) for bound in bounds)):
        vector = delta + np.asarray(image) @ structure.cell
        distances = np.linalg.norm(vector, axis=-1)
        keep = distances <= pair_cutoff + 1e-12
        if not self_interaction and image == (0, 0, 0):
            np.fill_diagonal(keep, False)
        i, j = np.where(keep)
        if len(i):
            offsets = np.asarray(image, int)[None, :] + wraps[i] - wraps[j]
            chunks.append((i, j, offsets, distances[i, j], vector[i, j]))
    if not chunks:
        return np.empty(0, int), np.empty(0, int), np.empty((0, 3), int), np.empty(0), np.empty((0, 3))
    arrays = [np.concatenate([chunk[index] for chunk in chunks]) for index in range(5)]
    i, j, offsets, _, _ = arrays
    order = np.lexsort((offsets[:, 2], offsets[:, 1], offsets[:, 0], j, i))
    return tuple(array[order] for array in arrays)


def minimum_image_vectors(vectors, cell, pbc=(True, True, True)):
    """Exact nearest images by bounded lattice search, including triclinic cells."""
    vectors = np.asarray(vectors, float)
    original_shape = vectors.shape
    if original_shape[-1:] != (3,) or not np.isfinite(vectors).all():
        raise ValueError("vectors must be finite with trailing dimension three")
    if not any(pbc):
        return vectors.copy()
    cell = np.asarray(cell, float)
    inverse = np.linalg.inv(cell)
    output = []
    for vector in vectors.reshape(-1, 3):
        fractional = vector @ inverse
        shift = np.array([int(np.rint(value)) if periodic else 0 for value, periodic in zip(fractional, pbc)])
        best = vector - shift @ cell
        radius = np.linalg.norm(best)
        limits = radius * np.linalg.norm(inverse, axis=0) + 1e-12
        ranges = [
            range(int(np.ceil(f - bound)), int(np.floor(f + bound)) + 1) if periodic else [0]
            for f, bound, periodic in zip(fractional, limits, pbc)
        ]
        for candidate in product(*ranges):
            current = vector - np.asarray(candidate) @ cell
            distance = np.linalg.norm(current)
            if distance < radius - 1e-13:
                best, radius = current, distance
        output.append(best)
    return np.asarray(output).reshape(original_shape)


def minimum_distance(structure: Structure, *, include_self_images=True) -> float:
    if structure.natoms == 0:
        return 0.0
    best = float("inf")
    if structure.natoms > 1:
        i, j = np.triu_indices(structure.natoms, 1)
        differences = minimum_image_vectors(
            structure.positions[j] - structure.positions[i], structure.cell, structure.pbc
        )
        best = float(np.linalg.norm(differences, axis=1).min())
    if include_self_images and any(structure.pbc):
        cutoff = min(
            np.linalg.norm(structure.cell[axis]) for axis, periodic in enumerate(structure.pbc) if periodic
        )
        singleton = Structure(structure.cell, [[0, 0, 0]], [1], pbc=structure.pbc)
        _, _, _, distances, _ = neighbor_list(singleton, cutoff)
        if len(distances):
            best = min(best, float(distances.min()))
    return 0.0 if not np.isfinite(best) else best


def compute_bonds(structure: Structure, *, scale=1.15, max_distance=3.6):
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("bond scale and max_distance must be finite and positive")
    radii = COVALENT_RADII[structure.atomic_numbers] * scale
    i, j, offsets, distances, _ = neighbor_list(structure, radii)
    result, seen = [], set()
    for a, b, image, distance in zip(i, j, offsets, distances):
        if distance <= 1e-6 or distance > max_distance:
            continue
        if a > b or (a == b and tuple(image) < tuple(-image)):
            a, b, image = b, a, -image
        key = (int(a), int(b), *map(int, image))
        if key not in seen:
            seen.add(key)
            result.append(
                {"i": int(a), "j": int(b), "distance": round(float(distance), 4), "offset": image.tolist()}
            )
    return sorted(result, key=lambda bond: (bond["i"], bond["j"], bond["offset"]))


def min_neighbor_distance_per_species(structure: Structure, *, cutoff=6.0):
    i, _, _, distances, _ = neighbor_list(structure, cutoff)
    result: dict[str, float] = {}
    for index, distance in zip(i, distances):
        if distance > 1e-6:
            symbol = structure.symbols[index]
            result[symbol] = min(result.get(symbol, float("inf")), float(distance))
    return result


def _take(structure, indices, **changes):
    indices = np.asarray(indices, int)
    return structure.copy(
        positions=structure.positions[indices],
        atomic_numbers=structure.atomic_numbers[indices],
        masses=structure.masses[indices],
        magnetic_moments=None if structure.magnetic_moments is None else structure.magnetic_moments[indices],
        fixed_coordinates=None
        if structure.fixed_coordinates is None
        else structure.fixed_coordinates[indices],
        arrays={key: value[indices] for key, value in structure.arrays.items()},
        metadata={
            **structure.metadata,
            "source_indices": np.asarray(
                structure.metadata.get("source_indices", np.arange(structure.natoms))
            )[indices].tolist(),
        },
        **changes,
    )


def _constraints_for_cell(structure: Structure, cell):
    """Preserve Cartesian tangent subspaces across a lattice basis change."""
    fixed = structure.fixed_coordinates
    if fixed is None or structure.metadata.get("constraint_basis", "cartesian") == "cartesian":
        return fixed
    if structure.metadata.get("constraint_basis") != "fractional":
        raise ValueError("constraint_basis must be cartesian or fractional")
    cell = np.asarray(cell, float)
    mapped = []
    for mask in fixed:
        free = structure.cell[~mask]
        q, _ = np.linalg.qr(free.T, mode="reduced")
        projection = q @ q.T
        residual = np.linalg.norm(cell - cell @ projection, axis=1)
        allowed = residual <= 1e-10 * np.linalg.norm(cell, axis=1) + 1e-12
        if int(allowed.sum()) != len(free):
            raise ValueError(
                "Fractional constraint subspace cannot be represented by coordinate flags in the new cell; "
                "use a compatible cell matrix or explicitly choose a different constraint subspace"
            )
        mapped.append(~allowed)
    return np.asarray(mapped, bool).reshape(-1, 3)


def make_supercell(structure: Structure, matrix):
    matrix = np.asarray(matrix, float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.equal(matrix, np.rint(matrix)).all():
        raise ValueError("supercell matrix must be an integer (3,3) array")
    matrix = matrix.astype(int)
    determinant = int(round(np.linalg.det(matrix)))
    if determinant == 0:
        raise ValueError("supercell matrix must be nonsingular")
    inverse = np.linalg.inv(matrix)
    corners = np.asarray(list(product((0, 1), repeat=3))) @ matrix
    ranges = [
        range(int(np.floor(corners[:, axis].min())) - 1, int(np.ceil(corners[:, axis].max())) + 1)
        for axis in range(3)
    ]
    translations = []
    for translation in product(*ranges):
        frac = np.asarray(translation) @ inverse
        if np.all(frac >= -1e-10) and np.all(frac < 1 - 1e-10):
            translations.append(translation)
    if len(translations) != abs(determinant):
        raise ValueError("Could not enumerate the integer supercell lattice")
    indices = np.tile(np.arange(structure.natoms), len(translations))
    shifts = np.repeat(np.asarray(translations) @ structure.cell, structure.natoms, axis=0)
    result = _take(structure, indices)
    cell = matrix @ structure.cell
    fixed = _constraints_for_cell(structure, cell)
    positions = structure.positions[indices] + shifts
    fractional = positions @ np.linalg.inv(cell)
    for axis, periodic in enumerate(structure.pbc):
        if periodic:
            fractional[:, axis] %= 1
    return result.copy(
        cell=cell,
        positions=fractional @ cell,
        fixed_coordinates=None if fixed is None else fixed[indices],
        metadata={
            **result.metadata,
            "supercell_matrix": matrix.tolist(),
        },
    )


def lattice_parameters(cell):
    if isinstance(cell, Structure):
        cell = cell.cell
    cell = np.asarray(cell, float)
    lengths = np.linalg.norm(cell, axis=1)
    if cell.shape != (3, 3) or np.any(lengths <= 0):
        raise ValueError("lattice parameters need three nonzero row vectors")
    angles = [
        np.degrees(np.arccos(np.clip(np.dot(cell[a], cell[b]) / (lengths[a] * lengths[b]), -1, 1)))
        for a, b in ((1, 2), (0, 2), (0, 1))
    ]
    return dict(zip(("a", "b", "c", "alpha", "beta", "gamma"), map(float, [*lengths, *angles])))


def _sigma(params, index):
    if params.get("includePristine", True) and index == 0:
        return 0.0, True
    low, high = float(params.get("sigmaMin", 0)), float(params.get("sigmaMax", params.get("sigmaMin", 0)))
    if low < 0 or high < low:
        raise ValueError("Gaussian sigma range must be nonnegative and ordered")
    return float(np.random.default_rng([int(params.get("seed", 0)), index]).uniform(low, high)), False


def expand_variants(transforms):
    recipes: list[list[dict[str, Any]]] = [[]]
    for transform in transforms or []:
        kind = transform["type"]
        params: dict[str, Any] = dict(transform.get("params") or {})
        variants = []
        if kind == "strain":
            minimum = float(params.get("min", params.get("percent", params.get("value", 0))))
            values = np.linspace(
                minimum,
                float(params.get("max", minimum)),
                max(1, int(params.get("steps", 1))),
            )
            variants = [{"type": kind, "params": {**params, "percent": float(value)}} for value in values]
        elif kind == "randomDistortion":
            variants = [
                {"type": kind, "params": {**params, "seed": int(params.get("seed", 0)) + index}}
                for index in range(max(1, int(params.get("count", 1))))
            ]
        elif kind == "gaussianDistortion":
            for index in range(max(1, int(params.get("count", 1)))):
                sigma, pristine = _sigma(params, index)
                variants.append(
                    {
                        "type": kind,
                        "params": {**params, "sigma": sigma, "variantIndex": index, "pristine": pristine},
                    }
                )
        else:
            variants = [transform]
        recipes = [recipe + [variant] for recipe in recipes for variant in variants]
    return recipes


def strain(structure: Structure, deformation, *, scale_positions=True):
    """Apply a Cartesian deformation gradient without changing atom identities."""
    deformation = np.asarray(deformation, float)
    if deformation.shape != (3, 3) or not np.isfinite(deformation).all() or np.linalg.det(deformation) <= 0:
        raise ValueError("deformation must be a finite orientation-preserving (3,3) matrix")
    return structure.copy(
        cell=structure.cell @ deformation.T,
        positions=structure.positions @ deformation.T if scale_positions else structure.positions,
    )


def add_vacuum(structure: Structure, vacuum_A: float, *, axis=2, center=True):
    """Extend a cell perpendicular to its other faces, preserving atom distances."""
    if not np.isfinite(vacuum_A) or vacuum_A < 0 or axis not in {0, 1, 2}:
        raise ValueError("vacuum must be finite nonnegative Å and axis in 0..2")
    other = [index for index in range(3) if index != axis]
    normal = np.cross(structure.cell[other[0]], structure.cell[other[1]])
    normal /= np.linalg.norm(normal)
    if np.dot(normal, structure.cell[axis]) < 0:
        normal = -normal
    cell = structure.cell.copy()
    cell[axis] += vacuum_A * normal
    return structure.copy(
        cell=cell,
        positions=structure.positions + (0.5 * vacuum_A * normal if center else 0),
        fixed_coordinates=_constraints_for_cell(structure, cell),
    )


def distort(structure: Structure, amplitude_A: float, *, distribution="gaussian", seed=0, min_distance_A=0):
    if not np.isfinite(amplitude_A) or amplitude_A < 0:
        raise ValueError("distortion amplitude must be finite nonnegative Å")
    rng = np.random.default_rng(seed)
    if distribution == "gaussian":
        shifts = rng.normal(0, amplitude_A, (structure.natoms, 3))
    elif distribution == "uniform":
        shifts = rng.uniform(-amplitude_A, amplitude_A, (structure.natoms, 3))
    else:
        raise ValueError("distribution must be gaussian or uniform")
    result = structure.copy(positions=structure.positions + shifts)
    if min_distance_A > 0 and result.natoms and minimum_distance(result) < min_distance_A:
        raise ValueError("distorted structure violates minimum distance")
    return result


def substitute(structure: Structure, indices, element):
    from ..elements import ATOMIC_MASSES

    number = atomic_number(element) if isinstance(element, str) else int(element)
    indices = np.asarray(indices, int)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= structure.natoms):
        raise ValueError("substitution indices must be valid atom indices")
    numbers, masses = structure.atomic_numbers.copy(), structure.masses.copy()
    numbers[indices], masses[indices] = number, ATOMIC_MASSES[number]
    return structure.copy(atomic_numbers=numbers, masses=masses)


def remove_atoms(structure: Structure, indices):
    indices = np.asarray(indices, int)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= structure.natoms):
        raise ValueError("removed indices must be valid atom indices")
    keep = np.ones(structure.natoms, bool)
    keep[indices] = False
    return _take(structure, np.where(keep)[0])


def apply_transforms(structure: Structure, transforms, *, legacy=False):
    result = structure.copy()
    for transform in transforms or []:
        kind, params = transform["type"], transform.get("params") or {}
        if kind in {"strain", "layer"}:
            cell = result.cell.copy()
            if kind == "strain":
                factor = 1 + float(params.get("percent", params.get("value", params.get("min", 0)))) / 100
                axes = (
                    [{"x": 0, "y": 1, "z": 2}[params.get("axis", "x")]]
                    if params.get("mode", "biaxial") == "uniaxial"
                    else [0, 1]
                )
                cell[axes] *= factor
            else:
                vacuum = float(params.get("vacuumAng", 0))
                if vacuum <= 0:
                    continue
                if not legacy:
                    result = add_vacuum(result, vacuum)
                    continue
                cell[2] *= 1 + vacuum / np.linalg.norm(cell[2])
            result = result.copy(cell=cell, positions=result.fractional_positions @ cell)
        elif kind in {"randomDistortion", "gaussianDistortion"}:
            seed = int(params.get("seed", 0))
            if kind == "randomDistortion":
                amp = float(params.get("amplitude", 0))
                if amp < 0:
                    raise ValueError("distortion amplitude must be nonnegative")
                displacement = np.random.default_rng(seed).uniform(-amp, amp, (result.natoms, 3))
                pristine = False
            else:
                index = int(params.get("variantIndex", 0))
                sigma, pristine = (
                    (float(params["sigma"]), bool(params.get("pristine", params["sigma"] == 0)))
                    if "sigma" in params
                    else _sigma(params, index)
                )
                displacement = (
                    np.zeros((result.natoms, 3))
                    if pristine
                    else np.random.default_rng([seed, index, 1]).normal(0, sigma, (result.natoms, 3))
                )
            result = result.copy(positions=result.positions + displacement)
            threshold = float(params.get("minDistance", 0))
            if (
                not pristine
                and threshold > 0
                and (result.natoms > 1 if legacy else result.natoms > 0)
                and minimum_distance(result, include_self_images=not legacy) < threshold
            ):
                raise ValueError(f"distorted atoms violate minimum distance {threshold} Å")
        elif kind == "supercell":
            matrix = params.get("matrix", np.diag([params.get(axis, 1) for axis in ("a", "b", "c")]))
            result = make_supercell(result, matrix)
        elif kind == "doping":
            source, target = atomic_number(params["elementFrom"]), atomic_number(params["elementTo"])
            fraction = float(params.get("fraction", 0))
            if not 0 <= fraction <= 1:
                raise ValueError("doping fraction must be in [0,1]")
            indices = np.where(result.atomic_numbers == source)[0]
            count = max(1, int(round(fraction * len(indices)))) if fraction > 0 and len(indices) else 0
            numbers, masses = result.atomic_numbers.copy(), result.masses.copy()
            numbers[indices[:count]] = target
            from ..elements import ATOMIC_MASSES

            masses[indices[:count]] = ATOMIC_MASSES[target]
            result = result.copy(atomic_numbers=numbers, masses=masses)
        elif kind == "vacancy":
            fraction = float(params.get("fraction", 0))
            if not 0 <= fraction <= 1:
                raise ValueError("vacancy fraction must be in [0,1]")
            count = max(1, int(round(fraction * result.natoms))) if fraction > 0 else 0
            result = _take(result, np.arange(max(0, result.natoms - count)))
        elif kind == "primitiveCell":
            from .symmetry import primitive_structure

            result = primitive_structure(result, symprec=float(params.get("symprec", 1e-5)))
        else:
            raise ValueError(f"Unknown structure transform {kind!r}")
    return result
