"""Native HPKOT path construction over an optional Spglib symmetry primitive.

Only the MIT-licensed published point/path/formula tables are reused. Path
classification, evaluation, primitive atom mapping and basis conversion are
implemented here. Fractional points always state their reciprocal cell basis.
"""

from __future__ import annotations

import json
import warnings
from importlib.resources import files
from itertools import product
from typing import Any

import numpy as np

from ..structures import Structure
from .lattice import reciprocal_lattice


def _spglib():
    try:
        import spglib
    except ImportError as error:
        raise ImportError("Symmetry detection requires queer[symmetry] (spglib)") from error
    return spglib


def _cell_tuple(structure):
    if isinstance(structure, Structure):
        return structure.cell, structure.fractional_positions, structure.atomic_numbers
    if isinstance(structure, dict):
        from ..elements import atomic_number

        cell = np.asarray(structure["lattice"], float)
        return (
            cell,
            np.asarray(structure["positions"]) @ np.linalg.inv(cell),
            np.array([atomic_number(s) for s in structure["species"]]),
        )
    cell, fractional, identifiers = structure
    return np.asarray(cell, float), np.asarray(fractional, float), np.asarray(identifiers, int)


def symmetry_dataset(structure, *, symprec=1e-5, angle_tolerance=-1):
    if not np.isfinite(symprec) or symprec <= 0:
        raise ValueError("symprec must be finite and positive")
    dataset = _spglib().get_symmetry_dataset(
        _cell_tuple(structure), symprec=symprec, angle_tolerance=angle_tolerance
    )
    if dataset is None:
        raise ValueError("Could not identify crystallographic symmetry")
    return dataset


def spacegroup_operations(*, number=None, name=None, hall_number=None):
    spglib = _spglib()
    if hall_number is None:
        normalized = re_group_name(name) if name else None
        matches = []
        for hall in range(1, 531):
            group = spglib.get_spacegroup_type(hall)
            if (number is not None and group.number == number) or (
                normalized
                and normalized
                in {
                    re_group_name(group.international_short),
                    re_group_name(group.international_full),
                    re_group_name(group.international),
                }
            ):
                matches.append(hall)
        if not matches:
            raise ValueError(f"Unknown crystallographic space group {number or name!r}")
        hall_number = matches[0]
    result = spglib.get_symmetry_from_database(int(hall_number))
    if result is None:
        raise ValueError("Unknown Hall number")
    return result["rotations"], result["translations"]


def re_group_name(name):
    return str(name).replace(" ", "").replace("_", "").lower()


def primitive_structure(structure: Structure, *, symprec=1e-5):
    """Reduce translationally equivalent sites while retaining all atom fields."""
    spglib = _spglib()
    # Magnetic/constraint/per-atom data are part of site identity: a reduction
    # must not merge an antiferromagnetic or otherwise decorated sublattice.
    identities: list[int] = []
    lookup: dict[tuple, int] = {}
    for index, number in enumerate(structure.atomic_numbers):
        signature: list[Any] = [int(number), round(float(structure.masses[index]), 10)]
        if structure.magnetic_moments is not None:
            signature.extend(np.round(structure.magnetic_moments[index], 8))
        if structure.fixed_coordinates is not None:
            signature.extend(structure.fixed_coordinates[index].astype(int))
        for key in sorted(structure.arrays):
            signature.append(repr(structure.arrays[key][index].tolist()))
        identities.append(lookup.setdefault(tuple(signature), len(lookup) + 1))
    result = spglib.standardize_cell(
        (structure.cell, structure.fractional_positions, identities),
        to_primitive=True,
        no_idealize=True,
        symprec=symprec,
    )
    if result is None:
        raise ValueError("Could not find a primitive structure")
    cell, fractional, types = result
    # Locate the representative in the original decorated motif. Spglib's
    # no_idealize cell has the original Cartesian orientation and origin.
    original_in_new = structure.positions @ np.linalg.inv(cell)
    indices = []
    for point, identity in zip(fractional, types):
        candidates = np.where(np.asarray(identities) == identity)[0]
        differences = original_in_new[candidates] - point
        differences -= np.rint(differences)
        distance = np.linalg.norm(differences @ cell, axis=1)
        best = candidates[np.argmin(distance)]
        if distance.min() > symprec * 4:
            raise ValueError("Primitive atom mapping could not be resolved")
        indices.append(best)
    from .structures import _constraints_for_cell, _take

    reduced = _take(structure, indices)
    return reduced.copy(
        cell=cell,
        positions=np.asarray(fractional) @ cell,
        fixed_coordinates=_constraints_for_cell(reduced, cell),
        metadata=reduced.metadata,
    )


def _family(number):
    for maximum, family in ((2, "a"), (15, "m"), (74, "o"), (142, "t"), (194, "h"), (230, "c")):
        if number <= maximum:
            return family
    raise ValueError("space-group number must be in 1..230")


def _primitive_matrix(bravais):
    centering = bravais[1]
    if centering == "P":
        return np.eye(3)
    if centering == "F":
        return (np.ones((3, 3)) - np.eye(3)) / 2
    if centering == "I":
        return (np.ones((3, 3)) - 2 * np.eye(3)) / 2
    if centering == "R":
        return np.array([[2, -1, -1], [1, 1, -2], [1, 1, 1]]) / 3
    if bravais == "oA":
        return np.array([[0, 0, 2], [1, 1, 0], [-1, 1, 0]]) / 2
    if bravais == "mC":
        return np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2]]) / 2
    if centering == "C":
        return np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 2]]) / 2
    raise ValueError(f"Unsupported standardized centering {bravais!r}")


def _deduplicate(fractional, identifiers, cell, tolerance):
    points: list[np.ndarray] = []
    types: list[int] = []
    mapping: list[int] = []
    for point, identifier in zip(fractional, identifiers):
        point = point % 1
        matches = [
            index
            for index, (previous, z) in enumerate(zip(points, types))
            if z == identifier and np.linalg.norm(((point - previous + 0.5) % 1 - 0.5) @ cell) < tolerance
        ]
        if matches:
            mapping.append(matches[0])
        else:
            mapping.append(len(points))
            points.append(point)
            types.append(int(identifier))
    return np.asarray(points), np.asarray(types), np.asarray(mapping)


def _triclinic_cell(cell, fractional):
    reciprocal = _spglib().niggli_reduce(reciprocal_lattice(cell))
    if reciprocal is None:
        raise ValueError("Reciprocal Niggli reduction failed")
    # HPKOT puts the smallest absolute reciprocal off-diagonal metric at gamma.
    metric = reciprocal @ reciprocal.T
    smallest = np.argmin([abs(metric[1, 2]), abs(metric[2, 0]), abs(metric[0, 1])])
    orders = ((1, 2, 0), (2, 0, 1), (0, 1, 2))
    reciprocal = reciprocal[list(orders[smallest])]
    for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
        candidate = reciprocal * np.asarray(signs)[:, None]
        metric = candidate @ candidate.T
        off = np.array([metric[1, 2], metric[2, 0], metric[0, 1]])
        if np.all(off >= -1e-12) or np.all(off <= 1e-12):
            direct = reciprocal_lattice(candidate)
            new_fractional = fractional @ cell @ np.linalg.inv(direct)
            return direct, new_fractional, "aP2" if np.all(off <= 1e-12) else "aP3"
    raise ValueError("Could not orient reciprocal triclinic cell")


def _variant(bravais, number, lengths, cosine):
    a, b, c = lengths
    beta = cosine[1]
    if bravais in {"cP", "cF"}:
        return bravais + ("1" if number <= 206 else "2")
    if bravais in {"cI", "tP", "oP", "mP"}:
        return bravais + "1"
    if bravais == "tI":
        return "tI1" if c <= a else "tI2"
    if bravais == "oF":
        return "oF1" if 1 / a**2 > 1 / b**2 + 1 / c**2 else "oF2" if 1 / c**2 > 1 / a**2 + 1 / b**2 else "oF3"
    if bravais == "oI":
        return "oI" + str((2, 3, 1)[int(np.argmax(lengths))])
    if bravais == "oC":
        return "oC1" if a <= b else "oC2"
    if bravais == "oA":
        return "oA1" if b <= c else "oA2"
    if bravais == "hR":
        return "hR1" if 3 * a * a <= 2 * c * c else "hR2"
    if bravais == "hP":
        return (
            "hP1"
            if number in {143, 144, 145, 146, 147, 148, 149, 151, 153, 157, 159, 160, 161, 162, 163}
            else "hP2"
        )
    if bravais == "mC":
        return (
            "mC1"
            if b < a * np.sqrt(1 - beta * beta)
            else "mC2"
            if -a * beta / c + a * a * (1 - beta * beta) / b**2 <= 1
            else "mC3"
        )
    raise ValueError(f"Unknown HPKOT Bravais lattice {bravais!r}")


def compute_band_path(
    structure,
    points_per_segment=20,
    with_time_reversal=True,
    *,
    symprec=1e-5,
    angle_tolerance=-1,
    threshold=1e-7,
    recipe="hpkot",
    original_basis=True,
):
    """Return HPKOT segments and primitive metadata, mapped to the input cell.

    Set ``original_basis=False`` to obtain crystallographic primitive reciprocal
    coordinates. Default input-cell coordinates prevent supercell/rotation
    mistakes when evaluating a path against an existing Hamiltonian.
    """
    if recipe != "hpkot":
        raise ValueError("Only the HPKOT path convention is supported")
    if isinstance(points_per_segment, bool) or int(points_per_segment) < 2:
        raise ValueError("points_per_segment must be at least two")
    source_cell, _, _ = _cell_tuple(structure)
    dataset = symmetry_dataset(structure, symprec=symprec, angle_tolerance=angle_tolerance)
    conventional, positions, identifiers = (
        dataset.std_lattice.copy(),
        dataset.std_positions.copy(),
        dataset.std_types.copy(),
    )
    number = int(dataset.number)
    bravais = _family(number) + dataset.international[0]
    if bravais == "aP":
        conventional, positions, variant = _triclinic_cell(conventional, positions)
    else:
        lengths = np.linalg.norm(conventional, axis=1)
        cosine = np.array(
            [
                np.dot(conventional[a], conventional[b]) / (lengths[a] * lengths[b])
                for a, b in ((1, 2), (0, 2), (0, 1))
            ]
        )
        variant = _variant(bravais, number, lengths, cosine)
    lengths = np.linalg.norm(conventional, axis=1)
    cosine = np.array(
        [
            np.dot(conventional[a], conventional[b]) / (lengths[a] * lengths[b])
            for a, b in ((1, 2), (0, 2), (0, 1))
        ]
    )
    matrix = _primitive_matrix(bravais)
    primitive = matrix.T @ conventional
    prim_positions, prim_types, mapping = _deduplicate(
        positions @ np.linalg.inv(matrix).T, identifiers, primitive, symprec * 2
    )
    tables = json.loads(files("queer").joinpath("resources/symmetry/hpkot.json").read_text())
    table = tables[variant]
    a, b, c = lengths
    variables = {
        "a": a,
        "b": b,
        "c": c,
        "cosalpha": cosine[0],
        "cosbeta": cosine[1],
        "cosgamma": cosine[2],
        "sinbeta": np.sqrt(max(0, 1 - cosine[1] ** 2)),
    }
    from ..io.atomistic import _expression

    for name, formula in table["parameters"]:
        variables[name] = _expression(formula, variables)
    coordinates = {
        label: np.array([_expression(expression, variables) for expression in expressions], float)
        for label, expressions in table["points"].items()
    }
    path = [tuple(segment) for segment in table["path"]]
    inversion = any(np.array_equal(rotation, -np.eye(3, dtype=int)) for rotation in dataset.rotations)
    augmented = not inversion and not with_time_reversal
    if augmented:
        for label, point in list(coordinates.items()):
            if label != "GAMMA":
                coordinates[label + "'"] = -point
        path += [(a if a == "GAMMA" else a + "'", b if b == "GAMMA" else b + "'") for a, b in path.copy()]
    primitive_points = {label: point.tolist() for label, point in coordinates.items()}
    primitive_reciprocal = reciprocal_lattice(primitive)
    rotation = np.asarray(dataset.std_rotation_matrix)
    if original_basis:
        conversion = primitive_reciprocal @ rotation @ np.linalg.inv(reciprocal_lattice(source_cell))
        coordinates = {label: point @ conversion for label, point in coordinates.items()}
    if not all(np.isfinite(point).all() for point in coordinates.values()):
        raise ValueError("Degenerate lattice produced nonfinite HPKOT coordinates")
    volume_ratio = abs(np.linalg.det(source_cell) / np.linalg.det(primitive))
    if original_basis and volume_ratio > 1 + threshold:
        warnings.warn(
            "The input cell is nonprimitive; HPKOT points describe the primitive Brillouin zone in the input reciprocal basis",
            RuntimeWarning,
            stacklevel=2,
        )
    from ..elements import SYMBOLS

    return {
        "points_per_segment": int(points_per_segment),
        "segments": [
            {
                "start": coordinates[a].tolist(),
                "end": coordinates[b].tolist(),
                "start_label": a,
                "end_label": b,
            }
            for a, b in path
        ],
        "point_coords": {label: point.tolist() for label, point in coordinates.items()},
        "path": path,
        "primitive_point_coords": primitive_points,
        "primitive_lattice": primitive.tolist(),
        "primitive_positions": prim_positions.tolist(),
        "primitive_types": prim_types.tolist(),
        "primitive_species": [SYMBOLS[int(z)] if 0 < z < len(SYMBOLS) else str(z) for z in prim_types],
        "reciprocal_primitive_lattice": primitive_reciprocal.tolist(),
        "conv_lattice": conventional.tolist(),
        "conv_positions": positions.tolist(),
        "conv_types": identifiers.tolist(),
        "conv_to_primitive": mapping.tolist(),
        "rotation_matrix": rotation.tolist(),
        "spacegroup_international": dataset.international,
        "spacegroup_number": number,
        "has_inversion_symmetry": inversion,
        "augmented_path": augmented,
        "bravais_lattice": bravais,
        "bravais_lattice_extended": variant,
        "coordinate_basis": "original" if original_basis else "primitive",
        "volume_original_wrt_prim": float(volume_ratio),
    }


def brillouin_zone_faces(reciprocal, *, shells=2):
    """Wigner–Seitz faces from reciprocal halfspaces, with complete shell checks."""
    from scipy.spatial import ConvexHull, HalfspaceIntersection

    reciprocal = np.asarray(reciprocal, float)
    if (
        reciprocal.shape != (3, 3)
        or not np.isfinite(reciprocal).all()
        or abs(np.linalg.det(reciprocal)) < 1e-14
    ):
        raise ValueError("reciprocal basis must be nonsingular finite (3,3)")
    # Reduce the basis for a bounded, reliable lattice-vector enumeration.
    basis = reciprocal.copy()
    for _ in range(100):
        previous = basis.copy()
        basis = basis[np.argsort(np.linalg.norm(basis, axis=1))]
        for i in range(1, 3):
            for j in range(i - 1, -1, -1):
                basis[i] -= np.rint(np.dot(basis[i], basis[j]) / np.dot(basis[j], basis[j])) * basis[j]
        if np.allclose(previous, basis, atol=1e-12):
            break
    previous_volume = None
    for extent in range(max(1, int(shells)), 13):
        vectors = (
            np.array([index for index in product(range(-extent, extent + 1), repeat=3) if index != (0, 0, 0)])
            @ basis
        )
        norms = np.linalg.norm(vectors, axis=1)
        halfspaces = np.column_stack([vectors, -(norms**2) / 2])
        vertices = HalfspaceIntersection(halfspaces, np.zeros(3)).intersections
        volume = ConvexHull(vertices).volume
        # Every unenumerated vector has magnitude >= (extent+1)*sigma_min.
        # A face at |G|/2 cannot cut a polytope wholly inside that radius.
        radius = np.linalg.norm(vertices, axis=1).max()
        outside_bound = (extent + 1) * np.linalg.svd(basis, compute_uv=False)[-1]
        if outside_bound > 2 * radius or (
            previous_volume is not None and np.isclose(volume, abs(np.linalg.det(reciprocal)), rtol=1e-9)
        ):
            result = []
            for vector, norm in zip(vectors, norms):
                on = np.abs(vertices @ vector - norm**2 / 2) <= 1e-8 * max(1, norm**2)
                points = vertices[on]
                if len(points) < 3:
                    continue
                normal = vector / norm
                center = points.mean(axis=0)
                u = points[0] - center
                u /= np.linalg.norm(u)
                v = np.cross(normal, u)
                angles = np.arctan2((points - center) @ v, (points - center) @ u)
                result.append((points[np.argsort(angles)], normal))
            return result
        previous_volume = volume
    raise ValueError("Brillouin-zone lattice enumeration did not converge")
