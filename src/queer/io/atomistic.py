"""Native POSCAR, CIF/MCIF, extended XYZ and ASE ULM trajectory formats.

No atomistic engine is required. All in-memory coordinates are Cartesian Å;
trajectory velocities are Å/fs and stresses are eV/Å³.
"""

from __future__ import annotations

import ast
import base64
import io
import json
import math
import re
import shlex
import struct
from pathlib import Path
from typing import Any

import numpy as np

from ..elements import atomic_number
from ..physics.constants import ASE_FS as ASE_FS
from ..structures import Evaluation, Frame, Structure
from ._poscar import parse_poscar_record

ULM_ARRAYS_INFO_KEY = "_queer_atom_arrays"


def parse_poscar(text, *, species=None):
    """Read rich VASP 4/5 geometry, including selective coordinate constraints."""
    try:
        record = parse_poscar_record(text)
        symbols = record.symbols
        if symbols is None:
            symbols = tuple(species or re.findall(r"[A-Z][a-z]?", record.comment))
            if len(symbols) != len(record.counts):
                raise ValueError("VASP 4 POSCAR requires species= or element symbols in its comment")
        numbers = np.repeat([atomic_number(symbol) for symbol in symbols], record.counts)
        fixed = (
            np.array([[flag.lower().startswith("f") for flag in row] for row in record.selective_flags])
            if record.selective_flags is not None
            else None
        )
        return Structure(
            record.cell,
            record.positions,
            numbers,
            fixed_coordinates=fixed,
            metadata={
                "comment": record.comment,
                "source_format": "poscar",
                **({"constraint_basis": "fractional"} if record.selective_flags is not None else {}),
            },
        )
    except (ValueError, IndexError, np.linalg.LinAlgError, ZeroDivisionError) as error:
        raise ValueError(f"Malformed POSCAR: {error}") from error


def write_poscar(structure: Structure, *, direct=True, sort=False, return_order=False):
    """Serialize geometry; optional stable Z sorting returns its atom permutation."""
    fixed = structure.fixed_coordinates
    if fixed is not None:
        basis = structure.metadata.get("constraint_basis", "cartesian")
        if basis == "cartesian":
            converted = []
            tolerance = np.linalg.norm(structure.cell, axis=1) * 1e-10
            for mask in fixed:
                allowed_rows = np.all(np.abs(structure.cell[:, mask]) <= tolerance[:, None], axis=1)
                if int(allowed_rows.sum()) != int((~mask).sum()):
                    raise ValueError(
                        "Cartesian coordinate constraints are incompatible with POSCAR's fractional flags; "
                        "use a trajectory/extXYZ format or an equivalent fractional constraint subspace"
                    )
                converted.append(~allowed_rows)
            fixed = np.asarray(converted, bool).reshape(-1, 3)
        elif basis != "fractional":
            raise ValueError("constraint_basis must be cartesian or fractional")
    order = np.argsort(structure.atomic_numbers, kind="stable") if sort else np.arange(structure.natoms)
    symbols = [structure.symbols[index] for index in order]
    groups: list[str] = []
    counts: list[int] = []
    for symbol in symbols:
        if not groups or groups[-1] != symbol:
            groups.append(symbol)
            counts.append(0)
        counts[-1] += 1
    lines = [str(structure.metadata.get("comment", "QUEER structure")).replace("\n", " "), "1.0"]
    lines += [" ".join(f"{value:.16g}" for value in row) for row in structure.cell]
    lines += [" ".join(groups), " ".join(map(str, counts))]
    if fixed is not None:
        lines.append("Selective dynamics")
    lines.append("Direct" if direct else "Cartesian")
    coordinates = structure.fractional_positions if direct else structure.positions
    for index in order:
        row = " ".join(f"{value:.16g}" for value in coordinates[index])
        if fixed is not None:
            row += " " + " ".join("F" if flag else "T" for flag in fixed[index])
        lines.append(row)
    text = "\n".join(lines) + "\n"
    return (text, order) if return_order else text


def poscar_text(structure: Structure, *, sort=True, direct=True):
    return write_poscar(structure, direct=direct, sort=sort, return_order=True)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _xyz_scalar(value):
    if isinstance(value, (bool, np.bool_)):
        return "T" if value else "F"
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f"{value:.16g}"
    return str(value)


def write_extxyz(frames):
    """Serialize frames and arbitrary per-atom/metadata fields, including REF_* keys."""
    if isinstance(frames, (Structure, Frame)):
        frames = [frames]
    output = []
    for item in frames:
        frame = Frame(item) if isinstance(item, Structure) else item
        atoms = frame.structure
        arrays = {"species": np.array(atoms.symbols), "pos": atoms.positions, **atoms.arrays}
        arrays.setdefault("mass", atoms.masses)
        if atoms.magnetic_moments is not None:
            arrays.setdefault("magmom", atoms.magnetic_moments)
        if atoms.fixed_coordinates is not None:
            arrays.setdefault("fixed_coordinates", atoms.fixed_coordinates)
        if frame.velocities is not None:
            arrays["velocities"] = frame.velocities
        metadata = {**atoms.metadata, **frame.metadata}
        if frame.step is not None:
            metadata["step"] = frame.step
        if frame.time_fs is not None:
            metadata["time_fs"] = frame.time_fs
        if frame.evaluation is not None:
            metadata["energy"] = frame.evaluation.energy_eV
            arrays["forces"] = frame.evaluation.forces_eV_A
            if frame.evaluation.stress_eV_A3 is not None:
                metadata["stress"] = frame.evaluation.stress_eV_A3
        descriptors: list[str] = []
        columns = []
        for name, values in arrays.items():
            values = np.asarray(values)
            if values.ndim > 2 or values.shape[0] != atoms.natoms:
                raise ValueError(f"extxyz field {name!r} must be scalar/vector per atom")
            width = 1 if values.ndim == 1 else values.shape[1]
            kind = (
                "L"
                if values.dtype.kind == "b"
                else "I"
                if values.dtype.kind in "iu"
                else "R"
                if values.dtype.kind in "fc"
                else "S"
            )
            if values.dtype.kind == "c":
                raise ValueError("extxyz does not support complex atom properties")
            descriptors.extend((name, kind, str(width)))
            columns.append(values.reshape(atoms.natoms, width))
        header = [
            "Lattice=" + json.dumps(" ".join(_xyz_scalar(value) for value in atoms.cell.ravel())),
            "Properties=" + ":".join(descriptors),
            "pbc=" + json.dumps(" ".join(_xyz_scalar(value) for value in atoms.pbc)),
        ]
        for name, value in metadata.items():
            if name in {"Lattice", "Properties", "pbc"}:
                continue
            if not re.fullmatch(r"[A-Za-z_][\w.-]*", str(name)):
                raise ValueError(f"Invalid extxyz metadata key {name!r}")
            array = np.asarray(value) if isinstance(value, (np.ndarray, list, tuple)) else None
            # extXYZ reserves column-major tensors only for stress and virial.
            # Arbitrary tensors (including MACE's REF_stress) use _JSON so
            # independent readers retain their shape and Cartesian components.
            if array is not None and array.shape == (3, 3) and name in {"stress", "virial"}:
                encoded = json.dumps(" ".join(_xyz_scalar(v) for v in array.ravel(order="F")))
            elif isinstance(value, (bool, int, float, np.generic)):
                encoded = _xyz_scalar(value)
            elif isinstance(value, str):
                encoded = json.dumps(value)
            else:
                encoded = json.dumps("_JSON " + json.dumps(_jsonable(value), separators=(",", ":")))
            header.append(f"{name}={encoded}")
        output += [str(atoms.natoms), " ".join(header)]
        for index in range(atoms.natoms):
            output.append(
                " ".join(shlex.quote(_xyz_scalar(value)) for column in columns for value in column[index])
            )
    return "\n".join(output) + ("\n" if output else "")


def _header_value(text):
    if text.startswith("_JSON "):
        return json.loads(text[6:])
    fields = text.split()
    values: list[Any]
    if all(value in {"T", "F", "True", "False"} for value in fields):
        values = [value in {"T", "True"} for value in fields]
    else:
        try:
            values = [float(value) for value in fields]
        except ValueError:
            return text
    return values[0] if len(values) == 1 else values


def read_extxyz(text):
    lines, index, frames = str(text).splitlines(), 0, []
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        try:
            n = int(lines[index])
            tokens = shlex.split(lines[index + 1])
            header = {
                token.split("=", 1)[0]: _header_value(token.split("=", 1)[1])
                for token in tokens
                if "=" in token
            }
            properties = str(header.pop("Properties", "species:S:1:pos:R:3")).split(":")
            if len(properties) % 3:
                raise ValueError("Properties must contain name:type:width triples")
            rows = [shlex.split(line) for line in lines[index + 2 : index + 2 + n]]
            if len(rows) != n:
                raise ValueError("truncated frame")
            arrays, cursor = {}, 0
            for name, kind, raw_width in zip(properties[::3], properties[1::3], properties[2::3]):
                width = int(raw_width)
                if width < 1 or any(len(row) < cursor + width for row in rows):
                    raise ValueError("truncated atom property")
                values = [row[cursor : cursor + width] for row in rows]
                dtype = {"R": float, "I": int, "S": str, "L": bool}.get(kind)
                if dtype is None:
                    raise ValueError(f"Unknown property type {kind!r}")
                array = (
                    np.array([[v in {"T", "True", "1"} for v in row] for row in values], bool)
                    if kind == "L"
                    else np.array(values, dtype=dtype)
                )
                arrays[name] = array[:, 0] if width == 1 else array
                cursor += width
            cell = np.asarray(header.pop("Lattice", np.zeros(9)), float).reshape(3, 3)
            pbc = header.pop("pbc", (np.linalg.det(cell) != 0,) * 3)
            numbers = arrays.pop("Z", None)
            symbols = arrays.pop("species", None)
            if numbers is None:
                if symbols is None:
                    raise ValueError("atom properties require species or Z")
                numbers = np.asarray([atomic_number(symbol) for symbol in symbols])
            positions = arrays.pop("pos", arrays.pop("positions", None))
            if positions is None:
                raise ValueError("atom properties require Cartesian positions")
            for name in ("stress", "virial", "REF_stress"):
                if name in header:
                    header[name] = np.asarray(header[name]).reshape(3, 3, order="F")
            structure = Structure(
                cell,
                positions,
                numbers,
                pbc=pbc,
                masses=arrays.pop("mass", arrays.pop("masses", None)),
                magnetic_moments=arrays.pop("magmom", None),
                fixed_coordinates=arrays.pop("fixed_coordinates", None),
                arrays=arrays,
                metadata=header,
            )
            energy = header.get("energy", header.get("REF_energy"))
            forces = arrays.get("forces", arrays.get("REF_forces"))
            evaluation = (
                Evaluation(energy, forces, header.get("stress", header.get("REF_stress")))
                if energy is not None and forces is not None
                else None
            )
            frames.append(
                Frame(
                    structure,
                    velocities=arrays.get("velocities"),
                    step=header.get("step"),
                    time_fs=header.get("time_fs"),
                    evaluation=evaluation,
                )
            )
            index += n + 2
        except (ValueError, IndexError, TypeError, np.linalg.LinAlgError) as error:
            raise ValueError(f"Malformed extxyz at line {index + 1}: {error}") from error
    return frames


def _cif_tokens(text):
    class Token(str):
        quoted: bool

        def __new__(cls, value, quoted=False):
            result = super().__new__(cls, value)
            result.quoted = quoted
            return result

    text, tokens, index = str(text), [], 0
    while index < len(text):
        if text[index].isspace():
            index += 1
        elif text[index] == "#":
            newline = text.find("\n", index)
            index = len(text) if newline < 0 else newline + 1
        elif text[index] == ";" and (index == 0 or text[index - 1] == "\n"):
            end = text.find("\n;", index + 1)
            if end < 0:
                raise ValueError("Unterminated CIF semicolon string")
            tokens.append(Token(text[index + 1 : end], True))
            index = end + 2
        elif text[index] in "\"'":
            quote, start = text[index], index + 1
            end = start
            while end < len(text):
                if text[end] == quote and (end + 1 == len(text) or text[end + 1].isspace()):
                    break
                end += 1
            if end == len(text):
                raise ValueError("Unterminated CIF quoted string")
            tokens.append(Token(text[start:end], True))
            index = end + 1
        else:
            end = index
            while end < len(text) and not text[end].isspace():
                end += 1
            tokens.append(Token(text[index:end]))
            index = end
    return tokens


def _cif_fields(text):
    tokens, index = _cif_tokens(text), 0
    fields: dict[str, Any] = {}
    while index < len(tokens):
        token = tokens[index]
        if not token.quoted and token.lower().startswith("data_"):
            if "_atom_site_fract_x" in fields or "_atom_site_cartn_x" in fields:
                break  # A single structure never merges sites across data blocks.
            fields = {}
            index += 1
        elif token.lower() == "loop_" and not token.quoted:
            index += 1
            keys = []
            while index < len(tokens) and tokens[index].startswith("_") and not tokens[index].quoted:
                keys.append(tokens[index].lower().replace(".", "_"))
                index += 1
            if not keys:
                raise ValueError("CIF loop has no column names")
            values = []
            while index < len(tokens) and (
                tokens[index].quoted
                or (
                    not tokens[index].startswith("_")
                    and tokens[index].lower() not in {"loop_", "stop_"}
                    and not tokens[index].lower().startswith(("data_", "save_"))
                )
            ):
                values.append(tokens[index])
                index += 1
            if len(values) % len(keys):
                raise ValueError("CIF loop row does not match its columns")
            fields.update({key: values[column :: len(keys)] for column, key in enumerate(keys)})
        elif token.startswith("_"):
            if index + 1 >= len(tokens):
                raise ValueError("CIF field has no value")
            fields[token.lower().replace(".", "_")] = tokens[index + 1]
            index += 2
        else:
            index += 1
    return fields


def _number(value):
    if value in {"?", "."}:
        raise ValueError("CIF numeric value is missing")
    return float(re.sub(r"\(\d+\)$", "", str(value)).replace("D", "E").replace("d", "e"))


def _expression(expression, variables):
    """Small arithmetic grammar for symmetry operations and tabulated formulas."""
    node = ast.parse(expression.replace("^", "**"), mode="eval").body

    def visit(item):
        if isinstance(item, ast.Constant) and isinstance(item.value, (int, float)):
            return item.value
        if isinstance(item, ast.Name) and item.id in variables:
            return variables[item.id]
        if isinstance(item, ast.UnaryOp) and isinstance(item.op, (ast.UAdd, ast.USub)):
            return visit(item.operand) * (-1 if isinstance(item.op, ast.USub) else 1)
        if isinstance(item, ast.BinOp):
            left, right = visit(item.left), visit(item.right)
            if isinstance(item.op, ast.Add):
                return left + right
            if isinstance(item.op, ast.Sub):
                return left - right
            if isinstance(item.op, ast.Mult):
                return left * right
            if isinstance(item.op, ast.Div):
                return left / right
            if isinstance(item.op, ast.Pow):
                return left**right
        raise ValueError(f"Unsupported arithmetic expression {expression!r}")

    return visit(node)


def _symmetry_operation(text):
    expressions = [part.strip() for part in str(text).split(",")]
    if len(expressions) not in {3, 4}:
        raise ValueError("Only three-dimensional CIF symmetry operations are supported")

    def affine(expression):
        # Symmetry operations are affine, unlike the trusted HPKOT formulas.
        # Carry [x,y,z,constant] coefficients instead of estimating them from
        # sample points, which would misidentify nonlinear expressions.
        def visit(node):
            if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
                return np.array([0, 0, 0, node.value], float)
            if isinstance(node, ast.Name) and node.id in {"x", "y", "z"}:
                return np.eye(4)[("x", "y", "z").index(node.id)]
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                return visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
            if isinstance(node, ast.BinOp):
                left, right = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    if not np.any(left[:3]):
                        return left[3] * right
                    if not np.any(right[:3]):
                        return right[3] * left
                if isinstance(node.op, ast.Div) and not np.any(right[:3]) and right[3] != 0:
                    return left / right[3]
            raise ValueError(f"Unsupported non-affine CIF symmetry expression {expression!r}")

        try:
            return visit(ast.parse(expression, mode="eval").body)
        except SyntaxError as error:
            raise ValueError(f"Invalid CIF symmetry expression {expression!r}") from error

    coefficients = np.array([affine(expr) for expr in expressions[:3]])
    matrix, translation = coefficients[:, :3], coefficients[:, 3]
    if not np.allclose(matrix, np.rint(matrix)) or not np.isclose(abs(np.linalg.det(matrix)), 1):
        raise ValueError("CIF operation must have an integer unimodular rotation")
    reversal = _number(expressions[3]) if len(expressions) == 4 else 1
    if reversal not in {-1, 1}:
        raise ValueError("Magnetic time-reversal sign must be +1 or -1")
    return matrix, translation, reversal


def cell_from_parameters(a, b, c, alpha=90, beta=90, gamma=90):
    alpha, beta, gamma = np.radians([alpha, beta, gamma])
    y = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    z2 = c * c - (c * np.cos(beta)) ** 2 - y * y
    if z2 <= 0:
        raise ValueError("Invalid CIF lattice parameters")
    return np.array(
        [[a, 0, 0], [b * np.cos(gamma), b * np.sin(gamma), 0], [c * np.cos(beta), y, np.sqrt(z2)]]
    )


def parse_cif(text, *, fractional_tolerance=1e-4):
    """Expand ordinary/magnetic crystallographic operations into a full cell.

    Magnetic moments are axial vectors in Cartesian coordinates. Incommensurate
    superspace operations cannot define a finite periodic cell and raise an
    explicit error rather than silently returning an asymmetric unit.
    """
    if not np.isfinite(fractional_tolerance) or fractional_tolerance <= 0:
        raise ValueError("fractional_tolerance must be finite and positive")
    fields = _cif_fields(text)
    cell = cell_from_parameters(
        *[_number(fields[f"_cell_length_{axis}"]) for axis in "abc"],
        *[_number(fields.get(f"_cell_angle_{axis}", 90)) for axis in ("alpha", "beta", "gamma")],
    )
    labels = fields.get("_atom_site_label", [])
    if isinstance(labels, str):
        labels = [labels]
    symbols = []
    for symbol in fields.get("_atom_site_type_symbol", labels):
        match = re.match(r"([A-Z][a-z]?)", symbol)
        if match is None:
            raise ValueError(f"Invalid CIF atom-site symbol {symbol!r}")
        symbols.append(match.group(1))
    if "_atom_site_fract_x" in fields:
        fractional = np.column_stack(
            [[_number(value) for value in fields[f"_atom_site_fract_{axis}"]] for axis in "xyz"]
        )
    elif "_atom_site_cartn_x" in fields:
        positions = np.column_stack(
            [[_number(value) for value in fields[f"_atom_site_cartn_{axis}"]] for axis in "xyz"]
        )
        fractional = positions @ np.linalg.inv(cell)
    else:
        raise ValueError("CIF requires fractional or Cartesian atom-site coordinates")
    if len(fractional) != len(symbols):
        raise ValueError("CIF atom-site columns have inconsistent lengths")
    occupancies = [_number(value) for value in fields.get("_atom_site_occupancy", [1] * len(symbols))]
    unit = cell / np.linalg.norm(cell, axis=1)[:, None]
    moment_labels = fields.get("_atom_site_moment_label", [])
    moments = np.zeros((len(symbols), 3))
    if moment_labels:
        raw = np.column_stack(
            [[_number(value) for value in fields[f"_atom_site_moment_crystalaxis_{axis}"]] for axis in "xyz"]
        )
        for label, value in zip(moment_labels, raw @ unit):
            moments[labels.index(label)] = value
    magnetic = fields.get("_space_group_symop_magn_operation_xyz")
    normal = fields.get("_space_group_symop_operation_xyz", fields.get("_symmetry_equiv_pos_as_xyz"))
    operations = magnetic or normal
    if operations is None:
        number = fields.get("_space_group_it_number", fields.get("_symmetry_int_tables_number"))
        group_name = fields.get("_space_group_name_h-m_alt", fields.get("_symmetry_space_group_name_h-m"))
        if number is None and group_name is None:
            operations = ["x,y,z"]
        else:
            from ..geometry.symmetry import spacegroup_operations

            rotations, translations = spacegroup_operations(
                number=int(_number(number)) if number is not None else None, name=group_name
            )
            operations = [
                (rotation, translation, 1) for rotation, translation in zip(rotations, translations)
            ]
    if isinstance(operations, str):
        operations = [operations]
    parsed = [_symmetry_operation(value) if isinstance(value, str) else value for value in operations]
    centerings = fields.get("_space_group_symop_magn_centering_xyz", ["x,y,z,+1"])
    if isinstance(centerings, str):
        centerings = [centerings]
    combined = []
    for rotation, translation, reversal in parsed:
        for centering in centerings:
            cr, ct, cs = _symmetry_operation(centering)
            combined.append((cr @ rotation, cr @ translation + ct, cs * reversal))
    expanded: list[np.ndarray] = []
    expanded_moments: list[np.ndarray] = []
    expanded_numbers: list[int] = []
    expanded_occupancy: list[float] = []
    for site, symbol in enumerate(symbols):
        number = atomic_number(symbol)
        for rotation, translation, reversal in combined:
            position = (rotation @ fractional[site] + translation) % 1
            position[np.isclose(position, 1, rtol=0, atol=1e-12)] = 0
            cart_rotation = cell.T @ rotation @ np.linalg.inv(cell.T)
            moment = reversal * np.linalg.det(cart_rotation) * (cart_rotation @ moments[site])
            duplicate = next(
                (
                    index
                    for index, (old, z) in enumerate(zip(expanded, expanded_numbers))
                    if z == number and np.linalg.norm((position - old + 0.5) % 1 - 0.5) < fractional_tolerance
                ),
                None,
            )
            if duplicate is None:
                expanded.append(position)
                expanded_numbers.append(number)
                expanded_moments.append(moment)
                expanded_occupancy.append(occupancies[site])
            elif moment_labels and not np.allclose(moment, expanded_moments[duplicate], atol=1e-5):
                raise ValueError("Magnetic symmetry operations give incompatible moments on the same site")
    return Structure(
        cell,
        np.asarray(expanded) @ cell,
        expanded_numbers,
        magnetic_moments=expanded_moments if moment_labels else None,
        arrays={"occupancy": np.asarray(expanded_occupancy)},
        metadata={"source_format": "mcif" if magnetic or moment_labels else "cif"},
    )


def write_cif(structure: Structure):
    from ..geometry.structures import lattice_parameters

    parameters = lattice_parameters(structure.cell)
    lines = ["data_queer", "_space_group_IT_number 1"]
    lines += [f"_cell_length_{axis} {parameters[axis]:.16g}" for axis in "abc"]
    lines += [f"_cell_angle_{axis} {parameters[axis]:.16g}" for axis in ("alpha", "beta", "gamma")]
    lines += [
        "loop_",
        "_space_group_symop_operation_xyz",
        "'x,y,z'",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]
    labels = [f"{symbol}{index + 1}" for index, symbol in enumerate(structure.symbols)]
    for index, (label, symbol, position) in enumerate(
        zip(labels, structure.symbols, structure.fractional_positions)
    ):
        occupancy = structure.arrays.get("occupancy", np.ones(structure.natoms))[index]
        lines.append(
            f"{label} {symbol} " + " ".join(f"{value:.16g}" for value in position) + f" {occupancy:.16g}"
        )
    if structure.magnetic_moments is not None:
        lines += [
            "loop_",
            "_atom_site_moment.label",
            "_atom_site_moment.crystalaxis_x",
            "_atom_site_moment.crystalaxis_y",
            "_atom_site_moment.crystalaxis_z",
        ]
        unit = structure.cell / np.linalg.norm(structure.cell, axis=1)[:, None]
        for label, moment in zip(labels, structure.magnetic_moments @ np.linalg.inv(unit)):
            lines.append(label + " " + " ".join(f"{value:.16g}" for value in moment))
    return "\n".join(lines) + "\n"


def read_trajectory(data):
    """Read ASE ULM v1–3 bytes without ASE; never unpickle input data."""
    data = Path(data).read_bytes() if isinstance(data, (str, Path)) else bytes(data)
    if len(data) < 48 or data[:8] not in {b"- of Ulm", b"AFFormat"}:
        raise ValueError("Not an ASE ULM trajectory")
    version, count, offset_table = struct.unpack_from("<3q", data, 24)
    if version not in {1, 2, 3} or count < 0 or offset_table < 48 or offset_table + 8 * count > len(data):
        raise ValueError("Invalid ULM header or offsets")
    offsets = struct.unpack_from(f"<{count}q", data, offset_table)

    def decode(item, little=True):
        output = {}
        little = item.get("_little_endian", little)
        for key, value in item.items():
            if key == "_little_endian":
                continue
            if key.endswith("."):
                key = key[:-1]
                if "ndarray" in value:
                    shape, dtype, offset = value["ndarray"]
                    dtype = np.dtype(dtype).newbyteorder("<" if little else ">")
                    if dtype.hasobject or any(type(size) is not int or size < 0 for size in shape):
                        raise ValueError("Unsafe ULM array dtype/shape")
                    count = math.prod(shape)
                    length = count * dtype.itemsize
                    if type(offset) is not int or offset < 0 or offset + length > len(data):
                        raise ValueError("Truncated ULM array")
                    output[key] = (
                        np.frombuffer(data, dtype=dtype, count=count, offset=offset)
                        .reshape(shape)
                        .astype(dtype.newbyteorder("="), copy=True)
                    )
                else:
                    output[key] = decode(value, little)
            else:
                output[key] = value
        return output

    frames, header = [], {}
    for offset in offsets:
        if offset < 0 or offset + 8 > len(data):
            raise ValueError("Invalid ULM record offset")
        length = struct.unpack_from("<q", data, offset)[0]
        if length < 0 or offset + 8 + length > len(data):
            raise ValueError("Truncated ULM JSON record")
        record = decode(json.loads(data[offset + 8 : offset + 8 + length]))
        if "numbers" in record:
            header = {
                key: record[key] for key in ("numbers", "pbc", "masses", "constraints") if key in record
            }
        info = dict(record.get("info", {}))
        native_arrays = {}
        extension = info.pop(ULM_ARRAYS_INFO_KEY, None)
        if extension is not None:
            if not isinstance(extension, dict) or extension.get("schema_version") != 1:
                raise ValueError("Unsupported QUEER trajectory array extension")
            for name, field in extension.get("arrays", {}).items():
                dtype = np.dtype(field["dtype"])
                shape = field["shape"]
                if dtype.hasobject or any(type(size) is not int or size < 0 for size in shape):
                    raise ValueError("Unsafe QUEER trajectory array dtype/shape")
                raw = base64.b64decode(field["data"], validate=True)
                if len(raw) != math.prod(shape) * dtype.itemsize:
                    raise ValueError("Invalid QUEER trajectory array byte count")
                native_arrays[name] = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
        constraints = json.loads(header.get("constraints", "[]"))
        fixed = np.zeros((len(header["numbers"]), 3), bool)
        bases = set()
        for constraint in constraints:
            name, kwargs = constraint.get("name"), constraint.get("kwargs", {})
            if name == "FixAtoms":
                fixed[np.asarray(kwargs.get("indices", []), int)] = True
            elif name in {"FixCartesian", "FixScaled"}:
                fixed[np.atleast_1d(kwargs.get("a", kwargs.get("indices", []))).astype(int)] = kwargs.get(
                    "mask", [True] * 3
                )
                bases.add("fractional" if name == "FixScaled" else "cartesian")
            else:
                raise ValueError(f"Unsupported trajectory constraint {name!r}")
        if len(bases) > 1:
            raise ValueError(
                "Mixed Cartesian/fractional trajectory constraints need an explicit constraint projector"
            )
        if bases:
            info = {**info, "constraint_basis": next(iter(bases))}
        calculator = record.get("calculator", {})
        moments = record.get("magmoms")
        if moments is not None and np.asarray(moments).ndim == 1:
            moments = np.column_stack([np.zeros_like(moments), np.zeros_like(moments), moments])
        atoms = Structure(
            record["cell"],
            record["positions"],
            header["numbers"],
            pbc=header.get("pbc", True),
            masses=header.get("masses"),
            magnetic_moments=moments,
            fixed_coordinates=fixed if constraints else None,
            arrays=native_arrays,
            metadata=info,
        )
        forces, energy = calculator.get("forces"), calculator.get("energy")
        stress = calculator.get("stress")
        if stress is not None and np.asarray(stress).shape == (6,):
            xx, yy, zz, yz, xz, xy = stress
            stress = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
        evaluation = Evaluation(energy, forces, stress) if energy is not None and forces is not None else None
        velocities = (
            np.asarray(record["momenta"]) / atoms.masses[:, None] * ASE_FS if "momenta" in record else None
        )
        frames.append(Frame(atoms, velocities, info.get("step"), info.get("time_fs"), evaluation))
    return frames


def write_trajectory(destination_or_frames, frames=None):
    """Produce interoperable ASE ULM v3 trajectories with complete frame headers."""
    from .. import __version__

    destination = Path(destination_or_frames) if frames is not None else None
    frames = list(destination_or_frames if frames is None else frames)
    buffer = io.BytesIO()
    buffer.write(b"- of Ulm" + b"ASE-Trajectory  " + struct.pack("<3q", 3, len(frames), 48))
    # Tags are exactly sixteen bytes.
    if buffer.tell() != 48:
        raise RuntimeError("invalid trajectory tag length")
    buffer.write(b"\0" * (8 * len(frames)))
    offsets = []

    def encode(item):
        output = {}
        for key, value in item.items():
            if isinstance(value, np.ndarray):
                padding = (-buffer.tell()) % 8
                buffer.write(b"\0" * padding)
                value = np.ascontiguousarray(value.astype(value.dtype.newbyteorder("<")))
                offset = buffer.tell()
                buffer.write(value.tobytes())
                output[key + "."] = {"ndarray": [list(value.shape), value.dtype.name, offset]}
            elif isinstance(value, dict) and key == "calculator":
                output[key + "."] = encode(value)
            else:
                output[key] = _jsonable(value)
        return output

    for frame in frames:
        if isinstance(frame, Structure):
            frame = Frame(frame)
        atoms = frame.structure
        info = {**atoms.metadata, **frame.metadata}
        if ULM_ARRAYS_INFO_KEY in info:
            raise ValueError(f"Trajectory metadata key {ULM_ARRAYS_INFO_KEY!r} is reserved")
        if atoms.arrays:
            packed = {}
            for name, values in atoms.arrays.items():
                values = np.ascontiguousarray(values)
                if values.dtype.hasobject or values.dtype.fields is not None:
                    raise ValueError("Trajectory atom arrays require non-object, unstructured NumPy dtypes")
                packed[name] = {
                    "dtype": values.dtype.str,
                    "shape": list(values.shape),
                    "data": base64.b64encode(values.tobytes()).decode("ascii"),
                }
            info[ULM_ARRAYS_INFO_KEY] = {"schema_version": 1, "arrays": packed}
        if frame.step is not None:
            info["step"] = frame.step
        if frame.time_fs is not None:
            info["time_fs"] = frame.time_fs
        record = {
            "version": 1,
            "queer_producer": {"name": "queer", "version": __version__},
            "pbc": list(atoms.pbc),
            "numbers": atoms.atomic_numbers,
            "masses": atoms.masses,
            "positions": atoms.positions,
            "cell": atoms.cell.tolist(),
            "info": info,
        }
        if atoms.fixed_coordinates is not None:
            constraint_name = (
                "FixScaled" if atoms.metadata.get("constraint_basis") == "fractional" else "FixCartesian"
            )
            record["constraints"] = json.dumps(
                [
                    {"name": constraint_name, "kwargs": {"a": [int(index)], "mask": row.tolist()}}
                    for index, row in enumerate(atoms.fixed_coordinates)
                    if row.any()
                ]
            )
        if atoms.magnetic_moments is not None:
            record["magmoms"] = atoms.magnetic_moments
        if frame.velocities is not None:
            record["momenta"] = frame.velocities * atoms.masses[:, None] / ASE_FS
        if frame.evaluation is not None:
            ev = frame.evaluation
            record["calculator"] = {
                "name": "queer",
                "parameters": {},
                "energy": ev.energy_eV,
                "forces": ev.forces_eV_A,
            }
            if ev.stress_eV_A3 is not None:
                s = ev.stress_eV_A3
                if not np.allclose(s, s.T, rtol=1e-10, atol=1e-12):
                    raise ValueError(
                        "ULM trajectory stress must be symmetric for six-component Voigt storage"
                    )
                record["calculator"]["stress"] = np.array(
                    [s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]]
                )
        encoded = json.dumps(encode(record), separators=(",", ":")).encode()
        offsets.append(buffer.tell())
        buffer.write(struct.pack("<q", len(encoded)) + encoded)
    buffer.seek(48)
    for offset in offsets:
        buffer.write(struct.pack("<q", offset))
    data = buffer.getvalue()
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return destination
    return data


def parse_structure(content, *, format=None, filename="", species=None):
    format = (
        format or Path(filename).suffix.lstrip(".") or ("cif" if "data_" in str(content)[:200] else "poscar")
    ).lower()
    if format in {"vasp", "poscar", "contcar"}:
        return parse_poscar(content, species=species)
    if format in {"cif", "mcif"}:
        return parse_cif(content)
    if format in {"xyz", "extxyz"}:
        frames = read_extxyz(content)
    elif format in {"traj", "ulm"}:
        frames = read_trajectory(content)
    else:
        raise ValueError(f"Unsupported structure format {format!r}")
    if len(frames) != 1:
        raise ValueError("Expected one structure; use read_frames for multi-frame input")
    return frames[0].structure


def read_structure(path, **kwargs):
    path = Path(path)
    format = str(kwargs.get("format") or path.suffix.lstrip(".")).lower()
    content = path.read_bytes() if format in {"traj", "ulm"} else path.read_text()
    return parse_structure(content, filename=str(path), **kwargs)


def read_frames(path, *, format=None):
    path = Path(path)
    format = (format or path.suffix.lstrip(".")).lower()
    if format in {"traj", "ulm"}:
        return read_trajectory(path.read_bytes())
    if format in {"xyz", "extxyz"}:
        return read_extxyz(path.read_text())
    return [Frame(read_structure(path, format=format))]


def write_structure(destination_or_structure, structure=None, *, format=None, **kwargs):
    destination = Path(destination_or_structure) if structure is not None else None
    structure = destination_or_structure if structure is None else structure
    format = (
        format or (destination.suffix.lstrip(".") if destination is not None else "") or "poscar"
    ).lower()
    if format in {"poscar", "vasp", "contcar"}:
        content = write_poscar(structure, **kwargs)
    elif format in {"cif", "mcif"}:
        content = write_cif(structure)
    elif format in {"xyz", "extxyz"}:
        content = write_extxyz(structure)
    elif format in {"traj", "ulm"}:
        content = write_trajectory([Frame(structure)])
    else:
        raise ValueError(f"Unsupported structure format {format!r}")
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            destination.write_bytes(content)
        else:
            destination.write_text(content)
        return destination
    return content
