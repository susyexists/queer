"""Pickle-free structure/frame records shared by workflows and exporters."""

from __future__ import annotations

import json

import numpy as np

from ..structures import Evaluation, Frame, Structure
from .artifacts import canonical_json


def record_units(arrays):
    """Unit annotations for native structure, trajectory and graph fields."""
    units = {}
    for key in arrays:
        if key.endswith(("positions", "cells", "cell")):
            units[key] = "Å"
        elif key.endswith("velocities"):
            units[key] = "Å/fs"
        elif key.endswith("times_fs"):
            units[key] = "fs"
        elif key.endswith(("energy_eV", "hamiltonian", "h0")):
            units[key] = "eV"
        elif key.endswith("forces_eV_A"):
            units[key] = "eV/Å"
        elif key.endswith("stress_eV_A3"):
            units[key] = "eV/Å³"
        elif key.endswith("masses"):
            units[key] = "amu"
        elif key.endswith("magnetic_moments"):
            units[key] = "µB"
        elif key.endswith(
            (
                "atomic_numbers",
                "pbc",
                "fixed_coordinates",
                "overlap",
                "orbital_counts",
                "orbital_slots",
                "edge_index",
                "cell_shift",
                "inverse_edges",
                "steps",
            )
        ):
            units[key] = "1"
    return units


def structure_arrays(structure, *, prefix=""):
    result = {
        prefix + name: np.asarray(getattr(structure, name))
        for name in ("cell", "positions", "atomic_numbers", "pbc", "masses")
    }
    for name in ("magnetic_moments", "fixed_coordinates"):
        value = getattr(structure, name)
        if value is not None:
            result[prefix + name] = np.asarray(value)
    for name, values in structure.arrays.items():
        if values.dtype.hasobject:
            raise ValueError("Native per-atom records cannot contain Python objects")
        result[prefix + "atom_array_" + name] = values
    result[prefix + "structure_metadata_json"] = np.asarray(canonical_json(structure.metadata))
    return result


def structure_from_arrays(arrays, *, prefix=""):
    required = {name: arrays[prefix + name] for name in ("cell", "positions", "atomic_numbers")}
    optional = {
        name: arrays[prefix + name]
        for name in ("pbc", "masses", "magnetic_moments", "fixed_coordinates")
        if prefix + name in arrays
    }
    stem = prefix + "atom_array_"
    per_atom = {name[len(stem) :]: value for name, value in arrays.items() if name.startswith(stem)}
    raw = arrays.get(prefix + "structure_metadata_json")
    metadata = {} if raw is None else json.loads(str(raw))
    return Structure(**required, **optional, arrays=per_atom, metadata=metadata)


def frames_arrays(frames):
    """Encode a fixed-composition atomistic trajectory with explicit missing masks."""
    frames = tuple(frames)
    if not frames:
        return {}
    first = frames[0].structure
    if any(not np.array_equal(f.structure.atomic_numbers, first.atomic_numbers) for f in frames):
        raise ValueError("Workflow trajectories require fixed composition and atom order")
    n = first.natoms
    result = structure_arrays(first, prefix="trajectory_")
    result.update(
        {
            "trajectory_positions": np.asarray([f.structure.positions for f in frames]),
            "trajectory_cells": np.asarray([f.structure.cell for f in frames]),
            "trajectory_steps": np.asarray([-1 if f.step is None else f.step for f in frames], np.int64),
            "trajectory_times_fs": np.asarray([np.nan if f.time_fs is None else f.time_fs for f in frames]),
            "trajectory_velocities": np.asarray(
                [np.full((n, 3), np.nan) if f.velocities is None else f.velocities for f in frames]
            ),
            "trajectory_has_velocities": np.asarray([f.velocities is not None for f in frames]),
            "trajectory_energy_eV": np.asarray(
                [np.nan if f.evaluation is None else f.evaluation.energy_eV for f in frames]
            ),
            "trajectory_forces_eV_A": np.asarray(
                [
                    np.full((n, 3), np.nan) if f.evaluation is None else f.evaluation.forces_eV_A
                    for f in frames
                ]
            ),
            "trajectory_has_evaluation": np.asarray([f.evaluation is not None for f in frames]),
            "trajectory_stress_eV_A3": np.asarray(
                [
                    np.full((3, 3), np.nan)
                    if f.evaluation is None or f.evaluation.stress_eV_A3 is None
                    else f.evaluation.stress_eV_A3
                    for f in frames
                ]
            ),
            "trajectory_frame_metadata_json": np.asarray([canonical_json(f.metadata) for f in frames]),
        }
    )
    return result


def frames_from_arrays(arrays):
    if "trajectory_cells" not in arrays:
        return ()
    values = dict(arrays)
    # Base composition/constraints are static; positions and cells are per frame.
    values["trajectory_positions"] = arrays["trajectory_positions"][0]
    base = structure_from_arrays(values, prefix="trajectory_")
    output = []
    for i, positions in enumerate(arrays["trajectory_positions"]):
        structure = base.copy(positions=positions, cell=arrays["trajectory_cells"][i])
        step, time = arrays["trajectory_steps"][i], arrays["trajectory_times_fs"][i]
        evaluation = None
        if arrays["trajectory_has_evaluation"][i]:
            stress = arrays["trajectory_stress_eV_A3"][i]
            evaluation = Evaluation(
                arrays["trajectory_energy_eV"][i],
                arrays["trajectory_forces_eV_A"][i],
                stress if np.isfinite(stress).all() else None,
            )
        velocities = arrays["trajectory_velocities"][i] if arrays["trajectory_has_velocities"][i] else None
        output.append(
            Frame(
                structure,
                velocities=velocities,
                step=None if step < 0 else int(step),
                time_fs=None if not np.isfinite(time) else float(time),
                evaluation=evaluation,
                metadata=json.loads(str(arrays["trajectory_frame_metadata_json"][i])),
            )
        )
    return tuple(output)
