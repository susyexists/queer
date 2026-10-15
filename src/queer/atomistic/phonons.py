"""Native central-displacement force constants and periodic phonon dispersion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..geometry.lattice import reciprocal_lattice
from ..geometry.structures import expand_diagonal
from ..observables.result import ObservableResult
from ..physics.constants import ACCELERATION
from ..structures import Structure
from ._common import evaluate, integer, positive


@dataclass
class DisplacementPlan:
    primitive: Structure
    reference: Structure
    configurations: tuple[Structure, ...]
    entries: tuple[tuple[int, int, int], ...]
    cell_indices: np.ndarray
    supercell: tuple[int, int, int]
    delta_A: float

    @property
    def structures(self):
        """The ordered force-evaluation structures for distributed workflows."""
        return self.configurations


@dataclass
class ForceConstants:
    structure: Structure
    translations: np.ndarray
    values_eV_A2: np.ndarray
    metadata: dict

    @property
    def observable(self):
        return ObservableResult(
            {"force_constants_eV_A2": self.values_eV_A2, "cell_translations": self.translations},
            dims={
                "force_constants_eV_A2": ("cell", "displacement", "force_component"),
                "cell_translations": ("cell", "cartesian"),
            },
            units={"force_constants_eV_A2": "eV/Å²"},
            metadata=self.metadata,
        )


def plan_displacements(structure: Structure, *, supercell=(2, 2, 2), delta_A=0.01):
    """Plan ±δ for every primitive atom/axis, displaced in the origin image.

    Supercell atoms are ordered by image then primitive atom. Force arrays must
    retain that order. The supplied-forces interface permits any force producer
    and independent distributed scheduling of these configurations.
    """
    delta_A = positive(delta_A, "delta_A")
    if structure.natoms < 1:
        raise ValueError("phonons require at least one atom")
    if structure.fixed_coordinates is not None and structure.fixed_coordinates.any():
        raise ValueError("periodic phonons require unconstrained displacement coordinates")
    reference, indices = expand_diagonal(structure, supercell)
    configurations, entries = [], []
    for atom in range(structure.natoms):
        for axis in range(3):
            for sign in (-1, 1):
                positions = reference.positions.copy()
                positions[atom, axis] += sign * delta_A
                configurations.append(reference.copy(positions=positions))
                entries.append((atom, axis, sign))
    return DisplacementPlan(
        structure.copy(),
        reference,
        tuple(configurations),
        tuple(entries),
        indices,
        (int(supercell[0]), int(supercell[1]), int(supercell[2])),
        delta_A,
    )


def assemble_force_constants(
    plan: DisplacementPlan,
    forces,
    *,
    frederiksen=True,
    acoustic=True,
    symmetrize_iterations=3,
) -> ForceConstants:
    """Assemble derivatives from provided forces, with optional sum-rule projection.

    Forces have shape ``(6*Nprimitive, Nsupercell, 3)`` in plan order. Net-force
    correction is applied to the displaced atom, before central differences.
    The acoustic rule is an orthogonal translational projection of the Γ force
    matrix; the correction is placed in the zero-translation block.
    """
    forces = np.asarray(forces, float)
    n, nc = plan.primitive.natoms, len(plan.cell_indices)
    if forces.shape != (len(plan.entries), plan.reference.natoms, 3) or not np.isfinite(forces).all():
        raise ValueError("forces must be finite (n_displacements,n_supercell_atoms,3) in plan order")
    symmetrize_iterations = integer(symmetrize_iterations, "symmetrize_iterations", 0)
    pair_forces = forces.copy().reshape(n, 3, 2, nc * n, 3)
    if frederiksen:
        for atom in range(n):
            pair_forces[atom, :, :, atom, :] -= pair_forces[atom].sum(axis=-2)
    derivatives = (pair_forces[:, :, 0] - pair_forces[:, :, 1]) / (2 * plan.delta_A)
    values = derivatives.reshape(3 * n, nc, 3 * n).transpose(1, 0, 2)
    sc = np.asarray(plan.supercell)
    translations = (plan.cell_indices + sc // 2) % sc - sc // 2
    by_index = {tuple(index): i for i, index in enumerate(plan.cell_indices)}
    inverse = np.asarray([by_index[tuple((-index) % sc)] for index in plan.cell_indices])
    origin = by_index[(0, 0, 0)]
    modes = np.tile(np.eye(3), (n, 1)) / np.sqrt(n)
    projection = np.eye(3 * n) - modes @ modes.T
    for _ in range(max(symmetrize_iterations, int(acoustic))):
        if symmetrize_iterations:
            values = 0.5 * (values + values[inverse].transpose(0, 2, 1))
        if acoustic:
            gamma = values.sum(axis=0)
            values[origin] += projection @ gamma @ projection - gamma
    residual = float(np.linalg.norm(values.sum(axis=0) @ modes))
    metadata = {
        "kind": "force_constants",
        "supercell": list(plan.supercell),
        "delta_A": plan.delta_A,
        "frederiksen": bool(frederiksen),
        "acoustic": bool(acoustic),
        "symmetrize_iterations": symmetrize_iterations,
        "acoustic_residual_eV_A2": residual,
        "atom_order": "cell-major,primitive-atom-minor",
        "force_constant_convention": "C[R,displaced,responding]",
    }
    return ForceConstants(plan.primitive.copy(), translations, values, metadata)


def phonon_dispersion(constants: ForceConstants, qpoints) -> ObservableResult:
    """Return signed frequencies: negative values denote imaginary modes.

    Fractional reciprocal q points retain the input ordering. Eigenvectors are
    mass-weighted and have unit Euclidean norm; frequencies are ordinary cycles
    per second (THz), rather than angular frequencies.
    """
    qpoints = np.asarray(qpoints, float)
    if qpoints.ndim != 2 or qpoints.shape[1] != 3 or not np.isfinite(qpoints).all() or not len(qpoints):
        raise ValueError("qpoints must be a nonempty finite (Nq,3) array")
    n = constants.structure.natoms
    masses: np.ndarray = np.repeat(constants.structure.masses, 3)
    dynamical = constants.values_eV_A2 / np.sqrt(masses[None, :, None] * masses[None, None, :])
    frequencies, eigenvectors, hermitian_errors = [], [], []
    for point in qpoints:
        phase = np.exp(-2j * np.pi * (constants.translations @ point))
        matrix = np.einsum("r,rij->ij", phase, dynamical)
        hermitian_errors.append(float(np.linalg.norm(matrix - matrix.conj().T)))
        # The finite supercell interpolation can have even-image alias errors;
        # report them and diagonalize its physical Hermitian component.
        values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
        frequencies.append(np.sign(values) * np.sqrt(np.abs(values) * ACCELERATION) * 1000 / (2 * np.pi))
        eigenvectors.append(vectors)
    return ObservableResult(
        {
            "qpoints": qpoints,
            "frequencies_THz": np.asarray(frequencies).T,
            "eigenvectors": np.asarray(eigenvectors),
            "force_constants_eV_A2": constants.values_eV_A2,
            "cell_translations": constants.translations,
            "hermitian_residual": np.asarray(hermitian_errors),
        },
        dims={
            "qpoints": ("qpoint", "reciprocal"),
            "frequencies_THz": ("mode", "qpoint"),
            "eigenvectors": ("qpoint", "coordinate", "mode"),
        },
        units={"frequencies_THz": "THz", "force_constants_eV_A2": "eV/Å²"},
        metadata={
            **constants.metadata,
            "kind": "phonon",
            "n_bands": 3 * n,
            "n_qpoints": len(qpoints),
            "imaginary_frequency_convention": "signed-negative",
        },
    )


def _qpath(structure, band_path, npoints):
    segments = (band_path or {}).get("segments", [])
    if not segments:
        from ..geometry.symmetry import compute_band_path

        segments = compute_band_path(structure, points_per_segment=npoints)["segments"]
    reciprocal = reciprocal_lattice(structure.cell)
    qs: list[np.ndarray] = []
    distances: list[float] = []
    metadata: list[dict[str, Any]] = []
    cumulative = 0.0
    for index, segment in enumerate(segments):
        start, end = np.asarray(segment["start"], float), np.asarray(segment["end"], float)
        if start.shape != (3,) or end.shape != (3,) or not np.isfinite([start, end]).all():
            raise ValueError("phonon path endpoints must be finite three-vectors")
        points = np.linspace(start, end, npoints)
        length = float(np.linalg.norm((end - start) @ reciprocal))
        qs.extend(points)
        distances.extend(cumulative + np.linspace(0, length, npoints))
        metadata.append(
            {
                "segment": index,
                "start": start.tolist(),
                "end": end.tolist(),
                "start_label": str(segment.get("start_label", "")),
                "end_label": str(segment.get("end_label", "")),
                "n_points": npoints,
                "start_index": index * npoints,
                "end_index": (index + 1) * npoints - 1,
                "cumulative_at_start": cumulative,
                "cumulative_at_end": cumulative + length,
            }
        )
        cumulative += length
    return np.asarray(qs), np.asarray(distances), metadata


def phonon_bands(
    structure: Structure,
    provider=None,
    *,
    forces=None,
    supercell=(2, 2, 2),
    delta_A=0.01,
    band_path=None,
    qpoints=None,
    n_qpoints_per_segment=50,
    frederiksen=True,
    acoustic=True,
    symmetrize_iterations=3,
) -> ObservableResult:
    """Compute phonon bands from a force callback or independently supplied forces."""
    if (provider is None) == (forces is None):
        raise ValueError("provide exactly one force producer or provided-forces array")
    plan = plan_displacements(structure, supercell=supercell, delta_A=delta_A)
    if forces is None:
        forces = np.asarray(
            [evaluate(provider, configuration).forces_eV_A for configuration in plan.configurations]
        )
    constants = assemble_force_constants(
        plan, forces, frederiksen=frederiksen, acoustic=acoustic, symmetrize_iterations=symmetrize_iterations
    )
    npoints = integer(n_qpoints_per_segment, "n_qpoints_per_segment", 2)
    segments = []
    if qpoints is None:
        qpoints, distances, segments = _qpath(structure, band_path, npoints)
    else:
        qpoints = np.asarray(qpoints, float)
        reciprocal = reciprocal_lattice(structure.cell)
        distances = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(qpoints, axis=0) @ reciprocal, axis=1)))
        )
    result = phonon_dispersion(constants, qpoints)
    summary = {
        "kind": "phonon",
        "parameters": {
            "supercell": list(plan.supercell),
            "delta": plan.delta_A,
            "n_qpoints_per_segment": npoints,
        },
        "segments": segments,
        "n_bands": structure.natoms * 3,
        "n_qpoints": len(qpoints),
        "n_qpoints_per_segment": npoints,
        "converged": True,
        "acoustic_residual_eV_A2": constants.metadata["acoustic_residual_eV_A2"],
    }
    return ObservableResult(
        {**result.arrays, "q_distance": distances},
        dims=result.dims,
        units={**result.units, "q_distance": "Å⁻¹"},
        metadata={**result.metadata, "segments": segments, "summary": summary},
    )
