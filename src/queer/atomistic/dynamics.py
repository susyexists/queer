"""Native fixed-cell FIRE and BAOAB Langevin dynamics in Å, eV, amu and fs.

Force evaluation is an explicit producer boundary; no atomistic engine supplies
the integrator. Restart states include the RNG and cadence histories so split
runs reproduce uninterrupted runs.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..geometry.structures import expand_diagonal as _expand
from ..observables.result import ObservableResult
from ..physics.constants import ACCELERATION as ACCELERATION
from ..physics.constants import KB_EV_K as KB_EV_K
from ..structures import Evaluation, Frame, Structure
from ._common import evaluate as _evaluate
from ._common import integer as _integer
from ._common import kinetic_energy
from ._common import positive as _positive

# Generator's 0.01 inverse ASE time converted using its CODATA-2014 time unit.
DEFAULT_FRICTION_FS_INV = 0.0009822694788464063


@dataclass
class SimulationResult:
    frames: tuple[Frame, ...]
    observable: ObservableResult
    state: dict[str, Any]
    final_frame: Frame

    @property
    def summary(self):
        return self.observable.metadata["summary"]


def _free(structure):
    return (
        np.ones((structure.natoms, 3), bool)
        if structure.fixed_coordinates is None
        else ~structure.fixed_coordinates
    )


def _project(velocities, structure, remove_com):
    velocities = np.asarray(velocities).copy()
    free = _free(structure)
    basis = structure.metadata.get("constraint_basis", "cartesian")
    if basis not in ("cartesian", "fractional"):
        raise ValueError("constraint_basis must be cartesian or fractional")
    if basis == "fractional" and structure.fixed_coordinates is not None:
        projectors = _tangent_projectors(structure)
        velocities = np.einsum("nij,nj->ni", projectors, velocities)
        if remove_com and structure.natoms > 1:
            # Solve the mass-weighted constrained COM correction in the
            # allowed Cartesian tangent space. Cartesian component masking
            # would change the fixed fractional coordinates in a skew cell.
            metric = np.sum(structure.masses[:, None, None] * projectors, axis=0)
            momentum = np.sum(structure.masses[:, None] * velocities, axis=0)
            correction = np.linalg.pinv(metric, rcond=1e-12) @ momentum
            velocities -= np.einsum("nij,j->ni", projectors, correction)
        return velocities
    velocities[~free] = 0
    if remove_com:
        for axis in range(3):
            moving = free[:, axis]
            if moving.sum() > 1:
                m = structure.masses[moving]
                velocities[moving, axis] -= np.dot(m, velocities[moving, axis]) / m.sum()
    return velocities


def _tangent_projectors(structure):
    projectors = []
    for free in _free(structure):
        allowed = structure.cell[free]
        if len(allowed) == 0:
            projectors.append(np.zeros((3, 3)))
        else:
            if np.linalg.matrix_rank(allowed) != len(allowed):
                raise ValueError("fractional constraints require independent cell vectors")
            projectors.append(allowed.T @ np.linalg.pinv(allowed @ allowed.T, rcond=1e-12) @ allowed)
    return np.asarray(projectors)


def _remove_rotation(velocities, structure):
    if structure.natoms < 2 or structure.fixed_coordinates is not None:
        return velocities
    m = structure.masses
    r = structure.positions - np.average(structure.positions, axis=0, weights=m)
    inertia = np.sum(
        m[:, None, None] * (np.sum(r * r, axis=1)[:, None, None] * np.eye(3) - r[:, :, None] * r[:, None, :]),
        axis=0,
    )
    angular = np.sum(np.cross(r, m[:, None] * velocities), axis=0)
    omega = np.linalg.pinv(inertia, rcond=1e-12) @ angular
    return velocities - np.cross(omega, r)


def _ndof(structure, remove_com):
    free = _free(structure)
    if structure.metadata.get("constraint_basis") == "fractional" and structure.fixed_coordinates is not None:
        projectors = _tangent_projectors(structure)
        removed = np.linalg.matrix_rank(projectors.sum(axis=0)) if remove_com and structure.natoms > 1 else 0
        return int(free.sum() - removed)
    return int(free.sum() - (sum(free[:, axis].sum() > 1 for axis in range(3)) if remove_com else 0))


def _sample(frame, remove_com):
    evaluation = frame.evaluation
    kinetic = kinetic_energy(frame.structure.masses, frame.velocities)
    dof = _ndof(frame.structure, remove_com)
    energy = evaluation.energy_eV
    return {
        "step": int(frame.step),
        "time_ps": float(frame.time_fs / 1000),
        "epot_eV": energy,
        "ekin_eV": kinetic,
        "etot_eV": energy + kinetic,
        "epot_per_atom_eV": energy / frame.structure.natoms,
        "etot_per_atom_eV": (energy + kinetic) / frame.structure.natoms,
        "temperature_K": 2 * kinetic / (max(dof, 1) * KB_EV_K),
        "max_force_eVA": float(np.linalg.norm(evaluation.forces_eV_A, axis=1).max(initial=0)),
    }


def _frame_record(frame):
    evaluation = frame.evaluation
    return {
        "positions_A": frame.structure.positions.tolist(),
        "velocities_A_fs": None if frame.velocities is None else frame.velocities.tolist(),
        "step": frame.step,
        "time_fs": frame.time_fs,
        "energy_eV": None if evaluation is None else evaluation.energy_eV,
        "forces_eV_A": None if evaluation is None else evaluation.forces_eV_A.tolist(),
        "stress_eV_A3": None
        if evaluation is None or evaluation.stress_eV_A3 is None
        else evaluation.stress_eV_A3.tolist(),
    }


def _record_frame(record, template):
    evaluation = (
        None
        if record["energy_eV"] is None
        else Evaluation(record["energy_eV"], record["forces_eV_A"], record.get("stress_eV_A3"))
    )
    return Frame(
        template.copy(positions=record["positions_A"]),
        record.get("velocities_A_fs"),
        record["step"],
        record.get("time_fs"),
        evaluation,
    )


def frames_from_state(state, structure):
    """Restore typed trajectory frames from a JSON simulation checkpoint."""
    template, _ = _expand(structure, state["identity"]["parameters"].get("supercell", (1, 1, 1)))
    frames = [_record_frame(record, template) for record in state.get("frames", [])]
    current = _record_frame(state["current"], template)
    if not frames or frames[-1].step != current.step:
        frames.append(current)
    return tuple(frames)


def _identity(structure, parameters):
    return {
        "cell": structure.cell.tolist(),
        "atomic_numbers": structure.atomic_numbers.tolist(),
        "masses": structure.masses.tolist(),
        "pbc": list(structure.pbc),
        "fixed_coordinates": _free(structure).tolist(),
        "constraint_basis": structure.metadata.get("constraint_basis", "cartesian"),
        "parameters": parameters,
    }


def _minimum_distance(structure):
    from ..geometry.structures import minimum_distance

    return minimum_distance(structure)


def _nvt_observable(frames, series, final, summary):
    n = final.structure.natoms
    arrays = {
        "positions_A": np.asarray([f.structure.positions for f in frames]).reshape(-1, n, 3),
        "velocities_A_fs": np.asarray([f.velocities for f in frames]).reshape(-1, n, 3),
        "frame_steps": np.asarray([f.step for f in frames], dtype=np.int64),
        "sample_steps": np.asarray([row["step"] for row in series], dtype=np.int64),
    }
    for name in ("time_ps", "epot_eV", "ekin_eV", "etot_eV", "temperature_K", "max_force_eVA"):
        arrays[name] = np.asarray([row[name] for row in series], dtype=float)
    if final.evaluation is not None:
        arrays["final_forces_eV_A"] = final.evaluation.forces_eV_A
    units = {
        "positions_A": "Å",
        "velocities_A_fs": "Å/fs",
        "time_ps": "ps",
        "temperature_K": "K",
        "epot_eV": "eV",
        "ekin_eV": "eV",
        "etot_eV": "eV",
        "max_force_eVA": "eV/Å",
    }
    if "final_forces_eV_A" in arrays:
        units["final_forces_eV_A"] = "eV/Å"
    return ObservableResult(
        arrays, units=units, metadata={"kind": "nvt", "summary": summary, "integrator": "BAOAB"}
    )


def run_nvt(
    structure: Structure,
    provider,
    *,
    temperature_K=300.0,
    timestep_fs=1.0,
    n_steps=2000,
    friction_fs_inv=DEFAULT_FRICTION_FS_INV,
    log_interval=10,
    traj_interval=None,
    seed=None,
    supercell=(1, 1, 1),
    initial_velocities=None,
    state=None,
    remove_com=True,
    remove_rotation=True,
    checkpoint=None,
) -> SimulationResult:
    """Integrate fixed-cell Langevin dynamics using the BAOAB splitting.

    ``n_steps`` is the final step number, including on restart. A seed controls
    both initial Maxwell velocities and all thermostat noise. Failures produce
    a result and a restart state at the last valid step. Frame/sample schedules
    are independent and the final valid frame and sample are always returned.
    """
    temperature_K = _positive(temperature_K, "temperature_K", zero=True)
    timestep_fs = _positive(timestep_fs, "timestep_fs")
    friction_fs_inv = _positive(friction_fs_inv, "friction_fs_inv", zero=True)
    n_steps = _integer(n_steps, "n_steps", 0)
    log_interval = _integer(log_interval, "log_interval")
    traj_interval = log_interval if traj_interval is None else _integer(traj_interval, "traj_interval")
    structure, _ = _expand(structure, supercell)
    if structure.natoms == 0:
        raise ValueError("dynamics require at least one atom")
    parameters = {
        "temperature_K": temperature_K,
        "timestep_fs": timestep_fs,
        "friction_fs_inv": friction_fs_inv,
        "log_interval": log_interval,
        "traj_interval": traj_interval,
        "remove_com": bool(remove_com),
        "remove_rotation": bool(remove_rotation),
        "supercell": [int(value) for value in supercell],
    }
    identity = _identity(structure, parameters)
    rng = np.random.default_rng(seed)
    samples, records = [], []
    failure = None
    if state is not None:
        if (
            state.get("kind") != "nvt"
            or state.get("schema_version") != 1
            or state.get("identity") != identity
        ):
            raise ValueError("NVT restart structure or scientific parameters do not match")
        final = _record_frame(state["current"], structure)
        rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        samples, records = copy.deepcopy(state["samples"]), copy.deepcopy(state["frames"])
        if final.step > n_steps:
            raise ValueError("n_steps cannot precede the restart step")
    else:
        if initial_velocities is None:
            velocities = rng.normal(size=(structure.natoms, 3)) * np.sqrt(
                ACCELERATION * KB_EV_K * temperature_K / structure.masses[:, None]
            )
            velocities = _project(velocities, structure, remove_com)
            if remove_rotation:
                velocities = _remove_rotation(velocities, structure)
        else:
            velocities = np.asarray(initial_velocities, float)
            if velocities.shape != (structure.natoms, 3) or not np.isfinite(velocities).all():
                raise ValueError("initial_velocities must be finite (natoms,3) in Å/fs")
        velocities = _project(velocities, structure, remove_com)
        final = Frame(structure, velocities, 0, 0.0)
    if final.evaluation is None:
        try:
            final.evaluation = _evaluate(provider, final.structure)
        except Exception as error:
            failure = {"type": type(error).__name__, "message": str(error), "step": final.step}
    if final.evaluation is not None and not samples:
        samples.append(_sample(final, remove_com))
        records.append(_frame_record(final))

    def restart_state():
        return {
            "schema_version": 1,
            "kind": "nvt",
            "identity": identity,
            "current": _frame_record(final),
            "rng_state": copy.deepcopy(rng.bit_generator.state),
            "samples": copy.deepcopy(samples),
            "frames": copy.deepcopy(records),
        }

    if checkpoint is not None:
        checkpoint(restart_state())
    h = timestep_fs
    damping = np.exp(-friction_fs_inv * h)
    noise_scale = np.sqrt(
        ACCELERATION * KB_EV_K * temperature_K * (1 - damping**2) / structure.masses[:, None]
    )
    while final.step < n_steps and failure is None:
        before_rng = copy.deepcopy(rng.bit_generator.state)
        try:
            current_evaluation = final.evaluation
            assert current_evaluation is not None
            acceleration = ACCELERATION * current_evaluation.forces_eV_A / structure.masses[:, None]
            velocities = _project(final.velocities + 0.5 * h * acceleration, structure, remove_com)
            positions = final.structure.positions + 0.5 * h * velocities
            noise = _project(rng.normal(size=velocities.shape) * noise_scale, structure, remove_com)
            velocities = damping * velocities + noise
            positions += 0.5 * h * velocities
            candidate = final.structure.copy(positions=positions)
            evaluation = _evaluate(provider, candidate)
            velocities = _project(
                velocities + 0.5 * h * ACCELERATION * evaluation.forces_eV_A / structure.masses[:, None],
                structure,
                remove_com,
            )
            final = Frame(candidate, velocities, final.step + 1, (final.step + 1) * h, evaluation)
        except Exception as error:
            rng.bit_generator.state = before_rng
            failure = {"type": type(error).__name__, "message": str(error), "step": final.step + 1}
            break
        if final.step % log_interval == 0:
            samples.append(_sample(final, remove_com))
        if final.step % traj_interval == 0:
            records.append(_frame_record(final))
        if checkpoint is not None and (final.step % log_interval == 0 or final.step % traj_interval == 0):
            checkpoint(restart_state())
    restart = restart_state()
    if checkpoint is not None:
        checkpoint(copy.deepcopy(restart))
    frames = [_record_frame(record, structure) for record in records]
    if not frames or frames[-1].step != final.step:
        frames.append(final)
    if final.evaluation is not None and (not samples or samples[-1]["step"] != final.step):
        samples.append(_sample(final, remove_com))
    completed = failure is None and final.step == n_steps
    summary = {
        "kind": "nvt",
        "parameters": parameters,
        "completed": completed,
        "converged": completed,
        "blew_up": failure is not None,
        "crash_step": None if failure is None else failure["step"],
        "failure": failure,
        "n_steps_run": final.step,
        "n_steps_requested": n_steps,
        "natoms": structure.natoms,
        "temperature_setpoint_K": temperature_K,
        "min_interatomic_distance": _minimum_distance(final.structure),
        "degrees_of_freedom": _ndof(structure, remove_com),
        "series": samples,
        "frame_steps": [frame.step for frame in frames],
    }
    return SimulationResult(tuple(frames), _nvt_observable(frames, samples, final, summary), restart, final)


def relax_fire(
    structure: Structure,
    provider,
    *,
    fmax_eV_A=0.02,
    max_steps=500,
    supercell=(1, 1, 1),
    state=None,
    dt=0.1,
    dtmax=1.0,
    max_displacement_A=0.2,
    n_min=5,
    increase=1.1,
    decrease=0.5,
    alpha_start=0.1,
    alpha_decay=0.99,
    checkpoint=None,
    checkpoint_interval=10,
) -> SimulationResult:
    """Relax ionic coordinates with native FIRE; the cell remains fixed.

    FIRE's fictitious velocity and step are optimizer quantities, not physical
    atomic velocities or fs. Restart includes all adaptive optimizer state.
    """
    fmax_eV_A = _positive(fmax_eV_A, "fmax_eV_A")
    max_steps = _integer(max_steps, "max_steps", 0)
    n_min = _integer(n_min, "n_min", 0)
    checkpoint_interval = _integer(checkpoint_interval, "checkpoint_interval")
    for value, name in ((dt, "dt"), (dtmax, "dtmax"), (max_displacement_A, "max_displacement_A")):
        _positive(value, name)
    if (
        dt > dtmax
        or increase <= 1
        or not 0 < decrease < 1
        or not 0 < alpha_start <= 1
        or not 0 < alpha_decay <= 1
    ):
        raise ValueError("invalid FIRE adaptation parameters")
    structure, _ = _expand(structure, supercell)
    if structure.natoms == 0:
        raise ValueError("relaxation requires at least one atom")
    parameters = {
        "fmax_eV_A": fmax_eV_A,
        "dt": float(dt),
        "dtmax": float(dtmax),
        "max_displacement_A": float(max_displacement_A),
        "n_min": n_min,
        "increase": float(increase),
        "decrease": float(decrease),
        "alpha_start": float(alpha_start),
        "alpha_decay": float(alpha_decay),
        "supercell": [int(value) for value in supercell],
    }
    identity = _identity(structure, parameters)
    records = []
    velocity = np.zeros_like(structure.positions)
    adaptive_dt, alpha, positive_steps = float(dt), float(alpha_start), 0
    failure = None
    if state is not None:
        if (
            state.get("kind") != "fire"
            or state.get("schema_version") != 1
            or state.get("identity") != identity
        ):
            raise ValueError("FIRE restart structure or scientific parameters do not match")
        final = _record_frame(state["current"], structure)
        velocity = np.asarray(state["velocity"], float)
        adaptive_dt, alpha, positive_steps = state["dt"], state["alpha"], state["positive_steps"]
        records = copy.deepcopy(state["frames"])
        if final.step > max_steps:
            raise ValueError("max_steps cannot precede the restart step")
    else:
        final = Frame(structure, step=0)
    if final.evaluation is None:
        try:
            final.evaluation = _evaluate(provider, final.structure)
        except Exception as error:
            failure = {"type": type(error).__name__, "message": str(error), "step": final.step}
    if not records:
        records.append(_frame_record(final))

    def restart_state():
        return {
            "schema_version": 1,
            "kind": "fire",
            "identity": identity,
            "current": _frame_record(final),
            "velocity": velocity.tolist(),
            "dt": adaptive_dt,
            "alpha": alpha,
            "positive_steps": positive_steps,
            "frames": copy.deepcopy(records),
        }

    if checkpoint is not None:
        checkpoint(restart_state())

    def forces():
        return _project(final.evaluation.forces_eV_A, structure, False)

    def maximum():
        return (
            float(np.linalg.norm(forces(), axis=1).max(initial=0)) if final.evaluation is not None else None
        )

    while failure is None and final.step < max_steps and maximum() > fmax_eV_A:
        previous_optimizer = velocity.copy(), adaptive_dt, alpha, positive_steps
        force = forces()
        power = float(np.sum(velocity * force))
        if power > 0:
            velocity = (1 - alpha) * velocity + alpha * force * np.linalg.norm(velocity) / np.linalg.norm(
                force
            )
            positive_steps += 1
            if positive_steps > n_min:
                adaptive_dt = min(dtmax, adaptive_dt * increase)
                alpha *= alpha_decay
        elif final.step > 0:
            velocity.fill(0)
            adaptive_dt *= decrease
            alpha, positive_steps = alpha_start, 0
        next_velocity = _project(velocity + adaptive_dt * force, structure, False)
        displacement = adaptive_dt * next_velocity
        size = np.linalg.norm(displacement)
        if size > max_displacement_A:
            displacement *= max_displacement_A / size
        try:
            candidate = final.structure.copy(positions=final.structure.positions + displacement)
            evaluation = _evaluate(provider, candidate)
        except Exception as error:
            velocity, adaptive_dt, alpha, positive_steps = previous_optimizer
            failure = {"type": type(error).__name__, "message": str(error), "step": final.step + 1}
            break
        velocity = next_velocity
        final = Frame(candidate, step=final.step + 1, evaluation=evaluation)
        records.append(_frame_record(final))
        if checkpoint is not None and final.step % checkpoint_interval == 0:
            checkpoint(restart_state())
    converged = failure is None and maximum() <= fmax_eV_A
    summary = {
        "kind": "relax",
        "parameters": parameters,
        "converged": converged,
        "failure": failure,
        "final_energy_eV": None if final.evaluation is None else final.evaluation.energy_eV,
        "final_energy_per_atom_eV": None
        if final.evaluation is None
        else final.evaluation.energy_eV / structure.natoms,
        "fmax_eVA": maximum(),
        "fmax_target_eVA": fmax_eV_A,
        "n_steps": final.step,
        "natoms": structure.natoms,
    }
    restart = restart_state()
    if checkpoint is not None:
        checkpoint(copy.deepcopy(restart))
    frames = tuple(_record_frame(record, structure) for record in records)
    arrays = {
        "positions_A": np.asarray([frame.structure.positions for frame in frames]),
        "steps": np.asarray([frame.step for frame in frames], dtype=np.int64),
        "energy_eV": np.asarray(
            [np.nan if frame.evaluation is None else frame.evaluation.energy_eV for frame in frames]
        ),
        "max_force_eV_A": np.asarray(
            [
                np.nan
                if frame.evaluation is None
                else np.linalg.norm(_project(frame.evaluation.forces_eV_A, structure, False), axis=1).max(
                    initial=0
                )
                for frame in frames
            ]
        ),
    }
    observable = ObservableResult(
        arrays,
        units={"positions_A": "Å", "energy_eV": "eV", "max_force_eV_A": "eV/Å"},
        metadata={"kind": "fire", "summary": summary, "fixed_cell": True},
    )
    return SimulationResult(frames, observable, restart, final)
