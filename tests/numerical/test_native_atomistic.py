"""Independent analytic atomistic references and deterministic restart checks."""

import json

import numpy as np
import pytest

from queer.atomistic import (
    Evaluation,
    Frame,
    Structure,
    assemble_force_constants,
    frames_from_state,
    nvt_metrics,
    phonon_bands,
    phonon_dispersion,
    phonon_metrics,
    plan_displacements,
    relax_fire,
    run_nvt,
    select_min_energy_frame,
)
from queer.atomistic.datasets import KBAR_TO_EV_A3, dataset_items_to_extxyz, split_dataset
from queer.atomistic.dynamics import ACCELERATION, KB_EV_K
from queer.io.atomistic import read_extxyz, write_poscar


def atom(positions=((0.4, 0.2, 0),), mass=10):
    n = len(positions)
    return Structure(np.diag([20.0, 20, 20]), positions, np.ones(n, int), pbc=False, masses=np.full(n, mass))


class Harmonic:
    def __init__(self, spring=0.5):
        self.spring = spring

    def evaluate(self, structure):
        return Evaluation(
            0.5 * self.spring * float(np.sum(structure.positions**2)), -self.spring * structure.positions
        )


def test_baoab_zero_friction_conserves_analytic_oscillator_energy():
    initial = atom()
    run = run_nvt(
        initial,
        Harmonic(),
        temperature_K=0,
        timestep_fs=0.2,
        n_steps=1000,
        friction_fs_inv=0,
        log_interval=10,
        initial_velocities=np.zeros((1, 3)),
    )
    assert run.summary["completed"]
    energy = run.observable["etot_eV"]
    assert np.ptp(energy) / energy[0] < 1e-5
    angular = np.sqrt(ACCELERATION * 0.5 / 10)
    exact = initial.positions * np.cos(angular * 200)
    np.testing.assert_allclose(run.final_frame.structure.positions, exact, atol=2e-6)


def test_langevin_thermal_ensemble_has_requested_kinetic_temperature():
    # Free independent atoms: the exact OU velocity update is an independent
    # canonical reference, without potential/finite-step discretization errors.
    initial = atom(np.zeros((32, 3)), mass=12)
    run = run_nvt(
        initial,
        Harmonic(0),
        temperature_K=250,
        timestep_fs=1,
        friction_fs_inv=0.05,
        n_steps=3000,
        log_interval=5,
        traj_interval=100,
        seed=402,
        remove_com=False,
        remove_rotation=False,
    )
    temperatures = run.observable["temperature_K"][100:]
    assert abs(temperatures.mean() - 250) < 12
    velocities = run.final_frame.velocities
    assert np.isfinite(velocities).all()
    expected_variance = ACCELERATION * KB_EV_K * 250 / 12
    assert abs(np.mean(velocities**2) / expected_variance - 1) < 0.4


def test_seeded_restart_matches_uninterrupted_run_with_independent_cadences():
    kwargs = dict(
        temperature_K=50,
        timestep_fs=0.5,
        friction_fs_inv=0.02,
        log_interval=7,
        traj_interval=11,
        seed=34,
        remove_rotation=False,
    )
    initial = atom(((0.2, 0.3, 0.4), (-0.2, -0.3, -0.4)))
    uninterrupted = run_nvt(initial, Harmonic(), n_steps=47, **kwargs)
    partial = run_nvt(initial, Harmonic(), n_steps=19, **kwargs)
    resumed = run_nvt(initial, Harmonic(), n_steps=47, state=json.loads(json.dumps(partial.state)), **kwargs)
    assert [f.step for f in resumed.frames] == [0, 11, 22, 33, 44, 47]
    assert list(resumed.observable["sample_steps"]) == [0, 7, 14, 21, 28, 35, 42, 47]
    for key in uninterrupted.observable.arrays:
        np.testing.assert_array_equal(uninterrupted.observable[key], resumed.observable[key])
    np.testing.assert_allclose(
        np.sum(initial.masses[:, None] * resumed.final_frame.velocities, axis=0), 0, atol=1e-15
    )
    restored = frames_from_state(partial.state, initial)
    assert restored[-1].step == 19


def test_checkpoint_callback_interrupts_and_resumes_without_noise_loss():
    checkpoints = []

    class Stop(Exception):
        pass

    def checkpoint(state):
        checkpoints.append(json.loads(json.dumps(state, allow_nan=False)))
        if state["current"]["step"] >= 9:
            raise Stop

    kwargs = dict(n_steps=25, log_interval=3, traj_interval=7, seed=23)
    with pytest.raises(Stop):
        run_nvt(atom(), Harmonic(), checkpoint=checkpoint, **kwargs)
    resumed = run_nvt(atom(), Harmonic(), state=checkpoints[-1], **kwargs)
    complete = run_nvt(atom(), Harmonic(), **kwargs)
    np.testing.assert_array_equal(resumed.observable["positions_A"], complete.observable["positions_A"])
    np.testing.assert_array_equal(
        resumed.observable["velocities_A_fs"], complete.observable["velocities_A_fs"]
    )


def test_force_failure_retains_last_valid_state_and_finite_summary():
    class Fails(Harmonic):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def evaluate(self, structure):
            self.calls += 1
            if self.calls == 5:
                raise FloatingPointError("producer became unstable")
            return super().evaluate(structure)

    run = run_nvt(atom(), Fails(), n_steps=20, seed=5, log_interval=3)
    assert not run.summary["completed"]
    assert run.summary["n_steps_run"] == 3
    assert run.summary["crash_step"] == 4
    json.dumps(run.summary, allow_nan=False)
    resumed = run_nvt(atom(), Harmonic(), n_steps=20, seed=5, log_interval=3, state=run.state)
    reference = run_nvt(atom(), Harmonic(), n_steps=20, seed=5, log_interval=3)
    np.testing.assert_array_equal(resumed.observable["positions_A"], reference.observable["positions_A"])


def test_nvt_restart_rejects_parameter_changes_and_invalid_parameters():
    run = run_nvt(atom(), Harmonic(), n_steps=1, seed=3)
    with pytest.raises(ValueError, match="parameters"):
        run_nvt(atom(), Harmonic(), n_steps=2, state=run.state, timestep_fs=2)
    with pytest.raises(ValueError, match="log_interval"):
        run_nvt(atom(), Harmonic(), log_interval=0)


def test_fire_converges_fixed_cell_and_restart_is_equivalent():
    initial = atom()
    kwargs = dict(fmax_eV_A=1e-7, max_steps=1000)
    full = relax_fire(initial, Harmonic(), **kwargs)
    assert full.summary["converged"]
    assert full.summary["fmax_eVA"] <= 1e-7
    np.testing.assert_array_equal(full.final_frame.structure.cell, initial.cell)
    partial = relax_fire(initial, Harmonic(), fmax_eV_A=1e-7, max_steps=13)
    resumed = relax_fire(initial, Harmonic(), state=json.loads(json.dumps(partial.state)), **kwargs)
    np.testing.assert_array_equal(full.observable["positions_A"], resumed.observable["positions_A"])
    checkpoints = []
    relax_fire(initial, Harmonic(), max_steps=12, checkpoint=checkpoints.append, checkpoint_interval=4)
    assert [state["current"]["step"] for state in checkpoints] == [0, 4, 8, 12, 12]


def test_fire_checkpoint_callback_interruption_restores_adaptive_state():
    class Stop(Exception):
        pass

    checkpoints = []

    def checkpoint(state):
        checkpoints.append(json.loads(json.dumps(state, allow_nan=False)))
        if state["current"]["step"] >= 12:
            raise Stop

    parameters = dict(fmax_eV_A=1e-7, max_steps=1000, checkpoint_interval=4)
    with pytest.raises(Stop):
        relax_fire(atom(), Harmonic(), checkpoint=checkpoint, **parameters)
    resumed = relax_fire(atom(), Harmonic(), state=checkpoints[-1], **parameters)
    full = relax_fire(atom(), Harmonic(), **parameters)
    assert resumed.summary["converged"]
    np.testing.assert_array_equal(resumed.observable["positions_A"], full.observable["positions_A"])
    assert resumed.state == full.state


def test_frozen_coordinates_and_initial_force_failure_are_reported():
    initial = atom().copy(fixed_coordinates=[[False, True, True]])
    run = relax_fire(initial, Harmonic(), fmax_eV_A=1e-6)
    assert run.summary["converged"]
    np.testing.assert_array_equal(run.final_frame.structure.positions[:, 1:], initial.positions[:, 1:])

    def wrong(structure):
        return Evaluation(0, np.zeros((structure.natoms + 1, 3)))

    run = relax_fire(initial, wrong)
    assert not run.summary["converged"]
    assert run.summary["final_energy_eV"] is None
    assert "force shape" in run.summary["failure"]["message"]


def test_fractional_constraints_preserve_skew_cell_coordinates():
    cell = np.array([[4.0, 1.5, 0], [0.0, 5.0, 1.0], [0.0, 0.0, 6.0]])
    fractional = np.array([[0.2, 0.3, 0.4], [0.4, 0.1, 0.2]])
    initial = Structure(
        cell,
        fractional @ cell,
        [1, 1],
        masses=[2, 3],
        fixed_coordinates=[[False, True, True], [False, True, True]],
        metadata={"constraint_basis": "fractional"},
    )
    relaxed = relax_fire(initial, Harmonic(), fmax_eV_A=1e-7, max_steps=1000)
    assert relaxed.summary["converged"]
    np.testing.assert_allclose(relaxed.final_frame.structure.fractional_positions[:, 1:], fractional[:, 1:])
    # The exact minimum along each allowed lattice vector is an independent
    # constrained quadratic minimizer: displacement orthogonal to final r.
    np.testing.assert_allclose(relaxed.final_frame.structure.positions @ cell[0], 0, atol=1e-6)
    run = run_nvt(initial, Harmonic(), n_steps=50, seed=22, log_interval=5, temperature_K=120)
    positions = run.observable["positions_A"] @ np.linalg.inv(cell)
    np.testing.assert_allclose(
        positions[:, :, 1:], np.broadcast_to(fractional[:, 1:], positions[:, :, 1:].shape)
    )
    np.testing.assert_allclose(
        np.sum(initial.masses[:, None] * run.final_frame.velocities, axis=0), 0, atol=1e-15
    )
    assert run.summary["degrees_of_freedom"] == 1


class PeriodicChain:
    def __init__(self, spring=2):
        self.spring = spring

    def evaluate(self, structure):
        # Equilibrium sites retain the cell-major ordering from the plan.
        n = structure.natoms
        reference = np.zeros((n, 3))
        reference[:, 0] = np.arange(n) * structure.cell[0, 0] / n
        displacement = structure.positions - reference
        differences = displacement - np.roll(displacement, 1, axis=0)
        forces = self.spring * (
            np.roll(displacement, 1, axis=0) + np.roll(displacement, -1, axis=0) - 2 * displacement
        )
        return Evaluation(0.5 * self.spring * float(np.sum(differences**2)), forces)


def chain_cell():
    return Structure(np.diag([2.0, 10, 10]), [[0, 0, 0]], [1], masses=[4.0])


def test_central_force_constants_recover_analytic_chain_and_acoustic_modes():
    primitive = chain_cell()
    plan = plan_displacements(primitive, supercell=(3, 1, 1), delta_A=0.002)
    provider = PeriodicChain()
    forces = np.asarray(
        [provider.evaluate(configuration).forces_eV_A for configuration in plan.configurations]
    )
    constants = assemble_force_constants(plan, forces)
    qpoints = np.array([[0.0, 0, 0], [0.25, 0, 0], [0.5, 0, 0]])
    result = phonon_dispersion(constants, qpoints)
    expected = np.sqrt(ACCELERATION * 4 * (1 - np.cos(2 * np.pi * qpoints[:, 0])) / 4) * 1000 / (2 * np.pi)
    np.testing.assert_allclose(result["frequencies_THz"], np.tile(expected, (3, 1)), atol=1e-7)
    assert constants.metadata["acoustic_residual_eV_A2"] < 1e-12
    assert np.max(result["hermitian_residual"]) < 1e-12
    native = phonon_bands(primitive, provider, supercell=(3, 1, 1), qpoints=qpoints)
    supplied = phonon_bands(primitive, forces=forces, supercell=(3, 1, 1), delta_A=0.002, qpoints=qpoints)
    np.testing.assert_allclose(native["frequencies_THz"], supplied["frequencies_THz"], atol=1e-7)


def test_frederiksen_corrects_displaced_atom_bias_and_diatomic_mass_weighting():
    primitive = atom(((0.0, 0, 0), (0.8, 0.3, 0.2))).copy(masses=[2, 5])
    plan = plan_displacements(primitive, supercell=(1, 1, 1), delta_A=0.001)
    spring = 1.3
    forces = []
    for structure, (displaced, axis, sign) in zip(plan.configurations, plan.entries, strict=True):
        u = structure.positions - primitive.positions
        first = -spring * (u[0] - u[1])
        force = np.asarray([first, -first])
        force[displaced] += sign * np.array([0.3, -0.2, 0.1])
        forces.append(force)
    constants = assemble_force_constants(plan, forces, frederiksen=True, acoustic=False)
    exact = np.kron([[spring, -spring], [-spring, spring]], np.eye(3))
    np.testing.assert_allclose(constants.values_eV_A2[0], exact, atol=1e-12)
    frequencies = phonon_dispersion(constants, [[0, 0, 0]])["frequencies_THz"][:, 0]
    optical = np.sqrt(ACCELERATION * spring * (1 / 2 + 1 / 5)) * 1000 / (2 * np.pi)
    np.testing.assert_allclose(frequencies[:3], 0, atol=2e-6)
    np.testing.assert_allclose(frequencies[3:], optical, atol=1e-10)
    projected = assemble_force_constants(plan, forces, frederiksen=False, acoustic=True)
    assert projected.metadata["acoustic_residual_eV_A2"] < 1e-10


def test_imaginary_modes_keep_negative_signed_frequencies_and_segment_distance():
    path = {
        "segments": [
            {"start": [0, 0, 0], "end": [0.5, 0, 0], "start_label": "GAMMA", "end_label": "X"},
            {"start": [0.2, 0, 0], "end": [0.4, 0, 0], "start_label": "A", "end_label": "B"},
        ]
    }
    result = phonon_bands(
        chain_cell(), PeriodicChain(-2), supercell=(3, 1, 1), band_path=path, n_qpoints_per_segment=4
    )
    assert result["frequencies_THz"].min() < 0
    assert len(result["qpoints"]) == 8
    assert result["q_distance"][3] == result["q_distance"][4]
    assert result["q_distance"][-1] == pytest.approx(0.7 * np.pi)
    assert phonon_metrics(result)["phonon_n_imaginary"] > 0
    with pytest.raises(ValueError, match="force producer"):
        phonon_bands(chain_cell())


def test_default_phonon_path_uses_native_crystallographic_points():
    pytest.importorskip("spglib")
    primitive = Structure(np.eye(3) * 2, [[0, 0, 0]], [1])
    result = phonon_bands(primitive, Harmonic(0), n_qpoints_per_segment=3)
    assert len(result.metadata["segments"]) > 0
    assert result.metadata["segments"][0]["start_label"] == "GAMMA"
    assert result["frequencies_THz"].shape[0] == 3
    np.testing.assert_array_equal(result["frequencies_THz"], 0)


def test_selection_considers_recorded_geometry_and_reports_explicit_preceding_mode():
    frames = [Frame(atom(), step=step) for step in (0, 11, 22)]
    samples = [
        {"step": 0, "etot_eV": 10},
        {"step": 7, "etot_eV": -100},
        {"step": 11, "etot_eV": 2},
        {"step": 22, "etot_eV": 4},
    ]
    exact = select_min_energy_frame(frames, samples)
    assert exact.frame.step == 11
    assert exact.summary["selected_energy_eV"] == 2
    previous = select_min_energy_frame(frames, samples, alignment="previous")
    assert previous.frame.step == 0
    assert previous.summary["selected_sample_step"] == 7
    assert not previous.summary["energy_matches_geometry"]
    fallback = select_min_energy_frame(frames, [{"step": 0, "etot_eV": np.nan}])
    assert fallback.summary["passthrough"]
    with pytest.raises(ValueError, match="usable frame"):
        select_min_energy_frame([], [])


def test_dataset_partition_has_no_leakage_and_extxyz_keeps_reference_units():
    items = list(range(13))
    splits = split_dataset(items, seed=2)
    assert splits == split_dataset(items, seed=2)
    assert sorted(sum(splits.values(), [])) == items
    assert all(splits.values())
    assert split_dataset([1, 2, 3])["test"]
    with pytest.raises(ValueError, match="not enough"):
        split_dataset([1, 2])
    reference = {
        "outputs": {
            "poscar": write_poscar(atom()),
            "energy": -4.5,
            "forces": [[0.1, 0.2, 0.3]],
            "stress": np.diag([1.0, 2.0, 3.0]).tolist(),
        }
    }
    text = dataset_items_to_extxyz([reference], stress_unit="kbar")
    frame = read_extxyz(text)[0]
    assert frame.structure.metadata["REF_energy"] == -4.5
    np.testing.assert_allclose(frame.structure.arrays["REF_forces"], [[0.1, 0.2, 0.3]])
    np.testing.assert_allclose(
        frame.structure.metadata["REF_stress"], np.diag([1.0, 2.0, 3.0]) * KBAR_TO_EV_A3
    )
    canonical = read_extxyz(dataset_items_to_extxyz([reference]))[0]
    np.testing.assert_allclose(canonical.structure.metadata["REF_stress"], np.diag([1.0, 2.0, 3.0]))


def test_nvt_metric_regression_pairs_times_and_energies_without_missing_row_skew():
    summary = {
        "completed": True,
        "n_steps_requested": 10,
        "n_steps_run": 10,
        "temperature_setpoint_K": 300,
        "series": [
            {"time_ps": 0, "etot_per_atom_eV": 1, "temperature_K": 290, "max_force_eVA": 0.1},
            {"time_ps": 1, "temperature_K": 310, "max_force_eVA": 0.2},
            {"time_ps": 3, "etot_per_atom_eV": 1.6, "temperature_K": 300, "max_force_eVA": 0.3},
        ],
    }
    metrics = nvt_metrics(summary)
    assert metrics["nvt_energy_drift_mev_per_atom_ps"] == pytest.approx(200)
    assert metrics["nvt_temp_mean_dev_k"] == 0
    assert metrics["nvt_completed"] == 1
