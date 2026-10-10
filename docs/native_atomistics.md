# Native atomistic APIs

Queer implements fixed-cell dynamics, ionic relaxation, displacement phonons,
frame selection, trajectory metrics, and reference-dataset export. Force and
energy prediction enters through an explicit callback or `ForceProvider`; the
integrators, force-constant assembly, periodic geometry, and file handling run
inside Queer.

## Structures, evaluations, and units

Import the public types and calculators from `queer.atomistic`.

| Quantity | Native convention |
| --- | --- |
| `Structure.cell` | `(3, 3)` row lattice vectors, Å |
| `Structure.positions` | `(N, 3)` Cartesian positions, Å |
| `Structure.atomic_numbers` | `(N,)` actual atomic numbers |
| `Structure.masses` | `(N,)` masses, amu; elemental defaults when omitted |
| `Frame.velocities` | `(N, 3)`, Å/fs |
| `Evaluation.energy_eV` | Finite total energy, eV |
| `Evaluation.forces_eV_A` | Finite `(N, 3)` forces, eV/Å |
| `Evaluation.stress_eV_A3` | Optional finite `(3, 3)` tensile stress, eV/Å³ |
| Dynamics time, temperature, friction | fs, K, fs⁻¹ |
| Phonon reciprocal points | Fractional coordinates in the input cell's reciprocal basis |

A provider implements `evaluate(structure) -> Evaluation`, or is a callable with
the same input and output. Queer passes a copied structure, validates the returned
evaluation, and rejects a changed atom count. The provider must support the
expanded atom count when `supercell` is supplied.

`SimulationResult` contains `frames`, `final_frame`, `observable`, and a
JSON-serializable `state`. Its `summary` property contains completion, failure,
and convergence information. `ObservableResult` carries arrays, dimensions,
units, and JSON metadata; retrieve arrays with `result["array_name"]`.

The following example is runnable with the base numerical dependencies. Its
harmonic potential is an analytical trap used to demonstrate the API, rather
than a material force field.

```python
import json

import numpy as np

from queer.atomistic import Evaluation, Structure, relax_fire, run_nvt


def harmonic(structure):
    spring_eV_A2 = 0.5
    positions = structure.positions
    return Evaluation(
        energy_eV=0.5 * spring_eV_A2 * float(np.sum(positions**2)),
        forces_eV_A=-spring_eV_A2 * positions,
    )


structure = Structure(
    cell=np.eye(3) * 20,
    positions=[[0.4, 0.2, 0], [-0.4, -0.2, 0]],
    atomic_numbers=[6, 6],
    pbc=False,
    masses=[12, 12],
)
parameters = dict(
    temperature_K=100,
    timestep_fs=0.5,
    friction_fs_inv=0.02,
    seed=7,
    log_interval=7,
    traj_interval=11,
    remove_rotation=False,
)
partial = run_nvt(structure, harmonic, n_steps=19, **parameters)
saved_state = json.loads(json.dumps(partial.state, allow_nan=False))
resumed = run_nvt(structure, harmonic, n_steps=40, state=saved_state, **parameters)
complete = run_nvt(structure, harmonic, n_steps=40, **parameters)
np.testing.assert_array_equal(
    resumed.observable["positions_A"], complete.observable["positions_A"]
)
assert [frame.step for frame in resumed.frames] == [0, 11, 22, 33, 40]

relaxed = relax_fire(structure, harmonic, fmax_eV_A=1e-7, max_steps=1000)
assert relaxed.summary["converged"]
print(relaxed.final_frame.structure.positions)
```

## Langevin NVT and FIRE

`run_nvt(structure, provider, *, temperature_K=300, timestep_fs=1,
n_steps=2000, friction_fs_inv=..., log_interval=10, traj_interval=None,
seed=None, supercell=(1, 1, 1), initial_velocities=None, state=None,
remove_com=True, remove_rotation=True, checkpoint=None)` uses the BAOAB splitting:
half force kick, half position drift, exact Ornstein–Uhlenbeck thermostat update,
half drift, and half kick. At zero friction it reduces to velocity Verlet.
The default friction is approximately `0.00098226948 fs⁻¹`; pass a physical
`friction_fs_inv` explicitly when selecting another damping time.

Initial velocities are Maxwell samples in Å/fs, using the same seeded generator
as the thermostat. They are projected through constraints and, by default,
center-of-mass motion is removed. Initial angular momentum is removed when
`remove_rotation=True` for an unconstrained structure with at least two atoms.
Rotation removal applies to initialization; it is not a persistent rotational
constraint or an exact-temperature rescaling. Supplied `initial_velocities`
must have the expanded structure's `(N, 3)` shape and are constraint/COM
projected without rotational initialization.

`log_interval` controls thermodynamic samples and `traj_interval` controls
recorded frames independently. A missing trajectory interval defaults to the
log interval. Results include step zero and the final valid step, even when the
final step falls between cadence points. Explicit `sample_steps` and
`frame_steps` arrays preserve their alignment. Energies are totals in eV;
`summary["series"]` also includes per-atom energies and time in ps.

`relax_fire(structure, provider, *, fmax_eV_A=0.02, max_steps=500,
supercell=(1, 1, 1), state=None, dt=0.1, dtmax=1, max_displacement_A=0.2,
n_min=5, increase=1.1, decrease=0.5, alpha_start=0.1, alpha_decay=0.99,
checkpoint=None, checkpoint_interval=10)` implements FIRE with adaptive
fictitious velocity, step, and force mixing. `dt` and `dtmax` are optimizer
quantities; they are not physical femtoseconds. The cell stays fixed. The
global displacement norm is bounded by `max_displacement_A`, and convergence
uses the largest atomic norm of the allowed force component. Every accepted
optimizer step is retained as a frame.

`Structure.fixed_coordinates` is an `(N, 3)` Boolean mask, with `True` meaning
fixed. The default basis is Cartesian. With
`metadata={"constraint_basis": "fractional"}`, the allowed motion is the span
of the unfixed row lattice vectors. This preserves selective-dynamics
coordinates in skew cells; Queer projects forces and velocities into that
Cartesian tangent space. POSCAR selective-dynamics imports record the
fractional convention. Only fixed-coordinate constraints and a fixed cell are
implemented.

The kinetic temperature uses the number of allowed coordinate degrees of
freedom after persistent COM removal. For Cartesian constraints, one COM degree
is removed per axis containing at least two moving atoms. Fractional constraints
use the rank of the allowed constrained COM correction for a structure with
more than one atom. A single atom retains its allowed motion. Initial rotation
removal does not subtract persistent degrees of freedom. An entirely fixed
structure has zero kinetic energy and reported temperature zero.

## Checkpoints and restart behavior

Pass `checkpoint=save_state` to persist JSON states during either simulation.
NVT invokes the callback initially, at logging or trajectory cadence points,
and finally. FIRE invokes it initially, every `checkpoint_interval` accepted
steps, and finally. The last callback can repeat the last cadence step.
Callbacks can raise to interrupt execution; exceptions from checkpoint callbacks
propagate. States retain the current geometry and evaluation, cadence histories,
and either velocities/RNG state or FIRE's adaptive optimizer state. Their size
grows with the retained history.

`n_steps` and `max_steps` are final total step numbers, including when `state`
is supplied. Passing a smaller target than the saved step is rejected.
Structure identity, masses, constraints, and scientific controls must match.
`frames_from_state(state, original_structure)` restores typed trajectory frames,
including the current valid frame. Terminal frames and samples added for display
are kept out of cadence history, so extending a partial API run gives the same
recorded history as an uninterrupted run.

An ordinary force-producer exception produces a result with failure information
and a state at the last accepted step. An unsuccessful NVT step restores its RNG
state; FIRE restores its adaptive optimizer state. Check `summary["completed"]`
for NVT and `summary["converged"]` for FIRE. `KeyboardInterrupt` and `SystemExit`
propagate, leaving earlier callbacks available for restart. Exact restart
agreement assumes a deterministic provider and the same numerical environment;
Queer does not serialize a provider's private random state.

The `run_workflow(..., force_provider=provider,
force_provider_id="stable-producer-identity")` boundary persists states through
`RunStore`. Resume uses the original configuration and input checksums, with the
same provider identity. A failed simulation retains a diagnostic and restart
states without publishing a completed result. Unlike direct API extension,
workflow resume requires the original configured step target.

## Displacement phonons and supplied forces

The force-evaluation and assembly stages are public so displacement work can be
distributed or evaluated by an independent producer:

```python
from queer.atomistic import (
    assemble_force_constants,
    phonon_dispersion,
    plan_displacements,
)

# Reuse the analytical structure and callback above for a pinned Γ-point fixture.
plan = plan_displacements(structure, supercell=(1, 1, 1), delta_A=0.01)
forces = np.stack([harmonic(s).forces_eV_A for s in plan.configurations])
constants = assemble_force_constants(
    plan, forces, frederiksen=False, acoustic=False
)
dispersion = phonon_dispersion(constants, qpoints=[[0, 0, 0]])
assert dispersion["frequencies_THz"].shape == (6, 1)
```

The trap has no translational invariance, so this analytical example disables
net-force correction and the acoustic rule. Periodic solid calculations normally
use the defaults described below.

`plan_displacements(structure, *, supercell=(2, 2, 2), delta_A=0.01)` creates
`6*Nprimitive` structures. For each primitive atom, axes are ordered x, y, z;
each axis uses `-delta_A` followed by `+delta_A`. The displaced atom is in the
origin image. `plan.entries` records `(atom_index, axis, sign)`;
`plan.configurations` and its `structures` alias give the evaluation structures.
Fixed displacement coordinates are rejected.

Supercell atoms are ordered by cell image, then primitive atom. Images follow
`np.ndindex(supercell)`, with the final image index varying fastest.
`plan.cell_indices` and `plan.reference` make that ordering explicit. Supply
finite forces in eV/Å with shape
`(6*Nprimitive, Nsupercell_atoms, 3)`, preserving both orders exactly.

`assemble_force_constants(plan, forces, *, frederiksen=True, acoustic=True,
symmetrize_iterations=3)` returns `ForceConstants`. Central differences form
`C[R, 3*i+a, 3*j+b] = -∂F[R,j,b]/∂u[0,i,a]`, in eV/Å².
Translations are centered integer primitive-cell images. Frederiksen correction
subtracts the total force from the displaced atom in each displacement before
differentiation. Reciprocal image blocks are symmetrized under
`C[R] = C[-R].T`. The acoustic rule projects the Γ force matrix orthogonally
against uniform translations and places its correction in the origin block.
Metadata records the controls, atom order, and acoustic residual.

`phonon_dispersion(constants, qpoints)` applies mass weighting, Fourier phase
`exp(-2πi R·q)`, and Hermitian diagonalization. It returns
`frequencies_THz` with shape `(3*Nprimitive, Nq)` and mass-weighted normalized
`eigenvectors` with shape `(Nq, 3*Nprimitive, 3*Nprimitive)`. Frequencies are
ordinary frequencies in THz. Negative values mean imaginary modes; they are
not discarded. The `hermitian_residual` array reports the interpolation residual,
including possible finite-supercell even-image aliasing, before diagonalizing
the Hermitian component. Converge supercell size and displacement magnitude for
the physical system.

`phonon_bands(structure, provider=None, *, forces=None, supercell=(2, 2, 2),
delta_A=0.01, band_path=None, qpoints=None, n_qpoints_per_segment=50,
frederiksen=True, acoustic=True, symmetrize_iterations=3)` combines these stages.
Supply exactly one provider or force array. Explicit `qpoints` avoid symmetry
detection. Otherwise, provide `band_path={"segments": [...]}` with each segment's
fractional `start`, `end`, `start_label`, and `end_label`, or use the native
crystallographic path through the optional Spglib primitive. Each segment includes
both endpoints. `q_distance` uses physical reciprocal distance in Å⁻¹ and does
not add an artificial jump between disconnected segments. Segment indices and
labels remain in metadata. The generator export supplies its historical path
coordinate separately.

## Frame selection, metrics, and datasets

`select_min_energy_frame(frames, samples=None, *, energy_key="etot_eV",
fallback=None, alignment="exact")` returns `SelectionResult(frame, summary)`.
With telemetry, samples must identify their simulation `step`. Exact alignment
considers only finite energies whose steps have recorded geometry. With no
telemetry, it uses stored evaluation energies and includes kinetic energy when
velocities are present and total energy is requested. `epot_eV` selects potential
energy instead. Ties preserve the first occurrence.

`alignment="previous"` permits a telemetry sample to select the preceding
recorded geometry; the summary identifies both `selected_step` and
`selected_sample_step` and sets `energy_matches_geometry` accordingly. Missing
usable energies pass through the first frame, or the supplied fallback structure
when there are no frames. Missing both geometry and fallback raises an error.
`queer.compat.generator.select_legacy_min_energy_frame` is the explicit bridge
for legacy cadence inference and its previous-frame behavior; native selection
requires step alignment.

`phonon_metrics` reports finite minimum/maximum frequencies and counts values
below `-0.001 THz` as imaginary. `nvt_metrics` reports temperature statistics,
maximum force, completion, minimum distance, and the absolute least-squares slope
of total energy per atom in meV/atom/ps. Time and energy are paired from the same
finite rows. The aliases `benchmark_metrics_from_phonon` and
`benchmark_metrics_from_nvt` expose the same implementations.

`dataset_item_to_frame`, `dataset_items_to_extxyz`, and `export_dataset_splits`
accept reference items with an `outputs` mapping containing POSCAR text,
`energy`, `forces`, and optional `(3, 3)` `stress`. Native defaults use eV,
eV/Å, and tensile eV/Å³. Other producers must declare `stress_unit="kbar"`
and the appropriate `stress_sign` when necessary. Exports preserve `REF_energy`,
`REF_forces`, and `REF_stress` for external training producers.
`queer.compat.generator.dataset_exports` retains the audited legacy kbar/sign
conversion explicitly.

`split_dataset` uses a seeded permutation, with default train/valid/test ratios
`0.8/0.1/0.1` and seed `42`. Ratios must sum to one. Every item appears in one
split, and each positive-ratio split contains at least one item. Too few items
for the requested nonempty splits raise an error. `export_dataset_splits`
returns `train.xyz`, `valid.xyz`, and `test.xyz` text mappings.

## Optional direct MACE producer

```python
from queer.adapters.mace import MACEForceProvider

provider = MACEForceProvider(
    checkpoint="/resolved/path/to/trained.model",
    device="cpu",
    compute_stress=False,
)
# result = run_nvt(structure, provider, n_steps=100, seed=7)
```

Provide exactly one resolved full-module checkpoint or loaded `model`. Checkpoint
model classes must be installed in the matching producer environment; state-only
weight dictionaries and foundation-model catalog names are not resolved by this
adapter. `from_checkpoint(path, **options)` is a convenience constructor.
Multi-head checkpoints require an explicit `head` when more than one head exists.
The checkpoint must support every atomic number in the evaluated structure.

The adapter builds native periodic neighbors and Torch position/cell/shift,
one-hot species, edge, batch, and head tensors. It invokes the model's direct
forward method and requires energy and force outputs; it never constructs an
ASE calculator. Position gradients remain enabled during force prediction, even
if the caller uses `torch.no_grad()`. Model parameters are frozen for inference.
Torch and producer classes load lazily; importing the adapter does not load
Torch or ASE.

Native input/output units remain Å and eV. For another producer convention,
declare positive `energy_units_to_eV` and `length_units_to_A` factors; the adapter
converts geometry and applies the corresponding force and stress factors.
Stress evaluation is optional and must be supported by the model.
`provider.provenance` records the exact checkpoint SHA-256, or loaded model-state
checksum, together with model class, Torch version, device, dtype, head, and unit
options. `provider.fingerprint` supplies a stable identifier for that record.

Independent analytical tests cover harmonic dynamics and constrained relaxation,
thermal sampling, displacement derivatives, acoustic modes, unequal masses,
signed unstable modes, cadence alignment, dataset reference interoperability,
and exact interrupted/resumed trajectories. Direct Torch tests cover graph
construction, force gradients, unit conversions, stress, heads, and checkpoint
checksums. No actual trained MACE checkpoint was available for validation;
production-model accuracy and checkpoint/version compatibility remain properties
of the selected producer environment.

An additional isolated producer smoke test used Python 3.13.13,
Torch `2.14.0+cu130`, PyG `2.8.0.post1`, MACE `0.3.16`, and e3nn `0.6.0`.
All ten optional adapter tests ran successfully, including exact float32
Hartree/Bohr graph serialization and Hamiltonian prediction conversion with
`atol=1e-6 eV`, `rtol=1e-7`. A small, randomly initialized actual MACE architecture
with two interaction layers, correlation order two, and `8x0e + 8x1o` hidden
representations produced finite energy, forces, and stress through the native
adapter. A Cartesian energy finite difference with `delta=1e-5 Å` agreed with
its force to approximately `3e-9 eV/Å` (test tolerance `atol=1e-7 eV/Å`,
`rtol=1e-6`); total force vanished within `1e-12 eV/Å`. Saving and reloading that
untrained full-module checkpoint reproduced energy and forces exactly. This
checks architecture/serialization interoperability and gradients, without
establishing trained-model accuracy.
