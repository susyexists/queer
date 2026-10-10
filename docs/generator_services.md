# Native generator scientific services

Queer owns the numerical postprocessing and atomistic algorithms described here.
The generator remains unchanged and continues using its existing providers until
its integration is migrated separately. No Queer module imports the generator.
DFT solvers, Wannierisation, OpenMX overlap production, trained model architectures
and training remain external producers.

## Install and run

Core structure, file, graph, dataset and force-callback calculations require only
Queer's numerical dependencies. Install `queer[symmetry]` for spglib-assisted
primitive reduction, crystallographic database operations and HPKOT paths. The
path construction, interpolation, reciprocal-basis mappings and Brillouin-zone
geometry are implemented in Queer. `queer[plot]` adds optional rendering.
Neither extra requires ASE, PyATB, SeekPath or Dans_Diffraction.

```bash
queer structure --config configs/examples/native_structure.toml
queer nvt --config configs/examples/native_nvt.toml
queer relax --config configs/examples/native_relax.toml
queer phonons --config configs/examples/native_phonons.toml
queer export /path/to/completed/run --profile generator
```

These examples use a synthetic onsite harmonic potential. They demonstrate the
algorithms and storage contract; they do not represent a material force model.
[The Python example](../examples/native_services.py) uses a plain force callback.

## Scientific interfaces

`Structure`, `Frame`, `Evaluation` and `ForceProvider` are exported by `queer`.
Structure cells are row vectors; positions are Cartesian Å. Energies, forces,
tensile stresses, masses, time and temperature use eV, eV/Å, eV/Å³, amu, fs and K.
Frames record simulation steps explicitly. `Evaluation` accepts an optional
Cartesian stress tensor. The existing tuple-returning `read_poscar` is preserved.

Imported `BandData` holds absolute energies as `(spin, kpoint, band)`, optional
occupations, reciprocal coordinates, segmented paths and reference provenance.
It is separate from Hamiltonian models. Existing model `energies` still returns
Fermi-relative `(band, kpoint)` arrays. New numerical services use
`ObservableResult`; workflows use `ExecutionOptions` and `RunStore`.

Read [structures and symmetry](native_structures.md),
[atomistic methods](native_atomistics.md), and [electronic responses and readers](native_responses.md)
for conventions, public APIs, algorithm validation and limitations.

## Workflow inputs and persistence

Every service has a declared input kind, allowed parameters, consumed files,
checkpoint kind and required final products in `workflows.services.SERVICE_SPECS`.
All configurations contain `inputs.material_dir`; file paths resolve against it.
These workflows do not load Wannier inputs:

| CLI workflow | Input files | Purpose |
| --- | --- | --- |
| `structure` | `structure` | Native parsing, transforms, variants, bonds and optional symmetry |
| `phonons` | `structure`, optional `forces` or `checkpoint` | Finite displacements or analysis of already calculated forces |
| `nvt`, `relax` | `structure`, optional model `checkpoint` | Native BAOAB NVT or fixed-cell FIRE |
| `select` | `trajectory`, optional `series`, fallback `structure` | Select by recorded step and energy |
| `postprocess` | Explicit output files or a calculation directory | Native VASP, OpenMX and Wannier reports/bands |
| `hamiltonian_pack` | `scfout`, optional `h0`, `openmx_input` | Native OpenMX-to-graph preprocessing |
| `hamiltonian_bands` | `graph`, optional `predictions` | Native generalized eigenproblem reconstruction |
| `band_compare` | `reference`, `candidate` | Segment-aware energy/band agreement |
| `dataset` | `dataset` JSON | Reproducible train/valid/test extended XYZ |

Configure a force model with `[parameters.provider] kind="mace"` and an explicitly
resolved `inputs.checkpoint`. Optional `queer[mace]` supplies Torch and MACE model
classes; Queer builds neighbors and calls the model tensors directly. It does not
construct ASE atoms or use `MACECalculator`. Existing producer checkpoints may
require their matching producer environment. Catalog downloads and training are
external concerns.

Python callers can supply a provider directly:

```python
run = run_workflow(config, force_provider=provider, force_provider_id="resolved-model-sha256")
```

A callback must supply a stable `force_provider_id` or string `fingerprint`.
Input/checkpoint checksums and scientific parameters are checked on resume.
NVT/FIRE state includes RNG/optimizer state, retained frames and independent log
and trajectory schedules. Displacement forces and local electronic integration
cells are committed independently. Failed calculations retain diagnostics and
valid checkpoints without creating a successful final result. Resume uses the
same configuration and an explicit run directory:

```bash
queer nvt --config simulation.toml --run-dir /path/to/run --resume
```

`postprocess` discovers only the standard files for its selected engine if no
explicit files are given. Every discovered file is checksummed. Missing Fermi
levels and unknown convergence remain unknown; they are not replaced by zero or
assumed success. OUTCAR finite-temperature free energies remain distinct from
sigma-to-zero energies; VASP compressive kbar stress is converted to tensile
eV/Å³. Native datasets default to tensile eV/Å³. Declare the input stress unit
and sign for other source records.

## Hamiltonian graphs and model bridges

`queer.io.pack_openmx` consumes the native SCFOUT reader directly. It preserves
original atom and neighbor order, onsite rows before offsite rows, reversed-edge
indices and integer cell translations. The padded non-SOC bases are 14, 19 and
26 slots. Placement is derived from OpenMX orbital quantum labels or an explicit
checkpoint `basis_layout`, rather than guessed from element identity.

The reserved shell placement is `s:0..2`, `p:3..8`, `d:9..18`, `f:19..25`;
the 14-slot basis has one d shell. Occupied basis slots are declared per atom and
padding is masked per block. SOC is rejected by this generator graph contract.
Collinear sources require explicit `spin="up"` or `spin="down"`; one selected
channel has degeneracy one and does not acquire a guessed channel filling.

```python
from queer.io import pack_openmx, write_graphs
from queer import GraphHamiltonianModel

graph = pack_openmx("calculation.scfout", openmx_input="calculation.dat", nao_max=14)
write_graphs("graphs.npz", {"structure-0": graph})
model = GraphHamiltonianModel(graph)
energies = model.energies([[0, 0, 0]])
```

Native graph archives contain numerical/string arrays and JSON metadata, with
eV and Å units and no pickle. Their generalized eigensolver retains overlaps;
translation Hermiticity is validated unless the caller explicitly requests
averaging. Band workflows accept an explicit Fermi level, the SCF reference
Fermi level, or `electron_num` with `filling_grid` and `temperature_kelvin` for
native integration of electron filling. The selected reference is recorded.

`HamiltonianGraph.with_predictions` accepts eV or Hartree blocks, masks padding,
and marks the resulting Hamiltonian as predicted. A replaced prediction is not
a retained SCF reference. `masked_hamiltonian_errors` accepts real reference
graphs and computes padding-aware MAE/RMSE, optionally by structure split. An
overlap/H0-only archive cannot claim a reference error.
Its file chemical potential is retained as a reported value, without treating
it as an established SCF Fermi reference; supply explicit filling or a reference
energy for predicted band reconstruction.

`queer.adapters.hamgnn.HamGNNAdapter` calls an externally supplied predictor;
its checkpoint loader is supplied by the producer integration. It returns a
Hamiltonian graph and records checkpoint provenance. It does not provide forces.
`queer[hamgnn]` enables optional PyG serialization. Its bridge converts graph
geometry to Bohr and H/H0 to Hartree, with checkpoint-compatible float32 tensors.
Legacy `graph_data.npz` contains pickle; its reader requires `trusted=True` and
an explicit orbital layout. The scientific archive remains numeric-only.

## Compatibility and exported products

`queer.compat.generator` contains parameter, payload, recipe and artifact
adapters. `translate_pyatb` returns a native workflow, native parameters,
execution controls and `export_metadata`. Put `export_metadata` in `inputs` when
constructing the native run configuration. Legacy mappings include:

| Legacy field | Native meaning |
| --- | --- |
| `fermi_energy`, `fermi_energy_unit=Ry` | Absolute eV, with explicit Ry conversion |
| FHS `fhs_energy_window` | Fermi-relative bounds; upper-exclusive count scan |
| FHS `fhs_dimension` | Selected Hall components; bulk conductivity remains S/cm |
| Hall `openmx`/`fhs` sign conventions | Export sign only for direct/Kubo legacy recipes |
| Shift `eta` | Legacy Gaussian width divided by √2 for native standard deviation |
| `n_occ`, `m_unocc` | A single one-based occupied/unoccupied band pair; native indices are zero-based |
| `occ_band` | Occupied count and count-based scissors boundary |
| Shift methods `0`, `1` | Audited legacy sum-over-states computation mapped to native analytic derivatives |
| Threshold `adaptive_grid_threshold` | Curvature threshold converted from Bohr² to Å² |
| ASE `friction` | Legacy inverse ASE time converted to fs⁻¹ |
| Layer and phonon path display recipes | Historical behavior confined to compatibility adapters |

The native shift-current Sternheimer option performs its own constrained linear
solve; the legacy method translation does not claim to select that solver.
Local threshold subdivision and error-controlled cubature are deterministic,
persist error estimates and enforce point/cell budgets. A threshold-only run
does not establish continuum convergence. Uniform global refinement remains
available separately.

`select_legacy_min_energy_frame` requires an explicit trajectory interval to
infer absent historical steps, uses declared preceding-frame alignment and
reports that inference. Native selection defaults to exact recorded steps.
`transform_structures` enables historical layer scaling. `dataset_exports`
preserves the generator's historical kbar/sign conversion explicitly.

`queer export RUN_DIR --profile generator` verifies a completed manifest,
artifact checksums, required numerical products and known convergence before
writing success. An unconverged FIRE run remains available for analysis but
cannot export success. Exports include a product-checksum manifest and units,
input/model provenance and calculation status in JSON summaries.

Expected filenames include `Out/AHC/ahc.dat`,
`Out/Shift_Current/shift_current.dat`, `Out/Chern_Number/chern_number.dat`,
`pyatb_results.json`, `phonon_band.dat`, `phonon_results.json`,
`trajectory.traj`, `nvt_results.json`, `relax.traj`, `relax_results.json`,
`select_results.json`, `POSCAR`, `CONTCAR`, `results.json`, `band_1.dat`,
and dataset `train.xyz`, `valid.xyz`, `test.xyz`. Graph compatibility export
requires the optional PyG bridge. Band payload axes retain the existing
`(spin, kpoint, band)` shape; Fermi-relative display is explicitly marked.
Canonical phonon arrays retain physical reciprocal distances and signed THz;
the compatibility payload alone uses equal segment display widths.
DFT compatibility summaries retain native `energy_eV` and tensile `stress_eV_A3`
and explicitly label legacy fields: VASP `energy` prefers its force-consistent
free energy and `stress` uses compressive kbar; OpenMX `stress` retains its
historical tensile kbar convention. Unknown reference-dependent gaps remain
unknown rather than preventing band import.

## Migration and validation

Migrate a generator provider by translating its recipe, resolving producer
checkpoints and explicit references, running a Queer workflow and exporting its
completed artifacts. Compare analytical fixtures before evaluating real
producer outputs. Keep generators' job management, artifact collection and
model downloads outside the scientific package.

Committed fixtures are synthetic. Tests cover analytical spectra, Hall/Chern,
nonlinear responses, harmonic phonons/relaxation, thermostat statistics,
constraints, restart equality, skew neighbors, supercells, magnetic expansion,
recorded-step alignment, native archives and legacy payloads. Bidirectional ASE
ULM/extXYZ and optional model tensor checks provide interchange evidence.
Actual trained MACE and HamGNN checkpoints are not committed; producer-specific
checkpoint validation requires the model owner's environment. The maintained
pytest suite, Ruff, mypy and isolated wheel/sdist checks cover packaging.
A separate optional environment also validates an actual untrained MACE
architecture against finite-difference forces and exact checkpoint reload.

Implementation is independently written under Queer's MIT license. Redistributed
HPKOT scientific tables retain their MIT license and attribution in
`queer/resources/symmetry/`. No ASE or HamGNN implementation was copied.
