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

### Generator command entry

`python -m queer generator run --step STEP --params FILE [--workspace DIR]`,
`python -m queer generator capabilities` and `python -m queer generator compare ...`
start in `queer._generator_boot`, which imports only the standard library. It
compares the installed versions of the core dependencies with its embedded copy of
`[project].dependencies` (no `packaging` needed) before anything imports NumPy.
A gap makes `run` write `queer_results.json` atomically with
`{"status": "failed", "engine": "queer", "error": {"code": "queer_env_missing"}}`,
the step, QUEER and Python versions, and exit 2; `capabilities` prints the same
error with `"steps": []`. Otherwise each command is delegated to
`queer.compat.generator.runner.main`; a build without that runner reports
`queer_internal_error` and exits 1. Exit codes follow the Generator PythTB
driver: 0 success, 1 internal error, 2 reported failure, 3 out of memory.

### Wannier90 inputs of the PythTB engine

`queer.io.wannier.read_win` returns a `WinData` record of the `.win` settings that
the Generator's PythTB engine consumes, read with its lenient rules: case-insensitive
keywords in `=`, `:` or blank-separated form, `!`/`#` comments, Fortran `d`
exponents and logical spellings, and skipped malformed `kpoints`/`kpoint_path`
rows. A missing or malformed `mp_grid` is inferred from the distinct coordinates of
the `kpoints` block. Bohr cells are converted with `WANNIER_BOHR_ANGSTROM`
(CODATA 2018), which differs from the Generator's CODATA 2010 factor by about
3e-11 relative; `cell_unit` records the declared unit. `read_eig`,
`read_band_kpt`, `read_labelinfo` and `read_band_dat` read the other staged files,
and `band_path_breaks` marks a path jump where a step exceeds three times the
previous one. Unlike the validating `queer.io.bands.read_wannier_bands`, these
readers follow the producer's semantics, so a replay sees the same rows it did.

`periodic_dirs_from_table` reads the periodic axes from the table as Wannier90
wrote it: Wigner-Seitz images move a hop between centres more than half a cell
apart through the neighbouring cell, so a corrected slab table also carries
translations across its vacuum. `fold_vacuum_axes` sums those components onto
zero, which reproduces the corrected Fourier sum exactly at zero vacuum momentum.
`unfold_from_wsvec` applies wsvec file images with the producer's file-mode rules:
pairs absent from the file keep their hop at `R`, entries outside the table are
ignored, and images already applied to the HR weights are refused. The strict
`ws_distance` mode of `normalized_operators` requires a complete mapping instead.

### Numerical ports of the PythTB engine

The Generator's PythTB engine computes most of its results with its own NumPy
code around a few PythTB calls. QUEER carries those algorithms. Each port is
tested against the original function, vendored verbatim in `tests/oracles`, on
randomized and structured inputs, and independently against analytic models.

`queer.observables.topology.wilson_plane` evaluates the Wilson loops of the plane
k_c = value: loops along k_a = i/n (endpoint excluded) at transverse points
k_b = j/n_t, with (a, b, c) cyclic so the plane's Chern number is the component
C_c. Its links are those of `queer.observables.geometry.wilson_loop`: occupied
overlaps through the exact diagonal basis overlap of the Wannier centres, closed
on the first point's eigenvectors at k + G without any extra phase. These equal
the Generator's links on PythTB's cell-periodic eigenvectors, whose closing link
carries exp(-2πiτ_a). Applying that phase to QUEER's periodic-gauge eigenvectors
instead would be wrong: for a two-orbital chain with τ = (0, ½) the closing link
becomes singular. `queer.physics.topology` holds the phase-winding Chern number
C = -(1/2π) Σ wrap(Δφ), the Soluyanov-Vanderbilt largest-gap Z2 over t in [0, ½]
with its ambiguity flag, the Kramers pairing check at the TRIMs and the strong and
weak indices of a 3D model. A plane is resolved when every Berry-phase step is
below π/2 and C lies within 0.05 of an integer, and Z2 is unambiguous; otherwise
n_t doubles, at most `max_refinements` times, reusing the loops already computed.
`topology_planes` and `trim_points` give the Generator's plane list (the layer
plane in 2D; k_c = 0, ½ for c = 3, 1, 2 in 3D) and its TRIMs.

### Replaying saved Generator PythTB calculations

`scripts/validation/generator_replay.py` compares an already downloaded successful
`PYTHTB_BANDS` result against native QUEER calculations. It needs the original
`pythtb_results.json`, `pythtb_params.json`, Wannier HR, centres XYZ and WIN files.
The utility performs no authentication or network requests and does not import
PythTB, ASE, PyATB or the Generator producer. Save account-scoped workflow/job
metadata and artifact retrieval provenance separately, then pass them explicitly:

```bash
python scripts/validation/generator_replay.py \
  --producer-dir data/replay/original-pythtb-job \
  --wannier-dir data/replay/source-wannier-job \
  --output-dir results/replay/comparison \
  --job-metadata results/replay/job.json \
  --workflow-metadata results/replay/workflow.json \
  --download-manifest data/replay/download_manifest.json \
  --producer-core data/replay/original-pythtb-job/_pythtb_core.py \
  --with-dos --batch-size 64 --memory-mb 256
```

Bands use the saved fractional points, absolute energy reference, periodic axes
and interpolation settings. Missing WS tables can be reconstructed with
`queer.io.wannier.recompute_wsvec`, which retains unwrapped Cartesian centres,
all tied nearest supercell images and HR degeneracy weights. Its Cartesian
distance search checks that omitted images cannot be closer and stores orbital
images in compact numerical arrays. It can also normalize position operators
with the same weights. The model's own input arrays and WS construction storage
are additional to the execution working-memory budget.

The optional DOS comparison evaluates every original mesh point with the native
eigensolver. It preserves the historical histogram bins and discrete Gaussian
smoothing; this compatibility estimator differs from direct per-level Gaussian
broadening in `queer.observables.dos`. No implicit spin factor, fitted energy shift,
sign correction or grid resampling is added. Numerical arrays, checksums, coverage,
absolute/RMS errors, resource settings and producer provenance are saved as NPZ,
JSON and Markdown. A completed calculation can fail the requested comparison
tolerance; completion and agreement are reported separately. Band/DOS agreement
does not establish quantum-geometry or finite-field response accuracy.

Prefer the completed calculation's original staged inputs as a coherent snapshot.
Source-job artifacts may have been collected at different times: matching an HR
checksum alone does not establish that its centres or reference bands are the
ones used in the historical calculation. The parameter file and optional producer
core are verified against recorded SHA256 values; the core is read for provenance
and is never imported or executed. DOS-at-Fermi diagnostics explicitly distinguish
the historical first nonnegative bin, nearest bin and linear interpolation.

### Replaying saved response calculations

Retrieve saved scientific inputs and output tables through the Generator's
configured SSH broker when a historical HPC directory is outside the public
artifact inventory. Use the authenticated user's usable host configuration and
reuse its existing SSH session. Record remote paths, full-content SHA256 hashes,
resolved inputs, completion logs and producer source fingerprints alongside the
downloaded files. The replay utilities operate offline and never execute the
third-party producer. A current source checkout alone does not establish the
version used when an old result was generated.

`scripts/validation/generator_response_replay.py` handles PyATB's single-value
AHC prose and its 18-component shift-current table, using an explicitly verified
SCFOUT checksum and resolved `Out/input.json`. Native workflows retain resumable
RunStore checkpoints and resource limits. `generator_shift_replay.py` additionally
compares every frequency/component in the recovered GaAs benchmark cases and
records the source's estimator differences.

The adapter accepts historical `engine=pyatb` as the direct/Kubo family, with
`method` selecting the calculation, and `n_occ=m_unocc=-1` as unrestricted
occupied-to-unoccupied transitions. Hall exports order their pseudovector as
`yz,zx,xy`; shift-current exports contain the 18 symmetric-input components
`xxx,xxy,xxz,xyy,xyz,xzz,yxx,yxy,yxz,yyy,yyz,yzz,zxx,zxy,zxz,zyy,zyz,zzz`.

Use `generator_omx_replay.py` for a saved OMXsigmaxy two-dimensional Hall scan:

```bash
python scripts/validation/generator_omx_replay.py \
  --scfout data/replay/test.scfout \
  --expected-scfout-sha256 VERIFIED_SHA256 \
  --recipe data/replay/test.in \
  --reference data/replay/test_AHC.dat \
  --output-dir results/replay/omxsigmaxy --workers 4
```

It preserves the original energy recipe, including its explicit Hartree
conversion, and compares every original energy row in e²/h. Add `--resume` to
reuse unchanged completed checkpoints. All utilities report calculation
completion separately from agreement. Response comparisons and explicit
finite-step metric checks return failure when their tolerances are exceeded.
A fixed-grid comparison does not establish continuum convergence.

`generator_metric_replay.py` projects the native Cartesian metric into the
producer's normalized reciprocal directions. Supply the actual `--grid` and
`--dimension`, the saved average and integrated tables, and `--runtime-log` or
`--runtime-input-json` when available. `--binding inferred` records an unverified
historical Hamiltonian association even when its mesh is recovered from a log.
The differential metric and finite-step determinant distance have distinct
definitions, so their comparison reports acceptance as JSON `null`. Optional
`--finite-step` additionally evaluates determinant distances with native basis
links and reports a numerical pass/fail. First-order links can have large
finite-step errors; using the same mesh and determinant formula alone does not
establish operator equivalence with the producer.
Select `--finite-step --finite-basis-link centre_resummed` to evaluate the
audited midpoint reference approximation with exact orbital-centre phases.
It derives centres from native home-cell position-operator diagonals and uses
native overlap and position matrices. This explicit reference estimator retains
the native first-order tangent without changing the physical differential
metric. Its finite-step value depends on the declared orbital basis, and source
fingerprints still need to distinguish an audited checkout from an unverified
historical generation revision.

QUEER's physical shift-current calculation uses unshifted-Hamiltonian geometry
and applies scissors to transition energies. The audited historical PyATB
producer applies scissors inside derivative kernels, regularizes small
denominators, and truncates its Gaussian tails. Those choices can differ
substantially near small band splittings. Replay reports record these scientific
differences; comparison tolerances are not adjusted to hide them. A saved metric
must also declare its finite-difference steps and Cartesian or reciprocal
directional basis before comparison with the native finite-basis interband
metric.

Implementation is independently written under Queer's MIT license. Redistributed
HPKOT scientific tables retain their MIT license and attribution in
`queer/resources/symmetry/`. No ASE or HamGNN implementation was copied.
