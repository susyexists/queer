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
| `fermi_energy`, `fermi_energy_unit=Ry` | PyATB's eV (its Rydberg is 13.605698066 eV) moved to QUEER's Hartree scale: `E * HARTREE_EV / 27.211324570273` |
| FHS `fhs_energy_window` | Fermi-relative bounds; upper-exclusive count scan, offsets on the same scale |
| FHS `fhs_dimension` | Selected Hall components; bulk conductivity remains S/cm |
| `hall_convention` (default `auto`, `openmx` for OpenMX input) | `hall_export_sign`: -1 for `pyatb` with the direct/Kubo engine, +1 for `openmx`, `fhs`, `auto` and the FHS engine |
| Shift `eta` | Legacy Gaussian width divided by √2 for native standard deviation |
| `n_occ`, `m_unocc` | A single one-based occupied/unoccupied band pair; native indices are zero-based |
| `occ_band` | Occupied count and count-based scissors boundary |
| Shift methods `0`, `1` | Audited legacy sum-over-states computation mapped to native analytic derivatives |
| Threshold `adaptive_grid_threshold` | Curvature threshold in Å², unconverted; 1e30 or more turns the refinement off (plain Gamma-origin grid) |
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

Expected filenames include `Out/AHC/ahc.dat` (PyATB's one-line vector, or its FHS
table), `Out/Shift_Current/shift_current.dat`, `Out/Chern_Num/chern_number.dat`
(PyATB's one-line value),
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

`queer.observables.geometry.berry_plane` evaluates the Berry maps of the
PYTHTB_BERRY step at fractional k on a reciprocal plane (a, b). With an occupied
band count it gives that manifold's Kubo curvature, as
`berry_curvature(engine="kubo")`, and optionally the in-plane trace
tr g - n·g·n of its finite-basis interband metric, as `quantum_metric`. Without
one it gives the Generator's Fermi-sea curvature
Ω = -2 Im Σ v^a_nm v^b_mn/(E_n - E_m)² over n below and m at or above E_F, with
pairs closer than 1e-6 eV skipped (`queer.physics.geometry.fermi_sea_curvature`).
Its velocities come from `queer.models.evaluation.velocities`, which include the
i[H, τ] term of the centres and equal PythTB's cell-periodic velocities; the bare
`derivatives()["H"]` lacks that term and gives a different curvature. Reduced
quantities use b = 2π inv(A)ᵀ. Ω_red = b_a·Ω·b_b is dimensionless, the plane's
Chern number or Fermi-sea integral is Σ Ω_red/(n²·2π) over an n×n mesh
(`plane_flux`), Ω_red/|b_a×b_b| is the Cartesian curvature in Å², and the
quantum weight is Σ tr g·|b_a×b_b|/(n²·2π) (`quantum_weight`).
`rotate_velocity` keeps the Generator's rotation of orbital-basis velocities into
row eigenvectors.

`queer.physics.slab` builds the PYTHTB_SLAB Hamiltonians. `layer_blocks` sums
H(R) e^{2πik∥·R∥} into inter-cell blocks B_d by the offset d along the finite
axis, dropping hops along an axis that is neither finite nor periodic, and
`slab_hamiltonian` places them block-Toeplitz over whole cells,
H[w, w + d] = B_d, with no change of termination.
`queer.observables.surface.slab_eigensystem` solves one slab for its levels with
E - E_F in (-w, w] (`scipy.linalg.eigh` with `subset_by_value` and the `evr`
driver, E_F taken off the home-cell block). `slab_edge_spectrum` adds each
degenerate cluster's total probability on the first and last edge cells. That
cluster sum is the physical edge weight and does not depend on the basis inside
the cluster. The Generator's per-state weights first rotate each cluster (levels
closer than 1e-5 eV) to diagonalize the first-edge projector; inside a cluster
whose edge projector is degenerate they still depend on the basis.

The Generator's edge-crossing count (`queer.compat.generator.estimators.edge_crossings`,
the driver's rule since core f4a73779…) is a weighted spectral flow checked against the
bulk. Every level carries its weight on the first half of the cells, P_L (an odd ribbon's
middle cell counts half; P_R = 1 - P_L), and on the middle half, I (cells n/4 to 3n/4,
cut cells by their overlap), from `half_ribbon_weights`, which rotates each degenerate
cluster to diagonalize the first-half projector. Both are stored per level in `Out/{kind}_bands.json`
(`left_half_weight`, `interior_weight`). A level with I ≥ 1/4 is bulk-like.

The bulk check (`slab.bulk_at_fermi`, `bulk_row` per in-plane k and `bulk_summary`) solves
the bulk H(k) = Σ_R e^{2πik·R} H(R) at the ribbon's in-plane k-points on a mesh along the
finite axis of `bulk_k_perp_points` points: a whole multiple of twice the cells, and at
least the ribbon's own k-points, so the finite axis is resolved as finely as the ribbon
resolves its periodic one (`k_points`: [in-plane, finite]). It reports `metal` when the
number of levels below E_F varies, otherwise the bulk gap around E_F (`gap_eV`) and, over
a ribbon's closed zone, the Chern number of the bands below E_F with the periodic axis
first (`chern`): the winding along the zone of their Berry phase around the finite axis,
-Σ arg det of neighbouring overlaps, with its largest step (`berry_phase_step`, trusted
below π/2 as in TOPOLOGY). A resolved left net equals it.

The in-plane mesh is offset by half a step and never samples k∥ = 0 or 1/2, so a band
extremum between samples lies beyond them (`bulk_overshoot`). Near an extremum a band is
convex or concave: over the half step to it a band changes by at most half its larger
step to a neighbouring sample, and a sample's bound adds both axes. The steps show only
the hoppings the mesh resolves (fewer than half its points per period); the rest add their
Weyl bound 2π Σ|R| ‖H(R)‖ per point (`bulk_aliasing`; on 2 in-plane points every
nearest-neighbour hop). `overshoot_eV` is [valence, conduction], the most any band below
E_F can rise above the highest sample below it and any band above fall below the lowest
sample above it; `local_overshoot_eV` is [steps, gap] of the band pair across E_F whose
bounds at one point come closest to half their own gap there. The mesh resolves the gap
(`mesh_resolution`) when E_F lies inside the gap the overshoot leaves (the guaranteed
gap), the two overshoots together, v·dk, are at most half the sampled gap, and every band
pair's are at most half its own gap. The true gap is then at least v·dk and at least half
the sampled one, and a gapped Dirac point puts at most 2·arctan(v·dk/gap) ≤ π/2 of Berry
flux into one strip of the winding (π/3 under the pair bound), the bound the Berry-phase
step must meet: same-sign Dirac points sharing a strip fail that step instead of wrapping
to a small one, and the pair bound catches them away from the gap edges. Otherwise the
check `k_mesh` fails and neither the gap nor the Chern number is trusted. Still missed: a
band feature narrower than one step that the neighbouring steps do not show, other than a
convex or concave extremum, and five or more same-sign Dirac points in one strip, each
near the bound.

Per step k_j → k_j+1 (a ribbon's last point back to its first) the levels are ranked from
the reference energy, and the offset o that minimizes the mean step over the ranks -3…2
says how many crossed and which way; the offsets tried lie within 3 of 0 and within 3 of
the change in the number of levels below the reference (`_offsets`), so any number of
copies crossing in one step is followed, and a level without a partner costs its
distance to the window edge it must have crossed (`align_ranks`). The flow adds the P_L
that moves through the reference over the levels taking part (`taking_part`): the crossed
ranks; with o = 0, ranks -1 and 0 when each could end where the other ends within a
step's reach (two levels crossing each other at the reference within one step, which
energy order hides); then every neighbour that could swap with an end of that set. The
reach is 1.5 times the largest step of an aligned pair seen (`step_shift_eV`). So a rising
level adds the mean P_L of its endpoints, two chiral modes crossing at a TRIM add one
each, and |C| = 2 or 4 copies count with no swap repair. The right flow is the net level
count minus the left, and both close to zero around the zone. A level within 1e-5 eV of
the reference (the weights' degeneracy tolerance) counts as at it, as one exactly at it
does, so a flat band at E_F does not cross it back and forth with rounding noise. Per
step, each edge's crossings count the levels taking part with more than half of them on
its half that end on the other side of the reference, as many exchanges as those levels
allow (`exchanges`): two helical modes of one edge crossing in the same step count one
each way. They are upper bounds: two levels of one edge that only pass near the reference
in a step count so too, since energies alone cannot tell the two pairings apart (two
Kramers pairs per edge with E_F midway between their Dirac points read 4 up and 4 down for
2 and 2). Their difference is the net.

The flows are evaluated at E_F and at 9 energies evenly across the guaranteed gap within
the window (`window_eV`), a reach inside the sampled edges, past the ribbon's own levels
there (`per_energy`: `left_flow`, `right_flow`, `bulk_flow`, the last weighted by I); a
sample within 1e-5 eV of E_F is E_F's row. The status is `resolved` when every flow lies
within 0.1 (`tolerance`) of the same integer and that integer is the bulk Chern number: a
pair of edge modes then holds at most 5% of its weight on the far half. A gap that holds
no edge level, which nothing crosses, is `resolved` on a trusted bulk Chern number of 0
(the mesh resolves the gap and the Berry-phase step is below π/2): "the bulk Chern number
is 0 and no level crosses the gap"; a nonzero count is never certified from absence. It
is `metal` when the bulk bands cross E_F, `open_path` for a slab's surface path, and
`unresolved` otherwise, with a `reason` and the failed `checks`, each with the energies it
failed at (none for a check of the whole gap): `k_mesh` (the bulk mesh does not resolve
the gap, above), `hybridised` (a level inside the bulk gap,
between the lowest and highest energy evaluated, that is bulk-like or shared by the two
halves, `talking_levels`: inside a bulk gap only edge states live, so the edges talk
across the ribbon; the reason gives the finite-size gap they open where sampled), `chern`
(a clean flow other than the bulk Chern number: a mode hybridized out of the gap, which
leaves nothing in it to see), `berry_step` (a Chern number the bulk mesh does not
resolve), `gap_edge`, `absence` (no edge level in the gap without a trusted bulk Chern
number of 0), `strays` and `loose` (a level entering the window beside the levels
crossing), `touches`, `open_loops`, `fractional`, `varying`, `interior`, `empty_window`
(certified only on a trusted bulk Chern number of 0) and `unfollowed`. `left` and `right`
hold the crossings at E_F and `net` the rounded flow, certified only when `resolved`.

Validated on numpy ribbons against the FHS Chern number on fine meshes (analytic where
the gap lies below that mesh's resolution): 374 named cases and 7416 sweep ribbons (plain
Haldane and QWZ grids; random perturbed 2- to 6-orbital models with dangling edge
orbitals), with no wrong `resolved` count and no insulator called `metal`, and 13 440
bulk checks of models with up to eight Dirac points of one sign per line, besides steep
Dirac points away from the gap edges, with no trusted wrong Chern number. The hybridized
ribbons that the gap of the ribbon's own bulk-like levels used to certify wrongly
(Haldane with Δ = 0.45, t2 = 0.1 at 12 cells, and similar at 6 to 11 cells; a C = -2 pair
with one mode out of the gap; a chiral mode beside a dangling edge band) are unresolved;
the Haldane case resolves +1 from 24 cells and the C = -2 pair from 80. So are those that
core d75fb8fc… still certified wrongly: two Dirac points of one sign on a line the mesh
never samples (QWZ cut along (1, 1) at m = -9 meV with 3 eV hops, whose sampled gap was
±47.9 meV for a true ±9 meV and whose winding missed a unit; QWZ with a two-cell hop; an
armchair Haldane ribbon), 2 in-plane points and a pocket between samples, all `k_mesh`;
the (1, 1) cut resolves -1 from 200 k-points at m = -50 meV and from 400 at -30 meV. A
trivial gap with no level in it now resolves 0 on a trusted bulk Chern number. QWZ at
m = -1.9 still resolves from 34 cells at 200 k-points. The drivers up to core 56fda912…
matched each edge state to the nearest-energy state of the same edge, unbounded and many
to one; `legacy_edge_crossings` keeps that rule for comparing their references, and
neither the rank-continuity count of core 93167386…, the first weighted flow of core
dbdf34d1… (gap from the ribbon's bulk-like levels, no Chern check) nor the bulk-gap check
of core d75fb8fc… (the sampled gap, its Chern number trusted on the Berry-phase step
alone) was released.

`queer.compat.generator.estimators` keeps the Generator's own estimators with
their exact semantics. The DOS histograms the eigenvalues on bins from
min(E) - 2 eV to max(E) + 2 eV in `dos_step_eV`, divides by N_k and smooths with
a Gaussian sampled on the bins, truncated at ⌈5σ/step⌉ bins, normalized to unit
sum and applied with `np.convolve(..., "same")`; each Wannier band counts as one
state. DOS(E_F) interpolates linearly between bin centres, or, for the first
driver revisions, takes the first bin whose centre is at or above E_F
(`dos_at_fermi(..., "first_nonnegative_bin")`). `BandEdges` and `classify_gap`
pick a gap above 1 meV within `fermi_tolerance_eV` of E_F, the most central one
winning. The module also holds the band.dat window (frozen, else outer, else
±2 eV) and deviation metrics, the wannier90-style path sampling, `k_distance`
(with 2π), `plane_frame` (without 2π), the surface paths, the orbital species of
the nearest atom image, centres unwrapped across a slab's vacuum and the
hopping-decay fit.

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

### Parity comparator

`queer.compat.generator.parity.compare(step, reference_dir, candidate_dir)` judges a
candidate run of a Generator Wannier step (BANDS, TOPOLOGY, BERRY or SLAB) against a
reference run of the same step and returns a JSON-able report; `render_markdown`
renders it. Each directory holds one results document (`queer_results.json` or the
PythTB driver's `pythtb_results.json`), optionally its parameter file
(`queer_params.json` or `pythtb_params.json`) and the step's `Out/*.json` data files.
Model files are never read. The OpenMX families (AHC, CHERN_NUMBER) compare a QUEER
run with a PyATB run directory; see [OpenMX steps of the PyATB
counterpart](#openmx-steps-of-the-pyatb-counterpart).

```bash
python -m queer generator compare --step QUEER_BERRY \
  --reference runs/pythtb-berry --candidate runs/queer-berry \
  --report parity.json --markdown parity.md
```

The command prints the verdict and every failed check, writes the requested
reports and exits 0 for `pass` or `pass_with_documented_differences`, 2 otherwise.

**Preconditions.** A failed precondition makes the verdict `not_comparable` and
nothing is compared:

- one readable results document per directory, both of the step's family
  (`PYTHTB_X` and `QUEER_X` are one family);
- the same source WANNIER job when both runs name one;
- equal SHA-256 digests in `provenance.inputs` for every file both runs recorded.
  Without recorded inputs, a model fingerprint must match and the report sets
  `inputs_verified_by_fingerprint`: `num_wann`, the lattice to 1e-9 Å, `n_R`,
  `n_R_after_ws`, `n_R_model`, `folded_axes`, `periodic_dirs`, the Wigner-Seitz
  mode and its scores to 1e-6 meV, and the `_band.dat` deviations to 1e-6 meV;
- equal effective step parameters when both parameter files are present: the step's
  keys with the driver defaults, ignoring `workers`, `chunk_size`, `time_budget_s`
  and the order of the structure's atoms. When either file is missing, equal
  `params_sha256` digests are reported, otherwise the check is skipped;
- the same Fermi energy and the same derived decisions: the Wigner-Seitz mode, metal
  or insulator, the occupied band count and its isolation, the occupation mode, the
  topology's applicability, its planes and their transverse sampling, and the slab
  kind, cells and finite axis. A threshold decision that flipped is reported as
  not comparable, not as a failure.

**Reference semantics.** A registry keyed by the producer's `core_sha256`, and the
reference's own structure where it tells, adapt the comparison to the revision that
wrote the reference:

| Producer | Rules |
|---|---|
| driver 9f43886 (`f6c9b7af…`) | DOS(E_F) compared with the candidate's DOS recomputed under the first-nonnegative-bin rule; Fermi surface by its crossing bands only (no plane frame); localization not compared; the DOS mesh recorded on all three axes is read at the candidate's mesh axes; inputs verified by fingerprint |
| a SLAB reference whose `edge_crossings_at_fermi` records no `method` (every driver up to core 56fda912…, whatever its digest) | edge-crossing counts compared with the nearest-energy count recomputed from the candidate's own levels; the status and flows, which it did not record, are not applicable; the candidate's weighted flow is checked only against its own levels and weights (`agree_after_convention`, never `agree`): its inputs are unverifiable against such a reference |
| a fermi-mode BERRY reference without `berry.conventions` | written before the sign fix: its maps, path curvature and integrals are compared after a factor -1 |
| QUEER and the vendored driver | current semantics; compared as recorded |

A difference removed by one of these rules is classed `agree_after_convention`.

**Tolerances.** They follow from the data, never from the material. With δ_W the
`hermiticity_weyl_bound` that QUEER records in the model block (0 when absent):

| Quantity | Rule |
|---|---|
| Band eigenvalues | τ_E = max(1e-8 eV, δ_W) + 1e-13 max\|E\| |
| Gap, band edges, isolation, Kramers splitting | 2 τ_E; a band edge found at another k-point with an energy within 2 τ_E is a documented tie, and so is the `is_direct` flag it flips when both runs' smallest direct gap equals their gap |
| DOS, PDOS, DOS(E_F) | 1e-9 of the row's largest value. When δ_W > 0, eigenvalues may cross bin edges: a difference that conserves the states and stays within an L1 budget of 2 (N_bands N_k 2τ_E/ΔE + 1)/N_k is documented |
| Fermi-surface contours | the same crossing bands; symmetric Hausdorff distance ≤ max(1e-9 \|b\|, 1e-6) in the stored coordinates (1/Å without 2π) |
| Chern, Z2, convergence, crossings | equal, per plane matched by (fixed axis, value) |
| Wannier charge centres | optimal circular matching ≤ 1e-9 per loop; Berry phases modulo 2π |
| BERRY, bands mode | each map ≤ 1e-8 max\|X\| of its own quantity; Kubo Chern number ≤ 1e-8; quantum weight ≤ 1e-6 relative |
| BERRY, fermi mode | the flagged set F holds the points where the maps disagree beyond 1e-8 max\|Ω\|: a level within τ_E of E_F, or an occupied-empty pair at the 1e-6 eV skip threshold, decides such a point. Its size is reported, it must be a minority of the points carrying curvature, and the integral must agree to 1e-8 + Σ_F (\|Ω_Q\| + \|Ω_P\|)/(n² 2π). Each document's integral must equal its own map |
| Path curvature | pointwise ≤ 1e-8 max\|Ω\| |
| SLAB levels | τ_E per k-point; a level within τ_E of the window edge may be missing on one side |
| SLAB edge and half-ribbon weights | left and right, first-half and middle-half weights summed over clusters of levels within 1e-5 eV ≤ 1e-9; clusters within 1e-5 eV of the window edge are excluded |
| SLAB bulk check | the same mesh (`k_points`) and `metal` flag (a flip with a gap edge within 2 τ_E of E_F is documented); the bulk gap ≤ 2 τ_E; the overshoot of its edges and how close the closest band pair comes to the mesh's bound (its steps less half its gap) ≤ 4 τ_E (each a level plus half of two steps); the same Chern number (a difference where a run's Berry-phase step lies within 1e-6 rad of π is documented) and Berry-phase step ≤ 1e-6 rad |
| SLAB edge crossings | the candidate's block equal to the flow recomputed from its own stored levels and weights, always. Against a weighted-flow reference: counts at E_F equal; status, method, tolerance and the failed `checks` equal (a reason that differs only in the values it prints is documented); flows ≤ 1e-9 × the larger stored level count; gap, step shift and sampled energies ≤ 4 τ_E (each is a level plus 1.5 differences of two). A difference is documented only where a decision of either run lies within tolerance of flipping there (`crossing_margins`, on both runs' levels): at an energy, an offset cost, a swap test of the set or its extension, a stray's or an unpartnered level's test against the reach, any level's distance from the reference the flow is taken at (the energy less 1e-5 eV), or a sample's distance from E_F against 1e-5 eV, ≤ 5 τ_E; a middle-half weight of a level taking part within 1e-6 of 1/4 or its first-half weight within 1e-6 of 1/2; a flow within the flow tolerance of a half-integer or of 0.1 from an integer. For a check of the whole gap its own decisions: E_F against the reach inside the sampled edges and the guaranteed ones (`gap_edge`), a non-bulk-like level entering or leaving the gap (`absence`), each level of the `hybridised` test against the gap, the evaluated span, 1/4 and its cluster's bound, each cluster cut against the reach, and any level in the gap against its edges, the flows against a half-integer (`varying`, `chern`), the bulk Berry-phase step against π/2 and π (`chern`, `berry_step`), the mesh's resolution (`k_mesh`: the two overshoots against half the sampled gap and the closest band pair's steps against half its gap, ≤ 10 τ_E, each guaranteed edge against E_F, ≤ 5 τ_E; also for `chern`, and for `absence` and `empty_window` with the Berry-phase step where the bulk Chern number is 0), and the guaranteed gap against empty (`guaranteed`). A status difference is documented only when every check that differs between the runs has such a decision, at the energies where it differs, or an alignment that sets the step shift could give another step shift (`step_shift`: a near-tie of offset costs or a level at E_F or at a provisional energy, giving a larger step or losing the one step that sets it), or, without a bulk gap, a middle-half weight that sets the gap lies within 1e-6 of 1/4 (`gap`), or the guaranteed gap could empty (`guaranteed`): each moves every sampled energy; or the bulk checks' metal flags flip at a gap edge. A check that differs only at a row one run has and the other has not is documented where both runs' `rows` decisions are near. The counts at E_F are documented on the decisions at E_F and on `step_shift`, `gap` and `guaranteed`. Sampled energies that differ are documented on the same grounds (status, `step_shift`, `gap`, `guaranteed`), and a different number of rows only where both runs place a sample within 5 τ_E of 1e-5 eV from E_F or of another sample, or the sampled range within 5 τ_E of empty (`rows`, both runs), or on `step_shift`, `gap`, `guaranteed`, a metal flag that flips at a gap edge, or a level at the window edge (`empty_window`). Against a nearest-energy reference: its counts equal the recount, documented only where moving the 1/2 threshold by 1e-6 changes the recount |
| Model and validation blocks | float leaves ≤ 1e-6 relative (the worst is reported), other leaves equal; producer keys are skipped |
| Warnings | the same codes: `warning_codes` when both runs record them, else codes classified from the messages. Reworded messages with the same codes, such as "PythTB bands" and "Model bands", are documented; plot warnings depend on the plotting environment and are not compared |

**Classes and verdicts.** Every quantity is `agree`, `agree_after_convention`,
`differs_documented`, `differs_unexplained` or `not_applicable`. The verdict is
`not_comparable` when a precondition fails, `fail` when any quantity
`differs_unexplained`, `pass_with_documented_differences` when any
`differs_documented`, and `pass` otherwise. When both runs failed, equivalent failure
codes (`pythtb_oom` and `queer_oom`) pass. A document whose blocks do not have the
schema's shape fails the check that reads it instead of stopping the comparison. The
report also lists the preconditions, the semantics profile, every flagged set and the
quantities closest to their tolerance.

**Goldens.** `tests/fixtures/generator_goldens` holds the driver's results on six
analytic cases (Haldane bands with a Fermi surface, Chern topology, bands-mode and
fermi-mode Berry curvature and a ribbon; Kane-Mele Z2). The case module
`tests/numerical/generator_golden_cases.py` rebuilds their inputs byte for byte,
and the tests require every golden to pass against a fresh QUEER run with all input
digests equal. `scripts/validation/generate_generator_goldens.py` regenerates them
with the vendored driver under `uv run --no-project --python 3.12 --with
pythtb==2.0.2 --with scipy --with matplotlib`.

### OpenMX steps of the PyATB counterpart

`QUEER_AHC` and `QUEER_CHERN_NUMBER` (aliases `PYATB_AHC`, `PYATB_CHERN_NUMBER`) run
the calculation of PyATB's `AHC` or `CHERN_NUMBER` block on the staged OpenMX scfout
with QUEER's own model (`queer.compat.generator.openmx_steps`):

```bash
python -m queer generator run --step QUEER_AHC --params queer_params.json
```

**Parameter file.** The Generator writes `queer_params.json` beside the scfout
(`queer.compat.generator.openmx_params`):

| Key | Meaning |
|---|---|
| `step_type`, `counterpart_step_type` | `QUEER_AHC`/`PYATB_AHC` or `QUEER_CHERN_NUMBER`/`PYATB_CHERN_NUMBER` |
| `input_parameters` | PyATB's INPUT_PARAMETERS as written to its `Input`, without `package` (always OPENMX) and `scfout_route`: `nspin`, `fermi_energy`, `fermi_energy_unit`, `hall_convention`, `max_kpoint_num`, `sparse_format` |
| `parameters` | the function block as written to PyATB's `Input` |
| `workers` | processes; absent or null: every CPU of the allocation |
| `scfout` | the staged scfout relative to the workspace (default `scf.scfout`) |
| `self_convergence` | `true` also evaluates a refining integration on the doubled grid |

Unknown keys, `package`, `scfout_route` and values PyATB refuses fail with
`queer_bad_params` before any calculation, as do an `nspin` the scfout's
`SpinP_switch` contradicts (4 with spin-orbit coupling, else 1) and a zero
`relative_error` or `absolute_error` of an `Adaptive` integral (PyATB never meets
a zero tolerance and runs to its generation cap).
`params_from_pyatb_input` builds the file from a parsed PyATB `Input` (or, with
`echo=True`, from PyATB's resolved `Out/input.json`) and always writes
`hall_convention`: a missing key is `auto` in an `Input` and `pyatb` in an echo,
which names the key whenever PyATB has one, so an echo without it comes from an
older PyATB that printed its native sign.
`workers`, `max_kpoint_num` (a cap on the k-points of one task) and
`sparse_format` never change a result.

**Conventions,** measured against PyATB (build 60fc6b2a) on an OpenMX scfout:

| Quantity | PyATB | QUEER |
|---|---|---|
| Hartree | 27.211324570273 eV | CODATA `HARTREE_EV` |
| `fermi_energy Auto` | `ChemP * 27.211324570273` | `ChemP * HARTREE_EV`; `chemp_hartree` recorded |
| explicit `fermi_energy` (eV; Ry at 13.605698066 eV) | sampled at `E / 27.211324570273` Hartree | sampled at `E * HARTREE_EV / 27.211324570273` eV; both values recorded |
| Hall sign of the direct/Kubo engine | native: minus the physical value; `openmx`, `fhs`, `auto` (OpenMX input) flip it; the prefactor carries the signed reciprocal volume, so a left-handed lattice flips it again | physical `-(e²/ħ) ∫ f Ω`; `Out/AHC/ahc.dat` holds `export_sign` times it: the sign table's factor times the lattice's handedness (`model.lattice_handedness`) |
| Hall sign of the FHS engine | physical, whatever `hall_convention` says | physical |
| FHS links | `W(k)^+ [sum_R exp(2πi k.R) (S_R + i dk.r_R)] W(k+dk)`: the raw position operator with `+i dk.r`, the opposite sign of the first-order term of the cell-periodic overlap `<u_k|u_(k+dk)> = <psi_k|exp(-i dk.r)|psi_(k+dk)>`, at the link's start; leading minors without pivoting. Inside a band this offsets the plaquette sum from the Fermi-sea flux by an amount that does not shrink with the grid (60fc6b2a on the kagome scan: up to 0.15 S/cm above the Fermi-sea sum on the same nodes, and at -2.12 eV 0.119 S/cm above its own direct engine) | `W(k)^+ [S(m) - i dk.A(m)] W(k+dk)` at the link's midpoint m (`basis_overlap`), A the hermitian part of the position operator: exp(-i dk.r) to first order; the same integer in a gap, and inside a band the Fermi-sea flux, which its FHS meets |
| FHS Cartesian output | `sum_d (C_d e²/h 1e8/abs(a_d)) normalize(b_(d+1) x b_(d+2))`, the Hall vector only for orthogonal lattice vectors | `sigma_S_cm` the physical `-g (e²/h) sign(det A) sum_i C_i b_i/2π`; `pyatb_construction_S_cm` PyATB's construction, which `Out/AHC/ahc.dat` holds |
| Chern number | plane flux over 2π, no Hall sign | the same orientation |
| Grid nodes | `k_start + Σ v_i n_i / N_i`, upper end excluded | the same nodes (`kmesh`) |
| Grid refinement | a node whose occupied curvature norm reaches the threshold (Å², raw, before prefactor and sign; for a Chern number its normal component) is replaced by an `adaptive_grid` sub-grid starting half a cell below it | refinement off (threshold ≥ 1e30): the plain grid; otherwise QUEER's threshold quadrature on cell midpoints |
| Spin without spin-orbit coupling | `Hks[0]` only: the spin-up channel of a collinear SCF, no spin factor | the physical value, and `legacy_single_channel` (channel 0, occupied once) |
| Chern `integrate_mode Adaptive` | stops with an AttributeError | QUEER's cubature (a gapped plane) |

A 1 x 1 x 1 `adaptive_grid` does not switch PyATB's replacement off: a flagged
node is still replaced by the node half a cell below it.

**Integration.** A Grid with the refinement off is the plain Gamma-origin sum,
evaluated point by point with the functions `queer.observables.integrated.ahc`
sums (the same result to the last bit); each point's occupation facts give the
flagged set `F`: points with a level within `delta_EF` of E_F and, for the direct
engine, an occupied-empty pair in one spin channel closer than 1e-4 eV, where
`delta_EF = 1e-6 eV * HARTREE_EV / 27.211324570273` (the Fermi-energy agreement
allowed between the codes) `+ max(1e-8 eV, degeneracy tolerance)`. PyATB's direct
engine sets the coefficient derivative of such a pair to zero (`get_D`); its Kubo
engine sums every occupied-empty pair with no cut, so Kubo pairs are not flagged.
`per_k_flagged` lists each point of `F` (the largest first) with its contribution
`c_k` (S/cm per component; dimensionless for a Chern number), `c_pyatb_rule` (the
contribution with PyATB's cut, evaluated with QUEER's per-k functions),
`flip_allowance` (the curvature of the degenerate blocks within `delta_EF`, which
either code may occupy) and `allowance = |c_k - c_pyatb_rule| + flip_allowance`;
`flagged` sums them (`pair_shift`, `flip_allowance`, `allowance`). A refining Grid
runs QUEER's threshold quadrature and an Adaptive integral QUEER's cubature with
PyATB's tolerances (`absolute_error` in S/cm). Their nodes differ from PyATB's, so
their comparison needs `self_convergence: true`, which records `e_Q = |sigma(N) -
sigma(2N)|` (the grid doubled on its active axes; 4 or 8 times the cost); a
cubature also records its own embedded error estimate. PyATB writes no error
estimate for a Grid, so with `self_convergence` a refining Grid also records
`pyatb_emulation.refined_grid`: PyATB's own refined sum, evaluated with QUEER's
per-k curvature on the model PyATB reads (Gamma-origin nodes; a node whose
occupied curvature norm, in Å² on PyATB's Bohr radius, reaches the threshold is
replaced by the `adaptive_grid` sub-grid starting half a coarse cell below it;
PyATB's pair cut at flagged points; PyATB's e, ħ and Bohr radius) on the grid and
its double: `printed_S_cm`, `printed_doubled_S_cm`, `e_P_S_cm` (their difference,
PyATB's discretisation error), the refined node counts, the threshold ties and the
occupation windows. Their budgets: QUEER's quadrature holds about 1 kB per cell it
keeps (0.7-0.9 kB measured with every cell refined), and those cells may use half
of the planned 60% of the available memory (cgroup-aware, as for the Wannier
steps), so it may hold `0.5 * 0.6 * available memory / 1024` cells. A threshold
quadrature keeps only its coarse cells: the refined cells' sub-grids are evaluated
in batches of whole cells (at most 65,536 points) and each refined cell keeps its
mean, as the emulation of PyATB's refined sum keeps one mean per refined node. Its
budget is the request's worst case, every cell refined (`max_cells = coarse cells x
sub-grid cells`, enough points for all of them), so a refinement PyATB can run is
never refused for memory: PyATB's 3D Grid of 100^3 nodes with 20^3 sub-grids fits
in 8 GiB (with `self_convergence`, its 200^3 double from about 26 GiB). A cubature
keeps every cell it makes and gets all of the cells the memory holds; past them it
stops with `queer_validation_failed`. A coarse grid, or with `self_convergence` its
double, that needs more cells than the memory holds is refused before the model is
read (`queer_bad_params`, "exceeds QUEER's integration budget"); PyATB has no such
limit, so a comparison of either refusal is `not_comparable`. The FHS engine scans
PyATB's energies (the window offsets, upper end excluded) with QUEER's FHS
plaquettes; `fhs_dimension 0` evaluates the k1-k2 plane only. Its physical vector
carries `sign(det A)`: the plaquettes of family i are oriented by
`b_(i+1) x b_(i+2)`, parallel to `sign(det A) a_i`, so a crystal described by a
left-handed lattice keeps its Hall vector. Where an energy shows a varying occupied
rank at plaquette vertices (`integration.fhs.metallic_vertex_rank_variation`),
`pyatb_emulation.fhs_links` records PyATB's scan with its own links
(`printed_S_cm`, PyATB's e²/h and Bohr radius) and QUEER's midpoint links on the
same eigenvectors (`queer_links_S_cm`), and `ahc.fermi_sea_check` records QUEER's
own physical check of its FHS at those energies: `fermi_sea_S_cm`, its direct-engine
Fermi-sea sum FS(N) on the FHS grid's nodes (a level below the energy is occupied,
the count a plaquette vertex takes) as the evaluated plane families see it
(`sum_d (sigma . a_d / 2 pi) b_d`), and `fhs_half_grid_S_cm`, its FHS on the halved
grid FHS(N/2); null at the other energies, and `error` when the Fermi-sea sum
cannot be evaluated (a connection that violates dipole covariance with the
overlap). QUEER's midpoint links converge to the direct engine's curvature, which
holds the basis curvature the Kubo sum leaves out. A Chern number on a plane that
is metallic at E_F is the Fermi-sea flux over 2π (PyATB's definition) with the
warning `chern_plane_metallic`; a refining Chern integral requires a gapped plane.

**Results.** `queer_results.json` holds `engine`, `step_type`,
`counterpart_step_type`, `provenance` (QUEER version, `queer_record_sha256`, the
bytes and SHA-256 of `scf.scfout` and `queer_params.json` in `inputs`, the parallel
plans), `model` (`scfout_sha256`, `spin_mode`, `spin_p_switch`, `nspin`,
`fermi_energy` sampled, `fermi_source`, `fermi_energy_pyatb_eV`, `chemp_hartree`,
both Hartree values, the lattice and volume), `ahc` (`mu_eV`, `mu_pyatb_eV`,
`energy_offsets_eV`, `sigma_S_cm {yz, zx, xy}` per energy, `at_fermi`,
`pyatb_construction_S_cm` (FHS), `convention {requested, resolved, sign_table,
lattice_handedness, export_sign}`, `legacy_single_channel`, `per_k_flagged`,
`flagged`, `raw_curvature_to_S_cm`, `pyatb_emulation` with `refined_grid` or
`fhs_links`) or `chern` (`value`, `integer`, `gapped`, `method`,
`integrate_mode`, `occupied_bands`, `plane`, `flux`, `legacy_single_channel`,
`per_k_flagged`, `flagged`, `pyatb_emulation`), `integration` (mode, refinement,
grid, threshold in Å², node placement, points, error estimate, `budget`,
`self_convergence`), `warnings` with their `warning_codes` and `outputs`.
`pyatb_emulation` holds comparison diagnostics only: PyATB's own schemes evaluated
with QUEER's per-k functions, recorded so a parity comparison can bound PyATB's
error or explain a difference. QUEER keeps its physical estimators as the result
(`sigma_S_cm`, `chern.value`) and never reports an emulated legacy value as one. The
workspace also receives PyATB's `Out/AHC/ahc.dat` or `Out/Chern_Num/chern_number.dat`
in PyATB's layout and requested sign, and neither `pyatb_results.json` nor
`manifest.json`. Failures write the same document with `status: failed` and
`error {code, message}`: `queer_bad_params` (also an integration larger than the
job's memory allows), `queer_missing_input` (the scfout is missing or cannot be
read), `queer_validation_failed` (a refining Chern integral on a metallic plane, a
cubature that outgrew the cells the memory holds), `queer_env_missing`, `queer_oom` and
`queer_internal_error`, with exit codes 2, 2, 2, 2, 3 and 1.

**Parity.** `compare` reaches `queer.compat.generator.parity_pyatb` for the
`AHC` and `CHERN_NUMBER` families. The reference is a PyATB run directory: `Input`,
optional `inputs.sha256`, `Out/input.json`, `Out/running.log`, the product and,
for an adaptive integral, `Out/AHC/integraton.log`. The command recognises such a
directory and needs no `--step`; `--scfout FILE` names the scfout to hash when the
reference has no `inputs.sha256`:

```bash
python -m queer generator compare --reference runs/pyatb-ahc --candidate runs/queer-ahc \
  --scfout runs/pyatb-ahc/scf.scfout --report parity.json --markdown parity.md
```

The preconditions are the family, the scfout digest (and the reference `Input`
against its recorded digest), the effective parameters (both sides read as the
runner reads them, text vectors and numbers included; PyATB's defaults applied,
keys of the other engine and integration mode and the execution-only keys left
out; `hall_convention` only for the direct or Kubo AHC, since the FHS engine
ignores it and a Chern number carries no Hall sign; an echo without
`hall_convention` is `pyatb`), then, for two completed runs,
the Fermi point `|E_F^Q / HARTREE_EV - E_F^P / 27.211324570273| <= 1e-6 /
27.211324570273` Hartree (E_F^P from the input echo, or the six decimals
`running.log` prints for `Auto`; the eV difference is `agree_after_convention`),
the resolved Hall convention, and an integration that can be compared: an adaptive
integral from at least 512 cells and an error estimate for a refining one.
PyATB's printed values are compared with QUEER's physical value times the sign
table's factor (`agree_after_convention` when it is -1) and, without spin-orbit
coupling, with `legacy_single_channel` (`agree_after_convention`):

| Tier | When | Rule per component |
|---|---|---|
| A | Grid with the refinement off in both runs | `abs(d sigma) <= base + sum_F flip_allowance`, `base = 2e-7 abs(sigma) + 5e-7 S/cm`; PyATB equal to QUEER's value with PyATB's pair cut at the flagged points (`sigma - pair_shift`) within the same bound is `differs_documented`, naming the points; agreement that relies on an allowance above `base` is `differs_documented`, never `agree` |
| B | a Grid threshold below 1e30, or Adaptive | outside Tier A, `differs_documented` only if `abs(d sigma) <= 3 max(e_Q, e_P)`; `e_P` from `integraton.log` (Adaptive) or from QUEER's emulation of PyATB's refined Grid sum, used only when it reproduces PyATB's printed value within Tier A (`pyatb_refined_grid`, else `differs_unexplained`); without `e_Q` or `e_P`, `not_comparable` |
| FHS | the FHS engine | the energies equal to the printed six decimals; QUEER's values in PyATB's Cartesian construction (`agree_after_convention` where it differs from the physical vector); every energy without a varying occupied rank (a gap) meets Tier A; inside a band, where PyATB's FHS carries an offset from the Fermi-sea flux, a gap is `differs_documented` with its size only when QUEER's emulation of PyATB's links reproduces PyATB and QUEER's own midpoint links reproduce QUEER, each within Tier A, and QUEER's FHS meets its own Fermi-sea sum at every such energy, `abs(FHS(N) - FS(N)) <= 3 abs(FHS(N) - FHS(N/2)) + base` (`ahc.fermi_sea_check`, with the numbers in the report's `tolerance_basis.fermi_sea_guard`; a converging FHS meets it, a grid-independent offset such as a defect in QUEER's connection does not); otherwise, and for a value that is not finite, `differs_unexplained` |
| Chern | gapped plane, both values within 0.05 of an integer | the integers equal; otherwise the floats meet Tier A with a 1e-9 floor (Tier B when refining) |

2e-7 covers the two codes' physical constants (PyATB's e and ħ differ from CODATA's
by up to 1.8e-7 in e²/ħ) and 5e-7 S/cm the six decimals PyATB prints. For an
Adaptive integral `e_P` is PyATB's last-generation error in `integraton.log`,
converted with `raw_curvature_to_S_cm`. For a refined Grid it is `|sigma_E(N) -
sigma_E(2N)|` of the emulation; `|sigma_P - sigma_E(N)|` is its fidelity, reported
with the refined node counts of both. A refined Grid without `self_convergence`
has neither estimate and is `not_comparable`: run it with the refinement off
(`adaptive_grid_threshold 1e30`) for Tier A, or set `self_convergence`. The pair
cut is applied as an exact shift rather than a window of its size, so a wrong
candidate cannot pass where the cut points dominate sigma. The candidate's own product file must hold what its
document says. Two failed runs pass when their failures are equivalent: PyATB's
are classified from its logs, Slurm's `slurm-*.err` and `slurm-*.out` included
(the Generator's batch script does not redirect PyATB's output; an nspin conflict
is `bad_params`, a memory failure `oom`). A QUEER refusal at a documented limit of
QUEER's (`queer_validation_failed`, or `queer_bad_params` for an integration
larger than the job's memory or a zero Adaptive tolerance) against a completed
PyATB run is `not_comparable` with the reason. The runner records the scfout's
size and SHA-256 before it translates the parameters, so a run that fails that
early is still compared on the same scfout; only a missing scfout leaves no
digest (`queer_missing_input`).

**Calibration.** On a kagome Chern insulator with spin-orbit coupling (101 kB
scfout; E_F -2.35 eV on its sigma_xy plateau, -2.12 eV in a band, -2.60 eV below
it) every Tier A case passes, direct and Kubo, in each convention: QUEER's 38.740461
and 33.739453 S/cm (direct; plateau and metal) against PyATB's 38.740468 and
33.739459 are the constant 1.7e-7 of the physical constants. The Chern numbers are
-1.0000000000000013 (QUEER) and -0.9999999999999967 (PyATB). On the refined Grids
(thresholds 10 and 20 Å², 2 x 2 sub-grids) PyATB refines the nodes whose occupied
curvature vector has a norm of at least the threshold in Å² (raw, before the
prefactor and sign): 1757 and 471 nodes, which QUEER's emulation of its sum
reproduces exactly, printing -38.724467275 and -38.725572875 S/cm (PyATB:
-38.724467 and -38.725573). PyATB's own grid error is e_P 0.0120 and 0.0112 S/cm,
larger than QUEER's e_Q (0.0048 and 0.0044 S/cm; QUEER's midpoint quadrature
refines 1754 and 460 cells), so both are `differs_documented` within 3 max(e_Q,
e_P) (|d sigma| 0.0096 and 0.0090 S/cm). On the metal (E_F -2.12 eV) PyATB's
plain 100 x 100 Grid relabelled to threshold 100 Å² (no node reaches it; the largest
norm is 21.5 Å²) prints 33.739459 S/cm, 0.0174 S/cm from QUEER's midpoint
quadrature (33.722013): 100 times QUEER's e_Q (1.7e-4 S/cm), so the rule without
PyATB's error failed this correct run; PyATB's own e_P (0.0123 S/cm; the emulation
reproduces its print to 1.9e-7) documents it.
The FHS scan (201 energies) agrees to Tier A at the 159 energies in a gap. At the 42
energies inside a band QUEER's FHS stays within 0.0002-0.042 S/cm of its direct
Fermi-sea sum on the same 100 x 100 nodes while PyATB's table lies 0.016-0.146 S/cm
above it (correlation 0.999998), and PyATB's own outputs show the same offset: at
-2.12 eV its FHS scan interpolates to 33.858 S/cm against its own direct 33.739459
and Kubo 33.734532 S/cm on the same grid. The offset is the sign of PyATB's
first-order link term: on the same eigenvectors, at the three energies with the
largest gaps, PyATB's links minus QUEER's give 0.131-0.134 S/cm on both the 100 x
100 and the 200 x 200 grid, and QUEER's midpoint links with the sign of their `dk.A`
term flipped reproduce PyATB's table to 1e-4 S/cm. The emulation of PyATB's links
reproduces PyATB's table at all 201 energies within Tier A (at most 5.0e-7 S/cm, its
six printed decimals), QUEER's midpoint links on the same eigenvectors reproduce
QUEER's values to 2.1e-13 S/cm, and QUEER's FHS meets its own Fermi-sea sum at all
42 band energies (largest 0.79 of `3 abs(FHS(N) - FHS(N/2)) + base`), so the gap, up
to 0.1325 S/cm, is `differs_documented`. A defect in QUEER's connection fails:
scaling QUEER's connection (the position operator it reads from the scfout) by -1 or
1.5 breaks its dipole covariance with the overlap, so the Fermi-sea sum cannot be
evaluated, and scaling only the `-i dk.A` term of the midpoint links, in the FHS and
its own-link emulation alike, leaves the Fermi-sea sum intact but fails the guard at
16 and 8 of the 42 band energies (largest ratios 22.6 and 23.9). Both passed as
documented without the guard. Sign-flipped, doubled, zeroed or NaN candidates fail
at the gapped energies. `tests/integration/test_generator_pyatb_calibration.py`
replays the twelve cases and the relabelled metal with `QUEER_PYATB_CALIBRATION=1`.

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
