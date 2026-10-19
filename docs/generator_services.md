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
the hoppings the mesh resolves (more than 8 points per period of their Bloch phase,
`CROSSING_ALIAS_POINTS`: with fewer, a band can turn between samples by more than its
steps to the neighbouring samples show); the rest add their Weyl bound 2π Σ|R| ‖H(R)‖ per
point (`bulk_aliasing`; on 8 or fewer in-plane points every nearest-neighbour hop). `overshoot_eV` is [valence, conduction], the most any band below
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
pair of edge modes then holds at most 5% of its weight on the far half. No count is
certified from absence: a gap that holds no edge level stays `unresolved` even when the
bulk Chern number is 0, because a Chern number from the ribbon's own mesh can be wrong
while the mesh looks resolved (an engineered C = 2 ribbon with strongly anisotropic Dirac
points read 0 at the default 200 k-points). It
is `metal` when the bulk bands cross E_F, `open_path` for a slab's surface path, and
`unresolved` otherwise, with a `reason` and the failed `checks`, each with the energies it
failed at (none for a check of the whole gap): `k_mesh` (the bulk mesh does not resolve
the gap, above), `hybridised` (a level inside the bulk gap,
between the lowest and highest energy evaluated, that is bulk-like or shared by the two
halves, `talking_levels`: inside a bulk gap only edge states live, so the edges talk
across the ribbon; the reason gives the finite-size gap they open where sampled), `chern`
(a clean flow other than the bulk Chern number: a mode hybridized out of the gap, which
leaves nothing in it to see), `berry_step` (a Chern number the bulk mesh does not
resolve), `gap_edge`, `absence` (no edge level in the gap), `strays` and `loose` (a level entering the window beside the levels
crossing), `touches`, `open_loops`, `fractional`, `varying`, `interior`, `empty_window`
and `unfollowed`. `left` and `right`
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
the (1, 1) cut resolves -1 from 200 k-points at m = -50 meV and from 400 at -30 meV. Core
f4a73779… also certified an empty gap on a bulk Chern number of 0 and counted a hopping as
resolved from 2 points per period, which let 105 coarse or clipped ribbons (4 to 8 cells,
or 4 to 6 in-plane points) and the engineered C = 2 ribbon read a wrong 0; they are
unresolved now, and the sweeps resolve 2466 of 5016 plain and 1035 of 2400 random
ribbons, all with the right count. Still missed: on a mesh far below the default, a band
that turns between samples by more than its steps show (a 4-band C = 1 model read 0 at 20
in-plane points; it is unresolved at 200 and resolves +1 at 400). QWZ at
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
candidate run of a Generator Wannier step (BANDS, TOPOLOGY, BERRY, SLAB or QGEOM) against a
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

### Quantum geometry of the PythTB engine

`QUEER_QGEOM` (alias `PYTHTB_QGEOM`) runs the Generator's quantum-geometry step: the
band and manifold geometry of a 2D Wannier model at two position levels, the
Berry-connection-polarizability (BCP) dipole M_abc and the intrinsic nonlinear Hall
response chi_abc, integrated over the zone on an adaptive mesh.

```bash
python -m queer generator run --step QUEER_QGEOM --params pythtb_params.json
```

**The port.** The Generator's `pythtb_qgeom` kernel is numpy code throughout (PythTB
enters only its probe), so QUEER carries it operation for operation, and each
function is tested against the vendored kernel (`tests/oracles/generator_pythtb_qgeom.py`)
to 1e-12 on randomized and structured inputs, whole adaptive integrations included
(`tests/numerical/test_qgeom_ports.py`):

| Kernel (`pythtb_qgeom`) | QUEER |
|---|---|
| `layer_frame`, `a3_misalignment`, `cell_area`, `zone_area`, `reciprocal_in_plane`, `kcart`, `is_hexagonal`, `c3_rotate_frac` | `queer.physics.qgeom` (same names; `kcart` is `frame_k`) |
| `degenerate_clusters`, `_rows`, `_rows_diag`, `_diag_sandwich`, `geometry` | `queer.physics.qgeom.degenerate_clusters`, `layer_geometry` |
| `_fields`, `_blocks`, `dipole_tensor`, `symmetric_tensor`, the `X_*`/`B_*` columns | `integrand_fields`, `manifold_columns`, `dipole_tensor`, `symmetric_tensor` |
| `fermi`, `minus_fprime`, `density_cm2`, `_is_uniform`, `thermal_grid`, `mu_subset`, `curves_at` | `fermi_function`, `minus_fermi_derivative`, `density_cm2`, `is_uniform_grid`, `thermal_grid`, `mu_subset`, `curves_from_integrals` |
| `chi_from_dipole`, `chi_all_conventions`, `trigonal`, `trigonal_tensor`, `rotate_tensor`, `c3_residual_tensor`, `c3_residual_matrix`, `gap_bin_labels` | same names in `queer.physics.qgeom` |
| `_sorted_triangles`, `_theta_one`, `_delta_one`, `triangle_theta`, `triangle_delta`, `_accumulate`, `_range_pairs`, `_piece_coefficients`, `_add_cubic_pieces`, `_grid_rule`, `_pair_values`, `theta_on_grid`, `delta_on_grid`, `_grouped` | `sorted_triangles`, `theta_triangles`, `delta_triangles`, `triangle_theta`, `triangle_delta`, `accumulate_rows`, `range_pairs`, `piece_coefficients`, `add_cubic_pieces`, `grid_rule`, `rule_at`, `theta_on_grid`, `delta_on_grid`, `grouped_rule` |
| `hermitize`, `inplane_mismatch`, `dft_exactness` | `hermitize_translations`, `inplane_mismatch`, `dft_exactness` |
| `Table`, `evaluate`, `evaluate_chunks`, `sum_rule_residual`, `uniform_grid`, `uniform_response`, `make_payload`, `_local_extrema`, `occupation_pass`, `AdaptiveMesh`, `_cell_block_integral`, `Integrator`, `source_fields`, `dipole_sources`, `adaptive_response` | `queer.observables.qgeom`: `QgeomTable` (also `QgeomTable.from_model` of a `WannierModel`), the same names, `local_extrema`, `cell_block_integral` |
| `tb_layout`, `read_tb`, `parse_eig` | `queer.io.wannier.read_tb_layout`, `read_tb_records`, `read_eig` |
| `compact`, `write_json_compact`, `decimate`, `units` | `queer.compat.generator.qgeom_step` |

`queer.observables.qgeom.layer_band_geometry` and `nonlinear_hall_response` return
`ObservableResult` arrays (dimensions, units, the integration record). The driver's
step (`run_qgeom` and its helpers in `pythtb_core`) is
`queer.compat.generator.qgeom_step.run_qgeom`; `WannierStepContext.ws_apply` applies
the Hamiltonian's Wigner-Seitz images and vacuum fold to the stacked position table,
as the driver's `Context.ws_apply` does. Analytic tests (`test_qgeom_physics.py`)
check the triangle rules against exact polygon clipping, the diagonal level against
QUEER's own Kubo curvature and finite-basis metric, the Haldane Chern number and
Hall plateau, and the thermal integrals against a fine uniform grid.

**Runner contract.** The step builds its model as every Wannier step does and then
follows the driver:

- refusals (`queer_bad_params`): a model periodic along anything but a1 and a2,
  hopping cutoffs, a3 off the layer normal, a full level whose normal is not along z,
  more than 4001 mu points; the catalog bounds of `translate_pythtb` (temperatures at
  least 0.5 K, base mesh 16 to 512, 10 000 to 5 000 000 vertices, ...);
- the position levels: the diagonal level is PythTB's tight-binding model, tau the
  centres as PythTB places the orbitals; the full level comes from the WANNIER run's
  `wannier90_tb.dat`, read with the driver's fixed-layout size check. `auto` takes it
  when staged and otherwise warns (`position_diagonal_only`); `full` without it is
  `queer_position_unavailable`; a file of another size, R set or degeneracies, H more
  than 2e-6 eV off the model after the same Wigner-Seitz unfold, or r_ii(0) more than
  2e-5 Å off the centres in-plane is `queer_position_mismatch`;
- the table is the model stage's folded table with E_F on its R = 0 diagonal,
  hermitized at k as the driver does, so a replay sees the driver's rounding;
- `queer_results.json` holds the driver's `qgeom` block in the `pythtb_results.json`
  schema (every key of the production job job-1cd360d9 and of its two data files is
  present, checked against their key structure in
  `tests/fixtures/generator_schema/qgeom_keys.json`; the exceptions are the renamed
  producer digests, the largest reaped worker's RSS, which QUEER does not report, and
  the skipped `.eig` check), with the
  engine, counterpart and QUEER provenance, `provenance.queer_qgeom_sha256` (the
  digest of QUEER's quantum-geometry sources) and the inputs it read, including the
  `_tb.dat` and the `.eig`. `Out/qgeom_response.json` and `Out/qgeom_path.json` are
  written as the driver writes them (5 significant digits, decimated to the driver's
  10 MB budget) with the figures `qgeom_dipole.png`, `qgeom_ahc_dos.png` and
  `qgeom_path.png`;
- QUEER adds `integration.replay` (the stop controls), `integration.stop_vertices`
  (the vertex count where the integration stopped) and `integration.vertex_set`
  (count and SHA-256 of the sorted integer vertex keys), and writes the keys to
  `Out/qgeom_vertices.json` (up to 250 000 of them; the digest always);
- QUEER also writes `Out/qgeom_response_full.json`: the curves of
  `Out/qgeom_response.json` (`mu_eV` and `per_T`, at its stride) unrounded, about
  twice its size (some 5 MB for the production job's 2001 mu at four temperatures,
  under 20 MB at the 10 MB budget). Two QUEER runs are compared on it;
- the probe: where the driver compares its diagonal level with PythTB's own metric
  and curvature at 32 k-points, QUEER compares one band's metric and curvature with
  its own `band_pair_geometry(bands=...)` and `berry_curvature(bands=[b])`;
  `qgeom.parity` labels it `internal_consistency` and a deviation above 1e-8 warns;
- the `.eig` check: QUEER reads the `.win`'s `kpoints` block, so it compares the
  model's eigenvalues with the frozen DFT states where the driver, whose `.win`
  parser never stores that block, always reports it skipped. The check is a
  diagnostic run after the integration: a `.eig` QUEER cannot read (a repeated
  band/k-point record, a short line, an index that is not one-based) skips it with
  the reason in `dft_check.skipped` and the warning `dft_eig_unreadable`, and the
  step's result stands.

**Deterministic replay.** The integration stops at its vertex budget (`budget`), at
the depth limit (`depth`) or past its time budget (`time`, from `time_budget_s`).
Only the time stop depends on the machine. Two QUEER-only parameters make a run
reproduce a reference's sampling exactly: `qgeom_time_stop: false` turns the time
budget off (a reference that never stopped on time then replays as it ran), and
`qgeom_replay_vertices: N` stops the integration where the time budget would have,
the first time the mesh holds at least N vertices at one of the driver's checks,
reported as `time`, without a clock. A reference's final vertex count is the count
at its stop (no vertex is added after a stop), so a reference stopped on time at N
vertices replays with `qgeom_replay_vertices = N`. Setting `qgeom_max_points` to N
instead would not: the vertex budget also caps the gap rule at 30 % of it, and it
reports `budget`. The production reference job-1cd360d9 stopped on its vertex budget
(1 041 183 vertices of 1 000 000 allowed, the must-split phase overshooting), so it
replays with its own parameters and `qgeom_time_stop: false`. Rounding can still
steer a decision that lies exactly at its threshold: a strict local extremum of the
occupation pass whose neighbour ties it by symmetry (the seeds), a velocity that
vanishes by symmetry in the band-edge rule, the error estimates of symmetry-related
cells that tie in the refinement order, an A3 maximum that ties; and a controlled
quantity that vanishes identically (by symmetry) has a rounding-level scale, so its
normalised error estimate is rounding noise that orders the refinement. Another BLAS
kernel breaks such ties another way. A Haldane metal (C3-symmetric bands, the test's
`SYMMETRIC` settings) gives 127 band-edge vertices under OpenBLAS's Sandybridge,
Prescott, Nehalem, Barcelona and Haswell kernels and 150 under SkylakeX (10 014 vs
10 037 vertices); a Rice-Mele stack (`rice_mele(1.0, 0.6, staggered=0.15)`, E_F 0.9 eV,
whose Berry curvature and off-diagonal metric vanish identically) gives 10 108
vertices under Sandybridge, Prescott, Nehalem and Barcelona, 10 310 under Haswell and
10 564 under SkylakeX. `OPENBLAS_CORETYPE=Haswell` and `Zen` crash in the development
environment's OpenBLAS 0.3.34 (NumPy 2.5.3) on this Zen 4 host; Haswell runs with
NumPy 2.4.6's bundled OpenBLAS (a `PYTHONPATH` overlay). The goldens, the QWZ square
metal and the production-like insulator keep one vertex set on all six kernels. A
symmetric model replays reproducibly only with the reference's own NumPy and BLAS, so
the comparator decides a mesh difference by the runs' numeric environments (below).

**Comparator** (`queer.compat.generator.parity_qgeom`, through `compare` and the
`compare` command). On top of the Wannier family's preconditions (inputs by SHA-256,
parameters, E_F; derived decisions: the position level, the band window, the
occupied count and its source, the occupied manifold's applicability):

- *producer*: a PythTB reference must come from the vendored kernel (4fb51a38…) and a
  core whose quantum-geometry code is the vendored one (56fda912… to 0efcc4b3…);
- *stop rule*: a reference stopped on its vertex budget, the depth limit or not at all
  is deterministic and the candidate must not have stopped on its own time budget; a
  reference stopped on its time budget needs a candidate replayed at its vertex
  count; otherwise, or without the count, `not_comparable` with the reason;
- *replay check*: equal vertex sets (equal SHA-256 of the sorted integer keys when
  both runs recorded them, which are exact coordinates; against a reference without
  the record, the recorded mesh: vertex count, vertices per rule, levels, finest
  cell, seeds, controlled mu, the capped gap rule and the history), the same stop
  reason and convergence flags. A record must be its own run's mesh (its count that
  run's vertex count, its keys, when written, hashing to its digest, a QUEER run's
  digest its `vertex_set`'s), and a QUEER candidate without one fails against a
  reference that has one;
- *numeric environment* (precondition `mesh_determined`): every QUEER run records
  `provenance.numeric_environment` (Python, NumPy, SciPy, the machine, the SIMD
  extensions NumPy dispatches to, and each BLAS library's build, version, core type
  and threading layer; the thread count is kept but not compared). Where the vertex
  sets differ between runs that recorded one environment, rounding is the same in
  both, so the difference is compared and fails (`qgeom.mesh` `differs_unexplained`).
  Where the environments differ, or are not both recorded (the driver records its
  Python, NumPy and SciPy versions but no BLAS build), the difference is
  `not_comparable`, naming the differing or unrecorded fields, the first divergence
  and the remedy: rerun the candidate in the reference's numeric environment (the same
  Python, NumPy, SciPy and BLAS builds on the same core type). The decision records
  only describe the divergence (below); they decide nothing. The C3 gate (both runs'
  bands C3-symmetric on a hexagonal lattice) is retired: it let a real defect on an
  exactly C3 model hide behind `not_comparable` and failed correct runs of a
  mirror-symmetric model. `symmetry.lattice_hexagonal` and `c3_band_floor_meV` stay
  informational;
- *quantities*, with δ(T) = 1e-8 max(max A3, max |M|) over both runs: C, C_in to
  1e-8 max(1, |C|); K_in, K_diag, K_upper_loose to 1e-8 of their largest element;
  the headline A3 to δ; theta0 to asin(δ/A3)/3 on 3 theta0 modulo 360° where
  A3 > δ, and not judged where A3 <= δ (inside a gap A3 is rounding, 1e-14 against
  δ = 4e-9 in the insulator golden, and M's tolerance turns the trigonal part to any
  direction: Prescott and Nehalem kernels put theta0 37 to 40° away there). Such an
  entry is recorded as undetermined (`not_applicable`, listed under
  `flagged_sets.qgeom_undetermined`); a row would be `not_comparable` only if nothing
  else in it were judged. The chi amplitudes: chi's trigonal part is exactly
  -(c_out + 2 c_in) CHI_PER_M times M's, so each agrees to
  |c_out + 2 c_in| CHI_PER_M sqrt(2) δ plus the rounding of building and projecting
  chi (64 ulp of the M scale, weighted by |c_out| + 2 |c_in|); a turn of M's trigonal
  direction does not change the amplitude, so no direction term enters (the wpd set,
  c_out + 2 c_in = 0, is rounding only). sigma_xy to 1e-8 max(1, |sigma_xy|), the
  density to the count tolerance at each mu over the cell area; the 5-digit curves of the response and
  path files to half a unit in the 5th digit of each run plus the quantity's
  tolerance; the dipole sources,
  symmetry residuals, position checks, grid, window, electron count and verdict
  exactly or to the bounds derived from these;
- *mu_CNP* by the electron count: each run's count (n_occ plus its density curve times
  the cell area, interpolated linearly in mu on the grid block's full-precision grid,
  the rule that placed its mu_CNP) must be n_occ at its own mu_CNP and at the other
  run's, within the count tolerance there plus the print rounding. The count
  tolerance is the rounding of the count's sum, 4 ulp of the band count per vertex
  (the triangle rule's `bincount` and `cumsum` add the terms one by one), plus the
  count's slope |dN/dmu| times the eigenvalue tolerance: the count falls as any
  eigenvalue rises (Fermi weights and triangle areas alike), and those partial
  derivatives sum to -dN/dmu, so eigenvalues determined to τ_E determine the count
  to |dN/dmu| τ_E. No fixed floor enters: an earlier 1e-8 of the band count was
  2.3e-6 electrons on the 230-band production reference, where that run's own count
  residual is 3e-14, and let a mu_CNP move 1.5 meV along the 2 K thermal tail with
  its CNP row. With the derived tolerance (2.1e-7 there) the 2 K value may move
  0.71 meV, inside which the count is n_occ to its rounding, and the 10 K and 30 K
  values 0.007 and 0.005 meV. Two values in one
  gap, where the count is n_occ over a range of mu and rounding picks the interpolated
  value (another BLAS kernel picks another), are equal by this rule, and the detail
  says so; a mu_CNP on a band fails, whatever the density of states between the two
  values. Each run's CNP row must sit at the grid mu nearest its mu_CNP, and
  mu - mu_CNP must differ between the runs by exactly the mu_CNP difference. On a
  decimated file the count at a skipped grid point is not recorded, but the count is
  monotonic in mu, so it lies between the counts of the file points around it, and
  n_occ must lie in that bracket (to the count tolerance); the detail says when a
  bracket was used. A
  run without a mu_CNP fails unless its count misses n_occ at the edge of the range by
  no more than the count tolerance (then rounding decides whether mu_CNP lies on the
  grid: `not_comparable`). A headline point the runs place at different grid mu (two
  mu_CNP that agree, or a near-tie of the A3 maximum) is documented, and each run's
  row is checked field by field at its own mu: A3, A3_err, theta0 (where A3 > δ),
  sigma_xy, the density and full_over_diag_A3 against the other run's curves there,
  and chi and chi_err against their exact multiples -(c_out + 2 c_in) CHI_PER_M A3 and
  |c_out + 2 c_in| CHI_PER_M A3_err of the row's own A3 and A3_err (to the rounding
  term of the chi tolerance). A row mu a decimated file skipped is recorded as
  undetermined, since the other run's curve does not hold it, and so is a near-tie
  of the A3 maximum that a decimated file cannot show;
- *unrounded curves*: each `Out/qgeom_response_full.json` must round to its run's
  5-digit file (`qgeom.response.full_precision`); where both runs wrote one, the
  response curves, the count and the CNP checks use it without print rounding;
- *no reported error enters a tolerance*: the stage differences (`A3_err`, `err`) and
  the smeared check are compared as values, never used as error bars, so a forged
  estimate fails instead of widening anything;
- *producer gaps*: `dft_check` is documented (the driver's skipped `.eig` check;
  `not_applicable` when QUEER could not read the `.eig` either); `parity` (each
  producer's probe) is not compared, but the candidate's must pass; their warnings
  are left out of the warning comparison, and the QUEER-only replay and unrounded
  records out of the output comparison.

Negative controls fail (`tests/numerical/test_generator_qgeom.py`): theta0 shifted by
1° where A3 is determined, a doubled Chern number, a sign-flipped K_in, a
sign-flipped or zeroed chi at E_F and at a CNP with a determined A3, a dropped vertex
(with or without the vertex keys, whatever the model's symmetry), vertex keys that do
not hash to their digest, another stop reason, forged error estimates, a mu_CNP moved
onto a band (at 0, 5, 20 and 60 K, with its row and curves moved consistently and with
the gap's DOS zeroed), a CNP row off its mu_CNP, a missing mu_CNP whose count crosses
n_occ, unrounded curves that are not the run's 5-digit file, an A3 move of 1e-6 that
the 5 digits hide, a Weyl bound the candidate reports (it widens nothing), and mesh
defects in the reference's numeric environment (a gap-rule ratio a fifth low, one
gap-flagged leaf left unsplit, a first stage that never refines its worst leaf, a
dropped vertex, a forged decision record); an unknown producer, a clock-stopped
reference without its vertex count or without a matching replay, a candidate stopped
on its own clock and a mesh difference between runs of different or unrecorded
numeric environments are `not_comparable`. Rerun under OpenBLAS kernels
(`OPENBLAS_CORETYPE`, in a subprocess, skipped where a kernel cannot run): the
insulator under Prescott, Nehalem and Sandybridge passes against its golden, under
Prescott also against QUEER's own default-kernel run unrounded; the Haldane metal and
the Rice-Mele stack under Sandybridge and SkylakeX are `not_comparable`, naming the
BLAS core types (on this host their vertex sets differ), and the QWZ square metal
passes. The insulator's verdict loses `quotable` at its in-gap CNP rows, where A3,
A3_err and M are rounding: where the verdicts differ only in that reason and every
row behind it lies within 3 δ(T) of its threshold, `qgeom.verdict` and
`verdict_reasons` are undetermined (on a 1-meV grid Sandybridge writes `indicative`
and SkylakeX `quotable` on one mesh); a reason decided at a determined row fails. The
eigenvalue tolerance comes from the reference alone and carries no Weyl bound: both
producers hermitize H(k) alike, and no number a run reports widens a tolerance.

#### Why two QGEOM meshes differ

*The user's decisions (2026-10-18).* (1) When a QGEOM comparison cannot be made
exactly, the comparator identifies the reason of the mismatch; the plan's error-bar
fallback (|ΔX| ≤ err_ref + err_Q) is not used. (2) A mesh mismatch identified as
rounding at tied decisions ends as `not_comparable` with that reason (the point, the
rule and the tie count) and the remedy (run both on one BLAS kernel); an unidentified
mismatch is compared and fails. (3) `Out/qgeom_response_full.json` stays always
written. (4) After a review showed that the tie attribution misjudges exactly symmetric
models both ways (a refinement defect on a model with an identically vanishing
quantity was reported as a tie even on one kernel, and correct runs of exactly
degenerate models failed across kernels, where the eigensolver's basis inside a
degenerate cluster is kernel-dependent), the user decided that a mesh mismatch is
decided by the numeric environment: in one environment it fails; in different or
unrecorded environments it is `not_comparable`, with the environments, the first
divergence and its recorded ties as the reason and the same-environment rerun as the
remedy. The tie analysis describes, never decides.

*Why the environment decides.* On one kernel, in the PythTB environment (Python
3.12, NumPy 2.5.4), the Generator driver and QUEER give identical vertex sets on every
exactly symmetric model tried, and across kernels both change alike:

| model | Sandybridge (driver = QUEER) | SkylakeX (driver = QUEER) |
| --- | --- | --- |
| Haldane metal (C3) | 10 014 | 10 037 |
| Rice-Mele stack (mirror) | 10 108 | 10 564 |
| Kane-Mele with Rashba (C3, time reversal) | 10 506 | 10 556 |
| Kane-Mele, PT-symmetric (degenerate everywhere) | 10 777 | 11 304 |

QUEER with 4 workers, and with 8 or 16 OpenBLAS threads, gives the single-thread
mesh. So rounding at ties differs only between environments, and a difference within
one environment is a defect.

*The decision record* (`qgeom.integration.decision_ties`, QUEER-only,
`queer.observables.qgeom.Integrator.decision_record`). Beside the kernel's
arithmetic, without changing the mesh or the history (the port tests compare QUEER's
integration with the vendored kernel's exactly, and a 305 631-vertex run gives the
same vertex set, history and wall time with and without the record, which takes
27.5 kB), QUEER computes each decision's margin to its threshold and records the
decisions within their rounding band:

- *seeds*: the strict local extrema of each pair's gap and each window band on the
  occupation grid and their thresholds (hot gap, degeneracy, the extended range) and
  the controlled-mu range; the record gives the seed count, the digest of the seed
  ids and the tied ones (grid index, kind, bands, taken);
- *must-split rounds*, one per round, also the last that splits nothing: each judged
  leaf's possible outcomes (the first rule that holds, in the order seed, gap, edge,
  below the depth limit, a capped gap rule deferring its leaves) from the margins of
  the seed diameter, the gap rule's d vmax > gap_ratio max(gap, gap_floor), its
  extended range and 1e-9 eV degeneracy floor, the occupied-block rule and the edge
  rule's velocity signs, reach and span; the tied leaves (centre key, mask of possible
  outcomes, the run's outcome), the tie count per rule, the leaves decided outside
  their bounds (a QUEER bug, never expected), and per rule the number of leaves split,
  an additive digest of their centre keys (the sum of their splitmix64 hashes modulo
  2^64: digests of disjoint sets add) and the range of vertex ids their splits created;
- *refinement steps*, one per history entry: the stop test worst <= target, the
  batch's count (the cumulative indicator against worst - 0.75 target) and the leaves
  whose membership rounding can flip (indicators within the band of the batch's cut);
  per stage the controlled quantities whose scale is rounding (below), and per step
  the leaves whose indicator is one of theirs and the worst summed error from them and
  from the others;
- *the second stage's A3 maximum*: the grid mu within its band of the maximum.

`Out/qgeom_vertices.json` holds the keys in creation order (`order: creation`; the
digest stays that of the sorted keys), so each round's splits are the cells whose
child centres its vertex range created: a vertex whose two coordinates share their
lowest set bit l is the centre of the cell of size 2 l, which exists only once its
parent is split.

*The bands* are derived from the run's own scales. An energy comparison is tied within
TIE_ROUNDING = 1e-12 of the spectral radius over the occupation grid (a gap or a
neighbour difference within twice that, a difference of gaps within four times); a
velocity comparison within 1e-12 of the largest band, interband or occupied-block
velocity of the base mesh (a gap-rule margin d vmax - gap_ratio max(gap, floor)
within d times the velocity band plus gap_ratio times twice the energy band); the
refinement's indicators, their sums and the A3 maximum within TIE_INDICATOR = 1e-7 of
the cut value, of max(worst, target) and of max(max A3, max |M|) (twice that between
two leaves or two mu). A quantity's scale is rounding when it is nonzero and at most
1e-12 of its family's (a Fermi-surface v_a G_bc or v_a omega column against the largest
of those at the same mu; the sea and block omega and metric columns against the
largest area-integrated geometric scale; a headline's alpha and beta where one of its
v_a G_bc is). Measured on the six models (Haldane, Rice-Mele, QWZ square, the three
goldens) under Sandybridge, SkylakeX, Prescott, Nehalem, Barcelona and Haswell, for
every decision of the rounds before the first divergence:

| decisions | largest kernel-to-kernel margin difference | closest margin of a generic model |
| --- | --- | --- |
| seed extrema and thresholds | 5.8e-4 band (1.2e-15 of the spectral radius) | 1.5e6 bands |
| gap rule (ratio, range, degeneracy, occupied block) | 1.3e-3 band | 2.2e7 bands; 131 to 210 bands for the 1e-9 eV floor against the top band's zero gap |
| edge rule (velocity signs, reach, span) | 2.1e-3 band (2.1e-15 of the velocity scale) | 4.5e6 bands (signs), 4.9e3 bands (span) |
| batch cut (indicators) | 3.8e-3 band (3.8e-10 of the cut value) | 148 bands between neighbouring indicators at the cut |
| stop and batch count (worst, cumulative indicator) | 9.4e-4 band | |
| A3 maximum | 9.7e-7 band | |
| indicators of rounding-level quantities (Rice-Mele) | 3.75e6 bands (0.38 of the cut) | |

So each band exceeds the largest measured difference 260 to 10^6 times, and the
closest margins of models without exact symmetries lie 130 bands or more beyond it
(a symmetric model's exact ties sit at margin 0). The closest is the 1e-9 eV
degeneracy floor, a fixed energy against bands that grow with the spectral radius:
131 bands at the Haldane metal's 3.8 eV, 25 at 20 eV. A flip in the other run puts this
run's margin within twice the rounding, inside its band, which is why one run's
record can speak for a decision.

*The description of a divergence* (`queer.compat.generator.qgeom_mesh.attribute`). It
is written into the `not_comparable` reason and the `qgeom.mesh` detail and decides
nothing; the rule it applies, below, is what it reports as "identified". A record is
first checked against its own run: its rounds chain from the base mesh (2 base^2
vertices, base^2 leaves; a split adds 3 leaves and 4 to 8 vertices,
`AdaptiveMesh.split`) to the run's vertex count, its history entries are the run's,
its counts are its lists, it decides no leaf outside its bounds and, with the keys,
each round's splits are what its vertices show and each tied leaf was a leaf then
with the outcome its vertices show. A record that is not its run's (a forged tie, a
count that is not its list, a dropped vertex) is reported in the detail. Then:

- *two records* (two QUEER runs): the seeds and the seed energies inside the mu range
  (a count that differs needs that many energies tied at the range's ends in both
  records), then each round in order; a round agrees
  when its splits per rule (count and digest), stop and A3 maximum agree, and the
  meshes before the first round that does not are then the same. There the
  difference is identified only if the leaves split outside the decisions tied in
  both records are the same (the tied leaves are taken out of each digest), each tied
  decision decided differently was possible the other way in both records, a stop
  decided differently is tied in both, an A3 maximum decided differently is tied in
  both, or the stage normalises rounding-level quantities that set an indicator or
  the worst sum in both runs (the refinement is then ordered by rounding noise);
- *one record* (the driver's reference has neither keys nor a record, and keeps its
  last 12 history entries): the seed count (at most the tied seed decisions may
  differ), then the first history entry that differs, aligned to the record's full
  history. The round that produced it must tie at least |Δleaves|/3 split decisions
  and no more than 8 vertices per tied decision may differ; a phase or stage that
  ended in one run only needs a tied end; a refinement step ordered by rounding-level
  quantities explains any count. Where the reference's 12 entries begin at an entry
  the candidate never reached, the difference is identified only if the must-split
  vertices per rule agree and every refinement step of the candidate is ordered by
  rounding;
- *equal records without keys* (a driver reference whose recorded mesh equals the
  candidate's): `qgeom.mesh` agrees and says what is established (vertex count,
  vertices per rule, seeds, controlled mu, deepest level and finest cell, the last
  history entries) and what is not: that the vertex sets are equal, since an
  exchange of leaves that keeps these counts is not seen; the candidate's record
  counts the tied decisions that could do so.

*Scenarios* (this host). Across kernels, the Haldane metal, the Rice-Mele stack and
both Kane-Mele models are `not_comparable`, naming the BLAS core types (and against
Haswell the NumPy 2.4.6 overlay); the description places the Haldane metal's
divergence at must-split round 2 (one of 14 tied edge decisions). In one environment,
a gap-rule ratio a fifth low, one gap-flagged leaf left unsplit, an error estimate
three times too large, an integrand distorted by up to 30 %, an edge resolution 1.5
times too coarse (Kane-Mele) and a first stage that never refines its worst leaf all
fail, also on models whose refinement is ordered by rounding-level quantities; a
forged decision record changes nothing. Against a driver reference, which records no
BLAS build, every mesh difference is `not_comparable` with the unrecorded fields.

*Out of reach, and what closes it.* Between environments a mesh defect is
`not_comparable`, never a pass; the remedy, a rerun in the reference's environment,
then decides it. Against a driver reference that is every mesh difference, until the
driver also records its BLAS build (a change to the Generator's `_pythtb_core.py`,
for backend R2). For a QGEOM pair to be decisive on the Generator, the QUEER
environment on the host must equal the PythTB one (the same Python, NumPy and SciPy
wheels, which carry their OpenBLAS builds) on the same node type. Real Wannier models
carry no exact symmetry (the production model is 9.2 meV from C3) and replay exactly
across kernels, so this matters for synthetic symmetric models and any future
symmetric reference.

**Goldens.** `qgeom_full_metal`, `qgeom_full_insulator`, `qgeom_diagonal_budget` and
`qgeom_position_unavailable` (a generic three-band layer without exact symmetries, so
no refinement decision depends on rounding) are the vendored driver's results under
PythTB 2.0.2; the golden script also records the kernel's final vertex keys. QUEER
reproduces every vertex set, stop reason and quantity, also with OpenBLAS's
Sandybridge, SkylakeX, Prescott, Nehalem and Barcelona kernels forced
(`OPENBLAS_CORETYPE`); the `.eig` check differs, as documented, and under another
kernel the insulator's mu_CNP inside its gap may too, which agrees by the count and
moves its CNP row to another grid mu (documented).

**Deviations from the plan.** A reference that cannot be replayed (stopped on its
time budget with an unknown vertex count, or a candidate not replayed at it) is
`not_comparable` with the reason, not compared under the plan's error-bar fallback
|ΔX| ≤ err_ref + err_Q: that fallback puts both runs' own error estimates into the
tolerance, and the user decided (2026-10-18) that the comparator identifies the reason
of a mismatch instead. The per-T curves are compared at the driver's 5-digit print
precision plus 1e-8 against the driver, which writes nothing finer, and unrounded
between two QUEER runs. The replay is a vertex-count stop reported as `time`, not
`qgeom_max_points = N`.

**QUEER-only records and the Generator.** `Out/qgeom_vertices.json` (the replay
record, keys in creation order), `Out/qgeom_response_full.json` (the unrounded curves,
always written) and `qgeom.integration.decision_ties` (the decision record, in the
results document) are comparison records, not products. The backend picks them up with the job's `Out/**` like every
data file, and its parity service already fetches every `Out/*.json` of
both runs (within its 64 MiB budget per run), which is how a comparison finds them;
deleting either only makes a later comparison fall back (to the recorded mesh, or to
5-digit curves). The backend collector and the frontend should list them as data of
kind `data`, apart from the user-facing products (the response and path files and the
figures), or hide them. For the 1.04M-vertex production replay QUEER writes the
digest without keys (cap 250 000), and the driver's reference records neither: two
QUEER runs still compare their vertex sets exactly by digest, and against the driver
the recorded mesh decides. Uncapped keys would only name which vertices differ, so
they are not written.

### OpenMX steps of the PyATB counterpart

`QUEER_AHC` and `QUEER_CHERN_NUMBER` (aliases `PYATB_AHC`, `PYATB_CHERN_NUMBER`) run
the calculation of PyATB's `AHC` or `CHERN_NUMBER` block on the staged OpenMX scfout
with QUEER's own model (`queer.compat.generator.openmx_steps`); `QUEER_SHIFT_CURRENT`
has its own section ([Shift current of the PyATB
counterpart](#shift-current-of-the-pyatb-counterpart)):

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
reference has no `inputs.sha256`. A `SHIFT_CURRENT` comparison requires `--scfout`
(it recomputes both spectra from it, on `--workers N` processes, refusing more than
`--recompute-budget N` of work, k-points x bands^2 x (bands + photon energies / 8);
see the shift current below):

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

### Shift current of the PyATB counterpart

`QUEER_SHIFT_CURRENT` (alias `PYATB_SHIFT_CURRENT`) runs PyATB's `SHIFT_CURRENT`
block on the staged scfout with QUEER's resonant shift conductivity
`sigma^abc(0; w, -w)` (`queer.observables.nonlinear.shift_current`, microA/V^2;
`queer.compat.generator.openmx_shift`):

```bash
python -m queer generator run --step QUEER_SHIFT_CURRENT --params queer_params.json
```

**Parameters.** The file has the keys of the OpenMX steps; `parameters` is the
`SHIFT_CURRENT` block. `occ_band`, `omega`, `domega` and `grid` are required (PyATB
has no defaults for them); the defaults are PyATB's own (its resolved echo prints
them): `smearing_method 1`, `eta 0.01`, `method 1`, `n_occ -1`, `m_unocc -1`,
`scissors 0`. `n_occ`/`m_unocc` (one-based) name one band pair only as PyATB
reads them, both at least 1 and `m_unocc > n_occ`; any other values mean every
occupied-empty pair. Missing required keys, an `occ_band` below 0 or at least the
number of bands PyATB reads (the spinor bands, or the bands of the spin-up channel
without spin-orbit coupling), a nonpositive `domega` or `eta`, a decreasing
`omega` and unknown keys are `queer_bad_params`; a negative first photon energy is
taken as PyATB takes it. A negative `scissors`
(QUEER's scissors is a rigid upward shift) and a pair that does not lead from an
occupied band to an empty one (`n_occ <= occ_band < m_unocc`), both of which PyATB
evaluates with fixed occupations, are refused at documented limits of QUEER's
(`not_comparable` in a parity comparison), as is an occupation boundary that splits
a degenerate level at a sampled k (`queer_validation_failed`).
`self_convergence: true` also evaluates the doubled grid (the axes with more than
one point).

**Conventions,** read from PyATB 60fc6b2a (`pyatb/berry/shift_current.py`,
`cpp/core/shift_current_solver.cpp`, `cpp/core/linear_response.cpp`,
`pyatb/io/openmx_read_scfout.py`):

| Quantity | PyATB | QUEER |
|---|---|---|
| photon energies | `start + i*domega`, `i < int((end - start)/domega) + 1` | the same list (`omega_eV`) |
| occupation | the lowest `occ_band` levels (`fn = 0`, `fm = 1`) | the same, per spin channel without spin-orbit coupling |
| Gaussian (`smearing_method 1`) | `exp(-((E - w)/eta)^2)/(eta sqrt(pi))` on the rows within `int(5 eta/domega)` of the transition's row `int((E - w_0)/domega)` | the untruncated Gaussian of standard deviation `eta/sqrt(2)` |
| no smearing (`0`) | the transition's weight added to row `int((E - w_0)/domega)`, no `1/domega` | the centred histogram of width `domega` per eV, row `i` holding `w_i - domega/2 < E <= w_i + domega/2`; `binned_uA_V2` holds PyATB's normalization from the same estimator: `domega` times the histogram with its rows above each photon energy, the weight in `(w_i, w_i + domega]`; each transition's row from one integer index (`ceil(x - 1/2)` and `ceil(x) - 1`, `x = (E - w_0)/domega`) |
| adaptive (`2`) | `min(1, max(1e-4, abs(grad(E_n - E_m)) dk eta))`, `dk = max_i abs(b_i)/100` | QUEER's adaptive Gaussian (RMS transition change across a reciprocal cell over sqrt(12), clipped to [1e-4, 1] eV) |
| window | without a band pair only the transitions in `[w_0, w_0 + N_w domega)` count | every transition |
| `method` | 0: every energy denominator of its eigenvector derivatives `d/(d^2 + 0.04^2)`, at every k-point (`get_partial_eigenvectors_1k_SumOverStates`); 1: `get_D_degenerate` divides by `E_l - E_j` for every level `l` and every band `j` of a selected pair and uses `d/(d^2 + 0.04^2)` for `abs(d) <= 1e-4` eV, which is zero for an exact degeneracy | QUEER's analytic derivatives (degenerate blocks covariant); at an exact degeneracy PyATB's treatment (no derivative inside the block, the block's velocity eigenvalues) meets QUEER's covariant sum on the models without on-site dipoles tested, Kramers pairs among them, and with on-site dipoles depends on its eigensolver's basis inside the block; a spacing above QUEER's degeneracy tolerance and within 1e-4 eV is regularized |
| scissors | added to the eigenvalues from `occ_band` on before its derivative kernels | moves the transition energies; geometry from the unshifted Hamiltonian |
| prefactor | `-pi t e^3/(2 hbar^2 V)` with its `t` = 6.582119e-16 eV s, e = 1.602176565e-19 C, hbar = 1.054571726e-34 J s, and the signed cell volume | `-pi e^2/(hbar V)` (CODATA), `V = abs(det A)`; the two agree to 4e-8 (`pyatb_shift_constant_ratio`) |
| spin | `nspin 1` (no spin-orbit coupling): only `Hks[0]` is read (`openmx_read_scfout.py` lines 218-223, `nspin` inferred at line 358) and the value is multiplied by 2 (`shift_current_solver.cpp` lines 103-110); `nspin 4`: by 1 | the spinor model; a non-spin-polarized SCF: twice its channel (equal to PyATB's); a collinear SCF: channel 0 plus channel 1, each with `occ_band` levels, and `legacy_single_channel` = 2 x channel 0 (warning `shift_collinear_channels`) |
| energies | the scfout's Hartree times 27.211324570273 | CODATA's Hartree (a transition 2.3e-6 of its energy higher) |

The 18 independent components `a(bc)`, `b <= c`, are in PyATB's order `xxx, xxy,
xxz, xyy, xyz, xzz, yxx, ..., zzz`.

**Results.** `queer_results.json` holds `model`, `integration` (the Gamma-origin
Grid, its points and the doubled grid with `self_convergence`) and
`shift_current`: `omega_eV`, `components`, `components_computed` (all 18),
`sigma_uA_V2` (`{component: column}`, QUEER's physical value), `binned_uA_V2`
(without smearing), `smearing` (`eta_pyatb_eV`, `gaussian_sigma_eV`,
`bin_width_eV`), `method`, `occupation` (`occ_band`, `n_occ`, `m_unocc`, the
zero-based `band_pair`, `scissors_eV`), `grid`, `convention` (`lattice_handedness`,
`export_sign`, `pyatb_spin_factor`, `pyatb_nspin`, `pyatb_constant_ratio`),
`legacy_single_channel` (a collinear SCF: `spin_factor`, `sigma_uA_V2`,
`binned_uA_V2`), `build` (`queer_source_sha256`: the SHA-256 of every `.py` file
of the installed queer package, sorted relative paths and bytes,
`results.package_source_sha256`) and `diagnostics`, plain facts of the run on the
model PyATB reads with QUEER's own transitions, which no comparison reads:
`min_pair_gap_eV` and `min_pair_gap_k` (the smallest spacing, scissored as PyATB's
derivatives see them, between a band of a pair PyATB selects at a k-point and any
other level not exactly degenerate with it), `exact_degenerate_kpoints`,
`kpoints_at_pyatb_threshold` (those with such a spacing within 1e-4 eV: PyATB's
`method 1` regularizes them; warning `shift_pair_gap_at_pyatb_threshold`),
`transitions` (`count`, `inside_window`, `below_window`, `above_window`, their
extremes on both Hartree scales and the margins to the first and last photon
energy, also in units of eta; warning `shift_transitions_outside_window` when
PyATB's window drops some), `gaussian_truncation` (`interval_bins`, `cut_eV`,
`tail_relative`: the largest share of a transition's peak PyATB's truncation
drops) and `self_convergence` (`physical` and, for a collinear SCF,
`legacy_single_channel`: `sigma_doubled_uA_V2`, `e_inf_uA_V2`,
`max_doubled_uA_V2`, `ratio`, `converged`; warning `shift_current_unconverged`).
The records that served only the earlier comparator (`pyatb_scheme`,
`regularisation` and its flagged records, `rounding`, `exact_degeneracies`,
`spin_channels`, `clean_control`, `kpoint_order`) are gone: a comparison recomputes
everything it judges from the scfout.
`Out/Shift_Current/shift_current.dat` is PyATB's table (`%10.5f` and 18 x `%15.6e`
under PyATB's header) of `export_sign` times `sigma_uA_V2`, or times `binned_uA_V2`
without smearing. Failures write the document with the codes of the OpenMX steps.

*Estimator fix, 2026-10-18 (physics, smearing 0 only).* QUEER's histogram
(`queer.observables.nonlinear.shift_current_spectrum`, for `sigma_uA_V2` and
`binned_uA_V2` without smearing) now places each transition by one integer index
(`queer.observables.nonlinear.histogram_rows`): per eV row `ceil(x - 1/2)`, binned row
`ceil(x) - 1`, `x = (E - w_0)/domega`, the half-open rows of the table above, so every
transition lands in exactly one row (or none of the table). It used to test each row's
bounds in floating point (`w_i - E` against `+-domega/2`), which can drop a transition
that lies exactly on a bin edge or count it in both rows: on six photon grids, 9633 of
20889 energies placed exactly on the edges (4429 dropped, 5204 doubled). Away from the
edges the values are bit-identical, and a Gaussian or adaptive smearing is unchanged: an
earlier result can differ only where a transition lies within rounding of a bin edge
(the 48-run edge sweep of 2026-10-17, analytic models with transitions on bin edges,
found the old rule placing each of them once). Photon energies that are not
`w_0 + i bin_width` to rounding keep the bounds' test (they do not tile). The regression
test is
`tests/numerical/test_native_postprocessing.py::test_a_histogram_places_every_transition_in_exactly_one_row`.

**Parity rule** (`queer.compat.generator.parity_shift`, reached through `compare`
and the command, which recognises the reference). QUEER's reported values are its
physical estimator, and they are what is judged. **The user decided on 2026-10-18**
to stop patching the earlier design (energy-scale moves, rounding floors,
candidate-supplied records, partial recomputations and a per-release clean-control
report each opened new holes) and to replace it by a full independent recomputation:
every number a comparison compares QUEER's and PyATB's tables with, and every
tolerance, comes from its own evaluation of the scfout; it reads no record of the
candidate's but its tables, so a forged or wrong diagnostic changes nothing. Each
comparison verifies itself: the per-release clean-control report, its
`--clean-control` option and its gate are retired, with the runner records that
served them.

*The recomputation* (`queer.compat.generator.pyatb_shift_integrand`) evaluates, on
the run's own Gamma-origin grid and photon energies, two complete spectra of all 18
components from the scfout. It reads the scfout itself
(`queer.compat.generator.shift_scfout`, written from OpenMX's version-3 record layout:
`Hks`, `iHks`, `OLP`, the first `OLPpo` moments, the lattice and atomic positions in
atomic units; the file must end where its records and embedded input lines say) and
shares only numpy and scipy with QUEER: none of QUEER's scfout reader, runner or
observables. A defect in QUEER's reader (its Hartree constant, its reconciliation of
the records, its spin-down channel) therefore moves QUEER's value and not the
physical mode's, and check A fails; a defect in this parse moves the literal table and
not PyATB's, and check B is not met. (Until 2026-10-18 the physical mode was built from
QUEER's reader; three such defects then passed both checks.)

- the **physical** mode, QUEER's shift current written from its equations on QUEER's
  physical operators: CODATA's Hartree and Bohr (`scipy.constants`); a collinear
  scfout's spin-down channel from its own `Hks` records; with spin-orbit coupling the
  spinor `[[uu, ud], [ud(-R)^+, dd]]` (`uu = Hks0 + i iHks0`, `dd = Hks1 + i iHks1`,
  `ud = Hks2 + i (Hks3 + iHks2)`); the absolute position operator
  `r_a(R) = OLPpo_a(R) + tau_a S(R)` (`tau` the bra orbital's atom); every record
  projected onto the translation identities, `H(R) = (H(R) + H(-R)^+)/2`, the same for
  `S`, and `r_a(R) = (r_a(R) + r_a(-R)^+ + R_a S(R))/2`. Then
  `sigma^abc(w) = -(pi e^2/hbar) (g/(V N_k)) sum_k sum_{m occ, n empty} f_mn
  Im[r^b_mn r^c_nm;a]_(bc) delta(w - w_nm)` with the exact SI 2019 `e` and `hbar`,
  `V = abs(det A)`, `g = 2` without spin polarization, a collinear scfout's two
  channels summed (each with `occ_band` levels) and the spinor bands with spin-orbit
  coupling; the scissors moving the transition energies only; every transition above
  the degeneracy tolerance (`max(1e-9 eV, 10 x)` the largest adjoint residual of the
  scfout's own Hamiltonian records, measured here); the Gaussian
  `exp(-(x/eta)^2)/(eta sqrt(pi))` (standard deviation `eta/sqrt(2)`) or, without
  smearing, QUEER's histogram with each transition's row from one integer index, so
  every transition lands in exactly one row: per eV row `i` holds
  `w_i - domega/2 < E <= w_i + domega/2` (`ceil(x - 1/2)`, weight `1/domega`) and
  `binned_uA_V2`'s row `i` holds `w_i < E <= w_i + domega` (`ceil(x) - 1`), with
  `x = (E - start)/domega` (a test of each row's centre in floating point dropped or
  doubled 5-15 % of the transitions exactly on an edge); an exact degeneracy treated
  covariantly (parallel transport inside the degenerate levels,
  `D_a = -Sbar_a/2 + i (rbar_a + rbar_a^+)/2` there, and their velocity matrix), so a
  degenerate level's transitions sum to a value independent of the basis inside it;
- the **PyATB-literal** mode, what PyATB 60fc6b2a prints, transcribed from its
  sources (`shift_current_solver.cpp`, `linear_response.cpp`, `xr_operation.cpp`,
  `base_data.cpp`, `openmx_read_scfout.py`; read only, written anew): its reader's
  operators (Hartree 27.211324570273 eV, Bohr 0.52917721 Angstrom; for `nspin 1` the
  spin-up `Hks` only, with `H` and `S` replaced by `(X(R) + X(-R)^+)/2` over the
  translations and their negatives before anything else; for `nspin 4` the spinor with
  its `H21` block from `H12(-R)^+` and the recorded overlap; the real part of the
  position operator built with the recorded overlap), its Bloch sums of their upper
  triangles (Hermitian completion; the position operator's lower triangle from
  `(r + i dS)^+`), the scissored levels in its derivative kernels, its method-1
  integrand (`get_Inm_ik_sumOver` with `get_D_degenerate`: `1/dE`, or
  `dE/(dE^2 + 0.04^2)` within 1e-4 eV, `i rbar(j, j)` on the diagonal, and the band
  velocities of `get_partial_eigenvalue_fromBar`), its window
  `[start, start + count domega)` without a band pair, its rows `int((E - start)/domega)`
  (C's truncation toward zero), its Gaussian on the rows within `int(5 eta/domega)`
  or its bare row weight without smearing, and its prefactor
  `-t pi e^3/(2 hbar^2 V N_k) 1e6` with its constants (`t` = 6.582119e-16 eV s,
  e = 1.602176565e-19 C, hbar = 1.054571726e-34 J s), the signed cell volume and its
  spin factor (2 for `nspin 1`). `method 0` and the adaptive smearing are not
  transcribed. A collinear scfout whose records are Hermitian only to 1e-10 to 3e-9 eV
  was not reproduced before the Hermitization was transcribed.

Each mode is evaluated in four orbital bases: the first and three fixed random
orthogonal changes of it, each with the translations of its Bloch sums in another
fixed random order, which reach the same levels and integrand through other
floating-point operations. A table's **rounding tolerance** at each entry is

- 100 times its envelope, the largest over the other bases of `sum_k abs(T_k - T'_k)`
  (taken per k-point, so no cancellation between k-points hides it);
- plus 32 eps times the broadened first-order rounding of its transitions'
  integrands: every quantity of the formula carries its rounding forward (a product
  `x y` by `N_x |y| + |x| N_y`, a sum by its parts' sum, the second-order product too)
  from three sources: the operators' Bloch sums and projections
  (`|C|^T (sum_R |w_R| |X(R)|) |C|`); the eigenvectors, which a backward-stable solver
  rotates into each other level `j` by up to `s/|E_m - E_j|`
  (`s = max |E| max(1, max_n |C_n|^2)`), moving a projection by
  `s (G |Xbar| + |Xbar| G)` with `G_mj = 1/|E_m - E_j|` (a symmetry-forbidden factor is
  noise of the size of the elements that rotation mixes in, not of its own size); and
  the energies (`max |E|` each, through every `1/dE`). Levels closer than 1 % of the
  kernel's width (eta, or domega without smearing) are one group: for a transition
  between two groups that lie wholly on their sides of the occupation boundary (and no
  band pair) the table sums every transition between them, so the rotation inside a
  group cancels there but for the group's spread through the kernel's slope (or whole
  at a bin edge within that spread); a transition inside a group keeps all of it;
- plus the kernel's own rounding and its slope times the transition energies'
  rounding (each `64 eps`);
- plus, for a transition within 1e-9 eV of an interior discontinuity of the kernel (a bin
  edge, PyATB's rows), the difference of placing it either side; summed over the rows
  that choice moves only the edge's own row, so the sums conserve the transition.

**Outer edges.** On the first or last edge of a histogram, or at PyATB's window ends,
one side lies outside the table: the placement decides whether a transition is in the
table at all and moves the sum of every row beyond. There is no allowance there (until
2026-10-18 the edge row could hold anything within the transition's weight, twice it,
minus it or half of it). Each k-point's transitions on such an edge are one group with
exactly two placements, inside the table (with their own rounding in the row they take:
like the table's envelope, the basis difference of the group's summed weight, which a
rotation inside a degenerate group leaves unchanged, not transition by transition, which
priced a Kramers group on the binned first edge at about 100 times its weight)
or outside it, and checks A and B test the table against every combination of the groups:
the recomputation's own first, up to 10 groups (1024 combinations). Beyond 10, or where
the recomputation's own placement splits a k-point's group across the edge, those
transitions keep the allowance and their entries are not determined (`not_comparable`,
never a pass). Interior edges are exact in the sums over the rows (beyond the edge's
row they move nothing) and keep their allowance: a transition there may be split
between its two rows (each part within its weight), never dropped or counted twice in
total.

No number of the candidate's enters a tolerance. Measured transition by transition
against QUEER's estimator on thirteen models (clean, nonorthogonal with dipoles,
Kramers pairs, pairs split by 4e-5 and 1e-7 eV, weakly broken symmetries, the real
calibration scfout at occ_band 2, 4 and 16), QUEER's difference is at most 4.2e-16 of
that first-order rounding, so 32 eps leaves a factor of 17. A single second basis is
not a bound where integrands are rounding (the earlier design's 100-fold two-basis
envelope plus `1e-13` of the conditioned magnitudes `abs(r^b) abs(r^c_;a)`): QUEER's
honest value then failed at PyATB's eta 0.01 on 10 of 20 real-scfout configurations and
on 12 of 48 weakly broken analytic models. Where PyATB's own value depends on its
eigensolver's basis inside degenerate levels (on-site dipoles at exact Kramers pairs, its
regularized block at natural near-degeneracies), the bases differ by that dependence and
the tolerance says so.

In order:

1. *Preconditions* (`not_comparable`): the scfout is given (`scfout=`, `--scfout`;
   without it "the shift-current comparison recomputes the spectrum from the scfout")
   and its SHA-256 equals both the reference's `inputs.sha256` entry and the
   candidate's provenance; PyATB's resolved settings equal QUEER's (its defaults and
   band-pair rule, `nspin` as the scfout's layout reads it: 4 with spin-orbit
   coupling, else 1); PyATB's printed photon energies are its `start + i domega` to
   its five decimals; `method 1` and `smearing_method` 0 or 1; the work, k-points x
   bands^2 x (bands + photon energies / 8) summed over the channels, within the
   recompute budget (`recompute_budget=`, `--recompute-budget`, default 8e9: else "too
   large to recompute here: run the comparison as a separate job"); a scfout the parse
   and the transcription evaluate.
2. *Determined*, judged first: an entry of a table checks A and B judge (QUEER's
   physical tables and PyATB's literal one) is determined when its rounding without
   the edge placements (which the sums bound) lies within 0.1 times 1e-4 of the
   table's largest value; every entry is when the table vanishes (its values and
   rounding below 1e-6 of the broadened magnitudes it cancels from, and its values
   within 1e3 times its rounding: near-degenerate partners' transitions carry
   magnitudes far above what their sum cancels from, so the share alone would call a
   nonzero table vanishing; see *Vanishing tables*); an entry at an outer edge whose
   placements are not tested exactly is not determined.
3. *Check A, QUEER's physical value* (a mismatch is `differs_unexplained`, a QUEER
   defect, and fails whatever PyATB printed): `sigma_uA_V2`, without smearing
   `binned_uA_V2`, and for a collinear scfout `legacy_single_channel` (PyATB's spin
   factor times the spin-up channel; `sigma_uA_V2` is both channels) each equal the
   physical mode at every determined entry within its rounding tolerance, and their
   sums over the photon energies up to each one within the summed tolerance: a
   transition on an interior bin edge may land on either side of it (its two rows allow
   either placement), never vanish or count twice; on the table's first or last edge
   each k-point's transitions lie inside the table or outside it (*Outer edges*), nothing
   in between. An entry the scfout does not determine
   counts at most its tolerance, so it never fails a comparison; it makes the
   comparison `not_comparable` (the scfout does not determine the values to the
   agreement tolerance: QUEER's value ill-conditioned at near-degenerate levels, or
   PyATB's own value changing with its eigensolver's basis inside degenerate levels).
   QUEER's photon energies are PyATB's to 1e-9 eV, and its own `shift_current.dat`
   holds the lattice's handedness times its table. A failure here stands; the report
   still records the determination and whether PyATB's table was reproduced.
4. *Check B, PyATB's table* (`not_comparable`, "PyATB's table is not reproduced",
   with the gap): PyATB's printed table equals the PyATB-literal mode within its print
   precision (half a unit in the seventh significant digit of each `%15.6e` value) and
   the literal mode's rounding tolerance, with the same condition on the sums over the
   photon energies and the same placements at its window's ends. A reader or
   transcription difference is never a pass.
5. *Classes*, per component, with `tol = 1e-4 max abs(sigma_P) + print precision +`
   both rounding tolerances (`+ 2e-6` of the cancelled magnitudes when both tables
   vanish): `agree` when QUEER's value in PyATB's layout (the lattice's handedness,
   the legacy channel of a collinear scfout, the bins without smearing) meets PyATB's
   table, `agree_after_convention` when that layout is not the identity;
   `agree_after_convention` when it meets it once the recomputation's own convention
   step is applied (PyATB's SI constants and Hartree constant on QUEER's transitions);
   otherwise `differs_documented` with the measured gap and an attribution computed by
   the recomputation, which switches PyATB's ingredients on one after another
   (`constants`, `hartree`, `window`, `bins`: its rows and truncation, `scissors`: its
   scissored derivative kernels, `cut`: its method-1 integrand and the Bloch sums of
   its reader's operators) and reports each step's largest change and the residual
   (QUEER's and PyATB's gaps to the recomputation, bounded by checks A and B). Because
   checks A and B hold, the steps telescope from QUEER's value to PyATB's table: every
   documented difference is explained by construction. **Approved deviation from the
   plan (the user, 2026-10-17)**: a difference fully explained by a reproduced, bounded
   mechanism is documented without an open-ended cause.

*Vanishing tables.* A table vanishes when its values and its rounding both lie below
1e-6 of the broadened magnitudes it cancels from (`cancelled_scale_uA_V2`: the same
kernel of `|r^b_mn| |r^c_nm;a|`, what the k-points' contributions cancel to zero), and its
values within 1e3 times its rounding: a response a symmetry forbids, zero but for
rounding and the scfout's own asymmetry. Its values are then rounding of that cancelled
scale, and values below that rounding cannot be judged: the comparison judges such a
table as zero, not its values. On the real calibration scfout at 18 x 18 x 1 with
`occ_band` 16 the physical table peaks at 1.6e-6 microA/V^2 against a cancelled scale of
322, and its rounding, which four natural near-degenerate k-points dominate, reaches
6e-7: QUEER's estimator scaled by 1 + 1e-2 at those four k-points still passes, while
1 + 0.3 there fails. A pass on a
vanishing table therefore says that both tables are zero within about 1e-6 of what
cancels in them, nothing about QUEER's estimator below that; it is not chased further.
The report says so wherever it judges one: `rules.determined.tables.<table>.vanishing`
and `rules.determined.vanishing_tables`, the `determined` precondition (`<table>
vanishing (largest ... of the ... microA/V^2 that cancel in it: judged as zero, not its
values below that rounding)`), the check-A quantity of that table, which begins with the
same words, and, when both tables vanish, every component's class (`both vanish and
agree as zero`).

*Outcomes confirmed on 2026-10-18*, two that the four-basis first-order rounding moved
or kept (measured with the comparison itself, PyATB stood in for by the loop port):

- Kramers pairs split by 1e-8, 1e-7, 1e-6 and 1e-5 eV at eta 0.1 (spinor model, 4 x 4 x 4)
  are `pass_with_documented_differences` by `cut`, no longer `not_comparable`. QUEER's
  physical value is determined and continuous into the exact degeneracy: its peak is
  4.485971737525 microA/V^2 at a split of 1e-8 eV and at exact Kramers pairs, the same
  to 5e-16 of it (3e-10 at 1e-5 eV); check A holds at 1 % of its tolerance or less; its
  rounding (9.6e-6 at 1e-8 eV) lies within the determined share (4.5e-5). PyATB's table is
  reproduced (its four bases agree to 3e-14). PyATB's method 1 regularizes every such
  pair (`dE/(dE^2 + 0.04^2)` within 1e-4 eV) and so moves its table by 0.0737 microA/V^2,
  1.6 % of the peak, at every split from 1e-8 to 4e-5 eV, while at the exact degeneracy
  its cut changes nothing (1.6e-14, a `pass`): the documented gap (0.0737) is that cut
  step to a residual of 5e-7 (the constants and Hartree steps). This is the intended
  outcome: a reproduced, bounded PyATB mechanism. At eta 0.01 the split of 1e-8 eV stays
  `not_comparable`: QUEER's rounding there (8.3e-4) exceeds the determined share (2.0e-4).
- The real calibration scfout at 18 x 18 x 1 with `occ_band` 6 stays `not_comparable`
  (`determined`) at eta 0.1 and 0.01. QUEER's physical table vanishes (peak 1.6e-6 and
  1.4e-5 microA/V^2) and passes check A (0.5 % of its tolerance); PyATB's cut at the four
  natural 5.8e-5 eV pairs breaks the inversion, so its table does not vanish (peak 9.4e-5
  and 4.7e-4), and its rounding (1.0e-7 and 6.9e-7) is about 100 times the determined
  share (9.4e-10 and 4.7e-9) at 402 and 70 entries; both the hundredfold per-k-point
  basis envelope (3.1e-8) and the first-order rounding (7.1e-8) exceed it on their own.
  What was measured lies below the share: the four bases' spread 9.3e-11, another BLAS
  kernel (`OPENBLAS_CORETYPE=Sandybridge`) moving PyATB's literal table by 6.8e-11 and the
  loop port differing from it by 5e-11 (at eta 0.01: 8e-10, 4.4e-10, 5e-10). The bound is
  not met, so the comparison does not claim PyATB's 9.4e-5 artifact table to 1e-5 of its
  size: QUEER's value is verified by check A, the attribution of PyATB's table is not
  made. This is intended; widening the bound for this scfout would be a tuned constant.

*Information, not gates*: the symmetry of the two tables (the components above 1e-4
of each table's largest value and the signs of their integrals over omega), the
candidate's self-convergence estimate, for a collinear scfout the gap between QUEER's
physical value (both channels) and PyATB's table, and the recomputation's facts
(transitions PyATB selects, the k-points its method 1 regularizes and those with an
exact degeneracy, the smallest such spacing, the records' distances from their
translation identities, the bases' seed, workers and seconds).

*Cost.* The recomputation is a full grid evaluation, four times per mode, vectorized
over batches of 32 k-points and spread over `--workers` processes (QUEER's process
executor; default every CPU), its memory planned like the runner's. Its work is
k-points x bands^2 x (bands + photon energies / 8), summed over the channels: the
eigensystems and integrands grow as the cube of the bands, the kernels as the square of
the bands times the photon energies. One core does about 3e6 of it a second on the
calibration scfout (30 bands; 51, 601 and 2401 photon energies alike), so the default
budget of 8e9 is some 45 minutes on one core; its 18 x 18 grid with 601 photon
energies takes 10 s, 36 x 36 takes 39 s.

*Scenarios* (`tests/numerical/test_generator_parity_pyatb_shift.py`, with
`tests/numerical/pyatb_shift_port.py`, a loop-by-loop port of PyATB's reader and method
1 that shares no code with QUEER's reader or the recomputation, as PyATB): a wrong
QUEER value (negated, doubled, zeroed, scaled by 1 + 1e-6, columns swapped, rows
rolled, one component zeroed) fails with or without diagnostics that agree with it and
whether PyATB printed its own table or the wrong value, also on weakly broken symmetries
and on pairs split across the occupation boundary; a defect in QUEER's scfout reader
(its Hartree constant 27.0 eV, its reconciled Hamiltonian 1 % off, its collinear
spin-down channel parsed from the spin-up records) fails; forged diagnostics change
nothing; QUEER reporting PyATB's window, Hartree constant or truncation as its physical
value fails; an error in either table without smearing, a wrong spin-down channel or a
wrong legacy channel fails; a degeneracy tolerance too large (5e-5 eV for pairs 4e-5
eV apart) fails; an error confined to the k-points PyATB's method 1 regularizes fails;
a transition on a bin edge may change rows but not vanish or count twice; on a
histogram's first or last edge each k-point's transitions inside the table or outside it
pass in every combination, and twice, minus, a quarter or one and a half times their weight
in that row fails, as PyATB's table holding twice or a quarter of them at its window's
edge is `not_comparable` (more groups there than are tested exactly: `not_comparable`);
a vanishing table is reported as vanishing in the determination, check A and the classes;
QUEER's value
beyond its rounding only at entries the scfout does not determine is
`not_comparable`, at a determined entry it fails; a PyATB table scaled, shifted, rolled
or with one component zeroed is `not_comparable`, also at exact Kramers pairs; correct
candidates pass or are documented (scissors: `scissors`, also with transitions exactly
on bin edges; PyATB's window: `window`; pairs split by 4e-5 eV: `cut`, also across the
occupation boundary and for one band pair of them), including PyATB's default eta 0.01
on steps of 0.01 and 0.02 eV, narrow eta on fine steps, left-handed and collinear
lattices, records Hermitian only to 1e-9 eV, weakly broken symmetries whose integrands
are rounding at TRIM points, Kramers pairs at the TRIM points in orthogonal and
nonorthogonal bases, and responses that vanish; a correct candidate whose levels are
split by 1e-8 eV at eta 0.01 (QUEER's value ill-conditioned) or whose exact Kramers
pairs carry on-site dipoles (PyATB's value basis-dependent) is `not_comparable`. On the
real calibration scfout its vanishing response passes (12 x 12, occ_band 3; and occ_band
4 at eta 0.01 and 0.1, whose integrands at the TRIM points are rounding, with PyATB's
table reproduced), and on 18 x 18 with occ_band 16, where four k-points hold levels
5.8e-5 eV apart, PyATB's cut breaks the inversion that makes the physical value vanish:
12 components are documented, each by `cut`.

*Literature anchor*, on its own: `literature_anchor(run, anchor)` takes GaAs
sigma^xyz of Ibanez-Azpiroz, Tsirkin and Souza, PRB 97, 245143 (2018) at the
paper's scissors and broadening and passes a peak within 2 eta and a height
within 20 %. `tests/fixtures/literature/shift_current_gaas_anchor.json` holds
placeholders to fill from the paper; `QUEER_SHIFT_LITERATURE_RUN=<run>` runs
`tests/integration/test_generator_shift_literature_anchor.py`, which skips while
the anchor is a placeholder.

**Calibration kit.** No PyATB shift-current output of a known model existed, so
`scripts/validation/generator_pyatb_shift_kit.py --output DIR` writes one to run:
`soc.scfout` (a gapped 3D two-orbital spinor model with neither inversion nor time
reversal; every level at least 0.28 eV from every other, transitions 1.06-5.32 eV),
`soc_left.scfout` (the same on a left-handed lattice) and `collinear.scfout` (a
gapped 2D two-orbital collinear model without inversion whose channels differ),
all written by `tests/numerical/openmx_files.py` and read by PyATB's own reader,
and `test.scfout`, the real OpenMX calibration scfout (honeycomb with inversion,
spin-orbit coupling, a nonorthogonal basis with position moments), whose shift
current vanishes; PyATB `Input`s for nine cases (a Gaussian on 16^3 k-points,
`method 0`, no smearing, scissors 0.4 eV, the collinear scfout, one band pair, the
left-handed lattice, the real scfout, PyATB's default eta 0.01 on its default step
0.01); the Slurm job `shift-kit-job.sh` (8 MPI ranks, writing each case's
`inputs.sha256` first); QUEER's runs under `queer-side/` with `summary.json`, whose
`prediction` is each case's comparison against the loop port's table standing in for
PyATB's; and a `README.md` with the SHA-256 of every file and the expected verdicts
(the predictions: `pass` for all but `method 0`, `not_comparable` as not transcribed,
and the scissors, `pass_with_documented_differences` by `scissors`). Once PyATB's
outputs are back in the case directories, `QUEER_PYATB_SHIFT_CALIBRATION=1` runs
`tests/integration/test_generator_pyatb_shift_calibration.py` (data directory
`QUEER_PYATB_SHIFT_CALIBRATION_DIR`, default
`~/.cache/queer-parity-data/pyatb-shift-kit`): QUEER runs each case afresh, each must
meet its expected verdict and classes (and its documented cause), and wherever the
comparison recomputes PyATB's table its check B must hold: PyATB's printed table equal
to the transcription's PyATB-literal table, PyATB's reader and Bloch sums of a
nonorthogonal basis with position moments included.

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
