# Native quantum geometry and finite-field Hall workflows

Queer computes band-pair quantum geometry and nonperturbative, quasistatic
two-band Hall responses in a specified Cartesian momentum plane. These
calculations use the native Wannier/OpenMX covariant operators and do not import
HHH at runtime. Energies use eV, momentum uses Å⁻¹, electric fields use V/m,
and the Hall output is a sheet current in A/m.

Install the optional extended-precision tables and plotting tools when needed:

```bash
uv sync --locked --extra precision --extra plot
.venv/bin/queer crossing_scan --config configs/examples/finite_field_crossings.toml --render
.venv/bin/queer landau_zener_hall --config configs/examples/finite_field_hall.toml
.venv/bin/queer hall_harmonics --config configs/examples/finite_field_harmonics.toml
```

The default profile is `paper`, matching HHH: conduction-band curvature,
band-pair Eq8 metric exponents, gap-minimum crossings, positive paper charge
direction, an e²/h prefactor, and T1=T2=2 ps. `paper_figures` applies the
figure scale 1/(4π) once. Select `physical` explicitly for physical electron
charge, conduction-minus-valence curvature, DDP exponents, and geometric,
Stokes, and twisted Landau–Zener corrections. `physical_eq8` keeps the physical charge, curvature, and
current normalization, explicitly selects gap-minimum crossings and Eq8 metric
exponents, and disables geometric, Stokes, and twisted Landau–Zener phase
corrections. Every response output records the resolved conventions,
relaxation settings, crossing policy, quadrature, field range, and engine source
hash. HHH's fitted DDP and Eq8 fallback exponents retain their source and
validity flags; an unavailable diabatic projector falls back to gap minima
with `diabatic_unavailable`. Wavefunction phases still require wavefunctions,
and unsupported generalized-model operations raise capability errors.

Berry curvature uses Ω = −2 Im⟨∂ₓu|∂ᵧu⟩ throughout. The Hall convention is
σ_xy=C e²/h, so `hall_chern(flux)` returns −flux for flux normalized by 2π.
The Fourier outputs obey `c=(a-i*b)/2`; a signed sine coefficient is distinct
from the magnitude of its complex coefficient.

The general `queer.observables.chern_number` API reports the geometric
number `C_geo=∫Ω/(2π)`. Convert it with `hall_chern(C_geo)` when comparing
HHH's Hall number. Both APIs use the same Berry curvature; equilibrium Hall
conductivity already uses σ_xy=−C_geo e²/h=C_H e²/h.

Critical diagnostics report the conduction strength `C_s` and manuscript
strength `C_paper=-C_s`. `transport.critical.c_from_harmonic` defaults to the
printed large-order constant; `finite_N=True` selects the exact finite-order
factor. `c_from_slope` implements the fitted device-signal inversion without
including its fitted intercept.

Relaxation times use seconds. `parameters.relaxation.coherence="paper"` treats
`T2` as pure dephasing, while `"bloch"` treats it as the total coherence time
and requires `T2 <= 2*T1`. Set both `T1="inf"` and `T2="inf"` explicitly
for no relaxation; this setting remains distinct from unavailable diagnostics,
which are stored as JSON `null`.

Physical geometric phases use the native dipoles and physical basis links.
`parameters.phase.phase_nodes`, `phase_max_nodes`, and `phase_tolerance` control
refinement; outputs retain each error and exhausted budget. Shift-vector checks
also compare two step sizes. Model-validity flags and numerical-convergence
flags remain separate, including strong-twist, unresolved phase, and finite
window diagnostics.

## Workflow inputs

| Workflow | Numerical product |
| --- | --- |
| `geometry_grid` | Cartesian or polar sampled pair geometry |
| `crossing_scan` | Ragged crossings with gaps, metrics, Eq8/DDP fields, and validity flags |
| `landau_zener_hall` | Signed-field sheet Hall current |
| `hall_harmonics` | Sine, cosine, complex Fourier coefficients, and DC current |
| `harmonic_map` | Harmonics over a model family or analytical parameter scan |
| `family_report` | Basis consistency and critical displacement-field diagnostics |
| `critical_hall` | Verified critical-point asymptotes and cutoff diagnostics |
| `fit_hall` | Activation/oscillation fits over an explicit field window |
| `houston_line` | Houston-state propagation for a selected field line |
| `lindblad_line` | Two-band time-domain propagation with T1/T2 relaxation |
| `berry_flux` | Finite-window occupied-subspace flux and grid-refinement error |
| `paper_figure` | Named paper panels with their explicit conventions |

Use exactly one of `inputs.model`, `inputs.geometry_grid`, or
`inputs.family_manifest`. Analytical models, grids, families, and artifact-based
fits do not require a material directory. File-backed models use
`inputs.material_dir` and the same native files as other Queer workflows.

For an analytical reference:

```toml
workflow = "crossing_scan"
[inputs]
model = "fourband"
[parameters]
offsets = [0.0, 0.0001]
[parameters.model]
x = 0.9
```

`x` is the four-band displacement ratio. Other `parameters.model` keys are
native `FourBandParams` values; `warping`, `warping_matrix`, `d3_layer`, and
`include_identity` configure the analytical model.

For Wannier or OpenMX calculations, specify the band pair or occupied count and
an oriented plane. The origin uses fractional reciprocal coordinates; the axes
are Cartesian orthonormal unit vectors. The axis orientation fixes the Hall
sign, and `parameters.drive.angle` rotates the drive within this plane.

```toml
workflow = "landau_zener_hall"
[inputs]
model = "wannier"
material_dir = "../../data/materials/my_slab"
hr = "slab_hr.dat"
tb = "slab_tb.dat"
[parameters]
band_pair = [9, 10]
fields = [-10000.0, 0.0, 10000.0]
[parameters.plane]
origin = [0.0, 0.0, 0.0]
direction = [1.0, 0.0, 0.0]
transverse = [0.0, 1.0, 0.0]
```

Native Wannier loading discovers companion position/centre/lattice files by
seed. Hamiltonian-only, centre-embedded, and full-position geometry retain
their explicit completeness metadata. Full position operators contribute
basis curvature. Quantum metrics describe the retained-basis interband metric;
they do not include discarded continuum orbital spread. All consumed
sidecars enter the input checksum list, including auto-discovered files.

Nonorthogonal OpenMX calculations retain `S(k)`, position operators, spin
channels, and their derivatives. Complex continuation and near-critical patches
require validated support; ordinary geometry does not imply support for these
precision-sensitive operations. Orthogonal Wannier models can derive a layer
projector from their slab-unwrapped centres; an explicitly supplied projector
takes precedence. Nonorthogonal models need a supplied covariant layer operator.
Cross-family character tracking needs a validated
orbital correspondence; nonorthogonal cross-model overlaps are not inferred
from coefficient dot products.

Physical sheet currents include the model's spin multiplicity: nonmagnetic
OpenMX spatial bands represent two identical spins, while explicit collinear
channels and SOC spinors have multiplicity one. Geometry and crossing
probabilities remain per band pair. Geometry archives retain this multiplicity;
older archives without it describe a single pair with multiplicity one.

`houston_line` requires certified dynamics in a fixed orthonormal basis;
`lindblad_line` further requires exactly two bands. Their signed fields must
be nonzero. Lindblad parameters include positive `dphi` and `basis` equal to
`adiabatic` or `superadiabatic`, with relaxation times in seconds.
`finite_field_lindblad.toml` is a template for a supplied two-band Wannier model.
`berry_flux` includes native overlap/position links and compares grids of
`n` and `2*n-1` points. Its one- or two-component positive `window` is a plane
half-width in Å⁻¹; this finite-window flux is not a whole-Brillouin-zone Chern
certificate. Geometry grids cannot supply these wavefunction-based validators.

A geometry-grid NPZ contains model-frame tensor fields and JSON metadata.
Polar sampling also stores optional Γ fields (`gamma_*`) and the independent
Γ gap. Below the first radial node, quadratic interpolation in r² preserves
the patched core. Older grids without Γ data use quadratic extrapolation with
a cubic consistency certificate and an unresolved-core flag when necessary.
Interpolation uses gap², curvature × gap³, and metric × gap⁴. Polar grids are
periodic in angle; extrapolation and outside-domain samples have flags.
Unresolved nonpositive interpolated gap² fails with a refinement message.
Geometry-only grids cannot supply wavefunction phases or Houston validation.

Family manifests accept `n_occ` and `members`, with each member specifying
`label`, numerical `D`, and `files` relative to the manifest. Supported file
keys are `hr`, `tb`, `centres`, `wout`, `wsvec`, `poscar`, `scfout`, and `grid`.
`harmonic_map` accepts `normalize_by_Dc=true` for validated tracking or a
specified nonzero numerical critical field. Every member retains its own
diagnostics. Analytical maps instead specify `model_values` and
`model_parameter` (default `x`).
`family_report` accepts `reference`, `min_weight`, `jump_tol`, and `mix_tol`
for explicit tracking/critical-field acceptance settings; unavailable
cross-model overlaps remain reported as unavailable.

Family tracking removes the Bloch phases from detected orbital home-cell
changes before comparing states. At a nonzero plane origin this requires
matched Wannier centres, a common lattice, and equivalent fractional origins.
Reports retain orbital permutations, integer home-cell shifts, tracked
character, and accepted or rejected roots. Centered spectra preserve small
gaps separately from absolute energy references; reported assembly precision
floors describe terms that double-precision Hamiltonian assembly may lose.

## Checkpoints and reusable artifacts

Hall and harmonic workflows commit the transverse layout, complete solved
lines and their Laplace tables, field evaluations, and Fourier products through
Queer's exclusive-owner `RunStore`. Geometry, crossing, fitting, flux, and other
small workflows commit their complete numerical product as a final stage.
A line checkpoint contains ordinary arrays and
JSON metadata; ragged segment/crossing data never require pickle. Interrupted
runs reuse validated lines and completed stages:

```bash
.venv/bin/queer landau_zener_hall --config calculation.toml --run-dir results/my-run --resume
.venv/bin/queer render results/my-run
.venv/bin/queer export results/my-run --profile finite_field
```

Rendering and exports consume checksummed artifacts and do not reload input
models. The export profile copies `result.npz`, scientific `metadata.json`, and
a product checksum manifest. A sampled `geometry_grid` result NPZ can be used
directly as `inputs.geometry_grid`. `fit_hall` accepts inline `fields`/`values`
or `inputs.response_artifact`, which may name an NPZ or a completed run. Fits
require `window`; multidimensional artifacts also require `member` and/or
`harmonic` selections as applicable.

The run identity hashes every family manifest leaf and consumed model sidecar.
Scientific input changes reject resume. Execution batch size and worker
allocation may change without changing the scientific identity.

`parameters.cache_dir` optionally reuses preparation across separate runs.
Its path resolves relative to the source TOML and is recorded as a resource
location. Cache entries use model-content and preparation-option signatures;
each run still commits owner-verified line checkpoints. The cache directory
does not alter the run's scientific identity.

## Paper figure recipes

The `finite_field_paper_*.toml` examples cover these numerical products:

| Recipe | Product |
| --- | --- |
| `fig4b` | Single-passage probability versus field |
| `fig4d` | Two-passage interference probability and envelope |
| `fig5ab` | Energy bands colored by pair metric and curvature maps |
| `fig5c` | Harmonic maps over displacement ratio and field amplitude |
| `fig6bd` | Critical curvature C/k reference and gap-edge pair metric |
| `ext7a` | Extended harmonic parameter maps |
| `ext7e` | Gap-minimum and diabatic critical-field/interference-frequency scans |
| `ext9b` | Scaled critical harmonics N bN versus amplitude/N |

Run `.venv/bin/queer paper_figure --config configs/examples/finite_field_paper_fig4d.toml
--render` for a small probability recipe. `parameters.figure` selects the
recipe and `parameters.recipe` supplies its overrides, such as `x`, `fields`,
`xs`, `k`, `kmap`, `model_parameters`, or `options` as accepted by that recipe.
The harmonic recipes use the native layout/line/Fourier checkpoints and
configured execution limits. The `fig5c` and `ext7a` example TOMLs use
`harmonic_map` with the named profile, retaining the Slurm `--split-members`
route and all Fourier orders. Increase the sample lists in these example maps
for publication resolution, or use `paper_figure` to select the complete
reference grids and recipe orders.

Render a completed run to inspect the result. Named profiles retain explicit
paper normalizations and relaxation assumptions; every artifact records these
choices. `benchmarks/finite_field.py` reports preparation, field-evaluation,
and Fourier timings plus numerical diagnostics. Optional compiled response
kernels require `uv sync --extra acceleration` and `parameters.acceleration=true`;
unsupported phase corrections produce a capability error.

The explicit analysis helpers
`queer.finite_field.analysis.paper.load_paper_maps(path)`, `model_maps(...)`,
and `compare_maps(...)` support comparisons with supplied digitized NPZ maps.
The repository bundles digitized curve anchors in
`tests/fixtures/finite_field/paper_targets.json`, with a pinned checksum and
source provenance. Map NPZ inputs are supplied separately. Pass their file path
explicitly; production calculations never load paper data automatically.
Run the corresponding scientific checks:

```bash
uv sync --locked --group dev --extra precision --extra plot
.venv/bin/python -m pytest -m paper
.venv/bin/python -m pytest -m slow
.venv/bin/python -m pytest -m precision
```

The `precision` checks need the optional `mpmath` extra. Slow convergence and
time-domain acceptance checks can take longer than ordinary workflow tests.

The packaged `queer/finite_field/SOURCE_MANIFEST.json` pins the HHH source
snapshot by content SHA256. Reference fixtures retain the corresponding test
and digitized-data fingerprints. Result metadata separately records the
installed QUEER engine's source hash.
