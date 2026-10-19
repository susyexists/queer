# Migration notes

## QUEER 1.6.1 Generator driver environment

The Generator's PythTB driver records `provenance.numeric_environment` from core
`497beb4c…` on, field for field as QUEER does. The quantum-geometry comparator accepts
that core, so a mesh difference between a driver reference and a QUEER candidate of
one numeric environment fails instead of being not comparable. No values change, and
the algorithm identity stays `1.6.0-histogram-rows-v1`.

1.6.1 also carries the 1.5.1 Wigner-Seitz certification, which 1.6.0 lacks: on 1.6.0,
`io.wannier_ws.recompute_wsvec` still refuses a correct image search in a supercell much
longer along one axis than the others.

## QUEER 1.6.0 Generator OpenMX and quantum-geometry runners

QUEER now also runs the Generator's PyATB steps and its quantum-geometry step:

- `QUEER_AHC` and `QUEER_CHERN_NUMBER` read the OpenMX `scf.scfout` through
  `OpenMXModel` and integrate the anomalous Hall conductivity and the Chern number
  with QUEER's own engines (direct, Kubo, FHS). `export_run` writes PyATB's layouts:
  the one-line AHC vector, the FHS table and `Out/Chern_Num/chern_number.dat`.
- `QUEER_SHIFT_CURRENT` computes the physical shift current (both spin channels of a
  collinear run, with the spin-up channel reported beside it as PyATB's legacy single
  channel) and writes PyATB's `shift_current.dat` layout.
- `QUEER_QGEOM` is a port of the Generator's quantum-geometry driver: the layer
  geometry, the triangle rules and the adaptive integrator (`physics.qgeom`,
  `observables.qgeom`), with the `_tb.dat` full position level and the `.eig` check.
  `qgeom_replay_vertices` and `qgeom_time_stop` make a run replay a reference's
  sampling. The run also writes QUEER-only comparison records: `Out/qgeom_vertices.json`
  (the mesh's vertex keys), `Out/qgeom_response_full.json` (the unrounded curves) and
  `qgeom.integration.decision_ties`.

Every Generator run records `provenance.numeric_environment`: the Python, NumPy and
SciPy versions, the machine, NumPy's SIMD targets and each BLAS library's build and
core type.

The comparator (`python -m queer generator compare`) covers the new steps:

- PyATB's AHC and Chern runs, in their Grid and refined-Grid tiers;
- the shift current, by a full independent recomputation from the scfout (it needs
  `--scfout`);
- the quantum geometry, by deterministic replay. A mesh difference fails between runs
  of one numeric environment and is not comparable between different ones.

`docs/generator_services.md` gives each rule, its tolerances and its measured
outcomes.

Corrections that change values:

- `translate_pyatb`: the Hall export sign was inverted, and the default convention is
  now `auto`, resolved per package. `adaptive_grid_threshold` is in square angstroms
  and is no longer converted from Bohr^2. Explicit Fermi energies move from PyATB's
  Hartree (27.211324570273 eV) to QUEER's.
- `shift_current_spectrum` without smearing places each transition in one histogram
  row by an integer index. Before, a transition exactly on a bin edge could land in no
  row or in two. Away from the edges, and with every smearing, the values are
  bit-identical. `ShiftCurrentSettings` gains `bin_rows`.
- The Generator SLAB edge count follows the driver's rewritten rule: the weighted
  flow of half-ribbon weight with rank alignment, counted only where the bulk k mesh
  resolves the gap. A gap without edge levels is never certified.

Algorithm identity is `1.6.0-histogram-rows-v1`: older partial runs require a fresh
run; completed artifacts remain renderable.

## QUEER 1.5.1 Wigner-Seitz certification in long supercells

`io.wannier_ws.recompute_wsvec` no longer refuses a correct image search in a
supercell much longer along one axis than the others, such as a 6x6x6 mesh on a
54 Å cell. The per-axis plane bound that proves no image outside the
`[-search_size, search_size]` cube can tie or beat the selected one could not
decide such cells. Pairs it leaves open now go through an exact lattice walk over
the images within that distance. The selected images are unchanged, so models that
certified before give identical tables, and a search that is genuinely too small
still raises `increase search_size`.

Existing workflows keep their values. Checkpoints keep the `1.4.1-refactor-v1`
algorithm identity.

## QUEER 1.5.0 Generator parity runner

QUEER now runs the Generator's PythTB steps itself. `python -m queer generator run
--step QUEER_BANDS|QUEER_TOPOLOGY|QUEER_BERRY|QUEER_SLAB --params ... --workspace ...`
reads the same Wannier90 inputs and parameter file as the Generator PythTB driver and
writes `queer_results.json` in the driver's `pythtb_results.json` schema, with the
same `Out/` files. Where PythTB was called, QUEER's own models and observables are
used instead:

- `WannierModel.from_table`, with averaging hermitization and PythTB's cutoff rules;
- eigensystems;
- `wilson_loop` with exact diagonal links;
- `band_geometry` Kubo curvature and the finite-basis metric;
- `velocities` for the Fermi sea.

The driver's own numerical rules moved into QUEER:

- Wigner-Seitz images and vacuum folding (`io.wannier_ws`);
- Chern windings and the Z2 largest-gap rule (`physics.topology`, `observables.topology`);
- Fermi-sea curvature (`physics.geometry`);
- block-Toeplitz slabs with cluster-summed edge weights (`observables.surface`).

The historical estimators stay in `compat.generator`. Every port is checked against
the driver's original functions, which are vendored as test oracles.

The command starts from a standard-library boot, so a host without NumPy or SciPy
gets a `queer_env_missing` result instead of a traceback. It also reads `.win`,
`.eig` and the `wannier90_band.*` files. The comparator
(`queer.compat.generator.parity`, `python -m queer generator compare`) judges a QUEER
run against the run it reproduces. Its report records:

- preconditions;
- a semantics profile for older driver revisions;
- data-driven tolerances;
- classes per quantity.

Other changes:

- `compat.generator` is now a package. Its historical names import lazily and are
  unchanged.
- TOML configurations accept `win`, `tb`, `centres` and `wout` inputs,
  `interpolation = "auto"` and a `position` mode.
- `QUEER_RUNSTORE_LOCK=none` disables advisory locking where a file system lacks
  `flock`. Run manifests are written in batches.

Existing workflows keep their values. Checkpoints keep the `1.4.1-refactor-v1`
algorithm identity.

## QUEER 1.4.1 shared-kernel and workflow refactor

Repeated occupation, broadening, Green-function, Lindhard, EPW, covariant geometry
and length-gauge response arithmetic now have native owners in `queer.physics`.
Models and observable adapters retain their existing public signatures, shapes,
defaults and scientific metadata. Legacy top-level imports remain available in 1.x.
The [equation registry](equations.toml) records specialized backends separately.

Thermal-energy occupations retain the half-filled zero-energy limit; native Kelvin
occupations retain an unoccupied state exactly at the chemical potential. EPW keeps
its negative imaginary regulator and documented reduced-coupling normalization.
Paper/physical finite-field choices and historical conversion constants remain
explicit. This reorganization does not select new scientific defaults.

Structure readers share one POSCAR record parser. Tuple readers continue returning
fractional coordinates and species group IDs, including VASP4 records without
element labels; native structures still require chemical species. Simulation
expansion preserves unwrapped positions, image-major atom order and restart metadata,
while structure preprocessing wraps periodic coordinates and records provenance.

Workflow registration, input discovery, domain handlers and result reduction now
have separate owners. Execution and artifact APIs retain their existing contracts.
Algorithm identity is `1.4.1-refactor-v1`: older partial runs require a fresh run;
completed historical runs remain renderable from their saved artifacts. Preparation
identities include shared numerical source owners. The packaged finite-field
`SOURCE_MANIFEST.json` continues to record the original HHH donor snapshot; it is
distinct from the current QUEER engine fingerprint.

## Finite-field parity with current HHH

Omitted finite-field options now select `paper`, matching HHH, instead of
`physical`. This applies to library options, direct calculations, workflows,
and canonical examples. Request `preset="physical"` explicitly to retain the
corrected physical convention. `physical_eq8` remains available for the
physical charge/curvature normalization with metric exponents.

`fig5c` and `ext7a` now use T1=2 ps and T2=∞. `fig4d` defaults to V_L=1.6 meV
and diabatic crossings. These corrections change interference fringes and
some high-harmonic signs. The default Laplace-table density is eight per decade.
Critical inversion uses the printed asymptotic constant by default; specify
`finite_N=True` for the exact finite-order inversion. Physical inversion also
undoes the conduction-minus-valence factor, and critical diagnostics include
`C_paper=-C_s`.

HHH's fitted DDP/Eq8 and unavailable-diabatic fallback paths now carry the same
explicit source and validity flags. Missing wavefunctions and unsupported
native operators still fail rather than supplying fabricated corrections.
`vlc_reference="paper"` is rejected: construct
`FourBandParams.paper_vlc_matched()` instead.

New polar grids retain Γ anchors; old grids remain readable with core
extrapolation diagnostics. Changed scientific settings and source fingerprints
invalidate preparation caches. Completed older artifacts keep their resolved
settings and remain renderable; start a fresh calculation to use these fixes.
Split family jobs preserve each member's interpolation, plane, and band selection.

## QUEER 1.2 migration

Native OpenMX support adds `OpenMXModel`, `read_scfout`, `read_pao` and typed
`ObservableResult` APIs, plus electronic, surface, unfolding, geometry, transport
and optical workflows. Select `inputs.model="openmx"` and `inputs.scfout`.
File EF is the default; `parameters.ef` overrides the band reference. OpenMX
does not use Wannier interpolation modes. Existing `energies()` axes and energy
zero are preserved; `eigenstates().energies` and `eig()` use absolute eV.

New response chemical potentials use absolute eV and new temperatures use
`temperature_kelvin`; older `temperature` parameters retain thermal-energy eV.
Nonmagnetic states carry occupation degeneracy two; explicit spin channels carry
one. Collinear bands remain channel-major and SOC orbitals are all-up followed
by all-down. Signed Mulliken weights are not clipped. These conventions correct
upstream spin-channel loss and interleaved-basis assumptions.

Generalized Green functions use `(omega+EF+i eta)S-H` and `Tr(SG)` spectra.
Derivatives include overlap and position connections. Full direct curvature
and projected velocity-Kubo curvature are distinct; quantum metric uses the
retained-basis interband definition. SciPy constants replace legacy rounded
conversions. Optical response follows retarded absorption signs. See
[native workflows](openmx_workflows.md) and scientific conventions for details.

Algorithm identity is `1.2.0-native-services-v2`; older checkpoints require fresh runs.
Completed older artifacts remain renderable. Slurm accepts every workflow's
complete configuration, including repeated `--config` arguments for case arrays;
the original momentum/V0 array remains supported.

Existing supported model/module calls and script arguments remain available in 1.x.
Use `WannierModel`, `EPWModel`, validated TOML workflows, and installed `queer` commands
for new consumers. Legacy adapters are scheduled for removal in 2.0; undeclared
third-party objects accidentally exported by wildcard imports are no longer public.

Numerical corrections intentionally change historical outputs. Conventional Wannier
interpolation now parses orbital indices, divides degeneracies, and uses the positive
Fourier phase. Hamiltonian and eigenvalue methods share one implementation. Invalid
Hamiltonians fail instead of being silently symmetrized or reduced to real eigenvalues.
Named paths use the input reciprocal basis rather than an unrelated standardized cell.
Spectral ARPES uses signed omega = E-EF and kinetic energy photon_energy + omega,
so occupied states lower kinetic energy consistently with positive binding-energy meshes.

Specify the interpolation mode explicitly in production configuration. `standard`
uses conventional HR weights; `ws_distance` requires complete pair-specific shift
mappings; `ndegen_applied` handles expanded, already-weighted exports and does not
apply shifts again. Unit degeneracies alone cannot establish export provenance.

First-band selection and empty arrays now behave consistently. ARPES inaccessible
points use NaNs/masks; positive broadening is required. EPW metadata overrides,
file/branch ordering, and self-energy execution are corrected. Legacy temperature
arguments remain thermal energy in eV. Susceptibility and nesting are distinct outputs.

Local execution now defaults to one worker, with explicit CPU/native-thread limits.
Data/results helpers use working-directory defaults with environment overrides.
Install extras for named paths, plotting/tabular compatibility, animation, and notebooks.
Supported scripts require an installed package and no longer modify `sys.path`.
Legacy V0 script outputs now include isolated run manifests/raw arrays and a `figures/`
directory; explicit output directories must be new. Historical scientific defaults
remain confined to compatibility scripts.

CI validates clean installations, supported notebooks, and synthetic workflows. Release
artifacts are built after acceptance gates and attached to CI runs; remote publication
is configured separately when a distribution destination is chosen.
## Native generator scientific services

Generator provider integration remains a separate project. Queer now supplies
native structures and format I/O, symmetry paths, phonons, force-callback NVT
and FIRE, recorded-step frame selection, dataset export, imported band analysis
and OpenMX Hamiltonian graph preprocessing. Existing `read_poscar` tuples and
model energy-array conventions remain available.

Use [the service migration guide](generator_services.md) to translate legacy
parameters and export completed runs with `queer export RUN_DIR --profile generator`.
Scientific arrays use explicit native units; historical units, signs and display
recipes live in `queer.compat.generator`. Optional model bridges continue using
external trained architectures. Core/plot/symmetry installations no longer
require ASE or SeekPath.
