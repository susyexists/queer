# Native OpenMX workflows

SCFOUT input is selected explicitly. The file chemical potential is the default
band reference. Cartesian workflows use its embedded lattice and atoms. Named
paths require complete species labels or a matching supplementary POSCAR.

```toml
workflow = "optical_conductivity"
[inputs]
model = "openmx"
material_dir = "/path/to/calculation"
scfout = "material.scfout"
# openmx_input = "material.dat"  # validated supplemental orbital labels
# pao_dir = "PAO"               # actual matching radial orbitals for unfolding
[parameters]
grid = [8, 8, 8]
omega = [0.5, 1.0, 1.5, 2.0]
eta = 0.05
temperature_kelvin = 300.0
# chemical_potential = -5.0     # absolute eV; omitted means file EF
[execution]
workers = 2
native_threads = 1
batch_size = 4
memory_mb = 1024
```

Run `queer optical_conductivity --config calculation.toml`. Rendering is optional
(`--render` or `queer render RUN`) and uses only stored artifacts. Python
functions are exported from `queer.observables`. Their keyword parameters also
form the strict TOML interface; `grid` supplies `kpoints` when required. Names
follow lowercase upstream task names, with `bands` retaining the existing command.
`dos`, `quantum_metric` and `second_order_static` are additional explicit commands.

| Group | Workflow names | Additional inputs or controls |
| --- | --- | --- |
| Electronic | `bands`, `fat_band`, `dos`, `pdos`, `jdos`, `fermi_energy`, `fermi_surface`, `find_nodes`, `spin_texture`, `reduce_basis` | Bands: `kpoints` or `kpath`/`n_points`, collinear `spin`; scissors requires `n_occ`. Densities: `omega`, `eta`, optional orbital `groups`. Fermi solve: `electron_num`. |
| Surface | `surface_state` | `kpoints`, `surface_direction`; `method="bands"` gives slab bands. `omega` is required for `"green"`, `"slab"` or `"direct_green"` spectral modes; `slab_layers`, `eta`. |
| Unfolding | `bandunfolding`, `bandunfolding_spin_texture` | Integer `supercell_matrix`, `ecut` in eV; orbital labels and matching PAO radial functions. |
| Geometry | `berry_curvature`, `quantum_metric`, `chern_number`, `chirality`, `wilson_loop`, `polarization` | Complete optional `occupied_bands` manifold. Chern: `grid`, `plane`, `engine`. Chirality: `kpoint`, Cartesian `radius`. Wilson: ordered `kpoints`. Polarization: optional explicit `ionic_charges`. |
| Integrated | `ahc`, `anc`, `berry_curvature_dipole`, `orbital_magnetization`, `shc`, `drude_weight` | `grid` or `kpoints`/`weights`; absolute `chemical_potentials`; `temperature_kelvin`; appropriate broadening; AHC direct/Kubo/FHS. |
| Transport | `boltz_transport` | Absolute `chemical_potentials`, `temperature_kelvin` scalar/scan; CRTA `relaxation_time` in fs or EMPC `deformation_potential` in eV and `young_modulus` in GPa. |
| Optical | `optical_conductivity`, `shift_current`, `cpge`, `shg`, `pockels`, `second_order_static` | `omega` is photon energy in eV; `eta`, absolute `chemical_potential`, `temperature_kelvin`, additive optional `scissors`. Static omits `omega` and requires an insulating manifold. Optional CPGE `relaxation_time_seconds`. |

`optical_conductivity` stores conductivity and dielectric tensors together;
`dielectric_function` is a Python convenience API. Surface output contains slab
bands, semi-infinite surface spectra or layer-resolved finite-slab spectra.
Fermi surfaces are sampled energy and crossing masks. Node searches minimize
adjacent same-channel gaps and deduplicate periodic images; they are heuristics
and can miss nodes. Basis reduction reports orbital importance and
retained indices, rather than claiming a newly fitted Hamiltonian.

PAO files are supplied through `inputs.pao_dir` or a `[inputs.pao_files]` mapping
from the input basis's PAO key to a file. Optimized/contracted orbitals require
explicit matching radial functions; angular multiplicities and radial grids are
validated. Uncontracted functions are never substituted for a contracted basis.
Index projections work without species labels; physical unfolding and named
paths report missing metadata as `CapabilityError` (a `ValueError` subclass).

DOS Gaussian `eta` is its standard deviation in eV (default 0.05); Lorentzian
`eta` is the half-width. Physical spin texture is in units of ħ, with fully
polarized spin ±1/2. Plane-wave unfolding retains actual cutoff-dependent radial
weights without rescaling them to sum to one. EMPC transport is the isotropic
equipartition acoustic deformation-potential approximation. Effective masses
use a Cartesian finite-difference step in inverse Å and a carrier-weighted
pseudoinverse; unresolved crossings are excluded and carry validity masks.

Position moments remain accessible in `model.data`. Momentum and real/imaginary
density matrices retain native component order. Energies and lengths use SciPy
CODATA conversions. Raw records remain available when bounded integral noise is
reconciled with translation adjoints; correction sizes and precision used to
group near-degenerate states are recorded. Unsupported binary versions, bad
counts, truncation, nonpositive overlap and invalid residuals fail explicitly.

## Checkpoints and convergence

BZ sums use normalized weights, explicit spin degeneracy and fixed 128-point
scientific tiles, independent of worker/batch size. Arrays, definitions, units
and algorithm versions are committed atomically and verified on resume. Pockels
and CRTA transport average additive tensors/moments before nonlinear
reconstruction. EMPC depends on whole-mesh DOS, and FHS Hall couples adjacent
mesh vertices; each commits one durable complete-mesh case per refinement
level. Interruption can repeat the unfinished level. Loop, search and slab
tasks likewise commit complete deterministic cases.

Optional `parameters.integration` performs adaptive **nested uniform mesh
refinement**, preserving axes whose initial grid size is one when any other
axis is resolved. An initial `[1, 1, 1]` grid instead refines all three axes
(only the selected plane for FHS sheet Hall).
It is not local tetrahedral refinement. Each mesh and reduction is an independent checkpoint;
only a converged reduction becomes `result`:

```toml
[parameters.integration]
method = "adaptive"
absolute_tolerance = 0.000001
relative_tolerance = 0.0001
max_levels = 4
max_points = 1000000
```

Every response array must satisfy its own absolute-plus-relative mesh-change
criterion. Absolute tolerances use reported units. Exhausted limits fail while
retaining partials. Also vary broadening, loop spacing, radial cutoff and
derivative step where applicable; mesh differences do not prove convergence
near singularities. FHS topology stores discrete-flux diagnostics. Changes to
consumed SCFOUT, sidecar, POSCAR or PAO inputs reject resume. PAO directory inputs
are conservatively hashed in full.

## Scientific scope

Full nonorthogonal direct curvature includes basis curvature; projected
velocity-Kubo curvature is separately identified. Quantum metric is
`finite_basis_interband`: a positive occupied-to-unoccupied covariant Cartesian
sum within the retained basis. It omits external-basis contributions and is not
the full continuum second-moment metric. Orbital moments and optical responses
have the corresponding retained-basis scope. Finite-distance links use a
midpoint first-order position expansion; loop phases require mesh convergence
and are not exact finite-distance fidelities.

Mulliken weights are signed and obey total normalization; negative individual
weights are retained. Fat bands, spin texture, unfolding and slab-band
projections report the invariant mean over each complete same-channel
degenerate manifold, repeated for its member states. Their `manifold_sizes`
arrays distinguish these means from isolated-state projections. Requests that
cut a degenerate manifold, including a scissors occupation boundary, fail.
Complete equally occupied subspaces and explicit broadening are required.

Electronic polarization unwraps continuous Wilson phases and requires zero
occupied-manifold Chern numbers; it has lattice branches. Total polarization
additionally requires explicit ionic valences. Chirality requires a constant
occupied rank on the enclosing sphere. Berry-curvature dipoles reject
unresolved dispersive degeneracies; Drude and transport tensors use coherent
complete-subspace velocity products. Two-dimensional Hall responses use the
actual reciprocal-plane area and report sheet conductance. Susceptibility
retains its energy-only approximation.

Definitions, signs, units and independent references are in
[scientific conventions](scientific_conventions.md). The audit inventory is the
[pinned PyATB task definitions](https://github.com/susyexists/pyatb/blob/60fc6b2acafac36b9c8334208326937fef3366c5/src/pyatb/io/default_input.py).
