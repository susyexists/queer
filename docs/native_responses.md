# Native response and output processing

Queer's electronic response calculators use NumPy/SciPy and its own model
geometry. Native output readers operate on completed or partial calculation
files without importing ASE, pymatgen, Sumo, PyProcar, PyATB or Wannier90.
They retain missing physical information as missing.

## Oriented reciprocal planes and Hall conventions

`chern_number(model, k_vectors=[[1, 0, 0], [0, 1, 0]],
k_start=[0, 0, 0], grid=[24, 24, 1])` accepts two independent fractional
reciprocal spans. `plane` selects the two mesh axes assigned to those spans;
the remaining mesh dimension must be one. Swapping spans reverses orientation.
Integer spans describe closed reciprocal surfaces and may cover more than one
primitive surface. For noninteger spans the result is surface Berry flux divided
by `2*pi`; quantization is not asserted. Metadata records the spans and closure.

`engine="fhs"` uses occupied-overlap plaquettes following
[Fukui, Hatsugai and Suzuki](https://arxiv.org/abs/cond-mat/0503172).
`direct` and `kubo` integrate Cartesian curvature contracted with the oriented
spans and report both definitions. A constant sampled occupied rank is required;
a count boundary may not split a degenerate subspace. Sampled rank and branch
admissibility do not prove a globally gapped or adequately resolved surface.

The native convention is `A=i<u|du>`, `Omega_ab=-2 Im Q_ab`, and
`sigma_ab=-e^2/hbar integral(f*Omega_ab)`. `ahc` returns bulk S/cm for
`dimension=3` or sheet Siemens for `dimension=2`. A sheet tensor lies in the
selected reciprocal plane. Native calculations apply this physical convention
consistently; legacy service sign choices belong to the compatibility adapter.

## Shift-current manifolds and solvers

`shift_current` accepts `occupied_bands`, `band_pairs`, and
`scissors_occupied_bands`. A scalar occupied count selects the lowest states
of the complete model. A mapping such as `{"0": 2, "1": 2}` selects counts
independently in every collinear channel. Counts require zero temperature.
Pairs are distinct **zero-based** `[occupied, empty]` model band indices and
must stay in one channel. Partial selections of a degenerate transition block
are rejected. They select transitions while intermediate-state derivatives
retain the complete model basis.

`scissors` is an additive nonnegative conduction-energy shift in eV. Its
boundary defaults to `occupied_bands`, or to the chemical potential when no
count is supplied. `scissors_occupied_bands` overrides that boundary. Wavefunctions
and dipoles remain fixed; interband velocities use the shifted energy differences.

`method="analytic"` and its alias `sum_over_states` use generalized eigenvalue
perturbation identities. `method="sternheimer"` actually solves augmented
orbital-space systems for `(H-E*S)` with the entire degenerate nullspace
constrained. It checks residuals and uses the same normalization gauge. Both
methods then form analytic connection gradients and the same resonant
Sipe--Shkrebtii tensor in microA/V². Sternheimer here still diagonalizes the
finite model and uses its full retained basis; it is not a solver for external
plane-wave states or omitted high-energy bands.

## Smearing and local quadrature

Lorentzian `eta` is a half-width; Gaussian `eta` is a standard deviation, both
in eV. `smearing="none"` represents delta peaks by normalized bins centered
on the supplied photon energies. Uniform ascending energies determine the
bin width, or set positive `bin_width` explicitly. Bin edges are half-open.

`smearing="adaptive"` uses a Gaussian standard deviation from the RMS
transition-energy variation across the reciprocal cell, multiplied by
`adaptive_scale` and clipped to `adaptive_min`/`adaptive_max`. Explicit kpoints
require `adaptive_spacing`, a `(3,3)` array of Cartesian reciprocal cell steps
in Å⁻¹. Otherwise the base `grid` determines these steps. Local quadrature
refines the integral at that defined linewidth; its error estimate does not
measure linewidth or frequency-grid convergence.

`ahc`, `chern_number` and `shift_current` accept, for example:

```python
integration = {
    "method": "cubature",
    "absolute_tolerance": 1e-5,
    "relative_tolerance": 1e-4,
    "max_points": 100000,
    "max_cells": 20000,
}
```

`threshold` replaces selected coarse midpoint cells with a midpoint
`subdivision` grid; replacement weights retain unit total measure. The
threshold compares response magnitude in output units. For Hall and Chern,
`threshold_quantity="curvature"` instead defines the threshold in Å².
Shift-current thresholds use response units. Threshold subdivision reports
the change from each replaced cell, without claiming tolerance convergence.

`cubature` compares midpoint and tensor two-node Gauss rules, bisects the cell
with the largest normalized error, and requires every integrated component
to satisfy `absolute_tolerance + relative_tolerance*abs(value)`. Absolute
tolerances use S/cm, S, microA/V² or dimensionless Chern units as appropriate.
`max_points` counts all rule nodes, including nodes restored on resume;
local Chern also includes its single cached occupied-rank reference point.
`max_cells` limits final active cells. Inactive axes require one grid point.
Exhausted budgets raise and preserve completed checkpoints. Workflow cell
artifacts are immutable and verified by checksum before reuse.

`adaptive` performs nested uniform mesh refinement, limited by `max_levels`
and `max_points`. FHS needs linked uniform meshes and supports this route;
it rejects local threshold/cubature. Every error estimate is a numerical
comparison, not a bound proving continuum convergence. Sharp unresolved
features can defeat embedded rules and coarse FHS meshes can alias topology.

## Imported bands and calculation results

`queer.io.bands.BandData` stores absolute eV energies with axes
`(spin, kpoint, band)`, optional occupations, source weights, fractional
kpoints, reciprocal vectors, Fermi energy, labels and distances. Segment
bounds are half-open and partition the original rows; repeated endpoints
and disconnected paths are preserved. `.relative_energies()` requires an
explicit Fermi reference; `.as_result()` supplies array dimensions and units.

Readers include `read_eigenval`, `read_vasp_outcar`, `read_vasprun`,
`read_vasp_kpoints`, `read_openmx_bands`, `read_wannier_bands`, and
`read_wannier_report`. OUTCAR/XML use their declared scalar, collinear or
spinor occupation capacity. A one-channel EIGENVAL does not encode spinor
status: supply `occupation_capacity=1` for SOC/noncollinear states; the
default scalar capacity is two. Incomplete final OUTCAR spin cycles fail.
Cartesian KPOINTS retain their stated scale and require caller lattice
conversion. Output bands do not create orbital projectors or a Hamiltonian.

The OpenMX reader converts Hartree energies and inverse-Bohr reciprocal
vectors and supports repeated per-spin headers as well as grouped energies.
Its fractional-coordinate contract follows the
[official OpenMX format description](https://www.openmx-square.org/forum/patio.cgi?mode=view&no=2937).
Wannier blank-separated paths and labelinfo discontinuities follow the
[Wannier90 output specification](https://wannier90.readthedocs.io/en/latest/user_guide/wannier90/files/).

`queer.io.calculations` exposes `read_vasp_results`, `read_vasprun_results`
and `read_openmx_results`. Each returns `CalculationResult(arrays, summary,
bands=None)` with final arrays, energy histories, JSON summaries and optional
bands. `energy_eV` prefers VASP's sigma-to-zero energy;
`free_energy_eV` retains TOTEN separately. Forces use eV/Å. Stress uses tensile
Cartesian eV/Å³: VASP's compressive kbar tensor is negated and converted,
consistent with its [reported pressure convention](https://vasp.at/forum/viewtopic.php?p=23484).
OpenMX cell gradients are transformed as `sym(A.T @ dE/dA)/abs(det(A))`,
after converting gradients from Hartree/Bohr. Missing gradient records do
not produce a fabricated stress tensor.

SCF outcomes use explicit final markers where available and retain marker
provenance. A new OpenMX SCF cycle resets an earlier success to unknown.
Missing evidence is `None`. XML iteration-count and EDIFFG convergence checks
are explicitly marked as inferred; force checks assume unconstrained atoms.
Positive EDIFFG uses reported free-energy changes when available.

`band_gap` reports sampled VBM/CBM, direct gap, metal classification and edge
indices/coordinates from occupations or an explicit Fermi level. Unsampled
extrema remain possible. `compare_bands` aligns explicit Fermi references,
validates matching path segments, interpolates within each segment and uses
unique energy matching per kpoint to report RMSE/MAE/max error and coverage.
It compares energies, not wavefunction character or orbital correspondence.

## Independent checks

[The native numerical tests](../tests/numerical/test_native_postprocessing.py)
include a QWZ Hamiltonian `sin(qx)*sx + sin(qy)*sy +
(-1+cos(qx)+cos(qy))*sz`: a 12×12 FHS plane gives +1 and swapped spans give
−1 within `1e-12`. A nonzero two-band dimer shift response agrees between
analytic and constrained Sternheimer methods at `rtol=2e-12, atol=2e-9`.
Integrated bin and Gaussian weights agree at `rtol=2e-5, atol=1e-7`.

Independent polynomial quadrature gives `(1/3, 1/2)` within `1e-13`, with
embedded error below `2e-5`; threshold midpoint replacement agrees within
`4e-4`. Synthetic VASP/OpenMX/Wannier records check spin and energy axes,
SOC capacities, unit/sign conversions, disconnected paths and incomplete
files. A real workflow interruption proves resumed Hall integration skips
completed cells and preserves their checksums, modification times and
manifest records. These checks establish the tested contracts and small
reference problems; they do not certify convergence for a new material.
