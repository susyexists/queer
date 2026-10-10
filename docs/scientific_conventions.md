# Scientific contracts

Compute with float64/complex128. Direct/reciprocal lattice vectors are rows:
`B = 2*pi*inv(A).T`, `A @ B.T = 2*pi*I`, and `k_cart = k_frac @ B`.
Direct coordinates are angstroms; Cartesian momenta are inverse angstroms.
K-points have shape `(Nk,3)`, band energies `(Nb,Nk)`, and spectra `(Nomega,Nk)`.
Energy parameters and bands/spectral results use eV unless explicitly labeled.
EPW phonons and couplings use meV; its supported reduced-coupling response preserves
the legacy normalization, which is recorded explicitly rather than labeled as a
universal electron self-energy in eV.

Conventional HR interpolation is
`H(k) = sum_R H(R)*exp(2*pi*i*k_frac.R)/degeneracy(R)`.
Pair-dependent Wigner–Seitz mappings and expanded exports are separate explicit modes.
Reference definitions: [Wannier90 file formats](https://wannier90.readthedocs.io/en/latest/user_guide/wannier90/files/)
and [interpolation source](https://raw.githubusercontent.com/wannier-developers/wannier90/v3.1.0/src/plot.F90).

Check Hermiticity using `||H-H†||inf <= 1e-10 eV + 1e-10*||H||inf`.
Only within-tolerance noise may be symmetrized; otherwise evaluation fails with
k-point context. Synthetic references use `atol=rtol=1e-12`; serial/parallel energies
use `1e-10`. Degenerate eigenvectors should be compared through subspace projectors.

The model Fermi energy defines the zero of returned bands. Photoemission's kinetic
offset and workflow energy shifts are separate explicit parameters. Thermal energy
is `k_B*T` in eV; zero temperature uses the occupation limit. Broadening and Gaussian
widths must be positive. Binding-energy sweep intervals are half-open.

ARPES points outside the real hemisphere carry NaNs and valid-point masks; computations
must not replace inaccessible points by `kz=0`. Named paths use SeekPath's original-cell
basis. Static susceptibility and Fermi-surface nesting must retain separate names and
units; the historical two-row response adapter is documented in its numerical API.

`arpes_mesh` and `arpes_path` accept `return_mask=True` to return points together
with a Boolean mask aligned to their original rows. Spectral artifacts save a
`valid_mask` aligned to their `(Nomega,Nk)` array. The mesh kinetic energy is
`photon_energy - binding_energy - fermi_energy`; this historical `fermi_energy`
argument is a kinetic offset, separate from the model's Fermi energy. Curved
spectral workflows use `photon_energy + omega` with a zero additional kinetic
offset. Adjust the effective photon-energy parameter explicitly when using a
different work-function/reference convention.

Run manifests retain the environment, source version, scheduler allocation,
resolved execution budget, status, and elapsed time for every resume attempt.
Each checkpoint identifies its attempt and elapsed time since the previous
checkpoint, including calculation and persistence. Release changes to numerical
algorithms must update `ALGORITHM_VERSION` to prevent incompatible resumes.

Fixtures are analytical models for tests. The Ag-labeled fcc fixture exercises
crystallographic labels and notebook integration; it does not model actual silver.
Large real-material comparisons require matching interpolation/export provenance
and tolerances accounting for serialized HR precision.
