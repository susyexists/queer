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

## Native OpenMX model

`OpenMXModel.from_scfout` reads the OpenMX version-3 SCFOUT interface directly.
Hamiltonian, overlap, position moments, momentum/density records, embedded input,
atom positions, and available orbital labels remain associated with the model.
Public Hamiltonians are eV and position/lattice matrices are angstroms; raw
momentum records retain their named atomic-unit derivative representation.
No PyATB installation or compiled PyATB library is required.

The interpolated generalized problem is `H(k) C = S(k) C E`, with
`C.conj().T @ S @ C = I`. Real-space operators stay sparse; dense solves operate
on bounded k-point batches. Nonmagnetic spatial Hamiltonians carry spin
degeneracy two. Collinear models retain separate up/down channels; SOC models
retain a full spinor Hamiltonian. Explicit spin channels have multiplicity one,
and their orbital ordering is all up followed by all down. Band indices are
zero-based. `eigenstates` and `solver` return absolute eigenvalues;
`energies` and band/spectral plots subtract `model.fermi_energy`.

Native SCFOUT export can contain small inconsistencies between reverse
translation blocks or between position/overlap derivative identities. Import
reconciles only bounded source noise and records the original residuals in
`source_operator_identity_residuals`. A model's default degeneracy tolerance is
`max(1e-9, 10 * hamiltonian_source_residual_eV)`; response APIs can override it
with `degeneracy_tol`. This tolerance describes unresolved energy differences
in the source, independently of an optical linewidth. A positive-definite
overlap and valid Hermitian/covariant operators are required.

All integration weights are nonnegative and normalized to sum to one.
For bulk responses the normalized BZ average is divided by the cell volume;
specified two-dimensional Hall responses use the reciprocal-plane area and
report sheet conductance. Cartesian axes always refer to physical x/y/z, even
for skew direct or reciprocal cells. Named reciprocal-plane indices are
fractional axes and are converted through the reciprocal lattice.

## Electronic, surface, unfolding, and transport observables

Each native observable returns `ObservableResult`, containing numeric arrays,
axis names in `dims`, physical `units`, and JSON metadata. Scan axes precede
Cartesian response axes. Electronic band arrays preserve `(band,k)`.

| Calculator | Definition and interpretation |
| --- | --- |
| `band_structure` | Interpolated bands in eV relative to the model Fermi energy. Its optional scissors shift uses an explicit `n_occ` threshold within each channel. |
| `fat_band`, `pdos` | Mulliken orbital/group projection `Re(C_i* (S C)_i)`. Weights sum to one for a normalized eigenstate when groups partition the basis; individual orbital weights can be signed. PDOS is states/eV/cell. |
| `dos` | Normalized Gaussian or Lorentzian spectral density, states/eV/cell, including the model's spin multiplicity. Integrating over a sufficiently broad energy interval recovers the basis capacity. |
| `jdos` | Positive-energy same-spin-channel transitions weighted by occupation differences, transitions/eV/cell. Photon energies describe band differences. |
| `fermi_energy` | Absolute chemical potential for a specified electron count per cell. At zero temperature a partially filled sampled level can require fractional occupation; finite-temperature counting uses Fermi functions. |
| `fermi_surface` | Sampled band-energy mask and Gaussian proximity weight around an absolute chemical potential. The result is a sampled surface representation. |
| `find_nodes` | Local optimization of candidate adjacent-band gaps seeded by a reciprocal grid. A finer initial grid is necessary to establish completeness. |
| `spin_texture` | Physical spin expectation in units of hbar, including the factor one half multiplying Pauli matrices. |
| `reduce_basis` | Maximum absolute Mulliken participation in selected states and a proposed keep/remove mask. This diagnostic does not construct a contracted Hamiltonian. |
| `surface_state` | Nonorthogonal principal-layer Green-function decimation, finite-slab generalized diagonalization, or direct finite-slab Green inversion. Spectra include neighboring overlap contributions to layer Mulliken density, states/eV/principal layer. Principal layers span the longest perpendicular coupling. |
| `bandunfolding` | Primitive-momentum projection onto plane waves using the actual PAO radial functions and real spherical harmonics. `ecut` is an eV kinetic cutoff; weights require convergence in this cutoff. |
| `bandunfolding_spin_texture` | The same plane-wave projection with spin-weighted amplitudes in hbar. Dividing by a nonzero unfolding weight gives the projected normalized spin. |
| `boltz_transport` | Semiclassical velocity moments with the Fermi derivative: conductivity S/m, Seebeck V/K, electronic thermal conductivity W/(m K), mobility m²/(V s), and carrier densities m⁻³. CRTA takes a lifetime in fs. Optional effective masses use finite differences of Cartesian band velocities and require derivative-step convergence. |

Unfolding requires PAO radial data matching the original OpenMX species and
orbital labels. Species/centers alone do not determine an orbital Fourier
transform. Queer reads the documented logarithmic radial PAO grids and converts
their units; missing radial functions or labels produce an error.
The native radial reader follows the
[OpenMX PAO format documentation](https://www.openmx-square.org/adpack_man2.2/node22.html).
OpenMX distribution data and independent user inputs retain their own licenses;
the library's analytical test orbitals do not redistribute those datasets.

Transport's optional `EMPC` lifetime is the isotropic equipartition acoustic
deformation-potential approximation
`tau = hbar*C/(2*pi*k_B*T*D²*DOS_per_volume)`, with Young modulus in GPa and
deformation potential in eV. It assumes that scattering model explicitly;
it does not determine a general electron-phonon lifetime from SCFOUT.
Electron/hole densities are partitioned relative to the model's reference
Fermi energy. Ionic thermal transport, magnetic-field transport, and
interaction corrections are outside these independent-electron definitions.

## Nonorthogonal geometry and integrated responses

The physical band connection is
`A_a = i C† S (partial_a C) + (C† r_a C)†`, with units angstroms.
The first term alone omits position matrix elements. Cartesian derivatives
come from analytic Fourier translation moments. The position convention obeys
`r_a-r_a† = -i partial_a S`, which makes the complete connection Hermitian.
Physical `hbar*v_a` is eV angstroms, including overlap and position corrections.
These distinctions follow the published
[nonorthogonal optical matrix-element derivation](https://arxiv.org/abs/1805.08918)
and [full NAO Berry-curvature formula](https://arxiv.org/abs/2105.14662).

The default `engine="direct"` Berry tensor is
`Omega_ab = partial_a A_b - partial_b A_a`. The retained-basis Kubo expression
`engine="kubo"` uses interband connection products. Their difference is the
diagonal of the projected-coordinate curvature
`F_ab = partial_a A_b-partial_b A_a-i[A_a,A_b]`.
This term can matter in a reduced or incomplete localized basis; agreement
between the engines is a convergence diagnostic rather than an identity
imposed by changing a Hall sign. Degenerate eigenvectors are handled through
complete subspaces; an occupied boundary may not split an unresolved same-spin
degenerate block. Individual-band Berry curvature at such a block is undefined.

| Calculator | Definition, units, and limits |
| --- | --- |
| `berry_curvature` | Cartesian antisymmetric tensor in Å², with individual-band or occupied-manifold axes. |
| `quantum_metric` | Explicit `definition="finite_basis_interband"`: `g_ab=Re sum_(n occupied,m unoccupied) A_a,nm A_b,mn`, in Å². This is positive semidefinite and invariant under occupied/unoccupied block rotations. |
| `chern_number` | Geometric `C_geo`: occupied-manifold Berry flux divided by `2*pi`; no extra spin multiplicity is applied. Supports direct/Kubo integration and FHS plaquettes. HHH's Hall number is `C_H=-C_geo`, obtained with `queer.finite_field.geometry.topology.hall_chern(C_geo)`. |
| `chirality` | Occupied Berry flux through an outward Cartesian sphere divided by `2*pi`. Radius is inverse angstroms, and angular quadrature must converge. |
| `wilson_loop` | Unitarized products of occupied overlaps, with reciprocal-boundary sewing. Hybrid-center phases are modulo one; the determinant Berry phase is radians. |
| `polarization` | Electronic Berry-phase polarization in C/m², modulo the returned lattice polarization quanta. Optional ionic charges are pseudopotential valences in units of e. |
| `ahc` | `sigma_ab = -e²/hbar * integral_BZ f*Omega_ab`, bulk S/cm or sheet S, including spin multiplicity. |
| `anc` | Berry-curvature integral weighted by Fermi entropy, A/(K cm) or sheet A/K. It tends to the low-temperature Mott relation `alpha = -(pi²*k_B²*T/3e) d sigma/d mu`. |
| `berry_curvature_dipole` | Dimensionless bulk Fermi-surface `D_ab=integral(-df/dE)*(partial_a E)*Omega_b`, with energy-gradient and curvature-pseudovector axes. |
| `orbital_magnetization` | Retained-interband wave-packet orbital moment plus the corresponding Berry-curvature Fermi-sea term, A/m. Separate local/itinerant arrays are returned. |
| `shc` | Conventional spin-current Kubo response, `(hbar/2e) S/cm`, axes chemical potential, spin polarization, current, field. The spin current is half the anticommutator of physical velocity and spin; no spin-torque correction is added. |
| `drude_weight` | Positive intraband tensor defined by `sigma(E)=i*D/(E+i*eta)`, eV S/cm or sheet eV S; this definition has no extra factor pi. |

The finite-basis metric excludes orbital spread into discarded states. In
particular it vanishes when the entire retained basis is occupied, while a
physical continuum projector can retain internal atomic-orbital spread.
First position moments cannot supply that missing information. Raw SCFOUT
higher powers along x/y/z are retained, but a complete physical metric also
requires mixed second moments. Queer therefore rejects an unsupported full
metric definition instead of identifying normalized overlap-link singular
values with an exact continuum metric. Orbital magnetization likewise carries
an explicit retained-interband definition.

OpenMX basis overlaps use a midpoint expansion through the first position
moment. They are suitable for converged loop phases and FHS curvature when
the k spacing tends to zero; they do not give exact fidelity at a finite
separation. Refine Wilson/FHS meshes and avoid singular occupied links.
The plaquette construction follows
[Fukui, Hatsugai, and Suzuki](https://arxiv.org/abs/cond-mat/0503172).
Polarization additionally needs a globally gapped manifold and a physically
consistent choice of branch across reciprocal strings.

## Optical and second-order response conventions

`omega` is nonnegative photon energy in eV. Time dependence is
`exp(-i*omega*t)`, with retarded `z=omega+i*eta`; `eta` defaults to 0.05 eV.
Chemical potentials are absolute eV, and temperature is Kelvin (default zero).
At zero temperature a normalized Gaussian of width `eta` represents Fermi
surface derivatives; finite-temperature derivatives use the Fermi function.
A rigid optional `scissors` shifts states above the chemical potential upward
by that many eV, keeping dipoles fixed and updating off-diagonal velocities
to the shifted gaps. It is an additive shift intended for a gapped conduction
manifold. A metallic chemical-potential crossing does not define a smooth
rigid conduction projector for this derivative prescription.

Linear and nonlinear responses use the physical connection in the length-gauge
density-matrix recursion. Within degenerate blocks, energy gradients and
Hessians are matrices; internal block connections/velocities are retained in
shift and injection contractions. The current includes the projected-coordinate
field term from `[R_a,R_b]=i F_ab`. This preserves the finite-basis Heisenberg
identity even when projected position operators do not commute. The recursion
is based on [Ventura et al.](https://arxiv.org/abs/1703.07796); the projected
field-current term also follows directly from `j=-e*(i/hbar)[H+e E·R,R]`.

| Calculator / array | Field convention and units |
| --- | --- |
| `optical_conductivity` / `conductivity` | Complex Cartesian `j_a=sigma_ab E_b`, S/cm, including interband and Drude terms. |
| `dielectric_function` / `dielectric` | Dimensionless `epsilon=I+i*sigma_SI/(epsilon0*z*e/hbar)` using the same analytic retarded frequency, including at zero real photon energy. Also returned by optical conductivity. |
| `shg` | Complex `chi2(-2w;w,w)` in pm/V, `P_a(2w)=epsilon0*chi2_abc E_b(w)E_c(w)` with no factorial. Incoming linewidths are eta and output linewidth is `2*eta`. |
| `pockels` | Electronic electro-optic impermeability change `delta(epsilon^-1)_ab=r_abc E_dc,c`, pm/V. It computes `r_abc=-2*(epsilon^-1)_ai*chi2_ijc(-w;w,0)*(epsilon^-1)_jb` with full tensors and also returns `chi2` and `dielectric`. |
| `second_order_static` | Real adiabatic `chi2(0;0,0)` in pm/V; default regulator eta is `1e-4` eV. Requires constant occupied rank, chemical potential farther than `10*eta` from sampled bands, and occupations within `1e-10` of zero/one. Reduce eta and refine the grid to verify convergence. |
| `shift_current` | Real symmetric-input dc conductivity `sigma_abc(0;w,-w)`, microA/V², using the generalized dipole derivative and normalized Lorentzian/Gaussian resonance. Electron charge is `-e` and the shift-vector convention is `R_nm=-partial arg(r_nm)+A_nn-A_mm`. |
| `cpge` | Purely imaginary antisymmetric-input circular injection tensor, A/(V² s), `d j_a/dt=cpge_abc E_b conj(E_c)`. `helicity_rate` is the equivalent real tensor multiplying `i E cross conj(E)`. A positive `relaxation_time_seconds` additionally produces `steady_cpge` in microA/V². |

Shift and circular injection expressions follow
[Sipe and Shkrebtii](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.61.5337)
and [de Juan et al.](https://arxiv.org/abs/1611.05887), respectively.
These are independent-electron electronic responses: excitons, local-field
screening, ionic/phonon electro-optic contributions, and general scattering
vertex corrections are not inferred from the exported tight-binding matrices.
CPGE without a supplied lifetime is a rate rather than a steady conductivity.

## Numerical acceptance and material convergence

The optical/nonlinear suite checks nonzero two-level retarded line shapes,
dielectric passivity, spin and occupation factors, and shifted resonances against
closed forms at approximately `1e-11` relative tolerance. SHG and Pockels match
closed two-level polarization formulas. Three-level static tensors match
Rayleigh coefficients and independent finite-field diagonalization within
`2e-6` relative error, including noncommuting Cartesian projected positions.
Rice–Mele shift current agrees with independently differentiated aligned
wavefunctions within `2e-6`, and its inversion-symmetric limit vanishes.
A spherical Weyl model has injection trace `-pi*e³/h²` for `H=k·sigma` under
the stated electron-charge/helicity convention, within `2e-7` relative error.
Block rotations, nonorthogonal basis changes, origin/Cartesian transformations,
unequal-weight workflow tiles, process workers, and bounded-memory batches
provide separate covariance and execution checks.

Topology/Hall tests include analytic Chern phases, outward Weyl flux, Wilson
winding, polarization quanta, sheet/bulk conversion, finite-basis curvature
corrections, entropy/Mott limits, Drude normalization, and conserved-spin Hall
identities. Electronic and surface tests compare sum rules and independently
solved chains/slabs; radial unfolding uses analytical orbitals. These synthetic
references establish the algorithmic contracts without depending on upstream
output from the same formulas.

For real materials, record the input checksum, OpenMX basis and SCF settings,
spin mode, chemical potential, source operator residuals, mesh, broadening,
temperature, and any scissors/lifetime choices. Refine the k grid independently
of linewidth and temperature, especially at metallic Fermi surfaces and sharp
resonances. Finite output or qualitative plot agreement is insufficient to
establish convergence. A material regression is meaningful only at a stated
quantitative tolerance with matching scientific definitions and numerical
settings; passing an analytical test does not establish that a coarse material
mesh has converged.
