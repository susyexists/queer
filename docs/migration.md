# QUEER 1.2 migration

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
