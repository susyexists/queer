# QUEER 1.1 migration

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
file/branch ordering, and self-energy execution are corrected. Temperature arguments
remain thermal energy in eV, not kelvin. Susceptibility and nesting are distinct outputs.

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
