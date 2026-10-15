# Contributing scientific changes

Start with the [scientific contracts](scientific_conventions.md) and
[equation registry](equations.toml). Reuse an existing owner when the mathematical
definition and conventions match. A specialized precision or compiled implementation
needs a documented reason and independent parity/convergence tests.

## Ownership and interfaces

- Put reusable array arithmetic in `physics`; pass evaluated operators, energies,
  occupations and explicit conventions. Keep model discovery, paths and executors
  outside these kernels.
- Models own representation, interpolation and eigensystem evaluation. Observables
  select bands/manifolds and construct `ObservableResult` arrays, dimensions, units
  and diagnostics. `observables._tb` remains a compatibility facade over native
  operators/evaluation helpers.
- Keep field-specific crossings, propagation and harmonic algorithms in
  `finite_field`. Reuse native geometry arithmetic; preserve precision and accelerated
  backends with parity tests and explicit capability checks.
- Atomistic algorithms share force-provider validation, kinetic energy and ordered
  expansion. Keep format syntax in `io`, native records in `io.records`, numerical
  identities in `io.numerical`, and historical unit/sign conversions in `compat`.
- Register workflows once in the catalog/specifications and reduction policies.
  Domain handlers consume resolved sources and numerical results; the runner owns
  locks, attempts, checkpoint identity and completion. Plotting reads saved results.

Preserve public signatures and reexports throughout 1.x, including historical dotted
imports, callable `queer.model`/`queer.epw`, axes and sentinel handling. Moving a class
or function requires checking every package export surface and installed distributions.
Core imports must work without visualization, symmetry, precision or producer extras.

`models.contracts` defines the shared eigenstate schema: absolute energies `(Nb,Nk)`,
coefficients `(Nk,Nbasis,Nb)`, overlaps `(Nk,Nbasis,Nbasis)` and band channels `(Nb,)`.
Wannier fields remain assignable; OpenMX retains its historical frozen fields.
Numerical arrays in either variant retain their existing mutation behavior.

## Equations, provenance and compatibility

Add or amend registry entries with a stable ID, expression, input/output units,
assumptions, conventions, canonical `module:symbol`, specialized backend symbols,
source references and independent pytest node IDs. Registry checks resolve source
and test definitions statically; they perform no imports or downloads.

Do not merge thermal-energy and Kelvin defaults, direct and projected Kubo curvature,
pair and all-band metrics, paper and physical profiles, or EPW and retarded regulators.
Treat degenerate states through complete subspaces. Record numerical corrections
separately from structural refactors and document their changed values.

Update `ALGORITHM_VERSION` when checkpoint compatibility changes and include new
numerical owners in preparation fingerprints. Completed old artifacts remain
renderable. Preserve the finite-field `SOURCE_MANIFEST.json` donor hashes; current
QUEER source identity belongs in engine fingerprints. Historical archives are
checkout provenance and are verified against their disposition ledger.

## Validation

Run the affected independent numerical, recovery and installed-package tests, then
the required repository Ruff, formatting, mypy and pytest gates. Add tests for a
meaningful analytical reference or public boundary; avoid tests that merely reproduce
the implementation. Optional backends need a job that actually installs their extras
and checks parity, alongside core-only distribution acceptance.

Preserve bounded working batches, cache shares, executor reuse and streaming call
counts. Compare benchmark wall time and peak RSS on the same machine with identical
inputs, runtime and execution settings; report baseline ratios and run-to-run noise.
See [benchmark protocols](../benchmarks/README.md). Wall-time thresholds do not
transfer between machines; resource and scientific contracts remain hard gates.
