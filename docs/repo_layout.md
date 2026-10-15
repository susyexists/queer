# Repository responsibilities

- `src/queer/models`: native OpenMX, Wannier and EPW representations and model evaluation.
- `physics`: shared array equations, units, occupations, line shapes, geometry and response contractions.
- `geometry`: lattice/coordinate transformations, rotations, meshes, named paths.
- `io`: validated format adapters, neutral POSCAR records, numerical identities and atomic arrays/manifests.
- `observables`: evaluate models, apply scientific selections and assemble named observable arrays.
- `atomistic`: force-provider validation, dynamics, phonons, selection and dataset services.
- `finite_field`: field-specific model adapters, crossings, propagation, quadrature and harmonic methods.
- `workflows`: shared catalog/reduction policies, domain handlers and recoverable orchestration.
- `plotting`: optional visualization from computed results.
- `config.py`, `execution.py`, `cli.py`: configuration, bounded resources, installed commands.
- Legacy top-level modules and `scripts/runs` forward supported 1.x interfaces.
- `configs/examples`: complete small workloads; `configs/hpc`: customizable site resources.
- `tests/{fixtures,unit,numerical,integration,packaging}`: independent acceptance gates.
- `tests/oracles`: verbatim Generator sources that ported functions are tested against; not tests.
- `benchmarks`: reproducible time/memory evidence and optional real-data runs.
- `notebooks/active`: maintained consumers; notebook/script archives: unsupported provenance.
- `data` and `results`: ignored inputs and generated artifacts, with trackable manifests.

Numerical code must not select storage paths, save figures, or discover personal
inputs. Workflow boundaries resolve configuration and resources; numerical methods
receive arrays and explicit options. Plotting consumes already-computed results.
Package resources are loaded with `importlib.resources`, not checkout-relative paths.

Shared equations flow from `physics` into models/observables; compatibility modules
forward outward to native owners. Specialized finite-field precision and compiled
backends retain explicit conventions and parity tests. Force producers consume
native structures; shared ordered expansion lives in `geometry`, independently of
the dynamics integrators. Readers own external syntax and conversions; `io.records`
owns durable structure/frame arrays and `io.numerical` owns numerical metadata/hash
adapters. Public historical modules retain imports during the 1.x migration.

The [equation registry](equations.toml) connects canonical implementation owners,
units, assumptions and independent tests. See the [contributor guide](development.md)
before adding another implementation or moving a public export.

The historical `artifact_manifest.tsv` remains unchanged. Archive dispositions
are in `archive_dispositions.tsv`. Historical notebook outputs remain intact;
unsupported figure scripts have been moved without altering their bytes.
