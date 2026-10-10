# Repository responsibilities

- `src/queer/models`: native OpenMX, Wannier and EPW representations and model evaluation.
- `geometry`: lattice/coordinate transformations, rotations, meshes, named paths.
- `io`: validated readers and atomic arrays/manifests.
- `observables`: shared nonorthogonal operators, electronic/surface/unfolding,
  geometric/integrated/optical kernels, spectral, ARPES and momentum calculations.
- `workflows`: calculation cases and recoverable orchestration; `plotting`: optional visualization.
- `config.py`, `execution.py`, `cli.py`: configuration, bounded resources, installed commands.
- Legacy top-level modules and `scripts/runs` forward supported 1.x interfaces.
- `configs/examples`: complete small workloads; `configs/hpc`: customizable site resources.
- `tests/{fixtures,unit,numerical,integration,packaging}`: independent acceptance gates.
- `benchmarks`: reproducible time/memory evidence and optional real-data runs.
- `notebooks/active`: maintained consumers; notebook/script archives: unsupported provenance.
- `data` and `results`: ignored inputs and generated artifacts, with trackable manifests.

Numerical code must not select storage paths, save figures, or discover personal
inputs. Workflow boundaries resolve configuration and resources; numerical methods
receive arrays and explicit options. Plotting consumes already-computed results.
Package resources are loaded with `importlib.resources`, not checkout-relative paths.

The historical `artifact_manifest.tsv` remains unchanged. Archive dispositions
are in `archive_dispositions.tsv`. Historical notebook outputs remain intact;
unsupported figure scripts have been moved without altering their bytes.
