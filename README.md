# QUEER

QUEER is a Python library for Wannier Hamiltonians, bands, static susceptibility,
spectral/ARPES calculations, momentum microscopy, and EPW self-energy workflows.
It supports local, notebook, and Slurm execution with durable numerical results.

## Install

Python 3.11+ is required. For development and HPC environment preparation:

```bash
uv sync --locked --all-extras --group dev --python 3.13
```

For ordinary package installation, use `python -m pip install .` for numerical
work or `python -m pip install '.[plot,symmetry]'` for figures and named k-paths.
Optional extras are `plot`, `symmetry`, `animation`, `notebook`, and `all`.
`requirements.txt` is a compatibility installer; `pyproject.toml` is authoritative.

## Reproducible workflows

The committed example configurations use tiny analytical fixtures, not personal
material datasets:

```bash
uv run --locked queer bands --config configs/examples/bands.toml --render
uv run --locked queer susceptibility --config configs/examples/susceptibility.toml
uv run --locked queer spectral --config configs/examples/spectral.toml
uv run --locked queer spectral --config configs/examples/spectral_arpes.toml
uv run --locked queer momentum --config configs/examples/momentum.toml
uv run --locked queer epw --config configs/examples/epw.toml
```

Each command prints its run directory. A run contains resolved configuration,
input checksums, version/resource/scheduler metadata, a status manifest, and
incremental NPZ arrays. Figures can be regenerated without calculation:

```bash
queer render /path/to/completed/run
queer momentum --config calculation.toml --run-dir /path/to/interrupted/run --resume
```

Configurations resolve relative input/output paths against the TOML directory.
CLI path overrides resolve against the working directory. Explicit configuration
roots override `QUEER_DATA_DIR` and `QUEER_RESULTS_DIR`; defaults are working-directory
`data/` and `results/`. Runtime storage never depends on the installation location.
Use `inputs.material` to select `data/materials/<name>` through `QUEER_DATA_DIR`.

## Library and compatibility

```python
from queer import ExecutionOptions, WannierModel

model = WannierModel.from_directory("my-material", ef=0.0, interpolation="standard")
bands = model.energies([[0.0, 0.0, 0.0]], execution=ExecutionOptions(batch_size=256))
```

K-points are `(Nk, 3)` fractional reciprocal coordinates; bands are `(Nb, Nk)` in eV.
Cartesian momenta use inverse angstroms. The existing `queer.model(...)` and legacy
module functions remain available during the 1.x migration. Scientific corrections
change historical values; see [migration notes](docs/migration.md).

## Notebooks and Slurm

The maintained notebook is `notebooks/active/Ag_primitive_single.ipynb`. Install the
notebook extra and select the prepared environment's kernel. Set
`QUEER_NOTEBOOK_CONFIG` to a TOML configuration for any supported workflow. The
notebook delegates calculation, recovery, and rendering to the workflow APIs; it
keeps scientific parameters in configuration and saves the same run artifacts.
Archives retain historical notebooks and outputs and are excluded from execution.

Prepare the environment once, customize `configs/hpc/default.toml`, and submit:

```bash
.venv/bin/python scripts/hpc/submit_sweep.py --config configs/examples/momentum.toml \
  --profile configs/hpc/default.toml --dry-run
```

Remove `--dry-run` to submit. See [HPC operations](docs/hpc_usage.md) for site resources,
array failures, recovery, and real-cluster acceptance.

## Verification and layout

```bash
uv run --locked pytest
uv run --locked ruff check src tests scripts/runs scripts/hpc examples benchmarks
uv run --locked mypy
```

The tests include synthetic numerical references, workflow recovery, notebook
execution, scheduler adapters, and isolated wheel/sdist installation. Optional personal
material smoke checks may skip; numerical acceptance uses committed fixtures.

See [repository boundaries](docs/repo_layout.md), [scientific conventions](docs/scientific_conventions.md),
and [reference benchmarks](benchmarks/README.md). Large inputs and generated results
remain ignored; tracked dataset/run manifests provide provenance.
