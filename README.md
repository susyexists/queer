# QUEER

QUEER is a Python library for Wannier-Hamiltonian workflows in condensed
matter calculations, including band energies, susceptibility, self-energy, and
momentum-microscopy energy-surface runs.

## Repository Layout

- `src/queer/`: importable library code.
- `notebooks/active/`: current working notebooks. The active Ag primitive
  notebook is `notebooks/active/Ag_primitive_single.ipynb`.
- `notebooks/archive/`: preserved development notebooks by material/topic.
- `data/`: local calculation inputs such as `POSCAR`, `OUTCAR`, `ef.txt`, and
  `wannier90_hr.dat`. These files are ignored by Git.
- `results/`: generated figures, animations, and logs. These files are ignored
  by Git.
- `scripts/runs/`: runnable Python entrypoints.
- `scripts/hpc/`: Slurm launchers for HPC runs.
- `docs/artifact_manifest.tsv`: old-path to new-path inventory for moved
  notebooks, inputs, scripts, and results.
- `tests/`: smoke tests for imports, paths, mesh conversions, ARPES helpers,
  and small model calculations.

## Install

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

For local notebook work, launch Jupyter from the repository root after the
editable install so `queer` resolves consistently.

The project uses a `src/` layout, so the local development environment should
be installed editably with:

```bash
.venv/bin/python -m pip install -e .
.venv/bin/python -m ipykernel install --user --name queer-dev --display-name "Python (queer editable)"
```

Select `Python (queer editable)` as the notebook kernel.

## Data And Results

By default, QUEER looks for data under `./data` and writes results under
`./results`. On HPC systems or scratch filesystems, override those roots:

```bash
export QUEER_DATA_DIR=/path/to/data
export QUEER_RESULTS_DIR=/path/to/results
```

The helper functions `queer.paths.data_path(...)` and
`queer.paths.results_path(...)` should be used by notebooks and scripts instead
of root-relative strings.

## Momentum-Microscopy Runs

Run a single V0 sweep locally:

```bash
python scripts/runs/v0_sweep/run_v0.py --V0 12 --material ag_primitive --ef 8.310342
```

Submit the multi-V0 Slurm workflow:

```bash
sbatch scripts/hpc/run_v0_sweep.sbatch
```

The Slurm script honors `QUEER_DATA_DIR`, `QUEER_RESULTS_DIR`, and
`QUEER_VENV`.

## Quick Smoke Check

```bash
PYTHONDONTWRITEBYTECODE=1 python -m compileall -q queer examples scripts
python -m pytest
```
