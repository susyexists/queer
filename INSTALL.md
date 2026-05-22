# Installation And HPC Notes

## Local Editable Install

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

The editable install is recommended for notebooks and HPC scripts because the
repository is still under active development. The package lives under
`src/queer`, so installing editably is the intended way to make notebooks and
scripts see source-code edits immediately.

For this checkout, the local `.venv` has also been installed as an editable
package and registered as the Jupyter kernel `Python (queer editable)`.
Select that kernel in notebooks so edits under `queer/` are imported directly.

## Dependencies

Runtime packages are listed in `requirements.txt`: NumPy, SciPy, Matplotlib,
Pandas, Joblib, psutil, tqdm, Pillow, ImageIO, Jupyter, nbformat, and pytest for
smoke testing.

## Organized Paths

Local data and results are intentionally ignored by Git:

- Inputs: `data/materials/<material>/`
- Legacy large inputs: `data/archive/`
- Generated figures/logs: `results/`

Use environment variables to point runs at shared or scratch storage:

```bash
export QUEER_DATA_DIR=/scratch/$USER/queer/data
export QUEER_RESULTS_DIR=/scratch/$USER/queer/results
```

The package resolves these through `queer.paths.data_path()` and
`queer.paths.results_path()`.

## Running On Slurm

Use the provided launcher:

```bash
sbatch scripts/hpc/run_v0_sweep.sbatch
```

Optional environment variables:

- `QUEER_VENV`: virtual environment to activate before running.
- `QUEER_DATA_DIR`: data root containing `materials/`.
- `QUEER_RESULTS_DIR`: output root for generated frames and logs.

## Verification

After moving or restoring artifacts, compare against
`docs/artifact_manifest.tsv`. The manifest records old path, new path, size,
category, and checksums for notebooks, scripts, docs, and data files.
