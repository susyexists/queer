# HPC Usage

The Slurm entrypoint is:

```bash
sbatch scripts/hpc/run_v0_sweep.sbatch
```

Set these environment variables when data or results live outside the checkout:

```bash
export QUEER_DATA_DIR=/scratch/$USER/queer/data
export QUEER_RESULTS_DIR=/scratch/$USER/queer/results
export QUEER_VENV=/scratch/$USER/envs/queer
```

The launcher runs:

```bash
python scripts/runs/v0_sweep/run_v0.py --V0 <value>
```

For Ag primitive momentum-microscopy runs, pass the Ag material and Fermi
energy explicitly:

```bash
python scripts/runs/v0_sweep/run_v0.py \
  --V0 12 \
  --material ag_primitive \
  --ef 8.310342 \
  --binding-min 4.7 \
  --binding-max 6.7 \
  --binding-step 0.5 \
  --align-from 0 0 1 \
  --align-to 1 1 1
```
