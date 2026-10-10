#!/bin/bash
#SBATCH --job-name=queer-momentum
set -euo pipefail
: "${QUEER_SWEEP_FILE:?Submit with scripts/hpc/submit_sweep.py}"
: "${QUEER_PYTHON:?Pinned environment interpreter is required}"
: "${QUEER_SWEEP_RUNNER:?Array task runner is required}"
: "${SLURM_ARRAY_TASK_ID:?This launcher requires an array task}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
exec srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
    "$QUEER_PYTHON" "$QUEER_SWEEP_RUNNER" "$QUEER_SWEEP_FILE" "$SLURM_ARRAY_TASK_ID"
