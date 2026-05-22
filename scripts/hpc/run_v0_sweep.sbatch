#!/bin/bash
#SBATCH --partition=bansil2
#SBATCH --nodes=5
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=128
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=run_V0_multi
#SBATCH --output=results/v0_sweep/logs/run_V0_multi_%j.out
#SBATCH --time=30-00:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if [ -f /shared/centos7/intel/oneapi/2022.1.0/setvars.sh ]; then
    source /shared/centos7/intel/oneapi/2022.1.0/setvars.sh
fi

VENV_DIR="${QUEER_VENV:-${PROJECT_ROOT}/.venv}"
if [ -d "${VENV_DIR}" ]; then
    source "${VENV_DIR}/bin/activate"
fi

export QUEER_PROJECT_ROOT="${QUEER_PROJECT_ROOT:-${PROJECT_ROOT}}"
export QUEER_DATA_DIR="${QUEER_DATA_DIR:-${PROJECT_ROOT}/data}"
export QUEER_RESULTS_DIR="${QUEER_RESULTS_DIR:-${PROJECT_ROOT}/results}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

mkdir -p "${QUEER_RESULTS_DIR}/v0_sweep/logs"

for V0 in 5 8 10 12 15; do
    srun -n1 python scripts/runs/v0_sweep/run_v0.py \
        --V0 "${V0}" \
        --num-core "${SLURM_CPUS_PER_TASK:-1}" \
        > "${QUEER_RESULTS_DIR}/v0_sweep/logs/output_V0_${V0}.log" 2>&1 &
done

wait
