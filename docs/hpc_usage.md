# Slurm operations

Prepare the pinned environment on storage visible to compute nodes:

```bash
uv sync --locked --extra plot --extra symmetry --extra notebook --python 3.13
.venv/bin/python scripts/hpc/submit_sweep.py --config calculation.toml \
  --profile configs/hpc/default.toml --dry-run
```

Copy/customize the profile's CPU allocation, memory, wall time, array concurrency,
partition, and account. Remove `--dry-run` to submit. The submission adapter creates
an isolated submission directory and logs before calling `sbatch`. It exports the
prepared interpreter and explicit dispatch/runner paths. The launcher does not
infer checkout paths from Slurm's spooled script and has no site-specific Intel or
CentOS assumptions. Submission copies the source TOML to `config.toml`, records its
SHA-256 in `dispatch.json`, and records resolved material/results roots. Every task
verifies the snapshot and passes those roots explicitly, so relative paths retain
their submission-time meaning. Later edits to the original TOML do not alter queued
tasks; a changed submitted snapshot is rejected.

Input data are hashed when each task starts, rather than copied or hashed at
submission. Input files, the prepared interpreter, runner, and submission directory
must be visible from every node. Keep inputs and the prepared environment unchanged
while the array is queued/running so all tasks use the same scientific data.

Each task executes one resolved workflow case through the installed CLI, writes its own run
manifest and numerical checkpoints, and exits nonzero on failure. Logs use `%A_%a`.
Native thread pools default to one; process workers are capped by scheduler allocation
and affinity. Choose `execution.memory_mb` and `batch_size` to bound concurrent working
arrays; this budget excludes model storage and returned result arrays. Scheduler
memory must also cover those allocations and process-worker copies. Use node-local
`JOBLIB_TEMP_FOLDER` when appropriate for the site's shared-memory policy.

The adapter supports all registered native OpenMX observables and their complete
configuration, as well as the existing momentum/V0 sweep. Repeat `--config` to
submit distinct configurations in one array:

```bash
.venv/bin/python scripts/hpc/submit_sweep.py \
  --config configs/examples/openmx_bands.toml \
  --config configs/examples/openmx_dos.toml \
  --profile configs/hpc/default.toml --dry-run
```

Each generic case gets an immutable `config-NNNN.toml` snapshot, resolved roots,
SHA256 and a separate `task-NNNN` run. Fixed integration tiles and refinement
levels survive execution batch changes. Pockels and CRTA transport checkpoint
additive quantities before global reconstruction. Whole-mesh EMPC and loop/search
tasks checkpoint complete cases. Rendering can be done later on a login node
using stored artifacts, without the original SCFOUT or PAO files.

For scratch storage:

```bash
export QUEER_DATA_DIR=/scratch/$USER/queer/data
export QUEER_RESULTS_DIR=/scratch/$USER/queer/results
```

Set these before submission, or supply explicit roots in the configuration. CLI
roots take precedence. No environment setup occurs inside task execution.

## Recovery

Scheduler requeues with a positive `SLURM_RESTART_COUNT` automatically resume an
existing task manifest. The CLI verifies configuration, inputs and checkpoint
checksums before continuing. Restarts without a prior manifest request a fresh run;
fresh duplicate task invocations retain the existing-directory failure behavior.

Inspect `manifest.json` and task logs, correct transient environment/storage issues,
and resume the task directory using the submitted snapshot and V0. Read the resolved
`material_dir` and `results_dir` values from the submission's `dispatch.json` and pass
them explicitly; the copied TOML can contain paths relative to its original location:

```bash
queer momentum --config /path/to/submission/config.toml --v0 12 \
  --material-dir /resolved/material/root --results-dir /resolved/results/root \
  --run-dir /path/to/submission/task-0001 --resume
```

Resume verifies scientific parameters, algorithm version, input checksums, and
completed artifact checksums. Modified scientific inputs require a new run. Advisory
locks exclude concurrent writers and release when a process exits or is killed;
shared storage must support advisory file locking. SIGTERM records interruption;
SIGKILL can leave status `running`, but completed atomic checkpoints remain resumable.
Completed runs can be resumed idempotently. Rendering consumes completed artifacts.

For generic arrays, recovery can also call the adapter with the original index:

```bash
SLURM_RESTART_COUNT=1 .venv/bin/python scripts/hpc/array_task.py \
  /path/to/submission/dispatch.json 1
```

Use the same prepared interpreter and scheduler worker allocation when reproducing
resource behavior. A failed convergence gate requires scientific investigation;
modifying the mesh or tolerance starts a new scientific run.

## Cluster acceptance

Local tests emulate submission and task failure; they do not establish behavior on a
real scheduler. Before production cluster rollout, run the tiny momentum array and
confirm logs, task isolation, accounting, and numerical output. Force one invalid
input and confirm its failed task status. Terminate/resume a larger task and inspect
checkpoint reuse. Compare serial/parallel outputs, time, and aggregate memory with
`benchmarks/reference.py` and scheduler accounting. Real-material datasets are opt-in.
