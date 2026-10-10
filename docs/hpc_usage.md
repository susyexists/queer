# Slurm operations

Prepare the pinned environment on storage visible to compute nodes:

```bash
uv sync --locked --all-extras --python 3.13
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

Each task executes one V0 case through the installed CLI, writes its own run
manifest and numerical checkpoints, and exits nonzero on failure. Logs use `%A_%a`.
Native thread pools default to one; process workers are capped by scheduler allocation
and affinity. Choose `execution.memory_mb` and `batch_size` to bound concurrent working
arrays; this budget excludes model storage and returned result arrays. Scheduler
memory must also cover those allocations and process-worker copies. Use node-local
`JOBLIB_TEMP_FOLDER` when appropriate for the site's shared-memory policy.

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

## Cluster acceptance

Local tests emulate submission and task failure; they do not establish behavior on a
real scheduler. Before production cluster rollout, run the tiny momentum array and
confirm logs, task isolation, accounting, and numerical output. Force one invalid
input and confirm its failed task status. Terminate/resume a larger task and inspect
checkpoint reuse. Compare serial/parallel outputs, time, and aggregate memory with
`benchmarks/reference.py` and scheduler accounting. Real-material datasets are opt-in.
