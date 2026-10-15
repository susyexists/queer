"""CPU and memory planning of the Generator Wannier steps on a batch node.

These are the Generator PythTB driver's rules. CPUs come from Slurm's node
allocation, else this process's affinity. Available memory is the smaller of
``MemAvailable`` and the room left under the tightest memory limit of this
process's cgroups (v1 or v2, any ancestor), and a stage may use 60% of it. A
k-point keeps ``MATRICES_PER_K[stage]`` complex N x N matrices alive at peak
and every worker also holds a fixed share (for a model stage, 32 bytes per
entry of the hopping table); workers are lowered until their chunks fit.
Stages run on QUEER's :class:`~queer.execution.ExecutionOptions` with one
native thread per worker, and each plan is recorded for provenance.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable
from typing import Any

from ...execution import ExecutionOptions, execution_session

# Complex N x N matrices one k-point keeps alive at peak, per stage.
MATRICES_PER_K: dict[str, int] = {
    "validation": 2,
    "band path": 4,
    "DOS": 2,
    "DOS with PDOS": 4,
    "occupation": 3,
    "Fermi surface": 2,
    "Wilson loops": 3,
    "Berry bands": 14,
    "Berry fermi": 8,
    "slab": 4,
}
FALLBACK_MEMORY_BYTES = 8 * 1024**3
MEMORY_FRACTION = 0.6
MAX_CHUNK = 4096
MIN_AUTOMATIC_CHUNK = 4
MAX_WORKER_BYTES = 4.0 * 1024**3
# A worker of a model stage rebuilds the per-translation hopping matrices.
TABLE_BYTES_PER_ENTRY = 32.0
# Limits at or above this many bytes mean "unlimited" (cgroup v1 writes 2**63 rounded to a page).
UNLIMITED_BYTES = 1 << 60


def available_cpus() -> int:
    """CPUs of this job: Slurm's node allocation, else the process affinity."""
    for key in ("SLURM_CPUS_ON_NODE", "SLURM_CPUS_PER_TASK"):
        value = os.environ.get(key)
        if value and value.split("(")[0].isdigit():
            return max(1, int(value.split("(")[0]))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def cgroup_memory_room(proc_cgroup: str = "/proc/self/cgroup", root: str = "/sys/fs/cgroup") -> int | None:
    """Bytes left under the tightest memory limit of this process's cgroups.

    Each ``/proc/self/cgroup`` entry is ``hierarchy:controllers:path``. The v2
    entry (``0::path``) reads ``memory.max`` and ``memory.current`` under
    ``root``; a v1 entry with the ``memory`` controller reads
    ``memory.limit_in_bytes`` and ``memory.usage_in_bytes`` under
    ``root/memory``. Slurm's cgroup enforcement caps a job well below the
    node's MemAvailable, and a limit can sit on any ancestor (job, step, task),
    so every level up to the root counts; ``max`` and v1's near-2**63 value mean
    no limit, and an unreadable level is skipped. None when no level has a limit.
    """
    try:
        with open(proc_cgroup) as handle:
            entries = [line.rstrip("\n").split(":", 2) for line in handle]
    except OSError:
        return None
    rooms: list[int] = []
    for entry in entries:
        if len(entry) != 3:
            continue
        hierarchy, controllers, path = entry
        if hierarchy == "0" and not controllers:
            base, limit_name, usage_name = root, "memory.max", "memory.current"
        elif "memory" in controllers.split(","):
            base = os.path.join(root, "memory")
            limit_name, usage_name = "memory.limit_in_bytes", "memory.usage_in_bytes"
        else:
            continue
        parts = [part for part in path.split("/") if part]
        for depth in range(len(parts), -1, -1):
            directory = os.path.join(base, *parts[:depth])
            try:
                with open(os.path.join(directory, limit_name)) as handle:
                    raw = handle.read().strip()
                with open(os.path.join(directory, usage_name)) as handle:
                    usage = int(handle.read().strip())
                limit = None if raw == "max" else int(raw)
            except (OSError, ValueError):
                continue
            if limit is None or limit >= UNLIMITED_BYTES:
                continue
            rooms.append(max(0, limit - usage))
    return min(rooms) if rooms else None


def available_memory_bytes(
    meminfo: str = "/proc/meminfo",
    proc_cgroup: str = "/proc/self/cgroup",
    root: str = "/sys/fs/cgroup",
) -> int:
    """MemAvailable, or less when this process's cgroups allow less; 8 GiB when neither is known."""
    found: list[int] = []
    try:
        with open(meminfo) as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    found.append(int(line.split()[1]) * 1024)
                    break
    except (OSError, ValueError, IndexError):
        pass
    room = cgroup_memory_room(proc_cgroup, root)
    if room is not None:
        found.append(room)
    return min(found) if found else FALLBACK_MEMORY_BYTES


def peak_rss() -> dict[str, int]:
    """Peak resident memory of this process and of its largest reaped child (Linux KiB counters)."""
    try:
        import resource

        own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
    except (ImportError, OSError, ValueError):
        return {}
    return {"peak_rss_bytes": int(own), "peak_worker_rss_bytes": int(children)}


def model_fixed_bytes(n_translations: int, n_orb: int) -> float:
    """What every worker of a model stage holds whatever its chunk: the hopping table's matrices."""
    return TABLE_BYTES_PER_ENTRY * int(n_translations) * int(n_orb) ** 2


def plan_parallelism(
    n_orb: int,
    per_k_matrices: int,
    workers: int | None,
    chunk: int | None,
    *,
    fixed_bytes: float = 0.0,
    n_items: int | None = None,
    memory_bytes: float | None = None,
) -> tuple[int, int]:
    """(workers, k-points per chunk) so that every worker's arrays fit in memory.

    ``per_k_matrices`` complex ``n_orb x n_orb`` matrices stay alive per k-point
    and ``fixed_bytes`` per worker whatever its chunk. An automatic chunk is what
    fits one worker's share of the budget (at most 4 GiB, between 4 and 4096
    k-points), capped at the even share ceil(n_items / workers); a requested
    chunk size is kept. Workers, requested or all CPUs, are lowered until every
    worker fits. ``memory_bytes`` defaults to :func:`available_memory_bytes`.
    """
    per_k = 16.0 * n_orb * n_orb * max(1, per_k_matrices)
    room = available_memory_bytes() if memory_bytes is None else float(memory_bytes)
    budget = MEMORY_FRACTION * room
    workers = max(1, int(workers if workers is not None else available_cpus()))
    if chunk is None:
        per_worker = min(budget / workers - fixed_bytes, MAX_WORKER_BYTES)
        chunk = int(max(MIN_AUTOMATIC_CHUNK, min(MAX_CHUNK, per_worker // per_k)))
        if n_items:
            chunk = min(chunk, math.ceil(int(n_items) / workers))
    chunk = max(1, int(chunk))
    while workers > 1 and workers * (chunk * per_k + fixed_bytes) > budget:
        workers -= 1
    return workers, chunk


def pool_processes(workers: int, n_items: int) -> int:
    """Processes a stage of ``n_items`` tasks runs on: 1 runs them in this process."""
    return min(workers, n_items) if workers > 1 and n_items > 1 else 1


class ParallelPlanner:
    """Stage plans of one run and their record for ``provenance.parallel``.

    ``workers`` and ``chunk_size`` are the run's requests (None: automatic);
    ``log`` receives one line per stage; ``records`` (a new list by default)
    receives one ``{stage, k_points, chunk, tasks, workers, processes}`` entry
    per stage, ``workers`` being the plan and ``processes`` what runs once the
    CPU allocation caps it. ``memory_bytes`` fixes the memory every plan sees.
    """

    def __init__(
        self,
        workers: int | None = None,
        chunk_size: int | None = None,
        log: Callable[[str], None] | None = None,
        *,
        records: list[dict[str, Any]] | None = None,
        memory_bytes: float | None = None,
    ):
        self.workers = workers
        self.chunk_size = chunk_size
        self.log = log
        self.records: list[dict[str, Any]] = [] if records is None else records
        self.memory_bytes = memory_bytes

    def plan(
        self,
        stage: str,
        per_k_matrices: int,
        n_items: int,
        *,
        n_orb: int,
        fixed_bytes: float = 0.0,
        chunk: int | None = None,
    ) -> ExecutionOptions:
        """Execution options of a stage solving ``n_items`` k-points; logged and recorded.

        ``chunk`` fixes the k-points of one task when a task cannot be split (a
        Wilson loop, one slab per k-point); otherwise the run's ``chunk_size`` or
        the automatic size applies.
        """
        workers, size = plan_parallelism(
            n_orb,
            per_k_matrices,
            self.workers,
            chunk if chunk is not None else self.chunk_size,
            fixed_bytes=fixed_bytes,
            n_items=n_items,
            memory_bytes=self.memory_bytes,
        )
        tasks = math.ceil(n_items / size) if n_items else 0
        options = ExecutionOptions(workers=workers, batch_size=size, native_threads=1).resolved()
        processes = pool_processes(options.workers, tasks)
        self.records.append(
            {
                "stage": stage,
                "k_points": int(n_items),
                "chunk": int(size),
                "tasks": int(tasks),
                "workers": int(workers),
                "processes": int(processes),
            }
        )
        if self.log is not None:
            self.log(
                f"{stage}: {n_items} k-points, {tasks} task(s) of up to {size} k on {processes} process(es)"
            )
        return options


def chunks(points, size: int) -> list:
    """Consecutive slices of at most ``size`` rows."""
    step = max(1, int(size))
    return [points[start : start + step] for start in range(0, len(points), step)]


def pmap(function: Callable[[Any], Any], items: Iterable[Any], options: ExecutionOptions) -> list[Any]:
    """Ordered map over QUEER's process executor; one task per item.

    Large NumPy arrays reachable from ``function`` (a model's tables) reach the
    workers memory-mapped rather than copied per task. With one worker or one
    item everything runs in this process under the native-thread limit.
    """
    tasks = list(items)
    resolved = options.resolved()
    if pool_processes(resolved.workers, len(tasks)) == 1:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=resolved.native_threads):
            return [function(item) for item in tasks]
    from joblib import delayed

    with execution_session(resolved) as session:
        return list(session.parallel()(delayed(function)(item) for item in tasks))
