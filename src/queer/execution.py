"""Explicit CPU and memory budgets for scientific calculations."""

from __future__ import annotations

import os
import threading
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any, Iterator


def cpu_budget() -> int:
    """Return the smallest available scheduler, affinity, and host CPU budget."""
    limits = [os.cpu_count() or 1]
    if hasattr(os, "sched_getaffinity"):
        try:
            limits.append(len(os.sched_getaffinity(0)))
        except OSError:
            pass
    allocation = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocation:
        try:
            cpus = int(allocation)
        except ValueError as error:
            raise ValueError("SLURM_CPUS_PER_TASK must be a positive integer") from error
        if cpus < 1:
            raise ValueError("SLURM_CPUS_PER_TASK must be a positive integer")
        limits.append(cpus)
    return max(1, min(limits))


@dataclass(frozen=True)
class ExecutionOptions:
    """Process, native-thread, and temporary-memory limits.

    ``workers`` is an upper bound; scheduler/CPU affinity can reduce it. The
    memory budget applies to concurrent numerical working arrays, excluding
    the input model and the returned result array. Serial execution is the
    default so notebooks do not unexpectedly start a process pool.
    """

    workers: int = 1
    batch_size: int = 256
    native_threads: int = 1
    memory_mb: float | None = None

    def __post_init__(self):
        for name in ("workers", "batch_size", "native_threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.memory_mb is not None:
            import math

            if isinstance(self.memory_mb, bool) or not math.isfinite(self.memory_mb) or self.memory_mb <= 0:
                raise ValueError("memory_mb must be a positive finite number")

    def resolved(self) -> ExecutionOptions:
        """Cap workers so workers × native threads fits the CPU allocation."""
        cpus = cpu_budget()
        if self.native_threads > cpus:
            raise ValueError(f"native_threads={self.native_threads} exceeds CPU allocation {cpus}")
        return ExecutionOptions(
            workers=min(self.workers, cpus // self.native_threads),
            batch_size=self.batch_size,
            native_threads=self.native_threads,
            memory_mb=self.memory_mb,
        )

    def bounded_batch_size(self, bytes_per_point: int) -> int:
        """Bound per-worker batches with a 20% allowance for solver overhead."""
        if self.memory_mb is None:
            return self.batch_size
        count = int(self.memory_mb * 1024**2 * 0.8 / (self.workers * bytes_per_point))
        if count < 1:
            raise ValueError("memory_mb is too small for one k-point per worker")
        return min(self.batch_size, count)

    def as_dict(self) -> dict:
        return asdict(self)


class _ExecutionSession:
    """Own a lazily opened executor for one synchronous calculation scope."""

    def __init__(self, options: ExecutionOptions):
        self.options = options
        self.owner_thread = threading.get_ident()
        self._resources = ExitStack()
        self._parallel: Any = None

    def parallel(self) -> Any:
        if self._parallel is None:
            from joblib import Parallel, parallel_config

            self._resources.enter_context(
                parallel_config(backend="loky", inner_max_num_threads=self.options.native_threads)
            )
            self._parallel = self._resources.enter_context(
                Parallel(
                    n_jobs=self.options.workers,
                    return_as="generator",
                    pre_dispatch=self.options.workers,
                    batch_size=1,
                    max_nbytes="1M",
                    mmap_mode="r",
                )
            )
        return self._parallel

    def close(self) -> None:
        self._resources.close()


_current_session: ContextVar[_ExecutionSession | None] = ContextVar("queer_execution_session", default=None)


@contextmanager
def execution_session(options: ExecutionOptions) -> Iterator[_ExecutionSession]:
    """Reuse an executor across energy calls in a workflow and close it on exit.

    Matching nested scopes reuse their parent. A changed budget or a context
    copied into another thread gets an independent executor; resetting the
    ContextVar restores the parent even when a calculation raises. Executors
    are opened only for parallel work, so serial/resumed runs start no pool.
    Callers must consume result generators before requesting another batch.
    """
    resolved = options.resolved()
    active = _current_session.get()
    if active is not None and active.options == resolved and active.owner_thread == threading.get_ident():
        yield active
        return
    session = _ExecutionSession(resolved)
    token = _current_session.set(session)
    try:
        yield session
    finally:
        _current_session.reset(token)
        session.close()
