"""CPU, cgroup memory and per-stage plans follow the Generator PythTB driver."""

import itertools

import pytest

from queer.compat.generator import memory
from queer.compat.generator.memory import (
    FALLBACK_MEMORY_BYTES,
    ParallelPlanner,
    available_cpus,
    available_memory_bytes,
    cgroup_memory_room,
    chunks,
    model_fixed_bytes,
    plan_parallelism,
    pmap,
)
from queer.execution import ExecutionOptions

GIB = 1024**3


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def cgroup_v2(tmp_path, levels, path="/slurm/job_1/step_0/task_0"):
    """A fake v2 hierarchy; ``levels`` maps a relative directory to (memory.max, memory.current)."""
    proc = tmp_path / "proc_cgroup"
    write(proc, f"0::{path}\n")
    root = tmp_path / "cgroup"
    for directory, (limit, usage) in levels.items():
        write(root / directory / "memory.max", f"{limit}\n")
        write(root / directory / "memory.current", f"{usage}\n")
    return str(proc), str(root)


def test_v2_room_is_the_tightest_ancestor(tmp_path):
    proc, root = cgroup_v2(
        tmp_path,
        {
            "": ("max", 50 * GIB),
            "slurm/job_1": (8 * GIB, 2 * GIB),
            "slurm/job_1/step_0": (7 * GIB, 3 * GIB),
            "slurm/job_1/step_0/task_0": ("max", 3 * GIB),
        },
    )
    assert cgroup_memory_room(proc, root) == 4 * GIB


def test_v2_usage_above_the_limit_leaves_no_room(tmp_path):
    proc, root = cgroup_v2(tmp_path, {"slurm/job_1": (2 * GIB, 3 * GIB)})
    assert cgroup_memory_room(proc, root) == 0


def test_v1_memory_controller_and_its_unlimited_value(tmp_path):
    proc = tmp_path / "proc_cgroup"
    write(proc, "12:cpu,cpuacct:/slurm/uid_1/job_2\n5:memory:/slurm/uid_1/job_2\n1:name=systemd:/\nbroken\n")
    root = tmp_path / "cgroup"
    write(root / "memory" / "memory.limit_in_bytes", "9223372036854771712\n")
    write(root / "memory" / "memory.usage_in_bytes", f"{40 * GIB}\n")
    job = root / "memory" / "slurm" / "uid_1" / "job_2"
    write(job / "memory.limit_in_bytes", f"{4 * GIB}\n")
    write(job / "memory.usage_in_bytes", f"{1 * GIB}\n")
    write(root / "memory" / "slurm" / "memory.limit_in_bytes", "garbage\n")
    write(root / "memory" / "slurm" / "memory.usage_in_bytes", "0\n")
    assert cgroup_memory_room(str(proc), str(root)) == 3 * GIB


def test_no_limit_or_no_cgroup_file_means_none(tmp_path):
    proc, root = cgroup_v2(tmp_path, {"": ("max", GIB), "slurm/job_1": ("max", GIB)})
    assert cgroup_memory_room(proc, root) is None
    assert cgroup_memory_room(str(tmp_path / "absent"), root) is None
    write(tmp_path / "other", "3:cpu:/x\n")
    assert cgroup_memory_room(str(tmp_path / "other"), root) is None


def test_available_memory_is_the_smaller_of_memavailable_and_the_cgroup_room(tmp_path):
    meminfo = tmp_path / "meminfo"
    write(meminfo, "MemTotal:       67108864 kB\nMemAvailable:   16777216 kB\n")
    proc, root = cgroup_v2(tmp_path, {"slurm/job_1": (10 * GIB, 4 * GIB)})
    assert available_memory_bytes(str(meminfo), proc, root) == 6 * GIB
    proc, root = cgroup_v2(tmp_path / "loose", {"slurm/job_1": (100 * GIB, 4 * GIB)})
    assert available_memory_bytes(str(meminfo), proc, root) == 16 * GIB
    absent = str(tmp_path / "absent")
    assert available_memory_bytes(absent, absent, absent) == FALLBACK_MEMORY_BYTES
    assert available_memory_bytes(absent, proc, root) == 96 * GIB


def test_cpus_come_from_slurm_then_the_affinity(monkeypatch):
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "16(x2)")
    assert available_cpus() == 16
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "many")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    assert available_cpus() == 4
    monkeypatch.delenv("SLURM_CPUS_ON_NODE")
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    monkeypatch.setattr(memory.os, "sched_getaffinity", lambda pid: {0, 1, 2})
    assert available_cpus() == 3


CASES = list(
    itertools.product(
        [8, 230],  # orbitals
        [2, 14],  # matrices per k-point
        [None, 3, 64],  # requested workers
        [None, 7],  # requested chunk
        [0.0, 5e8],  # fixed bytes per worker
        [None, 201, 22500],  # k-points of the stage
        [2 * GIB, 64 * GIB],  # available memory
    )
)


@pytest.mark.parametrize(("n_orb", "per_k", "workers", "chunk", "fixed", "n_items", "room"), CASES)
def test_plans_equal_the_drivers(
    generator_core, monkeypatch, n_orb, per_k, workers, chunk, fixed, n_items, room
):
    monkeypatch.setattr(generator_core, "_available_memory_bytes", lambda: room)
    monkeypatch.setattr(generator_core, "_available_cpus", lambda: 24)
    monkeypatch.setattr(memory, "available_cpus", lambda: 24)
    expected = generator_core.plan_parallelism(
        n_orb, per_k, workers, chunk, fixed_bytes=fixed, n_items=n_items
    )
    assert plan_parallelism(
        n_orb, per_k, workers, chunk, fixed_bytes=fixed, n_items=n_items, memory_bytes=room
    ) == (expected)


def test_planner_records_and_logs_every_stage(monkeypatch):
    monkeypatch.setattr(memory, "available_cpus", lambda: 4)
    monkeypatch.setattr(memory.ExecutionOptions, "resolved", lambda self: self)
    lines, records = [], []
    planner = ParallelPlanner(None, None, lines.append, records=records, memory_bytes=64 * GIB)
    options = planner.plan("validation", 2, 201, n_orb=230, fixed_bytes=model_fixed_bytes(189, 230))
    assert isinstance(options, ExecutionOptions)
    assert (options.workers, options.batch_size, options.native_threads) == (4, 51, 1)
    planner.plan("Wilson loops, k3=0", 3, 61 * 81, n_orb=8, chunk=81)
    planner.plan("empty", 2, 0, n_orb=8)
    assert records == [
        {"stage": "validation", "k_points": 201, "chunk": 51, "tasks": 4, "workers": 4, "processes": 4},
        {
            "stage": "Wilson loops, k3=0",
            "k_points": 4941,
            "chunk": 81,
            "tasks": 61,
            "workers": 4,
            "processes": 4,
        },
        {"stage": "empty", "k_points": 0, "chunk": 4096, "tasks": 0, "workers": 4, "processes": 1},
    ]
    assert lines[0] == "validation: 201 k-points, 4 task(s) of up to 51 k on 4 process(es)"
    assert planner.records is records


def test_the_cpu_allocation_caps_the_processes_not_the_recorded_plan(monkeypatch):
    monkeypatch.setattr(memory, "available_cpus", lambda: 1)
    planner = ParallelPlanner(workers=8, chunk_size=10, memory_bytes=64 * GIB)
    monkeypatch.setattr(
        memory.ExecutionOptions,
        "resolved",
        lambda self: ExecutionOptions(1, self.batch_size, self.native_threads),
    )
    options = planner.plan("DOS", 2, 100, n_orb=4)
    assert options.workers == 1
    assert planner.records[0]["workers"] == 8 and planner.records[0]["processes"] == 1


def test_model_fixed_bytes_is_the_drivers_table_estimate():
    assert model_fixed_bytes(189, 230) == 32.0 * 189 * 230 * 230


def test_chunks_and_serial_map_keep_order():
    assert [list(part) for part in chunks(list(range(7)), 3)] == [[0, 1, 2], [3, 4, 5], [6]]
    assert pmap(lambda value: value * value, range(5), ExecutionOptions(workers=1)) == [0, 1, 4, 9, 16]
