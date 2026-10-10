"""Reproducible wall-time and memory evidence; run after installing QUEER."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from queer import ExecutionOptions, WannierModel
from queer.io.artifacts import ALGORITHM_VERSION, digest_file, environment_versions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--material-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests/fixtures/wannier/chain",
    )
    parser.add_argument("--ef", type=float, default=0.0)
    parser.add_argument("--points", nargs="+", type=int, default=[1000, 100000])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--memory-mb", type=float, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    model = WannierModel.from_directory(args.material_dir, ef=args.ef, interpolation="standard")
    execution = ExecutionOptions(
        workers=args.workers, batch_size=args.batch_size, memory_mb=args.memory_mb
    ).resolved()
    results = []
    for count in args.points:
        if count < 1:
            raise ValueError("Benchmark sizes must be positive")
        points = np.zeros((count, 3))
        points[:, 0] = np.linspace(0, 1, count, endpoint=False)
        start = time.perf_counter()
        bands = model.energies(points, execution=execution)
        elapsed = time.perf_counter() - start
        serial = model.energies(points[: min(count, 32)])
        np.testing.assert_allclose(bands[:, : serial.shape[1]], serial, rtol=1e-10, atol=1e-10)
        results.append(
            {
                "k_points": count,
                "bands": model.nbnd,
                "wall_seconds": elapsed,
                "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "execution": execution.as_dict(),
                "finite": bool(np.isfinite(bands).all()),
            }
        )
        del bands, points, serial
    hr_file = args.material_dir.resolve() / "wannier90_hr.dat"
    payload = json.dumps(
        {
            "schema_version": 1,
            "algorithm_version": ALGORITHM_VERSION,
            "environment": environment_versions(),
            "input": {"path": str(hr_file), "sha256": digest_file(hr_file)},
            "model": {"ef": args.ef, "interpolation": "standard", "dtype": "float64/complex128"},
            "workloads": results,
        },
        indent=2,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
