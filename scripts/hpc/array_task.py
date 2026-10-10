#!/usr/bin/env python
"""Execute one pre-resolved parameter case and propagate its exit status."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise ValueError("expected dispatch JSON and array index")
    dispatch = json.loads(Path(args[0]).read_text())
    if "config_sha256" in dispatch:
        actual = hashlib.sha256(Path(dispatch["config"]).read_bytes()).hexdigest()
        if actual != dispatch["config_sha256"]:
            raise ValueError("Submitted configuration snapshot was changed")
    index = int(args[1])
    if index < 0 or index >= len(dispatch["values"]):
        raise ValueError("array index outside parameter cases")
    try:
        restart_count = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    except ValueError as error:
        raise ValueError("SLURM_RESTART_COUNT must be a nonnegative integer") from error
    if restart_count < 0:
        raise ValueError("SLURM_RESTART_COUNT must be a nonnegative integer")
    run_dir = Path(dispatch["run_root"]) / f"task-{index:04d}"
    command = [
        sys.executable,
        "-m",
        "queer",
        "momentum",
        "--config",
        dispatch["config"],
        "--v0",
        str(dispatch["values"][index]),
        "--run-dir",
        str(run_dir),
        "--workers",
        os.environ.get("SLURM_CPUS_PER_TASK", "1"),
        "--native-threads",
        "1",
    ]
    if restart_count > 0 and (run_dir / "manifest.json").is_file():
        command.append("--resume")
    for field in ("material_dir", "results_dir"):
        if field in dispatch:
            command.extend(["--" + field.replace("_", "-"), dispatch[field]])
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
