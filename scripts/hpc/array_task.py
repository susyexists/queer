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
    case = dispatch.get("cases", [])[index] if "cases" in dispatch else dispatch
    for source in case.get("inputs", []):
        path = Path(source["path"])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"Submitted scientific input was changed or removed: {path}")
    if "config_sha256" in case:
        actual = hashlib.sha256(Path(case["config"]).read_bytes()).hexdigest()
        if actual != case["config_sha256"]:
            raise ValueError("Submitted configuration snapshot was changed")
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
        case.get("workflow", "momentum"),
        "--config",
        case["config"],
        "--run-dir",
        str(run_dir),
        "--workers",
        os.environ.get("SLURM_CPUS_PER_TASK", "1"),
        "--native-threads",
        "1",
    ]
    if "cases" not in dispatch:
        command.extend(["--v0", str(dispatch["values"][index])])
    if "config_base_dir" in case:
        command.extend(["--config-base-dir", case["config_base_dir"]])
    if restart_count > 0 and (run_dir / "manifest.json").is_file():
        command.append("--resume")
    for field in ("material_dir", "results_dir"):
        if field in case:
            command.extend(["--" + field.replace("_", "-"), case[field]])
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
