#!/usr/bin/env python
"""Submit resolved workflow cases from a prepared environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import tomllib
import uuid
from pathlib import Path

from queer.config import config_from_mapping


def _toml_text(data):
    """Serialize resolved workflow tables without an optional TOML writer."""

    def literal(value):
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (str, Path)):
            return json.dumps(str(value))
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(literal(item) for item in value) + "]"
        if isinstance(value, (int, float)):
            return repr(value)
        raise ValueError(f"Cannot serialize resolved TOML value: {type(value).__name__}")

    lines = []

    def table(values, prefix=""):
        if prefix:
            lines.extend(["", f"[{prefix}]"])
        for key, value in values.items():
            if value is not None and not isinstance(value, dict):
                lines.append(f"{key} = {literal(value)}")
        for key, value in values.items():
            if isinstance(value, dict):
                table(value, f"{prefix}.{key}" if prefix else key)

    table(data)
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--split-members", action="store_true", help="Submit isolated hall_harmonics tasks for a harmonic_map"
    )
    args = parser.parse_args(argv)
    config_path = args.config[0].expanduser().resolve()
    config_text = config_path.read_text()
    config = config_from_mapping(tomllib.loads(config_text), base_dir=config_path.parent)
    profile = tomllib.loads(args.profile.read_text())
    unknown = set(profile) - {"cpus_per_task", "memory", "time", "concurrency", "partition", "account"}
    if unknown:
        raise ValueError(f"Unknown HPC options: {sorted(unknown)}")
    cpus = profile.get("cpus_per_task", 1)
    concurrency = profile.get("concurrency", 1)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (cpus, concurrency)
    ):
        raise ValueError("CPU allocation and concurrency must be positive integers")
    legacy = config.workflow == "momentum" and len(args.config) == 1
    cases = []
    aggregation = None
    if args.split_members:
        if len(args.config) != 1:
            raise ValueError("--split-members accepts exactly one harmonic_map configuration")
        from queer.workflows.finite_field import finite_field_input_files, split_harmonic_map

        children, aggregation = split_harmonic_map(config)
        parent_inputs = finite_field_input_files(config)
        for index, child in enumerate(children):
            leaves = sorted(set(parent_inputs + finite_field_input_files(child)))
            cases.append(
                {
                    "workflow": child.workflow,
                    "results_dir": str(child.results_dir),
                    **({"material_dir": str(child.material_dir)} if "material_dir" in child.inputs else {}),
                    "config_base_dir": str(config_path.parent),
                    "source": str(config_path),
                    "text": _toml_text(child.as_dict()),
                    "resolved_config": child.as_dict(),
                    "member_index": index,
                    "parameter": aggregation["parameters"][index],
                    "label": aggregation["labels"][index],
                    "inputs": [
                        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                        for path in leaves
                    ],
                }
            )
        values = list(range(len(cases)))
    elif legacy:
        values = config.parameters.get("v0_values", [config.parameters.get("v0")])
        if not values or any(value is None or not math.isfinite(float(value)) for value in values):
            raise ValueError("Provide finite v0 or v0_values in the workflow configuration")
    else:
        for source in args.config:
            source = source.expanduser().resolve()
            text = source.read_text()
            resolved = config_from_mapping(tomllib.loads(text), base_dir=source.parent)
            cases.append(
                {
                    "workflow": resolved.workflow,
                    **(
                        {"material_dir": str(resolved.material_dir)}
                        if "material_dir" in resolved.inputs
                        else {}
                    ),
                    "results_dir": str(resolved.results_dir),
                    "config_base_dir": str(source.parent),
                    "source": str(source),
                    "text": text,
                }
            )
            from queer.workflows.finite_field import FINITE_FIELD_NAMES, finite_field_input_files

            if resolved.workflow in FINITE_FIELD_NAMES:
                leaves = finite_field_input_files(resolved)
                cases[-1]["inputs"] = [
                    {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in leaves
                ]
        values = list(range(len(cases)))
    if not args.python.absolute().is_file():
        raise ValueError("Prepared Python interpreter does not exist")
    root = config.results_dir / ("submission-preview" if args.dry_run else f"submission-{uuid.uuid4().hex}")
    runner = Path(__file__).with_name("array_task.py").resolve()
    launcher = Path(__file__).with_name("run_v0_sweep.sbatch").resolve()
    command = [
        "sbatch",
        f"--array=0-{len(values) - 1}%{concurrency}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={cpus}",
        f"--mem={profile.get('memory', '4G')}",
        f"--time={profile.get('time', '01:00:00')}",
        f"--chdir={Path.cwd().resolve()}",
        f"--output={root / 'logs' / '%A_%a.out'}",
        f"--error={root / 'logs' / '%A_%a.err'}",
        "--export=ALL",
    ]
    for option in ("partition", "account"):
        if profile.get(option):
            command.append(f"--{option}={profile[option]}")
    command.append(str(launcher))
    print(shlex.join(command))
    if args.dry_run:
        return 0
    (root / "logs").mkdir(parents=True, exist_ok=False)
    dispatch = root / "dispatch.json"
    snapshot = root / "config.toml"
    snapshot.write_text(config_text)
    if not legacy:
        for index, case in enumerate(cases):
            target = root / f"config-{index:04d}.toml"
            target.write_text(case.pop("text"))
            case["config"] = str(target)
            case["config_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    dispatch.write_text(
        json.dumps(
            {
                "config": str(snapshot),
                "config_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                **({"material_dir": str(config.material_dir)} if "material_dir" in config.inputs else {}),
                "results_dir": str(config.results_dir),
                "values": values,
                "run_root": str(root),
                **({"cases": cases} if not legacy else {}),
                **({"aggregation": aggregation} if aggregation is not None else {}),
            },
            indent=2,
        )
    )
    environment = dict(
        os.environ,
        QUEER_SWEEP_FILE=str(dispatch),
        QUEER_PYTHON=str(args.python.absolute()),
        QUEER_SWEEP_RUNNER=str(runner),
    )
    return subprocess.run(command, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
