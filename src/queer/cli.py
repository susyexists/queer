"""Installed command interface for local and HPC scientific workflows."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

from .config import WORKFLOWS, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="queer", description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in WORKFLOWS:
        command = commands.add_parser(name, help=f"Compute a {name} run from TOML configuration")
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--config-base-dir", type=Path, help=argparse.SUPPRESS)
        command.add_argument("--material-dir", type=Path)
        command.add_argument("--results-dir", type=Path)
        command.add_argument("--workers", type=int)
        command.add_argument("--native-threads", type=int)
        command.add_argument("--batch-size", type=int)
        command.add_argument("--memory-mb", type=float)
        command.add_argument("--run-dir", type=Path)
        command.add_argument("--resume", action="store_true")
        command.add_argument(
            "--render", action="store_true", help="Also render the persisted numerical result"
        )
        if name == "momentum":
            command.add_argument("--v0", type=float, help="Run this V0 instead of the configured sweep")
    render = commands.add_parser("render", help="Render an existing completed run without computation")
    render.add_argument("run_dir", type=Path)
    export = commands.add_parser("export", help="Export a completed run for a consuming application")
    export.add_argument("run_dir", type=Path)
    export.add_argument("--profile", choices=["generator", "finite_field"], default="generator")
    export.add_argument("--output-dir", type=Path)
    return parser


def _terminated(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


def main(argv=None) -> int:
    arguments = sys.argv[1:] if argv is None else list(argv)
    if arguments[:1] == ["generator"]:
        from ._generator_boot import main as generator_main

        return generator_main(arguments[1:])
    parser = build_parser()
    args = parser.parse_args(arguments)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    from .workflows import render_run, run_workflow

    previous = signal.signal(signal.SIGTERM, _terminated)
    try:
        if args.command == "render":
            output = render_run(args.run_dir)
        elif args.command == "export":
            if args.profile == "finite_field":
                from .compat.finite_field import export_run
            else:
                from .compat.generator import export_run

            output = export_run(args.run_dir, output_dir=args.output_dir, profile=args.profile)
        else:
            overrides = {
                key: getattr(args, key, None)
                for key in (
                    "material_dir",
                    "results_dir",
                    "workers",
                    "native_threads",
                    "batch_size",
                    "memory_mb",
                    "v0",
                )
            }
            config = load_config(
                args.config,
                workflow=args.command,
                overrides=overrides,
                base_dir=args.config_base_dir,
            )
            output = run_workflow(config, run_dir=args.run_dir, resume=args.resume)
            if args.render:
                render_run(output)
        print(output)
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        logging.error("%s", error, exc_info=args.verbose)
        return 1
    except KeyboardInterrupt:
        logging.error("Run interrupted; use --resume with its run directory")
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)
