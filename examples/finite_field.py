"""Run native finite-field geometry and signed Hall calculations.

Usage: python examples/finite_field.py --config configs/examples/finite_field_crossings.toml
"""

from argparse import ArgumentParser
from pathlib import Path

from queer.config import load_config
from queer.workflows import render_run, run_workflow


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/examples/finite_field_crossings.toml",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    run = run_workflow(load_config(args.config), run_dir=args.run_dir, resume=args.resume)
    if args.render:
        render_run(run)
    print(run)


if __name__ == "__main__":
    main()
