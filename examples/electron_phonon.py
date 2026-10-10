"""Run the maintained epw example; replace the synthetic configuration for real data."""

import argparse
from pathlib import Path

from queer.config import load_config
from queer.workflows import render_run, run_workflow


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/examples/epw.toml"
    )
    parser.add_argument("--results-dir", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config, overrides={"results_dir": args.results_dir})
    run_dir = run_workflow(config)
    print(render_run(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
