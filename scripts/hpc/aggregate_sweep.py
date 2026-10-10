#!/usr/bin/env python
"""Assemble a checksum-verified harmonic map from completed isolated Slurm members."""

import argparse
from pathlib import Path

from queer.workflows.finite_field import aggregate_member_runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dispatch", type=Path)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    print(aggregate_member_runs(args.dispatch, run_dir=args.run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
