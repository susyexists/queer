#!/usr/bin/env python
"""Compatibility wrapper for the historical V0=10 run."""

from __future__ import annotations

import sys

from run_v0 import main


if __name__ == "__main__":
    raise SystemExit(main(["--V0", "10", *sys.argv[1:]]))
