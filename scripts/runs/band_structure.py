#!/usr/bin/env python
"""Calculate and plot an electron band structure along a named k path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists() or (candidate / "setup.py").exists():
            return candidate
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _project_root()
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib.pyplot as plt

import queer
from queer.paths import data_path, results_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", default="ag_primitive", help="Material folder under data/materials.")
    parser.add_argument("--material-path", type=Path, help="Explicit directory containing POSCAR and HR file.")
    parser.add_argument("--hr", default="wannier90_hr.dat", help="Wannier Hamiltonian filename.")
    parser.add_argument("--poscar", default="POSCAR", help="POSCAR filename.")
    parser.add_argument("--ef", type=float, default=8.310342, help="Fermi energy in eV.")
    parser.add_argument("--kpath", default="GAMMA-X-W-K-GAMMA-L", help="Named SeekPath route, e.g. GAMMA-X-W-K-GAMMA-L.")
    parser.add_argument("--k-points", type=int, default=200, help="Number of interpolated points along the path.")
    parser.add_argument("--ylim", nargs=2, type=float, default=(-10.0, 10.0), metavar=("YMIN", "YMAX"))
    parser.add_argument("--num-core", type=int, default=1, help="Number of joblib workers.")
    parser.add_argument("--output", type=Path, help="Output image path.")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively after saving.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    material_dir = args.material_path or data_path("materials", args.material)
    output = args.output or results_path("bands", f"{args.material}_{args.kpath.replace('-', '_')}.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    model = queer.model(
        path=material_dir,
        hr=args.hr,
        ef=args.ef,
        poscar=args.poscar,
        num_core=args.num_core,
    )

    model.plot_band_path(args.kpath, n_points=args.k_points, ylim=tuple(args.ylim), save=output)
    if args.show:
        plt.show()
    else:
        plt.close()

    print(f"Saved band structure to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
