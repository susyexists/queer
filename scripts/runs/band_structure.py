#!/usr/bin/env python
"""Calculate and plot an electron band structure along a named k path."""

from __future__ import annotations

import argparse
from pathlib import Path

from queer.config import config_from_mapping
from queer.paths import data_path
from queer.workflows import run_workflow


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", default="ag_primitive", help="Material folder under data/materials.")
    parser.add_argument(
        "--material-path", type=Path, help="Explicit directory containing POSCAR and HR file."
    )
    parser.add_argument("--hr", default="wannier90_hr.dat", help="Wannier Hamiltonian filename.")
    parser.add_argument("--poscar", default="POSCAR", help="POSCAR filename.")
    parser.add_argument("--ef", type=float, default=8.310342, help="Fermi energy in eV.")
    parser.add_argument(
        "--kpath", default="GAMMA-X-W-K-GAMMA-L", help="Named SeekPath route, e.g. GAMMA-X-W-K-GAMMA-L."
    )
    parser.add_argument(
        "--k-points", type=int, default=200, help="Number of interpolated points along the path."
    )
    parser.add_argument("--ylim", nargs=2, type=float, default=(-10.0, 10.0), metavar=("YMIN", "YMAX"))
    parser.add_argument("--num-core", type=int, default=1, help="Number of joblib workers.")
    parser.add_argument("--output", type=Path, help="Output image path.")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively after saving.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import matplotlib.pyplot as plt
    import numpy as np

    material_dir = args.material_path or data_path("materials", args.material)
    config = config_from_mapping(
        {
            "workflow": "bands",
            "inputs": {"material_dir": str(material_dir), "hr": args.hr, "poscar": args.poscar},
            "parameters": {
                "ef": args.ef,
                "interpolation": "standard",
                "kpath": args.kpath,
                "n_points": args.k_points,
            },
            "execution": {"workers": args.num_core},
        }
    )
    run_dir = run_workflow(config)
    output = args.output or run_dir / "figures" / "bands.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(run_dir / "arrays" / "bands.npz", allow_pickle=False) as artifact:
        fig, ax = plt.subplots()
        ax.plot(artifact["energies"].T, color="blue")
        ax.set_xticks(artifact["symmetry"], artifact["labels"])
        ax.set(ylim=tuple(args.ylim), ylabel="Energy (eV)", xlabel="k-point index")
        ax.axhline(0.0, linestyle="--", color="red")
        fig.savefig(output, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)
    print(f"Saved {output}; numerical run {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
