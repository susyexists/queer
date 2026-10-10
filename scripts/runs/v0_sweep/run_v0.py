#!/usr/bin/env python
"""Run a V0 momentum-microscopy sweep from the organized data/results layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from queer.config import config_from_mapping
from queer.paths import data_path
from queer.workflows import run_workflow


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run momentum-microscopy surfaces for a V0 value.")
    parser.add_argument("--V0", "--v0", dest="v0", type=float, required=True, help="Inner potential in eV.")
    parser.add_argument("--material", default="vineet", help="Material folder under data/materials.")
    parser.add_argument("--material-path", type=Path, help="Explicit material input directory.")
    parser.add_argument("--output-dir", type=Path, help="Explicit output directory.")
    parser.add_argument("--hr", default="wannier90_hr.dat", help="Wannier HR filename.")
    parser.add_argument("--poscar", default="POSCAR", help="POSCAR filename.")
    parser.add_argument("--ef", type=float, default=5.8554, help="Fermi energy used by the model.")
    parser.add_argument("--photon-energy", type=float, default=21.2)
    parser.add_argument("--fermi-energy", type=float, default=0.2)
    parser.add_argument("--binding-min", type=float, default=0.0)
    parser.add_argument("--binding-max", type=float, default=3.1)
    parser.add_argument("--binding-step", type=float, default=0.1)
    parser.add_argument("--n-points", type=int, default=200)
    parser.add_argument("--factor", type=float, default=4.0)
    parser.add_argument("--energy-shift", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--num-core", type=int, default=None)
    parser.add_argument("--align-from", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--align-to", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--make-animation", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if (args.align_from is None) != (args.align_to is None):
        raise SystemExit("--align-from and --align-to must be provided together.")

    material_dir = args.material_path or data_path("materials", args.material)
    parameters = {
        "ef": args.ef,
        "interpolation": "standard",
        "photon_energy": args.photon_energy,
        "fermi_energy": args.fermi_energy,
        "binding_min": args.binding_min,
        "binding_max": args.binding_max,
        "binding_step": args.binding_step,
        "v0": args.v0,
        "n_points": args.n_points,
        "factor": args.factor,
        "energy_shift": args.energy_shift,
        "sigma": args.sigma,
    }
    if args.align_from is not None:
        parameters["align"] = [args.align_from, args.align_to]
    config = config_from_mapping(
        {
            "workflow": "momentum",
            "inputs": {"material_dir": str(material_dir), "hr": args.hr, "poscar": args.poscar},
            "parameters": parameters,
            "execution": {"workers": args.num_core or 1},
        }
    )
    run_dir = run_workflow(config, run_dir=args.output_dir)
    # Preserve historical render arguments, using the durable arrays as source.
    import json

    import pandas as pd

    from queer.momentum import save_surface_plots, write_animation

    manifest = json.loads((run_dir / "manifest.json").read_text())
    output_dir = run_dir / "figures"
    frames = []
    for record in manifest["cases"].values():
        with np.load(run_dir / record["path"], allow_pickle=False) as artifact:
            energy = float(artifact["binding_energy"][0])
            surface = pd.DataFrame(artifact["surface"], columns=["kx", "ky", "kz", "e", "I"])
            frames.extend(
                save_surface_plots(
                    {energy: surface},
                    output_dir,
                    v0=args.v0,
                    energy_shift=args.energy_shift,
                    rotate_from=args.align_from,
                    rotate_to=args.align_to,
                    transparent=args.transparent,
                )
            )
    if args.make_animation:
        write_animation(frames, output_dir)
    print(f"Saved run and {len(frames)} frames to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
