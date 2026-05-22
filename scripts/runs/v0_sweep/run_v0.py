#!/usr/bin/env python
"""Run a V0 momentum-microscopy sweep from the organized data/results layout."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import queer
from queer.momentum import binding_energy_surfaces, format_time, save_surface_plots, write_animation
from queer.paths import data_path, results_path


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

    start_total = time.time()
    material_dir = args.material_path or data_path("materials", args.material)
    output_dir = args.output_dir or results_path("v0_sweep", f"V0_{args.v0:g}")
    align = [args.align_from, args.align_to] if args.align_from is not None else None

    start_model = time.time()
    model = queer.model(
        path=material_dir,
        hr=args.hr,
        ef=args.ef,
        poscar=args.poscar,
        num_core=args.num_core or False,
    )
    print(f"Model loading took {format_time(time.time() - start_model)}")

    start_energy = time.time()
    surfaces = binding_energy_surfaces(
        model=model,
        g_vec=model.g_vec,
        photon_energy=args.photon_energy,
        fermi_energy=args.fermi_energy,
        binding_range=[args.binding_min, args.binding_max],
        binding_step=args.binding_step,
        v0=args.v0,
        n_points=args.n_points,
        factor=args.factor,
        align=align,
        energy_shift=args.energy_shift,
        sigma=args.sigma,
    )
    print(f"Energy and intensity processing took {format_time(time.time() - start_energy)}")

    start_plot = time.time()
    frames = save_surface_plots(
        surfaces,
        output_dir=output_dir,
        v0=args.v0,
        energy_shift=args.energy_shift,
        rotate_from=args.align_from,
        rotate_to=args.align_to,
        transparent=args.transparent,
    )
    print(f"Plotting took {format_time(time.time() - start_plot)}")

    if args.make_animation:
        start_video = time.time()
        write_animation(frames, output_dir)
        print(f"Animation creation took {format_time(time.time() - start_video)}")

    print(f"Saved {len(frames)} frames to {output_dir}")
    print(f"Total execution time: {format_time(time.time() - start_total)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
