"""Offline native replay of the recovered OMXsigmaxy sheet-Hall benchmark.

The reference input format is four lines: relative energy bounds/count, grid,
unit flag, and dimension flag. Only the explicit two-dimensional e²/h case is
supported. No third-party reader or calculator is imported or executed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from queer import OpenMXModel
from queer.config import config_from_mapping
from queer.io.artifacts import atomic_json, atomic_npz, digest_file, environment_versions, load_manifest
from queer.io.openmx import HARTREE_EV
from queer.observables._tb import E_CHARGE, HBAR
from queer.workflows import run_workflow

# Original OMXsigmaxy recipe converts its eV energy bounds to Hartree with this
# declared constant. QUEER's operators retain their native physical constants.
OMX_HARTREE_EV = 27.2113845


def replay(args):
    started = time.monotonic()
    actual = digest_file(args.scfout)
    if actual != args.expected_scfout_sha256:
        raise ValueError("SCFOUT differs from the verified historical input SHA256")
    lines = [line.split() for line in args.recipe.read_text().splitlines() if line.strip()]
    if len(lines) != 4 or list(map(int, lines[2])) != [1] or list(map(int, lines[3])) != [0]:
        raise ValueError("Only the original four-line 2D e²/h OMXsigmaxy recipe is supported")
    if len(lines[0]) != 3 or len(lines[1]) != 3:
        raise ValueError("Invalid original OMXsigmaxy energy/grid recipe")
    low, high = map(float, lines[0][:2])
    count = int(lines[0][2])
    grid = list(map(int, lines[1]))
    if not np.isfinite([low, high]).all() or high <= low or count < 1 or min(grid) < 1 or grid[2] != 1:
        raise ValueError("Invalid 2D reference energy scan/grid")
    energies = low + np.arange(count) * (high - low) / count
    reference = np.loadtxt(args.reference, ndmin=2)
    if reference.shape != (count, 4) or not np.isfinite(reference).all():
        raise ValueError("Expected all original energy rows and three Hall components")
    if not np.allclose(reference[:, 0], energies, atol=5.01e-7, rtol=0):
        raise ValueError("Reference rows differ from the declared energy recipe")
    if not np.isfinite([args.atol, args.rtol]).all() or min(args.atol, args.rtol) < 0:
        raise ValueError("Comparison tolerances must be finite and nonnegative")
    model = OpenMXModel.from_scfout(args.scfout)
    config = config_from_mapping(
        {
            "workflow": "ahc",
            "inputs": {
                "material_dir": str(args.scfout.resolve().parent),
                "model": "openmx",
                "scfout": args.scfout.name,
            },
            "parameters": {
                "engine": "fhs",
                "dimension": 2,
                "plane": [0, 1],
                "grid": grid,
                "chemical_potentials": (model.fermi_energy + energies * HARTREE_EV / OMX_HARTREE_EV).tolist(),
            },
            "execution": {"workers": args.workers, "native_threads": 1, "batch_size": 64, "memory_mb": 512},
            "output": {"results_dir": str(args.output_dir.resolve().parent)},
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = run_workflow(config, run_dir=args.output_dir / "native-run", resume=args.resume)
    manifest = load_manifest(run)
    with np.load(run / "arrays/result.npz", allow_pickle=False) as result:
        native = result["ahc"][:, 0, 1] / (E_CHARGE**2 / (2 * np.pi * HBAR))
    delta = native - reference[:, 3]
    report = {
        "status": "completed",
        "third_party": "OMXsigmaxy",
        "quantity": "ahc",
        "units": "e2/h",
        "original_input": {
            name: {"path": str(path.resolve()), "sha256": digest_file(path)}
            for name, path in (
                ("recipe", args.recipe),
                ("reference", args.reference),
                ("scfout", args.scfout),
            )
        },
        "native_run": str(run.resolve()),
        "versions": environment_versions(),
        "energy_conversion": {
            "original_Hartree_eV": OMX_HARTREE_EV,
            "native_Hartree_eV": HARTREE_EV,
            "reference": "relative to original SCFOUT ChemP",
        },
        "comparison": {
            "compared_values": count,
            "components": ["xy"],
            "atol": args.atol,
            "rtol": args.rtol,
            "max_absolute_error": float(np.abs(delta).max()),
            "rms_error": float(np.sqrt(np.mean(delta**2))),
            "passed": bool(np.all(np.abs(delta) <= args.atol + args.rtol * np.abs(reference[:, 3]))),
        },
        "limitations": [
            "native midpoint first-order basis links differ from original atom-centered position-moment links",
            "fixed-mesh agreement does not establish continuum convergence",
        ],
        "diagnostics": manifest["cases"]["result"]["metadata"],
        "wall_seconds": time.monotonic() - started,
    }
    atomic_npz(
        args.output_dir / "comparison.npz",
        {"relative_energy_eV": energies, "native": native, "reference": reference[:, 3], "difference": delta},
    )
    atomic_json(args.output_dir / "replay.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scfout", type=Path, required=True)
    parser.add_argument("--expected-scfout-sha256", required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.02)
    args = parser.parse_args(argv)
    report = replay(args)
    print(json.dumps(report["comparison"], indent=2))
    return 0 if report["comparison"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
