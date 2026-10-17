"""Write the PyATB shift-current calibration kit: synthetic scfouts, PyATB Inputs, a Slurm job and QUEER's side.

No PyATB shift-current output of a known model exists, so this kit makes one
possible. It writes, under ``--output``:

- ``soc.scfout``, ``soc_left.scfout`` (the same model on a left-handed lattice) and
  ``collinear.scfout``: the gapped models without inversion of
  ``tests/numerical/openmx_files.py`` (every level at least 0.28 eV, 0.81 eV for
  the collinear spin-up channel, from every other);
- one directory per case holding PyATB's ``Input`` (``scfout_route ../<model>.scfout``);
- ``shift-kit-job.sh``: the Slurm job that runs PyATB on every case (8 MPI ranks),
  after writing each case's ``inputs.sha256`` (``sha256sum ../<model>.scfout Input``);
- ``queer-side/<case>/``: QUEER's own run of each case (``queer_params.json``,
  ``queer_results.json``, its ``shift_current.dat``) and ``queer-side/summary.json``;
  ``diagnostics.pyatb_scheme.sigma_uA_V2`` of each results file is the table PyATB
  should print from QUEER's transitions;
- ``README.md``: the SHA-256 of every file, how to run the kit and the expected
  classes.

Run it with the repository's environment::

    python scripts/validation/generator_pyatb_shift_kit.py \\
      --output ~/.cache/queer-parity-data/pyatb-shift-kit --workers 8

``--no-queer`` writes the PyATB side only. The opt-in test
``tests/integration/test_generator_pyatb_shift_calibration.py`` judges PyATB's
outputs, copied back into the case directories, against fresh QUEER runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "numerical"))

from openmx_files import shift_collinear_scfout, shift_spinor_scfout  # noqa: E402

from queer.compat.generator.openmx_params import params_from_pyatb_input  # noqa: E402
from queer.compat.generator.pyatb_files import SHIFT_COMPONENTS, read_input  # noqa: E402
from queer.compat.generator.runner import run  # noqa: E402

BOHR = 0.529177210903
MODELS = {
    "soc.scfout": (shift_spinor_scfout, {}),
    "soc_left.scfout": (shift_spinor_scfout, {"lattice": np.diag([3.0, 3.2, -3.5])}),
    "collinear.scfout": (shift_collinear_scfout, {}),
}
SPINOR = {
    "occ_band": 2,
    "omega": [0.3, 6.5],
    "domega": 0.02,
    "smearing_method": 1,
    "eta": 0.1,
    "grid": [16, 16, 16],
    "method": 1,
    "n_occ": -1,
    "m_unocc": -1,
    "scissors": 0.0,
}
COLLINEAR = {**SPINOR, "occ_band": 1, "omega": [0.2, 4.5], "grid": [48, 48, 1]}
# Case: (model, nspin, SHIFT_CURRENT block, self_convergence, expected verdict, expected classes, purpose).
CASES = {
    "a_clean_control": (
        "soc.scfout",
        4,
        SPINOR,
        False,
        "pass",
        ["agree"],
        "the clean control: scissors 0, Gaussian, method 1, levels 0.28 eV apart, 5 eta of margin; every "
        "component must meet QUEER's physical value directly",
    ),
    "b_method0": (
        "soc.scfout",
        4,
        {**SPINOR, "method": 0},
        True,
        "pass_with_documented_differences",
        ["agree", "agree_after_convention", "differs_documented"],
        "method 0: PyATB regularizes every energy denominator with 0.04 eV (unbounded cause)",
    ),
    "c_no_smearing": (
        "soc.scfout",
        4,
        {**SPINOR, "smearing_method": 0},
        False,
        "pass_with_documented_differences",
        ["agree_after_convention", "differs_documented"],
        "smearing 0: PyATB's integer bins without 1/domega (x domega, half a bin); its Hartree scale moves a "
        "few transitions across bin edges (bounded by bin_moves, documented)",
    ),
    "d_scissors": (
        "soc.scfout",
        4,
        {**SPINOR, "scissors": 0.4},
        True,
        "pass_with_documented_differences",
        ["agree", "agree_after_convention", "differs_documented"],
        "scissors 0.4 eV: PyATB shifts eigenvalues inside its derivative kernels (unbounded cause)",
    ),
    "e_collinear": (
        "collinear.scfout",
        1,
        COLLINEAR,
        False,
        "pass",
        ["agree_after_convention"],
        "a collinear scfout: PyATB doubles its spin-up channel (legacy_single_channel)",
    ),
    "f_band_pair": (
        "soc.scfout",
        4,
        {**SPINOR, "omega": [1.5, 6.5], "n_occ": 2, "m_unocc": 3},
        False,
        "pass",
        ["agree"],
        "one band pair, part of it below the first photon energy (PyATB keeps it, C's int() bins)",
    ),
    "g_left_handed": (
        "soc_left.scfout",
        4,
        SPINOR,
        False,
        "pass",
        ["agree_after_convention"],
        "a left-handed lattice: PyATB's signed cell volume flips every component",
    ),
}
JOB = """#!/bin/bash -l
#SBATCH -p bansil
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH -J pyatb-shift-kit
#SBATCH -o /home/s.sevim/pyatb-shift-kit/shift-kit-%j.log
# PyATB 60fc6b2a SHIFT_CURRENT on the synthetic scfouts of the QUEER shift-current kit.
set -o pipefail
module purge
module load python/3.13.5 OpenMPI/4.1.6 intel/mkl-2025.0
export LD_LIBRARY_PATH=/shared/EL9/explorer/python/3.13.5/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1
PYATB=/home/s.sevim/pyatb-env/bin/pyatb
cd /home/s.sevim/pyatb-shift-kit
hostname; date -u
for d in */; do
  d=${d%/}
  test -f "$d/Input" || continue
  route=$(awk '$1 == "scfout_route" {print $2}' "$d/Input")
  ( cd "$d" && rm -rf Out && sha256sum "$route" Input > inputs.sha256 && start=$(date +%s) \\
      && mpirun -n 8 "$PYATB" > pyatb.out 2>&1; echo "exit=$? seconds=$(( $(date +%s) - start ))" > status )
done
echo "== results"
for d in */; do
  d=${d%/}
  test -f "$d/Input" || continue
  echo "## $d $(cat "$d/status" 2>/dev/null)"
  head -3 "$d/Out/Shift_Current/shift_current.dat" 2>/dev/null | cut -c1-100
  tail -2 "$d/pyatb.out" 2>/dev/null | cut -c1-200
done
echo "SHIFT KIT DONE"
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def block_text(name: str, values: dict) -> str:
    lines = [name, "{"]
    for key, value in values.items():
        text = " ".join(str(v) for v in value) if isinstance(value, list) else str(value)
        lines.append(f"    {key:<32}{text}")
    return "\n".join([*lines, "}", ""])


def lattice_block(scfout: Path) -> str:
    from queer import OpenMXModel

    rows = np.asarray(OpenMXModel.from_scfout(scfout).lattice, float) / BOHR
    vectors = "\n".join("     " + "  ".join(f"{value:.12f}" for value in row) for row in rows)
    return (
        "LATTICE\n{\n    lattice_constant                1.0\n    lattice_constant_unit           Bohr\n"
        f"    lattice_vector\n{vectors}\n}}\n"
    )


def input_text(model: str, nspin: int, block: dict, scfout: Path) -> str:
    inputs = {"nspin": nspin, "package": "OPENMX", "scfout_route": f"../{model}", "max_kpoint_num": 8000}
    return "\n".join(
        [block_text("INPUT_PARAMETERS", inputs), lattice_block(scfout), block_text("SHIFT_CURRENT", block)]
    )


def queer_side(output: Path, case: str, model: str, self_convergence: bool, workers: int) -> dict:
    target = output / "queer-side" / case
    with tempfile.TemporaryDirectory() as scratch:
        workspace = Path(scratch)
        shutil.copy(output / model, workspace / "scf.scfout")
        params = params_from_pyatb_input(
            read_input(output / case / "Input"), workers=workers, self_convergence=self_convergence or None
        )
        (workspace / "queer_params.json").write_text(json.dumps(params, indent=1))
        code = run(None, workspace / "queer_params.json", workspace, log=lambda line: None)
        document = json.loads((workspace / "queer_results.json").read_text())
        if target.exists():
            shutil.rmtree(target)
        (target / "Out" / "Shift_Current").mkdir(parents=True)
        for name in ("queer_params.json", "queer_results.json", "Out/Shift_Current/shift_current.dat"):
            if (workspace / name).exists():
                shutil.copy(workspace / name, target / name)
    if code != 0:
        return {"status": document.get("status"), "error": document.get("error")}
    block = document["shift_current"]
    physical = np.column_stack([block["sigma_uA_V2"][name] for name in SHIFT_COMPONENTS])
    variant = block["legacy_single_channel"] or block
    matched = variant["binned_uA_V2"] if block["binned_uA_V2"] is not None else variant["sigma_uA_V2"]
    convention = block["convention"]["export_sign"] * np.column_stack(
        [matched[name] for name in SHIFT_COMPONENTS]
    )
    scheme = block["diagnostics"]["pyatb_scheme"].get("sigma_uA_V2")
    printed = None if scheme is None else np.column_stack([scheme[name] for name in SHIFT_COMPONENTS])
    peak = int(np.argmax(np.max(np.abs(physical), axis=1)))
    convergence = block["diagnostics"].get("self_convergence") or {}
    compared = convergence.get("legacy_single_channel") or convergence.get("physical") or {}
    return {
        "status": "ok",
        "max_abs_sigma_uA_V2": float(np.max(np.abs(physical))),
        "peak_omega_eV": block["omega_eV"][peak],
        "pyatb_scheme_max_abs_uA_V2": None if printed is None else float(np.max(np.abs(printed))),
        "pyatb_scheme_minus_convention_uA_V2": None
        if printed is None
        else float(np.max(np.abs(printed - convention))),
        "convention_table": ("legacy_single_channel." if block["legacy_single_channel"] else "")
        + ("binned_uA_V2" if block["binned_uA_V2"] is not None else "sigma_uA_V2"),
        "min_pair_gap_eV": block["diagnostics"]["min_pair_gap_eV"],
        "exact_degenerate_kpoints": block["diagnostics"]["exact_degenerate_kpoints"],
        "regularised_kpoints": block["diagnostics"]["regularisation"]["flagged_kpoints"],
        "transitions": block["diagnostics"]["transitions"]["count"],
        "clean_control_eligible": block["diagnostics"]["clean_control"]["eligible"],
        "queer_source_sha256": block["build"]["queer_source_sha256"],
        "self_convergence_ratio": compared.get("ratio"),
        "scfout_sha256": document["model"]["scfout_sha256"],
        "queer_version": document["provenance"]["queer_version"],
    }


def readme(output: Path, summary: dict) -> str:
    lines = [
        "# PyATB shift-current calibration kit",
        "",
        "Written by `scripts/validation/generator_pyatb_shift_kit.py` of the QUEER checkout. The scfouts are the",
        "synthetic models of `tests/numerical/openmx_files.py` (one atom, two orbitals, identity overlap, no",
        "position moments). PyATB 60fc6b2a's own reader (`pyatb/io/openmx_read_scfout.py`, loaded from its source",
        "with its multiXR container stubbed) parses them: nspin 4 (SpinP_switch 3) for `soc*.scfout`, nspin 1",
        "(SpinP_switch 1) for `collinear.scfout`, a negative det(tv) for `soc_left.scfout`, and its eigenvalues",
        "equal QUEER's times 27.211324570273 / HARTREE_EV (its Hartree constant) to 1e-12 eV.",
        "",
        "## Files (SHA-256)",
        "",
        "| File | SHA-256 | Bytes |",
        "|---|---|---|",
    ]
    summary_file = output / "queer-side" / "summary.json"
    for path in (
        sorted(output.glob("*.scfout"))
        + sorted(output.glob("*/Input"))
        + [output / "shift-kit-job.sh"]
        + ([summary_file] if summary_file.exists() else [])
    ):
        lines.append(f"| `{path.relative_to(output)}` | `{sha256(path)}` | {path.stat().st_size} |")
    builds = sorted(
        {facts.get("queer_source_sha256") for facts in summary.values() if facts.get("queer_source_sha256")}
    )
    if builds:
        lines += [
            "",
            f"QUEER's side was run by the build with `queer_source_sha256` {', '.join(builds)} (every `.py` file "
            "of the queer package).",
        ]
    lines += [
        "",
        "## Cases and the expected comparison",
        "",
        "| Case | scfout | Expected verdict | Expected classes | QUEER max abs sigma (uA/V^2) | Purpose |",
        "|---|---|---|---|---|---|",
    ]
    for case, (model, _, _, _, verdict, kinds, purpose) in CASES.items():
        facts = summary.get(case, {})
        value = facts.get("max_abs_sigma_uA_V2")
        shown = "" if value is None else f"{value:.6g}"
        lines.append(f"| `{case}` | `{model}` | `{verdict}` | {', '.join(kinds)} | {shown} | {purpose} |")
    lines += [
        "",
        "`queer-side/<case>/queer_results.json` holds QUEER's run of each case: `shift_current.sigma_uA_V2` is",
        "QUEER's physical value, the estimator judged, and `shift_current.diagnostics.pyatb_scheme.sigma_uA_V2`",
        "the table PyATB should print (its Hartree scale, window rule, int() bins, Gaussian truncation,",
        "constants, spin factor and signed volume applied to QUEER's transitions), a diagnostic only.",
        "`queer-side/summary.json` lists the headline numbers and the build (`queer_source_sha256`) the runs",
        "come from.",
        "",
        "## Running PyATB (Explorer)",
        "",
        "1. Copy this directory to `/home/s.sevim/pyatb-shift-kit` (the job's working directory).",
        "2. `sbatch shift-kit-job.sh` (8 MPI ranks; each case takes seconds to a few minutes).",
        "3. Copy the case directories back here (each gains `inputs.sha256`, `status`, `pyatb.out` and `Out/`).",
        "",
        "## Judging PyATB's outputs",
        "",
        "```bash",
        "QUEER_PYATB_SHIFT_CALIBRATION=1 QUEER_REAL_REPLAY_WORKERS=8 \\",
        "  python -m pytest -q tests/integration/test_generator_pyatb_shift_calibration.py",
        "```",
        "",
        "`QUEER_PYATB_SHIFT_CALIBRATION_DIR` overrides the default data directory",
        "`~/.cache/queer-parity-data/pyatb-shift-kit` and `QUEER_PARITY_REPORT_DIR` receives every report. The",
        "clean control (`a_clean_control`) is judged first: every component must meet QUEER's physical value",
        "directly (to 1e-4 of the peak beside the derived effect of PyATB's Hartree scale), and the emulation",
        "must reproduce PyATB's table to its print precision (1e-6 of each value). Its report, for the same",
        "build (queer_version, RECORD, build_sha, and the digest of every .py file of the queer package that ran",
        "the case and that compares it), is applied to every other case: a difference is documented only after",
        "it passed.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-queer", action="store_true")
    options = parser.parse_args(argv)
    output = options.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    for name, (writer, arguments) in MODELS.items():
        writer(output / name, **arguments)
    for case, (model, nspin, block, *_) in CASES.items():
        (output / case).mkdir(exist_ok=True)
        (output / case / "Input").write_text(input_text(model, nspin, block, output / model))
    (output / "shift-kit-job.sh").write_text(JOB)
    summary: dict = {}
    if not options.no_queer:
        for case, (model, _, _, self_convergence, *_) in CASES.items():
            summary[case] = queer_side(output, case, model, self_convergence, options.workers)
            print(case, json.dumps(summary[case]), flush=True)
        (output / "queer-side" / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    elif (output / "queer-side" / "summary.json").exists():
        summary = json.loads((output / "queer-side" / "summary.json").read_text())
    (output / "README.md").write_text(readme(output, summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
