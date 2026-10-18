"""Write the PyATB shift-current calibration kit: synthetic scfouts, PyATB Inputs, a Slurm job and QUEER's side.

No PyATB shift-current output of a known model exists, so this kit makes one
possible. It writes, under ``--output``:

- ``soc.scfout``, ``soc_left.scfout`` (the same model on a left-handed lattice) and
  ``collinear.scfout``: the gapped models without inversion of
  ``tests/numerical/openmx_files.py`` (every level at least 0.28 eV, 0.81 eV for
  the collinear spin-up channel, from every other), and ``test.scfout``: the real
  OpenMX calibration scfout of ``pyatb-calib-60fc6b2a`` (``--calibration-scfout``;
  honeycomb with inversion, spin-orbit coupling, a nonorthogonal basis with position
  moments), whose shift current vanishes;
- one directory per case holding PyATB's ``Input`` (``scfout_route ../<model>.scfout``);
- ``shift-kit-job.sh``: the Slurm job that runs PyATB on every case (8 MPI ranks),
  after writing each case's ``inputs.sha256`` (``sha256sum ../<model>.scfout Input``);
- ``queer-side/<case>/``: QUEER's own run of each case (``queer_params.json``,
  ``queer_results.json``, its ``shift_current.dat``) and ``queer-side/summary.json``
  with each case's prediction: the comparison of QUEER's run against the table of
  ``tests/numerical/pyatb_shift_port.py`` (a loop-by-loop port of PyATB's reader and
  method 1, sharing no code with QUEER's reader or the recomputation) standing in for
  PyATB's;
- ``README.md``: the SHA-256 of every file, how to run the kit and the expected
  verdicts.

Run it with the repository's environment::

    python scripts/validation/generator_pyatb_shift_kit.py \\
      --output ~/.cache/queer-parity-data/pyatb-shift-kit --workers 8

``--no-queer`` writes the PyATB side only. Every comparison recomputes QUEER's
physical spectrum and PyATB's table from the scfout
(:mod:`queer.compat.generator.parity_shift`). The opt-in test
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
import time
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
# The real calibration scfout, copied into the kit.
CALIBRATION = "test.scfout"
CALIBRATION_SOURCE = Path.home() / ".cache" / "queer-parity-data" / "pyatb-calib-60fc6b2a" / "test.scfout"
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
# Case: (model, nspin, SHIFT_CURRENT block, self_convergence, expected verdict, expected classes, the step of the
# recomputation's attribution that explains the documented components (None without one), purpose).
CASES = {
    "a_clean_control": (
        "soc.scfout",
        4,
        SPINOR,
        False,
        "pass",
        ["agree"],
        None,
        "scissors 0, Gaussian, method 1, levels 0.28 eV apart, 5 eta of margin: every component meets QUEER's "
        "physical value directly (PyATB's Hartree constant moves it by less than 1e-4 of the peak)",
    ),
    "b_method0": (
        "soc.scfout",
        4,
        {**SPINOR, "method": 0},
        True,
        "not_comparable",
        [],
        None,
        "method 0 (every energy denominator regularized with 0.04 eV) is not transcribed: the comparison does not "
        "recompute it",
    ),
    "c_no_smearing": (
        "soc.scfout",
        4,
        {**SPINOR, "smearing_method": 0},
        False,
        "pass",
        ["agree_after_convention"],
        None,
        "smearing 0: PyATB's integer bins (no 1/domega) against QUEER's binned_uA_V2, the weight in (w, w + domega]",
    ),
    "d_scissors": (
        "soc.scfout",
        4,
        {**SPINOR, "scissors": 0.4},
        True,
        "pass_with_documented_differences",
        ["differs_documented"],
        "scissors",
        "scissors 0.4 eV: PyATB feeds the scissored levels to its derivative kernels, QUEER moves the transition "
        "energies only",
    ),
    "e_collinear": (
        "collinear.scfout",
        1,
        COLLINEAR,
        False,
        "pass",
        ["agree_after_convention"],
        None,
        "a collinear scfout: PyATB doubles its spin-up channel (legacy_single_channel); QUEER's physical value adds "
        "the spin-down channel, which the comparison recomputes too",
    ),
    "f_band_pair": (
        "soc.scfout",
        4,
        {**SPINOR, "omega": [1.5, 6.5], "n_occ": 2, "m_unocc": 3},
        False,
        "pass",
        ["agree"],
        None,
        "one band pair, part of it below the first photon energy (PyATB keeps it, C's int() rows)",
    ),
    "g_left_handed": (
        "soc_left.scfout",
        4,
        SPINOR,
        False,
        "pass",
        ["agree_after_convention"],
        None,
        "a left-handed lattice: PyATB's signed cell volume flips every component",
    ),
    "h_real_zero": (
        CALIBRATION,
        4,
        {**SPINOR, "occ_band": 3, "omega": [0.0, 12.0], "grid": [12, 12, 1]},
        False,
        "pass",
        ["agree"],
        None,
        "the real calibration scfout, whose shift current vanishes by inversion (1e-12 microA/V^2 against "
        "broadened magnitudes of order 10): both codes print rounding, which the recomputation reproduces; PyATB's "
        "own reader and Bloch sums of a nonorthogonal basis with position moments are checked here",
    ),
    "i_pyatb_default": (
        "soc.scfout",
        4,
        {**SPINOR, "eta": 0.01, "domega": 0.01},
        False,
        "pass",
        ["agree", "agree_after_convention"],
        None,
        "PyATB's default eta 0.01 on its default step domega 0.01: its Hartree constant moves narrow peaks by more "
        "than 1e-4 of the peak (agree_after_convention, recomputed)",
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


def stand_in_reference(
    directory: Path, case_input: Path, model: str, scfout: Path, table: np.ndarray
) -> Path:
    """A PyATB run directory printing ``table`` (the port's) for a case: its ``Input``, ``inputs.sha256``,
    ``Out/input.json``, ``Out/running.log`` and ``Out/Shift_Current/shift_current.dat``."""
    from queer.compat.generator.openmx_params import SHIFT_DEFAULTS
    from queer.compat.generator.pyatb_files import shift_current_text

    blocks = read_input(case_input)
    block = {**SHIFT_DEFAULTS, **blocks["SHIFT_CURRENT"]}
    shutil.copy(case_input, directory / "Input")
    (directory / "inputs.sha256").write_text(f"{sha256(scfout)}  ../{model}\n{sha256(case_input)}  Input\n")
    out = directory / "Out"
    (out / "Shift_Current").mkdir(parents=True)
    (out / "input.json").write_text(
        json.dumps({"INPUT_PARAMETERS": blocks["INPUT_PARAMETERS"], "SHIFT_CURRENT": block}, indent=1)
    )
    (out / "running.log").write_text("|                  Shift Current                     |\n")
    start, end = (float(value) for value in block["omega"])
    domega = float(block["domega"])
    omega = start + np.arange(int((end - start) / domega) + 1) * domega
    (out / "Shift_Current" / "shift_current.dat").write_text(shift_current_text(omega, table))
    return directory


def predict(output: Path, case: str, model: str, nspin: int, workspace: Path, workers: int) -> dict:
    """The comparison of QUEER's run against the loop port's table standing in for PyATB's."""
    from pyatb_shift_port import pyatb_shift

    from queer.compat.generator.openmx_params import SHIFT_DEFAULTS
    from queer.compat.generator.parity import compare

    block = {**SHIFT_DEFAULTS, **read_input(output / case / "Input")["SHIFT_CURRENT"]}
    scfout = output / model
    started = time.perf_counter()
    if int(block.get("method", 1)) == 1:
        _, table = pyatb_shift(scfout, block, nspin=nspin)
    else:
        start, end = (float(value) for value in block["omega"])
        table = np.zeros((int((end - start) / float(block["domega"])) + 1, len(SHIFT_COMPONENTS)))
    port_seconds = time.perf_counter() - started
    with tempfile.TemporaryDirectory() as scratch:
        reference = stand_in_reference(Path(scratch), output / case / "Input", model, scfout, table)
        started = time.perf_counter()
        report = compare("SHIFT_CURRENT", reference, workspace, scfout=scfout, workers=workers)
        seconds = time.perf_counter() - started
    classes = sorted({item["class"] for item in report["quantities"] if item["name"].startswith("sigma_")})
    attribution = (report["semantics"] or {}).get("rules", {}).get("attribution") or {}
    causes = sorted(
        {
            max(values["steps_uA_V2"], key=values["steps_uA_V2"].get)
            for values in attribution.values()
            if values["class"] == "differs_documented"
        }
    )
    recomputation = (report.get("tolerance_basis") or {}).get("recomputation") or {}
    return {
        "verdict": report["verdict"],
        "classes": classes,
        "documented_causes": causes,
        "failed_preconditions": [item["name"] for item in report["preconditions"] if not item["ok"]],
        "comparison_seconds": round(seconds, 2),
        "port_seconds": round(port_seconds, 2),
        "recompute_seconds": None if not recomputation else round(recomputation.get("seconds", 0.0), 2),
        "recompute_work": recomputation.get("work"),
        "flagged_kpoints": recomputation.get("flagged_kpoints"),
    }


def queer_side(
    output: Path, case: str, model: str, nspin: int, self_convergence: bool, workers: int, prediction: bool
) -> dict:
    target = output / "queer-side" / case
    predicted = None
    with tempfile.TemporaryDirectory() as scratch:
        workspace = Path(scratch)
        shutil.copy(output / model, workspace / "scf.scfout")
        params = params_from_pyatb_input(
            read_input(output / case / "Input"), workers=workers, self_convergence=self_convergence or None
        )
        (workspace / "queer_params.json").write_text(json.dumps(params, indent=1))
        started = time.perf_counter()
        code = run(None, workspace / "queer_params.json", workspace, log=lambda line: None)
        seconds = time.perf_counter() - started
        document = json.loads((workspace / "queer_results.json").read_text())
        if code == 0 and prediction:
            predicted = predict(output, case, model, nspin, workspace, workers)
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
    peak = int(np.argmax(np.max(np.abs(physical), axis=1)))
    diagnostics = block["diagnostics"]
    convergence = diagnostics.get("self_convergence") or {}
    compared = convergence.get("legacy_single_channel") or convergence.get("physical") or {}
    return {
        "status": "ok",
        "max_abs_sigma_uA_V2": float(np.max(np.abs(physical))),
        "peak_omega_eV": block["omega_eV"][peak],
        "file_table": ("legacy_single_channel." if block["legacy_single_channel"] else "")
        + ("binned_uA_V2" if block["binned_uA_V2"] is not None else "sigma_uA_V2"),
        "min_pair_gap_eV": diagnostics["min_pair_gap_eV"],
        "exact_degenerate_kpoints": diagnostics["exact_degenerate_kpoints"],
        "kpoints_at_pyatb_threshold": diagnostics["kpoints_at_pyatb_threshold"],
        "transitions": diagnostics["transitions"]["count"],
        "queer_seconds": round(seconds, 2),
        "queer_source_sha256": block["build"]["queer_source_sha256"],
        "self_convergence_ratio": compared.get("ratio"),
        "scfout_sha256": document["model"]["scfout_sha256"],
        "queer_version": document["provenance"]["queer_version"],
        "prediction": predicted,
    }


def readme(output: Path, summary: dict) -> str:
    lines = [
        "# PyATB shift-current calibration kit",
        "",
        "Written by `scripts/validation/generator_pyatb_shift_kit.py` of the QUEER checkout. The synthetic scfouts are",
        "the models of `tests/numerical/openmx_files.py` (one atom, two orbitals, identity overlap, no position",
        "moments). PyATB 60fc6b2a's own reader (`pyatb/io/openmx_read_scfout.py`, loaded from its source with its",
        "multiXR container stubbed) parses them: nspin 4 (SpinP_switch 3) for `soc*.scfout`, nspin 1 (SpinP_switch 1)",
        "for `collinear.scfout`, a negative det(tv) for `soc_left.scfout`. `test.scfout` is the real OpenMX",
        "calibration scfout of `pyatb-calib-60fc6b2a` (honeycomb with inversion, spin-orbit coupling, a nonorthogonal",
        "basis with position moments, 30 spinor bands), whose shift current vanishes.",
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
        "The expected verdict is what the comparison gives with the loop port of PyATB's reader and method 1",
        "(`tests/numerical/pyatb_shift_port.py`, sharing no code with QUEER's reader or the recomputation) standing in",
        "for PyATB (`prediction` in `queer-side/summary.json`).",
        "",
        "| Case | scfout | Expected verdict | Expected classes | Documented by | QUEER max abs sigma (uA/V^2) | Purpose |",
        "|---|---|---|---|---|---|---|",
    ]
    for case, (model, _, _, _, verdict, kinds, cause, purpose) in CASES.items():
        facts = summary.get(case, {})
        value = facts.get("max_abs_sigma_uA_V2")
        shown = "" if value is None else f"{value:.6g}"
        lines.append(
            f"| `{case}` | `{model}` | `{verdict}` | {', '.join(kinds) or '-'} | {cause or '-'} | {shown} | {purpose} |"
        )
    lines += [
        "",
        "`queer-side/<case>/queer_results.json` holds QUEER's run of each case: `shift_current.sigma_uA_V2` is QUEER's",
        "physical value, the value judged, and its diagnostics are plain facts of the run. A comparison reads none of",
        "them: it parses the scfout itself (`queer.compat.generator.shift_scfout`, no code of QUEER's reader) and",
        "recomputes QUEER's physical spectrum and the table PyATB prints from it, each in four orbital bases",
        "(`queer.compat.generator.pyatb_shift_integrand`); it judges first which entries the scfout determines, then",
        "checks QUEER's tables against the first at those entries (a mismatch fails) and PyATB's printed table",
        "against the second (a mismatch is `not_comparable`), and classes each",
        "component, documenting a difference with the recomputation's own attribution (PyATB's SI constants, Hartree",
        "constant, window, rows and truncation, scissored kernels, method-1 integrand). `queer-side/summary.json` lists",
        "the headline numbers, the prediction and the build (`queer_source_sha256`) the runs come from.",
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
        "`~/.cache/queer-parity-data/pyatb-shift-kit` and `QUEER_PARITY_REPORT_DIR` receives every report. Each case",
        "is run by QUEER afresh and compared with its scfout (`scfout=`): it must meet its expected verdict and classes,",
        "and wherever the comparison recomputes PyATB's table (every case but `b_method0`), PyATB's printed table must",
        "equal the recomputation's PyATB-literal table to its print precision and rounding: the check of the",
        "transcription, PyATB's reader included, against PyATB itself.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-queer", action="store_true")
    parser.add_argument(
        "--no-prediction", action="store_true", help="skip the comparisons against the loop port"
    )
    parser.add_argument("--calibration-scfout", type=Path, default=CALIBRATION_SOURCE)
    options = parser.parse_args(argv)
    output = options.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    for name, (writer, arguments) in MODELS.items():
        writer(output / name, **arguments)
    source = options.calibration_scfout.expanduser()
    if not source.is_file():
        parser.error(f"no calibration scfout at {source} (--calibration-scfout)")
    if not (output / CALIBRATION).is_file() or sha256(output / CALIBRATION) != sha256(source):
        shutil.copy(source, output / CALIBRATION)
    for case, (model, nspin, block, *_) in CASES.items():
        (output / case).mkdir(exist_ok=True)
        (output / case / "Input").write_text(input_text(model, nspin, block, output / model))
    (output / "shift-kit-job.sh").write_text(JOB)
    summary: dict = {}
    if not options.no_queer:
        for case, (model, nspin, _, self_convergence, *_) in CASES.items():
            summary[case] = queer_side(
                output, case, model, nspin, self_convergence, options.workers, not options.no_prediction
            )
            print(case, json.dumps(summary[case]), flush=True)
        (output / "queer-side" / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    elif (output / "queer-side" / "summary.json").exists():
        summary = json.loads((output / "queer-side" / "summary.json").read_text())
    (output / "README.md").write_text(readme(output, summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
