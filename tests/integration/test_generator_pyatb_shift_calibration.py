"""QUEER's shift current against PyATB's runs of the shift-current calibration kit.

Opt-in: ``QUEER_PYATB_SHIFT_CALIBRATION=1``. ``QUEER_PYATB_SHIFT_CALIBRATION_DIR``
names the kit (default ``~/.cache/queer-parity-data/pyatb-shift-kit``), written by
``scripts/validation/generator_pyatb_shift_kit.py``: the scfouts and one directory
per case holding PyATB's ``Input`` and, once PyATB has run it, its
``inputs.sha256``, ``Out/input.json``, ``Out/running.log`` and
``Out/Shift_Current/shift_current.dat`` (the layout of the AHC calibration
``pyatb-calib-60fc6b2a``). Each case's ``queer_params.json`` is built from its
``Input`` as the Generator builds it; QUEER runs it afresh and the comparison
recomputes both spectra from the case's scfout. Each case must meet its expected
verdict and classes, and wherever the comparison recomputes PyATB's table, PyATB's
printed table must equal the recomputation's PyATB-literal table: the check of the
transcription, PyATB's reader included, against PyATB itself. A case PyATB has not
run yet is skipped. ``QUEER_REAL_REPLAY_WORKERS`` sets the workers (default 4) and
``QUEER_PARITY_REPORT_DIR`` receives each report as JSON and Markdown.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

from queer.compat.generator.openmx_params import params_from_pyatb_input
from queer.compat.generator.parity import DOCUMENTED, compare, render_markdown
from queer.compat.generator.pyatb_files import read_input
from queer.compat.generator.results import EXIT_OK
from queer.compat.generator.runner import run

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(
    os.environ.get("QUEER_PYATB_SHIFT_CALIBRATION_DIR")
    or Path.home() / ".cache" / "queer-parity-data" / "pyatb-shift-kit"
)
SPEC = importlib.util.spec_from_file_location(
    "shift_kit", ROOT / "scripts/validation/generator_pyatb_shift_kit.py"
)
KIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(KIT)
WORKERS = int(os.environ.get("QUEER_REAL_REPLAY_WORKERS") or 4)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("QUEER_PYATB_SHIFT_CALIBRATION") != "1",
        reason="the PyATB shift-current calibration runs with QUEER_PYATB_SHIFT_CALIBRATION=1",
    ),
]


def save_parity_report(report, name):
    directory = os.environ.get("QUEER_PARITY_REPORT_DIR")
    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / f"{name}.json").write_text(json.dumps(report, indent=1) + "\n")
        (Path(directory) / f"{name}.md").write_text(render_markdown(report))


def judge(case, tmp_path):
    reference = DATA / case
    if not (reference / "Out" / "Shift_Current" / "shift_current.dat").is_file():
        pytest.skip(f"PyATB has not run {case} yet ({reference}/Out is missing)")
    model, _, _, self_convergence, *_ = KIT.CASES[case]
    workspace = tmp_path / case
    workspace.mkdir()
    (workspace / "scf.scfout").symlink_to(DATA / model)
    params = params_from_pyatb_input(
        read_input(reference / "Input"), workers=WORKERS, self_convergence=True if self_convergence else None
    )
    (workspace / "queer_params.json").write_text(json.dumps(params, indent=1))
    code = run(None, workspace / "queer_params.json", workspace, log=lambda line: None)
    document = json.loads((workspace / "queer_results.json").read_text())
    assert code == EXIT_OK, document.get("error")
    report = compare("SHIFT_CURRENT", reference, workspace, scfout=DATA / model, workers=WORKERS)
    save_parity_report(report, case)
    return report


@pytest.mark.parametrize("case", sorted(KIT.CASES))
def test_queer_meets_pyatbs_shift_current_case(case, tmp_path):
    report = judge(case, tmp_path)
    _, _, block, _, verdict, kinds, cause, purpose = KIT.CASES[case]
    rules = (report.get("semantics") or {}).get("rules") or {}
    if int(block.get("method", 1)) == 1:
        # PyATB's own table is the recomputation's PyATB-literal one: the transcription and PyATB's reader agree.
        reproduced = rules.get("pyatb_reproduced") or {}
        assert reproduced.get("reproduced"), f"{case}: {json.dumps(reproduced)}\n{render_markdown(report)}"
    assert report["verdict"] == verdict, f"{purpose}\n{render_markdown(report)}"
    compared = {item["class"] for item in report["quantities"] if item["name"].startswith("sigma_")}
    assert compared <= set(kinds) and bool(compared) == bool(kinds), render_markdown(report)
    if cause is not None:
        attribution = rules["attribution"]
        documented = [values for values in attribution.values() if values["class"] == DOCUMENTED]
        assert documented and all(
            max(values["steps_uA_V2"], key=values["steps_uA_V2"].get) == cause for values in documented
        ), render_markdown(report)
