"""QUEER's OpenMX steps against PyATB's calibration runs on a kagome scfout.

Opt-in: ``QUEER_PYATB_CALIBRATION=1``. ``QUEER_PYATB_CALIBRATION_DIR`` names the
calibration directory (default ``data/pyatb-calib-60fc6b2a`` of this checkout): one
directory per case holding PyATB's ``Input``, ``Out/input.json``,
``Out/running.log`` and its product, and ``test.scfout``, the scfout every case's
``Input`` names (``../test.scfout``). Each case's ``queer_params.json`` is built
from its ``Input`` as the Generator builds it (the INPUT_PARAMETERS without
``package`` and ``scfout_route``, and the function block); the refining Grid cases
also ask for the self-convergence estimate a comparison of them needs.
``QUEER_REAL_REPLAY_WORKERS`` sets the workers (default 4) and
``QUEER_PARITY_REPORT_DIR`` receives each report as JSON and Markdown.

The scfout is a kagome Chern insulator with spin-orbit coupling: E_F -2.35 eV
sits on its sigma_xy plateau, -2.12 eV in a band and -2.60 eV below it.
"""

import json
import os
from pathlib import Path

import pytest

from queer.compat.generator.openmx_params import params_from_pyatb_input
from queer.compat.generator.parity import (
    AFTER_CONVENTION,
    AGREE,
    DOCUMENTED,
    PASS,
    PASS_DOCUMENTED,
    compare,
    render_markdown,
)
from queer.compat.generator.pyatb_files import read_input
from queer.compat.generator.results import EXIT_OK
from queer.compat.generator.runner import run

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("QUEER_PYATB_CALIBRATION_DIR") or ROOT / "data" / "pyatb-calib-60fc6b2a")
SCFOUT_SHA256 = "9309730433e6b54d439aad25ca3dbcc57eb75c705ad1684cdfb310be8b6ea0cb"
# Case: (expected verdicts, expected class of every compared sigma or Chern quantity).
CASES = {
    "a_fhs_scan": ({PASS, PASS_DOCUMENTED}, {AGREE, DOCUMENTED}),
    "b_direct_pyatb": ({PASS}, {AFTER_CONVENTION}),
    "c_direct_openmx": ({PASS}, {AGREE}),
    "d_kubo_pyatb": ({PASS}, {AFTER_CONVENTION}),
    "e_kubo_openmx": ({PASS}, {AGREE}),
    "f_kubo_auto": ({PASS}, {AGREE}),
    "g_kubo_openmx_offplateau": ({PASS}, {AGREE}),
    "h_chern_fermi": ({PASS}, {AGREE}),
    "i_threshold_10": ({PASS, PASS_DOCUMENTED}, {AFTER_CONVENTION, DOCUMENTED}),
    "i_threshold_20": ({PASS, PASS_DOCUMENTED}, {AFTER_CONVENTION, DOCUMENTED}),
    "j_direct_metal": ({PASS}, {AGREE}),
    "k_kubo_metal": ({PASS}, {AGREE}),
}
REFINING = {"i_threshold_10", "i_threshold_20"}

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("QUEER_PYATB_CALIBRATION") != "1",
        reason="the PyATB calibration replay runs with QUEER_PYATB_CALIBRATION=1",
    ),
]


def save_parity_report(report, name):
    directory = os.environ.get("QUEER_PARITY_REPORT_DIR")
    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / f"{name}.json").write_text(json.dumps(report, indent=1) + "\n")
        (Path(directory) / f"{name}.md").write_text(render_markdown(report))


@pytest.mark.parametrize("case", sorted(CASES))
def test_queer_reproduces_the_pyatb_calibration_case(case, tmp_path):
    reference = DATA / case
    workspace = tmp_path / case
    workspace.mkdir()
    (workspace / "scf.scfout").symlink_to(DATA / "test.scfout")
    params = params_from_pyatb_input(
        read_input(reference / "Input"),
        workers=int(os.environ.get("QUEER_REAL_REPLAY_WORKERS") or 4),
        self_convergence=True if case in REFINING else None,
    )
    (workspace / "queer_params.json").write_text(json.dumps(params, indent=1))
    code = run(None, workspace / "queer_params.json", workspace, log=lambda line: None)
    document = json.loads((workspace / "queer_results.json").read_text())
    assert code == EXIT_OK, document.get("error")
    assert document["model"]["scfout_sha256"] == SCFOUT_SHA256

    report = compare(
        "AHC" if "AHC" in params["counterpart_step_type"] else "CHERN_NUMBER",
        reference,
        workspace,
        scfout=DATA / "test.scfout",
    )
    save_parity_report(report, case)
    verdicts, classes = CASES[case]
    assert report["verdict"] in verdicts, render_markdown(report)
    compared = {
        item["class"] for item in report["quantities"] if item["name"].startswith(("sigma_", "chern_number"))
    }
    assert compared and compared <= classes, render_markdown(report)
