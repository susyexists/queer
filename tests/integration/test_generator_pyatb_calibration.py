"""QUEER's OpenMX steps against PyATB's calibration runs on a kagome scfout.

Opt-in: ``QUEER_PYATB_CALIBRATION=1``. ``QUEER_PYATB_CALIBRATION_DIR`` names the
calibration directory (default ``data/pyatb-calib-60fc6b2a`` of this checkout): one
directory per case holding PyATB's ``Input``, ``Out/input.json``,
``Out/running.log`` and its product, and ``test.scfout``, the scfout every case's
``Input`` names (``../test.scfout``). Each case's ``queer_params.json`` is built
from its ``Input`` as the Generator builds it (the INPUT_PARAMETERS without
``package`` and ``scfout_route``, and the function block); the refining Grid cases
also ask for the self-convergence estimate a comparison of them needs, and their
emulation of PyATB's refined sum must refine exactly PyATB's nodes. The FHS scan's
emulation of PyATB's links must reproduce PyATB's table at every energy, and QUEER's
FHS must meet its own Fermi-sea sum at every band energy. A metal
relabelled to a refining Grid that refines nothing checks that PyATB's own grid
error bounds a gap QUEER's alone cannot.
``QUEER_REAL_REPLAY_WORKERS`` sets the workers (default 4) and
``QUEER_PARITY_REPORT_DIR`` receives each report as JSON and Markdown.

The scfout is a kagome Chern insulator with spin-orbit coupling: E_F -2.35 eV
sits on its sigma_xy plateau, -2.12 eV in a band and -2.60 eV below it.
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
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
from queer.compat.generator.pyatb_files import parse_ahc, read_input
from queer.compat.generator.results import EXIT_OK
from queer.compat.generator.runner import run

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("QUEER_PYATB_CALIBRATION_DIR") or ROOT / "data" / "pyatb-calib-60fc6b2a")
SCFOUT_SHA256 = "9309730433e6b54d439aad25ca3dbcc57eb75c705ad1684cdfb310be8b6ea0cb"
# Case: (expected verdicts, expected class of every compared sigma or Chern quantity).
CASES = {
    "a_fhs_scan": ({PASS_DOCUMENTED}, {AGREE, DOCUMENTED}),
    "b_direct_pyatb": ({PASS}, {AFTER_CONVENTION}),
    "c_direct_openmx": ({PASS}, {AGREE}),
    "d_kubo_pyatb": ({PASS}, {AFTER_CONVENTION}),
    "e_kubo_openmx": ({PASS}, {AGREE}),
    "f_kubo_auto": ({PASS}, {AGREE}),
    "g_kubo_openmx_offplateau": ({PASS}, {AGREE}),
    "h_chern_fermi": ({PASS}, {AGREE}),
    "i_threshold_10": ({PASS_DOCUMENTED}, {AFTER_CONVENTION, DOCUMENTED}),
    "i_threshold_20": ({PASS_DOCUMENTED}, {AFTER_CONVENTION, DOCUMENTED}),
    "j_direct_metal": ({PASS}, {AGREE}),
    "k_kubo_metal": ({PASS}, {AGREE}),
    # A Fermi level where 8 grid points sit inside pairs closer than 1e-4 eV: PyATB's direct engine zeroes
    # them (its documented cut, reproduced at those points), its Kubo engine does not.
    "l_direct_paircut": ({PASS_DOCUMENTED}, {DOCUMENTED}),
    "m_kubo_paircut": ({PASS}, {AGREE}),
}
REFINING = {"i_threshold_10", "i_threshold_20"}
# PyATB's own refinement counts (running.log), which QUEER's emulation of its refined sum must reproduce.
REFINED_NODES = {"i_threshold_10": 1757, "i_threshold_20": 471}

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
    if case in REFINING:
        emulation = document["ahc"]["pyatb_emulation"]["refined_grid"]
        assert emulation["flagged"] == REFINED_NODES[case]
        assert {item["name"]: item["class"] for item in report["quantities"]}["pyatb_refined_grid"] == AGREE
    if case == "a_fhs_scan":
        # QUEER's emulation of PyATB's FHS links reproduces PyATB's table at every energy (Tier A).
        table = parse_ahc((reference / "Out/AHC/ahc.dat").read_text())
        printed = np.asarray(table["sigma"])[:, 2]
        emulated = np.asarray(document["ahc"]["pyatb_emulation"]["fhs_links"]["printed_S_cm"]["xy"])
        assert np.all(np.abs(printed - emulated) <= 2e-7 * np.abs(emulated) + 5e-7)
        # QUEER's FHS meets its own Fermi-sea sum at every band energy; PyATB's lies above it.
        guard = report["tolerance_basis"]["fermi_sea_guard"]["xy"]
        assert guard["energies"] == 42 and guard["failed"] == 0, guard
        sea = np.asarray(document["ahc"]["fermi_sea_check"]["fermi_sea_S_cm"]["xy"], dtype=float)
        band = np.isfinite(sea)
        queer = np.asarray(document["ahc"]["sigma_S_cm"]["xy"])[band] - sea[band]
        assert abs(np.mean(queer)) < 0.03 and np.mean(printed[band] - sea[band]) > 0.08


def _relabel(text, replacements):
    for old, new in replacements:
        assert old in text, old
        text = text.replace(old, new)
    return text


def test_a_refined_metal_is_bounded_by_both_codes_discretisation_errors(tmp_path):
    """A correct refined Grid on a metal passes only with PyATB's own error e_P.

    PyATB's j_direct_metal run (E_F -2.12 eV, 100 x 100 x 1, refinement off) relabelled
    to threshold 100 A^2 with 20 x 20 x 1 sub-grids: no node's curvature reaches 100 A^2
    (the largest is 21.5), so PyATB prints the same plain Gamma-origin sum. QUEER's
    midpoint sum differs from it by more than 3 e_Q; PyATB's own grid error, e_P from
    QUEER's emulation of its refined sum on the grid and its double, bounds the gap.
    """
    reference = tmp_path / "reference"
    shutil.copytree(DATA / "j_direct_metal", reference)
    swaps = [
        ("adaptive_grid                   1 1 1", "adaptive_grid                   20 20 1"),
        ("adaptive_grid_threshold         1e30", "adaptive_grid_threshold         100"),
    ]
    (reference / "Input").write_text(_relabel((reference / "Input").read_text(), swaps))
    echo = json.loads((reference / "Out/input.json").read_text())
    echo["AHC"].update(adaptive_grid=[20, 20, 1], adaptive_grid_threshold=100.0)
    (reference / "Out/input.json").write_text(json.dumps(echo, indent=4))
    log = _relabel(
        (reference / "Out/running.log").read_text(),
        [
            (
                ">> adaptive_grid           : 1        1        1       ",
                ">> adaptive_grid           : 20       20       1       ",
            ),
            ("1000000000000000019884624838656.0000", "100.0000"),
        ],
    )
    (reference / "Out/running.log").write_text(log)
    workspace = tmp_path / "queer"
    workspace.mkdir()
    (workspace / "scf.scfout").symlink_to(DATA / "test.scfout")
    params = params_from_pyatb_input(
        read_input(reference / "Input"),
        workers=int(os.environ.get("QUEER_REAL_REPLAY_WORKERS") or 4),
        self_convergence=True,
    )
    (workspace / "queer_params.json").write_text(json.dumps(params, indent=1))
    assert run(None, workspace / "queer_params.json", workspace, log=lambda line: None) == EXIT_OK
    report = compare("AHC", reference, workspace, scfout=DATA / "test.scfout")
    save_parity_report(report, "j_direct_metal_threshold_100")
    classes = {item["name"]: item["class"] for item in report["quantities"]}
    assert report["verdict"] == PASS_DOCUMENTED and classes["sigma_xy"] == DOCUMENTED, render_markdown(report)
    assert classes["pyatb_refined_grid"] == AGREE
    e_q, e_p = report["tolerance_basis"]["e_Q"][2], report["tolerance_basis"]["e_P"][2]
    gap = [item for item in report["quantities"] if item["name"] == "sigma_xy"][0]["metric"]
    assert 3 * e_q < gap <= 3 * e_p, (gap, e_q, e_p)
