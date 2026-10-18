"""QUEER's shift current against PyATB's runs of the shift-current calibration kit.

Opt-in: ``QUEER_PYATB_SHIFT_CALIBRATION=1``. ``QUEER_PYATB_SHIFT_CALIBRATION_DIR``
names the kit (default ``~/.cache/queer-parity-data/pyatb-shift-kit``), written by
``scripts/validation/generator_pyatb_shift_kit.py``: the synthetic scfouts and one
directory per case holding PyATB's ``Input`` and, once PyATB has run it, its
``inputs.sha256``, ``Out/input.json``, ``Out/running.log`` and
``Out/Shift_Current/shift_current.dat`` (the layout of the AHC calibration
``pyatb-calib-60fc6b2a``). Each case's ``queer_params.json`` is built from its
``Input`` as the Generator builds it, with the self-convergence estimate the cases
with an unbounded cause need; the clean control is judged first and its report
applies to every other case. A case PyATB has not run yet is skipped.
``QUEER_REAL_REPLAY_WORKERS`` sets the workers (default 4) and
``QUEER_PARITY_REPORT_DIR`` receives each report as JSON and Markdown.
"""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from queer.compat.generator.openmx_params import params_from_pyatb_input
from queer.compat.generator.parity import AGREE, PASS, compare, render_markdown
from queer.compat.generator.pyatb_files import SHIFT_COMPONENTS, parse_shift_current, read_input
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
CLEAN = "a_clean_control"
# Cases where PyATB should print QUEER's emulation of its scheme to its own print precision.
EMULATED = {"a_clean_control", "c_no_smearing", "e_collinear", "f_band_pair", "g_left_handed"}
# PyATB prints %15.6e: seven significant digits, a rounding of at most 5e-7 of each value.
PRINT_PRECISION = 1e-6

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


def judge(case, tmp_path, clean_control=None):
    reference = DATA / case
    if not (reference / "Out" / "Shift_Current" / "shift_current.dat").is_file():
        pytest.skip(f"PyATB has not run {case} yet ({reference}/Out is missing)")
    model, _, _, self_convergence, *_ = KIT.CASES[case]
    workspace = tmp_path / case
    workspace.mkdir()
    (workspace / "scf.scfout").symlink_to(DATA / model)
    params = params_from_pyatb_input(
        read_input(reference / "Input"),
        workers=int(os.environ.get("QUEER_REAL_REPLAY_WORKERS") or 4),
        self_convergence=True if self_convergence else None,
    )
    (workspace / "queer_params.json").write_text(json.dumps(params, indent=1))
    code = run(None, workspace / "queer_params.json", workspace, log=lambda line: None)
    document = json.loads((workspace / "queer_results.json").read_text())
    assert code == EXIT_OK, document.get("error")
    report = compare("SHIFT_CURRENT", reference, workspace, scfout=DATA / model, clean_control=clean_control)
    save_parity_report(report, case)
    return report, document


def assert_emulation_reproduces_pyatb(case, document):
    """QUEER's emulation of PyATB's scheme reproduces PyATB's table to PyATB's print precision.

    Each printed value within 1e-6 of itself (``%15.6e`` rounds to 5e-7), plus 1e-10
    of the peak for summation order; no transition lies within 1e-9 eV of a bin or
    window edge, where the two codes' rounding could place it apart.
    """
    printed = np.asarray(
        parse_shift_current((DATA / case / "Out/Shift_Current/shift_current.dat").read_text())["sigma"]
    )
    scheme = document["shift_current"]["diagnostics"]["pyatb_scheme"]
    assert scheme["edge_transitions"] == 0, case
    emulated = np.column_stack([scheme["sigma_uA_V2"][name] for name in SHIFT_COMPONENTS])
    allowed = PRINT_PRECISION * np.abs(emulated) + 1e-10 * np.max(np.abs(emulated))
    excess = np.abs(printed - emulated) - allowed
    assert np.all(excess <= 0), f"{case}: worst excess {np.max(excess):.3g} microA/V^2"


@pytest.fixture(scope="module")
def clean_run(tmp_path_factory):
    return judge(CLEAN, tmp_path_factory.mktemp("clean"))


def test_the_clean_control_agrees_with_queers_physical_value_directly(clean_run):
    report, document = clean_run
    rules = report["semantics"]["rules"]
    assert rules["clean_control"]["eligible"], rules["clean_control"]
    assert report["verdict"] == PASS, render_markdown(report)
    # Every component meets QUEER's physical value itself, not only its emulation of PyATB.
    compared = {item["class"] for item in report["quantities"] if item["name"].startswith("sigma_")}
    assert compared == {AGREE}, render_markdown(report)
    assert_emulation_reproduces_pyatb(CLEAN, document)


@pytest.mark.parametrize("case", sorted(set(KIT.CASES) - {CLEAN}))
def test_queer_meets_pyatbs_shift_current_case(case, tmp_path, clean_run):
    report, document = judge(case, tmp_path, clean_control=clean_run[0])
    _, _, _, _, verdict, kinds, purpose = KIT.CASES[case]
    assert report["verdict"] == verdict, f"{purpose}\n{render_markdown(report)}"
    compared = {item["class"] for item in report["quantities"] if item["name"].startswith("sigma_")}
    assert compared and compared <= set(kinds), render_markdown(report)
    assert report["semantics"]["rules"]["clean_control"]["release"] == "passed"
    if case in EMULATED:
        assert_emulation_reproduces_pyatb(case, document)
