"""QUEER against the Generator PythTB driver's goldens: every case must pass the parity comparator.

``tests/fixtures/generator_goldens`` holds the driver's results on the analytic
cases of ``generator_golden_cases`` (``scripts/validation/generate_generator_goldens.py``,
PythTB 2.0.2). Each case's inputs are rebuilt byte for byte and run with QUEER;
the comparator must find every recorded input digest and the parameter file equal
and every quantity in agreement. The goldens are live PythTB results, so this is
the library-replacement check that runs without PythTB.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GOLDENS = ROOT / "tests" / "fixtures" / "generator_goldens"
sys.path.insert(0, str(ROOT / "tests" / "numerical"))

from generator_golden_cases import CASES, write_case  # noqa: E402

from queer.compat.generator.parity import (  # noqa: E402
    AGREE,
    DOCUMENTED,
    NOT_APPLICABLE,
    PASS,
    PASS_DOCUMENTED,
    compare,
)
from queer.compat.generator.runner import run  # noqa: E402

DRIVER_CORE_SHA256 = "56fda912ba570e2293b3a9b4bf20d11038c64efa5d2f192c8ac03c615034e215"
# Extrema tied at several k-points (K and K' of the honeycomb models) are picked
# by rounding, which differs between BLAS kernels and machines. Only those picks
# may differ from a golden, as documented ties; everything else must agree.
TIE_BREAKS = ("edge_kpoints", "is_direct", "gap_midpoints")


def test_every_case_has_a_golden_and_nothing_else_is_stored():
    assert sorted(path.name for path in GOLDENS.iterdir() if path.is_dir()) == sorted(CASES)
    stored = sorted(str(path.relative_to(GOLDENS)) for path in GOLDENS.rglob("*") if path.is_file())
    assert all(name == "README.md" or name.endswith(".json") for name in stored)
    assert sum((GOLDENS / name).stat().st_size for name in stored) < 1 << 20


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_driver_golden_of_a_case_passes_against_queer(tmp_path, name):
    golden = GOLDENS / name
    reference = json.loads((golden / "pythtb_results.json").read_text())
    assert reference["step_type"] == CASES[name].step and reference["status"] == "ok"
    assert reference["provenance"]["core_sha256"] == DRIVER_CORE_SHA256
    assert reference["provenance"]["pythtb"] == "2.0.2"
    params = write_case(name, tmp_path)
    assert run(None, params, tmp_path, log=lambda line: None) == 0
    report = compare(CASES[name].step, golden, tmp_path)
    assert report["verdict"] in (PASS, PASS_DOCUMENTED), report["summary"]
    assert {item["class"] for item in report["quantities"]} <= {AGREE, NOT_APPLICABLE, DOCUMENTED}
    documented = [item["name"] for item in report["quantities"] if item["class"] == DOCUMENTED]
    assert all(item.rsplit(".", 1)[-1] in TIE_BREAKS for item in documented), documented
    assert report["reference"]["engine"] == "pythtb" and report["candidate"]["engine"] == "queer"
    inputs = next(item for item in report["preconditions"] if item["name"] == "inputs")
    assert inputs["ok"] and inputs["detail"].startswith("SHA-256 equal for 6 files")
    assert not report["inputs_verified_by_fingerprint"]
    parameters = next(item for item in report["preconditions"] if item["name"] == "parameters")
    assert parameters["detail"] == "identical parameter files (params_sha256)"
