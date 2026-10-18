"""QUEER against the Generator PythTB driver's goldens: every case must pass the parity comparator.

``tests/fixtures/generator_goldens`` holds the driver's results on the analytic
cases of ``generator_golden_cases`` (``scripts/validation/generate_generator_goldens.py``,
PythTB 2.0.2). Each case's inputs are rebuilt byte for byte and run with QUEER;
the comparator must find every recorded input digest and the parameter file equal
and every quantity in agreement. The goldens are live PythTB results, so this is
the library-replacement check that runs without PythTB.
"""

import json
import re
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
from queer.compat.generator.parity_qgeom import QGEOM_KERNEL_SHA256  # noqa: E402
from queer.compat.generator.runner import run  # noqa: E402

DRIVER_CORE_SHA256 = "56fda912ba570e2293b3a9b4bf20d11038c64efa5d2f192c8ac03c615034e215"
# The slab golden is the driver's that counts edge crossings as a weighted flow of
# half-ribbon weight across the bulk gap, checked against the bulk Chern number; the
# other cases' results did not change with it.
CASE_CORE_SHA256 = {"haldane_slab": "f4a73779a891de356cc53e55d8c077f0fdc6befce931a6ed0bf9782973f5551b"}
# Extrema tied at several k-points (K and K' of the honeycomb models) are picked
# by rounding, which differs between BLAS kernels and machines. Only those picks
# may differ from a golden, as documented ties; everything else must agree.
TIE_BREAKS = ("edge_kpoints", "is_direct", "gap_midpoints")
QGEOM_CASES = sorted(name for name, case in CASES.items() if case.step == "PYTHTB_QGEOM")
# Two mu_CNP in one gap of the count agree (the count decides); the CNP rows they snap to
# different grid mu are documented and checked against the other run's curves.
CNP_TIES = (
    r"qgeom\.headline\.\d+K\.CNP\.mu_meV",
    r"qgeom\.dipole_sources\.CNP",
)
WANNIER_CASES = sorted(set(CASES) - set(QGEOM_CASES))
# The quantum-geometry goldens are the vendored core's (its quantum-geometry code is 56fda912's).
QGEOM_CORE_SHA256 = "0efcc4b3079266a8d35644d3d06f648857e31d86b12be173c6f03298a5d505c0"


def test_every_case_has_a_golden_and_nothing_else_is_stored():
    assert sorted(path.name for path in GOLDENS.iterdir() if path.is_dir()) == sorted(CASES)
    stored = sorted(str(path.relative_to(GOLDENS)) for path in GOLDENS.rglob("*") if path.is_file())
    assert all(name == "README.md" or name.endswith(".json") for name in stored)
    assert sum((GOLDENS / name).stat().st_size for name in stored) < 1 << 20


@pytest.mark.parametrize("name", WANNIER_CASES)
def test_the_driver_golden_of_a_case_passes_against_queer(tmp_path, name):
    golden = GOLDENS / name
    reference = json.loads((golden / "pythtb_results.json").read_text())
    assert reference["step_type"] == CASES[name].step and reference["status"] == "ok"
    assert reference["provenance"]["core_sha256"] == CASE_CORE_SHA256.get(name, DRIVER_CORE_SHA256)
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
    if CASES[name].step == "PYTHTB_SLAB":
        kinds = {item["name"]: item["class"] for item in report["quantities"]}
        for quantity in ("at_fermi", "status", "per_energy", "recomputed"):
            assert kinds[f"slab.edge_crossings_{quantity}"] == AGREE
        assert kinds["slab.half_ribbon_weights"] == kinds["slab.bulk_at_fermi"] == AGREE
        crossings = json.loads((golden / "pythtb_results.json").read_text())["slab"][
            "edge_crossings_at_fermi"
        ]
        assert crossings["status"] == "resolved" and crossings["method"] == "half_ribbon_flow"
        bulk = json.loads((golden / "pythtb_results.json").read_text())["slab"]["bulk_at_fermi"]
        assert crossings["checks"] == {} and bulk["chern"] == crossings["left"]["net"]


@pytest.mark.parametrize("name", QGEOM_CASES)
def test_the_driver_golden_of_a_quantum_geometry_case_passes_against_queer(tmp_path, name):
    case = CASES[name]
    golden = GOLDENS / name
    reference = json.loads((golden / "pythtb_results.json").read_text())
    provenance = reference["provenance"]
    assert reference["step_type"] == "PYTHTB_QGEOM" and provenance["pythtb"] == "2.0.2"
    assert (
        provenance["core_sha256"] == QGEOM_CORE_SHA256 and provenance["qgeom_sha256"] == QGEOM_KERNEL_SHA256
    )
    params = write_case(name, tmp_path)
    code = run(None, params, tmp_path, log=lambda line: None)
    report = compare("QUEER_QGEOM", golden, tmp_path)
    if case.expect != "ok":
        assert code == 2 and reference["error"]["code"] == case.expect
        assert report["verdict"] == PASS and [item["class"] for item in report["quantities"]] == [AGREE]
        return
    assert code == 0 and reference["status"] == "ok"
    assert report["verdict"] in (PASS, PASS_DOCUMENTED), report["summary"]
    kinds = {item["name"]: item["class"] for item in report["quantities"]}
    documented = sorted(name for name, kind in kinds.items() if kind == DOCUMENTED)
    # The driver never reads the .win kpoints block, so only QUEER checks the .eig. Inside a gap
    # rounding picks the interpolated mu_CNP, which differs between BLAS kernels, and with it the CNP row.
    ties = [name for name in documented if any(re.fullmatch(pattern, name) for pattern in CNP_TIES)]
    assert sorted(set(documented) - set(ties)) == (["qgeom.dft_check"] if case.eig else [])
    assert set(kinds.values()) <= {AGREE, NOT_APPLICABLE, DOCUMENTED}
    assert kinds["qgeom.mesh"] == kinds["qgeom.stop"] == AGREE
    assert all(
        kind == AGREE for name, kind in kinds.items() if name.startswith("qgeom.electron_count.mu_cnp")
    )
    mesh = next(item for item in report["quantities"] if item["name"] == "qgeom.mesh")
    assert mesh["detail"].startswith("identical vertex sets")
    inputs = next(item for item in report["preconditions"] if item["name"] == "inputs")
    assert inputs["ok"] and inputs["detail"].startswith(f"SHA-256 equal for {6 + case.tb} files")
    stop = next(item for item in report["preconditions"] if item["name"] == "stop_rule")
    assert stop["ok"] and stop["detail"].startswith("deterministic")
