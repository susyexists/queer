"""The shift-current parity rule: PyATB SHIFT_CURRENT run directories against QUEER runs.

Candidates are real runner outputs on the analytic models of ``openmx_files``. A
reference prints the candidate's own emulation of PyATB's scheme (what PyATB would
print from QUEER's transitions) unless a test gives it other numbers, so every
faithful pair passes and each negative control must not: a flipped sign, a zeroed
component, photon energies shifted by one step, another eta or occ_band, and a
candidate that is not converged.
"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from openmx_files import shift_collinear_scfout, shift_spinor_scfout

from queer.compat.generator.openmx_params import SHIFT_DEFAULTS
from queer.compat.generator.parity import (
    AFTER_CONVENTION,
    AGREE,
    DOCUMENTED,
    FAIL,
    NOT_APPLICABLE,
    NOT_COMPARABLE,
    PASS,
    PASS_DOCUMENTED,
    UNEXPLAINED,
    compare,
    render_markdown,
)
from queer.compat.generator.parity_shift import literature_anchor, symmetry_gate
from queer.compat.generator.pyatb_files import SHIFT_COMPONENTS, shift_current_text
from queer.compat.generator.runner import main, run

SPINOR = {"occ_band": 2, "omega": [0.3, 6.5], "domega": 0.1, "eta": 0.1, "grid": [3, 3, 2]}
COLLINEAR = {"occ_band": 1, "omega": [0.2, 4.5], "domega": 0.05, "eta": 0.1, "grid": [12, 12, 1]}
RUNS = {
    "clean": (shift_spinor_scfout, {}, 4, SPINOR, None),
    "binned": (shift_spinor_scfout, {}, 4, {**SPINOR, "smearing_method": 0, "domega": 0.05}, None),
    "collinear": (shift_collinear_scfout, {}, 1, {**COLLINEAR, "grid": [6, 6, 1]}, None),
    "nonmagnetic": (shift_collinear_scfout, {"spin": 0}, 1, {**COLLINEAR, "grid": [6, 6, 1]}, None),
    "method0": (shift_collinear_scfout, {}, 1, {**COLLINEAR, "method": 0}, True),
    "scissors": (shift_collinear_scfout, {}, 1, {**COLLINEAR, "scissors": 0.3}, True),
    "unconverged": (shift_spinor_scfout, {}, 4, {**SPINOR, "method": 0, "grid": [2, 2, 1]}, True),
    "left": (shift_spinor_scfout, {"lattice": np.diag([3.0, 3.2, -3.5])}, 4, SPINOR, None),
    "pair": (shift_spinor_scfout, {}, 4, {**SPINOR, "omega": [1.5, 6.5], "n_occ": 2, "m_unocc": 3}, None),
    "coarse": (shift_spinor_scfout, {}, 4, {**SPINOR, "eta": 0.02, "domega": 0.2}, None),
    "window": (shift_spinor_scfout, {}, 4, {**SPINOR, "omega": [2.0, 4.0]}, None),
    "adaptive": (shift_spinor_scfout, {}, 4, {**SPINOR, "smearing_method": 2, "grid": [2, 2, 1]}, True),
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=1))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def columns(block):
    return np.column_stack([block[name] for name in SHIFT_COMPONENTS])


def candidate(directory, scfout, model, nspin, block, self_convergence=None):
    directory.mkdir(parents=True)
    scfout(directory / "scf.scfout", **model)
    params = {
        "step_type": "QUEER_SHIFT_CURRENT",
        "counterpart_step_type": "PYATB_SHIFT_CURRENT",
        "input_parameters": {"nspin": nspin},
        "parameters": dict(block),
        "workers": 1,
    }
    if self_convergence is not None:
        params["self_convergence"] = self_convergence
    write(directory / "queer_params.json", params)
    run(None, directory / "queer_params.json", directory, log=lambda line: None)
    return directory


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("shift")
    return {name: candidate(root / name, *spec) for name, spec in RUNS.items()}


def block_text(name, values):
    lines = [name, "{"]
    for key, value in values.items():
        text = " ".join(str(v) for v in value) if isinstance(value, list) else str(value)
        lines.append(f"    {key:<32}{text}")
    return "\n".join(lines + ["}", ""])


def scheme_table(run_dir):
    block = read(run_dir / "queer_results.json")["shift_current"]
    return columns(block["diagnostics"]["pyatb_scheme"]["sigma_uA_V2"])


def reference_from(directory, run_dir, *, sigma=None, echo=None, omega=None, digests=True):
    """A PyATB-layout run directory printing ``sigma`` (default: the candidate's emulation of PyATB)."""
    document = read(run_dir / "queer_results.json")
    params = read(run_dir / "queer_params.json")
    block = {**SHIFT_DEFAULTS, **params["parameters"], **(echo or {})}
    inputs = {"package": "OPENMX", "scfout_route": "scf.scfout", **params["input_parameters"]}
    out = directory / "Out"
    (out / "Shift_Current").mkdir(parents=True)
    text = block_text("INPUT_PARAMETERS", inputs)
    text += "\n" + block_text("LATTICE", {"lattice_constant": 1.0, "lattice_constant_unit": "Angstrom"})
    text += "\n" + block_text("SHIFT_CURRENT", block)
    (directory / "Input").write_text(text)
    write(
        out / "input.json", {"INPUT_PARAMETERS": {**inputs, "max_kpoint_num": 8000}, "SHIFT_CURRENT": block}
    )
    (out / "running.log").write_text("|                  Shift Current                     |\n")
    shift = document["shift_current"]
    values = scheme_table(run_dir) if sigma is None else np.asarray(sigma)
    (out / "Shift_Current" / "shift_current.dat").write_text(
        shift_current_text(shift["omega_eV"] if omega is None else omega, values)
    )
    if digests:
        (directory / "inputs.sha256").write_text(
            f"{document['model']['scfout_sha256']}  scf.scfout\n{sha256(directory / 'Input')}  Input\n"
        )
    return directory


def classes(report):
    return {item["name"]: item["class"] for item in report["quantities"]}


def components(report):
    return {item["class"] for item in report["quantities"] if item["name"].startswith("sigma_")}


def judge(runs, tmp_path, name, **reference):
    return compare(
        "QUEER_SHIFT_CURRENT", reference_from(tmp_path / "reference", runs[name], **reference), runs[name]
    )


def test_every_candidate_ran(runs):
    for name, directory in runs.items():
        document = read(directory / "queer_results.json")
        assert document["status"] == "ok", (name, document.get("error"))


@pytest.mark.parametrize(
    ("name", "kinds"),
    [
        ("clean", {AGREE}),
        ("binned", {AFTER_CONVENTION}),
        ("collinear", {AFTER_CONVENTION}),
        ("nonmagnetic", {AGREE}),
        ("left", {AFTER_CONVENTION}),
        ("pair", {AGREE, AFTER_CONVENTION}),
        ("coarse", {AGREE, AFTER_CONVENTION}),
        ("window", {AGREE, AFTER_CONVENTION}),
    ],
)
def test_a_reference_printing_pyatbs_scheme_of_the_candidate_passes(runs, tmp_path, name, kinds):
    report = judge(runs, tmp_path, name)
    assert report["verdict"] == PASS, render_markdown(report)
    assert components(report) <= kinds and classes(report)["symmetry"] == AGREE
    assert classes(report)["candidate_file"] == AGREE
    rules = report["semantics"]["rules"]
    assert rules["legacy_single_channel"] is (name == "collinear")
    assert rules["export_sign"] == (-1 if name == "left" else 1)
    assert rules["spin_factor"] == (2 if name in {"collinear", "nonmagnetic"} else 1)
    assert report["tolerance_basis"]["relative"] == 1e-4 and report["tolerance_basis"]["emulated"]


def test_the_clean_control_is_recognised_from_the_data_and_must_agree(runs, tmp_path):
    report = judge(runs, tmp_path / "pass", "clean")
    clean = report["semantics"]["rules"]["clean_control"]
    assert clean["eligible"] and all(clean["conditions"].values()) and clean["release"] == "not_supplied"
    assert classes(report)["clean_control"] == AGREE
    for name in ("binned", "window"):
        rules = judge(runs, tmp_path / name, name)["semantics"]["rules"]["clean_control"]
        assert not rules["eligible"]
    failing = rules["conditions"]
    assert not failing["window_margin_5_eta"] and failing["gaussian"]
    # A clean control that differs by 2e-4 of its peak fails, whatever the causes.
    report = judge(runs, tmp_path / "off", "clean", sigma=scheme_table(runs["clean"]) * (1 + 2e-4))
    assert report["verdict"] == FAIL and classes(report)["clean_control"] == UNEXPLAINED
    # PyATB's truncated Gaussian on a coarse step is measured by the emulation, not forgiven.
    coarse = judge(runs, tmp_path / "coarse", "coarse")
    assert coarse["semantics"]["rules"]["clean_control"]["eligible"]
    truncation = read(runs["coarse"] / "queer_results.json")["shift_current"]["diagnostics"][
        "gaussian_truncation"
    ]
    assert truncation["interval_bins"] == 0 and truncation["max_effect_uA_V2"] > 1e-2
    assert AFTER_CONVENTION in components(coarse)


@pytest.mark.parametrize(
    ("change", "verdict"),
    [
        ("flip", FAIL),
        ("zero_xxx", FAIL),
        ("omega_shift", NOT_COMPARABLE),
        ("eta", NOT_COMPARABLE),
        ("occ_band", NOT_COMPARABLE),
        ("unconverged", NOT_COMPARABLE),
    ],
)
def test_negative_controls_never_pass(runs, tmp_path, change, verdict):
    table = scheme_table(runs["clean"])
    if change == "flip":
        report = judge(runs, tmp_path, "clean", sigma=-table)
        assert classes(report)["symmetry"] == UNEXPLAINED
    elif change == "zero_xxx":
        zeroed = table.copy()
        zeroed[:, 0] = 0
        report = judge(runs, tmp_path, "clean", sigma=zeroed)
        assert "xxx" in report["semantics"]["rules"]["symmetry"]["components_differ"]
    elif change == "omega_shift":
        omega = (
            np.asarray(read(runs["clean"] / "queer_results.json")["shift_current"]["omega_eV"]) + 0.1
        ).tolist()
        report = judge(runs, tmp_path, "clean", echo={"omega": [0.4, 6.6]}, omega=omega)
    elif change == "eta":
        report = judge(runs, tmp_path, "clean", echo={"eta": 0.12})
    elif change == "occ_band":
        report = judge(runs, tmp_path, "clean", echo={"occ_band": 1})
    else:
        # A real coarse candidate with a documented cause (method 0) whose grid has not converged.
        report = judge(runs, tmp_path, "unconverged", sigma=scheme_table(runs["unconverged"]) * 1.2)
        failed = [item for item in report["preconditions"] if not item["ok"]]
        assert [item["name"] for item in failed] == ["self_convergence"]
        assert failed[0]["detail"].startswith("queer_unconverged")
    assert report["verdict"] == verdict, render_markdown(report)
    if verdict == NOT_COMPARABLE and change != "unconverged":
        failed = {item["name"] for item in report["preconditions"] if not item["ok"]}
        assert failed & {"parameters", "omega"}, report["preconditions"]


def test_an_unbounded_cause_documents_a_difference_only_on_a_converged_candidate(runs, tmp_path):
    for name, cause in (("method0", "regularisation"), ("scissors", "scissors")):
        convergence = read(runs[name] / "queer_results.json")["shift_current"]["diagnostics"][
            "self_convergence"
        ]
        assert convergence["legacy_single_channel"]["converged"], convergence["legacy_single_channel"][
            "ratio"
        ]
        report = judge(runs, tmp_path / name, name, sigma=scheme_table(runs[name]) * 0.97)
        assert report["verdict"] == PASS_DOCUMENTED, render_markdown(report)
        # The 2D model's components with a z index vanish in both runs and agree.
        kinds = classes(report)
        assert {kinds[f"sigma_{name}"] for name in ("xxx", "xxy", "xyy", "yxx", "yxy", "yyy")} == {DOCUMENTED}
        assert components(report) == {DOCUMENTED, AFTER_CONVENTION}
        assert report["semantics"]["rules"]["unbounded_causes"] == [cause]
        assert [item for item in report["preconditions"] if item["name"] == "self_convergence"][0]["ok"]
        # The same numbers agree when PyATB prints QUEER's emulation.
        assert judge(runs, tmp_path / f"{name}-same", name)["verdict"] == PASS
    # A documented cause never excuses a broken symmetry.
    flipped = scheme_table(runs["method0"]) * 0.97
    flipped[:, 0] *= -1
    assert judge(runs, tmp_path / "flipped", "method0", sigma=flipped)["verdict"] == FAIL
    # Without its self-convergence estimate the candidate cannot be documented.
    copy = shutil.copytree(runs["method0"], tmp_path / "no-estimate")
    document = read(copy / "queer_results.json")
    document["shift_current"]["diagnostics"]["self_convergence"] = None
    write(copy / "queer_results.json", document)
    reference = reference_from(tmp_path / "no-estimate-ref", copy, sigma=scheme_table(copy) * 0.97)
    report = compare("SHIFT_CURRENT", reference, copy)
    assert report["verdict"] == NOT_COMPARABLE and "self_convergence: true" in json.dumps(
        report["preconditions"]
    )
    # Ten times PyATB's largest value is not a convergence question.
    report = judge(runs, tmp_path / "huge", "method0", sigma=scheme_table(runs["method0"]) / 20)
    assert report["verdict"] == NOT_COMPARABLE and "max|sigma_Q|" in json.dumps(report["preconditions"])


def test_adaptive_smearing_is_an_unbounded_cause_without_an_emulation(runs, tmp_path):
    document = read(runs["adaptive"] / "queer_results.json")
    scheme = document["shift_current"]["diagnostics"]["pyatb_scheme"]
    assert scheme["sigma_uA_V2"] is None and "not emulated" in scheme["reason"]
    physical = columns(document["shift_current"]["sigma_uA_V2"])
    report = judge(runs, tmp_path / "same", "adaptive", sigma=physical)
    assert report["verdict"] == PASS and not report["tolerance_basis"]["emulated"]
    report = judge(runs, tmp_path / "other", "adaptive", sigma=physical * 1.05)
    assert "adaptive_smearing" in report["semantics"]["rules"]["unbounded_causes"]
    # Its 2 x 2 x 1 grid is far from converged: the difference cannot be documented.
    failed = [item["name"] for item in report["preconditions"] if not item["ok"]]
    assert report["verdict"] == NOT_COMPARABLE and failed == ["self_convergence"]


def test_pyatbs_spin_and_bin_conventions_are_required(runs, tmp_path):
    collinear = read(runs["collinear"] / "queer_results.json")["shift_current"]
    # PyATB doubling the spin-up channel is the legacy value: the physical sum of both channels is not.
    report = judge(runs, tmp_path / "physical", "collinear", sigma=columns(collinear["sigma_uA_V2"]))
    assert report["verdict"] == FAIL
    binned = read(runs["binned"] / "queer_results.json")["shift_current"]
    # Without smearing PyATB prints bin weights: QUEER's histogram per eV is not what it prints.
    report = judge(runs, tmp_path / "density", "binned", sigma=columns(binned["sigma_uA_V2"]))
    assert report["verdict"] == FAIL
    # A left-handed lattice: PyATB's signed volume flips what it prints.
    left = read(runs["left"] / "queer_results.json")["shift_current"]
    report = judge(runs, tmp_path / "unflipped", "left", sigma=columns(left["sigma_uA_V2"]))
    assert report["verdict"] == FAIL


def test_the_inputs_and_the_candidate_file_are_checked(runs, tmp_path):
    reference = reference_from(tmp_path / "reference", runs["clean"])
    (tmp_path / "other.scfout").write_bytes(b"other")
    digests = reference / "inputs.sha256"
    digests.write_text(
        digests.read_text().replace(
            read(runs["clean"] / "queer_results.json")["model"]["scfout_sha256"],
            sha256(tmp_path / "other.scfout"),
        )
    )
    report = compare("SHIFT_CURRENT", reference, runs["clean"])
    assert report["verdict"] == NOT_COMPARABLE and "different scfout" in json.dumps(report["preconditions"])
    copy = shutil.copytree(runs["clean"], tmp_path / "edited")
    target = copy / "Out/Shift_Current/shift_current.dat"
    target.write_text(
        shift_current_text(
            read(copy / "queer_results.json")["shift_current"]["omega_eV"], scheme_table(copy) * 2
        )
    )
    report = compare("SHIFT_CURRENT", reference_from(tmp_path / "reference-2", copy), copy)
    assert report["verdict"] == FAIL and classes(report)["candidate_file"] == UNEXPLAINED


def test_failures_and_queers_limits(runs, tmp_path):
    refused = tmp_path / "refused"
    candidate(refused, shift_spinor_scfout, {}, 4, {**SPINOR, "scissors": -0.2})
    document = read(refused / "queer_results.json")
    assert document["status"] == "failed" and document["error"]["code"] == "queer_bad_params"
    reference = reference_from(tmp_path / "ok-reference", runs["clean"], echo={"scissors": -0.2})
    report = compare("SHIFT_CURRENT", reference, refused)
    assert report["verdict"] == NOT_COMPARABLE and "documented limit" in json.dumps(report["preconditions"])
    # Both runs refusing the same nspin conflict agree.
    conflict = tmp_path / "conflict"
    candidate(conflict, shift_spinor_scfout, {}, 1, SPINOR)
    assert read(conflict / "queer_results.json")["error"]["code"] == "queer_bad_params"
    failed = tmp_path / "failed-reference"
    reference_from(failed, runs["clean"], echo={})
    (failed / "Out/Shift_Current/shift_current.dat").unlink()
    for name in ("Input", "Out/input.json"):
        path = failed / name
        path.write_text(
            path.read_text()
            .replace('"nspin": 4', '"nspin": 1')
            .replace("nspin                           4", "nspin                           1")
        )
    (failed / "inputs.sha256").write_text(
        f"{read(conflict / 'queer_results.json')['provenance']['inputs']['scf.scfout']['sha256']}  scf.scfout\n"
    )
    (failed / "pyatb.out").write_text("ValueError: nspin=1 is inconsistent with scfout SpinP_switch=3\n")
    report = compare("SHIFT_CURRENT", failed, conflict)
    assert report["verdict"] == PASS and classes(report)["status"] == AGREE, render_markdown(report)


def test_the_release_clean_control_report_applies_to_every_comparison(runs, tmp_path):
    clean = judge(runs, tmp_path / "clean", "clean")
    assert clean["verdict"] == PASS
    failed = {**clean, "verdict": FAIL}
    report = compare(
        "SHIFT_CURRENT",
        reference_from(tmp_path / "ref", runs["binned"]),
        runs["binned"],
        clean_control=failed,
    )
    assert report["verdict"] == FAIL and components(report) == {UNEXPLAINED}
    assert classes(report)["clean_control_release"] == UNEXPLAINED
    path = tmp_path / "clean.json"
    write(path, clean)
    report = compare(
        "SHIFT_CURRENT",
        reference_from(tmp_path / "ref-2", runs["binned"]),
        runs["binned"],
        clean_control=path,
    )
    assert report["verdict"] == PASS and classes(report)["clean_control_release"] == AGREE
    other = json.loads(json.dumps(clean))
    other["candidate"]["queer_version"] = "0.0.0"
    report = compare(
        "SHIFT_CURRENT",
        reference_from(tmp_path / "ref-3", runs["binned"]),
        runs["binned"],
        clean_control=other,
    )
    assert classes(report)["clean_control_release"] == NOT_APPLICABLE and report["verdict"] == PASS
    with pytest.raises(ValueError, match="SHIFT_CURRENT comparison only"):
        compare("AHC", tmp_path, tmp_path, clean_control=clean)
    # The command passes the report through.
    arguments = ["compare", "--reference", str(reference_from(tmp_path / "ref-4", runs["binned"]))]
    arguments += ["--candidate", str(runs["binned"]), "--clean-control", str(path)]
    assert main(arguments) == 0


def test_the_symmetry_gate_ignores_thresholds_the_values_agree_across():
    rng = np.random.default_rng(3)
    reference = rng.normal(size=(30, 18))
    reference[:, 5] = 1.01e-4 * np.max(np.abs(reference)) * np.sin(np.arange(30))
    candidate = reference.copy()
    candidate[:, 5] *= 0.98
    ok, facts = symmetry_gate(reference, candidate, 0.1, np.full(18, 1e-4 * np.max(np.abs(reference))))
    assert ok and facts["components_differ"] == []
    ok, facts = symmetry_gate(reference, candidate, 0.1)
    assert not ok and facts["components_differ"] == ["xzz"]
    flipped = reference.copy()
    flipped[:, 3] *= -1
    ok, facts = symmetry_gate(reference, flipped, 0.1)
    assert not ok and facts["integral_sign_differs"] == ["xyy"]


def test_the_literature_anchor_checks_peak_position_and_height(runs):
    document = read(runs["clean"] / "queer_results.json")
    block = document["shift_current"]
    omega = np.asarray(block["omega_eV"])
    values = np.asarray(block["sigma_uA_V2"]["xyz"])
    index = int(np.argmax(np.abs(values)))
    # A synthetic anchor built from the candidate itself (not published data) exercises the rule.
    anchor = {
        "component": "xyz",
        "peak_eV": float(omega[index]) + 0.15,
        "height_uA_V2": float(values[index]) * 1.1,
        "scissors_eV": 0.0,
        "eta_eV": 0.1,
        "smearing": "gaussian",
    }
    assert literature_anchor(document, anchor)["status"] == "pass"
    assert (
        literature_anchor(runs["clean"], {**anchor, "peak_eV": float(omega[index]) + 0.25})["status"]
        == "fail"
    )
    assert (
        literature_anchor(document, {**anchor, "height_uA_V2": float(values[index]) * 1.3})["status"]
        == "fail"
    )
    assert literature_anchor(document, {**anchor, "scissors_eV": 1.0})["status"] == "not_comparable"
    assert literature_anchor(document, {**anchor, "eta_eV": 0.05})["status"] == "not_comparable"
    placeholder = literature_anchor(document, {**anchor, "peak_eV": None})
    assert placeholder["status"] == "placeholder" and placeholder["missing"] == ["peak_eV"]
