"""The OpenMX-family parity comparator: PyATB run directories against QUEER runs.

Candidates are real runner outputs on the analytic QWZ scfout; their references
are PyATB-layout directories holding the candidate's own numbers as PyATB would
print them, so every pair must pass and each negative control (one altered
quantity or input) must not. The committed PyATB outputs of the kagome
calibration are then judged against candidates carrying QUEER's calibration
values.
"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from openmx_files import collinear_scfout, qwz_spinor_scfout

from queer.compat.generator.parity import (
    AFTER_CONVENTION,
    AGREE,
    DOCUMENTED,
    FAIL,
    NOT_COMPARABLE,
    PASS,
    PASS_DOCUMENTED,
    UNEXPLAINED,
    compare,
    render_markdown,
)
from queer.compat.generator.parity_pyatb import is_pyatb_directory, openmx_family, pyatb_step
from queer.compat.generator.pyatb_conventions import HARTREE_EV, PYATB_HARTREE_EV, pyatb_energy
from queer.compat.generator.pyatb_files import ahc_scan_text, ahc_vector_text, chern_text
from queer.compat.generator.runner import main, run

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "pyatb"
SCALE = HARTREE_EV / PYATB_HARTREE_EV
KAGOME_SHA256 = "9309730433e6b54d439aad25ca3dbcc57eb75c705ad1684cdfb310be8b6ea0cb"
GRID = {"integrate_mode": "Grid", "integrate_grid": [12, 12, 1], "adaptive_grid": [1, 1, 1]}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=1))


def classes(report):
    return {item["name"]: item["class"] for item in report["quantities"]}


def copy(source, target):
    shutil.copytree(source, target, symlinks=True)
    return target


def candidate(
    directory, step="QUEER_AHC", inputs=None, scfout=qwz_spinor_scfout, self_convergence=None, **block
):
    """A QUEER run of an OpenMX step on an analytic scfout."""
    directory.mkdir(parents=True)
    scfout(directory / "scf.scfout")
    function = {"engine": "kubo", "method": 0, **GRID, "adaptive_grid_threshold": 1e30, **block}
    if step == "QUEER_CHERN_NUMBER":
        function.pop("engine")
        function.setdefault("occ_band", -1)
    params = {
        "step_type": step,
        "counterpart_step_type": step.replace("QUEER_", "PYATB_"),
        "input_parameters": {"nspin": 4, "fermi_energy": "Auto", **(inputs or {})},
        "parameters": function,
        "workers": 1,
    }
    if self_convergence is not None:
        params["self_convergence"] = self_convergence
    write(directory / "queer_params.json", params)
    assert run(None, directory / "queer_params.json", directory, log=lambda line: None) == 0
    return directory


def block_text(name, values):
    lines = [name, "{"]
    for key, value in values.items():
        text = " ".join(str(v) for v in value) if isinstance(value, list) else str(value)
        lines.append(f"    {key:<32}{text}")
    return "\n".join(lines + ["}", ""])


def reference_from(directory, run_dir, *, sigma=None, chern=None, digests=True):
    """A PyATB-layout run directory holding the candidate's own numbers as PyATB prints them."""
    document = read(run_dir / "queer_results.json")
    params = read(run_dir / "queer_params.json")
    family = "AHC" if document["step_type"] == "QUEER_AHC" else "CHERN_NUMBER"
    inputs = dict(params["input_parameters"])
    function = dict(params["parameters"])
    out = directory / "Out"
    out.mkdir(parents=True)
    text = block_text("INPUT_PARAMETERS", {"package": "OPENMX", "scfout_route": "scf.scfout", **inputs})
    text += "\n" + block_text("LATTICE", {"lattice_constant": 1.0, "lattice_constant_unit": "Angstrom"})
    text += "\n" + block_text(family, function)
    (directory / "Input").write_text(text)
    echo_inputs = {**inputs, "package": "OPENMX", "scfout_route": "scf.scfout", "max_kpoint_num": 8000}
    echo_inputs["hall_convention"] = {"auto": "openmx"}.get(
        inputs.get("hall_convention", "auto"), inputs.get("hall_convention", "auto")
    )
    write(out / "input.json", {"INPUT_PARAMETERS": echo_inputs, family: function})
    fermi = pyatb_energy(document["model"]["fermi_energy"])
    log = [
        "program start",
        f" >> fermi_energy            : {fermi:<15.6f}",
        "!!The number of rough k-grid : 144",
        "!!The number of k points for adaptive refinement : 0",
    ]
    if family == "AHC":
        block = document["ahc"]
        sign = block["convention"]["export_sign"]
        values = np.column_stack([block["sigma_S_cm"][name] for name in ("yz", "zx", "xy")]) * sign
        if block.get("legacy_single_channel"):
            legacy = block["legacy_single_channel"]["sigma_S_cm"]
            values = np.column_stack([legacy[name] for name in ("yz", "zx", "xy")]) * sign
        if sigma is not None:
            values = np.atleast_2d(sigma)
        product = (
            ahc_scan_text(block["energy_offsets_eV"], values)
            if block["engine"] == "fhs"
            else ahc_vector_text(values[0])
        )
        (out / "AHC").mkdir()
        (out / "AHC" / "ahc.dat").write_text(product)
    else:
        value = document["chern"]["value"] if chern is None else chern
        (out / "Chern_Num").mkdir()
        (out / "Chern_Num" / "chern_number.dat").write_text(chern_text(value, "Grid"))
    (out / "running.log").write_text("\n".join(log + ["Total time : 0 h 0 m 1 s"]) + "\n")
    if digests:
        scfout = document["model"]["scfout_sha256"]
        (directory / "inputs.sha256").write_text(
            f"{scfout}  scf.scfout\n{sha256(directory / 'Input')}  Input\n"
        )
    return directory


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("pyatb-parity")
    return {
        "openmx": candidate(base / "openmx", inputs={"hall_convention": "openmx"}),
        "pyatb": candidate(base / "pyatb", inputs={"hall_convention": "pyatb", "fermi_energy": 0.1}),
        "chern": candidate(base / "chern", "QUEER_CHERN_NUMBER"),
        "metal": candidate(base / "metal", "QUEER_CHERN_NUMBER", inputs={"fermi_energy": -2.0}),
        "fhs": candidate(
            base / "fhs",
            engine="fhs",
            fhs_grid=[8, 8, 1],
            fhs_energy_window=[-2.5, 0.5],
            fhs_energy_points=6,
        ),
        "refining": candidate(
            base / "refining",
            adaptive_grid=[2, 2, 1],
            adaptive_grid_threshold=0.0,
            integrate_grid=[6, 6, 1],
            self_convergence=True,
        ),
        "unconverged": candidate(
            base / "unconverged",
            adaptive_grid=[2, 2, 1],
            adaptive_grid_threshold=0.0,
            integrate_grid=[6, 6, 1],
        ),
        "collinear": candidate(
            base / "collinear", inputs={"nspin": 1, "fermi_energy": 0.0}, scfout=collinear_scfout
        ),
    }


@pytest.mark.parametrize(
    ("name", "kind"),
    [("openmx", AGREE), ("pyatb", AFTER_CONVENTION), ("chern", AGREE), ("metal", AGREE), ("fhs", AGREE)],
)
def test_a_reference_printing_the_candidates_numbers_passes(runs, tmp_path, name, kind):
    reference = reference_from(tmp_path / "reference", runs[name])
    step = "PYATB_AHC" if name not in {"chern", "metal"} else "PYATB_CHERN_NUMBER"
    report = compare(step, reference, runs[name])
    assert report["verdict"] == PASS, render_markdown(report)
    compared = {c for n, c in classes(report).items() if n.startswith(("sigma", "chern"))}
    assert compared == {kind}
    assert classes(report)["candidate_file"] == AGREE and classes(report)["fermi_energy"] in {
        AGREE,
        AFTER_CONVENTION,
    }
    assert report["schema"] == "queer-generator-parity/1" and report["reference"]["engine"] == "pyatb"
    assert "## Preconditions" in render_markdown(report)


def test_the_single_channel_of_a_collinear_run_is_compared_after_convention(runs, tmp_path):
    reference = reference_from(tmp_path / "reference", runs["collinear"])
    report = compare("AHC", reference, runs["collinear"])
    assert report["verdict"] == PASS, render_markdown(report)
    assert report["semantics"]["rules"]["legacy_single_channel"]
    assert {c for n, c in classes(report).items() if n.startswith("sigma")} == {AFTER_CONVENTION}


def test_a_refining_grid_is_documented_within_its_error_estimate(runs, tmp_path):
    document = read(runs["refining"] / "queer_results.json")
    physical = document["ahc"]["sigma_S_cm"]["xy"][0]
    e_q = document["integration"]["self_convergence"]["e_Q"][2]
    assert e_q > 1e-6
    shifted = [0.0, 0.0, physical + 2 * e_q]
    report = compare(
        "AHC", reference_from(tmp_path / "near", runs["refining"], sigma=shifted), runs["refining"]
    )
    assert report["verdict"] == PASS_DOCUMENTED and classes(report)["sigma_xy"] == DOCUMENTED
    assert report["tolerance_basis"]["tier"] == "B"
    far = [0.0, 0.0, physical + 4 * e_q]
    report = compare("AHC", reference_from(tmp_path / "far", runs["refining"], sigma=far), runs["refining"])
    assert report["verdict"] == FAIL and classes(report)["sigma_xy"] == UNEXPLAINED
    report = compare("AHC", reference_from(tmp_path / "plain", runs["unconverged"]), runs["unconverged"])
    assert report["verdict"] == NOT_COMPARABLE
    assert "self_convergence" in [item for item in report["preconditions"] if not item["ok"]][0]["detail"]


def _preconditions(report):
    return {item["name"] for item in report["preconditions"] if not item["ok"]}


def test_negative_controls_do_not_pass(runs, tmp_path):
    good = reference_from(tmp_path / "good", runs["pyatb"])
    assert compare("AHC", good, runs["pyatb"])["verdict"] == PASS

    document = read(runs["pyatb"] / "queer_results.json")
    physical = [document["ahc"]["sigma_S_cm"][name][0] for name in ("yz", "zx", "xy")]
    flipped = compare(
        "AHC", reference_from(tmp_path / "flipped", runs["pyatb"], sigma=physical), runs["pyatb"]
    )
    assert flipped["verdict"] == FAIL and classes(flipped)["sigma_xy"] == UNEXPLAINED

    shifted = copy(good, tmp_path / "fermi")
    echo = read(shifted / "Out/input.json")
    echo["INPUT_PARAMETERS"]["fermi_energy"] += 1e-4
    write(shifted / "Out/input.json", echo)
    report = compare("AHC", shifted, runs["pyatb"])
    assert report["verdict"] == NOT_COMPARABLE and _preconditions(report) == {"fermi_energy"}

    regridded = copy(good, tmp_path / "grid")
    echo = read(regridded / "Out/input.json")
    echo["AHC"]["integrate_grid"] = [14, 12, 1]
    write(regridded / "Out/input.json", echo)
    report = compare("AHC", regridded, runs["pyatb"])
    assert report["verdict"] == NOT_COMPARABLE and _preconditions(report) == {"parameters"}
    assert "integrate_grid" in [item for item in report["preconditions"] if not item["ok"]][0]["detail"]

    other = copy(good, tmp_path / "scfout")
    (other / "inputs.sha256").write_text(f"{'0' * 64}  scf.scfout\n{sha256(other / 'Input')}  Input\n")
    report = compare("AHC", other, runs["pyatb"])
    assert report["verdict"] == NOT_COMPARABLE and _preconditions(report) == {"inputs"}

    convention = copy(good, tmp_path / "convention")
    echo = read(convention / "Out/input.json")
    echo["INPUT_PARAMETERS"]["hall_convention"] = "openmx"
    write(convention / "Out/input.json", echo)
    report = compare("AHC", convention, runs["pyatb"])
    assert report["verdict"] == NOT_COMPARABLE and _preconditions(report) == {"parameters"}
    assert "hall_convention: 'openmx' vs 'pyatb'" in report["preconditions"][3]["detail"]
    claimed = copy(runs["pyatb"], tmp_path / "claimed")
    document = read(claimed / "queer_results.json")
    document["ahc"]["convention"]["resolved"] = "openmx"
    write(claimed / "queer_results.json", document)
    report = compare("AHC", good, claimed)
    assert report["verdict"] == NOT_COMPARABLE and _preconditions(report) == {"hall_convention"}

    chern = read(runs["chern"] / "queer_results.json")["chern"]["value"]
    report = compare(
        "CHERN_NUMBER", reference_from(tmp_path / "chern", runs["chern"], chern=chern + 1), runs["chern"]
    )
    assert report["verdict"] == FAIL and classes(report)["chern_number"] == UNEXPLAINED

    metal = read(runs["metal"] / "queer_results.json")["chern"]["value"]
    report = compare(
        "CHERN_NUMBER", reference_from(tmp_path / "m", runs["metal"], chern=metal + 1e-6), runs["metal"]
    )
    assert report["verdict"] == FAIL and classes(report)["chern_number"] == UNEXPLAINED


def test_inputs_are_proven_by_digests_or_by_the_named_scfout(runs, tmp_path):
    bare = reference_from(tmp_path / "bare", runs["openmx"], digests=False)
    report = compare("AHC", bare, runs["openmx"])
    assert report["verdict"] == NOT_COMPARABLE and "no inputs.sha256" in report["preconditions"][2]["detail"]
    assert compare("AHC", bare, runs["openmx"], scfout=runs["openmx"] / "scf.scfout")["verdict"] == PASS
    edited = reference_from(tmp_path / "edited", runs["openmx"])
    (edited / "Input").write_text((edited / "Input").read_text() + "\n")
    report = compare("AHC", edited, runs["openmx"])
    assert (
        report["verdict"] == NOT_COMPARABLE
        and "changed after the run" in report["preconditions"][2]["detail"]
    )


def test_an_adaptive_integral_from_few_cells_is_not_comparable(runs, tmp_path):
    reference = reference_from(tmp_path / "adaptive", runs["openmx"])
    echo = read(reference / "Out/input.json")
    echo["AHC"].update(integrate_mode="Adaptive", initial_grid=[1, 1, 1])
    write(reference / "Out/input.json", echo)
    candidate_dir = copy(runs["openmx"], tmp_path / "candidate")
    params = read(candidate_dir / "queer_params.json")
    params["parameters"].update(integrate_mode="Adaptive", initial_grid=[1, 1, 1])
    write(candidate_dir / "queer_params.json", params)
    report = compare("AHC", reference, candidate_dir)
    assert report["verdict"] == NOT_COMPARABLE and "fewer than 512" in report["preconditions"][-1]["detail"]


def test_failures_pass_only_when_both_runs_failed_alike(runs, tmp_path):
    reference = reference_from(tmp_path / "reference", runs["openmx"])
    (reference / "Out/AHC/ahc.dat").unlink()
    (reference / "pyatb.out").write_text(
        "ValueError: nspin=1 is inconsistent with scfout SpinP_switch=3 (expected nspin=4)\n"
    )
    failed = copy(runs["openmx"], tmp_path / "failed")
    document = read(failed / "queer_results.json")
    document.update(status="failed", error={"code": "queer_bad_params", "message": "nspin conflicts"})
    write(failed / "queer_results.json", document)
    report = compare("AHC", reference, failed)
    assert report["verdict"] == PASS and classes(report) == {"status": AGREE}
    report = compare("AHC", reference, runs["openmx"])
    assert report["verdict"] == FAIL and classes(report) == {"status": UNEXPLAINED}
    document["error"]["code"] = "queer_oom"
    write(failed / "queer_results.json", document)
    assert compare("AHC", reference, failed)["verdict"] == FAIL


def test_a_generator_workspace_failure_is_read_from_slurms_logs(runs, tmp_path):
    # The Generator's run.sh launches pyatb without redirection: its traceback lands
    # in slurm-<job>.err and nothing reaches Out/running.log or a pyatb.out.
    reference = reference_from(tmp_path / "reference", runs["openmx"])
    (reference / "Out/AHC/ahc.dat").unlink()
    (reference / "run.sh").write_text("#!/bin/bash\n#SBATCH --error=slurm-%j.err\nmpirun -n 4 pyatb\n")
    (reference / "slurm-4242.out").write_text("PyATB 60fc6b2a\nreading the scfout\n")
    (reference / "slurm-4242.err").write_text(
        'Traceback (most recent call last):\n  File "main.py", line 37, in main\n'
        "ValueError: nspin=1 is inconsistent with scfout SpinP_switch=3 (expected nspin=4)\n"
    )
    failed = copy(runs["openmx"], tmp_path / "failed")
    document = read(failed / "queer_results.json")
    document.update(status="failed", error={"code": "queer_bad_params", "message": "nspin conflicts"})
    write(failed / "queer_results.json", document)
    report = compare("AHC", reference, failed)
    assert report["verdict"] == PASS and classes(report) == {"status": AGREE}, render_markdown(report)
    (reference / "slurm-4242.err").write_text("srun: error: task 0: Out Of Memory\noom-kill event\n")
    document["error"] = {"code": "queer_oom", "message": "out of memory"}
    write(failed / "queer_results.json", document)
    assert compare("AHC", reference, failed)["verdict"] == PASS


def test_a_capability_refusal_is_not_comparable_and_other_failures_fail(runs, tmp_path):
    reference = reference_from(tmp_path / "reference", runs["openmx"])
    refused = copy(runs["openmx"], tmp_path / "refused")
    document = read(refused / "queer_results.json")
    cases = (
        ("queer_validation_failed", "QUEER refines a Chern-number integral only on a gapped plane", None),
        ("queer_bad_params", "the doubled grid exceeds QUEER's integration budget of 10 cells", None),
        ("queer_bad_params", "Invalid PYATB_AHC settings: method must be 0 (direct) or 1 (Kubo)", FAIL),
        ("queer_internal_error", "RuntimeError: boom", FAIL),
    )
    for code, message, verdict in cases:
        document.update(status="failed", error={"code": code, "message": message})
        write(refused / "queer_results.json", document)
        report = compare("AHC", reference, refused)
        if verdict is None:
            assert report["verdict"] == NOT_COMPARABLE and _preconditions(report) == {"capability"}
            assert message in report["preconditions"][-1]["detail"]
        else:
            assert report["verdict"] == verdict and classes(report) == {"status": UNEXPLAINED}
    # A failed candidate given other settings than the reference is not comparable, never a fail.
    params = read(refused / "queer_params.json")
    params["parameters"]["integrate_grid"] = [14, 12, 1]
    write(refused / "queer_params.json", params)
    report = compare("AHC", reference, refused)
    assert report["verdict"] == NOT_COMPARABLE and _preconditions(report) == {"parameters"}


def test_a_budget_refusal_of_a_real_run_is_not_comparable(runs, tmp_path, monkeypatch):
    import queer.compat.generator.openmx_steps as steps

    # 256 KiB holds 76 cells: the 6x6 grid fits, its 12x12 self-convergence double does not.
    monkeypatch.setattr(steps, "available_memory_bytes", lambda: 256 * 1024)
    refused = tmp_path / "refused"
    refused.mkdir()
    qwz_spinor_scfout(refused / "scf.scfout")
    write(refused / "queer_params.json", read(runs["refining"] / "queer_params.json"))
    assert run(None, refused / "queer_params.json", refused, log=lambda line: None) == 2
    assert read(refused / "queer_results.json")["error"]["code"] == "queer_bad_params"
    report = compare("AHC", reference_from(tmp_path / "reference", runs["refining"]), refused)
    assert report["verdict"] == NOT_COMPARABLE and _preconditions(report) == {"capability"}, render_markdown(
        report
    )


def test_a_metallic_chern_refinement_refused_by_queer_is_not_comparable(runs, tmp_path):
    directory = tmp_path / "metal-refining"
    directory.mkdir()
    qwz_spinor_scfout(directory / "scf.scfout")
    function = {
        "method": 0,
        "occ_band": -1,
        **GRID,
        "adaptive_grid": [2, 2, 1],
        "adaptive_grid_threshold": 0.0,
    }
    params = {
        "step_type": "QUEER_CHERN_NUMBER",
        "counterpart_step_type": "PYATB_CHERN_NUMBER",
        "input_parameters": {"nspin": 4, "fermi_energy": -2.0},
        "parameters": function,
        "workers": 1,
        "self_convergence": True,
    }
    write(directory / "queer_params.json", params)
    assert run(None, directory / "queer_params.json", directory, log=lambda line: None) == 2
    document = read(directory / "queer_results.json")
    assert document["error"]["code"] == "queer_validation_failed"
    # PyATB prints the Fermi-sea flux of the metallic plane.
    reference = reference_from(tmp_path / "reference", runs["metal"])
    echo = read(reference / "Out/input.json")
    echo["CHERN_NUMBER"].update(adaptive_grid=[2, 2, 1], adaptive_grid_threshold=0.0)
    write(reference / "Out/input.json", echo)
    report = compare("CHERN_NUMBER", reference, directory)
    assert report["verdict"] == NOT_COMPARABLE and _preconditions(report) == {"capability"}, render_markdown(
        report
    )


def _rehash(reference):
    """Record the reference's current Input in its inputs.sha256."""
    lines = (reference / "inputs.sha256").read_text().splitlines()
    kept = [line for line in lines if not line.endswith("  Input")]
    (reference / "inputs.sha256").write_text(
        "\n".join([*kept, f"{sha256(reference / 'Input')}  Input"]) + "\n"
    )


def test_an_echo_without_hall_convention_is_pyatbs_native_sign(runs, tmp_path):
    # A PyATB build older than hall_convention printed its native sign and echoed no key.
    reference = reference_from(tmp_path / "reference", runs["pyatb"])
    echo = read(reference / "Out/input.json")
    del echo["INPUT_PARAMETERS"]["hall_convention"]
    write(reference / "Out/input.json", echo)
    text = (reference / "Input").read_text().splitlines()
    (reference / "Input").write_text("\n".join(line for line in text if "hall_convention" not in line) + "\n")
    _rehash(reference)
    report = compare("AHC", reference, runs["pyatb"])
    assert report["verdict"] == PASS and classes(report)["sigma_xy"] == AFTER_CONVENTION, render_markdown(
        report
    )
    detail = [item for item in report["preconditions"] if item["name"] == "hall_convention"][0]["detail"]
    assert "older than the key" in detail and "factor -1" in detail
    # A candidate built from the key-less Input file (auto, so openmx) is not comparable, never a fail.
    auto = copy(runs["pyatb"], tmp_path / "auto")
    params = read(auto / "queer_params.json")
    params["input_parameters"]["hall_convention"] = "auto"
    write(auto / "queer_params.json", params)
    report = compare("AHC", reference, auto)
    assert report["verdict"] == NOT_COMPARABLE and _preconditions(report) == {"parameters"}
    assert "hall_convention: 'pyatb' vs 'openmx'" in report["preconditions"][3]["detail"]


def test_text_values_in_the_candidates_parameter_file_compare_as_the_runner_reads_them(runs, tmp_path):
    reference = reference_from(tmp_path / "reference", runs["openmx"])
    textual = copy(runs["openmx"], tmp_path / "textual")
    params = read(textual / "queer_params.json")
    params["parameters"].update(
        integrate_grid="12 12 1", adaptive_grid="1 1 1", adaptive_grid_threshold="1e30", method="0"
    )
    params["input_parameters"]["nspin"] = "4"
    write(textual / "queer_params.json", params)
    report = compare("AHC", reference, textual)
    assert report["verdict"] == PASS, render_markdown(report)


def left_handed_qwz_scfout(path):
    """The QWZ spinor model with a3 reversed: the same bands on a left-handed lattice."""
    from openmx_files import SQUARE_TRANSLATIONS, write_scfout_model

    uu = np.array([1.0, 0.5, 0.5, 0.5, 0.5])
    up_down_real = np.array([0.0, 0.0, 0.0, -0.5, 0.5])
    up_down_imaginary = np.array([0.0, -0.5, 0.5, 0.0, 0.0])
    hamiltonian = np.stack((uu, -uu, up_down_real, up_down_imaginary))[:, :, None, None]
    return write_scfout_model(
        path,
        lattice=np.diag([2.0, 2.0, -10.0]),
        translations=SQUARE_TRANSLATIONS,
        hamiltonian=hamiltonian,
        spin=3,
    )


def test_a_left_handed_lattice_flips_pyatbs_direct_and_kubo_sign(runs, tmp_path):
    left = candidate(tmp_path / "left", inputs={"hall_convention": "openmx"}, scfout=left_handed_qwz_scfout)
    document = read(left / "queer_results.json")
    right = read(runs["openmx"] / "queer_results.json")
    assert document["model"]["lattice_handedness"] == -1 and right["model"]["lattice_handedness"] == 1
    # The physical conductivity does not depend on the lattice's handedness ...
    assert document["ahc"]["sigma_S_cm"]["xy"][0] == pytest.approx(
        right["ahc"]["sigma_S_cm"]["xy"][0], rel=1e-12
    )
    # ... but PyATB's signed reciprocal volume flips what it prints, and the file follows it.
    convention = document["ahc"]["convention"]
    assert convention["sign_table"] == 1 and convention["export_sign"] == -1
    physical = [document["ahc"]["sigma_S_cm"][name][0] for name in ("yz", "zx", "xy")]
    assert (left / "Out/AHC/ahc.dat").read_text() == ahc_vector_text([-value for value in physical])
    report = compare("AHC", reference_from(tmp_path / "pyatb-like", left), left)
    assert report["verdict"] == PASS and classes(report)["sigma_xy"] == AFTER_CONVENTION, render_markdown(
        report
    )
    unsigned = compare("AHC", reference_from(tmp_path / "unsigned", left, sigma=physical), left)
    assert unsigned["verdict"] == FAIL and classes(unsigned)["sigma_xy"] == UNEXPLAINED


def test_a_candidate_file_disagreeing_with_its_document_fails(runs, tmp_path):
    reference = reference_from(tmp_path / "reference", runs["openmx"])
    tampered = copy(runs["openmx"], tmp_path / "tampered")
    (tampered / "Out/AHC/ahc.dat").write_text(ahc_vector_text([0.0, 0.0, 1.0]))
    report = compare("AHC", reference, tampered)
    assert report["verdict"] == FAIL and classes(report)["candidate_file"] == UNEXPLAINED


def test_a_value_on_a_rounding_boundary_still_matches_its_six_decimals(runs, tmp_path):
    halfway = copy(runs["openmx"], tmp_path / "halfway")
    document = read(halfway / "queer_results.json")
    document["ahc"]["sigma_S_cm"] = {"yz": [0.0], "zx": [0.0], "xy": [1.5e-6]}
    write(halfway / "queer_results.json", document)
    (halfway / "Out/AHC/ahc.dat").write_text(ahc_vector_text([0.0, 0.0, 1.5e-6]))
    assert "0.000002" in (halfway / "Out/AHC/ahc.dat").read_text()
    report = compare("AHC", reference_from(tmp_path / "reference", halfway), halfway)
    assert classes(report)["candidate_file"] == AGREE


def template(runs, directory, fixture, *, sigma=None, chern=None, offsets=None):
    """A candidate run of the kagome calibration: a runner document carrying QUEER's calibration values."""
    family = "chern" if chern is not None else "ahc"
    source = runs["chern"] if family == "chern" else runs["fhs" if offsets is not None else "openmx"]
    directory = copy(source, directory)
    blocks = json.loads((FIXTURES / fixture / "Out/input.json").read_text())
    from queer.compat.generator.openmx_params import params_from_pyatb_input
    from queer.compat.generator.pyatb_files import read_input

    params = params_from_pyatb_input(read_input(FIXTURES / fixture / "Input"), workers=1)
    write(directory / "queer_params.json", params)
    document = read(directory / "queer_results.json")
    document["model"].update(
        scfout_sha256=KAGOME_SHA256, fermi_energy=blocks["INPUT_PARAMETERS"]["fermi_energy"] * SCALE
    )
    document["provenance"]["inputs"]["scf.scfout"]["sha256"] = KAGOME_SHA256
    if family == "ahc":
        convention = blocks["INPUT_PARAMETERS"]["hall_convention"]
        sign = -1 if convention == "pyatb" and offsets is None else 1
        document["ahc"]["convention"].update(requested=convention, resolved=convention, export_sign=sign)
        rows = np.atleast_2d(sigma)
        document["ahc"]["sigma_S_cm"] = {
            name: rows[:, axis].tolist() for axis, name in enumerate(("yz", "zx", "xy"))
        }
        if offsets is not None:
            document["ahc"]["energy_offsets_eV"] = offsets
            text = ahc_scan_text(offsets, rows * sign)
        else:
            text = ahc_vector_text(rows[0] * sign)
        (directory / "Out/AHC/ahc.dat").write_text(text)
    else:
        document["chern"]["value"] = chern
        (directory / "Out/Chern_Num/chern_number.dat").write_text(chern_text(chern, "Grid"))
    write(directory / "queer_results.json", document)
    return directory


def calibration_reference(fixture, directory):
    reference = copy(FIXTURES / fixture, directory)
    (reference / "inputs.sha256").write_text(f"{KAGOME_SHA256}  ../test.scfout\n")
    return reference


def test_the_kagome_calibration_outputs_pass_against_queers_values(runs, tmp_path):
    direct = [1.4207299834736195e-16, -5.784907531812258e-15, 38.74046146463808]
    for fixture, kind in (("c_direct_openmx", AGREE), ("b_direct_pyatb", AFTER_CONVENTION)):
        reference = calibration_reference(fixture, tmp_path / f"{fixture}-reference")
        candidate_dir = template(runs, tmp_path / fixture, fixture, sigma=direct)
        report = compare("QUEER_AHC", reference, candidate_dir)
        assert report["verdict"] == PASS, render_markdown(report)
        assert classes(report)["sigma_xy"] == kind
        ratio = {entry["name"]: entry["ratio"] for entry in report["summary"]["worst_relative_to_tolerance"]}
        assert 0.5 < ratio["sigma_xy"] < 1.0
    reference = calibration_reference("h_chern_fermi", tmp_path / "chern-reference")
    candidate_dir = template(runs, tmp_path / "chern", "h_chern_fermi", chern=-0.9999999999999989)
    report = compare("CHERN_NUMBER", reference, candidate_dir)
    assert report["verdict"] == PASS and classes(report)["chern_number"] == AGREE
    candidate_dir = template(runs, tmp_path / "chern-plus", "h_chern_fermi", chern=0.0000000000000011)
    assert compare("CHERN_NUMBER", reference, candidate_dir)["verdict"] == FAIL


def test_an_fhs_table_agrees_point_by_point_or_is_documented(runs, tmp_path):
    from queer.compat.generator.pyatb_files import parse_ahc

    table = parse_ahc((FIXTURES / "a_fhs_scan/Out/AHC/ahc.dat").read_text())
    reference = calibration_reference("a_fhs_scan", tmp_path / "reference")
    rows = np.asarray(table["sigma"])
    same = template(runs, tmp_path / "same", "a_fhs_scan", sigma=rows, offsets=table["energy_eV"])
    report = compare("AHC", reference, same)
    assert report["verdict"] == PASS and classes(report)["sigma_xy(E)"] == AGREE
    moved = rows.copy()
    moved[150:, 2] *= 0.996
    differing = template(runs, tmp_path / "differing", "a_fhs_scan", sigma=moved, offsets=table["energy_eV"])
    report = compare("AHC", reference, differing)
    assert report["verdict"] == PASS_DOCUMENTED and classes(report)["sigma_xy(E)"] == DOCUMENTED
    detail = [item for item in report["quantities"] if item["name"] == "sigma_xy(E)"][0]["detail"]
    assert "correlation" in detail and "beyond Tier A" in detail
    shifted = template(
        runs, tmp_path / "shifted", "a_fhs_scan", sigma=rows, offsets=[e + 1e-3 for e in table["energy_eV"]]
    )
    assert classes(compare("AHC", reference, shifted))["energies"] == UNEXPLAINED


def test_step_names_and_reference_detection():
    assert openmx_family("QUEER_AHC") == "AHC" and openmx_family("pyatb_chern_number") == "CHERN_NUMBER"
    assert openmx_family("QUEER_BANDS") is None and openmx_family("AHC") == "AHC"
    assert is_pyatb_directory(FIXTURES / "c_direct_openmx") and not is_pyatb_directory(FIXTURES)
    assert pyatb_step(FIXTURES / "h_chern_fermi") == "PYATB_CHERN_NUMBER"
    with pytest.raises(ValueError, match="AHC or CHERN_NUMBER"):
        from queer.compat.generator.parity_pyatb import compare as compare_pyatb

        compare_pyatb("SHIFT_CURRENT", FIXTURES, FIXTURES)


def test_the_compare_command_detects_a_pyatb_reference(runs, tmp_path, capsys):
    reference = reference_from(tmp_path / "reference", runs["openmx"], digests=False)
    report, markdown = tmp_path / "report.json", tmp_path / "report.md"
    arguments = ["compare", "--reference", str(reference), "--candidate", str(runs["openmx"])]
    code = main(
        [
            *arguments,
            "--scfout",
            str(runs["openmx"] / "scf.scfout"),
            "--report",
            str(report),
            "--markdown",
            str(markdown),
        ]
    )
    assert code == 0 and capsys.readouterr().out.startswith("pass")
    assert read(report)["step"] == "AHC" and markdown.read_text().startswith("# Parity of AHC: pass")
    assert main(arguments) == 2
    assert "precondition inputs" in capsys.readouterr().out
