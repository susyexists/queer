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


def test_a_refining_grid_is_judged_with_both_codes_error_estimates(runs, tmp_path):
    document = read(runs["refining"] / "queer_results.json")
    emulation = document["ahc"]["pyatb_emulation"]["refined_grid"]
    # threshold 0 refines every node: PyATB's sum is then its 2 x 2 sub-grids of every node.
    assert emulation["flagged"] == 36 and emulation["flagged_doubled"] == 144
    assert emulation["grid"] == [6, 6, 1] and emulation["doubled_grid"] == [12, 12, 1]
    printed = emulation["printed_S_cm"]
    e_p = emulation["e_P_S_cm"][2]
    assert e_p == pytest.approx(abs(printed[2] - emulation["printed_doubled_S_cm"][2]))
    # PyATB printing its own refined sum (as QUEER emulates it) is documented within 3 max(e_Q, e_P).
    report = compare(
        "AHC", reference_from(tmp_path / "pyatb", runs["refining"], sigma=printed), runs["refining"]
    )
    assert report["verdict"] in {PASS, PASS_DOCUMENTED}, render_markdown(report)
    assert classes(report)["pyatb_refined_grid"] == AGREE and report["tolerance_basis"]["tier"] == "B"
    assert report["tolerance_basis"]["e_P"] == pytest.approx(emulation["e_P_S_cm"])
    assert "emulation" in report["tolerance_basis"]["e_P_source"]
    # A reference the emulation does not reproduce is unexplained, whatever the error estimates.
    off = [printed[0], printed[1], printed[2] + 10 * max(e_p, 1e-3)]
    report = compare("AHC", reference_from(tmp_path / "off", runs["refining"], sigma=off), runs["refining"])
    assert report["verdict"] == FAIL and classes(report)["pyatb_refined_grid"] == UNEXPLAINED
    assert report["tolerance_basis"]["e_P"] is None
    # Without self_convergence neither e_Q nor e_P is known: not comparable, with the way out.
    report = compare("AHC", reference_from(tmp_path / "plain", runs["unconverged"]), runs["unconverged"])
    assert report["verdict"] == NOT_COMPARABLE and _preconditions(report) == {"integration"}
    detail = [item for item in report["preconditions"] if not item["ok"]][0]["detail"]
    assert "adaptive_grid_threshold 1e30" in detail and "self_convergence" in detail and "e_P" in detail


@pytest.mark.parametrize(("e_q", "e_p", "verdict"), [(1e-9, 0.5, PASS_DOCUMENTED), (1e-9, 0.1, FAIL)])
def test_tier_b_bounds_the_gap_by_both_codes_discretisation_errors(runs, tmp_path, e_q, e_p, verdict):
    # A correct refined run whose gap to PyATB is PyATB's own discretisation error: QUEER's
    # e_Q alone cannot bound it, PyATB's e_P (from QUEER's emulation of its sum) can.
    candidate_dir = copy(runs["refining"], tmp_path / "candidate")
    document = read(candidate_dir / "queer_results.json")
    physical = document["ahc"]["sigma_S_cm"]["xy"][0]
    printed = [0.0, 0.0, physical + 1.2]
    document["integration"]["self_convergence"]["e_Q"] = [e_q, e_q, e_q]
    emulation = document["ahc"]["pyatb_emulation"]["refined_grid"]
    emulation.update(printed_S_cm=printed, e_P_S_cm=[e_p, e_p, e_p])
    write(candidate_dir / "queer_results.json", document)
    report = compare(
        "AHC", reference_from(tmp_path / "reference", runs["refining"], sigma=printed), candidate_dir
    )
    assert report["verdict"] == verdict, render_markdown(report)
    assert classes(report)["sigma_xy"] == (DOCUMENTED if verdict == PASS_DOCUMENTED else UNEXPLAINED)
    assert classes(report)["pyatb_refined_grid"] == AGREE


def _preconditions(report):
    return {item["name"] for item in report["preconditions"] if not item["ok"]}


def near_gap_scfout(path):
    """QWZ with its gap almost closed at M: the two levels sit 3e-5 eV either side of E_F = 0 there.

    A weak spin-orbit coupling spreads the curvature, so the M node holds a small share of sigma.
    """
    return qwz_spinor_scfout(path, mass=2.0 - 3e-5, coupling=1e-4)


NEAR_GAP_GRID = {"integrate_grid": [24, 24, 1]}


def _scaled(directory, factor):
    """A copy of a candidate whose document and file hold ``factor`` times its conductivity."""
    target = copy(directory, directory.parent / f"{directory.name}-x{factor}")
    document = read(target / "queer_results.json")
    for name in ("yz", "zx", "xy"):
        document["ahc"]["sigma_S_cm"][name] = [
            factor * value for value in document["ahc"]["sigma_S_cm"][name]
        ]
    write(target / "queer_results.json", document)
    sign = document["ahc"]["convention"]["export_sign"]
    values = [sign * document["ahc"]["sigma_S_cm"][name][0] for name in ("yz", "zx", "xy")]
    (target / "Out/AHC/ahc.dat").write_text(ahc_vector_text(values))
    return target


def test_pyatbs_kubo_engine_has_no_pair_cut_so_tier_a_is_not_widened(tmp_path):
    kubo = candidate(
        tmp_path / "kubo",
        inputs={"hall_convention": "openmx"},
        scfout=near_gap_scfout,
        method=1,
        **NEAR_GAP_GRID,
    )
    document = read(kubo / "queer_results.json")
    # The 6e-5 eV pair at M is not flagged for the Kubo engine: PyATB's Kubo kernel has no cut.
    assert document["ahc"]["flagged"]["count"] == 0 and document["ahc"]["flagged"]["pair_gap_eV"] is None
    reference = reference_from(tmp_path / "reference", kubo)
    assert compare("AHC", reference, kubo)["verdict"] == PASS
    for factor in (0.5, 0.0, 1.9, -1.0):
        report = compare("AHC", reference, _scaled(kubo, factor))
        assert report["verdict"] == FAIL and classes(report)["sigma_xy"] == UNEXPLAINED, factor


def test_pyatbs_direct_pair_cut_is_an_exact_documented_allowance(tmp_path):
    direct = candidate(
        tmp_path / "direct", inputs={"hall_convention": "openmx"}, scfout=near_gap_scfout, **NEAR_GAP_GRID
    )
    document = read(direct / "queer_results.json")
    flagged, listed = document["ahc"]["flagged"], document["ahc"]["per_k_flagged"]
    assert flagged["count"] == 1 and listed[0]["k"] == [0.5, 0.5, 0.0]
    assert listed[0]["reasons"] == ["pyatb_pair_cut"] and listed[0]["pair_gap_eV"] < 1e-4
    point = listed[0]
    # With the only pair cut, PyATB's direct engine sees no curvature at M: c_pyatb_rule is ~0.
    assert point["c_pyatb_rule"][2] == pytest.approx(0.0, abs=1e-6 * abs(point["c"][2]))
    assert point["allowance"][2] == pytest.approx(abs(point["c"][2] - point["c_pyatb_rule"][2]), rel=1e-12)
    assert flagged["pair_shift_S_cm"][2] == pytest.approx(point["c"][2] - point["c_pyatb_rule"][2], rel=1e-12)
    assert flagged["flip_allowance_S_cm"] == [0.0, 0.0, 0.0]
    physical = [document["ahc"]["sigma_S_cm"][name][0] for name in ("yz", "zx", "xy")]
    # PyATB's direct value: QUEER's sum with the contribution at M replaced by PyATB's.
    pyatb = [value - c + p for value, c, p in zip(physical, point["c"], point["c_pyatb_rule"], strict=True)]
    reference = reference_from(tmp_path / "pyatb", direct, sigma=pyatb)
    report = compare("AHC", reference, direct)
    assert report["verdict"] == PASS_DOCUMENTED and classes(report)["sigma_xy"] == DOCUMENTED, (
        render_markdown(report)
    )
    detail = [item for item in report["quantities"] if item["name"] == "sigma_xy"][0]["detail"]
    assert "[0.5, 0.5, 0.0]" in detail and "1e-4 eV" in detail and "PyATB's cut" in detail
    # The exact shift, not a window of its size: wrong candidates fail even where M dominates sigma.
    for factor in (0.5, 0.0, -1.0, 2.0):
        report = compare("AHC", reference, _scaled(direct, factor))
        assert report["verdict"] == FAIL and classes(report)["sigma_xy"] == UNEXPLAINED, factor


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


def test_a_run_failing_before_its_scfout_is_staged_is_judged_on_the_same_scfout(runs, tmp_path, monkeypatch):
    import queer.compat.generator.runner as runner

    # PyATB runs a zero absolute_error to its generation cap; QUEER refuses it while it
    # translates the parameters, before the scfout is staged.
    adaptive = {"integrate_mode": "Adaptive", "initial_grid": [32, 32, 1], "absolute_error": 0.0}
    reference = reference_from(tmp_path / "reference", runs["openmx"])
    echo = read(reference / "Out/input.json")
    for key in ("integrate_grid", "adaptive_grid", "adaptive_grid_threshold"):
        echo["AHC"].pop(key)
    echo["AHC"].update(adaptive, relative_error=1e-4)
    write(reference / "Out/input.json", echo)
    refused = tmp_path / "refused"
    refused.mkdir()
    qwz_spinor_scfout(refused / "scf.scfout")
    params = read(runs["openmx"] / "queer_params.json")
    for key in ("integrate_grid", "adaptive_grid", "adaptive_grid_threshold"):
        params["parameters"].pop(key)
    params["parameters"].update(adaptive, relative_error=1e-4)
    write(refused / "queer_params.json", params)
    assert run(None, refused / "queer_params.json", refused, log=lambda line: None) == 2
    document = read(refused / "queer_results.json")
    assert document["error"]["code"] == "queer_bad_params" and "model" not in document
    assert document["provenance"]["inputs"]["scf.scfout"]["sha256"] == sha256(refused / "scf.scfout")
    report = compare("AHC", reference, refused)
    assert report["verdict"] == NOT_COMPARABLE and _preconditions(report) == {"capability"}, render_markdown(
        report
    )
    assert "zero tolerance" in report["preconditions"][-1]["detail"]

    # A defect in QUEER's parameter translation fails against the completed PyATB run.
    def broken(params, step=None):
        raise KeyError("integrate_grid")

    monkeypatch.setattr(runner, "translate_openmx", broken)
    crashed = copy(runs["openmx"], tmp_path / "crashed")
    (crashed / "queer_results.json").unlink()
    assert run(None, crashed / "queer_params.json", crashed, log=lambda line: None) == 1
    document = read(crashed / "queer_results.json")
    assert document["error"]["code"] == "queer_internal_error"
    assert document["provenance"]["inputs"]["scf.scfout"]["sha256"] == sha256(crashed / "scf.scfout")
    report = compare("AHC", reference_from(tmp_path / "plain", runs["openmx"]), crashed)
    assert report["verdict"] == FAIL and classes(report) == {"status": UNEXPLAINED}, render_markdown(report)
    # Without a staged scfout there is no digest: the missing input is the failure compared.
    (crashed / "scf.scfout").unlink()
    monkeypatch.undo()
    assert run(None, crashed / "queer_params.json", crashed, log=lambda line: None) == 2
    document = read(crashed / "queer_results.json")
    assert (
        document["error"]["code"] == "queer_missing_input"
        and "scf.scfout" not in document["provenance"]["inputs"]
    )
    report = compare("AHC", reference_from(tmp_path / "missing", runs["openmx"]), crashed)
    assert report["verdict"] == FAIL and "before it read a scfout" in report["preconditions"][2]["detail"]


def test_a_chern_reference_without_hall_convention_compares(runs, tmp_path):
    # A Chern number carries no Hall sign: a pre-key echo (no hall_convention, read as pyatb)
    # against a candidate built from its Input (auto, read as openmx) still compares.
    reference = calibration_reference("h_chern_fermi", tmp_path / "reference")
    echo = read(reference / "Out/input.json")
    assert echo["INPUT_PARAMETERS"].pop("hall_convention") == "openmx"
    write(reference / "Out/input.json", echo)
    candidate_dir = template(runs, tmp_path / "chern", "h_chern_fermi", chern=-0.9999999999999989)
    assert read(candidate_dir / "queer_params.json")["input_parameters"]["hall_convention"] == "auto"
    report = compare("CHERN_NUMBER", reference, candidate_dir)
    assert report["verdict"] == PASS and classes(report)["chern_number"] == AGREE, render_markdown(report)


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
    """The QWZ spinor model with a3 reversed: the same crystal on a left-handed lattice."""
    return qwz_spinor_scfout(path, lattice=np.diag([2.0, 2.0, -10.0]))


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


def tilted_qwz_scfout(path):
    """The QWZ spinor model with a3 tilted off the layer normal: a non-orthogonal cell."""
    return qwz_spinor_scfout(path, lattice=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.5, 0.0, 10.0]])


def test_pyatbs_fhs_construction_is_compared_after_convention_on_a_tilted_cell(tmp_path):
    scan = {"fhs_grid": [8, 8, 1], "fhs_energy_window": [-0.2, 0.2], "fhs_energy_points": 4}
    tilted = candidate(tmp_path / "tilted", scfout=tilted_qwz_scfout, engine="fhs", **scan)
    document = read(tilted / "queer_results.json")
    names = ("yz", "zx", "xy")
    construction = np.column_stack([document["ahc"]["pyatb_construction_S_cm"][name] for name in names])
    physical = np.column_stack([document["ahc"]["sigma_S_cm"][name] for name in names])
    assert np.all(np.abs(construction[:, 0]) > 1.0) and np.all(physical[:, 0] == 0)
    report = compare("AHC", reference_from(tmp_path / "pyatb", tilted, sigma=construction), tilted)
    assert report["verdict"] == PASS, render_markdown(report)
    assert (
        classes(report)["sigma_xy(E)"] == AFTER_CONVENTION
        and classes(report)["sigma_yz(E)"] == AFTER_CONVENTION
    )
    detail = [item for item in report["quantities"] if item["name"] == "sigma_xy(E)"][0]["detail"]
    assert "Cartesian construction" in detail
    # PyATB printing the physical vector would contradict its own construction: in a gap that fails.
    report = compare("AHC", reference_from(tmp_path / "physical", tilted, sigma=physical), tilted)
    assert report["verdict"] == FAIL and classes(report)["sigma_yz(E)"] == UNEXPLAINED


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


def template(runs, directory, fixture, *, sigma=None, chern=None, offsets=None, fhs=None):
    """A candidate run of the kagome calibration: a runner document carrying QUEER's calibration values.

    ``fhs`` (an FHS scan) holds QUEER's per-energy facts: ``rank_variation``, the
    ``emulated_xy`` and ``queer_links_xy`` values of its link emulation and the
    ``fermi_sea_xy`` and ``half_grid_xy`` values of its Fermi-sea check.
    """
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
            document["ahc"]["pyatb_construction_S_cm"] = document["ahc"]["sigma_S_cm"]
            facts = fhs or {}
            zeros = [0.0] * len(offsets)
            document["integration"]["fhs"]["metallic_vertex_rank_variation"] = facts.get(
                "rank_variation", [False] * len(offsets)
            )
            document["ahc"]["pyatb_emulation"] = (
                {
                    "fhs_links": {
                        "printed_S_cm": {"yz": zeros, "zx": zeros, "xy": facts["emulated_xy"]},
                        "queer_links_S_cm": {"yz": zeros, "zx": zeros, "xy": facts["queer_links_xy"]},
                    }
                }
                if "emulated_xy" in facts
                else None
            )
            document["ahc"]["fermi_sea_check"] = None
            if "fermi_sea_xy" in facts:
                band = [None if value is None else 0.0 for value in facts["fermi_sea_xy"]]
                document["ahc"]["fermi_sea_check"] = {
                    "engine": "direct",
                    "grid": [100, 100, 1],
                    "half_grid": [50, 50, 1],
                    "error": facts.get("fermi_sea_error"),
                    "fermi_sea_S_cm": {"yz": band, "zx": band, "xy": facts["fermi_sea_xy"]},
                    "fhs_half_grid_S_cm": {"yz": band, "zx": band, "xy": facts["half_grid_xy"]},
                }
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


def kagome_fhs():
    """PyATB's a_fhs_scan table and QUEER's values on the same scan (fixtures)."""
    from queer.compat.generator.pyatb_files import parse_ahc

    table = parse_ahc((FIXTURES / "a_fhs_scan/Out/AHC/ahc.dat").read_text())
    queer = read(FIXTURES / "a_fhs_scan_queer.json")
    rows = np.zeros((len(queer["construction_xy"]), 3))
    rows[:, 2] = queer["construction_xy"]
    return table, queer, rows


def test_an_fhs_scan_agrees_in_gaps_and_is_explained_by_the_link_schemes_in_bands(runs, tmp_path):
    table, queer, rows = kagome_fhs()
    reference = calibration_reference("a_fhs_scan", tmp_path / "reference")
    offsets = table["energy_eV"]
    same = template(runs, tmp_path / "same", "a_fhs_scan", sigma=np.asarray(table["sigma"]), offsets=offsets)
    report = compare("AHC", reference, same)
    assert report["verdict"] == PASS and classes(report)["sigma_xy(E)"] == AGREE
    # QUEER's real values: Tier A at the 159 gapped energies, and at the 42 in bands the gap
    # (up to 0.13 S/cm) is the offset of PyATB's links (+i dk.r) from the Fermi-sea flux, which
    # QUEER's emulation reproduces while QUEER's FHS meets its own Fermi-sea sum.
    real = template(runs, tmp_path / "real", "a_fhs_scan", sigma=rows, offsets=offsets, fhs=queer)
    report = compare("AHC", reference, real)
    assert report["verdict"] == PASS_DOCUMENTED and classes(report)["sigma_xy(E)"] == DOCUMENTED, (
        render_markdown(report)
    )
    item = [item for item in report["quantities"] if item["name"] == "sigma_xy(E)"][0]
    assert 0.13 < item["metric"] < 0.14 and "0 in a gap, 42 with a varying occupied rank" in item["detail"]
    assert "+i dk.r" in item["detail"] and "-i dk.r" in item["detail"] and "Fermi-sea" in item["detail"]
    assert "link's start" not in item["detail"]
    guard = report["tolerance_basis"]["fermi_sea_guard"]["xy"]
    assert guard["energies"] == 42 and guard["failed"] == 0 and 0.5 < guard["max_ratio"] < 1.0
    assert 0.03 < guard["max_gap_S_cm"] < 0.05
    shifted = template(
        runs, tmp_path / "shifted", "a_fhs_scan", sigma=rows, offsets=[e + 1e-3 for e in offsets]
    )
    assert classes(compare("AHC", reference, shifted))["energies"] == UNEXPLAINED


@pytest.mark.parametrize("factor", [-1.0, 2.0, 0.0, float("nan")])
def test_fhs_negative_controls_do_not_pass(runs, tmp_path, factor):
    table, queer, rows = kagome_fhs()
    reference = calibration_reference("a_fhs_scan", tmp_path / "reference")
    wrong = template(
        runs, tmp_path / "wrong", "a_fhs_scan", sigma=rows * factor, offsets=table["energy_eV"], fhs=queer
    )
    report = compare("AHC", reference, wrong)
    assert report["verdict"] == FAIL and classes(report)["sigma_xy(E)"] == UNEXPLAINED, factor


def test_an_fhs_gap_needs_both_link_schemes_reproduced(runs, tmp_path):
    table, queer, rows = kagome_fhs()
    reference = calibration_reference("a_fhs_scan", tmp_path / "reference")
    offsets = table["energy_eV"]
    band = queer["rank_variation"].index(True)
    gap = queer["rank_variation"].index(False)
    # One gapped energy off by 1e-5 S/cm: in a gap only Tier A passes.
    moved = rows.copy()
    moved[gap, 2] += 1e-5
    report = compare(
        "AHC",
        reference,
        template(runs, tmp_path / "gap", "a_fhs_scan", sigma=moved, offsets=offsets, fhs=queer),
    )
    assert report["verdict"] == FAIL and "1 in a gap" in report["quantities"][-2]["detail"]
    # An emulation that misses PyATB's value at one band energy explains nothing.
    missed = {**queer, "emulated_xy": list(queer["emulated_xy"])}
    missed["emulated_xy"][band] += 1e-3
    report = compare(
        "AHC",
        reference,
        template(runs, tmp_path / "missed", "a_fhs_scan", sigma=rows, offsets=offsets, fhs=missed),
    )
    assert classes(report)["sigma_xy(E)"] == UNEXPLAINED
    assert (
        "emulation of PyATB's links misses"
        in [i for i in report["quantities"] if i["name"] == "sigma_xy(E)"][0]["detail"]
    )
    # A QUEER value that its own midpoint links do not give is unexplained even inside a band.
    strayed = rows.copy()
    strayed[band, 2] += 0.05
    report = compare(
        "AHC",
        reference,
        template(runs, tmp_path / "strayed", "a_fhs_scan", sigma=strayed, offsets=offsets, fhs=queer),
    )
    assert classes(report)["sigma_xy(E)"] == UNEXPLAINED
    assert (
        "not its own midpoint links'"
        in [i for i in report["quantities"] if i["name"] == "sigma_xy(E)"][0]["detail"]
    )
    # Without the emulation a band gap cannot be documented.
    bare = {"rank_variation": queer["rank_variation"]}
    report = compare(
        "AHC",
        reference,
        template(runs, tmp_path / "bare", "a_fhs_scan", sigma=rows, offsets=offsets, fhs=bare),
    )
    assert classes(report)["sigma_xy(E)"] == UNEXPLAINED
    # Nor without QUEER's own Fermi-sea check, or with one that could not be evaluated.
    unchecked = {key: value for key, value in queer.items() if key not in ("fermi_sea_xy", "half_grid_xy")}
    failed = {
        **queer,
        "fermi_sea_error": "ValueError: position/overlap derivatives violate dipole covariance",
    }
    for name, facts, reason in (
        ("unchecked", unchecked, "no Fermi-sea check"),
        ("failed", failed, "dipole covariance"),
    ):
        report = compare(
            "AHC",
            reference,
            template(runs, tmp_path / name, "a_fhs_scan", sigma=rows, offsets=offsets, fhs=facts),
        )
        detail = [i for i in report["quantities"] if i["name"] == "sigma_xy(E)"][0]["detail"]
        assert classes(report)["sigma_xy(E)"] == UNEXPLAINED and reason in detail, detail


@pytest.mark.parametrize("factor", ["-1", "1.5"])
def test_a_link_defect_shared_by_queers_fhs_and_its_emulation_fails_the_fermi_sea_guard(
    runs, tmp_path, factor
):
    # Measured runs with the -i dk.A term of QUEER's midpoint links scaled by -1 (PyATB's sign)
    # or 1.5 in its FHS and its own-link emulation alike: the emulation still explains every
    # band gap, but the FHS no longer meets QUEER's own Fermi-sea sum, which the connection
    # data still give.
    table, queer, _ = kagome_fhs()
    defect = read(FIXTURES / "a_fhs_scan_link_defects.json")[factor]
    rows = np.zeros((len(defect["construction_xy"]), 3))
    rows[:, 2] = defect["construction_xy"]
    facts = {**queer, "queer_links_xy": defect["queer_links_xy"], "half_grid_xy": defect["half_grid_xy"]}
    reference = calibration_reference("a_fhs_scan", tmp_path / "reference")
    candidate_dir = template(
        runs, tmp_path / "defect", "a_fhs_scan", sigma=rows, offsets=table["energy_eV"], fhs=facts
    )
    report = compare("AHC", reference, candidate_dir)
    detail = [item for item in report["quantities"] if item["name"] == "sigma_xy(E)"][0]["detail"]
    assert report["verdict"] == FAIL and classes(report)["sigma_xy(E)"] == UNEXPLAINED, render_markdown(
        report
    )
    assert "departs from its own Fermi-sea sum" in detail and "0 in a gap" in detail
    assert "misses" not in detail and "midpoint links'" not in detail
    guard = report["tolerance_basis"]["fermi_sea_guard"]["xy"]
    assert guard["energies"] == 42 and guard["failed"] >= 8 and guard["max_ratio"] > 10, guard


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
