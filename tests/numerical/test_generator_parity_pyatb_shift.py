"""The shift-current parity rule: PyATB SHIFT_CURRENT run directories against QUEER runs.

Candidates are real runner outputs on the analytic models of ``openmx_files``. A
reference prints the candidate's own emulation of PyATB's scheme (what PyATB would
print from QUEER's transitions) unless a test gives it other numbers. QUEER's
physical value is what is judged: a faithful pair agrees directly (PyATB's table
moved to QUEER's energy scale from its own curve) or differs by PyATB's scheme
reproduced from QUEER's value (its window as QUEER records it, its bins' recorded
moves, its method-1 cut recomputed from the scfout; documented only once this
build's clean control has passed), and each negative control must not pass: a
flipped sign, a zeroed component, photon energies shifted by one step, another eta
or occ_band, a candidate that is not converged, a candidate whose physical value is
wrong while its emulation of PyATB is right, any record inflated, missing or
contradicting its recomputation.
"""

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from openmx_files import shift_collinear_scfout, shift_spinor_scfout

import queer
from queer import OpenMXModel
from queer.compat.generator.openmx_params import SHIFT_DEFAULTS
from queer.compat.generator.openmx_shift import (
    PyatbShiftScheme,
    conversion,
    pyatb_scheme_spectrum,
)
from queer.compat.generator.openmx_steps import single_channel_model
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
from queer.compat.generator.parity_shift import (
    NO_CLEAN_CONTROL,
    energy_scale_reach,
    literature_anchor,
    move_to_queer_scale,
    symmetry_gate,
)
from queer.compat.generator.pyatb_conventions import (
    HARTREE_EV,
    PYATB_HARTREE_EV,
    pyatb_shift_constant_ratio,
)
from queer.compat.generator.pyatb_files import SHIFT_COMPONENTS, shift_current_text
from queer.compat.generator.pyatb_shift_integrand import queer_weights, shift_integrand
from queer.compat.generator.results import package_source_sha256, package_sources
from queer.compat.generator.runner import main, run
from queer.models.evaluation import derivatives

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
    # Time reversal (no exchange): exact Kramers pairs at the 8 TRIM points of a 4 x 4 x 4 grid.
    "kramers": (shift_spinor_scfout, {"exchange": 0.0}, 4, {**SPINOR, "grid": [4, 4, 4]}, None),
    # A 2e-5 eV exchange splits those pairs by 4e-5 eV: PyATB's method 1 regularizes them.
    "split": (shift_spinor_scfout, {"exchange": 2e-5}, 4, {**SPINOR, "grid": [4, 4, 4]}, None),
    "negative": (shift_spinor_scfout, {}, 4, {**SPINOR, "omega": [-0.5, 6.5]}, None),
    # PyATB's narrow default-like broadening: its Hartree constant moves each transition by 2.3e-6 of its
    # energy, which moves a correct table by up to 3.8e-4 of its peak at eta 0.01.
    "sharp05": (shift_spinor_scfout, {}, 4, {**SPINOR, "eta": 0.05, "domega": 0.01}, None),
    "sharp02": (shift_spinor_scfout, {}, 4, {**SPINOR, "eta": 0.02, "domega": 0.004}, None),
    "sharp01": (shift_spinor_scfout, {}, 4, {**SPINOR, "eta": 0.01, "domega": 0.002}, None),
    "sharp_collinear": (
        shift_collinear_scfout,
        {},
        1,
        {**COLLINEAR, "eta": 0.01, "domega": 0.002, "omega": [0.2, 6.0], "grid": [6, 6, 1]},
        None,
    ),
    # PyATB's window at its narrow eta: the Hartree scale and the window's dropped transitions together.
    "window_sharp": (
        shift_spinor_scfout,
        {},
        4,
        {**SPINOR, "omega": [2.0, 4.0], "eta": 0.01, "domega": 0.002},
        None,
    ),
    # The split pairs without smearing and with a window: the cut beside PyATB's bins or its window.
    "split_binned": (
        shift_spinor_scfout,
        {"exchange": 2e-5},
        4,
        {**SPINOR, "grid": [4, 4, 4], "smearing_method": 0, "domega": 0.05},
        None,
    ),
    "split_window": (
        shift_spinor_scfout,
        {"exchange": 2e-5},
        4,
        {**SPINOR, "grid": [4, 4, 4], "omega": [2.0, 4.0]},
        None,
    ),
    # A clean-configuration method-1 candidate with its self-convergence estimate (converged).
    "converged": (shift_collinear_scfout, {}, 1, COLLINEAR, True),
    # Narrower still: PyATB's Hartree constant moves the table by about 1 % of its peak at eta 0.002.
    "sharp002": (shift_spinor_scfout, {}, 4, {**SPINOR, "eta": 0.002, "domega": 0.0004}, None),
    # Pairs split by 9.9e-5 eV (flagged) and 1.01e-4 eV (not flagged) on PyATB's scale, and the split pairs at eta 0.01.
    "split_near": (shift_spinor_scfout, {"exchange": 4.95e-5}, 4, {**SPINOR, "grid": [4, 4, 4]}, None),
    "split_far": (shift_spinor_scfout, {"exchange": 5.06e-5}, 4, {**SPINOR, "grid": [4, 4, 4]}, None),
    "split_sharp": (
        shift_spinor_scfout,
        {"exchange": 2e-5},
        4,
        {**SPINOR, "grid": [4, 4, 4], "eta": 0.01, "domega": 0.002},
        None,
    ),
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


@pytest.fixture(scope="module")
def clean_report(runs, tmp_path_factory):
    """The passed clean-control report of this build (the clean run against its own emulation)."""
    report = compare(
        "SHIFT_CURRENT",
        reference_from(tmp_path_factory.mktemp("clean-control"), runs["clean"]),
        runs["clean"],
    )
    assert report["verdict"] == PASS and report["semantics"]["rules"]["clean_control"]["eligible"]
    return report


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


def judge(runs, tmp_path, name, clean_control=None, scfout=False, **reference):
    """``name``'s run against a reference printing its emulation (or ``sigma``); ``scfout`` passes its scfout."""
    return compare(
        "QUEER_SHIFT_CURRENT",
        reference_from(tmp_path / "reference", runs[name], **reference),
        runs[name],
        clean_control=clean_control,
        scfout=runs[name] / "scf.scfout" if scfout else None,
    )


def failed_preconditions(report):
    return [item["name"] for item in report["preconditions"] if not item["ok"]]


def tampered(directory, run_dir, transform, legacy=False):
    """A copy of a candidate whose physical (or legacy) table is ``transform``ed (a linear map) over every k-point,
    so its value without the flagged k-points too; its other diagnostics untouched."""
    copy = shutil.copytree(run_dir, directory)
    document = read(copy / "queer_results.json")
    block = document["shift_current"]
    target = block["legacy_single_channel"] if legacy else block
    key = "binned_uA_V2" if block["binned_uA_V2"] is not None else "sigma_uA_V2"
    values = transform(columns(target[key]).copy())
    target[key] = {name: values[:, index].tolist() for index, name in enumerate(SHIFT_COMPONENTS)}
    regularisation = block["diagnostics"]["regularisation"]
    record = "binned_unflagged_uA_V2" if key == "binned_uA_V2" else "sigma_unflagged_uA_V2"
    regularisation[record] = as_columns(transform(columns(regularisation[record]).copy()))
    write(copy / "queer_results.json", document)
    exported = columns(block[key]) * block["convention"]["export_sign"]
    (copy / "Out/Shift_Current/shift_current.dat").write_text(shift_current_text(block["omega_eV"], exported))
    return copy


def as_columns(values):
    return {name: np.asarray(values)[:, index].tolist() for index, name in enumerate(SHIFT_COMPONENTS)}


def edited(directory, run_dir, edit):
    """A copy of a candidate whose ``shift_current`` block ``edit`` changes; its file follows the block."""
    copy = shutil.copytree(run_dir, directory)
    document = read(copy / "queer_results.json")
    block = document["shift_current"]
    edit(block)
    write(copy / "queer_results.json", document)
    key = "binned_uA_V2" if block["binned_uA_V2"] is not None else "sigma_uA_V2"
    exported = columns(block[key]) * block["convention"]["export_sign"]
    (copy / "Out/Shift_Current/shift_current.dat").write_text(shift_current_text(block["omega_eV"], exported))
    return copy


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
        ("pair", {AGREE}),
        ("kramers", {AGREE}),
        ("negative", {AGREE}),
    ],
)
def test_a_reference_printing_pyatbs_scheme_of_the_candidate_passes(runs, tmp_path, name, kinds):
    report = judge(runs, tmp_path, name)
    assert report["verdict"] == PASS, render_markdown(report)
    assert components(report) == kinds and classes(report)["symmetry"] == AGREE
    assert classes(report)["candidate_file"] == AGREE == classes(report)["emulation_consistency"]
    rules = report["semantics"]["rules"]
    assert rules["legacy_single_channel"] is (name == "collinear")
    assert rules["export_sign"] == (-1 if name == "left" else 1)
    assert rules["spin_factor"] == (2 if name in {"collinear", "nonmagnetic"} else 1)
    assert report["tolerance_basis"]["relative"] == 1e-4 and report["tolerance_basis"]["emulated"]


def test_pyatbs_scheme_explains_a_difference_only_as_documented_after_a_passed_clean_control(
    runs, tmp_path, clean_report
):
    # PyATB's window drops the transitions outside it, which moves its table near the window's edges
    # away from QUEER's physical value: never an agreement.
    report = judge(runs, tmp_path / "alone", "window")
    assert report["verdict"] == NOT_COMPARABLE and failed_preconditions(report) == ["clean_control"]
    assert NO_CLEAN_CONTROL in json.dumps(report["preconditions"])
    report = judge(runs, tmp_path / "released", "window", clean_control=clean_report)
    assert report["verdict"] == PASS_DOCUMENTED, render_markdown(report)
    assert DOCUMENTED in components(report) and components(report) <= {DOCUMENTED, AGREE}
    assert classes(report)["emulation_consistency"] == AGREE
    documented = [item for item in report["quantities"] if item["class"] == DOCUMENTED]
    assert all("PyATB's scheme reproduces it" in item["detail"] for item in documented)
    assert all("transitions its window drops" in item["detail"] for item in documented)
    # The rows the dropped transitions cannot reach carry enough of each component to check QUEER's value there.
    scheme = report["tolerance_basis"]["scheme"]
    assert scheme["rows_beyond_reach"] > 0 and min(scheme["informative_share"].values()) >= 0.1
    # The documented metric is the physical gap (to PyATB's print rounding), not the emulation's.
    physical = columns(read(runs["window"] / "queer_results.json")["shift_current"]["sigma_uA_V2"])
    gap = np.max(np.abs(scheme_table(runs["window"]) - physical), axis=0)
    for item in documented:
        assert item["metric"] == pytest.approx(gap[SHIFT_COMPONENTS.index(item["name"][6:])], rel=1e-6)
    # A Gaussian truncated at int(5 eta/domega) = 0 bins moves PyATB's table so far that its emulation
    # cannot check QUEER's value: the run cannot judge QUEER at all.
    report = judge(runs, tmp_path / "coarse", "coarse", clean_control=clean_report)
    assert report["verdict"] == NOT_COMPARABLE and failed_preconditions(report) == ["pyatb_scheme_resolution"]
    assert (
        report["tolerance_basis"]["truncation_tail"] > 1e-6 and classes(report)["symmetry"] == NOT_APPLICABLE
    )
    # At PyATB's narrow eta its Hartree scale and its window act together: still reproduced exactly.
    report = judge(runs, tmp_path / "window-sharp", "window_sharp", clean_control=clean_report)
    assert report["verdict"] == PASS_DOCUMENTED, render_markdown(report)
    assert report["tolerance_basis"]["energy_scale"]["applied"]


def test_the_clean_control_is_recognised_from_the_data_and_must_agree(runs, tmp_path):
    report = judge(runs, tmp_path / "pass", "clean")
    clean = report["semantics"]["rules"]["clean_control"]
    assert clean["eligible"] and all(clean["conditions"].values()) and clean["release"] == "not_supplied"
    assert classes(report)["clean_control"] == AGREE and components(report) == {AGREE}
    for name, condition in (
        ("binned", "gaussian"),
        ("window", "window_margin_5_eta"),
        ("coarse", "truncation_tail_below_1e-6"),
        ("split", "no_pair_within_1e-4_eV"),
        ("method0", "method_1"),
    ):
        rules = judge(runs, tmp_path / name, name)["semantics"]["rules"]["clean_control"]
        assert not rules["eligible"] and not rules["conditions"][condition], (name, rules)
    # A clean control that differs by 2e-4 of its peak fails, whatever the causes.
    report = judge(runs, tmp_path / "off", "clean", sigma=scheme_table(runs["clean"]) * (1 + 2e-4))
    assert report["verdict"] == FAIL and classes(report)["clean_control"] == UNEXPLAINED
    # Its tolerance is 1e-4 of PyATB's peak and PyATB's print rounding: no number of the candidate widens it.
    basis = report["tolerance_basis"]
    assert basis["base_uA_V2"] == pytest.approx(
        1e-4 * np.max(np.abs(scheme_table(runs["clean"]) * (1 + 2e-4)))
    )
    assert "edge_allowance_uA_V2" not in basis and "consistency" not in basis
    # PyATB's truncated Gaussian on a coarse step (interval 0) is measured, never a clean control.
    truncation = read(runs["coarse"] / "queer_results.json")["shift_current"]["diagnostics"][
        "gaussian_truncation"
    ]
    assert truncation["interval_bins"] == 0 and truncation["max_effect_uA_V2"] > 1e-2


def test_queers_physical_value_must_meet_pyatb_not_its_emulation_alone(runs, tmp_path, clean_report):
    """A wrong physical value beside a right emulation of PyATB, with PyATB printing the truth, never passes."""
    xxy, yxx = SHIFT_COMPONENTS.index("xxy"), SHIFT_COMPONENTS.index("yxx")

    def swap(values):
        values[:, [xxy, yxx]] = values[:, [yxx, xxy]]
        return values

    cases = {
        "negated": ("clean", lambda values: -values),
        "doubled": ("clean", lambda values: 2 * values),
        "zeroed": ("clean", lambda values: 0 * values),
        "swapped": ("clean", swap),
        "rolled": ("clean", lambda values: np.roll(values, 3, axis=0)),
        "halved": ("nonmagnetic", lambda values: 0.5 * values),
        "unsigned": ("left", lambda values: -values),
        "window-doubled": ("window", lambda values: 2 * values),
        "binned-doubled": ("binned", lambda values: 2 * values),
        "legacy-halved": ("collinear", lambda values: 0.5 * values),
    }
    for label, (name, transform) in cases.items():
        legacy = name == "collinear"
        copy = tampered(tmp_path / label, runs[name], transform, legacy=legacy)
        truth = scheme_table(runs[name])
        for supplied in (None, clean_report):
            reference = reference_from(tmp_path / f"{label}-{supplied is None}", copy, sigma=truth)
            report = compare("SHIFT_CURRENT", reference, copy, clean_control=supplied)
            assert report["verdict"] == FAIL, (label, render_markdown(report))
            assert classes(report)["emulation_consistency"] == UNEXPLAINED, label
            assert failed_preconditions(report) == []
    # Where PyATB's scheme is too coarse to check QUEER's value, a wrong value still contradicts the emulation's
    # own sum of its transitions (an exact identity at any step); doubled everywhere it is not comparable.
    copy = tampered(tmp_path / "coarse-doubled", runs["coarse"], lambda values: 2 * values)
    reference = reference_from(tmp_path / "coarse-doubled-ref", copy, sigma=scheme_table(runs["coarse"]))
    report = compare("SHIFT_CURRENT", reference, copy, clean_control=clean_report)
    assert report["verdict"] == FAIL and classes(report)["emulation_consistency"] == UNEXPLAINED

    def everywhere(block):
        scheme = block["diagnostics"]["pyatb_scheme"]
        scheme["transitions_uA_V2"] = as_columns(2 * columns(scheme["transitions_uA_V2"]))

    consistent = edited(tmp_path / "coarse-consistent", copy, everywhere)
    reference = reference_from(
        tmp_path / "coarse-consistent-ref", consistent, sigma=scheme_table(runs["coarse"])
    )
    report = compare("SHIFT_CURRENT", reference, consistent, clean_control=clean_report)
    assert report["verdict"] == NOT_COMPARABLE and failed_preconditions(report) == ["pyatb_scheme_resolution"]
    # A clean control whose physical value is wrong fails, so its report licenses nothing.
    broken = tampered(tmp_path / "broken-control", runs["clean"], lambda values: 2 * values)
    control = compare(
        "SHIFT_CURRENT",
        reference_from(tmp_path / "broken-control-ref", broken, sigma=scheme_table(runs["clean"])),
        broken,
    )
    assert control["verdict"] == FAIL
    report = judge(runs, tmp_path / "licensed", "window", clean_control=control)
    assert report["verdict"] == FAIL and classes(report)["clean_control_release"] == UNEXPLAINED


def test_pyatbs_table_moves_to_queers_energy_scale_from_its_own_curve():
    """PyATB places a transition E at s E (its Hartree constant) and multiplies by its constants r: its Gaussian table
    P(w) = (r/s) X_{eta/s}(w/s). move_to_queer_scale undoes that from P's own curve (no candidate number); on random
    transitions with near-cancelling pairs it meets QUEER's X to well within 1e-4 of the peak wherever its own
    uncertainty is below half of that, and the reach bounds the move where the step is too coarse to resolve it."""
    rng = np.random.default_rng(11)
    scale = PYATB_HARTREE_EV / HARTREE_EV
    ratio = pyatb_shift_constant_ratio()

    def kernel(offset, eta):
        return np.exp(-((offset / eta) ** 2)) / (eta * math.sqrt(math.pi))

    for eta, step in ((0.1, 0.02), (0.01, 0.002), (0.01, 0.004), (0.002, 0.0004), (0.002, 0.0008)):
        omega = 0.3 + step * np.arange(int(6.2 / step) + 1)
        energies = rng.uniform(0.6, 6.2, 300)
        weights = rng.normal(size=(300, 3))
        pairs = energies[:100] + rng.uniform(-0.3, 0.3, 100) * eta
        energies = np.concatenate((energies, pairs))
        weights = np.concatenate((weights, -weights[:100] * rng.uniform(0.9, 1.1, (100, 1))))
        queer = kernel(omega[:, None] - energies[None, :], eta) @ weights
        pyatb = ratio * kernel(omega[:, None] - scale * energies[None, :], eta) @ weights
        moved, uncertainty = move_to_queer_scale(pyatb, omega=omega, eta=eta, domega=step)
        tolerance = 1e-4 * np.max(np.abs(queer))
        resolved = uncertainty <= 0.5 * tolerance
        assert resolved.mean() > 0.9, (eta, step)
        error = np.abs(moved - queer)
        assert np.all(error[resolved] <= tolerance + uncertainty[resolved]), (eta, step)
        # The move matters: at eta 0.002 PyATB's Hartree constant alone moves the table by about 1 % of its peak.
        if eta <= 0.01:
            move = np.max(np.abs(pyatb / ratio - queer)) / np.max(np.abs(queer))
            assert move > (5e-3 if eta == 0.002 else 1e-4), (eta, move)
        # The reach covers what PyATB's Hartree constant does, and a wrong table (twice the truth) lies beyond it
        # wherever the truth is a tenth of its peak.
        reach = energy_scale_reach(pyatb, omega=omega, eta=eta, domega=step)
        assert np.all(np.abs(pyatb - queer) <= reach + 1e-12 * np.max(np.abs(queer))), (eta, step)
        large = np.abs(queer) > 0.1 * np.max(np.abs(queer))
        assert np.all((np.abs(2 * queer - queer) > tolerance + reach)[large]), (eta, step)
    # On a step of eta the move is not attempted; its reach still covers what PyATB's Hartree constant does.
    omega = 0.3 + 0.1 * np.arange(63)
    energies = rng.uniform(0.6, 6.2, 200)
    weights = rng.normal(size=(200, 2))
    queer = kernel(omega[:, None] - energies[None, :], 0.1) @ weights
    pyatb = ratio * kernel(omega[:, None] - scale * energies[None, :], 0.1) @ weights
    reach = energy_scale_reach(pyatb, omega=omega, eta=0.1, domega=0.1)
    assert np.all(np.abs(pyatb - queer) <= reach + 1e-12 * np.max(np.abs(queer)))


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
def test_negative_controls_never_pass(runs, tmp_path, clean_report, change, verdict):
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
        report = judge(
            runs,
            tmp_path,
            "unconverged",
            clean_control=clean_report,
            sigma=scheme_table(runs["unconverged"]) * 1.2,
        )
        failed = [item for item in report["preconditions"] if not item["ok"]]
        assert [item["name"] for item in failed] == ["self_convergence"]
        assert failed[0]["detail"].startswith("queer_unconverged")
    assert report["verdict"] == verdict, render_markdown(report)
    if verdict == NOT_COMPARABLE and change != "unconverged":
        failed = {item["name"] for item in report["preconditions"] if not item["ok"]}
        assert failed & {"parameters", "omega"}, report["preconditions"]


def test_an_unbounded_cause_documents_a_difference_only_on_a_converged_candidate(
    runs, tmp_path, clean_report
):
    for name, cause in (("method0", "regularisation"), ("scissors", "scissors")):
        convergence = read(runs[name] / "queer_results.json")["shift_current"]["diagnostics"][
            "self_convergence"
        ]
        assert convergence["legacy_single_channel"]["converged"], convergence["legacy_single_channel"][
            "ratio"
        ]
        # Without a passed clean control of this build the difference is not judged.
        report = judge(runs, tmp_path / f"{name}-alone", name, sigma=scheme_table(runs[name]) * 0.97)
        assert report["verdict"] == NOT_COMPARABLE and failed_preconditions(report) == ["clean_control"]
        report = judge(
            runs, tmp_path / name, name, clean_control=clean_report, sigma=scheme_table(runs[name]) * 0.97
        )
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
    report = judge(runs, tmp_path / "flipped", "method0", clean_control=clean_report, sigma=flipped)
    assert report["verdict"] == FAIL
    # Without its self-convergence estimate the candidate cannot be documented.
    copy = shutil.copytree(runs["method0"], tmp_path / "no-estimate")
    document = read(copy / "queer_results.json")
    document["shift_current"]["diagnostics"]["self_convergence"] = None
    write(copy / "queer_results.json", document)
    reference = reference_from(tmp_path / "no-estimate-ref", copy, sigma=scheme_table(copy) * 0.97)
    report = compare("SHIFT_CURRENT", reference, copy, clean_control=clean_report)
    assert report["verdict"] == NOT_COMPARABLE and "self_convergence: true" in json.dumps(
        report["preconditions"]
    )
    # Ten times PyATB's largest value either way is not a convergence question.
    for label, factor in (("huge", 1 / 20), ("tiny", 20)):
        sigma = scheme_table(runs["method0"]) * factor
        report = judge(runs, tmp_path / label, "method0", clean_control=clean_report, sigma=sigma)
        assert report["verdict"] == NOT_COMPARABLE and "max|sigma_Q|" in json.dumps(report["preconditions"])


def test_adaptive_smearing_is_an_unbounded_cause_without_an_emulation(runs, tmp_path, clean_report):
    document = read(runs["adaptive"] / "queer_results.json")
    scheme = document["shift_current"]["diagnostics"]["pyatb_scheme"]
    assert scheme["sigma_uA_V2"] is None and "not emulated" in scheme["reason"]
    physical = columns(document["shift_current"]["sigma_uA_V2"])
    report = judge(runs, tmp_path / "same", "adaptive", sigma=physical)
    assert report["verdict"] == PASS and not report["tolerance_basis"]["emulated"]
    report = judge(runs, tmp_path / "other", "adaptive", clean_control=clean_report, sigma=physical * 1.05)
    assert "adaptive_smearing" in report["semantics"]["rules"]["unbounded_causes"]
    # Its 2 x 2 x 1 grid is far from converged: the difference cannot be documented.
    assert report["verdict"] == NOT_COMPARABLE and failed_preconditions(report) == ["self_convergence"]


def test_pyatbs_spin_and_bin_conventions_are_required(runs, tmp_path, clean_report):
    collinear = read(runs["collinear"] / "queer_results.json")["shift_current"]
    # PyATB doubling the spin-up channel is the legacy value: the physical sum of both channels is not.
    report = judge(runs, tmp_path / "physical", "collinear", sigma=columns(collinear["sigma_uA_V2"]))
    assert report["verdict"] == FAIL
    binned = read(runs["binned"] / "queer_results.json")["shift_current"]
    # Without smearing PyATB prints bin weights: QUEER's histogram per eV is not what it prints.
    report = judge(
        runs, tmp_path / "density", "binned", clean_control=clean_report, sigma=columns(binned["sigma_uA_V2"])
    )
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


def test_the_release_clean_control_report_applies_to_every_comparison(runs, tmp_path, clean_report):
    clean = clean_report
    assert clean["semantics"]["rules"]["build"]["queer_source_sha256"]
    failed = {**clean, "verdict": FAIL}
    report = compare(
        "SHIFT_CURRENT",
        reference_from(tmp_path / "ref", runs["window"]),
        runs["window"],
        clean_control=failed,
    )
    assert report["verdict"] == FAIL and components(report) == {UNEXPLAINED}
    assert classes(report)["clean_control_release"] == UNEXPLAINED
    path = tmp_path / "clean.json"
    write(path, clean)
    report = compare(
        "SHIFT_CURRENT",
        reference_from(tmp_path / "ref-2", runs["window"]),
        runs["window"],
        clean_control=path,
    )
    assert report["verdict"] == PASS_DOCUMENTED and classes(report)["clean_control_release"] == AGREE
    # A report of another build (another queer_version, the shift path's or the comparator's source, or a
    # report without a build) is not this build's: the difference is not judged.
    for key, value in (
        ("queer_version", "0.0.0"),
        ("queer_source_sha256", "0" * 64),
        ("comparator_sha256", "0" * 64),
        ("build_sha", "deadbeef"),
        (None, None),
    ):
        other = json.loads(json.dumps(clean))
        if key is None:
            del other["semantics"]["rules"]["build"]
        else:
            other["semantics"]["rules"]["build"][key] = value
        report = compare(
            "SHIFT_CURRENT",
            reference_from(tmp_path / f"ref-{key}", runs["window"]),
            runs["window"],
            clean_control=other,
        )
        assert classes(report)["clean_control_release"] == NOT_APPLICABLE, key
        assert report["verdict"] == NOT_COMPARABLE and NO_CLEAN_CONTROL in json.dumps(report["preconditions"])
    # A report whose components agree only through a cause is no passed clean control.
    documented = json.loads(json.dumps(clean))
    for item in documented["quantities"]:
        if item["name"] == "sigma_xyz":
            item["class"] = DOCUMENTED
    report = compare(
        "SHIFT_CURRENT",
        reference_from(tmp_path / "ref-documented", runs["window"]),
        runs["window"],
        clean_control=documented,
    )
    assert classes(report)["clean_control_release"] == UNEXPLAINED and report["verdict"] == FAIL
    # A direct agreement needs no report.
    assert judge(runs, tmp_path / "direct", "binned")["verdict"] == PASS
    with pytest.raises(ValueError, match="SHIFT_CURRENT comparison only"):
        compare("AHC", tmp_path, tmp_path, clean_control=clean)
    # The command passes the report through.
    arguments = ["compare", "--reference", str(reference_from(tmp_path / "ref-4", runs["window"]))]
    arguments += ["--candidate", str(runs["window"]), "--clean-control", str(path)]
    assert main(arguments) == 0
    arguments = ["compare", "--reference", str(reference_from(tmp_path / "ref-5", runs["window"]))]
    assert main([*arguments, "--candidate", str(runs["window"])]) == 2


def pyatb_method_1_table(run_dir):
    """PyATB 60fc6b2a's method-1 shift current of a spinor run, ported independently of QUEER's geometry.

    ``get_shift_current_conductivity_ik`` with ``get_Inm_ik_sumOver`` and
    ``linear_response::get_D_degenerate`` (shift_current_solver.cpp, linear_response.cpp)
    for an orthonormal basis without position moments (identity overlap, r = 0, as
    the analytic scfouts are): ``D = -H_a / dE`` with ``dE / (dE^2 + 0.04^2)`` for
    ``|dE| <= 1e-4`` eV (zero for an exact degeneracy), the band slopes on the
    diagonal only, PyATB's window on its Hartree scale, and its per-pair integrand
    ``-(r^b_mn d_r^c_nm,a + r^c_mn d_r^b_nm,a) / 2`` in QUEER's normalization; then
    PyATB's broadening (:func:`pyatb_scheme_spectrum`), constants and spin factor.
    """
    document = read(run_dir / "queer_results.json")
    block = document["shift_current"]
    model = OpenMXModel.from_scfout(run_dir / "scf.scfout")
    params = read(run_dir / "queer_params.json")["parameters"]
    occupied = int(params["occ_band"])
    start, domega, eta = float(params["omega"][0]), float(params["domega"]), float(params["eta"])
    scheme = PyatbShiftScheme(start, domega, len(block["omega_eV"]), eta, 1, False)
    scale = PYATB_HARTREE_EV / HARTREE_EV
    grid = [int(value) for value in params["grid"]]
    points = np.stack(np.meshgrid(*[np.arange(n) / n for n in grid], indexing="ij"), -1).reshape(-1, 3)
    index = [(a, b, c) for a in range(3) for b, c in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))]
    total = np.zeros((scheme.count, len(SHIFT_COMPONENTS)))
    for k in points:
        assert np.allclose(model.overlap(k), np.eye(model.nbnd), atol=1e-14)
        energy, vectors = np.linalg.eigh(model.hamiltonian(k))
        first, second = derivatives(model, k)["H"], derivatives(model, k, order=2)["H"]
        bar = np.einsum("in,aij,jm->anm", vectors.conj(), first, vectors)
        bar2 = np.einsum("in,abij,jm->abnm", vectors.conj(), second, vectors)
        delta = energy[:, None] - energy[None, :]
        inverse = np.where(
            np.abs(delta) > 1e-4, 1 / np.where(delta == 0, 1, delta), delta / (delta**2 + 0.04**2)
        )
        d = -bar * inverse[None]
        d[:, range(len(energy)), range(len(energy))] = 0
        slope = np.real(np.diagonal(bar, axis1=1, axis2=2))
        gaps, weights = [], []
        for n in range(occupied, len(energy)):
            for m in range(occupied):
                if not start <= (energy[n] - energy[m]) * scale < scheme.end:
                    continue
                r = 1j * d[:, m, n]
                step = energy[m] - energy[n]
                dr = np.zeros((3, 3), complex)
                for a in range(3):
                    for b in range(3):
                        dbh = np.vdot(d[b][:, n], bar[a][:, m]) + bar2[a, b, n, m] + bar[a][n, :] @ d[b][:, m]
                        dr[a, b] = (dbh * step - bar[a, n, m] * (slope[b, m] - slope[b, n])) / step**2 * 1j
                weights.append([-0.5 * (r[b] * dr[c, a] + r[c] * dr[b, a]).imag for a, b, c in index])
                gaps.append(energy[n] - energy[m])
        spectrum, _ = pyatb_scheme_spectrum(scheme, np.asarray(gaps), np.asarray(weights).reshape(-1, 18))
        total += spectrum
    return pyatb_shift_constant_ratio() * conversion(model) / len(points) * total


def pyatb_table(run_dir):
    """PyATB's method-1 table from the compat transcription of its integrand (its window on its scale, its 1e-4 eV
    threshold and 0.04 eV regularization), its broadening, constants, spin factor and signed volume."""
    document = read(run_dir / "queer_results.json")
    block = document["shift_current"]
    params = {**SHIFT_DEFAULTS, **read(run_dir / "queer_params.json")["parameters"]}
    model = OpenMXModel.from_scfout(run_dir / "scf.scfout")
    view = model if model.spin_mode == "soc" else single_channel_model(model)
    occupied, domega = int(params["occ_band"]), float(params["domega"])
    start = float(params["omega"][0])
    count = len(block["omega_eV"])
    pair_mode = int(params["n_occ"]) >= 1 and int(params["m_unocc"]) > int(params["n_occ"])
    pairs = [(int(params["n_occ"]) - 1, int(params["m_unocc"]) - 1)] if pair_mode else None
    scheme = PyatbShiftScheme(
        start, domega, count, float(params.get("eta", 0.01)), int(params["smearing_method"]), pair_mode
    )
    grid = [int(value) for value in params["grid"]]
    points = np.stack(np.meshgrid(*[np.arange(n) / n for n in grid], indexing="ij"), -1).reshape(-1, 3)
    total = np.zeros((count, len(SHIFT_COMPONENTS)))
    for k in points:
        result = shift_integrand(
            view,
            k,
            pairs,
            energy_scale=PYATB_HARTREE_EV / HARTREE_EV,
            threshold_eV=1e-4,
            regularization_eV=0.04,
            occupied=occupied,
            window=None if pair_mode else (start, start + count * domega),
        )
        if result["pairs"]:
            spectrum, _ = pyatb_scheme_spectrum(
                scheme, result["energies_eV"], queer_weights(result["integrand"])
            )
            total += spectrum
    factor = block["convention"]["export_sign"] * block["convention"]["pyatb_spin_factor"]
    return factor * pyatb_shift_constant_ratio() * conversion(view) / len(points) * total


def test_exact_kramers_pairs_are_no_cause_and_a_near_degenerate_pair_is_documented_by_its_recomputed_cut(
    runs, tmp_path, clean_report
):
    """PyATB's method 1 drops a pair's 1/dE for |dE| <= 1e-4 eV: an exact (Kramers) degeneracy changes nothing,
    a split one changes the flagged k-points, which the comparison recomputes from the scfout; without the scfout
    the cut is never documented on trust."""
    kramers = read(runs["kramers"] / "queer_results.json")["shift_current"]["diagnostics"]
    assert kramers["exact_degenerate_kpoints"] == 8 and kramers["kpoints_at_pyatb_threshold"] == 0
    assert kramers["regularisation"]["flagged_kpoints"] == 0 and kramers["clean_control"]["eligible"]
    split = read(runs["split"] / "queer_results.json")["shift_current"]["diagnostics"]
    assert split["regularisation"]["flagged_kpoints"] == 8 == split["regularisation"]["emulated_kpoints"]
    assert split["min_pair_gap_eV"] == pytest.approx(4e-5, rel=1e-6)
    for name in ("kramers", "split"):
        # The candidate's emulation (its cut at the flagged k-points) is what the test's own port prints, and the
        # compat transcription prints that too.
        port = pyatb_method_1_table(runs[name])
        assert np.max(np.abs(port - scheme_table(runs[name]))) <= 1e-6 * np.max(np.abs(port)), name
        assert np.max(np.abs(port - pyatb_table(runs[name]))) <= 1e-8 * np.max(np.abs(port)), name
    physical = columns(read(runs["split"] / "queer_results.json")["shift_current"]["sigma_uA_V2"])
    shift = columns(split["regularisation"]["shift_uA_V2"])
    assert np.max(np.abs(shift)) > 1e-2 * np.max(np.abs(physical))
    # A time-reversal-symmetric SOC model is a clean control: any symmetric error fails.
    assert components(judge(runs, tmp_path / "kramers", "kramers")) == {AGREE}
    for factor in (2.0, 0.5, 1 / 8):
        report = judge(
            runs, tmp_path / f"kramers-{factor}", "kramers", sigma=scheme_table(runs["kramers"]) * factor
        )
        assert report["verdict"] == FAIL and classes(report)["clean_control"] == UNEXPLAINED, factor
    truth = pyatb_table(runs["split"])
    report = judge(runs, tmp_path / "split-alone", "split", sigma=truth)
    assert report["verdict"] == NOT_COMPARABLE and "clean_control" in failed_preconditions(report)
    # Without the scfout the cut cannot be recomputed: not documented, not comparable.
    report = judge(runs, tmp_path / "split-unverified", "split", clean_control=clean_report, sigma=truth)
    assert report["verdict"] == NOT_COMPARABLE and failed_preconditions(report) == ["regularisation"]
    assert "no scfout" in json.dumps(report["preconditions"]) and DOCUMENTED not in components(report)
    # With it, every flagged k-point meets QUEER's estimator and PyATB's transcribed integrand: documented, at
    # PyATB's default-like eta 0.01 too, beside its bins or its window, and for a spacing of 9.9e-5 eV.
    for name in ("split", "split_binned", "split_window", "split_sharp", "split_near"):
        report = judge(
            runs,
            tmp_path / f"{name}-scfout",
            name,
            clean_control=clean_report,
            scfout=True,
            sigma=pyatb_table(runs[name]),
        )
        assert report["verdict"] == PASS_DOCUMENTED, (name, render_markdown(report))
        assert classes(report)["regularisation_recomputed"] == AGREE
        documented = [item for item in report["quantities"] if item["class"] == DOCUMENTED]
        assert documented and all("recomputed from the scfout" in item["detail"] for item in documented)
        assert report["semantics"]["rules"]["unbounded_causes"] == []
        assert report["semantics"]["rules"]["flagged_kpoints"]["verified"]
    # A spacing of 1.01e-4 eV on PyATB's scale is no flagged k-point: a clean control that agrees directly.
    far = read(runs["split_far"] / "queer_results.json")["shift_current"]["diagnostics"]
    assert far["regularisation"]["flagged_kpoints"] == 0
    assert far["min_pair_gap_eV"] * PYATB_HARTREE_EV / HARTREE_EV > 1e-4
    report = judge(runs, tmp_path / "far", "split_far", sigma=pyatb_table(runs["split_far"]))
    assert report["verdict"] == PASS and components(report) <= {AGREE, AFTER_CONVENTION}
    near = read(runs["split_near"] / "queer_results.json")["shift_current"]["diagnostics"]
    assert near["regularisation"]["flagged_kpoints"] == 8
    # The same cause cannot absorb anything beyond PyATB's cut.
    for factor in (1.5, 0.9, 1.001):
        report = judge(
            runs,
            tmp_path / f"split-{factor}",
            "split",
            clean_control=clean_report,
            scfout=True,
            sigma=truth * factor,
        )
        assert report["verdict"] == FAIL, factor
    # Another scfout is not the candidate's: nothing is recomputed from it.
    report = compare(
        "SHIFT_CURRENT",
        reference_from(tmp_path / "split-other", runs["split"], sigma=truth),
        runs["split"],
        clean_control=clean_report,
        scfout=runs["clean"] / "scf.scfout",
    )
    assert report["verdict"] == NOT_COMPARABLE and "is not the candidate's" in json.dumps(
        report["preconditions"]
    )


@pytest.mark.parametrize("name", ["sharp05", "sharp02", "sharp01", "sharp002", "sharp_collinear"])
def test_a_correct_candidate_is_a_clean_control_at_pyatbs_narrow_broadening(runs, tmp_path, name):
    """PyATB's Hartree constant moves every transition by 2.3e-6 of its energy: at eta 0.01 a correct table
    differs from PyATB's by 3.6e-4 of the peak. PyATB's table moved back to QUEER's scale from its own curve meets
    QUEER's value to 1e-4 (agree_after_convention); no candidate number enters, and the 1e-4 itself stays."""
    run = runs[name]
    collinear = name == "sharp_collinear"
    # PyATB: the independent port of its method-1 kernel, or (collinear) the emulation, which meets it to 1e-13.
    pyatb = scheme_table(run) if collinear else pyatb_method_1_table(run)
    block = read(run / "queer_results.json")["shift_current"]
    physical = columns((block["legacy_single_channel"] or block)["sigma_uA_V2"])
    peak = np.max(np.abs(pyatb))
    report = judge(runs, tmp_path / "same", name, sigma=pyatb)
    assert report["verdict"] == PASS and report["semantics"]["rules"]["clean_control"]["eligible"], (
        render_markdown(report)
    )
    assert components(report) <= {AGREE, AFTER_CONVENTION} and classes(report)["clean_control"] == AGREE
    if name in ("sharp02", "sharp01", "sharp002"):
        # The Hartree scale alone moves PyATB's table beyond 1e-4 of its peak; moved back to QUEER's scale from
        # its own curve, it meets QUEER's value.
        assert np.max(np.abs(pyatb - physical)) > 1e-4 * peak
        moved = [
            item for item in report["quantities"] if "moved to QUEER's Hartree constant" in item["detail"]
        ]
        assert moved and all(item["metric"] > report["tolerance_basis"]["base_uA_V2"] for item in moved)
        assert all(item["class"] == AFTER_CONVENTION for item in moved)
    # A wrong table never passes: doubled, negated, its largest component zeroed.
    zeroed = pyatb.copy()
    zeroed[:, int(np.argmax(np.max(np.abs(pyatb), axis=0)))] = 0.0
    for label, wrong in (("doubled", 2 * pyatb), ("negated", -pyatb), ("zeroed", zeroed)):
        report = judge(runs, tmp_path / label, name, sigma=wrong)
        assert report["verdict"] == FAIL and classes(report)["clean_control"] == UNEXPLAINED, label
    # Nor does a physical value 1 % off beside a right emulation.
    copy = tampered(tmp_path / "off", run, lambda values: 1.01 * values, legacy=collinear)
    report = compare("SHIFT_CURRENT", reference_from(tmp_path / "off-ref", copy, sigma=pyatb), copy)
    assert report["verdict"] == FAIL and classes(report)["emulation_consistency"] == UNEXPLAINED


def test_an_error_confined_to_the_flagged_kpoints_never_passes(runs, tmp_path, clean_report):
    """A QUEER error at the k-points PyATB's method 1 cuts moves QUEER's value and the measured shift together
    (shift = spectrum - cut spectrum there). The flagged k-points' records must sum to both, and with the scfout
    each one is recomputed: such an error fails at any size, never documented."""
    run = runs["split"]
    document = read(run / "queer_results.json")
    regularisation = document["shift_current"]["diagnostics"]["regularisation"]
    shift = columns(regularisation["shift_uA_V2"])
    physical = columns(document["shift_current"]["sigma_uA_V2"])
    truth = pyatb_table(run)
    peak = np.max(np.abs(truth))
    assert np.max(np.abs(physical + 2.0 * shift - truth)) > 1e-2 * peak
    for factor in (2.0, 4.0, 8.0, -4.0, -8.0, 50.0, -20.0):

        def bug(block, factor=factor):
            block["sigma_uA_V2"] = as_columns(physical + factor * shift)
            block["diagnostics"]["regularisation"]["shift_uA_V2"] = as_columns((1 + factor) * shift)

        def follow(block, factor=factor):
            # The records follow the error too: each flagged k-point's physical transitions carry it.
            bug(block)
            for item in block["diagnostics"]["regularisation"]["flagged_records"]["kpoints"]:
                own, cut = item["physical"], item["cut"]
                own["weight_uA_V2_eV"] = (
                    np.asarray(own["weight_uA_V2_eV"]).tolist()
                    + (factor * np.asarray(own["weight_uA_V2_eV"])).tolist()
                    + (-factor * np.asarray(cut["weight_uA_V2_eV"])).tolist()
                )
                own["energy_eV"] = own["energy_eV"] * 2 + cut["energy_eV"]

        for label, edit in (("bug", bug), ("follow", follow)):
            copy = edited(tmp_path / f"{label}{factor:g}", run, edit)
            reference = reference_from(tmp_path / f"{label}{factor:g}-ref", copy, sigma=truth)
            for scfout in (None, run / "scf.scfout"):
                report = compare("SHIFT_CURRENT", reference, copy, clean_control=clean_report, scfout=scfout)
                assert DOCUMENTED not in components(report), (label, factor, render_markdown(report))
                if label == "bug":
                    # The records still hold the true transitions: the value contradicts them.
                    assert (
                        report["verdict"] == FAIL and classes(report)["regularisation_record"] == UNEXPLAINED
                    )
                elif scfout is not None:
                    assert report["verdict"] == FAIL, (factor, render_markdown(report))
                    assert classes(report)["regularisation_recomputed"] == UNEXPLAINED
                else:
                    # Consistent records are not trusted: the cut needs the scfout.
                    assert report["verdict"] in (NOT_COMPARABLE, FAIL), (factor, render_markdown(report))
                    if report["verdict"] == NOT_COMPARABLE:
                        assert failed_preconditions(report) == ["regularisation"]
    # A forged flag cannot open the cut on a clean run: its pair spacings flag nothing.
    clean_physical = columns(read(runs["clean"] / "queer_results.json")["shift_current"]["sigma_uA_V2"])

    def forged(block):
        block["sigma_uA_V2"] = as_columns(2 * clean_physical)
        block["diagnostics"]["regularisation"].update(
            flagged_kpoints=1,
            emulated_kpoints=1,
            not_emulated_kpoints=0,
            shift_uA_V2=as_columns(clean_physical),
        )

    copy = edited(tmp_path / "forged", runs["clean"], forged)
    report = compare(
        "SHIFT_CURRENT",
        reference_from(tmp_path / "forged-ref", copy, sigma=scheme_table(runs["clean"])),
        copy,
        clean_control=clean_report,
        scfout=runs["clean"] / "scf.scfout",
    )
    assert report["verdict"] == FAIL and classes(report)["regularisation_record"] == UNEXPLAINED
    # The value without the flagged k-points, or a record of a k-point that is not on the grid, or one missing.
    unflagged = columns(regularisation["sigma_unflagged_uA_V2"])

    def moved_k(block):
        block["diagnostics"]["regularisation"]["flagged_records"]["kpoints"][0]["k"] = [0.1, 0.2, 0.3]

    def unflag(block):
        block["diagnostics"]["regularisation"]["sigma_unflagged_uA_V2"] = as_columns(1.5 * unflagged)

    def dropped(block):
        block["diagnostics"]["regularisation"]["flagged_records"]["kpoints"].pop()

    for label, change in (("unflagged", unflag), ("off-grid", moved_k), ("dropped", dropped)):
        copy = edited(tmp_path / label, run, change)
        reference = reference_from(tmp_path / f"{label}-ref", copy, sigma=truth)
        report = compare(
            "SHIFT_CURRENT", reference, copy, clean_control=clean_report, scfout=run / "scf.scfout"
        )
        assert report["verdict"] == FAIL and classes(report)["regularisation_record"] == UNEXPLAINED, label


def test_no_record_of_the_candidate_widens_a_tolerance(runs, tmp_path, clean_report):
    """A physical value off by 0.5 % to 100 % beside records scaled by a normalization fault must never pass as a
    clean control or be documented. No candidate magnitude enters a tolerance: the window's dropped
    transitions, the emulation's transitions, the bins' moves and the flagged k-points' transitions are exact terms
    that must add up, so inflating any of them fails, with or without a wrong physical value."""
    for name in ("sharp01", "clean"):
        for factor in (1.005, 1.02, 1.1, 2.0):
            copy = tampered(
                tmp_path / f"{name}-{factor}", runs[name], lambda values, factor=factor: factor * values
            )
            reference = reference_from(
                tmp_path / f"{name}-{factor}-ref", copy, sigma=scheme_table(runs[name])
            )
            report = compare("SHIFT_CURRENT", reference, copy)
            assert report["verdict"] == FAIL and classes(report)["clean_control"] == UNEXPLAINED, (
                name,
                factor,
            )

    def inflate(key, factor, error):
        def change(block):
            scheme = block["diagnostics"]["pyatb_scheme"]
            scheme[key] = as_columns(factor * columns(scheme[key]))
            if error != 1.0:
                block["sigma_uA_V2"] = as_columns(error * columns(block["sigma_uA_V2"]))
                regularisation = block["diagnostics"]["regularisation"]
                regularisation["sigma_unflagged_uA_V2"] = as_columns(
                    error * columns(regularisation["sigma_unflagged_uA_V2"])
                )

        return change

    for name, key in (
        ("window", "window_dropped_uA_V2"),
        ("window_sharp", "window_dropped_uA_V2"),
        ("clean", "transitions_uA_V2"),
    ):
        for factor in (10.0, 1e3):
            for error in (1.0, 1.01, 1.1):
                label = f"inflate-{name}-{factor:g}-{error:g}"
                copy = edited(tmp_path / label, runs[name], inflate(key, factor, error))
                reference = reference_from(tmp_path / f"{label}-ref", copy, sigma=scheme_table(runs[name]))
                report = compare("SHIFT_CURRENT", reference, copy, clean_control=clean_report)
                assert report["verdict"] == FAIL and DOCUMENTED not in components(report), label

    def moved(block):
        record = block["diagnostics"]["pyatb_scheme"]["moved_transitions"]
        record["weight_uA_V2_eV"] = (1e3 * np.asarray(record["weight_uA_V2_eV"])).tolist()
        block["binned_uA_V2"] = as_columns(1.1 * columns(block["binned_uA_V2"]))

    def records(block):
        for item in block["diagnostics"]["regularisation"]["flagged_records"]["kpoints"]:
            for part in ("physical", "cut"):
                item[part]["weight_uA_V2_eV"] = (10 * np.asarray(item[part]["weight_uA_V2_eV"])).tolist()

    for name, change in (("split_binned", moved), ("split", records)):
        copy = edited(tmp_path / f"records-{name}", runs[name], change)
        reference = reference_from(tmp_path / f"records-{name}-ref", copy, sigma=pyatb_table(runs[name]))
        for scfout in (None, runs[name] / "scf.scfout"):
            report = compare("SHIFT_CURRENT", reference, copy, clean_control=clean_report, scfout=scfout)
            assert report["verdict"] == FAIL and DOCUMENTED not in components(report), name


def test_a_bug_over_every_kpoint_never_passes(runs, tmp_path, clean_report):
    """A normalization bug over every k-point: every table the runner writes (value, its unflagged part and shift, the
    emulation, its transitions, the window's dropped transitions, the bins' moves and the flagged k-points'
    transitions) twice the truth, consistently. PyATB prints the truth: no scheme reproduces a factor of two."""

    def doubled(block):
        for holder in (block, block["legacy_single_channel"] or {}):
            for key in ("sigma_uA_V2", "binned_uA_V2"):
                if holder.get(key) is not None:
                    holder[key] = as_columns(2 * columns(holder[key]))
        diagnostics = block["diagnostics"]
        regularisation, scheme = diagnostics["regularisation"], diagnostics["pyatb_scheme"]
        for key in ("shift_uA_V2", "sigma_unflagged_uA_V2", "binned_shift_uA_V2", "binned_unflagged_uA_V2"):
            if regularisation.get(key) is not None:
                regularisation[key] = as_columns(2 * columns(regularisation[key]))
        for key in ("sigma_uA_V2", "transitions_uA_V2", "window_dropped_uA_V2"):
            if scheme.get(key) is not None:
                scheme[key] = as_columns(2 * columns(scheme[key]))
        for item in regularisation["flagged_records"]["kpoints"]:
            for part in ("physical", "cut"):
                item[part]["weight_uA_V2_eV"] = (2 * np.asarray(item[part]["weight_uA_V2_eV"])).tolist()
        if scheme.get("moved_transitions"):
            record = scheme["moved_transitions"]
            record["weight_uA_V2_eV"] = (2 * np.asarray(record["weight_uA_V2_eV"])).tolist()

    for name in ("window", "split", "split_window", "clean", "sharp01", "binned", "split_binned"):
        copy = edited(tmp_path / name, runs[name], doubled)
        reference = reference_from(tmp_path / f"{name}-ref", copy, sigma=scheme_table(runs[name]))
        for scfout in (None, runs[name] / "scf.scfout"):
            report = compare("SHIFT_CURRENT", reference, copy, clean_control=clean_report, scfout=scfout)
            assert DOCUMENTED not in components(report), (name, render_markdown(report))
            if name.startswith("split") and scfout is None:
                # Consistent records of flagged k-points are not trusted: undecided until recomputed.
                assert report["verdict"] == NOT_COMPARABLE and failed_preconditions(report) == [
                    "regularisation"
                ]
            else:
                assert report["verdict"] == FAIL, (name, render_markdown(report))
    # With an unbounded cause and a converged candidate the plan documents it, as before.
    copy = edited(tmp_path / "scissors", runs["scissors"], doubled)
    reference = reference_from(tmp_path / "scissors-ref", copy, sigma=scheme_table(runs["scissors"]))
    report = compare("SHIFT_CURRENT", reference, copy, clean_control=clean_report)
    assert report["verdict"] == PASS_DOCUMENTED and report["semantics"]["rules"]["unbounded_causes"] == [
        "scissors"
    ]


def test_noise_on_pyatbs_table_counts_against_the_tolerance(runs, tmp_path):
    """PyATB's own table with noise of a share of its peak on every value: a clean control holds to 1e-4 of the
    peak beside the Hartree scale (about 3e-5 of it at eta 0.1 with domega = eta, not removed on that step)."""
    truth = scheme_table(runs["clean"])
    peak = np.max(np.abs(truth))
    signs = np.random.default_rng(4).choice([-1.0, 1.0], size=truth.shape)
    for share, verdict in ((1e-6, PASS), (3e-5, PASS), (1.2e-4, FAIL), (1e-3, FAIL)):
        report = judge(runs, tmp_path / f"noise-{share:g}", "clean", sigma=truth + share * peak * signs)
        assert report["verdict"] == verdict, (share, render_markdown(report))


def test_a_missing_record_is_a_contradiction(runs, tmp_path, clean_report):
    """A candidate without diagnostics.regularisation (with or without pyatb_scheme) must not open an unmeasured
    cause. The runner always writes its records; one missing is a contradiction that fails, here on a
    converged clean-configuration method-1 candidate whose physical value is 2 or 0.3 times the truth."""
    run = runs["converged"]
    convergence = read(run / "queer_results.json")["shift_current"]["diagnostics"]["self_convergence"]
    assert convergence["legacy_single_channel"]["converged"]
    truth = scheme_table(run)
    for drop in (
        ("regularisation",),
        ("regularisation", "pyatb_scheme"),
        ("pyatb_scheme",),
        ("pyatb_scheme.transitions_uA_V2",),
        ("pyatb_scheme.window_dropped_uA_V2",),
    ):
        for factor in (2.0, 0.3):

            def change(block, drop=drop, factor=factor):
                legacy = block["legacy_single_channel"]
                legacy["sigma_uA_V2"] = as_columns(factor * columns(legacy["sigma_uA_V2"]))
                for key in drop:
                    if "." in key:
                        outer, inner = key.split(".")
                        block["diagnostics"][outer].pop(inner)
                    else:
                        block["diagnostics"].pop(key)

            label = "-".join(drop) + f"-{factor:g}"
            copy = edited(tmp_path / label, run, change)
            reference = reference_from(tmp_path / f"{label}-ref", copy, sigma=truth)
            report = compare("SHIFT_CURRENT", reference, copy, clean_control=clean_report)
            assert report["verdict"] == FAIL, (label, render_markdown(report))
            assert classes(report)["candidate_record"] == UNEXPLAINED and DOCUMENTED not in components(report)
            assert report["semantics"]["rules"]["unbounded_causes"] == []
    # Without smearing the moved transitions, with flagged k-points their records.
    for name, key in (
        ("binned", "pyatb_scheme.moved_transitions"),
        ("split", "regularisation.flagged_records"),
    ):

        def missing(block, key=key):
            outer, inner = key.split(".")
            block["diagnostics"][outer].pop(inner)

        copy = edited(tmp_path / f"missing-{name}", runs[name], missing)
        reference = reference_from(tmp_path / f"missing-{name}-ref", copy)
        report = compare("SHIFT_CURRENT", reference, copy, clean_control=clean_report)
        assert report["verdict"] == FAIL and classes(report)["candidate_record"] == UNEXPLAINED, name


def test_the_build_identity_hashes_every_source_file_of_the_package(runs, tmp_path, clean_report):
    """A clean control vouches for the build of every module a run executes, not a hand-picked list: a change to
    any .py file of the package (here the delta kernels of physics/broadening.py) makes another build."""
    root = Path(queer.__file__).resolve().parent
    sources = package_sources()
    assert sources == sorted(path.relative_to(root).as_posix() for path in root.rglob("*.py"))
    for module in (
        "io/openmx.py",
        "physics/broadening.py",
        "observables/_response_common.py",
        "physics/statistics.py",
        "geometry/lattice.py",
        "models/contracts.py",
        "execution.py",
        "compat/generator/runner.py",
        "compat/generator/results.py",
        "compat/generator/pyatb_conventions.py",
    ):
        assert module in sources, module
    build = clean_report["semantics"]["rules"]["build"]
    assert build["queer_source_sha256"] == build["comparator_sha256"] == package_source_sha256() is not None
    # A copy of the package is the same build until one module changes.
    devbuild = tmp_path / "devbuild"
    copy = shutil.copytree(root, devbuild / "queer", ignore=shutil.ignore_patterns("__pycache__"))
    assert package_source_sha256(copy) == package_source_sha256()
    kernel = copy / "physics" / "broadening.py"
    kernel.write_text(kernel.read_text() + "\n# another build\n")
    assert package_source_sha256(copy) != package_source_sha256()
    # A run by that build: this build's clean control is not its clean control.
    directory = tmp_path / "window-devbuild"
    directory.mkdir()
    scfout, model = RUNS["window"][:2]
    scfout(directory / "scf.scfout", **model)
    params = read(runs["window"] / "queer_params.json")
    write(directory / "queer_params.json", params)
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); import queer; "
        "assert queer.__file__.startswith(sys.argv[1]), queer.__file__; "
        "from queer.compat.generator.runner import run; "
        "run(None, sys.argv[2], sys.argv[3], log=lambda line: None)"
    )
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    subprocess.run(
        [sys.executable, "-c", script, str(devbuild), str(directory / "queer_params.json"), str(directory)],
        check=True,
        env=environment,
        cwd=tmp_path,
    )
    produced = read(directory / "queer_results.json")["shift_current"]["build"]["queer_source_sha256"]
    assert produced == package_source_sha256(copy) != build["queer_source_sha256"]
    report = compare(
        "SHIFT_CURRENT",
        reference_from(tmp_path / "devbuild-ref", directory),
        directory,
        clean_control=clean_report,
    )
    assert classes(report)["clean_control_release"] == NOT_APPLICABLE
    assert report["verdict"] == NOT_COMPARABLE and NO_CLEAN_CONTROL in json.dumps(report["preconditions"])


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
