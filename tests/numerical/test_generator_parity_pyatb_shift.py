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
from queer.observables.nonlinear import shift_current_point, shift_current_settings

SPINOR = {"occ_band": 2, "omega": [0.3, 6.5], "domega": 0.02, "eta": 0.1, "grid": [3, 3, 2]}
COLLINEAR = {"occ_band": 1, "omega": [0.2, 4.5], "domega": 0.04, "eta": 0.1, "grid": [12, 12, 1]}
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
    # PyATB's default broadening (eta 0.01) on its default step (domega 0.01), and on half of it: steps wider than
    # 0.4 eta, where its table cannot be moved to QUEER's energy scale from its own curve.
    "pyatb_default": (shift_spinor_scfout, {}, 4, {**SPINOR, "eta": 0.01, "domega": 0.01}, None),
    "half_step": (shift_spinor_scfout, {}, 4, {**SPINOR, "eta": 0.01, "domega": 0.005}, None),
    # A response that vanishes: no odd mixing and no spin-orbit coupling leave every transition's integrand zero.
    "centro": (
        shift_spinor_scfout,
        {"mixing_odd": (0.0, 0.0, 0.0), "soc": 0.0, "soc_z": 0.0},
        4,
        SPINOR,
        None,
    ),
    # Time reversal in a nonorthogonal basis with position moments: exact Kramers pairs at the 8 TRIM points;
    # with on-site dipoles PyATB's own treatment of them depends on its eigensolver's basis inside them.
    "kramers_nonorthogonal": (
        shift_spinor_scfout,
        {"exchange": 0.0, "overlap_hopping": (0.06, 0.04, 0.03)},
        4,
        {**SPINOR, "grid": [4, 4, 4]},
        None,
    ),
    "kramers_dipole": (
        shift_spinor_scfout,
        {"exchange": 0.0, "overlap_hopping": (0.06, 0.04, 0.03), "dipole": 0.6},
        4,
        {**SPINOR, "grid": [2, 2, 2]},
        None,
    ),
}
CALIBRATION_SCFOUT = Path(
    os.environ.get("QUEER_PYATB_CALIBRATION_SCFOUT")
    or Path.home() / ".cache" / "queer-parity-data" / "pyatb-calib-60fc6b2a" / "test.scfout"
)


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
    report = judge(runs, tmp_path / "alone", "window", scfout=True)
    assert report["verdict"] == NOT_COMPARABLE and failed_preconditions(report) == ["clean_control"]
    assert NO_CLEAN_CONTROL in json.dumps(report["preconditions"])
    # What the window drops is QUEER's own per-transition record: documented only once it is recomputed from the
    # scfout.
    report = judge(runs, tmp_path / "unrecomputed", "window", clean_control=clean_report)
    assert report["verdict"] == NOT_COMPARABLE and failed_preconditions(report) == ["regularisation"]
    assert "window record is not recomputed" in json.dumps(report["preconditions"])
    report = judge(runs, tmp_path / "released", "window", clean_control=clean_report, scfout=True)
    assert report["verdict"] == PASS_DOCUMENTED, render_markdown(report)
    assert DOCUMENTED in components(report) and components(report) <= {DOCUMENTED, AGREE}
    assert classes(report)["emulation_consistency"] == AGREE == classes(report)["window_recomputed"]
    documented = [item for item in report["quantities"] if item["class"] == DOCUMENTED]
    assert all("PyATB's scheme reproduces it" in item["detail"] for item in documented)
    assert all("transitions its window drops" in item["detail"] for item in documented)
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
    report = judge(runs, tmp_path / "window-sharp", "window_sharp", clean_control=clean_report, scfout=True)
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
    report = judge(
        runs, tmp_path / "physical", "collinear", sigma=columns(collinear["sigma_uA_V2"]), scfout=True
    )
    assert report["verdict"] == FAIL and report["semantics"]["rules"]["clean_control"]["eligible"]
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
        scfout=runs["window"] / "scf.scfout",
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
    assert main([*arguments, "--scfout", str(runs["window"] / "scf.scfout")]) == 0
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
    # A time-reversal-symmetric SOC model is a clean control once its exact degeneracies are recomputed from the
    # scfout (QUEER's tolerance and the spacings it treats as exact, by the comparator's own levels): any symmetric
    # error fails. Without the scfout its exclusion of the Kramers pairs rests on QUEER's tolerance: no clean control.
    report = judge(runs, tmp_path / "kramers", "kramers", scfout=True)
    assert components(report) == {AGREE} and classes(report)["degeneracies_recomputed"] == AGREE
    assert report["semantics"]["rules"]["clean_control"]["eligible"]
    report = judge(runs, tmp_path / "kramers-alone", "kramers")
    conditions = report["semantics"]["rules"]["clean_control"]["conditions"]
    assert report["verdict"] == PASS and not conditions["exact_degeneracies_verified"]
    for factor in (2.0, 0.5, 1 / 8):
        report = judge(
            runs,
            tmp_path / f"kramers-{factor}",
            "kramers",
            sigma=scheme_table(runs["kramers"]) * factor,
            scfout=True,
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
    # A collinear run is a clean control once its spin-down channel, which PyATB never reads, is recomputed.
    report = judge(runs, tmp_path / "same", name, sigma=pyatb, scfout=collinear)
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
        report = judge(runs, tmp_path / label, name, sigma=wrong, scfout=collinear)
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
        for key in ("moved_transitions", "window_transitions"):
            if scheme.get(key):
                record = scheme[key]
                record["weight_uA_V2_eV"] = (2 * np.asarray(record["weight_uA_V2_eV"])).tolist()
        channels = diagnostics["spin_channels"] or {}
        for key in ("up_uA_V2", "down_uA_V2", "binned_up_uA_V2", "binned_down_uA_V2"):
            if channels.get(key) is not None:
                channels[key] = as_columns(2 * columns(channels[key]))
        for rounding in (
            diagnostics["rounding"],
            channels.get("down_rounding") or {"kpoint_scale_uA_V2_eV": []},
        ):
            rounding["kpoint_scale_uA_V2_eV"] = (2 * np.asarray(rounding["kpoint_scale_uA_V2_eV"])).tolist()

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
            if scfout is not None:
                # Recomputed from the scfout, the doubled records contradict it.
                assert {"rounding_recomputed"} <= {
                    item["name"] for item in report["quantities"] if item["class"] == UNEXPLAINED
                }, name
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


def kpoint_record(run_dir, k, *, cut=False):
    """QUEER's own transitions at ``k`` of a run as its flagged-k-point record holds them (PyATB's conventions)."""
    import dataclasses

    from queer.compat.generator.openmx_shift import _transitions

    block = read(run_dir / "queer_results.json")["shift_current"]
    params = {**SHIFT_DEFAULTS, **read(run_dir / "queer_params.json")["parameters"]}
    model = OpenMXModel.from_scfout(run_dir / "scf.scfout")
    view = model if model.spin_mode == "soc" else single_channel_model(model)
    occupied = int(params["occ_band"])
    settings = shift_current_settings(
        view,
        np.asarray(block["omega_eV"]),
        0.0,
        eta=float(params["eta"]) / math.sqrt(2),
        smearing="gaussian",
        occupied_bands=occupied,
        method="analytic",
        scissors_occupied_bands=occupied,
    )
    if cut:
        settings = dataclasses.replace(settings, degeneracy_tol=1e-4 * HARTREE_EV / PYATB_HARTREE_EV)
    energies, weights = _transitions(shift_current_point(view, np.asarray(k, float), settings))
    grid = [int(value) for value in params["grid"]]
    factor = block["convention"]["export_sign"] * block["convention"]["pyatb_spin_factor"]
    factor *= conversion(view) / int(np.prod(grid))
    return {"energy_eV": energies.tolist(), "weight_uA_V2_eV": (factor * weights).tolist()}


def test_a_response_that_vanishes_agrees_within_its_rounding_floor_once_recomputed(runs, tmp_path):
    """A shift current that vanishes by symmetry: QUEER prints its zeros, PyATB its rounding noise. They agree within
    the rounding floor (the sum of the k-points' rounding scales, each the conditioning times its transitions' product
    moduli), recomputed from the scfout; the symmetry gate has no sign to compare; such a table is no clean control
    (it vouches for no build); an inflated floor contradicts the scfout."""
    run = runs["centro"]
    block = read(run / "queer_results.json")["shift_current"]
    assert np.max(np.abs(columns(block["sigma_uA_V2"]))) < 1e-12
    port = pyatb_method_1_table(run)
    assert np.max(np.abs(port)) < 1e-20
    # PyATB's rounding noise (the port's own may vanish exactly): 1e-30 microA/V^2 of either sign.
    noise = port + 1e-30 * np.random.default_rng(3).choice([-1.0, 1.0], size=port.shape)
    for label, pyatb in (("noise", noise), ("zeros", np.zeros_like(noise))):
        report = judge(runs, tmp_path / label, "centro", sigma=pyatb, scfout=True)
        assert report["verdict"] == PASS and components(report) <= {AGREE}, render_markdown(report)
        assert (
            classes(report)["symmetry"] == NOT_APPLICABLE and classes(report)["rounding_recomputed"] == AGREE
        )
        clean = report["semantics"]["rules"]["clean_control"]
        assert not clean["eligible"] and not clean["conditions"]["nontrivial_table"]
        # Without the scfout the floor rests on the candidate's own scales: not judged (unless the two runs print
        # the same zeros).
        report = judge(runs, tmp_path / f"{label}-alone", "centro", sigma=pyatb)
        if label == "noise":
            assert report["verdict"] == NOT_COMPARABLE and "rounding" in failed_preconditions(report)
        else:
            assert report["verdict"] == PASS and not report["semantics"]["rules"]["clean_control"]["eligible"]
    # A trivial table licenses nothing: as a release report it is no clean control.
    trivial = judge(runs, tmp_path / "trivial", "centro", sigma=noise, scfout=True)
    report = judge(runs, tmp_path / "licensed", "window", clean_control=trivial, scfout=True)
    assert report["semantics"]["rules"]["clean_control"]["release"] == "not_a_clean_control"
    assert report["verdict"] == NOT_COMPARABLE and NO_CLEAN_CONTROL in json.dumps(report["preconditions"])

    def inflate(block):
        rounding = block["diagnostics"]["rounding"]
        rounding["kpoint_scale_uA_V2_eV"] = (1e6 * np.asarray(rounding["kpoint_scale_uA_V2_eV"])).tolist()

    for name, sigma in (("centro", noise), ("clean", 1.01 * scheme_table(runs["clean"]))):
        copy = edited(tmp_path / f"inflated-{name}", runs[name], inflate)
        reference = reference_from(tmp_path / f"inflated-{name}-ref", copy, sigma=sigma)
        report = compare("SHIFT_CURRENT", reference, copy, scfout=copy / "scf.scfout")
        assert report["verdict"] == FAIL and classes(report)["rounding_recomputed"] == UNEXPLAINED, name


@pytest.mark.skipif(
    not CALIBRATION_SCFOUT.is_file(), reason=f"no OpenMX calibration scfout at {CALIBRATION_SCFOUT}"
)
def test_the_real_calibration_scfout_whose_shift_current_vanishes_is_compared_honestly(tmp_path_factory):
    """The real OpenMX calibration scfout (honeycomb with inversion, spin-orbit coupling, a nonorthogonal basis with
    position moments): its shift current vanishes (about 1e-9 microA/V^2 against k-point weights of order 1-10). An
    honest run against its own PyATB scheme is no contradiction: with the scfout it agrees within the recomputed
    rounding floor (the window's dropped tails lie below it), without it it is not judged."""
    root = tmp_path_factory.mktemp("real")
    for occupied in (3, 4):
        directory = root / f"occ{occupied}"
        directory.mkdir()
        shutil.copy(CALIBRATION_SCFOUT, directory / "scf.scfout")
        params = {
            "step_type": "QUEER_SHIFT_CURRENT",
            "counterpart_step_type": "PYATB_SHIFT_CURRENT",
            "input_parameters": {"nspin": 4},
            "parameters": {
                "occ_band": occupied,
                "omega": [0.0, 12.0],
                "domega": 0.02,
                "eta": 0.1,
                "grid": [6, 6, 1],
            },
            "workers": 1,
        }
        write(directory / "queer_params.json", params)
        run(None, directory / "queer_params.json", directory, log=lambda line: None)
        reference = reference_from(root / f"ref{occupied}", directory)
        report = compare("SHIFT_CURRENT", reference, directory, scfout=directory / "scf.scfout")
        assert report["verdict"] == PASS and components(report) <= {AGREE, AFTER_CONVENTION}, render_markdown(
            report
        )
        assert classes(report)["emulation_consistency"] == AGREE == classes(report)["rounding_recomputed"]
        assert classes(report)["symmetry"] == NOT_APPLICABLE
        report = compare("SHIFT_CURRENT", reference, directory)
        assert report["verdict"] == NOT_COMPARABLE and "rounding" in failed_preconditions(report)
        assert classes(report)["emulation_consistency"] == AGREE


@pytest.mark.skipif(
    not CALIBRATION_SCFOUT.is_file(), reason=f"no OpenMX calibration scfout at {CALIBRATION_SCFOUT}"
)
def test_a_flagged_record_at_a_time_reversal_invariant_point_of_the_real_scfout_meets_its_recomputation(
    tmp_path,
):
    """Where the integrand vanishes by symmetry (Gamma and M of the centrosymmetric calibration scfout) its rounding
    is that of the terms that cancel, measured by the transcription's two orbital bases: an honest record there meets
    QUEER's estimator and the transcription; at K and generic k-points a 1e-6 error in the record does not."""
    from queer.compat.generator.parity_pyatb import PyatbRun, QueerRun
    from queer.compat.generator.parity_shift import _ShiftComparison

    directory = tmp_path / "real"
    directory.mkdir()
    shutil.copy(CALIBRATION_SCFOUT, directory / "scf.scfout")
    params = {
        "step_type": "QUEER_SHIFT_CURRENT",
        "counterpart_step_type": "PYATB_SHIFT_CURRENT",
        "input_parameters": {"nspin": 4},
        "parameters": {"occ_band": 3, "omega": [0.0, 12.0], "domega": 0.02, "eta": 0.1, "grid": [6, 6, 1]},
        "workers": 1,
    }
    write(directory / "queer_params.json", params)
    run(None, directory / "queer_params.json", directory, log=lambda line: None)
    comparison = _ShiftComparison(
        PyatbRun.load(reference_from(tmp_path / "ref", directory)),
        QueerRun.load(directory),
        directory / "scf.scfout",
    )
    assert comparison.check_documents() and comparison.check_family()
    assert comparison.guarded(comparison.check_inputs) and comparison.guarded(comparison.check_parameters)
    assert comparison.check_spin() and comparison.check_sign()
    recompute = comparison.recomputation()
    assert recompute is not None
    points = {"Gamma": [0, 0, 0], "M": [0.5, 0, 0], "M'": [0.5, 0.5, 0], "K": [1 / 3, 1 / 3, 0]}
    points.update({"generic": [0.086, 0.237, 0.0], "near Gamma": [0.005, 0.0, 0.0]})
    for name, k in points.items():
        item = {
            "k": k,
            "pair_gap_eV": 5e-5,
            "physical": kpoint_record(directory, k),
            "cut": kpoint_record(directory, k, cut=True),
        }
        ratios = comparison.recompute_items([item], recompute)["kpoints"][0]["ratios"]
        assert max(ratios.values()) < 0.05, (name, ratios)
        if name in ("K", "generic"):
            item["physical"]["weight_uA_V2_eV"] = (
                (1 + 1e-6) * np.asarray(item["physical"]["weight_uA_V2_eV"])
            ).tolist()
            ratios = comparison.recompute_items([item], recompute)["kpoints"][0]["ratios"]
            assert ratios["transcription"] > 1 and ratios["queer"] > 1, (name, ratios)


@pytest.mark.parametrize("name", ["pyatb_default", "half_step"])
def test_a_step_wider_than_0_4_eta_is_no_clean_control(runs, tmp_path, clean_report, name):
    """PyATB's default eta 0.01 with domega 0.01 or 0.005: its Hartree constant moves a correct table beyond 1e-4 of
    its peak, and on such a step its table cannot be moved back from its own curve. Such a run is no clean control: a
    correct candidate is not comparable (never failed), a wrong one beyond the Hartree scale's reach fails."""
    truth = pyatb_method_1_table(runs[name])
    for supplied in (None, clean_report):
        report = judge(
            runs,
            tmp_path / f"truth-{supplied is None}",
            name,
            clean_control=supplied,
            sigma=truth,
            scfout=True,
        )
        conditions = report["semantics"]["rules"]["clean_control"]["conditions"]
        assert (
            not conditions["step_at_most_0.4_eta"]
            and not report["semantics"]["rules"]["clean_control"]["eligible"]
        )
        assert report["verdict"] == NOT_COMPARABLE, render_markdown(report)
        assert "pyatb_scheme_resolution" in failed_preconditions(
            report
        ) or "clean_control" in failed_preconditions(report)
        assert DOCUMENTED not in components(report)
    for factor in (1.1, -1.0):
        report = judge(
            runs,
            tmp_path / f"wrong-{factor}",
            name,
            clean_control=clean_report,
            sigma=truth * factor,
            scfout=True,
        )
        assert report["verdict"] == FAIL, (factor, render_markdown(report))


def test_a_degeneracy_tolerance_bug_contradicts_the_scfout(tmp_path, monkeypatch, clean_report):
    """QUEER's degeneracy tolerance too large (5e-5 eV for split Kramers pairs 4e-5 eV apart): it counts them exact
    and flags nothing, so its diagnostics agree with its value. The comparator measures the scfout's own tolerance
    and levels: a contradiction; without the scfout such a run is no clean control."""
    original = OpenMXModel.from_scfout.__func__

    def patched(cls, path, *args, **kwargs):
        model = original(cls, path, *args, **kwargs)
        model.degeneracy_tolerance = 5e-5
        model.metadata["degeneracy_tolerance_eV"] = 5e-5
        return model

    monkeypatch.setattr(OpenMXModel, "from_scfout", classmethod(patched))
    bugged = candidate(
        tmp_path / "bugged", shift_spinor_scfout, {"exchange": 2e-5}, 4, {**SPINOR, "grid": [4, 4, 4]}
    )
    monkeypatch.undo()
    diagnostics = read(bugged / "queer_results.json")["shift_current"]["diagnostics"]
    assert (
        diagnostics["exact_degenerate_kpoints"] == 8 and diagnostics["regularisation"]["flagged_kpoints"] == 0
    )
    truth = pyatb_table(bugged)
    for supplied in (None, clean_report):
        reference = reference_from(tmp_path / f"ref-{supplied is None}", bugged, sigma=truth)
        report = compare(
            "SHIFT_CURRENT", reference, bugged, clean_control=supplied, scfout=bugged / "scf.scfout"
        )
        assert report["verdict"] == FAIL and classes(report)["degeneracies_recomputed"] == UNEXPLAINED
        report = compare("SHIFT_CURRENT", reference, bugged, clean_control=supplied)
        clean = report["semantics"]["rules"]["clean_control"]
        assert not clean["eligible"] and not clean["conditions"]["exact_degeneracies_verified"]
    # A record of the exact degeneracies that names spacings the scfout does not have contradicts it too.

    def forge(block):
        for item in block["diagnostics"]["exact_degeneracies"]["kpoints"]:
            item["pairs"] = [[0, 2, 0.0]]

    kramers = candidate(
        tmp_path / "kramers", shift_spinor_scfout, {"exchange": 0.0}, 4, {**SPINOR, "grid": [4, 4, 4]}
    )
    copy = edited(tmp_path / "forged", kramers, forge)
    report = compare(
        "SHIFT_CURRENT", reference_from(tmp_path / "forged-ref", copy), copy, scfout=copy / "scf.scfout"
    )
    assert report["verdict"] == FAIL and classes(report)["degeneracies_recomputed"] == UNEXPLAINED


def test_a_collinear_spin_down_bug_never_passes_as_a_clean_control(runs, tmp_path):
    """PyATB reads only the spin-up channel of a collinear scfout. QUEER's physical value must be its two channels'
    sum (channel_sum), and its spin-down channel is recomputed from the scfout (spin_down_recomputed): a bug there
    fails, and without the scfout the run is no clean control."""
    run_dir = runs["collinear"]

    def physical_only(block):
        channels = block["diagnostics"]["spin_channels"]
        up, down = columns(channels["up_uA_V2"]), columns(channels["down_uA_V2"])
        block["sigma_uA_V2"] = as_columns(up + 3 * down)

    def consistent(block):
        physical_only(block)
        channels = block["diagnostics"]["spin_channels"]
        channels["down_uA_V2"] = as_columns(3 * columns(channels["down_uA_V2"]))

    copy = edited(tmp_path / "physical-only", run_dir, physical_only)
    report = compare("SHIFT_CURRENT", reference_from(tmp_path / "physical-only-ref", copy), copy)
    assert report["verdict"] == FAIL and classes(report)["channel_sum"] == UNEXPLAINED
    copy = edited(tmp_path / "consistent", run_dir, consistent)
    reference = reference_from(tmp_path / "consistent-ref", copy)
    report = compare("SHIFT_CURRENT", reference, copy, scfout=copy / "scf.scfout")
    assert report["verdict"] == FAIL and classes(report)["spin_down_recomputed"] == UNEXPLAINED
    report = compare("SHIFT_CURRENT", reference, copy)
    clean = report["semantics"]["rules"]["clean_control"]
    assert not clean["eligible"] and not clean["conditions"]["spin_down_recomputed"]
    # The honest run is a clean control with the scfout.
    report = judge(runs, tmp_path / "honest", "collinear", scfout=True)
    assert report["verdict"] == PASS and classes(report)["spin_down_recomputed"] == AGREE
    assert report["semantics"]["rules"]["clean_control"]["eligible"]


def test_the_window_and_bin_records_are_recomputed_from_the_scfout(runs, tmp_path, clean_report):
    """What PyATB's window drops and its bins move is QUEER's own per-transition record. A record forged to carry a
    value error (consistently with the window table, the emulation's transitions or the moves) contradicts the
    scfout; without it nothing rests on the record: not comparable, never documented."""
    run_dir = runs["window"]

    def outside(block):
        # A bug in the weights of the transitions outside PyATB's window, carried by every table and record.
        scheme = block["diagnostics"]["pyatb_scheme"]
        dropped = columns(scheme["window_dropped_uA_V2"])
        block["sigma_uA_V2"] = as_columns(columns(block["sigma_uA_V2"]) + 2 * dropped)
        regularisation = block["diagnostics"]["regularisation"]
        regularisation["sigma_unflagged_uA_V2"] = as_columns(
            columns(regularisation["sigma_unflagged_uA_V2"]) + 2 * dropped
        )
        scheme["window_dropped_uA_V2"] = as_columns(3 * dropped)
        scheme["transitions_uA_V2"] = as_columns(columns(scheme["transitions_uA_V2"]) + 2 * dropped)
        record = scheme["window_transitions"]
        record["weight_uA_V2_eV"] = (3 * np.asarray(record["weight_uA_V2_eV"])).tolist()

    copy = edited(tmp_path / "outside", run_dir, outside)
    reference = reference_from(tmp_path / "outside-ref", copy, sigma=scheme_table(run_dir))
    report = compare("SHIFT_CURRENT", reference, copy, clean_control=clean_report, scfout=copy / "scf.scfout")
    assert report["verdict"] == FAIL and classes(report)["window_recomputed"] == UNEXPLAINED
    report = compare("SHIFT_CURRENT", reference, copy, clean_control=clean_report)
    assert report["verdict"] == NOT_COMPARABLE and DOCUMENTED not in components(report)
    # A forged window table alone contradicts the record it must be the Gaussian of.

    def table_only(block):
        scheme = block["diagnostics"]["pyatb_scheme"]
        scheme["window_dropped_uA_V2"] = as_columns(10 * columns(scheme["window_dropped_uA_V2"]))

    copy = edited(tmp_path / "table-only", run_dir, table_only)
    report = compare(
        "SHIFT_CURRENT", reference_from(tmp_path / "table-only-ref", copy), copy, clean_control=clean_report
    )
    assert report["verdict"] == FAIL and classes(report)["emulation_consistency"] == UNEXPLAINED
    # Bins: a chain of moves just above PyATB's bin edges that carries an error in the histogram away (each entry's
    # rows follow from its energy), and an entry claimed to lie at an edge that does not.
    binned = runs["binned"]
    document = read(binned / "queer_results.json")["shift_current"]
    histogram = columns(document["binned_uA_V2"])
    params = read(binned / "queer_params.json")["parameters"]
    start, domega = float(params["omega"][0]), float(params["domega"])
    bump = len(histogram) // 2
    error = 3 * np.max(np.abs(histogram))
    scale = PYATB_HARTREE_EV / HARTREE_EV

    def chain(block):
        values = columns(block["binned_uA_V2"])
        values[bump] += error
        block["binned_uA_V2"] = as_columns(values)
        regularisation = block["diagnostics"]["regularisation"]
        unflagged = columns(regularisation["binned_unflagged_uA_V2"])
        unflagged[bump] += error
        regularisation["binned_unflagged_uA_V2"] = as_columns(unflagged)
        record = block["diagnostics"]["pyatb_scheme"]["moved_transitions"]
        for row in range(bump, -1, -1):
            # Just above QUEER's edge start + row domega, below PyATB's: QUEER's row, PyATB's one lower.
            edge = start + row * domega
            energy = edge * (1 + 0.5 * (1 - scale))
            assert energy > edge > energy * scale
            record["energy_eV"].append(energy)
            record["weight_uA_V2_eV"].append([error] * len(SHIFT_COMPONENTS))
            record["queer_row"].append(row)
            record["pyatb_row"].append(row - 1)
            record["ambiguous"].append(False)
            record["k_index"].append(0)

    copy = edited(tmp_path / "chain", binned, chain)
    reference = reference_from(tmp_path / "chain-ref", copy, sigma=scheme_table(binned))
    report = compare("SHIFT_CURRENT", reference, copy, clean_control=clean_report, scfout=copy / "scf.scfout")
    assert report["verdict"] == FAIL, render_markdown(report)
    report = compare("SHIFT_CURRENT", reference, copy, clean_control=clean_report)
    assert report["verdict"] != PASS and DOCUMENTED not in components(report), render_markdown(report)

    def stray(block):
        record = block["diagnostics"]["pyatb_scheme"]["moved_transitions"]
        record["energy_eV"].append(start + 10.5 * domega)
        record["weight_uA_V2_eV"].append([1.0] * len(SHIFT_COMPONENTS))
        record["queer_row"].append(10)
        record["pyatb_row"].append(-1)
        record["ambiguous"].append(True)
        record["k_index"].append(0)

    copy = edited(tmp_path / "stray", binned, stray)
    report = compare(
        "SHIFT_CURRENT", reference_from(tmp_path / "stray-ref", copy), copy, clean_control=clean_report
    )
    assert report["verdict"] == FAIL and classes(report)["emulation_consistency"] == UNEXPLAINED
    assert "lie at none" in json.dumps(report["quantities"])


def test_a_transition_at_an_edge_is_placed_on_either_side(tmp_path, clean_report):
    """A transition within 1e-9 eV of PyATB's window end (its Gaussian), or of one of its bin edges (its bins), may
    fall on either side in PyATB's arithmetic: either PyATB table is reproduced, each placement tried."""
    from queer.compat.generator.openmx_shift import _transitions

    probe = shift_spinor_scfout(tmp_path / "probe.scfout")
    model = OpenMXModel.from_scfout(probe)
    settings = shift_current_settings(
        model, np.array([1.0]), 0.0, eta=0.07, smearing="gaussian", occupied_bands=2, method="analytic"
    )
    energies, weights = _transitions(shift_current_point(model, np.zeros(3), settings))
    level = float(np.sort(energies)[1])
    members = np.flatnonzero(np.abs(energies - level) < 1e-12)
    scale = PYATB_HARTREE_EV / HARTREE_EV
    conversion_k = pyatb_shift_constant_ratio() * conversion(model) / 18
    # The Gaussian: PyATB's window [start, start + 40 domega) ends at s E.
    domega = 0.02
    start = scale * level - 40 * domega
    window = candidate(
        tmp_path / "window",
        shift_spinor_scfout,
        {},
        4,
        {**SPINOR, "omega": [start, start + 39.5 * domega], "domega": domega},
    )
    truth = pyatb_table(window)
    contribution, _ = pyatb_scheme_spectrum(
        PyatbShiftScheme(start, domega, 40, 0.1, 1, True), energies[members], weights[members]
    )
    kept = start + 40 * domega > scale * level
    other = truth - conversion_k * contribution if kept else truth + conversion_k * contribution
    assert np.max(np.abs(other - truth)) > 1e-2 * np.max(np.abs(truth))
    for label, table in (("pyatb", truth), ("other", other)):
        reference = reference_from(tmp_path / f"window-{label}", window, sigma=table)
        report = compare(
            "SHIFT_CURRENT", reference, window, clean_control=clean_report, scfout=window / "scf.scfout"
        )
        assert report["verdict"] == PASS_DOCUMENTED, (label, render_markdown(report))
        assert report["tolerance_basis"]["scheme"]["edge_transitions"] == len(members)
    # The bins: PyATB's bin edge start + 60 domega at s E.
    domega = 0.05
    start = scale * level - 60 * domega
    binned = candidate(
        tmp_path / "binned",
        shift_spinor_scfout,
        {},
        4,
        {**SPINOR, "smearing_method": 0, "omega": [start, start + 100.5 * domega], "domega": domega},
    )
    record = read(binned / "queer_results.json")["shift_current"]["diagnostics"]["pyatb_scheme"][
        "moved_transitions"
    ]
    assert sum(record["ambiguous"]) == len(members)
    truth = pyatb_table(binned)
    weight = conversion_k * weights[members].sum(axis=0)
    rows = (60, 59) if np.trunc((scale * level - start) / domega) == 60 else (59, 60)
    other = truth.copy()
    other[rows[0]] -= weight
    other[rows[1]] += weight
    impossible = truth.copy()
    impossible[rows[1]] -= weight
    impossible[rows[0]] += weight
    for label, table, verdict in (
        ("pyatb", truth, PASS_DOCUMENTED),
        ("other", other, PASS_DOCUMENTED),
        ("impossible", impossible, FAIL),
    ):
        reference = reference_from(tmp_path / f"bins-{label}", binned, sigma=table)
        report = compare(
            "SHIFT_CURRENT", reference, binned, clean_control=clean_report, scfout=binned / "scf.scfout"
        )
        assert report["verdict"] == verdict, (label, render_markdown(report))


def test_pyatbs_own_value_at_an_exact_degeneracy_depends_on_its_eigensolver_in_a_nonorthogonal_basis(
    runs, tmp_path
):
    """Exact Kramers pairs in a nonorthogonal basis: PyATB zeroes its derivative inside the degenerate levels and
    takes the block's velocity eigenvalues, which is the physical value only where the parallel transport inside them
    is zero. Without on-site dipoles it is: a clean control. With them PyATB's value there depends on its
    eigensolver's basis, beyond a tenth of the tolerance on this grid: no clean control, and a difference from PyATB
    is not comparable rather than failed or documented."""
    report = judge(
        runs,
        tmp_path / "plain",
        "kramers_nonorthogonal",
        sigma=pyatb_table(runs["kramers_nonorthogonal"]),
        scfout=True,
    )
    assert report["verdict"] == PASS and report["semantics"]["rules"]["clean_control"]["eligible"], (
        render_markdown(report)
    )
    assert not report["semantics"]["rules"]["recomputed"]["exact_degeneracy_differs"]
    report = judge(
        runs, tmp_path / "dipole", "kramers_dipole", sigma=pyatb_table(runs["kramers_dipole"]), scfout=True
    )
    rules = report["semantics"]["rules"]
    assert rules["recomputed"]["exact_degeneracy_differs"] and rules["causes"]["exact_degeneracy"]["active"]
    assert not rules["clean_control"]["eligible"]
    assert report["verdict"] == NOT_COMPARABLE and DOCUMENTED not in components(report), render_markdown(
        report
    )


def test_an_inflated_rounding_floor_hides_no_value_error(runs, tmp_path, clean_report):
    """A value 1 % off beside honest records and rounding scales inflated a millionfold: the records then meet the
    value only within that floor, which rests on the candidate until the scfout recomputes it. Without the scfout no
    difference is judged on such records (never documented, even through an unbounded cause); with it the floor
    contradicts the scfout."""

    def value_and_scales(block):
        for holder in (block, block["legacy_single_channel"] or {}):
            for key in ("sigma_uA_V2", "binned_uA_V2"):
                if holder.get(key) is not None:
                    holder[key] = as_columns(1.01 * columns(holder[key]))
        for rounding in (
            block["diagnostics"]["rounding"],
            (block["diagnostics"]["spin_channels"] or {}).get("down_rounding")
            or {"kpoint_scale_uA_V2_eV": []},
        ):
            rounding["kpoint_scale_uA_V2_eV"] = (1e6 * np.asarray(rounding["kpoint_scale_uA_V2_eV"])).tolist()

    for name in ("clean", "window", "scissors", "binned"):
        copy = edited(tmp_path / name, runs[name], value_and_scales)
        reference = reference_from(tmp_path / f"{name}-ref", copy, sigma=scheme_table(runs[name]))
        report = compare("SHIFT_CURRENT", reference, copy, clean_control=clean_report)
        assert report["verdict"] != PASS and DOCUMENTED not in components(report), (
            name,
            render_markdown(report),
        )
        report = compare(
            "SHIFT_CURRENT", reference, copy, clean_control=clean_report, scfout=copy / "scf.scfout"
        )
        assert report["verdict"] == FAIL and classes(report)["rounding_recomputed"] == UNEXPLAINED, name
