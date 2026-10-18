"""The shift-current parity rule: PyATB SHIFT_CURRENT run directories against QUEER runs, by a full recomputation.

Candidates are real runner outputs on the analytic models of ``openmx_files`` (and the real calibration scfout
when it is present). PyATB is stood in for by ``pyatb_shift_port``, a loop-by-loop port of PyATB 60fc6b2a's
method 1 that shares no code with the comparator's transcription: a reference prints the port's table unless a
test gives it other numbers. Every scenario of the earlier verifications comes out as the rule says: a wrong
QUEER value fails (with or without diagnostics that agree with it), a correct one passes or is documented with
an attribution, and a PyATB table the recomputation does not reproduce is not comparable, never a pass.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from openmx_files import shift_collinear_scfout, shift_spinor_scfout
from pyatb_shift_port import pyatb_shift

from queer import OpenMXModel
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
from queer.compat.generator.parity_shift import (
    AGREEMENT_RELATIVE,
    DETERMINED_SHARE,
    NO_SCFOUT,
    NOT_REPRODUCED,
    TOO_LARGE,
    literature_anchor,
    print_precision,
    symmetry_gate,
)
from queer.compat.generator.pyatb_conventions import pyatb_shift_constant_ratio
from queer.compat.generator.pyatb_files import SHIFT_COMPONENTS, shift_current_text
from queer.compat.generator.pyatb_shift_integrand import PYATB_HARTREE, ShiftRun, recompute
from queer.compat.generator.runner import main, run
from queer.compat.generator.shift_scfout import codata_hartree, read_records

SPINOR = {"occ_band": 2, "omega": [0.3, 6.5], "domega": 0.02, "eta": 0.1, "grid": [3, 3, 2]}
COLLINEAR = {"occ_band": 1, "omega": [0.2, 4.5], "domega": 0.04, "eta": 0.1, "grid": [6, 6, 1]}
RUNS = {
    "clean": (shift_spinor_scfout, {}, 4, SPINOR),
    "binned": (shift_spinor_scfout, {}, 4, {**SPINOR, "smearing_method": 0, "domega": 0.05}),
    "collinear": (shift_collinear_scfout, {}, 1, COLLINEAR),
    "collinear_binned": (shift_collinear_scfout, {}, 1, {**COLLINEAR, "smearing_method": 0}),
    "nonmagnetic": (shift_collinear_scfout, {"spin": 0}, 1, COLLINEAR),
    "scissors": (shift_collinear_scfout, {}, 1, {**COLLINEAR, "scissors": 0.3}),
    "left": (shift_spinor_scfout, {"lattice": np.diag([3.0, 3.2, -3.5])}, 4, SPINOR),
    "pair": (shift_spinor_scfout, {}, 4, {**SPINOR, "omega": [1.5, 6.5], "n_occ": 2, "m_unocc": 3}),
    "window": (shift_spinor_scfout, {}, 4, {**SPINOR, "omega": [2.0, 4.0]}),
    "negative": (shift_spinor_scfout, {}, 4, {**SPINOR, "omega": [-0.5, 6.5]}),
    # Time reversal (no exchange): exact Kramers pairs at the 8 TRIM points of a 4 x 4 x 4 grid, in an orthogonal and
    # in a nonorthogonal basis with position moments.
    "kramers": (shift_spinor_scfout, {"exchange": 0.0}, 4, {**SPINOR, "grid": [4, 4, 4]}),
    "kramers_nonorthogonal": (
        shift_spinor_scfout,
        {"exchange": 0.0, "overlap_hopping": (0.06, 0.04, 0.03)},
        4,
        {**SPINOR, "grid": [4, 4, 4]},
    ),
    # With on-site dipoles PyATB's own value at those pairs depends on its eigensolver's basis inside them.
    "kramers_dipole": (
        shift_spinor_scfout,
        {"exchange": 0.0, "overlap_hopping": (0.06, 0.04, 0.03), "dipole": 0.6},
        4,
        {**SPINOR, "grid": [2, 2, 2]},
    ),
    # A 2e-5 eV exchange splits those pairs by 4e-5 eV: PyATB's method 1 regularizes them.
    "split": (shift_spinor_scfout, {"exchange": 2e-5}, 4, {**SPINOR, "grid": [4, 4, 4]}),
    "split_binned": (
        shift_spinor_scfout,
        {"exchange": 2e-5},
        4,
        {**SPINOR, "grid": [4, 4, 4], "smearing_method": 0, "domega": 0.05},
    ),
    "split_window": (
        shift_spinor_scfout,
        {"exchange": 2e-5},
        4,
        {**SPINOR, "grid": [4, 4, 4], "omega": [2.0, 4.0]},
    ),
    # Split by 1e-8 eV only: QUEER's decomposition of those levels' transitions is ill-conditioned.
    "split_tiny": (
        shift_spinor_scfout,
        {"exchange": 5e-9},
        4,
        {**SPINOR, "grid": [4, 4, 4], "eta": 0.01, "domega": 0.002},
    ),
    # PyATB's narrow broadening on coarse steps (its default eta 0.01 on domega 0.01 and 0.02).
    "pyatb_default": (shift_spinor_scfout, {}, 4, {**SPINOR, "eta": 0.01, "domega": 0.01}),
    "coarse_step": (shift_spinor_scfout, {}, 4, {**SPINOR, "eta": 0.01, "domega": 0.02}),
    "sharp": (shift_spinor_scfout, {}, 4, {**SPINOR, "eta": 0.002, "domega": 0.0004}),
    # A response that vanishes: no odd mixing and no spin-orbit coupling leave every transition's integrand zero.
    "centro": (shift_spinor_scfout, {"mixing_odd": (0.0, 0.0, 0.0), "soc": 0.0, "soc_z": 0.0}, 4, SPINOR),
    # Integrands at the level of rounding: a weakly broken symmetry (odd mixing 4e-7, spin-orbit 1e-4) on the TRIM
    # points of a 2 x 2 x 2 grid at PyATB's narrow broadening.
    "weak": (
        shift_spinor_scfout,
        {"mixing_odd": (4e-7, 5e-7, 3e-7), "soc": 1e-4, "soc_z": 5e-5, "exchange": 0.1},
        4,
        {**SPINOR, "grid": [2, 2, 2], "eta": 0.01, "domega": 0.002},
    ),
    # The 4e-5 eV pairs split across the occupation boundary (a transition inside a pair), and one band pair of them.
    "split_occupation": (
        shift_spinor_scfout,
        {"exchange": 2e-5},
        4,
        {**SPINOR, "grid": [4, 4, 4], "occ_band": 1},
    ),
    "split_pair": (
        shift_spinor_scfout,
        {"exchange": 2e-5},
        4,
        {**SPINOR, "grid": [4, 4, 4], "omega": [1.5, 6.5], "n_occ": 2, "m_unocc": 3},
    ),
    # Scissored transitions exactly on bin edges without smearing (QUEER's histogram must tile its rows).
    "scissors_binned": (
        shift_collinear_scfout,
        {},
        1,
        {**COLLINEAR, "scissors": 0.3, "smearing_method": 0, "domega": 0.02},
    ),
    "scissors_edge": (
        shift_collinear_scfout,
        {},
        1,
        {**COLLINEAR, "scissors": 0.1, "smearing_method": 0, "domega": 0.04},
    ),
    # Records Hermitian only to 1e-9 eV, as an SCF's quadrature leaves them: PyATB's reader Hermitizes nspin 1.
    "nonhermitian": (
        shift_collinear_scfout,
        {"asymmetry": 1e-9},
        1,
        {**COLLINEAR, "grid": [12, 12, 1], "eta": 0.01, "domega": 0.002, "omega": [0.2, 6.0]},
    ),
    "method0": (shift_collinear_scfout, {}, 1, {**COLLINEAR, "method": 0}),
    "adaptive": (shift_spinor_scfout, {}, 4, {**SPINOR, "smearing_method": 2, "grid": [2, 2, 1]}),
    "converged": (shift_spinor_scfout, {}, 4, SPINOR, True),
}
# What a correct candidate against PyATB's table comes to, and the steps a documented difference rests on.
EXPECTED = {
    "clean": (PASS, {AGREE}, None),
    "binned": (PASS, {AGREE, AFTER_CONVENTION}, None),
    "collinear": (PASS, {AGREE, AFTER_CONVENTION}, None),
    "collinear_binned": (PASS, {AGREE, AFTER_CONVENTION}, None),
    "nonmagnetic": (PASS, {AGREE}, None),
    "scissors": (PASS_DOCUMENTED, {AGREE, AFTER_CONVENTION, DOCUMENTED}, "scissors"),
    "left": (PASS, {AGREE, AFTER_CONVENTION}, None),
    "pair": (PASS, {AGREE}, None),
    "window": (PASS_DOCUMENTED, {AGREE, DOCUMENTED}, "window"),
    "negative": (PASS, {AGREE}, None),
    "kramers": (PASS, {AGREE}, None),
    "kramers_nonorthogonal": (PASS, {AGREE}, None),
    "split": (PASS_DOCUMENTED, {AGREE, DOCUMENTED}, "cut"),
    "split_binned": (PASS_DOCUMENTED, {AGREE, AFTER_CONVENTION, DOCUMENTED}, "cut"),
    "split_window": (PASS_DOCUMENTED, {AGREE, DOCUMENTED}, None),
    "pyatb_default": (PASS, {AGREE, AFTER_CONVENTION}, None),
    "coarse_step": (PASS, {AGREE, AFTER_CONVENTION}, None),
    "sharp": (PASS, {AGREE, AFTER_CONVENTION}, None),
    "centro": (PASS, {AGREE}, None),
    "converged": (PASS, {AGREE}, None),
    "weak": (PASS, {AGREE, AFTER_CONVENTION}, None),
    "split_occupation": (PASS_DOCUMENTED, {AGREE, DOCUMENTED}, None),
    "split_pair": (PASS_DOCUMENTED, {AGREE, DOCUMENTED}, "cut"),
    "scissors_binned": (PASS_DOCUMENTED, {AFTER_CONVENTION, DOCUMENTED}, "scissors"),
    "scissors_edge": (PASS_DOCUMENTED, {AFTER_CONVENTION, DOCUMENTED}, "scissors"),
    "nonhermitian": (PASS, {AGREE, AFTER_CONVENTION}, None),
}
CALIBRATION_SCFOUT = Path(
    os.environ.get("QUEER_PYATB_CALIBRATION_SCFOUT")
    or Path.home() / ".cache" / "queer-parity-data" / "pyatb-calib-60fc6b2a" / "test.scfout"
)
REAL = {**SPINOR, "occ_band": 3, "omega": [0.0, 12.0], "grid": [12, 12, 1]}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=1))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def columns(block):
    return np.column_stack([block[name] for name in SHIFT_COMPONENTS])


def as_columns(values):
    return {name: np.asarray(values)[:, index].tolist() for index, name in enumerate(SHIFT_COMPONENTS)}


def candidate(directory, scfout, model, nspin, block, self_convergence=None):
    directory.mkdir(parents=True)
    if callable(scfout):
        scfout(directory / "scf.scfout", **model)
    else:
        shutil.copy(scfout, directory / "scf.scfout")
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


def port_table(run_dir):
    """What PyATB prints for a run (the port), unprinted."""
    params = read(run_dir / "queer_params.json")
    block = {**SHIFT_DEFAULTS, **params["parameters"]}
    return pyatb_shift(run_dir / "scf.scfout", block, nspin=params["input_parameters"]["nspin"])[1]


@pytest.fixture(scope="module")
def printed(runs):
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = port_table(runs[name])
        return cache[name]

    return get


def block_text(name, values):
    lines = [name, "{"]
    for key, value in values.items():
        text = " ".join(str(v) for v in value) if isinstance(value, list) else str(value)
        lines.append(f"    {key:<32}{text}")
    return "\n".join(lines + ["}", ""])


def reference_from(directory, run_dir, sigma, *, echo=None, omega=None, digests=True):
    """A PyATB-layout run directory printing ``sigma`` for the run's settings."""
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
    (out / "Shift_Current" / "shift_current.dat").write_text(
        shift_current_text(shift["omega_eV"] if omega is None else omega, np.asarray(sigma))
    )
    if digests:
        (directory / "inputs.sha256").write_text(
            f"{sha256(run_dir / 'scf.scfout')}  scf.scfout\n{sha256(directory / 'Input')}  Input\n"
        )
    return directory


def judge(run_dir, reference, *, scfout=True, **options):
    return compare(
        "QUEER_SHIFT_CURRENT",
        reference,
        run_dir,
        scfout=run_dir / "scf.scfout" if scfout else None,
        workers=1,
        **options,
    )


def judge_run(runs, printed, tmp_path, name, sigma=None, **options):
    """``name``'s run against a reference printing PyATB's table (or ``sigma``)."""
    table = printed(name) if sigma is None else sigma
    return judge(runs[name], reference_from(tmp_path / f"reference-{name}", runs[name], table), **options)


def classes(report):
    return {item["name"]: item["class"] for item in report["quantities"]}


def components(report):
    return {item["class"] for item in report["quantities"] if item["name"].startswith("sigma_")}


def failed_or_passed(report, name):
    """A precondition's detail."""
    return next(item["detail"] for item in report["preconditions"] if item["name"] == name)


def failed_preconditions(report):
    return {item["name"]: item["detail"] for item in report["preconditions"] if not item["ok"]}


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


def transformed(key, transform):
    def edit(block):
        block[key] = as_columns(transform(columns(block[key]).copy()))

    return edit


def test_every_candidate_ran(runs):
    for name, directory in runs.items():
        document = read(directory / "queer_results.json")
        assert document["status"] == "ok", (name, document.get("error"))


def test_the_port_is_pyatbs_loop_order():
    """The port's own pieces: PyATB's print precision, constants and rows."""
    assert print_precision([1.0, 9.9999, -2.5e-12, 0.0]).tolist() == pytest.approx([5e-7, 5e-7, 5e-19, 0.0])
    from pyatb_shift_port import AU2EV, E_CHARGE, EV_SECONDS, HBAR

    assert AU2EV == 27.211324570273 and (E_CHARGE, EV_SECONDS, HBAR) == (
        1.602176565e-19,
        6.582119e-16,
        1.054571726e-34,
    )
    ratio = (
        (E_CHARGE**2 / HBAR)
        * (E_CHARGE * EV_SECONDS / HBAR)
        / (1.602176634e-19**2 / (6.62607015e-34 / (2 * np.pi)))
    )
    assert ratio == pytest.approx(pyatb_shift_constant_ratio(), rel=1e-15)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_a_correct_candidate_passes_or_is_documented_by_the_recomputation(runs, printed, tmp_path, name):
    verdict, kinds, cause = EXPECTED[name]
    report = judge_run(runs, printed, tmp_path, name)
    assert report["verdict"] == verdict, render_markdown(report)
    found = components(report)
    assert found <= kinds and found, (name, found)
    rules = report["semantics"]["rules"]
    assert rules["pyatb_reproduced"]["reproduced"] and rules["determined"]
    checked = [item for item in report["quantities"] if item["name"].startswith("physical.")]
    assert checked and all(item["class"] == AGREE for item in checked)
    if cause is not None:
        documented = [values for values in rules["attribution"].values() if values["class"] == DOCUMENTED]
        assert documented and all(
            max(values["steps_uA_V2"], key=values["steps_uA_V2"].get) == cause for values in documented
        ), documented
    # Every documented gap is the recomputation's own path from QUEER's value to PyATB's table.
    for values in rules["attribution"].values():
        total = sum(values["steps_uA_V2"].values())
        assert values["gap_uA_V2"] <= total + values["residual_uA_V2"] + 1e-9 * max(1.0, values["gap_uA_V2"])


@pytest.mark.parametrize(
    "transform",
    [
        ("negated", lambda t: -t),
        ("doubled", lambda t: 2 * t),
        ("zeroed", lambda t: 0 * t),
        ("scaled 1.001", lambda t: 1.001 * t),
        ("scaled 1+1e-6", lambda t: (1 + 1e-6) * t),
        ("xxy and yxx swapped", lambda t: t[:, [0, 6, 2, 3, 4, 5, 1, *range(7, 18)]]),
        ("rolled 3 rows", lambda t: np.roll(t, 3, axis=0)),
        ("xyz zeroed", lambda t: np.where(np.arange(18) == 4, 0.0, t)),
    ],
    ids=lambda item: item[0],
)
@pytest.mark.parametrize(
    "name", ["clean", "kramers", "split", "left", "pyatb_default", "weak", "split_occupation"]
)
def test_a_wrong_physical_value_fails_whatever_pyatb_printed(runs, printed, tmp_path, name, transform):
    """QUEER's value changed, its file consistent with it: check A fails, whether PyATB printed its own table or the
    wrong value itself (correlated diagnostics or a PyATB that agrees with the wrong value change nothing)."""
    label, change = transform
    copy = edited(tmp_path / "copy", runs[name], transformed("sigma_uA_V2", change))
    honest = judge(copy, reference_from(tmp_path / "honest", copy, printed(name)))
    assert honest["verdict"] == FAIL and classes(honest)["physical.sigma_uA_V2"] == UNEXPLAINED, label
    sign = read(copy / "queer_results.json")["shift_current"]["convention"]["export_sign"]
    wrong = sign * columns(read(copy / "queer_results.json")["shift_current"]["sigma_uA_V2"])
    agreeing = judge(copy, reference_from(tmp_path / "agreeing", copy, wrong))
    assert agreeing["verdict"] == FAIL and classes(agreeing)["physical.sigma_uA_V2"] == UNEXPLAINED, label


def test_forged_diagnostics_change_nothing(runs, printed, tmp_path):
    """The comparison reads no record of the candidate's but its tables."""

    def forge(block):
        diagnostics = block["diagnostics"]
        diagnostics["min_pair_gap_eV"] = 1e-12
        diagnostics["kpoints_at_pyatb_threshold"] = 10**6
        diagnostics["exact_degenerate_kpoints"] = 10**6
        diagnostics["regularisation"] = {"shift_uA_V2": as_columns(np.ones((len(block["omega_eV"]), 18)))}
        diagnostics["rounding"] = {"kpoint_scale_uA_V2_eV": [1e30] * 100}

    for name in ("clean", "split"):
        copy = edited(tmp_path / name, runs[name], forge)
        report = judge(copy, reference_from(tmp_path / f"ref-{name}", copy, printed(name)))
        assert report["verdict"] == EXPECTED[name][0]
        forged = edited(
            tmp_path / f"{name}-wrong",
            runs[name],
            lambda block: (forge(block), transformed("sigma_uA_V2", lambda t: 1.01 * t)(block)),
        )
        report = judge(forged, reference_from(tmp_path / f"ref-{name}-wrong", forged, printed(name)))
        assert report["verdict"] == FAIL


def test_queer_reporting_pyatbs_scheme_as_its_physical_value_fails(runs, printed, tmp_path):
    """A QUEER bug that applies PyATB's window, Hartree constant or truncation to its own physical value: the
    recomputed physical value has none of them."""
    for name in ("window", "pyatb_default", "split"):
        sign = read(runs[name] / "queer_results.json")["shift_current"]["convention"]["export_sign"]
        scheme = sign * printed(name) / pyatb_shift_constant_ratio()

        def mimic(block):
            block["sigma_uA_V2"] = as_columns(scheme)

        copy = edited(tmp_path / name, runs[name], mimic)
        report = judge(copy, reference_from(tmp_path / f"ref-{name}", copy, printed(name)))
        assert report["verdict"] == FAIL and classes(report)["physical.sigma_uA_V2"] == UNEXPLAINED, name


def test_without_smearing_both_tables_are_judged(runs, printed, tmp_path):
    """Without smearing QUEER's per-eV histogram and its bins (PyATB's normalization) are both its physical
    value: an error in either fails."""
    for key in ("sigma_uA_V2", "binned_uA_V2"):
        for change in (lambda t: 2 * t, lambda t: -t, lambda t: 0 * t):
            copy = edited(tmp_path / f"{key}-{id(change)}", runs["binned"], transformed(key, change))
            report = judge(
                copy, reference_from(tmp_path / f"ref-{key}-{id(change)}", copy, printed("binned"))
            )
            assert report["verdict"] == FAIL and classes(report)[f"physical.{key}"] == UNEXPLAINED

    # The collinear channels' bins too.
    def legacy(block):
        block["legacy_single_channel"]["binned_uA_V2"] = as_columns(
            1.01 * columns(block["legacy_single_channel"]["binned_uA_V2"])
        )

    copy = edited(tmp_path / "legacy-binned", runs["collinear_binned"], legacy)
    report = judge(copy, reference_from(tmp_path / "ref-legacy-binned", copy, printed("collinear_binned")))
    assert report["verdict"] == FAIL
    assert classes(report)["physical.legacy_single_channel.binned_uA_V2"] == UNEXPLAINED


def test_a_collinear_spin_down_bug_fails(runs, printed, tmp_path):
    """PyATB reads only the spin-up channel: QUEER's physical value adds the spin-down one, which the recomputation
    evaluates too. A wrong spin-down channel, a wrong legacy channel or a sum of the wrong channels fails."""
    block = read(runs["collinear"] / "queer_results.json")["shift_current"]
    up = columns(block["legacy_single_channel"]["sigma_uA_V2"]) / 2
    down = columns(block["sigma_uA_V2"]) - up
    for label, physical, legacy_value in (
        ("down x3", up + 3 * down, 2 * up),
        ("up twice", 2 * up, 2 * up),
        ("legacy of the down channel", up + down, 2 * down),
    ):

        def edit(target, physical=physical, legacy_value=legacy_value):
            target["sigma_uA_V2"] = as_columns(physical)
            target["legacy_single_channel"]["sigma_uA_V2"] = as_columns(legacy_value)

        copy = edited(tmp_path / label.replace(" ", "-"), runs["collinear"], edit)
        report = judge(
            copy, reference_from(tmp_path / f"ref-{label.replace(' ', '-')}", copy, printed("collinear"))
        )
        assert report["verdict"] == FAIL, label


@pytest.mark.parametrize("defect", ["hartree", "reconciliation", "spin-down"])
def test_a_defect_in_queers_scfout_reader_fails(runs, printed, tmp_path, monkeypatch, defect):
    """QUEER's runner reads the scfout through QUEER's reader; the recomputation parses it itself and PyATB (the port)
    with a reader of its own. A wrong Hartree constant (27.0 eV), a reconciliation whose Hamiltonian is 1 % off, or a
    collinear spin-down channel parsed from the spin-up records moves QUEER's value and nothing else: check A fails.
    (While the recomputation built its operators from QUEER's reader, these passed both checks.)"""
    import queer.io.openmx as reader
    import queer.models.openmx as models

    names = ["clean", "collinear"]
    if defect == "hartree":
        monkeypatch.setattr(reader, "HARTREE_EV", 27.0)
    elif defect == "reconciliation":
        original = models._canonical_native_operators

        def scaled(data):
            h, s, r, errors = original(data)
            return {key: 1.01 * value for key, value in h.items()}, s, r, errors

        monkeypatch.setattr(models, "_canonical_native_operators", scaled)
    else:
        original_spin = reader._spin_hamiltonian

        def spin_up_twice(real, imaginary, mode):
            if mode == "collinear":
                real = (real[0], real[0], *real[2:])
            return original_spin(real, imaginary, mode)

        monkeypatch.setattr(reader, "_spin_hamiltonian", spin_up_twice)
        names = ["collinear"]
    bugged = {name: candidate(tmp_path / name, *RUNS[name]) for name in names}
    monkeypatch.undo()
    for name, directory in bugged.items():
        report = judge(directory, reference_from(tmp_path / f"ref-{name}", directory, printed(name)))
        assert report["verdict"] == FAIL and classes(report)["physical.sigma_uA_V2"] == UNEXPLAINED, name


def test_an_entry_the_scfout_does_not_determine_is_never_a_failure(runs, printed, tmp_path):
    """What the scfout determines is judged before check A: QUEER's value beyond its rounding at entries whose
    recomputation is not determined to the agreement tolerance (Kramers pairs split by 1e-8 eV, a transition inside a
    pair) makes the comparison not comparable; the same excess at a determined entry fails."""
    from queer.compat.generator.pyatb_shift_integrand import ShiftRun, recompute
    from queer.compat.generator.shift_scfout import read_records

    directory = runs["split_tiny"]
    params = read(directory / "queer_params.json")
    run_settings = ShiftRun.from_block({**SHIFT_DEFAULTS, **params["parameters"]})
    table = recompute(read_records(directory / "scf.scfout"), run_settings, workers=1).physical["sigma"]
    peak = np.max(np.abs(table.value))
    undetermined = table.smooth > DETERMINED_SHARE * AGREEMENT_RELATIVE * peak
    assert undetermined.any() and not undetermined.all()
    row, column = np.unravel_index(
        int(np.argmax(np.where(undetermined, 0.0, np.abs(table.value)))), table.value.shape
    )
    determined = np.zeros_like(undetermined)
    determined[row, column] = True
    for label, where, verdict in (
        ("undetermined", undetermined, NOT_COMPARABLE),
        ("determined", determined, FAIL),
    ):
        copy = edited(
            tmp_path / label,
            directory,
            transformed("sigma_uA_V2", lambda t, w=where: t + np.where(w, 3 * table.rounding + 1e-30, 0.0)),
        )
        report = judge(copy, reference_from(tmp_path / f"ref-{label}", copy, printed("split_tiny")))
        assert report["verdict"] == verdict, (label, render_markdown(report))
        if verdict == NOT_COMPARABLE:
            assert "does not determine" in failed_preconditions(report)["determined"]


def test_a_degeneracy_tolerance_bug_fails(tmp_path, monkeypatch):
    """QUEER's degeneracy tolerance too large (5e-5 eV for Kramers pairs split by 4e-5 eV): it sums them as exact
    degeneracies. The recomputation measures the scfout's own tolerance and treats them as the levels they are."""
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
    assert (
        read(bugged / "queer_results.json")["shift_current"]["diagnostics"]["exact_degenerate_kpoints"] == 8
    )
    report = judge(bugged, reference_from(tmp_path / "reference", bugged, port_table(bugged)))
    assert report["verdict"] == FAIL and classes(report)["physical.sigma_uA_V2"] == UNEXPLAINED


def test_a_transition_on_a_bin_edge_may_change_rows_but_not_vanish_or_count_twice(runs, printed, tmp_path):
    """The collinear model has a transition at exactly 1.64 eV at two k-points, the edge between the bins (1.60, 1.64]
    and (1.64, 1.68] (rows 35 and 36), which hold nothing else: either placement is QUEER's binned value, so rows 35
    and 36 exchanged still pass; the transition dropped or counted in both rows fails, although every entry stays
    within the placement's allowance, because the rows' sum is conserved."""
    run = runs["collinear_binned"]
    report = judge_run(runs, printed, tmp_path / "correct", "collinear_binned")
    facts = report["semantics"]["rules"]["determined"]["tables"]["physical.binned_uA_V2"]
    assert report["verdict"] == PASS and facts["edge_uA_V2"] > 0.1 * facts["largest_uA_V2"]
    lower, upper = 35, 36
    omega = np.asarray(read(run / "queer_results.json")["shift_current"]["omega_eV"])
    assert omega[upper] == pytest.approx(1.64, abs=1e-12)

    def rows(change):
        def edit(block):
            for owner in (block, block["legacy_single_channel"]):
                table = columns(owner["binned_uA_V2"])
                pair = table[[lower, upper]]
                table[[lower, upper]] = change(pair)
                owner["binned_uA_V2"] = as_columns(table)

        return edit

    for label, change, verdict in (
        ("exchanged", lambda pair: pair[::-1], PASS),
        ("dropped", lambda pair: 0 * pair, FAIL),
        ("counted twice", lambda pair: np.repeat(pair.sum(axis=0, keepdims=True), 2, axis=0), FAIL),
    ):
        copy = edited(tmp_path / label.replace(" ", "-"), run, rows(change))
        report = judge(
            copy,
            reference_from(tmp_path / f"ref-{label.replace(' ', '-')}", copy, printed("collinear_binned")),
        )
        assert report["verdict"] == verdict, (label, render_markdown(report))
        if verdict == FAIL:
            item = next(q for q in report["quantities"] if q["name"] == "physical.binned_uA_V2")
            assert item["class"] == UNEXPLAINED and "sum over the photon energies" in item["detail"], item


def outer_edge_run(directory, omega):
    """The collinear model without smearing on ``omega``, with its recomputation."""
    block = {**COLLINEAR, "smearing_method": 0, "omega": omega}
    run_dir = candidate(directory, shift_collinear_scfout, {}, 1, block)
    full = {**SHIFT_DEFAULTS, **block}
    return run_dir, recompute(read_records(run_dir / "scf.scfout"), ShiftRun.from_block(full), workers=1)


def moved(table, row, chosen):
    """A table's row with the groups ``chosen`` of its outer edges on the other side."""
    return table.value[row] + sum((table.placements[index].change[row] for index in chosen), np.zeros(18))


def inside(table, row):
    """A table's row with every group of its outer edges inside the table."""
    return moved(table, row, [index for index, item in enumerate(table.placements) if not item.inside])


@pytest.mark.parametrize(
    ("label", "omega", "key", "row"),
    [
        ("binned-first", [1.64, 3.0], "binned_uA_V2", 0),
        ("per-eV-first", [1.66, 3.0], "sigma_uA_V2", 0),
        ("binned-last", [0.2, 1.605], "binned_uA_V2", -1),
    ],
)
def test_a_transition_on_an_outer_edge_lies_inside_the_table_or_outside_it(tmp_path, label, omega, key, row):
    """The collinear model's transitions at exactly 1.64 eV (two k-points of equal weight; nothing else shares their
    row) on the first or the last edge of QUEER's histogram: each k-point's transitions inside the table or outside it
    pass, in every combination (half their weight is one k-point's inside); the edge row holding twice, minus, a
    quarter or one and a half times their weight fails. The earlier allowance passed those within their weight, since
    on an outer edge it moved the sum of every row beyond."""
    run_dir, recomputation = outer_edge_run(tmp_path / label, omega)
    reference = reference_from(tmp_path / f"reference-{label}", run_dir, port_table(run_dir))
    name = "binned" if key == "binned_uA_V2" else "sigma"
    tables = {
        key: recomputation.physical[name],
        f"legacy_single_channel.{key}": recomputation.physical[
            "legacy_binned" if name == "binned" else "legacy"
        ],
    }
    assert all(table.groups == 2 and table.placements_exact for table in tables.values())

    def at_edge(target):
        def edit(block):
            for owner, table in tables.items():
                holder = block["legacy_single_channel"] if owner.startswith("legacy") else block
                values = columns(holder[key])
                values[row] = target(table)
                holder[key] = as_columns(values)

        return edit

    cases = [
        (f"placed {chosen}", lambda table, chosen=chosen: moved(table, row, chosen), True)
        for chosen in ([], [0], [1], [0, 1])
    ]
    cases += [
        (f"{factor} x inside", lambda table, factor=factor: factor * inside(table, row), False)
        for factor in (2.0, -1.0, 0.25, 1.5)
    ]
    assert judge(run_dir, reference)["verdict"] != FAIL
    for case, target, honest in cases:
        copy = edited(tmp_path / f"{label}-{case}".replace(" ", "_"), run_dir, at_edge(target))
        report = judge(copy, reference)
        quantities = {item["name"]: item for item in report["quantities"]}
        judged = [quantities[f"physical.{owner}"] for owner in tables]
        if honest:
            assert report["verdict"] != FAIL and all(item["class"] == AGREE for item in judged), (
                case,
                judged,
            )
        else:
            assert report["verdict"] == FAIL and all(item["class"] == UNEXPLAINED for item in judged), case
            assert "outer edges tested inside and outside the table" in judged[0]["detail"]


def test_outer_edge_placements_not_tested_exactly_are_not_determined(tmp_path, monkeypatch):
    """Beyond the groups whose placements are tested exactly (here a limit of one against the two k-points on the
    binned table's first edge) those transitions keep their allowance and their entries are not determined: honest or
    with row 0 holding twice their absolute weight, the comparison is not comparable, never a pass."""
    from queer.compat.generator import pyatb_shift_integrand

    monkeypatch.setattr(pyatb_shift_integrand, "OUTER_PLACEMENTS", 1)
    run_dir, recomputation = outer_edge_run(tmp_path / "run", [1.64, 3.0])
    table = recomputation.physical["binned"]
    assert table.groups == 2 and not table.placements_exact and not table.placements
    reference = reference_from(tmp_path / "reference", run_dir, port_table(run_dir))

    def doubled(block):
        for holder, name in ((block, "binned"), (block["legacy_single_channel"], "legacy_binned")):
            values = columns(holder["binned_uA_V2"])
            values[0] = 2 * (
                recomputation.physical[name].value[0] + recomputation.physical[name].outer_edge[0]
            )
            holder["binned_uA_V2"] = as_columns(values)

    for copy in (run_dir, edited(tmp_path / "doubled", run_dir, doubled)):
        report = judge(copy, reference)
        assert report["verdict"] == NOT_COMPARABLE, render_markdown(report)
        assert "not placed exactly" in failed_preconditions(report)["determined"]


def test_a_transition_on_pyatbs_window_edge_lies_inside_it_or_outside_it(tmp_path):
    """The collinear model's 1.64 eV transitions on PyATB's energy scale (2.3e-6 of their energy lower) at exactly the
    first photon energy without smearing: PyATB's window holds them or not, and each k-point's choice reproduces its
    table; row 0 holding twice or a quarter of their weight is not reproduced (not comparable, never a pass)."""
    run_dir, recomputation = outer_edge_run(tmp_path / "run", [1.64 * PYATB_HARTREE / codata_hartree(), 3.0])
    literal = recomputation.literal
    assert literal.groups == 2 and literal.placements_exact
    weight = inside(literal, 0)
    assert np.max(np.abs(weight)) > 0.1 * np.max(np.abs(literal.value))
    own = port_table(run_dir)
    for case, row, verdict in (
        ("own", None, None),
        ("placed [0]", moved(literal, 0, [0]), None),
        ("placed [0, 1]", moved(literal, 0, [0, 1]), None),
        ("twice", 2 * weight, NOT_COMPARABLE),
        ("a quarter", 0.25 * weight, NOT_COMPARABLE),
    ):
        table = own.copy()
        if row is not None:
            table[0] = row
        report = judge(run_dir, reference_from(tmp_path / case.replace(" ", "_"), run_dir, table))
        reproduced = report["semantics"]["rules"]["pyatb_reproduced"]["reproduced"]
        if verdict is None:
            assert report["verdict"] in {PASS, PASS_DOCUMENTED} and reproduced, (
                case,
                render_markdown(report),
            )
        else:
            assert report["verdict"] == verdict and not reproduced, case
            assert NOT_REPRODUCED in failed_preconditions(report)["pyatb_reproduced"]


def test_a_vanishing_table_is_reported_as_vanishing(runs, printed, tmp_path):
    """A response that vanishes passes as zero, and the report says so wherever it judges it: the determination, check
    A's quantity and every component's class name the table vanishing, so the pass is not read as a statement about
    its values below the rounding of what cancels in it."""
    report = judge_run(runs, printed, tmp_path, "centro")
    assert report["verdict"] == PASS, render_markdown(report)
    determined = report["semantics"]["rules"]["determined"]
    assert determined["vanishing"] and set(determined["vanishing_tables"]) == {
        "physical.sigma_uA_V2",
        "literal",
    }
    detail = failed_or_passed(report, "determined")
    assert "every table vanishes" in detail and "physical.sigma_uA_V2 vanishing (largest" in detail
    quantities = {item["name"]: item["detail"] for item in report["quantities"]}
    assert quantities["physical.sigma_uA_V2"].startswith("vanishing (largest ")
    assert all("both vanish" in quantities[f"sigma_{name}"] for name in SHIFT_COMPONENTS)
    assert "vanishing" in render_markdown(report)


def test_a_queer_error_at_the_near_degenerate_kpoints_alone_fails(runs, printed, tmp_path, monkeypatch):
    """QUEER's estimator wrong only at the 8 TRIM points of the split model, where Kramers pairs 4e-5 eV apart make
    PyATB's method 1 regularize its denominators: every record of the run follows the error, and the recomputation
    still sees it."""
    from queer.compat.generator import openmx_shift

    original = openmx_shift.shift_current_point

    def wrong(model, k, settings):
        point = original(model, k, settings)
        if np.all(np.isclose((2 * np.asarray(k)) % 1, 0)):
            point = {**point, "integrand": 1.5 * point["integrand"]}
        return point

    monkeypatch.setattr(openmx_shift, "shift_current_point", wrong)
    for name in ("split", "split_binned"):
        bugged = candidate(tmp_path / name, *RUNS[name])
        report = judge(bugged, reference_from(tmp_path / f"ref-{name}", bugged, printed(name)))
        assert report["verdict"] == FAIL and classes(report)["physical.sigma_uA_V2"] == UNEXPLAINED, name


def test_the_lattice_handedness_and_the_candidate_file(runs, printed, tmp_path):
    """PyATB's signed volume flips every component on a left-handed lattice (agree_after_convention); QUEER's file
    must hold the handedness times its value, else the candidate contradicts itself."""
    report = judge_run(runs, printed, tmp_path, "left")
    assert report["verdict"] == PASS and components(report) == {AFTER_CONVENTION}
    copy = shutil.copytree(runs["left"], tmp_path / "unsigned")
    block = read(copy / "queer_results.json")["shift_current"]
    (copy / "Out/Shift_Current/shift_current.dat").write_text(
        shift_current_text(block["omega_eV"], columns(block["sigma_uA_V2"]))
    )
    report = judge(copy, reference_from(tmp_path / "ref-unsigned", copy, printed("left")))
    assert report["verdict"] == FAIL and classes(report)["candidate_file"] == UNEXPLAINED


@pytest.mark.parametrize(
    "change",
    [
        ("doubled", lambda t: 2 * t),
        ("halved", lambda t: 0.5 * t),
        ("negated", lambda t: -t),
        ("scaled 1.001", lambda t: 1.001 * t),
        ("noise 1e-5", lambda t: t + 1e-5 * np.max(np.abs(t)) * np.sin(np.arange(t.size)).reshape(t.shape)),
        ("rolled", lambda t: np.roll(t, 1, axis=0)),
        ("xyz zeroed", lambda t: np.where(np.arange(18) == 4, 0.0, t)),
    ],
    ids=lambda item: item[0],
)
@pytest.mark.parametrize("name", ["clean", "kramers", "window", "binned"])
def test_a_pyatb_table_the_recomputation_does_not_reproduce_is_never_a_pass(
    runs, printed, tmp_path, name, change
):
    label, transform = change
    report = judge_run(runs, printed, tmp_path, name, sigma=transform(printed(name)))
    assert report["verdict"] == NOT_COMPARABLE, (label, render_markdown(report))
    assert failed_preconditions(report)["pyatb_reproduced"].startswith(NOT_REPRODUCED)


def test_pyatbs_print_precision_is_allowed_and_no_more(runs, printed, tmp_path):
    """PyATB's %15.6e moves each value by at most half a unit in its seventh significant digit: its printed table
    meets the recomputation, one entry off by two units in that digit does not."""
    table = printed("clean")
    report = judge_run(runs, printed, tmp_path / "within", "clean", sigma=table)
    assert report["verdict"] == PASS
    reproduced = report["semantics"]["rules"]["pyatb_reproduced"]
    assert 0 < reproduced["largest_gap_uA_V2"] <= np.max(print_precision(table))
    row, column = np.unravel_index(int(np.argmax(np.abs(table))), table.shape)
    off = table.copy()
    off[row, column] += 4 * print_precision(table[row, column])
    report = judge_run(runs, printed, tmp_path / "beyond", "clean", sigma=off)
    assert report["verdict"] == NOT_COMPARABLE
    assert failed_preconditions(report)["pyatb_reproduced"].startswith(NOT_REPRODUCED)


def test_the_scfout_is_required_and_must_be_both_runs(runs, printed, tmp_path):
    reference = reference_from(tmp_path / "reference", runs["clean"], printed("clean"))
    report = judge(runs["clean"], reference, scfout=False)
    assert report["verdict"] == NOT_COMPARABLE and failed_preconditions(report)["inputs"] == NO_SCFOUT
    other = tmp_path / "other.scfout"
    shift_spinor_scfout(other, exchange=0.1)
    report = compare("SHIFT_CURRENT", reference, runs["clean"], scfout=other, workers=1)
    assert report["verdict"] == NOT_COMPARABLE and "the scfout" in failed_preconditions(report)["inputs"]
    bare = reference_from(tmp_path / "bare", runs["clean"], printed("clean"), digests=False)
    report = judge(runs["clean"], bare)
    assert report["verdict"] == NOT_COMPARABLE and "inputs.sha256" in failed_preconditions(report)["inputs"]
    # The candidate's provenance must name the same scfout.
    copy = shutil.copytree(runs["clean"], tmp_path / "provenance")
    document = read(copy / "queer_results.json")
    document["provenance"]["inputs"]["scf.scfout"]["sha256"] = "0" * 64
    document["model"]["scfout_sha256"] = "0" * 64
    write(copy / "queer_results.json", document)
    report = judge(copy, reference)
    assert report["verdict"] == NOT_COMPARABLE and "provenance" in failed_preconditions(report)["inputs"]


@pytest.mark.parametrize(
    ("echo", "failed"),
    [({"eta": 0.12}, "parameters"), ({"occ_band": 1}, "parameters"), ({"omega": [0.4, 6.6]}, "parameters")],
)
def test_other_settings_are_not_comparable(runs, printed, tmp_path, echo, failed):
    reference = reference_from(tmp_path / "reference", runs["clean"], printed("clean"), echo=echo)
    report = judge(runs["clean"], reference)
    assert report["verdict"] == NOT_COMPARABLE and failed in failed_preconditions(report)


def test_photon_energies_pyatb_and_queer(runs, printed, tmp_path):
    omega = np.asarray(read(runs["clean"] / "queer_results.json")["shift_current"]["omega_eV"])
    reference = reference_from(
        tmp_path / "printed", runs["clean"], printed("clean"), omega=(omega + 0.02).tolist()
    )
    report = judge(runs["clean"], reference)
    assert report["verdict"] == NOT_COMPARABLE and "omega" in failed_preconditions(report)

    def shifted(block):
        block["omega_eV"] = (np.asarray(block["omega_eV"]) + 0.02).tolist()

    copy = edited(tmp_path / "shifted", runs["clean"], shifted)
    report = judge(
        copy, reference_from(tmp_path / "ref-shifted", copy, printed("clean"), omega=omega.tolist())
    )
    assert report["verdict"] == FAIL and classes(report)["physical.omega_eV"] == UNEXPLAINED


def test_method_0_and_the_adaptive_smearing_are_not_recomputed(runs, tmp_path):
    for name, failed in (("method0", "method"), ("adaptive", "smearing")):
        block = read(runs[name] / "queer_results.json")["shift_current"]
        report = judge(runs[name], reference_from(tmp_path / name, runs[name], columns(block["sigma_uA_V2"])))
        assert report["verdict"] == NOT_COMPARABLE and failed in failed_preconditions(report)


def test_a_comparison_beyond_the_budget_is_not_comparable(runs, printed, tmp_path):
    reference = reference_from(tmp_path / "reference", runs["clean"], printed("clean"))
    report = judge(runs["clean"], reference, recompute_budget=100)
    detail = failed_preconditions(report)["recompute_budget"]
    assert report["verdict"] == NOT_COMPARABLE and detail.startswith(TOO_LARGE)
    assert judge(runs["clean"], reference, recompute_budget=1e6)["verdict"] == PASS
    with pytest.raises(ValueError, match="SHIFT_CURRENT comparison only"):
        compare("AHC", tmp_path, tmp_path, recompute_budget=1)
    arguments = ["compare", "--reference", str(reference), "--candidate", str(runs["clean"])]
    arguments += ["--scfout", str(runs["clean"] / "scf.scfout"), "--workers", "1"]
    assert main(arguments) == 0
    assert main([*arguments, "--recompute-budget", "100"]) == 2
    assert main(arguments[:4]) == 2


def test_pyatbs_basis_dependent_value_is_not_determined(runs, printed, tmp_path):
    """Exact Kramers pairs with on-site dipoles: PyATB's own value there changes with its eigensolver's basis inside
    them, which the recomputation's two bases measure. A correct candidate is not comparable; a wrong one fails."""
    report = judge_run(runs, printed, tmp_path, "kramers_dipole")
    assert report["verdict"] == NOT_COMPARABLE and "eigensolver" in failed_preconditions(report)["determined"]
    copy = edited(tmp_path / "wrong", runs["kramers_dipole"], transformed("sigma_uA_V2", lambda t: 1.01 * t))
    report = judge(copy, reference_from(tmp_path / "ref-wrong", copy, printed("kramers_dipole")))
    assert report["verdict"] == FAIL


def test_an_ill_conditioned_near_degeneracy_is_not_determined_but_an_error_elsewhere_fails(
    runs, printed, tmp_path
):
    """Kramers pairs split by 1e-8 eV: QUEER's value at their transitions is determined only to its conditioning.
    The correct candidate is not comparable; an error at the other photon energies fails, a PyATB of another table
    is not reproduced."""
    report = judge_run(runs, printed, tmp_path, "split_tiny")
    assert (
        report["verdict"] == NOT_COMPARABLE
        and "ill-conditioned" in failed_preconditions(report)["determined"]
    )
    for factor in (1.001, 2.0, -1.0, 0.0):
        copy = edited(
            tmp_path / f"x{factor}", runs["split_tiny"], transformed("sigma_uA_V2", lambda t, f=factor: f * t)
        )
        report = judge(copy, reference_from(tmp_path / f"ref-x{factor}", copy, printed("split_tiny")))
        assert report["verdict"] == FAIL, factor


def test_kramers_pairs_at_time_reversal_invariant_points(runs, printed, tmp_path):
    """Exact Kramers pairs at the TRIM points: QUEER's covariant sum meets the recomputation, PyATB's own treatment of
    them meets its transcription; PyATB's table scaled is not reproduced, QUEER's scaled fails."""
    for factor in (2.0, 0.5, 0.125, -1.0):
        report = judge_run(
            runs, printed, tmp_path / f"pyatb{factor}", "kramers", sigma=factor * printed("kramers")
        )
        assert report["verdict"] == NOT_COMPARABLE, factor
        copy = edited(
            tmp_path / f"queer{factor}",
            runs["kramers"],
            transformed("sigma_uA_V2", lambda t, f=factor: f * t),
        )
        report = judge(copy, reference_from(tmp_path / f"ref{factor}", copy, printed("kramers")))
        assert report["verdict"] == FAIL, factor


def test_failures_and_queers_limits(runs, tmp_path):
    refused = tmp_path / "refused"
    candidate(refused, shift_spinor_scfout, {}, 4, {**SPINOR, "scissors": -0.2})
    document = read(refused / "queer_results.json")
    assert document["status"] == "failed" and document["error"]["code"] == "queer_bad_params"
    count = len(read(runs["clean"] / "queer_results.json")["shift_current"]["omega_eV"])
    reference = reference_from(
        tmp_path / "ok-reference", runs["clean"], np.zeros((count, 18)), echo={"scissors": -0.2}
    )
    report = judge(refused, reference)
    assert report["verdict"] == NOT_COMPARABLE and "documented limit" in json.dumps(report["preconditions"])
    # Both runs refusing the same nspin conflict agree.
    conflict = tmp_path / "conflict"
    candidate(conflict, shift_spinor_scfout, {}, 1, SPINOR)
    assert read(conflict / "queer_results.json")["error"]["code"] == "queer_bad_params"
    failed = tmp_path / "failed-reference"
    reference_from(failed, runs["clean"], np.zeros((count, 18)))
    (failed / "Out/Shift_Current/shift_current.dat").unlink()
    for name in ("Input", "Out/input.json"):
        path = failed / name
        path.write_text(
            path.read_text()
            .replace('"nspin": 4', '"nspin": 1')
            .replace("nspin                           4", "nspin                           1")
        )
    (failed / "inputs.sha256").write_text(f"{sha256(conflict / 'scf.scfout')}  scf.scfout\n")
    (failed / "pyatb.out").write_text("ValueError: nspin=1 is inconsistent with scfout SpinP_switch=3\n")
    report = judge(conflict, failed)
    assert report["verdict"] == PASS and classes(report)["status"] == AGREE, render_markdown(report)


def test_the_symmetry_gate_and_self_convergence_are_information(runs, printed, tmp_path):
    report = judge_run(runs, printed, tmp_path, "converged")
    rules = report["semantics"]["rules"]
    assert rules["symmetry"]["same"] and rules["self_convergence"]["converged"] is not None
    assert "symmetry" not in classes(report) and "self_convergence" not in failed_preconditions(report)
    rng = np.random.default_rng(3)
    reference = rng.normal(size=(30, 18))
    flipped = reference.copy()
    flipped[:, 3] *= -1
    facts = symmetry_gate(reference, flipped, 0.1)
    assert not facts["same"] and facts["integral_sign_differs"] == ["xyy"]
    zeroed = reference.copy()
    zeroed[:, 5] = 0
    assert symmetry_gate(reference, zeroed, 0.1)["components_differ"] == ["xzz"]


def test_the_attribution_telescopes_from_queers_value_to_pyatbs_table(runs, printed, tmp_path):
    """The steps the recomputation switches on one after another sum to its PyATB-literal table less its QUEER value:
    a documented difference is the recomputation's own, ingredient by ingredient."""
    report = judge_run(runs, printed, tmp_path, "split_window")
    rules = report["semantics"]["rules"]
    assert rules["path"] == ["queer", "constants", "hartree", "window", "bins", "scissors", "cut"]
    steps = {
        step
        for values in rules["attribution"].values()
        if values["class"] == DOCUMENTED
        for step, change in values["steps_uA_V2"].items()
        if change > 1e-6 * max(values["gap_uA_V2"], 1e-30)
    }
    assert {"window", "cut"} <= steps and "scissors" not in steps


def test_the_report_renders(runs, printed, tmp_path):
    report = judge_run(runs, printed, tmp_path, "window")
    text = render_markdown(report)
    assert "differs_documented" in text and report["tolerance_basis"]["recomputation"]["kpoints"] == 18
    assert classes(report).get("pyatb_reproduced") is None
    wrong = edited(tmp_path / "wrong", runs["window"], transformed("sigma_uA_V2", lambda t: 2 * t))
    report = judge(wrong, reference_from(tmp_path / "ref-wrong", wrong, 3 * printed("window")))
    # QUEER's own value is wrong: that fails, and PyATB's table is not judged.
    assert report["verdict"] == FAIL and classes(report)["pyatb_reproduced"] == NOT_APPLICABLE


@pytest.mark.skipif(not CALIBRATION_SCFOUT.is_file(), reason=f"no calibration scfout at {CALIBRATION_SCFOUT}")
def test_the_real_calibration_scfout_whose_shift_current_vanishes(tmp_path):
    """The nonorthogonal SOC calibration scfout with position moments and inversion: its shift current vanishes (the
    k-points' contributions cancel to 1e-12 of their size). QUEER and PyATB both print rounding, which the
    recomputation reproduces: a pass; a value above the rounding fails."""
    directory = candidate(tmp_path / "real", CALIBRATION_SCFOUT, {}, 4, REAL)
    table = port_table(directory)
    report = judge(directory, reference_from(tmp_path / "reference", directory, table))
    assert report["verdict"] == PASS and components(report) == {AGREE}, render_markdown(report)
    assert report["semantics"]["rules"]["determined"]["vanishing"]
    peak = np.max(np.abs(columns(read(directory / "queer_results.json")["shift_current"]["sigma_uA_V2"])))
    for label, change in (
        ("x2", lambda t: 2 * t),
        ("plus", lambda t: t + 1e-9),
        ("zxx up", lambda t: t + np.where(np.arange(18) == 12, 10 * peak, 0.0)),
    ):
        copy = edited(tmp_path / label, directory, transformed("sigma_uA_V2", change))
        report = judge(copy, reference_from(tmp_path / f"ref-{label}", copy, table))
        assert report["verdict"] == FAIL, label


@pytest.mark.skipif(not CALIBRATION_SCFOUT.is_file(), reason=f"no calibration scfout at {CALIBRATION_SCFOUT}")
def test_pyatbs_cut_at_natural_near_degeneracies_of_the_real_scfout_is_documented(tmp_path):
    """The calibration scfout on 18 x 18 x 1 with occ_band 16: four k-points hold levels 5.8e-5 eV apart, where PyATB's
    method 1 replaces 1/dE by dE/(dE^2 + 0.04^2) and so breaks the inversion that makes the physical value vanish.
    QUEER's vanishing value meets the recomputation, PyATB's table (0.019 microA/V^2) meets its transcription, and every
    component that differs is documented with PyATB's cut as its cause; QUEER's value moved by 1e-3 of PyATB's peak
    fails."""
    directory = candidate(
        tmp_path / "real", CALIBRATION_SCFOUT, {}, 4, {**REAL, "occ_band": 16, "grid": [18, 18, 1]}
    )
    table = port_table(directory)
    report = judge(directory, reference_from(tmp_path / "reference", directory, table))
    assert report["verdict"] == PASS_DOCUMENTED, render_markdown(report)
    rules = report["semantics"]["rules"]
    assert rules["determined"]["tables"]["physical.sigma_uA_V2"]["vanishing"]
    # The physical table vanishes (1.6e-6 of the 322 microA/V^2 that cancel in it) and is reported so: check A
    # judges it as zero, not its values below that rounding.
    assert classes(report)["physical.sigma_uA_V2"] == AGREE
    detail = next(item["detail"] for item in report["quantities"] if item["name"] == "physical.sigma_uA_V2")
    assert detail.startswith("vanishing (largest ")
    assert "physical.sigma_uA_V2 vanishing" in failed_or_passed(report, "determined")
    assert report["tolerance_basis"]["recomputation"]["flagged_kpoints"] == 4
    documented = [values for values in rules["attribution"].values() if values["class"] == DOCUMENTED]
    assert len(documented) >= 6 and all(
        max(values["steps_uA_V2"], key=values["steps_uA_V2"].get) == "cut" for values in documented
    )
    peak = np.max(np.abs(table))
    copy = edited(
        tmp_path / "wrong",
        directory,
        transformed("sigma_uA_V2", lambda t: t + np.where(np.arange(18) == 6, 1e-3 * peak, 0.0)),
    )
    report = judge(copy, reference_from(tmp_path / "ref-wrong", copy, table))
    assert report["verdict"] == FAIL and classes(report)["physical.sigma_uA_V2"] == UNEXPLAINED


@pytest.mark.skipif(not CALIBRATION_SCFOUT.is_file(), reason=f"no calibration scfout at {CALIBRATION_SCFOUT}")
@pytest.mark.parametrize(
    ("grid", "eta", "domega"),
    [([6, 6, 1], 0.01, 0.01), ([9, 9, 1], 0.1, 0.02)],
    ids=["6x6-eta0.01", "9x9-eta0.1"],
)
def test_noise_level_integrands_of_the_real_scfout_agree_as_noise(tmp_path, grid, eta, domega):
    """The calibration scfout with occ_band 4: at Gamma and the other TRIM points its near inversion leaves integrands
    of rounding size, which the transcription and QUEER compute differently. At PyATB's eta 0.01 such entries failed
    check A against a rounding taken from one second basis; at eta 0.1 PyATB's printed table was not reproduced at
    that noise level (2.98e-26 against 2.66e-26). The rounding now holds every factor's own rounding: a pass, with
    PyATB's table reproduced; QUEER's value doubled, or moved by 1e-7 of the magnitudes it cancels from, fails."""
    block = {**REAL, "occ_band": 4, "grid": grid, "eta": eta, "domega": domega}
    directory = candidate(tmp_path / "real", CALIBRATION_SCFOUT, {}, 4, block)
    table = port_table(directory)
    report = judge(directory, reference_from(tmp_path / "reference", directory, table))
    assert report["verdict"] == PASS, render_markdown(report)
    rules = report["semantics"]["rules"]
    assert rules["pyatb_reproduced"]["reproduced"]
    cancelled = rules["determined"]["tables"]["physical.sigma_uA_V2"]["cancelled_scale_uA_V2"]
    rows = len(table)
    for label, change in (
        ("x2", lambda t: 2 * t),
        ("plus 1e-7", lambda t: t + 1e-7 * cancelled * (np.arange(rows)[:, None] == rows // 3)),
    ):
        copy = edited(tmp_path / label.replace(" ", "-"), directory, transformed("sigma_uA_V2", change))
        report = judge(copy, reference_from(tmp_path / f"ref-{label.replace(' ', '-')}", copy, table))
        assert report["verdict"] == FAIL, label


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
