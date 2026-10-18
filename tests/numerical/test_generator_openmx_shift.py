"""QUEER's OpenMX shift-current step on analytic scfout files without inversion.

The spinor model (``shift_spinor_scfout``) breaks inversion and time reversal and
keeps every level at least 0.28 eV from every other; the collinear model breaks
inversion with different channels. The runner must report QUEER's own
``shift_current`` (to rounding), PyATB's photon energies, component order, file
layout, spin factor and sign, and plain diagnostics of its transitions (a
comparison recomputes everything it judges from the scfout).
"""

import json
import math

import numpy as np
import pytest
from openmx_files import shift_collinear_scfout, shift_spinor_scfout

from queer import ExecutionOptions, OpenMXModel
from queer.compat.generator.openmx_params import limit_refusal
from queer.compat.generator.openmx_shift import component_table
from queer.compat.generator.openmx_steps import single_channel_model
from queer.compat.generator.pyatb_files import SHIFT_COMPONENTS, parse_shift_current
from queer.compat.generator.results import EXIT_FAILED, EXIT_OK, package_source_sha256
from queer.compat.generator.runner import capabilities, run
from queer.observables.nonlinear import shift_current

SERIAL = ExecutionOptions(workers=1)


def shift_params(nspin=4, **block):
    function = {
        "occ_band": 2,
        "omega": [0.3, 6.5],
        "domega": 0.1,
        "smearing_method": 1,
        "eta": 0.1,
        "grid": [3, 3, 2],
        "method": 1,
        **block,
    }
    return {
        "step_type": "QUEER_SHIFT_CURRENT",
        "counterpart_step_type": "PYATB_SHIFT_CURRENT",
        "input_parameters": {"nspin": nspin},
        "parameters": {key: value for key, value in function.items() if value is not None},
        "workers": 1,
    }


def execute(directory, payload, scfout=shift_spinor_scfout, **model):
    directory.mkdir(parents=True, exist_ok=True)
    if scfout is not None:
        scfout(directory / "scf.scfout", **model)
    (directory / "queer_params.json").write_text(json.dumps(payload))
    code = run(None, directory / "queer_params.json", directory, log=lambda line: None)
    return code, json.loads((directory / "queer_results.json").read_text())


def columns(block):
    return np.column_stack([block[name] for name in SHIFT_COMPONENTS])


def pyatb_omega(start, end, step):
    return np.array([start + step * index for index in range(int((end - start) / step) + 1)])


def test_capabilities_list_the_shift_current_step():
    assert "QUEER_SHIFT_CURRENT" in capabilities()["steps"]


def test_the_shift_current_is_queers_own_estimator_in_pyatbs_layout(tmp_path):
    code, document = execute(tmp_path, shift_params())
    assert code == EXIT_OK and document["status"] == "ok", document.get("error")
    block = document["shift_current"]
    omega = pyatb_omega(0.3, 6.5, 0.1)
    assert block["omega_eV"] == omega.tolist() and len(omega) == int(6.2 / 0.1) + 1
    assert block["components"] == list(SHIFT_COMPONENTS) == block["components_computed"]
    assert (
        SHIFT_COMPONENTS[:6] == ("xxx", "xxy", "xxz", "xyy", "xyz", "xzz") and SHIFT_COMPONENTS[-1] == "zzz"
    )
    model = OpenMXModel.from_scfout(tmp_path / "scf.scfout")
    expected = shift_current(
        model,
        omega,
        eta=0.1 / math.sqrt(2),
        smearing="gaussian",
        grid=(3, 3, 2),
        occupied_bands=2,
        scissors_occupied_bands=2,
        execution=SERIAL,
    )["shift_current"]
    physical = columns(block["sigma_uA_V2"])
    assert np.max(np.abs(physical - component_table(expected))) <= 1e-12 * np.max(np.abs(physical))
    assert np.max(np.abs(physical)) > 0.1 and np.all(np.max(np.abs(physical), axis=0) > 1e-4 * physical.max())
    assert block["smearing"]["gaussian_sigma_eV"] == pytest.approx(0.1 / math.sqrt(2), rel=1e-15)
    assert block["convention"] == {
        **block["convention"],
        "export_sign": 1,
        "pyatb_spin_factor": 1,
        "pyatb_nspin": 4,
    }
    assert block["legacy_single_channel"] is None and block["binned_uA_V2"] is None
    written = parse_shift_current((tmp_path / "Out/Shift_Current/shift_current.dat").read_text())
    assert written["omega_eV"] == pytest.approx(omega.tolist(), abs=5e-6)
    assert np.allclose(written["sigma"], physical, rtol=5.01e-7, atol=0)
    assert {
        "path": "Out/Shift_Current/shift_current.dat",
        "title": "Shift current (PyATB layout)",
        "kind": "data",
    } in (document["outputs"])
    diagnostics = block["diagnostics"]
    assert diagnostics["min_pair_gap_eV"] > 0.28 and diagnostics["kpoints_at_pyatb_threshold"] == 0
    assert diagnostics["exact_degenerate_kpoints"] == 0
    transitions = diagnostics["transitions"]
    assert transitions["count"] == 18 * 4 == transitions["inside_window"] and transitions["margin_eta"] > 5
    truncation = diagnostics["gaussian_truncation"]
    assert truncation["interval_bins"] == 5 and truncation["tail_relative"] == pytest.approx(math.exp(-25))
    assert document["model"]["spin_mode"] == "soc" and document["integration"]["mode"] == "Grid"
    # The build a comparison names: every source file of the installed package.
    assert block["build"]["queer_source_sha256"] == package_source_sha256() is not None
    # Nothing that only served the earlier comparator: a comparison recomputes what it judges from the scfout.
    retired = {
        "pyatb_scheme",
        "regularisation",
        "rounding",
        "clean_control",
        "exact_degeneracies",
        "spin_channels",
    }
    assert not retired & set(diagnostics) and "transition_bounds" not in diagnostics


def test_a_collinear_scfout_reports_both_channels_and_pyatbs_doubled_spin_up_channel(tmp_path):
    payload = shift_params(nspin=1, occ_band=1, omega=[0.2, 4.5], grid=[6, 6, 1])
    code, document = execute(tmp_path, payload, scfout=shift_collinear_scfout)
    assert code == EXIT_OK, document.get("error")
    block = document["shift_current"]
    model = OpenMXModel.from_scfout(tmp_path / "scf.scfout")
    omega = np.asarray(block["omega_eV"])
    arguments = dict(eta=0.1 / math.sqrt(2), smearing="gaussian", grid=(6, 6, 1), execution=SERIAL)
    channels = [
        component_table(
            shift_current(single_channel_model(model, c), omega, occupied_bands=1, **arguments)[
                "shift_current"
            ]
        )
        for c in (0, 1)
    ]
    physical, legacy = columns(block["sigma_uA_V2"]), block["legacy_single_channel"]
    scale = np.max(np.abs(physical))
    assert np.max(np.abs(physical - channels[0] - channels[1])) <= 1e-12 * scale
    assert legacy["spin_factor"] == 2 and legacy["channel"] == 0
    assert np.max(np.abs(columns(legacy["sigma_uA_V2"]) - 2 * channels[0])) <= 1e-12 * scale
    assert np.max(np.abs(columns(legacy["sigma_uA_V2"]) - physical)) > 0.05 * scale
    assert block["convention"]["pyatb_spin_factor"] == 2 and block["convention"]["pyatb_nspin"] == 1
    assert "shift_collinear_channels" in document["warning_codes"]
    # The file holds the physical value, not PyATB's legacy channel.
    written = parse_shift_current((tmp_path / "Out/Shift_Current/shift_current.dat").read_text())
    assert np.allclose(written["sigma"], physical, rtol=5.01e-7, atol=0)
    with pytest.raises(ValueError, match="channels 0 and 1"):
        single_channel_model(OpenMXModel.from_scfout(tmp_path / "scf.scfout"), 2)


def test_a_non_spin_polarized_scfout_doubles_its_channel_as_pyatb_does(tmp_path):
    payload = shift_params(nspin=1, occ_band=1, omega=[0.2, 4.5], grid=[5, 5, 1])
    code, document = execute(tmp_path, payload, scfout=shift_collinear_scfout, spin=0)
    assert code == EXIT_OK, document.get("error")
    block = document["shift_current"]
    model = OpenMXModel.from_scfout(tmp_path / "scf.scfout")
    assert model.spin_mode == "nonmagnetic" and model.spin_degeneracy == 2
    expected = shift_current(
        model,
        np.asarray(block["omega_eV"]),
        eta=0.1 / math.sqrt(2),
        smearing="gaussian",
        grid=(5, 5, 1),
        occupied_bands=1,
        execution=SERIAL,
    )["shift_current"]
    physical = columns(block["sigma_uA_V2"])
    assert np.max(np.abs(physical - component_table(expected))) <= 1e-12 * np.max(np.abs(physical))
    assert block["legacy_single_channel"] is None and block["convention"]["pyatb_spin_factor"] == 2


def test_without_smearing_the_file_holds_pyatbs_bin_weights(tmp_path):
    code, document = execute(tmp_path, shift_params(smearing_method=0, domega=0.05))
    assert code == EXIT_OK, document.get("error")
    block = document["shift_current"]
    omega = np.asarray(block["omega_eV"])
    model = OpenMXModel.from_scfout(tmp_path / "scf.scfout")
    common = dict(
        smearing="none", bin_width=0.05, eta=0.01, grid=(3, 3, 2), occupied_bands=2, execution=SERIAL
    )
    histogram = component_table(shift_current(model, omega, **common)["shift_current"])
    assert np.max(np.abs(columns(block["sigma_uA_V2"]) - histogram)) <= 1e-12 * np.max(np.abs(histogram))
    # PyATB's bin [w_i, w_i + domega) is domega times QUEER's histogram half a bin higher.
    shifted = component_table(shift_current(model, omega + 0.025, **common)["shift_current"])
    binned = columns(block["binned_uA_V2"])
    assert np.allclose(binned, 0.05 * shifted, rtol=1e-12, atol=1e-12 * np.max(np.abs(binned)))
    assert block["smearing"]["bin_width_eV"] == 0.05 and block["diagnostics"]["gaussian_truncation"] is None
    written = parse_shift_current((tmp_path / "Out/Shift_Current/shift_current.dat").read_text())
    assert np.allclose(written["sigma"], binned, rtol=5.01e-7, atol=0)


def test_a_left_handed_lattice_flips_what_the_file_holds(tmp_path):
    lattice = np.diag([3.0, 3.2, -3.5])
    code, document = execute(tmp_path, shift_params(), lattice=lattice)
    assert code == EXIT_OK, document.get("error")
    block = document["shift_current"]
    assert document["model"]["lattice_handedness"] == -1 and block["convention"]["export_sign"] == -1
    physical = columns(block["sigma_uA_V2"])
    written = parse_shift_current((tmp_path / "Out/Shift_Current/shift_current.dat").read_text())
    assert np.allclose(written["sigma"], -physical, rtol=5.01e-7, atol=0)


def test_self_convergence_adds_the_doubled_grid(tmp_path):
    payload = {**shift_params(grid=[2, 2, 1], omega=[0.3, 6.5]), "self_convergence": True}
    code, document = execute(tmp_path, payload)
    assert code == EXIT_OK, document.get("error")
    convergence = document["shift_current"]["diagnostics"]["self_convergence"]
    assert convergence["grid"] == [2, 2, 1] and convergence["grid_doubled"] == [4, 4, 1]
    model = OpenMXModel.from_scfout(tmp_path / "scf.scfout")
    omega = np.asarray(document["shift_current"]["omega_eV"])
    doubled = component_table(
        shift_current(
            model,
            omega,
            eta=0.1 / math.sqrt(2),
            smearing="gaussian",
            grid=(4, 4, 1),
            occupied_bands=2,
            execution=SERIAL,
        )["shift_current"]
    )
    physical = convergence["physical"]
    assert np.max(np.abs(columns(physical["sigma_doubled_uA_V2"]) - doubled)) <= 1e-12 * np.max(
        np.abs(doubled)
    )
    coarse = columns(document["shift_current"]["sigma_uA_V2"])
    assert physical["e_inf_uA_V2"] == pytest.approx(np.max(np.abs(coarse - doubled)), rel=1e-12)
    assert physical["converged"] is (physical["e_inf_uA_V2"] <= 0.1 * physical["max_doubled_uA_V2"])
    assert convergence["legacy_single_channel"] is None
    assert ("shift_current_unconverged" in document["warning_codes"]) is (not physical["converged"])
    assert document["integration"]["self_convergence"] == {"grid": [2, 2, 1], "grid_doubled": [4, 4, 1]}


@pytest.mark.parametrize(
    ("block", "code", "message", "limit"),
    [
        ({"occ_band": None}, "queer_bad_params", "occ_band is required", False),
        ({"grid": None}, "queer_bad_params", "grid is required", False),
        ({"occ_band": 4}, "queer_bad_params", "below the 4 spinor bands PyATB reads", False),
        ({"scissors": -0.1}, "queer_bad_params", "rigid upward shift", True),
        ({"n_occ": 3, "m_unocc": 4}, "queer_bad_params", "occupied band to an empty one", True),
        ({"eta": 0.0}, "queer_bad_params", "eta must be positive", False),
        ({"smearing_method": 3}, "queer_bad_params", "smearing_method must be", False),
    ],
)
def test_shift_current_refusals_have_stable_codes(tmp_path, block, code, message, limit):
    exit_code, document = execute(tmp_path, shift_params(**block))
    assert exit_code == EXIT_FAILED and document["status"] == "failed"
    error = document["error"]
    assert error["code"] == code and message in error["message"], error
    assert limit_refusal(error["code"], error["message"]) is limit


def test_a_negative_photon_energy_start_is_computed_as_pyatb_takes_it(tmp_path):
    code, document = execute(tmp_path / "negative", shift_params(omega=[-0.5, 6.5]))
    assert code == EXIT_OK, document.get("error")
    block = document["shift_current"]
    assert block["omega_eV"] == pyatb_omega(-0.5, 6.5, 0.1).tolist()
    code, document = execute(tmp_path / "decreasing", shift_params(omega=[2.0, 1.0]))
    assert code == EXIT_FAILED and "increasing photon-energy range" in document["error"]["message"]
    assert "omega is required" not in document["error"]["message"]


def test_a_missing_scfout_and_a_split_degenerate_level_fail_with_their_codes(tmp_path):
    code, document = execute(tmp_path / "missing", shift_params(), scfout=None)
    assert code == EXIT_FAILED and document["error"]["code"] == "queer_missing_input"
    # Without exchange the spinor model keeps time reversal: its Kramers pairs at Gamma cannot be cut.
    code, document = execute(tmp_path / "kramers", shift_params(occ_band=1), exchange=0.0)
    assert code == EXIT_FAILED and document["error"]["code"] == "queer_validation_failed", document["error"]
    assert "degenerate" in document["error"]["message"]


def test_pyatbs_band_pair_rule_names_one_pair_or_every_pair(tmp_path):
    code, document = execute(tmp_path / "pair", shift_params(n_occ=2, m_unocc=3))
    assert code == EXIT_OK, document.get("error")
    occupation = document["shift_current"]["occupation"]
    assert occupation["band_pair"] == [1, 2] and occupation["n_occ"] == 2
    assert document["shift_current"]["diagnostics"]["transitions"]["count"] == 18
    code, document = execute(tmp_path / "all", shift_params(n_occ=0, m_unocc=3))
    assert code == EXIT_OK and document["shift_current"]["occupation"]["band_pair"] is None
    assert document["shift_current"]["occupation"]["n_occ"] == -1
    assert document["shift_current"]["diagnostics"]["transitions"]["count"] == 72


def test_the_diagnostics_are_plain_facts_of_the_run(tmp_path):
    """The transitions PyATB's window drops (with a warning), the k-points where an exact degeneracy takes part and
    those where a band of a selected pair lies within PyATB's 1e-4 eV of another level (with a warning)."""
    code, document = execute(tmp_path / "window", shift_params(omega=[2.0, 4.0]))
    assert code == EXIT_OK, document.get("error")
    transitions = document["shift_current"]["diagnostics"]["transitions"]
    assert transitions["below_window"] + transitions["above_window"] > 0
    assert (
        transitions["count"]
        == transitions["inside_window"] + transitions["below_window"] + transitions["above_window"]
    )
    assert "shift_transitions_outside_window" in document["warning_codes"]
    # Kramers pairs at the 8 TRIM points of a 4 x 4 x 4 grid: exact degeneracies, nothing PyATB regularizes.
    code, document = execute(tmp_path / "kramers", shift_params(grid=[4, 4, 4]), exchange=0.0)
    diagnostics = document["shift_current"]["diagnostics"]
    assert diagnostics["exact_degenerate_kpoints"] == 8 and diagnostics["kpoints_at_pyatb_threshold"] == 0
    # Split by 4e-5 eV: PyATB's method 1 regularizes those 8 k-points.
    code, document = execute(tmp_path / "split", shift_params(grid=[4, 4, 4]), exchange=2e-5)
    diagnostics = document["shift_current"]["diagnostics"]
    assert diagnostics["kpoints_at_pyatb_threshold"] == 8 and diagnostics["exact_degenerate_kpoints"] == 0
    assert diagnostics["min_pair_gap_eV"] == pytest.approx(4e-5, rel=1e-3)
    assert "shift_pair_gap_at_pyatb_threshold" in document["warning_codes"]
