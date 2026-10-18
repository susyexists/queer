"""QUEER's OpenMX shift-current step on analytic scfout files without inversion.

The spinor model (``shift_spinor_scfout``) breaks inversion and time reversal and
keeps every level at least 0.28 eV from every other; the collinear model breaks
inversion with different channels. The runner must report QUEER's own
``shift_current`` (to rounding), PyATB's photon energies, component order, file
layout, spin factor and sign, and diagnostics that follow PyATB's rules
transition by transition.
"""

import json
import math

import numpy as np
import pytest
from openmx_files import shift_collinear_scfout, shift_spinor_scfout

from queer import ExecutionOptions, OpenMXModel
from queer.compat.generator.openmx_params import limit_refusal
from queer.compat.generator.openmx_shift import (
    FLAGGED_RECORD_LIMIT,
    PyatbShiftScheme,
    bin_rows,
    component_table,
    edge_transitions,
    pyatb_scheme_spectrum,
)
from queer.compat.generator.openmx_steps import single_channel_model
from queer.compat.generator.pyatb_conventions import (
    HARTREE_EV,
    PYATB_HARTREE_EV,
    pyatb_shift_constant_ratio,
)
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
    transitions = diagnostics["transitions"]
    assert transitions["count"] == 18 * 4 == transitions["inside_window"] and transitions["margin_eta"] > 5
    assert diagnostics["clean_control"]["eligible"] and all(
        diagnostics["clean_control"]["conditions"].values()
    )
    truncation = diagnostics["gaussian_truncation"]
    assert truncation["interval_bins"] == 5 and truncation["tail_relative"] == pytest.approx(math.exp(-25))
    assert truncation["max_effect_uA_V2"] < 1e-9 * np.max(np.abs(physical))
    scheme = columns(diagnostics["pyatb_scheme"]["sigma_uA_V2"])
    # PyATB's Hartree scale moves each transition by 2.3e-6 of its energy: far below 1e-4 of the peak.
    assert 0 < np.max(np.abs(scheme - physical)) < 1e-4 * np.max(np.abs(physical))
    assert document["model"]["spin_mode"] == "soc" and document["integration"]["mode"] == "Grid"
    # The build a clean control vouches for: every source file of the installed package.
    assert block["build"]["queer_source_sha256"] == package_source_sha256() is not None
    regularisation = diagnostics["regularisation"]
    assert regularisation["flagged_kpoints"] == 0 and regularisation["kpoints"] == 18
    assert not np.any(columns(regularisation["shift_uA_V2"]))
    # Without a flagged k-point the value without them is the value, and none is recorded.
    assert np.allclose(columns(regularisation["sigma_unflagged_uA_V2"]), physical, rtol=1e-13, atol=0)
    records = regularisation["flagged_records"]
    assert (
        records["recorded"] and records["kpoints"] == [] and records["limit_kpoints"] == FLAGGED_RECORD_LIMIT
    )
    # The emulation's own sum of its transitions under QUEER's Gaussian is the value; nothing lies outside the
    # window, so PyATB's window drops nothing.
    pyatb = diagnostics["pyatb_scheme"]
    listed = columns(pyatb["transitions_uA_V2"])
    assert np.max(np.abs(listed - physical)) <= 1e-12 * np.max(np.abs(physical))
    assert not np.any(columns(pyatb["window_dropped_uA_V2"]))
    assert pyatb["moved_transitions"] is None and pyatb["edge_transitions"] == 0
    assert "transition_bounds" not in diagnostics and "edge_allowance_uA_V2" not in pyatb


def pyatb_loops(scheme, energies, weights):
    """PyATB's shift-current broadening written loop by loop (its window, int() bins and truncation)."""
    table = np.zeros((scheme.count, len(SHIFT_COMPONENTS)))
    interval = int(5 * scheme.eta / scheme.domega)
    for energy, weight in zip(np.asarray(energies) * PYATB_HARTREE_EV / HARTREE_EV, weights):
        if not scheme.pair_mode and not (
            scheme.start <= energy < scheme.start + scheme.count * scheme.domega
        ):
            continue
        index = int((energy - scheme.start) / scheme.domega)
        if scheme.smearing == 0:
            if 0 <= index < scheme.count:
                table[index] += weight
            continue
        for row in range(index - interval, index + interval + 1):
            if 0 <= row < scheme.count:
                omega = scheme.start + row * scheme.domega
                table[row] += (
                    weight
                    * math.exp(-(((energy - omega) / scheme.eta) ** 2))
                    / scheme.eta
                    / math.sqrt(math.pi)
                )
    return table


@pytest.mark.parametrize("smearing", [0, 1])
@pytest.mark.parametrize("pair_mode", [False, True])
def test_the_pyatb_scheme_follows_pyatbs_loops_transition_by_transition(smearing, pair_mode):
    rng = np.random.default_rng(7)
    # Transitions inside, just below (C's int() truncates toward zero) and beyond the window.
    energies = np.concatenate((rng.uniform(1.0, 2.0, 40), [0.995, 0.97, 2.3, 2.6, 1.0000001]))
    weights = rng.normal(size=(len(energies), len(SHIFT_COMPONENTS)))
    scheme = PyatbShiftScheme(1.0, 0.04, 26, 0.05, smearing, pair_mode)
    spectrum, untruncated = pyatb_scheme_spectrum(scheme, energies, weights)
    assert np.allclose(spectrum, pyatb_loops(scheme, energies, weights), rtol=1e-13, atol=1e-13)
    if smearing == 1:
        grid = 1.0 + 0.04 * np.arange(26)
        scaled = energies * PYATB_HARTREE_EV / HARTREE_EV
        gauss = np.exp(-(((scaled[:, None] - grid) / 0.05) ** 2)) / 0.05 / math.sqrt(math.pi)
        assert np.allclose(untruncated, gauss.T @ weights, rtol=1e-13, atol=1e-13)
    else:
        assert untruncated is None
    if pair_mode:
        # A named pair keeps every transition: 0.995 eV lands in bin 0, as int(-0.125) == 0.
        assert np.any(spectrum[0]) and pyatb_loops(scheme, energies[[0]], weights[[0]]).any()


def test_edge_transitions_counts_transitions_on_bin_and_window_edges():
    scheme = PyatbShiftScheme(1.0, 0.1, 10, 0.05, 0, False, energy_scale=1.0)
    energies = np.array([1.2 + 1e-12, 1.25, 1.0 - 1e-12, 1.5])
    assert edge_transitions(scheme, energies) == 3
    assert edge_transitions(scheme, energies[[1]]) == 0


def test_bin_rows_name_the_transitions_queers_and_pyatbs_bins_place_apart():
    scheme = PyatbShiftScheme(1.0, 0.1, 10, 0.05, 0, False)
    half = 1.0 + 0.1 * np.arange(10) + 0.05
    # Inside a bin, on a bin edge, moved across the window's end by PyATB's Hartree scale (2.0000001 eV is
    # 1.9999956 eV on its scale), and below and above the window.
    energies = np.array([1.25, 1.3, 2.0000001, 0.95, 2.05])
    own, pyatb, ambiguous = bin_rows(scheme, half, 0.1, energies)
    assert own[0] == pyatb[0] == 2 and not ambiguous[0]
    assert ambiguous[1]
    assert own[2] == -1 and pyatb[2] == 9 and not ambiguous[2]
    assert own[3] == pyatb[3] == -1 and own[4] == pyatb[4] == -1
    # A named pair keeps 0.95 eV: PyATB's int(-0.5) == 0 puts it in bin 0, QUEER's histogram drops it.
    own, pyatb, _ = bin_rows(PyatbShiftScheme(1.0, 0.1, 10, 0.05, 0, True), half, 0.1, energies)
    assert own[3] == -1 and pyatb[3] == 0
    # PyATB's bins are QUEER's histogram half a bin higher plus the moves of the transitions the two place apart.
    rng = np.random.default_rng(5)
    for pair_mode in (False, True):
        scheme = PyatbShiftScheme(1.0, 0.1, 10, 0.05, 0, pair_mode)
        energies = np.concatenate((rng.uniform(0.8, 2.2, 3000), 1.0 + 0.1 * np.arange(12)))
        weights = rng.normal(size=(len(energies), len(SHIFT_COMPONENTS)))
        pyatb_table, _ = pyatb_scheme_spectrum(scheme, energies, weights)
        lines = (half[:, None] - energies >= -0.05) & (half[:, None] - energies < 0.05)
        queer = lines.astype(float) @ weights
        own, pyatb, ambiguous = bin_rows(scheme, half, 0.1, energies)
        moved = (own != pyatb) | ambiguous
        moves = np.zeros_like(queer)
        np.add.at(moves, pyatb[moved & (pyatb >= 0)], weights[moved & (pyatb >= 0)])
        np.subtract.at(moves, own[moved & (own >= 0)], weights[moved & (own >= 0)])
        assert np.allclose(pyatb_table, queer + moves, rtol=0, atol=1e-12) and moved.any()


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
    scheme = columns(block["diagnostics"]["pyatb_scheme"]["sigma_uA_V2"])
    assert np.max(np.abs(scheme - 2 * channels[0])) < 1e-4 * scale
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
    scheme = columns(block["diagnostics"]["pyatb_scheme"]["sigma_uA_V2"])
    assert np.max(np.abs(scheme - physical)) < 1e-4 * np.max(np.abs(physical))


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
    assert not block["diagnostics"]["clean_control"]["eligible"]
    written = parse_shift_current((tmp_path / "Out/Shift_Current/shift_current.dat").read_text())
    assert np.allclose(written["sigma"], binned, rtol=5.01e-7, atol=0)
    scheme = columns(block["diagnostics"]["pyatb_scheme"]["sigma_uA_V2"])
    # PyATB's bins and QUEER's differ only by the transitions the record names, each in its two rows.
    record = block["diagnostics"]["pyatb_scheme"]["moved_transitions"]
    assert record["recorded"] and block["diagnostics"]["pyatb_scheme"]["window_dropped_uA_V2"] is None
    moves = np.zeros_like(binned)
    for weight, own, theirs in zip(record["weight_uA_V2_eV"], record["queer_row"], record["pyatb_row"]):
        if theirs >= 0:
            moves[theirs] += weight
        if own >= 0:
            moves[own] -= weight
    ratio = pyatb_shift_constant_ratio()
    assert np.allclose(scheme, ratio * (binned + moves), rtol=0, atol=1e-9 * np.max(np.abs(binned)))


def gaussian_rows(omega, energies, weights, eta):
    """QUEER's Gaussian (standard deviation eta/sqrt(2)) of recorded transitions on the photon energies."""
    energies = np.asarray(energies)
    kernel = np.exp(-(((np.asarray(omega)[:, None] - energies[None, :]) / eta) ** 2)) / (
        eta * math.sqrt(math.pi)
    )
    return kernel @ np.asarray(weights).reshape(len(energies), len(SHIFT_COMPONENTS))


def test_every_flagged_kpoint_is_recorded_with_its_transitions(tmp_path, monkeypatch):
    """A comparison recomputes PyATB's method-1 cut from the scfout at the k-points the runner flags: it records
    each one (fractional coordinates) with QUEER's transitions there, physical and with the cut, which sum to
    the value less the value without them and to the shift; beyond the record's limit it keeps none."""
    payload = shift_params(grid=[4, 4, 4])
    code, document = execute(tmp_path / "split", payload, exchange=2e-5)
    assert code == EXIT_OK, document.get("error")
    block = document["shift_current"]
    regularisation = block["diagnostics"]["regularisation"]
    records = regularisation["flagged_records"]
    assert regularisation["flagged_kpoints"] == 8 and records["recorded"] and records["reason"] is None
    omega = np.asarray(block["omega_eV"])
    physical = columns(block["sigma_uA_V2"])
    total = np.zeros_like(physical)
    cut = np.zeros_like(physical)
    keys = set()
    for item in records["kpoints"]:
        keys.add(tuple(np.round(np.asarray(item["k"]) * 4).astype(int)))
        assert item["pair_gap_eV"] == pytest.approx(4e-5, rel=1e-6)
        spectrum = gaussian_rows(
            omega, item["physical"]["energy_eV"], item["physical"]["weight_uA_V2_eV"], 0.1
        )
        total += spectrum
        cut += spectrum - gaussian_rows(omega, item["cut"]["energy_eV"], item["cut"]["weight_uA_V2_eV"], 0.1)
    assert len(keys) == 8
    scale = np.max(np.abs(physical))
    unflagged = columns(regularisation["sigma_unflagged_uA_V2"])
    assert np.max(np.abs(physical - unflagged - total)) <= 1e-12 * scale
    assert np.max(np.abs(columns(regularisation["shift_uA_V2"]) - cut)) <= 1e-12 * scale
    # The emulation's transitions under QUEER's Gaussian are the value less the measured cut.
    listed = columns(block["diagnostics"]["pyatb_scheme"]["transitions_uA_V2"])
    assert np.max(np.abs(listed - (physical - columns(regularisation["shift_uA_V2"])))) <= 1e-12 * scale
    # Past the record's limit nothing is kept: a comparison then cannot recompute the cut.
    import queer.compat.generator.openmx_shift as shift_module

    monkeypatch.setattr(shift_module, "FLAGGED_RECORD_LIMIT", 4)
    code, document = execute(tmp_path / "limited", payload, exchange=2e-5)
    records = document["shift_current"]["diagnostics"]["regularisation"]["flagged_records"]
    assert code == EXIT_OK and not records["recorded"] and records["kpoints"] == []
    assert "8 flagged k-points" in records["reason"]


def test_the_window_record_holds_what_pyatbs_window_drops(tmp_path):
    """QUEER's value of the transitions outside PyATB's window, at the k-points it does not regularize: it reaches
    only the photon energies within a few eta of the window's ends."""
    code, document = execute(tmp_path, shift_params(omega=[2.0, 4.0]))
    assert code == EXIT_OK, document.get("error")
    block = document["shift_current"]
    pyatb = block["diagnostics"]["pyatb_scheme"]
    dropped = columns(pyatb["window_dropped_uA_V2"])
    omega = np.asarray(block["omega_eV"])
    end = 2.0 + len(omega) * 0.1
    far = np.minimum(omega - 2.0, end - omega) >= 0.6
    physical = columns(block["sigma_uA_V2"])
    assert np.max(np.abs(dropped)) > 0.1 * np.max(np.abs(physical))
    assert np.max(np.abs(dropped[far])) <= 1e-12 * np.max(np.abs(physical))
    model = OpenMXModel.from_scfout(tmp_path / "scf.scfout")
    # QUEER's own estimator without the transitions outside [2.0, 4.1) eV on PyATB's scale: the value less the record.
    from queer.compat.generator.openmx_shift import PyatbShiftScheme as Scheme
    from queer.observables.nonlinear import shift_current_point, shift_current_settings

    scheme = Scheme(2.0, 0.1, len(omega), 0.1, 1, False)
    settings = shift_current_settings(
        model, omega, model.fermi_energy, eta=0.1 / math.sqrt(2), smearing="gaussian", occupied_bands=2
    )
    kept = np.zeros_like(physical)
    points = np.stack(np.meshgrid(*[np.arange(n) / n for n in (3, 3, 2)], indexing="ij"), -1).reshape(-1, 3)
    for k in points:
        point = shift_current_point(model, k, settings)
        source, target = np.nonzero(point["factor"])
        energies = point["gap"][source, target]
        weights = component_table(
            np.moveaxis(point["factor"][source, target] * point["integrand"][..., source, target], -1, 0)
        )
        inside = scheme.inside(energies)
        kept += gaussian_rows(omega, energies[inside], weights[inside], 0.1)
    from queer.compat.generator.openmx_shift import conversion

    kept *= conversion(model) / len(points)
    assert np.max(np.abs(physical - dropped - kept)) <= 1e-12 * np.max(np.abs(physical))


def test_a_left_handed_lattice_flips_what_the_file_holds(tmp_path):
    lattice = np.diag([3.0, 3.2, -3.5])
    code, document = execute(tmp_path, shift_params(), lattice=lattice)
    assert code == EXIT_OK, document.get("error")
    block = document["shift_current"]
    assert document["model"]["lattice_handedness"] == -1 and block["convention"]["export_sign"] == -1
    physical = columns(block["sigma_uA_V2"])
    written = parse_shift_current((tmp_path / "Out/Shift_Current/shift_current.dat").read_text())
    assert np.allclose(written["sigma"], -physical, rtol=5.01e-7, atol=0)
    scheme = columns(block["diagnostics"]["pyatb_scheme"]["sigma_uA_V2"])
    assert np.max(np.abs(scheme + physical)) < 1e-4 * np.max(np.abs(physical))


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


def test_the_runner_records_what_a_comparison_recomputes(tmp_path):
    """Per k-point, in the grid's C order: each k-point's rounding scale (its conditioning times its transitions'
    product moduli, which bound their weights); the k-points with an exact degeneracy and its spacings; the
    transitions PyATB's window drops within reach of a photon energy (their Gaussian is the window table) and those
    it keeps at its ends; the moved transitions' k-points; and a collinear scfout's two channels."""
    from queer.compat.generator.openmx_shift import WINDOW_REACH_ETA

    code, document = execute(tmp_path / "window", shift_params(omega=[2.0, 4.0], domega=0.02))
    assert code == EXIT_OK, document.get("error")
    block = document["shift_current"]
    diagnostics = block["diagnostics"]
    rounding = diagnostics["rounding"]
    assert rounding["recorded"] and len(rounding["kpoint_scale_uA_V2_eV"]) == 18
    assert min(rounding["kpoint_scale_uA_V2_eV"]) > 0
    window = diagnostics["pyatb_scheme"]["window_transitions"]
    assert window["recorded"] and window["energy_eV"] and not any(window["kept"])
    energies = np.asarray(window["energy_eV"])
    omega = np.asarray(block["omega_eV"])
    scale = PYATB_HARTREE_EV / HARTREE_EV
    outside = (energies * scale < 2.0) | (energies * scale >= 2.0 + len(omega) * 0.02)
    assert np.all(outside)
    assert np.all(
        (energies >= omega[0] - WINDOW_REACH_ETA * 0.1) & (energies <= omega[-1] + WINDOW_REACH_ETA * 0.1)
    )
    weights = np.asarray(window["weight_uA_V2_eV"])
    record = gaussian_rows(omega, energies, weights, 0.1)
    dropped = columns(diagnostics["pyatb_scheme"]["window_dropped_uA_V2"])
    assert np.max(np.abs(record - dropped)) <= 1e-12 * np.max(np.abs(dropped))
    assert set(window["k_index"]) <= set(range(18))
    # Kramers pairs at the 8 TRIM points of a 4 x 4 x 4 grid: each recorded with its exact pairs.
    code, document = execute(tmp_path / "kramers", shift_params(grid=[4, 4, 4]), exchange=0.0)
    exact = document["shift_current"]["diagnostics"]["exact_degeneracies"]
    assert (
        exact["recorded"]
        and len(exact["kpoints"]) == 8 == document["shift_current"]["diagnostics"]["exact_degenerate_kpoints"]
    )
    points = np.stack(np.meshgrid(*[np.arange(4) / 4] * 3, indexing="ij"), -1).reshape(-1, 3)
    for item in exact["kpoints"]:
        assert np.allclose(points[item["k_index"]], item["k"]) and np.all(np.asarray(item["k"]) * 2 % 1 == 0)
        assert item["pairs"] and all(
            spacing <= exact["degeneracy_tolerance_eV"] for _, _, spacing in item["pairs"]
        )
    # The bins' moved transitions name their k-points.
    code, document = execute(tmp_path / "binned", shift_params(smearing_method=0, domega=0.05))
    moved = document["shift_current"]["diagnostics"]["pyatb_scheme"]["moved_transitions"]
    assert len(moved["k_index"]) == len(moved["energy_eV"])
    # A collinear scfout: its physical value is its two channels' sum, the legacy one twice the spin-up channel.
    code, document = execute(
        tmp_path / "collinear",
        shift_params(nspin=1, occ_band=1, omega=[0.2, 4.5], domega=0.04, grid=[6, 6, 1]),
        scfout=shift_collinear_scfout,
    )
    block = document["shift_current"]
    channels = block["diagnostics"]["spin_channels"]
    up, down = columns(channels["up_uA_V2"]), columns(channels["down_uA_V2"])
    assert np.max(np.abs(columns(block["sigma_uA_V2"]) - up - down)) <= 1e-12 * np.max(np.abs(up))
    assert np.max(np.abs(columns(block["legacy_single_channel"]["sigma_uA_V2"]) - 2 * up)) <= 1e-12 * np.max(
        np.abs(up)
    )
    assert np.max(np.abs(down)) > 1e-2 * np.max(np.abs(up))
    assert (
        channels["down_rounding"]["recorded"]
        and len(channels["down_rounding"]["kpoint_scale_uA_V2_eV"]) == 36
    )
