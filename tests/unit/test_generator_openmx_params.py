"""The ``queer_params.json`` contract of the OpenMX steps: validation, normalization and PyATB defaults."""

from pathlib import Path

import pytest

from queer.compat.generator.openmx_params import (
    STEP_AHC,
    STEP_CHERN,
    effective_parameters,
    is_openmx_step,
    limit_refusal,
    params_from_pyatb_input,
    staged_scfout,
    translate_openmx,
)
from queer.compat.generator.pyatb_files import read_input
from queer.compat.generator.results import RunnerError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "pyatb"


def ahc_params(**parameters):
    return {
        "step_type": "QUEER_AHC",
        "counterpart_step_type": "PYATB_AHC",
        "input_parameters": {"nspin": 4, "fermi_energy": -2.35, "fermi_energy_unit": "eV"},
        "parameters": {
            "engine": "kubo",
            "method": 0,
            "integrate_mode": "Grid",
            "integrate_grid": [100, 100, 1],
            "adaptive_grid": [1, 1, 1],
            "adaptive_grid_threshold": 1e30,
            **parameters,
        },
        "workers": 4,
        "scfout": "scf.scfout",
    }


def test_a_valid_file_resolves_and_normalizes_text_values():
    params = ahc_params(integrate_grid="100 100 1", method="1")
    params["input_parameters"]["nspin"] = "4"
    resolved = translate_openmx(params)
    assert resolved.step == STEP_AHC and resolved.counterpart == "PYATB_AHC" and resolved.workers == 4
    assert resolved.parameters["integrate_grid"] == [100, 100, 1] and resolved.parameters["method"] == 1
    assert resolved.input_parameters["nspin"] == 4 and resolved.scfout == "scf.scfout"
    assert not resolved.self_convergence and resolved.function == "AHC"
    assert translate_openmx(ahc_params(), "PYATB_AHC").step == STEP_AHC
    assert is_openmx_step("pyatb_chern_number") and not is_openmx_step("QUEER_BANDS")


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"extra": 1}, "unknown keys"),
        ({"counterpart_step_type": "PYATB_SHIFT_CURRENT"}, "counterpart_step_type"),
        ({"input_parameters": {"package": "OPENMX"}}, "fixed by the runner"),
        ({"input_parameters": {"scfout_route": "x.scfout"}}, "fixed by the runner"),
        ({"input_parameters": {"rR_route": "x"}}, "unknown input_parameters"),
        ({"input_parameters": {"nspin": 2}}, "nspin must be 1 or 4"),
        ({"input_parameters": {"fermi_energy": "high"}}, "fermi_energy"),
        ({"input_parameters": {"fermi_energy_unit": "Hartree"}}, "fermi_energy_unit"),
        ({"input_parameters": {"hall_convention": "physical"}}, "hall_convention"),
        ({"parameters": {"occ_band": 3}}, "unknown AHC parameters: occ_band"),
        ({"parameters": {"method": 2}}, "method must be 0"),
        ({"parameters": {"integrate_grid": [10, 10]}}, "integrate_grid must hold 3"),
        ({"parameters": {"adaptive_grid_threshold": -1}}, "nonnegative"),
        ({"parameters": {"integrate_mode": "Cubature"}}, "integrate_mode"),
        ({"workers": 0}, "workers"),
        ({"scfout": "../scf.scfout"}, "relative to the workspace"),
        ({"self_convergence": "yes"}, "self_convergence"),
    ],
)
def test_invalid_files_are_bad_parameters(change, fragment):
    params = ahc_params()
    for key, value in change.items():
        if key in {"input_parameters", "parameters"}:
            params[key] = {**params[key], **value}
        else:
            params[key] = value
    with pytest.raises(RunnerError, match=fragment) as raised:
        translate_openmx(params)
    assert raised.value.code == "queer_bad_params"


def test_step_names_must_agree():
    with pytest.raises(RunnerError, match="differs from the requested"):
        translate_openmx(ahc_params(), "QUEER_CHERN_NUMBER")
    with pytest.raises(RunnerError, match="the OpenMX steps"):
        translate_openmx({**ahc_params(), "step_type": "QUEER_BANDS"})
    with pytest.raises(RunnerError, match="no step"):
        translate_openmx({key: value for key, value in ahc_params().items() if key != "step_type"})


def test_an_nspin_conflict_with_the_scfout_is_a_bad_parameter():
    resolved = translate_openmx(ahc_params())
    with pytest.raises(RunnerError, match="nspin conflicts") as raised:
        resolved.recipe(fermi_energy=0.0, spin_mode="collinear")
    assert raised.value.code == "queer_bad_params"
    assert resolved.recipe(fermi_energy=0.0, spin_mode="soc")["parameters"]["engine"] == "direct"


def test_params_from_the_generator_input_file():
    params = params_from_pyatb_input(read_input(FIXTURES / "h_chern_fermi" / "Input"), workers=2)
    assert params["step_type"] == STEP_CHERN and params["counterpart_step_type"] == "PYATB_CHERN_NUMBER"
    # An Input without hall_convention is PyATB's default auto; the file names it explicitly.
    assert params["input_parameters"] == {
        "nspin": 4,
        "fermi_energy": -2.35,
        "fermi_energy_unit": "eV",
        "hall_convention": "auto",
    }
    assert params["parameters"]["integrate_grid"] == [100, 100, 1] and params["workers"] == 2
    assert translate_openmx(params).step == STEP_CHERN


@pytest.mark.parametrize("key", ["relative_error", "absolute_error"])
def test_a_zero_adaptive_tolerance_is_refused(key):
    adaptive = {"integrate_mode": "Adaptive", "initial_grid": [8, 8, 8], key: 0}
    with pytest.raises(RunnerError, match=f"parameters.{key} = 0") as raised:
        translate_openmx(ahc_params(**adaptive))
    assert raised.value.code == "queer_bad_params"
    # A Grid ignores the adaptive tolerances, so a zero there is harmless.
    assert translate_openmx(ahc_params(**{key: 0})).parameters[key] == 0.0


def test_only_queers_documented_limits_are_limit_refusals():
    zero = {"integrate_mode": "Adaptive", "initial_grid": [8, 8, 8], "absolute_error": 0, "relative_error": 0}
    with pytest.raises(RunnerError) as both:
        translate_openmx(ahc_params(**zero))
    assert limit_refusal(both.value.code, str(both.value))
    # A zero tolerance beside a genuine problem is a bad parameter, not a limit of QUEER's.
    with pytest.raises(RunnerError) as mixed:
        translate_openmx(ahc_params(**{**zero, "initial_grid": "7 7"}))
    assert "initial_grid" in str(mixed.value) and not limit_refusal(mixed.value.code, str(mixed.value))
    budget = "the integration grid 9x9x9 (729 cells) exceeds QUEER's integration budget of 10 cells"
    assert limit_refusal("queer_bad_params", budget)
    assert not limit_refusal("queer_bad_params", "Invalid QUEER parameters: unknown keys for QUEER_AHC: x")
    assert not limit_refusal("queer_internal_error", str(both.value))


def test_the_staged_scfout_is_named_before_the_parameters_are_translated():
    assert staged_scfout({}) == ["scf.scfout"] and staged_scfout({"scfout": "run/a.scfout"}) == [
        "run/a.scfout"
    ]
    for unsafe in ("/abs/scf.scfout", "../scf.scfout", 7):
        assert staged_scfout({"scfout": unsafe}) == []
        with pytest.raises(RunnerError, match="a file name relative to the workspace"):
            translate_openmx({**ahc_params(), "scfout": unsafe})


def test_a_key_less_echo_is_a_pre_key_pyatb_with_its_native_sign():
    # PyATB resolves hall_convention before it writes Out/input.json, so an echo
    # without the key comes from a build older than the key: the native sign.
    echo = {"INPUT_PARAMETERS": {"nspin": 4, "fermi_energy": 0.1}, "AHC": {"engine": "kubo", "method": 0}}
    params = params_from_pyatb_input(echo, echo=True)
    assert params["input_parameters"]["hall_convention"] == "pyatb"
    assert translate_openmx(params).recipe(fermi_energy=0.0)["export_metadata"]["hall_export_sign"] == -1
    assert params_from_pyatb_input(echo)["input_parameters"]["hall_convention"] == "auto"
    explicit = {**echo, "INPUT_PARAMETERS": {**echo["INPUT_PARAMETERS"], "hall_convention": "openmx"}}
    assert params_from_pyatb_input(explicit, echo=True)["input_parameters"]["hall_convention"] == "openmx"
    function = {"engine": "kubo"}
    assert effective_parameters(STEP_AHC, {}, function)["input"]["hall_convention"] == "openmx"
    keyless = effective_parameters(STEP_AHC, {}, function, missing_convention="pyatb")
    assert keyless["input"]["hall_convention"] == "pyatb"


def test_effective_parameters_read_text_values_as_the_runner_does():
    # The Generator's raw parameter JSON may hold vectors and numbers as text.
    text = effective_parameters(
        STEP_AHC,
        {"nspin": "4", "hall_convention": "OpenMX"},
        {
            "integrate_grid": "100 100 1",
            "adaptive_grid": "1, 1, 1",
            "adaptive_grid_threshold": "1e30",
            "method": "1",
        },
    )
    listed = effective_parameters(
        STEP_AHC,
        {"nspin": 4, "hall_convention": "openmx"},
        {
            "integrate_grid": [100, 100, 1],
            "adaptive_grid": [1, 1, 1],
            "adaptive_grid_threshold": 1e30,
            "method": 1,
        },
    )
    assert text == listed
    fhs = {"engine": "fhs", "fhs_energy_window": "-0.5 0.5", "fhs_grid": "100 100 1"}
    assert effective_parameters(STEP_AHC, {}, fhs)["function"]["fhs_energy_window"] == [-0.5, 0.5]
    chern = effective_parameters(STEP_CHERN, {}, {"k_start": "0 0 0", "k_vect1": "1 0 0"})
    assert chern["function"]["k_start"] == [0.0, 0.0, 0.0] and chern["function"]["k_vect1"] == [1.0, 0.0, 0.0]


def test_effective_parameters_apply_pyatb_defaults_and_drop_inapplicable_keys():
    echo_like = effective_parameters(
        STEP_AHC,
        {"nspin": 4, "hall_convention": "auto", "max_kpoint_num": 8000},
        {
            "engine": "kubo",
            "method": 1,
            "integrate_mode": "Grid",
            "integrate_grid": [100, 100, 1],
            "fhs_grid": [100, 100, 1],
            "relative_error": 1e-6,
        },
    )
    assert echo_like == {
        "function": {
            "integrate_mode": "Grid",
            "method": 1,
            "integrate_grid": [100, 100, 1],
            "adaptive_grid": [4, 4, 4],
            "adaptive_grid_threshold": 50.0,
            "engine": "kubo",
        },
        "input": {"nspin": 4, "hall_convention": "openmx"},
    }
    fhs = effective_parameters(STEP_AHC, {"hall_convention": "pyatb"}, {"engine": "fhs", "method": 1})
    assert fhs["function"] == {
        "fhs_grid": [100, 100, 1],
        "fhs_energy_window": [-0.5, 0.5],
        "fhs_energy_points": 101,
        "fhs_dimension": 0,
        "engine": "fhs",
    }
    assert fhs["input"] == {}
    chern = effective_parameters(STEP_CHERN, {}, {"integrate_grid": [10, 10, 7], "integrate_mode": "Grid"})
    assert chern["function"]["integrate_grid"] == [10, 10, 1] and chern["function"]["occ_band"] == -1
    # A Chern number carries no Hall sign: whatever the convention or its absence, it is not compared.
    for inputs, missing in (({"hall_convention": "pyatb"}, "auto"), ({}, "pyatb"), ({}, "auto")):
        resolved = effective_parameters(STEP_CHERN, {"nspin": 4, **inputs}, {}, missing_convention=missing)
        assert resolved["input"] == {"nspin": 4}


def shift_params(**parameters):
    block = {"occ_band": 2, "omega": "0.5 3.0", "domega": "0.01", "grid": "8 8 8", **parameters}
    return {
        "step_type": "QUEER_SHIFT_CURRENT",
        "counterpart_step_type": "PYATB_SHIFT_CURRENT",
        "input_parameters": {"nspin": 4},
        "parameters": {key: value for key, value in block.items() if value is not None},
    }


def test_shift_current_parameters_follow_pyatbs_input():
    from queer.compat.generator.openmx_params import SHIFT_DEFAULTS, STEP_SHIFT
    from queer.compat.generator.pyatb_files import read_input_echo

    parameters = translate_openmx(shift_params(eta="0.05", scissors=0.5))
    assert parameters.step == STEP_SHIFT and parameters.counterpart == "PYATB_SHIFT_CURRENT"
    assert parameters.function == "SHIFT_CURRENT" and is_openmx_step("pyatb_shift_current")
    assert dict(parameters.parameters) == {
        "occ_band": 2,
        "omega": [0.5, 3.0],
        "domega": 0.01,
        "grid": [8, 8, 8],
        "eta": 0.05,
        "scissors": 0.5,
    }
    native = parameters.recipe(fermi_energy=0.0)["parameters"]
    assert native["eta"] == pytest.approx(0.05 / 2**0.5) and native["smearing"] == "gaussian"
    assert len(native["omega"]) == int((3.0 - 0.5) / 0.01) + 1 and native["scissors_occupied_bands"] == 2
    # PyATB's own resolved echo holds exactly these defaults.
    echo = read_input_echo(FIXTURES / "c_direct_openmx" / "Out" / "input.json")["SHIFT_CURRENT"]
    assert {key: value for key, value in echo.items() if value is not None} == dict(SHIFT_DEFAULTS)


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"occ_band": None}, "occ_band is required"),
        ({"omega": None, "domega": None}, "omega is required"),
        ({"occ_band": -1}, "occ_band must count the occupied bands"),
        ({"omega": "3.0 0.5"}, "increasing photon-energy range"),
        ({"domega": 0}, "domega must be positive"),
        ({"eta": -0.1}, "eta must be positive"),
        ({"smearing_method": 4}, "smearing_method must be 0"),
        ({"method": 2}, "method must be 0"),
        ({"grid": "8 8"}, "grid must hold 3 numbers"),
        ({"integrate_grid": "8 8 8"}, "unknown SHIFT_CURRENT parameters: integrate_grid"),
    ],
)
def test_invalid_shift_current_files_are_bad_parameters(change, fragment):
    with pytest.raises(RunnerError) as caught:
        translate_openmx(shift_params(**change))
    assert caught.value.code == "queer_bad_params" and fragment in str(caught.value)
    assert not limit_refusal(caught.value.code, str(caught.value))
    # A key that is present but invalid is reported as invalid, never also as missing.
    for key, value in change.items():
        if value is not None:
            assert f"{key} is required" not in str(caught.value)


def test_a_negative_photon_energy_start_is_taken_as_pyatb_takes_it():
    # PyATB uses start_omega as given; QUEER computes the same photon energies.
    native = translate_openmx(shift_params(omega="-0.5 3.0")).recipe(fermi_energy=0.0)["parameters"]
    assert native["omega"][0] == -0.5 and len(native["omega"]) == int(3.5 / 0.01) + 1


def test_a_negative_scissors_and_a_pair_across_no_gap_are_queers_limits():
    for change in ({"scissors": -0.2}, {"n_occ": 3, "m_unocc": 4}, {"n_occ": 1, "m_unocc": 2}):
        with pytest.raises(RunnerError) as caught:
            translate_openmx(shift_params(**change))
        assert limit_refusal(caught.value.code, str(caught.value)), caught.value
    with pytest.raises(RunnerError) as caught:
        translate_openmx(shift_params(scissors=-0.2, eta=-1))
    assert not limit_refusal(caught.value.code, str(caught.value))


def test_pyatbs_band_pair_rule_and_the_effective_shift_settings():
    from queer.compat.generator.openmx_params import STEP_SHIFT, pyatb_band_pair

    assert pyatb_band_pair({"n_occ": 2, "m_unocc": 3}) == (2, 3)
    for pair in ({"n_occ": 0, "m_unocc": 3}, {"n_occ": 3, "m_unocc": 2}, {"n_occ": 2}, {}):
        assert pyatb_band_pair(pair) == (-1, -1)
    # A pair PyATB does not read as one is every pair: no translation error.
    native = translate_openmx(shift_params(n_occ=3, m_unocc=1)).recipe(fermi_energy=0.0)["parameters"]
    assert "band_pairs" not in native
    native = translate_openmx(shift_params(n_occ=2, m_unocc=3)).recipe(fermi_energy=0.0)["parameters"]
    assert native["band_pairs"] == [[1, 2]]
    resolved = effective_parameters(STEP_SHIFT, {"nspin": "4"}, {**shift_params()["parameters"], "n_occ": 0})
    assert resolved == {
        "function": {
            "occ_band": 2,
            "omega": [0.5, 3.0],
            "domega": 0.01,
            "grid": [8, 8, 8],
            "smearing_method": 1,
            "eta": 0.01,
            "method": 1,
            "n_occ": -1,
            "m_unocc": -1,
            "scissors": 0.0,
        },
        "input": {"nspin": 4},
    }
    without = effective_parameters(
        STEP_SHIFT, {}, {**shift_params()["parameters"], "smearing_method": 0, "eta": 9}
    )
    assert "eta" not in without["function"] and without["input"] == {}
    params = params_from_pyatb_input({"INPUT_PARAMETERS": {"nspin": 4}, "SHIFT_CURRENT": {"occ_band": 2}})
    assert params["step_type"] == STEP_SHIFT and params["counterpart_step_type"] == "PYATB_SHIFT_CURRENT"
