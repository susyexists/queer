"""The ``queer_params.json`` contract of the OpenMX steps: validation, normalization and PyATB defaults."""

from pathlib import Path

import pytest

from queer.compat.generator.openmx_params import (
    STEP_AHC,
    STEP_CHERN,
    effective_parameters,
    is_openmx_step,
    params_from_pyatb_input,
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
    assert params["input_parameters"] == {"nspin": 4, "fermi_energy": -2.35, "fermi_energy_unit": "eV"}
    assert params["parameters"]["integrate_grid"] == [100, 100, 1] and params["workers"] == 2
    assert translate_openmx(params).step == STEP_CHERN


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
