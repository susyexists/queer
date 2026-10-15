"""QUEER Wannier step parameters keep the Generator PythTB driver's defaults, keys and refusals."""

from types import MappingProxyType

import pytest

from queer.compat.generator.pythtb_params import (
    CONTEXT_PARAM_KEYS,
    COUNTERPARTS,
    DEFAULT_PARAMS,
    SHARED_PARAM_KEYS,
    STEP_PARAM_KEYS,
    STEP_TYPES,
    canonical_step,
    step_keys,
    translate_pythtb,
)
from queer.compat.generator.results import ERR_BAD_PARAMS, RunnerError

# The settings of a Generator PYTHTB_BANDS parameter file (job-e228d1af), structure shortened.
GENERATOR_BANDS = {
    "chunk_size": None,
    "dos_mesh": None,
    "dos_sigma_eV": 0.03,
    "dos_step_eV": 0.005,
    "energy_window_eV": [-3.0, 3.0],
    "fermi_energy": 4.70845361,
    "fermi_source": "reference_band",
    "fermi_surface": True,
    "fermi_surface_mesh": None,
    "fermi_tolerance_eV": 0.02,
    "kpath_points": 400,
    "max_distance": None,
    "min_hopping_norm": None,
    "pdos": True,
    "periodic_dirs": None,
    "reference_band_job_id": "job-9a4971ae",
    "source_wannier_job_id": "job-48ec1fe7",
    "spinors": True,
    "step_type": "PYTHTB_BANDS",
    "structure": {"cart": [[2.1545, 1.243901, 13.5595]], "species": ["Mn"]},
    "validation_fail_meV": 100.0,
    "validation_warn_meV": 5.0,
    "workers": None,
    "ws_correction": "auto",
    "ws_hint": True,
}


def bad_params(params, step=None):
    with pytest.raises(RunnerError) as caught:
        translate_pythtb(params, step)
    assert caught.value.code == ERR_BAD_PARAMS
    return str(caught.value)


def test_defaults_and_key_sets_are_the_drivers(generator_core):
    core = generator_core
    for key, value in DEFAULT_PARAMS.items():
        assert (list(value) if isinstance(value, tuple) else value) == core.DEFAULT_PARAMS[key], key
    assert SHARED_PARAM_KEYS == core.SHARED_PARAM_KEYS
    assert CONTEXT_PARAM_KEYS == core.CONTEXT_PARAM_KEYS
    for step, counterpart in COUNTERPARTS.items():
        assert STEP_PARAM_KEYS[step] == core.STEP_PARAM_KEYS[counterpart]
        assert (
            step_keys(step) == core.SHARED_PARAM_KEYS + core.STEP_PARAM_KEYS[counterpart] + CONTEXT_PARAM_KEYS
        )
    driver_keys = set(core.SHARED_PARAM_KEYS) | set(core.CONTEXT_PARAM_KEYS)
    for counterpart in COUNTERPARTS.values():
        driver_keys |= set(core.STEP_PARAM_KEYS[counterpart])
    assert set(DEFAULT_PARAMS) == driver_keys
    assert set(COUNTERPARTS.values()) == set(core.STEP_TYPES) - {core.STEP_QGEOM}


def test_a_generator_bands_file_resolves_with_the_driver_defaults():
    resolved = translate_pythtb(GENERATOR_BANDS)
    assert (resolved.step, resolved.counterpart) == ("QUEER_BANDS", "PYTHTB_BANDS")
    assert isinstance(resolved.values, MappingProxyType)
    assert resolved["step_type"] == "QUEER_BANDS"
    assert resolved["fermi_energy"] == 4.70845361 and resolved["spinors"] is True
    assert resolved["ws_hint"] is True and resolved["ws_correction"] == "auto"
    assert resolved["structure"] == GENERATOR_BANDS["structure"]
    # A null keeps the driver default, here "automatic".
    assert resolved["dos_mesh"] is None and resolved["workers"] is None and resolved["periodic_dirs"] is None
    # Keys of other steps keep their defaults without being accepted from the file.
    assert resolved["loop_points"] == 81 and resolved["slab_kpoints"] == 200
    assert resolved["energy_window_eV"] == [-3.0, 3.0]
    assert resolved.get("kpath", {"segments": []}) == {"segments": []}


@pytest.mark.parametrize(
    ("name", "step"),
    [
        ("PYTHTB_BANDS", "QUEER_BANDS"),
        ("pythtb_topology", "QUEER_TOPOLOGY"),
        (" Queer_Berry ", "QUEER_BERRY"),
        ("QUEER_SLAB", "QUEER_SLAB"),
    ],
)
def test_step_names_and_their_pythtb_aliases(name, step):
    assert canonical_step(name) == step
    assert translate_pythtb({"step_type": name}).step == step
    assert translate_pythtb({}, name).step == step
    assert step in STEP_TYPES


@pytest.mark.parametrize("name", ["PYTHTB_QGEOM", "QUEER_QGEOM", "WANNIER", "", None])
def test_other_steps_are_refused(name):
    with pytest.raises(RunnerError) as caught:
        canonical_step(name)
    assert caught.value.code == ERR_BAD_PARAMS
    message = bad_params({"step_type": name}) if name is not None else bad_params({})
    assert "step" in message


def test_a_requested_step_must_agree_with_the_file():
    assert "differs from the requested" in bad_params({"step_type": "PYTHTB_BANDS"}, "QUEER_SLAB")
    assert translate_pythtb({"step_type": "PYTHTB_SLAB"}, "queer_slab").step == "QUEER_SLAB"


def test_unknown_and_foreign_keys_are_refused_together():
    params = {
        "step_type": "PYTHTB_SLAB",
        "spin_mode": "noncollinear",
        "qgeom_tolerance": 1e-3,
        "kpath_points": 400,
        "validation_warn_meV": 5.0,
    }
    message = bad_params(params)
    assert "unknown parameters for QUEER_SLAB: kpath_points, qgeom_tolerance, spin_mode" in message
    assert "validation_warn_meV" not in message


def test_values_outside_the_catalog_kinds_and_bounds_are_named():
    params = {
        "step_type": "PYTHTB_BANDS",
        "kpath_points": 1,
        "ws_correction": "maybe",
        "energy_window_eV": [1.0, -1.0],
        "workers": 0,
        "pdos": "perhaps",
        "periodic_dirs": [0, 0],
        "max_distance": 0.0,
        "fermi_energy": float("nan"),
        "dos_mesh": 2.5,
        "chunk_size": True,
    }
    message = bad_params(params)
    for key in params:
        if key != "step_type":
            assert f"{key} = " in message, key


def test_catalog_values_are_coerced_to_their_kinds():
    resolved = translate_pythtb(
        {
            "step_type": "PYTHTB_BERRY",
            "fermi_energy": "4.25",
            "berry_mesh": 120.0,
            "quantum_metric": "false",
            "occupation": "Bands",
            "energy_window_eV": "-1, 2",
            "ws_correction": "FILE",
            "periodic_dirs": [1, 0],
            "berry_plane": {"fixed_axis": 2, "value": 0.5},
            "occupied_bands": 4,
            "chunk_size": 16,
        }
    )
    assert resolved["fermi_energy"] == 4.25 and resolved["berry_mesh"] == 120
    assert isinstance(resolved["berry_mesh"], int)
    assert resolved["quantum_metric"] is False and resolved["occupation"] == "bands"
    assert resolved["energy_window_eV"] == [-1.0, 2.0] and resolved["ws_correction"] == "file"
    assert resolved["periodic_dirs"] == [0, 1]
    assert resolved["berry_plane"] == {"fixed_axis": 2, "value": 0.5}
    assert resolved["occupied_bands"] == 4 and resolved["chunk_size"] == 16
    slab = translate_pythtb({"step_type": "PYTHTB_SLAB", "slab_axis": "1"})
    assert slab["slab_axis"] == 1


def test_an_empty_periodic_list_means_automatic_like_the_driver():
    assert translate_pythtb({"step_type": "PYTHTB_BANDS", "periodic_dirs": []})["periodic_dirs"] is None


def test_driver_value_falls_back_on_zero_like_the_driver():
    resolved = translate_pythtb(
        {"step_type": "PYTHTB_TOPOLOGY", "kramers_tol_meV": 0, "max_refinements": 0, "fermi_tolerance_eV": 0}
    )
    assert resolved.get("kramers_tol_meV", 1.0) == 0
    assert resolved.driver_value("kramers_tol_meV", 1.0) == 1.0
    assert resolved.driver_value("max_refinements", 0) == 0
    assert resolved.driver_value("fermi_tolerance_eV", 0.02) == 0.02


def test_planes_and_paths_keep_the_shapes_the_steps_read():
    resolved = translate_pythtb(
        {
            "step_type": "PYTHTB_TOPOLOGY",
            "planes": [{"fixed_axis": 2, "value": 0.5}, [0, 0]],
            "kpath": {"segments": [["G", [0, 0, 0], "M", [0.5, 0, 0]]], "source": "seekpath"},
        }
    )
    assert resolved["planes"] == [{"fixed_axis": 2, "value": 0.5}, {"fixed_axis": 0, "value": 0.0}]
    assert resolved["kpath"] == {
        "segments": [["G", [0.0, 0.0, 0.0], "M", [0.5, 0.0, 0.0]]],
        "source": "seekpath",
    }
    message = bad_params({"step_type": "PYTHTB_TOPOLOGY", "planes": [{"fixed_axis": 3}], "kpath": [1]})
    assert "planes = " in message and "kpath = " in message


def test_a_parameter_file_must_be_an_object():
    with pytest.raises(RunnerError) as caught:
        translate_pythtb(["PYTHTB_BANDS"])  # type: ignore[arg-type]
    assert caught.value.code == ERR_BAD_PARAMS
