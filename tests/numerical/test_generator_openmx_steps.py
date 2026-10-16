"""QUEER's OpenMX steps on analytic scfout files.

The Qi-Wu-Zhang model written as one spinor orbital (``openmx_files``) is a Chern
insulator for 0 < |m| < 2t with E_F = 0 in its gap: its Hall conductivity sits
near -(e^2/h) C / c and its FHS plaquette sum is exactly -(e^2/h) C / c. The runner
must reproduce QUEER's own observables on the same nodes, write PyATB's layouts in
the requested sign, record the flagged k-points and fail with stable codes.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from openmx_files import collinear_scfout, qwz_spinor_scfout

from queer import ExecutionOptions, OpenMXModel
from queer.compat.generator.openmx_steps import hall_rows, single_channel_model
from queer.compat.generator.pyatb_conventions import HARTREE_EV, PYATB_HARTREE_EV
from queer.compat.generator.pyatb_files import (
    ahc_scan_text,
    ahc_vector_text,
    chern_text,
    parse_ahc,
)
from queer.compat.generator.results import EXIT_FAILED, EXIT_OK
from queer.compat.generator.runner import capabilities, run
from queer.observables.geometry import chern_number
from queer.observables.integrated import ahc
from queer.physics.constants import E2_H

ROOT = Path(__file__).resolve().parents[2]
C_ANGSTROM = 10.0
PLATEAU = E2_H / C_ANGSTROM * 1e8


def params(step="QUEER_AHC", inputs=None, **function):
    blocks = {
        "QUEER_AHC": {
            "engine": "kubo",
            "method": 0,
            "integrate_mode": "Grid",
            "integrate_grid": [12, 12, 1],
            "adaptive_grid": [1, 1, 1],
            "adaptive_grid_threshold": 1e30,
        },
        "QUEER_CHERN_NUMBER": {
            "method": 0,
            "occ_band": -1,
            "integrate_mode": "Grid",
            "integrate_grid": [12, 12, 1],
            "adaptive_grid": [1, 1, 1],
            "adaptive_grid_threshold": 1e30,
        },
    }
    return {
        "step_type": step,
        "counterpart_step_type": step.replace("QUEER_", "PYATB_"),
        "input_parameters": {"nspin": 4, "fermi_energy": "Auto", **(inputs or {})},
        "parameters": {**blocks[step], **function},
        "workers": 1,
        "scfout": "scf.scfout",
    }


def stage(directory, payload, scfout=qwz_spinor_scfout, **model):
    directory.mkdir(parents=True, exist_ok=True)
    if scfout is not None:
        scfout(directory / "scf.scfout", **model)
    (directory / "queer_params.json").write_text(json.dumps(payload))
    return directory / "queer_params.json"


def execute(directory, payload, scfout=qwz_spinor_scfout, **model):
    code = run(None, stage(directory, payload, scfout, **model), directory, log=lambda line: None)
    return code, json.loads((directory / "queer_results.json").read_text())


def test_capabilities_list_the_openmx_steps():
    steps = capabilities()["steps"]
    assert {"QUEER_AHC", "QUEER_CHERN_NUMBER"} <= set(steps) and steps == sorted(steps)


def test_the_grid_ahc_is_queers_own_sum_and_pyatbs_layout(tmp_path):
    code, document = execute(tmp_path, params(inputs={"hall_convention": "openmx"}))
    assert code == EXIT_OK and document["status"] == "ok", document.get("error")
    assert document["step_type"] == "QUEER_AHC" and document["counterpart_step_type"] == "PYATB_AHC"
    model = OpenMXModel.from_scfout(tmp_path / "scf.scfout")
    native = ahc(
        model, grid=(12, 12, 1), chemical_potentials=[0.0], engine="direct", execution=ExecutionOptions()
    )
    tensor = np.asarray(native["ahc"])[0]
    sigma = document["ahc"]["sigma_S_cm"]
    assert sigma["xy"] == [tensor[0, 1]] and sigma["yz"] == [tensor[1, 2]] and sigma["zx"] == [tensor[2, 0]]
    assert abs(sigma["xy"][0]) == pytest.approx(PLATEAU, rel=0.05)
    assert document["ahc"]["convention"] == {
        "requested": "openmx",
        "resolved": "openmx",
        "sign_table": 1,
        "lattice_handedness": 1,
        "export_sign": 1,
        "physical": "sigma_ab = -(e^2/hbar) integral(f Omega_ab), carrier charge -e",
        "file": "Out/AHC/ahc.dat holds export_sign times sigma_S_cm",
    }
    written = (tmp_path / "Out/AHC/ahc.dat").read_text()
    assert written == ahc_vector_text([sigma["yz"][0], sigma["zx"][0], sigma["xy"][0]])
    model_block = document["model"]
    assert model_block["fermi_source"] == "auto" and model_block["fermi_energy"] == 0.0
    assert model_block["spin_mode"] == "soc" and model_block["nspin"] == 4 and model_block["n_basis"] == 2
    inputs = document["provenance"]["inputs"]
    assert set(inputs) == {"scf.scfout", "queer_params.json"}
    assert inputs["scf.scfout"]["sha256"] == model_block["scfout_sha256"]
    assert document["integration"]["refinement"] == "off" and document["integration"]["points"] == 144
    assert document["ahc"]["flagged"]["count"] == 0 and document["warning_codes"] == []
    assert document["ahc"]["legacy_single_channel"] is None
    assert not (tmp_path / "pyatb_results.json").exists() and not (tmp_path / "manifest.json").exists()
    assert document["outputs"] == [
        {"path": "Out/AHC/ahc.dat", "title": "Anomalous Hall conductivity (PyATB layout)", "kind": "data"}
    ]


def test_the_pyatb_convention_writes_the_negative_and_an_explicit_fermi_energy_is_rescaled(tmp_path):
    code, document = execute(tmp_path, params(inputs={"hall_convention": "pyatb", "fermi_energy": 0.25}))
    assert code == EXIT_OK
    assert document["model"]["fermi_energy"] == pytest.approx(0.25 * HARTREE_EV / PYATB_HARTREE_EV, rel=1e-15)
    assert (
        document["model"]["fermi_source"] == "input_eV" and document["model"]["fermi_energy_pyatb_eV"] == 0.25
    )
    sigma = document["ahc"]["sigma_S_cm"]["xy"][0]
    assert document["ahc"]["convention"]["export_sign"] == -1
    assert parse_ahc((tmp_path / "Out/AHC/ahc.dat").read_text())["sigma"][2] == pytest.approx(
        -sigma, abs=5e-7
    )


def test_execution_only_settings_never_change_the_result(tmp_path):
    code, plain = execute(tmp_path / "plain", params())
    assert code == EXIT_OK
    tuned = params(inputs={"max_kpoint_num": 7, "sparse_format": 1})
    tuned["workers"] = 2
    code, other = execute(tmp_path / "tuned", tuned)
    assert code == EXIT_OK
    assert other["ahc"]["sigma_S_cm"] == plain["ahc"]["sigma_S_cm"]
    assert other["integration"]["max_kpoint_num"] == 7 and other["integration"]["sparse_format"] == 1
    plans = other["provenance"]["parallel"]
    assert plans and all(plan["chunk"] <= 7 for plan in plans)


def test_a_level_at_the_fermi_energy_flags_its_point_with_its_contribution(tmp_path):
    model = OpenMXModel.from_scfout(qwz_spinor_scfout(tmp_path / "probe.scfout", mass=1.0))
    level = float(model.energies(np.zeros((1, 3)))[0, 0])
    pyatb_value = level * PYATB_HARTREE_EV / HARTREE_EV
    code, document = execute(tmp_path / "run", params(inputs={"fermi_energy": pyatb_value}))
    assert code == EXIT_OK
    flagged = document["ahc"]["flagged"]
    assert flagged["count"] >= 1 and flagged["points"] == 144
    listed = document["ahc"]["per_k_flagged"]
    assert [0.0, 0.0, 0.0] in [entry["k"] for entry in listed]
    for axis in range(3):
        assert flagged["sum_abs_S_cm"][axis] == pytest.approx(sum(abs(entry["c"][axis]) for entry in listed))
    assert "flagged_kpoints" in document["warning_codes"]
    point = [entry for entry in listed if entry["k"] == [0.0, 0.0, 0.0]][0]
    assert point["closest_level_eV"] < flagged["delta_ef_eV"]


def test_the_chern_number_is_queers_own_and_a_metal_gives_the_fermi_sea_flux(tmp_path):
    code, document = execute(tmp_path / "gap", params("QUEER_CHERN_NUMBER"))
    assert code == EXIT_OK, document.get("error")
    model = OpenMXModel.from_scfout(tmp_path / "gap" / "scf.scfout")
    native = float(chern_number(model, grid=(12, 12, 1), fermi_energy=0.0, engine="direct")["chern_number"])
    chern = document["chern"]
    assert chern["value"] == pytest.approx(native, abs=1e-12) and chern["gapped"]
    assert chern["integer"] == round(native) and abs(chern["integer"]) == 1
    assert chern["occupied_count"] == 1 and chern["occupied_bands"] == "fermi" and chern["method"] == "direct"
    assert (tmp_path / "gap/Out/Chern_Num/chern_number.dat").read_text() == chern_text(chern["value"], "Grid")
    assert document["warning_codes"] == []

    code, document = execute(tmp_path / "metal", params("QUEER_CHERN_NUMBER", inputs={"fermi_energy": -2.0}))
    assert code == EXIT_OK
    chern = document["chern"]
    assert not chern["gapped"] and chern["integer"] is None and chern["occupied_count"] is None
    assert "chern_plane_metallic" in document["warning_codes"]
    assert 0.0 < abs(chern["value"]) < 1.0


def test_a_band_count_occupation_ignores_the_fermi_energy(tmp_path):
    code, document = execute(tmp_path, params("QUEER_CHERN_NUMBER", occ_band=1, method=1))
    assert code == EXIT_OK
    assert document["chern"]["occupied_bands"] == 1 and document["chern"]["method"] == "kubo"
    assert abs(document["chern"]["integer"]) == 1


def test_a_refining_grid_records_its_self_convergence_or_warns(tmp_path):
    refining = {"integrate_grid": [6, 6, 1], "adaptive_grid": [2, 2, 1], "adaptive_grid_threshold": 0.0}
    code, document = execute(tmp_path / "plain", params(**refining))
    assert code == EXIT_OK
    integration = document["integration"]
    assert integration["refinement"] == "threshold" and integration["threshold_A2"] == 0.0
    assert (
        integration["node_placement"] == "cell_midpoints" and integration["self_convergence"]["e_Q"] is None
    )
    assert "self_convergence_skipped" in document["warning_codes"]
    payload = {**params(**refining), "self_convergence": True}
    code, document = execute(tmp_path / "converged", payload)
    assert code == EXIT_OK
    convergence = document["integration"]["self_convergence"]
    assert convergence["grid"] == [12, 12, 1] and len(convergence["e_Q"]) == 3
    finer = convergence["sigma_S_cm"]["xy"][0]
    assert convergence["e_Q"][2] == pytest.approx(abs(document["ahc"]["sigma_S_cm"]["xy"][0] - finer))
    assert "self_convergence_skipped" not in document["warning_codes"]


def test_an_adaptive_integral_from_few_cells_warns(tmp_path):
    adaptive = {"integrate_mode": "Adaptive", "initial_grid": [4, 4, 1], "absolute_error": 50.0}
    code, document = execute(tmp_path, params(**adaptive))
    assert code == EXIT_OK, document.get("error")
    assert document["integration"]["refinement"] == "cubature"
    assert document["integration"]["error_estimate"] is not None
    assert "adaptive_initial_grid_coarse" in document["warning_codes"]


def test_the_fhs_scan_gives_the_quantized_plateau_on_pyatbs_energies(tmp_path):
    fhs = {"engine": "fhs", "fhs_grid": [12, 12, 1], "fhs_energy_window": [-0.2, 0.2], "fhs_energy_points": 4}
    code, document = execute(tmp_path, params(inputs={"hall_convention": "pyatb"}, **fhs))
    assert code == EXIT_OK, document.get("error")
    block = document["ahc"]
    assert block["engine"] == "fhs" and block["convention"]["export_sign"] == 1
    assert block["energy_offsets_eV"] == pytest.approx([-0.2, -0.1, 0.0, 0.1], abs=1e-15)
    assert block["components_computed"] == ["xy"]
    xy = np.asarray(block["sigma_S_cm"]["xy"])
    assert np.allclose(np.abs(xy), PLATEAU, rtol=1e-12) and np.all(xy == xy[0])
    assert block["sigma_S_cm"]["yz"] == [0.0] * 4 and block["at_fermi"]["xy"] == xy[2]
    rows = np.column_stack([block["sigma_S_cm"][name] for name in ("yz", "zx", "xy")])
    assert (tmp_path / "Out/AHC/ahc.dat").read_text() == ahc_scan_text(block["energy_offsets_eV"], rows)


def test_a_collinear_scfout_reports_pyatbs_single_channel_beside_the_physical_value(tmp_path):
    payload = params(inputs={"nspin": 1, "fermi_energy": 0.0})
    code, document = execute(tmp_path / "ahc", payload, scfout=collinear_scfout)
    assert code == EXIT_OK, document.get("error")
    assert document["model"]["spin_mode"] == "collinear" and document["model"]["n_basis"] == 4
    legacy = document["ahc"]["legacy_single_channel"]
    assert legacy["channel"] == 0 and "spin-up" in legacy["reason"]
    assert set(legacy["sigma_S_cm"]) == {"yz", "zx", "xy"} and legacy["flagged"]["points"] == 144
    assert "pyatb_single_channel" in document["warning_codes"]
    payload = params("QUEER_CHERN_NUMBER", inputs={"nspin": 1, "fermi_energy": 0.0})
    code, document = execute(tmp_path / "chern", payload, scfout=collinear_scfout)
    assert code == EXIT_OK
    assert document["chern"]["legacy_single_channel"]["value"] == pytest.approx(0.0, abs=1e-12)


def _qwz_blocks(mass):
    """Two-orbital QWZ hoppings (eV) on a square lattice, keyed by translation."""
    sx, sy, sz = (np.array(m, complex) for m in ([[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]))
    return {
        (0, 0, 0): mass * sz,
        (1, 0, 0): (sz - 1j * sx) / 2,
        (-1, 0, 0): (sz + 1j * sx) / 2,
        (0, 1, 0): (sz - 1j * sy) / 2,
        (0, -1, 0): (sz + 1j * sy) / 2,
    }


def test_the_single_channel_model_is_the_spin_up_hamiltonian_occupied_once():
    up, down = _qwz_blocks(1.0), _qwz_blocks(-1.5)
    keys = list(up)
    zero = np.zeros((2, 2), complex)
    hopping = {key: np.block([[up[key], zero], [zero, down[key]]]) for key in keys}
    lattice = np.diag([2.0, 2.0, C_ANGSTROM])
    model = OpenMXModel(
        np.array(keys), hopping, {(0, 0, 0): np.eye(4)}, lattice=lattice, spin_mode="collinear"
    )
    channel = single_channel_model(model)
    assert channel.spin_mode == "nonmagnetic" and channel.spin_degeneracy == 1 and channel.nbnd == 2

    def alone(blocks):
        return OpenMXModel(np.array(keys), blocks, {(0, 0, 0): np.eye(2)}, lattice=lattice, spin_degeneracy=1)

    points = np.array([[0.1, 0.2, 0.0], [0.4, 0.05, 0.0], [0.25, 0.5, 0.0]])
    rows = {
        name: hall_rows(m, 0.0, "kubo", None, points)
        for name, m in (("channel", channel), ("up", alone(up)), ("down", alone(down)), ("both", model))
    }
    assert np.array_equal(rows["channel"][:, :9], rows["up"][:, :9])
    assert np.allclose(rows["both"][:, :9], rows["up"][:, :9] + rows["down"][:, :9], atol=1e-12)
    assert not np.allclose(rows["both"][:, 1], rows["up"][:, 1])
    with pytest.raises(ValueError, match="spin-orbit"):
        soc = OpenMXModel(np.array(keys), hopping, {(0, 0, 0): np.eye(4)}, lattice=lattice, spin_mode="soc")
        single_channel_model(soc)


@pytest.mark.parametrize(
    ("payload_change", "scfout", "code", "fragment"),
    [
        ({}, None, "queer_missing_input", "was not staged"),
        ({}, "garbage", "queer_missing_input", "cannot be read as an OpenMX v3 scfout"),
        ({"input_parameters": {"nspin": 1}}, "qwz", "queer_bad_params", "nspin conflicts"),
        ({"parameters": {"method": 5}}, "qwz", "queer_bad_params", "method must be 0"),
        ({"step_type": "QUEER_SHIFT"}, "qwz", "queer_bad_params", "unknown step_type"),
    ],
)
def test_failures_write_a_coded_document_and_exit_two(tmp_path, payload_change, scfout, code, fragment):
    payload = params()
    for key, value in payload_change.items():
        payload[key] = {**payload[key], **value} if isinstance(value, dict) else value
    tmp_path.mkdir(exist_ok=True)
    if scfout == "qwz":
        qwz_spinor_scfout(tmp_path / "scf.scfout")
    elif scfout == "garbage":
        (tmp_path / "scf.scfout").write_bytes(b"\x01\x02 not a scfout")
    (tmp_path / "queer_params.json").write_text(json.dumps(payload))
    exit_code = run(
        "QUEER_AHC" if "step_type" not in payload_change else None,
        tmp_path / "queer_params.json",
        tmp_path,
        log=lambda line: None,
    )
    document = json.loads((tmp_path / "queer_results.json").read_text())
    assert exit_code == EXIT_FAILED and document["status"] == "failed"
    assert document["error"]["code"] == code and fragment in document["error"]["message"]
    assert document["engine"] == "queer" and "provenance" in document and "outputs" in document


def test_the_module_entry_runs_an_openmx_step(tmp_path):
    stage(tmp_path, params(integrate_grid=[4, 4, 1]))
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT / "src"), MPLBACKEND="Agg", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    finished = subprocess.run(
        [
            sys.executable,
            "-m",
            "queer",
            "generator",
            "run",
            "--step",
            "PYATB_AHC",
            "--params",
            "queer_params.json",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert finished.returncode == 0, finished.stderr
    document = json.loads((tmp_path / "queer_results.json").read_text())
    assert document["status"] == "ok" and document["step_type"] == "QUEER_AHC"
    assert (tmp_path / "Out/AHC/ahc.dat").is_file() and (tmp_path / "queer.log").is_file()
