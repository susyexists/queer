"""Native services, explicit legacy translations, and completion-gated exports."""

import json
from pathlib import Path

import numpy as np
import pytest

from queer.cli import main
from queer.compat.generator import band_payload, export_run, translate_ase, translate_pyatb
from queer.config import config_from_mapping
from queer.io.atomistic import read_frames, write_structure
from queer.io.bands import BandData
from queer.io.hamiltonian import pack_openmx, write_graphs
from queer.structures import Structure
from queer.workflows import run_workflow

ROOT = Path(__file__).resolve().parents[2]


def config(tmp_path, name, inputs, parameters=None):
    return config_from_mapping(
        {
            "workflow": name,
            "inputs": {"material_dir": str(tmp_path), **inputs},
            "parameters": parameters or {},
            "output": {"results_dir": str(tmp_path / "runs")},
        }
    )


def test_legacy_parameter_translations_preserve_energy_units_and_component_selection():
    translated = translate_pyatb(
        "PYATB_AHC",
        {
            "engine": "fhs",
            "fhs_dimension": 0,
            "fhs_energy_window": [-1, 1],
            "fhs_energy_points": 4,
            "fermi_energy": 1,
            "fermi_energy_unit": "Ry",
        },
    )
    parameters = translated["parameters"]
    np.testing.assert_allclose(
        parameters["chemical_potentials"], parameters["ef"] + np.array([-1, -0.5, 0, 0.5])
    )
    assert parameters["dimension"] == 3
    assert translated["export_metadata"]["hall_components"] == ["xy"]
    assert parameters["ef"] == pytest.approx(13.60569312299, rel=1e-7)
    shift = translate_pyatb(
        "PYATB_SHIFT_CURRENT",
        {
            "grid": [2, 2, 2],
            "omega": [0, 0.3],
            "domega": 0.1,
            "eta": 0.2,
            "occ_band": 2,
            "n_occ": 1,
            "m_unocc": 3,
            "method": 0,
        },
    )
    assert shift["parameters"]["band_pairs"] == [[0, 2]]
    assert shift["parameters"]["scissors_occupied_bands"] == 2
    assert shift["parameters"]["eta"] == pytest.approx(0.2 / np.sqrt(2))
    assert shift["parameters"]["method"] == "analytic"
    assert len(shift["parameters"]["omega"]) == int(0.3 / 0.1) + 1
    assert translate_ase("ASE_NVT", {"steps": 2, "friction": 0.01})["parameters"][
        "friction_fs_inv"
    ] == pytest.approx(0.0009822694788464063)
    with pytest.raises(ValueError, match="engine"):
        translate_pyatb("AHC", {"engine": "invalid"})


def test_historical_pyatb_engine_and_unrestricted_pair_defaults():
    ahc = translate_pyatb("AHC", {"engine": "pyatb", "method": 0})
    assert ahc["parameters"]["engine"] == "direct"
    assert ahc["export_metadata"]["legacy_engine"] == "pyatb"
    assert "family alias" in ahc["export_metadata"]["engine_translation"]
    parameters = {
        "grid": [4, 4, 4],
        "omega": [0, 8],
        "domega": 0.01,
        "occ_band": 28,
        "n_occ": -1,
        "m_unocc": -1,
        "scissors": 1.15,
    }
    shift = translate_pyatb("SHIFT_CURRENT", parameters)
    assert "band_pairs" not in shift["parameters"]
    assert shift["export_metadata"]["band_pair_selection"] == "all occupied-unoccupied pairs"
    assert "unshifted Hamiltonian" in shift["export_metadata"]["scientific_corrections"][0]
    with pytest.raises(ValueError, match="Band pair"):
        translate_pyatb("SHIFT_CURRENT", {**parameters, "m_unocc": 29})


@pytest.mark.parametrize("xy_only", [False, True])
def test_pyatb_hall_export_matches_recorded_pseudovector_order(tmp_path, xy_only):
    from queer.io.artifacts import RunStore

    tensor = np.array([[[0.0, 3.0, -2.0], [-3.0, 0.0, 1.0], [2.0, -1.0, 0.0]]])
    conventions = {"hall_components": ["xy"]} if xy_only else {}
    store = RunStore(
        tmp_path / "run",
        {
            "workflow": "ahc",
            "inputs": {"export_metadata": conventions},
            "parameters": {},
            "execution": {},
        },
        [],
    )
    try:
        store.commit("result", {"ahc": tensor, "chemical_potentials": [-5.0]})
        store.finish()
    finally:
        store.release()
    output = export_run(store.root) / "Out/AHC/ahc.dat"
    np.testing.assert_array_equal(
        np.loadtxt(output), [-5.0, 0.0, 0.0, 3.0] if xy_only else [-5.0, 1.0, 2.0, 3.0]
    )
    assert "sigma_yz sigma_zx sigma_xy" in output.read_text().splitlines()[0]


def test_pyatb_shift_export_matches_recorded_eighteen_component_columns(tmp_path):
    from queer.io.artifacts import RunStore

    tensor = np.arange(27.0, dtype=float).reshape(1, 3, 3, 3)
    store = RunStore(
        tmp_path / "run",
        {
            "workflow": "shift_current",
            "inputs": {},
            "parameters": {},
            "execution": {},
        },
        [],
    )
    try:
        store.commit("result", {"shift_current": tensor, "omega": [2.0]})
        store.finish()
    finally:
        store.release()
    output = export_run(store.root) / "Out/Shift_Current/shift_current.dat"
    np.testing.assert_array_equal(
        np.loadtxt(output),
        [
            2.0,
            0.0,
            1.0,
            2.0,
            4.0,
            5.0,
            8.0,
            9.0,
            10.0,
            11.0,
            13.0,
            14.0,
            17.0,
            18.0,
            19.0,
            20.0,
            22.0,
            23.0,
            26.0,
        ],
    )
    assert (
        "xxx xxy xxz xyy xyz xzz yxx yxy yxz yyy yyz yzz zxx zxy zxz zyy zyz zzz"
        in output.read_text().splitlines()[0]
    )


def test_imported_band_payload_has_declared_reference_and_segment_partition():
    data = BandData(
        np.array([[[-1, 2], [0, 3]]]),
        fermi_energy=0.5,
        labels=("G", "X"),
        label_indices=(0, 1),
        metadata={"fermi_source": "explicit"},
    )
    payload = band_payload(data)
    np.testing.assert_allclose(payload["energies"], [[[-1.5, 1.5], [-0.5, 2.5]]])
    assert payload["fermi_source"] == "explicit"
    assert payload["segments"] == [(0, 2)]
    assert band_payload(BandData(data.energies))["shifted_to_fermi"] is False


def test_phonons_from_provided_forces_need_no_force_model(tmp_path):
    atom = Structure(np.eye(3) * 5, [[0, 0, 0]], [1])
    write_structure(tmp_path / "POSCAR", atom)
    forces = np.zeros((6, 1, 3))
    for axis in range(3):
        forces[2 * axis, 0, axis] = 0.02
        forces[2 * axis + 1, 0, axis] = -0.02
    np.save(tmp_path / "forces.npy", forces)
    specification = config(
        tmp_path,
        "phonons",
        {"structure": "POSCAR", "forces": "forces.npy"},
        {
            "supercell": [1, 1, 1],
            "delta_A": 0.01,
            "qpoints": [[0, 0, 0], [0.1, 0, 0]],
            "frederiksen": False,
            "acoustic": False,
        },
    )
    run = run_workflow(specification)
    assert run_workflow(specification, run_dir=run, resume=True) == run
    with np.load(run / "arrays/result.npz", allow_pickle=False) as arrays:
        assert arrays["frequencies_THz"].shape == (3, 2)
        assert np.all(arrays["frequencies_THz"] > 0)
    assert (export_run(run) / "phonon_results.json").is_file()


def test_nvt_cli_export_preserves_steps_and_rejects_changed_artifacts(tmp_path):
    atom = Structure(np.eye(3) * 5, [[0, 0, 0]], [1])
    write_structure(tmp_path / "POSCAR", atom)
    specification = config(
        tmp_path,
        "nvt",
        {"structure": "POSCAR"},
        {
            "provider": {"kind": "harmonic"},
            "n_steps": 6,
            "seed": 3,
            "remove_com": False,
            "remove_rotation": False,
            "log_interval": 2,
            "traj_interval": 3,
        },
    )
    run = run_workflow(specification)
    target = tmp_path / "export"
    assert main(["export", str(run), "--profile", "generator", "--output-dir", str(target)]) == 0
    frames = read_frames(target / "trajectory.traj")
    assert [frame.step for frame in frames] == [0, 3, 6]
    summary = json.loads((target / "nvt_results.json").read_text())
    assert [row["step"] for row in summary["series"]] == [0, 2, 4, 6]
    assert summary["queer_provenance"]["calculation_status"] == "completed"
    with (run / "arrays/result.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        export_run(run)


def test_unconverged_relaxation_cannot_export_success(tmp_path):
    write_structure(tmp_path / "POSCAR", Structure(np.eye(3) * 5, [[1, 0, 0]], [1]))
    run = run_workflow(
        config(
            tmp_path,
            "relax",
            {"structure": "POSCAR"},
            {"provider": {"kind": "harmonic", "reference_positions": [[0, 0, 0]]}, "max_steps": 0},
        )
    )
    with pytest.raises(ValueError, match="converged"):
        export_run(run)


def test_graph_bands_filling_and_dataset_splits(tmp_path):
    graph = pack_openmx(
        ROOT / "tests/fixtures/openmx/chain/chain.scfout", atomic_numbers=[1], basis_layout=[[0]], spin="up"
    )
    write_graphs(tmp_path / "graphs.npz", {"sample": graph})
    run = run_workflow(
        config(
            tmp_path,
            "hamiltonian_bands",
            {"graph": "graphs.npz"},
            {"kpoints": [[0, 0, 0], [0.25, 0, 0]], "electron_num": 0.5, "filling_grid": [4, 1, 1]},
        )
    )
    result = json.loads((export_run(run) / "results.json").read_text())
    assert result["band_structure"]["fermi_source"] == "native_electron_filling"
    assert result["band_structure"]["n_bands"] == 1
    text = "fixture\n1\n5 0 0\n0 5 0\n0 0 5\nH\n1\nCartesian\n0 0 0\n"
    items = [
        {
            "id": i,
            "outputs": {
                "poscar": text,
                "energy": float(i),
                "forces": [[0, 0, 0]],
                "stress": np.eye(3).tolist(),
            },
        }
        for i in range(5)
    ]
    (tmp_path / "dataset.json").write_text(json.dumps(items))
    data_run = run_workflow(config(tmp_path, "dataset", {"dataset": "dataset.json"}, {"seed": 3}))
    target = export_run(data_run)
    frames = [frame for stem in ("train", "valid", "test") for frame in read_frames(target / f"{stem}.xyz")]
    assert len(frames) == 5
    assert all("REF_forces" in frame.structure.arrays for frame in frames)
    assert all("REF_stress" in frame.structure.metadata for frame in frames)


def test_output_only_postprocess_declares_stress_and_convergence(tmp_path):
    (tmp_path / "OUTCAR").write_text(
        "NIONS = 1\nfree  energy TOTEN = -2.0 eV\n"
        "energy(sigma->0) = -1.9\nEDIFF is reached\n in kB 1 2 3 4 5 6\n"
        " POSITION TOTAL-FORCE (eV/Angst)\n----------------\n0 0 0 .1 .2 .3\n"
    )
    run = run_workflow(config(tmp_path, "postprocess", {}, {"engine": "vasp"}))
    result = json.loads((export_run(run) / "results.json").read_text())
    assert result["energy_eV"] == -1.9
    assert result["electronic_converged"] is True
    assert result["stress"][0][0] == pytest.approx(1)
    assert result["stress_eV_A3"][0][0] < 0
    assert result["energy"] == -2.0
    assert result["legacy_units"]["stress"] == "kbar"
    assert result["queer_provenance"]["units"]["stress"] == "eV/angstrom^3"


def test_wannier_import_without_fermi_exports_unknown_gap(tmp_path):
    (tmp_path / "wannier90_band.dat").write_text("0 -1\n1 0\n\n0 2\n1 3\n")
    run = run_workflow(config(tmp_path, "postprocess", {}, {"engine": "wannier"}))
    result = json.loads((export_run(run) / "results.json").read_text())
    assert result["bandgap"] is None
    assert result["band_structure"]["fermi_energy"] is None
    assert result["band_structure"]["shifted_to_fermi"] is False
    assert "Fermi" in result["bandgap_unavailable_reason"]
