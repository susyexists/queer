"""QUEER_TOPOLOGY reproduces the Generator PythTB driver's run_topology on analytic wannier90 files.

The Haldane flow is compared loop by loop with the vendored driver's
``wilson_loop`` on PythTB's cell-periodic eigenvectors closed by its boundary
phase; the Kane-Mele layers and their stacked crystal check the Z2 policy and
indices, and the refusals keep the driver's reasons and codes.
"""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose
from tight_binding_models import cell_periodic_operators, haldane, kane_mele
from wannier90_files import Wannier90Job

from queer.compat.generator import topology as topology_step
from queer.compat.generator.results import EXIT_FAILED, EXIT_OK, RESULTS_FILENAME
from queer.compat.generator.runner import run
from queer.models.wannier import WannierModel
from queer.physics.validation import NumericalError

FAST = {"loop_points": 21, "transverse_points": 12}
NOT_GAPPED = (
    "No gapped occupied manifold at E_F (the model is metallic); set occupied_bands to the band count "
    "of an isolated group."
)
PLANE_KEYS = {
    "fixed_axis",
    "value",
    "loop_axis",
    "transverse_axis",
    "chern",
    "chern_raw",
    "max_phase_step",
    "converged",
    "transverse_points",
    "loop_points",
    "file",
}
Z2_KEYS = {"z2", "z2_ambiguous", "z2_crossings"}


def write_model(directory, model, **options):
    job = Wannier90Job(model.translations, model.hopping_matrices, model.lattice, model.centres, **options)
    job.write(directory)
    return job


def run_topology(directory, **settings):
    values = {"step_type": "PYTHTB_TOPOLOGY", "fermi_energy": 0.0, "workers": 1, "ws_correction": "off"}
    values.update(settings)
    params_path = directory / "pythtb_params.json"
    params_path.write_text(json.dumps(values))
    code = run(None, params_path, directory, log=lambda line: None)
    return code, json.loads((directory / RESULTS_FILENAME).read_text())


def driver_flow(core, job, plane_value, loop_points, transverse, occupied):
    """The driver's Berry phases and WCCs of the loops along k1 on k3 = value, from PythTB's eigenvectors."""
    translations, table = job.driver_table()
    model = WannierModel(translations, table, lattice=job.lattice, centres=job.centres)
    reduced = job.centres @ np.linalg.inv(job.lattice)
    boundary = np.exp(-2j * np.pi * reduced[:, 0])
    phases, centres = [], []
    for t in transverse:
        rows = []
        for j in range(loop_points):
            hamiltonian, _ = cell_periodic_operators(model, [j / loop_points, t, plane_value])
            rows.append(np.linalg.eigh(hamiltonian)[1][:, :occupied].T)
        phase, wcc = core.wilson_loop(rows, boundary)
        phases.append(phase)
        centres.append(wcc)
    return np.asarray(phases), np.asarray(centres)


def circular(a, b, period):
    difference = np.mod(np.asarray(a) - np.asarray(b) + period / 2, period) - period / 2
    return np.abs(difference).max()


def test_a_haldane_layer_has_the_drivers_wilson_flow_and_chern_number(generator_core, tmp_path):
    job = write_model(tmp_path, haldane(staggered=0.2))
    code, document = run_topology(tmp_path, **FAST)
    assert code == EXIT_OK, document.get("error")
    topology = document["topology"]
    assert topology["applicable"] is True
    assert topology["occupation"]["n_occ"] == 1 and topology["occupation"]["isolated"] is True
    trims = np.array([[0, 0, 0], [0, 0.5, 0], [0.5, 0, 0], [0.5, 0.5, 0]])
    _, split = generator_core.kramers_degenerate(
        generator_core.eigvals_from_table(*job.driver_table(), trims), 1e-3
    )
    assert topology["time_reversal"] == {
        "spinors": False,
        "kramers_degenerate_at_trims": False,
        "max_kramers_split_meV": pytest.approx(split * 1000.0, abs=1e-9),
        "z2_computed": False,
    }
    assert "z2_indices" not in topology
    (plane,) = topology["planes"]
    assert set(plane) == PLANE_KEYS
    assert [plane[key] for key in ("fixed_axis", "value", "loop_axis", "transverse_axis")] == [2, 0.0, 0, 1]
    assert abs(plane["chern"]) == 1 and plane["chern_raw"] == pytest.approx(plane["chern"], abs=1e-6)
    assert plane["converged"] is True and plane["max_phase_step"] < np.pi / 2
    assert plane["loop_points"] == 21 and plane["transverse_points"] % 12 == 0
    assert plane["file"] == "Out/wcc_k3_0.json"

    data = json.loads((tmp_path / "Out" / "wcc_k3_0.json").read_text())
    assert set(data) == {"t", "berry_phase", "wcc", "gap_midpoints"} and data["gap_midpoints"] is None
    assert_allclose(data["t"], np.arange(plane["transverse_points"]) / plane["transverse_points"])
    phases, centres = driver_flow(generator_core, job, 0.0, 21, data["t"], 1)
    assert circular(data["berry_phase"], phases, 2 * np.pi) < 1e-9
    assert circular(data["wcc"], centres, 1.0) < 1e-9
    chern, step = generator_core.chern_from_phases(phases)
    assert plane["chern_raw"] == pytest.approx(chern, abs=1e-9)
    assert plane["max_phase_step"] == pytest.approx(step, abs=1e-9)

    assert [(item["path"], item["title"], item["kind"]) for item in document["outputs"]] == [
        ("Out/hopping_decay.png", "Hopping decay with distance", "plot"),
        ("Out/wcc_k3_0.json", "Wilson loop, k3 = 0 (data)", "data"),
        ("Out/wcc_k3_0.png", "Wannier charge centres, k3 = 0", "plot"),
    ]
    stages = [plan["stage"] for plan in document["provenance"]["parallel"]]
    assert stages == ["occupation", "Wilson loops, k3=0"]


def test_a_trivial_haldane_layer_has_chern_zero(tmp_path):
    write_model(tmp_path, haldane(staggered=1.0))
    code, document = run_topology(tmp_path, **FAST)
    assert code == EXIT_OK
    (plane,) = document["topology"]["planes"]
    assert plane["chern"] == 0 and abs(plane["chern_raw"]) < 1e-6 and plane["converged"] is True


@pytest.mark.parametrize(("onsite", "mode", "nu"), [(1.0, "auto", 1), (2.5, "auto", 0), (1.0, "off", None)])
def test_kane_mele_z2_follows_the_mode(tmp_path, onsite, mode, nu):
    write_model(tmp_path, kane_mele(onsite), win={"spinors": ".true."})
    code, document = run_topology(tmp_path, z2=mode, **FAST)
    assert code == EXIT_OK, document.get("error")
    topology = document["topology"]
    reversal = topology["time_reversal"]
    assert reversal["spinors"] is True and reversal["kramers_degenerate_at_trims"] is True
    assert reversal["max_kramers_split_meV"] < 1e-6 and reversal["z2_computed"] is (mode != "off")
    (plane,) = topology["planes"]
    assert plane["chern"] == 0 and topology["occupation"]["n_occ"] == 2
    data = json.loads((tmp_path / "Out" / "wcc_k3_0.json").read_text())
    if nu is None:
        assert set(plane) == PLANE_KEYS and "z2_indices" not in topology
        assert data["gap_midpoints"] is None
        return
    assert set(plane) == PLANE_KEYS | Z2_KEYS
    assert plane["z2"] == nu and plane["z2_ambiguous"] is False and plane["z2_crossings"] % 2 == nu
    assert topology["z2_indices"] == {"nu": nu}
    assert len(data["gap_midpoints"]) == plane["transverse_points"] // 2 + 1


def test_z2_forced_on_a_spinless_model_warns_and_reports_the_raw_plane_z2(tmp_path):
    write_model(tmp_path, haldane(staggered=0.2))
    code, document = run_topology(tmp_path, z2="on", **FAST)
    assert code == EXIT_OK
    topology = document["topology"]
    assert topology["time_reversal"]["z2_computed"] is True
    assert document["warnings"][-1] == topology_step.Z2_FORCED_WARNING
    assert document["warning_codes"][-1] == "z2_without_kramers"
    (plane,) = topology["planes"]
    assert topology["z2_indices"] == {"nu": plane["z2"]}


def test_a_stacked_kane_mele_crystal_scans_six_planes_for_weak_indices(tmp_path):
    write_model(
        tmp_path, kane_mele(1.0, interlayer=0.1, height=4.0), win={"spinors": ".true."}, mp_grid=(2, 2, 2)
    )
    code, document = run_topology(tmp_path, loop_points=31, transverse_points=20)
    assert code == EXIT_OK, document.get("error")
    topology = document["topology"]
    planes = topology["planes"]
    assert [(plane["fixed_axis"], plane["value"]) for plane in planes] == [
        (2, 0.0),
        (2, 0.5),
        (0, 0.0),
        (0, 0.5),
        (1, 0.0),
        (1, 0.5),
    ]
    assert [plane["file"] for plane in planes] == [
        f"Out/wcc_{name}.json" for name in ("k3_0", "k3_0p5", "k1_0", "k1_0p5", "k2_0", "k2_0p5")
    ]
    assert [(plane["loop_axis"], plane["transverse_axis"]) for plane in planes] == [
        (0, 1),
        (0, 1),
        (1, 2),
        (1, 2),
        (2, 0),
        (2, 0),
    ]
    assert [plane["z2"] for plane in planes] == [1, 1, 0, 0, 0, 0]
    assert not any(plane["z2_ambiguous"] for plane in planes)
    assert all(plane["chern"] == 0 and plane["converged"] for plane in planes)
    assert topology["z2_indices"] == {"nu1": 0, "nu2": 0, "nu3": 1, "nu0": 0, "consistent": True}

    requested = tmp_path / "requested"
    write_model(
        requested, kane_mele(1.0, interlayer=0.1, height=4.0), win={"spinors": ".true."}, mp_grid=(2, 2, 2)
    )
    code, document = run_topology(
        requested, planes=[{"fixed_axis": 2, "value": 0.5}], loop_points=31, transverse_points=20
    )
    assert code == EXIT_OK
    assert [plane["file"] for plane in document["topology"]["planes"]] == ["Out/wcc_k3_0p5.json"]
    assert document["topology"]["z2_indices"] == {
        "nu1": None,
        "nu2": None,
        "nu3": 1,
        "nu0": None,
        "consistent": True,
    }


def test_a_set_touching_the_next_band_is_not_applicable(tmp_path):
    write_model(tmp_path, haldane(staggered=0.2))
    code, document = run_topology(tmp_path, isolation_threshold_meV=1e6, **FAST)
    assert code == EXIT_OK and document["status"] == "ok"
    topology = document["topology"]
    isolation = topology["occupation"]["isolation_meV"]
    assert topology["occupation"]["isolated"] is False
    assert topology["applicable"] is False and topology["planes"] == []
    assert topology["reason"] == (
        f"Bands 1 and 2 touch somewhere in the Brillouin zone (smallest direct gap {isolation:.2f} meV): "
        "invariants of a non-isolated band set are undefined."
    )
    assert "time_reversal" not in topology and not list((tmp_path / "Out").glob("wcc_*"))


def test_a_metal_has_no_topology(tmp_path):
    translations = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
    table = np.array([[[0.0]], [[-1.0]], [[-1.0]], [[-1.0]], [[-1.0]]], complex)
    Wannier90Job(translations, table, np.diag([2.0, 2.0, 10.0]), np.array([[1.0, 1.0, 5.0]])).write(tmp_path)
    code, document = run_topology(tmp_path, fermi_energy=-1.0, **FAST)
    assert code == EXIT_OK
    topology = document["topology"]
    assert topology["applicable"] is False and topology["reason"] == NOT_GAPPED
    assert topology["occupation"]["metal"] is True and topology["planes"] == []


@pytest.mark.parametrize(
    "refusal",
    [
        ValueError("singular Wilson overlap; refine the path"),
        ValueError("occupied/unoccupied boundary splits a degenerate subspace"),
        NumericalError("Hermitian eigensolver failed for k-points starting at [0.0, 0.0, 0.0]"),
    ],
)
def test_a_refused_wilson_loop_is_a_numerical_error(tmp_path, monkeypatch, refusal):
    def refuse(*args, **kwargs):
        raise refusal

    monkeypatch.setattr(topology_step, "wilson_plane", refuse)
    write_model(tmp_path, haldane(staggered=0.2))
    code, document = run_topology(tmp_path, **FAST)
    assert code == EXIT_FAILED and document["status"] == "failed"
    assert document["error"]["code"] == "queer_numerical_error"
    assert str(refusal) in document["error"]["message"]
    assert document["topology"]["applicable"] is True and document["topology"]["planes"] == []
