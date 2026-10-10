"""Durable native simulation state and graph-filling workflow contracts."""

import json

import numpy as np
import pytest

from queer.io.atomistic import write_poscar
from queer.io.hamiltonian import HamiltonianGraph, write_graphs
from queer.io.openmx import HARTREE_EV
from queer.structures import Evaluation, Structure
from queer.workflows import run_workflow


class HarmonicProvider:
    def __init__(self, fail_at=None, failure=RuntimeError):
        self.calls = 0
        self.fail_at = fail_at
        self.failure = failure

    def evaluate(self, structure):
        self.calls += 1
        if self.calls == self.fail_at:
            raise self.failure("transient producer interruption")
        return Evaluation(0.25 * float(np.sum(structure.positions**2)), -0.5 * structure.positions)


def artifact(run):
    manifest = json.loads((run / "manifest.json").read_text())
    record = manifest["cases"]["result"]
    with np.load(run / record["path"], allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return manifest, arrays


@pytest.mark.parametrize("workflow", ["nvt", "relax"])
@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_force_workflow_resumes_only_missing_steps_after_transient_failure(tmp_path, workflow, failure):
    material = tmp_path / "material"
    material.mkdir()
    structure = Structure(np.eye(3) * 10, [[0.5, 0.2, 0.1]], [1])
    (material / "POSCAR").write_text(write_poscar(structure))
    parameters = (
        {
            "n_steps": 37,
            "seed": 25,
            "log_interval": 7,
            "traj_interval": 11,
            "temperature_K": 50,
            "remove_rotation": False,
        }
        if workflow == "nvt"
        else {"max_steps": 400, "fmax_eV_A": 1e-7}
    )
    config = {
        "workflow": workflow,
        "inputs": {"material_dir": material, "structure": "POSCAR"},
        "parameters": parameters,
        "output": {"results_dir": tmp_path / "results"},
    }
    interrupted_run = tmp_path / "interrupted"
    interrupted_provider = HarmonicProvider(fail_at=15, failure=failure)
    with pytest.raises(failure):
        run_workflow(
            config,
            run_dir=interrupted_run,
            force_provider=interrupted_provider,
            force_provider_id="analytic-harmonic-v1",
        )
    interrupted_manifest = json.loads((interrupted_run / "manifest.json").read_text())
    assert interrupted_manifest["status"] == ("interrupted" if failure is KeyboardInterrupt else "failed")
    assert "result" not in interrupted_manifest["cases"]
    checkpoints = [
        case
        for case, record in interrupted_manifest["cases"].items()
        if record["metadata"].get("kind") == "restart_state"
    ]
    assert len(checkpoints) > 1
    with np.load(
        interrupted_run / interrupted_manifest["cases"][max(checkpoints)]["path"], allow_pickle=False
    ) as data:
        state = json.loads(str(data["state_json"]))
    assert state["current"]["step"] > 0
    resumed_provider = HarmonicProvider()
    run_workflow(
        config,
        run_dir=interrupted_run,
        resume=True,
        force_provider=resumed_provider,
        force_provider_id="analytic-harmonic-v1",
    )
    reference_provider = HarmonicProvider()
    complete = run_workflow(
        config,
        run_dir=tmp_path / "reference",
        force_provider=reference_provider,
        force_provider_id="analytic-harmonic-v1",
    )
    resumed_manifest, resumed_arrays = artifact(interrupted_run)
    reference_manifest, reference_arrays = artifact(complete)
    assert resumed_manifest["status"] == "completed"
    assert resumed_provider.calls < reference_provider.calls
    assert resumed_arrays.keys() == reference_arrays.keys()
    for key in reference_arrays:
        np.testing.assert_array_equal(resumed_arrays[key], reference_arrays[key])
    assert (
        resumed_manifest["cases"]["result"]["metadata"]["summary"]
        == reference_manifest["cases"]["result"]["metadata"]["summary"]
    )
    with pytest.raises(ValueError, match="Resume rejected"):
        run_workflow(
            config,
            run_dir=interrupted_run,
            resume=True,
            force_provider=HarmonicProvider(),
            force_provider_id="different-producer",
        )


@pytest.mark.parametrize("spin_degeneracy,electron_num", [(2, 2), (1, 1)])
def test_predicted_graph_bands_fill_declared_electron_degeneracy_and_report_source(
    tmp_path, spin_degeneracy, electron_num
):
    width = 14
    hamiltonian = np.zeros((1, width, width))
    overlap = np.zeros_like(hamiltonian)
    hamiltonian[0, 0, 0], hamiltonian[0, 3, 3] = -1, 1
    overlap[0, 0, 0] = overlap[0, 3, 3] = 1
    graph = HamiltonianGraph(
        np.array([1]),
        np.eye(3) * 4,
        np.zeros((1, 3)),
        np.array([2]),
        np.array([[0, 3] + [-1] * (width - 2)]),
        np.zeros((2, 0), int),
        np.zeros((0, 3), int),
        np.zeros(0, int),
        hamiltonian,
        overlap,
        metadata={
            "has_scf_reference": True,
            "fermi_energy_eV": 0.0,
            "electron_count": electron_num,
            "spin_degeneracy": spin_degeneracy,
        },
    )
    write_graphs(tmp_path / "graph.npz", {"toy": graph})
    predictions = hamiltonian.copy()
    predictions[0, 0, 0] += 2
    predictions[0, 3, 3] += 2
    np.save(tmp_path / "predictions.npy", predictions.reshape(1, -1) / HARTREE_EV)
    run = run_workflow(
        {
            "workflow": "hamiltonian_bands",
            "inputs": {"material_dir": tmp_path, "graph": "graph.npz", "predictions": "predictions.npy"},
            "parameters": {
                "kpoints": [[0, 0, 0], [0.5, 0, 0]],
                "electron_num": electron_num,
                "filling_grid": [1, 1, 1],
                "prediction_unit": "Ha",
            },
            "output": {"results_dir": tmp_path / "results"},
        }
    )
    manifest, arrays = artifact(run)
    np.testing.assert_allclose(arrays["energies"], [[[1, 3], [1, 3]]])
    metadata = manifest["cases"]["result"]["metadata"]
    assert metadata["fermi_energy_eV"] == pytest.approx(2)
    assert metadata["fermi_source"] == "native_electron_filling"
    assert metadata["hamiltonian_source"] == "prediction"
    assert not metadata["has_scf_reference"]
    assert metadata["source_has_scf_reference"]
