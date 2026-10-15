"""TOML-level Wannier sidecars, interpolation selection and position modes."""

import json
from itertools import product

import numpy as np
import pytest

from queer.config import config_from_mapping
from queer.workflows import run_workflow
from queer.workflows.sources import load_electronic_model, model_input_files

LATTICE = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 10.0]])
CENTRES = np.array([[0.1, 0.0, 5.0], [1.1, 0.0, 5.0]])
ONSITE = np.diag([-0.5, 0.5])
HOP = np.array([[0.1, 0.05], [0.05, -0.1]])
TRANSLATIONS = [(-1, 0, 0), (0, 0, 0), (1, 0, 0)]


def analytic_energies(points):
    values = []
    for point in np.asarray(points):
        phase = np.exp(2j * np.pi * point[0])
        values.append(np.linalg.eigvalsh(ONSITE + phase * HOP + np.conj(phase) * HOP.T))
    return np.asarray(values).T


def write_material(root):
    """Two-band chain in HR, TB (with positions), centres, WIN and a disabled wsvec."""
    blocks = {(-1, 0, 0): HOP.T, (0, 0, 0): ONSITE, (1, 0, 0): HOP}
    hr = ["analytic chain", "2", "3", "1 1 1"]
    tb = ["analytic chain", *(" ".join(f"{x:.10f}" for x in row) for row in LATTICE), "2", "3", "1 1 1"]
    for translation in TRANSLATIONS:
        tb.append(" ".join(map(str, translation)))
        for j, i in product(range(2), repeat=2):
            value = blocks[translation][i, j]
            hr.append(f"{translation[0]} {translation[1]} {translation[2]} {i + 1} {j + 1} {value:.12f} 0.0")
            tb.append(f"{i + 1} {j + 1} {value:.12f} 0.0")
    for translation in TRANSLATIONS:
        tb.append(" ".join(map(str, translation)))
        for j, i in product(range(2), repeat=2):
            position = CENTRES[i] if i == j and translation == (0, 0, 0) else [0.0, 0.0, 0.0]
            off = 0.02 if i != j else 0.0
            values = [position[0] + off, 0.0, position[1], 0.0, position[2], 0.0]
            tb.append(f"{i + 1} {j + 1} " + " ".join(f"{value:.12f}" for value in values))
    (root / "wannier90_hr.dat").write_text("\n".join(hr) + "\n")
    (root / "wannier90_tb.dat").write_text("\n".join(tb) + "\n")
    (root / "wannier90_centres.xyz").write_text(
        "2\nanalytic centres\n" + "\n".join("X " + " ".join(map(str, centre)) for centre in CENTRES) + "\n"
    )
    cell = "\n".join(" ".join(map(str, row)) for row in LATTICE)
    (root / "wannier90.win").write_text(
        f"mp_grid = 4 1 1\nbegin unit_cell_cart\nang\n{cell}\nend unit_cell_cart\n"
    )
    wsvec = ["## written with use_ws_distance=.false."]
    for translation in TRANSLATIONS:
        for i, j in product(range(1, 3), repeat=2):
            wsvec.extend([f"{translation[0]} {translation[1]} {translation[2]} {i} {j}", "1", "0 0 0"])
    (root / "wannier90_wsvec.dat").write_text("\n".join(wsvec) + "\n")
    return root


def config(root, workflow="bands", inputs=None, **parameters):
    values = {"ef": 0.0, "interpolation": "standard", **parameters}
    if workflow == "bands" and "kpoints" not in values:
        values["kpoints"] = [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]
    return config_from_mapping(
        {
            "workflow": workflow,
            "inputs": {"material_dir": str(root), **(inputs or {})},
            "parameters": values,
            "output": {"results_dir": str(root / "results")},
        }
    )


def test_named_sidecars_auto_interpolation_and_full_positions(tmp_path):
    root = write_material(tmp_path)
    resolved = config(
        root,
        inputs={"win": "wannier90.win", "tb": "wannier90_tb.dat", "centres": "wannier90_centres.xyz"},
        interpolation="auto",
        position="full",
    )
    files = {path.name for path in model_input_files(resolved.inputs, resolved.parameters)}
    assert files == {
        "wannier90_hr.dat",
        "wannier90_tb.dat",
        "wannier90_centres.xyz",
        "wannier90.win",
        "wannier90_wsvec.dat",
    }
    model = load_electronic_model(resolved)
    assert model.interpolation == "standard"
    assert model.position_matrices is not None and model.metadata["position_link"] == "midpoint_first_order"
    np.testing.assert_allclose(model.lattice, LATTICE)
    assert {"win", "tb", "wsvec"} <= set(model.source_files)
    points = [[0.0, 0.0, 0.0], [0.3, 0.1, 0.0]]
    np.testing.assert_allclose(model.energies(points), analytic_energies(points), atol=1e-10)


def test_diagonal_position_ignores_an_unnamed_tb_file_and_takes_the_win_lattice(tmp_path):
    root = write_material(tmp_path)
    resolved = config(root, inputs={"win": "wannier90.win"}, position="diagonal")
    files = {path.name for path in model_input_files(resolved.inputs, resolved.parameters)}
    assert files == {"wannier90_hr.dat", "wannier90_centres.xyz", "wannier90.win"}
    model = load_electronic_model(resolved)
    assert model.position_matrices is None and model.metadata["position_link"] == "exact_diagonal"
    np.testing.assert_allclose(model.lattice, LATTICE)
    np.testing.assert_allclose(model.centres, CENTRES)
    run = run_workflow(resolved)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert sorted(entry["path"].rsplit("/", 1)[-1] for entry in manifest["inputs"]) == sorted(files)
    assert manifest["config"]["parameters"]["position"] == "diagonal"
    with np.load(run / manifest["cases"]["bands"]["path"]) as arrays:
        np.testing.assert_allclose(
            arrays["energies"], analytic_energies(resolved.parameters["kpoints"]), atol=1e-10
        )


def test_registered_observables_accept_model_loading_parameters(tmp_path):
    root = write_material(tmp_path)
    resolved = config(
        root,
        workflow="berry_curvature",
        inputs={"win": "wannier90.win"},
        interpolation="auto",
        position="diagonal",
        kpoints=[[0.1, 0.2, 0.0]],
        occupied_bands=1,
    )
    run = run_workflow(resolved)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "completed" and "result" in manifest["cases"]
    assert manifest["scientific_model"]["definitions"]["position_link"] == "exact_diagonal"


@pytest.mark.parametrize(
    "inputs,parameters,match",
    [
        ({}, {"position": "exact"}, "parameters.position must be auto, full or diagonal"),
        ({}, {"interpolation": "nearest"}, "standard, ws_distance, ndegen_applied or auto"),
        ({"win": ""}, {}, "inputs.win must be a nonempty filename"),
        ({"model": "openmx", "scfout": "x.scfout", "tb": "wannier90_tb.dat"}, {}, "Wannier inputs: tb"),
    ],
)
def test_invalid_wannier_configuration_is_refused(tmp_path, inputs, parameters, match):
    with pytest.raises(ValueError, match=match):
        config(tmp_path, inputs=inputs, **parameters)


def test_openmx_and_services_refuse_wannier_modes(tmp_path):
    with pytest.raises(ValueError, match="OpenMX matrices do not use Wannier position modes"):
        config_from_mapping(
            {
                "workflow": "fat_band",
                "inputs": {"material_dir": str(tmp_path), "model": "openmx", "scfout": "x.scfout"},
                "parameters": {"position": "full", "kpoints": [[0.0, 0.0, 0.0]]},
            }
        )
    with pytest.raises(ValueError, match="Unknown input/output options"):
        config_from_mapping(
            {
                "workflow": "structure",
                "inputs": {"material_dir": str(tmp_path), "structure": "POSCAR", "win": "x"},
            }
        )
    with pytest.raises(ValueError, match="Unknown input/output options"):
        config_from_mapping({"workflow": "epw", "inputs": {"material_dir": str(tmp_path), "tb": "x_tb.dat"}})
