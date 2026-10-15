"""QUEER_BERRY reproduces the Generator PythTB driver's run_berry on analytic wannier90 files.

The vendored driver runs its own ``run_berry`` in ``fermi`` mode on a stand-in
context whose PythTB model gives convention-I eigenvectors and reduced
velocities (:func:`cell_periodic_operators`), so the maps, the path, the result
block, the warnings and the outputs are compared one to one. ``bands`` mode,
which the driver hands to PythTB's own Kubo curvature, is checked by the Chern
numbers and quantum weights of gapped layers and by its equality with the
Fermi-sea sum in the gap.
"""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose
from tight_binding_models import cell_periodic_operators, haldane, kane_mele, qwz
from wannier90_files import Wannier90Job

from queer.compat.generator import berry as berry_step
from queer.compat.generator.pythtb_params import translate_pythtb
from queer.compat.generator.results import (
    BERRY_CONVENTIONS,
    EXIT_FAILED,
    EXIT_OK,
    RESULTS_FILENAME,
    to_jsonable,
)
from queer.compat.generator.runner import run
from queer.compat.generator.wannier_steps import WannierStepContext
from queer.models.wannier import WannierModel

# The vendored driver draws its maps of a sheared plane without silencing this notice.
pytestmark = pytest.mark.filterwarnings("ignore:The input coordinates to pcolormesh:UserWarning")
LAYER_PATH = {
    "segments": [
        ["G", [0, 0, 0], "M", [0.5, 0, 0]],
        ["M", [0.5, 0, 0], "K", [1 / 3, 2 / 3, 0]],
        ["K", [1 / 3, 2 / 3, 0], "G", [0, 0, 0]],
    ]
}
BASE = {
    "step_type": "PYTHTB_BERRY",
    "fermi_energy": 0.0,
    "workers": 1,
    "ws_correction": "off",
    "kpath": LAYER_PATH,
    "kpath_points": 30,
    "berry_mesh": 16,
}
MODEL_OUTPUTS = 1  # the hopping-decay figure of the model stage
BANDS_OUTPUTS = [
    ("Out/berry_k3_0.json", "Berry curvature map (data)", "data"),
    ("Out/berry_k3_0.png", "Berry curvature Ω (Å²)", "plot"),
    ("Out/metric_k3_0.png", "Quantum metric tr g (Å²)", "plot"),
    ("Out/berry_path.json", "Berry curvature along the path (data)", "data"),
    ("Out/berry_path.png", "Berry curvature along the path", "plot"),
]


def qwz_layer(mass):
    """The QWZ layer with both orbitals on one site mid-cell."""
    return qwz(mass, centres=np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 5.0]]))


def write_model(directory, model, **options):
    job = Wannier90Job(model.translations, model.hopping_matrices, model.lattice, model.centres, **options)
    job.write(directory)
    return job


def settings(**overrides):
    values = dict(BASE)
    values.update(overrides)
    return {key: value for key, value in values.items() if value is not None}


def run_berry(directory, **overrides):
    values = settings(**overrides)
    params_path = directory / "pythtb_params.json"
    params_path.write_text(json.dumps(values))
    code = run(None, params_path, directory, log=lambda line: None)
    return code, json.loads((directory / RESULTS_FILENAME).read_text())


def read_out(directory, name):
    return json.loads((directory / "Out" / name).read_text())


def outputs(document):
    return [(entry["path"], entry["title"], entry["kind"]) for entry in document["outputs"]][MODEL_OUTPUTS:]


class PythtbConvention:
    """PythTB's ``solve_ham`` and ``velocity(cartesian=False)`` on the driver's table, E_F off the onsite block."""

    def __init__(self, job, fermi):
        translations, table = job.driver_table()
        self.model = WannierModel(translations, table, lattice=job.lattice, centres=job.centres)
        self.fermi = fermi

    def operators(self, k):
        hamiltonian, velocity = cell_periodic_operators(self.model, k)
        return hamiltonian - self.fermi * np.eye(len(hamiltonian)), velocity

    def solve_ham(self, kpoints, return_eigvecs=False):
        values, rows = [], []
        for k in np.asarray(kpoints, float).reshape(-1, 3):
            energies, vectors = np.linalg.eigh(self.operators(k)[0])
            values.append(energies)
            rows.append(vectors.T)
        return (np.asarray(values), np.asarray(rows)) if return_eigvecs else np.asarray(values)

    def velocity(self, kpoints, cartesian=False):
        assert not cartesian
        return np.stack([self.operators(k)[1] for k in np.asarray(kpoints, float).reshape(-1, 3)], axis=1)


def driver_context_class(core):
    """A stand-in for the driver's Context: QUEER's occupation and path, the driver's own plane helpers."""

    class DriverContext:
        plane_points = core.Context.plane_points
        default_plane = core.Context.default_plane
        add_output = core.Context.add_output
        _plt = core.Context._plt
        _save = core.Context._save
        _ticks = core.Context._ticks

        def __init__(self, context, values):
            self.params = core.merged_params(values)
            self.periodic = list(context.periodic)
            self.lattice = context.lattice
            self.recip = core.reciprocal_lattice(context.lattice)
            self.results, self.warnings, self.outputs = {}, [], []
            self.occupation = context.resolve_occupation()
            self.path = context.band_path()

        def resolve_occupation(self):
            return dict(self.occupation)

        def band_path(self):
            return self.path

        def parallel(self, stage, per_k, n_items, chunk=None, n_orb=None):
            return 1, max(1, n_items)

    return DriverContext


def run_driver_berry(core, monkeypatch, directory, job, values):
    """The vendored driver's run_berry on ``directory``'s model; returns its context."""
    params = translate_pythtb(values)
    context = WannierStepContext(params, directory, log=lambda line: None)
    context.build_model()
    driver = driver_context_class(core)(context, values)
    monkeypatch.setitem(core._STATE, "model", PythtbConvention(job, context.fermi))
    work = directory / "driver"
    (work / "Out").mkdir(parents=True)
    monkeypatch.chdir(work)
    core.run_berry(driver)
    return driver, work


@pytest.mark.parametrize(
    ("builder", "fermi"),
    [(lambda: haldane(staggered=0.2), 0.8), (lambda: qwz_layer(-1.0), 1.5)],
    ids=["haldane", "qwz"],
)
def test_fermi_mode_reproduces_the_drivers_maps_path_and_block(
    generator_core, monkeypatch, tmp_path, builder, fermi
):
    job = write_model(tmp_path, builder())
    values = settings(fermi_energy=fermi)
    code, document = run_berry(tmp_path, fermi_energy=fermi)
    assert code == EXIT_OK, document.get("error")
    driver, work = run_driver_berry(generator_core, monkeypatch, tmp_path, job, values)

    berry, expected = document["berry"], driver.results["berry"]
    assert berry["occupation"] == "fermi" and berry["occupation_info"]["metal"] is True
    assert list(berry) == list(expected)
    for key in ("fixed_axis", "value", "plane_axes", "occupation", "occupation_info", "mesh", "path_file"):
        assert berry[key] == to_jsonable(expected[key]), key
    assert berry["conventions"] == expected["conventions"] == generator_core.BERRY_CONVENTIONS
    for key in ("fermi_sea_integral_e2_h", "sigma_xy_e2_h", "omega_cart_max_A2"):
        assert berry[key] == pytest.approx(expected[key], rel=1e-10, abs=1e-12), key
    assert berry["sigma_xy_e2_h"] == -berry["fermi_sea_integral_e2_h"]
    assert abs(berry["fermi_sea_integral_e2_h"]) > 1e-3
    assert document["warnings"][-1:] == driver.warnings == [berry_step.METRIC_SKIPPED_WARNING]
    assert document["warning_codes"][-1] == "quantum_metric_skipped"
    assert outputs(document) == [(entry["path"], entry["title"], entry["kind"]) for entry in driver.outputs]

    plane, reference = (
        read_out(tmp_path, "berry_k3_0.json"),
        json.loads((work / "Out" / "berry_k3_0.json").read_text()),
    )
    assert list(plane) == list(reference) == ["k_frac", "omega_reduced", "omega_cart_A2"]
    assert plane["k_frac"] == reference["k_frac"]
    scale = np.abs(reference["omega_reduced"]).max()
    assert_allclose(plane["omega_reduced"], reference["omega_reduced"], rtol=0, atol=1e-11 * scale)
    assert_allclose(plane["omega_cart_A2"], reference["omega_cart_A2"], rtol=1e-12, atol=1e-11 * scale)

    path, reference = (
        read_out(tmp_path, "berry_path.json"),
        json.loads((work / "Out" / "berry_path.json").read_text()),
    )
    assert list(path) == list(reference)
    assert (
        path["tick_indices"] == reference["tick_indices"] and path["tick_labels"] == reference["tick_labels"]
    )
    assert_allclose(path["k_distance"], reference["k_distance"], rtol=1e-14, atol=0)
    scale = np.abs(reference["omega_cart_A2"]).max()
    assert_allclose(path["omega_cart_A2"], reference["omega_cart_A2"], rtol=0, atol=1e-11 * scale)


@pytest.mark.parametrize(
    ("builder", "chern", "mesh"),
    [(lambda: haldane(staggered=0.2), -1, 24), (lambda: qwz_layer(-1.0), 1, 16)],
    ids=["haldane", "qwz"],
)
def test_bands_mode_gives_the_kubo_chern_number_bounded_by_the_quantum_weight(tmp_path, builder, chern, mesh):
    write_model(tmp_path, builder())
    code, document = run_berry(tmp_path, berry_mesh=mesh)
    assert code == EXIT_OK, document.get("error")
    berry = document["berry"]
    assert berry["occupation"] == "bands" and berry["occupation_info"]["isolated"] is True
    assert list(berry) == [
        "fixed_axis",
        "value",
        "plane_axes",
        "occupation",
        "occupation_info",
        "mesh",
        "approximation",
        "chern_kubo",
        "chern_kubo_integer_distance",
        "conventions",
        "omega_cart_max_A2",
        "quantum_weight",
        "quantum_weight_bound_ok",
        "path_file",
    ]
    assert berry["chern_kubo"] == pytest.approx(chern, abs=1e-3)
    assert berry["chern_kubo_integer_distance"] == pytest.approx(abs(berry["chern_kubo"] - chern), abs=1e-15)
    assert berry["quantum_weight"] >= abs(chern) - 1e-3 and berry["quantum_weight_bound_ok"] is True
    assert berry["conventions"] == dict(BERRY_CONVENTIONS)
    assert "chern_kubo_not_integer" not in document["warning_codes"]
    assert outputs(document) == BANDS_OUTPUTS
    plane = read_out(tmp_path, "berry_k3_0.json")
    assert list(plane) == ["k_frac", "omega_reduced", "omega_cart_A2", "metric_trace_A2"]
    assert len(plane["k_frac"]) == len(plane["metric_trace_A2"]) == mesh * mesh
    area = np.linalg.norm(
        np.cross(*(2 * np.pi * np.linalg.inv(np.asarray(document["model"]["lattice"])).T)[:2])
    )
    assert berry["quantum_weight"] == pytest.approx(
        np.sum(plane["metric_trace_A2"]) * area / mesh**2 / (2 * np.pi)
    )


def test_fermi_sea_in_the_gap_equals_the_band_set(tmp_path):
    maps = {}
    for mode in ("bands", "fermi"):
        directory = tmp_path / mode
        directory.mkdir()
        write_model(directory, haldane(staggered=0.2))
        code, document = run_berry(directory, occupation=mode, path_curvature=False)
        assert code == EXIT_OK, document.get("error")
        maps[mode] = (document["berry"], read_out(directory, "berry_k3_0.json"))
    (bands, band_map), (fermi, fermi_map) = maps["bands"], maps["fermi"]
    assert fermi["occupation"] == "fermi" and "path_file" not in fermi
    assert fermi["fermi_sea_integral_e2_h"] == pytest.approx(bands["chern_kubo"], abs=1e-12)
    assert fermi["sigma_xy_e2_h"] == pytest.approx(-bands["chern_kubo"], abs=1e-12)
    scale = np.abs(band_map["omega_reduced"]).max()
    assert_allclose(fermi_map["omega_reduced"], band_map["omega_reduced"], rtol=0, atol=1e-10 * scale)
    assert "quantum_weight" not in fermi and "metric_trace_A2" not in fermi_map


def test_bands_mode_needs_an_occupied_set(tmp_path):
    write_model(tmp_path, haldane(staggered=0.2))
    code, document = run_berry(tmp_path, fermi_energy=0.8, occupation="bands")
    assert code == EXIT_FAILED and document["error"]["code"] == "queer_bad_params"
    assert document["error"]["message"] == "occupation='bands' needs a gapped model or occupied_bands"
    assert "berry" not in document


def test_a_coarse_mesh_warns_that_the_chern_number_is_not_an_integer(tmp_path):
    write_model(tmp_path, haldane(staggered=0.2))
    code, document = run_berry(tmp_path, berry_mesh=3, quantum_metric=False, path_curvature=False)
    assert code == EXIT_OK
    berry = document["berry"]
    distance = berry["chern_kubo_integer_distance"]
    assert distance > 0.05 and "quantum_weight" not in berry
    assert document["warning_codes"][-1] == "chern_kubo_not_integer"
    assert document["warnings"][-1] == (
        f"the Kubo Chern number {berry['chern_kubo']:.3f} is {distance:.3f} from an integer: a small gap "
        "is under-resolved on the 3x3 mesh (raise berry_mesh)"
    )
    assert [path for path, _, _ in outputs(document)] == ["Out/berry_k3_0.json", "Out/berry_k3_0.png"]


def test_a_missing_path_skips_the_path_curvature_with_a_warning(tmp_path):
    write_model(tmp_path, qwz_layer(-1.0))
    code, document = run_berry(tmp_path, kpath=None, kpath_points=None, quantum_metric=False)
    assert code == EXIT_OK
    assert "path_file" not in document["berry"]
    assert document["warning_codes"][-1] == "path_curvature_skipped"
    assert document["warnings"][-1] == (
        "path curvature skipped: No k-path: the WANNIER run had no kpoint_path and none was resolved"
    )
    assert not (tmp_path / "Out" / "berry_path.json").exists()


@pytest.mark.parametrize(
    ("requested", "periodic", "plane"),
    [
        (None, [0, 1], (2, 0.0)),
        (None, [0, 2], (1, 0.0)),
        (None, [1, 2], (0, 0.0)),
        (None, [0, 1, 2], (2, 0.0)),
        ({"fixed_axis": 0, "value": 0.5}, [0, 1, 2], (0, 0.5)),
    ],
)
def test_the_plane_is_the_requested_one_else_the_drivers_default(requested, periodic, plane):
    missing = [axis for axis in range(3) if axis not in periodic]
    if requested is None:
        assert plane == ((missing[0] if missing else 2), 0.0)
    assert berry_step.selected_plane(requested, periodic) == plane


def test_a_requested_plane_of_a_crystal_names_its_files_by_the_plane(tmp_path):
    write_model(tmp_path, kane_mele(1.0, interlayer=0.2))
    code, document = run_berry(
        tmp_path, berry_plane={"fixed_axis": 0, "value": 0.5}, berry_mesh=6, path_curvature=False
    )
    assert code == EXIT_OK, document.get("error")
    berry = document["berry"]
    assert document["model"]["dimension"] == 3
    assert (berry["fixed_axis"], berry["value"], berry["plane_axes"]) == (0, 0.5, [1, 2])
    plane = read_out(tmp_path, "berry_k1_0p5.json")
    assert np.all(np.asarray(plane["k_frac"])[:, 0] == 0.5)
    assert outputs(document)[0] == ("Out/berry_k1_0p5.json", "Berry curvature map (data)", "data")


def test_a_refused_curvature_is_a_numerical_error(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise ValueError("occupied/unoccupied boundary splits a degenerate subspace")

    monkeypatch.setattr(berry_step, "berry_plane", refuse)
    write_model(tmp_path, haldane(staggered=0.2))
    code, document = run_berry(tmp_path)
    assert code == EXIT_FAILED and document["error"]["code"] == "queer_numerical_error"
    assert document["error"]["message"] == (
        "Berry curvature (bands), k3=0 failed: occupied/unoccupied boundary splits a degenerate subspace"
    )
    assert "berry" not in document
