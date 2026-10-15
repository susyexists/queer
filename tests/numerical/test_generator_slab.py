"""QUEER_SLAB reproduces the Generator PythTB driver's run_slab on analytic wannier90 files.

The vendored driver runs its own ``run_slab`` on a stand-in context holding the
step's folded table, so the levels, the per-state edge weights, the crossing
counts, the path and the outputs are compared one to one. A Chern layer's ribbon
carries one chiral mode per edge and a trivial one none.
"""

import json

import numpy as np
import pytest
from tight_binding_models import haldane, kane_mele
from wannier90_files import Wannier90Job

from queer.compat.generator import slab as slab_step
from queer.compat.generator.pythtb_params import translate_pythtb
from queer.compat.generator.results import EXIT_FAILED, EXIT_OK, RESULTS_FILENAME, to_jsonable
from queer.compat.generator.runner import run
from queer.compat.generator.wannier_steps import WannierStepContext

BASE = {"step_type": "PYTHTB_SLAB", "fermi_energy": 0.0, "workers": 1, "ws_correction": "off"}
SLAB_KEYS = [
    "kind",
    "finite_axis",
    "periodic_axes",
    "cells",
    "states",
    "window_eV",
    "edge_cells",
    "edge_crossings_at_fermi",
    "note",
]
FILE_KEYS = [
    "k_distance",
    "kpoints_frac",
    "energies_eV",
    "left_weight",
    "right_weight",
    "tick_indices",
    "tick_labels",
]


def write_model(directory, model):
    job = Wannier90Job(model.translations, model.hopping_matrices, model.lattice, model.centres)
    job.write(directory)
    return job


def settings(**overrides):
    values = dict(BASE)
    values.update(overrides)
    return values


def run_slab(directory, **overrides):
    params_path = directory / "pythtb_params.json"
    params_path.write_text(json.dumps(settings(**overrides)))
    code = run(None, params_path, directory, log=lambda line: None)
    return code, json.loads((directory / RESULTS_FILENAME).read_text())


def driver_slab(core, monkeypatch, directory, values):
    """The vendored driver's run_slab on the folded table of ``directory``'s model; returns its context."""

    class DriverContext:
        add_output = core.Context.add_output
        _plt = core.Context._plt
        _save = core.Context._save
        _ticks = core.Context._ticks

        def parallel(self, stage, per_k, n_items, chunk=None, n_orb=None):
            return 1, 1

    context = WannierStepContext(translate_pythtb(values), directory, log=lambda line: None)
    context.build_model()
    driver = DriverContext()
    driver.params = core.merged_params(values)
    driver.periodic = list(context.periodic)
    driver.n_orb = context.n_orb
    driver.lattice = context.lattice
    driver.table_R, driver.table_H = context.table_R, context.table_H
    driver.fermi = context.fermi
    driver.results, driver.warnings, driver.outputs = {}, [], []
    work = directory / "driver"
    (work / "Out").mkdir(parents=True)
    monkeypatch.chdir(work)
    core.run_slab(driver)
    return driver, work


def assert_matches_the_driver(core, monkeypatch, directory, document, values):
    driver, work = driver_slab(core, monkeypatch, directory, values)
    expected = to_jsonable(driver.results["slab"])
    assert list(document["slab"]) == SLAB_KEYS
    assert document["slab"] == expected
    kind = expected["kind"]
    produced = json.loads((directory / "Out" / f"{kind}_bands.json").read_text())
    reference = json.loads((work / "Out" / f"{kind}_bands.json").read_text())
    assert list(produced) == list(reference) == FILE_KEYS
    assert produced == reference
    titles = [(entry["path"], entry["title"], entry["kind"]) for entry in document["outputs"][1:]]
    assert titles == [(entry["path"], entry["title"], entry["kind"]) for entry in driver.outputs]
    assert driver.warnings == []
    return produced


@pytest.mark.parametrize(("staggered", "net"), [(0.2, 1), (1.5, 0)], ids=["chern", "trivial"])
def test_a_haldane_ribbon_has_one_chiral_mode_per_edge_exactly_as_the_driver(
    generator_core, monkeypatch, tmp_path, staggered, net
):
    write_model(tmp_path, haldane(staggered=staggered))
    values = settings(slab_kpoints=41)
    code, document = run_slab(tmp_path, slab_kpoints=41)
    assert code == EXIT_OK, document.get("error")
    slab = document["slab"]
    assert (slab["kind"], slab["finite_axis"], slab["periodic_axes"]) == ("ribbon", 1, [0])
    assert (slab["cells"], slab["states"], slab["window_eV"], slab["edge_cells"]) == (20, 40, 0.5, 2)
    crossings = slab["edge_crossings_at_fermi"]
    # The Haldane layer's C = -1 puts a left-moving mode on the first edge.
    assert crossings["left"]["net"] == net and crossings["right"]["net"] == -net
    assert crossings["left"]["up"] + crossings["left"]["down"] == abs(net)
    produced = assert_matches_the_driver(generator_core, monkeypatch, tmp_path, document, values)
    # 41 points round up to 42, half a step off -1/2, 0 and 1/2.
    k = np.asarray(produced["kpoints_frac"])
    assert len(k) == 42 and np.all(k[:, 1:] == 0)
    assert np.allclose(k[:, 0], -0.5 + (np.arange(42) + 0.5) / 42, atol=0, rtol=0)
    assert produced["k_distance"] == list(k[:, 0])
    assert produced["tick_indices"] == [] and produced["tick_labels"] == []


def test_levels_stay_in_the_window_with_ragged_weights(tmp_path):
    write_model(tmp_path, haldane(staggered=0.2))
    code, document = run_slab(tmp_path, slab_kpoints=24, slab_cells=8, slab_window_eV=0.8, edge_cells=1)
    assert code == EXIT_OK, document.get("error")
    slab = document["slab"]
    assert (slab["cells"], slab["states"], slab["window_eV"], slab["edge_cells"]) == (8, 16, 0.8, 1)
    produced = json.loads((tmp_path / "Out" / "ribbon_bands.json").read_text())
    counts = [len(levels) for levels in produced["energies_eV"]]
    assert len(set(counts)) > 1 and max(counts) < 16
    for levels, left, right in zip(
        produced["energies_eV"], produced["left_weight"], produced["right_weight"]
    ):
        levels, left, right = np.asarray(levels), np.asarray(left), np.asarray(right)
        assert len(left) == len(right) == len(levels)
        assert np.all(np.abs(levels) <= 0.8) and np.all(np.diff(levels) >= 0)
        assert np.all(left >= -1e-12) and np.all(right >= -1e-12) and np.all(left + right <= 1 + 1e-12)
    titles = [entry["title"] for entry in document["outputs"][1:]]
    assert titles == ["Ribbon bands (data)", "Ribbon bands with edge weight"]


def test_a_crystal_slab_follows_the_surface_path_exactly_as_the_driver(generator_core, monkeypatch, tmp_path):
    # PT-symmetric Kane-Mele layers stacked along a3: degenerate pairs at every k exercise the
    # driver's rotation of each cluster onto the first edge.
    write_model(tmp_path, kane_mele(0.0, interlayer=0.3, rashba=0.0))
    values = settings(slab_kpoints=25, slab_cells=6, slab_window_eV=1.5)
    code, document = run_slab(tmp_path, slab_kpoints=25, slab_cells=6, slab_window_eV=1.5)
    assert code == EXIT_OK, document.get("error")
    slab = document["slab"]
    assert document["model"]["dimension"] == 3
    assert (slab["kind"], slab["finite_axis"], slab["periodic_axes"], slab["states"]) == (
        "slab",
        2,
        [0, 1],
        24,
    )
    produced = assert_matches_the_driver(generator_core, monkeypatch, tmp_path, document, values)
    path = generator_core.surface_path(np.asarray(document["model"]["lattice"]), 0, 1, 25)
    assert produced["kpoints_frac"] == to_jsonable(path["kpoints"])
    assert produced["tick_labels"] == ["G", "M", "K", "G", "M'", "M''", "G"]
    assert produced["tick_indices"] == path["tick_indices"]
    distance = generator_core.k_distance(path["kpoints"], document["model"]["lattice"], path["break_indices"])
    assert produced["k_distance"] == to_jsonable(distance)
    levels = [np.asarray(values) for values in produced["energies_eV"]]
    assert any(np.any(np.diff(values) < 1e-5) for values in levels)


def test_a_layer_cut_along_its_first_axis_is_a_ribbon_along_the_second(generator_core, monkeypatch, tmp_path):
    write_model(tmp_path, haldane(staggered=0.2))
    values = settings(slab_axis=0, slab_kpoints=12, slab_cells=6)
    code, document = run_slab(tmp_path, slab_axis=0, slab_kpoints=12, slab_cells=6)
    assert code == EXIT_OK, document.get("error")
    assert (document["slab"]["finite_axis"], document["slab"]["periodic_axes"]) == (0, [1])
    assert_matches_the_driver(generator_core, monkeypatch, tmp_path, document, values)


def test_a_finite_axis_outside_the_periodic_ones_is_a_bad_parameter(tmp_path):
    write_model(tmp_path, haldane(staggered=0.2))
    code, document = run_slab(tmp_path, slab_axis=2)
    assert code == EXIT_FAILED and document["error"]["code"] == "queer_bad_params"
    assert document["error"]["message"] == "slab_axis 2 is not a periodic direction of the model"
    assert "slab" not in document


@pytest.mark.parametrize(
    ("two_d", "n_orb", "cells"), [(True, 60, 20), (True, 61, 12), (False, 60, 12), (False, 61, 6)]
)
def test_default_cells_follow_the_drivers_size_rule(two_d, n_orb, cells):
    assert slab_step.default_cells(two_d, n_orb) == cells
