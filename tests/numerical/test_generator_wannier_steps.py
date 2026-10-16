"""The Wannier model stage reproduces the Generator PythTB driver on analytic wannier90 files.

Each test writes a small analytic model from ``tight_binding_models`` as the files
a WANNIER job stages and compares :class:`WannierStepContext` with the vendored
driver's pure functions (``generator_core``) on the same tables. The driver's
``Context`` needs PythTB and is never built.
"""

import hashlib
import importlib.util
import json
import re

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from tight_binding_models import haldane, kane_mele
from wannier90_files import Wannier90Job, exact, line_path, write_band_files

import queer
from queer.compat.generator import wannier_steps
from queer.compat.generator.pythtb_params import translate_pythtb
from queer.compat.generator.results import (
    ERR_BAD_PARAMS,
    ERR_MISSING_INPUT,
    ERR_TRANSLATE_HOME_CELL,
    ERR_VALIDATION,
    EXIT_FAILED,
    EXIT_OK,
    RESULTS_FILENAME,
    RunnerError,
    execute,
)
from queer.compat.generator.wannier_steps import (
    WS_SCORE_KEYS,
    WannierStepContext,
    model_stage,
    pythtb_lattice,
)
from queer.io.wannier import WannierData, read_win, recompute_wsvec

FERMI = 0.1
MODEL_KEYS = {
    "approximation",
    "cutoffs",
    "dimension",
    "fermi_energy",
    "fermi_source",
    "folded_axes",
    "hermiticity_weyl_bound",
    "lattice",
    "mp_grid",
    "n_R",
    "n_R_after_ws",
    "n_R_model",
    "n_hops",
    "num_wann",
    "orbital_species",
    "periodic_dirs",
    "pythtb_version",
    "queer_version",
    "reference_band_job_id",
    "source_wannier_job_id",
    "species",
    "spinors",
    "win_scalars",
    "ws_correction",
}
VALIDATION_KEYS = {"band_dat", "fermi_in_frozen_window", "hermiticity_error", "localization"}
BAND_DAT_KEYS = {
    "available",
    "in_window_max_meV",
    "in_window_points",
    "in_window_rms_meV",
    "max_meV",
    "n_k",
    "rms_meV",
    "window_eV",
}
LAYER_PATH = [[0, 0, 0], [0.5, 0, 0], [1 / 3, 2 / 3, 0], [0, 0, 0]]
BULK_PATH = [[0, 0, 0], [0.5, 0, 0], [1 / 3, 2 / 3, 0], [0, 0, 0], [0, 0, 0.5]]
STEPS = 8


def layer_job(**settings):
    """A gapped Haldane layer in a 10 Å cell whose two centres sit on either side of the cell boundary."""
    model = haldane(staggered=0.3, height=10.0)
    reduced = np.array([[1 / 3, 1 / 3, 0.02], [2 / 3, 2 / 3, 0.97]])
    options = {
        "degeneracies": [1 if not np.any(r) else 2 for r in model.translations],
        "mp_grid": (2, 2, 1),
        "win": {"dis_froz_min": -1.0, "dis_froz_max": 0.6},
    }
    options.update(settings)
    return Wannier90Job(
        model.translations, model.hopping_matrices, model.lattice, reduced @ model.lattice, **options
    )


def bulk_job(**settings):
    """Stacked Kane-Mele layers with spinors (a 3D model) on a 2x2x2 mesh."""
    model = kane_mele(1.0, interlayer=0.2, height=4.0)
    reduced = np.repeat([[1 / 3, 1 / 3, 0.1], [2 / 3, 2 / 3, 0.8]], 2, axis=0)
    options = {
        "degeneracies": [1 + index % 3 for index in range(len(model.translations))],
        "mp_grid": (2, 2, 2),
        "win": {"spinors": ".true.", "dis_win_min": -4.0, "dis_win_max": 4.0},
    }
    options.update(settings)
    return Wannier90Job(
        model.translations, model.hopping_matrices, model.lattice, reduced @ model.lattice, **options
    )


def chain_job():
    """One orbital hopping along a1 only: periodic along a single axis."""
    translations = [[0, 0, 0], [1, 0, 0], [-1, 0, 0]]
    table = np.array([[[0.0]], [[-1.0]], [[-1.0]]], complex)
    return Wannier90Job(translations, table, np.diag([2.0, 10.0, 10.0]), np.array([[0.5, 5.0, 5.0]]))


def decaying_job(decay=1.5):
    """Two orbitals on a square layer, |t| = exp(-d / decay) up to three cells, centres across the vacuum."""
    lattice = np.diag([2.0, 2.0, 12.0])
    centres = np.array([[0.2, 0.3, 0.5], [1.1, 1.3, 11.6]])
    translations = np.array([(a, b, 0) for a in range(-3, 4) for b in range(-3, 4)])
    near = centres - np.array([0.0, 0.0, 12.0]) * np.array([[0], [1]])
    span = (translations @ lattice)[:, None, None, :] + near[None, None, :, :] - near[None, :, None, :]
    table = np.exp(-np.linalg.norm(span, axis=-1) / decay).astype(complex)
    return Wannier90Job(translations, table, lattice, centres, mp_grid=(7, 7, 1))


def params(**settings):
    values = {
        "step_type": "PYTHTB_BANDS",
        "fermi_energy": FERMI,
        "fermi_source": "reference_band",
        "workers": 1,
    }
    values.update(settings)
    return translate_pythtb(values)


def build(directory, **settings):
    context = WannierStepContext(params(**settings), directory, log=lambda line: None)
    context.build_model()
    return context


def driver_tables(core, job, directory):
    """The driver's parsed .win, PythTB's lattice and the Wigner-Seitz candidates of a written job."""
    win = core.parse_win((directory / "wannier90.win").read_text())
    lattice = np.asarray(core.parse_win(core.sanitized_win(win))["lattice"], float)
    translations, table = job.driver_table()
    candidates = {"off": (translations, table)}
    if win["mp_grid"]:
        search = int(win.get("ws_search_size") or 2)
        tolerance = float(win.get("ws_distance_tol") or 1e-5)
        candidates["recompute"] = core.ws_unfold(
            translations, table, job.centres, lattice, win["mp_grid"], search_size=search, tol=tolerance
        )
    return win, lattice, candidates


def stage_bands(core, directory, table, path, ticks=True):
    """wannier90's band files from ``table`` along ``path``; the k-points and energies relative to E_F."""
    kpoints = exact(line_path(path, STEPS))
    labels = [(chr(ord("A") + index), index * STEPS) for index in range(len(path))] if ticks else ()
    energies = write_band_files(directory, kpoints, core.eigvals_from_table(*table, kpoints), labels)
    return kpoints, energies - FERMI


def driver_decision(core, win, candidates, kpoints, reference):
    window = core.comparison_window(win, FERMI)
    scores = {
        name: core.band_deviation(core.eigvals_from_table(*table, kpoints) - FERMI, reference, window)
        for name, table in candidates.items()
    }
    best = min(scores, key=lambda name: (scores[name]["in_window_max_meV"] or 0.0, name != "recompute"))
    return best, scores


def independent_hops(translations, n):
    """Hops of R = 0 above the diagonal plus every hop of the positive half (PythTB's count)."""
    keys = {tuple(int(value) for value in row) for row in translations}
    count = 0
    for key in keys:
        if not any(key):
            count += n * (n - 1) // 2
        elif tuple(-value for value in key) not in keys or next(value for value in key if value) > 0:
            count += n * n
    return count


def test_layer_model_block_reproduces_the_driver_tables(generator_core, tmp_path):
    core = generator_core
    job = layer_job()
    job.write(tmp_path)
    win, lattice, candidates = driver_tables(core, job, tmp_path)
    stage_bands(core, tmp_path, candidates["recompute"], LAYER_PATH)
    atoms = np.array([[1 / 3, 1 / 3, 0.0], [2 / 3, 2 / 3, 0.0]]) @ job.lattice
    structure = {"species": ["B", "N"], "cart": atoms.tolist()}
    context = build(
        tmp_path, structure=structure, source_wannier_job_id="job-w", reference_band_job_id="job-b"
    )
    block = context.results["model"]
    assert set(block) == MODEL_KEYS
    folded_R, folded_H = core.fold_axes(*candidates["recompute"], [2])
    assert (block["num_wann"], block["n_R"]) == (2, 7)
    assert block["n_R_after_ws"] == len(candidates["recompute"][0]) == 19
    assert block["n_R_model"] == len(folded_R) == 9
    assert block["folded_axes"] == [2] and block["periodic_dirs"] == [0, 1] and block["dimension"] == 2
    assert block["n_hops"] == independent_hops(folded_R, 2) == 17
    assert block["cutoffs"] == {
        "min_hopping_norm": None,
        "max_distance": None,
        "hops": 17,
        "dropped_by_distance": 0,
        "dropped_by_norm": 0,
        "kept_hops": 17,
    }
    assert_array_equal(np.asarray(block["lattice"]), lattice)
    assert_array_equal(context.table_R, folded_R)
    assert_allclose(context.table_H, folded_H, rtol=1e-13, atol=1e-15)
    assert block["mp_grid"] == [2, 2, 1] and block["win_scalars"] == {
        "dis_froz_min": -1.0,
        "dis_froz_max": 0.6,
    }
    assert block["species"] == ["B", "N"] and block["orbital_species"] == ["B", "N"]
    assert block["spinors"] is False
    assert block["fermi_energy"] == FERMI and block["fermi_source"] == "reference_band"
    assert block["source_wannier_job_id"] == "job-w" and block["reference_band_job_id"] == "job-b"
    assert block["pythtb_version"] is None and block["queer_version"] == queer.__version__
    assert block["approximation"].startswith("Tight-binding (diagonal position) approximation")
    assert block["ws_correction"]["mode"] == "recompute"
    assert context.model.metadata["hermitization"] == "average"
    assert_allclose(
        context.model.energies(np.zeros((1, 3)))[:, 0], np.linalg.eigvalsh(folded_H.sum(0)) - FERMI
    )


def test_bulk_model_block_keeps_every_axis_and_the_win_spinors(generator_core, tmp_path):
    core = generator_core
    job = bulk_job()
    job.write(tmp_path)
    _, lattice, candidates = driver_tables(core, job, tmp_path)
    stage_bands(core, tmp_path, candidates["recompute"], BULK_PATH)
    context = build(tmp_path, spinors=False)
    block = context.results["model"]
    recompute_R, recompute_H = candidates["recompute"]
    assert (block["n_R"], block["n_R_after_ws"], block["n_R_model"]) == (
        9,
        len(recompute_R),
        len(recompute_R),
    )
    assert block["folded_axes"] == [] and block["periodic_dirs"] == [0, 1, 2] and block["dimension"] == 3
    assert block["n_hops"] == independent_hops(recompute_R, 4)
    assert block["spinors"] is True
    assert block["species"] == ["all"] and block["orbital_species"] == ["all"] * 4
    assert block["win_scalars"] == {"dis_win_min": -4.0, "dis_win_max": 4.0}
    assert_array_equal(np.asarray(block["lattice"]), lattice)
    assert_array_equal(context.table_R, recompute_R)
    assert_allclose(context.table_H, recompute_H, rtol=1e-13, atol=1e-15)


def test_spinors_fall_back_to_the_step_context(tmp_path):
    layer_job().write(tmp_path)
    assert build(tmp_path, spinors=True).results["model"]["spinors"] is True
    assert build(tmp_path).results["model"]["spinors"] is False


@pytest.mark.parametrize("truth", ["recompute", "off"])
@pytest.mark.parametrize(("make", "path"), [(layer_job, LAYER_PATH), (bulk_job, BULK_PATH)])
def test_automatic_correction_scores_and_choice_are_the_drivers(generator_core, tmp_path, truth, make, path):
    core = generator_core
    job = make()
    job.write(tmp_path)
    win, _, candidates = driver_tables(core, job, tmp_path)
    kpoints, reference = stage_bands(core, tmp_path, candidates[truth], path)
    best, scores = driver_decision(core, win, candidates, kpoints, reference)
    assert best == truth
    context = build(tmp_path)
    report = context.results["model"]["ws_correction"]
    assert report["requested"] == "auto" and report["mode"] == best and report["selected_by"] == "band_dat"
    assert list(report["scores"]) == ["off", "recompute"]
    for name, score in scores.items():
        assert set(report["scores"][name]) == set(WS_SCORE_KEYS)
        for key in WS_SCORE_KEYS:
            assert report["scores"][name][key] == pytest.approx(score[key], rel=1e-9, abs=1e-9), (name, key)
    if best == "off":
        assert "hermiticity_error" not in report
        assert context.results["model"]["n_R_after_ws"] == context.results["model"]["n_R"]
    else:
        assert report["hermiticity_error"] == pytest.approx(
            core.hermiticity_error(*candidates[best]), abs=1e-15
        )
    assert context.record.warnings == []


def test_a_tie_goes_to_the_recomputed_correction(generator_core, tmp_path, monkeypatch):
    job = layer_job()
    job.write(tmp_path)
    stage_bands(generator_core, tmp_path, job.driver_table(), LAYER_PATH)
    monkeypatch.setattr(wannier_steps, "table_band_energies", lambda R, H, k: np.full((len(k), 2), FERMI))
    report = build(tmp_path).results["model"]["ws_correction"]
    assert report["mode"] == "recompute"
    assert report["scores"]["off"] == report["scores"]["recompute"]


@pytest.mark.parametrize(
    ("hint", "use_ws_distance", "mode", "selected_by"),
    [
        (None, None, "recompute", "wannier90_3_default"),
        (None, ".false.", "off", "wannier90_3_default"),
        (True, None, "recompute", "ws_hint"),
        (True, "F", "off", "ws_hint"),
        (False, ".true.", "off", "ws_hint"),
    ],
)
def test_without_band_files_the_hint_and_the_win_decide(tmp_path, hint, use_ws_distance, mode, selected_by):
    win = {"dis_froz_max": 0.6}
    if use_ws_distance is not None:
        win["use_ws_distance"] = use_ws_distance
    layer_job(win=win).write(tmp_path)
    context = build(tmp_path, ws_hint=hint)
    report = context.results["model"]["ws_correction"]
    assert {key: report[key] for key in ("requested", "mode", "selected_by")} == {
        "requested": "auto",
        "mode": mode,
        "selected_by": selected_by,
    }
    assert ("hermiticity_error" in report) == (mode == "recompute")
    expected = ["ws_assumed_wannier90_default"] if mode == "recompute" else []
    assert context.record.warning_codes() == expected
    assert context.results["model"]["n_R_after_ws"] == (19 if mode == "recompute" else 7)


@pytest.mark.parametrize("requested", ["auto", "recompute", "file", "off"])
def test_without_mp_grid_the_correction_is_off(tmp_path, requested):
    layer_job(mp_grid=None).write(tmp_path)
    context = build(tmp_path, ws_correction=requested)
    block = context.results["model"]
    assert block["ws_correction"] == {"requested": requested, "mode": "off"}
    assert block["mp_grid"] is None and block["n_R_after_ws"] == block["n_R"] == block["n_R_model"] == 7
    assert context.record.warning_codes() == ([] if requested == "off" else ["ws_no_mp_grid"])


def test_a_kpoints_block_supplies_the_mesh_as_in_the_driver(generator_core, tmp_path):
    layer_job(mp_grid=None).write(tmp_path)
    rows = [f"{a / 2} {b / 2} 0.0" for a in range(2) for b in range(2)]
    with (tmp_path / "wannier90.win").open("a") as stream:
        stream.write("begin kpoints\n" + "\n".join(rows) + "\nend kpoints\n")
    block = build(tmp_path).results["model"]
    assert block["mp_grid"] == generator_core.parse_win((tmp_path / "wannier90.win").read_text())["mp_grid"]
    assert block["mp_grid"] == [2, 2, 1] and block["ws_correction"]["mode"] == "recompute"


def test_requested_modes_skip_the_scoring(generator_core, tmp_path):
    job = layer_job()
    job.write(tmp_path)
    _, _, candidates = driver_tables(generator_core, job, tmp_path)
    stage_bands(generator_core, tmp_path, candidates["off"], LAYER_PATH)
    recompute = build(tmp_path, ws_correction="recompute")
    report = recompute.results["model"]["ws_correction"]
    assert set(report) == {"requested", "mode", "hermiticity_error"} and report["mode"] == "recompute"
    assert recompute.results["model"]["n_R_after_ws"] == 19
    off = build(tmp_path, ws_correction="off")
    assert off.results["model"]["ws_correction"] == {"requested": "off", "mode": "off"}
    assert off.record.warnings == []


def write_wsvec(directory, job, lattice):
    """wannier90's ``_wsvec.dat`` with the images QUEER recomputes for ``job``."""
    data = WannierData(job.translations, job.raw, job.degeneracies)
    images = recompute_wsvec(data, job.centres, lattice, job.mp_grid)
    lines = ["## written by the QUEER tests with use_ws_distance=.true."]
    for key, shifts in images.shifts.items():
        lines += [" ".join(str(int(value)) for value in key), str(len(shifts))]
        lines += [" ".join(str(int(value)) for value in shift) for shift in shifts]
    (directory / "wannier90_wsvec.dat").write_text("\n".join(lines) + "\n")


def test_a_staged_wsvec_file_serves_auto_and_file_but_not_recompute(generator_core, tmp_path):
    core = generator_core
    job = layer_job()
    job.write(tmp_path)
    _, lattice, candidates = driver_tables(core, job, tmp_path)
    write_wsvec(tmp_path, job, lattice)
    stage_bands(core, tmp_path, candidates["off"], LAYER_PATH)
    text = (tmp_path / "wannier90_wsvec.dat").read_text()
    expected = core.fold_axes(*core.ws_unfold_from_wsvec(*job.driver_table(), core.parse_wsvec(text)), [2])
    for requested in ("auto", "file"):
        context = build(tmp_path, ws_correction=requested)
        report = context.results["model"]["ws_correction"]
        assert report == {"requested": requested, "mode": "wsvec_file", "hermiticity_error": 0.0}
        assert_array_equal(context.table_R, expected[0])
        assert_allclose(context.table_H, expected[1], rtol=1e-13, atol=1e-15)
        assert "wannier90_wsvec.dat" in context.record.inputs
    recompute = build(tmp_path, ws_correction="recompute")
    assert recompute.results["model"]["ws_correction"]["mode"] == "recompute"
    assert "wannier90_wsvec.dat" not in recompute.record.inputs


@pytest.mark.parametrize(
    ("case", "code"),
    [
        ("missing_hr", ERR_MISSING_INPUT),
        ("missing_centres", ERR_MISSING_INPUT),
        ("missing_win", ERR_MISSING_INPUT),
        ("translate_home_cell", ERR_TRANSLATE_HOME_CELL),
        ("no_cell", ERR_MISSING_INPUT),
        ("no_fermi", ERR_BAD_PARAMS),
        ("chain", ERR_BAD_PARAMS),
        ("one_axis", ERR_BAD_PARAMS),
    ],
)
def test_the_drivers_refusals_keep_their_codes(tmp_path, case, code):
    settings = {}
    if case == "chain":
        chain_job().write(tmp_path)
    elif case == "translate_home_cell":
        layer_job(win={"translate_home_cell": ".true."}).write(tmp_path)
    else:
        layer_job().write(
            tmp_path, hr=case != "missing_hr", centres=case != "missing_centres", win=case != "missing_win"
        )
    if case == "no_cell":
        (tmp_path / "wannier90.win").write_text("num_wann = 2\nmp_grid = 2 2 1\n")
    if case == "no_fermi":
        settings["fermi_energy"] = None
        # The driver refuses before it reads the hopping table.
        (tmp_path / "wannier90_hr.dat").write_text("not a hopping table\n")
    if case == "one_axis":
        settings["periodic_dirs"] = [0]
    with pytest.raises(RunnerError) as caught:
        build(tmp_path, **settings)
    assert caught.value.code == code
    if case == "chain":
        assert "periodic along [0] only" in str(caught.value)


def test_validation_reports_the_drivers_band_deviation(generator_core, tmp_path):
    core = generator_core
    job = layer_job()
    job.write(tmp_path)
    win, _, candidates = driver_tables(core, job, tmp_path)
    translations, table = candidates["recompute"]
    perturbed = table.copy()
    perturbed[np.flatnonzero(~translations.any(axis=1))[0]] += np.diag([0.003, -0.002])
    kpoints, reference = stage_bands(core, tmp_path, (translations, perturbed), LAYER_PATH)
    folded = core.fold_axes(translations, table, [2])
    expected = core.band_deviation(
        core.eigvals_from_table(*folded, kpoints) - FERMI, reference, core.comparison_window(win, FERMI)
    )
    context = build(tmp_path, validation_warn_meV=1.0)
    context.validate()
    validation = context.results["validation"]
    assert set(validation) == VALIDATION_KEYS
    band_dat = validation["band_dat"]
    assert set(band_dat) == BAND_DAT_KEYS
    for key in ("max_meV", "rms_meV", "in_window_max_meV", "in_window_rms_meV"):
        assert band_dat[key] == pytest.approx(expected[key], rel=1e-9), key
    assert band_dat["window_eV"] == pytest.approx(expected["window_eV"])
    assert band_dat["in_window_points"] == expected["in_window_points"] > 0
    assert band_dat["available"] is True and band_dat["n_k"] == len(kpoints)
    assert 1.0 < band_dat["in_window_max_meV"] < 4.0
    assert validation["fermi_in_frozen_window"] is True and validation["hermiticity_error"] == 0.0
    assert context.record.warning_codes() == ["band_dat_deviation"]
    assert f"{band_dat['in_window_max_meV']:.1f} meV" in context.record.warning_messages()[0]
    assert context.record.parallel[-1]["stage"] == "validation"

    failing = build(tmp_path, validation_fail_meV=2.0)
    with pytest.raises(RunnerError) as caught:
        failing.validate()
    assert caught.value.code == ERR_VALIDATION and "(limit 2 meV)" in str(caught.value)
    assert failing.results["validation"]["band_dat"] == band_dat


def test_missing_band_files_and_a_fermi_level_outside_the_frozen_window_warn(tmp_path):
    layer_job(win={"dis_froz_min": -1.0, "dis_froz_max": 0.05}).write(tmp_path)
    context = build(tmp_path)
    context.validate()
    validation = context.results["validation"]
    assert validation["band_dat"] == {"available": False}
    assert validation["fermi_in_frozen_window"] is False
    assert context.record.warning_codes() == [
        "ws_assumed_wannier90_default",
        "band_dat_missing",
        "fermi_outside_frozen_window",
    ]
    assert context.record.parallel == []
    open_window = tmp_path / "open"
    layer_job(win={}).write(open_window)
    unbounded = build(open_window)
    unbounded.validate()
    assert "fermi_in_frozen_window" not in unbounded.results["validation"]


def test_localization_is_the_drivers_decay_across_the_vacuum(generator_core, tmp_path):
    core = generator_core
    job = decaying_job()
    job.write(tmp_path)
    _, lattice, candidates = driver_tables(core, job, tmp_path)
    folded = core.fold_axes(*candidates["recompute"], [2])
    decay = core.hopping_decay(*folded, core.unwrap_centres(job.centres, lattice, [2]), lattice)
    context = build(tmp_path)
    context.validate()
    localization = context.results["validation"]["localization"]
    assert localization["decay_length_A"] == pytest.approx(decay["decay_length_A"], rel=1e-12)
    assert localization["max_distance_A"] == pytest.approx(decay["max_distance_A"], rel=1e-12)
    assert localization["decay_length_A"] == pytest.approx(1.5, rel=0.25)
    assert localization["max_distance_A"] < 10.0
    if importlib.util.find_spec("matplotlib") is not None:
        assert (tmp_path / "Out" / "hopping_decay.png").stat().st_size > 0
        assert context.record.outputs == [
            {"path": "Out/hopping_decay.png", "title": "Hopping decay with distance", "kind": "plot"}
        ]


def test_a_failed_decay_plot_only_warns(tmp_path, monkeypatch):
    figure = pytest.importorskip("matplotlib.figure")

    def refuse(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(figure.Figure, "savefig", refuse)
    decaying_job().write(tmp_path)
    context = build(tmp_path)
    context.validate()
    assert context.record.outputs == []
    assert context.record.warnings[-1] == ("plot_skipped", "hopping-decay plot skipped: disk full")


def test_without_matplotlib_the_decay_plot_is_skipped_silently(tmp_path, monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "matplotlib.figure", None)
    decaying_job().write(tmp_path)
    context = build(tmp_path)
    context.validate()
    assert context.record.outputs == [] and "plot_skipped" not in context.record.warning_codes()
    assert context.results["validation"]["localization"]["decay_length_A"] > 0


def test_occupation_reproduces_the_drivers_gap_classification(generator_core, tmp_path):
    core = generator_core
    job = layer_job()
    job.write(tmp_path)
    _, _, candidates = driver_tables(core, job, tmp_path)
    kpoints, _ = stage_bands(core, tmp_path, candidates["recompute"], LAYER_PATH)
    context = build(tmp_path)
    info = context.resolve_occupation()
    folded = core.fold_axes(*candidates["recompute"], [2])
    points = np.concatenate([core.uniform_mesh([36, 36, 36], [0, 1]), kpoints])
    edges = core.BandEdges(2)
    edges.add(points, core.eigvals_from_table(*folded, points) - FERMI)
    expected = core.classify_gap(edges, 0.02, None)
    assert info["n_occ"] == expected["n_occ"] == 1 and info["metal"] is expected["metal"] is False
    assert info["k_points_sampled"] == expected["k_points_sampled"] == 36 * 36 + len(kpoints)
    for key in ("gap_eV", "direct_gap_eV", "vbm_eV", "cbm_eV"):
        assert info[key] == pytest.approx(expected[key], abs=1e-12), key
    for key in ("vbm_k", "cbm_k", "direct_gap_k"):
        assert_allclose(info[key], expected[key], atol=1e-12)
    assert info["isolation_meV"] == pytest.approx(float(edges.direct[0]) * 1000.0, abs=1e-9)
    assert info["isolated"] is True
    assert context.record.parallel[-1]["stage"] == "occupation"
    assert context.record.parallel[-1]["k_points"] == 36 * 36 + len(kpoints)
    narrow = build(tmp_path, step_type="PYTHTB_TOPOLOGY", isolation_threshold_meV=1e6).resolve_occupation()
    assert narrow["isolated"] is False
    with pytest.raises(RunnerError) as caught:
        build(tmp_path, step_type="PYTHTB_TOPOLOGY", occupied_bands=2).resolve_occupation()
    assert caught.value.code == ERR_BAD_PARAMS


def test_a_bulk_occupation_mesh_and_the_win_path(generator_core, tmp_path):
    core = generator_core
    path = (("G", (0.0, 0.0, 0.0), "Z", (0.0, 0.0, 0.5)), ("Z", (0.0, 0.0, 0.5), "X", (0.5, 0.0, 0.5)))
    job = bulk_job(kpoint_path=path)
    job.write(tmp_path)
    context = build(tmp_path, step_type="PYTHTB_TOPOLOGY")
    band_path = context.band_path()
    expected = core.sample_path([list(row) for row in path], context.lattice, 400)
    assert band_path["source"] == "win_kpoint_path"
    assert_allclose(band_path["kpoints"], expected["kpoints"])
    assert band_path["tick_indices"] == expected["tick_indices"]
    info = context.resolve_occupation()
    assert info["k_points_sampled"] == 16**3 + len(expected["kpoints"])


def test_a_layer_path_from_the_params_drops_its_vacuum_component(generator_core, tmp_path):
    layer_job().write(tmp_path)
    segments = [["G", [0, 0, 0], "A", [0, 0, 0.5]], ["A", [0, 0, 0.5], "M", [0.5, 0, 0.5]]]
    context = build(tmp_path, kpath={"segments": segments, "source": "seekpath"})
    band_path = context.band_path()
    expected = generator_core.sample_path(generator_core.project_path(segments, [0, 1]), context.lattice, 400)
    assert band_path["source"] == "seekpath" and not np.any(band_path["kpoints"][:, 2])
    assert_allclose(band_path["kpoints"], expected["kpoints"])
    bare = tmp_path / "bare"
    layer_job().write(bare)
    with pytest.raises(RunnerError) as caught:
        build(bare).band_path()
    assert caught.value.code == ERR_BAD_PARAMS
    assert build(bare).resolve_occupation()["k_points_sampled"] == 36 * 36


def test_inputs_are_hashed_and_the_stages_logged(generator_core, tmp_path, capsys):
    job = layer_job()
    job.write(tmp_path)
    _, _, candidates = driver_tables(generator_core, job, tmp_path)
    kpoints, _ = stage_bands(generator_core, tmp_path, candidates["recompute"], LAYER_PATH)
    context = WannierStepContext(params(), tmp_path)
    context.build_model()
    context.validate()
    assert list(context.record.inputs) == [
        "wannier90.win",
        "wannier90_hr.dat",
        "wannier90_centres.xyz",
        "wannier90_band.kpt",
        "wannier90_band.dat",
        "wannier90_band.labelinfo.dat",
    ]
    for name, entry in context.record.inputs.items():
        content = (tmp_path / name).read_bytes()
        assert entry == {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    lines = (tmp_path / "queer.log").read_text().splitlines()
    stamp = r"\[\s*\d+\.\ds\] "
    assert re.fullmatch(
        stamp + r"read wannier90_hr\.dat: 2 Wannier functions, 7 R vectors \(\d+\.\ds\)", lines[0]
    )
    assert re.fullmatch(
        stamp + r"model: N=2, periodic \[0, 1\], spinors=False, E_F=0\.1000 eV \(reference_band\), "
        r"WS=recompute \(\d+\.\ds\)",
        lines[1],
    )
    n_k = len(kpoints)
    assert re.fullmatch(
        stamp + rf"validation: {n_k} k-points, 1 task\(s\) of up to {n_k} k on 1 process\(es\)", lines[2]
    )
    assert capsys.readouterr().out.splitlines() == lines


def test_the_lattice_is_the_one_pythtb_reads_from_the_drivers_win(generator_core, tmp_path):
    core = generator_core
    rows = ["4.30912345678912 0.0 0.0", "-2.15456172839456d0 3.73170351234567 0.0", "0 0 81.3580049123456789"]
    for unit in ("ang", "Bohr", None):
        lines = [
            "begin unit_cell_cart",
            *([unit] if unit else []),
            *rows,
            "end unit_cell_cart",
            "mp_grid 2 2 1",
        ]
        text = "\n".join(lines) + "\n"
        path = tmp_path / f"{unit}.win"
        path.write_text(text)
        expected = np.asarray(core.parse_win(core.sanitized_win(core.parse_win(text)))["lattice"])
        assert_array_equal(pythtb_lattice(read_win(path)), expected)
    (tmp_path / "none.win").write_text("num_wann = 2\n")
    assert pythtb_lattice(read_win(tmp_path / "none.win")) is None


def test_the_model_stage_writes_the_generator_schema(generator_core, tmp_path):
    job = layer_job()
    job.write(tmp_path)
    _, _, candidates = driver_tables(generator_core, job, tmp_path)
    stage_bands(generator_core, tmp_path, candidates["recompute"], LAYER_PATH)
    settings = {"step_type": "PYTHTB_BANDS", "fermi_energy": FERMI, "workers": 1, "ws_hint": True}
    params_path = tmp_path / "pythtb_params.json"
    params_path.write_text(json.dumps(settings))
    code = execute(tmp_path, params_path, model_stage, log=lambda line: None)
    document = json.loads((tmp_path / RESULTS_FILENAME).read_text())
    assert code == EXIT_OK, document.get("error")
    assert set(document) == {
        "approximation",
        "citation",
        "counterpart_step_type",
        "engine",
        "model",
        "outputs",
        "provenance",
        "schema_version",
        "status",
        "step_type",
        "validation",
        "warning_codes",
        "warnings",
    }
    assert set(document["model"]) == MODEL_KEYS
    assert set(document["model"]["ws_correction"]) == {
        "requested",
        "mode",
        "selected_by",
        "scores",
        "hermiticity_error",
    }
    assert set(document["validation"]) == VALIDATION_KEYS
    assert set(document["validation"]["band_dat"]) == BAND_DAT_KEYS
    assert list(document["provenance"]["inputs"])[:3] == [
        "wannier90.win",
        "wannier90_hr.dat",
        "wannier90_centres.xyz",
    ]
    assert [plan["stage"] for plan in document["provenance"]["parallel"]] == ["validation"]
    assert document["warnings"] == [] and document["warning_codes"] == []

    with (tmp_path / "wannier90.win").open("a") as stream:
        stream.write("translate_home_cell = true\n")
    code = execute(tmp_path, params_path, model_stage, log=lambda line: None)
    document = json.loads((tmp_path / RESULTS_FILENAME).read_text())
    assert code == EXIT_FAILED and document["status"] == "failed"
    assert document["error"]["code"] == ERR_TRANSLATE_HOME_CELL
    assert "model" not in document and list(document["provenance"]["inputs"]) == ["wannier90.win"]
