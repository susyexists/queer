"""QUEER_BANDS reproduces the Generator PythTB driver's run_bands on analytic wannier90 files.

Each test writes a small analytic model as the files a WANNIER job stages, runs
the step through the runner and compares the bands, DOS, PDOS, gap and Fermi
surface with the vendored driver's pure functions (``generator_core``) applied
to eigenvalues of the same table.
"""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from tight_binding_models import haldane
from wannier90_files import Wannier90Job, exact, line_path, write_band_files

from queer.compat.generator.bands import FERMI_SURFACE_UNITS, dos_divisions
from queer.compat.generator.estimators import dos_at_fermi
from queer.compat.generator.results import EXIT_OK, RESULTS_FILENAME
from queer.compat.generator.runner import run

STEPS = 8
LAYER_PATH = [[0, 0, 0], [0.5, 0, 0], [1 / 3, 2 / 3, 0], [0, 0, 0]]
LAYER_LABELS = ["G", "M", "K", "G"]
INSULATOR_KEYS = {
    "n_bands",
    "k_points_sampled",
    "n_occ_source",
    "metal",
    "n_occ",
    "gap_eV",
    "direct_gap_eV",
    "direct_gap_k",
    "is_direct",
    "vbm_eV",
    "cbm_eV",
    "vbm_k",
    "cbm_k",
    "dos_at_fermi_per_eV",
}
METAL_KEYS = {
    "n_bands",
    "k_points_sampled",
    "metal",
    "n_occ",
    "gap_eV",
    "bands_crossing_fermi",
    "fermi_level_outside_model",
    "dos_at_fermi_per_eV",
}
POCKET_FERMI = -3.5


def run_step(directory, **settings):
    """Run the step of ``settings`` in ``directory``; the exit code and the results document."""
    values = {"workers": 1, "ws_correction": "off"}
    values.update(settings)
    params_path = directory / "pythtb_params.json"
    params_path.write_text(json.dumps(values))
    code = run(None, params_path, directory, log=lambda line: None)
    return code, json.loads((directory / RESULTS_FILENAME).read_text())


def layer_job(directory, core):
    """A gapped Haldane layer of two species with wannier90's bands along G-M-K-G."""
    model = haldane(staggered=0.3)
    job = Wannier90Job(model.translations, model.hopping_matrices, model.lattice, model.centres)
    job.write(directory)
    kpoints = exact(line_path(LAYER_PATH, STEPS))
    ticks = [(label, index * STEPS) for index, label in enumerate(LAYER_LABELS)]
    write_band_files(directory, kpoints, core.eigvals_from_table(*job.driver_table(), kpoints), ticks)
    structure = {"species": ["B", "N"], "cart": job.centres.tolist()}
    return job, kpoints, structure


def pocket_job(directory):
    """One orbital on a square layer with nearest-neighbour hops: an electron pocket around Gamma."""
    translations = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
    table = np.array([[[0.0]], [[-1.0]], [[-1.0]], [[-1.0]], [[-1.0]]], complex)
    job = Wannier90Job(translations, table, np.diag([2.0, 2.0, 10.0]), np.array([[1.0, 1.0, 5.0]]))
    job.write(directory)
    return job


def pocket_energy(kpoints):
    """E(k) - E_F of the pocket model at fractional ``kpoints``."""
    k = 2 * np.pi * np.asarray(kpoints, float)
    return -2.0 * (np.cos(k[..., 0]) + np.cos(k[..., 1])) - POCKET_FERMI


POCKET_PATH = {
    "segments": [
        ["G", [0, 0, 0], "X", [0.5, 0, 0]],
        ["X", [0.5, 0, 0], "M", [0.5, 0.5, 0]],
        ["M", [0.5, 0.5, 0], "G", [0, 0, 0]],
    ]
}


def driver_dos(core, table, path_kpoints, divisions, species_index, fermi, step=0.005, sigma=0.03):
    """The driver's DOS estimators on eigenvalues of ``table``: grid, centres, DOS, PDOS and edges."""
    mesh = core.uniform_mesh(divisions, [0, 1])
    path_energies = core.eigvals_from_table(*table, path_kpoints) - fermi
    values, vectors = np.linalg.eigh(core.hk_from_table(*table, mesh))
    energies = values - fermi
    weights = np.abs(np.swapaxes(vectors, 1, 2)) ** 2  # (k, band, orbital), as PythTB returns them
    grid = np.arange(path_energies.min() - 2.0, path_energies.max() + 2.0 + step, step)
    total = np.histogram(energies.ravel(), bins=grid)[0]
    parts = np.stack(
        [
            np.histogram(
                energies.ravel(), bins=grid, weights=weights[:, :, species_index == s].sum(-1).ravel()
            )[0]
            for s in range(int(species_index.max()) + 1)
        ]
    )
    edges = core.BandEdges(energies.shape[1])
    edges.add(mesh, energies)
    on_path = core.BandEdges(energies.shape[1])
    on_path.add(path_kpoints, path_energies)
    edges.merge(on_path)
    return {
        "mesh": mesh,
        "centres": 0.5 * (grid[1:] + grid[:-1]),
        "dos": core.gaussian_dos(total / len(mesh), step, sigma),
        "pdos": core.gaussian_dos(parts / len(mesh), step, sigma),
        "gap": core.classify_gap(edges, 0.02, None),
    }


@pytest.mark.parametrize(
    ("n_orb", "dimension", "requested", "expected"),
    [
        (2, 2, None, 300),
        (90, 2, None, 225),
        (230, 2, None, 150),
        (2, 3, None, 60),
        (85, 3, None, 42),
        (200, 3, None, 24),
        (230, 2, 12, 12),
    ],
)
def test_the_dos_mesh_follows_the_drivers_size_rule(n_orb, dimension, requested, expected):
    assert dos_divisions(n_orb, dimension, requested) == [expected] * 3


def test_a_gapped_layer_reproduces_the_drivers_bands_dos_and_gap(generator_core, tmp_path):
    core = generator_core
    job, kpoints, structure = layer_job(tmp_path, core)
    code, document = run_step(
        tmp_path, step_type="PYTHTB_BANDS", fermi_energy=0.0, dos_mesh=30, structure=structure
    )
    assert code == EXIT_OK, document.get("error")
    assert document["status"] == "ok" and document["step_type"] == "QUEER_BANDS"
    assert document["counterpart_step_type"] == "PYTHTB_BANDS"

    table = job.driver_table()
    bands = document["bands"]
    assert_allclose(bands["eigenvalues_eV"], core.eigvals_from_table(*table, kpoints), atol=1e-12)
    assert_array_equal(bands["kpoints_frac"], kpoints)
    assert bands["tick_indices"] == [0, 8, 16, 24] and bands["tick_labels"] == LAYER_LABELS
    assert bands["break_indices"] == [] and bands["source"] == "wannier90_band_kpt"
    assert bands["fermi_energy"] == 0.0

    expected = driver_dos(core, table, kpoints, [30, 30, 30], np.array([0, 1]), 0.0)
    data = json.loads((tmp_path / "Out" / "dos.json").read_text())
    assert_allclose(data["energy_eV"], expected["centres"], rtol=0, atol=1e-12)
    assert_allclose(data["dos_per_eV"], expected["dos"], rtol=0, atol=1e-9)
    assert list(data["pdos_per_eV"]) == ["B", "N"]
    for index, name in enumerate(["B", "N"]):
        assert_allclose(data["pdos_per_eV"][name], expected["pdos"][index], rtol=0, atol=1e-9)
    assert_allclose(np.sum(list(data["pdos_per_eV"].values()), axis=0), data["dos_per_eV"], atol=1e-9)
    assert document["dos"] == {
        "mesh": [30, 30],
        "mesh_axes": [0, 1],
        "k_points": 900,
        "sigma_eV": 0.03,
        "integral": pytest.approx(2.0, abs=1e-9),
        "species": ["B", "N"],
        "file": "Out/dos.json",
    }

    electronic = document["electronic"]
    assert set(electronic) == INSULATOR_KEYS
    gap = expected["gap"]
    assert electronic["n_occ"] == gap["n_occ"] == 1 and electronic["metal"] is gap["metal"] is False
    assert electronic["k_points_sampled"] == gap["k_points_sampled"] == 900 + len(kpoints)
    for key in ("gap_eV", "direct_gap_eV", "vbm_eV", "cbm_eV"):
        assert electronic[key] == pytest.approx(gap[key], abs=1e-12), key
    for key in ("vbm_k", "cbm_k", "direct_gap_k"):
        assert_allclose(electronic[key], gap[key], atol=1e-12)
    assert electronic["dos_at_fermi_per_eV"] == pytest.approx(
        float(np.interp(0.0, expected["centres"], expected["dos"])), abs=1e-12
    )

    assert "fermi_surface" not in document
    assert [(item["path"], item["title"], item["kind"]) for item in document["outputs"]] == [
        ("Out/hopping_decay.png", "Hopping decay with distance", "plot"),
        ("Out/dos.json", "DOS and species PDOS (data)", "data"),
        ("Out/bands.png", "Band structure", "plot"),
        ("Out/dos.png", "Density of states", "plot"),
    ]
    for item in document["outputs"]:
        assert (tmp_path / item["path"]).stat().st_size > 0
    stages = [plan["stage"] for plan in document["provenance"]["parallel"]]
    assert stages == ["validation", "band path", "DOS"]
    assert document["warnings"] == []


def test_a_metal_with_a_gamma_pocket_has_one_closed_fermi_contour(tmp_path):
    pocket_job(tmp_path)
    code, document = run_step(
        tmp_path,
        step_type="QUEER_BANDS",
        fermi_energy=POCKET_FERMI,
        kpath=POCKET_PATH,
        kpath_points=60,
        dos_mesh=40,
        fermi_surface_mesh=60,
    )
    assert code == EXIT_OK, document.get("error")
    electronic = document["electronic"]
    assert set(electronic) == METAL_KEYS
    assert electronic["metal"] is True and electronic["n_occ"] is None
    assert electronic["bands_crossing_fermi"] == [0] and electronic["fermi_level_outside_model"] is False

    bands = document["bands"]
    assert bands["source"] == "params" and bands["tick_labels"] == ["G", "X", "M", "G"]
    assert_allclose(
        np.asarray(bands["eigenvalues_eV"])[:, 0],
        pocket_energy(bands["kpoints_frac"]) + POCKET_FERMI,
        atol=1e-12,
    )

    data = json.loads((tmp_path / "Out" / "dos.json").read_text())
    centres, dos = np.asarray(data["energy_eV"]), np.asarray(data["dos_per_eV"])
    linear = dos_at_fermi(centres, dos, "linear_interpolation")
    assert electronic["dos_at_fermi_per_eV"] == pytest.approx(linear, abs=1e-15)
    assert linear != dos_at_fermi(centres, dos, "first_nonnegative_bin")
    assert document["dos"]["species"] == [] and data["pdos_per_eV"] == {}
    assert document["dos"]["integral"] == pytest.approx(1.0, abs=1e-9)

    assert document["fermi_surface"] == {
        "fixed_axis": 2,
        "value": 0.0,
        "bands_crossing": [0],
        "file": "Out/fermi_surface.json",
    }
    surface = json.loads((tmp_path / "Out" / "fermi_surface.json").read_text())
    assert surface["mesh"] == 60 and surface["bands"] == [0]
    assert surface["k_frac_range"] == [-0.5, 0.5] and surface["k_frac_range_end"] == "excluded"
    assert surface["units"] == FERMI_SURFACE_UNITS
    axes = np.asarray(surface["axes_cart"])
    assert_allclose(axes, [[1, 0, 0], [0, 1, 0]], atol=1e-12)
    (contour,) = surface["contours_cart"]["0"]
    points = np.asarray(contour)
    assert_allclose(points[0], points[-1], atol=1e-12)
    # Back to fractional k: the plane frame's k carries no 2 pi, so k_frac = k_cart . a_i.
    lattice = np.diag([2.0, 2.0, 10.0])
    kpoints = (points @ axes) @ lattice.T
    assert np.abs(pocket_energy(kpoints)).max() < 5e-3
    radius = np.linalg.norm(points, axis=1)
    assert_allclose(radius, np.arccos(0.75) / (2 * np.pi * 2.0), rtol=0.02)

    assert [item["path"] for item in document["outputs"]] == [
        "Out/hopping_decay.png",
        "Out/dos.json",
        "Out/bands.png",
        "Out/dos.png",
        "Out/fermi_surface.png",
        "Out/fermi_surface.json",
    ]
    assert document["outputs"][4]["title"] == "Fermi surface contours"
    assert document["outputs"][5]["title"] == "Fermi-surface contours (data)"
    stages = [plan["stage"] for plan in document["provenance"]["parallel"]]
    assert stages == ["band path", "DOS", "Fermi surface"]
    assert document["warning_codes"] == ["band_dat_missing"]


def test_the_fermi_surface_can_be_turned_off(tmp_path):
    pocket_job(tmp_path)
    code, document = run_step(
        tmp_path,
        step_type="QUEER_BANDS",
        fermi_energy=POCKET_FERMI,
        kpath=POCKET_PATH,
        dos_mesh=12,
        fermi_surface=False,
    )
    assert code == EXIT_OK and document["electronic"]["metal"] is True
    assert "fermi_surface" not in document and not (tmp_path / "Out" / "fermi_surface.json").exists()


def test_failed_plots_only_warn_and_keep_the_contours(tmp_path, monkeypatch):
    figure = pytest.importorskip("matplotlib.figure")

    def refuse(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(figure.Figure, "savefig", refuse)
    pocket_job(tmp_path)
    code, document = run_step(
        tmp_path,
        step_type="QUEER_BANDS",
        fermi_energy=POCKET_FERMI,
        kpath=POCKET_PATH,
        dos_mesh=12,
        fermi_surface_mesh=40,
    )
    assert code == EXIT_OK
    assert [item["path"] for item in document["outputs"]] == ["Out/dos.json", "Out/fermi_surface.json"]
    plot_warnings = [
        message
        for code, message in zip(document["warning_codes"], document["warnings"])
        if code == "plot_skipped"
    ]
    assert plot_warnings == [
        "hopping-decay plot skipped: disk full",
        "band plot skipped: disk full",
        "DOS plot skipped: disk full",
        "Fermi-surface plot skipped: disk full",
    ]
    surface = json.loads((tmp_path / "Out" / "fermi_surface.json").read_text())
    assert len(surface["contours_cart"]["0"]) == 1


def test_without_matplotlib_the_plots_are_skipped_and_the_contours_empty(tmp_path, monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "matplotlib.figure", None)
    pocket_job(tmp_path)
    code, document = run_step(
        tmp_path,
        step_type="QUEER_BANDS",
        fermi_energy=POCKET_FERMI,
        kpath=POCKET_PATH,
        dos_mesh=12,
        fermi_surface_mesh=40,
    )
    assert code == EXIT_OK and "plot_skipped" not in document["warning_codes"]
    assert [item["path"] for item in document["outputs"]] == ["Out/dos.json", "Out/fermi_surface.json"]
    surface = json.loads((tmp_path / "Out" / "fermi_surface.json").read_text())
    assert surface["contours_cart"] == {} and surface["bands"] == [0]
    assert document["fermi_surface"]["bands_crossing"] == [0]
