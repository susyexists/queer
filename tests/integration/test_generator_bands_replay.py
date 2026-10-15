"""QUEER_BANDS on a real Generator PYTHTB_BANDS job reproduces the driver's bands, DOS and gap.

Opt-in: ``QUEER_REAL_REPLAY=1``. ``QUEER_REAL_REPLAY_DIR`` names the job directory
(default ``data/generator_replay_20261010/job-e228d1af`` of this checkout) holding
``staged-inputs``, ``pythtb_params.json`` and the driver's ``pythtb_results.json``
and ``Out/dos.json``; ``QUEER_REAL_REPLAY_WORKERS`` overrides the file's worker
count. The job is a 230-orbital MnBi2Te4 slab whose ``_hr.dat`` is 389 MB.

The job ran a driver revision that took DOS(E_F) from the first bin centred at
or above E_F and wrote its Fermi-surface contours without a plane frame. The
step interpolates DOS(E_F) linearly, so the job's value is recovered with the
historical rule on the step's own DOS, and the Fermi surface is compared by its
crossing bands only.

The job's band path is wannier90's own ``_band.kpt`` samples, which the driver
takes only with the ticks of ``_band.labelinfo.dat``. A replay copy without that
file gets it back from the ticks the job's results record; the reader uses only
each row's label and index.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from queer.compat.generator.estimators import DOS_AT_FERMI_RULES, dos_at_fermi
from queer.compat.generator.results import EXIT_OK, RESULTS_FILENAME
from queer.compat.generator.runner import run
from queer.io.wannier import read_band_kpt

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOB = ROOT / "data" / "generator_replay_20261010" / "job-e228d1af"
STAGED = (
    "wannier90.win",
    "wannier90_hr.dat",
    "wannier90_centres.xyz",
    "wannier90_band.kpt",
    "wannier90_band.dat",
    "wannier90_band.labelinfo.dat",
    "wannier90_wsvec.dat",
)
ELECTRONIC_KEYS = (
    "n_bands",
    "k_points_sampled",
    "metal",
    "n_occ",
    "gap_eV",
    "bands_crossing_fermi",
    "fermi_level_outside_model",
)
EIGENVALUE_TOL_EV = 1e-8
# The same estimator on eigenvalues equal to rounding: a mesh eigenvalue that
# crossed a bin edge would move the DOS by about 1e-4 states/eV.
DOS_TOL = 1e-9


def restore_labelinfo(workspace, bands):
    """``wannier90_band.labelinfo.dat`` with the ticks of the job's ``bands`` block, rows ``LABEL index x k``."""
    kpoints = read_band_kpt(workspace / "wannier90_band.kpt")
    rows = [
        f"{label} {index + 1} 0.0 " + " ".join(f"{value:.6f}" for value in kpoints[index])
        for index, label in zip(bands["tick_indices"], bands["tick_labels"])
    ]
    (workspace / "wannier90_band.labelinfo.dat").write_text("\n".join(rows) + "\n")
    return np.asarray(kpoints)


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("QUEER_REAL_REPLAY") != "1",
        reason="the real Generator replay runs with QUEER_REAL_REPLAY=1",
    ),
]


def test_e228d1af_bands_step_matches_the_drivers_results(tmp_path):
    job = Path(os.environ.get("QUEER_REAL_REPLAY_DIR") or DEFAULT_JOB)
    staged = job / "staged-inputs"
    assert staged.is_dir(), f"no staged inputs at {staged}; set QUEER_REAL_REPLAY_DIR"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name in STAGED:
        if (staged / name).exists():
            (workspace / name).symlink_to(staged / name)
    expected = json.loads((job / "pythtb_results.json").read_text())
    if not (workspace / "wannier90_band.labelinfo.dat").exists():
        kpoints = restore_labelinfo(workspace, expected["bands"])
        # The recorded ticks sit on the .win path's corners K2 = -(1/6, 1/6), G and K2 = (1/6, 1/6).
        corners = kpoints[expected["bands"]["tick_indices"]]
        assert np.allclose(corners, [[-1 / 6, -1 / 6, 0], [0, 0, 0], [1 / 6, 1 / 6, 0]], atol=1e-5)
    params = json.loads((job / "pythtb_params.json").read_text())
    if os.environ.get("QUEER_REAL_REPLAY_WORKERS"):
        params["workers"] = int(os.environ["QUEER_REAL_REPLAY_WORKERS"])
    params_path = tmp_path / "pythtb_params.json"
    params_path.write_text(json.dumps(params))

    code = run("PYTHTB_BANDS", params_path, workspace)
    document = json.loads((workspace / RESULTS_FILENAME).read_text())
    assert code == EXIT_OK, document.get("error")
    dos_file = json.loads((workspace / "Out" / "dos.json").read_text())
    expected_dos_file = json.loads((job / "Out" / "dos.json").read_text())

    bands, expected_bands = document["bands"], expected["bands"]
    eigenvalue_error = float(
        np.abs(np.asarray(bands["eigenvalues_eV"]) - np.asarray(expected_bands["eigenvalues_eV"])).max()
    )
    centres = np.asarray(dos_file["energy_eV"])
    expected_centres = np.asarray(expected_dos_file["energy_eV"])
    same_grid = centres.shape == expected_centres.shape
    dos = np.asarray(dos_file["dos_per_eV"])
    expected_dos = np.asarray(expected_dos_file["dos_per_eV"])
    rules = {rule: dos_at_fermi(centres, dos, rule) for rule in DOS_AT_FERMI_RULES}
    reference_value = expected["electronic"]["dos_at_fermi_per_eV"]
    comparison = {
        "max_abs_eigenvalue_diff_eV": eigenvalue_error,
        "grid_points": [len(centres), len(expected_centres)],
        "max_abs_grid_diff_eV": float(np.abs(centres - expected_centres).max()) if same_grid else None,
        "max_abs_dos_diff_per_eV": float(np.abs(dos - expected_dos).max()) if same_grid else None,
        "max_dos_per_eV": float(expected_dos.max()),
        "max_abs_pdos_diff_per_eV": {
            name: float(np.abs(np.asarray(values) - np.asarray(expected_dos_file["pdos_per_eV"][name])).max())
            for name, values in dos_file["pdos_per_eV"].items()
        }
        if same_grid
        else None,
        "dos_at_fermi_per_eV": {
            "reported": document["electronic"]["dos_at_fermi_per_eV"],
            **rules,
            "job": reference_value,
        },
        "electronic": {
            key: [document["electronic"][key], expected["electronic"][key]] for key in ELECTRONIC_KEYS
        },
        "dos_block": {"queer": document["dos"], "job": expected["dos"]},
        "fermi_surface_bands": [
            document["fermi_surface"]["bands_crossing"],
            expected["fermi_surface"]["bands_crossing"],
        ],
        "warnings": document["warnings"],
        "parallel": document["provenance"]["parallel"],
        "wall_seconds": document["provenance"]["wall_seconds"],
        "peak_rss_bytes": document["provenance"]["peak_rss_bytes"],
    }
    print(json.dumps(comparison, indent=1))

    assert eigenvalue_error <= EIGENVALUE_TOL_EV
    assert np.array_equal(bands["kpoints_frac"], expected_bands["kpoints_frac"])
    for key in ("tick_indices", "tick_labels", "break_indices", "source", "fermi_energy"):
        assert bands[key] == expected_bands[key], key
    assert same_grid and comparison["max_abs_grid_diff_eV"] <= EIGENVALUE_TOL_EV
    assert comparison["max_abs_dos_diff_per_eV"] <= DOS_TOL
    assert list(dos_file["pdos_per_eV"]) == list(expected_dos_file["pdos_per_eV"])
    assert max(comparison["max_abs_pdos_diff_per_eV"].values()) <= DOS_TOL
    for key in ("k_points", "sigma_eV", "species", "file"):
        assert document["dos"][key] == expected["dos"][key], key
    assert document["dos"]["integral"] == pytest.approx(expected["dos"]["integral"], abs=1e-9)
    for key in ELECTRONIC_KEYS:
        assert document["electronic"][key] == expected["electronic"][key], key
    assert document["electronic"]["dos_at_fermi_per_eV"] == rules["linear_interpolation"]
    assert rules["first_nonnegative_bin"] == pytest.approx(reference_value, abs=DOS_TOL)
    assert document["fermi_surface"]["bands_crossing"] == expected["fermi_surface"]["bands_crossing"]
    assert document["warnings"] == expected["warnings"]
