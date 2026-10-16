"""QUEER_BERRY on a real Generator PYTHTB_BERRY job reproduces the driver's Fermi-sea curvature.

Opt-in: ``QUEER_REAL_REPLAY=1``. ``QUEER_BERRY_REPLAY_JOB`` names the job directory
(default ``data/generator_replay_20261010/job-8621166b`` of this checkout) holding
the driver's ``pythtb_params.json``, ``pythtb_results.json``, ``Out/berry_k3_0.json``
and ``Out/berry_path.json``. The job staged the WANNIER job's files that the BANDS
replay stages (``QUEER_REAL_REPLAY_DATA``, default ``data/generator_replay_20261010``
of this checkout: ``job-e228d1af/staged-inputs``) plus that WANNIER job's
``_band.labelinfo.dat`` (``job-48ec1fe7``); their SHA-256 must equal the job's
provenance. ``QUEER_REAL_REPLAY_WORKERS`` overrides the file's worker count and
``QUEER_REPLAY_REPORT`` names a file that receives the comparison as JSON.

The model is a 230-orbital MnBi2Te4 slab with E_F in a band, so the job summed
the Fermi sea. It ran the vendored driver revision: every block compares
directly. The Kubo sum skips occupied-empty pairs closer than 1e-6 eV, so at a
k-point with a level within 1e-6 eV of E_F, or an occupied and an empty level
within 3e-6 eV (the skip threshold plus a 2e-6 eV margin), the last bits of
the eigenvalues can decide which pairs count; the maps and the path are also
compared without those points, and the plane integral is checked to differ by
their share alone.

The parity comparator then judges the job's directory against the replay's: the
verdict must be ``pass`` with every recorded input digest equal.
``QUEER_PARITY_REPORT_DIR`` names a directory that receives the report as JSON
and Markdown.
"""

import json
import os
import shutil
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from queer.compat.generator import runner
from queer.compat.generator.berry import run_berry
from queer.compat.generator.parity import compare, render_markdown
from queer.compat.generator.results import EXIT_OK, RESULTS_FILENAME

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("QUEER_REAL_REPLAY_DATA") or ROOT / "data" / "generator_replay_20261010")
JOB = Path(os.environ.get("QUEER_BERRY_REPLAY_JOB") or DATA / "job-8621166b")
WANNIER_FILES = (
    "wannier90.win",
    "wannier90_hr.dat",
    "wannier90_centres.xyz",
    "wannier90_band.kpt",
    "wannier90_band.dat",
)
LABELINFO = "wannier90_band.labelinfo.dat"
# Leading SHA-256 digits of the inputs the job recorded.
INPUT_DIGESTS = {
    "wannier90.win": "721f89a7",
    "wannier90_hr.dat": "e7cf3d96",
    "wannier90_centres.xyz": "f9537579",
    "wannier90_band.kpt": "43539c72",
    "wannier90_band.dat": "5b8d99f0",
    LABELINFO: "74cf0396",
}
SKIP_GAP_EV = 1e-6
MARGIN_EV = 2e-6
# Away from the points above the two codes evaluate the same sums in another order.
MAP_RELATIVE_TOL = 1e-6
INTEGRAL_TOL = 1e-9
BLOCK_RELATIVE_TOL = 1e-9
QUEER_ONLY_MODEL_KEYS = ("approximation", "pythtb_version", "queer_version")


def save_parity_report(report, name):
    """``<QUEER_PARITY_REPORT_DIR>/<name>.json`` and ``.md`` when the variable is set."""
    directory = os.environ.get("QUEER_PARITY_REPORT_DIR")
    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / f"{name}.json").write_text(json.dumps(report, indent=1) + "\n")
        (Path(directory) / f"{name}.md").write_text(render_markdown(report))


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("QUEER_REAL_REPLAY") != "1",
        reason="the real Generator replay runs with QUEER_REAL_REPLAY=1",
    ),
]


def staged_sources():
    sources = {name: DATA / "job-e228d1af" / "staged-inputs" / name for name in WANNIER_FILES}
    sources[LABELINFO] = DATA / "job-48ec1fe7" / LABELINFO
    return sources


def near_degenerate(levels):
    """k-points where a level is within the skip threshold of E_F or a straddling pair is near it."""
    levels = np.asarray(levels, dtype=float)
    below = np.where(levels < 0.0, levels, -np.inf).max(axis=1)
    above = np.where(levels >= 0.0, levels, np.inf).min(axis=1)
    closest = np.minimum(np.abs(below), np.abs(above))
    return (closest < SKIP_GAP_EV) | (above - below < SKIP_GAP_EV + MARGIN_EV), closest, above - below


def map_difference(produced, reference, near):
    produced, reference = np.asarray(produced, float), np.asarray(reference, float)
    difference = np.abs(produced - reference)
    scale = float(np.abs(reference).max())
    away = difference[~near]
    worst = np.argsort(difference)[::-1][:5]
    return {
        "max_abs_diff": float(difference.max()),
        "max_abs": scale,
        "relative": float(difference.max() / scale),
        "near_degenerate_points": int(near.sum()),
        "max_abs_diff_away": float(away.max()) if len(away) else 0.0,
        "relative_away": float(away.max() / scale) if len(away) else 0.0,
        "largest": [
            {"index": int(index), "abs_diff": float(difference[index]), "near": bool(near[index])}
            for index in worst
        ],
    }


def leaf_differences(produced, expected, prefix=""):
    """``(path, produced, expected)`` of every leaf of ``expected``."""
    if isinstance(expected, dict):
        rows = []
        for key, value in expected.items():
            rows += leaf_differences(
                produced.get(key) if isinstance(produced, dict) else None, value, f"{prefix}.{key}"
            )
        return rows
    if isinstance(expected, list) and expected and all(isinstance(item, (int, float)) for item in expected):
        return [(f"{prefix}[]", produced, expected)]
    return [(prefix, produced, expected)]


def compare_block(produced, expected, skip=()):
    """Exact for every non-float leaf, the largest relative difference of the floats."""
    worst = {"path": None, "relative": 0.0}
    for path, value, reference in leaf_differences(
        {k: v for k, v in produced.items() if k not in skip},
        {k: v for k, v in expected.items() if k not in skip},
    ):
        floats = isinstance(reference, float) or (
            isinstance(reference, list) and any(isinstance(item, float) for item in reference)
        )
        if not floats:
            assert value == reference, (path, value, reference)
            continue
        a, b = np.asarray(value, float), np.asarray(reference, float)
        assert a.shape == b.shape, path
        # Relative to the job's value; absolute where that value is zero.
        scale = np.where(np.abs(b) > 0, np.abs(b), 1.0)
        relative = float(np.max(np.abs(a - b) / scale)) if a.size else 0.0
        if relative > worst["relative"]:
            worst = {"path": path, "relative": relative, "queer": value, "job": reference}
    return worst


def test_8621166b_berry_step_matches_the_drivers_results(tmp_path, monkeypatch):
    sources = staged_sources()
    for name, source in sources.items():
        assert source.is_file(), f"{source} is missing; set QUEER_REAL_REPLAY_DATA"
    assert (JOB / "pythtb_results.json").is_file(), f"no job at {JOB}; set QUEER_BERRY_REPLAY_JOB"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name, source in sources.items():
        (workspace / name).symlink_to(source)
    expected = json.loads((JOB / "pythtb_results.json").read_text())
    params = json.loads((JOB / "pythtb_params.json").read_text())
    assert params["step_type"] == "PYTHTB_BERRY"
    if os.environ.get("QUEER_REAL_REPLAY_WORKERS"):
        params["workers"] = int(os.environ["QUEER_REAL_REPLAY_WORKERS"])
    params_path = tmp_path / "pythtb_params.json"
    params_path.write_text(json.dumps(params))
    captured = {}

    def berry_keeping_the_context(context):
        captured["context"] = context
        run_berry(context)

    monkeypatch.setattr(
        runner,
        "STEP_RUNNERS",
        MappingProxyType({**runner.STEP_RUNNERS, "QUEER_BERRY": berry_keeping_the_context}),
    )

    code = runner.run("PYTHTB_BERRY", params_path, workspace)
    document = json.loads((workspace / RESULTS_FILENAME).read_text())
    assert code == EXIT_OK, document.get("error")
    inputs = document["provenance"]["inputs"]
    assert inputs == expected["provenance"]["inputs"]
    assert {name: entry["sha256"][:8] for name, entry in inputs.items()} == INPUT_DIGESTS

    berry, job_berry = document["berry"], expected["berry"]
    plane = json.loads((workspace / "Out" / "berry_k3_0.json").read_text())
    job_plane = json.loads((JOB / "Out" / "berry_k3_0.json").read_text())
    path = json.loads((workspace / "Out" / "berry_path.json").read_text())
    job_path = json.loads((JOB / "Out" / "berry_path.json").read_text())
    assert plane["k_frac"] == job_plane["k_frac"]

    context = captured["context"]
    band_path = context.band_path()
    points = np.concatenate([np.asarray(plane["k_frac"], float), np.asarray(band_path["kpoints"], float)])
    options = context.parallel("levels at E_F for the replay check", 3, len(points))
    levels = np.asarray(context.model.energies(points, execution=options)).T
    near, closest, gap = near_degenerate(levels)
    mesh = int(berry["mesh"])
    on_plane = slice(0, mesh * mesh)
    on_path = slice(mesh * mesh, None)

    maps = {
        key: map_difference(plane[key], job_plane[key], near[on_plane])
        for key in ("omega_reduced", "omega_cart_A2")
    }
    along = map_difference(path["omega_cart_A2"], job_path["omega_cart_A2"], near[on_path])
    reduced_difference = np.asarray(plane["omega_reduced"]) - np.asarray(job_plane["omega_reduced"])
    near_share = float(reduced_difference[near[on_plane]].sum() / (mesh * mesh * 2.0 * np.pi))
    integral_difference = berry["fermi_sea_integral_e2_h"] - job_berry["fermi_sea_integral_e2_h"]
    largest = int(np.argmax(np.abs(job_plane["omega_cart_A2"])))
    model_worst = compare_block(document["model"], expected["model"], skip=QUEER_ONLY_MODEL_KEYS)
    validation_worst = compare_block(document["validation"], expected["validation"])
    report = {
        "occupation": [berry["occupation"], job_berry["occupation"]],
        "occupation_info_equal": berry["occupation_info"] == job_berry["occupation_info"],
        "fermi_sea_integral_e2_h": [berry["fermi_sea_integral_e2_h"], job_berry["fermi_sea_integral_e2_h"]],
        "fermi_sea_integral_abs_diff": abs(integral_difference),
        "sigma_xy_e2_h_abs_diff": abs(berry["sigma_xy_e2_h"] - job_berry["sigma_xy_e2_h"]),
        "integral_share_of_near_degenerate_points": near_share,
        "integral_abs_diff_away_from_them": abs(integral_difference - near_share),
        "omega_cart_max_A2": [berry["omega_cart_max_A2"], job_berry["omega_cart_max_A2"]],
        "omega_cart_max_abs_diff": abs(berry["omega_cart_max_A2"] - job_berry["omega_cart_max_A2"]),
        "omega_cart_max_at_near_degenerate_point": bool(near[on_plane][largest]),
        "plane_maps": maps,
        "path_omega_cart_A2": along,
        "path_k_distance_max_abs_diff": float(
            np.abs(np.asarray(path["k_distance"]) - np.asarray(job_path["k_distance"])).max()
        ),
        "smallest_level_distance_to_fermi_eV": float(closest.min()),
        "smallest_straddling_gap_eV": float(gap.min()),
        "model_worst_float": model_worst,
        "validation_worst_float": validation_worst,
        "warnings": [document["warnings"], expected["warnings"]],
        "parallel": document["provenance"]["parallel"],
        "wall_seconds": document["provenance"]["wall_seconds"],
        "peak_rss_bytes": document["provenance"]["peak_rss_bytes"],
    }
    text = json.dumps(report, indent=1, default=str)
    print(text)
    if os.environ.get("QUEER_REPLAY_REPORT"):
        Path(os.environ["QUEER_REPLAY_REPORT"]).write_text(text + "\n")

    assert berry["occupation"] == job_berry["occupation"] == "fermi"
    assert berry["occupation_info"] == job_berry["occupation_info"]
    for key in ("fixed_axis", "value", "plane_axes", "mesh", "conventions", "path_file"):
        assert berry[key] == job_berry[key], key
    assert list(berry) == list(job_berry)
    assert berry["sigma_xy_e2_h"] == -berry["fermi_sea_integral_e2_h"]
    assert report["integral_abs_diff_away_from_them"] <= INTEGRAL_TOL
    for key, difference in maps.items():
        assert difference["relative_away"] <= MAP_RELATIVE_TOL, key
    assert along["relative_away"] <= MAP_RELATIVE_TOL
    if not report["omega_cart_max_at_near_degenerate_point"]:
        assert berry["omega_cart_max_A2"] == pytest.approx(
            job_berry["omega_cart_max_A2"], rel=MAP_RELATIVE_TOL
        )
    assert path["tick_indices"] == job_path["tick_indices"] and path["tick_labels"] == job_path["tick_labels"]
    assert report["path_k_distance_max_abs_diff"] <= 1e-12
    assert model_worst["relative"] <= BLOCK_RELATIVE_TOL, model_worst
    assert validation_worst["relative"] <= BLOCK_RELATIVE_TOL, validation_worst
    assert document["warnings"] == expected["warnings"]
    assert [(o["path"], o["title"], o["kind"]) for o in document["outputs"]] == [
        (o["path"], o["title"], o["kind"]) for o in expected["outputs"]
    ]

    shutil.copyfile(params_path, workspace / "pythtb_params.json")
    verdict = compare("QUEER_BERRY", JOB, workspace)
    save_parity_report(verdict, f"{JOB.name}-QUEER_BERRY")
    print(
        json.dumps(
            {key: verdict[key] for key in ("verdict", "semantics", "summary", "flagged_sets")}, indent=1
        )
    )
    assert verdict["verdict"] == "pass", verdict["summary"]
    assert verdict["inputs_verified_by_fingerprint"] is False
    inputs = next(item for item in verdict["preconditions"] if item["name"] == "inputs")
    assert inputs["ok"] and inputs["detail"].startswith("SHA-256 equal for 6 files")
