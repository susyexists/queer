"""The model stage on a real Generator PYTHTB_BANDS job reproduces the driver's model block.

Opt-in: ``QUEER_REAL_REPLAY=1``. ``QUEER_REAL_REPLAY_DIR`` names the job directory
(default ``data/generator_replay_20261010/job-e228d1af`` of this checkout) holding
``staged-inputs``, ``pythtb_params.json`` and the driver's ``pythtb_results.json``;
``QUEER_REAL_REPLAY_WORKERS`` overrides the file's worker count. The job is a
230-orbital MnBi2Te4 slab whose ``_hr.dat`` is 389 MB.

The job ran a driver revision that measured the hopping decay between the centres
as written, so a hop between orbitals on either side of the wrap line spanned the
vacuum. The stage follows the vendored revision, which unwraps the centres across
the vacuum first; the job's numbers are recovered from the stage's own table with
the centres as written.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from queer.compat.generator.estimators import hopping_decay
from queer.compat.generator.pythtb_params import translate_pythtb
from queer.compat.generator.wannier_steps import WannierStepContext

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
EXACT_MODEL_KEYS = (
    "num_wann",
    "n_R",
    "n_R_after_ws",
    "n_R_model",
    "n_hops",
    "dimension",
    "folded_axes",
    "periodic_dirs",
    "mp_grid",
    "spinors",
    "lattice",
    "species",
    "orbital_species",
    "win_scalars",
    "fermi_energy",
    "fermi_source",
    "source_wannier_job_id",
    "reference_band_job_id",
)
BAND_DAT_VALUES = ("max_meV", "rms_meV", "in_window_max_meV", "in_window_rms_meV")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("QUEER_REAL_REPLAY") != "1",
        reason="the real Generator replay runs with QUEER_REAL_REPLAY=1",
    ),
]


def test_e228d1af_model_stage_matches_the_drivers_results(generator_core, tmp_path):
    job = Path(os.environ.get("QUEER_REAL_REPLAY_DIR") or DEFAULT_JOB)
    staged = job / "staged-inputs"
    assert staged.is_dir(), f"no staged inputs at {staged}; set QUEER_REAL_REPLAY_DIR"
    for name in STAGED:
        if (staged / name).exists():
            (tmp_path / name).symlink_to(staged / name)
    params = json.loads((job / "pythtb_params.json").read_text())
    if os.environ.get("QUEER_REAL_REPLAY_WORKERS"):
        params["workers"] = int(os.environ["QUEER_REAL_REPLAY_WORKERS"])
    expected = json.loads((job / "pythtb_results.json").read_text())

    context = WannierStepContext(translate_pythtb(params), tmp_path)
    context.build_model()
    context.validate()
    model, reference = context.results["model"], expected["model"]
    validation, checked = context.results["validation"], expected["validation"]
    ws, ws_reference = model["ws_correction"], reference["ws_correction"]
    table = (context.table_R, context.table_H)
    unwrapped = generator_core.unwrap_centres(context.centres, context.lattice, context.folded_axes)
    vendored = generator_core.hopping_decay(*table, unwrapped, context.lattice)
    as_written = hopping_decay(*table, context.centres, context.lattice)
    assert as_written is not None
    comparison = {
        "model": {key: [model[key], reference[key]] for key in EXACT_MODEL_KEYS if key != "orbital_species"},
        "ws_correction": {"queer": ws, "driver": ws_reference},
        "band_dat": {key: [validation["band_dat"][key], checked["band_dat"][key]] for key in BAND_DAT_VALUES},
        "localization": {
            "queer": validation["localization"],
            "job_driver": checked["localization"],
            "queer_table_centres_as_written": {key: as_written[key] for key in checked["localization"]},
        },
        "warnings": context.record.warning_messages(),
    }
    print(json.dumps(comparison, indent=1))

    for key in EXACT_MODEL_KEYS:
        assert model[key] == reference[key], key
    assert {key: ws[key] for key in ("requested", "mode", "selected_by")} == {
        key: ws_reference[key] for key in ("requested", "mode", "selected_by")
    }
    for name, scores in ws_reference["scores"].items():
        for key, value in scores.items():
            assert ws["scores"][name][key] == pytest.approx(value, rel=1e-6, abs=1e-9), (name, key)
    assert ws["hermiticity_error"] == pytest.approx(ws_reference["hermiticity_error"], abs=1e-12)
    for key in BAND_DAT_VALUES:
        assert validation["band_dat"][key] == pytest.approx(checked["band_dat"][key], rel=1e-6, abs=1e-9), key
    for key in ("available", "n_k", "in_window_points"):
        assert validation["band_dat"][key] == checked["band_dat"][key], key
    assert validation["band_dat"]["window_eV"] == pytest.approx(checked["band_dat"]["window_eV"], abs=1e-12)
    assert validation["fermi_in_frozen_window"] is checked["fermi_in_frozen_window"]
    assert validation["hermiticity_error"] == pytest.approx(checked["hermiticity_error"], abs=1e-12)
    for key, value in checked["localization"].items():
        assert validation["localization"][key] == pytest.approx(vendored[key], rel=1e-12), key
        assert as_written[key] == pytest.approx(value, rel=1e-9), key
    # Unwrapped, the longest hop stays shorter than the 81 Å cell: the 5-SL slab is about 65 Å thick.
    assert validation["localization"]["max_distance_A"] < float(np.asarray(model["lattice"])[2, 2])
    assert context.record.warning_messages() == expected["warnings"]
