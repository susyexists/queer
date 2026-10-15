"""QUEER's results documents keep the keys of the Generator PythTB driver's documents.

``tests/fixtures/generator_schema`` holds the key structure, without values, of
the ``pythtb_results.json`` of a real PYTHTB_BANDS job and a real PYTHTB_BERRY
job (a list element is ``[]``). Every key path must appear in the QUEER document
of the same step on an analytic model staged as those jobs were: all six
wannier90 files, the Wigner-Seitz choice scored against ``_band.dat``, the
disentanglement windows and the submitter context. The exceptions are the
documented renames and the one value QUEER does not report. The keys the
Generator's consumers read are checked on every step.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "generator_schema"
sys.path.insert(0, str(ROOT / "tests" / "numerical"))

from tight_binding_models import haldane, kane_mele  # noqa: E402
from wannier90_files import Wannier90Job, exact, line_path, write_band_files  # noqa: E402

from queer.compat.generator.results import RESULTS_FILENAME  # noqa: E402
from queer.compat.generator.runner import run  # noqa: E402

# Driver keys QUEER writes under another name.
RENAMED = {
    ("provenance", "core_sha256"): ("provenance", "queer_record_sha256"),
    ("provenance", "pythtb"): ("provenance", "queer_version"),
}
# The largest reaped worker: QUEER's worker pool outlives the run, so none is reaped when the
# document is written.
NOT_REPORTED = {("provenance", "peak_worker_rss_bytes")}
# Keys QUEER adds to every document.
ADDED = [
    ("engine",),
    ("counterpart_step_type",),
    ("warning_codes",),
    ("provenance", "queer_version"),
    ("provenance", "queer_record_sha256"),
    ("provenance", "build_sha"),
    ("model", "queer_version"),
]
# The keys the Generator reads, per step; every step also has the model-stage ones.
CONSUMED_BY_EVERY_STEP = [
    ("model", "fermi_energy"),
    ("model", "fermi_source"),
    ("validation", "band_dat", "available"),
    ("validation", "band_dat", "in_window_max_meV"),
    ("validation", "band_dat", "in_window_rms_meV"),
    ("outputs", "[]", "path"),
    ("outputs", "[]", "title"),
    ("outputs", "[]", "kind"),
]
CONSUMED = {
    "bands_gapped": [
        ("electronic", "metal"),
        ("electronic", "gap_eV"),
        ("electronic", "is_direct"),
        ("electronic", "n_occ"),
        ("electronic", "dos_at_fermi_per_eV"),
    ],
    "bands_metal": [
        ("electronic", "metal"),
        ("electronic", "gap_eV"),
        ("electronic", "n_occ"),
        ("electronic", "dos_at_fermi_per_eV"),
    ],
    "topology": [
        ("topology", "applicable"),
        ("topology", "planes", "[]", "fixed_axis"),
        ("topology", "planes", "[]", "value"),
        ("topology", "z2_indices"),
    ],
    "berry_bands": [
        ("berry", "occupation"),
        ("berry", "chern_kubo"),
        ("berry", "quantum_weight"),
        ("berry", "conventions"),
    ],
    "berry_fermi": [
        ("berry", "occupation"),
        ("berry", "fermi_sea_integral_e2_h"),
        ("berry", "sigma_xy_e2_h"),
        ("berry", "conventions"),
    ],
    "slab": [
        ("slab", "edge_crossings_at_fermi", "left", "net"),
        ("slab", "edge_crossings_at_fermi", "right", "net"),
    ],
}
CORNERS = [[0, 0, 0], [0.5, 0, 0], [1 / 3, 2 / 3, 0], [0, 0, 0]]
LABELS = ["G", "M", "K", "G"]
STEPS = 6
WIN = {"dis_win_min": -6.0, "dis_win_max": 6.0, "dis_froz_min": -1.0, "dis_froz_max": 1.5}
CONTEXT = {
    "workers": 1,
    "ws_correction": "auto",
    "fermi_source": "reference_band",
    "source_wannier_job_id": "job-wannier",
    "reference_band_job_id": "job-band",
}
SCENARIOS = {
    "bands_metal": (
        "PYTHTB_BANDS",
        "haldane",
        {"fermi_energy": 0.8, "dos_mesh": 8, "fermi_surface_mesh": 12},
    ),
    "bands_gapped": ("PYTHTB_BANDS", "haldane", {"fermi_energy": 0.0, "dos_mesh": 8}),
    "berry_fermi": ("PYTHTB_BERRY", "haldane", {"fermi_energy": 0.8, "berry_mesh": 8}),
    "berry_bands": ("PYTHTB_BERRY", "haldane", {"fermi_energy": 0.0, "berry_mesh": 8}),
    "topology": (
        "PYTHTB_TOPOLOGY",
        "kane_mele",
        {"fermi_energy": 0.0, "spinors": True, "loop_points": 11, "transverse_points": 6},
    ),
    "slab": ("PYTHTB_SLAB", "haldane", {"fermi_energy": 0.0, "slab_kpoints": 12, "slab_cells": 4}),
}
MODELS = {"haldane": lambda: haldane(staggered=0.2), "kane_mele": lambda: kane_mele(1.0)}


def key_paths(value, prefix=()):
    """Every path to a leaf: dict keys, ``[]`` for the elements of a list of objects."""
    if isinstance(value, dict):
        if not value:
            return {prefix}
        paths = set()
        for key, item in value.items():
            paths |= key_paths(item, prefix + (str(key),))
        return paths
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        paths = set()
        for item in value:
            paths |= key_paths(item, prefix + ("[]",))
        return paths
    return {prefix}


def present(value, path):
    """Whether ``path`` exists in ``value``, in every element of a ``[]`` list, which must not be empty."""
    if not path:
        return True
    head, rest = path[0], path[1:]
    if head == "[]":
        return isinstance(value, list) and bool(value) and all(present(item, rest) for item in value)
    return isinstance(value, dict) and head in value and present(value[head], rest)


def run_scenario(core, directory, name):
    """The QUEER document of one scenario on a model staged as a Generator WANNIER job stages it."""
    step, model_name, settings = SCENARIOS[name]
    model = MODELS[model_name]()
    job = Wannier90Job(
        model.translations, model.hopping_matrices, model.lattice, model.centres, mp_grid=(4, 4, 1), win=WIN
    )
    job.write(directory)
    kpoints = exact(line_path(CORNERS, STEPS))
    ticks = [(label, index * STEPS) for index, label in enumerate(LABELS)]
    write_band_files(directory, kpoints, core.eigvals_from_table(*job.driver_table(), kpoints), ticks)
    species = ["B", "N"] if len(job.centres) == 2 else ["B", "B", "N", "N"]
    structure = {"species": species[: len(job.centres)], "cart": job.centres.tolist()}
    values = {"step_type": step, **CONTEXT, "structure": structure, **settings}
    params_path = directory / "pythtb_params.json"
    params_path.write_text(json.dumps(values))
    code = run(None, params_path, directory, log=lambda line: None)
    document = json.loads((directory / RESULTS_FILENAME).read_text())
    assert code == 0, document.get("error")
    return document


@pytest.mark.parametrize(("fixture", "scenario"), [("bands", "bands_metal"), ("berry", "berry_fermi")])
def test_every_driver_key_appears_in_the_queer_document(generator_core, tmp_path, fixture, scenario):
    reference = json.loads((FIXTURES / f"{fixture}_keys.json").read_text())
    document = run_scenario(generator_core, tmp_path, scenario)
    assert document["counterpart_step_type"] == reference["step_type"]
    produced = key_paths(document)
    expected = {tuple(path) for path in reference["key_paths"]}
    missing = sorted(path for path in expected if path not in produced and RENAMED.get(path) not in produced)
    assert missing == sorted(path for path in expected if path in NOT_REPORTED)
    for path in ADDED:
        assert path in produced, path
    assert document["engine"] == "queer"
    assert document["model"]["pythtb_version"] is None
    assert len(document["warning_codes"]) == len(document["warnings"])


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_the_keys_the_generator_reads_are_written(generator_core, tmp_path, scenario):
    document = run_scenario(generator_core, tmp_path, scenario)
    for path in CONSUMED_BY_EVERY_STEP + CONSUMED[scenario]:
        assert present(document, path), path
    assert document["validation"]["band_dat"]["available"] is True
