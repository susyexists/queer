"""QUEER_QGEOM: the runner's contract, failure codes and replay switch, and the comparator's rules.

The step runs on the analytic quantum-geometry cases of ``generator_golden_cases``
(staged as a Generator WANNIER job stages them). The comparator is checked against
the vendored driver's goldens (``tests/fixtures/generator_goldens``): each negative
control (a theta0 shifted by one degree, a doubled Chern number, a flipped K_in, a
flipped or zeroed chi, a mu_CNP moved onto a band, a dropped vertex, vertex keys off
their digest, another stop reason) must not pass, and a reference whose sampling
cannot be replayed is not comparable. What only rounding decides (theta0 where A3 is
within M's tolerance, mu_CNP inside a gap, the mesh of an exactly symmetric model) is
never a failure: some cases rerun under other OpenBLAS kernels (``OPENBLAS_CORETYPE``)
in a subprocess.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from tight_binding_models import haldane, qwz, rice_mele
from wannier90_files import Wannier90Job
from wannier90_tb import write_tb

ROOT = Path(__file__).resolve().parents[2]
GOLDENS = ROOT / "tests" / "fixtures" / "generator_goldens"
SCHEMA = ROOT / "tests" / "fixtures" / "generator_schema" / "qgeom_keys.json"
REAL_REFERENCE = Path.home() / ".cache" / "queer-parity-data" / "job-1cd360d9"
sys.path.insert(0, str(ROOT / "tests" / "numerical"))

from generator_golden_cases import write_case  # noqa: E402

from queer.compat.generator.environment import (  # noqa: E402
    BLAS_FIELDS,
    ENVIRONMENT_FIELDS,
    environment_differences,
)
from queer.compat.generator.parity import (  # noqa: E402
    AGREE,
    DOCUMENTED,
    FAIL,
    NOT_APPLICABLE,
    NOT_COMPARABLE,
    PASS,
    PASS_DOCUMENTED,
    UNEXPLAINED,
    compare,
)
from queer.compat.generator.parity_qgeom import CHI_PER_M, CHI_SETS, MESH_ROUNDING  # noqa: E402
from queer.compat.generator.pythtb_params import translate_pythtb  # noqa: E402
from queer.compat.generator.qgeom_mesh import (  # noqa: E402
    REMEDY,
    MeshRun,
    attribute,
    creating_splits,
    history_from_record,
    record_problems,
)
from queer.compat.generator.qgeom_step import compact, decimate  # noqa: E402
from queer.compat.generator.runner import run  # noqa: E402
from queer.compat.generator.wannier_steps import WannierStepContext  # noqa: E402
from queer.observables.qgeom import OUTCOMES, TIE_ROUNDING, seed_id, set_digest  # noqa: E402

PASSING = (PASS, PASS_DOCUMENTED)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value))


def run_case(name, directory, **overrides):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    params = write_case(name, directory, **overrides)
    code = run(None, params, directory, log=lambda line: None)
    return code, read(directory / "queer_results.json")


@pytest.fixture(scope="module")
def metal(tmp_path_factory):
    """QUEER's run of the full-level metal case next to the driver's golden."""
    directory = tmp_path_factory.mktemp("qgeom") / "metal"
    code, document = run_case("qgeom_full_metal", directory)
    assert code == 0, document.get("error")
    return GOLDENS / "qgeom_full_metal", directory


@pytest.fixture(scope="module")
def insulator(tmp_path_factory):
    directory = tmp_path_factory.mktemp("qgeom") / "insulator"
    code, document = run_case("qgeom_full_insulator", directory)
    assert code == 0, document.get("error")
    return GOLDENS / "qgeom_full_insulator", directory


def write_response(directory, response):
    """An edited response written as a run writes it: 5 significant digits, and unrounded where it has that."""
    write(Path(directory) / "Out" / "qgeom_response.json", compact(response))
    full_path = Path(directory) / "Out" / "qgeom_response_full.json"
    if full_path.exists():
        full = read(full_path)
        full.update(mu_eV=response["mu_eV"], per_T=response["per_T"])
        write(full_path, full)


def copy(source, target):
    shutil.copytree(source, target)
    return target


def edit(directory, mutate, name="queer_results.json"):
    path = Path(directory) / name
    document = read(path)
    mutate(document)
    write(path, document)


def quantities(report):
    return {item["name"]: item["class"] for item in report["quantities"]}


def preconditions(report):
    return {item["name"]: item for item in report["preconditions"]}


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


def test_the_step_writes_the_driver_blocks_and_queers_replay_record(metal):
    _, directory = metal
    document = read(directory / "queer_results.json")
    assert document["engine"] == "queer" and document["step_type"] == "QUEER_QGEOM"
    assert document["counterpart_step_type"] == "PYTHTB_QGEOM"
    qgeom = document["qgeom"]
    assert (
        qgeom["position"]["mode"] == "full"
        and qgeom["approximation_level"] == "full_position_in_wannier_space"
    )
    integration = qgeom["integration"]
    assert integration["replay"] == {"time_stop": True, "time_budget_s": 3600.0, "stop_vertices": None}
    assert (
        integration["vertex_set"]["count"] == integration["vertices"] and integration["stop_vertices"] is None
    )
    record = read(directory / "Out" / "qgeom_vertices.json")
    assert record["count"] == integration["vertices"] and len(record["keys"]) == record["count"]
    assert record["sha256"] == integration["vertex_set"]["sha256"]
    assert qgeom["parity"]["check"] == "internal_consistency" and qgeom["parity"]["max_rel"] < 1e-12
    assert qgeom["dft_check"]["counts_match"] and qgeom["dft_check"]["max_meV"] < 1e-3
    assert document["provenance"]["inputs"]["wannier90_tb.dat"]["sha256"]
    assert document["provenance"]["inputs"]["wannier90.eig"]["sha256"]
    assert len(document["provenance"]["queer_qgeom_sha256"]) == 64
    paths = [entry["path"] for entry in document["outputs"]]
    for name in (
        "qgeom_response.json",
        "qgeom_response_full.json",
        "qgeom_path.json",
        "qgeom_vertices.json",
        "qgeom_dipole.png",
    ):
        assert f"Out/{name}" in paths
    assert len(document["warning_codes"]) == len(document["warnings"])


def key_paths(value, prefix=()):
    """Every path to a leaf (``[]`` for list elements that are objects, ``<T>`` for temperature keys)."""
    import re

    if isinstance(value, dict):
        if not value:
            return {prefix}
        paths = set()
        for key, item in value.items():
            key = "<T>" if re.fullmatch(r"\d+(\.\d+)?K", str(key)) else str(key)
            paths |= key_paths(item, prefix + (key,))
        return paths
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        paths = set()
        for item in value:
            paths |= key_paths(item, prefix + ("[]",))
        return paths
    return {prefix}


def covered(path, produced):
    return path in produced or any(other[: len(path)] == path for other in produced)


# Driver keys QUEER writes under another name, and the ones it does not report.
RENAMED = {
    ("provenance", "core_sha256"): ("provenance", "queer_record_sha256"),
    ("provenance", "pythtb"): ("provenance", "queer_version"),
    ("provenance", "qgeom_sha256"): ("provenance", "queer_qgeom_sha256"),
}
NOT_REPORTED = {("provenance", "peak_worker_rss_bytes")}
# QUEER runs the .eig check the driver always skips (the insulator case stages no .eig).
PRODUCER_SPECIFIC = {("qgeom", "dft_check", "skipped")}
# Written by both producers only when a Wigner-Seitz correction is applied.
SCENARIO_DEPENDENT = {("model", "ws_correction", "hermiticity_error")}


def test_the_queer_document_carries_every_key_of_the_real_reference(tmp_path):
    """Every key path of job-1cd360d9 (PYTHTB_QGEOM, full level) appears in QUEER's documents.

    The case is run as that job ran: the full level, an integration that stopped
    before it was confirmed (here by the replay switch) and an occupied manifold with
    its gap open. Its Wigner-Seitz choice is off, where neither producer records a
    corrected table's Hermiticity error.
    """
    fixture = read(SCHEMA)
    assert fixture["step_type"] == "PYTHTB_QGEOM"
    code, document = run_case("qgeom_full_insulator", tmp_path, qgeom_replay_vertices=3000)
    assert code == 0, document.get("error")
    assert document["qgeom"]["occupied"]["verdict"] == "unconverged"
    produced = key_paths(document)
    missing = [
        tuple(path)
        for path in fixture["key_paths"]
        if not covered(tuple(path), produced)
        and (tuple(path) not in RENAMED or not covered(RENAMED[tuple(path)], produced))
    ]
    assert sorted(missing) == sorted(NOT_REPORTED | PRODUCER_SPECIFIC | SCENARIO_DEPENDENT)
    for name, paths in fixture["out_files"].items():
        produced = key_paths(read(tmp_path / "Out" / name))
        assert [path for path in paths if not covered(tuple(path), produced)] == [], name


@pytest.mark.skipif(not REAL_REFERENCE.is_dir(), reason="the real reference is not downloaded")
def test_the_real_reference_parameters_resolve_and_its_keys_are_the_fixtures():
    fixture = read(SCHEMA)
    reference = read(REAL_REFERENCE / "pythtb_results.json")
    assert sorted(map(list, key_paths(reference))) == fixture["key_paths"]
    for name, paths in fixture["out_files"].items():
        assert sorted(map(list, key_paths(read(REAL_REFERENCE / "Out" / name)))) == paths
    parameters = translate_pythtb(read(REAL_REFERENCE / "pythtb_params.json"))
    assert parameters.step == "QUEER_QGEOM" and parameters["time_budget_s"] == 10080
    integration = reference["qgeom"]["integration"]
    # It stopped on its vertex budget, not its clock: its sampling replays without a vertex count.
    assert integration["stop"] == "budget" and integration["vertices"] == 1041183
    assert reference["provenance"]["qgeom_sha256"].startswith("4fb51a38")


@pytest.mark.parametrize(
    ("settings", "code", "fragment"),
    [
        (
            {"position_operator": "full", "case": "qgeom_diagonal_budget"},
            "queer_position_unavailable",
            "no wannier90_tb",
        ),
        ({"periodic_dirs": [0, 1, 2]}, "queer_bad_params", "needs a 2D model"),
        ({"min_hopping_norm": 1e-4}, "queer_bad_params", "full hopping table"),
        ({"mu_range_meV": [-300.0, 300.0], "mu_step_meV": 0.1}, "queer_bad_params", "at most 4001"),
        ({"temperatures_K": [0.1, 1.0, 2.0]}, "queer_bad_params", "temperatures_K"),
        ({"position_operator": "exact"}, "queer_bad_params", "position_operator"),
        ({"qgeom_replay_vertices": 0}, "queer_bad_params", "qgeom_replay_vertices"),
    ],
)
def test_refusals_carry_the_drivers_codes(tmp_path, settings, code, fragment):
    settings = dict(settings)
    name = settings.pop("case", "qgeom_full_metal")
    exit_code, document = run_case(name, tmp_path, **settings)
    assert exit_code == 2 and document["status"] == "failed"
    assert document["error"]["code"] == code and fragment in document["error"]["message"]


def corrupt_tb(directory, how):
    path = Path(directory) / "wannier90_tb.dat"
    text = path.read_text()
    if how == "truncated":
        path.write_text(text[:-200])
    elif how == "degeneracies":
        lines = text.splitlines()
        lines[6] = lines[6].replace("    1", "    2", 1)
        path.write_text("\n".join(lines) + "\n")
    elif how == "lattice":
        lines = text.splitlines()
        lines[1] = "  3.0000000000  0.0000000000  0.0000000000"
        path.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize(
    ("how", "fragment"),
    [
        ("truncated", "expected"),
        ("degeneracies", "another R set or other degeneracies"),
        ("lattice", "lattice differs"),
    ],
)
def test_a_tb_file_that_does_not_belong_to_the_model_is_a_position_mismatch(tmp_path, how, fragment):
    params = write_case("qgeom_full_metal", tmp_path)
    corrupt_tb(tmp_path, how)
    assert run(None, params, tmp_path, log=lambda line: None) == 2
    document = read(tmp_path / "queer_results.json")
    assert document["error"]["code"] == "queer_position_mismatch"
    assert fragment in document["error"]["message"]


def haldane_full_job(directory, shift_h=0.0, shift_centre=0.0, staggered=0.3):
    """A Haldane layer whose centres straddle the cell boundary, its _tb.dat, band files and parameters."""
    model = haldane(staggered=staggered, height=10.0)
    reduced = np.array([[1 / 3, 1 / 3, 0.02], [2 / 3, 2 / 3, 0.97]])
    degeneracies = np.array([1 if not np.any(r) else 2 for r in model.translations])
    job = Wannier90Job(
        model.translations,
        model.hopping_matrices,
        model.lattice,
        reduced @ model.lattice,
        degeneracies=degeneracies,
        mp_grid=(2, 2, 1),
        win={"dis_froz_min": -1.0, "dis_froz_max": 0.6},
    )
    job.write(directory)
    rng = np.random.default_rng(5)
    n = job.num_wann
    r = np.zeros(job.raw.shape + (3,), complex)
    home = int(np.flatnonzero(~job.translations.any(axis=1))[0])
    r[home, np.arange(n), np.arange(n)] = job.centres + shift_centre
    index = {tuple(t): i for i, t in enumerate(job.translations)}
    for i, t in enumerate(job.translations):
        j = index[tuple(-t)]
        if j > i:
            block = 0.03 * (rng.normal(size=(n, n, 3)) + 1j * rng.normal(size=(n, n, 3)))
            r[i], r[j] = block, np.conj(np.swapaxes(block, 0, 1))
    H = job.raw.copy()
    H[home] += shift_h * np.eye(n)
    write_tb(
        directory / "wannier90_tb.dat",
        job.lattice,
        job.translations,
        degeneracies,
        H,
        r * degeneracies[:, None, None, None],
    )
    kpoints = np.array([[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0], [1 / 3, 1 / 3, 0], [0, 0, 0]], float)
    return job, kpoints


def qgeom_params(directory, **settings):
    values = {
        "step_type": "PYTHTB_QGEOM",
        "fermi_energy": 0.0,
        "workers": 1,
        "chunk_size": 256,
        "mu_range_meV": [-50.0, 50.0],
        "mu_step_meV": 10.0,
        "temperatures_K": [10.0, 20.0, 40.0],
        "qgeom_base_mesh": 16,
        "qgeom_max_points": 10000,
        "qgeom_tolerance": 0.05,
        "occupation_mesh": 21,
        "ws_correction": "recompute",
        "kpath": {
            "segments": [["G", [0, 0, 0], "M", [0.5, 0, 0]], ["M", [0.5, 0, 0], "K", [1 / 3, 1 / 3, 0]]]
        },
        "kpath_points": 20,
        **settings,
    }
    path = Path(directory) / "pythtb_params.json"
    path.write_text(json.dumps(values))
    return path


def test_the_position_table_unfolds_with_the_hamiltonians_wigner_seitz_images(generator_core, tmp_path):
    core = generator_core
    job, _ = haldane_full_job(tmp_path)
    params = qgeom_params(tmp_path)
    context = WannierStepContext(translate_pythtb(read(params)), tmp_path, log=lambda line: None)
    context.build_model()
    assert context.ws_report["mode"] == "recompute"
    assert context.hr_degeneracy == {tuple(r): int(d) for r, d in zip(job.translations, job.degeneracies)}
    translations, table = job.driver_table()
    rng = np.random.default_rng(3)
    stack = table[..., None] * np.r_[1.0, 0.5, -0.25] + 0.01 * rng.normal(size=table.shape + (3,))
    stack[..., 0] = table
    R_port, S_port = context.ws_apply(translations, stack)
    R_drv, S_drv = core.fold_axes(
        *core.ws_unfold(translations, stack, job.centres, context.lattice, [2, 2, 1]),
        [2],
    )
    order = np.lexsort(R_drv.T[::-1])
    np.testing.assert_array_equal(R_port, R_drv[order])
    np.testing.assert_allclose(S_port, S_drv[order], atol=1e-13)
    np.testing.assert_array_equal(R_port, context.table_R)
    np.testing.assert_allclose(S_port[..., 0], context.table_H, atol=1e-13)


def test_a_full_level_with_wigner_seitz_images_runs_and_a_tb_file_off_the_model_is_refused(tmp_path):
    haldane_full_job(tmp_path / "ok")
    params = qgeom_params(tmp_path / "ok")
    assert run(None, params, tmp_path / "ok", log=lambda line: None) == 0
    qgeom = read(tmp_path / "ok" / "queer_results.json")["qgeom"]
    assert qgeom["position"]["mode"] == "full" and qgeom["position"]["checks"]["h_vs_hr_max_meV"] < 1e-3
    # The Chern number does not depend on the position level.
    haldane_full_job(tmp_path / "diagonal")
    params = qgeom_params(tmp_path / "diagonal", position_operator="diagonal")
    assert run(None, params, tmp_path / "diagonal", log=lambda line: None) == 0
    diagonal = read(tmp_path / "diagonal" / "queer_results.json")["qgeom"]
    assert diagonal["position"]["mode"] == "diagonal"
    assert qgeom["occupied"]["applicable"] and diagonal["occupied"]["applicable"]
    assert qgeom["occupied"]["chern"] == pytest.approx(diagonal["occupied"]["chern"], abs=0.02)
    for name, shift, fragment in (
        ("h", {"shift_h": 1e-4}, "differs from wannier90_hr.dat"),
        ("c", {"shift_centre": 1e-3}, "Wannier centres"),
    ):
        directory = tmp_path / name
        haldane_full_job(directory, **shift)
        assert run(None, qgeom_params(directory), directory, log=lambda line: None) == 2
        error = read(directory / "queer_results.json")["error"]
        assert error["code"] == "queer_position_mismatch" and fragment in error["message"]


def test_the_replay_switch_stops_at_a_vertex_count_and_the_time_budget_can_be_turned_off(tmp_path, metal):
    _, free = metal
    history = read(free / "queer_results.json")["qgeom"]["integration"]["history"]
    target = history[len(history) // 2]["vertices"]
    code, replay = run_case("qgeom_full_metal", tmp_path / "replay", qgeom_replay_vertices=target)
    assert code == 0
    integration = replay["qgeom"]["integration"]
    assert integration["stop"] == "time" and integration["stop_vertices"] >= target
    assert integration["replay"] == {"time_stop": False, "time_budget_s": 3600.0, "stop_vertices": target}
    assert integration["vertices"] == integration["stop_vertices"]
    # The same replay twice is the same run, and a replay compares as one.
    code, again = run_case("qgeom_full_metal", tmp_path / "again", qgeom_replay_vertices=target)
    assert again["qgeom"]["integration"]["vertex_set"] == integration["vertex_set"]
    report = compare("QUEER_QGEOM", tmp_path / "replay", tmp_path / "again")
    assert report["verdict"] == PASS, report["summary"]
    # A clock stop: the time budget passes at once, unless it is turned off.
    code, clocked = run_case("qgeom_full_metal", tmp_path / "clock", time_budget_s=1e-9)
    assert clocked["qgeom"]["integration"]["stop"] == "time"
    code, unclocked = run_case(
        "qgeom_full_metal", tmp_path / "off", time_budget_s=1e-9, qgeom_time_stop=False
    )
    assert (
        unclocked["qgeom"]["integration"]["vertex_set"]
        == read(free / "queer_results.json")["qgeom"]["integration"]["vertex_set"]
    )
    report = compare("QUEER_QGEOM", GOLDENS / "qgeom_full_metal", tmp_path / "clock")
    assert report["verdict"] == NOT_COMPARABLE
    assert "own time budget" in preconditions(report)["stop_rule"]["detail"]


# ---------------------------------------------------------------------------
# The comparator
# ---------------------------------------------------------------------------


def candidate_copy(metal, tmp_path):
    reference, candidate = metal
    return reference, copy(candidate, tmp_path / "candidate")


def test_a_shifted_theta0_fails(metal, tmp_path):
    reference, candidate = candidate_copy(metal, tmp_path)

    def shift(document):
        for rows in document["qgeom"]["headline"].values():
            for row in rows.values():
                row["theta0_deg"] += 1.0

    edit(candidate, shift)
    report = compare("QUEER_QGEOM", reference, candidate)
    assert report["verdict"] == FAIL
    assert quantities(report)["qgeom.headline.0K.E_F.theta0_deg"] == UNEXPLAINED

    response = read(candidate / "Out" / "qgeom_response.json")
    for entry in response["per_T"].values():
        for level in entry["levels"].values():
            level["theta0_deg"] = [value + 1.0 for value in level["theta0_deg"]]
    write_response(candidate, response)
    edit(candidate, lambda d: None)
    report = compare("QUEER_QGEOM", reference, candidate)
    assert quantities(report)["qgeom.response.0K.full.theta0_deg"] == UNEXPLAINED
    assert quantities(report)["qgeom.response.full_precision"] == AGREE


@pytest.mark.parametrize(
    ("mutate", "quantity"),
    [
        (lambda q: q["occupied"].update(chern=2 * q["occupied"]["chern"]), "qgeom.occupied.chern"),
        (
            lambda q: q["occupied"].update(K_in=[[-x for x in row] for row in q["occupied"]["K_in"]]),
            "qgeom.occupied.K_in",
        ),
        (lambda q: q["integration"].update(stop="budget"), "qgeom.stop"),
    ],
    ids=["chern_doubled", "K_in_sign", "stop_reason"],
)
def test_wrong_manifold_values_and_another_stop_reason_fail(metal, tmp_path, mutate, quantity):
    reference, candidate = candidate_copy(metal, tmp_path)
    edit(candidate, lambda document: mutate(document["qgeom"]))
    report = compare("QUEER_QGEOM", reference, candidate)
    assert report["verdict"] == FAIL and quantities(report)[quantity] == UNEXPLAINED


def test_a_dropped_vertex_fails_with_or_without_the_vertex_keys(metal, tmp_path):
    reference, candidate = candidate_copy(metal, tmp_path)
    record = read(candidate / "Out" / "qgeom_vertices.json")
    record["keys"] = record["keys"][1:]
    record["count"] -= 1
    record["sha256"] = "0" * 64
    write(candidate / "Out" / "qgeom_vertices.json", record)
    report = compare("QUEER_QGEOM", reference, candidate)
    assert report["verdict"] == FAIL and quantities(report)["qgeom.mesh"] == UNEXPLAINED
    # A record that is not its run's mesh, or a candidate without one, fails too.
    for name, mutate in (
        ("stale", lambda record: record.update(count=record["count"] + 1)),
        ("missing", None),
    ):
        broken = copy(metal[1], tmp_path / name)
        if mutate is None:
            (broken / "Out" / "qgeom_vertices.json").unlink()
        else:
            data = read(broken / "Out" / "qgeom_vertices.json")
            mutate(data)
            write(broken / "Out" / "qgeom_vertices.json", data)
        report = compare("QUEER_QGEOM", reference, broken)
        assert report["verdict"] == FAIL and quantities(report)["qgeom.mesh"] == UNEXPLAINED, name
    # Against a reference without the record the recorded mesh decides.
    bare = copy(reference, tmp_path / "bare")
    (bare / "Out" / "qgeom_vertices.json").unlink()
    fresh = copy(metal[1], tmp_path / "fresh")
    assert compare("QUEER_QGEOM", bare, fresh)["verdict"] in PASSING
    edit(
        fresh,
        lambda document: document["qgeom"]["integration"].update(
            vertices=document["qgeom"]["integration"]["vertices"] - 1
        ),
    )
    report = compare("QUEER_QGEOM", bare, fresh)
    assert report["verdict"] == FAIL and quantities(report)["qgeom.mesh"] == UNEXPLAINED


def test_forged_error_estimates_change_nothing(metal, tmp_path):
    reference, candidate = candidate_copy(metal, tmp_path)

    def forge(document):
        row = document["qgeom"]["headline"]["0K"]["E_F"]
        row["A3_A_per_eV"] *= 1 + 1e-6
        row["A3_err"] = 1e6
        document["qgeom"]["integration"]["smeared_check_rel"] = 1e6

    edit(candidate, forge)
    response = read(candidate / "Out" / "qgeom_response.json")
    for entry in response["per_T"].values():
        entry["err"]["A3"] = [1e6 for _ in entry["err"]["A3"]]
    write_response(candidate, response)
    kinds = quantities(compare("QUEER_QGEOM", reference, candidate))
    assert kinds["qgeom.headline.0K.E_F.A3_A_per_eV"] == UNEXPLAINED
    assert kinds["qgeom.integration.smeared_check_rel"] == UNEXPLAINED
    assert kinds["qgeom.response.0K.err.A3"] == UNEXPLAINED


def test_a_reference_stopped_by_its_clock_needs_a_replay_of_its_vertex_count(metal, tmp_path):
    reference, candidate = metal
    clocked = copy(reference, tmp_path / "clocked")
    edit(
        clocked, lambda document: document["qgeom"]["integration"].update(stop="time"), "pythtb_results.json"
    )
    report = compare("QUEER_QGEOM", clocked, candidate)
    assert report["verdict"] == NOT_COMPARABLE
    vertices = read(clocked / "pythtb_results.json")["qgeom"]["integration"]["vertices"]
    assert f"qgeom_replay_vertices = {vertices}" in preconditions(report)["stop_rule"]["detail"]
    replayed = copy(candidate, tmp_path / "replayed")

    def as_replay(document):
        document["qgeom"]["integration"].update(stop="time", stop_vertices=vertices)
        document["qgeom"]["integration"]["replay"] = {
            "time_stop": False,
            "time_budget_s": 3600.0,
            "stop_vertices": vertices,
        }

    edit(replayed, as_replay)
    report = compare("QUEER_QGEOM", clocked, replayed)
    assert preconditions(report)["stop_rule"]["ok"] and report["verdict"] in PASSING
    unknown = copy(clocked, tmp_path / "unknown")
    edit(unknown, lambda document: document["qgeom"]["integration"].pop("vertices"), "pythtb_results.json")
    report = compare("QUEER_QGEOM", unknown, replayed)
    assert report["verdict"] == NOT_COMPARABLE
    assert "cannot be replayed" in preconditions(report)["stop_rule"]["detail"]


def test_an_unknown_producer_revision_is_not_comparable(metal, tmp_path):
    reference, candidate = metal
    other = copy(reference, tmp_path / "other")
    edit(other, lambda document: document["provenance"].update(qgeom_sha256="1" * 64), "pythtb_results.json")
    report = compare("QUEER_QGEOM", other, candidate)
    assert report["verdict"] == NOT_COMPARABLE and not preconditions(report)["qgeom_producer"]["ok"]
    older = copy(reference, tmp_path / "older")
    edit(
        older,
        lambda document: document["provenance"].update(core_sha256="45afbf42" + "0" * 56),
        "pythtb_results.json",
    )
    assert compare("QUEER_QGEOM", older, candidate)["verdict"] == NOT_COMPARABLE


def test_another_derived_decision_is_not_comparable(metal, tmp_path):
    reference, candidate = candidate_copy(metal, tmp_path)
    edit(candidate, lambda document: document["qgeom"]["window"].update(bands=[0, 3]))
    report = compare("QUEER_QGEOM", reference, candidate)
    assert report["verdict"] == NOT_COMPARABLE and not preconditions(report)["derived_decisions"]["ok"]


def curves_of(directory):
    """A run's response with its unrounded curves where it wrote them."""
    directory = Path(directory)
    response = read(directory / "Out" / "qgeom_response.json")
    full_path = directory / "Out" / "qgeom_response_full.json"
    if full_path.exists():
        full = read(full_path)
        response.update(mu_eV=full["mu_eV"], per_T=full["per_T"])
    return response


def row_from_curves(row, response, T, j):
    """A headline row as a run writes it at grid index ``j`` of its own curves."""
    entry = response["per_T"][T]
    level = entry["levels"]["full"]
    A3, err = level["A3"][j], entry["err"]["A3"][j]
    row.update(
        mu_meV=float(response["mu_eV"][j]) * 1e3,
        A3_A_per_eV=A3,
        A3_err=err,
        theta0_deg=level["theta0_deg"][j],
        chi_mA_nm_per_V2={k: -(c + 2.0 * d) * CHI_PER_M * A3 for k, (c, d) in CHI_SETS.items()},
        chi_err={k: abs(c + 2.0 * d) * CHI_PER_M * err for k, (c, d) in CHI_SETS.items()},
        sigma_xy_e2_h=level["sigma_xy_e2_h"][j],
        density_cm2=entry["density_cm2"][j],
    )
    if "full_over_diag_A3" in row:
        diag = entry["levels"]["diag"]["A3"][j]
        row["full_over_diag_A3"] = A3 / diag if diag > 0 else None


def move_cnp(directory, T, mu_meV, row_index=None):
    """Move a run's mu_CNP at T, its CNP row onto its own curves at the grid mu nearest it (or ``row_index``),
    and its mu - mu_CNP curve with it, as a run with that mu_CNP would have written them."""
    directory = Path(directory)
    response = curves_of(directory)
    mu = np.asarray(response["mu_eV"], float)
    j = int(np.argmin(np.abs(mu * 1e3 - mu_meV))) if row_index is None else row_index

    def mutate(document):
        qgeom = document["qgeom"]
        qgeom["electron_count"]["mu_cnp_meV"][T] = mu_meV
        row_from_curves(qgeom["headline"][T]["CNP"], response, T, j)

    edit(directory, mutate)
    response["per_T"][T]["mu_minus_cnp_eV"] = [float(x) for x in mu - mu_meV * 1e-3]
    write_response(directory, response)


def item(report, name):
    return next(entry for entry in report["quantities"] if entry["name"] == name)


def test_mu_cnp_moved_inside_its_gap_is_equal_and_its_row_is_checked_on_the_curves(insulator, tmp_path):
    reference, candidate = insulator
    golden = read(reference / "pythtb_results.json")["qgeom"]
    mu = golden["electron_count"]["mu_cnp_meV"]["0K"]
    assert golden["headline"]["0K"]["CNP"]["A3_A_per_eV"] < 1e-10
    # Inside the gap no triangle straddles mu_CNP: the dipole source row is null in both runs.
    assert golden["dipole_sources"]["points"]["CNP"] is None
    moved = copy(candidate, tmp_path / "moved")
    move_cnp(moved, "0K", mu - 10.0)
    report = compare("QUEER_QGEOM", reference, moved)
    assert report["verdict"] == PASS_DOCUMENTED, report["summary"]
    cnp = item(report, "qgeom.electron_count.mu_cnp_meV.0K")
    assert cnp["class"] == AGREE and "one gap of the count" in cnp["detail"]
    assert quantities(report)["qgeom.headline.0K.CNP.mu_meV"] == DOCUMENTED
    documented = {entry["name"] for entry in report["quantities"] if entry["class"] == DOCUMENTED}
    assert "qgeom.headline.0K.CNP.mu_meV" in documented
    assert all(re.fullmatch(r"qgeom\.headline\.\d+K\.CNP\.mu_meV", name) for name in documented)


@pytest.mark.parametrize(
    ("T", "mu_meV", "hide_gap_dos"),
    [
        ("5K", 250.0, False),
        ("0K", 280.0, False),
        ("5K", -250.0, False),
        ("20K", 250.0, True),
        ("60K", 0.0, False),
    ],
    ids=[
        "5K_in_the_conduction_band",
        "0K_in_the_band",
        "5K_in_the_valence_band",
        "20K_gap_dos_zeroed",
        "60K",
    ],
)
def test_a_wrong_mu_cnp_fails_by_the_electron_count(insulator, tmp_path, T, mu_meV, hide_gap_dos):
    """The verifier's mutations: a mu_CNP moved onto a band, with its row and curves moved consistently."""
    reference, candidate = insulator
    wrong = copy(candidate, tmp_path / "wrong")
    move_cnp(wrong, T, mu_meV)
    if hide_gap_dos:
        # DOS values in the gap below the DOS tolerance set to zero: the count still decides.
        response = read(wrong / "Out" / "qgeom_response.json")
        dos = response["per_T"][T]["dos"]
        for index in np.argsort(np.abs(dos))[:3]:
            dos[int(index)] = 0.0
        write_response(wrong, response)
    report = compare("QUEER_QGEOM", reference, wrong)
    assert report["verdict"] == FAIL
    cnp = item(report, f"qgeom.electron_count.mu_cnp_meV.{T}")
    assert cnp["class"] == UNEXPLAINED and "count at" in cnp["detail"], cnp
    assert quantities(report)["qgeom.response.full_precision"] == AGREE


def test_a_cnp_row_off_its_mu_cnp_fails(insulator, tmp_path):
    reference, candidate = insulator
    golden = read(reference / "pythtb_results.json")["qgeom"]
    mu = golden["electron_count"]["mu_cnp_meV"]["60K"]
    response = read(candidate / "Out" / "qgeom_response.json")
    j = int(np.argmin(np.abs(np.asarray(response["mu_eV"]) * 1e3 - mu)))
    shifted = copy(candidate, tmp_path / "shifted")
    move_cnp(shifted, "60K", mu, row_index=j + 1)
    report = compare("QUEER_QGEOM", reference, shifted)
    assert report["verdict"] == FAIL
    cnp = item(report, "qgeom.electron_count.mu_cnp_meV.60K")
    assert cnp["class"] == UNEXPLAINED and "not at the grid mu nearest its mu_CNP" in cnp["detail"]


def test_a_mu_cnp_moved_along_a_thermal_tail_fails_beyond_the_counts_rounding(insulator, tmp_path):
    """At 20 K the count leaves n_occ by about 1e-8 at 162 meV, 32 meV above its mu_CNP: inside 1e-8 per band,
    far outside the count's rounding over the vertices plus its slope times the eigenvalue tolerance."""
    reference, candidate = insulator
    moved = copy(candidate, tmp_path / "moved")
    move_cnp(moved, "20K", 162.0)
    report = compare("QUEER_QGEOM", reference, moved)
    assert report["verdict"] == FAIL
    cnp = item(report, "qgeom.electron_count.mu_cnp_meV.20K")
    assert cnp["class"] == UNEXPLAINED and "misses n_occ" in cnp["detail"], cnp


@pytest.mark.parametrize(
    ("forge", "name"),
    [
        (lambda row: row["chi_mA_nm_per_V2"].update(lka_2_m05=123.0), "chi.lka_2_m05@170meV.own_A3"),
        (lambda row: row["chi_err"].update(qk_05_m1=1e3), "chi_err.qk_05_m1@170meV.own_A3_err"),
        (lambda row: row.update(A3_err=999.0), "A3_err@170meV.reference_curve"),
        (
            lambda row: row.update(sigma_xy_e2_h=row["sigma_xy_e2_h"] + 1e-3),
            "sigma_xy_e2_h@170meV.reference_curve",
        ),
    ],
    ids=["chi", "chi_err", "A3_err", "sigma_xy"],
)
def test_a_tied_rows_own_fields_are_checked(insulator, tmp_path, forge, name):
    """A CNP row the runs place at different grid mu is checked field by field, not only by its A3."""
    reference, candidate = insulator
    mu = read(reference / "pythtb_results.json")["qgeom"]["electron_count"]["mu_cnp_meV"]["0K"]
    moved = copy(candidate, tmp_path / "moved")
    move_cnp(moved, "0K", mu - 10.0)
    assert compare("QUEER_QGEOM", reference, moved)["verdict"] == PASS_DOCUMENTED
    edit(moved, lambda document: forge(document["qgeom"]["headline"]["0K"]["CNP"]))
    report = compare("QUEER_QGEOM", reference, moved)
    assert report["verdict"] == FAIL
    assert quantities(report)[f"qgeom.headline.0K.CNP.{name}"] == UNEXPLAINED
    # Inside the gap A3 is within the M tolerance: the row's theta0 is recorded as undetermined.
    assert quantities(report)["qgeom.headline.0K.CNP.theta0_deg@170meV.reference_curve"] == NOT_APPLICABLE


def tie_max_a3(directory, T):
    """Raise A3 at a neighbour of the maximum to the maximum, and return that neighbour's index."""
    response = curves_of(directory)
    level = response["per_T"][T]["levels"]["full"]
    j = int(np.argmax(level["A3"]))
    k = j + 1 if j + 1 < len(level["A3"]) else j - 1
    level["A3"][k] = level["A3"][j]
    write_response(directory, response)
    return k


@pytest.mark.parametrize(
    ("forge", "name"),
    [
        (None, None),
        (lambda row: row.update(full_over_diag_A3=row["full_over_diag_A3"] * 1.01), "full_over_diag_A3"),
        (lambda row: row.update(theta0_deg=row["theta0_deg"] + 1.0), "theta0_deg"),
    ],
    ids=["consistent", "ratio", "theta0"],
)
def test_a_tied_max_a3_row_is_checked_on_the_other_runs_curves(metal, tmp_path, forge, name):
    """Two QUEER runs whose A3 maximum is tied: the run that places the row at the neighbour is checked there."""
    _, candidate = metal
    first, second = copy(candidate, tmp_path / "first"), copy(candidate, tmp_path / "second")
    tie_max_a3(first, "0K")
    k = tie_max_a3(second, "0K")
    response = curves_of(second)
    edit(
        second,
        lambda document: row_from_curves(document["qgeom"]["headline"]["0K"]["max_A3"], response, "0K", k),
    )
    if forge is not None:
        edit(second, lambda document: forge(document["qgeom"]["headline"]["0K"]["max_A3"]))
    report = compare("QUEER_QGEOM", first, second)
    assert quantities(report)["qgeom.headline.0K.max_A3.mu_meV"] == DOCUMENTED
    mu = float(response["mu_eV"][k]) * 1e3
    if forge is None:
        assert report["verdict"] == PASS_DOCUMENTED, report["summary"]
        judged = [
            entry["name"] for entry in report["quantities"] if f"@{mu:g}meV.reference_curve" in entry["name"]
        ]
        assert {"full_over_diag_A3", "theta0_deg", "A3_err"} <= {
            n.split(".")[4].split("@")[0] for n in judged
        }
    else:
        assert report["verdict"] == FAIL
        assert quantities(report)[f"qgeom.headline.0K.max_A3.{name}@{mu:g}meV.reference_curve"] == UNEXPLAINED


def decimate_run(source, target, stride):
    """A run's response files and grid record decimated as a run writes them past its size budget."""
    target = copy(source, target)
    response = read(target / "Out" / "qgeom_response.json")
    n = len(response["mu_eV"])
    full_path = target / "Out" / "qgeom_response_full.json"
    if full_path.exists():
        full = read(full_path)
        full.update(mu_eV=decimate(full["mu_eV"], stride, n), per_T=decimate(full["per_T"], stride, n))
        write(full_path, full)
        response.update(mu_eV=compact(full["mu_eV"]), per_T=compact(full["per_T"]))
    else:
        response.update(
            mu_eV=decimate(response["mu_eV"], stride, n), per_T=decimate(response["per_T"], stride, n)
        )
    write(target / "Out" / "qgeom_response.json", response)
    name = "queer_results.json" if (target / "queer_results.json").exists() else "pythtb_results.json"
    edit(target, lambda document: document["qgeom"]["grid"].update(downsampled=stride), name=name)
    return target


def test_a_mu_cnp_beside_a_gap_edge_on_a_decimated_file_is_equal(insulator, tmp_path):
    """90 meV is a grid point a stride-2 file skips, next to the valence band at 80 meV: the count there is only
    known to lie between the file points around it, which holds n_occ."""
    reference, candidate = insulator
    moved = copy(candidate, tmp_path / "moved")
    move_cnp(moved, "0K", 90.0)
    report = compare(
        "QUEER_QGEOM",
        decimate_run(reference, tmp_path / "reference", 2),
        decimate_run(moved, tmp_path / "decimated", 2),
    )
    assert report["verdict"] == PASS_DOCUMENTED, report["summary"]
    cnp = item(report, "qgeom.electron_count.mu_cnp_meV.0K")
    assert cnp["class"] == AGREE and "decimated" in cnp["detail"]
    # The candidate's row at 90 meV is not on the reference's decimated file: not judged there.
    assert quantities(report)["qgeom.headline.0K.CNP@90meV.reference_curve"] == NOT_APPLICABLE
    # A mu_CNP in the valence band still fails on the decimated files.
    wrong = copy(candidate, tmp_path / "wrong")
    move_cnp(wrong, "0K", 40.0)
    report = compare(
        "QUEER_QGEOM", tmp_path / "reference", decimate_run(wrong, tmp_path / "wrong_decimated", 2)
    )
    assert report["verdict"] == FAIL
    assert quantities(report)["qgeom.electron_count.mu_cnp_meV.0K"] == UNEXPLAINED


def drop_cnp(directory, T):
    """A run that records no mu_CNP at T (no CNP row, no mu - mu_CNP curve)."""

    def mutate(document):
        document["qgeom"]["electron_count"]["mu_cnp_meV"][T] = None
        document["qgeom"]["headline"][T].pop("CNP", None)

    edit(directory, mutate)
    response = read(directory / "Out" / "qgeom_response.json")
    response["per_T"][T]["mu_minus_cnp_eV"] = None
    write_response(directory, response)
    return response


def test_a_missing_mu_cnp_fails_unless_its_count_misses_n_occ_by_rounding(metal, insulator, tmp_path):
    reference, candidate = metal
    missing = copy(candidate, tmp_path / "missing")
    drop_cnp(missing, "60K")
    report = compare("QUEER_QGEOM", reference, missing)
    assert report["verdict"] == FAIL
    assert "reaches n_occ" in item(report, "qgeom.electron_count.mu_cnp_meV.60K")["detail"]
    # A count that stays above n_occ by less than the count tolerance: the miss is rounding.
    reference, candidate = insulator
    edge = copy(candidate, tmp_path / "edge")
    response = drop_cnp(edge, "0K")
    entry = response["per_T"]["0K"]
    entry["density_cm2"] = [value if value > 0 else 1.0 for value in entry["density_cm2"]]
    write_response(edge, response)
    report = compare("QUEER_QGEOM", reference, edge)
    assert report["verdict"] == NOT_COMPARABLE
    assert "decided by rounding" in preconditions(report)["mu_cnp_determined"]["detail"]


def test_theta0_where_a3_is_within_the_m_tolerance_is_not_judged(insulator, tmp_path):
    """Inside the gap A3 is rounding (1e-14 against δ = 4e-9): another BLAS kernel turns theta0 freely there."""
    reference, _ = insulator
    # A copy of the golden itself, so that only the turned angles differ on any BLAS kernel.
    assert compare("QUEER_QGEOM", reference, copy(reference, tmp_path / "same"))["verdict"] == PASS
    turned = copy(reference, tmp_path / "turned")
    response = read(turned / "Out" / "qgeom_response.json")
    level = response["per_T"]["0K"]["levels"]["full"]
    noise = [index for index, value in enumerate(level["A3"]) if value < 1e-10]
    assert noise
    for index in noise:
        level["theta0_deg"][index] += 40.0
    write(turned / "Out" / "qgeom_response.json", response)
    edit(
        turned,
        lambda document: document["qgeom"]["headline"]["0K"]["CNP"].update(theta0_deg=21.3),
        "pythtb_results.json",
    )
    report = compare("QUEER_QGEOM", reference, turned)
    assert report["verdict"] == PASS, report["summary"]
    row = item(report, "qgeom.headline.0K.CNP.theta0_deg")
    assert row["class"] == NOT_APPLICABLE and row["detail"].startswith("undetermined")
    assert "qgeom.headline.0K.CNP.theta0_deg" in report["flagged_sets"]["qgeom_undetermined"]
    curve = item(report, "qgeom.response.0K.full.theta0_deg")
    assert (
        curve["class"] == AGREE and f"{len(noise)} of" in curve["detail"] and "not judged" in curve["detail"]
    )
    # Where A3 is determined the asin bound still holds: one degree off fails.
    for index, value in enumerate(level["A3"]):
        if value > 1e-6:
            level["theta0_deg"][index] += 1.0
    write(turned / "Out" / "qgeom_response.json", response)
    report = compare("QUEER_QGEOM", reference, turned)
    assert (
        report["verdict"] == FAIL and quantities(report)["qgeom.response.0K.full.theta0_deg"] == UNEXPLAINED
    )


def test_chi_is_its_exact_multiple_of_a3_in_every_golden_row():
    """The chi tolerance rests on chi's trigonal part being -(c_out + 2 c_in) CHI_PER_M times M's."""
    for case in ("qgeom_full_metal", "qgeom_full_insulator", "qgeom_diagonal_budget"):
        headline = read(GOLDENS / case / "pythtb_results.json")["qgeom"]["headline"]
        for rows in headline.values():
            for row in rows.values():
                for convention, (c_out, c_in) in CHI_SETS.items():
                    exact = -(c_out + 2.0 * c_in) * CHI_PER_M * row["A3_A_per_eV"]
                    deviation = abs(row["chi_mA_nm_per_V2"][convention] - exact)
                    assert deviation <= 1e-14 * CHI_PER_M * max(row["A3_A_per_eV"], 1e-300), (
                        case,
                        convention,
                    )


@pytest.mark.parametrize(
    ("T", "point", "convention", "factor"),
    [
        ("0K", "E_F", "lka_2_m05", -1.0),
        ("0K", "E_F", "qk_05_m1", 0.0),
        ("60K", "CNP", "lka_2_m05", 0.0),
        ("60K", "CNP", "qk_05_m1", -1.0),
    ],
    ids=["E_F_sign", "E_F_zero", "CNP_zero", "CNP_sign"],
)
def test_a_sign_flipped_or_zeroed_chi_fails(insulator, tmp_path, T, point, convention, factor):
    reference, candidate = insulator
    golden = read(reference / "pythtb_results.json")["qgeom"]["headline"][T][point]
    assert golden["A3_A_per_eV"] > 1e-7  # determined there
    broken = copy(candidate, tmp_path / "broken")
    edit(
        broken,
        lambda document: document["qgeom"]["headline"][T][point]["chi_mA_nm_per_V2"].update(
            {convention: factor * document["qgeom"]["headline"][T][point]["chi_mA_nm_per_V2"][convention]}
        ),
    )
    report = compare("QUEER_QGEOM", reference, broken)
    assert report["verdict"] == FAIL
    assert quantities(report)[f"qgeom.headline.{T}.{point}.chi.{convention}"] == UNEXPLAINED


# Two metals whose exact symmetries let rounding steer the adaptive mesh. The Haldane layer has C3 bands
# whose band-edge velocities vanish by symmetry, so the edge rule ties at those vertices; the Rice-Mele
# stack's chains give an identically zero Berry curvature and off-diagonal metric, so its error estimate
# normalises rounding noise. OpenBLAS's Sandybridge and SkylakeX kernels break both differently here.
SYMMETRIC = {
    "fermi_energy": 0.8,
    "occupation_mesh": 61,
    "mu_range_meV": [-100.0, 100.0],
    "mu_step_meV": 20.0,
    "temperatures_K": [10.0, 20.0, 30.0],
    "qgeom_max_points": 10000,
    "qgeom_tolerance": 0.01,
}
MIRROR = {
    **SYMMETRIC,
    "fermi_energy": 0.9,
    "kpath": {"segments": [["G", [0, 0, 0], "X", [0.5, 0, 0]], ["X", [0.5, 0, 0], "M", [0.5, 0.5, 0]]]},
}
SQUARE = {**MIRROR, "fermi_energy": 0.5}


def mirror_full_job(directory, square=False):
    """A Rice-Mele stack (chains along a1, stacked along a2), or with ``square`` a QWZ layer on a 3 Å square
    lattice, with its _tb.dat at diagonal positions."""
    if square:
        model = qwz(-1.0, lattice=np.diag([3.0, 3.0, 10.0]))
        reduced = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])
    else:
        model = rice_mele(1.0, 0.6, staggered=0.15)
        reduced = np.array([[0.0, 0.0, 0.5], [0.5, 0.0, 0.5]])
    degeneracies = np.ones(len(model.translations), int)
    job = Wannier90Job(
        np.asarray(model.translations),
        np.asarray(model.hopping_matrices),
        np.asarray(model.lattice),
        reduced @ np.asarray(model.lattice),
        degeneracies=degeneracies,
        mp_grid=(2, 2, 1),
        win={"dis_froz_min": -5.0, "dis_froz_max": 5.0},
    )
    job.write(directory)
    n = job.num_wann
    r = np.zeros(job.raw.shape + (3,), complex)
    home = int(np.flatnonzero(~job.translations.any(axis=1))[0])
    r[home, np.arange(n), np.arange(n)] = job.centres
    write_tb(directory / "wannier90_tb.dat", job.lattice, job.translations, degeneracies, job.raw.copy(), r)


def inject_defect(name):
    """A defect in QUEER's mesh decisions, patched into this process (the parameters stay the same).

    ``threshold``: the gap rule's ratio a fifth low; ``dropped``: the first must-split
    round leaves one leaf the gap rule flags unsplit; ``unrefined``: the first refinement
    stage never refines the leaf with the largest error estimate (its indicator reads zero).
    """
    if not name:
        return
    from queer.observables import qgeom as obs

    if name == "threshold":
        must_split = obs.Integrator.must_split

        def defective_must_split(self, *args, **kwargs):
            self.gap_ratio *= 0.8
            return must_split(self, *args, **kwargs)

        obs.Integrator.must_split = defective_must_split
    elif name == "dropped":
        rule_flags = obs.Integrator._rule_flags
        state = {"done": False}

        def defective_rule_flags(self):
            flags = rule_flags(self)
            if not state["done"] and "gap" in flags and flags["gap"].any():
                flags["gap"] = flags["gap"].copy()
                flags["gap"][int(np.flatnonzero(flags["gap"])[0])] = False
                state["done"] = True
            return flags

        obs.Integrator._rule_flags = defective_rule_flags
    elif name == "unrefined":
        error_sums = obs.Integrator._error_sums
        state = {"done": False}

        def defective_error_sums(self, idx, scales, head):
            ind, total = error_sums(self, idx, scales, head)
            if not state["done"] and self._totals is None:
                ind = ind.copy()
                ind[int(np.argmax(ind))] = 0.0
                state["done"] = True
            return ind, total

        obs.Integrator._error_sums = defective_error_sums
    else:
        raise ValueError(f"unknown defect {name!r}")


KERNEL_RUN = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import test_generator_qgeom as t
from generator_golden_cases import write_case
from queer.compat.generator.runner import run
directory, what, defect = Path(sys.argv[2]), sys.argv[3], sys.argv[4]
directory.mkdir(parents=True, exist_ok=True)
t.inject_defect(defect)
if what == "symmetric":
    t.haldane_full_job(directory, staggered=0.2)
    params = t.qgeom_params(directory, **t.SYMMETRIC)
elif what == "mirror":
    t.mirror_full_job(directory)
    params = t.qgeom_params(directory, **t.MIRROR)
elif what == "square":
    t.mirror_full_job(directory, square=True)
    params = t.qgeom_params(directory, **t.SQUARE)
else:
    params = write_case(what, directory)
sys.exit(run(None, params, directory, log=lambda line: None))
"""


def run_under_kernel(kernel, directory, what, defect=""):
    """QUEER's step under a forced OpenBLAS kernel (in its own process), with an optional injected defect;
    skipped where the kernel cannot run."""
    env = {
        **os.environ,
        "OPENBLAS_CORETYPE": kernel,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
    }
    result = subprocess.run(
        [sys.executable, "-c", KERNEL_RUN, str(ROOT / "tests" / "numerical"), str(directory), what, defect],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode < 0:
        pytest.skip(f"OpenBLAS's {kernel} kernel does not run on this host (signal {-result.returncode})")
    assert result.returncode == 0, result.stderr[-3000:]
    return Path(directory)


def as_driver(source, target):
    """A QUEER run rewritten as the vendored driver writes it: pythtb_results.json with the vendored
    producer, no QUEER-only blocks (the decision record, the replay and vertex records) or files."""
    target = copy(source, target)
    document = read(target / "queer_results.json")
    (target / "queer_results.json").unlink()
    for key in ("engine", "counterpart_step_type", "warning_codes"):
        document.pop(key, None)
    document["step_type"] = "PYTHTB_QGEOM"
    golden = read(GOLDENS / "qgeom_full_metal" / "pythtb_results.json")["provenance"]
    document["provenance"].update(core_sha256=golden["core_sha256"], qgeom_sha256=golden["qgeom_sha256"])
    document["provenance"].pop("numeric_environment")
    for key in ("decision_ties", "replay", "stop_vertices", "vertex_set"):
        document["qgeom"]["integration"].pop(key)
    queer_only = ("Out/qgeom_vertices.json", "Out/qgeom_response_full.json")
    document["outputs"] = [entry for entry in document["outputs"] if entry["path"] not in queer_only]
    for name in queer_only:
        (target / name).unlink()
    write(target / "pythtb_results.json", document)
    return target


def symmetric_run(directory):
    haldane_full_job(directory, staggered=0.2)
    assert run(None, qgeom_params(directory, **SYMMETRIC), directory, log=lambda line: None) == 0
    return Path(directory)


@pytest.fixture(scope="module")
def symmetric(tmp_path_factory):
    return symmetric_run(tmp_path_factory.mktemp("qgeom") / "symmetric")


def drop_vertex(directory, keep_keys=True):
    """One vertex less, with the record, its digest and the integration block consistent."""
    record = read(directory / "Out" / "qgeom_vertices.json")
    keys = record["keys"][1:]
    digest = hashlib.sha256(np.sort(np.asarray(keys, "<i8")).tobytes()).hexdigest()
    record.update(count=len(keys), sha256=digest, keys=keys if keep_keys else None)
    write(directory / "Out" / "qgeom_vertices.json", record)

    def mutate(document):
        integration = document["qgeom"]["integration"]
        integration["vertices"] = len(keys)
        integration["vertex_set"].update(count=len(keys), sha256=digest)

    edit(directory, mutate)


def test_the_decision_record_is_its_runs_own(symmetric):
    """QUEER records its tied decisions beside the kernel's: the record chains its rounds from the base mesh
    to the run's vertex count, rebuilds the run's history and, with the keys in creation order, each round's
    splits and tied leaves; its bands come from the run's own scales."""
    document = read(symmetric / "queer_results.json")
    integration = document["qgeom"]["integration"]
    record = integration["decision_ties"]
    vertices = read(symmetric / "Out" / "qgeom_vertices.json")
    assert vertices["order"] == "creation" and len(vertices["keys"]) == integration["vertices"]
    run = MeshRun("candidate", integration, record, np.asarray(vertices["keys"]), vertices["unit"])
    assert record_problems(run) == []
    history = history_from_record(record)
    assert history[-len(integration["history"]) :] == [
        {key: entry.get(key) for key in ("phase", "vertices", "leaves", "target")}
        for entry in integration["history"]
    ]
    assert record["seeds"]["count"] == integration["seeds"] and not record["truncated"]
    bands = record["bands"]
    assert bands["energy_eV"] == TIE_ROUNDING * bands["energy_scale_eV"] and bands["energy_scale_eV"] > 1.0
    assert bands["velocity_eVA"] == TIE_ROUNDING * bands["velocity_scale_eVA"]
    # The C3 metal's band-edge velocities vanish at symmetric vertices: the edge rule ties there.
    tied = [entry for entry in record["rounds"] if entry["phase"] == "must_split" and entry["tied_count"]]
    assert tied and all(set(entry["tied_by_rule"]) <= {"seed", "gap", "edge"} for entry in tied)
    assert sum(entry["tied_by_rule"].get("edge", 0) for entry in tied) > 0
    # A forged tie (a leaf the mesh never had then) or a count that is not the list is never a reason.
    for forge, fragment in (
        (lambda entry: entry.update(tied=entry["tied"] + [[vertices["keys"][-1], 9, 0]]), "not a leaf"),
        (lambda entry: entry.update(tied_count=entry["tied_count"] + 1), "ties"),
    ):
        forged = json.loads(json.dumps(record))
        entry = next(
            entry for entry in forged["rounds"] if entry["phase"] == "must_split" and entry["tied_count"]
        )
        forge(entry)
        if fragment == "not a leaf":
            entry["tied_count"] = len(entry["tied"])
        problems = record_problems(run._replace(record=forged))
        assert problems and fragment in problems[0]


def test_a_dropped_vertex_fails_whatever_the_symmetry(symmetric, metal, tmp_path):
    """The C3 gate is retired: a vertex dropped from an exactly C3-symmetric model's mesh, with its vertex
    record, digest and integration block consistent, fails against a run of the same numeric environment;
    against a driver reference, which records no BLAS build, it is not comparable with the remedy."""
    qgeom = read(symmetric / "queer_results.json")["qgeom"]
    assert qgeom["symmetry"]["lattice_hexagonal"] and qgeom["symmetry"]["c3_band_floor_meV"] < 1e-5
    assert compare("QUEER_QGEOM", symmetric, copy(symmetric, tmp_path / "same"))["verdict"] == PASS
    other = copy(symmetric, tmp_path / "other")
    drop_vertex(other)
    report = compare("QUEER_QGEOM", symmetric, other)
    assert report["verdict"] == FAIL and quantities(report)["qgeom.mesh"] == UNEXPLAINED
    assert preconditions(report)["mesh_determined"]["ok"]
    assert "one numeric environment" in preconditions(report)["mesh_determined"]["detail"]
    assert "decision record" in item(report, "qgeom.mesh")["detail"]
    report = compare("QUEER_QGEOM", as_driver(symmetric, tmp_path / "driver"), other)
    assert report["verdict"] == NOT_COMPARABLE, report["summary"]
    detail = preconditions(report)["mesh_determined"]["detail"]
    assert "unrecorded" in detail and "blas (reference)" in detail and REMEDY in detail
    # A vertex record that is not its run's mesh fails whatever the symmetry.
    forged = copy(symmetric, tmp_path / "forged")
    record = read(forged / "Out" / "qgeom_vertices.json")
    record["keys"][-1] += 1
    write(forged / "Out" / "qgeom_vertices.json", record)
    report = compare("QUEER_QGEOM", symmetric, forged)
    assert report["verdict"] == FAIL and quantities(report)["qgeom.mesh"] == UNEXPLAINED
    # The generic layer: the driver's golden records another numpy build and no BLAS build.
    reference, candidate = metal
    asymmetric = copy(candidate, tmp_path / "asymmetric")
    drop_vertex(asymmetric)
    report = compare("QUEER_QGEOM", reference, asymmetric)
    assert (
        report["verdict"] == NOT_COMPARABLE and REMEDY in preconditions(report)["mesh_determined"]["detail"]
    )
    report = compare("QUEER_QGEOM", candidate, asymmetric)
    assert report["verdict"] == FAIL and quantities(report)["qgeom.mesh"] == UNEXPLAINED


@pytest.fixture(scope="module")
def kernel_run(tmp_path_factory):
    """QUEER's step of one model under one OpenBLAS kernel with an optional defect, each run once per module."""
    base = tmp_path_factory.mktemp("kernels")
    done: dict[tuple[str, str, str], Path] = {}

    def get(kernel, what, defect=""):
        key = (kernel, what, defect)
        if key not in done:
            done[key] = run_under_kernel(kernel, base / "_".join(filter(None, key)), what, defect)
        return done[key]

    return get


def vertex_sets(*runs):
    return [read(run / "queer_results.json")["qgeom"]["integration"]["vertex_set"]["sha256"] for run in runs]


@pytest.mark.parametrize(
    "what, point, mechanism",
    [
        ("symmetric", "must-split round", "edge"),
        ("mirror", "refinement step", "vanish identically"),
        ("square", "", ""),
    ],
)
def test_a_symmetric_model_on_two_blas_kernels_is_not_comparable_for_its_identified_reason(
    kernel_run, what, point, mechanism
):
    """OPENBLAS_CORETYPE=Sandybridge and SkylakeX break the Haldane metal's edge-rule ties and the Rice-Mele
    stack's rounding-noise error estimates differently here (the QWZ square metal keeps one mesh): the
    comparator names the first divergence, the tied decisions behind it and the remedy."""
    first, second = kernel_run("Sandybridge", what), kernel_run("SkylakeX", what)
    report = compare("QUEER_QGEOM", first, second)
    if len(set(vertex_sets(first, second))) == 1:
        assert report["verdict"] == PASS, report["summary"]
        return
    assert report["verdict"] == NOT_COMPARABLE, report["summary"]
    detail = preconditions(report)["mesh_determined"]["detail"]
    assert MESH_ROUNDING in detail and REMEDY in detail and point in detail and mechanism in detail
    assert "numeric environments differ" in detail and "SkylakeX" in detail


@pytest.mark.parametrize("what", ["symmetric", "mirror"])
@pytest.mark.parametrize(
    "defect, fragment",
    [("threshold", "differ outside the"), ("dropped", "outside their rules' bounds")],
)
def test_a_defect_in_the_candidates_mesh_decisions_fails_in_one_numeric_environment(
    kernel_run, what, defect, fragment
):
    """The candidate's gap-rule ratio a fifth low, or one gap-flagged leaf left unsplit: on the reference's
    kernel the mesh is compared and fails, with the decision records' description; on another kernel the
    difference is not comparable, and the remedy is the same-environment rerun that fails it."""
    reference = kernel_run("Sandybridge", what)
    report = compare("QUEER_QGEOM", reference, kernel_run("Sandybridge", what, defect))
    assert report["verdict"] == FAIL, report["summary"]
    assert preconditions(report)["mesh_determined"]["ok"]
    assert (
        quantities(report)["qgeom.mesh"] == UNEXPLAINED and fragment in item(report, "qgeom.mesh")["detail"]
    )
    report = compare("QUEER_QGEOM", reference, kernel_run("SkylakeX", what, defect))
    assert report["verdict"] == NOT_COMPARABLE, report["summary"]
    assert REMEDY in preconditions(report)["mesh_determined"]["detail"]


def test_a_refinement_the_candidate_drops_fails_in_one_numeric_environment(kernel_run):
    """The production-like insulator keeps one mesh on every kernel; a candidate whose first stage never
    refines its worst leaf fails against a run of its environment, and is not comparable against another
    kernel's run or the driver's golden (another numpy build, no BLAS record)."""
    candidate = kernel_run("Prescott", "qgeom_full_insulator", "unrefined")
    report = compare("QUEER_QGEOM", kernel_run("Prescott", "qgeom_full_insulator"), candidate)
    assert report["verdict"] == FAIL and quantities(report)["qgeom.mesh"] == UNEXPLAINED, report["summary"]
    assert "refinement step" in item(report, "qgeom.mesh")["detail"]
    for reference in (kernel_run("Sandybridge", "qgeom_full_insulator"), GOLDENS / "qgeom_full_insulator"):
        report = compare("QUEER_QGEOM", reference, candidate)
        assert report["verdict"] == NOT_COMPARABLE, report["summary"]
        assert REMEDY in preconditions(report)["mesh_determined"]["detail"]


def forge_ties(directory, rounds):
    """The candidate's record rewritten to call every leaf its given must-split rounds split tied, consistently
    with its vertex keys (each such leaf a leaf then, its outcome what the keys show)."""
    vertices = read(Path(directory) / "Out" / "qgeom_vertices.json")
    keys, unit = np.asarray(vertices["keys"]), vertices["unit"]
    creator = creating_splits(keys, unit)

    def mutate(document):
        for entry in document["qgeom"]["integration"]["decision_ties"]["rounds"]:
            if entry["phase"] != "must_split" or entry.get("history") not in rounds:
                continue
            listed = {row[0] for row in entry["tied"]}
            for rule, (lo, hi) in entry["ranges"].items():
                code = OUTCOMES.index(rule)
                for centre in np.unique(creator[lo:hi][creator[lo:hi] >= 0]).tolist():
                    if centre not in listed:
                        entry["tied"].append([centre, 1 | 1 << code, code])
            entry["tied_count"] = len(entry["tied"])

    edit(directory, mutate)


def test_a_forged_tie_record_decides_nothing(kernel_run, tmp_path):
    """A defective candidate whose record is forged to call its decisions tied still fails in the
    reference's environment, and an honest run of another kernel whose real tie is struck from its record is
    still not comparable: the decision records only describe."""
    reference = kernel_run("Sandybridge", "symmetric")
    defective = copy(kernel_run("Sandybridge", "symmetric", "threshold"), tmp_path / "defective")
    forge_ties(defective, range(200))
    report = compare("QUEER_QGEOM", reference, defective)
    assert report["verdict"] == FAIL and quantities(report)["qgeom.mesh"] == UNEXPLAINED, report["summary"]
    honest = kernel_run("SkylakeX", "symmetric")
    if len(set(vertex_sets(reference, honest))) == 1:
        pytest.skip("these kernels give the Haldane metal one mesh on this host")
    struck = copy(honest, tmp_path / "struck")

    def strike(document):
        rounds = document["qgeom"]["integration"]["decision_ties"]["rounds"]
        entry = next(entry for entry in rounds if entry["phase"] == "must_split" and entry["tied_count"])
        entry["tied"], entry["tied_count"] = [], 0

    edit(struck, strike)
    report = compare("QUEER_QGEOM", reference, struck)
    assert report["verdict"] == NOT_COMPARABLE, report["summary"]


@pytest.mark.parametrize("what", ["symmetric", "mirror"])
def test_a_driver_reference_leaves_a_mesh_difference_not_comparable(kernel_run, tmp_path, what):
    """The driver records its numpy build but no BLAS build: equal records pass and say what they leave
    unestablished, and any mesh difference (a defect or another kernel's rounding) is not comparable, with
    the unrecorded fields and the remedy."""
    sandybridge, skylakex = kernel_run("Sandybridge", what), kernel_run("SkylakeX", what)
    same = compare("QUEER_QGEOM", as_driver(sandybridge, tmp_path / "same"), sandybridge)
    assert same["verdict"] == PASS and "not established" in item(same, "qgeom.mesh")["detail"]
    for candidate in (kernel_run("Sandybridge", what, "threshold"), sandybridge):
        report = compare("QUEER_QGEOM", as_driver(skylakex, tmp_path / f"driver_{candidate.name}"), candidate)
        if candidate == sandybridge and len(set(vertex_sets(sandybridge, skylakex))) == 1:
            continue
        assert report["verdict"] == NOT_COMPARABLE, report["summary"]
        detail = preconditions(report)["mesh_determined"]["detail"]
        assert MESH_ROUNDING in detail and REMEDY in detail and "unrecorded" in detail


def test_each_run_records_its_numeric_environment(kernel_run):
    """Python, numpy, SciPy, the machine, numpy's SIMD targets and every BLAS library with its core type."""
    first, second = kernel_run("Sandybridge", "symmetric"), kernel_run("SkylakeX", "symmetric")
    environments = [
        read(run / "queer_results.json")["provenance"]["numeric_environment"] for run in (first, second)
    ]
    for environment in environments:
        assert set(ENVIRONMENT_FIELDS) <= set(environment) and environment["blas"]
        assert all(set(BLAS_FIELDS) <= set(library) for library in environment["blas"])
    assert {library["architecture"] for library in environments[0]["blas"]} == {"Sandybridge"}
    assert environment_differences(environments[0], environments[0]) == ([], [])
    differing, unrecorded = environment_differences(*environments)
    assert unrecorded == [] and len(differing) == 1 and "SkylakeX" in differing[0]
    driver = {"python": "3.12.12", "numpy": "2.5.4", "scipy": "1.18.1"}
    differing, unrecorded = environment_differences(driver, environments[1])
    assert any("numpy" in text for text in differing)
    assert unrecorded == ["machine (reference)", "simd (reference)", "blas (reference)"]


def test_a_reported_weyl_bound_widens_no_tolerance(insulator, tmp_path):
    """QGEOM hermitizes H(k) alike in both producers: a Weyl bound the candidate reports (and the driver never
    writes) leaves the count tolerance as it is, so the 20 K mu_CNP moved along its thermal tail still fails."""
    reference, candidate = insulator
    moved = copy(candidate, tmp_path / "moved")
    move_cnp(moved, "20K", 162.0)
    edit(moved, lambda document: document["model"].update(hermiticity_weyl_bound=1e-3))
    report = compare("QUEER_QGEOM", reference, moved)
    assert report["verdict"] == FAIL
    assert quantities(report)["qgeom.electron_count.mu_cnp_meV.20K"] == UNEXPLAINED


def test_a_quotable_flag_rounding_decides_is_undetermined(insulator, tmp_path):
    """The insulator golden loses 'quotable' at its in-gap CNP rows, where A3, A3_err and M are rounding; a
    run whose rounding there is zero is quotable. That difference is undetermined; the same difference at a
    determined row fails."""
    reference, candidate = insulator
    zeroed = copy(candidate, tmp_path / "zeroed")

    def quotable(document):
        qgeom = document["qgeom"]
        for T in ("0K", "5K", "20K"):
            row = qgeom["headline"][T]["CNP"]
            row.update(A3_A_per_eV=0.0, A3_err=0.0)
            row["chi_mA_nm_per_V2"] = {key: 0.0 for key in row["chi_mA_nm_per_V2"]}
            row["chi_err"] = {key: 0.0 for key in row["chi_err"]}
        qgeom["verdict"], qgeom["verdict_reasons"] = "quotable", []

    edit(zeroed, quotable)
    assert read(reference / "pythtb_results.json")["qgeom"]["verdict"] == "indicative"
    report = compare("QUEER_QGEOM", reference, zeroed)
    assert report["verdict"] in PASSING, report["summary"]
    assert quantities(report)["qgeom.verdict"] == NOT_APPLICABLE
    assert "rounding decides" in item(report, "qgeom.verdict")["detail"]
    determined = copy(candidate, tmp_path / "determined")
    edit(determined, lambda document: document["qgeom"].update(verdict="quotable", verdict_reasons=[]))
    # The reference's 0K E_F row now fails 'quotable' by 0.01 against a threshold of 0.0033: determined.
    loud = copy(reference, tmp_path / "reference")
    edit(
        loud,
        lambda document: document["qgeom"]["headline"]["0K"]["E_F"].update(A3_err=0.01),
        name="pythtb_results.json",
    )
    report = compare("QUEER_QGEOM", loud, determined)
    assert quantities(report)["qgeom.verdict"] == UNEXPLAINED


@pytest.mark.parametrize("kernel", ["Prescott", "Nehalem", "Sandybridge"])
def test_the_insulator_on_another_blas_kernel_passes(tmp_path, kernel):
    """Inside the gap theta0 (A3 ~ 1e-14) and mu_CNP are rounding, which these kernels round otherwise."""
    candidate = run_under_kernel(kernel, tmp_path / kernel, "qgeom_full_insulator")
    report = compare("QUEER_QGEOM", GOLDENS / "qgeom_full_insulator", candidate)
    assert report["verdict"] in PASSING, report["summary"]
    assert quantities(report)["qgeom.mesh"] == AGREE


def test_vertex_keys_that_do_not_hash_to_their_digest_fail(metal, tmp_path):
    """The verifier's forgery: a changed key under the reference's (and its own) digest."""
    reference, candidate = candidate_copy(metal, tmp_path)
    record = read(candidate / "Out" / "qgeom_vertices.json")
    assert record["sha256"] == read(reference / "Out" / "qgeom_vertices.json")["sha256"]
    record["keys"][-1] += 1
    write(candidate / "Out" / "qgeom_vertices.json", record)
    report = compare("QUEER_QGEOM", reference, candidate)
    assert report["verdict"] == FAIL
    assert "hash to" in item(report, "qgeom.mesh")["detail"]
    record["keys"][-1] = "x"
    write(candidate / "Out" / "qgeom_vertices.json", record)
    report = compare("QUEER_QGEOM", reference, candidate)
    assert report["verdict"] == FAIL and "are not" in item(report, "qgeom.mesh")["detail"]


@pytest.mark.parametrize(
    ("damage", "fragment"),
    [
        (lambda lines: lines + [lines[-1]], "repeated band/k-point record"),
        (lambda lines: lines[:-1] + ["1 2"], "expected band index"),
        (lambda lines: ["0" + lines[0][1:]] + lines[1:], "one-based"),
    ],
    ids=["repeated", "short_line", "zero_index"],
)
def test_an_unreadable_eig_skips_the_dft_check_and_keeps_the_result(tmp_path, damage, fragment):
    params = write_case("qgeom_diagonal_budget", tmp_path)
    eig = tmp_path / "wannier90.eig"
    eig.write_text("\n".join(damage(eig.read_text().splitlines())) + "\n")
    assert run(None, params, tmp_path, log=lambda line: None) == 0
    document = read(tmp_path / "queer_results.json")
    assert document["status"] == "ok" and fragment in document["qgeom"]["dft_check"]["skipped"]
    assert "dft_eig_unreadable" in document["warning_codes"]
    report = compare("QUEER_QGEOM", GOLDENS / "qgeom_diagonal_budget", tmp_path)
    assert report["verdict"] == PASS, report["summary"]
    assert quantities(report)["qgeom.dft_check"] == NOT_APPLICABLE


def test_two_queer_runs_compare_their_unrounded_curves(insulator, tmp_path):
    """QUEER's Out/qgeom_response_full.json: the 5-digit curves unrounded, compared when both runs have it."""
    _, first = insulator
    full = read(first / "Out" / "qgeom_response_full.json")
    assert compact(full["per_T"]) == read(first / "Out" / "qgeom_response.json")["per_T"]
    report = compare("QUEER_QGEOM", first, copy(first, tmp_path / "same"))
    assert report["verdict"] == PASS
    assert "without print rounding" in item(report, "qgeom.response.full_precision")["detail"]
    # A move of A3 by 1e-6 relative hides under the 5 digits but not in the unrounded curves.
    moved = copy(first, tmp_path / "moved")
    level = full["per_T"]["0K"]["levels"]["full"]
    j = int(np.argmax(level["A3"]))
    level["A3"][j] *= 1 + 1e-6
    write(moved / "Out" / "qgeom_response_full.json", full)
    response = read(moved / "Out" / "qgeom_response.json")
    response["per_T"] = compact(full["per_T"])
    write(moved / "Out" / "qgeom_response.json", response)
    report = compare("QUEER_QGEOM", first, moved)
    assert report["verdict"] == FAIL and quantities(report)["qgeom.response.0K.full.A3"] == UNEXPLAINED
    rounded = copy(moved, tmp_path / "rounded")
    (rounded / "Out" / "qgeom_response_full.json").unlink()
    report = compare("QUEER_QGEOM", first, rounded)
    assert quantities(report)["qgeom.response.0K.full.A3"] == AGREE
    # Unrounded curves that are not the run's 5-digit file fail.
    forged = copy(first, tmp_path / "forged")
    full = read(forged / "Out" / "qgeom_response_full.json")
    full["per_T"]["0K"]["dos"][0] *= 2.0
    write(forged / "Out" / "qgeom_response_full.json", full)
    report = compare("QUEER_QGEOM", first, forged)
    assert report["verdict"] == FAIL and quantities(report)["qgeom.response.full_precision"] == UNEXPLAINED


def test_two_queer_runs_on_different_blas_kernels_agree_unrounded(insulator, tmp_path):
    _, first = insulator
    second = run_under_kernel("Prescott", tmp_path / "prescott", "qgeom_full_insulator")
    report = compare("QUEER_QGEOM", first, second)
    assert report["verdict"] in PASSING, report["summary"]
    assert "without print rounding" in item(report, "qgeom.response.full_precision")["detail"]


def test_two_failed_runs_with_the_same_failure_pass(tmp_path):
    code, document = run_case("qgeom_position_unavailable", tmp_path)
    assert code == 2 and document["error"]["code"] == "queer_position_unavailable"
    report = compare("QUEER_QGEOM", GOLDENS / "qgeom_position_unavailable", tmp_path)
    assert report["verdict"] == PASS and quantities(report) == {"status": AGREE}


# ---------------------------------------------------------------------------
# The attribution rule on synthetic records
# ---------------------------------------------------------------------------


def decisions(rounds, seeds=(), seed_ties=()):
    """A decision record with the given rounds and seeds (seed ids, tied seed rows)."""
    return {
        "seeds": {
            "count": len(seeds),
            "digest": set_digest(np.asarray(seeds, np.int64)),
            "tied": [list(row) for row in seed_ties],
            "tied_count": len(seed_ties),
        },
        "rounds": rounds,
        "truncated": False,
    }


def split_round(rule, cells, tied=(), history=0, **extra):
    """A must-split (or, with rule error, a refinement) round splitting ``cells`` with the given tied rows."""
    entry = {
        "phase": "error" if rule == "error" else "must_split",
        "history": history,
        "splits": {rule: len(cells)} if cells else {},
        "digest": {rule: set_digest(np.asarray(cells, np.int64))} if cells else {},
        "tied": [list(row) for row in tied],
        "tied_count": len(tied),
        "stop": {"by": None, "tied": False},
        **extra,
    }
    return entry


def bits(*names):
    return sum(1 << OUTCOMES.index(name) for name in names)


def attributed(a, b, integration_a=None, integration_b=None):
    return attribute(
        MeshRun("reference", integration_a or {}, a, None, None),
        MeshRun("candidate", integration_b or {}, b, None, None),
    )


def test_a_difference_is_identified_only_where_both_records_tie_it():
    edge = OUTCOMES.index("edge")
    common = split_round("gap", [10, 20])
    flip_a = split_round("edge", [30, 40], tied=[(40, bits("none", "edge"), edge)], history=1)
    flip_b = split_round("edge", [30], tied=[(40, bits("none", "edge"), 0)], history=1)
    found = attributed(decisions([common, flip_a]), decisions([common, flip_b]))
    assert found.identified and "must-split round 1" in found.point and "none" in found.detail
    # Tied in one record only, outside the bounds the other allows, or another leaf split: never a reason.
    for other in (
        split_round("edge", [30], history=1),
        split_round("edge", [30], tied=[(40, bits("none", "gap"), 0)], history=1),
        split_round("edge", [30, 50], tied=[(40, bits("none", "edge"), 0)], history=1),
    ):
        assert not attributed(decisions([common, flip_a]), decisions([common, other])).identified
    # A seed decision tied in both runs that went the other way, and one that is not tied.
    seed = seed_id(7, "gap", 0, 1)
    found = attributed(
        decisions([common], seeds=[seed], seed_ties=[(7, 0, 0, 1, 1)]),
        decisions([common], seeds=[], seed_ties=[(7, 0, 0, 1, 0)]),
    )
    assert found.identified and "seeds" in found.point
    assert not attributed(decisions([common], seeds=[seed]), decisions([common])).identified
    # The same seeds with another count inside the mu range: their energies tie its ends, or not.
    a, b = decisions([common]), decisions([common])
    a["seeds"].update(mu_seeds=2, mu_seeds_tied=1)
    b["seeds"].update(mu_seeds=1, mu_seeds_tied=1)
    assert attributed(a, b).identified and "controlled mu" in attributed(a, b).point
    b["seeds"]["mu_seeds_tied"] = 0
    assert not attributed(a, b).identified


def test_a_refinement_stop_or_maximum_differs_only_where_tied_or_set_by_rounding():
    def step(by, tied, noise=None, cells=()):
        entry = split_round("error", list(cells), history=3, target=0.01)
        entry["stop"] = {"by": by, "tied": tied, "worst": 0.01, "band": 1e-9}
        if noise:
            entry["noise"] = noise
        return entry

    common = split_round("gap", [10])
    assert attributed(
        decisions([common, step("target", True)]), decisions([common, step(None, True, cells=[5])])
    ).identified
    assert not attributed(
        decisions([common, step("target", True)]), decisions([common, step(None, False, cells=[5])])
    ).identified
    rounding = {"columns": 8, "leaves": 3, "worst_real": 0.01, "worst_rounding": 0.2}
    found = attributed(
        decisions([common, step(None, False, rounding, [5, 6])]),
        decisions([common, step(None, False, rounding, [5, 7])]),
    )
    assert found.identified and "vanish identically" in found.detail
    assert not attributed(
        decisions([common, step(None, False, cells=[5, 6])]),
        decisions([common, step(None, False, cells=[5, 7])]),
    ).identified

    def restart(chosen, tied):
        return {
            "phase": "restart",
            "history": None,
            "chosen_mu_eV": chosen,
            "tied_mu_eV": tied,
            "tied_count": len(tied) - 1,
        }

    assert attributed(
        decisions([common, restart(0.01, [0.01, 0.02])]), decisions([common, restart(0.02, [0.01, 0.02])])
    ).identified
    assert not attributed(
        decisions([common, restart(0.01, [0.01])]), decisions([common, restart(0.02, [0.01, 0.02])])
    ).identified


def test_against_a_driver_reference_the_candidates_record_bounds_the_difference():
    """One record: the first differing history entry's round must tie at least |d leaves| / 3 split decisions
    and 8 vertices per tied decision; a count beyond them fails."""
    first = split_round("gap", [10, 20], history=0, vertices=[512, 528], leaves=[256, 262])
    second = split_round(
        "edge", [30], tied=[(40, bits("none", "edge"), 0)], history=1, vertices=[528, 536], leaves=[262, 265]
    )
    record = decisions([first, second])
    mine = {"seeds": 0, "history": history_from_record(record)}
    other = {
        "seeds": 0,
        "history": [dict(mine["history"][0]), {"phase": "must_split", "vertices": 542, "leaves": 268}],
    }
    found = attributed(None, record, other, mine)
    assert found.identified and "history entry 1" in found.point
    other["history"][1].update(vertices=560, leaves=271)
    assert not attributed(None, record, other, mine).identified
    other["history"][1].update(vertices=542, leaves=268)
    assert not attributed(None, record, {**other, "seeds": 2}, mine).identified


def test_the_second_stage_records_every_a3_maximum_within_its_band():
    from types import SimpleNamespace

    from queer.observables.qgeom import TIE_INDICATOR, Integrator

    integ = SimpleNamespace(
        mesh=SimpleNamespace(lvl=np.zeros(4), nv=9), mu_subset=np.zeros(3), decision_rounds=[]
    )
    mu = np.array([-0.01, 0.0, 0.01, 0.02])
    A3 = np.array([1.0, 1.0 + 0.5 * TIE_INDICATOR, 0.5, 1.0 - 3.0 * TIE_INDICATOR])
    Integrator.record_restart(integ, mu, A3, int(np.argmax(A3)), 1.0)
    entry = integ.decision_rounds[0]
    assert entry["chosen_mu_eV"] == 0.0 and entry["tied_mu_eV"] == [-0.01, 0.0] and entry["tied_count"] == 1


@pytest.fixture(scope="module")
def mirror(tmp_path_factory):
    directory = tmp_path_factory.mktemp("qgeom") / "mirror"
    mirror_full_job(directory)
    assert run(None, qgeom_params(directory, **MIRROR), directory, log=lambda line: None) == 0
    return directory


def test_quantities_that_vanish_identically_are_recorded_as_rounding(mirror, symmetric):
    """The Rice-Mele chains have no Berry curvature and no off-diagonal metric: the record names those
    controlled quantities, whose scales are rounding, and the leaves and worst sum they set; the Haldane
    metal controls none."""
    rounds = read(mirror / "queer_results.json")["qgeom"]["integration"]["decision_ties"]["rounds"]
    steps = [entry for entry in rounds if entry["phase"] == "error"]
    named = steps[0]["noise_quantities"]
    assert {"sea omega", "Fermi-surface v_x omega", "Fermi-surface v_x G_xy"} <= set(named)
    assert "Fermi-surface v_x G_xx" not in named and "Fermi-surface count" not in named
    assert all(entry["noise"]["leaves"] > 0 for entry in steps)
    assert all(entry["noise"]["worst_rounding"] > entry["noise"]["worst_real"] for entry in steps)
    rounds = read(symmetric / "queer_results.json")["qgeom"]["integration"]["decision_ties"]["rounds"]
    assert all(not entry.get("noise_quantities") and "noise" not in entry for entry in rounds)
