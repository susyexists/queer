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
from tight_binding_models import haldane
from wannier90_files import Wannier90Job
from wannier90_tb import write_tb

ROOT = Path(__file__).resolve().parents[2]
GOLDENS = ROOT / "tests" / "fixtures" / "generator_goldens"
SCHEMA = ROOT / "tests" / "fixtures" / "generator_schema" / "qgeom_keys.json"
REAL_REFERENCE = Path.home() / ".cache" / "queer-parity-data" / "job-1cd360d9"
sys.path.insert(0, str(ROOT / "tests" / "numerical"))

from generator_golden_cases import write_case  # noqa: E402

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
from queer.compat.generator.parity_qgeom import CHI_PER_M, CHI_SETS, SYMMETRIC_TIES  # noqa: E402
from queer.compat.generator.pythtb_params import translate_pythtb  # noqa: E402
from queer.compat.generator.qgeom_step import compact  # noqa: E402
from queer.compat.generator.runner import run  # noqa: E402
from queer.compat.generator.wannier_steps import WannierStepContext  # noqa: E402

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


def move_cnp(directory, T, mu_meV, row_index=None):
    """Move a run's mu_CNP at T, its CNP row onto its own curves at the grid mu nearest it (or ``row_index``),
    and its mu - mu_CNP curve with it, as a run with that mu_CNP would have written them."""
    directory = Path(directory)
    response = read(directory / "Out" / "qgeom_response.json")
    mu = np.asarray(response["mu_eV"], float)
    j = int(np.argmin(np.abs(mu * 1e3 - mu_meV))) if row_index is None else row_index
    entry = response["per_T"][T]
    level = entry["levels"]["full"]

    def mutate(document):
        qgeom = document["qgeom"]
        qgeom["electron_count"]["mu_cnp_meV"][T] = mu_meV
        qgeom["headline"][T]["CNP"].update(
            mu_meV=float(mu[j]) * 1e3,
            A3_A_per_eV=level["A3"][j],
            theta0_deg=level["theta0_deg"][j],
            sigma_xy_e2_h=level["sigma_xy_e2_h"][j],
            density_cm2=entry["density_cm2"][j],
        )

    edit(directory, mutate)
    entry["mu_minus_cnp_eV"] = [float(x) for x in mu - mu_meV * 1e-3]
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


# A Haldane metal: C3-symmetric bands whose band-edge velocities vanish by symmetry, so rounding decides
# some of its refinements (OpenBLAS's Sandybridge and SkylakeX kernels give different vertex sets).
SYMMETRIC = {
    "fermi_energy": 0.8,
    "occupation_mesh": 61,
    "mu_range_meV": [-100.0, 100.0],
    "mu_step_meV": 20.0,
    "temperatures_K": [10.0, 20.0, 30.0],
    "qgeom_max_points": 10000,
    "qgeom_tolerance": 0.01,
}
KERNEL_RUN = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import test_generator_qgeom as t
from generator_golden_cases import write_case
from queer.compat.generator.runner import run
directory, what = Path(sys.argv[2]), sys.argv[3]
directory.mkdir(parents=True, exist_ok=True)
if what == "symmetric":
    t.haldane_full_job(directory, staggered=0.2)
    params = t.qgeom_params(directory, **t.SYMMETRIC)
else:
    params = write_case(what, directory)
sys.exit(run(None, params, directory, log=lambda line: None))
"""


def run_under_kernel(kernel, directory, what):
    """QUEER's step under a forced OpenBLAS kernel (in its own process); skipped where the kernel cannot run."""
    env = {
        **os.environ,
        "OPENBLAS_CORETYPE": kernel,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
    }
    result = subprocess.run(
        [sys.executable, "-c", KERNEL_RUN, str(ROOT / "tests" / "numerical"), str(directory), what],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode < 0:
        pytest.skip(f"OpenBLAS's {kernel} kernel does not run on this host (signal {-result.returncode})")
    assert result.returncode == 0, result.stderr[-3000:]
    return Path(directory)


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


def test_a_symmetric_models_mesh_difference_is_not_comparable_and_an_asymmetric_ones_fails(
    symmetric, metal, tmp_path
):
    qgeom = read(symmetric / "queer_results.json")["qgeom"]
    assert qgeom["symmetry"]["lattice_hexagonal"] and qgeom["symmetry"]["c3_band_floor_meV"] < 1e-5
    assert compare("QUEER_QGEOM", symmetric, copy(symmetric, tmp_path / "same"))["verdict"] == PASS
    other = copy(symmetric, tmp_path / "other")
    drop_vertex(other)
    report = compare("QUEER_QGEOM", symmetric, other)
    assert report["verdict"] == NOT_COMPARABLE
    determined = preconditions(report)["mesh_determined"]
    assert not determined["ok"] and SYMMETRIC_TIES in determined["detail"]
    assert (
        "C3-symmetric" in determined["detail"] and "1 vertices only in the reference" in determined["detail"]
    )
    # Against a record without keys (the driver's), the recorded mesh decides, with the same verdict.
    bare = copy(symmetric, tmp_path / "bare")
    (bare / "Out" / "qgeom_vertices.json").unlink()
    edit(bare, lambda document: document.update(engine="pythtb"))
    report = compare("QUEER_QGEOM", bare, other)
    assert report["verdict"] == NOT_COMPARABLE and not preconditions(report)["mesh_determined"]["ok"]
    # A record that is not its run's mesh fails whatever the symmetry.
    forged = copy(symmetric, tmp_path / "forged")
    record = read(forged / "Out" / "qgeom_vertices.json")
    record["keys"][-1] += 1
    write(forged / "Out" / "qgeom_vertices.json", record)
    report = compare("QUEER_QGEOM", symmetric, forged)
    assert report["verdict"] == FAIL and quantities(report)["qgeom.mesh"] == UNEXPLAINED
    # The generic layer has no exact symmetry: a mesh difference there fails.
    reference, candidate = metal
    asymmetric = copy(candidate, tmp_path / "asymmetric")
    drop_vertex(asymmetric)
    report = compare("QUEER_QGEOM", reference, asymmetric)
    assert report["verdict"] == FAIL and quantities(report)["qgeom.mesh"] == UNEXPLAINED
    assert preconditions(report)["mesh_determined"]["ok"]


def test_a_symmetric_model_on_two_blas_kernels_is_never_a_failure(tmp_path):
    """OPENBLAS_CORETYPE=Sandybridge and SkylakeX break the Haldane metal's ties differently here."""
    first = run_under_kernel("Sandybridge", tmp_path / "sandybridge", "symmetric")
    second = run_under_kernel("SkylakeX", tmp_path / "skylakex", "symmetric")
    sets = [read(run / "queer_results.json")["qgeom"]["integration"]["vertex_set"] for run in (first, second)]
    report = compare("QUEER_QGEOM", first, second)
    if sets[0] == sets[1]:
        assert report["verdict"] == PASS, report["summary"]
    else:
        assert report["verdict"] == NOT_COMPARABLE, report["summary"]
        assert SYMMETRIC_TIES in preconditions(report)["mesh_determined"]["detail"]


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
