"""QUEER_QGEOM: the runner's contract, failure codes and replay switch, and the comparator's rules.

The step runs on the analytic quantum-geometry cases of ``generator_golden_cases``
(staged as a Generator WANNIER job stages them). The comparator is checked against
the vendored driver's goldens (``tests/fixtures/generator_goldens``): each negative
control (a theta0 shifted by one degree, a doubled Chern number, a flipped K_in, a
dropped vertex, another stop reason) must not pass, and a reference whose sampling
cannot be replayed is not comparable.
"""

import json
import shutil
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
    NOT_COMPARABLE,
    PASS,
    PASS_DOCUMENTED,
    UNEXPLAINED,
    compare,
)
from queer.compat.generator.pythtb_params import translate_pythtb  # noqa: E402
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
    for name in ("qgeom_response.json", "qgeom_path.json", "qgeom_vertices.json", "qgeom_dipole.png"):
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


def haldane_full_job(directory, shift_h=0.0, shift_centre=0.0):
    """A Haldane layer whose centres straddle the cell boundary, its _tb.dat, band files and parameters."""
    model = haldane(staggered=0.3, height=10.0)
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
    write(candidate / "Out" / "qgeom_response.json", response)
    edit(candidate, lambda d: None)
    report = compare("QUEER_QGEOM", reference, candidate)
    assert quantities(report)["qgeom.response.0K.full.theta0_deg"] == UNEXPLAINED


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
    write(candidate / "Out" / "qgeom_response.json", response)
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


def test_mu_cnp_inside_a_gap_without_states_is_a_documented_tie(insulator, tmp_path):
    reference, candidate = insulator
    golden = read(reference / "pythtb_results.json")["qgeom"]
    mu = golden["electron_count"]["mu_cnp_meV"]["0K"]
    assert golden["headline"]["0K"]["CNP"]["A3_A_per_eV"] < 1e-10
    moved = copy(candidate, tmp_path / "moved")
    shifted = mu - 10.0

    def move(document):
        qgeom = document["qgeom"]
        qgeom["electron_count"]["mu_cnp_meV"]["0K"] = shifted
        qgeom["headline"]["0K"]["CNP"]["mu_meV"] = shifted
        # Inside the gap no triangle straddles mu_CNP: the dipole source row is null in both runs.
        assert qgeom["dipole_sources"]["points"]["CNP"] is None

    edit(moved, move)
    response = read(moved / "Out" / "qgeom_response.json")
    entry = response["per_T"]["0K"]
    entry["mu_minus_cnp_eV"] = [value + 0.01 for value in entry["mu_minus_cnp_eV"]]
    write(moved / "Out" / "qgeom_response.json", response)
    report = compare("QUEER_QGEOM", reference, moved)
    kinds = quantities(report)
    assert report["verdict"] == PASS_DOCUMENTED, report["summary"]
    assert kinds["qgeom.electron_count.mu_cnp_meV.0K"] == DOCUMENTED
    assert kinds["qgeom.headline.0K.CNP.mu_meV"] == DOCUMENTED
    # A mu_CNP moved onto a band, where the count is not flat, fails.
    banded = copy(candidate, tmp_path / "banded")
    edit(banded, lambda document: document["qgeom"]["electron_count"]["mu_cnp_meV"].update({"60K": 0.0}))
    report = compare("QUEER_QGEOM", reference, banded)
    assert quantities(report)["qgeom.electron_count.mu_cnp_meV.60K"] == UNEXPLAINED


def test_two_failed_runs_with_the_same_failure_pass(tmp_path):
    code, document = run_case("qgeom_position_unavailable", tmp_path)
    assert code == 2 and document["error"]["code"] == "queer_position_unavailable"
    report = compare("QUEER_QGEOM", GOLDENS / "qgeom_position_unavailable", tmp_path)
    assert report["verdict"] == PASS and quantities(report) == {"status": AGREE}
