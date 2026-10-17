"""The Generator parity comparator on analytic runs, negative controls and reference semantics.

Every case of ``generator_golden_cases`` runs twice with QUEER; the two runs must
pass. Each negative control alters one quantity or input of a passing pair and
must not pass, for the expected reason. Rewritten references reproduce what
earlier driver revisions recorded and must pass under their registered
semantics.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "numerical"))

from generator_golden_cases import CASES, write_case  # noqa: E402

from queer.compat.generator import parity  # noqa: E402
from queer.compat.generator.estimators import (  # noqa: E402
    dos_at_fermi,
    edge_crossings,
    gaussian_dos,
    legacy_edge_crossings,
)
from queer.compat.generator.parity import (  # noqa: E402
    AFTER_CONVENTION,
    AGREE,
    DOCUMENTED,
    FAIL,
    NOT_APPLICABLE,
    NOT_COMPARABLE,
    PASS,
    PASS_DOCUMENTED,
    UNEXPLAINED,
    circular_matching_distance,
    compare,
    contour_hausdorff,
    render_markdown,
    step_family,
    warning_code,
)
from queer.compat.generator.results import to_jsonable  # noqa: E402
from queer.compat.generator.runner import main, run  # noqa: E402

LEGACY_CORE = "f6c9b7af239b530253cb7eeb9b209847e7d47baaf0237cb835a326c17dd5ec32"
# Driver cores that counted edge crossings by nearest energy: dcdc104 to c1c973b, and the two
# between 9f43886 and dcdc104 (aed1f70, ee04213) that were live on main.
NEAREST_ENERGY_CORES = (
    "56fda912ba570e2293b3a9b4bf20d11038c64efa5d2f192c8ac03c615034e215",
    "cfe3b5f6" + "0" * 56,
    "a171477a" + "0" * 56,
)
QUEER_RESULTS = "queer_results.json"
DRIVER_RESULTS = "pythtb_results.json"


def run_case(name, directory, **overrides):
    directory.mkdir(parents=True)
    params = write_case(name, directory, **overrides)
    code = run(None, params, directory, log=lambda line: None)
    assert code == 0, read(directory / QUEER_RESULTS).get("error")
    return directory


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Two QUEER runs of every case."""
    base = tmp_path_factory.mktemp("parity")
    return {name: (run_case(name, base / name / "a"), run_case(name, base / name / "b")) for name in CASES}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value))


def copy(source, target):
    shutil.copytree(source, target, symlinks=True)
    return target


def classes(report):
    return {item["name"]: item["class"] for item in report["quantities"]}


def failed(report):
    return {item["name"]: item["detail"] for item in report["preconditions"] if not item["ok"]}


def as_driver_document(directory, step, core="0" * 64):
    """Rewrite a QUEER run's document as the driver writes it: no QUEER keys, ``pythtb_results.json``."""
    document = read(directory / QUEER_RESULTS)
    (directory / QUEER_RESULTS).unlink()
    for key in ("engine", "counterpart_step_type", "warning_codes"):
        document.pop(key)
    document["step_type"] = step
    provenance = document["provenance"]
    document["provenance"] = {"core_sha256": core, "pythtb": "2.0.2", "inputs": provenance["inputs"]}
    for key in ("hermiticity_weyl_bound", "queer_version"):
        document["model"].pop(key)
    write(directory / DRIVER_RESULTS, document)
    return document


@pytest.mark.parametrize("name", sorted(CASES))
def test_two_queer_runs_of_a_case_pass(runs, name):
    reference, candidate = runs[name]
    report = compare(f"QUEER_{step_family(CASES[name].step)}", reference, candidate)
    assert report["verdict"] == PASS, report["summary"]
    assert set(classes(report).values()) <= {AGREE, NOT_APPLICABLE}
    assert not report["inputs_verified_by_fingerprint"]
    assert report["semantics"]["profile"] == "current"
    assert all(item["ok"] for item in report["preconditions"])
    assert json.loads(json.dumps(report)) == report


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "expected"), [("haldane_berry", ()), ("haldane_berry_fermi", ("sign",))])
def test_a_flipped_curvature_map_fails(runs, tmp_path, name, expected):
    reference, candidate = runs[name]
    candidate = copy(candidate, tmp_path / "candidate")
    plane = read(candidate / "Out" / "berry_k3_0.json")
    for key in ("omega_reduced", "omega_cart_A2"):
        plane[key] = [-value for value in plane[key]]
    write(candidate / "Out" / "berry_k3_0.json", plane)
    report = compare("QUEER_BERRY", reference, candidate)
    assert report["verdict"] == FAIL
    assert {"berry.omega_reduced", "berry.omega_cart_A2"} <= set(report["summary"]["unexplained"])
    assert "berry.candidate_integral_consistency" in report["summary"]["unexplained"]
    if expected:
        detail = next(
            item["detail"] for item in report["quantities"] if item["name"] == "berry.omega_reduced"
        )
        assert "negative of the reference" in detail


def test_wilson_loops_closed_without_the_boundary_phase_fail(runs, tmp_path):
    reference, candidate = runs["haldane_topology"]
    candidate = copy(candidate, tmp_path / "candidate")
    flow = read(candidate / "Out" / "wcc_k3_0.json")
    # Closing a loop without exp(-i G.tau) moves every centre by the orbital's
    # coordinate along the loop axis (tau_1 = 1/3 here) and the phase with it.
    flow["wcc"] = [sorted((value + 1 / 3) % 1.0 for value in row) for row in flow["wcc"]]
    flow["berry_phase"] = [
        float(np.angle(np.exp(1j * (value - 2 * np.pi / 3)))) for value in flow["berry_phase"]
    ]
    write(candidate / "Out" / "wcc_k3_0.json", flow)
    report = compare("QUEER_TOPOLOGY", reference, candidate)
    assert report["verdict"] == FAIL
    assert {"topology.plane[k3=0].wcc", "topology.plane[k3=0].berry_phase"} <= set(
        report["summary"]["unexplained"]
    )


def test_a_fermi_energy_one_mev_higher_is_not_comparable(runs, tmp_path):
    reference, _ = runs["haldane_bands"]
    candidate = run_case("haldane_bands", tmp_path / "candidate", fermi_energy=0.801)
    report = compare("QUEER_BANDS", reference, candidate)
    assert report["verdict"] == NOT_COMPARABLE
    assert "fermi_energy" in failed(report)
    assert report["quantities"] == []


def test_a_doubled_dos_fails(runs, tmp_path):
    reference, candidate = runs["haldane_bands"]
    candidate = copy(candidate, tmp_path / "candidate")
    dos = read(candidate / "Out" / "dos.json")
    dos["dos_per_eV"] = [2.0 * value for value in dos["dos_per_eV"]]
    write(candidate / "Out" / "dos.json", dos)
    report = compare("QUEER_BANDS", reference, candidate)
    assert report["verdict"] == FAIL
    assert report["summary"]["unexplained"] == ["dos.dos_per_eV"]


def test_a_flipped_wigner_seitz_mode_is_not_comparable(runs, tmp_path):
    reference, candidate = runs["haldane_bands"]
    candidate = copy(candidate, tmp_path / "candidate")
    document = read(candidate / QUEER_RESULTS)
    assert document["model"]["ws_correction"]["mode"] == "recompute"
    document["model"]["ws_correction"]["mode"] = "off"
    write(candidate / QUEER_RESULTS, document)
    report = compare("QUEER_BANDS", reference, candidate)
    assert report["verdict"] == NOT_COMPARABLE
    assert list(failed(report)) == ["derived_decisions"]
    assert "ws_mode" in failed(report)["derived_decisions"]


def test_a_changed_edge_crossing_count_fails(runs, tmp_path):
    reference, candidate = runs["haldane_slab"]
    candidate = copy(candidate, tmp_path / "candidate")
    document = read(candidate / QUEER_RESULTS)
    left = document["slab"]["edge_crossings_at_fermi"]["left"]
    left["up"] += 1
    left["net"] += 1
    write(candidate / QUEER_RESULTS, document)
    report = compare("QUEER_SLAB", reference, candidate)
    assert report["verdict"] == FAIL
    # The candidate's block no longer equals the flow recomputed from its own levels either.
    assert sorted(report["summary"]["unexplained"]) == [
        "slab.edge_crossings_at_fermi",
        "slab.edge_crossings_recomputed",
    ]


def tied_valence_maximum(runs, tmp_path, *, move_kpoint):
    """The Kane-Mele pair with the candidate's is_direct flipped, its VBM optionally at the other valley."""
    reference, candidate = runs["kane_mele_topology"]
    candidate = copy(candidate, tmp_path / "candidate")
    document = read(candidate / QUEER_RESULTS)
    occupation = document["topology"]["occupation"]
    assert occupation["direct_gap_eV"] == pytest.approx(occupation["gap_eV"])
    if move_kpoint:
        # K and K' hold the same valence maximum; the other one is picked.
        k1, k2, k3 = occupation["vbm_k"]
        occupation["vbm_k"] = [k2, k1, k3]
    # Which valley this machine's rounding picked does not matter: the flag flips.
    occupation["is_direct"] = not occupation["is_direct"]
    write(candidate / QUEER_RESULTS, document)
    return compare("QUEER_TOPOLOGY", reference, candidate)


def test_a_direct_flag_flipped_by_tied_extrema_is_documented(runs, tmp_path):
    report = tied_valence_maximum(runs, tmp_path, move_kpoint=True)
    assert report["verdict"] == PASS_DOCUMENTED, report["summary"]
    kinds = classes(report)
    assert kinds["topology.occupation.edge_kpoints"] == DOCUMENTED
    assert kinds["topology.occupation.is_direct"] == DOCUMENTED
    assert kinds["topology.occupation.counts"] == AGREE


def test_a_direct_flag_flipped_without_a_tie_fails(runs, tmp_path):
    report = tied_valence_maximum(runs, tmp_path, move_kpoint=False)
    assert report["verdict"] == FAIL
    assert report["summary"]["unexplained"] == ["topology.occupation.is_direct"]


def test_a_different_step_parameter_is_not_comparable(runs, tmp_path):
    reference, _ = runs["haldane_bands"]
    candidate = run_case("haldane_bands", tmp_path / "candidate", dos_sigma_eV=0.07)
    report = compare("QUEER_BANDS", reference, candidate)
    assert report["verdict"] == NOT_COMPARABLE
    assert list(failed(report)) == ["parameters"]
    assert "dos_sigma_eV" in failed(report)["parameters"]


def test_differing_parameter_digests_without_the_files_are_not_comparable(runs, tmp_path):
    reference, candidate = runs["haldane_bands"]
    reference = copy(reference, tmp_path / "reference")
    candidate = copy(candidate, tmp_path / "candidate")
    for directory in (reference, candidate):
        for params in directory.glob("*_params.json"):
            params.unlink()
    document = read(candidate / QUEER_RESULTS)
    document["provenance"]["params_sha256"] = "0" * 64
    write(candidate / QUEER_RESULTS, document)
    report = compare("QUEER_BANDS", reference, candidate)
    assert report["verdict"] == NOT_COMPARABLE
    assert list(failed(report)) == ["parameters"]


def test_execution_settings_and_the_atom_order_are_not_step_parameters(runs, tmp_path):
    reference, _ = runs["haldane_bands"]
    structure = read(reference / "pythtb_params.json")["structure"]
    reordered = {"species": structure["species"][::-1], "cart": structure["cart"][::-1]}
    candidate = run_case(
        "haldane_bands", tmp_path / "candidate", workers=2, chunk_size=7, structure=reordered
    )
    assert compare("QUEER_BANDS", reference, candidate)["verdict"] == PASS


def test_runs_of_other_steps_or_wannier_jobs_are_not_comparable(runs, tmp_path):
    reference, candidate = runs["haldane_bands"]
    report = compare("QUEER_BERRY", reference, candidate)
    assert report["verdict"] == NOT_COMPARABLE and list(failed(report)) == ["step"]
    other = run_case("haldane_bands", tmp_path / "other", source_wannier_job_id="job-other")
    report = compare("BANDS", reference, other)
    assert report["verdict"] == NOT_COMPARABLE
    assert {"source_wannier_job", "parameters"} == set(failed(report))


def test_different_inputs_are_not_comparable(runs, tmp_path):
    reference, candidate = runs["haldane_bands"]
    candidate = copy(candidate, tmp_path / "candidate")
    document = read(candidate / QUEER_RESULTS)
    document["provenance"]["inputs"]["wannier90_hr.dat"]["sha256"] = "0" * 64
    write(candidate / QUEER_RESULTS, document)
    report = compare("QUEER_BANDS", reference, candidate)
    assert report["verdict"] == NOT_COMPARABLE
    assert "wannier90_hr.dat" in failed(report)["inputs"]


def test_missing_or_ambiguous_documents_are_not_comparable(runs, tmp_path):
    reference, candidate = runs["haldane_bands"]
    report = compare("QUEER_BANDS", reference, tmp_path / "absent")
    assert report["verdict"] == NOT_COMPARABLE and "does not exist" in failed(report)["documents"]
    both = copy(candidate, tmp_path / "both")
    shutil.copyfile(both / QUEER_RESULTS, both / DRIVER_RESULTS)
    report = compare("QUEER_BANDS", reference, both)
    assert report["verdict"] == NOT_COMPARABLE and "holds both" in failed(report)["documents"]
    with pytest.raises(ValueError, match="unknown step"):
        compare("QUEER_QGEOM", reference, candidate)


def test_documents_of_an_unexpected_shape_are_reported_not_raised(runs, tmp_path):
    reference, candidate = runs["haldane_topology"]
    candidate = copy(candidate, tmp_path / "candidate")
    document = read(candidate / QUEER_RESULTS)
    document["topology"]["occupation"] = [1, 2]
    write(candidate / QUEER_RESULTS, document)
    report = compare("QUEER_TOPOLOGY", reference, candidate)
    assert report["verdict"] == NOT_COMPARABLE
    assert "could not be checked: AttributeError" in failed(report)["derived_decisions"]
    document = read(runs["haldane_topology"][1] / QUEER_RESULTS)
    document["topology"]["planes"][0]["file"] = 7
    candidate = copy(runs["haldane_topology"][1], tmp_path / "planes")
    write(candidate / QUEER_RESULTS, document)
    report = compare("QUEER_TOPOLOGY", reference, candidate)
    assert report["verdict"] == FAIL
    assert "topology.plane[k3=0].invariants" in report["summary"]["unexplained"]


# ---------------------------------------------------------------------------
# Reference semantics and documented differences
# ---------------------------------------------------------------------------


def legacy_bands_reference(source, target, rule="first_nonnegative_bin"):
    """A QUEER BANDS run rewritten as the first driver revision recorded it."""
    target = copy(source, target)
    document = as_driver_document(target, "PYTHTB_BANDS", LEGACY_CORE)
    del document["provenance"]["inputs"]
    dos_file = read(target / "Out" / "dos.json")
    document["electronic"]["dos_at_fermi_per_eV"] = dos_at_fermi(
        dos_file["energy_eV"], dos_file["dos_per_eV"], rule
    )
    mesh, axes = document["dos"].pop("mesh"), document["dos"].pop("mesh_axes")
    document["dos"]["mesh"] = [mesh[axes.index(axis)] if axis in axes else mesh[0] for axis in range(3)]
    document["validation"]["localization"] = {"decay_length_A": 71.0, "max_distance_A": 86.0}
    del document["fermi_surface"]["file"]
    write(target / DRIVER_RESULTS, document)
    surface = read(target / "Out" / "fermi_surface.json")
    for key in ("k_frac_range", "k_frac_range_end", "axes_cart", "units"):
        surface.pop(key)
    surface["contours_cart"] = {band: [[[0.0, 0.0], [1.0, 1.0]]] for band in surface["contours_cart"]}
    write(target / "Out" / "fermi_surface.json", surface)
    return target


def test_a_first_driver_reference_passes_under_its_semantics(runs, tmp_path):
    source, candidate = runs["haldane_bands"]
    reference = legacy_bands_reference(source, tmp_path / "legacy")
    linear = read(candidate / QUEER_RESULTS)["electronic"]["dos_at_fermi_per_eV"]
    legacy = read(reference / DRIVER_RESULTS)["electronic"]["dos_at_fermi_per_eV"]
    assert abs(linear - legacy) > 1e-3 * abs(linear)
    report = compare("PYTHTB_BANDS", reference, candidate)
    assert report["verdict"] == PASS, report["summary"]
    assert report["semantics"]["profile"] == "driver-9f43886" and report["semantics"]["key"] == LEGACY_CORE
    assert report["inputs_verified_by_fingerprint"] is True
    kinds = classes(report)
    assert kinds["electronic.dos_at_fermi_per_eV"] == AFTER_CONVENTION
    assert kinds["dos.mesh"] == AFTER_CONVENTION
    assert kinds["fermi_surface.contours"] == NOT_APPLICABLE
    assert kinds["fermi_surface.bands_crossing"] == AGREE
    assert "localization" in next(
        item["detail"] for item in report["quantities"] if item["name"] == "validation"
    )


def test_the_first_driver_rule_is_applied_not_ignored(runs, tmp_path):
    source, candidate = runs["haldane_bands"]
    reference = legacy_bands_reference(source, tmp_path / "legacy", rule="linear_interpolation")
    report = compare("PYTHTB_BANDS", reference, candidate)
    assert report["verdict"] == FAIL
    assert report["summary"]["unexplained"] == ["electronic.dos_at_fermi_per_eV"]


def test_a_fingerprint_mismatch_is_not_comparable(runs, tmp_path):
    source, candidate = runs["haldane_bands"]
    reference = legacy_bands_reference(source, tmp_path / "legacy")
    document = read(reference / DRIVER_RESULTS)
    document["model"]["lattice"][0][0] += 1e-8
    document["model"]["ws_correction"]["scores"]["off"]["max_meV"] += 1e-5
    write(reference / DRIVER_RESULTS, document)
    report = compare("PYTHTB_BANDS", reference, candidate)
    assert report["verdict"] == NOT_COMPARABLE and list(failed(report)) == ["inputs"]
    assert "lattice" in failed(report)["inputs"] and "scores.off.max_meV" in failed(report)["inputs"]
    assert report["inputs_verified_by_fingerprint"] is False


def prefix_berry_reference(source, target, *, conventions=False):
    """A fermi-mode QUEER BERRY run rewritten as a driver revision before the sign fix."""
    target = copy(source, target)
    document = as_driver_document(target, "PYTHTB_BERRY")
    berry = document["berry"]
    if not conventions:
        del berry["conventions"]
    for key in ("fermi_sea_integral_e2_h", "sigma_xy_e2_h"):
        berry[key] = -berry[key]
    write(target / DRIVER_RESULTS, document)
    for name, keys in (
        ("berry_k3_0.json", ("omega_reduced", "omega_cart_A2")),
        ("berry_path.json", ("omega_cart_A2",)),
    ):
        data = read(target / "Out" / name)
        for key in keys:
            data[key] = [-value for value in data[key]]
        write(target / "Out" / name, data)
    return target


def test_a_fermi_mode_reference_without_conventions_passes_after_the_sign(runs, tmp_path):
    source, candidate = runs["haldane_berry_fermi"]
    reference = prefix_berry_reference(source, tmp_path / "prefix")
    report = compare("PYTHTB_BERRY", reference, candidate)
    assert report["verdict"] == PASS, report["summary"]
    assert report["semantics"]["berry_sign_applied_to_reference"] is True
    kinds = classes(report)
    for name in (
        "berry.omega_reduced",
        "berry.omega_cart_A2",
        "berry.fermi_sea_integral_e2_h",
        "berry.sigma_xy_e2_h",
        "berry.path.omega_cart_A2",
        "berry.conventions",
    ):
        assert kinds[name] == AFTER_CONVENTION, name
    assert report["flagged_sets"]["berry.omega_reduced"]["size"] == 0


def test_flipped_signs_with_recorded_conventions_fail(runs, tmp_path):
    source, candidate = runs["haldane_berry_fermi"]
    reference = prefix_berry_reference(source, tmp_path / "flipped", conventions=True)
    report = compare("PYTHTB_BERRY", reference, candidate)
    assert report["verdict"] == FAIL
    assert {"berry.omega_reduced", "berry.fermi_sea_integral_e2_h"} <= set(report["summary"]["unexplained"])


def test_a_few_near_degenerate_points_are_flagged_and_bounded(runs, tmp_path):
    reference, candidate = runs["haldane_berry_fermi"]
    candidate = copy(candidate, tmp_path / "candidate")
    plane = read(candidate / "Out" / "berry_k3_0.json")
    document = read(candidate / QUEER_RESULTS)
    mesh = document["berry"]["mesh"]
    index = int(np.argmax(np.abs(plane["omega_reduced"])))
    # One point where an occupied-empty pair at the skip threshold counted in one run only.
    change = 0.5 * plane["omega_reduced"][index]
    plane["omega_reduced"][index] += change
    plane["omega_cart_A2"][index] *= 1.5
    write(candidate / "Out" / "berry_k3_0.json", plane)
    shift = change / (mesh * mesh * 2 * np.pi)
    document["berry"]["fermi_sea_integral_e2_h"] += shift
    document["berry"]["sigma_xy_e2_h"] -= shift
    document["berry"]["omega_cart_max_A2"] = float(np.abs(plane["omega_cart_A2"]).max())
    write(candidate / QUEER_RESULTS, document)
    report = compare("QUEER_BERRY", reference, candidate)
    assert report["verdict"] == PASS_DOCUMENTED, report["summary"]
    assert report["flagged_sets"]["berry.omega_reduced"]["size"] == 1
    assert report["flagged_sets"]["berry.omega_reduced"]["indices"] == [index]
    kinds = classes(report)
    assert kinds["berry.omega_reduced"] == kinds["berry.fermi_sea_integral_e2_h"] == DOCUMENTED


def test_bin_crossings_are_allowed_only_with_a_weyl_bound(runs, tmp_path):
    reference, candidate = runs["haldane_bands"]
    candidate = copy(candidate, tmp_path / "candidate")
    dos = read(candidate / "Out" / "dos.json")
    document = read(candidate / QUEER_RESULTS)
    step = float(np.diff(dos["energy_eV"][:2])[0])
    moved = np.zeros(len(dos["energy_eV"]))
    middle = len(moved) // 2
    moved[middle], moved[middle + 1] = -1.0, 1.0
    smear = gaussian_dos(moved / document["dos"]["k_points"], step, document["dos"]["sigma_eV"])
    dos["dos_per_eV"] = (np.asarray(dos["dos_per_eV"]) + smear).tolist()
    write(candidate / "Out" / "dos.json", dos)
    assert compare("QUEER_BANDS", reference, candidate)["summary"]["unexplained"] == ["dos.dos_per_eV"]
    document["model"]["hermiticity_weyl_bound"] = 1e-7
    write(candidate / QUEER_RESULTS, document)
    report = compare("QUEER_BANDS", reference, candidate)
    assert report["verdict"] == PASS_DOCUMENTED, report["summary"]
    assert classes(report)["dos.dos_per_eV"] == DOCUMENTED
    assert report["tolerance_basis"]["eigenvalue_eV"] == pytest.approx(1e-7, rel=1e-3)


def test_slab_levels_at_the_window_edge_on_one_side_pass(runs, tmp_path):
    reference, candidate = runs["haldane_slab"]
    candidate = copy(candidate, tmp_path / "candidate")
    ribbon = read(candidate / "Out" / "ribbon_bands.json")
    window = read(candidate / QUEER_RESULTS)["slab"]["window_eV"]
    # A level at the window edge, absent from the reference, at a k-point whose window is
    # otherwise empty: far from the levels taking part, it changes no flow.
    index = next(k for k, levels in enumerate(ribbon["energies_eV"]) if not levels)
    ribbon["energies_eV"][index].append(window)
    for key, value in (("left_weight", 0.9), ("right_weight", 0.0), ("left_half_weight", 1.0)):
        ribbon[key][index].append(value)
    ribbon["interior_weight"][index].append(0.0)
    write(candidate / "Out" / "ribbon_bands.json", ribbon)
    report = compare("QUEER_SLAB", reference, candidate)
    assert report["verdict"] == PASS, report["summary"]
    assert "window edge" in next(
        item["detail"] for item in report["quantities"] if item["name"] == "slab.levels"
    )
    ribbon["energies_eV"][index][-1] = 0.0
    write(candidate / "Out" / "ribbon_bands.json", ribbon)
    report = compare("QUEER_SLAB", reference, candidate)
    assert "slab.levels" in report["summary"]["unexplained"]


def synthetic_ribbon(source, target, touch):
    """A run whose ribbon is two chiral modes and a level reaching ``touch`` eV at one k-point.

    The document's crossings and bulk check are rewritten from the file, so the run is
    consistent with itself.
    """
    target = copy(source, target)
    ribbon = read(target / "Out" / "ribbon_bands.json")
    count = len(ribbon["energies_eV"])
    ks = -0.5 + (np.arange(count) + 0.5) / count
    energies, half, middle = [], [], []
    for index, k in enumerate(ks):
        top = touch if index == count // 4 else -0.06
        energies.append(sorted([3 * k * 0.1, -3 * k * 0.1, top]))
        weights = {3 * k * 0.1: 1.0, -3 * k * 0.1: 0.0, top: 1.0}
        half.append([weights[e] for e in energies[-1]])
        middle.append([0.0, 0.0, 0.0])
    ribbon.update(
        energies_eV=energies,
        left_weight=half,
        right_weight=[[1.0 - x for x in row] for row in half],
        left_half_weight=half,
        interior_weight=middle,
    )
    write(target / "Out" / "ribbon_bands.json", ribbon)
    document = read(target / QUEER_RESULTS)
    slab = document["slab"]
    slab["bulk_at_fermi"] = {"metal": False, "gap_eV": [-1.0, 1.0], "k_perp_points": 2 * slab["cells"]}
    slab["edge_crossings_at_fermi"] = to_jsonable(
        edge_crossings(energies, half, middle, periodic=True, window=slab["window_eV"], bulk_metal=False)
    )
    write(target / QUEER_RESULTS, document)
    return target


def test_a_crossing_decided_within_tolerance_of_e_f_is_documented(runs, tmp_path):
    # A level reaching just below E_F in one run and just above it in the other: one run
    # counts it up and down again at E_F, the other not at all. Within τ_E either is right.
    source = runs["haldane_slab"][1]
    reference = synthetic_ribbon(source, tmp_path / "below", -2e-9)
    candidate = synthetic_ribbon(source, tmp_path / "above", 2e-9)
    crossings = [
        read(run / QUEER_RESULTS)["slab"]["edge_crossings_at_fermi"] for run in (reference, candidate)
    ]
    assert crossings[0]["left"] != crossings[1]["left"]
    report = compare("QUEER_SLAB", reference, candidate)
    kinds = classes(report)
    assert kinds["slab.edge_crossings_at_fermi"] == DOCUMENTED, report["summary"]
    assert kinds["slab.edge_crossings_recomputed"] == AGREE
    assert report["verdict"] == PASS_DOCUMENTED, report["summary"]
    # Far from E_F the same difference is unexplained.
    reference = synthetic_ribbon(source, tmp_path / "far-below", -2e-3)
    candidate = synthetic_ribbon(source, tmp_path / "far-above", 2e-3)
    report = compare("QUEER_SLAB", reference, candidate)
    assert classes(report)["slab.edge_crossings_at_fermi"] == UNEXPLAINED


def test_reworded_driver_warnings_are_documented(runs, tmp_path):
    source, candidate = runs["haldane_bands"]
    reference = copy(source, tmp_path / "reference")
    document = as_driver_document(reference, "PYTHTB_BANDS")
    document["warnings"] = [
        "PythTB bands differ from wannier90_band.dat by up to 6.0 meV inside the frozen window."
    ]
    write(reference / DRIVER_RESULTS, document)
    candidate = copy(candidate, tmp_path / "candidate")
    queer = read(candidate / QUEER_RESULTS)
    queer["warnings"] = [
        "Model bands differ from wannier90_band.dat by up to 6.0 meV inside the frozen window."
    ]
    queer["warning_codes"] = ["band_dat_deviation"]
    write(candidate / QUEER_RESULTS, queer)
    report = compare("QUEER_BANDS", reference, candidate)
    assert report["verdict"] == PASS_DOCUMENTED
    assert classes(report)["warnings"] == DOCUMENTED
    queer["warnings"], queer["warning_codes"] = [], []
    write(candidate / QUEER_RESULTS, queer)
    assert compare("QUEER_BANDS", reference, candidate)["summary"]["unexplained"] == ["warnings"]


def test_two_failed_runs_with_equivalent_codes_pass(runs, tmp_path):
    failing = tmp_path / "queer"
    failing.mkdir()
    (failing / "pythtb_params.json").write_text(
        json.dumps({"step_type": "PYTHTB_BANDS", "fermi_energy": 0.0})
    )
    assert run(None, failing / "pythtb_params.json", failing, log=lambda line: None) == 2
    assert read(failing / QUEER_RESULTS)["error"]["code"] == "queer_missing_input"
    driver = copy(failing, tmp_path / "driver")
    document = read(driver / QUEER_RESULTS)
    (driver / QUEER_RESULTS).unlink()
    document.update(step_type="PYTHTB_BANDS", error={"code": "pythtb_missing_input", "message": "no _hr.dat"})
    write(driver / DRIVER_RESULTS, document)
    report = compare("PYTHTB_BANDS", driver, failing)
    assert report["verdict"] == PASS and classes(report) == {"status": AGREE}
    document["error"]["code"] = "pythtb_bad_params"
    write(driver / DRIVER_RESULTS, document)
    assert compare("PYTHTB_BANDS", driver, failing)["verdict"] == FAIL
    assert compare("PYTHTB_BANDS", runs["haldane_bands"][0], failing)["verdict"] == FAIL


# ---------------------------------------------------------------------------
# Helpers, report and command
# ---------------------------------------------------------------------------


def test_step_names_and_warning_codes():
    assert {step_family(name) for name in ("QUEER_BANDS", "pythtb_bands", "BANDS")} == {"BANDS"}
    with pytest.raises(ValueError):
        step_family("PYTHTB_QGEOM")
    assert (
        warning_code("PythTB bands differ from wannier90_band.dat by up to 6.0 meV") == "band_dat_deviation"
    )
    assert warning_code("path curvature plot skipped: no display") == "plot_skipped"
    assert warning_code("path curvature skipped: no path") == "path_curvature_skipped"
    assert warning_code("berry_k3_0.png skipped: no display") == "plot_skipped"
    assert warning_code("something new") == "message:something new"


def test_circular_matching_and_contour_distances():
    assert circular_matching_distance([0.05, 0.5], [0.4999999999, 0.0500000001]) == pytest.approx(
        1e-10, abs=1e-15
    )
    assert circular_matching_distance([0.999999, 0.3], [0.3, 0.000001]) == pytest.approx(2e-6)
    assert circular_matching_distance([0.1, 0.2], [0.1]) == np.inf
    circle = [[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 9)]
    finer = [[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 17)]
    assert contour_hausdorff([circle], [circle[4:], circle[:5]]) == 0.0
    sagitta = 1 - np.cos(np.pi / 8)
    assert contour_hausdorff([circle], [finer]) == pytest.approx(sagitta, rel=1e-12)
    shifted = [[x + 1e-3, y] for x, y in circle]
    assert contour_hausdorff([circle], [shifted]) == pytest.approx(1e-3, rel=1e-6)
    assert contour_hausdorff([], []) == 0.0 and contour_hausdorff([circle], []) == np.inf


def test_the_markdown_page_lists_the_checks(runs):
    report = compare("QUEER_BERRY", *runs["haldane_berry_fermi"])
    page = render_markdown(report)
    assert page.startswith("# Parity of BERRY: pass\n")
    assert "## Preconditions" in page and "| derived_decisions | yes |" in page
    assert "| berry.fermi_sea_integral_e2_h | agree |" in page
    assert "## Flagged sets" in page


def test_the_compare_command_prints_the_verdict_and_writes_reports(runs, tmp_path, capsys):
    reference, candidate = runs["haldane_topology"]
    arguments = [
        "compare",
        "--step",
        "QUEER_TOPOLOGY",
        "--reference",
        str(reference),
        "--candidate",
        str(candidate),
    ]
    report_path, page_path = tmp_path / "out" / "report.json", tmp_path / "out" / "report.md"
    assert main([*arguments, "--report", str(report_path), "--markdown", str(page_path)]) == 0
    assert capsys.readouterr().out.splitlines()[0] == PASS
    assert read(report_path)["verdict"] == PASS
    assert page_path.read_text().startswith("# Parity of TOPOLOGY: pass")
    other = run_case("haldane_topology", tmp_path / "other", loop_points=23)
    assert (
        main(["compare", "--step", "TOPOLOGY", "--reference", str(reference), "--candidate", str(other)]) == 2
    )
    output = capsys.readouterr().out.splitlines()
    assert output[0] == NOT_COMPARABLE and output[1].startswith(
        "  precondition parameters: differing: loop_points"
    )
    assert main(["compare", "--step", "TOPOLOGY", "--reference", str(reference)]) == 2
    assert "--candidate" in capsys.readouterr().err
    assert main(["compare", "--step", "QGEOM", "--reference", "a", "--candidate", "b"]) == 2
    assert "unknown step" in capsys.readouterr().err


def nearest_energy_slab_reference(source, target, core=NEAREST_ENERGY_CORES[0]):
    """A QUEER SLAB run rewritten as a nearest-energy driver recorded it: the counts and nothing else."""
    target = copy(source, target)
    document = as_driver_document(target, "PYTHTB_SLAB", core)
    ribbon = read(target / "Out" / "ribbon_bands.json")
    for key in ("left_half_weight", "interior_weight"):
        ribbon.pop(key)
    write(target / "Out" / "ribbon_bands.json", ribbon)
    document["slab"].pop("bulk_at_fermi")
    document["slab"]["edge_crossings_at_fermi"] = legacy_edge_crossings(
        ribbon["energies_eV"], ribbon["left_weight"], ribbon["right_weight"], periodic=True
    )
    write(target / DRIVER_RESULTS, document)
    return target


@pytest.mark.parametrize("core", [*NEAREST_ENERGY_CORES, LEGACY_CORE, "0" * 64])
def test_a_nearest_energy_reference_meets_its_count_recomputed_from_the_candidate(runs, tmp_path, core):
    # A reference whose crossings record no method counted by nearest energy, whatever its core.
    source, candidate = runs["haldane_slab"]
    reference = nearest_energy_slab_reference(source, tmp_path / "legacy", core)
    report = compare("PYTHTB_SLAB", reference, candidate)
    assert report["verdict"] == PASS, report["summary"]
    assert report["semantics"]["key"] == core
    kinds = classes(report)
    assert kinds["slab.edge_crossings_at_fermi"] == AFTER_CONVENTION
    assert kinds["slab.edge_crossings_status"] == kinds["slab.edge_crossings_per_energy"] == NOT_APPLICABLE
    assert kinds["slab.half_ribbon_weights"] == kinds["slab.bulk_at_fermi"] == NOT_APPLICABLE
    # The candidate's own weighted flow is still checked against its stored levels.
    assert kinds["slab.edge_crossings_recomputed"] == AGREE
    broken = copy(candidate, tmp_path / "candidate")
    document = read(broken / QUEER_RESULTS)
    document["slab"]["edge_crossings_at_fermi"]["left"]["up"] += 5
    write(broken / QUEER_RESULTS, document)
    report = compare("PYTHTB_SLAB", reference, broken)
    assert report["verdict"] == FAIL
    assert report["summary"]["unexplained"] == ["slab.edge_crossings_recomputed"]
    # The recount is applied, not skipped: other reference counts fail.
    document = read(reference / DRIVER_RESULTS)
    document["slab"]["edge_crossings_at_fermi"]["right"]["down"] += 1
    document["slab"]["edge_crossings_at_fermi"]["right"]["net"] -= 1
    write(reference / DRIVER_RESULTS, document)
    report = compare("PYTHTB_SLAB", reference, candidate)
    assert report["verdict"] == FAIL
    assert report["summary"]["unexplained"] == ["slab.edge_crossings_at_fermi"]


def test_the_crossing_status_and_flows_are_compared(runs, tmp_path):
    source, candidate = runs["haldane_slab"]

    def status(crossings):
        crossings["status"] = "unresolved"

    def flows(crossings):
        crossings["per_energy"][2]["left_flow"] += 0.01

    def shifted(crossings, by):
        crossings["window_eV"] = [value + by for value in crossings["window_eV"]]

    def rule(crossings):
        crossings["method"] = "rank_continuity"

    changes = [
        ("status", status, ["slab.edge_crossings_status"]),
        ("flows", flows, ["slab.edge_crossings_per_energy"]),
        ("far", lambda crossings: shifted(crossings, 1e-6), ["slab.edge_crossings_per_energy"]),
        ("near", lambda crossings: shifted(crossings, 1e-9), []),
        ("rule", rule, list(parity.CROSSING_QUANTITIES)),
    ]
    for name, change, expected in changes:
        reference = copy(source, tmp_path / name)
        document = read(reference / QUEER_RESULTS)
        change(document["slab"]["edge_crossings_at_fermi"])
        write(reference / QUEER_RESULTS, document)
        report = compare("QUEER_SLAB", reference, candidate)
        assert sorted(report["summary"]["unexplained"]) == sorted(expected), name
        assert report["verdict"] == (FAIL if expected else PASS), name


def test_the_bulk_check_and_half_ribbon_weights_are_compared(runs, tmp_path):
    source, candidate = runs["haldane_slab"]
    reference = copy(source, tmp_path / "bulk")
    document = read(reference / QUEER_RESULTS)
    document["slab"]["bulk_at_fermi"]["gap_eV"][1] += 1e-6
    write(reference / QUEER_RESULTS, document)
    assert compare("QUEER_SLAB", reference, candidate)["summary"]["unexplained"] == ["slab.bulk_at_fermi"]
    reference = copy(source, tmp_path / "weights")
    ribbon = read(reference / "Out" / "ribbon_bands.json")
    ribbon["interior_weight"][3][0] += 1e-6
    write(reference / "Out" / "ribbon_bands.json", ribbon)
    report = compare("QUEER_SLAB", reference, candidate)
    assert report["summary"]["unexplained"] == ["slab.half_ribbon_weights"]


def test_the_registry_names_the_first_driver_revision():
    assert parity.REFERENCE_SEMANTICS[LEGACY_CORE].dos_at_fermi_rule == "first_nonnegative_bin"
    assert parity.semantics_of({"provenance": {"core_sha256": LEGACY_CORE}})[0].name == "driver-9f43886"
    # The edge-crossing rule of a reference is read from its structure, not its core.
    assert list(parity.REFERENCE_SEMANTICS) == [LEGACY_CORE]
    for core in NEAREST_ENERGY_CORES:
        assert parity.semantics_of({"provenance": {"core_sha256": core}})[0] is parity.CURRENT_SEMANTICS
    assert "edge_crossings_rule" not in parity.CURRENT_SEMANTICS.rules()
    assert parity.semantics_of({"engine": "queer", "provenance": {}})[0] is parity.CURRENT_SEMANTICS
    assert parity.passing({"verdict": PASS_DOCUMENTED}) and not parity.passing({"verdict": UNEXPLAINED})
