"""Independent native-family acceptance for orbital identity and critical roots."""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from queer.capabilities import CapabilityError
from queer.execution import ExecutionOptions
from queer.finite_field.family import FamilyMember, GridLoader, ModelFamily, _find_roots, match_orbitals
from queer.finite_field.models.fourband import FourBandParams
from queer.finite_field.models.lattice_reg import lattice_regularized_fourband
from queer.finite_field.native import NativePlaneModel
from queer.models.openmx import OpenMXModel
from queer.models.wannier import WannierModel
from queer.observables.result import ObservableResult


def native_gamma(h):
    source = WannierModel([[0, 0, 0]], np.asarray(h, complex)[None], lattice=np.eye(3))
    return NativePlaneModel(source, band_pair=(0, 1))


def native_family(models, parameters):
    calls = []
    members = []
    for i, (model, parameter) in enumerate(zip(models, parameters, strict=True)):

        def load(model=model, i=i):
            calls.append(i)
            return model

        members.append(FamilyMember(f"m{i}", float(parameter), {}, load))
    return ModelFamily(members), calls


def lattice_family(xs, permutation=None):
    params = FourBandParams()
    models = []
    for i, x in enumerate(xs):
        source = lattice_regularized_fourband(params.at(x))
        if permutation is not None and i == 1:
            p = np.asarray(permutation)
            source = WannierModel(
                source.translations,
                source.hopping_matrices[:, p][:, :, p],
                lattice=source.lattice,
                centres=source.centres[p],
            )
        models.append(NativePlaneModel(source, band_pair=(1, 2)))
    family, calls = native_family(models, np.asarray(xs) * params.vlc)
    return family, params.vlc, calls


def test_orbital_matching_retains_colocated_identities_and_periodic_images():
    reference = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, -5.0], [0.0, 0.0, 5.0], [0.0, 0.0, -5.0]])
    permutation, drift = match_orbitals(reference, reference + 0.01)
    assert permutation.tolist() == [0, 1, 2, 3]
    assert_allclose(drift, 0.01)
    permutation, drift = match_orbitals(reference, reference[[3, 1, 2, 0]])
    assert permutation.tolist() == [3, 1, 2, 0]
    assert_allclose(drift, 0)
    lattice = np.array([[2.0, 0.3, 0.0], [0.4, 1.7, 0.0], [0.0, 0.0, 30.0]])
    shifted = reference + np.outer([1, -1, 2, 0], lattice[0])
    permutation, drift = match_orbitals(reference, shifted, lattice)
    assert permutation.tolist() == [0, 1, 2, 3]
    assert_allclose(drift, 0, atol=2e-15)


def test_detectable_orbital_permutation_keeps_critical_root_and_records_correspondence():
    family, scale, calls = lattice_family([0.5, 0.8, 0.95, 1.05, 1.2, 1.5], [3, 1, 2, 0])
    report = family.critical_D_report()
    assert report["D_c"] / scale == pytest.approx(1.0, abs=1e-4)
    assert report["orbital_permutations"] == {"m1": [3, 1, 2, 0]}
    assert "orbital_permutation" in report["flags"][1]
    consistency = family.consistency()
    assert not consistency["ok"]
    assert consistency["orbital_permutations"] == report["orbital_permutations"]
    assert consistency["max_z_centre_drift"] == consistency["max_inplane_centre_drift"] == 0
    assert len(family._models) <= 2
    assert sorted(calls) == list(range(len(family)))
    json.dumps(report, allow_nan=False)


def test_invisible_orbital_relabelling_rejects_spurious_roots():
    family, scale, _ = lattice_family([0.5, 0.8, 0.95, 1.05, 1.2, 1.5], [2, 1, 0, 3])
    report = family.critical_D_report()
    assert [root["accepted"] for root in report["roots"]] == [False, False, True]
    assert report["orbital_permutations"] == {}
    with pytest.warns(RuntimeWarning, match="2 sign change"):
        assert family.critical_D() / scale == pytest.approx(1.0, abs=1e-4)


def test_avoided_crossing_rejects_sign_jump_and_keeps_evidence():
    parameters = [0.0, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0]
    family, _ = native_family(
        [native_gamma([[d - 0.5, 0.05], [0.05, 0.5 - d]]) for d in parameters], parameters
    )
    report = family.critical_D_report()
    assert min(abs(np.asarray(report["signed_gap"]))) > 0.1
    (root,) = report["roots"]
    assert not root["accepted"] and "jumps" in root["reason"]
    assert root["mismatch"] > 0.2
    assert report["D_c"] is None
    with pytest.warns(RuntimeWarning, match="rejected"):
        assert family.critical_D() is None


def test_lost_character_inside_root_bracket_is_null_and_rejected():
    parameters = [0.0, 1.0, 1.8, 3.0, 4.0]
    weights = np.sqrt([0.45, 0.0, 0.35, 0.2])
    unit = (np.eye(4)[0] - weights) / np.linalg.norm(np.eye(4)[0] - weights)
    rotation = np.eye(4) - 2 * np.outer(unit, unit)
    models = []
    for i, d in enumerate(parameters):
        h = np.diag([d - 2, 2 - d, 5, 6])
        models.append(native_gamma(rotation @ h @ rotation.T if i == 2 else h))
    family, _ = native_family(models, parameters)
    report = family.critical_D_report()
    assert report["flags"][2] == ["character_lost"]
    assert report["signed_gap"][2] is None
    (root,) = report["roots"]
    assert not root["accepted"] and "lost" in root["reason"]
    assert root["D"] is None
    assert report["D_c"] is None
    json.dumps(report, allow_nan=False)


def test_multiple_inversions_preserve_all_roots_and_require_explicit_choice():
    family, scale, _ = lattice_family([-1.5, -1.2, -0.8, -0.5, 0, 0.5, 0.8, 1.2, 1.5])
    report = family.critical_D_report()
    assert_allclose(np.asarray(report["accepted_roots"]) / scale, [-1, 1], atol=2e-3)
    assert report["D_c"] is None
    with pytest.raises(ValueError, match="2 gap closings"):
        family.critical_D()


@pytest.mark.parametrize("xs", [[0.3, 0.7, 1.1, 1.5], [0.2, 0.6, 1.4, 1.8], [0, 0.5, 1.5, 2]])
def test_coarse_native_members_accept_clean_unmixed_inversion(xs):
    family, scale, _ = lattice_family(xs)
    report = family.critical_D_report()
    (root,) = report["roots"]
    assert root["accepted"]
    assert root["reason"].startswith("band inversion")
    assert root["mixing"] < 1e-20
    assert abs(report["D_c"] / scale - 1) < 0.03


def test_first_closed_member_uses_gapped_reference():
    parameters = [0.0, 1.0, 2.0]
    family, _ = native_family([native_gamma(np.diag([-d, d])) for d in parameters], parameters)
    report = family.critical_D_report()
    assert report["reference"] == "m1"
    assert "degenerate" in report["flags"][0]
    assert report["D_c"] == 0


@pytest.mark.parametrize(
    "gap,reason",
    [
        ([0.1, 0.2, -0.2, -0.1], "does not fall towards the bracket"),
        ([1.0, 0.99, -0.5, -1.0], "more than one member spacing outside"),
    ],
)
def test_clean_character_swap_still_requires_gross_closing_trend(gap, reason):
    (root,) = _find_roots(np.arange(4.0), gap, 0.1, mixing=np.zeros(4))
    assert not root["accepted"] and reason in root["reason"]
    (curved,) = _find_roots(np.arange(4.0), [0.9, 0.5, -0.5, -0.9], 0.1, mixing=np.zeros(4))
    assert curved["accepted"] and "outside the bracket" in curved["v_test"]


def test_character_loss_reason_preserves_indices_of_grid_only_members():
    (root,) = _find_roots(
        np.arange(5.0),
        [1.0, np.nan, 0.5, np.nan, -0.5],
        0.1,
        present=[True, False, True, True, True],
    )
    assert root["members"] == [2, 4] and "members [3]" in root["reason"]


def test_grid_only_member_does_not_fabricate_lost_wavefunction_character():
    family, scale, _ = lattice_family([0.5, 0.8, 1.05, 1.2, 1.5])
    member = family.members[2]
    family.members[2] = FamilyMember(
        member.label, member.D, {"grid": "unloaded.npz"}, GridLoader("unloaded.npz")
    )
    report = family.critical_D_report()
    assert report["flags"][2] == ["no_hamiltonian"]
    assert report["signed_gap"][2] is None
    (root,) = report["roots"]
    assert root["accepted"] and root["members"] == [1, 3]
    assert abs(report["D_c"] / scale - 1) < 0.005


@pytest.mark.parametrize("permutation,skew", [(None, False), ([1, 0], True)])
def test_nonzero_origin_home_cells_do_not_fabricate_critical_roots(permutation, skew):
    lattice = np.array([[2, 0.3, 0], [0.4, 1.7, 0], [0.1, 0.2, 3]]) if skew else np.eye(3)
    centres = np.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.6]]) @ lattice
    h = np.array([[0.3, 0.4 + 0.2j], [0.4 - 0.2j, -0.3]])
    source = WannierModel([[0, 0, 0]], [h], lattice=lattice, centres=centres)
    shifts = np.array([[1, -1, 0], [-1, 0, 1]])
    blocks = {}
    for i, j in np.ndindex(2, 2):
        key = tuple(shifts[i] - shifts[j])
        blocks.setdefault(key, np.zeros((2, 2), complex))[i, j] = h[i, j]
    p = np.arange(2) if permutation is None else np.asarray(permutation)
    changed = WannierModel(
        list(blocks),
        np.array(list(blocks.values()))[:, p][:, :, p],
        lattice=lattice,
        centres=(centres + shifts @ lattice)[p],
    )
    origin = [0.217, -0.13, 0.07]
    views = [
        NativePlaneModel(model, plane={"origin": origin}, band_pair=(0, 1)) for model in (source, changed)
    ]
    family, _ = native_family(views, [0, 1])
    report = family.critical_D_report()
    expected_gap = 2 * np.sqrt(0.3**2 + 0.4**2 + 0.2**2)
    assert_allclose(report["signed_gap"], expected_gap, atol=5e-16)
    assert_allclose(report["weight_v"], 1, atol=1e-15)
    assert_allclose(report["weight_c"], 1, atol=1e-15)
    assert report["roots"] == [] and report["D_c"] is None
    assert report["home_cell_shifts"]["m1"] == shifts[p].tolist()
    assert family.consistency()["home_cell_shifts"] == report["home_cell_shifts"]
    if permutation is not None:
        assert report["orbital_permutations"] == {"m1": permutation}
    else:
        assert family.consistency()["ok"]


@pytest.mark.parametrize("unavailable", ["centres", "origin", "lattice"])
def test_nonzero_origin_unavailable_tracking_raises_and_consistency_reports(unavailable):
    centres = None if unavailable == "centres" else [[0.1, 0.2, 0], [0.7, 0.3, 0]]
    source = WannierModel([[0, 0, 0]], [[[0, 1], [1, 0]]], lattice=np.eye(3), centres=centres)
    changed = WannierModel(
        [[0, 0, 0]],
        [[[0, 1], [1, 0]]],
        lattice=np.diag([2, 1, 1]) if unavailable == "lattice" else np.eye(3),
        centres=centres,
    )
    views = [
        NativePlaneModel(
            model,
            plane={"origin": [0.13 + (0.1 if i and unavailable == "origin" else 0), 0, 0]},
            band_pair=(0, 1),
        )
        for i, model in enumerate((source, changed))
    ]
    family, _ = native_family(views, [0, 1])
    with pytest.raises(CapabilityError):
        family.critical_D_report()
    consistency = family.consistency()
    assert not consistency["ok"] and consistency["issues"]
    assert consistency["gamma_character"]["available"] is False
    assert consistency["n_orb"] == [2] and consistency["band_pairs"] == [[0, 1]]
    assert consistency["max_lattice_diff"] == (1 if unavailable == "lattice" else 0)
    if unavailable == "centres":
        assert consistency["max_inplane_centre_drift"] is None
        assert consistency["max_z_centre_drift"] is None
    json.dumps(consistency, allow_nan=False)


def test_shared_single_level_is_lost_character_even_below_half_weight():
    phi = np.arccos(-2 / 3)
    a = np.sqrt([0.4, 0.3, 0.3]).astype(complex)
    b = a * np.exp(1j * np.array([0, phi, -phi]))
    c = np.conj(np.cross(a, b))
    c /= np.linalg.norm(c)
    rotation = np.stack([a, b, c])
    h = np.diag([-1, 1, 2])
    family, _ = native_family(
        [native_gamma(h), native_gamma(rotation @ h @ rotation.conj().T), native_gamma(h)], [0, 1, 2]
    )
    report = family.critical_D_report(min_weight=0.3)
    assert report["flags"][1] == ["character_lost"]
    assert report["signed_gap"][1] is None
    assert report["roots"] == []


def test_gamma_gap_is_centered_independently_of_large_absolute_offset():
    h = 5 * np.eye(2) + 1e-12 * np.array([[0, 1], [1, 0]])
    family, _ = native_family([native_gamma(h)], [0])
    gamma = family.gamma_data(0)
    assert gamma.energy_reference == 5
    assert gamma.relative_eps[1] - gamma.relative_eps[0] == pytest.approx(2e-12, rel=1e-14)
    report = family.critical_D_report()
    assert report["signed_gap"] == pytest.approx([2e-12], rel=1e-14)
    assert report["energy_references_eV"] == [5]
    assert report["assembly_precision_floors_eV"][0] > 0
    assert "assembly" in report["precision_caveat"]


def test_family_harmonic_members_inherit_thread_and_memory_budgets(monkeypatch):
    from queer.finite_field import api

    monkeypatch.setattr("queer.execution.cpu_budget", lambda: 4)
    seen = []
    options = ExecutionOptions(workers=2, batch_size=7, native_threads=2, memory_mb=18)
    family, _ = native_family([native_gamma(np.diag([-1, 1]))], [0])

    class FakePrepared:
        def __init__(self, model, options, execution, cache_dir):
            seen.append(execution)

        def prepare(self):
            return self

        def harmonics(self, amplitudes, n_max):
            return ObservableResult(
                {"amplitudes": np.asarray(amplitudes), "b": np.zeros((1, 2))},
                {"amplitudes": ("amplitude",), "b": ("amplitude", "harmonic")},
                {"amplitudes": "V/m", "b": "A/m"},
            )

    class SerialPool:
        def __call__(self, jobs):
            return [function(*args, **kwargs) for function, args, kwargs in jobs]

    monkeypatch.setattr(api, "PreparedHallResponse", FakePrepared)
    monkeypatch.setattr("queer.execution._ExecutionSession.parallel", lambda self: SerialPool())
    family.harmonic_map([1e4], cfg="physical_eq8", execution=options)
    assert seen == [ExecutionOptions(workers=1, batch_size=7, native_threads=2, memory_mb=9)]


def test_openmx_only_consistency_retains_static_data_without_eigenvector_comparisons(monkeypatch):
    source = OpenMXModel([[0, 0, 0]], [np.diag([-1.0, 1.0])], lattice=np.eye(3))

    def forbidden(*args, **kwargs):
        raise AssertionError("generalized eigenvectors must not enter cross-member character tracking")

    monkeypatch.setattr(source, "eigenstates", forbidden)
    family, _ = native_family([NativePlaneModel(source, band_pair=(0, 1))], [0])
    report = family.consistency()
    assert report["n_orb"] == [2] and report["band_pairs"] == [[0, 1]]
    assert report["max_lattice_diff"] == 0
    assert report["max_inplane_centre_drift"] is None and report["max_z_centre_drift"] is None
    assert report["no_hamiltonian"] == []
    assert report["tracking_unavailable"] == ["m0"]
    assert report["gamma_character"] == {
        "ok": False,
        "available": False,
        "min_weight": None,
        "max_mixing": None,
    }
    assert not report["ok"]
    json.dumps(report, allow_nan=False)


def test_mixed_family_distinguishes_openmx_tracking_from_grid_only_members():
    openmx = NativePlaneModel(
        OpenMXModel([[0, 0, 0]], [np.diag([-1.0, 1.0])], lattice=np.diag([2, 1, 1])), band_pair=(0, 1)
    )
    source = WannierModel(
        [[0, 0, 0]], [np.diag([-1.0, 1.0])], lattice=np.eye(3), centres=[[0.1, 0, 0], [0.7, 0, 0]]
    )
    wannier = NativePlaneModel(source, band_pair=(0, 1))
    family = ModelFamily(
        [
            FamilyMember("openmx", 0, {}, lambda: openmx),
            FamilyMember("wannier", 1, {}, lambda: wannier),
            FamilyMember("grid", 2, {"grid": "never-loaded.npz"}, GridLoader("never-loaded.npz")),
        ]
    )
    report = family.consistency()
    assert report["n_orb"] == [2] and report["band_pairs"] == [[0, 1]]
    assert report["max_lattice_diff"] == 1
    assert report["no_hamiltonian"] == ["grid"]
    assert report["tracking_unavailable"] == ["openmx"]
    assert not report["gamma_character"]["available"]
    assert report["gamma_character"]["tracked_members"] == ["wannier"]
    assert not report["ok"]
    json.dumps(report, allow_nan=False)
