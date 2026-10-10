"""Historical shift-current replay preserves coverage, axes and input provenance."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from queer.io.artifacts import digest_file
from queer.models.wannier import WannierModel

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "generator_shift_replay", ROOT / "scripts/validation/generator_shift_replay.py"
)
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


def case_fixture(tmp_path):
    case = tmp_path / "case"
    (case / "Out/Shift_Current").mkdir(parents=True)
    (case / "Input").write_text("historical input provenance\n")
    (case / "Out/running.log").write_text("historical producer completed\n")
    resolved = {
        "INPUT_PARAMETERS": {"package": "OPENMX", "nspin": 4, "fermi_energy": "Auto"},
        "LATTICE": {
            "lattice_constant": 1,
            "lattice_constant_unit": "Angstrom",
            "lattice_vector": np.eye(3).tolist(),
        },
        "SHIFT_CURRENT": {
            "omega": [0, 0.02],
            "domega": 0.01,
            "grid": [2, 2, 2],
            "occ_band": 28,
            "smearing_method": 1,
            "eta": 0.05,
            "method": 1,
            "scissors": 1.15,
            "n_occ": -1,
            "m_unocc": -1,
        },
    }
    (case / "Out/input.json").write_text(json.dumps(resolved))
    table = np.column_stack((np.arange(3) * 0.01, np.arange(54).reshape(3, 18)))
    np.savetxt(
        case / "Out/Shift_Current/shift_current.dat",
        table,
        header="omega(eV) " + " ".join(REPLAY.COMPONENTS) + " (uA/V^2)",
    )
    model = SimpleNamespace(fermi_energy=-4.5, spin_mode="soc", lattice=np.eye(3))
    return case, table, model


def test_disabled_pair_sentinel_preserves_complete_scan_and_width(tmp_path):
    case, table, model = case_fixture(tmp_path)
    parameters, reference, provenance = REPLAY.read_case(case, model)
    assert provenance["disabled_band_pair_sentinel"] == {"n_occ": -1, "m_unocc": -1}
    assert "band_pairs" not in parameters
    assert parameters["eta"] == pytest.approx(0.05 / np.sqrt(2))
    assert parameters["scissors_occupied_bands"] == parameters["occupied_bands"] == 28
    assert parameters["chemical_potential"] == -4.5
    np.testing.assert_array_equal(parameters["omega"], table[:, 0])
    np.testing.assert_array_equal(reference, table)


def test_component_extraction_uses_producer_order_not_flat_tensor_order():
    tensor = np.arange(54).reshape(2, 3, 3, 3)
    independent = REPLAY.independent_components(tensor)
    np.testing.assert_array_equal(
        independent[0], [0, 1, 2, 4, 5, 8, 9, 10, 11, 13, 14, 17, 18, 19, 20, 22, 23, 26]
    )


def test_all_components_and_missing_samples_cannot_silently_pass():
    reference = np.zeros((801, 18))
    native = reference.copy()
    native[-1, -1] = 1
    metrics = REPLAY.error_metrics(native, reference, atol=1e-5, rtol=2e-6)
    assert metrics["compared_values"] == metrics["expected_values"] == 801 * 18
    assert metrics["coverage"] == 1
    assert metrics["max_absolute_error"] == 1
    assert not metrics["passed"]
    native[0, 0] = np.nan
    metrics = REPLAY.error_metrics(native, reference, atol=1e-5, rtol=2e-6)
    assert metrics["compared_values"] == 801 * 18 - 1
    assert not metrics["passed"]


def test_broker_sidecar_rejects_changed_source_bytes(tmp_path):
    source = tmp_path / "source"
    source.write_text("original\n")
    Path(str(source) + ".broker.json").write_text(
        json.dumps({"sha256": digest_file(source), "size": source.stat().st_size})
    )
    assert REPLAY.fingerprint(source)["broker_sha256_verified"]
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        REPLAY.fingerprint(source)


def test_recorded_frequency_axis_cannot_be_resampled(tmp_path):
    case, _, model = case_fixture(tmp_path)
    source = case / "Out/Shift_Current/shift_current.dat"
    table = np.loadtxt(source)
    table[1, 0] = 0.015
    np.savetxt(source, table, header="omega(eV) " + " ".join(REPLAY.COMPONENTS) + " (uA/V^2)")
    with pytest.raises(ValueError, match="frequencies differ"):
        REPLAY.read_case(case, model)


def test_block_transport_derivative_check_on_nonzero_degenerate_response():
    """Two Rice-Mele copies test transport through complete degenerate manifolds."""
    translations = [[-1, 0, 0], [0, 0, 0], [1, 0, 0]]
    forward = np.asarray([[0, 0], [0.7, 0]], complex)
    scalar_h = np.asarray([forward.conj().T, [[0.4, 1.1], [1.1, -0.4]], forward])
    hopping = np.asarray([np.kron(np.eye(2), value) for value in scalar_h])
    position = np.zeros((3, 3, 4, 4), complex)
    position[0, 1] = np.kron(np.eye(2), np.diag([0, 0.5]))
    model = WannierModel(translations, hopping, lattice=np.eye(3), position=np.moveaxis(position, 0, -1))
    k = [0.13, 0, 0]
    levels = np.linalg.eigvalsh(model.hamiltonian(k))
    check = REPLAY.finite_difference_point_check(
        model,
        k,
        steps=(1e-4, 5e-5),
        omega=levels[-1] - levels[0],
        occupied_bands=2,
        scissors=0,
        eta=0.07,
    )
    assert np.max(np.abs(check["analytic_tensor"])) > 1
    first, second = check["checks"]
    assert second["tensor_max_difference_over_analytic_peak"] < 1e-7
    assert second["tensor_max_absolute_difference"] < first["tensor_max_absolute_difference"]

    # A constant unitary orbital transformation preserves the independently
    # transported derivative despite changing the degenerate eigenvector gauge.
    rng = np.random.default_rng(39)
    unitary, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    rotated = WannierModel(
        translations,
        unitary.conj().T @ hopping @ unitary,
        lattice=np.eye(3),
        position=np.moveaxis(unitary.conj().T @ position @ unitary, 0, -1),
    )
    actual = REPLAY.finite_difference_point_check(
        rotated,
        k,
        steps=(5e-5,),
        omega=levels[-1] - levels[0],
        occupied_bands=2,
        scissors=0,
        eta=0.07,
    )
    np.testing.assert_allclose(actual["analytic_tensor"], check["analytic_tensor"], rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(
        actual["checks"][0]["finite_difference_tensor"],
        second["finite_difference_tensor"],
        rtol=1e-7,
        atol=1e-6,
    )
