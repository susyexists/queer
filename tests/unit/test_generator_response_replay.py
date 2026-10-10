"""Original producer artifact layouts and explicit native replay boundaries."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[2] / "scripts/validation/generator_response_replay.py"
SPEC = importlib.util.spec_from_file_location("generator_response_replay", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_original_ahc_prose_maps_to_cartesian_pseudovector(tmp_path):
    path = tmp_path / "ahc.dat"
    path.write_text("AHC is [-0.171844, 0.092770, 5.650510] S/cm\n")
    reference = MODULE.read_ahc(path)
    tensor = np.zeros((1, 3, 3))
    tensor[0, 1, 2], tensor[0, 2, 0], tensor[0, 0, 1] = reference
    np.testing.assert_array_equal(MODULE.tensor_components(tensor, MODULE.AHC_COMPONENTS)[0], reference)
    path.write_text("AHC is [-0.171844, 0.092770, 5.650510] e2/h\n")
    with pytest.raises(ValueError, match="S/cm"):
        MODULE.read_ahc(path)


def test_shift_artifact_columns_cover_exact_symmetric_input_tensor(tmp_path):
    path = tmp_path / "shift_current.dat"
    tensor = np.empty((2, 3, 3, 3))
    for a in range(3):
        for b in range(3):
            for c in range(3):
                tensor[:, a, b, c] = 100 * a + 10 * min(b, c) + max(b, c)
    values = MODULE.tensor_components(tensor, MODULE.SHIFT_COMPONENTS)
    np.savetxt(
        path,
        np.column_stack(([0.1, 0.2], values)),
        header="omega(eV) " + " ".join(MODULE.SHIFT_COMPONENTS) + " (uA/V^2)",
    )
    omega, restored = MODULE.read_shift_current(path)
    np.testing.assert_array_equal(omega, [0.1, 0.2])
    np.testing.assert_array_equal(restored, values)
    np.testing.assert_array_equal(restored[:, 1], tensor[:, 0, 0, 1])
    path.write_text(path.read_text().replace("xxy", "xyx"))
    with pytest.raises(ValueError, match="tensor order"):
        MODULE.read_shift_current(path)


def test_declared_legacy_alias_and_unrestricted_pair_sentinel():
    global_input = {"nspin": 4, "package": "OPENMX", "fermi_energy": "Auto"}
    recipe, changes = MODULE.translate_resolved_input(
        {
            "INPUT_PARAMETERS": global_input,
            "AHC": {
                "engine": "pyatb",
                "method": 0,
                "integrate_mode": "Adaptive",
                "initial_grid": [1, 1, 1],
            },
        },
        "AHC",
        fermi_energy=-5.374507,
        spin_mode="soc",
    )
    assert recipe["parameters"]["engine"] == "direct"
    assert recipe["parameters"]["chemical_potentials"] == [-5.374507]
    assert recipe["parameters"]["integration"]["method"] == "cubature"
    assert changes
    recipe, changes = MODULE.translate_resolved_input(
        {
            "INPUT_PARAMETERS": global_input,
            "SHIFT_CURRENT": {
                "omega": [0, 1],
                "domega": 0.1,
                "grid": [4, 4, 4],
                "occ_band": 4,
                "n_occ": -1,
                "m_unocc": -1,
            },
        },
        "SHIFT_CURRENT",
        fermi_energy=-5,
        spin_mode="soc",
    )
    assert "band_pairs" not in recipe["parameters"]
    assert recipe["parameters"]["occupied_bands"] == 4
    assert changes


def test_comparison_rejects_missing_axes_and_nonfinite_values():
    with pytest.raises(ValueError, match="axes"):
        MODULE.comparison([1, 2], [1], atol=1e-6, rtol=0)
    with pytest.raises(ValueError, match="finite"):
        MODULE.comparison([np.nan], [1], atol=1e-6, rtol=0)
    with pytest.raises(ValueError, match="tolerances"):
        MODULE.comparison([1], [1], atol=-1, rtol=0)
    assert not MODULE.comparison([1.01], [1], atol=0.001, rtol=0)["passed"]


def test_checksum_mismatch_stops_before_model_loading(tmp_path, monkeypatch):
    source = tmp_path / "scf.scfout"
    source.write_bytes(b"different original input")
    args = type("Args", (), {"scfout": source, "expected_scfout_sha256": "0" * 64, "atol": 1e-6, "rtol": 0})()
    monkeypatch.setattr(MODULE.OpenMXModel, "from_scfout", lambda *a, **k: pytest.fail("model loaded"))
    with pytest.raises(ValueError, match="SHA256"):
        MODULE.replay(args)
