import shutil
from pathlib import Path

import numpy as np
import pytest

from queer.epw import epw
from queer.epw_utils import get_ef, get_nk, get_nph, get_nq
from queer.io.epw import read_epw_directory, read_epw_metadata
from queer.models.epw import EPWModel

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "epw"


def test_reader_numeric_order_and_metadata_overrides():
    model = EPWModel.from_directory(FIXTURE, nk=2, nq=2, nph=2, ef=0.0)
    assert model.ef == 0.0
    np.testing.assert_array_equal(model.ph, [[10, 11], [20, 21]])
    np.testing.assert_array_equal(model.g_re[0], [[1, 2], [3, 4]])
    np.testing.assert_array_equal(model.g_re[1], [[10, 20], [30, 40]])
    assert model.g_complex.shape == (2, 2, 2)
    assert [
        get_nk(FIXTURE / "gkk.out"),
        get_nq(FIXTURE / "gkk.out"),
        get_nph(FIXTURE / "gkk.out"),
        get_ef(FIXTURE / "gkk.out"),
    ] == [2, 2, 2, 0.1]


def test_header_scanner_no_line_limit_and_fortran_exponent(tmp_path):
    header = tmp_path / "gkk.out"
    header.write_text(
        "unrelated\n" * 1200
        + "Size of k mesh : 2\nSize of q mesh : 3\nFermi energy : 1.2D-1 eV\n1 1 4 0.1 0.2 0.3\n"
    )
    assert read_epw_metadata(header) == {"nk": 2, "nq": 3, "ef": 0.12, "nph": 4}
    with pytest.raises(ValueError, match="missing EPW"):
        get_nph(FIXTURE / "README.md")


def test_legacy_epw_path_and_false_sentinel_zero_override():
    with pytest.warns(DeprecationWarning):
        model = epw(path=FIXTURE, nq=2, nph=2, ef=0.0)
    assert model.ef == 0.0
    model.load_data()
    np.testing.assert_array_equal(model.ph, [[10, 11], [20, 21]])
    with pytest.warns(DeprecationWarning):
        alternate = epw(work_dir=FIXTURE.parent, outfolder="epw")
    alternate.load_data()
    np.testing.assert_allclose(
        alternate.calculate_self_energy(0), alternate.self_energy(0), atol=1e-12, rtol=1e-12
    )


def test_self_energy_independent_reference_and_empty_indices():
    model = EPWModel.from_directory(FIXTURE)
    temperature, broadening = 0.02, 1e-4
    expected = np.zeros((2, 2), dtype=complex)
    for q in range(model.nq):
        for mode in range(model.nph):
            for k in range(model.nk):
                initial, final = model.e_k[q, k], model.e_kq[q, k]
                occupation = 1 / (np.exp((initial - model.ef) / temperature) + 1) - 1 / (
                    np.exp((final - model.ef) / temperature) + 1
                )
                coupling = abs(model.g_complex[mode, q, k]) ** 2 * model.ph[mode, q] * 1e-6
                denominator = initial - final - 1j * broadening - model.ph[mode, q] * 1e-3
                expected[q, mode] += coupling * occupation / denominator / model.nk
    np.testing.assert_allclose(
        model.self_energy([0, 1], temperature, broadening), expected, atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(
        model.self_energy(1, temperature, broadening), expected[1], atol=1e-12, rtol=1e-12
    )
    assert model.self_energy([]).shape == (0, 2)
    with pytest.raises(ValueError, match="q"):
        model.self_energy([2])
    with pytest.raises(ValueError, match="broadening"):
        model.self_energy(0, broadening=0)


def test_swap_keeps_derived_and_reduced_arrays_aligned():
    model = EPWModel.from_directory(FIXTURE)
    model.reduce_g()
    original = {
        name: getattr(model, name).copy()
        for name in ("ph", "g_re", "g_im", "g_abs", "g_complex", "g_sq_mean", "g0")
    }
    model.swap_branch(0, 1)
    for name, data in original.items():
        np.testing.assert_allclose(getattr(model, name), data[::-1], atol=1e-12, rtol=1e-12)
    model.swap_branch(0, 1, qpoint=1)
    for name in original:
        data = getattr(model, name)
        np.testing.assert_allclose(data[:, :1], original[name][:, :1], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(data[:, 1:], original[name][::-1, 1:], atol=1e-12, rtol=1e-12)


def test_negative_phonon_reduction_rejected_and_zero_mode_valid():
    model = EPWModel([[-1]], [[[1]]], [[0]], [[0.1]])
    with pytest.raises(ValueError, match="nonnegative"):
        model.reduce_g()
    zero = EPWModel([[0]], [[[1]]], [[0]], [[0.1]])
    np.testing.assert_array_equal(zero.self_energy(0), [0j])


@pytest.mark.parametrize(
    "fault,match",
    [
        ("branch", "indices differ"),
        ("size", "size differs"),
        ("frequency", "varies with k"),
        ("header", "nk"),
    ],
)
def test_invalid_directory_fails_with_context(tmp_path, fault, match):
    root = tmp_path / "epw"
    shutil.copytree(FIXTURE, root)
    if fault == "branch":
        (root / "gkk_im" / "gkk_im_2.dat").rename(root / "gkk_im" / "gkk_im_3.dat")
    elif fault == "size":
        (root / "enk" / "enk_1.dat").write_text("1 2 3\n")
    elif fault == "frequency":
        (root / "omega" / "omega_2.dat").write_text("10 12\n11 11\n")
    else:
        (root / "gkk.out").unlink()
    with pytest.raises(ValueError, match=match):
        read_epw_directory(root)


def test_missing_header_supported_with_explicit_metadata(tmp_path):
    root = tmp_path / "epw"
    shutil.copytree(FIXTURE, root)
    (root / "gkk.out").unlink()
    model = EPWModel.from_directory(root, nk=2, nq=2, nph=2, ef=0.1)
    assert model.nph == 2
