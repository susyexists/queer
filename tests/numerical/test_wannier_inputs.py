"""Wannier90 .win, .eig and band-file inputs read with the post-processing rules."""

import numpy as np
import pytest

from queer.io.wannier import (
    BOHR_ANGSTROM,
    band_path_breaks,
    infer_mp_grid,
    read_band_dat,
    read_band_kpt,
    read_eig,
    read_labelinfo,
    read_win,
)

WIN = """! comment-only line
 Num_Wann = 2          # trailing comment
num_bands : 4
spinors = .true.
use_ws_distance F
translate_home_cell = false
write_tb = T
write_hr = yes
dis_win_min = -1.25d1
dis_win_max = 5.5
dis_froz_min = -2.0
dis_froz_max = 1.25D0
ws_search_size = 3
ws_distance_tol = 1e-4
fermi_energy = 4.5 eV
mp_grid = 4, 4 1

Begin Unit_Cell_Cart
Bohr
  2.0  0.0  0.0
  1.0  3.0  0.0   ! oblique
  0.0  0.0  20.0
End Unit_Cell_Cart

begin kpoint_path
G 0.0 0.0 0.0 M 0.5 0.0 0.0
M 0.5,0.0,0.0 K 0.3333333333 0.3333333333 0.0
too short 1 2
X 0.5 0.0 0.0 Y 0.0 bad 0.0
end kpoint_path
"""


def write(tmp_path, text, name="wannier90.win"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_win_keywords_bohr_cell_windows_and_path(tmp_path):
    win = read_win(write(tmp_path, WIN))
    expected = np.array([[2.0, 0, 0], [1.0, 3.0, 0], [0, 0, 20.0]]) * BOHR_ANGSTROM
    np.testing.assert_allclose(win.lattice, expected, rtol=1e-15, atol=0)
    assert win.cell_unit == "bohr"
    assert win.mp_grid == (4, 4, 1) and not win.mp_grid_inferred
    assert (win.num_wann, win.num_bands) == (2, 4)
    assert win.spinors is True and win.use_ws_distance is False
    assert win.translate_home_cell is False and win.write_tb is True and win.write_hr is True
    assert win.write_xyz is None and win.bands_plot is None
    assert (win.dis_win_min, win.dis_win_max) == (-12.5, 5.5)
    assert (win.dis_froz_min, win.dis_froz_max) == (-2.0, 1.25)
    assert (win.ws_search_size, win.ws_distance_tol, win.fermi_energy) == (3, 1e-4, 4.5)
    assert win.kpoint_path == (
        ("G", (0.0, 0.0, 0.0), "M", (0.5, 0.0, 0.0)),
        ("M", (0.5, 0.0, 0.0), "K", (0.3333333333, 0.3333333333, 0.0)),
    )
    assert win.keywords["num_wann"] == "2" and win.keywords["fermi_energy"] == "4.5 eV"
    assert set(win.blocks) == {"unit_cell_cart", "kpoint_path"}
    assert win.blocks["unit_cell_cart"][2] == "1.0  3.0  0.0"
    assert win.n_kpoints == 0 and win.kpoints.shape == (0, 3)
    assert not win.lattice.flags.writeable


@pytest.mark.parametrize(
    "unit,scale,expected_unit",
    [("ang", 1.0, "ang"), ("Angstrom", 1.0, "angstrom"), (None, 1.0, None), ("BOHR", BOHR_ANGSTROM, "bohr")],
)
def test_win_cell_units(tmp_path, unit, scale, expected_unit):
    rows = "1.5 0 0\n0 2.5 0\n0 0 3.5d0"
    header = "" if unit is None else f"{unit}\n"
    win = read_win(write(tmp_path, f"begin unit_cell_cart\n{header}{rows}\nend unit_cell_cart\n"))
    np.testing.assert_allclose(win.lattice, np.diag([1.5, 2.5, 3.5]) * scale, rtol=1e-15, atol=0)
    assert win.cell_unit == expected_unit


def test_win_incomplete_or_invalid_cells(tmp_path):
    incomplete = read_win(write(tmp_path, "begin unit_cell_cart\nang\n1 0 0\n0 1 0\nend unit_cell_cart\n"))
    assert incomplete.lattice is None and incomplete.cell_unit == "ang"
    short_row = read_win(write(tmp_path, "begin unit_cell_cart\n1 0 0\n0 1\n0 0 1\nend unit_cell_cart\n"))
    assert short_row.lattice is None
    unterminated = read_win(write(tmp_path, "begin unit_cell_cart\n1 0 0\n0 1 0\n0 0 1\n"))
    assert unterminated.lattice is None and not unterminated.blocks
    with pytest.raises(ValueError, match="unit_cell_cart"):
        read_win(write(tmp_path, "begin unit_cell_cart\n1 0 0\n0 x 0\n0 0 1\nend unit_cell_cart\n"))


def test_missing_mp_grid_is_inferred_from_the_kpoints_block(tmp_path):
    rows = []
    for i in range(3):
        for j in range(2):
            # Equivalent negative coordinates and a weight column are accepted.
            rows.append(f"{i / 3:.10f} {(-0.5 if j else 0.0):.10f} 0.0 0.1666666667")
    block = "begin kpoints\n" + "\n".join(rows) + "\n1.0 nan-ish 0\nend kpoints\n"
    win = read_win(write(tmp_path, block))
    assert win.mp_grid == (3, 2, 1) and win.mp_grid_inferred
    assert win.n_kpoints == 6
    np.testing.assert_allclose(win.kpoints[1], [0.0, -0.5, 0.0])
    malformed = read_win(write(tmp_path, "mp_grid = 4.0 4 1\n" + block))
    assert malformed.mp_grid == (3, 2, 1) and malformed.mp_grid_inferred
    explicit = read_win(write(tmp_path, "mp_grid 5 5 5\n" + block))
    assert explicit.mp_grid == (5, 5, 5) and not explicit.mp_grid_inferred
    assert read_win(write(tmp_path, "mp_grid = 4 4\n")).mp_grid is None
    assert infer_mp_grid([[0.0, 0.0, 0.0], [0.5, 0.25, 1.0], [-0.5, 0.75, 0.0]]) == (2, 3, 1)
    assert infer_mp_grid(np.zeros((0, 3))) is None


def test_win_block_and_keyword_rules(tmp_path):
    text = """num_wann = 2
num_wann = 3
begin_projections = ignored as a keyword
begin kpoint_path
G 0 0 0 X 0.5 0 0
end projections
end
spinors = maybe
endless = 1
"""
    win = read_win(write(tmp_path, text))
    assert win.num_wann == 3
    assert win.keywords["begin_projections"] == "ignored as a keyword"
    assert win.keywords["endless"] == "1"
    assert win.kpoint_path == (("G", (0.0, 0.0, 0.0), "X", (0.5, 0.0, 0.0)),)
    assert win.spinors is None


@pytest.mark.parametrize(
    "value,expected",
    [
        (".TRUE.", True),
        ("t", True),
        (".t.", True),
        ("1", True),
        ("Yes", True),
        (".False.", False),
        ("F", False),
        ("0", False),
        ("no", False),
        ("maybe", None),
        ("", None),
    ],
)
def test_win_logical_spellings(tmp_path, value, expected):
    assert read_win(write(tmp_path, f"use_ws_distance = {value}\n")).use_ws_distance is expected


def test_eig_roundtrip_missing_and_repeated_records(tmp_path):
    rng = np.random.default_rng(7)
    energies = rng.normal(scale=5.0, size=(3, 4))
    lines = [f"{b + 1:5d}{k + 1:5d}{energies[k, b]:18.12f}" for k in range(3) for b in range(4)]
    path = write(tmp_path, "\n".join(lines) + "\n", "wannier90.eig")
    np.testing.assert_allclose(read_eig(path), energies, rtol=0, atol=5e-13)
    gapped = write(tmp_path, "\n".join(lines[:-1]) + "\n", "gapped.eig")
    values = read_eig(gapped)
    assert values.shape == (3, 4) and np.isnan(values[2, 3])
    np.testing.assert_allclose(values[:2], energies[:2], atol=5e-13)
    with pytest.raises(ValueError, match="repeated"):
        read_eig(write(tmp_path, "\n".join(lines + lines[:1]) + "\n", "repeated.eig"))
    with pytest.raises(ValueError, match="integer"):
        read_eig(write(tmp_path, "1.0 1 0.5\n", "float_index.eig"))
    with pytest.raises(ValueError, match="one-based"):
        read_eig(write(tmp_path, "0 1 0.5\n", "zero_index.eig"))


def test_band_files_ticks_and_path_breaks(tmp_path):
    first = [[0.125 * i, 0.0, 0.0] for i in range(5)]
    second = [[0.0, 0.5 - 0.125 * i, 0.0] for i in range(5)]
    points = np.array(first + second)
    kpt = "10\n" + "\n".join(f"{a:.8f} {b:.8f} {c:.8f} 1.0" for a, b, c in points) + "\n"
    labels = "G 1 0.0 0 0 0\nX 5 0.5 0.5 0 0\nY 6 0.5 0 0.5 0\nG 10 1.0 0 0 0\nbad row 1\n"
    energies = np.stack([np.linspace(-1, 0, 10), np.linspace(1, 2, 10)], axis=1)
    distance = np.arange(10) / 9
    blocks = ["\n".join(f"{x:.6f} {e:.8E}" for x, e in zip(distance, energies[:, b])) for b in range(2)]
    dat = write(tmp_path, "\n\n".join(blocks) + "\n", "wannier90_band.dat")
    read_points = read_band_kpt(write(tmp_path, kpt, "wannier90_band.kpt"))
    np.testing.assert_allclose(read_points, points, atol=1e-12)
    ticks, names = read_labelinfo(write(tmp_path, labels, "wannier90_band.labelinfo.dat"))
    assert ticks == [0, 4, 5, 9] and names == ["G", "X", "Y", "G"]
    np.testing.assert_allclose(read_band_dat(dat, 2), energies, atol=1e-12)
    with pytest.raises(ValueError, match="multiple of 3 bands"):
        read_band_dat(dat, 3)
    assert band_path_breaks(read_points, ticks) == [5]
    assert band_path_breaks(read_points, [1, 5, 5, 12]) == [5]
    # Without a count line every row is a k-point.
    headless = write(tmp_path, "0 0 0\n0.5 0 0 1\n", "headless.kpt")
    np.testing.assert_allclose(read_band_kpt(headless), [[0, 0, 0], [0.5, 0, 0]])
