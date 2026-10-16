"""PyATB's run-directory files: readers against real outputs, writers reproducing them byte for byte."""

from pathlib import Path

import pytest

from queer.compat.generator.pyatb_files import (
    AHC_FILE,
    CHERN_FILE,
    INPUT_ECHO,
    RUNNING_LOG,
    ahc_scan_text,
    ahc_vector_text,
    chern_text,
    parse_ahc,
    parse_chern,
    parse_input,
    parse_inputs_sha256,
    parse_integration_log,
    parse_running_log,
    read_input,
    read_input_echo,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "pyatb"


def test_input_blocks_keep_numbers_lists_and_text():
    blocks = read_input(FIXTURES / "c_direct_openmx" / "Input")
    assert set(blocks) == {"INPUT_PARAMETERS", "LATTICE", "AHC"}
    inputs, ahc = blocks["INPUT_PARAMETERS"], blocks["AHC"]
    assert inputs == {
        "nspin": 4,
        "package": "OPENMX",
        "scfout_route": "../test.scfout",
        "fermi_energy": -2.35,
        "fermi_energy_unit": "eV",
        "hall_convention": "openmx",
    }
    assert ahc["integrate_grid"] == [100, 100, 1] and ahc["adaptive_grid"] == [1, 1, 1]
    assert ahc["adaptive_grid_threshold"] == 1e30 and ahc["method"] == 0 and ahc["engine"] == "kubo"
    assert len(blocks["LATTICE"]["lattice_vector"]) == 9
    chern = read_input(FIXTURES / "h_chern_fermi" / "Input")["CHERN_NUMBER"]
    assert chern["k_vect1"] == [1, 0, 0] and chern["occ_band"] == -1 and chern["integrate_mode"] == "Grid"


def test_input_syntax_errors_are_refused():
    assert parse_input("AHC {\n  method 1 # a comment\n}\n") == {"AHC": {"method": 1}}
    with pytest.raises(ValueError, match="not closed"):
        parse_input("AHC\n{\n method 0\n")
    with pytest.raises(ValueError, match="without a name"):
        parse_input("{\n method 0\n}\n")


def test_the_echo_resolves_the_convention_and_fills_every_default():
    echo = read_input_echo(FIXTURES / "c_direct_openmx" / INPUT_ECHO)
    assert echo["INPUT_PARAMETERS"]["hall_convention"] == "openmx"
    assert echo["AHC"]["fhs_energy_points"] == 101 and echo["AHC"]["adaptive_grid_threshold"] == 1e30
    assert echo["CHERN_NUMBER"]["integrate_mode"] is None


def test_the_one_line_hall_vector_and_its_writer():
    text = (FIXTURES / "b_direct_pyatb" / AHC_FILE).read_text()
    product = parse_ahc(text)
    assert product == {"format": "vector", "sigma": [-0.0, 0.0, -38.740468], "unit": "S/cm"}
    assert ahc_vector_text([-1e-17, 2e-16, -38.7404684]) == text
    assert (
        ahc_vector_text([1e-17, -2e-16, 38.7404684]) == (FIXTURES / "c_direct_openmx" / AHC_FILE).read_text()
    )
    with pytest.raises(ValueError, match="3 AHC components"):
        parse_ahc("AHC is [1.0, 2.0] S/cm")


def test_the_fhs_table_round_trips_byte_for_byte():
    text = (FIXTURES / "a_fhs_scan" / AHC_FILE).read_text()
    product = parse_ahc(text)
    assert product["format"] == "scan" and len(product["energy_eV"]) == 201
    assert product["energy_eV"][0] == -0.7 and product["energy_eV"][1] == pytest.approx(-0.693035)
    assert ahc_scan_text(product["energy_eV"], product["sigma"]) == text
    with pytest.raises(ValueError, match="4 numbers"):
        parse_ahc("# header\n 0.1 0.2\n")


def test_the_chern_line_keeps_full_precision():
    text = (FIXTURES / "h_chern_fermi" / CHERN_FILE).read_text()
    assert parse_chern(text) == {"value": -0.9999999999999967, "integrate_mode": "Grid"}
    assert chern_text(-0.9999999999999967, "Grid") == text
    assert chern_text(1e-05, "Adaptive") == "Chern number is 1e-05 by using the Adaptive integral\n"
    with pytest.raises(ValueError):
        chern_text(1.0, "Cubature")


def test_running_log_facts():
    grid = parse_running_log((FIXTURES / "c_direct_openmx" / RUNNING_LOG).read_text())
    assert grid["fermi_energy_eV"] == -2.35 and grid["fermi_energy_decimals"] == 6
    assert grid["rough_kpoints"] == 10000 and grid["refined_kpoints"] == 0
    assert grid["processes"] == 8 and grid["ahc"] == [0.0, -0.0, 38.740468] and grid["completed"]
    fhs = parse_running_log((FIXTURES / "a_fhs_scan" / RUNNING_LOG).read_text())
    assert fhs["fermi_energy_eV"] == -2.7 and "refined_kpoints" not in fhs
    chern = parse_running_log((FIXTURES / "h_chern_fermi" / RUNNING_LOG).read_text())
    assert chern["chern"] == -0.9999999999999967
    refined = parse_running_log(
        "!!The number of rough k-grid : 10000\n!!The number of k points for adaptive refinement : 471\n"
    )
    assert refined["refined_kpoints"] == 471 and not refined["completed"]


def test_integration_log_last_generation():
    log = parse_integration_log((FIXTURES / "adaptive_log" / "integraton.log").read_text())
    assert log["complete"] and log["generations"] == 30
    assert log["error"] == [3.21214591e-05, 3.21214591e-05, 1.11174239e-03]
    assert log["value"] == [-0.0012743, 0.00068793, 0.04190095]
    early = parse_integration_log(
        "0 generation    Tue\nintegration region:1\n0.0314 [-0.0018  0.0010 -0.1399]\n"
    )
    assert early == {
        "generations": 1,
        "complete": False,
        "error": [0.0314],
        "value": [-0.0018, 0.001, -0.1399],
    }
    wrapped = parse_integration_log("1 generation    x\n[1.0e-05 2.0e-05\n 3.0e-05] [0.1 0.2\n 0.3]\n")
    assert wrapped["error"] == [1e-05, 2e-05, 3e-05] and wrapped["value"] == [0.1, 0.2, 0.3]
    scalar = parse_integration_log("0 generation    x\n1.5e-06 0.98\nintegration complete!    x\n")
    assert scalar["error"] == [1.5e-06] and scalar["value"] == [0.98] and scalar["complete"]
    with pytest.raises(ValueError, match="no error/value"):
        parse_integration_log("use self adaptive integration method\n")


def test_input_digests_in_sha256sum_text_and_binary_modes():
    digest = "93" + "0" * 62
    text = f"{digest}  scf.scfout\n{'AB' * 32} *Input\nnot a digest line\n"
    assert parse_inputs_sha256(text) == {"scf.scfout": digest, "Input": "ab" * 32}
