"""PyATB's sign, unit and grid conventions in the legacy translation.

The sign table, the threshold unit and the energy scales are the ones measured
against PyATB on an OpenMX scfout: QUEER's physical Hall conductivity equals
PyATB's ``openmx`` (and ``auto``, ``fhs``, and the FHS engine) and is the negative
of PyATB's ``pyatb`` convention; ``adaptive_grid_threshold`` is in Å²; PyATB's
Hartree is 27.211324570273 eV.
"""

import pytest

from queer.compat.generator import translate_pyatb
from queer.compat.generator.pyatb_conventions import (
    FERMI_TOLERANCE_HARTREE,
    HARTREE_EV,
    PYATB_HARTREE_EV,
    PYATB_RYDBERG_EV,
    fermi_difference_hartree,
    hall_export_sign,
    pyatb_energy,
    pyatb_input_fermi_energy,
    queer_energy,
    refinement_off,
    resolve_hall_convention,
)

SCALE = HARTREE_EV / PYATB_HARTREE_EV


@pytest.mark.parametrize(
    ("convention", "engine", "sign"),
    [
        ("pyatb", "kubo", -1),
        ("openmx", "kubo", 1),
        ("fhs", "kubo", 1),
        ("auto", "kubo", 1),
        (None, "kubo", 1),
        ("pyatb", "pyatb", -1),
        ("pyatb", "fhs", 1),
        ("openmx", "fhs", 1),
    ],
)
def test_the_hall_sign_table(convention, engine, sign):
    assert hall_export_sign(convention, engine) == sign
    recipe = translate_pyatb(
        "PYATB_AHC",
        {"engine": engine, "method": 0, "integrate_grid": [2, 2, 1], "adaptive_grid_threshold": 1e30},
        input_parameters={"hall_convention": convention} if convention else {},
        fermi_energy=-1.0,
    )
    assert recipe["export_metadata"]["hall_export_sign"] == sign


def test_auto_resolves_per_package_and_unknown_conventions_are_refused():
    assert resolve_hall_convention("auto", "OPENMX") == "openmx"
    assert resolve_hall_convention("AUTO", "ABACUS") == "pyatb"
    assert resolve_hall_convention(None) == "openmx"
    assert hall_export_sign("auto", "kubo", package="WANNIER90") == -1
    with pytest.raises(ValueError, match="hall_convention"):
        resolve_hall_convention("physical")


def test_the_grid_threshold_is_in_square_angstrom_and_1e30_turns_refinement_off():
    refined = translate_pyatb(
        "AHC",
        {"integrate_grid": [100, 100, 1], "adaptive_grid": [2, 2, 1], "adaptive_grid_threshold": 10},
        fermi_energy=-2.0,
    )
    assert refined["parameters"]["integration"] == {
        "method": "threshold",
        "subdivision": [2, 2, 1],
        "threshold": 10.0,
        "threshold_quantity": "curvature",
    }
    assert refined["export_metadata"]["threshold_source_unit"] == "angstrom^2"
    assert refined["export_metadata"]["refinement"] == "threshold"
    plain = translate_pyatb(
        "AHC", {"integrate_grid": [100, 100, 1], "adaptive_grid": [1, 1, 1], "adaptive_grid_threshold": 1e30}
    )
    assert plain["parameters"]["integration"] is None and plain["export_metadata"]["refinement"] == "off"
    assert refinement_off(float("inf")) and refinement_off(1e30) and not refinement_off(9.9e29)
    with pytest.raises(ValueError):
        translate_pyatb("AHC", {"adaptive_grid_threshold": -1.0})


def test_explicit_fermi_energies_move_to_queer_hartree_scale():
    explicit = translate_pyatb("AHC", {"integrate_grid": [2, 2, 1]}, input_parameters={"fermi_energy": -2.35})
    assert explicit["parameters"]["ef"] == pytest.approx(-2.35 * SCALE, rel=1e-15)
    assert explicit["parameters"]["chemical_potentials"] == [explicit["parameters"]["ef"]]
    metadata = explicit["export_metadata"]
    assert metadata["fermi_source"] == "input_eV" and metadata["fermi_energy_pyatb_eV"] == -2.35
    rydberg = translate_pyatb(
        "AHC",
        {"integrate_grid": [2, 2, 1]},
        input_parameters={"fermi_energy": -0.5, "fermi_energy_unit": "Ry"},
    )
    assert rydberg["parameters"]["ef"] == pytest.approx(-0.5 * PYATB_RYDBERG_EV * SCALE, rel=1e-15)
    auto = translate_pyatb("AHC", {"integrate_grid": [2, 2, 1]}, fermi_energy=-4.5837)
    assert auto["parameters"]["ef"] == -4.5837 and auto["export_metadata"]["fermi_source"] == "auto"
    assert auto["export_metadata"]["fermi_energy_pyatb_eV"] == pytest.approx(-4.5837 / SCALE, rel=1e-15)
    assert queer_energy(pyatb_energy(-4.5837)) == pytest.approx(-4.5837, rel=1e-15)
    assert (
        pyatb_input_fermi_energy("Auto") is None
        and pyatb_input_fermi_energy("1.5", "Ry") == 1.5 * 13.605698066
    )


def test_the_fermi_rule_compares_the_two_hartree_scales():
    chemp = -0.16844932921230793
    queer, pyatb = chemp * HARTREE_EV, round(chemp * PYATB_HARTREE_EV, 6)
    assert fermi_difference_hartree(queer, pyatb) <= FERMI_TOLERANCE_HARTREE
    assert fermi_difference_hartree(queer + 1e-4, pyatb) > FERMI_TOLERANCE_HARTREE
    assert fermi_difference_hartree(-2.35 * SCALE, -2.35) < 1e-15


def test_fhs_scans_sample_pyatb_energies_on_queer_scale():
    recipe = translate_pyatb(
        "AHC",
        {
            "engine": "fhs",
            "fhs_energy_window": [-0.7, 0.7],
            "fhs_energy_points": 201,
            "fhs_grid": [10, 10, 1],
        },
        input_parameters={"fermi_energy": -2.7},
    )
    offsets = recipe["export_metadata"]["pyatb_energy_offsets_eV"]
    assert len(offsets) == 201 and offsets[0] == -0.7 and offsets[1] == pytest.approx(-0.7 + 1.4 / 201)
    mus = recipe["parameters"]["chemical_potentials"]
    assert mus[1] == pytest.approx((-2.7 + offsets[1]) * SCALE, rel=1e-14)
    assert recipe["export_metadata"]["hall_export_sign"] == 1


def test_chern_translation_keeps_the_plane_and_no_hall_sign():
    recipe = translate_pyatb(
        "PYATB_CHERN_NUMBER",
        {
            "method": 1,
            "occ_band": -1,
            "integrate_mode": "Grid",
            "integrate_grid": [8, 8, 5],
            "adaptive_grid": [2, 2, 3],
            "adaptive_grid_threshold": 20,
            "k_start": [0, 0, 0.5],
            "k_vect1": [1, 0, 0],
            "k_vect2": [0, 1, 0],
        },
        input_parameters={"hall_convention": "pyatb"},
        fermi_energy=0.1,
    )
    native = recipe["parameters"]
    assert native["grid"] == [8, 8, 1] and native["integration"]["subdivision"] == [2, 2, 1]
    assert native["engine"] == "kubo" and native["occupied_bands"] is None and native["fermi_energy"] == 0.1
    assert native["k_start"] == [0, 0, 0.5] and recipe["export_metadata"]["hall_export_sign"] == 1


def test_pyatbs_shift_current_spin_factor_and_constants():
    from queer.compat.generator.pyatb_conventions import (
        PYATB_SHIFT_E_CHARGE,
        PYATB_SHIFT_EV_SECONDS,
        PYATB_SHIFT_HBAR,
        pyatb_shift_constant_ratio,
        pyatb_shift_spin_factor,
    )

    # shift_current_solver.cpp: fac * 2 for nspin == 1 (one channel read), fac for nspin 4.
    assert pyatb_shift_spin_factor(1) == 2 and pyatb_shift_spin_factor("4") == 1
    with pytest.raises(ValueError):
        pyatb_shift_spin_factor(2)
    assert (PYATB_SHIFT_EV_SECONDS, PYATB_SHIFT_E_CHARGE, PYATB_SHIFT_HBAR) == (
        6.582119e-16,
        1.602176565e-19,
        1.054571726e-34,
    )
    assert abs(pyatb_shift_constant_ratio() - 1) < 1e-7
