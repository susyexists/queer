"""The independent transcription of PyATB's method-1 shift-current integrand, against QUEER's estimator.

A comparison recomputes the k-points PyATB's method 1 regularizes with
:mod:`queer.compat.generator.pyatb_shift_integrand`: its own Bloch sums of the
model's blocks, its own generalized eigenproblem and PyATB's sum-over-states route.
With an exact threshold it is the physical integrand, which QUEER's covariant
estimator must meet transition by transition on the analytic models; with PyATB's
1e-4 eV threshold and 0.04 eV regularization it is PyATB's cut, which differs from
QUEER's degeneracy tolerance at 1e-4 eV only by PyATB's leftover dE/(dE^2 + eta^2).
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest
from openmx_files import shift_collinear_scfout, shift_spinor_scfout

from queer import OpenMXModel
from queer.compat.generator.openmx_shift import component_table
from queer.compat.generator.openmx_steps import single_channel_model
from queer.compat.generator.pyatb_conventions import HARTREE_EV, PYATB_HARTREE_EV
from queer.compat.generator.pyatb_shift_integrand import bloch_operators, queer_weights, shift_integrand
from queer.models.evaluation import derivatives
from queer.observables.nonlinear import shift_current_point, shift_current_settings

SCALE = PYATB_HARTREE_EV / HARTREE_EV
CALIBRATION_SCFOUT = Path(
    os.environ.get("QUEER_PYATB_CALIBRATION_SCFOUT")
    or Path.home() / ".cache" / "queer-parity-data" / "pyatb-calib-60fc6b2a" / "test.scfout"
)


def queer_transitions(model, k, occupied, tolerance=None, pairs=None):
    settings = shift_current_settings(
        model,
        np.array([1.0]),
        model.fermi_energy,
        eta=0.05,
        smearing="gaussian",
        occupied_bands=occupied,
        band_pairs=pairs,
        degeneracy_tol=tolerance,
    )
    point = shift_current_point(model, k, settings)
    source, target = np.nonzero(point["factor"])
    weights = component_table(
        np.moveaxis(point["factor"][source, target] * point["integrand"][..., source, target], -1, 0)
    )
    return point["gap"][source, target], weights, list(zip(source.tolist(), target.tolist())), point


def spectrum(energies, weights, eta=0.01):
    omega = np.linspace(0.0, 8.0, 1601)
    kernel = np.exp(-(((omega[:, None] - np.asarray(energies)[None, :]) / eta) ** 2)) / (
        eta * math.sqrt(math.pi)
    )
    return kernel @ np.asarray(weights)


def test_its_bloch_sums_are_the_models(tmp_path):
    shift_spinor_scfout(tmp_path / "soc.scfout")
    model = OpenMXModel.from_scfout(tmp_path / "soc.scfout")
    k = np.array([0.13, 0.41, 0.77])
    operators = bloch_operators(model, k)
    first, second = derivatives(model, k), derivatives(model, k, order=2)
    assert np.allclose(operators.hamiltonian, model.hamiltonian(k), rtol=0, atol=1e-13)
    assert np.allclose(operators.overlap, model.overlap(k), rtol=0, atol=1e-13)
    assert np.allclose(operators.d_hamiltonian, first["H"], rtol=0, atol=1e-12)
    assert np.allclose(operators.d2_hamiltonian, second["H"], rtol=0, atol=1e-11)
    assert not np.any(operators.position) and not np.any(operators.d_position)


@pytest.mark.parametrize(("label", "exchange"), [("generic", None), ("split", 2e-5)])
def test_the_physical_integrand_meets_queers_estimator(tmp_path, label, exchange):
    """With an exact threshold the transcription is QUEER's physical integrand: at generic k-points and where an
    exchange of 2e-5 eV splits Kramers pairs by 4e-5 eV (PyATB's flagged k-points), to rounding."""
    model_args = {} if exchange is None else {"exchange": exchange}
    shift_spinor_scfout(tmp_path / "soc.scfout", **model_args)
    model = OpenMXModel.from_scfout(tmp_path / "soc.scfout")
    rng = np.random.default_rng(3)
    points = (
        rng.random((3, 3)) if exchange is None else [np.array([0.5, 0.5, 0.0]), np.array([0.0, 0.0, 0.5])]
    )
    for k in points:
        energies, weights, pairs, point = queer_transitions(model, k, 2)
        tolerance = float(point["raw"]["degeneracy_tol"])
        port = shift_integrand(
            model, k, pairs, energy_scale=1.0, threshold_eV=tolerance, regularization_eV=None, occupied=2
        )
        assert port["pairs"] == pairs
        assert np.max(np.abs(port["energies_eV"] - energies)) <= 1e-12
        absolute = np.max(np.abs(weights).sum(axis=0))
        assert np.max(np.abs(queer_weights(port["integrand"]) - weights)) <= 1e-9 * absolute, k
        if exchange is not None:
            assert port["min_spacing_eV"] == pytest.approx(4e-5, rel=1e-6)
        # QUEER's own selection when no pair is named: every occupied-empty pair above the threshold.
        selected = shift_integrand(
            model, k, None, energy_scale=1.0, threshold_eV=tolerance, regularization_eV=None, occupied=2
        )
        assert sorted(selected["pairs"]) == sorted(pairs)


def test_pyatbs_cut_differs_from_queers_tolerance_only_by_its_leftover(tmp_path):
    shift_spinor_scfout(tmp_path / "split.scfout", exchange=2e-5)
    model = OpenMXModel.from_scfout(tmp_path / "split.scfout")
    for k in (np.array([0.5, 0.5, 0.0]), np.array([0.0, 0.0, 0.5]), np.array([0.5, 0.5, 0.5])):
        energies, weights, pairs, _ = queer_transitions(model, k, 2)
        cut_energies, cut_weights, _, _ = queer_transitions(model, k, 2, tolerance=1e-4 / SCALE)
        common = {"energy_scale": SCALE, "threshold_eV": 1e-4, "occupied": 2}
        regularized = shift_integrand(model, k, pairs, regularization_eV=0.04, **common)
        zeroed = shift_integrand(model, k, pairs, regularization_eV=None, **common)
        physical = spectrum(energies, weights)
        queer_cut = spectrum(cut_energies, cut_weights)
        pyatb_cut = spectrum(regularized["energies_eV"], queer_weights(regularized["integrand"]))
        dropped = spectrum(zeroed["energies_eV"], queer_weights(zeroed["integrand"]))
        scale = np.max(np.abs(physical))
        # The cut removes a sizeable part of the k-point's spectrum; PyATB's regularized leftover is tiny, and
        # QUEER's tolerance at 1e-4 eV is PyATB's cut with that leftover set to zero.
        assert np.max(np.abs(physical - queer_cut)) > 1e-3 * scale
        assert 0 < np.max(np.abs(pyatb_cut - dropped)) <= 1e-5 * scale
        assert np.max(np.abs(queer_cut - dropped)) <= 1e-6 * scale


def test_pyatbs_window_selects_its_pairs_on_its_scale(tmp_path):
    shift_collinear_scfout(tmp_path / "col.scfout")
    model = single_channel_model(OpenMXModel.from_scfout(tmp_path / "col.scfout"))
    k = np.array([0.2, 0.3, 0.0])
    everything = shift_integrand(
        model, k, None, energy_scale=SCALE, threshold_eV=1e-4, regularization_eV=0.04, occupied=1
    )
    gap = float(everything["energies_eV"][0]) * SCALE
    inside = shift_integrand(
        model,
        k,
        None,
        energy_scale=SCALE,
        threshold_eV=1e-4,
        regularization_eV=0.04,
        occupied=1,
        window=(0.0, gap + 1e-9),
    )
    outside = shift_integrand(
        model,
        k,
        None,
        energy_scale=SCALE,
        threshold_eV=1e-4,
        regularization_eV=0.04,
        occupied=1,
        window=(0.0, gap),
    )
    assert inside["pairs"] == everything["pairs"] and outside["pairs"] == []
    assert not outside["integrand"].size


def test_a_nonorthogonal_model_needs_its_position_blocks():
    translations = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    hopping = np.zeros((3, 2, 2), complex)
    hopping[0] = np.diag([0.0, 2.0])
    hopping[1] = hopping[2] = 0.3 * np.ones((2, 2))
    overlap = np.zeros((3, 2, 2), complex)
    overlap[0] = np.eye(2)
    overlap[1] = overlap[2] = 0.05 * np.eye(2)
    model = OpenMXModel(translations, hopping, overlap, lattice=np.diag([3.0, 10.0, 10.0]))
    with pytest.raises(ValueError, match="position blocks"):
        bloch_operators(model, [0.1, 0.0, 0.0])


@pytest.mark.skipif(
    not CALIBRATION_SCFOUT.is_file(), reason=f"no OpenMX calibration scfout at {CALIBRATION_SCFOUT}"
)
def test_a_nonorthogonal_scfout_with_position_moments_meets_queer():
    """A real OpenMX scfout (overlap and position moments, spin-orbit coupling): the transcription's physical
    integrand meets QUEER's estimator, so its position and overlap terms follow the same conventions."""
    model = OpenMXModel.from_scfout(CALIBRATION_SCFOUT)
    assert model.position_blocks and np.abs(model.overlap([0.1, 0.2, 0.0]) - np.eye(model.nbnd)).max() > 0.1
    rng = np.random.default_rng(5)
    for k in rng.random((2, 3)) * [1, 1, 0]:
        energies, weights, pairs, point = queer_transitions(model, k, 3)
        port = shift_integrand(
            model,
            k,
            pairs,
            energy_scale=1.0,
            threshold_eV=float(point["raw"]["degeneracy_tol"]),
            regularization_eV=None,
            occupied=3,
        )
        absolute = np.max(np.abs(weights).sum(axis=0))
        assert np.max(np.abs(queer_weights(port["integrand"]) - weights)) <= 1e-10 * absolute
