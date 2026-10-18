"""The independent recomputation of a Generator shift current from its scfout (``pyatb_shift_integrand``).

Its physical mode is QUEER's shift current written from the equations (its own Bloch sums, generalized
eigenproblem and covariant treatment of exact degeneracies, no code of QUEER's observables): it must meet QUEER's
estimator transition by transition and table by table. Its PyATB-literal mode is what PyATB 60fc6b2a prints: it
must meet ``pyatb_shift_port``, a loop-by-loop port of PyATB's method 1 that shares no code with it. The two
orbital bases measure each table's rounding, and a transition on a bin edge is priced both ways.
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest
from openmx_files import shift_collinear_scfout, shift_spinor_scfout
from pyatb_shift_port import pyatb_shift

from queer import ExecutionOptions, OpenMXModel
from queer.compat.generator.openmx_params import SHIFT_DEFAULTS
from queer.compat.generator.openmx_shift import component_table
from queer.compat.generator.openmx_steps import single_channel_model
from queer.compat.generator.pyatb_conventions import pyatb_shift_constant_ratio
from queer.compat.generator.pyatb_shift_integrand import (
    PATH_STEPS,
    PYATB_REGULARIZATION,
    PYATB_THRESHOLD,
    ShiftRun,
    bloch_operators,
    channels,
    cluster_labels,
    exactness_bound,
    kpoint_grid,
    orthogonal_basis,
    point_integrand,
    pyatb_velocities,
    recompute,
    work,
)
from queer.models.evaluation import derivatives
from queer.observables.nonlinear import shift_current, shift_current_point, shift_current_settings

SERIAL = ExecutionOptions(workers=1)
CALIBRATION_SCFOUT = Path(
    os.environ.get("QUEER_PYATB_CALIBRATION_SCFOUT")
    or Path.home() / ".cache" / "queer-parity-data" / "pyatb-calib-60fc6b2a" / "test.scfout"
)
SPINOR = {"occ_band": 2, "omega": [0.3, 6.5], "domega": 0.02, "eta": 0.1, "grid": [3, 3, 2]}
COLLINEAR = {"occ_band": 1, "omega": [0.2, 4.5], "domega": 0.04, "eta": 0.1, "grid": [6, 6, 1]}


def queer_transitions(model, k, occupied):
    """QUEER's transitions at ``k``: their energies, weights ``(f_n - f_m) Im[r^b r^c_;a]_sym`` and (occupied, empty)
    pairs."""
    settings = shift_current_settings(
        model, np.array([1.0]), model.fermi_energy, eta=0.05, smearing="gaussian", occupied_bands=occupied
    )
    point = shift_current_point(model, k, settings)
    source, target = np.nonzero(point["factor"])
    weights = component_table(
        np.moveaxis(point["factor"][source, target] * point["integrand"][..., source, target], -1, 0)
    )
    return point["gap"][source, target], weights, source, target


def transcribed(model, k, occupied, **options):
    """The transcription's physical transitions at ``k`` in QUEER's weights (``-I / 2``)."""
    result = point_integrand(
        channels(model)[0].reconciled,
        k,
        occupied,
        treatment="covariant",
        threshold=exactness_bound(model),
        **options,
    )
    return result["energies"], -0.5 * result["integrand"], result


def spectrum(energies, weights, eta=0.01):
    omega = np.linspace(0.0, 8.0, 1601)
    energies = np.asarray(energies).reshape(-1)
    kernel = np.exp(-(((omega[:, None] - energies[None, :]) / eta) ** 2)) / (eta * math.sqrt(math.pi))
    return kernel @ np.asarray(weights).reshape(len(energies), -1)


def recomputed(path, block):
    model = OpenMXModel.from_scfout(path)
    full = {**SHIFT_DEFAULTS, **block}
    sign = 1 if np.linalg.det(np.asarray(model.lattice, float)) > 0 else -1
    return model, full, recompute(model, ShiftRun.from_block(full), handedness=sign, workers=1)


def test_its_bloch_sums_are_the_models_and_pyatbs_upper_triangles(tmp_path):
    shift_spinor_scfout(tmp_path / "soc.scfout", overlap_hopping=(0.06, 0.04, 0.03), dipole=0.3)
    model = OpenMXModel.from_scfout(tmp_path / "soc.scfout")
    k = np.array([0.13, 0.41, 0.77])
    channel = channels(model)[0]
    operators = bloch_operators(channel.reconciled, k)
    first, second = derivatives(model, k), derivatives(model, k, order=2)
    assert np.allclose(operators.h[0], model.hamiltonian(k), rtol=0, atol=1e-13)
    assert np.allclose(operators.s[0], model.overlap(k), rtol=0, atol=1e-13)
    assert np.allclose(operators.dh[0], first["H"], rtol=0, atol=1e-12)
    assert np.allclose(operators.d2h[0], second["H"], rtol=0, atol=1e-11)
    assert np.abs(operators.r).max() > 0.01
    # PyATB's sums of the records' upper triangles, completed Hermitian and the position operator's lower triangle
    # from (r + i dS)^dagger, are the same operators for records that hold their translation identities exactly.
    literal = bloch_operators(channel.recorded, k, construction="pyatb")
    for name in ("h", "s", "dh", "ds", "d2h", "d2s", "r", "dr"):
        assert np.allclose(getattr(literal, name), getattr(operators, name), rtol=0, atol=1e-12), name


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
        bloch_operators(channels(model)[0].reconciled, [0.1, 0.0, 0.0])


def test_pyatbs_grid_degenerate_velocities_and_levels():
    assert kpoint_grid([2, 1, 3]).tolist() == [[i / 2, 0.0, j / 3] for i in range(2) for j in range(3)]
    energies = np.array([[0.0, 5e-5, 1.0, 1.0 + 2e-4, 3.0]])
    assert cluster_labels(energies, 1e-4).tolist() == [[0, 0, 1, 2, 3]]
    # PyATB's band velocities: the diagonal, or inside levels within 1e-4 eV the block's eigenvalues in order.
    rng = np.random.default_rng(2)
    hbar = rng.normal(size=(1, 3, 5, 5)) + 1j * rng.normal(size=(1, 3, 5, 5))
    hbar = hbar + np.swapaxes(hbar.conj(), -1, -2)
    velocities = pyatb_velocities(hbar, np.zeros_like(hbar), energies, 1e-4)
    block = np.linalg.eigvalsh(hbar[0, :, :2, :2], UPLO="L")
    assert np.allclose(velocities[0, :, :2], block)
    assert np.allclose(velocities[0, :, 2:], np.diagonal(hbar[0], axis1=-2, axis2=-1)[:, 2:].real)


@pytest.mark.parametrize(
    ("label", "model_args", "points"),
    [
        ("generic", {}, np.random.default_rng(3).random((3, 3))),
        ("split", {"exchange": 2e-5}, [[0.5, 0.5, 0.0], [0.0, 0.0, 0.5]]),
        ("nonorthogonal", {"overlap_hopping": (0.06, 0.04, 0.03), "dipole": 0.4}, [[0.13, 0.27, 0.31]]),
    ],
)
def test_the_physical_mode_meets_queers_estimator_transition_by_transition(
    tmp_path, label, model_args, points
):
    """At generic k-points, where Kramers pairs are split by 4e-5 eV (PyATB's method 1 regularizes them) and in a
    nonorthogonal basis with position moments: the same energies and weights to rounding."""
    shift_spinor_scfout(tmp_path / "soc.scfout", **model_args)
    model = OpenMXModel.from_scfout(tmp_path / "soc.scfout")
    for k in points:
        energies, weights, source, target = queer_transitions(model, np.asarray(k), 2)
        mine_energies, mine, _ = transcribed(model, k, 2)
        assert np.max(np.abs(mine_energies[source, target - 2] - energies)) <= 1e-12
        absolute = np.max(np.abs(weights).sum(axis=0))
        assert np.max(np.abs(mine[source, target - 2] - weights)) <= 1e-9 * absolute, (label, k)


def test_the_covariant_treatment_of_an_exact_degeneracy_is_queers_in_any_basis(tmp_path):
    """Exact Kramers pairs at the TRIM points of a nonorthogonal spinor model with position moments and on-site
    dipoles: the physical mode (parallel transport and the velocity matrix inside the degenerate levels) meets QUEER's
    covariant estimator in any orbital basis, so in any eigenvector basis inside the levels. PyATB's literal treatment
    (no derivative inside them, the block's velocity eigenvalues) does not, and its value depends on that basis: what
    the two orbital bases of a comparison measure."""
    shift_spinor_scfout(
        tmp_path / "kramers.scfout", exchange=0.0, overlap_hopping=(0.06, 0.04, 0.03), dipole=0.6
    )
    model = OpenMXModel.from_scfout(tmp_path / "kramers.scfout")
    blocks = channels(model)[0].reconciled
    for k, degenerate in (([0.0, 0.0, 0.0], True), ([0.5, 0.0, 0.0], True), ([0.13, 0.27, 0.31], False)):
        energies, weights, _, _ = queer_transitions(model, np.asarray(k), 2)
        queer = spectrum(energies, weights)
        peak = np.max(np.abs(queer))
        bases = (None, orthogonal_basis(4), orthogonal_basis(4, seed=7))
        for basis in bases:
            mine_energies, mine, _ = transcribed(model, k, 2, basis=basis)
            assert np.max(np.abs(spectrum(mine_energies, mine) - queer)) <= 1e-12 * peak, k
        literal = []
        for basis in bases:
            result = point_integrand(
                blocks,
                k,
                2,
                treatment="pyatb",
                threshold=PYATB_THRESHOLD,
                regularization=PYATB_REGULARIZATION,
                basis=basis,
            )
            literal.append(spectrum(result["energies"], -0.5 * result["integrand"]))
        if degenerate:
            assert np.max(np.abs(literal[0] - queer)) > 1e-5 * peak, k
            assert max(np.max(np.abs(other - literal[0])) for other in literal[1:]) > 1e-6 * peak, k
        else:
            assert max(np.max(np.abs(other - queer)) for other in literal) <= 1e-12 * peak, k


@pytest.mark.parametrize(
    ("label", "writer", "model_args", "nspin", "block"),
    [
        ("spinor", shift_spinor_scfout, {}, 4, SPINOR),
        ("narrow", shift_spinor_scfout, {}, 4, {**SPINOR, "eta": 0.01, "domega": 0.015}),
        ("binned", shift_spinor_scfout, {}, 4, {**SPINOR, "smearing_method": 0, "domega": 0.05}),
        ("window", shift_spinor_scfout, {}, 4, {**SPINOR, "omega": [2.0, 4.0]}),
        ("pair", shift_spinor_scfout, {}, 4, {**SPINOR, "omega": [1.5, 6.5], "n_occ": 2, "m_unocc": 3}),
        ("left", shift_spinor_scfout, {"lattice": np.diag([3.0, 3.2, -3.5])}, 4, SPINOR),
        ("split", shift_spinor_scfout, {"exchange": 2e-5}, 4, {**SPINOR, "grid": [4, 4, 4]}),
        (
            "nonorthogonal",
            shift_spinor_scfout,
            {"overlap_hopping": (0.06, 0.04, 0.03), "dipole": 0.3},
            4,
            SPINOR,
        ),
        ("collinear", shift_collinear_scfout, {}, 1, COLLINEAR),
        ("nonmagnetic", shift_collinear_scfout, {"spin": 0}, 1, COLLINEAR),
        ("scissors", shift_collinear_scfout, {}, 1, {**COLLINEAR, "scissors": 0.3}),
    ],
)
def test_the_literal_mode_is_pyatbs_table(tmp_path, label, writer, model_args, nspin, block):
    """The PyATB-literal table equals the loop-by-loop port of PyATB's method 1 within its own rounding tolerance,
    which is far below PyATB's print precision."""
    writer(tmp_path / "model.scfout", **model_args)
    model, full, recomputation = recomputed(tmp_path / "model.scfout", block)
    _, port = pyatb_shift(model, full, nspin=nspin)
    literal = recomputation.literal
    peak = np.max(np.abs(port))
    assert peak > 0
    assert np.all(np.abs(literal.value - port) <= literal.rounding + 1e-15 * peak), label
    assert np.max(literal.rounding) <= 1e-9 * peak
    # The steps from QUEER's value in PyATB's layout to PyATB's table, one ingredient at a time.
    path = dict(recomputation.path)
    assert [name for name, _ in recomputation.path] == ["queer", *(name for name, _ in PATH_STEPS)]
    assert np.allclose(path["constants"], pyatb_shift_constant_ratio() * path["queer"], rtol=1e-15, atol=0)
    assert np.array_equal(path["cut"], literal.value)


def test_the_physical_tables_are_queers_shift_current(tmp_path):
    """The physical mode's tables against QUEER's observable: a collinear scfout's two channels, its legacy channel
    (PyATB's spin factor times the spin-up one), and the binned table without smearing."""
    shift_collinear_scfout(tmp_path / "collinear.scfout")
    block = {**COLLINEAR, "smearing_method": 0}
    model, full, recomputation = recomputed(tmp_path / "collinear.scfout", block)
    run = ShiftRun.from_block(full)
    common = {
        "occupied_bands": 1,
        "grid": (6, 6, 1),
        "execution": SERIAL,
        "smearing": "none",
        "bin_width": 0.04,
    }
    channel = [
        component_table(
            shift_current(single_channel_model(model, index), run.omega, **common)["shift_current"]
        )
        for index in (0, 1)
    ]
    binned = [
        0.04
        * component_table(
            shift_current(single_channel_model(model, index), run.omega + 0.02, **common)["shift_current"]
        )
        for index in (0, 1)
    ]
    physical = recomputation.physical
    for name, expected in (
        ("sigma", channel[0] + channel[1]),
        ("legacy", 2 * channel[0]),
        ("binned", binned[0] + binned[1]),
        ("legacy_binned", 2 * binned[0]),
    ):
        table = physical[name]
        assert np.all(np.abs(table.value - expected) <= table.rounding), name
        # Two transitions lie exactly on the bin edge at 1.64 eV: either placement is the binned value.
        assert bool(np.max(table.edge) > 0) == name.endswith("binned"), name


def test_a_transition_on_a_bin_edge_is_priced_both_ways(tmp_path):
    """The collinear model's two transitions at exactly 1.64 eV lie on the edge between rows 35 and 36 of the bins
    (w, w + domega]: the edge term holds their weight in both rows, and in the sum over the rows only in row 35."""
    shift_collinear_scfout(tmp_path / "collinear.scfout")
    _, _, recomputation = recomputed(tmp_path / "collinear.scfout", {**COLLINEAR, "smearing_method": 0})
    table = recomputation.physical["binned"]
    rows = np.flatnonzero(np.max(table.edge, axis=1))
    assert rows.tolist() == [35, 36]
    assert np.allclose(table.edge[35], table.edge[36]) and np.max(table.edge[35]) > 0.1 * np.max(
        np.abs(table.value)
    )
    assert np.flatnonzero(np.max(table.edge_cumulative, axis=1)).tolist() == [35]
    assert np.allclose(table.edge_cumulative[35], table.edge[35])
    assert np.all(table.smooth <= 1e-9 * np.max(np.abs(table.value)))


def test_its_workers_give_the_same_tables(tmp_path, monkeypatch):
    """The recomputation split over worker processes sums the same k-points (in another order)."""
    from queer.compat.generator import pyatb_shift_integrand

    monkeypatch.setattr(pyatb_shift_integrand, "SERIAL_WORK", 0.0)
    shift_spinor_scfout(tmp_path / "soc.scfout")
    model = OpenMXModel.from_scfout(tmp_path / "soc.scfout")
    run = ShiftRun.from_block({**SHIFT_DEFAULTS, **SPINOR, "grid": [4, 4, 3]})
    assert work(model, run) == 48 * 4**3
    serial = recompute(model, run, handedness=1, workers=1)
    parallel = recompute(model, run, handedness=1, workers=2, memory_bytes=4e9)
    assert serial.facts["workers"] == 1 and parallel.facts["workers"] == 2
    for name in ("sigma",):
        first, second = serial.physical[name], parallel.physical[name]
        assert np.all(np.abs(first.value - second.value) <= first.rounding)
    assert np.all(np.abs(serial.literal.value - parallel.literal.value) <= serial.literal.rounding)
    assert serial.facts["transitions"] == parallel.facts["transitions"] == 48 * 4


@pytest.mark.skipif(
    not CALIBRATION_SCFOUT.is_file(), reason=f"no OpenMX calibration scfout at {CALIBRATION_SCFOUT}"
)
def test_a_real_nonorthogonal_scfout_with_position_moments_meets_queer():
    """The real OpenMX calibration scfout (overlap, position moments, spin-orbit coupling): the physical mode meets
    QUEER's estimator at generic k-points; at Gamma and M, where its inversion makes every integrand vanish but for
    rounding, the two orbital bases' difference measures that rounding, and QUEER's estimator meets it."""
    model = OpenMXModel.from_scfout(CALIBRATION_SCFOUT)
    assert model.position_blocks and np.abs(model.overlap([0.1, 0.2, 0.0]) - np.eye(model.nbnd)).max() > 0.1
    rng = np.random.default_rng(5)
    for k in rng.random((2, 3)) * [1, 1, 0]:
        energies, weights, source, target = queer_transitions(model, k, 3)
        _, mine, _ = transcribed(model, k, 3)
        absolute = np.max(np.abs(weights).sum(axis=0))
        assert np.max(np.abs(mine[source, target - 3] - weights)) <= 1e-10 * absolute
    basis = orthogonal_basis(model.nbnd)
    for k in ([0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]):
        energies, weights, source, target = queer_transitions(model, np.asarray(k), 3)
        first_energies, first, _ = transcribed(model, k, 3)
        second_energies, second, _ = transcribed(model, k, 3, basis=basis)
        queer = spectrum(energies, weights)
        mine = spectrum(first_energies[source, target - 3], first[source, target - 3])
        spread = np.max(
            np.abs(mine - spectrum(second_energies[source, target - 3], second[source, target - 3]))
        )
        assert spread > 0 and np.max(np.abs(mine - queer)) <= 100 * spread, k
