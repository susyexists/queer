"""Independent finite-field invariants and small analytic reference problems."""

import json
import math

import numpy as np
import pytest

from queer.capabilities import CapabilityError
from queer.finite_field import (
    PAPER,
    PHYSICAL,
    PHYSICAL_EQ8,
    FiniteFieldOptions,
    FourBandModel,
    FourBandParams,
    PreparedHallResponse,
    berry_flux,
    crossing_scan,
    decode_line,
    encode_line,
    houston_line,
    lindblad_line,
)
from queer.finite_field.geometry.gridded import GriddedGeometry
from queer.finite_field.geometry.provider import ModelGeometry
from queer.finite_field.harmonics import harmonics, power_law_bN
from queer.finite_field.lz.propagation import p_int_two, propagate_crossings, stokes_phase
from queer.finite_field.models.toy import LinearLZ
from queer.finite_field.tdse.houston import houston_line as reference_houston
from queer.finite_field.transport.critical import c_from_device, c_from_harmonic, critical_harmonics_analytic
from queer.finite_field.transport.hall import HallCalculator
from queer.finite_field.transport.reference import naive_jh
from queer.finite_field.transport.tables import LambdaGrid, build_table, eval_tables


def test_default_paper_options_and_json_roundtrip():
    options = FiniteFieldOptions()
    assert options.conventions.charge_sign == 1
    assert options.conventions.ec_method == "eq8"
    assert options.crossings.definition == "gap_minima"
    options = options.with_(**{"relaxation.T1": math.inf, "relaxation.T2": math.inf})
    text = json.dumps(options.to_dict(), allow_nan=False)
    restored = FiniteFieldOptions.from_json(text)
    assert restored.relaxation.T1 == math.inf
    assert restored.digest() == options.digest()


def test_lz_exact_transition_and_unitarity():
    model = LinearLZ(delta=0.0001, alpha=0.2)
    field = np.array([1e3, 5e3, 2e4])
    ec = np.pi * model.delta**2 / model.alpha * 1e10
    result = reference_houston(model, 0.0, field, -0.05, 0.05, dphi=0.05)
    assert np.allclose(result.P, np.exp(-ec / field), atol=2e-5, rtol=0)
    assert result.info["unitarity"] < 1e-11
    assert np.allclose(result.T.sum(axis=1), 1, atol=1e-11)
    assert np.allclose(result.T.sum(axis=2), 1, atol=1e-11)
    public = houston_line(model, 0.0, field, window=(-0.05, 0.05), dphi=0.05)
    reverse = reference_houston(model, 0.0, -field, -0.05, 0.05, dphi=0.05)
    assert np.allclose(public["population"], result.P, atol=1e-12)
    physical = houston_line(model, 0.0, field, window=(-0.05, 0.05), options=PHYSICAL, dphi=0.05)
    assert np.allclose(physical["population"], reverse.P, atol=1e-12)
    assert np.max(public["remote_leakage"]) < 1e-11


def test_density_matrix_propagation_and_positivity():
    rng = np.random.default_rng(5)
    probabilities = rng.uniform(0, 1, size=(5, 23))
    segments = rng.uniform(0, 0.03, size=(4, 23))
    phase = rng.uniform(-10, 10, size=(4, 23))
    lengths = (0.02, 0.015)
    actual, coherence = propagate_crossings(probabilities, segments, phase, *lengths, return_coherence=True)
    rho = np.zeros((23, 2, 2), complex)
    rho[:, 0, 0] = 1
    expected = []
    for j, probability in enumerate(probabilities):
        q = np.sqrt(1 - probability)
        p = np.sqrt(probability)
        scattering = np.array([[q, -p], [p, q]]).transpose(2, 0, 1)
        rho = scattering @ rho @ scattering.conj().transpose(0, 2, 1)
        expected.append(rho[:, 1, 1].real.copy())
        if j < 4:
            d = segments[j]
            rho[:, 1, 1] *= np.exp(-d / lengths[0])
            rho[:, 0, 0] = 1 - rho[:, 1, 1]
            rho[:, 1, 0] *= np.exp(-d / (2 * lengths[0]) - d / lengths[1] - 1j * phase[j])
            rho[:, 0, 1] = rho[:, 1, 0].conj()
    assert np.allclose(actual, expected, atol=1e-14, rtol=0)
    assert np.all((actual >= -1e-15) & (actual <= 1 + 1e-15))
    assert np.all(np.abs(coherence) ** 2 <= actual * (1 - actual) + 1e-14)
    pair = propagate_crossings(probabilities[:2], segments[:1], phase[:1], *lengths)
    assert np.allclose(pair[1], p_int_two(*probabilities[:2], segments[0], *lengths, phase[0]), atol=1e-13)
    assert stokes_phase(0) == pytest.approx(np.pi / 4)
    assert stokes_phase(math.inf) == 0


def test_tables_against_independent_direct_sums():
    radii = np.geomspace(1e-12, 0.03, 401)
    coefficients = np.cos(500 * radii) * np.diff(np.r_[0, radii])
    grid = LambdaGrid.make(3e-11, 30.0, per_decade=12)
    table, limit, moment = build_table(radii, coefficients, grid)
    lengths = np.geomspace(1e-10, 10.0, 180)
    actual = eval_tables(table, limit, moment, grid, lengths)
    expected = np.array([np.sum(coefficients * np.exp(-radii / length)) for length in lengths])
    assert np.max(np.abs(actual - expected)) < 1e-7 * np.max(np.abs(expected))
    assert eval_tables(table, limit, moment, grid, [math.inf])[0] == pytest.approx(coefficients.sum())


def test_harmonic_polynomials_and_power_law():
    amplitude = np.array([0.2, 1.0, 4.0])
    cubic = harmonics(lambda fields: fields**3, amplitude, n_max=7)
    assert np.allclose(cubic.b[:, 1], 0.75 * amplitude**3, rtol=1e-14)
    assert np.allclose(cubic.b[:, 3], -0.25 * amplitude**3, rtol=1e-14)
    quadratic = harmonics(lambda fields: fields**2, amplitude, n_max=7)
    assert np.allclose(quadratic.c0, amplitude**2 / 2, rtol=1e-14)
    assert np.allclose(quadratic.a[:, 2], -(amplitude**2) / 2, rtol=1e-14)
    assert np.array_equal(quadratic.c, (quadratic.a - 1j * quadratic.b) / 2)
    power = harmonics(lambda fields: np.sign(fields) * np.abs(fields) ** 1.5, amplitude)
    assert power.converged
    assert np.allclose(power.b, amplitude[:, None] ** 1.5 * power_law_bN(power.N, 1.5), atol=1e-13, rtol=1e-8)


def test_oscillatory_fourier_independent_period_reference_and_budget():
    def current(fields):
        absolute = np.maximum(np.abs(fields), 1e-100)
        return np.sign(fields) * np.exp(-1 / absolute) * (1 + np.cos(8 / absolute))

    amplitudes = np.array([0.3, 1.0, 4.0])
    projected = harmonics(current, amplitudes, n_max=9)
    theta = 2 * np.pi * (np.arange(2**17) + 0.5) / 2**17
    reference = np.array(
        [
            [2 * np.mean(current(e0 * np.sin(theta)) * np.sin(n * theta)) for n in range(10)]
            for e0 in amplitudes
        ]
    )
    assert projected.converged
    assert np.max(np.abs(projected.b - reference)) < 1e-7 * np.max(np.abs(reference))
    capped = harmonics(current, amplitudes, n_theta=32, n_cap=32)
    assert not capped.converged


def test_physical_ddp_source_and_missing_capabilities():
    model = FourBandModel(FourBandParams().at(0.5))
    scan = crossing_scan(model, [0.0], options=PHYSICAL)
    assert scan.metadata["options"]["label"] == "physical"
    assert scan.metadata["ddp_source"] == ["exact"]
    assert scan["ddp_field"][0] == pytest.approx(16433.79366362, rel=1e-10)
    provider = GriddedGeometry.sample(
        ModelGeometry(model), kx=np.linspace(-0.03, 0.03, 7), ky=np.linspace(-0.03, 0.03, 7)
    )
    with pytest.raises(CapabilityError, match="wavefunction"):
        PreparedHallResponse(provider=provider, options=PHYSICAL)
    # Explicitly selecting the available approximation is permitted.
    available = PreparedHallResponse(provider=provider, options="paper")
    assert available.options.conventions.ec_method == "eq8"
    approximated = PreparedHallResponse(provider=provider, options="physical_eq8")
    assert approximated.options == PHYSICAL_EQ8
    assert approximated.options.conventions.charge_sign == -1
    assert approximated.options.conventions.berry == "c_minus_v"


def test_physical_eq8_generalized_model_and_adapter_overrides():
    from queer import OpenMXModel, WannierModel
    from queer.finite_field import FieldPlane, NativePlaneModel

    sx = np.array([[0, 1], [1, 0]], complex)
    sz = np.diag([1, -1])
    translations = [[0, 0, 0], [1, 0, 0], [-1, 0, 0]]
    hopping = np.array([0.3 * sz, sx / (2j), -sx / (2j)])
    metric = np.zeros_like(hopping)
    metric[0] = np.eye(2)
    model = OpenMXModel(
        translations,
        hopping,
        metric,
        lattice=np.eye(3),
        spin_degeneracy=1,
        position=np.zeros((3, *hopping.shape), complex),
    )
    scan = crossing_scan(model, [0.0], options="physical_eq8", plane=FieldPlane(), band_pair=(0, 1))
    assert scan.metadata["options"]["conventions"]["charge_sign"] == -1
    assert np.isfinite(scan["critical_field"]).all()
    native = NativePlaneModel(WannierModel(translations, hopping, lattice=np.eye(3)), band_pair=(0, 1))
    rotated = FieldPlane(direction=(0, 1, 0), transverse=(1, 0, 0))
    projector = np.diag([1.0, 0.0])

    def supplied():
        return projector

    overridden = PreparedHallResponse(native, "physical_eq8", plane=rotated, layer_projector=supplied)
    assert overridden.model.plane == rotated
    assert np.array_equal(overridden.model.layer_projector(), projector)


def test_native_flux_embedding_and_nonorthogonal_covariance():
    from numpy.polynomial.legendre import leggauss

    from queer import OpenMXModel, WannierModel
    from queer.finite_field import FieldPlane, NativePlaneModel
    from queer.finite_field.geometry.topology import fhs_flux

    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.diag([1, -1])
    translations = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    hopping = np.array([0.3 * sz, sx / (2j), -sx / (2j), sy / (2j), -sy / (2j)])
    plain = WannierModel(translations, hopping, lattice=np.eye(3))
    embedded = WannierModel(translations, hopping, lattice=np.eye(3), centres=[[0, 0, 0], [0.3, -0.5, 0]])
    plane = FieldPlane(origin=(0.06, 0.02, 0))
    view = NativePlaneModel(embedded, plane=plane, band_pair=(0, 1))
    nodes, weights = leggauss(28)
    x, y = np.meshgrid(0.15 * nodes, 0.15 * nodes, indexing="ij")
    curvature = view.geometry(x, y).omega_v
    direct = np.einsum("i,j,ij->", 0.15 * weights, 0.15 * weights, curvature) / (2 * np.pi)
    result = berry_flux(view, window=0.15, n=31, graded_scale=0.1, tolerance=1e-4)
    assert result["flux"][-1] == pytest.approx(direct, abs=2e-5)
    assert result.metadata["domain"] == "finite_window"
    assert not result.metadata["quantized_chern_certificate"]
    axes = np.linspace(-0.15, 0.15, 31)
    transform = np.array([[1.4, 0.2j], [0, 0.6]], complex)
    generalized_h = np.einsum("ai,raj,jb->rib", transform.conj(), hopping, transform)
    generalized_s = np.zeros_like(hopping)
    generalized_s[0] = transform.conj().T @ transform
    generalized = OpenMXModel(
        translations,
        generalized_h,
        generalized_s,
        lattice=np.eye(3),
        spin_degeneracy=1,
        position=np.zeros((3, *hopping.shape), complex),
    )
    a = NativePlaneModel(plain, plane=plane, band_pair=(0, 1))
    b = NativePlaneModel(generalized, plane=plane, band_pair=(0, 1))
    assert fhs_flux(a, [0], axes, axes) == pytest.approx(fhs_flux(b, [0], axes, axes), abs=2e-13)
    reversed_plane = FieldPlane(origin=plane.origin, direction=(0, 1, 0), transverse=(1, 0, 0))
    reversed_view = NativePlaneModel(plain, plane=reversed_plane, band_pair=(0, 1))
    assert fhs_flux(a, [0], axes, axes) == pytest.approx(-fhs_flux(reversed_view, [0], axes, axes), abs=2e-13)


def test_lindblad_unitary_limit_and_relaxation_validity():
    model = LinearLZ(delta=0.0001, alpha=0.2)
    fields = np.array([5e3, 2e4, -5e3])
    options = PHYSICAL.with_(**{"relaxation.T1": math.inf, "relaxation.T2": math.inf})
    unitary = houston_line(model, 0.0, fields, window=(-0.05, 0.05), options=options, dphi=0.05)
    relaxed = lindblad_line(model, 0.0, fields, window=(-0.05, 0.05), options=options, dphi=0.05)
    assert np.allclose(relaxed["population"], unitary["population"], atol=1e-10, rtol=0)
    assert relaxed["valid"].all()
    damped = lindblad_line(model, 0.0, fields, window=(-0.05, 0.05), dphi=0.05)
    assert damped["valid"].all()


def test_optional_compiled_response_matches_numpy(prepared_paper):
    pytest.importorskip("numba")
    accelerated = PreparedHallResponse(
        prepared_paper.model, prepared_paper.options, keep_nodes=True, acceleration=True
    )
    accelerated.prepare(
        load_layout=lambda: prepared_paper.layout, load_line=lambda i, t: prepared_paper.lines[i]
    )
    fields = np.r_[0.0, np.geomspace(1e3, 5e4, 37), -np.geomspace(1e3, 5e4, 37)]
    expected = prepared_paper.J(fields)
    assert np.allclose(accelerated.J(fields), expected, rtol=1e-12, atol=1e-14 * np.max(np.abs(expected)))


@pytest.fixture(scope="module")
def prepared_paper():
    options = PAPER.with_(**{"quadrature.du": 0.3}, use_symmetry_shortcuts=True)
    return PreparedHallResponse(FourBandModel(FourBandParams().at(1.4)), options, keep_nodes=True).prepare()


def test_prepared_direct_and_independent_nested_reference(prepared_paper):
    fields = np.array([1e4, 3e4, -1e4, -3e4, 0.0])
    optimized = prepared_paper.J(fields)
    direct = prepared_paper.calculator.J_direct(fields)
    assert np.allclose(optimized, direct, rtol=1e-7, atol=0)
    assert optimized[-1] == 0
    assert np.allclose(optimized[:2], -optimized[2:4], rtol=1e-8)
    independent = naive_jh(prepared_paper.model, 1e4, prepared_paper.options)
    assert optimized[0] == pytest.approx(independent, rel=1e-6)


def test_line_checkpoint_restores_without_numerical_recomputation(prepared_paper, monkeypatch):
    stored = {i: encode_line(line) for i, line in enumerate(prepared_paper.lines)}
    assert all(not array.dtype.hasobject for arrays, _ in stored.values() for array in arrays.values())
    restored = PreparedHallResponse(prepared_paper.model, prepared_paper.options, keep_nodes=True)
    monkeypatch.setattr(
        "queer.finite_field.transport.hall.solve_line",
        lambda *args, **kwargs: pytest.fail("A completed line was recomputed"),
    )
    restored.prepare(load_line=lambda i, t: decode_line(*stored[i]))
    assert np.array_equal(restored.J([1e4, -2e4]), prepared_paper.J([1e4, -2e4]))


def test_physical_shortcuts_and_corrected_interference():
    model = FourBandModel(FourBandParams().at(1.8))
    cfg = PHYSICAL.with_(**{"quadrature.du": 0.45, "relaxation.T1": math.inf, "relaxation.T2": math.inf})
    fields = np.linspace(4e3, 1.9e4, 21)
    actual = HallCalculator(model, cfg)
    from queer.finite_field.transport.lines import solve_line

    line = solve_line(actual.provider, 0.0, cfg, actual.grid)
    expected = reference_houston(model.traceless(), 0.0, fields, -0.02, 0.02, dphi=0.05)
    population = actual.line_populations(line, fields, 1)[-1]
    leakage = 1 - expected.T[:, expected.v, expected.v] - expected.T[:, expected.c, expected.v]
    valid = leakage < 0.005
    assert valid.all()
    assert np.sqrt(np.mean((population[valid] - expected.P[valid]) ** 2)) < 0.05
    certified = HallCalculator(model, cfg.with_(use_symmetry_shortcuts=True))
    assert certified.selection_rules()["e_odd"]
    assert not certified.selection_rules()["t_mirror"]


def test_phase_budget_and_checkpoint_keep_numerical_validity_separate():
    from queer.finite_field.transport.lines import solve_line

    model = FourBandModel(FourBandParams().at(1.8))
    options = PHYSICAL.with_(**{"phase.phase_nodes": 3, "phase.phase_max_nodes": 3})
    calculator = HallCalculator(model, options, use_numba=False)
    line = solve_line(calculator.provider, 0.0, options, calculator.grid)
    assert "phase_under_resolved" in line.flags
    restored = decode_line(*encode_line(line))
    assert restored.phase_refinement == line.phase_refinement
    calculator.lines = [restored]
    diagnostics = calculator.diagnostics()
    assert diagnostics["numerical_convergence_flags"]["lines"]["phase_under_resolved"] == 1
    assert "phase_under_resolved" not in diagnostics["model_validity_flags"]
    assert diagnostics["phase_refinement"]["unchecked"] == 2


def test_critical_finite_harmonic_and_device_roundtrips():
    velocity, strength = 0.147213, -140.45
    amplitudes = np.array([100.0, 1000.0])
    orders = np.array([1, 3, 5, 9])
    coefficient = critical_harmonics_analytic(amplitudes, orders, strength, velocity)
    assert np.allclose(
        c_from_harmonic(orders[None, :], coefficient, amplitudes[:, None], velocity, finite_N=True),
        strength,
        rtol=1e-12,
    )
    current = np.array([1e-7, 2e-7])
    length, width, resistance = 1e-5, 5e-6, 2e3
    order = np.array([3, 9])
    field = current * resistance / length
    response = critical_harmonics_analytic(field, order, abs(strength), velocity)[np.arange(2), np.arange(2)]
    voltage = response * resistance * width**2 / length
    assert np.allclose(
        c_from_device(order, voltage, current, length, width, resistance, velocity, finite_N=True),
        abs(strength),
        rtol=1e-10,
    )
