"""Production input and Fourier optimizations preserve native operators."""

from types import SimpleNamespace

import numpy as np
import pytest

from queer.execution import ExecutionOptions, execution_session
from queer.finite_field.native import FieldPlane, NativePlaneModel
from queer.io.wannier import WannierData, WignerSeitzData, normalized_operators, read_hr, read_wsvec
from queer.io.wannier_ws import PackedWignerSeitzShifts
from queer.models.wannier import WannierModel


def test_bulk_hr_reader_retains_explicit_indices_and_translation_order(tmp_path):
    n = 129
    rng = np.random.default_rng(12)
    permutation = rng.permutation(2 * n**2)
    first_translation = 1 if permutation[0] < n**2 else -1
    expected_order = [first_translation, -first_translation]
    path = tmp_path / "large_hr.dat"
    with path.open("w") as stream:
        stream.write(f"synthetic\n{n}\n2\n3 7\n")
        for index in permutation:
            translation = 1 if index < n**2 else -1
            m, band = divmod(index % n**2, n)
            stream.write(
                f"{translation} 0 0 {m + 1} {band + 1} {translation + m + band / 1000:.3f}D0 .25D0\n"
            )
    raw = read_hr(path)
    np.testing.assert_array_equal(raw.translations, [[r, 0, 0] for r in expected_order])
    np.testing.assert_array_equal(raw.degeneracies, [3, 7])
    for index, translation in enumerate(expected_order):
        expected = translation + np.arange(n)[:, None] + np.arange(n)[None, :] / 1000 + 0.25j
        np.testing.assert_allclose(raw.hopping[index], expected, rtol=2e-16, atol=1e-15)
    assert not raw.hopping.flags.writeable
    with path.open("a") as stream:
        stream.write(f"{first_translation} 0 0 1 1 0 0\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_hr(path)


@pytest.mark.parametrize("entry", ["0 0 0 1 1 + 0", "0 0 0.0 1 1 0 0", "0 0 0 1 1 1e 0"])
def test_bulk_hr_reader_rejects_permissive_numeric_parser_cases(tmp_path, entry):
    path = tmp_path / "bad_hr.dat"
    path.write_text(f"test\n1\n1\n1\n{entry}\n")
    with pytest.raises(ValueError):
        read_hr(path)


def test_compact_ws_reader_across_text_chunks(tmp_path):
    n = 150
    path = tmp_path / "large_wsvec.dat"
    with path.open("w") as stream:
        stream.write("use_ws_distance = .true.\n")
        for m in range(1, n + 1):
            for band in range(1, n + 1):
                stream.write(f"0 0 0 {m} {band}\n1\n0 0 0\n")
    ws = read_wsvec(path)
    assert isinstance(ws.shifts, PackedWignerSeitzShifts)
    assert len(ws.shifts) == n**2
    np.testing.assert_array_equal(ws.shifts[(0, 0, 0, n, n)], [[0, 0, 0]])
    with pytest.raises(KeyError):
        ws.shifts[(0, 0, 0, 0, 1)]
    hopping = np.arange(n**2, dtype=float).reshape(1, n, n).astype(complex)
    raw = WannierData(np.zeros((1, 3), int), hopping, np.array([3]))
    R, H, _ = normalized_operators(raw, "ws_distance", ws)
    np.testing.assert_array_equal(R, raw.translations)
    np.testing.assert_allclose(H, hopping / 3, rtol=2e-16)


@pytest.mark.parametrize(
    "body, message",
    [
        ("0 0 0 1 1\n1\n0 0\n", "three shift integers"),
        ("0 0 0 1 1\n1\n0 0 -\n", "integer"),
        ("0 0 0 1 1\n1\n0 0 999999999999999999999\n", "int64"),
        ("0 0 0 1 1\n1\n0 0 0.5\n", "integer"),
        ("0 0 0 1 1\n1 1\n0 0 0\n", "one shift count"),
        ("0 0 0 1 1\n0\n", "positive"),
        ("0 0 0 1 1\n2\n0 0 0\n", "truncated"),
        ("0 0 0 1 1\n", "truncated"),
        ("0 0 0 1 1\n2\n0 0 0\n0 0 0\n", "duplicate equivalent"),
        ("0 0 0 1 1\n1\n0 0 0\n0 0 0 1 1\n1\n0 0 0\n", "duplicate wsvec"),
    ],
)
def test_compact_ws_reader_retains_strict_validation(tmp_path, body, message):
    path = tmp_path / "invalid_wsvec.dat"
    path.write_text("use_ws_distance = .true.\n" + body)
    with pytest.raises(ValueError, match=message):
        read_wsvec(path)


def test_compact_ws_scatter_matches_independent_image_sum():
    rng = np.random.default_rng(9)
    R = np.array([[1, 0, 0], [0, 0, 0], [-1, 0, 0]])
    H = rng.normal(size=(3, 3, 3)) + 1j * rng.normal(size=(3, 3, 3))
    position = rng.normal(size=(3, 3, 3, 3)) + 1j * rng.normal(size=(3, 3, 3, 3))
    data = WannierData(R, H, np.array([2, 3, 2]), position)
    mapping = {
        (*translation, m + 1, n + 1): np.array([[0, 0, 0], [2, 0, 0]])
        for translation in R
        for m in range(3)
        for n in range(3)
    }
    compact = PackedWignerSeitzShifts.from_mapping(mapping)
    expected = {}
    for index, translation in enumerate(R):
        for m in range(3):
            for n in range(3):
                for image in mapping[(*translation, m + 1, n + 1)]:
                    key = tuple(translation + image)
                    if key not in expected:
                        expected[key] = [np.zeros((3, 3), complex), np.zeros((3, 3, 3), complex)]
                    expected[key][0][m, n] += H[index, m, n] / (2 * data.degeneracies[index])
                    expected[key][1][m, n] += position[index, m, n] / (2 * data.degeneracies[index])
    for chunk in (1, 7, 65536):
        compact.chunk_size = chunk
        vectors, h, r = normalized_operators(data, "ws_distance", WignerSeitzData(compact, True, False))
        np.testing.assert_array_equal(vectors, sorted(expected))
        np.testing.assert_allclose(h, [expected[key][0] for key in sorted(expected)], rtol=2e-16)
        np.testing.assert_allclose(r, [expected[key][1] for key in sorted(expected)], rtol=2e-16)


def _model(n=4):
    translations = np.array([(a, b, 0) for a in range(-3, 4) for b in range(-3, 4)])
    rng = np.random.default_rng(4)
    H = rng.normal(size=(len(translations), n, n)) + 1j * rng.normal(size=(len(translations), n, n))
    position = rng.normal(size=(len(translations), n, n, 3)) + 1j * rng.normal(
        size=(len(translations), n, n, 3)
    )
    H = (H + H[::-1].conj().transpose(0, 2, 1)) * 0.03
    position = (position + position[::-1].conj().transpose(0, 2, 1, 3)) * 0.02
    H[len(translations) // 2] += np.diag(np.arange(n) * 2.0)
    lattice = np.array([[4, 0, 0], [-2, np.sqrt(12), 0], [0, 0, 20]])
    centres = np.zeros((n, 3))
    centres[:, 2] = np.linspace(-5, 5, n)
    return WannierModel(translations, H, lattice=lattice, centres=centres, position=position)


def test_line_factorization_preserves_operators_and_complex_continuation():
    model = _model()
    plane = FieldPlane(origin=(0.04, 0.01, 0))
    view = NativePlaneModel(model, plane=plane, occupied_bands=2)
    s = np.linspace(-0.03, 0.03, 9)
    t = 0.003
    points = plane.fractional(model, s, t)
    ref_H = view.hamiltonian(s, t)
    ref_gap = view.gap(s, t)
    ref_geometry = view.geometry(s, t)
    ref_position = [model.position(k) for k in points]
    ref_derivatives = [model.derivatives(k) for k in points]
    ref_second = [model.derivatives(k, 2) for k in points]
    complex_points = plane.fractional(model, s + 0.001j, t)
    ref_complex = model.analytic_hamiltonian(complex_points)
    with view.on_line(0.0, t):
        np.testing.assert_allclose(view.hamiltonian(s, t), ref_H, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(view.gap(s, t), ref_gap, rtol=1e-12)
        got = view.geometry(s, t)
        for name in ("omega_v", "omega_c", "g_cv", "g_v_total", "g_c_total"):
            np.testing.assert_allclose(
                getattr(got, name), getattr(ref_geometry, name), rtol=1e-11, atol=1e-13
            )
        for index, point in enumerate(points):
            np.testing.assert_allclose(model.position(point), ref_position[index], rtol=1e-12, atol=1e-14)
            for order, reference in ((1, ref_derivatives), (2, ref_second)):
                got = model.derivatives(point, order)
                for key in ("H", "S", "r"):
                    np.testing.assert_allclose(got[key], reference[index][key], rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(
            model.analytic_hamiltonian(complex_points), ref_complex, rtol=1e-12, atol=1e-14
        )
        with view.on_line(0.0, t + 0.002):
            view.gap(s, t + 0.002)
        np.testing.assert_allclose(view.hamiltonian(s, t), ref_H, rtol=1e-12, atol=1e-14)
        off_line = view.hamiltonian(s, t + 0.004)
        assert model._line_cache
    assert model._line_context is None
    np.testing.assert_array_equal(off_line, view.hamiltonian(s, t + 0.004))


def test_wrapped_slab_and_explicit_layer_projector():
    model = _model()
    model.centres = np.array([[0, 0, -9], [0, 0, -4], [0, 0, 3], [0, 0, 8]])
    np.testing.assert_array_equal(model.slab_z(), [11, 16, 3, 8])
    expected = np.diag([1, 1, 0, 0])
    np.testing.assert_array_equal(model.layer_projector(), expected)
    view = NativePlaneModel(model, occupied_bands=2)
    np.testing.assert_array_equal(view.layer_projector(), expected)
    override = np.diag([1, 0, 1, 0])
    explicit = NativePlaneModel(model, occupied_bands=2, layer_projector=override)
    np.testing.assert_array_equal(explicit.layer_projector(), override)


@pytest.mark.parametrize("engine", ["direct", "kubo"])
@pytest.mark.parametrize("full_position", [False, True])
def test_batched_pair_geometry_matches_general_covariant_engine(engine, full_position):
    model = _model()
    if not full_position:
        model.position_matrices = None
        model.centres = model.centres + np.array([0.4, -0.2, 0])

    class GeneralView:
        def __getattr__(self, name):
            if name == "_fourier_values":
                raise AttributeError(name)
            return getattr(model, name)

    plane = FieldPlane(
        origin=(0.01, -0.03, 0.02),
        direction=(1 / np.sqrt(2), 0, 1 / np.sqrt(2)),
        transverse=(0, 1, 0),
    )
    fast = NativePlaneModel(model, plane=plane, occupied_bands=2)
    general = NativePlaneModel(GeneralView(), plane=plane, occupied_bands=2)
    s, t = np.array([0.01, 0.02, -0.03]), np.array([0.003, 0.005, -0.007])
    for angle in (0.0, 0.37):
        a = fast.geometry(s, t, angle, engine=engine, keep_eps=True)
        b = general.geometry(s, t, angle, engine=engine)
        for name in ("eps_v", "eps_c", "omega_v", "omega_c", "g_cv", "g_v_total", "g_c_total"):
            np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(a.eps_g, b.eps_g, rtol=1e-14)
        assert a.eps.shape == (3, 4)


@pytest.mark.parametrize("engine", ["direct", "kubo"])
@pytest.mark.parametrize("component", [0, 2])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("factorized", [False, True])
def test_pair_geometry_rejects_nonhermitian_position(engine, component, batched, factorized):
    from contextlib import nullcontext

    from queer.models.wannier import NumericalError

    translations = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])
    hopping = np.zeros((3, 2, 2), complex)
    hopping[1] = np.diag([-1, 1])
    position = np.zeros((3, 2, 2, 3), complex)
    position[1, 0, 1, component] = 0.1
    model = WannierModel(translations, hopping, lattice=np.eye(3), position=position)
    view = NativePlaneModel(model, occupied_bands=1)
    s = np.array([-0.01, 0.02]) if batched else 0.01
    with pytest.raises(NumericalError, match="Non-Hermitian"):
        model.position([0, 0, 0])
    context = view.on_line(0.0, 0.003) if factorized else nullcontext()
    with context, pytest.raises(NumericalError, match="Non-Hermitian"):
        view.geometry(s, 0.003, engine=engine)
    if factorized:
        assert model._line_cache


@pytest.mark.parametrize("engine", ["direct", "kubo"])
def test_pair_geometry_hermitizes_position_roundoff_like_native_engine(engine):
    position = np.zeros((1, 2, 2, 3), complex)
    position[0, 0, 1, 0], position[0, 1, 0, 0] = 0.1 + 0.04j, 0.1 - 0.04j + 8e-11
    position[0, 0, 1, 1], position[0, 1, 0, 1] = 0.02j, -0.02j
    model = WannierModel([[0, 0, 0]], [np.diag([-1, 1])], lattice=np.eye(3), position=position)

    class GeneralView:
        def __getattr__(self, name):
            if name == "_fourier_values":
                raise AttributeError(name)
            return getattr(model, name)

    fast = NativePlaneModel(model, occupied_bands=1).geometry([0.01, 0.02], 0.003, engine=engine)
    general = NativePlaneModel(GeneralView(), occupied_bands=1).geometry([0.01, 0.02], 0.003, engine=engine)
    for name in ("omega_v", "omega_c", "g_cv", "g_v_total", "g_c_total", "v_cv"):
        np.testing.assert_allclose(getattr(fast, name), getattr(general, name), rtol=1e-14, atol=1e-15)


def test_230_orbital_gap_scan_uses_bounded_batches_with_centered_spectra(monkeypatch):
    from queer.finite_field.geometry.kubo import band_gap

    n = 230
    R = np.array([(a, b, 0) for a in range(-1, 2) for b in range(-1, 2)])
    H = np.zeros((len(R), n, n), complex)
    H[4] = np.diag(5 + np.linspace(-2, 2, n))
    H[1, 144, 145] = 0.002 + 0.001j
    H[7] = H[1].conj().T
    model = WannierModel(R, H, lattice=np.diag([4, 4, 20]))
    view = NativePlaneModel(model, occupied_bands=145, plane={"origin": [0.03, -0.02, 0]})
    s = np.linspace(-0.06, 0.06, 17)
    t = 0.003
    reference = []
    for point in view.plane.fractional(model, s, t):
        matrix = model.hamiltonian(point)
        centered = matrix - np.trace(matrix).real / n * np.eye(n)
        energy = np.linalg.eigvalsh(centered)
        reference.append(energy[145] - energy[144])
    calls = []
    original = model._assemble_points

    def record(points, *args, **kwargs):
        calls.append(len(points))
        return original(points, *args, **kwargs)

    monkeypatch.setattr(model, "_assemble_points", record)
    budget = 20 * 1024**2
    with view.on_line(0.0, t, mem_bytes=budget):
        result = band_gap(view, s, np.full_like(s, t), mem_bytes=budget)
    limit = int((budget * 2 / 3) // (16 * n**2 * 6))
    assert max(calls) <= limit and sum(calls) == len(s)
    assert len(calls) == int(np.ceil(len(s) / limit))
    assert model._line_cache
    assert model._line_cache_bytes <= budget / 3
    np.testing.assert_allclose(result, reference, rtol=2e-13, atol=2e-14)


def test_small_provider_budget_falls_back_without_oversized_line_tables():
    from queer.finite_field.geometry.provider import ModelGeometry

    model = _model(64)
    view = NativePlaneModel(model, occupied_bands=32)
    provider = ModelGeometry(view)
    provider.mem_bytes = 3 * 1024**2
    s = np.linspace(-0.02, 0.02, 7)
    expected_gap = view.gap(s, 0.003)
    expected_geometry = view.geometry(s, 0.003)
    got_gap = provider.gap(s, 0.003)
    got_geometry = provider.geometry(s, 0.003)
    assert model._line_cache_bytes <= provider.mem_bytes / 3
    assert not model._line_cache  # One coefficient table exceeds the reserved share.
    np.testing.assert_allclose(got_gap, expected_gap, rtol=1e-13)
    for name in ("omega_v", "omega_c", "g_cv", "g_v_total", "g_c_total"):
        np.testing.assert_allclose(
            getattr(got_geometry, name), getattr(expected_geometry, name), rtol=1e-12, atol=1e-12
        )


def _worker_arrays(view):
    return (
        isinstance(view.H_R, np.memmap),
        isinstance(view.source_model.hopping_matrices, np.memmap),
        view.H_R.flags.writeable,
        view.source_model.raw_data is None,
    )


def test_light_worker_view_shares_operators_without_raw_inputs():
    from joblib import delayed

    model = _model(64)
    model.raw_data = SimpleNamespace(hopping=model.hopping_matrices)
    view = NativePlaneModel(model, occupied_bands=32)
    light = view.light_worker_view()
    assert light.H_R is view.H_R
    assert light.source_model.hopping_matrices is model.hopping_matrices
    assert model.raw_data is not None and light.source_model.raw_data is None
    np.testing.assert_array_equal(light.layer_projector(), view.layer_projector())
    with execution_session(ExecutionOptions(workers=2)) as session:
        rows = list(session.parallel()(delayed(_worker_arrays)(light) for _ in range(2)))
    assert rows == [(True, True, False, True)] * 2
