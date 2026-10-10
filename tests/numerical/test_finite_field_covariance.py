"""Independent precision and wavefunction-covariance checks for native adapters."""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from queer.finite_field import PreparedHallResponse
from queer.finite_field.family import ModelFamily
from queer.finite_field.geometry.gridded import GriddedGeometry
from queer.finite_field.geometry.lowdin import LowdinPatch
from queer.finite_field.geometry.provider import ModelGeometry
from queer.finite_field.geometry.wilson import geometric_phase, geometric_phase_refined, shift_vector_refined
from queer.finite_field.models.fourband import FourBandParams
from queer.finite_field.models.lattice_reg import lattice_regularized_fourband
from queer.finite_field.models.toy import LinearLZ
from queer.finite_field.native import NativePlaneModel
from queer.models.openmx import OpenMXModel
from queer.models.wannier import WannierModel

PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
TRANSLATIONS = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])


def test_native_spin_semantics_enter_model_and_preparation_fingerprints():
    h = [np.diag([-2.0, 1.0, -1.0, 2.0])]
    views = [
        NativePlaneModel(OpenMXModel([[0, 0, 0]], h, spin_mode=mode), band_pair=(0, 1))
        for mode in ("collinear", "soc")
    ]
    assert [float(view.gap(0, 0)) for view in views] == [3, 1]
    assert views[0].fingerprint() != views[1].fingerprint()
    prepared = [PreparedHallResponse(view, "physical_eq8") for view in views]
    assert prepared[0].signature() != prepared[1].signature()
    assert views[0].describe()["spin_mode"] == "collinear"
    assert views[1].describe()["spin_degeneracy"] == 1


@pytest.mark.parametrize("value", [True, 2.0, 3])
def test_native_spin_multiplicity_rejects_ambiguous_metadata(value):
    model = phase_model()
    model.spin_degeneracy = value
    with pytest.raises(ValueError, match="spin_degeneracy"):
        NativePlaneModel(model, band_pair=(0, 1))


def test_collinear_independent_gap_keeps_small_generalized_splitting_with_offset():
    h = np.diag([5.0, 10.0, 2.0, 6.0])
    h[0, 1] = h[1, 0] = np.sqrt(2) * 1e-12
    source = OpenMXModel([[0, 0, 0]], [h], [np.diag([1.0, 2.0, 1.0, 2.0])], spin_mode="collinear")
    view = NativePlaneModel(source, band_pair=(0, 1))
    assert view.gap(0, 0) == pytest.approx(2e-12, rel=1e-13)
    absolute = source.eigenstates([[0, 0, 0]]).energies[:2, 0]
    assert abs((absolute[1] - absolute[0]) / 2e-12 - 1) > 1e-5


def test_tdse_chunk_budgets_preserve_signed_unitary_and_relaxed_populations(monkeypatch):
    from queer.finite_field.tdse import houston

    model = LinearLZ(delta=0.001, alpha=0.2)
    fields = np.array([1e7, -1e7, 2e7])
    baseline = houston.houston_line(model, 0.0, fields, -0.01, 0.01)
    damped = houston.lindblad_two_level(model, 0.0, fields, -0.01, 0.01, 2e-12, 2e-12)
    widths = []
    original = houston._magnus_eig

    def record(model, t, angle, s_lo, h):
        widths.append(len(h))
        return original(model, t, angle, s_lo, h)

    monkeypatch.setattr(houston, "_magnus_eig", record)
    restricted = houston.houston_line(model, 0.0, fields, -0.01, 0.01, chunk_bytes=1152)
    restricted_damped = houston.lindblad_two_level(
        model, 0.0, fields, -0.01, 0.01, 2e-12, 2e-12, chunk_bytes=4608
    )
    assert widths and max(widths) == 1
    assert_allclose(restricted.T, baseline.T, atol=2e-14, rtol=0)
    assert_allclose(restricted_damped, damped, atol=2e-14, rtol=0)
    assert restricted.info["chunk_bytes"] == 1152


@pytest.mark.parametrize("budget", [0, -1, True, float("inf"), float("nan"), 1])
def test_tdse_rejects_invalid_or_exhausted_chunk_budget(budget):
    from queer.finite_field.tdse.houston import houston_line, lindblad_two_level

    model = LinearLZ(delta=0.001, alpha=0.2)
    with pytest.raises(ValueError, match="chunk_bytes"):
        houston_line(model, 0.0, [1e7], -0.01, 0.01, chunk_bytes=budget)
    with pytest.raises(ValueError, match="chunk_bytes"):
        lindblad_two_level(model, 0.0, [1e7], -0.01, 0.01, 2e-12, 2e-12, chunk_bytes=budget)


def critical_dirac(offset=0.0):
    x, y, z = PAULI
    a, b = 1.1, 0.25
    hopping = np.array(
        [
            offset * np.eye(2) + 2 * b * z,
            -b * z / 2 - 0.5j * a * x,
            -b * z / 2 + 0.5j * a * x,
            -b * z / 2 - 0.5j * a * y,
            -b * z / 2 + 0.5j * a * y,
        ]
    )
    return WannierModel(TRANSLATIONS, hopping, lattice=np.eye(3))


def mp_geometry(model, points, pair, angle=0.0):
    """Direct 80-digit Fourier/eigensystem/covariant geometry, independent of patches."""
    mp = pytest.importorskip("mpmath")
    n = model.nbnd
    vectors = model.translations @ model.lattice
    output = {name: [] for name in ("gap", "omega_v", "omega_c", "g_cv", "g_v_total", "g_c_total")}
    with mp.workdps(80):
        ca, sa = mp.cos(mp.mpf(float(angle))), mp.sin(mp.mpf(float(angle)))
        for kx, ky in points:
            h = mp.matrix(n)
            dh = [mp.matrix(n), mp.matrix(n)]
            pos = [mp.matrix(n), mp.matrix(n)]
            dp = [[mp.matrix(n), mp.matrix(n)] for _ in range(2)]
            for index, vector in enumerate(vectors):
                d = [mp.mpf(float(value)) for value in vector[:2]]
                phase = mp.expj(mp.mpf(float(kx)) * d[0] + mp.mpf(float(ky)) * d[1])
                for i, j in np.ndindex(n, n):
                    value = mp.mpc(complex(model.hopping_matrices[index, i, j])) * phase
                    h[i, j] += value
                    for axis in range(2):
                        dh[axis][i, j] += 1j * d[axis] * value
                        if model.position_matrices is not None:
                            p = mp.mpc(complex(model.position_matrices[index, i, j, axis])) * phase
                            pos[axis][i, j] += p
                            for derivative in range(2):
                                dp[derivative][axis][i, j] += 1j * d[derivative] * p
            if model.position_matrices is None and model.centres is not None:
                for i in range(n):
                    for axis in range(2):
                        pos[axis][i, i] = mp.mpf(float(model.centres[i, axis]))
            energies, u = mp.eighe(h)
            covariant = [dh[a] - 1j * (pos[a] * h - h * pos[a]) for a in range(2)]
            xs = u.H * (ca * covariant[0] + sa * covariant[1]) * u
            xt = u.H * (-sa * covariant[0] + ca * covariant[1]) * u
            f = dp[0][1] - dp[1][0] - 1j * (pos[0] * pos[1] - pos[1] * pos[0])
            f = u.H * f * u
            for name, band in zip(("v", "c"), pair, strict=True):
                omega = mp.re(f[band, band])
                metric = [mp.mpf(0), mp.mpf(0), mp.mpf(0)]
                for other in range(n):
                    if other == band:
                        continue
                    gap2 = (energies[band] - energies[other]) ** 2
                    cross = xs[band, other] * mp.conj(xt[band, other])
                    omega -= 2 * mp.im(cross) / gap2
                    metric[0] += abs(xs[band, other]) ** 2 / gap2
                    metric[1] += mp.re(cross) / gap2
                    metric[2] += abs(xt[band, other]) ** 2 / gap2
                output[f"omega_{name}"].append(float(omega))
                output[f"g_{name}_total"].append([float(value) for value in metric])
            v, c = pair
            gap2 = (energies[c] - energies[v]) ** 2
            cross = xs[v, c] * mp.conj(xt[v, c])
            output["g_cv"].append(
                [
                    float(abs(xs[v, c]) ** 2 / gap2),
                    float(mp.re(cross) / gap2),
                    float(abs(xt[v, c]) ** 2 / gap2),
                ]
            )
            output["gap"].append(float(energies[c] - energies[v]))
    return {name: np.asarray(values) for name, values in output.items()}


@pytest.mark.parametrize("offset", [0.0, 5.0])
@pytest.mark.precision
def test_native_lowdin_critical_geometry_matches_independent_mpmath(offset):
    model = critical_dirac(offset)
    view = NativePlaneModel(model, band_pair=(0, 1))
    patch = LowdinPatch(view, k_max=1e-6)
    assert patch.exact_pair and patch.snapped
    assert patch.E_ref == pytest.approx(offset)
    k = np.geomspace(1e-12, 1e-6, 4)
    points = np.column_stack([k, 0 * k])
    reference = mp_geometry(model, points, (0, 1))
    actual = patch.evaluate(points[:, 0], points[:, 1])
    assert_allclose(actual.eps_g, reference["gap"], rtol=2e-12)
    for name in ("omega_v", "omega_c", "g_cv", "g_v_total", "g_c_total"):
        assert_allclose(getattr(actual, name), reference[name], rtol=2e-10, atol=2e-11)
    provider = ModelGeometry(view, k_patch=2e-6)
    served = provider.geometry(points[:, 0], points[:, 1])
    assert served.extras["patched"].all()
    assert_allclose(served.eps_g, reference["gap"], rtol=2e-12)
    assert_allclose(served.g_cv[:, 0], reference["g_cv"][:, 0], rtol=2e-10)


@pytest.mark.precision
def test_native_fourband_lowdin_matches_mpmath_with_offset():
    original = lattice_regularized_fourband(FourBandParams().at(1 + 1e-6))
    hopping = original.hopping_matrices.copy()
    hopping[np.flatnonzero(np.all(original.translations == 0, axis=1))[0]] += 5 * np.eye(4)
    model = WannierModel(original.translations, hopping, lattice=original.lattice, centres=original.centres)
    view = NativePlaneModel(model, band_pair=(1, 2))
    patch = LowdinPatch(view, k_max=1e-6)
    assert not patch.snapped
    assert patch.exact_pair
    k = np.geomspace(1e-12, 1e-6, 4)
    points = np.column_stack([k * np.cos(0.7), k * np.sin(0.7)])
    reference = mp_geometry(model, points, (1, 2), angle=0.3)
    actual = patch.evaluate(points[:, 0], points[:, 1], 0.3)
    assert_allclose(actual.eps_g, reference["gap"], rtol=1e-10)
    for name in ("omega_v", "omega_c", "g_cv", "g_v_total", "g_c_total"):
        expected = reference[name]
        scale = abs(expected) if expected.ndim == 1 else np.max(abs(expected), axis=-1, keepdims=True)
        assert np.max(abs(getattr(actual, name) - expected) / scale) < 1e-9


@pytest.mark.precision
def test_skew_cell_rotated_plane_lowdin_matches_mpmath():
    base = critical_dirac(5.0)
    lattice = np.array([[2, 0.3, 0], [0.4, 1.7, 0], [0, 0, 4]])
    model = WannierModel(base.translations, base.hopping_matrices, lattice=lattice)
    angle = 0.4
    direction = [np.cos(angle), np.sin(angle), 0]
    transverse = [-np.sin(angle), np.cos(angle), 0]
    view = NativePlaneModel(model, band_pair=(0, 1), plane={"direction": direction, "transverse": transverse})
    patch = LowdinPatch(view, k_max=1e-6)
    k = np.geomspace(1e-12, 1e-6, 4)
    points = np.column_stack([k * np.cos(angle), k * np.sin(angle)])
    reference = mp_geometry(model, points, (0, 1), angle=angle)
    actual = patch.evaluate(k, 0 * k)
    assert_allclose(actual.eps_g, reference["gap"], rtol=2e-12)
    for name in ("omega_v", "omega_c"):
        assert_allclose(getattr(actual, name), reference[name], rtol=2e-9, atol=1e-9)
    for name in ("g_cv", "g_v_total", "g_c_total"):
        expected = reference[name]
        scale = np.max(abs(expected), axis=-1, keepdims=True)
        assert np.max(abs(getattr(actual, name) - expected) / scale) < 2e-9
        # Preserve the small radial metric; a float-rounded Cartesian rotation
        # perturbs the mixed element of the singular tensor by O(eps/k^2).
        assert_allclose(getattr(actual, name)[:, 0], expected[:, 0], rtol=2e-9, atol=1e-9)


def offset_dirac_grid(offset=5.0, scale=1e-12):
    """A Dirac reference whose interpolation surrogates are exact polynomials."""
    coordinates = np.linspace(-2 * scale, 2 * scale, 5)
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    mass = scale
    energy = np.sqrt(mass**2 + x**2 + y**2)
    metric = np.stack([mass**2 + y**2, -x * y, mass**2 + x**2], axis=-1) / (4 * energy[..., None] ** 4)
    curvature = mass / (2 * energy**3)
    return GriddedGeometry(
        {
            "kx": coordinates,
            "ky": coordinates,
            "eps_v": offset - energy,
            "eps_c": offset + energy,
            "gap": 2 * energy,
            "omega_v": curvature,
            "omega_c": -curvature,
            "g_cv": metric,
            "g_v_total": metric,
            "g_c_total": metric,
        }
    )


def test_archive_preserves_independent_gap_at_large_energy_offset(tmp_path):
    grid = offset_dirac_grid()
    path = tmp_path / "geometry.npz"
    grid.save(path)
    restored = GriddedGeometry.from_npz(path)
    x, y = np.array([0.3e-12, -1.2e-12]), np.array([-0.7e-12, 0.8e-12])
    expected_gap = 2 * np.sqrt(1e-24 + x**2 + y**2)
    sampled = restored.at_cartesian(x, y)
    assert_allclose(sampled.eps_g, expected_gap, rtol=2e-14)
    assert np.max(abs((sampled.eps_c - sampled.eps_v) / expected_gap - 1)) > 1e-5
    assert_allclose(sampled.omega_c, -4e-12 / expected_gap**3, rtol=2e-13)
    outside = restored.at_cartesian([3e-12], [0])
    assert outside.extras["outside_grid"].tolist() == [True]
    assert restored.fingerprint() == grid.fingerprint()


def test_grid_only_family_remains_usable_without_hamiltonians(tmp_path):
    grid_path = tmp_path / "geometry.npz"
    offset_dirac_grid().save(grid_path)
    manifest = tmp_path / "family.json"
    manifest.write_text(
        json.dumps({"members": [{"label": "grid", "D": 1.0, "files": {"grid": grid_path.name}}]})
    )
    family = ModelFamily.from_manifest(manifest)
    assert family.gamma_data(0) is None
    source = family.source(0, angle=0.2)
    assert isinstance(source, GriddedGeometry)
    assert source.angle == 0.2
    assert np.isfinite(source.gap([0.3e-12], [0.7e-12])).all()
    assert set(family.input_files()) == {manifest, grid_path}
    with pytest.raises(ValueError, match="no member has a Hamiltonian"):
        family.critical_D_report()


def test_family_fingerprints_auto_sidecars_without_loading_matrices(tmp_path, monkeypatch):
    hr = tmp_path / "seed_hr.dat"
    hr.write_text("synthetic\n2\n1\n1\n0 0 0 1 1 -1 0\n0 0 0 1 2 .3 0\n0 0 0 2 1 .3 0\n0 0 0 2 2 1 0\n")
    centres = tmp_path / "seed_centres.xyz"
    centres.write_text("2\ncentres\nX .2 .3 0\nX .7 -.2 0\n")
    wout = tmp_path / "seed.wout"
    wout.write_text("Lattice Vectors (Ang)\na_1 1 0 0\na_2 0 1 0\na_3 0 0 1\n")
    manifest = tmp_path / "family.json"
    manifest.write_text(
        json.dumps({"n_occ": 1, "members": [{"label": "member", "D": 0.0, "files": {"hr": hr.name}}]})
    )
    family = ModelFamily.from_manifest(manifest)
    loader = family.members[0].loader
    first = family.model(0)
    first_hash = loader.fingerprint()
    assert set(family.input_files()) == {manifest, hr, centres, wout}
    assert first.source_model.source_files == {
        key: str(path) for key, path in loader.effective_files().items()
    }
    with monkeypatch.context() as patch:
        patch.setattr(
            WannierModel,
            "from_directory",
            lambda *args, **kwargs: pytest.fail("fingerprinting loaded operators"),
        )
        assert loader.fingerprint() == first_hash
    centres.write_text("2\ncentres\nX .6 .3 0\nX .7 -.2 0\n")
    assert loader.fingerprint() != first_hash
    second = family.model(0)
    assert second is not first
    assert second.source_model.centres[0, 0] == 0.6


def test_physical_cell_changes_invalidate_native_preparation_identity():
    first = critical_dirac()
    second = WannierModel(first.translations, first.hopping_matrices, lattice=np.diag([2, 3, 4]))
    views = [NativePlaneModel(model, band_pair=(0, 1)) for model in (first, second)]
    assert views[0].fingerprint() != views[1].fingerprint()
    preparations = [PreparedHallResponse(view, options="physical_eq8") for view in views]
    assert preparations[0].signature() != preparations[1].signature()


def phase_model():
    x, y, z = PAULI
    hopping = np.array([-z, (z - 1j * x) / 2, (z + 1j * x) / 2, (z - 1j * y) / 2, (z + 1j * y) / 2])
    return WannierModel(TRANSLATIONS, hopping, lattice=np.eye(3), centres=[[0.2, 0.3, 0], [0.7, -0.2, 0]])


def test_native_geometric_phase_is_independent_of_home_cells():
    model = phase_model()
    shifts = np.array([[1, 0, 0], [0, -1, 0]])
    blocks = {}
    for translation, matrix in zip(model.translations, model.hopping_matrices, strict=True):
        for i, j in np.ndindex(2, 2):
            key = tuple(translation + shifts[i] - shifts[j])
            blocks.setdefault(key, np.zeros((2, 2), complex))[i, j] += matrix[i, j]
    keys = sorted(blocks)
    changed = WannierModel(
        keys, [blocks[key] for key in keys], lattice=model.lattice, centres=model.centres + shifts
    )
    views = [NativePlaneModel(m, band_pair=(0, 1)) for m in (model, changed)]
    phases = [geometric_phase(view, -0.8, 0.7, 0.17, n=101) for view in views]
    assert_allclose(np.exp(1j * phases[0]), np.exp(1j * phases[1]), atol=2e-12)
    reverse = geometric_phase(views[0], 0.7, -0.8, 0.17, n=101)
    assert_allclose(np.exp(1j * reverse), np.exp(-1j * phases[0]), atol=2e-12)


def test_native_geometric_phase_is_invariant_to_independent_solver_gauges(monkeypatch):
    model = phase_model()
    view = NativePlaneModel(model, band_pair=(0, 1))
    expected = geometric_phase(view, -0.8, 0.7, 0.17, n=51)
    original = model.eigenstates
    rng = np.random.default_rng(172)

    def randomized(*args, **kwargs):
        states = original(*args, **kwargs)
        phases = np.exp(1j * rng.uniform(-np.pi, np.pi, size=states.energies.T.shape))
        states.coefficients *= phases[:, None, :]
        return states

    monkeypatch.setattr(model, "eigenstates", randomized)
    actual = geometric_phase(view, -0.8, 0.7, 0.17, n=51)
    assert_allclose(np.exp(1j * actual), np.exp(1j * expected), atol=2e-12)


def test_native_phase_refinement_converges_against_independent_fine_links():
    view = NativePlaneModel(phase_model(), band_pair=(0, 1), plane={"origin": [0.03, -0.02, 0]})
    phase, diagnostics = geometric_phase_refined(view, -0.8, 0.7, 0.17, n=9, n_max=1025, tolerance=1e-6)
    reference = geometric_phase(view, -0.8, 0.7, 0.17, n=4097)
    assert diagnostics["converged"]
    assert diagnostics["error"] <= 1e-6
    assert abs(np.angle(np.exp(1j * (phase - reference)))) < 1e-6
    reverse, reversed_diagnostics = geometric_phase_refined(
        view, 0.7, -0.8, 0.17, n=9, n_max=1025, tolerance=1e-6
    )
    assert reversed_diagnostics["converged"]
    assert_allclose(np.exp(1j * reverse), np.exp(-1j * phase), atol=2e-12)


def test_phase_and_shift_refinement_report_exhausted_budgets():
    view = NativePlaneModel(phase_model(), band_pair=(0, 1))
    phase, diagnostics = geometric_phase_refined(view, -0.8, 0.7, 0.17, n=3, n_max=5, tolerance=1e-14)
    assert np.isfinite(phase)
    assert not diagnostics["converged"] and diagnostics["nodes"] == 5
    assert diagnostics["error"] > 1e-14
    _, unresolved = geometric_phase_refined(view, -0.8, 0.7, 0.17, n=3, n_max=3)
    assert not unresolved["converged"] and unresolved["error"] is None
    _, shift_diagnostics = shift_vector_refined(view, 0.2, 0.17, h=0.1, tolerance=1e-14)
    assert not shift_diagnostics["converged"]
    shift, fine_diagnostics = shift_vector_refined(view, 0.2, 0.17, h=1e-4, tolerance=1e-5)
    reference, _ = shift_vector_refined(view, 0.2, 0.17, h=1e-5)
    assert fine_diagnostics["converged"]
    assert shift == pytest.approx(reference, abs=1e-6)
    json.dumps(diagnostics, allow_nan=False)


def test_phase_refinement_rejects_invalid_settings():
    view = NativePlaneModel(phase_model(), band_pair=(0, 1))
    for settings in ({"n": 2}, {"n": True}, {"n_max": 5}, {"tolerance": 0}, {"tolerance": np.nan}):
        with pytest.raises(ValueError):
            geometric_phase_refined(view, -0.8, 0.7, 0.17, **settings)
    for h in (0, -1, np.inf, np.nan):
        with pytest.raises(ValueError):
            shift_vector_refined(view, 0.2, 0.17, h=h)
