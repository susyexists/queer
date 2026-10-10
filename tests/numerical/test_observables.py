import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import trapezoid

from queer.models.wannier import WannierModel
from queer.observables.arpes import arpes_equation, arpes_mesh, arpes_path
from queer.observables.momentum import fermi_surface_points, iter_binding_energy_surfaces, select_surface
from queer.observables.response import green_function, spectral_from_eig, spectral_from_green, susceptibility
from queer.observables.spectral import spectral_along_path, spectral_arpes_path_sweep
from queer.utils import fermi_bands, path_create, ph_cross, untangle


def analytic_model():
    return WannierModel(
        [[0, 0, 0], [1, 0, 0], [-1, 0, 0]], np.array([[[0]], [[0.5]], [[0.5]]]), reciprocal=np.eye(3)
    )


def test_retarded_green_matches_analytic_lorentzian_and_integral():
    hamiltonian = np.array([[0.2, 0.3j], [-0.3j, -0.2]])
    eigenvalues = np.array([-np.sqrt(0.13), np.sqrt(0.13)])
    frequencies = np.linspace(-10, 10, 20001)
    eta = 0.03
    np.testing.assert_allclose(
        spectral_from_green(hamiltonian, frequencies[::100], eta),
        spectral_from_eig(eigenvalues, frequencies[::100], eta),
        rtol=1e-12,
        atol=1e-12,
    )
    result = spectral_from_eig(eigenvalues, frequencies, eta)
    assert result.min() >= 0
    assert trapezoid(result, frequencies) == pytest.approx(2.0, abs=0.004)
    assert green_function(hamiltonian, np.zeros((2, 3)), eta).shape == (2, 3, 2, 2)
    assert np.ndim(spectral_from_eig(eigenvalues, 0.1, eta)) == 0
    assert spectral_from_eig(eigenvalues, [], eta).shape == (0,)
    for invalid in (0, -0.1, np.nan):
        with pytest.raises(ValueError, match="eta"):
            spectral_from_eig(eigenvalues, [0], invalid)


def test_spectral_path_empty_and_scalar_axes():
    model = analytic_model()
    path = np.array([[0.0, 0.0, 0.0], [0.25, 0, 0]])
    expected = np.array(
        [
            [0.1 / np.pi / ((omega - np.cos(2 * np.pi * k)) ** 2 + 0.1**2) for k in path[:, 0]]
            for omega in [-1.0, 0, 1.0]
        ]
    )
    np.testing.assert_allclose(
        spectral_along_path(model, path, [-1.0, 0, 1.0]), expected, atol=1e-12, rtol=1e-12
    )
    assert spectral_along_path(model, path, 0).shape == (1, 2)
    assert spectral_along_path(model, path, []).shape == (0, 2)
    assert spectral_along_path(model, np.empty((0, 3)), [0]).shape == (1, 0)


def test_arpes_invalid_domain_consistent_masks_and_spectral_gaps():
    assert np.isnan(arpes_equation(1, 0, [1.0, 0]))
    assert arpes_equation(1, 0, [0.0, 0]) == pytest.approx(np.sqrt(0.2625))
    path = arpes_path(np.array([[0.0, 0.0, 0.0], [10, 0, 0]]), np.eye(3), Ek=1, V0=0)
    assert np.isfinite(path[0]).all() and np.isnan(path[1]).all()
    spectral = spectral_along_path(analytic_model(), path, [0])
    assert np.isfinite(spectral[:, 0]).all() and np.isnan(spectral[:, 1]).all()
    invalid_mesh = arpes_mesh(1, 0, 5, 0, 4)
    assert np.isnan(invalid_mesh[:, 2]).all() and np.isfinite(invalid_mesh[:, :2]).all()
    sweep = spectral_arpes_path_sweep(
        analytic_model(), np.empty((0, 3)), np.eye(3), 1, [0], [], align_zero=False
    )
    assert sweep[0].shape == (0, 0)


def test_susceptibility_independent_one_band_reference_and_mesh_reuse():
    model = analytic_model()
    mesh = np.column_stack((np.arange(12) / 12, np.zeros((12, 2))))
    q = np.array([[0.125, 0, 0]])
    temperature, eta = 0.05, 0.01
    initial = np.cos(2 * np.pi * mesh[:, 0])
    final = np.cos(2 * np.pi * (mesh[:, 0] + 0.125))

    def occupation(energy):
        return 1 / (np.exp(energy / temperature) + 1)

    expected = -np.mean((occupation(initial) - occupation(final)) / (initial - final + 1j * eta))
    actual = susceptibility(
        model, q, mesh, temperature=temperature, broadening=eta, mesh_energies=initial[None, :]
    )
    np.testing.assert_allclose(actual, [expected], atol=1e-12, rtol=1e-12)
    assert susceptibility(model, np.empty((0, 3)), mesh).shape == (0,)


def test_numpy_surface_matches_dataframe_grouping_and_ties():
    pd = pytest.importorskip("pandas")
    mesh = np.array([[0.0, 0.0, 2.0], [0, 0, 3], [1, 1, 4]])
    bands = np.array([[-0.2, -0.2, 0.1], [0.2, -0.1, -0.1]])
    intensity = np.exp(-(bands.ravel() ** 2) / (2 * 0.1**2))
    table = pd.DataFrame(
        np.column_stack((np.tile(mesh, (2, 1)), bands.ravel(), intensity)),
        columns=["kx", "ky", "kz", "e", "I"],
    )
    expected = table.loc[table.groupby(["kx", "ky"])["I"].idxmax()].to_numpy()
    np.testing.assert_allclose(select_surface(mesh, bands, 0), expected, atol=1e-12, rtol=1e-12)
    # Third point's +.1/-.1 tie chooses the earlier (first) band.
    assert select_surface(mesh, bands, 0)[1, 3] == 0.1
    assert select_surface(np.empty((0, 3)), np.empty((2, 0)), 0).shape == (0, 5)


def test_momentum_stream_and_legacy_dataframe_compatibility():
    from queer.momentum import binding_energy_surfaces

    arguments = (analytic_model(), np.eye(3), 21.2, 0.0, [0.0, 0.2], 0.1, 10, 4, 2)
    stream = iter_binding_energy_surfaces(*arguments)
    energy, first = next(stream)
    assert energy == 0.0 and first.shape[1] == 5
    frames = binding_energy_surfaces(*arguments)
    np.testing.assert_allclose(frames[0.0].to_numpy(), first, atol=1e-12, rtol=1e-12)
    empty = list(iter_binding_energy_surfaces(analytic_model(), np.eye(3), 1, 0, [5.0, 6.0], 1, 0, 4))
    assert empty[0][1].shape == (0, 5)
    mask, x, y = fermi_surface_points(np.full((2, 3), np.nan), analytic_model(), np.eye(3))
    assert mask.size == x.size == y.size == 0


def test_legacy_helpers_correct_windows_and_integer_path_endpoints():
    assert fermi_bands([[0.4], [0.05]], window=0.1) == {1}
    ticks, points = path_create(6, [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    np.testing.assert_array_equal(points[-1], [3, 0, 0])
    assert points.shape == (7, 3)
    ph = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    assert ph_cross(ph, 0.1, offset=0) == [[0, 1, 1]]
    result = untangle(ph, [[0, 1, 1]])
    np.testing.assert_array_equal(result[:, 0], [3, 1])
    np.testing.assert_array_equal(ph[:, 0], [1, 3])


def test_numeric_imports_do_not_require_pandas_or_plotting():
    code = """
import importlib.abc, sys
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pandas','matplotlib','PIL','imageio'}:
            raise ImportError('blocked optional package ' + fullname)
sys.meta_path.insert(0, Blocker())
import queer
from queer.models.epw import EPWModel
from queer.observables import arpes,momentum,response,spectral
from queer import utils
assert not any(name.startswith(('pandas','matplotlib')) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parents[1])


def test_invalid_infinities_rejected_but_nan_spectral_gaps_preserved():
    with pytest.raises(ValueError, match="infinities"):
        spectral_from_eig([np.inf], [0], 0.1)
    assert np.isnan(spectral_from_eig([np.nan], [0], 0.1)[0])
    with pytest.raises(ValueError, match="infinities"):
        spectral_along_path(analytic_model(), [[np.inf, 0, 0]], [0])
    with pytest.raises(ValueError, match="finite"):
        select_surface([[0, 0, 1]], [[np.nan]], 0)
    with pytest.raises(ValueError, match="finite"):
        arpes_equation(np.inf, 0, [0, 0])
    gapped = arpes_path([[np.nan, 0, 0], [0, 0, 0]], np.eye(3), Ek=1, V0=0)
    assert np.isnan(gapped[0]).all() and np.isfinite(gapped[1]).all()


def test_signed_arpes_frequency_geometry_and_streaming_reference_reuse():
    from queer.observables.spectral import iter_spectral_arpes_path_sweep

    calls = []

    class CapturedModel:
        def energies(self, points, execution=None):
            calls.append(points.copy())
            return np.full((1, len(points)), -0.2)

    model = CapturedModel()
    stream = iter_spectral_arpes_path_sweep(
        model, [[0, 0, 0]], np.eye(3), 21.2, [10], [-1, -2], align_zero=True
    )
    _, index, first = next(stream)
    assert index == 0 and len(calls) == 2
    # Independent free-electron k_z expression: kinetic energy decreases at omega<0.
    assert calls[1][0, 2] == pytest.approx(np.sqrt(0.2625 * (21.2 - 1 + 10)))
    _, index, second = next(stream)
    assert index == 1 and len(calls) == 3  # one reference, two frequency points
    assert calls[2][0, 2] == pytest.approx(np.sqrt(0.2625 * (21.2 - 2 + 10)))
    with pytest.raises(StopIteration):
        next(stream)
    collected = spectral_arpes_path_sweep(model, [[0, 0, 0]], np.eye(3), 21.2, [10], [-1, -2])
    np.testing.assert_allclose(collected[10], np.vstack((first, second)), atol=1e-12, rtol=1e-12)


def test_momentum_sweep_reuses_each_geometry_across_shifts_without_eager_cache():
    from queer.momentum import sweep_binding_energy_surfaces as legacy_sweep
    from queer.observables.momentum import binding_energy_surfaces, iter_surface_sweep

    class CountedModel:
        def __init__(self):
            self.model = analytic_model()
            self.calls = 0

        def energies(self, points, execution=None):
            self.calls += 1
            return self.model.energies(points, execution=execution)

    model = CountedModel()
    shifts = [0.0, 0.1, -0.1]
    values = [5.0, 12.0]
    arguments = (model, np.eye(3), 21.2, 0.0, [0.0, 0.2], 0.1, values, 4)
    stream = iter_surface_sweep(*arguments, factor=1, energy_shifts=shifts)
    first_geometry = [next(stream) for _ in shifts]
    assert model.calls == 1
    assert [key for key, _ in first_geometry] == [(5.0, shift, 0.0) for shift in shifts]
    actual = dict([*first_geometry, *stream])
    assert model.calls == 4  # two V0 geometries times two binding energies
    assert len(actual) == 12
    for v0 in values:
        for shift in shifts:
            expected = binding_energy_surfaces(
                analytic_model(),
                np.eye(3),
                21.2,
                0.0,
                [0.0, 0.2],
                0.1,
                v0,
                4,
                factor=1,
                energy_shift=shift,
            )
            for energy, surface in expected.items():
                np.testing.assert_allclose(actual[v0, shift, energy], surface, atol=1e-12, rtol=1e-12)
    facade_model = CountedModel()
    collected = legacy_sweep(
        facade_model,
        np.eye(3),
        21.2,
        0.0,
        [0.0, 0.2],
        0.1,
        values,
        4,
        factor=1,
        energy_shifts=shifts,
        verbose=False,
    )
    assert facade_model.calls == 4
    for (v0, shift), surfaces in collected.items():
        for energy, frame in surfaces.items():
            np.testing.assert_allclose(frame.to_numpy(), actual[v0, shift, energy], atol=1e-12, rtol=1e-12)


def test_arpes_masks_preserve_boundaries_input_rows_and_rotations():
    kinetic = 1 / 0.2625  # hemisphere radius is exactly one inverse angstrom
    path = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [np.nextafter(1.0, np.inf), 0.0, 0.0], [np.nan, 0.0, 0.0]]
    )
    curved, valid = arpes_path(path, np.eye(3), Ek=kinetic, V0=0, return_mask=True)
    np.testing.assert_array_equal(valid, [True, True, False, False])
    assert valid.shape == (len(path),) and valid.dtype == bool
    assert curved[1, 2] == 0.0  # hemisphere boundary has a real solution
    np.testing.assert_allclose(
        arpes_path(path, np.eye(3), Ek=kinetic, V0=0), curved, equal_nan=True, atol=1e-12, rtol=1e-12
    )
    mesh, mask = arpes_mesh(kinetic, 0, 0, 0, 3, factor=3, return_mask=True)
    assert mask.sum() == 5  # origin plus four points on the boundary; four corners excluded
    rotation = [[0, 0, 1], [1, 1, 1]]
    rotated, rotated_mask = arpes_mesh(kinetic, 0, 0, 0, 3, factor=3, align=rotation, return_mask=True)
    np.testing.assert_array_equal(mask, rotated_mask)
    np.testing.assert_array_equal(rotated_mask, np.isfinite(rotated).all(axis=1))
    from queer.geometry.rotations import rotation_matrix

    np.testing.assert_allclose(
        rotated[mask], mesh[mask] @ rotation_matrix(*rotation).T, atol=1e-12, rtol=1e-12
    )
    empty, empty_mask = arpes_path(np.empty((0, 3)), np.eye(3), return_mask=True)
    assert empty.shape == (0, 3) and empty_mask.shape == (0,)
    from queer.arpes import arpes_mesh as legacy_mesh

    legacy_points, legacy_mask = legacy_mesh(kinetic, 0, 0, 0, 3, factor=3, return_mask=True)
    np.testing.assert_array_equal(legacy_mask, mask)
    np.testing.assert_allclose(legacy_points, mesh, equal_nan=True, atol=1e-12, rtol=1e-12)
