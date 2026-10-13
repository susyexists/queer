"""Flat native gap minima need band slopes rather than sub-roundoff gap differences."""

from types import SimpleNamespace

import numpy as np
import pytest

from queer.finite_field.geometry.provider import ModelGeometry
from queer.finite_field.lz.crossings import _polish_native_gap_minimum, gap_minima
from queer.finite_field.native import FieldPlane, NativePlaneModel
from queer.models.wannier import WannierModel

# Measured derivative root and bounded-HHH coordinate for the real 0.086 model.
ROOT = -1.6971454312e-5
HHH_COORDINATE = -1.6971426305e-5
GAP = 0.00086139980123
CURVATURE = 252.06597294
SX = np.array([[0, 1], [1, 0]], complex)
SZ = np.diag([1, -1])


def flat_model(*, energy_zero=0, rotation=False, angle=0, t=0, plane=None):
    plane = FieldPlane.from_mapping(plane)
    direction, transverse = plane.axes(angle)
    target = 2 * np.pi * plane.origin[0] + ROOT * direction[0] + t * transverse[0]
    speed = np.sqrt(GAP * CURVATURE / 4) / abs(direction[0])
    hopping = np.array(
        [
            0.5 * GAP * SZ + energy_zero * np.eye(2),
            -0.5j * speed * np.exp(-1j * target) * SX,
            0.5j * speed * np.exp(1j * target) * SX,
        ]
    )
    centres = np.array([[0.13, -0.29, 0], [-0.21, 0.18, 0]])
    if rotation:
        unitary = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
        hopping = unitary.conj().T @ hopping @ unitary
    model = WannierModel([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], hopping, lattice=np.eye(3), centres=centres)
    return NativePlaneModel(model, plane=plane, band_pair=(0, 1))


@pytest.mark.parametrize("energy_zero,rotation", [(0, False), (5, False), (5, True)])
def test_flat_noisy_minimum_polishes_covariantly_inside_same_basin(energy_zero, rotation):
    model = flat_model(energy_zero=energy_zero, rotation=rotation)

    class RoundedGap(ModelGeometry):
        def gap(self, s, t):
            # The observed real-model gap noise is 1e-14 eV. Reproduce its flat
            # objective while the analytic Hamiltonian retains its derivatives.
            return np.round(super().gap(s, t) / 2e-14) * 2e-14

    provider = RoundedGap(model)
    minima, flags = gap_minima(provider, 0, 0.001, scan_points=65)
    assert flags == set()
    assert len(minima) == 1
    assert minima[0] == pytest.approx(ROOT, abs=1e-14)
    # The existing replay tolerance remains unchanged; no acceptance widening.
    assert minima[0] == pytest.approx(HHH_COORDINATE, rel=1e-4, abs=1e-12)


def test_polish_uses_rotated_field_and_nonzero_native_plane_origin():
    angle, t = 0.27, 0.002
    phi = 0.41
    plane = FieldPlane(
        origin=(0.013, -0.021, 0.017),
        direction=(np.cos(phi), np.sin(phi), 0),
        transverse=(-np.sin(phi), np.cos(phi), 0),
    )
    model = flat_model(energy_zero=5, rotation=True, angle=angle, t=t, plane=plane)
    provider = ModelGeometry(model, angle=angle)
    minima, flags = gap_minima(provider, t, 0.001, scan_points=65)
    assert flags == set()
    assert len(minima) == 1
    assert minima[0] == pytest.approx(ROOT, abs=1e-14)


def test_lattice_fourband_minima_match_analytic_stationarity_with_energy_shift():
    from scipy.optimize import brentq

    from queer.finite_field.models.fourband import FourBandParams
    from queer.finite_field.models.lattice_reg import lattice_regularized_fourband

    p, a = FourBandParams().at(1.8), 4.33

    def slope(k):
        # Exact triangular-lattice substitution on ky=0, differentiated before
        # taking the small adiabatic gap. No eigenvalue subtraction is needed.
        kl = 2 / (3 * a) * (np.sin(a * k) + np.sin(a * k / 2))
        dkl = 2 / 3 * (np.cos(a * k) + 0.5 * np.cos(a * k / 2))
        k2 = 4 / (3 * a**2) * (3 - np.cos(a * k) - 2 * np.cos(a * k / 2))
        dk2 = 4 / (3 * a) * (np.sin(a * k) + np.sin(a * k / 2))
        mass = p.M - p.B * k2
        u, v = p.A**2 * kl**2, mass**2
        du, dv = 2 * p.A**2 * kl * dkl, -2 * p.B * mass * dk2
        root = np.sqrt(v * p.Bl**2 + p.VL**2 * (u + v))
        return du + dv - (dv * p.Bl**2 + p.VL**2 * (du + dv)) / root

    radius = brentq(slope, 1e-3, 1e-2, xtol=1e-15)
    source = lattice_regularized_fourband(p, a)
    shifted = source.hopping_matrices.copy()
    onsite = np.flatnonzero(np.all(source.translations == 0, axis=1))[0]
    shifted[onsite] += 5 * np.eye(4)
    for model in (
        source,
        WannierModel(source.translations, shifted, lattice=source.lattice, centres=source.centres),
    ):
        provider = ModelGeometry(NativePlaneModel(model, band_pair=(1, 2)))
        minima, flags = gap_minima(provider, 0, 0.015, scan_points=129)
        assert flags == set()
        np.testing.assert_allclose(minima, [-radius, radius], rtol=0, atol=1e-12)


def test_precision_patch_and_generalized_models_keep_existing_candidate():
    candidate = ROOT + 3e-9
    model = flat_model()
    provider = SimpleNamespace(model=model, angle=0, patch=object(), k_patch=1e-4)
    assert _polish_native_gap_minimum(provider, 0, ROOT - 1e-7, ROOT + 1e-7, candidate, 1e-12) == candidate
    model.supports_generalized = True
    provider.patch = None
    assert _polish_native_gap_minimum(provider, 0, ROOT - 1e-7, ROOT + 1e-7, candidate, 1e-12) == candidate


def test_unavailable_derivative_or_missing_stationary_bracket_keeps_candidate(monkeypatch):
    model = flat_model()
    provider = ModelGeometry(model)
    candidate = ROOT + 3e-9
    assert _polish_native_gap_minimum(provider, 0, ROOT + 1e-8, ROOT + 1e-7, candidate, 1e-12) == candidate

    def unavailable(*args):
        raise NotImplementedError("no analytic band derivative")

    monkeypatch.setattr(model, "dhamiltonian", unavailable)
    assert _polish_native_gap_minimum(provider, 0, ROOT - 1e-7, ROOT + 1e-7, candidate, 1e-12) == candidate


def test_touching_gap_keeps_existing_minimizer():
    model = flat_model()
    provider = ModelGeometry(model)
    # A configured touching threshold also disables individual-band polishing.
    candidate = ROOT + 3e-9
    assert _polish_native_gap_minimum(provider, 0, ROOT - 1e-7, ROOT + 1e-7, candidate, GAP * 2) == candidate


def test_polish_passes_provider_memory_budget_to_native_line_context(monkeypatch):
    model = flat_model()
    provider = ModelGeometry(model)
    provider.mem_bytes = 8e6
    budgets = []
    original = model.on_line

    def record(angle, t, *, mem_bytes=None):
        budgets.append(mem_bytes)
        return original(angle, t, mem_bytes=mem_bytes)

    monkeypatch.setattr(model, "on_line", record)
    position = _polish_native_gap_minimum(provider, 0, ROOT - 1e-7, ROOT + 1e-7, ROOT + 3e-9, 1e-12)
    assert position == pytest.approx(ROOT, abs=1e-14)
    assert budgets == [provider.mem_bytes]
