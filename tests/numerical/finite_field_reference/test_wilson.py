import math

import numpy as np
import pytest

from queer.finite_field.geometry.wilson import geometric_phase, shift_vector
from queer.finite_field.models import FourBandModel, FourBandParams
from queer.finite_field.models.toy import ParabolicDoubleCrossing

P = FourBandParams()


def test_real_two_level_model_has_pi():
    toy = ParabolicDoubleCrossing()
    for n in (2000, 2001):  # with and without a node exactly on the zero of v_cv
        assert abs(abs(geometric_phase(toy, -toy.k0, toy.k0, 0.0, n=n)) - math.pi) < 1e-10


@pytest.mark.parametrize(
    "t_over_k0, ref", [(0.0, 0.0), (0.25, -0.7571), (0.5, -1.4303), (-0.5, 1.4303), (0.8, -2.0009)]
)
def test_fourband_geometric_phase_values(t_over_k0, ref):
    # analysis workflow (TDSE-validated): VL/VLc = 1.8, crossings at +-sqrt(k0^2 - t^2)
    m = FourBandModel(P.at(1.8)).traceless()
    k0 = 3.852e-3
    t = t_over_k0 * k0
    s0 = math.sqrt(k0**2 - t**2)
    assert geometric_phase(m, -s0, s0, t) == pytest.approx(ref, abs=2e-4)
    # reversing the traversal flips the sign
    assert geometric_phase(m, s0, -s0, t) == pytest.approx(-ref, abs=2e-4)


def test_gauge_invariance_under_random_phases(monkeypatch):
    import queer.finite_field.geometry.wilson as w

    m = FourBandModel(P.at(1.8)).traceless()
    ref = geometric_phase(m, -3e-3, 3e-3, 1e-3)
    orig = np.linalg.eigh
    rng = np.random.default_rng(3)

    def scrambled(H):
        e, U = orig(H)
        return e, U * np.exp(1j * rng.uniform(0, 2 * math.pi, U.shape[:-2] + (1, U.shape[-1])))

    monkeypatch.setattr(w.np.linalg, "eigh", scrambled)
    assert geometric_phase(m, -3e-3, 3e-3, 1e-3) == pytest.approx(ref, abs=1e-9)


def test_critical_shift_vector_is_minus_4C():
    m = FourBandModel(P.at(1.0)).traceless()
    C = P.at(1.0).C_critical
    assert shift_vector(m, 0.0, 1e-5) == pytest.approx(-4 * C, rel=2e-3)
    assert shift_vector(m, 0.0, -1e-5) == pytest.approx(4 * C, rel=2e-3)
