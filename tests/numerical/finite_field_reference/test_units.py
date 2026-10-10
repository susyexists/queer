import math

import numpy as np
import pytest

from queer.finite_field import constants as c
from queer.finite_field import units as u


def test_codata_values():
    assert c.HBAR_EVS == pytest.approx(6.582119569e-16, rel=1e-10)
    assert c.E2_H == pytest.approx(3.874045865e-5, rel=1e-9)


def test_kappa_relaxation_length():
    # e E T1 / hbar at 1e4 V/m and 2 ps, in 1/A
    assert u.kappa(1e4, 2e-12) == pytest.approx(3.0385e-3, rel=1e-4)
    assert u.kappa(-1e4, 2e-12) == pytest.approx(3.0385e-3, rel=1e-4)
    assert u.kappa(1.0, 2e-12) == pytest.approx(3.0385e-7, rel=1e-4)
    assert math.isinf(u.kappa(1e4, math.inf))


def test_critical_field_vl0_closed_form():
    # four-band model at VL = 0: eps_g = 2(|M|-Bl), sqrt(g) = A/(2(|M|-Bl)) -> E_c = pi(|M|-Bl)^2/A
    M, Bl, A = 1.022e-3, 0.4573e-3, 0.3290
    m = M - Bl
    ec = u.critical_field(2 * m, (A / (2 * m)) ** 2)
    assert ec == pytest.approx(math.pi * m**2 / A * 1e10, rel=1e-12)
    assert ec == pytest.approx(30450.2, rel=1e-5)


def test_phase_and_frequency_consistency():
    integral = 3.7e-6  # eV/A
    lam = u.interference_frequency(integral)
    for E in (1e3, -2.5e4):
        assert u.dynamical_phase(integral, E) == pytest.approx(2 * math.pi * lam / abs(E), rel=1e-12)


def test_velocity_roundtrip():
    hv = u.velocity_to_hbar_v(2.24e4)
    assert hv == pytest.approx(0.147439, rel=1e-5)
    assert u.hbar_v_to_velocity(hv) == pytest.approx(2.24e4, rel=1e-12)


def test_prefactor_modes():
    assert u.hall_prefactor("e2_h") == pytest.approx(c.E2_H)
    assert u.hall_prefactor("e2_2pi_h") == pytest.approx(c.E2_H / (2 * math.pi))
    with pytest.raises(ValueError):
        u.hall_prefactor("bogus")


def test_vectorized():
    E = np.array([1e2, 1e3, 1e4])
    assert u.kappa(E, 2e-12).shape == (3,)
    assert np.allclose(u.delta_ky(E, 0.147213), np.sqrt(1e-10 * E / 0.147213))
