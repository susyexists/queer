import math

import numpy as np
import pytest

from queer.finite_field.analysis.fits import (
    fit_harmonic,
    fit_smooth,
    lambda_intercept,
    onset_slope,
    stationary_phase_fN,
)
from queer.finite_field.harmonics import harmonics

EC, LAM = 16.1e3, 13.1e3


def _activated(E):
    a = np.maximum(np.abs(E), 1e-30)
    return np.sign(E) * np.exp(-EC / a) * a**1.5


def test_onset_slope_depends_on_window():
    """a_F (Methods 4.2) is not universal: 0.22 / 0.14 / 0.055 for windows up to 1 / 1.4 / 2 N E_c."""
    N = np.arange(3, 22, 2)
    expected = {1.0: 0.222, 1.4: 0.140, 2.0: 0.055}
    for xmax, ref in expected.items():
        ecn = []
        for n in N:
            E0 = np.linspace(0.05, xmax, 80) * n * EC
            hr = harmonics(_activated, E0, n_max=n, odd_only=True)
            ecn.append(fit_smooth(E0, hr.b[:, n], (E0[0], E0[-1]), Ec_guess=0.1 * n * EC).Ec_N)
        assert onset_slope(N, ecn)["slope"] / EC == pytest.approx(ref, abs=0.01)


def test_lambda_from_joint_fits():
    def J(E):
        a = np.maximum(np.abs(E), 1e-30)
        return np.sign(E) * np.exp(-EC / a) * (1 + np.cos(2 * math.pi * LAM / a))

    E0 = np.linspace(6e3, 5e4, 400)
    N = (3, 5, 7, 9, 11)
    fN = []
    for n in N:
        hr = harmonics(J, E0, n_max=n, odd_only=True)
        fN.append(fit_harmonic(E0, hr.b[:, n], (E0[0], E0[-1]), f_bounds=(5e3, 6e4))["f"])
    assert np.all(np.diff(fN) > 0)  # f_N grows with N (Ext. Data Fig. 5b/7c)
    assert lambda_intercept(N, fN)["Lambda"] == pytest.approx(LAM, rel=0.10)


def test_stationary_phase_limits():
    f = stationary_phase_fN(LAM, [1, 3, 41], 1e3)
    assert f[0] == pytest.approx(LAM, rel=1e-3) and np.all(np.diff(f) > 0)
    # large N: f_N ~ sqrt(N Lambda E0 / (2 pi))
    big = stationary_phase_fN(LAM, [4001], 5e4)[0]
    assert big == pytest.approx(math.sqrt(4001 * LAM * 5e4 / (2 * math.pi)), rel=0.02)
