"""Symmetry shortcuts under the PHYSICAL wavefunction phases (geometric phase, twisted LZ).

Antiunitary operations map a path onto the mirrored path traversed backwards and flip the shift vector,
so with these phases only unitary operations certified on R_s may grant shortcuts (symmetry.py).
"""

import math

import numpy as np
import pytest

from queer.finite_field.config import PAPER, PHYSICAL
from queer.finite_field.geometry.provider import ModelGeometry
from queer.finite_field.geometry.wilson import shift_field, shift_vector
from queer.finite_field.models import FourBandModel, FourBandParams
from queer.finite_field.symmetry import certify, selection_rules
from queer.finite_field.transport.hall import HallCalculator

P = FourBandParams()


def _rules(model, cfg, angle):
    cfg = cfg.with_(**{"drive.angle": angle}, use_symmetry_shortcuts=True)
    return HallCalculator(model, cfg).selection_rules()


def test_shift_field_matches_scalar_shift_vector():
    m = FourBandModel(P.at(0.9)).traceless()
    rng = np.random.default_rng(3)
    k = rng.uniform(-3e-3, 3e-3, size=(2, 6))
    for angle in (0.0, 0.7):
        f = shift_field(m, k[0], k[1], angle)
        c, s = math.cos(angle), math.sin(angle)
        ref = [
            shift_vector(m, kx * c + ky * s, -kx * s + ky * c, angle, h=1e-3 * math.hypot(kx, ky))
            for kx, ky in k.T
        ]
        np.testing.assert_allclose(f, ref, rtol=1e-6)


def test_isotropic_model_keeps_c2_but_not_the_mirror():
    m = FourBandModel(P.at(1.4))
    paper, phys = _rules(m, PAPER, 0.0), _rules(m, PHYSICAL, 0.0)
    assert paper["t_mirror"] and paper["e_odd"]
    assert not phys["t_mirror"]  # a mirror fixing s is antiunitary (omega_sign = +1, det R = -1)
    assert phys["e_odd"]  # C2 is unitary and certified on R_s


def test_warped_model_loses_mirror_oddness():
    # 3m' emulator, field perpendicular to the M0 mirror line: E-oddness only from the antiunitary M0*K
    m = FourBandModel(P.at(0.5), warping=200.0, warping_matrix="sz_t0")
    assert _rules(m, PAPER, math.pi / 2)["e_odd"]
    phys = _rules(m, PHYSICAL, math.pi / 2)
    assert not phys["e_odd"] and not phys["t_mirror"]


def test_certificate_without_shift_test_grants_nothing_under_wavefunction_phases():
    prov = ModelGeometry(FourBandModel(P.at(1.4)))
    cert = certify(prov.at_cartesian, field_angles=(0.0,))
    assert selection_rules(cert, 0.0)["e_odd"]
    assert not any(selection_rules(cert, 0.0, wavefunction_phases=True)[k] for k in ("e_odd", "t_mirror"))
    cert = certify(
        prov.at_cartesian, field_angles=(0.0,), shift_fn=lambda kx, ky: shift_field(prov.model, kx, ky, 0.0)
    )
    assert selection_rules(cert, 0.0, wavefunction_phases=True)["e_odd"]
    assert not selection_rules(cert, math.pi / 2, wavefunction_phases=True)["e_odd"]  # tested at angle 0 only


@pytest.mark.slow
def test_physical_shortcuts_do_not_change_J():
    m = FourBandModel(P.at(1.4))
    E = np.array([1e4, 3e4, -1e4, -3e4])
    full = HallCalculator(m, PHYSICAL).prepare()
    short = HallCalculator(m, PHYSICAL.with_(use_symmetry_shortcuts=True)).prepare()
    assert not short.layout.mirrored
    np.testing.assert_allclose(short.J(E), full.J(E), rtol=1e-12)
    np.testing.assert_allclose(full.J(E[:2]), -full.J(E[2:]), rtol=1e-9)  # C2: odd


@pytest.mark.slow
def test_twisted_lz_gives_the_warped_model_an_even_part():
    # single gap-minimum crossings at x = 0.5: the even part comes from twisted LZ alone (~0.25% at 10 kV/m)
    m = FourBandModel(P.at(0.5), warping=200.0, warping_matrix="sz_t0")
    cfg = PAPER.with_(**{"drive.angle": math.pi / 2, "phase.twisted_lz": True}, use_symmetry_shortcuts=True)
    hc = HallCalculator(m, cfg).prepare()
    assert not hc.selection_rules()["e_odd"]
    J = hc.J(np.array([1e4, -1e4]))
    assert abs(J[0] + J[1]) > 1e-3 * abs(J[0] - J[1])
    paper = HallCalculator(m, PAPER.with_(**{"drive.angle": math.pi / 2})).prepare().J(np.array([1e4, -1e4]))
    assert abs(paper[0] + paper[1]) < 1e-7 * abs(paper[0] - paper[1])  # quadrature level
