import pytest

from queer.finite_field.geometry.topology import chern_radial, fhs_flux, graded_axis
from queer.finite_field.models import FourBandModel, FourBandParams

P = FourBandParams()


@pytest.mark.parametrize("x, c_band2", [(0.5, -1.0), (1.5, 0.0)])
def test_band2_chern_number(x, c_band2):
    # band 2 (index 1) carries the transition: -1 -> 0 in the Omega convention ("C = 1 -> 0" up to sign)
    m = FourBandModel(P.at(x)).traceless()
    c = chern_radial(m, 1, k_max=20.0, breakpoints=(P.k_inv,))
    assert c == pytest.approx(c_band2, abs=2e-3)


def test_lower_two_band_total_and_regularization():
    # band 1 carries the extra -1 of the B k^2 regularization (sgn M = sgn B)
    for x in (0.5, 1.5):
        m = FourBandModel(P.at(x)).traceless()
        assert chern_radial(m, 0, k_max=20.0, breakpoints=(P.k_inv,)) == pytest.approx(-1.0, abs=2e-3)


def test_fhs_box_flux_jump_is_one():
    ax = graded_axis(0.05, 241, a=2e-5)
    flux = {}
    for x in (0.9, 1.1):
        flux[x] = fhs_flux(FourBandModel(P.at(x)), [0, 1], ax, ax)
    assert flux[1.1] - flux[0.9] == pytest.approx(1.0, abs=5e-3)
