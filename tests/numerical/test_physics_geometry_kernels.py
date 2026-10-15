"""Independent references for model-free retained-basis physics kernels."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import eigh

from queer.models.contracts import EigenstateBatch
from queer.physics.eigensystems import centered_pair_gap
from queer.physics.geometry import (
    band_curvatures,
    band_geometry_from_operators,
    interband_products,
    pair_metrics,
    projected_basis_curvature,
)
from queer.physics.response import linear_density

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]])
SZ = np.diag([1, -1])


def geometry(transform=np.eye(2)):
    """H=kx*sigma_x+ky*sigma_y+sigma_z at k=0, in an arbitrary basis."""
    h = transform.conj().T @ SZ @ transform
    overlap = transform.conj().T @ transform
    energy, coeff = eigh(h, overlap)
    solution = EigenstateBatch(energy[:, None], coeff[None], overlap[None], np.zeros(2, int))
    derivatives = {
        "H": np.stack([transform.conj().T @ x @ transform for x in (SX, SY, np.zeros((2, 2)))]),
        "S": np.zeros((3, 2, 2), complex),
        "r": np.zeros((3, 3, 2, 2), complex),
    }
    data = band_geometry_from_operators(solution, derivatives, np.zeros((3, 2, 2)), 1e-9)
    return data, derivatives, h


def test_array_only_dirac_geometry_and_nonorthogonal_covariance():
    first, derivative, h = geometry()
    transformed, second_derivative, second_h = geometry(np.array([[1.2, 0.2j], [0.1, 0.7]]))
    for data, operators, hamiltonian in ((first, derivative, h), (transformed, second_derivative, second_h)):
        metric, totals = pair_metrics(data["connections"], (0, 1), data["band_channels"])
        assert_allclose(metric, np.diag([0.25, 0.25, 0]), atol=2e-15)
        assert_allclose(totals, np.broadcast_to(metric, (2, 3, 3)), atol=2e-15)
        projected = projected_basis_curvature(data, operators["r"])
        assert_allclose(projected, 0, atol=2e-15)
        assert_allclose(band_curvatures(data, "direct", projected)[:, 0, 1], [0.5, -0.5], atol=2e-15)
        assert_allclose(centered_pair_gap(hamiltonian, data, (0, 1))[0], 2, atol=2e-15)


def test_selected_row_reducer_matches_closed_dirac_geometry():
    data, _, _ = geometry()
    delta = data["energies"][:, None] - data["energies"][None, :]
    inverse = np.zeros_like(delta)
    np.divide(1, delta**2, out=inverse, where=delta != 0)
    omega, gss, gst, gtt = interband_products(*data["velocities"][:2], inverse)
    assert_allclose(omega, [0.5, -0.5], atol=2e-15)
    assert_allclose(gss.sum(axis=-1), [0.25, 0.25], atol=2e-15)
    assert_allclose(gst, 0, atol=2e-15)
    assert_allclose(gtt.sum(axis=-1), [0.25, 0.25], atol=2e-15)


def test_array_only_linear_density_retarded_resonances():
    data, _, _ = geometry()
    z = np.array([0, 2, 3]) + 0.05j
    rho = linear_density(data, z, 0, 0, 0.05)
    # With E_v=-1, E_c=1, occupations are (1,0); evaluate both poles independently.
    assert_allclose(rho[:, 0, 0, 1], -data["connections"][0, 0, 1] / (z + 2), atol=1e-14)
    assert_allclose(rho[:, 0, 1, 0], data["connections"][0, 1, 0] / (z - 2), atol=1e-14)
    with pytest.raises(ValueError, match="projected-coordinate"):
        band_curvatures(data, "direct")


@pytest.mark.parametrize("calculator_name", ["ahc", "chern_number", "shift_current"])
@pytest.mark.parametrize("method", ["cubature", "adaptive"])
def test_public_response_checkpoint_injection_and_resume(calculator_name, method):
    from queer.models.wannier import WannierModel
    from queer.observables.geometry import chern_number
    from queer.observables.integrated import ahc
    from queer.observables.nonlinear import shift_current

    class MemoryCheckpoint:
        def __init__(self):
            self.records = {}
            self.saves = 0

        def load(self, case):
            return self.records.get(case)

        def save(self, case, arrays, metadata):
            self.records[case] = ({k: np.array(v, copy=True) for k, v in arrays.items()}, dict(metadata))
            self.saves += 1

    position = np.zeros((1, 2, 2, 3), complex)
    position[0, :, :, :2] = np.stack((SX, SY), axis=-1)
    model = WannierModel([[0, 0, 0]], np.diag([-1.0, 1.0])[None], position=position, lattice=np.eye(3))
    calculator, parameters = {
        "ahc": (ahc, {"engine": "kubo", "dimension": 2}),
        "chern_number": (chern_number, {"engine": "kubo", "occupied_bands": 1}),
        "shift_current": (shift_current, {"omega": [1, 2]}),
    }[calculator_name]
    options = {
        "method": method,
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 1e-6,
        "max_levels": 2,
        "max_points": 1000,
    }
    expected = calculator(model, grid=(1, 1, 1), integration=options, **parameters)
    checkpoint = MemoryCheckpoint()
    actual = calculator(model, grid=(1, 1, 1), integration=options, checkpoint=checkpoint, **parameters)
    assert checkpoint.saves > 0
    committed = checkpoint.saves
    resumed = calculator(model, grid=(1, 1, 1), integration=options, checkpoint=checkpoint, **parameters)
    assert checkpoint.saves == committed
    for key in expected.arrays:
        assert_allclose(actual[key], expected[key], atol=0, rtol=0)
        assert_allclose(resumed[key], expected[key], atol=0, rtol=0)
    assert actual.dims == resumed.dims == expected.dims
    assert actual.units == resumed.units == expected.units
