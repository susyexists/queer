"""Independent statistical/normalization references and preserved model contracts."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from queer import OpenMXModel, WannierModel
from queer.finite_field._storage import source_files, source_hash
from queer.models.contracts import EigenstateBatch
from queer.physics.broadening import gaussian, lorentzian
from queer.physics.constants import KB_EV
from queer.physics.statistics import occupation_derivatives, occupations, thermal_occupation

ROOT = Path(__file__).resolve().parents[2]


def test_statistics_keep_temperature_units_and_zero_temperature_boundary():
    energy = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_array_equal(thermal_occupation(energy, 0), [1, 0.5, 0])
    np.testing.assert_array_equal(occupations(energy, 0, 0), [1, 0, 0])
    temperature = 310.0
    np.testing.assert_allclose(
        occupations(energy, 0, temperature), thermal_occupation(energy, KB_EV * temperature)
    )
    f, first, second = occupation_derivatives(energy * 0.04, 0, temperature, 0.05)
    assert f[1] == 0.5
    assert first[1] == pytest.approx(-1 / (4 * KB_EV * temperature))
    assert second[1] == 0
    np.testing.assert_allclose(first, first[::-1])
    np.testing.assert_allclose(second, -second[::-1])


@pytest.mark.parametrize("shape", [gaussian, lorentzian])
def test_broadening_integrates_to_one_and_has_explicit_width(shape):
    width = 0.13
    integral, error = quad(lambda x: float(shape(x, width)), -np.inf, np.inf)
    assert integral == pytest.approx(1.0, abs=1e-10)
    assert error < 1e-8
    for invalid in (0, -1, np.inf, np.nan):
        with pytest.raises(ValueError):
            shape(0, invalid)
    if shape is lorentzian:
        assert shape(width, width) == pytest.approx(shape(0, width) / 2)
    else:
        assert shape(width, width) == pytest.approx(shape(0, width) * np.exp(-0.5))


@pytest.mark.parametrize("shape", [gaussian, lorentzian])
def test_adaptive_broadening_accepts_pair_resolved_widths(shape):
    widths = np.array([[0.03, 0.1], [0.2, 0.7]])
    # Integrate every transition at its own linewidth, independently of the
    # broadcasting used by the adaptive response contraction.
    areas = np.array(
        [[quad(lambda x: float(shape(x, width)), -np.inf, np.inf)[0] for width in row] for row in widths]
    )
    np.testing.assert_allclose(areas, np.ones_like(widths), atol=1e-10)
    assert shape(np.zeros((3, 2, 2)), widths).shape == (3, 2, 2)
    with pytest.raises(ValueError):
        shape(0, [0.1, 0])


def test_eigenstate_batches_preserve_schema_normalization_and_mutability():
    models = (
        WannierModel.from_directory(ROOT / "tests/fixtures/wannier/chain"),
        OpenMXModel.from_scfout(ROOT / "tests/fixtures/openmx/chain/chain.scfout"),
    )
    for model in models:
        batch = model.eigenstates([[0, 0, 0], [0.25, 0, 0]])
        assert isinstance(batch, EigenstateBatch)
        assert batch.absolute_energies is batch.energies
        for c, s in zip(batch.coefficients, batch.overlaps, strict=True):
            np.testing.assert_allclose(c.conj().T @ s @ c, np.eye(c.shape[1]), atol=1e-12)
        if isinstance(model, WannierModel):
            batch.coefficients *= 1j
            batch.coefficients = batch.coefficients.copy()
            initial = batch.energies
            batch.absolute_energies = initial + 2
            assert batch.energies is initial
            np.testing.assert_array_equal(batch.absolute_energies, initial + 2)
            batch.energies = initial - 1
            np.testing.assert_array_equal(batch.absolute_energies, initial + 2)
        else:
            with pytest.raises(FrozenInstanceError):
                batch.coefficients = batch.coefficients.copy()
            with pytest.raises(FrozenInstanceError):
                batch.absolute_energies = batch.energies.copy()


def test_new_numerical_owners_participate_in_engine_fingerprint(monkeypatch):
    owners = source_files()
    assert any(path.name == "statistics.py" and path.parent.name == "physics" for path in owners)
    assert any(path.name == "reduction.py" and path.parent.name == "workflows" for path in owners)
    original = Path.read_bytes
    before = source_hash()
    target = next(path for path in owners if path.name == "statistics.py" and path.parent.name == "physics")

    def changed(path):
        data = original(path)
        return data + b"\n# fingerprint probe" if path == target else data

    monkeypatch.setattr(Path, "read_bytes", changed)
    assert source_hash() != before
