"""Population-only propagation avoids allocating unrequested coherence history."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from queer.finite_field.lz.propagation import propagate_crossings


@pytest.mark.parametrize("coherence", ["paper", "bloch"])
def test_population_only_propagation_allocates_no_complex_history(monkeypatch, coherence):
    rng = np.random.default_rng(32)
    probabilities = rng.uniform(0, 1, size=(7, 11))
    segments = rng.uniform(0, 0.02, size=(6, 11))
    phases = rng.uniform(-np.pi, np.pi, size=(6, 11))
    arguments = (probabilities, segments, phases, 0.02, 0.015)
    expected, expected_coherence = propagate_crossings(*arguments, coherence=coherence, return_coherence=True)
    allocated = []
    original_empty = np.empty

    def recording_empty(shape, dtype=float, *args, **kwargs):
        allocated.append((tuple(shape), np.dtype(dtype)))
        return original_empty(shape, dtype, *args, **kwargs)

    monkeypatch.setattr(np, "empty", recording_empty)
    actual = propagate_crossings(*arguments, coherence=coherence)
    assert_array_equal(actual, expected)
    assert (probabilities.shape, np.dtype(complex)) not in allocated
    allocated.clear()
    repeated, repeated_coherence = propagate_crossings(*arguments, coherence=coherence, return_coherence=True)
    assert_array_equal(repeated, expected)
    assert_array_equal(repeated_coherence, expected_coherence)
    assert allocated.count((probabilities.shape, np.dtype(complex))) == 1
