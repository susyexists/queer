"""Scalar validation independent of model, storage, and execution concerns."""

import numpy as np


class NumericalError(RuntimeError):
    """A scientific calculation could not produce validated numerical output."""


def positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value
