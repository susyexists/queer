import os

# Pin BLAS pools before numpy is imported anywhere in the test session.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def rng():
    return np.random.default_rng(20261010)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: convergence and independent TDSE acceptance")
    config.addinivalue_line("markers", "paper: finite-field paper reference anchor")
