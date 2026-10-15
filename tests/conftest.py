"""Session helpers shared by every test directory."""

import importlib.util
import sys
from pathlib import Path

import pytest

ORACLES = Path(__file__).resolve().parent / "oracles"


def load_oracle(name):
    """Import ``tests/oracles/<name>.py`` by path once per session, without PythTB.

    The module is registered under a private name and never added to a package,
    so it cannot shadow or be shadowed by an installed module.
    """
    module_name = f"_queer_oracle_{name}"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(module_name, ORACLES / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the oracle {name!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[module_name]
        raise
    return module


@pytest.fixture(scope="session")
def generator_core():
    """The Generator's ``pythtb_core`` driver, as copied into ``tests/oracles``."""
    return load_oracle("generator_pythtb_core")


@pytest.fixture(scope="session")
def generator_qgeom():
    """The Generator's ``pythtb_qgeom`` kernel, as copied into ``tests/oracles``."""
    return load_oracle("generator_pythtb_qgeom")
