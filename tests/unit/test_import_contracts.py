"""Callable 1.x facades survive either submodule import order."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("name", ["model", "epw"])
def test_direct_submodule_import_preserves_package_callable(name):
    probe = f'''
import importlib
import queer
module = importlib.import_module("queer.{name}")
assert callable(queer.{name})
assert queer.{name} is getattr(module, "{name}")
'''
    subprocess.run([sys.executable, "-c", probe], check=True, capture_output=True)


def test_ordinary_wannier_import_keeps_codata_table_lazy():
    probe = """
import sys
from queer.models.wannier import WannierModel
assert "scipy.constants" not in sys.modules
from queer.physics.constants import BOHR_ANGSTROM
assert "scipy.constants" in sys.modules
assert BOHR_ANGSTROM > .5
"""
    subprocess.run([sys.executable, "-c", probe], check=True, capture_output=True)
