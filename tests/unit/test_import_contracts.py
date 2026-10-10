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
