"""1.x facades and lazy compatibility packages keep their import contracts."""

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


GENERATOR_COMPAT_NAMES = [
    "ASE_FS",
    "Any",
    "BOHR_ANGSTROM",
    "HARTREE_EV",
    "Path",
    "annotations",
    "atomic_json",
    "band_payload",
    "calculation_payload",
    "dataset_exports",
    "digest_file",
    "export_run",
    "np",
    "select_legacy_min_energy_frame",
    "structure_payload",
    "transform_structures",
    "translate_ase",
    "translate_pyatb",
]


def test_generator_compat_package_keeps_its_public_names_lazily():
    probe = f"""
import sys
import queer.compat.generator as generator
assert "numpy" not in sys.modules
expected = {GENERATOR_COMPAT_NAMES!r}
assert sorted(name for name in dir(generator) if not name.startswith("_")) == expected
assert sorted(generator.__all__) == expected
from queer.compat.generator import export_run, translate_pyatb
import queer.compat.generator._legacy as legacy
assert all(getattr(generator, name) is getattr(legacy, name) for name in expected)
import queer
assert queer.export_run is export_run is legacy.export_run
assert translate_pyatb.__module__ == "queer.compat.generator._legacy"
star = {{}}
exec("from queer.compat.generator import *", star)
assert sorted(name for name in star if not name.startswith("_")) == expected
try:
    generator.translate_pythtb
except AttributeError:
    pass
else:
    raise AssertionError("an unknown name must raise AttributeError")
"""
    subprocess.run([sys.executable, "-c", probe], check=True, capture_output=True)


def test_generator_compat_package_and_runner_lookup_import_without_numpy():
    probe = """
import importlib.util
import sys
sys.modules["numpy"] = None
import queer.compat.generator
from queer import _generator_boot as boot
assert importlib.util.find_spec("queer.compat.generator.runner") is None
assert boot._runner() is None
"""
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
