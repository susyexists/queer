"""Dependency boundaries protect shared numerical owners from orchestration."""

import ast
from importlib.util import resolve_name
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "queer"


def imports(path):
    relative = path.relative_to(SOURCE).with_suffix("")
    parts = relative.parts
    package = ".".join(("queer", *parts[:-1]))
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            name = "." * node.level + (node.module or "")
            yield resolve_name(name, package) if node.level else name


def test_physics_kernels_depend_only_on_physics_and_numerical_libraries():
    forbidden = ("matplotlib", "pandas", "joblib", "threadpoolctl", "pathlib", "subprocess")
    for path in (SOURCE / "physics").glob("*.py"):
        for name in imports(path):
            assert not name.startswith(forbidden), (path, name)
            if name.startswith("queer."):
                assert name.startswith("queer.physics."), (path, name)


def test_native_code_does_not_depend_on_legacy_facades():
    legacy = {
        "queer.functions",
        "queer.utils",
        "queer.model",
        "queer.epw",
        "queer.response",
        "queer.spectral",
        "queer.plots",
        "queer.plot_param",
        "queer.epw_utils",
        "queer.epw_plots",
    }
    for directory in ("models", "physics", "observables", "atomistic", "geometry", "io", "finite_field"):
        for path in (SOURCE / directory).rglob("*.py"):
            assert legacy.isdisjoint(imports(path)), path


def test_model_and_field_evaluation_do_not_import_observable_implementation():
    for directory in ("models", "finite_field"):
        for path in (SOURCE / directory).rglob("*.py"):
            for name in imports(path):
                assert not name.startswith("queer.observables._"), (path, name)
                assert name not in {"queer.observables.geometry", "queer.observables.optical"}, (path, name)


def test_quadrature_has_no_filesystem_or_run_store_access():
    path = SOURCE / "observables" / "integration.py"
    names = set(imports(path))
    assert not names.intersection({"pathlib", "queer.io.artifacts", "queer.io.results"})
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"manifest", "commit"}, node.attr
            assert not (
                isinstance(node.value, ast.Name)
                and node.value.id == "np"
                and node.attr in {"load", "save", "savez"}
            )
