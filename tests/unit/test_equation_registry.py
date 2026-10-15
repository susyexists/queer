"""The equation index resolves real owners and independent tests without imports."""

import ast
import re
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = tomllib.loads((ROOT / "docs/equations.toml").read_text())
EQUATIONS = REGISTRY["equations"]


def definition(path, qualified_name):
    body = ast.parse(path.read_text()).body
    for name in qualified_name.split("."):
        matches = [
            node
            for node in body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name
        ]
        assert matches, f"{path}: missing definition {qualified_name}"
        body = matches[0].body


def implementation(target):
    module, symbol = target.split(":")
    assert module.startswith("queer.")
    path = ROOT / "src" / Path(*module.split(".")).with_suffix(".py")
    assert path.is_file(), f"Missing implementation source: {target}"
    definition(path, symbol)


def test_equation_registry_schema_and_unique_ids():
    assert REGISTRY["schema_version"] == 1
    identifiers = [equation["id"] for equation in EQUATIONS]
    assert identifiers and len(identifiers) == len(set(identifiers))
    assert all(re.fullmatch(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+", value) for value in identifiers)


@pytest.mark.parametrize("equation", EQUATIONS, ids=lambda equation: equation["id"])
def test_equation_registry_owners_and_references_exist(equation):
    for field in ("expression", "input_units", "output_units", "canonical"):
        assert isinstance(equation[field], str) and equation[field].strip()
    for field in ("assumptions", "conventions", "source_references", "independent_tests"):
        assert equation[field] and all(isinstance(value, str) and value.strip() for value in equation[field])
    implementation(equation["canonical"])
    for backend in equation["specialized_backends"]:
        implementation(backend)
    for reference in equation["source_references"]:
        address = urlsplit(reference)
        if address.scheme:
            assert address.scheme == "https" and address.netloc
        else:
            assert (ROOT / address.path).is_file(), f"Missing local source reference: {reference}"
    for node_id in equation["independent_tests"]:
        relative_path, symbol = node_id.split("::")
        assert relative_path.startswith("tests/") and symbol.startswith("test_")
        path = ROOT / relative_path
        assert path.is_file(), f"Missing independent test source: {node_id}"
        definition(path, symbol)
