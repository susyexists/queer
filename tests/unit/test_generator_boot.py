"""The generator boot checks its environment and reports failures without NumPy."""

import ast
import importlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

import queer
import queer.__main__ as entry
from queer import _generator_boot as boot

ROOT = Path(__file__).resolve().parents[2]
CORE = ("numpy", "scipy", "joblib", "threadpoolctl")


def read_results(workspace):
    return json.loads((workspace / "queer_results.json").read_text())


def test_boot_imports_with_core_dependencies_blocked(monkeypatch):
    for name in CORE:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delitem(sys.modules, "queer._generator_boot", raising=False)
    module = importlib.import_module("queer._generator_boot")
    assert module.main is not None
    assert not any(isinstance(value, type(sys)) and value.__name__ in CORE for value in vars(module).values())
    # Blocked modules cannot be found, so the check reports every core dependency.
    gaps = module.environment_gaps()
    assert [gap.split(" ", 1)[0].split(">=")[0] for gap in gaps] == list(CORE)


@pytest.mark.parametrize(
    "relative",
    [
        "__init__.py",
        "__main__.py",
        "_generator_boot.py",
        "compat/__init__.py",
        "compat/generator/__init__.py",
    ],
)
def test_boot_path_imports_only_the_standard_library_at_module_level(relative):
    tree = ast.parse((ROOT / "src/queer" / relative).read_text())
    for node in tree.body:
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names = [node.module or ""]
        elif isinstance(node, ast.ImportFrom):
            raise AssertionError(f"{relative} imports package code at module level: line {node.lineno}")
        for name in names:
            assert name.split(".")[0] in sys.stdlib_module_names | {"__future__"}, (relative, name)


def test_embedded_requirements_equal_the_project_dependencies():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    declared = {}
    for requirement in project["dependencies"]:
        name, minimum = re.fullmatch(r"([A-Za-z0-9_.-]+)>=([0-9.]+)", requirement).groups()
        declared[name] = minimum
    assert boot.REQUIREMENTS == declared


@pytest.mark.parametrize(
    "installed,minimum,expected",
    [
        ("1.26", "1.26", True),
        ("1.26.4", "1.26", True),
        ("2.0.0", "1.26", True),
        ("1.25.2", "1.26", False),
        ("1.9", "1.11", False),
        ("1.11.0rc1", "1.11", False),
        ("1.11.0.dev0+abc", "1.11", False),
        ("2.0.0rc1", "1.26", True),
        ("1.26.0.post1", "1.26", True),
        ("3.2+local", "3.2", True),
        ("v1.3.2", "1.3", True),
        ("1!2.0", "1.0", False),
        ("unknown", "1.0", False),
    ],
)
def test_release_comparison(installed, minimum, expected):
    assert boot.version_at_least(installed, minimum) is expected


def fake_versions(monkeypatch, versions):
    def version(name):
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(boot.importlib.metadata, "version", version)


def test_run_reports_a_missing_or_outdated_environment(tmp_path, monkeypatch):
    fake_versions(monkeypatch, {"numpy": "1.25.2", "joblib": "1.4.2", "threadpoolctl": "3.5.0"})
    workspace = tmp_path / "workspace"
    code = entry.main(
        [
            "generator",
            "run",
            "--step",
            "QUEER_BANDS",
            "--params",
            "queer_params.json",
            "--workspace",
            str(workspace),
        ]
    )
    assert code == 2
    record = read_results(workspace)
    message = record.pop("error")
    assert record == {
        "schema_version": 1,
        "status": "failed",
        "step_type": "QUEER_BANDS",
        "engine": "queer",
        "provenance": {"queer_version": queer.__version__, "python": record["provenance"]["python"]},
        "warnings": [],
        "outputs": [],
    }
    assert message["code"] == "queer_env_missing"
    assert "numpy 1.25.2 is older than the required 1.26" in message["message"]
    assert "scipy>=1.11 is not installed" in message["message"]
    assert not list(workspace.glob(".queer_results.json.*"))


def test_run_without_a_runner_reports_an_internal_error(tmp_path):
    code = boot.main(["run", "--step", "QUEER_TOPOLOGY", "--params", "p.json", "--workspace", str(tmp_path)])
    assert code == 1
    record = read_results(tmp_path)
    assert record["error"] == {
        "code": "queer_internal_error",
        "message": "generator runner not implemented in this build",
    }
    assert record["step_type"] == "QUEER_TOPOLOGY"


def test_invalid_run_arguments_still_write_a_results_file(tmp_path):
    assert boot.main(["run", "--params", "p.json", f"--workspace={tmp_path}"]) == 2
    record = read_results(tmp_path)
    assert record["error"]["code"] == "queer_bad_params" and "--step" in record["error"]["message"]
    assert record["step_type"] is None


def test_capabilities_and_runner_delegation(tmp_path, monkeypatch, capsys):
    assert boot.main(["capabilities"]) == 0
    assert json.loads(capsys.readouterr().out) == {"queer_version": queer.__version__, "steps": []}
    fake_versions(monkeypatch, {"numpy": "2.0", "scipy": "1.14", "joblib": "1.4"})
    assert boot.main(["capabilities"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["steps"] == [] and payload["error"]["code"] == "queer_env_missing"
    monkeypatch.undo()
    calls = []
    runner = SimpleNamespace(main=lambda argv: calls.append(list(argv)) or 7)
    monkeypatch.setitem(sys.modules, boot.RUNNER_MODULE, runner)
    arguments = ["run", "--step", "QUEER_BANDS", "--params", "p.json", "--workspace", str(tmp_path)]
    assert boot.main(arguments) == 7
    assert boot.main(["compare", "--reference", "a", "--candidate", "b"]) == 7
    assert boot.main(["capabilities"]) == 7
    assert calls == [arguments, ["compare", "--reference", "a", "--candidate", "b"], ["capabilities"]]
    assert boot.main([]) == 2 and boot.main(["deploy"]) == 2


def test_bare_interpreter_without_numpy_reports_missing_environment(tmp_path):
    venv = tmp_path / "bare"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True, capture_output=True
    )
    python = venv / "bin" / "python"
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONHOME", "VIRTUAL_ENV"}}
    env.update(PYTHONPATH=str(ROOT / "src"), PYTHONNOUSERSITE="1")
    probe = subprocess.run(
        [str(python), "-c", "import importlib.util; print(importlib.util.find_spec('numpy') is None)"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    if probe.stdout.strip() != "True":
        pytest.skip("the base interpreter provides NumPy to new environments")
    workspace = tmp_path / "workspace"
    result = subprocess.run(
        [
            str(python),
            "-m",
            "queer",
            "generator",
            "run",
            "--step",
            "QUEER_BANDS",
            "--params",
            "x.json",
            "--workspace",
            str(workspace),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, result.stderr
    record = read_results(workspace)
    assert record["status"] == "failed" and record["engine"] == "queer"
    assert record["error"]["code"] == "queer_env_missing"
    assert "numpy>=1.26 is not installed" in record["error"]["message"]
    assert "Traceback" not in result.stderr
