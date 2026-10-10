"""Configuration and installed command contract tests."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from queer.cli import main
from queer.config import config_from_mapping, load_config

ROOT = Path(__file__).resolve().parents[2]


def test_config_precedence_and_relative_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("QUEER_RESULTS_DIR", str(tmp_path / "environment"))
    source = ROOT / "configs/examples/bands.toml"
    config = load_config(source)
    assert config.material_dir == ROOT / "tests/fixtures/wannier/chain"
    assert config.results_dir == tmp_path / "environment"
    data = config.as_dict()
    data["output"]["results_dir"] = str(tmp_path / "configured")
    assert config_from_mapping(data).results_dir == tmp_path / "configured"
    assert (
        config_from_mapping(data, overrides={"results_dir": tmp_path / "cli"}).results_dir == tmp_path / "cli"
    )


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("parameters", "eta_typo", 1),
        ("execution", "workers", 0),
        ("parameters", "ef", float("nan")),
        ("parameters", "kpoints", [[0, 0]]),
    ],
)
def test_config_rejects_invalid_values(section, key, value):
    data = load_config(ROOT / "configs/examples/bands.toml").as_dict()
    data[section][key] = value
    with pytest.raises(ValueError):
        config_from_mapping(data)


def test_new_configs_require_explicit_scientific_convention():
    data = load_config(ROOT / "configs/examples/bands.toml").as_dict()
    del data["parameters"]["interpolation"]
    with pytest.raises(ValueError, match="interpolation"):
        config_from_mapping(data)


def test_momentum_v0_cli_override_replaces_configured_sweep():
    config = load_config(ROOT / "configs/examples/momentum.toml", overrides={"v0": 12.0})
    assert config.parameters["v0"] == 12.0
    assert "v0_values" not in config.parameters


def test_cli_runs_from_unrelated_working_directory(tmp_path):
    run = tmp_path / "bands"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "queer",
            "bands",
            "--config",
            str(ROOT / "configs/examples/bands.toml"),
            "--run-dir",
            str(run),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert str(run) in result.stdout
    assert json.loads((run / "manifest.json").read_text())["status"] == "completed"
    rendered = subprocess.run(
        [sys.executable, "-m", "queer", "render", str(run)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert rendered.returncode == 0, rendered.stderr
    assert (run / "figures/bands.png").is_file()


def test_cli_validation_returns_nonzero_without_partial_run(tmp_path):
    run = tmp_path / "invalid"
    assert (
        main(
            [
                "bands",
                "--config",
                str(ROOT / "configs/examples/bands.toml"),
                "--workers",
                "0",
                "--run-dir",
                str(run),
            ]
        )
        == 1
    )
    assert not run.exists()
