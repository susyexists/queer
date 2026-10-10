"""Supported scripts retain arguments and run outside the checkout."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONSUMERS = [
    ("scripts/runs/band_structure.py", "bands", None),
    ("scripts/runs/v0_sweep/run_v0.py", "momentum", 12.0),
    ("scripts/runs/v0_sweep/run_v0_fixed_5.py", "momentum", 5.0),
    ("scripts/runs/v0_sweep/run_v0_fixed_10.py", "momentum", 10.0),
    ("examples/band_structure.py", "bands", None),
    ("examples/susceptibility.py", "susceptibility", None),
    ("examples/electron_phonon.py", "epw", None),
]


def invoke(script, arguments, tmp_path):
    env = dict(os.environ, MPLBACKEND="Agg", QUEER_RESULTS_DIR=str(tmp_path / "results"))
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(ROOT / script), *arguments],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return result


def momentum_arguments(run_dir):
    return [
        "--material-path",
        str(ROOT / "tests/fixtures/wannier/chain"),
        "--ef",
        "0",
        "--fermi-energy",
        "0",
        "--binding-min",
        "0",
        "--binding-max",
        ".2",
        "--binding-step",
        ".1",
        "--n-points",
        "4",
        "--factor",
        "1",
        "--num-core",
        "1",
        "--output-dir",
        str(run_dir),
    ]


@pytest.mark.parametrize("script,workflow,v0", CONSUMERS, ids=[Path(item[0]).stem for item in CONSUMERS])
def test_supported_legacy_consumer(script, workflow, v0, tmp_path):
    if script.startswith("examples/"):
        arguments = ["--results-dir", str(tmp_path / "example-results")]
    elif workflow == "momentum":
        arguments = momentum_arguments(tmp_path / "momentum-run")
        if Path(script).stem == "run_v0":
            arguments.extend(["--V0", str(v0)])
    else:
        arguments = [
            "--material-path",
            str(ROOT / "tests/fixtures/wannier/fcc"),
            "--ef",
            "0",
            "--kpath",
            "GAMMA-X",
            "--k-points",
            "4",
            "--num-core",
            "1",
            "--output",
            str(tmp_path / "bands.png"),
        ]
    invoke(script, arguments, tmp_path)
    manifests = [path for path in tmp_path.rglob("manifest.json") if path.parent.name != "figures"]
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["workflow"] == workflow and manifest["status"] == "completed"
    assert manifest["cases"] and manifest["config"]["execution"]["workers"] == 1
    assert list(tmp_path.rglob("*.png"))
    if v0 is not None:
        assert manifest["config"]["parameters"]["v0"] == v0
    if "fixed_" in script:
        # argparse's existing last-argument precedence lets callers override the preset.
        override_dir = tmp_path / "override-run"
        invoke(script, [*momentum_arguments(override_dir), "--V0", "13"], tmp_path)
        override = json.loads((override_dir / "manifest.json").read_text())
        assert override["config"]["parameters"]["v0"] == 13.0
        assert override["status"] == "completed"
