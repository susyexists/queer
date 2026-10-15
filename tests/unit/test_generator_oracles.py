"""The vendored Generator oracles are pinned, isolated and never collected as tests."""

import hashlib
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORACLES = ROOT / "tests" / "oracles"


def recorded_hashes():
    table = (ORACLES / "README.md").read_text()
    return dict(re.findall(r"^\| `([^`]+\.py)` \| [^|]+ \| `([0-9a-f]{64})` \|$", table, re.MULTILINE))


def test_every_vendored_file_matches_its_recorded_sha256():
    recorded = recorded_hashes()
    vendored = {path.name for path in ORACLES.glob("*.py")}
    assert set(recorded) == vendored == {"generator_pythtb_core.py", "generator_pythtb_qgeom.py"}
    for name, digest in recorded.items():
        assert hashlib.sha256((ORACLES / name).read_bytes()).hexdigest() == digest, name


def test_oracles_load_by_path_without_pythtb_scipy_or_matplotlib():
    probe = f"""
import importlib.util
import sys
for name in ("pythtb", "scipy", "matplotlib"):
    sys.modules[name] = None
spec = importlib.util.spec_from_file_location("oracle_conftest", {str(ROOT / "tests/conftest.py")!r})
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
core = helpers.load_oracle("generator_pythtb_core")
qgeom = helpers.load_oracle("generator_pythtb_qgeom")
assert helpers.load_oracle("generator_pythtb_core") is core
assert callable(core.wilson_loop) and callable(core.fermi_sea_curvature) and callable(qgeom.geometry)
assert not any(name in sys.modules and sys.modules[name] is not None for name in ("pythtb", "queer"))
"""
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    subprocess.run([sys.executable, "-c", probe], check=True, capture_output=True, env=environment)


def test_oracles_are_excluded_from_lint_and_type_checks_but_not_collected():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "tests/oracles" in config["tool"]["ruff"]["extend-exclude"]
    assert any(
        re.search(pattern, "tests/oracles/generator_pythtb_core.py")
        for pattern in config["tool"]["mypy"]["exclude"]
    )
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(ORACLES)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert collected.returncode == 5, collected.stdout + collected.stderr
