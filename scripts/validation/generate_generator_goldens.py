"""Run the Generator PythTB driver on the analytic golden cases and keep its small outputs.

Each case of ``tests/numerical/generator_golden_cases.py`` is staged in a scratch
directory with the vendored driver (``tests/oracles/generator_pythtb_core.py``,
staged as ``_pythtb_core.py`` as the Generator stages it) and run as the
Generator runs it, ``python _pythtb_core.py pythtb-run --params
pythtb_params.json``, under

    uv run --no-project --python 3.12 --with pythtb==2.0.2 --with scipy --with matplotlib

(``--offline`` by default: from the uv cache). The driver's ``pythtb_results.json``
and its ``Out/*.json`` data files are copied to ``<output>/<case>/``; the inputs
are not, since the case module rebuilds them byte for byte, and the figures and
log are dropped. The machine's host name in the provenance is replaced by
``golden``. The PythTB environment never enters QUEER's own environment.

    python scripts/validation/generate_generator_goldens.py [--output DIR] [--case NAME ...] [--online]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "tests" / "oracles" / "generator_pythtb_core.py"
DEFAULT_OUTPUT = ROOT / "tests" / "fixtures" / "generator_goldens"
UV_ENVIRONMENT = ["--no-project", "--python", "3.12", "--with", "pythtb==2.0.2", "--with", "scipy"]
UV_ENVIRONMENT += ["--with", "matplotlib"]
HOST = "golden"


def load_cases():
    sys.path.insert(0, str(ROOT / "tests" / "numerical"))
    import generator_golden_cases

    return generator_golden_cases


def run_driver(workdir: Path, *, offline: bool, uv: str) -> subprocess.CompletedProcess:
    command = [uv, "run", *(["--offline"] if offline else []), *UV_ENVIRONMENT]
    command += ["python", "_pythtb_core.py", "pythtb-run", "--params", "pythtb_params.json"]
    environment = dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    return subprocess.run(command, cwd=workdir, env=environment, capture_output=True, text=True, timeout=1800)


def keep_outputs(workdir: Path, target: Path) -> list[Path]:
    """Copy the results document (host name replaced) and the Out/*.json files to ``target``."""
    document = json.loads((workdir / "pythtb_results.json").read_text())
    if document.get("status") != "ok":
        raise RuntimeError(f"the driver failed: {document.get('error')}")
    document.setdefault("provenance", {})["host"] = HOST
    if target.exists():
        shutil.rmtree(target)
    (target / "Out").mkdir(parents=True)
    kept = [target / "pythtb_results.json"]
    kept[0].write_text(json.dumps(document, indent=1) + "\n")
    for source in sorted((workdir / "Out").glob("*.json")):
        shutil.copyfile(source, target / "Out" / source.name)
        kept.append(target / "Out" / source.name)
    return kept


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case", action="append", dest="cases", help="a case name (default: every case)")
    parser.add_argument("--online", action="store_true", help="let uv download missing packages")
    parser.add_argument("--uv", default=shutil.which("uv") or "uv")
    options = parser.parse_args(argv)
    cases = load_cases()
    names = options.cases or sorted(cases.CASES)
    unknown = sorted(set(names) - set(cases.CASES))
    if unknown:
        parser.error(f"unknown cases: {', '.join(unknown)}")
    total = 0
    for name in names:
        with tempfile.TemporaryDirectory(prefix=f"golden-{name}-") as scratch:
            workdir = Path(scratch)
            cases.write_case(name, workdir)
            shutil.copyfile(DRIVER, workdir / "_pythtb_core.py")
            result = run_driver(workdir, offline=not options.online, uv=options.uv)
            if result.returncode != 0:
                print(result.stdout[-2000:], result.stderr[-2000:], sep="\n", file=sys.stderr)
                print(f"{name}: the driver exited {result.returncode}", file=sys.stderr)
                return 1
            kept = keep_outputs(workdir, options.output / name)
        size = sum(path.stat().st_size for path in kept)
        total += size
        print(f"{name}: {len(kept)} files, {size / 1024:.1f} KiB")
    print(f"total {total / 1024:.1f} KiB in {options.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
