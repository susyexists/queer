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

A ``PYTHTB_QGEOM`` case also stages the vendored kernel
(``tests/oracles/generator_pythtb_qgeom.py`` as ``_pythtb_qgeom.py``, as the
Generator stages it) and runs the driver through a small wrapper that keeps the
kernel's integrator and writes its final mesh's vertex count and the SHA-256 of
its sorted vertex keys to ``Out/qgeom_vertices.json`` (QUEER's replay record; the
driver records only counts), so a QUEER replay can be checked vertex for vertex. A
case that expects a failure keeps the driver's failure document when its code is
the expected one.

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
KERNEL = ROOT / "tests" / "oracles" / "generator_pythtb_qgeom.py"
QGEOM_STEP = "PYTHTB_QGEOM"
# Runs the driver in its own environment and records the kernel's final mesh vertices.
QGEOM_WRAPPER = """import hashlib, json, sys
import numpy as np
import _pythtb_qgeom as kernel

captured = {}
adaptive_response = kernel.adaptive_response


def capture(*args, **kwargs):
    result = adaptive_response(*args, **kwargs)
    captured["integrator"] = result["integrator"]
    return result


kernel.adaptive_response = capture
import _pythtb_core as core

code = core.run("pythtb_params.json")
integ = captured.get("integrator")
if integ is not None:
    mesh = integ.mesh
    keys = np.sort(np.asarray(mesh._buf["_key"][: mesh.nv], np.int64)).astype("<i8")
    record = {
        "description": "Vertex keys x * unit + y of the final quadtree, x and y in units of 1/unit from -1/2",
        "count": int(len(keys)),
        "base": int(mesh.base),
        "unit": int(mesh.unit),
        "sha256": hashlib.sha256(keys.tobytes()).hexdigest(),
        "keys": None,
        "recorded_by": "scripts/validation/generate_generator_goldens.py",
    }
    with open("Out/qgeom_vertices.json", "w") as handle:
        json.dump(record, handle, separators=(",", ":"))
sys.exit(code)
"""
DEFAULT_OUTPUT = ROOT / "tests" / "fixtures" / "generator_goldens"
UV_ENVIRONMENT = ["--no-project", "--python", "3.12", "--with", "pythtb==2.0.2", "--with", "scipy"]
UV_ENVIRONMENT += ["--with", "matplotlib"]
HOST = "golden"


def load_cases():
    sys.path.insert(0, str(ROOT / "tests" / "numerical"))
    import generator_golden_cases

    return generator_golden_cases


def run_driver(workdir: Path, *, offline: bool, uv: str, qgeom: bool = False) -> subprocess.CompletedProcess:
    command = [uv, "run", *(["--offline"] if offline else []), *UV_ENVIRONMENT]
    if qgeom:
        command += ["python", "_golden_qgeom.py"]
    else:
        command += ["python", "_pythtb_core.py", "pythtb-run", "--params", "pythtb_params.json"]
    environment = dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    return subprocess.run(command, cwd=workdir, env=environment, capture_output=True, text=True, timeout=1800)


def keep_outputs(workdir: Path, target: Path, expect: str = "ok") -> list[Path]:
    """Copy the results document (host name replaced) and the Out/*.json files to ``target``.

    ``expect`` is "ok" or the failure code the case must end with.
    """
    document = json.loads((workdir / "pythtb_results.json").read_text())
    if expect == "ok" and document.get("status") != "ok":
        raise RuntimeError(f"the driver failed: {document.get('error')}")
    if expect != "ok" and (document.get("error") or {}).get("code") != expect:
        raise RuntimeError(
            f"expected the failure {expect}, got {document.get('status')}: {document.get('error')}"
        )
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
            case = cases.CASES[name]
            qgeom = case.step == QGEOM_STEP
            shutil.copyfile(DRIVER, workdir / "_pythtb_core.py")
            if qgeom:
                shutil.copyfile(KERNEL, workdir / "_pythtb_qgeom.py")
                (workdir / "_golden_qgeom.py").write_text(QGEOM_WRAPPER)
            result = run_driver(workdir, offline=not options.online, uv=options.uv, qgeom=qgeom)
            expected_exit = 0 if case.expect == "ok" else 2
            if result.returncode != expected_exit:
                print(result.stdout[-2000:], result.stderr[-2000:], sep="\n", file=sys.stderr)
                print(f"{name}: the driver exited {result.returncode}", file=sys.stderr)
                return 1
            kept = keep_outputs(workdir, options.output / name, case.expect)
        size = sum(path.stat().st_size for path in kept)
        total += size
        print(f"{name}: {len(kept)} files, {size / 1024:.1f} KiB")
    print(f"total {total / 1024:.1f} KiB in {options.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
