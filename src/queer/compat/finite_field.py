"""Checksum-verified finite-field artifact exports without recomputation."""

from __future__ import annotations

import shutil
from pathlib import Path

from ..io.artifacts import atomic_json, digest_file, load_manifest
from ..workflows.finite_field import FINITE_FIELD_NAMES


def export_run(run_dir, *, output_dir=None, profile="finite_field"):
    if profile != "finite_field":
        raise ValueError("Unsupported finite-field export profile")
    root = Path(run_dir).expanduser().resolve()
    manifest = load_manifest(root)
    if manifest["workflow"] not in FINITE_FIELD_NAMES:
        raise ValueError("finite_field export requires a finite-field workflow")
    target = Path(output_dir).expanduser().resolve() if output_dir else root / "exports" / "finite_field"
    if target == root or root in target.parents and target == root / "arrays":
        raise ValueError("Export output must be separate from the numerical run")
    target.mkdir(parents=True, exist_ok=True)
    record = manifest["cases"].get("result")
    if record is None:
        raise ValueError("Export requires a final result")
    shutil.copyfile(root / record["path"], target / "result.npz")
    atomic_json(
        target / "metadata.json",
        {
            "workflow": manifest["workflow"],
            "scientific_model": manifest.get("scientific_model", {}),
            "config": manifest["config"],
            "result": record["metadata"],
        },
    )
    products = ["result.npz", "metadata.json"]
    atomic_json(
        target / "manifest.json",
        {
            "schema_version": 1,
            "profile": profile,
            "source_fingerprint": manifest["fingerprint"],
            "products": [{"path": name, "sha256": digest_file(target / name)} for name in products],
        },
    )
    return target
