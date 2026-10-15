"""Compatibility entrypoints and conservative finite-field engine fingerprint."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..io.numerical import json_safe as json_safe
from ..io.numerical import load_npz as load_npz
from ..io.numerical import save_npz as save_npz
from ..io.numerical import stable_hash as stable_hash


def source_files():
    """All current numerical owners; donor provenance is kept independently."""
    root = Path(__file__).parent.parent
    paths = set()
    for package in ("physics", "models", "geometry", "observables", "finite_field", "io", "workflows"):
        paths.update((root / package).rglob("*.py"))
    paths.update(root / name for name in ("capabilities.py", "execution.py", "structures.py", "elements.py"))
    return sorted(paths)


def source_hash():
    root = Path(__file__).parent.parent
    digest = hashlib.sha256()
    for path in source_files():
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
