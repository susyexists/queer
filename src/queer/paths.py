"""Storage paths are independent of installed package locations."""

from __future__ import annotations

import os
from pathlib import Path

PathLike = str | os.PathLike[str]


def project_root(start: PathLike | None = None) -> Path:
    """Discover a source checkout for compatibility, falling back to cwd."""
    if root := os.environ.get("QUEER_PROJECT_ROOT"):
        return Path(root).expanduser().resolve()
    anchor = Path(start).expanduser().resolve() if start else Path.cwd()
    if anchor.is_file():
        anchor = anchor.parent
    for candidate in (anchor, *anchor.parents):
        if (candidate / "src" / "queer").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()


def data_root() -> Path:
    return Path(os.environ.get("QUEER_DATA_DIR", Path.cwd() / "data")).expanduser().resolve()


def results_root() -> Path:
    return Path(os.environ.get("QUEER_RESULTS_DIR", Path.cwd() / "results")).expanduser().resolve()


def data_path(*parts: PathLike) -> Path:
    return data_root().joinpath(*parts)


def results_path(*parts: PathLike) -> Path:
    return results_root().joinpath(*parts)


def ensure_parent(path: PathLike) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
