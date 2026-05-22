"""Project path helpers for local workstations and HPC runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, os.PathLike]


def project_root(start: Optional[PathLike] = None) -> Path:
    """Return the repository root, honoring QUEER_PROJECT_ROOT when set."""
    env_root = os.environ.get("QUEER_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    anchor = Path(start).expanduser().resolve() if start else Path(__file__).resolve()
    if anchor.is_file():
        anchor = anchor.parent

    for candidate in (anchor, *anchor.parents):
        if (candidate / "queer").is_dir() and (
            (candidate / "setup.py").exists() or (candidate / "pyproject.toml").exists()
        ):
            return candidate
        if (candidate / ".git").exists():
            return candidate

    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    """Return the data root, using QUEER_DATA_DIR when provided."""
    return Path(os.environ.get("QUEER_DATA_DIR", project_root() / "data")).expanduser().resolve()


def results_root() -> Path:
    """Return the results root, using QUEER_RESULTS_DIR when provided."""
    return Path(os.environ.get("QUEER_RESULTS_DIR", project_root() / "results")).expanduser().resolve()


def data_path(*parts: PathLike) -> Path:
    return data_root().joinpath(*map(Path, parts))


def results_path(*parts: PathLike) -> Path:
    return results_root().joinpath(*map(Path, parts))


def ensure_parent(path: PathLike) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
