"""The numeric environment a run computed in, and whether two runs shared one.

Two runs of the adaptive quantum-geometry integrator on an exactly symmetric model
break its exact ties by rounding. The same numpy and BLAS builds on the same kernel
round alike (the Generator driver and QUEER give identical meshes there), while
another OpenBLAS core type or another build rounds otherwise. A run therefore records
what decides its rounding: the Python, numpy and SciPy versions, the machine, the
SIMD extensions numpy dispatches to, and every BLAS library loaded with its build,
version, core type and threading layer. The thread count is kept for the reader but
is not compared: 1, 8 and 16 OpenBLAS threads give one mesh on these models, and a
run changes its count inside ``threadpool_limits``. ``OPENBLAS_CORETYPE`` is kept as
set, for the reader; its effect is the libraries' ``architecture``.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Mapping
from typing import Any

# The fields that decide rounding; two runs share an environment when all are recorded and equal.
ENVIRONMENT_FIELDS = ("python", "numpy", "scipy", "machine", "simd", "blas")
BLAS_FIELDS = ("internal_api", "library", "version", "architecture", "threading_layer")


def _simd() -> list[str] | None:
    """The SIMD extensions numpy dispatches to on this machine (numpy 2 and 1.26 module paths)."""
    import importlib

    for name in ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        dispatch = getattr(module, "__cpu_dispatch__", None)
        features = getattr(module, "__cpu_features__", None)
        if dispatch is not None and features is not None:
            return sorted(str(item) for item in dispatch if features.get(item))
    return None


def _blas() -> list[dict[str, Any]] | None:
    try:
        import scipy.linalg  # noqa: F401  (loads SciPy's own BLAS, so it is listed)
        from threadpoolctl import threadpool_info
    except ImportError:
        return None
    libraries = []
    for entry in threadpool_info():
        if entry.get("user_api") != "blas":
            continue
        libraries.append(
            {
                "internal_api": entry.get("internal_api"),
                "library": os.path.basename(str(entry.get("filepath") or "")),
                "version": entry.get("version"),
                "architecture": entry.get("architecture"),
                "threading_layer": entry.get("threading_layer"),
                "num_threads": entry.get("num_threads"),
            }
        )
    return sorted(libraries, key=lambda library: str(library["library"]))


def numeric_environment() -> dict[str, Any]:
    """What decides this process's floating-point rounding."""
    import numpy as np

    try:
        from importlib.metadata import version

        scipy_version: str | None = version("scipy")
    except Exception:
        scipy_version = None
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy_version,
        "machine": platform.machine(),
        "simd": _simd(),
        "blas": _blas(),
        "openblas_coretype": os.environ.get("OPENBLAS_CORETYPE"),
    }


def recorded_environment(document: Mapping[str, Any]) -> dict[str, Any]:
    """A results document's environment: its record, or the versions a driver document lists."""
    provenance = document.get("provenance") or {}
    record = provenance.get("numeric_environment")
    if isinstance(record, Mapping):
        return dict(record)
    return {
        key: provenance.get(key) for key in ("python", "numpy", "scipy") if provenance.get(key) is not None
    }


def environment_differences(a: Mapping[str, Any], b: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """The fields two environments record differently, and the fields either leaves unrecorded."""
    differing, unrecorded = [], []
    for field in ENVIRONMENT_FIELDS:
        x, y = a.get(field), b.get(field)
        if x is None or y is None:
            missing = [role for role, value in (("reference", x), ("candidate", y)) if value is None]
            unrecorded.append(f"{field} ({' and '.join(missing)})")
        elif field == "blas":
            if [_blas_key(item) for item in x] != [_blas_key(item) for item in y]:
                differing.append(f"BLAS {_blas_text(x)} vs {_blas_text(y)}")
        elif field == "simd":
            if list(x) != list(y):
                only_a, only_b = sorted(set(x) - set(y)), sorted(set(y) - set(x))
                differing.append(
                    f"SIMD extensions {only_a} only in the reference, {only_b} only in the candidate"
                )
        elif x != y:
            differing.append(f"{field} {x} vs {y}")
    return differing, unrecorded


def _blas_key(item: Any) -> tuple[Any, ...]:
    return tuple(item.get(field) for field in BLAS_FIELDS) if isinstance(item, Mapping) else (item,)


def _blas_text(libraries: Any) -> str:
    parts = []
    for item in libraries or []:
        if isinstance(item, Mapping):
            parts.append(f"{item.get('library')} {item.get('version')} {item.get('architecture')}")
    return "[" + "; ".join(parts) + "]"
