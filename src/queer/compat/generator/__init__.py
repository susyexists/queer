"""Generator compatibility: legacy parameters, payloads, estimators and runners.

Native results retain scientific units and absolute energies. The Generator's
historical sign, time, plotting-axis and index conventions apply only inside
this package. Importing the package loads nothing beyond the standard library,
so ``python -m queer generator`` can locate its runner before NumPy is checked.
The historical module-level names resolve lazily from their defining modules.
"""

from __future__ import annotations

import importlib as _importlib
import typing as _typing

_EXPORTS = {
    name: "_legacy"
    for name in (
        "ASE_FS",
        "Any",
        "BOHR_ANGSTROM",
        "HARTREE_EV",
        "Path",
        "atomic_json",
        "band_payload",
        "calculation_payload",
        "dataset_exports",
        "digest_file",
        "export_run",
        "np",
        "select_legacy_min_energy_frame",
        "structure_payload",
        "transform_structures",
        "translate_ase",
        "translate_pyatb",
    )
}
__all__ = sorted({*_EXPORTS, "annotations"})


def __getattr__(name: str) -> _typing.Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
