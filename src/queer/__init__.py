"""Scientific models and workflows, with lazy 1.x compatibility exports."""

import sys as _sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType as _ModuleType

try:
    __version__ = version("queer")
except PackageNotFoundError:
    __version__ = "1.1.0"

_EXPORTS = {
    "WannierModel": (".models.wannier", "WannierModel"),
    "EPWModel": (".models.epw", "EPWModel"),
    "ExecutionOptions": (".execution", "ExecutionOptions"),
    "KPath": (".kpath", "KPath"),
    "model": (".model", "model"),
    "epw": (".epw", "epw"),
}
for _name in (
    "dotdict",
    "read_data",
    "read_parallel",
    "hexagon",
    "suscep_epw",
    "read_poscar",
    "crystal2reciprocal",
    "write_data",
    "path_create",
    "read_gvec",
    "fermi_bands",
    "ph_cross",
    "g_cross",
    "untangle",
    "phselfen",
):
    _EXPORTS[_name] = (".utils", _name)
_MODULES = {"plots", "functions", "paths", "kpath", "momentum", "spectral"}
__all__ = ["__version__", *_EXPORTS]


def __getattr__(name):
    if name in _EXPORTS:
        module, attribute = _EXPORTS[name]
        value = getattr(import_module(module, __name__), attribute)
    elif name in _MODULES:
        value = import_module(f".{name}", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS) | _MODULES)


# Python attaches imported child modules to their parent. Historically these two
# package attributes are callable classes; preserve that contract regardless of
# whether a consumer first imports queer.model/queer.epw or accesses the facade.


class _CompatibilityModule(_ModuleType):
    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        if name in {"model", "epw"} and isinstance(value, _ModuleType):
            value = getattr(value, name)
            self.__dict__[name] = value
        return value


_sys.modules[__name__].__class__ = _CompatibilityModule
