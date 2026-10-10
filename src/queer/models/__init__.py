"""Model representations independent of visualization and workflow storage."""

from importlib import import_module

__all__ = ["WannierModel", "EPWModel"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    module = ".wannier" if name == "WannierModel" else ".epw"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
