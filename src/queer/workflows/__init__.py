"""Reusable scientific workflows with reproducible numerical run artifacts."""

from importlib import import_module

__all__ = ["run_workflow", "render_run"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    module = ".runner" if name == "run_workflow" else ".render"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
