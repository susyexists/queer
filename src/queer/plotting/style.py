"""Bundled styles with scoped matplotlib state."""

from contextlib import contextmanager
from importlib.resources import as_file, files


@contextmanager
def style_context(name="neon"):
    """Apply a bundled style for a block, restoring the caller's settings."""
    if name != "neon":
        raise ValueError(f"Unknown bundled style {name!r}")
    import matplotlib.style

    resource = files("queer").joinpath("resources", f"{name}.mplstyle")
    with as_file(resource) as path, matplotlib.style.context(str(path)):
        yield
