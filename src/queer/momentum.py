"""Legacy DataFrame wrappers; new numerical workflows use observables.momentum."""

from .observables import momentum as _numeric
from .observables.momentum import fermi_surface_points, format_time
from .plotting.momentum import save_surface_plots, write_animation

__all__ = [
    "format_time",
    "binding_energy_surfaces",
    "sweep_binding_energy_surfaces",
    "fermi_surface_points",
    "save_surface_plots",
    "write_animation",
]


def _frames(surfaces):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "legacy momentum DataFrames require queer[plot]; use queer.observables.momentum for arrays"
        ) from exc
    return {
        energy: pd.DataFrame(array, columns=_numeric.SURFACE_COLUMNS) for energy, array in surfaces.items()
    }


def binding_energy_surfaces(*args, **kwargs):
    return _frames(_numeric.binding_energy_surfaces(*args, **kwargs))


def sweep_binding_energy_surfaces(*args, **kwargs):
    return {
        key: _frames(surfaces)
        for key, surfaces in _numeric.sweep_binding_energy_surfaces(*args, **kwargs).items()
    }
