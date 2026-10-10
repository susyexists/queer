"""Compatibility imports for spectral calculations and optional plotting."""

from .observables.spectral import (
    iter_spectral_arpes_path_sweep,
    spectral_along_path,
    spectral_arpes_path_sweep,
)
from .plotting.spectral import plot_spectral_path

__all__ = [
    "spectral_along_path",
    "spectral_arpes_path_sweep",
    "iter_spectral_arpes_path_sweep",
    "plot_spectral_path",
]
