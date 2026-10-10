"""Compatibility imports for queer.observables.response."""

from .observables.response import (
    green_function,
    spectral_from_eig,
    spectral_from_green,
    susceptibility,
    susceptibility_point,
)

__all__ = [
    "green_function",
    "spectral_from_green",
    "spectral_from_eig",
    "susceptibility",
    "susceptibility_point",
]
