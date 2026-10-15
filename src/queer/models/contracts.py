"""Leaf contracts for native eigenstates and capability-specific evaluation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from typing import Any, Protocol

import numpy as np


@dataclass
class EigenstateBatch:
    """Absolute E(B,K), coefficients C(K,N,B), metric S(K,N,N), channels(B)."""

    energies: np.ndarray
    coefficients: np.ndarray
    overlaps: np.ndarray
    band_channels: np.ndarray

    def __post_init__(self):
        # The historical Wannier namespace stored two initially aliased but
        # independently assignable fields. Keep that behavior in the shared type.
        self.__dict__["absolute_energies"] = self.energies

    @property
    def absolute_energies(self):
        if "absolute_energies" in self.__dict__:
            return self.__dict__["absolute_energies"]
        return self.energies

    @absolute_energies.setter
    def absolute_energies(self, value):
        self.__dict__["absolute_energies"] = value


class FrozenEigenstateBatch(EigenstateBatch):
    """OpenMX's historical frozen fields; contained numerical arrays stay mutable.

    Wannier batches retain assignable fields, including NumPy augmented assignment.
    Both variants implement the same native eigenstate contract.
    """

    def __init__(self, energies, coefficients, overlaps, band_channels):
        object.__setattr__(self, "energies", energies)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "overlaps", overlaps)
        object.__setattr__(self, "band_channels", band_channels)

    def __setattr__(self, name, value):
        raise FrozenInstanceError(f"cannot assign to field {name!r}")

    def __delattr__(self, name):
        raise FrozenInstanceError(f"cannot delete field {name!r}")


class EigenstateProvider(Protocol):
    nbnd: int
    fermi_energy: float

    def eigenstates(self, kpoints, bands=None, execution=None) -> EigenstateBatch: ...


class GeometryOperators(Protocol):
    """Fractional input points, Cartesian derivatives; availability is checked separately."""

    def hamiltonian(self, k) -> np.ndarray: ...
    def overlap(self, k) -> np.ndarray: ...
    def position(self, k) -> np.ndarray: ...
    def derivatives(self, k, order=1) -> dict[str, Any]: ...
