"""Typed, pickle-free observable arrays and their scientific interpretation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ObservableResult:
    """Arrays with explicit axis names, units, and JSON-compatible metadata.

    These mappings are directly usable by Queer's NPZ/manifest storage. Cartesian
    tensor axes follow scan axes; band energies retain the public ``(Nb,Nk)``
    convention. Individual arrays may include NaNs when accompanied by a mask.
    """

    arrays: Mapping[str, np.ndarray]
    dims: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arrays = {key: np.asarray(value) for key, value in self.arrays.items()}
        if any(array.dtype.hasobject for array in arrays.values()):
            raise ValueError("Observable arrays must not contain Python objects")
        for key, axes in self.dims.items():
            if key not in arrays or len(axes) != arrays[key].ndim:
                raise ValueError(f"Dimensions for {key!r} must match its array rank")
        if set(self.units) - set(arrays):
            raise ValueError("Units must refer to an observable array")
        json.dumps(dict(self.metadata), allow_nan=False)
        object.__setattr__(self, "arrays", arrays)
        object.__setattr__(self, "dims", {key: tuple(value) for key, value in self.dims.items()})
        object.__setattr__(self, "units", dict(self.units))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]

    def artifact_metadata(self) -> dict[str, Any]:
        return {"dims": dict(self.dims), "units": dict(self.units), **self.metadata}
