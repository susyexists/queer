"""Storage-independent checkpoints for deterministic numerical evaluations."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

import numpy as np


class Checkpoint(Protocol):
    """An absent record returns None; an invalid record raises an exception."""

    def load(self, case: str) -> tuple[Mapping[str, np.ndarray], Mapping[str, Any]] | None: ...

    def save(self, case: str, arrays: Mapping[str, Any], metadata: Mapping[str, Any]) -> None: ...
