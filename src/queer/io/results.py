"""Read and restore numerical results without reconstructing scientific models."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..observables.result import ObservableResult


def read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def restore_result(arrays, metadata):
    return ObservableResult(
        arrays,
        metadata.get("dims", {}),
        metadata.get("units", {}),
        {key: value for key, value in metadata.items() if key not in {"dims", "units"}},
    )


class RunStoreCheckpoint:
    """Adapt an owner-controlled RunStore to the numerical checkpoint protocol."""

    def __init__(self, store):
        self.store = store

    def load(self, case: str):
        if not self.store.completed(case):
            return None
        record = self.store.manifest["cases"][case]
        return read_arrays(self.store.root / record["path"]), dict(record["metadata"])

    def save(self, case: str, arrays: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
        self.store.commit(case, arrays, metadata=metadata)
