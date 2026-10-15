"""Adapter for the historical RunStore-based quadrature checkpoint scope."""

from contextlib import contextmanager

from ..io.results import RunStoreCheckpoint


@contextmanager
def quadrature_checkpoint(store):
    """Translate a historical run store into a storage-independent checkpoint."""
    from ..observables.integration import _CHECKPOINT

    checkpoint = store if hasattr(store, "load") else RunStoreCheckpoint(store)
    token = _CHECKPOINT.set(checkpoint)
    try:
        yield
    finally:
        _CHECKPOINT.reset(token)
