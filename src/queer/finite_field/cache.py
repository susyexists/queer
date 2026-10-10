"""Checked, immutable preparation artifacts shared by Fourier/field scans."""

from __future__ import annotations

import fcntl
from pathlib import Path

from ..io.artifacts import atomic_json, digest_file
from ._storage import load_npz, save_npz


class PreparationCache:
    """One locked preparation identity; numerical outputs stay pickle-free.

    The identity is content based and includes effective preparation options.
    A corrupt entry is an error, never a source of silently reused physics.
    """

    def __init__(self, root, fingerprint):
        if len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
            raise ValueError("preparation fingerprint must be a SHA256 hexadecimal digest")
        self.fingerprint = fingerprint
        self.root = Path(root).expanduser().resolve() / fingerprint[:2] / fingerprint
        self._lock = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = (self.root / ".cache.lock").open("a+")
        fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *args):
        if self._lock is not None:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None

    def _get(self, name):
        import json

        data, record = self.root / f"{name}.npz", self.root / f"{name}.json"
        if not record.exists():
            return None
        info = json.loads(record.read_text())
        if info.get("identity") != self.fingerprint or info.get("schema") != "queer.preparation.v1":
            raise ValueError("preparation cache identity/schema mismatch")
        if not data.is_file() or digest_file(data) != info.get("sha256"):
            raise ValueError(f"preparation cache checksum mismatch: {data}")
        return load_npz(data)

    def _put(self, name, arrays, metadata):
        if self._get(name) is not None:
            return
        data = self.root / f"{name}.npz"
        save_npz(data, arrays, metadata)
        atomic_json(
            self.root / f"{name}.json",
            {"schema": "queer.preparation.v1", "identity": self.fingerprint, "sha256": digest_file(data)},
        )

    def get(self, index, offset):
        from .api import decode_line

        packed = self._get(f"line_{index:06d}")
        if packed is None:
            return None
        arrays, metadata = packed
        if metadata["offset"] != float(offset):
            raise ValueError("preparation cache transverse-layout mismatch")
        return decode_line(arrays, metadata)

    def put(self, index, offset, line):
        from .api import encode_line

        if line.t != float(offset):
            raise ValueError("line offset does not match preparation cache case")
        arrays, metadata = encode_line(line)
        self._put(f"line_{index:06d}", arrays, metadata)

    def get_layout(self):
        from .transport.hall import TransverseLayout

        packed = self._get("layout")
        if packed is None:
            return None
        arrays, metadata = packed
        return TransverseLayout(
            arrays["nodes"],
            arrays["weights"],
            arrays["breaks"],
            tuple(metadata["t_max"]),
            metadata["mirrored"],
            set(metadata["flags"]),
        )

    def put_layout(self, layout):
        self._put(
            "layout",
            {"nodes": layout.t_nodes, "weights": layout.t_weights, "breaks": layout.breaks},
            {"t_max": list(layout.t_max), "mirrored": bool(layout.mirrored), "flags": sorted(layout.flags)},
        )
