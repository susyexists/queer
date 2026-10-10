"""Legacy metadata readers; prefer queer.io.epw.read_epw_metadata."""

from .io.epw import read_epw_metadata


def _field(path, name):
    metadata = read_epw_metadata(path)
    if name not in metadata:
        raise ValueError(f"missing EPW {name} header in {path}")
    return metadata[name]


def get_nk(path):
    return _field(path, "nk")


def get_nq(path):
    return _field(path, "nq")


def get_nph(path):
    return _field(path, "nph")


def get_ef(path):
    return _field(path, "ef")
