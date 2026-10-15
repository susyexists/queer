"""Checked assembly of independently computed member results."""

import numpy as np


def stack_member_results(results, *, shared):
    """Keep shared coordinates and stack compatible member-dependent arrays."""
    if not results:
        raise ValueError("Member assembly requires at least one result")
    first = results[0]
    for result in results[1:]:
        if (
            set(result.arrays) != set(first.arrays)
            or result.dims != first.dims
            or result.units != first.units
        ):
            raise ValueError("Member schemas, dimensions, or units differ")
        if result.metadata.get("source_sha256") != first.metadata.get("source_sha256"):
            raise ValueError("Member engine revisions differ")
        for key, value in first.arrays.items():
            other = result[key]
            if value.shape != other.shape or value.dtype != other.dtype:
                raise ValueError(f"Member array shapes or dtypes differ: {key}")
            if key in shared and not np.array_equal(value, other):
                raise ValueError(f"Member coordinate grids differ: {key}")
    arrays, dims = {}, {}
    for key, value in first.arrays.items():
        axes = first.dims.get(key, tuple(f"axis{i}" for i in range(value.ndim)))
        if key in shared:
            arrays[key], dims[key] = value, first.dims.get(key, ())
        else:
            arrays[key], dims[key] = np.stack([result[key] for result in results]), ("member", *axes)
    return arrays, dims, dict(first.units)
