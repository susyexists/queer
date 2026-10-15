"""Artifact-only rendering handlers; optional plotting loads at render time."""

from pathlib import Path

import numpy as np


def render(root, manifest, output) -> list[Path]:
    name = manifest["workflow"]
    files = []
    from .finite_field import render_finite_field

    record = manifest["cases"].get("result")
    if record is None:
        raise ValueError("Finite-field rendering requires a final result")
    with np.load(root / record["path"], allow_pickle=False) as data:
        if record["metadata"].get("figure"):
            from .paper_figures import render_paper_figure

            arrays = {key: data[key] for key in data.files}
            if "render_orders" in record["metadata"]:
                orders = np.asarray(record["metadata"]["render_orders"])
                harmonic_indices = [int(np.flatnonzero(arrays["orders"] == order)[0]) for order in orders]
                for coefficient in ("a", "b", "c"):
                    arrays[coefficient] = arrays[coefficient][..., harmonic_indices]
                arrays["orders"] = orders
            files = render_paper_figure(arrays, record["metadata"], output)
        else:
            files = render_finite_field(
                name, {key: data[key] for key in data.files}, record["metadata"], output
            )
    return [output / path for path in files]
