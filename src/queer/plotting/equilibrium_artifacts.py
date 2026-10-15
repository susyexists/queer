"""Render completed numerical artifacts without loading scientific models.

Optional plotting dependencies are loaded only when a renderer is called.
"""

from pathlib import Path

import numpy as np


def render(root, manifest, output) -> list[Path]:
    import matplotlib.pyplot as plt

    name = manifest["workflow"]
    files = []
    record = manifest["cases"].get("result")
    if record is None:
        raise ValueError(f"No final observable artifact for {name}")
    with np.load(root / record["path"], allow_pickle=False) as data:
        candidates = [
            key
            for key in data.files
            if data[key].dtype.kind in "fci"
            and key
            not in {
                "kpoints",
                "omega",
                "chemical_potentials",
                "temperatures",
                "groups",
                "band_indices",
                "band_channels",
                "polarization_quantum",
            }
        ]
        preferred = [name, "conductivity", "metric", "curvature", "energies", "electronic", "phases"]
        key = next(
            (value for value in preferred if value in candidates), candidates[0] if candidates else None
        )
        if key is None:
            raise ValueError(f"No numerical quantity to render for {name}")
        values = np.asarray(data[key])
        dimensions = record["metadata"].get("dims", {}).get(key, [])
        unit = record["metadata"].get("units", {}).get(key, "")
        fig, ax = plt.subplots()
        if not values.size:
            ax.text(0.5, 0.5, "No matching samples", ha="center", va="center", transform=ax.transAxes)
        elif values.ndim == 0:
            ax.bar([key], [float(values.real)])
        else:
            axis = next(
                (
                    i
                    for i, value in enumerate(dimensions)
                    if value in {"k", "kpoint", "omega", "energy", "chemical_potential", "temperature"}
                ),
                0,
            )
            quantity_rows = np.moveaxis(values, axis, 0).reshape(values.shape[axis], -1)
            coordinate = {
                "omega": "omega",
                "energy": "omega",
                "chemical_potential": "chemical_potentials",
                "temperature": "temperatures",
            }.get(dimensions[axis] if dimensions else "")
            if (
                coordinate not in data
                or data[coordinate].ndim != 1
                or len(data[coordinate]) != len(quantity_rows)
            ):
                coordinate = None
            x = np.arange(len(quantity_rows)) if coordinate is None else data[coordinate]
            ax.plot(x, quantity_rows.real)
            if np.iscomplexobj(quantity_rows) and np.any(quantity_rows.imag):
                ax.plot(x, quantity_rows.imag, linestyle="--")
            ax.set_xlabel(coordinate or (dimensions[axis] if dimensions else "sample"))
        ax.set_title(name.replace("_", " "))
        ax.set_ylabel(f"{key} ({unit})" if unit else key)
        target = output / f"{name}.png"
        fig.savefig(target, bbox_inches="tight")
        plt.close(fig)
        files.append(target)
    return files
