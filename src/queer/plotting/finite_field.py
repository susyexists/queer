"""Finite-field plots built entirely from persisted numerical products."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def _key(arrays, *candidates):
    for candidate in candidates:
        if candidate in arrays:
            return candidate
    skip = {
        "fields",
        "amplitudes",
        "harmonics",
        "orders",
        "parameter",
        "labels",
        "offsets",
        "kpoints",
        "r",
        "phi",
        "kx",
        "ky",
    }
    return next(
        (key for key, value in arrays.items() if key not in skip and np.asarray(value).dtype.kind in "fci"),
        None,
    )


def render_finite_field(name, arrays, metadata, output):
    """Return filenames; plotting never imports a model or response engine."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    artist: Any
    if name == "family_report":
        ax.axis("off")
        ax.text(
            0,
            1,
            json.dumps(metadata, indent=2),
            va="top",
            transform=ax.transAxes,
            family="monospace",
            fontsize=8,
        )
    elif name == "geometry_grid":
        key = _key(arrays, "omega_c", "gap", "eps_c", "g_cv")
        values = np.asarray(arrays[key])
        if values.ndim == 3:
            values = values[..., 0]
        if "r" in arrays:
            radius, angle = np.meshgrid(arrays["r"], arrays["phi"], indexing="ij")
            artist = ax.scatter(
                (radius * np.cos(angle)).ravel(),
                (radius * np.sin(angle)).ravel(),
                c=values.ravel(),
                s=5,
                cmap="coolwarm",
            )
        else:
            artist = ax.pcolormesh(arrays["kx"], arrays["ky"], values.T, shading="auto", cmap="coolwarm")
        ax.set(xlabel="kx (Å⁻¹)", ylabel="ky (Å⁻¹)", aspect="equal")
        fig.colorbar(artist, ax=ax, label=key)
    elif name == "harmonic_map":
        key = _key(arrays, "b", "sine", "coefficients")
        values = np.asarray(arrays[key])
        if values.ndim == 3:
            order = 3 if values.shape[-1] > 3 else 1
            values = values[..., order]
            ax.set_title(f"Harmonic {order}")
        artist = ax.pcolormesh(
            arrays["amplitudes"], arrays["parameter"], values.real, shading="auto", cmap="coolwarm"
        )
        ax.set(xlabel="Field amplitude (V/m)", ylabel="Family parameter")
        fig.colorbar(artist, ax=ax, label=key)
    elif name == "hall_harmonics":
        key = _key(arrays, "b", "sine", "coefficients")
        values = np.asarray(arrays[key])
        x = np.asarray(arrays["amplitudes"])
        rows = values.reshape(len(x), -1)
        for order in range(1, rows.shape[1], 2):
            ax.plot(x, rows[:, order].real, label=f"N={order}")
        ax.set(xlabel="Field amplitude (V/m)", ylabel=f"{key} ({metadata.get('units', {}).get(key, 'A/m')})")
        if rows.shape[1] > 1:
            ax.legend()
    else:
        key = _key(
            arrays,
            "current",
            "J",
            "probability",
            "population",
            "P",
            "critical_field",
            "ddp_field",
            "gap",
            "fit",
            "smooth",
            "flux",
        )
        if key is None:
            ax.axis("off")
            ax.text(0, 1, json.dumps(metadata, indent=2), va="top", transform=ax.transAxes, fontsize=8)
        else:
            values = np.asarray(arrays[key])
            if values.ndim == 0:
                ax.bar([key], [values.real.item()])
            else:
                dims = metadata.get("dims", {}).get(key, [])
                sample_axis = next(
                    (
                        i
                        for i, dim in enumerate(dims)
                        if dim in {"field", "amplitude", "offset", "line", "sample"}
                    ),
                    0,
                )
                rows = np.moveaxis(values, sample_axis, 0).reshape(values.shape[sample_axis], -1)
                xkey = next(
                    (
                        candidate
                        for candidate in ("fields", "amplitudes", "offsets", "s", "time")
                        if candidate in arrays and np.asarray(arrays[candidate]).shape == (len(rows),)
                    ),
                    None,
                )
                x = arrays[xkey] if xkey else np.arange(len(rows))
                ax.plot(x, rows.real)
                if np.iscomplexobj(rows) and np.any(rows.imag):
                    ax.plot(x, rows.imag, linestyle="--")
                ax.set_xlabel(f"{xkey} ({metadata.get('units', {}).get(xkey, '')})" if xkey else "Sample")
            ax.set_ylabel(f"{key} ({metadata.get('units', {}).get(key, '')})")
    if name != "harmonic_map":
        ax.set_title(name.replace("_", " "))
    target = output / f"{name}.png"
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return [target.name]
