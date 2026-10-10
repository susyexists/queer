"""Paper panels rendered from numerical artifacts with no model imports."""

from __future__ import annotations

import numpy as np


def render_paper_figure(arrays, metadata, output):
    import matplotlib.pyplot as plt

    name = metadata["figure"]
    if name in {"fig4b", "fig4d"}:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.plot(arrays["fields"] / 1e3, arrays["probability"], label="Probability")
        if "envelope" in arrays:
            ax.plot(arrays["fields"] / 1e3, arrays["envelope"], "--", label="Envelope")
            ax.legend()
        ax.set(xlabel="Field (kV/m)", ylabel="Transition probability")
    elif name == "fig5ab":
        count = len(arrays["parameter"])
        fig, axes = plt.subplots(2, count, figsize=(3 * count, 6), squeeze=False)
        for i, x in enumerate(arrays["parameter"]):
            energies = arrays["energies"][i] * 1e3
            axes[0, i].plot(arrays["k"] * 1e3, energies, "k", lw=0.8)
            for band in (1, 2):
                metric = np.log10(np.maximum(arrays["pair_metric"][i], 1e-30))
                axes[0, i].scatter(arrays["k"] * 1e3, energies[:, band], c=metric, s=2, cmap="magma_r")
            axes[0, i].set(xlabel="kx (10⁻³ Å⁻¹)", ylabel="Energy (meV)", title=f"VL/VLc={x:g}")
            curvature = np.where(arrays["valid"][i], arrays["curvature"][i], np.nan)
            axes[1, i].pcolormesh(
                arrays["kmap"] * 1e3,
                arrays["kmap"] * 1e3,
                np.arcsinh(curvature.T / 1e3),
                shading="auto",
                cmap="RdBu_r",
            )
            axes[1, i].set(xlabel="kx (10⁻³ Å⁻¹)", ylabel="ky (10⁻³ Å⁻¹)", aspect="equal")
    elif name == "fig6bd":
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        axes[0].loglog(arrays["ky"], abs(arrays["curvature"]), label="Curvature")
        axes[0].loglog(arrays["ky"], abs(arrays["critical_reference"]), "--", label="C/k")
        axes[0].set(xlabel="ky (Å⁻¹)", ylabel="|Curvature| (Å²)")
        axes[0].legend()
        axes[1].plot(arrays["parameter"], arrays["gap_edge_metric"] / 1e6)
        axes[1].set(xlabel="VL/VLc", ylabel="Pair metric (10⁶ Å²)")
    elif name == "ext7e":
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        for method in ("gap_minimum", "diabatic"):
            axes[0].plot(arrays["parameter"], arrays[f"Ec_{method}"] / 1e3, label=method.replace("_", " "))
            axes[1].plot(
                arrays["parameter"], arrays[f"Lambda_{method}"] / 1e3, label=method.replace("_", " ")
            )
        for ax, label in zip(axes, ("Ec (kV/m)", "Lambda (kV/m)")):
            ax.set(xlabel="VL/VLc", ylabel=label)
            ax.legend()
    elif name == "ext9b":
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        for order, values in zip(arrays["orders"], arrays["scaled_sine"]):
            ax.loglog(arrays["scaled_amplitudes"], abs(values), label=f"N={order}")
        ax.set(xlabel="Amplitude / N (V/m)", ylabel="N |bN| (A/m)")
        ax.legend()
    else:
        count = len(arrays["orders"])
        fig, axes = plt.subplots(1, count, figsize=(3.5 * count, 3.5), squeeze=False)
        for i, order in enumerate(arrays["orders"]):
            im = axes[0, i].pcolormesh(
                arrays["parameter"],
                arrays["amplitudes"] / 1e4,
                arrays["b"][..., i].T * 1e3,
                cmap="bwr",
                shading="auto",
            )
            axes[0, i].set(title=f"N={order}", xlabel="VL/VLc", ylabel="Amplitude (10⁴ V/m)")
            fig.colorbar(im, ax=axes[0, i], label="mA/m")
    fig.suptitle(name)
    fig.tight_layout()
    filename = f"{name}.png"
    fig.savefig(output / filename, dpi=130)
    plt.close(fig)
    return [filename]
