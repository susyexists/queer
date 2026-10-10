"""Headless-friendly spectral visualization."""

import numpy as np


def plot_spectral_path(
    spectral_2d,
    band_path,
    omega_axis,
    ax=None,
    cmap="magma",
    title=None,
    save=None,
    ylabel=r"$E - E_F$ (eV)",
    xlabel=r"High-symmetry Points [$\AA^{-1}$]",
):
    import matplotlib.pyplot as plt

    from ..kpath import KPath

    if not isinstance(band_path, KPath):
        raise TypeError("plot_spectral_path requires a KPath for sym/labels")
    spectral_2d = np.asarray(spectral_2d)
    omega_axis = np.asarray(omega_axis)
    if spectral_2d.shape != (len(omega_axis), len(band_path.path)):
        raise ValueError("spectral shape must match frequency and path axes")
    if not spectral_2d.size:
        raise ValueError("plotting requires nonempty frequency and path axes")
    if ax is None:
        _, ax = plt.subplots()
    ax.pcolormesh(np.arange(spectral_2d.shape[1]), omega_axis, spectral_2d, shading="auto", cmap=cmap)
    for sym in band_path.sym[1:-1]:
        ax.axvline(sym, c="white", linestyle="--")
    ax.set_xticks(band_path.sym, band_path.labels, fontsize=15)
    ax.set_xlim(band_path.sym[0], band_path.sym[-1])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title)
    ax.figure.tight_layout()
    if save is not None:
        ax.figure.savefig(save, dpi=300, bbox_inches="tight")
    return ax.figure, ax
