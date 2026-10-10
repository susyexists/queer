"""Optional response visualizations using caller-owned axes."""

from pathlib import Path

import numpy as np


def plot_susceptibility(suscep, sym, labels, show=False, save=False, path=False, ax=None):
    import matplotlib.pyplot as plt

    susceptibility = np.asarray(suscep)
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 4))
    if susceptibility.ndim == 1:
        real, imaginary = susceptibility.real, susceptibility.imag
    elif susceptibility.ndim == 2 and susceptibility.shape[0] == 2:
        real, imaginary = susceptibility
    else:
        raise ValueError("susceptibility must be complex (Nq,) or historical (2,Nq)")
    ax.plot(real, label="Re(χ)")
    ax.plot(imaginary, label="Im(χ)")
    ax.set_xticks(sym, labels, fontsize=15)
    ax.set_xlim(sym[0], sym[-1])
    for point in sym[1:-1]:
        ax.axvline(point, c="black")
    ax.set_ylabel("1/eV")
    ax.legend()
    ax.figure.tight_layout()
    if save:
        destination = Path(save) if not isinstance(save, bool) else Path(path or ".") / "susceptibility.pdf"
        ax.figure.savefig(destination)
    elif path:
        ax.figure.savefig(Path(path) / "susceptibility.pdf")
    if show:
        plt.show()
    return ax.figure, ax


def plot_electron_mesh(
    band, mesh, metallic_band_index, fig_size=(5, 5), save=None, temp=None, cmap="jet", s=1, ax=None
):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=fig_size)
    mesh = np.asarray(mesh)
    if mesh.ndim != 2:
        raise ValueError("mesh must be a two-dimensional coordinate array")
    if mesh.shape[0] == 3:
        mesh = mesh.T
    if mesh.shape[1] != 3:
        raise ValueError("mesh must contain three coordinates")
    artist = ax.scatter(mesh[:, 0], mesh[:, 1], c=np.asarray(band)[metallic_band_index], cmap=cmap, s=s)
    ax.figure.colorbar(artist, ax=ax)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$k_x$ [$\AA^{-1}$]")
    ax.set_ylabel(r"$k_y$ [$\AA^{-1}$]")
    ax.set_title("" if temp is None else f"σ = {temp}")
    if save is not None:
        ax.figure.savefig(save)
    return ax.figure, ax
