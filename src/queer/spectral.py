"""Spectral-function helpers along k-paths."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .arpes import arpes_path
from .kpath import KPath
from .response import spectral_from_eig


def spectral_along_path(model, path, omega_list, eta: float = 0.1):
    """Compute A(omega, k) along a k-path.

    ``path`` may be a ``KPath`` or a raw ``(Nk, 3)`` array. Returns an array of
    shape ``(len(omega_list), Nk)``.
    """
    bands = model.calculate_energy(path)
    spectral = [spectral_from_eig(eig, omega_list=omega_list, eta=eta) for eig in bands.T]
    return np.array(spectral).T


def spectral_arpes_path_sweep(
    model,
    band_path,
    g_vec,
    photon_energy: float,
    v0_list,
    omega_list,
    eta: float = 0.1,
    align_zero: bool = True,
):
    """Sweep ``V0`` over the curved ARPES path and return per-V0 spectral arrays.

    For each ``v0`` and each ``omega`` in ``omega_list``, the curved path is built
    with ``Ek = photon_energy - omega`` and bands are evaluated. When
    ``align_zero=True`` the bands are shifted by the largest negative band value
    on the curved path at ``omega = 0`` — computed once per v0, then applied to
    every omega. Works for any ``omega_list`` (whether 0 is the first element,
    a middle element, or absent entirely).

    Returns ``{v0: 2D array of shape (len(omega_list), Nk)}``.
    """
    path_points = band_path.path if isinstance(band_path, KPath) else np.asarray(band_path)

    results = {}
    for v0 in v0_list:
        max_neg = 0.0
        if align_zero:
            ref_path = arpes_path(path_points, g_vec, Ek=photon_energy, V0=v0)
            ref_bands = model.calculate_energy(ref_path)
            negatives = ref_bands[ref_bands < 0]
            if negatives.size:
                max_neg = negatives.max()

        rows = []
        for omega in omega_list:
            curved = arpes_path(path_points, g_vec, Ek=photon_energy - omega, V0=v0)
            bands = model.calculate_energy(curved)
            row = [
                spectral_from_eig(eig - max_neg, omega_list=[omega], eta=eta)[0]
                for eig in bands.T
            ]
            rows.append(row)
        results[v0] = np.array(rows)
    return results


def plot_spectral_path(
    spectral_2d,
    band_path,
    omega_axis,
    ax=None,
    cmap: str = "magma",
    title=None,
    save=None,
    ylabel: str = r"$E - E_F$ (eV)",
    xlabel: str = r"High-symmetry Points [$\AA^{-1}$]",
):
    """Plot a 2D spectral array as a pcolormesh with high-symmetry tick lines."""
    if not isinstance(band_path, KPath):
        raise TypeError("plot_spectral_path requires a KPath for sym/labels.")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    x = np.arange(spectral_2d.shape[1])
    ax.pcolormesh(x, omega_axis, spectral_2d, shading="auto", cmap=cmap)
    for sym in band_path.sym[1:-1]:
        ax.axvline(sym, c="white", linestyle="--")
    ax.set_xticks(band_path.sym)
    ax.set_xticklabels(band_path.labels, fontsize=15)
    ax.set_xlim(band_path.sym[0], band_path.sym[-1])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig, ax
