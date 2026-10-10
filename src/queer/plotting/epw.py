"""EPW dispersion and raw coupling plots; computations remain in EPWModel."""

import numpy as np


def _plot_modes(values, ylabel, ax=None, save=None):
    import matplotlib.pyplot as plt

    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("mode data require shape (Nmode,Nq)")
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(np.arange(values.shape[1]), values.T)
    ax.set(xlabel="q-point index", ylabel=ylabel)
    if save is not None:
        ax.figure.savefig(save, bbox_inches="tight")
    return ax.figure, ax


def plot_phonon_dispersion(model, ax=None, save=None):
    return _plot_modes(model.ph, "Phonon frequency (meV)", ax, save)


def plot_epw_coupling(model, ax=None, save=None):
    return _plot_modes(np.mean(np.abs(model.g_complex), axis=2), "Mean raw coupling |g| (meV)", ax, save)
