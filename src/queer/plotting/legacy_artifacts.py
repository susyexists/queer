"""Render completed numerical artifacts without loading scientific models.

Optional plotting dependencies are loaded only when a renderer is called.
"""

from pathlib import Path

import numpy as np


def render_bands(root, manifest, output):
    import matplotlib.pyplot as plt

    files = []
    with np.load(root / manifest["cases"]["bands"]["path"], allow_pickle=False) as data:
        fig, ax = plt.subplots()
        ax.plot(data["energies"].T)
        ax.set(xlabel="k-point index", ylabel="Energy (eV)")
        if len(data["symmetry"]):
            ax.set_xticks(data["symmetry"], data["labels"])
        target = output / "bands.png"
        fig.savefig(target, bbox_inches="tight")
        plt.close(fig)
        files.append(target)
    return files


def render_response(root, manifest, output):
    name = manifest["workflow"]
    import matplotlib.pyplot as plt

    files = []
    rows = []
    for record in manifest["cases"].values():
        with np.load(root / record["path"], allow_pickle=False) as data:
            rows.append(
                np.asarray(data["response" if name == "susceptibility" else "self_energy"]).reshape(-1)
            )
    values = np.asarray(rows)
    fig, ax = plt.subplots()
    ax.plot(values.real, label="Real")
    ax.plot(values.imag, linestyle="--", label="Imaginary")
    ax.set(
        xlabel="q-point index",
        ylabel="Response (1/eV)" if name == "susceptibility" else "EPW response (legacy normalization)",
    )
    ax.legend()
    target = output / f"{name}.png"
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    files.append(target)
    return files


def render_spectral(root, manifest, output):
    import matplotlib.pyplot as plt

    files = []
    spectral_groups: dict[int, tuple[list[np.ndarray], list[float], float | None]] = {}
    for record in manifest["cases"].values():
        with np.load(root / record["path"], allow_pickle=False) as data:
            group_index = record["metadata"].get("v0_index", -1)
            potential = float(data["v0"][0]) if "v0" in data else None
            group = spectral_groups.setdefault(group_index, ([], [], potential))
            group[0].append(data["spectrum"][0])
            group[1].append(float(data["omega"][0]))
    for group_index, (rows, omega, potential) in spectral_groups.items():
        fig, ax = plt.subplots()
        frequencies, indices = np.unique(omega, return_index=True)
        ordered_rows = np.asarray(rows)[indices]
        if len(frequencies) == 1:
            half_width = manifest["config"]["parameters"]["eta"]
            edges = np.asarray([frequencies[0] - half_width, frequencies[0] + half_width])
        else:
            midpoints = (frequencies[:-1] + frequencies[1:]) / 2
            edges = np.concatenate(
                (
                    [frequencies[0] - (midpoints[0] - frequencies[0])],
                    midpoints,
                    [frequencies[-1] + (frequencies[-1] - midpoints[-1])],
                )
            )
        spectral_artist = ax.pcolormesh(
            np.arange(ordered_rows.shape[1] + 1) - 0.5, edges, ordered_rows, shading="flat"
        )
        ax.set(xlabel="k-point index", ylabel="Energy (eV)")
        if potential is not None:
            ax.set_title(f"V0={potential:g} eV")
        fig.colorbar(spectral_artist, ax=ax, label="Spectral weight")
        target = output / ("spectral.png" if potential is None else f"spectral_v0_{group_index:04d}.png")
        fig.savefig(target, bbox_inches="tight")
        plt.close(fig)
        files.append(target)
    return files


def render_momentum(root, manifest, output):
    import matplotlib.pyplot as plt

    files = []
    for case, record in manifest["cases"].items():
        with np.load(root / record["path"], allow_pickle=False) as data:
            surface = data["surface"]
            fig, ax = plt.subplots()
            artist = ax.scatter(surface[:, 0], surface[:, 1], c=surface[:, 4], s=2, cmap="inferno")
            ax.set(
                xlabel="kx (angstrom^-1)",
                ylabel="ky (angstrom^-1)",
                title=f"V0={data['v0'][0]:g} eV, binding energy={data['binding_energy'][0]:g} eV",
            )
            fig.colorbar(artist, ax=ax, label="Intensity")
            target = output / f"{case}.png"
            fig.savefig(target, bbox_inches="tight")
            plt.close(fig)
            files.append(target)
    return files


def render(root, manifest, output) -> list[Path]:
    """Dispatch historical bands, response, spectrum and momentum artifacts."""
    name = manifest["workflow"]
    renderer = {
        "bands": render_bands,
        "susceptibility": render_response,
        "epw": render_response,
        "spectral": render_spectral,
        "momentum": render_momentum,
    }[name]
    return renderer(root, manifest, output)
