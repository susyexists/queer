"""Momentum-microscopy surface helpers shared by notebooks and HPC scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .arpes import binding_k
from .functions import get_rotation_matrix
from .mesh import angstrom2reciprocal


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    if seconds < 3600:
        return f"{seconds / 60:.2f} minutes"
    return f"{seconds / 3600:.2f} hours"


def binding_energy_surfaces(
    model,
    g_vec,
    photon_energy: float,
    fermi_energy: float,
    binding_range,
    binding_step: float,
    v0: float,
    n_points: int,
    factor: float,
    align=None,
    energy_shift: float = 0.0,
    sigma: float = 0.1,
):
    """Calculate intensity surfaces for a range of binding-energy slices."""
    kmesh_dict = binding_k(
        photon_energy,
        fermi_energy,
        binding_range,
        binding_step,
        v0,
        n_points,
        factor,
        align,
    )

    surfaces = {}
    for binding_energy, mesh_components in kmesh_dict.items():
        kmesh = np.array(mesh_components)
        inv_kmesh = angstrom2reciprocal(kmesh.T, g_vec)
        band = model.calculate_energy(inv_kmesh) + energy_shift

        n_bands = band.shape[0]
        mesh_tile = np.array([kmesh.T for _ in range(n_bands)]).reshape(-1, 3)
        df = pd.DataFrame(
            {
                "kx": mesh_tile[:, 0],
                "ky": mesh_tile[:, 1],
                "kz": mesh_tile[:, 2],
                "e": band.flatten(),
            }
        )
        df["I"] = np.exp(-((df.e + binding_energy) ** 2) / (2 * sigma**2))
        surfaces[binding_energy] = df.loc[df.groupby(["kx", "ky"])["I"].idxmax()]

    return surfaces


def save_surface_plots(
    surfaces,
    output_dir,
    v0: float,
    energy_shift: float = 0.0,
    rotate_from=None,
    rotate_to=None,
    cmap: str = "inferno",
    transparent: bool = False,
    close: bool = True,
):
    """Save one scatter plot for each binding-energy surface."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rotation = None
    if rotate_from is not None and rotate_to is not None:
        rotation = get_rotation_matrix(np.array(rotate_from), np.array(rotate_to))

    frame_paths = []
    for binding_energy, df in surfaces.items():
        coords = np.array([df.kx.values, df.ky.values, df.kz.values])
        if rotation is not None:
            coords = rotation.T @ coords

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(coords[0], coords[1], c=df.I, cmap=cmap, s=1, edgecolors="none")
        ax.set_xlabel(r"$k_{x}$ [$\AA^{-1}$]")
        ax.set_ylabel(r"$k_{y}$ [$\AA^{-1}$]")
        ax.set_title(
            rf"$E_B$={np.around(binding_energy, 2)} eV  $V_0$={v0:g}, shift={energy_shift:g}"
        )
        ax.set_facecolor("black")
        fig.tight_layout()

        frame_path = output_dir / f"{np.around(binding_energy, 2)}.png"
        fig.savefig(frame_path, dpi=300, bbox_inches="tight", transparent=transparent)
        frame_paths.append(frame_path)
        if close:
            plt.close(fig)

    return frame_paths


def write_animation(frame_paths, output_dir, fps: int = 2):
    """Create MP4 and GIF animations from saved frame paths when imageio/Pillow are available."""
    import imageio.v2 as imageio
    from PIL import Image

    output_dir = Path(output_dir)
    frame_paths = [Path(path) for path in frame_paths]
    if not frame_paths:
        return None, None

    mp4_path = output_dir / "animation.mp4"
    gif_path = output_dir / "animation.gif"

    with imageio.get_writer(mp4_path, fps=fps) as writer:
        for frame in frame_paths:
            img = Image.open(frame)
            writer.append_data(np.array(img))

    with imageio.get_writer(gif_path, mode="I", duration=1 / fps) as writer:
        for frame in frame_paths:
            writer.append_data(imageio.imread(frame))

    return mp4_path, gif_path
