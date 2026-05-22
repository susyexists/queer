"""Momentum-microscopy surface helpers shared by notebooks and HPC scripts."""

from __future__ import annotations

import time
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


def sweep_binding_energy_surfaces(
    model,
    g_vec,
    photon_energy: float,
    fermi_energy: float,
    binding_range,
    binding_step: float,
    v0_list,
    n_points: int,
    factor: float,
    align=None,
    energy_shifts=(0.0,),
    sigma: float = 0.1,
    verbose: bool = True,
):
    """Run `binding_energy_surfaces` over a grid of (v0, energy_shift) values.

    Returns ``{(v0, shift): surfaces_dict}``.
    """
    results = {}
    for shift in energy_shifts:
        for v0 in v0_list:
            if verbose:
                print(f"--- V0={v0}, shift={shift} ---")
            start = time.time()
            surfaces = binding_energy_surfaces(
                model,
                g_vec,
                photon_energy,
                fermi_energy,
                binding_range,
                binding_step,
                v0,
                n_points,
                factor,
                align=align,
                energy_shift=shift,
                sigma=sigma,
            )
            if verbose:
                print(f"Surfaces computed in {format_time(time.time() - start)}")
            results[(v0, shift)] = surfaces
    return results


def fermi_surface_points(mesh, model, g_vec, delta: float = 0.05):
    """Return the (kx, ky) points where any band sits within ``delta`` of E_F.

    ``mesh`` is the ARPES k-mesh in Å⁻¹ (shape ``(N, 3)``). Bands are computed
    via ``model.calculate_energy`` after dropping NaN rows; the returned mask
    is aligned to the valid rows only.
    """
    mesh = np.asarray(mesh)
    valid = ~np.isnan(mesh).any(axis=1)
    valid_mesh = mesh[valid]
    inv_mesh = angstrom2reciprocal(valid_mesh, g_vec)
    band = model.calculate_energy(inv_mesh)
    mask = np.any(np.abs(band) <= delta, axis=0)
    kx = valid_mesh[:, 0][mask]
    ky = valid_mesh[:, 1][mask]
    return mask, kx, ky


def save_surface_plots(
    surfaces,
    output_dir,
    v0: float,
    energy_shift: float = 0.0,
    rotate_from=None,
    rotate_to=None,
    style: str = "intensity",
    cmap: str = "inferno",
    point_size: float = 1.0,
    alpha: float = 1.0,
    tol: float = 0.1,
    transparent: bool = False,
    close: bool = True,
):
    """Save one scatter plot per binding-energy surface.

    Styles:
        - ``"intensity"``  : scatter colored by ``df.I`` on a black canvas.
        - ``"scatter_white"`` : white dots (``alpha=0.25`` default) on a black
          canvas; rows are filtered to ``|e + binding_energy| < tol``.
        - ``"scatter_black_transparent"`` : black dots on a fully transparent
          figure + axes patch; rows are filtered the same way.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rotation = None
    if rotate_from is not None and rotate_to is not None:
        rotation = get_rotation_matrix(np.array(rotate_from), np.array(rotate_to))

    frame_paths = []
    for binding_energy, df in surfaces.items():
        if style in ("scatter_white", "scatter_black_transparent"):
            df = df[np.abs(df.e + binding_energy) < tol]
            if len(df) == 0:
                continue

        coords = np.array([df.kx.values, df.ky.values, df.kz.values])
        if rotation is not None:
            coords = rotation.T @ coords

        fig, ax = plt.subplots(figsize=(5, 5))
        if style == "intensity":
            ax.scatter(
                coords[0], coords[1],
                c=df.I, cmap=cmap, s=point_size,
                alpha=alpha, edgecolors="none",
            )
            ax.set_facecolor("black")
        elif style == "scatter_white":
            scatter_alpha = alpha if alpha != 1.0 else 0.25
            ax.scatter(
                coords[0], coords[1],
                color="white", s=point_size if point_size != 1.0 else 2,
                alpha=scatter_alpha, edgecolors="none", rasterized=True,
            )
            ax.set_facecolor("black")
        elif style == "scatter_black_transparent":
            scatter_alpha = alpha if alpha != 1.0 else 0.25
            ax.scatter(
                coords[0], coords[1],
                color="black", s=point_size if point_size != 1.0 else 2,
                alpha=scatter_alpha, edgecolors="none", rasterized=True,
            )
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            transparent = True
        else:
            raise ValueError(f"Unknown style: {style!r}")

        ax.set_xlabel(r"$k_{x}$ [$\AA^{-1}$]")
        ax.set_ylabel(r"$k_{y}$ [$\AA^{-1}$]")
        ax.set_title(
            rf"$E_B$={np.around(binding_energy, 2)} eV  $V_0$={v0:g}, shift={energy_shift:g}"
        )
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
