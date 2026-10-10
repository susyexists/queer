"""Momentum surface rendering and animation, independent of computation."""

from pathlib import Path

import numpy as np

from ..geometry.rotations import rotation_matrix as get_rotation_matrix
from ..observables.momentum import SURFACE_COLUMNS


def _surface_array(surface):
    if hasattr(surface, "to_numpy"):
        surface = surface.loc[:, list(SURFACE_COLUMNS)].to_numpy()
    result = np.asarray(surface, dtype=float)
    if result.ndim != 2 or result.shape[1] != 5:
        raise ValueError("surfaces require (N,5) arrays or DataFrames with kx,ky,kz,e,I")
    return result


def save_surface_plots(
    surfaces,
    output_dir,
    v0,
    energy_shift=0.0,
    rotate_from=None,
    rotate_to=None,
    style="intensity",
    cmap="inferno",
    point_size=1.0,
    alpha=1.0,
    tol=0.1,
    transparent=False,
    close=True,
):
    import matplotlib.pyplot as plt

    styles = ("intensity", "scatter_white", "scatter_black_transparent")
    if style not in styles:
        raise ValueError(f"Unknown style: {style!r}")
    if (rotate_from is None) != (rotate_to is None):
        raise ValueError("rotate_from and rotate_to must both be supplied")
    rotation = None if rotate_from is None else get_rotation_matrix(rotate_from, rotate_to)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    items = surfaces.items() if hasattr(surfaces, "items") else surfaces
    for energy, surface in items:
        data = _surface_array(surface)
        if style != "intensity":
            data = data[np.abs(data[:, 3] + energy) < tol]
            if not len(data):
                continue
        coordinates = data[:, :3].T
        if rotation is not None:
            coordinates = rotation.T @ coordinates
        fig, ax = plt.subplots(figsize=(5, 5))
        try:
            frame_transparent = transparent
            if style == "intensity":
                ax.scatter(
                    coordinates[0],
                    coordinates[1],
                    c=data[:, 4],
                    cmap=cmap,
                    s=point_size,
                    alpha=alpha,
                    edgecolors="none",
                )
                ax.set_facecolor("black")
            else:
                color = "white" if style == "scatter_white" else "black"
                ax.scatter(
                    coordinates[0],
                    coordinates[1],
                    color=color,
                    s=2 if point_size == 1 else point_size,
                    alpha=0.25 if alpha == 1 else alpha,
                    edgecolors="none",
                    rasterized=True,
                )
                if style == "scatter_white":
                    ax.set_facecolor("black")
                else:
                    fig.patch.set_alpha(0.0)
                    ax.patch.set_alpha(0.0)
                    frame_transparent = True
            ax.set_xlabel(r"$k_x$ [$\AA^{-1}$]")
            ax.set_ylabel(r"$k_y$ [$\AA^{-1}$]")
            ax.set_title(rf"$E_B$={energy:.2f} eV  $V_0$={v0:g}, shift={energy_shift:g}")
            fig.tight_layout()
            destination = output_dir / f"{np.around(energy, 2)}.png"
            fig.savefig(destination, dpi=300, bbox_inches="tight", transparent=frame_transparent)
            frames.append(destination)
        finally:
            if close:
                plt.close(fig)
    return frames


def write_animation(frame_paths, output_dir, fps=2):
    """Write MP4/GIF with common even-sized frames; images are closed promptly."""
    import imageio.v2 as imageio
    from PIL import Image

    if isinstance(fps, bool) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    frame_paths = [Path(path) for path in frame_paths]
    if not frame_paths:
        return None, None
    sizes = []
    for path in frame_paths:
        with Image.open(path) as frame:
            sizes.append(frame.size)
    # Multiples of 16 work with ffmpeg's default macroblock configuration.
    width = ((max(size[0] for size in sizes) + 15) // 16) * 16
    height = ((max(size[1] for size in sizes) + 15) // 16) * 16
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mp4_path, gif_path = output_dir / "animation.mp4", output_dir / "animation.gif"

    def frames():
        for path in frame_paths:
            with Image.open(path) as frame:
                canvas = Image.new("RGB", (width, height), "white")
                converted = frame.convert("RGBA")
                canvas.paste(converted, ((width - frame.width) // 2, (height - frame.height) // 2), converted)
                yield np.asarray(canvas)

    with imageio.get_writer(mp4_path, fps=fps) as writer:
        for frame in frames():
            writer.append_data(frame)
    with imageio.get_writer(gif_path, mode="I", duration=1 / fps) as writer:
        for frame in frames():
            writer.append_data(frame)
    return mp4_path, gif_path
