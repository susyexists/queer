"""Band and Brillouin-zone plotting adapters; Matplotlib loads on demand."""

import numpy as np

from ..geometry.paths import KPath


def plot_electron_path(model, band, sym, labels, ylim=(-10, 10), save=None, temp=None, title=False, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    for i in np.atleast_2d(band):
        ax.plot(i, c="blue")
    ax.set_xticks(sym, labels, fontsize=15)
    ax.set_xlim(sym[0], sym[-1])
    for i in sym[1:-1]:
        ax.axvline(i, c="black", linestyle="--")
    ax.axhline(0, linestyle="--", color="red")
    if ylim is not None:
        ax.set_ylim(ylim)
    if title is not False and title is not None:
        ax.set_title(title)
    if temp is not None:
        ax.set_title(f"σ = {temp}", fontsize=15)
    if getattr(model, "shift", 0) != 0:
        ax.set_title(r"$\delta \epsilon_{Fermi} = $" f" {model.shift} eV", fontsize=15)
    ax.set_ylabel("Energy (eV)", fontsize=15)
    if save is not None:
        ax.figure.savefig(save)
    return ax


def plot_band_path(
    model,
    path_or_points,
    sym=None,
    labels=None,
    n_points=200,
    ylim=(-10, 10),
    save=None,
    band_index=False,
    ax=None,
    **seekpath_kwargs,
):
    """Calculate and plot electron bands along a path.

    ``path_or_points`` can be either an explicit k-point array, in which
    case ``sym`` and ``labels`` must be supplied, a :class:`queer.kpath.KPath`,
    or a named path string such as ``"GAMMA-X-W-K-GAMMA-L"``.
    """
    if isinstance(path_or_points, str):
        k_path = model.kpath(path_or_points, n_points, **seekpath_kwargs)
        sym, path, labels = k_path
    elif isinstance(path_or_points, KPath):
        path = path_or_points.path
        sym = path_or_points.sym if sym is None else sym
        labels = path_or_points.labels if labels is None else labels
    else:
        if sym is None or labels is None:
            raise ValueError("Explicit paths require sym and labels.")
        path = np.asarray(path_or_points)

    bands = model.calculate_energy(path, band_index=band_index)
    axis = plot_electron_path(model, bands, sym, labels, ylim=ylim, save=save, ax=ax)
    return bands, axis


def plot_brillouin_zone(
    model,
    mesh=None,
    mesh_repeat=0,
    align_dir=None,
    ax=None,
    figsize=(7, 7),
    title=None,
    max_mesh_pts=2000,
    mesh_clip_bz=False,
    mesh_style="scatter",
):
    """Plot the Brillouin zone with optional k-point mesh overlay.

    Requires a POSCAR (pass ``poscar=`` when constructing the model).

    Parameters
    ----------
    mesh : (N, 3) array, optional
        k-points in Å⁻¹ to scatter on top of the BZ.
    mesh_repeat : int
        Number of BZ shells over which to tile the mesh using G-vector
        offsets.  0 = original mesh only; 1 = adds the 26 nearest G-vector
        images, etc.
    align_dir : array-like of length 3, optional
        Cartesian direction (Å⁻¹) for an arrow drawn from Γ to the BZ
        boundary.  The arrow length is clipped to the BZ face it first hits.
    ax : Axes3D, optional
        Existing axes to draw into; a new figure is created if omitted.
    figsize, title : passed to plt.figure / ax.set_title.

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt

    try:
        import ase.io
        from ase.dft.bz import bz_vertices
        from ase.dft.kpoints import get_special_points
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        raise ImportError("plot_brillouin_zone requires ase: pip install ase")

    if model.poscar is None:
        raise ValueError("plot_brillouin_zone requires a POSCAR. Initialize the model with poscar=...")

    atoms = ase.io.read(model.path / model.poscar)

    # model.g_vec rows are b1, b2, b3 in 2π/Å — exactly what bz_vertices expects
    faces = bz_vertices(model.g_vec)
    polys = [f[0] for f in faces]

    # fractional → Cartesian: k_cart = k_frac @ g_vec
    sp_cart = {name: frac @ model.g_vec for name, frac in get_special_points(atoms.cell).items()}

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    ax.add_collection3d(
        Poly3DCollection(polys, alpha=0.08, facecolor="steelblue", edgecolor="steelblue", linewidth=0.8)
    )

    if mesh is not None:
        m = np.asarray(mesh, dtype=float)
        n_pts = len(m)
        n_side = int(round(n_pts**0.5))
        is_structured = n_side * n_side == n_pts

        if mesh_clip_bz:
            # Find the G-vector nearest to the mesh centroid, then fold
            # every point back into the first BZ by subtracting that G
            # (reduced zone scheme).
            centroid = m.mean(axis=0)
            G_near = np.zeros(3)
            best_d = np.inf
            for n1 in range(-3, 4):
                for n2 in range(-3, 4):
                    for n3 in range(-3, 4):
                        G = n1 * model.g_vec[0] + n2 * model.g_vec[1] + n3 * model.g_vec[2]
                        d = np.linalg.norm(centroid - G)
                        if d < best_d:
                            best_d = d
                            G_near = G.copy()
            m = m - G_near  # fold into first BZ

        if mesh_style in ("surface", "wireframe") and is_structured:
            stride = max(1, n_side // max(1, int(max_mesh_pts**0.5)))
            gs = m.reshape(n_side, n_side, 3)[::stride, ::stride].copy()
            if mesh_clip_bz:
                flat = gs.reshape(-1, 3)
                inside = np.ones(len(flat), dtype=bool)
                for verts, normal in faces:
                    inside &= flat @ normal <= verts[0] @ normal + 1e-10
                flat[~inside] = np.nan
                gs = flat.reshape(gs.shape)
            X, Y, Z = gs[..., 0], gs[..., 1], gs[..., 2]
            if mesh_style == "wireframe":
                ax.plot_wireframe(X, Y, Z, color="orange", alpha=0.6, linewidth=0.5, rstride=1, cstride=1)
            else:
                ax.plot_surface(X, Y, Z, color="orange", alpha=0.3, linewidth=0, antialiased=True)
        else:
            # Scatter fallback (style='scatter' or unstructured input).
            if mesh_clip_bz:
                inside = np.ones(len(m), dtype=bool)
                for verts, normal in faces:
                    inside &= m @ normal <= verts[0] @ normal + 1e-10
                m = m[inside]
            elif mesh_repeat > 0:
                bz_radius = np.linalg.norm(np.vstack(polys), axis=1).max()
                shells = [m]
                for n1 in range(-mesh_repeat, mesh_repeat + 1):
                    for n2 in range(-mesh_repeat, mesh_repeat + 1):
                        for n3 in range(-mesh_repeat, mesh_repeat + 1):
                            if n1 == n2 == n3 == 0:
                                continue
                            offset = n1 * model.g_vec[0] + n2 * model.g_vec[1] + n3 * model.g_vec[2]
                            if np.linalg.norm(offset) <= 2 * bz_radius:
                                shells.append(m + offset)
                m = np.vstack(shells)
            if len(m) > max_mesh_pts:
                rng = np.random.default_rng(0)
                m = m[rng.choice(len(m), max_mesh_pts, replace=False)]
            ax.scatter(m.T[0], m.T[1], m.T[2], s=3, alpha=0.4, color="orange", label="ARPES mesh")

    for name, pt in sp_cart.items():
        ax.scatter(*pt, s=40, color="red", zorder=5)
        ax.text(pt[0], pt[1], pt[2], f" {name}", fontsize=10, color="red")

    lim = np.abs(np.vstack(polys)).max() * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel(r"$k_x$ [Å$^{-1}$]")
    ax.set_ylabel(r"$k_y$ [Å$^{-1}$]")
    ax.set_zlabel(r"$k_z$ [Å$^{-1}$]")
    ax.set_title(title or "Brillouin Zone")

    if align_dir is not None:
        d = np.asarray(align_dir, dtype=float)
        if d.shape != (3,) or not np.isfinite(d).all() or np.linalg.norm(d) == 0:
            raise ValueError("align_dir must be a finite nonzero 3-vector")
        d_hat = d / np.linalg.norm(d)
        # Find how far the ray Γ + t*d_hat travels before hitting a BZ face
        t_bz = np.inf
        for verts, normal in faces:
            denom = d_hat @ normal
            if abs(denom) > 1e-10:
                t = (verts[0] @ normal) / denom
                if 0 < t < t_bz:
                    t_bz = t
        arrow_tip = d_hat * (t_bz if np.isfinite(t_bz) else lim * 0.8)
        ax.quiver(0, 0, 0, *arrow_tip, color="cyan", linewidth=2, arrow_length_ratio=0.15)

    return fig, ax
