import numpy as np

def mesh_cartesian(N, factor=1, center=True):
    """
    Build a Cartesian mesh in [0,1)^3 scaled by `factor`.

    Params
    ------
    N : int or (int, int, int)
        Number of points along each axis.
    factor : float
        Scale factor for the mesh.
    gamma : bool
        If True, build a Gamma-centered grid (each axis shifted to zero mean
        before scaling).
    center : bool
        If True, center the final mesh by subtracting its mean (after scaling).
    """
    if isinstance(N, int):
        N = [N, N, N]

    x = np.linspace(0, 1, N[0], endpoint=False)
    y = np.linspace(0, 1, N[1], endpoint=False)
    z = np.linspace(0, 1, N[2], endpoint=False)


    mesh = np.array([[i, j, k] for i in x for j in y for k in z], dtype=float)
    mesh *= factor

    # "check" / enforce centering of the final mesh
    if center:
        center_vec = mesh.mean(axis=0)
        mesh -= center_vec  # now mesh.mean(axis=0) ≈ [0, 0, 0]

    return mesh

def mesh_crystal(N,g_vec,dimension = 3,factor=1):
    mesh = mesh_cartesian(N, factor)
    if dimension==2:
        g_vec = g_vec.T[:2].T[:2]
    t_mesh = np.dot(g_vec.T, mesh.T)
    return t_mesh

def hexagon_crystal(N,g_vec):
    grid = np.dot(hexagon_cartesian(N).T,inverse_g)
    return grid

def cartesian2crystal(cartesian,g_vec):
    """Legacy helper: transform Cartesian/angstrom coordinates with g_vec.T."""
    crystal = np.dot(g_vec.T, np.asarray(cartesian).T)
    return crystal

def crystal2cartesian(crystal,g_vec):
    """Legacy helper kept for older scripts; returns points as (n, 3)."""
    g_inv = np.linalg.inv(g_vec)
    cartesian = np.dot(np.asarray(crystal).T,g_inv)
    return cartesian


def reciprocal2angstrom(reciprocal,g_vec):
    reciprocal = np.asarray(reciprocal)
    angstrom = np.dot(g_vec.T, reciprocal.T)
    return angstrom.T

def angstrom2reciprocal(angstrom,g_vec):
    angstrom = np.asarray(angstrom)
    g_inv = np.linalg.inv(g_vec)
    cartesian = np.dot(angstrom,g_inv)
    return cartesian
