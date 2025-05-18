import numpy as np

def mesh_cartesian(N, factor=1,gamma=False):
    if type(N)== int:
        N = [N,N,N]
    x = np.linspace(0, 1, N[0])
    y = np.linspace(0, 1, N[1])
    z = np.linspace(0, 1, N[2])
    if gamma==True:
        x = x - x.mean()
        y = y-  y.mean()
        z = z-  y.mean()
    mesh = np.array([[i, j,k] for i in x for j in y for k in z])

    return (mesh*factor)

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
    crystal = np.dot(g_vec.T, cartesian.T)
    return crystal

def crystal2cartesian(crystal,g_vec):
    g_inv = np.linalg.inv(g_vec)
    cartesian = np.dot(crystal.T,g_inv)
    return cartesian


def reciprocal2angstrom(reciprocal,g_vec):
    angstrom = np.dot(g_vec.T, reciprocal.T)
    return angstrom.T

def angstrom2reciprocal(angstrom,g_vec):
    g_inv = np.linalg.inv(g_vec)
    cartesian = np.dot(angstrom,g_inv)
    return cartesian