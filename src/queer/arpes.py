"""Compatibility imports for ARPES geometry and optional visualization."""

from .observables.arpes import aligned_arpes_mesh, arpes_equation, arpes_mesh, arpes_path, binding_k

__all__ = [
    "arpes_equation",
    "arpes_mesh",
    "aligned_arpes_mesh",
    "binding_k",
    "arpes_path",
    "arpes_mesh_plot",
    "arpes_mesh_plot_3d",
]


def arpes_mesh_plot(photon_energy, fermi_energy, binding_energy, V0, N, factor):
    import matplotlib.pyplot as plt

    mesh = arpes_mesh(photon_energy, fermi_energy, binding_energy, V0, N, factor)
    fig, ax = plt.subplots()
    scatter = ax.scatter(mesh[:, 0], mesh[:, 1], c=mesh[:, 2], cmap="jet", s=1)
    fig.colorbar(scatter, ax=ax)
    plt.show()
    return fig, ax


def arpes_mesh_plot_3d(photon_energy, fermi_energy, binding_energy, V0, N, factor):
    import matplotlib.pyplot as plt

    mesh = arpes_mesh(photon_energy, fermi_energy, binding_energy, V0, N, factor)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*mesh.T, c=mesh[:, 2], cmap="jet", s=1)
    return fig, ax
