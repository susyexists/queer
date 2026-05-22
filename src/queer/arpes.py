
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import electron_mass, Planck
from .mesh import mesh_cartesian, mesh_crystal,reciprocal2angstrom,angstrom2reciprocal
from .functions import get_rotation_matrix

def arpes_equation(Ek, V0, k_xy):
    kx, ky = k_xy
    """
    Calculate ARPES equation for given parameters
    
    Parameters:
    -----------
    Ek : float or array
        Kinetic energy in eV
    V0 : float
        Inner potential in eV 
    kx : float or array
        Momentum in x direction in Å⁻¹
    ky : float or array
        Momentum in y direction in Å⁻¹
        
    Returns:
    --------
    kz : float or array
        Momentum in z direction in Å⁻¹
    """
    # Constant 2m/ħ² in eV⁻¹ Å⁻²
    C = 0.2625  # Correct value for E in eV, k in Å⁻¹
    
    # Calculate kz using ARPES equation
    arg = C * (Ek + V0) - kx**2 - ky**2

    # Option A: clip (keeps array size fixed)
    arg = np.maximum(arg, 0.0)
    kz = np.sqrt(arg)
    return kz


def arpes_mesh(photon_energy,fermi_energy,binding_energy,V0,N,factor,align=None):
    Ek = photon_energy - binding_energy - fermi_energy
    k_cartesian = mesh_cartesian(N=[N,N,1],factor=factor,center=True)
    kx,ky,kz = k_cartesian.T
    kz_curve = arpes_equation(Ek, V0, [kx,ky])
    k_cartesian_curve = np.array([kx,ky,kz_curve])
    if align:
        align_k_cartesian_curve = aligned_arpes_mesh(align,k_cartesian_curve).T
        return align_k_cartesian_curve

    return k_cartesian_curve.T


def aligned_arpes_mesh(align,k_points):
    initial, final = align
    rot_mat = get_rotation_matrix(initial,final)
    return rot_mat@k_points

def arpes_mesh_plot(photon_energy,fermi_energy,binding_energy,V0,N,factor):
    kx,ky,kz = arpes_mesh(photon_energy,fermi_energy,binding_energy,V0,N,factor).T
    plt.scatter(kx, ky, c=kz, cmap='jet',s=1)
    plt.colorbar()
    plt.show()

def arpes_mesh_plot_3d(photon_energy,fermi_energy,binding_energy,V0,N,factor):
    kx,ky,kz = arpes_mesh(photon_energy,fermi_energy,binding_energy,V0,N,factor).T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(kx, ky, kz, c=kz, cmap='jet',s=1)

def binding_k(photon_energy,fermi_energy ,binding_range,binding_step, V0, N, factor,align=None):
    """
    Create a dictionary to store the k-mesh for each binding energy.
    
    Parameters:
    -----------
    photon_energy : float
        Photon energy in eV
    binding_array : array
        Array of binding energies in eV
    V0 : float
        Inner potential in eV
    N : int
        Number of points in the mesh
    factor : float
        Scale factor for the mesh
        
    Returns:
    --------
    kmesh_dict : dict
        Dictionary with binding energies as keys and (kx, ky, kz) tuples as values
    """
    # Create a dictionary to store the k-mesh for each binding energy
    kmesh_dict = {}
    binding_array = np.arange(binding_range[0],binding_range[1],binding_step)
    # Loop through each binding energy in the binding_array
    for be in binding_array:
        # Calculate the ARPES mesh for the current binding energy
        kx, ky, kz = arpes_mesh(photon_energy,fermi_energy, be, V0, N, factor,align).T
        
        # Store the results in the dictionary
        kmesh_dict[be] = (kx, ky, kz)
    
    return kmesh_dict


def arpes_path(path,g_vec,photon_energy=None,binding_energy=0,fermi_energy=0,V0=10,Ek=None):
    if Ek is None:
        if photon_energy is None:
            photon_energy = 21
        Ek = photon_energy - binding_energy - fermi_energy
    flat_angstrom = reciprocal2angstrom(path,g_vec=g_vec)
    z_curve = arpes_equation(Ek=Ek,V0=V0,k_xy=flat_angstrom.T[:2])
    # make a copy so flat_angstrom is untouched
    curved_angstrom = flat_angstrom.copy()
    # add the arpes result into the last (z-) column
    curved_angstrom[:, 2] = flat_angstrom[:, 2] + z_curve
    curved_reciprocal = angstrom2reciprocal(curved_angstrom,g_vec=g_vec)
    return curved_reciprocal
