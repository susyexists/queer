
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import electron_mass, Planck
from .model import mesh_cartesian, mesh_crystal, triangle_mesh, hexagon_crystal

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
    kz = np.sqrt(C * (Ek + V0) - kx**2 - ky**2)
    
    return kz


def arpes_mesh(light_energy,binding_energy,V0,N,factor):
    Ek = light_energy - binding_energy
    k_xy = mesh_cartesian(N=N, dimension=2,factor=factor).T-factor/2
    kx,ky = k_xy
    kz = arpes_equation(Ek, V0, k_xy)
    
    # Combine kx, ky, kz into a single array
    k_points = np.vstack((kx, ky, kz)).T
    
    # Remove points with NaN values
    k_points = k_points[~np.isnan(k_points).any(axis=1)]
    
    # Extract components for return
    kx, ky, kz = k_points.T
    return kx, ky, kz

def arpes_mesh_plot(light_energy,binding_energy,V0,N,factor):
    kx,ky,kz = arpes_mesh(light_energy,binding_energy,V0,N,factor)
    plt.scatter(kx, ky, c=kz, cmap='jet',s=1)
    plt.colorbar()
    plt.show()

def arpes_mesh_plot_3d(light_energy,binding_energy,V0,N,factor):
    kx,ky,kz = arpes_mesh(light_energy,binding_energy,V0,N,factor)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(kx, ky, kz, c=kz, cmap='jet',s=1)

def k_array(light_energy, binding_array, V0, N, factor):
    """
    Create a dictionary to store the k-mesh for each binding energy.
    
    Parameters:
    -----------
    light_energy : float
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
    
    # Loop through each binding energy in the binding_array
    for be in binding_array:
        # Calculate the ARPES mesh for the current binding energy
        kx, ky, kz = arpes_mesh(light_energy, be, V0, N, factor)
        
        # Store the results in the dictionary
        kmesh_dict[be] = (kx, ky, kz)
    
    return kmesh_dict
