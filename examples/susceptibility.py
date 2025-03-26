# Import QUEER library
from queer import *
import numpy as np
import matplotlib.pyplot as plt

"""
Example script for calculating electronic susceptibility.
This is useful for studying electronic instabilities, charge density waves,
and other electronic ordering phenomena.
"""

def main():
    # Input files
    file_path = "./data/"
    nscf = "nscf.out"
    wout = "NbSe2.wout"
    hr = "NbSe2_hr.dat"
    
    # Temperature (in eV)
    T = 0.001
    
    # Define which band crosses the Fermi level (metallic band)
    metallic_band_index = 6
    
    # Create path and mesh
    q_points = 1000  # points along q-path
    k_mesh_size = 300  # size of k-mesh for integration
    
    # Generate high-symmetry path for q-vectors
    path, sym, label = GMKG(q_points)
    
    # Generate mesh for k-space integration
    mesh = mesh_cartesian([k_mesh_size, k_mesh_size, 1])
    
    # Create model
    print("Initializing model...")
    model = queer(file_path, nscf, wout, hr)
    
    # Calculate energies on the mesh
    print("Calculating energies on k-mesh...")
    mesh_energy = model.calculate_energy(mesh)
    
    # Get values for the metallic band
    metallic_band = mesh_energy[metallic_band_index]
    
    # Calculate Fermi-Dirac distribution
    mesh_fermi = np.where(metallic_band <= 0, 1.0, 0.0)  # Simplified at T=0K
    # For finite temperature: mesh_fermi = model.fermi(metallic_band, T)
    
    # Calculate susceptibility along q-path
    print("Calculating susceptibility...")
    susceptibility = []
    for q in path:
        # Real and imaginary parts of susceptibility
        chi = model.suscep(q, mesh, metallic_band, mesh_fermi, [metallic_band_index], T=T)
        susceptibility.append(chi)
    
    susceptibility = np.array(susceptibility).T
    
    # Plot susceptibility
    print("Plotting susceptibility...")
    plt.figure(figsize=(10, 6))
    
    # Real part
    plt.plot(susceptibility[0], label='Re(χ)', color='blue')
    
    # Imaginary part
    plt.plot(susceptibility[1], label='Im(χ)', color='red')
    
    # Add high-symmetry point labels
    plt.xticks(ticks=sym, labels=label)
    plt.xlim(sym[0], sym[-1])
    
    # Add vertical lines at high-symmetry points
    for i in sym[1:-1]:
        plt.axvline(i, c="black", linestyle="--", alpha=0.5)
    
    plt.legend()
    plt.ylabel("Susceptibility (a.u.)")
    plt.title("Electronic Susceptibility")
    plt.tight_layout()
    plt.savefig("./output/susceptibility.png")
    
    # Optionally, study susceptibility vs Fermi level shift
    print("Calculating susceptibility vs. Fermi level shift...")
    shifts = np.linspace(-0.2, 0.2, 20)
    peak_chi = []
    
    for shift in shifts:
        # Create model with shifted Fermi level
        model = queer(file_path, nscf, wout, hr, shift=shift)
        
        # Calculate energies and susceptibility at a specific q-vector (e.g., at K point)
        q_vector = path[sym[2]]  # K point
        mesh_energy = model.calculate_energy(mesh)
        metallic_band = mesh_energy[metallic_band_index]
        mesh_fermi = np.where(metallic_band <= 0, 1.0, 0.0)
        
        chi = model.suscep(q_vector, mesh, metallic_band, mesh_fermi, [metallic_band_index])
        peak_chi.append(chi[0])  # Real part
    
    # Plot susceptibility vs. Fermi level shift
    plt.figure(figsize=(8, 6))
    plt.plot(shifts, peak_chi, 'o-')
    plt.xlabel("Fermi Level Shift (eV)")
    plt.ylabel("Re(χ) at K-point")
    plt.title("Susceptibility vs. Fermi Level")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./output/susceptibility_vs_shift.png")
    
    print("Done! Files saved to ./output/")

if __name__ == "__main__":
    main()