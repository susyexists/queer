# Import QUEER library
from queer import *
import matplotlib.pyplot as plt
import numpy as np

"""
Example script for calculating and plotting electronic band structure
along high-symmetry paths in the Brillouin zone.
"""

def main():
    # Input files
    file_path = "./input/"
    nscf = "nscf.out"
    wout = "NbSe2.wout"
    hr = "NbSe2_hr.dat"
    
    # Number of k points along the path
    k_points = 1000
    
    # Energy axis limits
    ylim = [-2, 2]
    
    # Generate path along GMKG points (standard path for hexagonal lattices)
    path, sym, label = GMKG(k_points)
    
    # Create model
    model = queer(file_path, nscf, wout, hr)
    
    # Optional: apply a shift to the Fermi energy if needed
    # model = queer(file_path, nscf, wout, hr, shift=0.1)
    
    # Calculate band dispersion along the path
    print("Calculating band structure...")
    bands = model.calculate_energy(path)
    
    # Plot bands
    print("Plotting band structure...")
    model.plot_electron_path(
        bands, sym, label, ylim, save="./output/band_structure.png")
    
    # Optional: Calculate and plot DOS
    print("Calculating density of states...")
    energies = np.linspace(ylim[0], ylim[1], 1000)
    dos = np.zeros_like(energies)
    
    for i, energy in enumerate(energies):
        # Use delta function approximation for DOS
        dos[i] = np.sum([density_of_states(b, dE=0.05) for b in bands])
    
    plt.figure(figsize=(8, 6))
    plt.plot(dos, energies)
    plt.xlabel("DOS (a.u.)")
    plt.ylabel("Energy (eV)")
    plt.axhline(0, color='red', linestyle='--', alpha=0.7)
    plt.title("Density of States")
    plt.tight_layout()
    plt.savefig("./output/dos.png")
    
    print("Done! Files saved to ./output/")

if __name__ == "__main__":
    main()