# Import QUEER library
from queer import *
import numpy as np
import matplotlib.pyplot as plt

"""
Example script for working with electron-phonon coupling calculations.
This script demonstrates how to read EPW output, fix phonon branch crossings,
and calculate self-energy contributions.
"""

def main():
    # Path to EPW calculation output
    epw_path = "./epw_output/"
    
    # Number of k-points, q-points, and phonon modes
    nk = 10000
    nq = 400
    nph = 6
    
    # Initialize EPW object
    print("Initializing EPW object...")
    epw_obj = epw(path=epw_path, nk=nk, nq=nq, nph=nph)
    
    # Load data
    print("Loading EPW data...")
    epw_obj.load_data()
    
    # Fix phonon branches (untangle crossings)
    print("Fixing phonon branch crossings...")
    epw_obj.fix_model(ph_tolerance=0.15, g1_tolerance=30, g2_tolerance=10, offset=50)
    
    # Calculate reduced coupling
    print("Calculating reduced electron-phonon coupling...")
    epw_obj.reduce_g()
    
    # Plot phonon dispersion
    plt.figure(figsize=(10, 6))
    
    # Plotting each phonon branch
    for i in range(epw_obj.nph):
        plt.plot(epw_obj.ph[i], label=f"Branch {i+1}")
    
    plt.xlabel("q-point index")
    plt.ylabel("Phonon frequency (meV)")
    plt.legend()
    plt.title("Phonon Dispersion")
    plt.tight_layout()
    plt.savefig("./output/phonon_dispersion.png")
    
    # Plot electron-phonon coupling strength
    plt.figure(figsize=(10, 6))
    
    # Plotting coupling strength for each phonon branch
    for i in range(epw_obj.nph):
        # Calculate average coupling strength over k-points
        avg_coupling = np.mean(np.abs(epw_obj.g_complex[i]), axis=1)
        plt.plot(avg_coupling, label=f"Branch {i+1}")
    
    plt.xlabel("q-point index")
    plt.ylabel("Coupling strength (meV)")
    plt.legend()
    plt.title("Electron-Phonon Coupling Strength")
    plt.tight_layout()
    plt.savefig("./output/epc_strength.png")
    
    # Calculate self-energy at a specific q-point
    print("Calculating electron self-energy...")
    q_index = nq // 2  # Middle q-point
    selfen = epw_obj.calculate_self_energy(q_index)
    
    # Plot self-energy contributions from each phonon mode
    plt.figure(figsize=(10, 6))
    
    # Real part
    plt.subplot(2, 1, 1)
    for i in range(epw_obj.nph):
        plt.bar(i, selfen[i].real, alpha=0.7)
    plt.ylabel("Re(Σ) (meV)")
    plt.title(f"Self-Energy at q-point {q_index}")
    
    # Imaginary part
    plt.subplot(2, 1, 2)
    for i in range(epw_obj.nph):
        plt.bar(i, selfen[i].imag, alpha=0.7)
    plt.xlabel("Phonon branch")
    plt.ylabel("Im(Σ) (meV)")
    
    plt.tight_layout()
    plt.savefig("./output/self_energy.png")
    
    print("Done! Files saved to ./output/")

if __name__ == "__main__":
    main()