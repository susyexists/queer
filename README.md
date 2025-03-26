=======
# QUEER: Quantum Utilities and Electron Engineering Resources

A comprehensive tight-binding numerical library for condensed matter physics calculations.

## Overview

QUEER (Quantum Utilities and Electron Engineering Resources) is a Python library designed for numerical simulations in condensed matter physics, focusing on tight-binding models. The package allows for efficient calculation of electronic band structures, Fermi surfaces, susceptibility calculations, and electron-phonon coupling effects.

## Key Features

- **Electronic Band Structure**: Calculate and visualize electronic band structures along high-symmetry paths
- **Fermi Surface Mapping**: Generate and visualize Fermi surfaces in 2D and 3D
- **Susceptibility Calculations**: Compute electronic susceptibility for studying electronic instabilities
- **Electron-Phonon Coupling**: Tools for analyzing electron-phonon interactions and self-energy calculations
- **Parallel Computing Support**: Efficient parallel implementations for computationally intensive tasks

## Installation

```bash
# Clone the repository
git clone https://github.com/susyexists/queer.git

# Navigate to the directory
cd queer

# Install the package
pip install -e .
```

## Dependencies

QUEER requires the following packages:
- NumPy
- SciPy
- Matplotlib
- Pandas
- Joblib

You can install all dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Quick Start

### Band Structure Calculation

```python
from queer import *

# Input files
file_path = "./input/"
nscf = "nscf.out"
wout = "NbSe2.wout"
hr = "NbSe2_hr.dat"

# Define path along high-symmetry points
k_points = 1000
path, sym, label = GMKG(k_points)

# Create model
model = queer(file_path, nscf, wout, hr)

# Calculate dispersion
band = model.parallel_solver(path)

# Plot bands
model.plot_electron_path(band, sym, label, ylim=[-2, 2], save="./output/band_path.png")
```

### Susceptibility Calculation

```python
# Import library
from queer import *

# Define input files
file_path = "./data/"
nscf = "nscf.out"
wout = "NbSe2.wout"
hr = "NbSe2_hr.dat"

# Parameters
T = 0.001
metallic_band_index = 6

# Create mesh grid
q_points = 1000
k_mesh = 300
path, sym, label = GMKG(q_points)
mesh = mesh_2d(k_mesh)

# Create model and calculate susceptibility
model = queer(file_path, nscf, wout, hr)
mesh_energy = model.parallel_solver(mesh)[metallic_band_index]
mesh_fermi = model.fermi(mesh_energy)
sus_mesh = [model.suscep(q, mesh, mesh_energy, mesh_fermi) for q in path]

# Plot susceptibility
plot_susceptibility(sus_mesh, sym, label, save=True)
```

## Documentation

For detailed documentation of all available functions and classes, please see the [docs](./docs) directory.

## Examples

The `scripts` directory contains example scripts demonstrating different functionalities:

- `plot_path.py`: Calculates and plots band structure along high-symmetry paths
- `plot_mesh.py`: Generates 2D k-space meshes and band calculations
- `susceptibility.py`: Computes electronic susceptibility for studying instabilities
- `selfen.py`: Calculates electron-phonon self-energy contributions

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use QUEER in your research, please cite:

```
@software{queer,
  author = {Susy Exists},
  title = {QUEER: Quantum Utilities and Electron Engineering Resources},
  url = {https://github.com/susyexists/queer},
  year = {2025},
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
>>>>>>> a7c92fa (initial migration)
