# Installation Guide for QUEER

This document provides detailed instructions for installing QUEER (Quantum Utilities and Electron Engineering Resources) and its dependencies.

## Prerequisites

QUEER requires Python 3.7 or newer. The following packages are required:

- NumPy
- SciPy
- Matplotlib
- Pandas
- Joblib
- tqdm
- psutil

## Basic Installation

### Using pip (recommended)

The simplest way to install QUEER is using pip:

```bash
pip install queer
```

### Installing from source

To install the latest development version from the source code:

```bash
# Clone the repository
git clone https://github.com/yourusername/queer.git

# Navigate to the directory
cd queer

# Install in development mode
pip install -e .
```

## Step-by-Step Installation with Conda

For reproducible environments, we recommend using Conda:

```bash
# Create a new conda environment
conda create -n queer python=3.9

# Activate the environment
conda activate queer

# Install required packages
conda install numpy scipy matplotlib pandas joblib tqdm psutil

# Install QUEER
pip install queer
```

Alternatively, you can use the provided environment file:

```bash
# Create environment from the file
conda env create -f environment.yml

# Activate the environment
conda activate queer
```

## Testing the Installation

To verify that QUEER has been installed correctly, run:

```python
import queer
print(queer.__version__)
```

You should see the version number printed.

## Common Issues

### Missing dependencies

If you encounter errors related to missing dependencies, ensure all required packages are installed:

```bash
pip install -r requirements.txt
```

### Import errors

If you encounter import errors when trying to use QUEER, check that your Python environment is properly set up:

```bash
# Check which Python is being used
which python

# Check if QUEER is installed
pip list | grep queer
```

### Performance issues

For optimal performance, especially for large calculations:

1. Ensure you have a recent version of NumPy with optimized BLAS/LAPACK
2. On Linux, consider installing `numpy` with:
   ```bash
   conda install numpy scipy -c conda-forge
   ```

## Using with Quantum ESPRESSO and Wannier90

QUEER works with output files from Quantum ESPRESSO (QE) and Wannier90. To use these features:

1. Run a QE calculation and save the `nscf.out` file
2. Run Wannier90 to generate the `*_hr.dat` file
3. Point QUEER to these files when initializing the model

## Advanced Configuration

### Multi-threading

QUEER automatically uses parallel processing for computationally intensive tasks. You can control the number of threads:

```python
from queer import queer

# Limit to 4 cores
model = queer(hr="file.dat", num_core=4)
```

### Memory optimization

For very large systems, you may need to optimize memory usage:

```python
import os
# Limit OpenBLAS threads (if you're using OpenBLAS)
os.environ["OMP_NUM_THREADS"] = "4"
```