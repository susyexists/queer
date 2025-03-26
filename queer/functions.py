def fd(energy, T=0.025):
    """Fermi-Dirac distribution function."""
    import numpy as np
    return 1.0 / (np.exp(energy / T) + 1.0)