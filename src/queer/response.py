import numpy as np
from numpy.linalg import inv, eigh

def green_function(H, omega, eta):
    """
    Compute the retarded Green's function G^R(ω) = [(ω + iη)I - H]^{-1}.
    H: Hamiltonian matrix (NxN)
    omega: frequency (float or array)
    eta: small broadening parameter (float)
    """
    I = np.eye(H.shape[0], dtype=complex)
    return inv((omega + 1j * eta) * I - H)

def spectral_from_green(H, omega_list, eta):
    """
    Spectral function A(ω) using Green's function inversion.
    Returns array of A(ω) = -1/π Tr[Im G^R(ω)] for each ω in omega_list.
    """
    A_inv = []
    for omega in omega_list:
        G = green_function(H, omega, eta)
        A_inv.append(-np.trace(G.imag) / np.pi)
    return np.array(A_inv)

def spectral_from_eig(evals, omega_list, eta):
    """
    Spectral function A(ω) using eigenvalue decomposition.
    Returns array of A(ω) = Σₗ [η/π]/[(ω - εₗ)² + η²] for each ω.
    """
    A_eig = []
    for omega in omega_list:
        lorentz = (eta / np.pi) / ((omega - evals)**2 + eta**2)
        A_eig.append(lorentz.sum())
    return np.array(A_eig)