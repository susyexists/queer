"""Dense orthogonal Fourier assembly and real-k Hermitian validation.

Complex-k analytic continuation is deliberately a separate model capability.
"""

import numpy as np

from .validation import NumericalError


def assemble(translations, hopping, points):
    phase = np.exp(2j * np.pi * (points @ translations.T))
    matrices = np.einsum("kr,rij->kij", phase, hopping, optimize=True)
    return validate_assembly(matrices, points)


def validate_assembly(matrices, points):
    if not np.isfinite(matrices).all():
        raise NumericalError("Hamiltonian assembly produced nonfinite values")
    adjoints = matrices.conj().transpose(0, 2, 1)
    residuals = np.linalg.norm(matrices - adjoints, ord=np.inf, axis=(1, 2))
    norms = np.linalg.norm(matrices, ord=np.inf, axis=(1, 2))
    invalid = np.flatnonzero(residuals > 1e-10 + 1e-10 * norms)
    if invalid.size:
        index = int(invalid[0])
        raise NumericalError(
            f"Non-Hermitian Hamiltonian at k={points[index].tolist()}; residual={residuals[index]:.6g} eV"
        )
    return (matrices + adjoints) * 0.5, float(residuals.max(initial=0.0))
