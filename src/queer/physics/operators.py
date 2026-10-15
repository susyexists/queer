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


def positive_translation(vector) -> bool:
    """Whether the first nonzero component is positive; the zero vector is not."""
    for value in vector:
        if value:
            return bool(value > 0)
    return False


def _translation_index(translations):
    lookup = {tuple(int(value) for value in row): index for index, row in enumerate(translations)}
    if len(lookup) != len(translations):
        raise ValueError("translations must be distinct")
    return lookup


def _adjoint(block):
    return np.swapaxes(block, 0, 1).conj()


def table_hermiticity(translations, matrices) -> tuple[float, float]:
    """Element and spectral measures of a table's departure from H(-R) = H(R)^dagger.

    ``matrices`` is (NR,NB,NB) on integer ``translations``. The first value is
    the largest element of ``|H(-R) - H(R)^dagger|`` over the table, a block
    without its partner counting with its own largest element. The second sums
    ``||H(-R)^dagger - H(R)||_F`` over ``R = 0`` and the positive half (first
    nonzero component positive); a block without its partner counts its own norm,
    twice in the negative half. By Weyl's inequality it bounds every eigenvalue
    difference between the averaged table and the table that keeps the positive
    half and the upper ``R = 0`` triangle with their conjugates and a real diagonal.
    """
    data = np.asarray(matrices)
    worst = bound = 0.0
    lookup = _translation_index(translations)
    for key, index in lookup.items():
        partner = lookup.get(tuple(-value for value in key))
        if partner is None:
            size = float(np.linalg.norm(data[index]))
            worst = max(worst, float(np.abs(data[index]).max(initial=0.0)))
            bound += size if positive_translation(key) else 2.0 * size
            continue
        difference = data[partner] - _adjoint(data[index])
        worst = max(worst, float(np.abs(difference).max(initial=0.0)))
        if partner == index or positive_translation(key):
            bound += float(np.linalg.norm(difference))
    return worst, bound


def hermitize_table(translations, matrices):
    """Average each ``H(R)`` with ``H(-R)^dagger`` on a translation table.

    Each pair becomes ``(H(R) + H(-R)^dagger)/2`` and its adjoint, and ``R = 0``
    becomes ``(H(0) + H(0)^dagger)/2``. A block without its partner is kept and
    its full adjoint is inserted at ``-R``. Trailing axes after the two orbital
    axes, such as stacked position components, are averaged componentwise.
    Returns the translations (input order, inserted partners appended), the
    hermitized matrices and the inserted translations.
    """
    vectors = np.asarray(translations, dtype=np.int64)
    data = np.asarray(matrices, dtype=np.complex128)
    output = data.copy()
    added: list[tuple[int, ...]] = []
    blocks = []
    lookup = _translation_index(vectors)
    for key, index in lookup.items():
        inverse = tuple(-value for value in key)
        partner = lookup.get(inverse)
        if partner is None:
            added.append(inverse)
            blocks.append(_adjoint(data[index]))
        elif partner >= index:
            average = (data[index] + _adjoint(data[partner])) / 2
            output[index] = average
            if partner != index:
                output[partner] = _adjoint(average)
    if added:
        vectors = np.concatenate((vectors, np.asarray(added, dtype=np.int64)))
        output = np.concatenate((output, np.asarray(blocks)))
    return vectors, output, added
