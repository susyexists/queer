"""Pure centered eigensystem calculations with explicit metric normalization."""

import numpy as np


def centered_pair_gap(hamiltonian, data, pair):
    """Solve an energy-centred generalized problem instead of subtracting absolute levels."""
    from scipy.linalg import eigvalsh

    v, c = pair
    reference = 0.5 * data["energies"][v] + 0.5 * data["energies"][c]
    h = np.asarray(hamiltonian)
    s = data["overlap"]
    centred = h - reference * s
    channels = data["band_channels"]
    members = np.flatnonzero(channels == channels[v])
    if len(members) != len(channels):
        # Generalized eigenvectors are only projected with the physical metric;
        # ordinary Euclidean overlaps are never used to compare channel states.
        coefficients = data["coefficients"][:, members]
        centred = coefficients.conj().T @ centred @ coefficients
        s = coefficients.conj().T @ s @ coefficients
    shifted = eigvalsh(centred, s)
    rank_v = int(np.flatnonzero(members == v)[0])
    rank_c = int(np.flatnonzero(members == c)[0])
    floor = (
        np.finfo(float).eps
        * h.shape[0]
        * (np.linalg.norm(h, ord=np.inf) + abs(reference) * np.linalg.norm(data["overlap"], ord=np.inf))
    )
    return float(shifted[rank_c] - shifted[rank_v]), float(reference), float(floor)
