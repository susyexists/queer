"""Green functions, Lorentzian spectra, and electronic susceptibility."""

import numpy as np

from ..physics.electronic import green_function as green_function
from ..physics.electronic import lindhard
from ..physics.electronic import spectral_from_eig as spectral_from_eig
from ..physics.electronic import spectral_from_green as spectral_from_green
from ..physics.statistics import thermal_occupation as fd
from ._common import model_energies, points_array, positive


def susceptibility(
    model,
    q_path,
    k_mesh,
    band_indices=None,
    temperature=0.025,
    broadening=1e-7,
    execution=None,
    mesh_energies=None,
):
    """Lindhard response in 1/eV at q points, with fractional reciprocal inputs.

    Returns a complex array (Nq,). This conventional response differs from the
    historical model.suscep pair, whose second component is a joint DOS.
    """
    broadening = positive(broadening, "broadening")
    q_path = points_array(q_path)
    k_mesh = points_array(k_mesh)
    if not len(k_mesh) or not np.isfinite(k_mesh).all() or not np.isfinite(q_path).all():
        raise ValueError("k_mesh must be nonempty and all momentum points finite")
    energies = (
        model_energies(model, k_mesh, execution)
        if mesh_energies is None
        else np.asarray(mesh_energies, dtype=float)
    )
    if energies.ndim != 2 or energies.shape[1] != len(k_mesh) or not np.isfinite(energies).all():
        raise ValueError("mesh_energies must be a finite (Nb,Nk) array")
    indices = np.arange(len(energies)) if band_indices is None else np.asarray(band_indices)
    if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
        raise ValueError("band_indices must be a one-dimensional integer list")
    indices = indices.astype(int)
    if np.any(indices < 0) or np.any(indices >= len(energies)):
        raise ValueError("band index outside the model")
    energies = energies[indices]
    occupations = fd(energies, temperature)
    result = np.empty(len(q_path), dtype=complex)
    for i, point in enumerate(q_path):
        shifted = model_energies(model, k_mesh + point, execution)[indices]
        shifted_fd = fd(shifted, temperature)
        result[i] = lindhard(energies, shifted, occupations, shifted_fd, broadening)
    return result


def susceptibility_point(model, point, k_mesh, **kwargs):
    return susceptibility(model, np.asarray(point).reshape(1, 3), k_mesh, **kwargs)[0]
