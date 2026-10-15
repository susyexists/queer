"""Pure length-gauge density-matrix response kernels.

Photon energies are eV, connections are Angstrom and time dependence is
exp(-i omega t). Model evaluation and physical unit prefactors stay in callers.
"""

import numpy as np

from .statistics import occupation_derivatives


def linear_density(data, z, mu, temperature, eta):
    energy = data["energies"]
    dipole = data["connections"]
    f, fp, _ = occupation_derivatives(energy, mu, temperature, eta)
    source = dipole * (f[None, :] - f[:, None])[None]
    delta = energy[:, None] - energy[None, :]
    block = (np.abs(delta) <= data["degeneracy_tol"]) & (
        data["band_channels"][:, None] == data["band_channels"][None, :]
    )
    source += 1j * 0.5 * (fp[:, None] + fp[None, :])[None] * data["velocities"] * block[None]
    return source[None] / (z[:, None, None, None] - delta[None, None])


def linear_density_and_derivative(data, gradient, energy_gradient, hessian, z, mu, temperature, eta):
    energy = data["energies"]
    connection = data["connections"]
    f, fp, fpp = occupation_derivatives(energy, mu, temperature, eta)
    difference = f[None, :] - f[:, None]
    first = 0.5 * (fp[:, None] + fp[None, :])
    second = 0.5 * (fpp[:, None] + fpp[None, :])
    df = first[None] * energy_gradient
    ddf = (
        0.5
        * second[None, None]
        * (
            np.einsum("anl,blm->abnm", energy_gradient, energy_gradient, optimize=True)
            + np.einsum("bnl,alm->abnm", energy_gradient, energy_gradient, optimize=True)
        )
        + first[None, None] * hessian
    )
    source = connection * difference[None]
    dsource = gradient * difference[None, None]
    dsource += np.einsum("bnl,alm->abnm", connection, df, optimize=True) - np.einsum(
        "anl,blm->abnm", df, connection, optimize=True
    )
    source += 1j * df
    dsource += 1j * ddf
    delta = energy[:, None] - energy[None, :]
    denominator = z[:, None, None, None] - delta[None, None]
    rho = source[None] / denominator
    drho = dsource[None] + np.einsum("anl,wblm->wabnm", energy_gradient, rho, optimize=True)
    drho -= np.einsum("wbnl,alm->wabnm", rho, energy_gradient, optimize=True)
    drho /= denominator[:, None]
    # [i partial_a + A_a, rho_b]. This includes both intraband derivatives and
    # interband commutators and is covariant under band-phase changes.
    covariant = 1j * drho
    covariant += np.einsum("anl,wblm->wabnm", connection, rho, optimize=True)
    covariant -= np.einsum("wbnl,alm->wabnm", rho, connection, optimize=True)
    return rho, covariant
