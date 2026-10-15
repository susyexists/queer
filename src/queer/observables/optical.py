"""Retarded independent-electron optical response in a nonorthogonal basis.

The length-gauge density matrix is evaluated with the physical Cartesian
connection, including position matrix elements and overlap derivatives. Photon
energies and linewidths are eV; fields use SI units. The response is evaluated
at z = omega + i*eta, with time dependence exp(-i*omega*t).
"""

from __future__ import annotations

from ..physics.response import linear_density as _linear_density  # noqa: F401
from ..physics.statistics import occupation_derivatives as _occupation_derivatives  # noqa: F401
from ._response_common import (  # noqa: F401 - historical private compatibility imports
    _frequency_batch,
    _linear_response,
    _occupied_manifold,
    _scissor_geometry,
    _settings,
)
from ._sampling import (
    _weighted_sum,  # noqa: F401 - historical private compatibility import
)
from .result import ObservableResult


def optical_conductivity(
    model,
    omega,
    *,
    eta=0.05,
    chemical_potential=None,
    temperature_kelvin=0.0,
    grid=(8, 8, 8),
    kpoints=None,
    weights=None,
    scissors=0.0,
    degeneracy_tol=None,
    execution=None,
):
    """Complex conductivity (Nomega,3,3), S/cm, including the Drude term.

    ``omega`` is photon energy rather than angular frequency. At zero Kelvin,
    ``eta`` also sets the Gaussian approximation to Fermi-surface delta
    functions. ``scissors`` is an additive conduction energy shift, not a target
    band gap. A single nonmagnetic spatial block is counted with its model's
    spin degeneracy; explicit spin channels are never multiplied twice.
    """
    omega, mu, points, weight, metadata = _settings(
        model,
        omega,
        eta,
        chemical_potential,
        temperature_kelvin,
        grid,
        kpoints,
        weights,
        scissors,
        degeneracy_tol,
    )
    conductivity, dielectric = _linear_response(
        model, omega, eta, mu, temperature_kelvin, points, weight, scissors, execution, degeneracy_tol
    )
    return ObservableResult(
        {"omega": omega, "conductivity": conductivity, "dielectric": dielectric},
        {
            "omega": ("omega",),
            "conductivity": ("omega", "cartesian_a", "cartesian_b"),
            "dielectric": ("omega", "cartesian_a", "cartesian_b"),
        },
        {"omega": "eV", "conductivity": "S/cm", "dielectric": "dimensionless"},
        {
            **metadata,
            "formula": "length-gauge retarded Kubo density matrix",
            "dielectric_frequency": "omega+i eta",
        },
    )


def dielectric_function(model, omega, **kwargs):
    """Relative dielectric tensor, using the same retarded Kubo response."""
    return optical_conductivity(model, omega, **kwargs)
