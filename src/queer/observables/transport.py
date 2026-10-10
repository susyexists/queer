"""Semiclassical Boltzmann transport in SI units with explicit scattering models."""

from __future__ import annotations

from copy import copy

import numpy as np

from ..capabilities import CapabilityError
from ..execution import ExecutionOptions
from ..geometry.lattice import validated_lattice, validated_points
from ._common import positive
from ._tb import (
    band_geometry,
    fermi_derivative,
    map_points,
    normalized_weights,
    occupations,
    velocities,
    velocity_products,
)
from .result import ObservableResult

ELECTRON_CHARGE = 1.602176634e-19
HBAR_J_SECOND = 6.62607015e-34 / (2 * np.pi)
HBAR_EV_SECOND = HBAR_J_SECOND / ELECTRON_CHARGE
KB_J_KELVIN = 1.380649e-23


def finalize_transport(arrays):
    """Reconstruct response tensors from normalized, additive CRTA moments.

    This also supports durable workflow reduction: moments and carrier densities
    must be averaged over the complete mesh before deriving nonlinear tensors.
    """
    result = dict(arrays)
    l0, l1, l2 = (np.asarray(arrays[f"transport_moment_{order}"]) for order in range(3))
    inverse = np.linalg.pinv(l0, rcond=1e-12, hermitian=True)
    temperature = np.asarray(arrays["temperatures"])[:, None, None, None]
    result["conductivity"] = ELECTRON_CHARGE * l0
    result["seebeck"] = -inverse @ l1 / temperature
    result["thermal_conductivity"] = ELECTRON_CHARGE * (l2 - l1 @ inverse @ l1) / temperature
    density = np.asarray(arrays["electron_density"]) + np.asarray(arrays["hole_density"])
    valid = density > 0
    result["mobility"] = np.divide(
        result["conductivity"],
        ELECTRON_CHARGE * density[:, :, None, None],
        out=np.full_like(l0, np.nan),
        where=valid[:, :, None, None],
    )
    result["mobility_valid_mask"] = valid
    if "inverse_mass_numerator" in arrays:
        normalization = np.asarray(arrays["inverse_mass_normalization"])
        valid = normalization > 0
        average = np.divide(
            arrays["inverse_mass_numerator"],
            normalization[:, :, None, None],
            out=np.zeros_like(l0),
            where=valid[:, :, None, None],
        )
        mass = np.linalg.pinv(average, rcond=1e-12, hermitian=True)
        result["effective_mass"] = np.where(valid[:, :, None, None], mass, np.nan)
        result["effective_mass_valid_mask"] = valid
    return result


def _axis(value, name):
    result = np.atleast_1d(np.asarray(value, dtype=float))
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite one-dimensional array")
    return result


def _band_inverse_mass(model, points, step, options):
    reciprocal = validated_lattice(model.g_vec)
    result = np.empty((model.nbnd, len(points), 3, 3))

    def at(point):
        mass = np.empty((model.nbnd, 3, 3))
        for direction in range(3):
            delta_cart = np.eye(3)[direction] * step
            delta_frac = np.linalg.solve(reciprocal.T, delta_cart)
            _, plus = velocities(model, point + delta_frac)
            _, minus = velocities(model, point - delta_frac)
            diagonal_plus = np.diagonal(plus, axis1=-2, axis2=-1).real
            diagonal_minus = np.diagonal(minus, axis1=-2, axis2=-1).real
            # d(hbar*v[eV Å])/dk[Å^-1] -> eV Å² -> inverse mass in kg^-1.
            mass[:, :, direction] = (
                (diagonal_plus - diagonal_minus).T / (2 * step) * ELECTRON_CHARGE * 1e-20 / HBAR_J_SECOND**2
            )
        return mass

    batch_size = options.bounded_batch_size(1024 * model.nbnd**2)
    for start in range(0, len(points), batch_size):
        values = map_points(model, points[start : start + batch_size], at, options)
        result[:, start : start + len(values)] = np.asarray(values).transpose(1, 0, 2, 3)
    return 0.5 * (result + result.swapaxes(-2, -1))


def boltz_transport(
    model,
    kpoints,
    chemical_potentials=None,
    temperatures=(300.0,),
    relaxation_time=None,
    transport_method="CRTA",
    deformation_potential=None,
    young_modulus=None,
    eta=0.05,
    weights=None,
    effective_mass=False,
    derivative_step=1e-5,
    electron_num=None,
    temperature_kelvin=None,
    execution=None,
):
    """Conductivity, Seebeck, electronic thermal conductivity and carrier mobility.

    Temperatures are K, chemical potentials are absolute eV, relaxation_time
    is fs, deformation_potential is eV, and young_modulus is GPa. ``EMPC`` uses
    the isotropic equipartition acoustic-deformation-potential lifetime
    ħ C / (2π k_B T D² DOS_per_volume), with an explicitly broadened DOS.
    """
    points = validated_points(kpoints)
    integration_weights = normalized_weights(points, weights)
    chemical_potentials = _axis(
        [model.fermi_energy] if chemical_potentials is None else chemical_potentials, "chemical_potentials"
    )
    if temperature_kelvin is not None:
        if not np.array_equal(np.atleast_1d(temperatures), [300.0]):
            raise ValueError("provide temperature_kelvin or temperatures, not both")
        temperatures = temperature_kelvin
    temperatures = _axis(temperatures, "temperature_kelvin")
    if np.any(temperatures <= 0):
        raise ValueError("transport temperatures must be positive kelvin")
    positive(eta, "eta")
    positive(derivative_step, "derivative_step")
    method = str(transport_method).upper()
    if method not in {"CRTA", "EMPC"}:
        raise ValueError("transport_method must be CRTA or EMPC")
    if method == "CRTA":
        if relaxation_time is None:
            raise CapabilityError("CRTA requires an explicit relaxation_time in femtoseconds")
        positive(relaxation_time, "relaxation_time")
    else:
        if deformation_potential is None or young_modulus is None:
            raise CapabilityError("EMPC requires deformation_potential and young_modulus")
        positive(deformation_potential, "deformation_potential")
        positive(young_modulus, "young_modulus")
    lattice = getattr(model, "lattice", None)
    if lattice is None:
        reciprocal = validated_lattice(model.g_vec)
        lattice = 2 * np.pi * np.linalg.inv(reciprocal).T
    volume = abs(float(np.linalg.det(validated_lattice(lattice)))) * 1e-30
    degeneracy = float(getattr(model, "spin_degeneracy", 1))
    options = (execution or getattr(model, "execution", ExecutionOptions())).resolved()
    batch_size = options.bounded_batch_size(1024 * model.nbnd**2)
    model = copy(model)
    model.execution = ExecutionOptions(
        workers=1, batch_size=1, native_threads=options.native_threads, memory_mb=options.memory_mb
    )
    energies = np.empty((model.nbnd, len(points)))
    speed = np.empty((model.nbnd, len(points), 3))
    products = np.empty((model.nbnd, len(points), 3, 3))
    resolved_bands = np.empty((model.nbnd, len(points)), dtype=bool)

    def at(point):
        data = band_geometry(model, point)
        values, matrix = data["energies"], data["velocities"]
        same_channel = data["band_channels"][:, None] == data["band_channels"][None, :]
        equal = (np.abs(values[:, None] - values[None, :]) <= data["degeneracy_tol"]) & same_channel
        np.fill_diagonal(equal, False)
        conversion = 1e-10 / HBAR_EV_SECOND
        return (
            values,
            np.diagonal(matrix, axis1=-2, axis2=-1).real.T * conversion,
            velocity_products(data) * conversion**2,
            ~equal.any(axis=1),
        )

    for start in range(0, len(points), batch_size):
        for offset, (values, diagonal, product, resolved) in enumerate(
            map_points(model, points[start : start + batch_size], at, options)
        ):
            energies[:, start + offset] = values
            speed[:, start + offset] = diagonal
            products[:, start + offset] = product
            resolved_bands[:, start + offset] = resolved
    shape = (len(temperatures), len(chemical_potentials))
    transport_moments = np.zeros((3, *shape, 3, 3))
    carrier_e, carrier_h, carrier_dos, electron_counts = (np.zeros(shape) for _ in range(4))
    relaxation = np.empty((len(temperatures), model.nbnd, len(points)))
    # For EMPC the DOS is evaluated at each band energy, not at the chemical potential.
    if method == "EMPC":
        density_at_energy = np.zeros_like(energies)
        for band in range(model.nbnd):
            for k in range(len(points)):
                for start in range(0, len(points), batch_size):
                    stop = start + batch_size
                    difference = energies[:, start:stop] - energies[band, k]
                    weighted = degeneracy * integration_weights[None, start:stop]
                    density_at_energy[band, k] += (
                        np.sum(weighted * np.exp(-0.5 * (difference / eta) ** 2) / (eta * np.sqrt(2 * np.pi)))
                        / volume
                        / ELECTRON_CHARGE
                    )
    for t, temperature in enumerate(temperatures):
        tau = relaxation[t]
        if method == "CRTA":
            tau.fill(relaxation_time * 1e-15)
        else:
            factor = (
                HBAR_J_SECOND
                * young_modulus
                * 1e9
                / (2 * np.pi * KB_J_KELVIN * temperature * (deformation_potential * ELECTRON_CHARGE) ** 2)
            )
            np.divide(factor, density_at_energy, out=tau)
        for m, mu in enumerate(chemical_potentials):
            for start in range(0, len(points), batch_size):
                stop = start + batch_size
                values = energies[:, start:stop]
                weighted = degeneracy * integration_weights[None, start:stop]
                population = occupations(values, mu, temperature)
                derivative = fermi_derivative(values, mu, temperature, broadening=eta)
                difference = values - mu
                for order in range(3):
                    prefactor = weighted * tau[:, start:stop] * derivative * difference**order / volume
                    transport_moments[order, t, m] += np.einsum(
                        "nk,nkab->ab", prefactor, products[:, start:stop], optimize=True
                    )
                carrier_e[t, m] += np.sum(weighted * population * (values >= model.fermi_energy)) / volume
                carrier_h[t, m] += (
                    np.sum(weighted * (1 - population) * (values < model.fermi_energy)) / volume
                )
                electron_counts[t, m] += np.sum(weighted * population)
                carrier_dos[t, m] += np.sum(weighted * derivative) / volume
    arrays = {
        "temperatures": temperatures,
        "chemical_potentials": chemical_potentials,
        "electron_density": carrier_e,
        "hole_density": carrier_h,
        "carrier_dos": carrier_dos,
        "electron_count": electron_counts,
        "relaxation_time": relaxation,
        "energies_absolute": energies,
        "velocities": speed,
        "intraband_velocity_products": products,
        "resolved_band_mask": resolved_bands,
    }
    arrays.update({f"transport_moment_{order}": transport_moments[order] for order in range(3)})
    dims: dict[str, tuple[str, ...]] = {
        "temperatures": ("temperature",),
        "chemical_potentials": ("chemical_potential",),
    }
    units = {
        "temperatures": "K",
        "chemical_potentials": "eV",
        "conductivity": "S/m",
        "seebeck": "V/K",
        "thermal_conductivity": "W/(m K)",
        "mobility": "m^2/(V s)",
        "electron_density": "m^-3",
        "hole_density": "m^-3",
        "carrier_dos": "states/(eV m^3)",
        "electron_count": "electrons/cell",
        "relaxation_time": "s",
        "energies_absolute": "eV",
        "velocities": "m/s",
        "intraband_velocity_products": "m^2/s^2",
        "resolved_band_mask": "1",
    }
    for name in ("conductivity", "seebeck", "thermal_conductivity", "mobility"):
        dims[name] = ("temperature", "chemical_potential", "cartesian", "cartesian_2")
    for order in range(3):
        name = f"transport_moment_{order}"
        dims[name] = ("temperature", "chemical_potential", "cartesian", "cartesian_2")
        units[name] = f"eV^{order - 1}/(m s)"
    for name in ("electron_density", "hole_density", "carrier_dos", "electron_count"):
        dims[name] = ("temperature", "chemical_potential")
    dims["relaxation_time"] = ("temperature", "band", "k")
    dims["energies_absolute"] = ("band", "k")
    dims["velocities"] = ("band", "k", "cartesian")
    dims["intraband_velocity_products"] = ("band", "k", "cartesian", "cartesian_2")
    dims["resolved_band_mask"] = ("band", "k")
    if method == "EMPC":
        arrays["scattering_dos"] = density_at_energy
        dims["scattering_dos"] = ("band", "k")
        units["scattering_dos"] = "states/(J m^3)"
    if electron_num is not None:
        if not np.isfinite(electron_num) or not 0 <= electron_num <= degeneracy * model.nbnd:
            raise ValueError("electron_num is outside the model capacity")
        arrays["net_carrier_density"] = (electron_counts - electron_num) / volume
        dims["net_carrier_density"] = ("temperature", "chemical_potential")
        units["net_carrier_density"] = "m^-3"
    if effective_mass:
        inverse_mass = _band_inverse_mass(model, points, derivative_step, options)
        inverse_mass[~resolved_bands] = np.nan
        mass_numerator = np.zeros((*shape, 3, 3))
        mass_normalization = np.zeros(shape)
        for t, temperature in enumerate(temperatures):
            for m, mu in enumerate(chemical_potentials):
                for start in range(0, len(points), batch_size):
                    stop = start + batch_size
                    weighted = degeneracy * integration_weights[None, start:stop]
                    derivative = weighted * fermi_derivative(
                        energies[:, start:stop], mu, temperature, broadening=eta
                    )
                    derivative *= resolved_bands[:, start:stop]
                    mass_normalization[t, m] += derivative.sum()
                    mass_numerator[t, m] += np.einsum(
                        "nk,nkab->ab", derivative, np.nan_to_num(inverse_mass[:, start:stop])
                    )
        arrays["inverse_mass"] = inverse_mass
        arrays["inverse_mass_numerator"] = mass_numerator
        arrays["inverse_mass_normalization"] = mass_normalization
        dims["inverse_mass"] = ("band", "k", "cartesian", "cartesian_2")
        dims["effective_mass"] = ("temperature", "chemical_potential", "cartesian", "cartesian_2")
        units["inverse_mass"], units["effective_mass"] = "kg^-1", "kg"
        dims["inverse_mass_numerator"] = dims["effective_mass"]
        dims["inverse_mass_normalization"] = ("temperature", "chemical_potential")
        units["inverse_mass_numerator"] = "1/(kg eV)"
        units["inverse_mass_normalization"] = "eV^-1"
    arrays = finalize_transport(arrays)
    dims["mobility_valid_mask"] = ("temperature", "chemical_potential")
    units["mobility_valid_mask"] = "1"
    if effective_mass:
        dims["effective_mass_valid_mask"] = ("temperature", "chemical_potential")
        units["effective_mass_valid_mask"] = "1"
    return ObservableResult(
        arrays,
        dims,
        units,
        {
            "transport_method": method,
            "cell_volume_m3": volume,
            "temperature_unit": "K",
            "chemical_potential_reference": "absolute",
            "carrier_partition": "electron_above_reference_and_hole_below_reference",
            "empc_model": "isotropic_equipartition_acoustic_deformation_potential"
            if method == "EMPC"
            else None,
            "effective_mass_definition": "fermi_derivative_weighted_inverse_curvature"
            if effective_mass
            else None,
            "velocity_products_definition": "equal_energy_same_channel_block_trace",
            "effective_mass_degeneracies": "unresolved_bands_excluded" if effective_mass else None,
        },
    )
