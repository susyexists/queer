"""Independent-particle second-order optical and photocurrent tensors.

SHG and electro-optic susceptibilities use the length-gauge density-matrix
recursion (Ventura et al., PRB 96, 035431). Shift and injection currents use
the resonant Sipe--Shkrebtii and de Juan et al. expressions. All Cartesian
dipoles include the nonorthogonal basis connection; no engine sign flip is
applied. Electric fields are complex Fourier amplitudes in V/m.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import epsilon_0

from ..execution import ExecutionOptions
from ..models.evaluation import (
    band_geometry,
    basis_curvature,
    derivatives,
    reciprocal,
    spin_degeneracy,
    volume,
)
from ..physics.broadening import gaussian, lorentzian
from ..physics.constants import E_CHARGE, HBAR
from ..physics.geometry import differentiated_geometry, energy_blocks
from ..physics.response import linear_density_and_derivative as _linear_density_and_derivative
from ..physics.statistics import occupation_derivatives as _occupation_derivatives
from ._response_common import (
    _frequency_batch,
    _linear_response,
    _occupied_manifold,
    _scissor_geometry,
    _settings,
)
from ._sampling import _weighted_sum, map_points
from .result import ObservableResult


def _differentiated_geometry(model, k, data):
    return differentiated_geometry(data, derivatives(model, k), derivatives(model, k, order=2))


def _second_order_response(
    model, omega1, omega2, eta, mu, temperature, points, weight, scissors, execution, degeneracy_tol=None
):
    z1 = omega1 + 1j * eta
    z2 = omega2 + 1j * eta
    zout = z1 + z2
    options = (execution or getattr(model, "execution", None) or ExecutionOptions()).resolved()
    frequency_batch = _frequency_batch(model, len(z1), options, 128, 60, 8)

    def point_response(k):
        raw = band_geometry(model, k, degeneracy_tol)
        gradient, energy_gradient, hessian = _differentiated_geometry(model, k, raw)
        curvature = basis_curvature(model, k, raw)
        data = _scissor_geometry(raw, mu, scissors)
        delta = data["energies"][:, None] - data["energies"][None, :]
        answer = np.empty((len(z1), 3, 3, 3), complex)
        for start in range(0, len(z1), frequency_batch):
            stop = start + frequency_batch
            rho1, first = _linear_density_and_derivative(
                data, gradient, energy_gradient, hessian, z1[start:stop], mu, temperature, eta
            )
            if np.array_equal(z1, z2):
                rho_other, second = rho1, first
            else:
                rho_other, second = _linear_density_and_derivative(
                    data, gradient, energy_gradient, hessian, z2[start:stop], mu, temperature, eta
                )
            # Intrinsic permutation exchanges the two input frequencies. The
            # tensor convention P=epsilon0*chi2:E:E has no extra factorial.
            rho2 = 0.5 * (second + first.swapaxes(1, 2)) / (zout[start:stop, None, None, None, None] - delta)
            answer[start:stop] = np.einsum("anm,wbcmn->wabc", data["velocities"], rho2)
            # Projecting coordinates into a finite basis can give [R_a,R_b]
            # = i F_ab. The electric-field part of the Heisenberg current must
            # therefore be retained alongside the zero-field velocity.
            answer[start:stop] += 0.5 * (
                np.einsum("abnm,wcmn->wabc", curvature, rho_other)
                + np.einsum("acnm,wbmn->wabc", curvature, rho1)
            )
        return answer

    bytes_per_point = 16 * (model.nbnd**2 * (128 + 60 * frequency_batch) + 27 * len(z1))
    current = _weighted_sum(
        model, points, weight, point_response, (len(z1), 3, 3, 3), options, bytes_per_point
    )
    chi = -1j * spin_degeneracy(model) * E_CHARGE / (epsilon_0 * volume(model))
    return chi * current / zout[:, None, None, None] * 1e12


def _result(omega, tensor, name, units, metadata, **extra):
    arrays = {"omega": omega, name: tensor, **extra}
    dims = {"omega": ("omega",), name: ("omega", "cartesian_a", "cartesian_b", "cartesian_c")}
    for key, value in extra.items():
        dims[key] = ("omega", "cartesian_a", "cartesian_b") if value.ndim == 3 else dims[name]
    return ObservableResult(
        arrays, dims, {"omega": "eV", name: units, **{k: "dimensionless" for k in extra}}, metadata
    )


def shg(
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
    """Second-harmonic susceptibility chi2(-2w;w,w), complex pm/V.

    P_a(2w)=epsilon0*chi2_abc*E_b(w)*E_c(w). The output linewidth is 2*eta,
    the sum of the two incoming retarded linewidths.
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
    tensor = _second_order_response(
        model, omega, omega, eta, mu, temperature_kelvin, points, weight, scissors, execution, degeneracy_tol
    )
    return _result(
        omega,
        tensor,
        "shg",
        "pm/V",
        {
            **metadata,
            "formula": "covariant length-gauge density-matrix recursion",
            "incoming_frequencies": "(omega,omega)",
            "output_linewidth_eV": 2 * eta,
            "polarization_convention": "P_a=epsilon0*chi2_abc*E_b*E_c; no factorial",
        },
    )


def pockels_from_tensors(chi2, dielectric):
    """Convert equally normalized Cartesian susceptibilities to pm/V.

    ``chi2`` has shape (Nomega,3,3,3), ``dielectric`` (Nomega,3,3).
    This contraction follows BZ integration and is not itself a linear sum.
    """
    chi2 = np.asarray(chi2)
    dielectric = np.asarray(dielectric)
    if chi2.ndim != 4 or chi2.shape[1:] != (3, 3, 3) or dielectric.shape != (len(chi2), 3, 3):
        raise ValueError("chi2/dielectric must have matching (Nomega,3,3,3)/(Nomega,3,3) shapes")
    if not np.isfinite(chi2).all() or not np.isfinite(dielectric).all():
        raise ValueError("chi2/dielectric must be finite")
    inverse = np.linalg.inv(dielectric)
    return -2 * np.einsum("wai,wijc,wjb->wabc", inverse, chi2, inverse)


def pockels(
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
    """Electro-optic impermeability tensor r_abc, complex pm/V.

    Delta(epsilon^-1)_ab=r_abc*E_dc,c. The full Cartesian contraction
    r=-2*epsilon^-1*chi2(-w;w,0)*epsilon^-1 is used, including anisotropic
    or nondiagonal dielectric tensors. The accompanying ``chi2`` is pm/V.
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
    chi = _second_order_response(
        model,
        omega,
        np.zeros_like(omega),
        eta,
        mu,
        temperature_kelvin,
        points,
        weight,
        scissors,
        execution,
        degeneracy_tol,
    )
    _, dielectric = _linear_response(
        model, omega, eta, mu, temperature_kelvin, points, weight, scissors, execution, degeneracy_tol
    )
    electrooptic = pockels_from_tensors(chi, dielectric)
    result = _result(
        omega,
        electrooptic,
        "pockels",
        "pm/V",
        {
            **metadata,
            "formula": "covariant length-gauge electro-optic susceptibility",
            "incoming_frequencies": "(omega,0)",
            "impermeability_convention": "delta epsilon^-1=r:E_dc",
        },
        chi2=chi,
        dielectric=dielectric,
    )
    return ObservableResult(result.arrays, result.dims, {**result.units, "chi2": "pm/V"}, result.metadata)


def second_order_static(
    model,
    *,
    eta=1e-4,
    chemical_potential=None,
    temperature_kelvin=0.0,
    grid=(8, 8, 8),
    kpoints=None,
    weights=None,
    scissors=0.0,
    degeneracy_tol=None,
    execution=None,
):
    """Static electric susceptibility chi2(0;0,0), real pm/V.

    ``eta`` is an adiabatic imaginary-frequency regulator; reduce it below
    the insulating gap and verify convergence. This electric susceptibility
    requires an insulating occupied manifold, with the chemical potential at
    least ten regulator widths from every sampled band.
    """
    omega, mu, points, weight, metadata = _settings(
        model,
        [0.0],
        eta,
        chemical_potential,
        temperature_kelvin,
        grid,
        kpoints,
        weights,
        scissors,
        degeneracy_tol,
    )
    occupied_counts = None
    for k in points:
        data = _scissor_geometry(band_geometry(model, k, degeneracy_tol), mu, scissors)
        energy = data["energies"]
        if np.min(np.abs(energy - mu), initial=np.inf) <= 10 * eta:
            raise ValueError(
                "second_order_static requires an insulating chemical potential separated from bands"
            )
        counts = {
            str(int(channel)): int(np.sum((energy < mu) & (data["band_channels"] == channel)))
            for channel in np.unique(data["band_channels"])
        }
        if occupied_counts is not None and counts != occupied_counts:
            raise ValueError(
                "second_order_static requires constant occupied rank throughout the sampled mesh"
            )
        occupied_counts = counts
        f, _, _ = _occupation_derivatives(energy, mu, temperature_kelvin, eta)
        if np.max(np.minimum(f, 1 - f), initial=0) > 1e-10:
            raise ValueError("second_order_static requires insulating occupations; reduce temperature_kelvin")
    tensor = _second_order_response(
        model, omega, omega, eta, mu, temperature_kelvin, points, weight, scissors, execution, degeneracy_tol
    )[0].real
    return ObservableResult(
        {"second_order_static": tensor},
        {"second_order_static": ("cartesian_a", "cartesian_b", "cartesian_c")},
        {"second_order_static": "pm/V"},
        {
            **metadata,
            "formula": "adiabatic covariant length-gauge static susceptibility",
            "occupied_rank_by_channel": occupied_counts,
        },
    )


def _delta(energy, eta, smearing):
    if smearing == "lorentzian":
        return lorentzian(energy, eta)
    if smearing == "gaussian":
        return gaussian(energy, eta)
    raise ValueError("smearing must be 'lorentzian' or 'gaussian'")


def _sternheimer_geometry(model, k, data):
    """Solve constrained generalized perturbation equations in orbital space."""
    coeff, metric = data["coefficients"], data["overlap"]
    energy = data["energies"]
    hamiltonian = model.hamiltonian(k)
    first = derivatives(model, k)
    derivative = np.empty_like(data["coefficient_derivatives"])
    for band, eigenvalue in enumerate(energy):
        # Constrain the entire nullspace, including equal opposite-spin states.
        block = np.flatnonzero(np.abs(energy - eigenvalue) <= data["degeneracy_tol"])
        basis = coeff[:, block]
        constraint = metric @ basis
        system = np.block(
            [
                [hamiltonian - eigenvalue * metric, constraint],
                [constraint.conj().T, np.zeros((len(block), len(block)), complex)],
            ]
        )
        for axis in range(3):
            rhs = -(first["H"][axis] - eigenvalue * first["S"][axis]) @ coeff[:, band]
            normalization = -0.5 * basis.conj().T @ first["S"][axis] @ coeff[:, band]
            try:
                solution = np.linalg.solve(system, np.concatenate((rhs, normalization)))
            except np.linalg.LinAlgError as error:
                raise ValueError("singular constrained Sternheimer system") from error
            residual = system @ solution - np.concatenate((rhs, normalization))
            if np.linalg.norm(residual) > 1e-9 * (1 + np.linalg.norm(rhs)):
                raise ValueError("Sternheimer derivative failed its residual check")
            derivative[axis, :, band] = coeff.conj().T @ metric @ solution[: len(energy)]
    same = data["band_channels"][:, None] == data["band_channels"][None, :]
    derivative[:, ~same] = 0
    connection = 1j * derivative + data["projected_derivatives"]["r"].conj().transpose(0, 2, 1)
    return {
        **data,
        "coefficient_derivatives": derivative,
        "connections": (connection + connection.conj().transpose(0, 2, 1)) / 2,
    }


def _transition_selection(data, pairs, occupied):
    n = len(data["energies"])
    mask = np.ones((n, n), bool)
    if pairs is None:
        return mask
    pairs = np.asarray(pairs)
    if pairs.shape == (2,):
        pairs = pairs[None]
    if (
        pairs.ndim != 2
        or pairs.shape[1] != 2
        or pairs.dtype.kind not in "iu"
        or (np.any(pairs < 0) or np.any(pairs >= n))
        or len(np.unique(pairs, axis=0)) != len(pairs)
    ):
        raise ValueError("band_pairs must be distinct zero-based occupied/unoccupied index pairs")
    mask[:] = False
    for source, target in pairs:
        if data["band_channels"][source] != data["band_channels"][target]:
            raise ValueError("band_pairs cannot connect different spin channels")
        if not occupied[source] or occupied[target]:
            raise ValueError("band_pairs must connect occupied to unoccupied states")
        mask[source, target] = True
    blocks = energy_blocks(data)
    for left in blocks:
        for right in blocks:
            selected = mask[np.ix_(left, right)]
            if selected.any() and not selected.all():
                raise ValueError("band_pairs cut a degenerate transition manifold")
    return mask


def shift_current(
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
    smearing="lorentzian",
    occupied_bands=None,
    band_pairs=None,
    method="analytic",
    scissors_occupied_bands=None,
    bin_width=None,
    adaptive_scale=1.0,
    adaptive_min=1e-4,
    adaptive_max=1.0,
    adaptive_spacing=None,
    integration=None,
    execution=None,
    checkpoint=None,
):
    """Resonant dc shift conductivity sigma_abc(0;w,-w), microA/V^2.

    The optical input indices b,c are symmetric. The shift-vector convention
    is R_nm= -partial(arg r_nm)+A_nn-A_mm; electron charge is -e.
    """
    from .integration import integrate_unit_box, nested_observable, validate_integration

    controls = validate_integration(integration)
    if controls.get("threshold_quantity") == "curvature":
        raise ValueError("shift-current thresholds use response units microA/V^2, not curvature")
    if controls.get("method") == "adaptive":
        if kpoints is not None or weights is not None:
            raise ValueError("adaptive shift-current integration requires a grid")
        import inspect

        values = locals().copy()
        arguments = {
            key: values[key]
            for key in inspect.signature(shift_current).parameters
            if key not in {"model", "omega", "grid", "integration"}
        }
        return nested_observable(
            lambda refined: shift_current(model, omega, grid=refined, **arguments),
            grid,
            controls,
            "shift_current",
            checkpoint=checkpoint,
        )
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
    if method not in {"analytic", "sum_over_states", "sternheimer"}:
        raise ValueError("shift-current method must be analytic, sum_over_states, or sternheimer")
    if occupied_bands is not None and temperature_kelvin:
        raise ValueError("explicit occupied_bands requires zero temperature")
    if smearing not in {"lorentzian", "gaussian", "none", "adaptive"}:
        raise ValueError("smearing must be lorentzian, gaussian, none, or adaptive")
    if smearing == "none":
        if bin_width is None:
            if len(omega) < 2 or not np.allclose(np.diff(omega), np.diff(omega)[0]) or np.diff(omega)[0] <= 0:
                raise ValueError(
                    "binned no-smearing requires uniform ascending omega or an explicit bin_width"
                )
            bin_width = float(np.diff(omega)[0])
        if not np.isfinite(bin_width) or bin_width <= 0:
            raise ValueError("bin_width must be finite and positive in eV")
    if smearing == "adaptive":
        if not np.isfinite([adaptive_scale, adaptive_min, adaptive_max]).all() or (
            adaptive_scale <= 0 or adaptive_min <= 0 or adaptive_max < adaptive_min
        ):
            raise ValueError("adaptive Gaussian scale and width bounds must be positive and ordered")
        if kpoints is not None and adaptive_spacing is None:
            raise ValueError("adaptive smearing on explicit kpoints requires adaptive_spacing")
        spacing = (
            reciprocal(model) / np.asarray(grid)[:, None]
            if adaptive_spacing is None
            else np.asarray(adaptive_spacing, float)
        )
        if spacing.shape != (3, 3) or not np.isfinite(spacing).all():
            raise ValueError("adaptive_spacing must contain three finite Cartesian reciprocal cell steps")
        if adaptive_spacing is None and np.any(np.asarray(grid) > 1):
            spacing[np.asarray(grid) == 1] = 0
    options = (execution or getattr(model, "execution", None) or ExecutionOptions()).resolved()
    frequency_batch = _frequency_batch(model, len(omega), options, 180, 2, 64)

    def point_response(k):
        raw = band_geometry(model, k, degeneracy_tol)
        if method == "sternheimer":
            raw = _sternheimer_geometry(model, k, raw)
        gradient, _, _ = _differentiated_geometry(model, k, raw)
        occupied = _occupied_manifold(raw, occupied_bands, mu)
        pair_mask = _transition_selection(raw, band_pairs, occupied)
        count = occupied_bands if scissors_occupied_bands is None else scissors_occupied_bands
        data = _scissor_geometry(raw, mu, scissors, count)
        energy = data["energies"]
        connection = data["connections"]
        f, _, _ = _occupation_derivatives(energy, mu, temperature_kelvin, eta)
        if occupied_bands is not None:
            f = occupied.astype(float)
        gap = energy[None, :] - energy[:, None]
        block = (np.abs(gap) <= raw["degeneracy_tol"]) & (
            raw["band_channels"][:, None] == raw["band_channels"][None, :]
        )
        block_connection = connection * block[None]
        generalized = gradient - 1j * (
            np.einsum("anl,blm->abnm", block_connection, connection, optimize=True)
            - np.einsum("bnl,alm->abnm", connection, block_connection, optimize=True)
        )
        factor = (f[:, None] - f[None, :]) * (gap > raw["degeneracy_tol"]) * pair_mask
        if smearing == "adaptive":
            velocities = np.diagonal(raw["velocities"], axis1=1, axis2=2).real.T
            cell_changes = (velocities[:, None] - velocities[None, :]) @ spacing.T
            width = np.clip(
                adaptive_scale * np.linalg.norm(cell_changes, axis=-1) / np.sqrt(12),
                adaptive_min,
                adaptive_max,
            )
        integrand = np.einsum("bmn,acnm->abcnm", connection, generalized)
        integrand = 0.5 * (integrand + integrand.swapaxes(1, 2)).imag
        answer = np.empty((len(omega), 3, 3, 3))
        for start in range(0, len(omega), frequency_batch):
            stop = start + frequency_batch
            difference = omega[start:stop, None, None] - gap
            if smearing == "none":
                lines = ((difference >= -bin_width / 2) & (difference < bin_width / 2)) / bin_width
            elif smearing == "adaptive":
                lines = _delta(difference, width, "gaussian")
            else:
                lines = _delta(difference, eta, smearing)
            lines = lines * factor
            answer[start:stop] = np.einsum("wnm,abcnm->wabc", lines, integrand)
        return answer

    bytes_per_point = 16 * (model.nbnd**2 * (180 + 2 * frequency_batch) + 27 * len(omega))
    conversion = -np.pi * E_CHARGE**2 / HBAR * 1e6 * spin_degeneracy(model) / volume(model)
    if controls.get("method") in {"threshold", "cubature"}:
        if kpoints is not None or weights is not None:
            raise ValueError(
                "local shift-current integration requires a grid without explicit kpoints/weights"
            )
        integral = integrate_unit_box(
            lambda points: conversion * np.asarray(map_points(model, points, point_response, options)),
            grid,
            controls,
            namespace="shift_current",
            checkpoint=checkpoint,
        )
        tensor = integral.value.real
        metadata["integration"] = integral.metadata
    else:
        tensor = (
            _weighted_sum(
                model, points, weight, point_response, (len(omega), 3, 3, 3), options, bytes_per_point
            ).real
            * conversion
        )
    return _result(
        omega,
        tensor,
        "shift_current",
        "microA/V^2",
        {
            **metadata,
            "formula": "Sipe-Shkrebtii resonant generalized-dipole expression",
            "smearing": smearing,
            "method": method,
            "occupied_bands": occupied_bands,
            "band_pairs": None if band_pairs is None else np.asarray(band_pairs).tolist(),
            "scissors_occupied_bands": scissors_occupied_bands,
            "bin_width_eV": bin_width,
            "adaptive_width_definition": "Gaussian stddev from RMS transition change across reciprocal cell"
            if smearing == "adaptive"
            else None,
            "electron_charge": "-e",
            "incoming_frequencies": "(omega,-omega)",
        },
    )


def cpge(
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
    smearing="lorentzian",
    relaxation_time_seconds=None,
    execution=None,
):
    """Circular injection-current rate, complex A/(V^2 s).

    dj_a/dt=cpge_abc*E_b*conj(E_c). The tensor is purely imaginary and
    antisymmetric in b,c. ``helicity_rate`` is the equivalent real tensor for
    dj_a/dt=helicity_rate_ab*[i E cross conj(E)]_b. A supplied positive
    relaxation time also yields the steady coefficient in microA/V^2.
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
    _delta(np.array([0.0]), eta, smearing)
    options = (execution or getattr(model, "execution", None) or ExecutionOptions()).resolved()
    frequency_batch = _frequency_batch(model, len(omega), options, 90, 2, 64)
    if relaxation_time_seconds is not None and (
        not np.isfinite(relaxation_time_seconds) or relaxation_time_seconds <= 0
    ):
        raise ValueError("relaxation_time_seconds must be positive and finite")

    def point_response(k):
        raw = band_geometry(model, k, degeneracy_tol)
        data = _scissor_geometry(raw, mu, scissors)
        energy = data["energies"]
        connection = data["connections"]
        f, _, _ = _occupation_derivatives(energy, mu, temperature_kelvin, eta)
        gap = energy[None, :] - energy[:, None]
        block = (np.abs(gap) <= raw["degeneracy_tol"]) & (
            raw["band_channels"][:, None] == raw["band_channels"][None, :]
        )
        block_velocity = data["velocities"] * block[None]
        factor = (f[:, None] - f[None, :]) * (gap > raw["degeneracy_tol"])
        velocity_commutator = np.einsum("cnl,alm->acnm", connection, block_velocity, optimize=True)
        velocity_commutator -= np.einsum("anl,clm->acnm", block_velocity, connection, optimize=True)
        circular = 1j * np.einsum("bmn,acnm->abcnm", connection, velocity_commutator).imag
        circular = 0.5 * (circular - circular.swapaxes(1, 2))
        answer = np.empty((len(omega), 3, 3, 3), complex)
        for start in range(0, len(omega), frequency_batch):
            stop = start + frequency_batch
            lines = _delta(omega[start:stop, None, None] - gap, eta, smearing) * factor
            answer[start:stop] = np.einsum("wnm,abcnm->wabc", lines, circular)
        return answer

    bytes_per_point = 16 * (model.nbnd**2 * (90 + 2 * frequency_batch) + 27 * len(omega))
    tensor = _weighted_sum(
        model, points, weight, point_response, (len(omega), 3, 3, 3), options, bytes_per_point
    )
    tensor *= -2 * np.pi * E_CHARGE**3 / HBAR**2 * spin_degeneracy(model) / volume(model)
    helicity = np.stack((tensor[:, :, 1, 2].imag, tensor[:, :, 2, 0].imag, tensor[:, :, 0, 1].imag), axis=-1)
    arrays = {"omega": omega, "cpge": tensor, "helicity_rate": helicity}
    dims = {
        "omega": ("omega",),
        "cpge": ("omega", "cartesian_a", "cartesian_b", "cartesian_c"),
        "helicity_rate": ("omega", "cartesian_a", "cartesian_b"),
    }
    units = {"omega": "eV", "cpge": "A/(V^2 s)", "helicity_rate": "A/(V^2 s)"}
    if relaxation_time_seconds is not None:
        arrays["steady_cpge"] = tensor * relaxation_time_seconds * 1e6
        dims["steady_cpge"] = dims["cpge"]
        units["steady_cpge"] = "microA/V^2"
    return ObservableResult(
        arrays,
        dims,
        units,
        {
            **metadata,
            "formula": "circular part of Fermi-golden-rule injection current",
            "smearing": smearing,
            "relaxation_time_seconds": relaxation_time_seconds,
            "field_convention": "dj/dt=cpge:E:conj(E); helicity=i E cross conj(E)",
        },
    )
