"""Brillouin-zone responses with explicit SI conversions and scan axes.

Hall conductivity follows sigma_ab=-e²/hbar integral(f Omega_ab). The
finite-basis orbital moment/metric omit contributions outside the retained
localized-orbital space. All energies and chemical potentials are absolute eV.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.special import entr

from ..models.evaluation import band_geometry, cell, reciprocal, spin_degeneracy, states, volume
from ..models.evaluation import chemical_potentials as _mus
from ..physics.constants import E2_H, E_CHARGE, HBAR, KB_EV
from ..physics.geometry import energy_blocks, velocity_products
from ..physics.statistics import fermi_derivative, occupations
from ._sampling import kmesh, map_points, normalized_weights
from .geometry import _curvatures, _flux_diagnostics, _plaquette, _result


def _mesh(grid, kpoints, weights):
    points = kmesh(grid) if kpoints is None else np.asarray(kpoints, float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("kpoints must be a finite (Nk,3) array")
    return points, normalized_weights(points, weights)


def _temperature(value):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError("temperature_kelvin must be finite and nonnegative")
    return value


def _measure(model, dimension, plane):
    plane = tuple(plane)
    if len(plane) != 2 or len(set(plane)) != 2 or any(axis not in (0, 1, 2) for axis in plane):
        raise ValueError("plane must specify two distinct reciprocal axes")
    if dimension == 3:
        return 1e8 / volume(model), "S/cm"
    if dimension == 2:
        g = reciprocal(model)
        return np.linalg.norm(np.cross(g[plane[0]], g[plane[1]])) / (2 * np.pi) ** 2, "S"
    raise ValueError("dimension must be 2 (sheet) or 3 (bulk)")


def _sheet_tensor(values, model, dimension, plane, *, antisymmetric=False):
    if dimension == 3:
        return values
    g = reciprocal(model)
    normal = np.cross(g[plane[0]], g[plane[1]])
    normal /= np.linalg.norm(normal)
    if antisymmetric:
        vector = _pseudovector(values)
        return _tensor(np.einsum("...a,a->...", vector, normal)[..., None] * normal)
    projection = np.eye(3) - np.outer(normal, normal)
    return np.einsum("ac,...cd,db->...ab", projection, values, projection)


def _tensor(vector):
    vector = np.asarray(vector)
    values = np.zeros((*vector.shape[:-1], 3, 3))
    values[..., 1, 2] = vector[..., 0]
    values[..., 2, 0] = vector[..., 1]
    values[..., 0, 1] = vector[..., 2]
    return values - values.swapaxes(-1, -2)


def _pseudovector(tensor):
    return np.stack((tensor[..., 1, 2], tensor[..., 2, 0], tensor[..., 0, 1]), axis=-1)


def _scan_result(name, mus, values, unit, **metadata):
    dims = ("chemical_potential", "cartesian", "cartesian2")
    return _result(
        {"chemical_potentials": mus, name: values},
        {"chemical_potentials": ("chemical_potential",), name: dims},
        {"chemical_potentials": "eV", name: unit},
        name,
        energy_reference="absolute",
        **metadata,
    )


def ahc(
    model,
    *,
    grid=(24, 24, 24),
    kpoints=None,
    weights=None,
    chemical_potentials=None,
    temperature_kelvin=0.0,
    engine="direct",
    dimension=3,
    plane=(0, 1),
    integration=None,
    execution=None,
    checkpoint=None,
):
    """Intrinsic Hall tensor, in bulk S/cm or two-dimensional sheet Siemens.

    Sheet current and electric field lie in the reciprocal plane specified by
    ``plane``. Both Cartesian axes are projected onto that plane.
    """
    temperature = _temperature(temperature_kelvin)
    mus = _mus(model, chemical_potentials)
    measure, unit = _measure(model, dimension, plane)
    degeneracy = spin_degeneracy(model)
    diagnostics = {}
    from .integration import integrate_unit_box, nested_observable, validate_integration

    controls = validate_integration(integration)
    if controls.get("method") == "adaptive":
        if kpoints is not None or weights is not None:
            raise ValueError("adaptive Hall integration requires a grid without explicit kpoints/weights")
        return nested_observable(
            lambda refined: ahc(
                model,
                grid=refined,
                chemical_potentials=chemical_potentials,
                temperature_kelvin=temperature_kelvin,
                engine=engine,
                dimension=dimension,
                plane=plane,
                execution=execution,
                checkpoint=checkpoint,
            ),
            grid,
            controls,
            "ahc",
            active_axes=list(plane) if dimension == 2 else None,
            checkpoint=checkpoint,
        )
    local = controls.get("method") in {"threshold", "cubature"}
    if local and (engine == "fhs" or kpoints is not None or weights is not None):
        raise ValueError("local Hall integration requires direct/kubo and a grid without explicit weights")
    if engine == "fhs":
        if temperature != 0 or kpoints is not None or weights is not None:
            raise ValueError("FHS Hall requires a uniform grid at zero temperature")
        mesh = np.asarray(grid)
        points = kmesh(mesh)
        if dimension == 2 and mesh[({0, 1, 2} - set(plane)).pop()] != 1:
            raise ValueError("sheet Hall requires one point on the perpendicular mesh axis")
        reciprocal_vectors = reciprocal(model)
        cherns = np.zeros((len(mus), 3))
        maximum_flux = np.zeros((3, len(mus)))
        occupation_variation = np.zeros((3, len(mus)), dtype=int)
        planes = [(1, 2), (2, 0), (0, 1)]
        indices = range(3) if dimension == 3 else [({0, 1, 2} - set(plane)).pop()]
        for index in indices:
            u, v = planes[index] if dimension == 3 else tuple(plane)
            du, dv = np.eye(3)[u] / mesh[u], np.eye(3)[v] / mesh[v]

            def at(k):
                corners = (k, k + du, k + du + dv, k + dv)
                solution = states(model, corners)
                total = np.zeros(len(mus))
                largest = np.zeros(len(mus))
                rank_variation = np.zeros(len(mus), dtype=int)
                # Collinear sectors have independent ordered energy manifolds.
                # Sorting them together creates artificial rank discontinuities
                # when opposite-spin bands cross in a metal.
                for channel in np.unique(solution.band_channels):
                    indices = np.flatnonzero(solution.band_channels == channel)
                    subspace = SimpleNamespace(
                        energies=solution.energies[indices],
                        coefficients=solution.coefficients[:, :, indices],
                        overlaps=solution.overlaps,
                        band_channels=solution.band_channels[indices],
                    )
                    counts = np.array(
                        [(energy[:, None] < mus[None]).sum(axis=0) for energy in subspace.energies.T]
                    )
                    phases = {
                        int(count): _plaquette(model, corners, int(count), solutions=subspace)
                        for count in np.unique(counts)
                    }
                    vertex_phases = np.array([[phases[int(count)] for count in row] for row in counts])
                    total += vertex_phases.mean(axis=0)
                    largest = np.maximum(largest, np.max(np.abs(vertex_phases), axis=0))
                    rank_variation = np.maximum(rank_variation, counts.max(axis=0) - counts.min(axis=0))
                return np.array([total, largest, rank_variation])

            sampled = np.asarray(map_points(model, points, at, execution))
            cherns[:, index] = -sampled[:, 0].sum(axis=0) / (2 * np.pi * mesh[index])
            maximum_flux[index] = sampled[:, 1].max(axis=0)
            occupation_variation[index] = sampled[:, 2].max(axis=0).astype(int)
        diagnostics["fhs_diagnostics"] = {
            **_flux_diagnostics(maximum_flux),
            "grid": mesh.astype(int).tolist(),
            "chern_vector": cherns.tolist(),
            "evaluated_reciprocal_planes": [list(planes[i] if dimension == 3 else plane) for i in indices],
            "max_abs_plaquette_flux_by_plane_and_mu": maximum_flux.tolist(),
            "minimum_branch_margin_by_plane_and_mu": (np.pi - maximum_flux).tolist(),
            "principal_branch_admissible_by_plane_and_mu": (
                np.pi - maximum_flux > 64 * np.finfo(float).eps * np.pi
            ).tolist(),
            "maximum_channel_occupation_rank_range_by_plane_and_mu": occupation_variation.tolist(),
            "metallic_vertex_rank_variation_detected": np.any(occupation_variation > 0, axis=0).tolist(),
            "flux_scope": "principal channel plaquettes before metallic vertex occupation averaging",
            "occupation_scope": "sampled vertex ranks; uniform ranks do not prove an insulating gap",
            "plane_index_order": [
                list(planes[i] if dimension == 3 or i not in indices else plane) for i in range(3)
            ],
            "sign_convention": "C=-sum principal occupied-overlap phases/(2pi); sigma=-e^2/h times C",
        }
        if dimension == 3:
            vector = -degeneracy * E2_H * (cherns @ reciprocal_vectors) / (2 * np.pi) * 1e8
        else:
            normal = np.cross(reciprocal_vectors[plane[0]], reciprocal_vectors[plane[1]])
            normal /= np.linalg.norm(normal)
            vector = -degeneracy * E2_H * cherns[:, index, None] * normal
        values = _tensor(vector)
    elif engine in {"direct", "kubo"}:
        points, weight = _mesh(grid, kpoints, weights)

        def at(k):
            data = band_geometry(model, k)
            curvature = _curvatures(model, k, data, engine)
            f = occupations(data["energies"][:, None], mus[None], temperature)
            return np.einsum("nm,nab->mab", f, curvature)

        factor = -degeneracy * E_CHARGE**2 / HBAR * measure
        if local:
            local_controls = dict(controls)
            if controls.get("threshold_quantity") == "curvature":
                local_controls["threshold"] = controls.get("threshold", 50.0) * abs(factor) * np.sqrt(2)
            integral = integrate_unit_box(
                lambda points: factor * np.asarray(map_points(model, points, at, execution)),
                grid,
                local_controls,
                namespace="ahc",
                active_axes=list(plane) if dimension == 2 else None,
                threshold_selector=lambda value: np.max(np.linalg.norm(value, axis=(-2, -1))),
                checkpoint=checkpoint,
            )
            values = integral.value
            diagnostics["integration"] = integral.metadata
        else:
            values = factor * np.einsum(
                "k,kmab->mab", weight, np.asarray(map_points(model, points, at, execution))
            )
    else:
        raise ValueError("Hall engine must be 'direct', 'kubo', or 'fhs'")
    values = _sheet_tensor(values, model, dimension, plane, antisymmetric=True)
    return _scan_result(
        "ahc",
        mus,
        values,
        unit,
        engine=engine,
        dimension=dimension,
        temperature_kelvin=temperature,
        sign="sigma_ab=-e^2/hbar integral(f Omega_ab)",
        **diagnostics,
    )


def anc(
    model,
    *,
    grid=(24, 24, 24),
    kpoints=None,
    weights=None,
    chemical_potentials=None,
    temperature_kelvin=300.0,
    engine="direct",
    dimension=3,
    plane=(0, 1),
    execution=None,
):
    """Intrinsic anomalous thermoelectric conductivity from Fermi entropy.

    The low-temperature limit is alpha=-(pi² kB² T/3e) d sigma/d mu.
    """
    temperature = _temperature(temperature_kelvin)
    mus = _mus(model, chemical_potentials)
    points, weight = _mesh(grid, kpoints, weights)
    measure, _ = _measure(model, dimension, plane)

    def at(k):
        data = band_geometry(model, k)
        curvature = _curvatures(model, k, data, engine)
        f = occupations(data["energies"][:, None], mus[None], temperature)
        entropy = entr(f) + entr(1 - f)
        return np.einsum("nm,nab->mab", entropy, curvature)

    values = (
        spin_degeneracy(model)
        * E_CHARGE**2
        / HBAR
        * KB_EV
        * measure
        * np.einsum("k,kmab->mab", weight, np.asarray(map_points(model, points, at, execution)))
    )
    values = _sheet_tensor(values, model, dimension, plane, antisymmetric=True)
    return _scan_result(
        "anc",
        mus,
        values,
        "A/(K cm)" if dimension == 3 else "A/K",
        temperature_kelvin=temperature,
        engine=engine,
        dimension=dimension,
    )


def berry_curvature_dipole(
    model,
    *,
    grid=(24, 24, 24),
    kpoints=None,
    weights=None,
    chemical_potentials=None,
    temperature_kelvin=0.0,
    broadening=0.05,
    engine="direct",
    execution=None,
):
    """Fermi-surface D_ab=integral(-df/dE partial_a E Omega_b), dimensionless."""
    temperature = _temperature(temperature_kelvin)
    mus = _mus(model, chemical_potentials)
    points, weight = _mesh(grid, kpoints, weights)

    def at(k):
        data = band_geometry(model, k)
        curvature = _pseudovector(_curvatures(model, k, data, engine))
        slope = np.diagonal(data["velocities"], axis1=1, axis2=2).real.T
        derivative = fermi_derivative(data["energies"][:, None], mus[None], temperature, broadening)
        for block in energy_blocks(data):
            if len(block) < 2 or np.max(derivative[block], initial=0) <= 1e-14 * np.max(
                derivative, initial=0
            ):
                continue
            velocity = data["velocities"][:, block][:, :, block]
            scalar = np.trace(velocity, axis1=1, axis2=2)[:, None, None] * np.eye(len(block)) / len(block)
            tolerance = max(
                data["degeneracy_tol"] * np.max(np.linalg.norm(cell(model), axis=1)),
                1e-10 * (1 + np.max(np.abs(velocity), initial=0)),
            )
            if np.max(np.abs(velocity - scalar), initial=0) > tolerance:
                raise ValueError(
                    "Berry-curvature dipole is undefined at an unresolved dispersive degeneracy; shift the mesh"
                )
        return np.einsum("nm,na,nb->mab", derivative, slope, curvature)

    values = (
        spin_degeneracy(model)
        / volume(model)
        * np.einsum("k,kmab->mab", weight, np.asarray(map_points(model, points, at, execution)))
    )
    return _scan_result(
        "berry_curvature_dipole",
        mus,
        values,
        "1",
        engine=engine,
        temperature_kelvin=temperature,
        broadening_eV=float(broadening),
        tensor_axes="energy-gradient direction, Berry-curvature direction",
    )


def drude_weight(
    model,
    *,
    grid=(24, 24, 24),
    kpoints=None,
    weights=None,
    chemical_potentials=None,
    temperature_kelvin=0.0,
    broadening=0.05,
    dimension=3,
    plane=(0, 1),
    execution=None,
):
    """Positive intraband pole weight: sigma(E)=i D/(E+i eta).

    D is in eV S/cm (bulk) or eV S (sheet); this convention includes no extra pi.
    """
    temperature = _temperature(temperature_kelvin)
    mus = _mus(model, chemical_potentials)
    points, weight = _mesh(grid, kpoints, weights)
    measure, unit = _measure(model, dimension, plane)

    def at(k):
        data = band_geometry(model, k)
        df = fermi_derivative(data["energies"][:, None], mus[None], temperature, broadening)
        return np.einsum("nm,nab->mab", df, velocity_products(data))

    values = (
        spin_degeneracy(model)
        * E_CHARGE**2
        / HBAR
        * measure
        * np.einsum("k,kmab->mab", weight, np.asarray(map_points(model, points, at, execution)))
    )
    values = _sheet_tensor(values, model, dimension, plane)
    return _scan_result(
        "drude_weight",
        mus,
        values,
        "eV " + unit,
        definition="sigma(E)=i D/(E+i eta); no pi",
        temperature_kelvin=temperature,
        broadening_eV=float(broadening),
        dimension=dimension,
    )


def orbital_magnetization(
    model,
    *,
    grid=(24, 24, 24),
    kpoints=None,
    weights=None,
    chemical_potentials=None,
    temperature_kelvin=0.0,
    execution=None,
):
    """Finite-basis modern orbital magnetization in A/m, including Fermi-sea term."""
    temperature = _temperature(temperature_kelvin)
    mus = _mus(model, chemical_potentials)
    points, weight = _mesh(grid, kpoints, weights)

    def at(k):
        data = band_geometry(model, k)
        e, a = data["energies"], data["connections"]
        delta = e[None, :] - e[:, None]
        moments = np.stack(
            [np.einsum("nm,mn,nm->n", a[u], a[v], delta).imag for u, v in ((1, 2), (2, 0), (0, 1))], axis=1
        )
        curvature = _pseudovector(_curvatures(model, k, data, "kubo"))
        f = occupations(e[:, None], mus[None], temperature)
        if temperature == 0:
            free_energy = np.maximum(mus[None] - e[:, None], 0)
        else:
            kt = KB_EV * temperature
            free_energy = kt * np.logaddexp(0, (mus[None] - e[:, None]) / kt)
        return np.einsum("nm,na->ma", f, moments), np.einsum("nm,na->ma", free_energy, curvature)

    values = np.asarray(map_points(model, points, at, execution))
    factor = spin_degeneracy(model) * E_CHARGE**2 / HBAR * 1e10 / volume(model)
    local = factor * np.einsum("k,kma->ma", weight, values[:, 0])
    itinerant = factor * np.einsum("k,kma->ma", weight, values[:, 1])
    arrays = {
        "chemical_potentials": mus,
        "orbital_magnetization": local + itinerant,
        "local": local,
        "itinerant": itinerant,
    }
    return _result(
        arrays,
        {
            key: (
                ("chemical_potential",)
                if key == "chemical_potentials"
                else ("chemical_potential", "cartesian")
            )
            for key in arrays
        },
        {key: ("eV" if key == "chemical_potentials" else "A/m") for key in arrays},
        "orbital_magnetization",
        definition="finite_basis_interband",
        temperature_kelvin=temperature,
        decomposition="wave-packet orbital moment plus Berry-curvature Fermi-sea term",
    )


def shc(
    model,
    *,
    grid=(24, 24, 24),
    kpoints=None,
    weights=None,
    chemical_potentials=None,
    temperature_kelvin=0.0,
    broadening=0.0,
    execution=None,
):
    """Conventional intrinsic spin Hall tensor, units (hbar/2e) S/cm.

    Axes are chemical potential, spin polarization, current direction, field
    direction. The conventional spin current is one half the anticommutator
    of physical velocity and physical spin; spin need not be conserved.
    """
    if not hasattr(model, "spin_operators") or getattr(model, "spin_mode", "nonmagnetic") == "nonmagnetic":
        raise ValueError("spin Hall conductivity requires explicit spin operators")
    temperature = _temperature(temperature_kelvin)
    if not np.isfinite(broadening) or broadening < 0:
        raise ValueError("broadening must be nonnegative eV")
    mus = _mus(model, chemical_potentials)
    points, weight = _mesh(grid, kpoints, weights)

    def at(k):
        data = band_geometry(model, k)
        e, c, v = data["energies"], data["coefficients"], data["velocities"]
        spin = np.einsum("in,aij,jm->anm", c.conj(), model.spin_operators(k), c, optimize=True)
        delta = e[None] - e[:, None]
        inv = np.zeros_like(delta)
        np.divide(1.0, delta**2 + broadening**2, out=inv, where=np.abs(delta) > data["degeneracy_tol"])
        kernel = np.empty((len(e), 3, 3, 3))
        for s in range(3):
            for a in range(3):
                current = (spin[s] @ v[a] + v[a] @ spin[s]) / 2
                for b in range(3):
                    kernel[:, s, a, b] = -2 * np.einsum("nm,mn,nm->n", current, v[b], inv).imag
        f = occupations(e[:, None], mus[None], temperature)
        return np.einsum("nm,nsab->msab", f, kernel)

    values = (
        -(E_CHARGE**2)
        / HBAR
        * 1e8
        / volume(model)
        * np.einsum("k,kmsab->msab", weight, np.asarray(map_points(model, points, at, execution)))
    )
    return _result(
        {"chemical_potentials": mus, "shc": values},
        {
            "chemical_potentials": ("chemical_potential",),
            "shc": ("chemical_potential", "spin", "cartesian", "cartesian2"),
        },
        {"chemical_potentials": "eV", "shc": "(hbar/2e) S/cm"},
        "shc",
        temperature_kelvin=temperature,
        broadening_eV=float(broadening),
        definition="conventional spin current {v,S}/2; Pauli S=hbar sigma/2",
    )
