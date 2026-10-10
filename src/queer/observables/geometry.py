"""Berry geometry of nonorthogonal localized-orbital models.

The orientation is A=i<u|du>, Omega_ab=partial_a A_b-partial_b A_a.
The metric is the interband metric within the retained finite basis, and does
not claim to include orbital spread into discarded continuum states.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ..execution import ExecutionOptions
from ._tb import (
    E_CHARGE,
    band_geometry,
    basis_curvature,
    basis_overlap,
    cell,
    degeneracy_tolerance,
    kmesh,
    map_points,
    reciprocal,
    spin_degeneracy,
    states,
    volume,
)
from .result import ObservableResult


def _result(arrays, dims, units, observable, **metadata):
    return ObservableResult(
        arrays,
        dims,
        units,
        {"observable": observable, "berry_convention": "A=i<u|du>; Omega_ab=-2 Im Q_ab", **metadata},
    )


def _curvatures(model, k, data, engine="direct"):
    """Per-band curvature (band,Cartesian,Cartesian), with block gauge."""
    if engine not in {"direct", "kubo"}:
        raise ValueError("curvature engine must be 'direct' or 'kubo'")
    a = data["connections"]
    n = len(data["energies"])
    result = np.zeros((n, 3, 3))
    if engine == "kubo":
        for i in range(3):
            for j in range(i + 1, 3):
                q = np.einsum("nm,mn->n", a[i], a[j])
                result[:, i, j] = -2 * q.imag
                result[:, j, i] = -result[:, i, j]
        return result
    f = basis_curvature(model, k, data)
    for i in range(3):
        for j in range(i + 1, 3):
            curl = f[i, j] + 1j * (a[i] @ a[j] - a[j] @ a[i])
            result[:, i, j] = curl.diagonal().real
            result[:, j, i] = -result[:, i, j]
    return result


def _occupied(data, occupied_bands=None, fermi_energy=0.0):
    energies = data["energies"]
    if occupied_bands is None:
        mask = energies < fermi_energy
    else:
        count = int(occupied_bands)
        if count != occupied_bands or count < 0 or count > len(energies):
            raise ValueError("occupied_bands must be an integer within the band count")
        mask = np.zeros(len(energies), dtype=bool)
        mask[np.argsort(energies, kind="stable")[:count]] = True
    equal = np.abs(energies[:, None] - energies[None, :]) < data["degeneracy_tol"]
    same = data["band_channels"][:, None] == data["band_channels"][None, :]
    if np.any(equal & same & (mask[:, None] != mask[None, :])):
        raise ValueError("occupied/unoccupied boundary splits a degenerate subspace")
    return mask


def _fermi_energy(model, value):
    result = float(model.fermi_energy if value is None else value)
    if not np.isfinite(result):
        raise ValueError("fermi_energy must be a finite absolute eV value")
    return result


def berry_curvature(
    model,
    kpoints,
    *,
    bands=None,
    occupied_bands=None,
    fermi_energy=None,
    engine="direct",
    degeneracy_tol=None,
    execution=None,
):
    """Cartesian curvature in Angstrom², with band-first array axes.

    ``occupied_bands`` returns the trace over a complete occupied manifold.
    Individual-band curvature is rejected at unresolved same-channel degeneracies.
    """
    points = np.asarray(getattr(kpoints, "path", kpoints), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or not len(points):
        raise ValueError("kpoints must be a nonempty finite (Nk,3) array")
    raw_selection = np.arange(model.nbnd) if bands is None else np.atleast_1d(bands)
    if raw_selection.ndim != 1 or raw_selection.dtype.kind not in "iu" or not raw_selection.size:
        raise ValueError("bands must contain integer indices")
    selection = raw_selection.astype(int)
    if np.any(selection < 0) or np.any(selection >= model.nbnd):
        raise ValueError("band indices are outside the model")
    mu = _fermi_energy(model, fermi_energy)

    def at(k):
        data = band_geometry(model, k, degeneracy_tol)
        curvature = _curvatures(model, k, data, engine)
        if occupied_bands is not None:
            return curvature[_occupied(data, occupied_bands, mu)].sum(axis=0)
        e = data["energies"]
        equal = np.abs(e[:, None] - e[None, :]) < data["degeneracy_tol"]
        same = data["band_channels"][:, None] == data["band_channels"][None, :]
        np.fill_diagonal(equal, False)
        if np.any((equal & same)[selection]):
            raise ValueError(
                "individual-band Berry curvature is undefined at a degeneracy; select a manifold"
            )
        return curvature[selection]

    values = np.asarray(map_points(model, points, at, execution))
    dims: tuple[str, ...]
    if occupied_bands is None:
        values = values.transpose(1, 0, 2, 3)
        dims = ("band", "kpoint", "cartesian", "cartesian2")
    else:
        dims = ("kpoint", "cartesian", "cartesian2")
    return _result(
        {"kpoints": points, "curvature": values},
        {"kpoints": ("kpoint", "fractional"), "curvature": dims},
        {"kpoints": "fractional", "curvature": "Angstrom^2"},
        "berry_curvature",
        engine=engine,
    )


def quantum_metric(
    model,
    kpoints,
    *,
    occupied_bands=None,
    fermi_energy=None,
    definition="finite_basis_interband",
    degeneracy_tol=None,
    execution=None,
):
    """Occupied-manifold interband metric, a positive Cartesian tensor in Å²."""
    if definition != "finite_basis_interband":
        raise ValueError(
            "only definition='finite_basis_interband' is supported; full metric needs second moments"
        )
    points = np.asarray(getattr(kpoints, "path", kpoints), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or not len(points):
        raise ValueError("kpoints must be a nonempty finite (Nk,3) array")
    mu = _fermi_energy(model, fermi_energy)

    def at(k):
        data = band_geometry(model, k, degeneracy_tol)
        occupied = _occupied(data, occupied_bands, mu)
        a = data["connections"][:, occupied][:, :, ~occupied]
        return np.einsum("anm,bnm->ab", a, a.conj()).real

    values = np.asarray(map_points(model, points, at, execution))
    return _result(
        {"kpoints": points, "metric": values, "trace": np.trace(values, axis1=1, axis2=2)},
        {
            "kpoints": ("kpoint", "fractional"),
            "metric": ("kpoint", "cartesian", "cartesian2"),
            "trace": ("kpoint",),
        },
        {"kpoints": "fractional", "metric": "Angstrom^2", "trace": "Angstrom^2"},
        "quantum_metric",
        definition=definition,
        completeness="retained-basis interband; excludes external orbital spread",
    )


def _occupied_link(model, k, kp, count, *, solutions=None):
    if count == 0:
        return np.empty((0, 0), complex)
    first = states(model, np.asarray(k)[None]) if solutions is None else solutions[0]
    second = states(model, np.asarray(kp)[None]) if solutions is None else solutions[1]
    first_indices = np.flatnonzero(_solution_occupied(first, 0, count, 0.0, degeneracy_tolerance(model)))
    second_indices = np.flatnonzero(_solution_occupied(second, 0, count, 0.0, degeneracy_tolerance(model)))
    c1 = first.coefficients[0][:, first_indices]
    c2 = second.coefficients[0][:, second_indices]
    return c1.conj().T @ basis_overlap(model, k, kp) @ c2


def _link_phase(matrix):
    if not matrix.size:
        return 0.0
    sign, _ = np.linalg.slogdet(matrix)
    if abs(sign) == 0:
        raise ValueError("singular occupied overlap; refine the k mesh")
    return float(np.angle(sign))


def _point_solution(solution, index):
    return SimpleNamespace(
        energies=solution.energies[:, index : index + 1],
        coefficients=solution.coefficients[index : index + 1],
        overlaps=solution.overlaps[index : index + 1],
        band_channels=solution.band_channels,
    )


def _solution_occupied(solution, index, occupied_bands, mu, tolerance=1e-9):
    return _occupied(
        {
            "energies": solution.energies[:, index],
            "band_channels": solution.band_channels,
            "degeneracy_tol": tolerance,
        },
        occupied_bands,
        mu,
    )


def _plaquette(model, corners, count, *, solutions=None):
    solution = states(model, corners) if solutions is None else solutions
    vertices = [_point_solution(solution, i) for i in range(4)]
    phase = sum(
        _link_phase(
            _occupied_link(
                model, corners[i], corners[(i + 1) % 4], count, solutions=(vertices[i], vertices[(i + 1) % 4])
            )
        )
        for i in range(4)
    )
    return float(np.angle(np.exp(1j * phase)))


def _flux_diagnostics(flux):
    maximum = float(np.max(np.abs(flux), initial=0))
    margin = float(np.pi - maximum)
    guard = 64 * np.finfo(float).eps * np.pi
    return {
        "max_abs_plaquette_flux": maximum,
        "minimum_branch_margin": margin,
        "principal_branch_admissible": bool(margin > guard),
        "admissibility_criterion": "abs(principal plaquette flux)<pi",
        "branch_guard": float(guard),
        "near_branch_cut": bool(margin <= 0.1 * np.pi),
        "near_branch_cut_threshold": float(0.1 * np.pi),
        "continuum_convergence_established": False,
        "diagnostic_scope": "principal-branch separation only; wrapped coarse flux can alias continuum topology",
    }


def chern_number(
    model,
    *,
    grid=(24, 24, 1),
    plane=(0, 1),
    k_start=None,
    k_vectors=None,
    occupied_bands=None,
    fermi_energy=None,
    engine="fhs",
    refinement=False,
    integration=None,
    execution=None,
):
    """Chern number on a fractional reciprocal plane, without spin multiplicity.

    FHS returns signed principal plaquette flux and branch diagnostics.
    ``refinement=True`` additionally compares a plane mesh with twice as many
    points along both directions. Branch admissibility and one refinement do
    not establish continuum convergence. Direct/Kubo report both definitions.
    """
    if not isinstance(refinement, (bool, np.bool_)):
        raise ValueError("refinement must be boolean")
    plane = tuple(plane)
    if len(plane) != 2 or len(set(plane)) != 2 or any(a not in (0, 1, 2) for a in plane):
        raise ValueError("plane must specify two distinct reciprocal axes")
    mesh = np.asarray(grid)
    other = ({0, 1, 2} - set(plane)).pop()
    if mesh.shape != (3,):
        raise ValueError("a Chern grid must have three positive integer dimensions")
    if mesh[other] != 1:
        raise ValueError("a Chern plane needs one point along the perpendicular reciprocal axis")
    spans = np.eye(3)[list(plane)] if k_vectors is None else np.asarray(k_vectors, float)
    if spans.shape != (2, 3) or not np.isfinite(spans).all() or np.linalg.matrix_rank(spans) != 2:
        raise ValueError("k_vectors must contain two independent finite reciprocal spans")
    vectors = np.zeros((3, 3))
    vectors[list(plane)] = spans
    from .integration import checkpoint_samples, integrate_unit_box, nested_observable, validate_integration

    controls = validate_integration(integration)
    if controls.get("method") == "adaptive":
        return nested_observable(
            lambda refined: chern_number(
                model,
                grid=refined,
                plane=plane,
                k_start=k_start,
                k_vectors=spans,
                occupied_bands=occupied_bands,
                fermi_energy=fermi_energy,
                engine=engine,
                execution=execution,
            ),
            mesh,
            controls,
            "chern_number",
            active_axes=list(plane),
        )
    kpoints = kmesh(mesh, k_start=k_start, k_vectors=vectors)
    mu = _fermi_energy(model, fermi_energy)
    local = controls.get("method") in {"threshold", "cubature"}
    if local and engine == "fhs":
        raise ValueError("local Chern integration requires direct or kubo; FHS needs a uniform linked mesh")
    if local:
        if controls.get("max_points", 1_000_000) <= 1:
            raise RuntimeError("Local Chern integration exhausted max_points before quadrature")
        reference = kpoints[:1]
        energy = checkpoint_samples(
            lambda points: states(model, points, execution=execution).energies.T,
            reference,
            namespace="chern_number",
            tag="occupied-rank-reference",
            value_units="eV",
        ).T
    elif hasattr(model, "energies"):
        energy = model.energies(kpoints, execution=execution) + model.fermi_energy
    else:
        energy = np.asarray(
            map_points(model, kpoints, lambda k: states(model, np.asarray(k)[None]).energies[:, 0], execution)
        ).T
    solution = SimpleNamespace(
        energies=energy, band_channels=getattr(model, "band_channels", np.zeros(model.nbnd, int))
    )
    counts = [
        int(_solution_occupied(solution, i, occupied_bands, mu, degeneracy_tolerance(model)).sum())
        for i in range(energy.shape[1])
    ]
    if len(set(counts)) != 1:
        raise ValueError("Chern number requires a constant occupied rank (a gapped plane)")
    count = counts[0]
    diagnostics = {
        "grid": mesh.astype(int).tolist(),
        "k_start": (np.zeros(3) if k_start is None else np.asarray(k_start)).tolist(),
        "continuum_convergence_established": False,
        "flux_convention": "positive flux is Omega_ab integrated along the ordered reciprocal plane",
        "k_vectors": spans.tolist(),
        "closed_reciprocal_surface": bool(np.allclose(spans, np.round(spans), atol=1e-12, rtol=0)),
    }
    arrays = {"kpoints": kpoints}
    dims: dict[str, tuple[str, ...]] = {"kpoints": ("kpoint", "fractional")}
    units = {"kpoints": "fractional"}
    if engine == "fhs":
        u, v = plane
        du = spans[0] / mesh[u]
        dv = spans[1] / mesh[v]
        phases = map_points(
            model, kpoints, lambda k: _plaquette(model, (k, k + du, k + du + dv, k + dv), count), execution
        )
        flux = -np.asarray(phases)
        number = flux.sum() / (2 * np.pi)
        arrays["plaquette_flux"] = flux
        dims["plaquette_flux"] = ("kpoint",)
        units["plaquette_flux"] = "1"
        diagnostics.update(_flux_diagnostics(flux))
    elif engine in {"direct", "kubo"}:

        def at(k):
            data = band_geometry(model, k)
            occupied = _occupied(data, occupied_bands, mu)
            if int(occupied.sum()) != count:
                raise ValueError("Chern number requires a constant occupied rank (a gapped plane)")
            return np.array(
                [_curvatures(model, k, data, method)[occupied].sum(axis=0) for method in ("direct", "kubo")]
            )

        g = reciprocal(model)
        cartesian_spans = spans @ g
        selected = 0 if engine == "direct" else 1
        if local:
            origin = np.zeros(3) if k_start is None else np.asarray(k_start, float)

            def integrand(points):
                values = np.asarray(map_points(model, origin + points @ spans, at, execution))
                return np.einsum("kmab,a,b->km", values, *cartesian_spans) / (2 * np.pi)

            local_controls = dict(controls)
            local_controls["max_points"] = controls.get("max_points", 1_000_000) - 1
            if controls.get("threshold_quantity") == "curvature":
                local_controls["threshold"] = (
                    controls.get("threshold", 50.0) * np.linalg.norm(np.cross(*cartesian_spans)) / (2 * np.pi)
                )
            for key in ("grid", "subdivision"):
                if key in local_controls and len(local_controls[key]) == 3:
                    local_controls[key] = np.asarray(local_controls[key])[list(plane)].tolist()
            integral = integrate_unit_box(
                integrand,
                mesh[list(plane)],
                local_controls,
                namespace="chern_number",
                active_axes=[0, 1],
                threshold_selector=lambda value: abs(value[selected]),
            )
            numbers = integral.value
            diagnostics["integration"] = {
                **integral.metadata,
                "points": integral.metadata["points"] + 1,
                "quadrature_points": integral.metadata["points"],
                "rank_check_points": 1,
            }
            arrays["integration_error"] = np.asarray(integral.error[selected])
            dims["integration_error"], units["integration_error"] = (), "1"
        else:
            values = np.asarray(map_points(model, kpoints, at, execution))
            fluxes = np.einsum("kmab,a,b->km", values, *cartesian_spans) / (mesh[plane[0]] * mesh[plane[1]])
            numbers = fluxes.sum(axis=0) / (2 * np.pi)
            arrays["quadrature_flux"] = fluxes[:, selected]
            dims["quadrature_flux"] = ("kpoint",)
            units["quadrature_flux"] = "1"
        number = numbers[selected]
        diagnostics["engine_comparison"] = {
            "direct_chern_number": float(numbers[0]),
            "kubo_chern_number": float(numbers[1]),
            "direct_minus_kubo": float(numbers[0] - numbers[1]),
            "diagnostic_scope": "basis-curvature contribution; agreement does not establish mesh convergence",
        }
    else:
        raise ValueError("chern engine must be 'fhs', 'direct', or 'kubo'")
    if refinement:
        finer = mesh.astype(int).copy()
        finer[list(plane)] *= 2
        refined = chern_number(
            model,
            grid=finer,
            plane=plane,
            k_start=k_start,
            k_vectors=spans,
            occupied_bands=occupied_bands,
            fermi_energy=fermi_energy,
            engine=engine,
            integration=integration,
            execution=execution,
        )
        refined_number = float(refined["chern_number"])
        diagnostics["refinement"] = {
            "grid": finer.tolist(),
            "chern_number": refined_number,
            "coarse_chern_number": float(number),
            "absolute_difference": float(abs(refined_number - number)),
            "continuum_convergence_established": False,
        }
    arrays["chern_number"] = np.asarray(number)
    dims["chern_number"] = ()
    units["chern_number"] = "1"
    return _result(
        arrays,
        dims,
        units,
        "chern_number",
        engine=engine,
        plane=list(plane),
        occupied_bands=count,
        **diagnostics,
    )


def chirality(
    model,
    kpoint,
    *,
    radius=0.05,
    occupied_bands=None,
    fermi_energy=None,
    polar_points=16,
    azimuth_points=32,
    engine="direct",
    execution=None,
):
    """Berry flux/(2pi) through an outward-oriented Cartesian sphere."""
    center = np.asarray(kpoint, float)
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(radius) or radius <= 0:
        raise ValueError("kpoint must be finite and radius positive in inverse Angstrom")
    if (
        int(polar_points) != polar_points
        or int(azimuth_points) != azimuth_points
        or min(polar_points, azimuth_points) < 4
    ):
        raise ValueError("sphere quadrature requires at least four points per angle")
    z, weights = np.polynomial.legendre.leggauss(int(polar_points))
    phi = 2 * np.pi * np.arange(int(azimuth_points)) / azimuth_points
    zz, pp = np.meshgrid(z, phi, indexing="ij")
    directions = np.stack(
        (np.sqrt(1 - zz**2) * np.cos(pp), np.sqrt(1 - zz**2) * np.sin(pp), zz), axis=-1
    ).reshape(-1, 3)
    points = center + radius * directions @ np.linalg.inv(reciprocal(model))
    mu = _fermi_energy(model, fermi_energy)

    def at(k):
        data = band_geometry(model, k)
        mask = _occupied(data, occupied_bands, mu)
        f = _curvatures(model, k, data, engine)[mask].sum(axis=0)
        return np.asarray([f[1, 2], f[2, 0], f[0, 1], mask.sum()])

    data = np.asarray(map_points(model, points, at, execution))
    if len(np.unique(data[:, 3])) != 1:
        raise ValueError("chirality requires a constant occupied rank on the enclosing sphere")
    curvature = data[:, :3]
    solid = np.repeat(weights, int(azimuth_points)) * 2 * np.pi / azimuth_points
    charge = radius**2 * np.sum(solid * np.einsum("ka,ka->k", curvature, directions)) / (2 * np.pi)
    return _result(
        {"chirality": np.asarray(charge)},
        {"chirality": ()},
        {"chirality": "1"},
        "chirality",
        radius_inverse_angstrom=float(radius),
        engine=engine,
    )


def wilson_loop(model, kpoints, *, occupied_bands=None, fermi_energy=None, execution=None):
    """Hybrid centers modulo one from unitarized occupied overlap products."""
    paths = np.asarray(kpoints, dtype=float)
    if paths.ndim == 2:
        paths = paths[None]
    if paths.ndim != 3 or paths.shape[2] != 3 or paths.shape[1] < 2 or not np.isfinite(paths).all():
        raise ValueError("Wilson paths must have shape (Nk,3) or (Npath,Nk,3)")
    mu = _fermi_energy(model, fermi_energy)
    options = (execution or getattr(model, "execution", None) or ExecutionOptions()).resolved()
    size = options.bounded_batch_size(1024 * model.nbnd**2)

    def at(path):
        first = previous = previous_k = product = None
        count = None

        def accumulate(k, kp, left, right):
            link = _occupied_link(model, k, kp, count, solutions=(left, right))
            if not count:
                return np.eye(0, dtype=complex)
            u, singular, vh = np.linalg.svd(link)
            if np.min(singular) < 1e-12:
                raise ValueError("singular Wilson overlap; refine the path")
            return u @ vh

        for start in range(0, len(path), size):
            solution = states(model, path[start : start + size])
            for index in range(solution.energies.shape[1]):
                current = _point_solution(solution, index)
                current_count = int(
                    _solution_occupied(current, 0, occupied_bands, mu, degeneracy_tolerance(model)).sum()
                )
                k = path[start + index]
                if first is None:
                    first = current
                    count = current_count
                    product = np.eye(count, dtype=complex)
                else:
                    if current_count != count:
                        raise ValueError("Wilson loop requires constant occupied rank along the path")
                    product = product @ accumulate(previous_k, k, previous, current)
                previous, previous_k = current, k
        winding = np.rint(path[-1] - path[0])
        closing = path[0] + winding
        # Sew to the original first eigenvectors at the nearby lifted point.
        product = product @ accumulate(previous_k, closing, previous, first)
        return (
            np.sort((-np.angle(np.linalg.eigvals(product)) / (2 * np.pi)) % 1),
            -_link_phase(product),
            count,
        )

    results = map_points(model, paths, at, execution)
    ranks = {item[2] for item in results}
    if len(ranks) != 1:
        raise ValueError("Wilson loops must have a common occupied rank")
    rank = ranks.pop()
    return _result(
        {
            "phases": np.asarray([item[0] for item in results]),
            "berry_phase": np.asarray([item[1] for item in results]),
        },
        {"phases": ("loop", "occupied_band"), "berry_phase": ("loop",)},
        {"phases": "1", "berry_phase": "radian"},
        "wilson_loop",
        occupied_bands=rank,
    )


def polarization(
    model, *, grid=(8, 8, 8), occupied_bands=None, fermi_energy=None, ionic_charges=None, execution=None
):
    """Electronic Berry-phase polarization and optional ionic/total terms.

    Values are defined modulo ``polarization_quantum``. Ionic charges are
    pseudopotential valences in units of e, not atomic numbers.
    """
    dimensions = np.asarray(grid)
    kmesh(dimensions)  # validate before constructing strings
    reduced = np.zeros(3)
    for direction in range(3):
        if dimensions[direction] < 2:
            raise ValueError("polarization strings require at least two points along each axis")
        others = [a for a in range(3) if a != direction]
        paths = []
        for i in range(int(dimensions[others[0]])):
            for j in range(int(dimensions[others[1]])):
                path = np.zeros((int(dimensions[direction]) + 1, 3))
                path[:, direction] = np.linspace(0, 1, len(path))
                path[:, others[0]] = i / dimensions[others[0]]
                path[:, others[1]] = j / dimensions[others[1]]
                paths.append(path)
        phases = wilson_loop(
            model,
            np.asarray(paths),
            occupied_bands=occupied_bands,
            fermi_energy=fermi_energy,
            execution=execution,
        ).arrays["berry_phase"]
        phases = phases.reshape(tuple(dimensions[others].astype(int)))
        for axis in range(2):
            winding = np.angle(np.exp(1j * (np.roll(phases, -1, axis=axis) - phases))).sum(axis=axis) / (
                2 * np.pi
            )
            if np.max(np.abs(winding), initial=0) > 1e-6:
                raise ValueError(
                    "polarization requires a zero-Chern insulating manifold; transverse Berry phases wind"
                )
        continuous = np.unwrap(np.unwrap(phases, axis=0), axis=1)
        mean = np.angle(np.exp(1j * continuous.mean()))
        reduced[direction] = -spin_degeneracy(model) * mean / (2 * np.pi)
    quantum = E_CHARGE * 1e20 * cell(model) / volume(model)
    electronic = reduced @ quantum
    arrays = {"electronic": electronic, "reduced_electronic": reduced, "polarization_quantum": quantum}
    dims: dict[str, tuple[str, ...]] = {
        "electronic": ("cartesian",),
        "reduced_electronic": ("reciprocal_axis",),
        "polarization_quantum": ("lattice_axis", "cartesian"),
    }
    units = {"electronic": "C/m^2", "reduced_electronic": "1", "polarization_quantum": "C/m^2"}
    if ionic_charges is not None:
        positions = np.asarray(getattr(model, "atom_positions", None), float)
        charges = np.asarray(ionic_charges, float)
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or charges.shape != (len(positions),)
            or not np.isfinite(charges).all()
        ):
            raise ValueError("ionic_charges require finite per-atom charges and model atom positions")
        arrays["ionic"] = charges @ positions * E_CHARGE * 1e20 / volume(model)
        arrays["total"] = electronic + arrays["ionic"]
        dims.update(ionic=("cartesian",), total=("cartesian",))
        units.update(ionic="C/m^2", total="C/m^2")
    return _result(
        arrays, dims, units, "polarization", branch="principal; modulo lattice polarization quanta"
    )
