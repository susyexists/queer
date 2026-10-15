"""Plane-wave unfolding using the actual OpenMX numerical radial basis."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np

from ..capabilities import CapabilityError
from ..execution import ExecutionOptions
from ..geometry.lattice import reciprocal_lattice, validated_lattice, validated_points
from ..io.pao import PAORadialData, read_pao, real_harmonics
from ..physics.constants import KINETIC_EV_ANGSTROM2 as KINETIC_EV_ANGSTROM2
from ._common import positive
from ._tb import degeneracy_tolerance, states
from .electronic import _mean_manifolds, _projection_manifolds, _selection
from .result import ObservableResult


def _radial_basis(model, radial_data=None, pao_dir=None):
    metadata = getattr(model, "orbital_metadata", None)
    spatial = getattr(model, "n_spatial", model.nbnd)
    if not metadata or len(metadata) < spatial:
        raise CapabilityError("radial unfolding requires native orbital metadata")
    rows = metadata[:spatial]
    sources = dict(radial_data or {})
    basis = []
    loaded = {}
    for row in rows:
        key = row.get("pao_key")
        if key is None:
            raise CapabilityError("every unfolding orbital needs a PAO radial key")
        if row.get("contracted") and sources.get(key) is None:
            raise CapabilityError("contracted SCF orbitals require explicit matching radial_data files")
        if key not in loaded:
            supplied = sources.get(key)
            if supplied is None and pao_dir is not None:
                supplied = Path(pao_dir) / f"{key}.pao"
            if supplied is None:
                raise CapabilityError(f"missing actual PAO radial functions for {key}")
            if not isinstance(supplied, PAORadialData) and not Path(supplied).is_file():
                raise CapabilityError(f"missing actual PAO radial file for {key}: {supplied}")
            loaded[key] = supplied if isinstance(supplied, PAORadialData) else read_pao(supplied)
        ell, zeta, m = (row.get(name) for name in ("l", "zeta", "m"))
        if any(value is None for value in (ell, zeta, m)):
            raise CapabilityError("unfolding metadata needs l, zeta and OpenMX harmonic index")
        if any(int(value) != value for value in (ell, zeta, m)):
            raise ValueError("unfolding metadata needs integer l, zeta and OpenMX harmonic index")
        if (
            not 0 <= m <= 2 * ell
            or ell not in loaded[key].radial
            or not 0 <= zeta < loaded[key].radial[ell].shape[1]
        ):
            raise ValueError(f"PAO {key} does not match the SCF orbital indices")
        center = np.asarray(row.get("center"), dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("unfolding orbital centers must be finite Cartesian angstroms")
        basis.append((loaded[key], int(ell), int(zeta), int(m), center))
    return basis


def _integer_matrix(matrix):
    array = np.asarray(matrix)
    if array.shape != (3, 3) or not np.isfinite(array).all() or not np.equal(array, np.round(array)).all():
        raise ValueError("supercell_matrix must be a finite integer (3,3) array")
    matrix = array.astype(int)
    if abs(np.linalg.det(matrix)) < 0.5:
        raise ValueError("supercell_matrix must be nonsingular")
    return matrix


def _plane_wave_chunks(point, direct, reciprocal, ecut, chunk_size):
    qmax = np.sqrt(ecut / KINETIC_EV_ANGSTROM2)
    bounds = np.ceil(qmax * np.linalg.norm(direct, axis=1) / (2 * np.pi) + np.abs(point)).astype(int)
    pending = []
    for indices in product(*(range(-int(bound), int(bound) + 1) for bound in bounds)):
        momentum = (point + indices) @ reciprocal
        if momentum @ momentum <= qmax * qmax * (1 + 1e-12):
            pending.append(momentum)
            if len(pending) == chunk_size:
                yield np.asarray(pending)
                pending = []
    if pending:
        yield np.asarray(pending)


def _unfold(model, kpoints, supercell_matrix, ecut, bands, radial_data, pao_dir, execution, with_spin):
    points = validated_points(kpoints)
    matrix = _integer_matrix(supercell_matrix)
    positive(ecut, "ecut")
    lattice = validated_lattice(getattr(model, "lattice", None))
    primitive = np.linalg.solve(matrix, lattice)
    reciprocal = reciprocal_lattice(primitive)
    volume = abs(float(np.linalg.det(lattice)))
    basis = _radial_basis(model, radial_data, pao_dir)
    n_spatial = len(basis)
    options = (execution or getattr(model, "execution", ExecutionOptions())).resolved()
    # Projection functions and amplitudes are streamed over plane waves.
    chunk_size = options.bounded_batch_size(64 * n_spatial + 64 * model.nbnd)
    superpoints = np.mod(points @ matrix.T, 1)
    selected = _selection(model, bands)
    energies = np.empty((len(selected), len(points)))
    weights = np.zeros_like(energies)
    spin = np.zeros((*energies.shape, 3))
    manifold_sizes = np.ones_like(energies, dtype=int)
    plane_wave_counts = np.zeros(len(points), dtype=int)
    mode = getattr(model, "spin_mode", "nonmagnetic")
    for k, point in enumerate(points):
        solution = states(model, superpoints[k : k + 1], execution=options)
        blocks, sizes = _projection_manifolds(
            model, solution.energies[:, 0], solution.band_channels, selected
        )
        energies[:, k] = solution.energies[selected, 0] - model.fermi_energy
        manifold_sizes[:, k] = sizes[selected]
        vectors = solution.coefficients[0]
        spinful = vectors.shape[0] == 2 * n_spatial
        full_weights = np.zeros(model.nbnd)
        full_spin = np.zeros((model.nbnd, 3))
        for momenta in _plane_wave_chunks(point, primitive, reciprocal, ecut, chunk_size):
            plane_wave_counts[k] += len(momenta)
            magnitude = np.linalg.norm(momenta, axis=1)
            projections = np.empty((len(momenta), n_spatial), dtype=complex)
            radial_cache, angular_cache = {}, {}
            for orbital, (data, ell, zeta, m, center) in enumerate(basis):
                radial_key = (id(data), ell, zeta)
                if radial_key not in radial_cache:
                    radial_cache[radial_key] = data.fourier(magnitude, ell, zeta)
                if ell not in angular_cache:
                    angular_cache[ell] = real_harmonics(ell, momenta)
                projections[:, orbital] = (
                    4
                    * np.pi
                    * (-1j) ** ell
                    * radial_cache[radial_key]
                    * angular_cache[ell][:, m]
                    * np.exp(-1j * (momenta @ center))
                    / np.sqrt(volume)
                )
            up = projections @ vectors[:n_spatial]
            if spinful:
                down = projections @ vectors[n_spatial:]
                norm_up, norm_down = np.sum(np.abs(up) ** 2, axis=0), np.sum(np.abs(down) ** 2, axis=0)
                cross = np.sum(up.conj() * down, axis=0)
                full_weights += norm_up + norm_down
                # Weighted physical spin: hbar/2 times the Pauli expectation.
                full_spin[:, 0] += cross.real
                full_spin[:, 1] += cross.imag
                full_spin[:, 2] += 0.5 * (norm_up - norm_down)
            else:
                probability = np.sum(np.abs(up) ** 2, axis=0)
                full_weights += probability
                if mode == "collinear":
                    full_spin[:, 2] += 0.5 * np.where(solution.band_channels == 0, 1, -1) * probability
        weights[:, k] = _mean_manifolds(full_weights, blocks, 0)[selected]
        spin[:, k] = _mean_manifolds(full_spin, blocks, 0)[selected]
    arrays = {
        "kpoints": points,
        "supercell_kpoints": superpoints,
        "energies": energies,
        "weights": weights,
        "plane_wave_counts": plane_wave_counts,
        "manifold_sizes": manifold_sizes,
    }
    dims = {
        "kpoints": ("k", "cartesian"),
        "supercell_kpoints": ("k", "cartesian"),
        "energies": ("band", "k"),
        "weights": ("band", "k"),
        "plane_wave_counts": ("k",),
        "manifold_sizes": ("band", "k"),
    }
    units = {
        "kpoints": "fractional_primitive_reciprocal",
        "supercell_kpoints": "fractional_reciprocal",
        "energies": "eV",
        "weights": "1",
        "plane_wave_counts": "1",
        "manifold_sizes": "1",
    }
    if with_spin:
        arrays["spin"] = spin
        dims["spin"] = ("band", "k", "cartesian")
        units["spin"] = "hbar"
    return ObservableResult(
        arrays,
        dims,
        units,
        {
            "projection": "actual_radial_plane_waves",
            "ecut_eV": float(ecut),
            "supercell_matrix": matrix.tolist(),
            "finite_cutoff": True,
            "spin_weighted": bool(with_spin),
            "normalization": "plane_wave_1/sqrt_supercell_volume",
            "degenerate_state_interpretation": "complete_subspace_mean_per_state",
            "degeneracy_tolerance_eV": degeneracy_tolerance(model),
        },
    )


def bandunfolding(
    model, kpoints, supercell_matrix, ecut=100.0, bands=None, radial_data=None, pao_dir=None, execution=None
):
    """Unfold primitive reciprocal kpoints with an eV plane-wave kinetic cutoff."""
    return _unfold(model, kpoints, supercell_matrix, ecut, bands, radial_data, pao_dir, execution, False)


def bandunfolding_spin_texture(
    model, kpoints, supercell_matrix, ecut=100.0, bands=None, radial_data=None, pao_dir=None, execution=None
):
    """Return unfolding weights and their physical spin-weighted projections."""
    return _unfold(model, kpoints, supercell_matrix, ecut, bands, radial_data, pao_dir, execution, True)
