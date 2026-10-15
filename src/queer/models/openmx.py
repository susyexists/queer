"""Nonorthogonal OpenMX models, evaluated without a converter or runtime bridge."""

from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix

from ..capabilities import CapabilityError
from ..execution import ExecutionOptions, execution_session
from ..geometry.lattice import reciprocal_lattice, validated_lattice, validated_points
from ..geometry.paths import KPath, interpolate_path, parse_point_names
from ..io.openmx import OpenMXData, read_scfout
from ..physics.validation import NumericalError as NumericalError
from .contracts import FrozenEigenstateBatch as EigenstateBatch


def _blocks(translations, matrices, name):
    if isinstance(matrices, dict):
        if any(len(key) != 3 or any(not np.isfinite(x) or int(x) != x for x in key) for key in matrices):
            raise ValueError(f"{name} translation keys must contain three finite integers")
        out = {
            tuple(map(int, key)): csr_matrix(value, dtype=complex, copy=True)
            for key, value in matrices.items()
        }
    else:
        array = np.asarray(matrices, dtype=complex)
        if array.ndim != 3 or array.shape[0] != len(translations):
            raise ValueError(f"{name} must have shape (NR,N,N)")
        out = {}
        for translation, matrix in zip(translations, array, strict=True):
            key = tuple(map(int, translation))
            out[key] = out.get(key, csr_matrix(matrix.shape, dtype=complex)) + csr_matrix(matrix)
    if not out:
        raise ValueError(f"{name} must contain at least one translation")
    n = next(iter(out.values())).shape[0]
    for value in out.values():
        if value.shape != (n, n) or n < 1 or not np.isfinite(value.data).all():
            raise ValueError(f"{name} must contain finite equally sized square matrices")
        value.sum_duplicates()
        value.sort_indices()
    return out


def _hermitian(matrix, k, name, tolerance=1e-10):
    difference = np.linalg.norm(matrix - matrix.conj().T, ord=np.inf)
    scale = np.linalg.norm(matrix, ord=np.inf)
    if difference > tolerance * (1 + scale):
        raise NumericalError(f"Non-Hermitian {name} at k={np.asarray(k).tolist()}; residual={difference:g}")
    return (matrix + matrix.conj().T) / 2


def _canonical_native_operators(data):
    """Reconcile finite-grid integrals with exact translation adjoint identities.

    Raw records remain in ``model.data``. A measured, bounded correction fixes
    SCF quadrature noise, including the position derivative identity; it never
    replaces a missing operator. Large inconsistencies are rejected.
    """
    n = next(iter(data.hamiltonian.values())).shape[0]
    zero = csr_matrix((n, n), dtype=complex)
    errors = {}

    def magnitude(matrix):
        return float(np.max(np.asarray(abs(matrix).sum(axis=1)), initial=0))

    def reconcile(blocks, name, tolerance):
        keys = set(blocks) | {tuple(-x for x in key) for key in blocks}
        output, maximum = {}, 0.0
        for key in keys:
            other = tuple(-x for x in key)
            a, b = blocks.get(key, zero), blocks.get(other, zero).getH()
            error = magnitude(a - b)
            if error > tolerance * (1 + magnitude(a) + magnitude(b)):
                raise NumericalError(f"SCF {name} translation identity failed at R={key}: {error:g}")
            output[key] = ((a + b) * 0.5).tocsr()
            maximum = max(maximum, error)
        errors[name] = maximum
        return output

    h = reconcile(data.hamiltonian, "hamiltonian_eV", 1e-8)
    s = reconcile(data.overlap, "overlap", 1e-8)
    r = []
    if data.metadata["position_order"]:
        for axis, blocks in enumerate(data.position):
            keys = set(blocks) | set(s) | {tuple(-x for x in key) for key in blocks}
            output, maximum = {}, 0.0
            for key in keys:
                other = tuple(-x for x in key)
                a, b = blocks.get(key, zero), blocks.get(other, zero).getH()
                translation = (np.asarray(key) @ data.lattice)[axis] * s.get(key, zero)
                error = magnitude(a - b - translation)
                if error > 1e-3 * (1 + magnitude(a) + magnitude(b) + magnitude(translation)):
                    raise NumericalError(
                        f"SCF position covariance failed at R={key}, axis={axis}: {error:g} A"
                    )
                output[key] = ((a + b + translation) * 0.5).tocsr()
                maximum = max(maximum, error)
            r.append(output)
            errors[f"position_{axis}_Angstrom"] = maximum
    return h, s, tuple(r) if r else None, errors


def _solve_points(model, points, selection):
    values = np.empty((len(selection), len(points)))
    vectors = np.empty((len(points), model.nbnd, len(selection)), complex)
    overlaps = np.empty((len(points), model.nbnd, model.nbnd), complex)
    for i, point in enumerate(points):
        h, s = model.hamiltonian(point), model.overlap(point)
        try:
            if model.spin_mode == "collinear":
                n = model.n_spatial
                eu, cu = eigh(h[:n, :n], s[:n, :n], check_finite=False)
                ed, cd = eigh(h[n:, n:], s[n:, n:], check_finite=False)
                energy = np.concatenate((eu, ed))
                coeff = np.zeros_like(h)
                coeff[:n, :n], coeff[n:, n:] = cu, cd
            else:
                energy, coeff = eigh(h, s, check_finite=False)
        except np.linalg.LinAlgError as error:
            raise NumericalError(
                f"Positive-definite overlap/generalized solver failed at k={point.tolist()}"
            ) from error
        residual = h @ coeff - (s @ coeff) * energy[None, :]
        norm = max(
            1.0,
            np.linalg.norm(h, ord=np.inf) * np.linalg.norm(coeff, ord=np.inf),
            np.linalg.norm(s @ coeff, ord=np.inf) * np.max(np.abs(energy), initial=0),
        )
        normalization = coeff.conj().T @ s @ coeff - np.eye(model.nbnd)
        if np.linalg.norm(residual, ord=np.inf) / norm > 1e-10 or np.max(np.abs(normalization)) > 1e-10:
            raise NumericalError(f"Generalized eigenstate validation failed at k={point.tolist()}")
        values[:, i], vectors[i], overlaps[i] = energy[selection], coeff[:, selection], s
    return values, vectors, overlaps


def _energy_points(model, points, selection):
    # Validate eigenvectors and the generalized residual inside each bounded
    # solve, but retain only band energies for energy-only calculations.
    return _solve_points(model, points, selection)[0]


class OpenMXModel:
    """A sparse localized-orbital model with ``H C = S C E``.

    Direct construction supports analytical models. ``hopping`` and
    ``overlap_matrices`` are either (NR,N,N) arrays or sparse dictionaries keyed
    by integer translations. Position matrices use (3,NR,N,N) or three sparse
    dictionaries. Eigenstates retain absolute eV; ``energies`` subtracts EF.
    """

    def __init__(
        self,
        translations,
        hopping,
        overlap_matrices=None,
        *,
        position=None,
        lattice=None,
        reciprocal=None,
        fermi_energy=0.0,
        spin_mode="nonmagnetic",
        spin_degeneracy=None,
        atom_positions=None,
        orbital_metadata=None,
        execution=None,
        hermiticity_tolerance=1e-10,
    ):
        translations = np.asarray(translations)
        if (
            translations.ndim != 2
            or translations.shape[1] != 3
            or not len(translations)
            or not np.isfinite(translations).all()
            or not np.equal(translations, np.round(translations)).all()
        ):
            raise ValueError("translations must be a nonempty (NR,3) integer array")
        self.hamiltonian_blocks = _blocks(translations, hopping, "Hamiltonian")
        self.nbnd = next(iter(self.hamiltonian_blocks.values())).shape[0]
        if overlap_matrices is None:
            overlap_matrices = {(0, 0, 0): csr_matrix(np.eye(self.nbnd))}
        self.overlap_blocks = _blocks(translations, overlap_matrices, "overlap")
        if any(block.shape != (self.nbnd, self.nbnd) for block in self.overlap_blocks.values()):
            raise ValueError("Hamiltonian and overlap basis sizes differ")
        if spin_mode not in {"nonmagnetic", "collinear", "soc"}:
            raise ValueError("spin_mode must be nonmagnetic, collinear, or soc")
        if spin_mode != "nonmagnetic" and self.nbnd % 2:
            raise ValueError("collinear/SOC models require an even basis size")
        self.spin_mode = spin_mode
        self.n_spatial = self.nbnd if spin_mode == "nonmagnetic" else self.nbnd // 2
        self.spin_degeneracy = (
            (2 if spin_mode == "nonmagnetic" else 1) if spin_degeneracy is None else spin_degeneracy
        )
        if self.spin_degeneracy not in (1, 2):
            raise ValueError("spin_degeneracy must be one or two")
        if spin_mode != "nonmagnetic" and self.spin_degeneracy != 1:
            raise ValueError("Explicit collinear/SOC spin channels have occupation degeneracy one")
        self.band_channels = (
            np.repeat((0, 1), self.n_spatial) if spin_mode == "collinear" else np.zeros(self.nbnd, dtype=int)
        )
        self.lattice = validated_lattice(
            lattice
            if lattice is not None
            else (
                2 * np.pi * np.linalg.inv(validated_lattice(reciprocal)).T
                if reciprocal is not None
                else np.eye(3)
            )
        )
        self.g_vec = reciprocal_lattice(self.lattice)
        self.fermi_energy = fermi_energy
        self.execution = execution or ExecutionOptions()
        if not np.isfinite(hermiticity_tolerance) or hermiticity_tolerance <= 0:
            raise ValueError("hermiticity_tolerance must be positive")
        self.hermiticity_tolerance = float(hermiticity_tolerance)
        self.atom_positions = np.asarray(
            atom_positions if atom_positions is not None else [], dtype=float
        ).reshape(-1, 3)
        self.orbital_metadata = list(orbital_metadata or [])
        self.position_blocks = (
            tuple(_blocks(translations, axis, "position") for axis in position)
            if position is not None
            else ()
        )
        if self.position_blocks and (
            len(self.position_blocks) != 3
            or any(
                block.shape != (self.nbnd, self.nbnd)
                for axis in self.position_blocks
                for block in axis.values()
            )
        ):
            raise ValueError("position must contain three operators matching the model basis")
        self.translations = np.array(
            sorted(set(self.hamiltonian_blocks) | set(self.overlap_blocks)), dtype=int
        )
        self.translations.setflags(write=False)
        self.data: OpenMXData | None = None
        self.electron_count: float | None = None
        self.temperature_kelvin: float | None = None
        self.path: Path | None = None
        self.poscar = None
        self.interpolation = "localized_orbitals"
        self.degeneracy_tolerance = 1e-9
        self.metadata: dict[str, Any] = {
            "position_link": "midpoint_first_order",
            "energy_unit": "eV",
            "length_unit": "Angstrom",
        }
        self.capabilities = {"hamiltonian", "overlap", "derivatives", "spin"}
        if self.position_blocks:
            self.capabilities |= {"position", "basis_overlap"}
        if self.orbital_metadata:
            self.capabilities.add("orbital_metadata")
        if self.spin_mode != "nonmagnetic":
            for block in self.overlap_blocks.values():
                n = self.n_spatial
                if (
                    block[:n, n:].nnz
                    or block[n:, :n].nnz
                    or np.max(np.abs((block[:n, :n] - block[n:, n:]).data), initial=0) > 1e-12
                ):
                    raise ValueError("Spinful OpenMX requires a duplicated spin-independent spatial overlap")
        if self.spin_mode == "collinear":
            for operator in (self.hamiltonian_blocks, self.overlap_blocks):
                for block in operator.values():
                    if (
                        block[: self.n_spatial, self.n_spatial :].nnz
                        or block[self.n_spatial :, : self.n_spatial].nnz
                    ):
                        raise ValueError("collinear matrices must be spin-block diagonal")

    @property
    def fermi_energy(self):
        return self._fermi_energy

    @fermi_energy.setter
    def fermi_energy(self, value):
        if not np.isfinite(value):
            raise ValueError("Fermi energy must be finite")
        self._fermi_energy = float(value)

    @classmethod
    def from_scfout(cls, path, *, ef=None, openmx_input=None, execution=None):
        data = read_scfout(path, openmx_input=openmx_input)
        h, s, r, residuals = _canonical_native_operators(data)
        instance = cls(
            np.array(list(h)),
            h,
            s,
            position=r,
            lattice=data.lattice,
            fermi_energy=data.fermi_energy if ef is None else ef,
            spin_mode=data.spin_mode,
            atom_positions=data.atom_positions,
            orbital_metadata=data.orbital_metadata,
            execution=execution,
        )
        instance.data = data
        instance.path = Path(path).expanduser().resolve().parent
        instance.electron_count = data.electron_count
        instance.temperature_kelvin = data.temperature_kelvin
        instance.metadata.update(data.metadata)
        instance.metadata["source_operator_identity_residuals"] = residuals
        instance.metadata["source_identity_reconciliation"] = "bounded real-space translation adjoints"
        instance.degeneracy_tolerance = max(1e-9, 10 * residuals["hamiltonian_eV"])
        instance.metadata["degeneracy_tolerance_eV"] = instance.degeneracy_tolerance
        instance.metadata["degeneracy_tolerance_definition"] = (
            "max(1e-9 eV, 10 times source Hamiltonian adjoint residual)"
        )
        return instance

    @classmethod
    def from_directory(cls, directory, scfout=None, **kwargs):
        root = Path(directory).expanduser().resolve()
        if scfout is None:
            candidates = list(root.glob("*.scfout"))
            if len(candidates) != 1:
                raise ValueError("Provide scfout when a directory does not contain exactly one SCFOUT file")
            scfout = candidates[0]
        if kwargs.get("openmx_input") is not None:
            kwargs["openmx_input"] = root / kwargs["openmx_input"]
        return cls.from_scfout(root / scfout, **kwargs)

    def _point(self, k):
        point = np.asarray(k, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("k must be a finite (3,) fractional reciprocal point")
        return point

    def _fourier(self, blocks, k, order=0):
        point = self._point(k)
        out = np.zeros((3,) * order + (self.nbnd, self.nbnd), dtype=complex)
        for key, block in blocks.items():
            phase = np.exp(2j * np.pi * np.dot(point, key))
            vector = np.asarray(key) @ self.lattice
            coo = block.tocoo(copy=False)
            factor = phase
            if order == 1:
                factor = 1j * vector * phase
            elif order == 2:
                factor = -np.outer(vector, vector) * phase
            if order:
                out[..., coo.row, coo.col] += np.asarray(factor)[..., None] * coo.data
            else:
                out[coo.row, coo.col] += phase * coo.data
        return out

    def hamiltonian(self, k):
        return _hermitian(
            self._fourier(self.hamiltonian_blocks, k), k, "Hamiltonian", self.hermiticity_tolerance
        )

    def overlap(self, k):
        return _hermitian(self._fourier(self.overlap_blocks, k), k, "overlap", self.hermiticity_tolerance)

    def position(self, k):
        if not self.position_blocks:
            if len(self.overlap_blocks) == 1 and (0, 0, 0) in self.overlap_blocks:
                return np.zeros((3, self.nbnd, self.nbnd), complex)
            raise CapabilityError(
                "Position matrices are required for nonorthogonal geometric/optical observables"
            )
        return np.asarray([self._fourier(blocks, k) for blocks in self.position_blocks])

    def derivatives(self, k, order=1):
        if order not in (1, 2):
            raise ValueError("derivative order must be one or two")
        derivative_position = (
            np.stack([self._fourier(blocks, k, order) for blocks in self.position_blocks], axis=order)
            if self.position_blocks
            else np.zeros((3,) * (order + 1) + (self.nbnd, self.nbnd), complex)
        )
        h, s = (self._fourier(blocks, k, order) for blocks in (self.hamiltonian_blocks, self.overlap_blocks))
        return {
            "H": (h + h.conj().swapaxes(-1, -2)) / 2,
            "S": (s + s.conj().swapaxes(-1, -2)) / 2,
            "r": derivative_position,
        }

    def _band_indices(self, bands, spin=None):
        available = np.arange(self.nbnd)
        if spin is not None:
            if self.spin_mode != "collinear" or spin not in {"up", "down"}:
                raise ValueError("spin selection is up/down and requires a collinear model")
            available = available[self.band_channels == (0 if spin == "up" else 1)]
        if bands is None:
            return available
        if isinstance(bands, bool):
            raise ValueError("bands must be integer indices")
        selected = np.atleast_1d(bands)
        if (
            selected.ndim != 1
            or selected.dtype.kind not in "iu"
            or np.any(selected < 0)
            or np.any(selected >= len(available))
        ):
            raise ValueError("band indices are outside the selected model channel")
        return available[selected.astype(int)]

    def eigenstates(self, kpoints, bands=None, execution=None, spin=None):
        points = validated_points(kpoints.path if isinstance(kpoints, KPath) else kpoints)
        selected = self._band_indices(bands, spin)
        energies = np.empty((len(selected), len(points)))
        coeff = np.empty((len(points), self.nbnd, len(selected)), complex)
        overlaps = np.empty((len(points), self.nbnd, self.nbnd), complex)
        options = (execution or self.execution).resolved()
        size = options.bounded_batch_size(192 * self.nbnd**2 + 16 * self.nbnd)
        starts = range(0, len(points), size)
        if options.workers == 1:
            from threadpoolctl import threadpool_limits

            with threadpool_limits(limits=options.native_threads):
                for start in starts:
                    values, vectors, metric = _solve_points(self, points[start : start + size], selected)
                    end = start + values.shape[1]
                    energies[:, start:end], coeff[start:end], overlaps[start:end] = values, vectors, metric
        else:
            from joblib import delayed

            with execution_session(options) as session:
                remaining = iter(starts)
                while window := tuple(islice(remaining, options.workers)):
                    results = session.parallel()(
                        delayed(_solve_points)(self, points[start : start + size], selected)
                        for start in window
                    )
                    for start, (values, vectors, metric) in zip(window, results, strict=True):
                        end = start + values.shape[1]
                        energies[:, start:end], coeff[start:end], overlaps[start:end] = (
                            values,
                            vectors,
                            metric,
                        )
        return EigenstateBatch(energies, coeff, overlaps, self.band_channels[selected])

    def energies(self, kpoints, bands=None, execution=None, spin=None):
        points = validated_points(kpoints.path if isinstance(kpoints, KPath) else kpoints)
        selected = self._band_indices(bands, spin)
        energies = np.empty((len(selected), len(points)))
        options = (execution or self.execution).resolved()
        size = options.bounded_batch_size(192 * self.nbnd**2 + 16 * self.nbnd)
        starts = range(0, len(points), size)
        if options.workers == 1:
            from threadpoolctl import threadpool_limits

            with threadpool_limits(limits=options.native_threads):
                for start in starts:
                    values = _energy_points(self, points[start : start + size], selected)
                    energies[:, start : start + values.shape[1]] = values
        else:
            from joblib import delayed

            with execution_session(options) as session:
                remaining = iter(starts)
                while window := tuple(islice(remaining, options.workers)):
                    results = session.parallel()(
                        delayed(_energy_points)(self, points[start : start + size], selected)
                        for start in window
                    )
                    for start, values in zip(window, results, strict=True):
                        energies[:, start : start + values.shape[1]] = values
        return energies - self.fermi_energy

    def eig(self, kpoints):
        result = self.eigenstates(kpoints)
        return list(result.energies.T), list(result.coefficients)

    def solver(self, k):
        return self.eigenstates(np.asarray(k).reshape(1, 3)).energies[:, 0]

    def basis_overlap(self, k, kprime):
        """Midpoint first-order overlap between cell-periodic localized bases.

        The expansion is used for converged loop phases, not as an exact
        finite-distance fidelity. Higher raw position moments remain in data.
        """
        k, kp = self._point(k), self._point(kprime)
        midpoint = (k + kp) / 2
        delta = (kp - k) @ self.g_vec
        position = self.position(midpoint)
        connection = (position + position.conj().transpose(0, 2, 1)) / 2
        return self.overlap(midpoint) - 1j * np.einsum("a,aij->ij", delta, connection)

    def spin_operators(self, k):
        """Pauli matrices covariant in the all-up/all-down spatial basis."""
        s = self.overlap(k)
        result = np.zeros((3, self.nbnd, self.nbnd), complex)
        if self.spin_mode == "nonmagnetic":
            return result
        n = self.n_spatial
        spatial = s[:n, :n]
        result[0, :n, n:], result[0, n:, :n] = spatial, spatial
        result[1, :n, n:], result[1, n:, :n] = -1j * spatial, 1j * spatial
        result[2, :n, :n], result[2, n:, n:] = spatial, -spatial
        return result

    def kpath(self, point_names, n_points=200, poscar=None, **kwargs):
        if poscar is not None:
            from ..geometry.paths import named_k_path
            from ..io.structure import read_poscar

            supplied_cell, fractional, _ = read_poscar(poscar)
            if not np.allclose(supplied_cell, self.lattice, atol=1e-6, rtol=1e-8):
                raise ValueError("Supplementary POSCAR lattice does not match the OpenMX structure")
            if len(self.atom_positions):
                if len(fractional) != len(self.atom_positions):
                    raise ValueError("Supplementary POSCAR atom count does not match OpenMX")
                difference = fractional - self.atom_positions @ np.linalg.inv(self.lattice)
                difference -= np.rint(difference)
                if np.max(np.linalg.norm(difference @ self.lattice, axis=1), initial=0) > 1e-5:
                    raise ValueError("Supplementary POSCAR positions/order do not match OpenMX")
            return named_k_path(point_names, n_points, poscar, **kwargs)
        from ..geometry.symmetry import compute_band_path

        if self.data is None or not self.metadata.get("orbital_labels_complete"):
            raise CapabilityError("Named paths require complete embedded atom/species metadata or POSCAR")
        symbols = []
        aliases = sorted(self.metadata["species"])
        for atom in range(len(self.atom_positions)):
            row = next(row for row in self.orbital_metadata if row["atom_index"] == atom)
            symbols.append(aliases.index(row["species"]) + 1)
        fractional = self.atom_positions @ np.linalg.inv(self.lattice)
        coords = compute_band_path((self.lattice, fractional, symbols), **kwargs)["point_coords"]
        names = parse_point_names(point_names)
        sym, points = interpolate_path(n_points, [coords[name] for name in names], self.g_vec)
        distances = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(points @ self.g_vec, axis=0), axis=1)))
        )
        return KPath(
            sym, points, ["Γ" if name == "GAMMA" else name for name in names], names, coords, distances
        )
