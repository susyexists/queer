"""Wannier Hamiltonians with validated, bounded numerical evaluation."""

from __future__ import annotations

from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..execution import ExecutionOptions, execution_session
from ..geometry.lattice import reciprocal_lattice, validated_lattice, validated_points
from ..geometry.paths import KPath, named_k_path
from ..io.structure import read_poscar
from ..io.wannier import (
    WannierData,
    normalized_operators,
    read_centres_xyz,
    read_hr,
    read_tb,
    read_wout_lattice,
    read_wsvec,
    resolve_wannier_files,
)


class NumericalError(RuntimeError):
    """A scientific calculation could not produce validated numerical output."""


def _assemble(translations, hopping, points):
    phase = np.exp(2j * np.pi * (points @ translations.T))
    matrices = np.einsum("kr,rij->kij", phase, hopping, optimize=True)
    if not np.isfinite(matrices).all():
        raise NumericalError("Hamiltonian assembly produced nonfinite values")
    adjoints = matrices.conj().transpose(0, 2, 1)
    residuals = np.linalg.norm(matrices - adjoints, ord=np.inf, axis=(1, 2))
    norms = np.linalg.norm(matrices, ord=np.inf, axis=(1, 2))
    invalid = np.flatnonzero(residuals > 1e-10 + 1e-10 * norms)
    if invalid.size:
        index = int(invalid[0])
        raise NumericalError(
            f"Non-Hermitian Hamiltonian at k={points[index].tolist()}; residual={residuals[index]:.6g} eV"
        )
    return (matrices + adjoints) * 0.5, float(residuals.max(initial=0.0))


def _solve_batch(translations, hopping, points):
    matrices, residual = _assemble(translations, hopping, points)
    try:
        values = np.linalg.eigvalsh(matrices)
    except np.linalg.LinAlgError as error:
        raise NumericalError(
            f"Hermitian eigensolver failed for k-points starting at {points[0].tolist()}"
        ) from error
    if not np.isfinite(values).all():
        raise NumericalError("Hermitian eigensolver produced nonfinite values")
    return values.T, residual


def _solve_state_batch(translations, hopping, points):
    matrices, residual = _assemble(translations, hopping, points)
    try:
        values, vectors = np.linalg.eigh(matrices)
    except np.linalg.LinAlgError as error:
        raise NumericalError(
            f"Hermitian eigensolver failed for k-points starting at {points[0].tolist()}"
        ) from error
    if not np.isfinite(values).all() or not np.isfinite(vectors).all():
        raise NumericalError("Hermitian eigensolver produced nonfinite eigenstates")
    return values.T, vectors, residual


class WannierModel:
    """A Hamiltonian whose positive-phase real-space weights are already applied.

    Construct from files with :meth:`from_directory`. Direct construction is
    also supported for analytic models: ``translations`` is (NR,3), and
    ``hopping`` is (NR,NB,NB), with degeneracy weights already applied.
    Reciprocal coordinates use row vectors. Energies are measured relative
    to ``fermi_energy`` in eV; Hamiltonian matrices retain their absolute zero.
    """

    def __init__(
        self,
        translations,
        hopping,
        *,
        reciprocal=None,
        lattice=None,
        centres=None,
        position=None,
        fermi_energy=0.0,
        execution: ExecutionOptions | None = None,
    ):
        lattice_indices = np.asarray(translations)
        if lattice_indices.ndim != 2 or lattice_indices.shape[1] != 3 or not len(lattice_indices):
            raise ValueError("translations must be a nonempty (NR,3) integer array")
        if (
            not np.isfinite(lattice_indices).all()
            or not np.equal(lattice_indices, np.round(lattice_indices)).all()
        ):
            raise ValueError("translations must contain finite integers")
        matrices = np.asarray(hopping, dtype=np.complex128)
        if (
            matrices.ndim != 3
            or matrices.shape[0] != len(lattice_indices)
            or matrices.shape[1] != matrices.shape[2]
            or matrices.shape[1] < 1
        ):
            raise ValueError("hopping must have shape (NR,NB,NB)")
        if not np.isfinite(matrices).all():
            raise ValueError("hopping must be finite")
        self.translations = np.array(lattice_indices, dtype=np.int64, copy=True)
        self.hopping_matrices = np.array(matrices, dtype=np.complex128, copy=True)
        self.translations.setflags(write=False)
        self.hopping_matrices.setflags(write=False)
        self.nbnd = matrices.shape[1]
        self.lattice = None if lattice is None else validated_lattice(lattice).copy()
        self.g_vec = None if reciprocal is None else validated_lattice(reciprocal).copy()
        if self.lattice is not None:
            expected = reciprocal_lattice(self.lattice)
            if self.g_vec is not None and not np.allclose(self.g_vec, expected, rtol=1e-10, atol=1e-10):
                raise ValueError("lattice and reciprocal vectors are inconsistent")
            self.g_vec = expected
        elif self.g_vec is not None:
            self.lattice = 2 * np.pi * np.linalg.inv(self.g_vec).T
        self.centres = None if centres is None else np.asarray(centres, float).copy()
        if self.centres is not None:
            if self.centres.shape != (self.nbnd, 3) or not np.isfinite(self.centres).all():
                raise ValueError("centres must be a finite (NB,3) Cartesian Angstrom array")
            self.centres.setflags(write=False)
        self.position_matrices = None if position is None else np.asarray(position, complex).copy()
        if self.position_matrices is not None:
            if (
                self.position_matrices.shape != matrices.shape + (3,)
                or not np.isfinite(self.position_matrices).all()
            ):
                raise ValueError("position must be a finite (NR,NB,NB,3) Angstrom array")
            self.position_matrices.setflags(write=False)
        self.band_channels = np.zeros(self.nbnd, dtype=int)
        self.band_channels.setflags(write=False)
        self.capabilities = {
            "hamiltonian",
            "overlap",
            "derivatives",
            "position",
            "basis_overlap",
            "analytic_hamiltonian",
        }
        self.metadata = {
            "embedding": "full_position"
            if position is not None
            else ("centres" if centres is not None else "fixed_origin"),
            "position_link": "midpoint_first_order" if position is not None else "exact_diagonal",
            "energy_unit": "eV",
            "length_unit": "Angstrom",
        }
        self.source_files: dict[str, str] = {}
        self.fermi_energy = fermi_energy
        self.execution = execution or ExecutionOptions()
        self.path: Path | None = None
        self.poscar: Path | None = None
        self.hr: Path | None = None
        self.raw_data: WannierData | None = None
        self.atom_positions = np.empty((0, 3), float)
        self.interpolation = "standard"
        self.max_hermiticity_residual = 0.0

    @property
    def fermi_energy(self) -> float:
        return self._fermi_energy

    @fermi_energy.setter
    def fermi_energy(self, value):
        if not np.isfinite(value):
            raise ValueError("Fermi energy must be finite")
        self._fermi_energy = float(value)

    @classmethod
    def from_directory(
        cls,
        directory,
        hr="wannier90_hr.dat",
        poscar=None,
        ef=0.0,
        interpolation="standard",
        wsvec=None,
        *,
        tb=None,
        centres=None,
        wout=None,
    ):
        """Load HR and optional native position/centre/lattice sidecars.

        A companion ``_tb.dat`` supplies full position operators while HR remains
        authoritative for its Hamiltonian. TB-only directories are also supported.
        Sidecars are discovered by seed; explicit missing filenames are errors.
        """
        root = Path(directory).expanduser().resolve()
        source_inventory = resolve_wannier_files(
            root,
            hr=hr,
            tb=tb,
            centres=centres,
            wout=wout,
            poscar=poscar,
            wsvec=wsvec,
            interpolation=interpolation,
        )
        if hr is None:
            if tb is None:
                raise ValueError("Wannier loading requires hr or tb")
            hr = tb
        hr_path = root / hr
        seed = hr_path.name.removesuffix("_hr.dat").removesuffix("_tb.dat")
        tb_path = root / tb if tb is not None else root / f"{seed}_tb.dat"
        if tb is not None and not hr_path.exists():
            seed = tb_path.name.removesuffix("_tb.dat")
        if hr_path.name.endswith("_tb.dat"):
            tb_path = hr_path
        tb_data = read_tb(tb_path) if tb is not None or tb_path.exists() else None
        source_files = {}
        if hr_path.exists() and not hr_path.name.endswith("_tb.dat"):
            raw = read_hr(hr_path)
            source_files["hr"] = str(hr_path)
        elif tb_data is not None:
            raw = tb_data
        else:
            raw = read_hr(hr_path)  # contextual FileNotFoundError
        if tb_data is not None:
            source_files["tb"] = str(tb_path)
            lookup = {tuple(row): index for index, row in enumerate(tb_data.translations)}
            if raw.nbnd != tb_data.nbnd or set(lookup) != set(map(tuple, raw.translations)):
                raise ValueError("HR and TB translation/orbital data do not match")
            order = [lookup[tuple(row)] for row in raw.translations]
            if not np.array_equal(raw.degeneracies, tb_data.degeneracies[order]) or not np.allclose(
                raw.hopping, tb_data.hopping[order], atol=1e-6, rtol=1e-8
            ):
                raise ValueError("HR and TB Hamiltonians/degeneracies do not match")
            raw = WannierData(
                raw.translations,
                raw.hopping,
                raw.degeneracies,
                None if tb_data.position is None else tb_data.position[order],
                tb_data.lattice,
            )
        ws_path = root / wsvec if wsvec is not None else root / f"{seed}_wsvec.dat"
        mapping = None
        if interpolation in {"ws_distance", "auto"}:
            if not ws_path.exists():
                raise ValueError(
                    f"{ws_path}: missing interpolation metadata; explicitly select standard if appropriate"
                )
            mapping = read_wsvec(ws_path)
            source_files["wsvec"] = str(ws_path)
        if interpolation == "auto":
            assert mapping is not None
            interpolation = (
                "ndegen_applied"
                if mapping.ndegen_applied
                else ("ws_distance" if mapping.use_ws_distance else "standard")
            )
        translations, hopping, position = normalized_operators(raw, interpolation, mapping)
        poscar_path = None if poscar is None else root / poscar
        lattice = raw.lattice
        wout_path = root / wout if wout is not None else root / f"{seed}.wout"
        if wout is not None or (lattice is None and poscar_path is None and wout_path.exists()):
            wout_lattice = read_wout_lattice(wout_path)
            if lattice is not None and not np.allclose(lattice, wout_lattice, atol=1e-6, rtol=1e-8):
                raise ValueError("TB and WOUT lattice vectors do not match")
            lattice = wout_lattice
            source_files["wout"] = str(wout_path)
        if poscar_path is not None:
            poscar_lattice = read_poscar(poscar_path)[0]
            if lattice is not None and not np.allclose(lattice, poscar_lattice, atol=1e-6, rtol=1e-8):
                raise ValueError("Wannier and POSCAR lattice vectors do not match")
            lattice = poscar_lattice
            source_files["poscar"] = str(poscar_path)
        centres_path = root / centres if centres is not None else root / f"{seed}_centres.xyz"
        centre_values = None
        atoms: list[tuple[str, np.ndarray]] = []
        if centres is not None or centres_path.exists():
            centre_values, atoms = read_centres_xyz(centres_path)
            source_files["centres"] = str(centres_path)
        elif position is not None:
            zero = np.flatnonzero(np.all(translations == 0, axis=1))
            if len(zero):
                centre_values = np.diagonal(position[zero].sum(axis=0), axis1=0, axis2=1).real.T
        instance = cls(
            translations, hopping, lattice=lattice, centres=centre_values, position=position, fermi_energy=ef
        )
        instance.path = root
        instance.hr = Path(hr)
        instance.poscar = None if poscar is None else Path(poscar)
        instance.interpolation = interpolation
        instance.raw_data = raw
        instance.source_files = {key: str(path) for key, path in source_inventory.items()}
        instance.atom_positions = np.asarray([point for _, point in atoms], float).reshape(-1, 3)
        return instance

    def hamiltonian(self, k) -> np.ndarray:
        point = self._point(k)
        matrices, residual = _assemble(
            self.translations, self.hopping_matrices, validated_points(point[None, :])
        )
        self.max_hermiticity_residual = max(self.max_hermiticity_residual, residual)
        return matrices[0]

    @staticmethod
    def _point(k):
        raw = np.asarray(k)
        if np.iscomplexobj(raw) and np.any(raw.imag != 0):
            raise ValueError("real-k operators reject complex k; use analytic_hamiltonian")
        point = np.asarray(raw.real, float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("one k-point must have finite shape (3,)")
        return point

    @staticmethod
    def _points(kpoints):
        raw = np.asarray(kpoints.path if isinstance(kpoints, KPath) else kpoints)
        if np.iscomplexobj(raw) and np.any(raw.imag != 0):
            raise ValueError("real-k operators reject complex k; use analytic_hamiltonian")
        return validated_points(raw.real)

    def analytic_hamiltonian(self, k):
        """Raw entire Fourier sum at complex fractional k, without hermitization.

        This separate interface is for complex branch calculations. Applying an
        adjoint or the real-k validation here would destroy analytic continuation.
        It accepts a single (3,) point or a batch (Nk,3).
        """
        points = np.asarray(k, complex)
        single = points.ndim == 1
        if single:
            points = points[None]
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("analytic k must have finite shape (3,) or (Nk,3)")
        phase = np.exp(2j * np.pi * (points @ self.translations.T))
        matrices = np.einsum("kr,rij->kij", phase, self.hopping_matrices, optimize=True)
        if not np.isfinite(matrices).all():
            raise NumericalError("analytic Hamiltonian assembly produced nonfinite values")
        return matrices[0] if single else matrices

    def overlap(self, k):
        self._point(k)
        return np.eye(self.nbnd, dtype=complex)

    def position(self, k):
        """Wannier connection (Cartesian,NB,NB) in Å, with HR phases unchanged."""
        point = self._point(k)
        if self.position_matrices is not None:
            return np.stack(
                [
                    _assemble(self.translations, self.position_matrices[..., axis], point[None])[0][0]
                    for axis in range(3)
                ]
            )
        result = np.zeros((3, self.nbnd, self.nbnd), complex)
        if self.centres is not None:
            result[:, np.arange(self.nbnd), np.arange(self.nbnd)] = self.centres.T
        return result

    def derivatives(self, k, order=1):
        """Cartesian Fourier moments, following the native ``H,S,r`` schema."""
        if order not in (1, 2):
            raise ValueError("derivative order must be one or two")
        point = self._point(k)
        if self.lattice is None:
            raise ValueError("Cartesian derivatives require lattice/reciprocal vectors")
        vectors = self.translations @ self.lattice
        phase = np.exp(2j * np.pi * (self.translations @ point))
        if order == 1:
            h = np.einsum("r,ra,rij->aij", 1j * phase, vectors, self.hopping_matrices)
            r = (
                np.zeros((3, 3, self.nbnd, self.nbnd), complex)
                if self.position_matrices is None
                else np.einsum("r,ra,rijb->abij", 1j * phase, vectors, self.position_matrices)
            )
        else:
            h = -np.einsum("r,ra,rb,rij->abij", phase, vectors, vectors, self.hopping_matrices)
            r = (
                np.zeros((3, 3, 3, self.nbnd, self.nbnd), complex)
                if self.position_matrices is None
                else -np.einsum("r,ra,rb,rijc->abcij", phase, vectors, vectors, self.position_matrices)
            )
        if not np.isfinite(h).all() or not np.isfinite(r).all():
            raise NumericalError("Cartesian operator derivatives produced nonfinite values")
        return {"H": h, "S": np.zeros_like(h), "r": r}

    def basis_overlap(self, k, kprime):
        """Exact diagonal embedding links, or midpoint first-order full links.

        Full-position links need path refinement; higher raw position moments
        are absent from TB files and finite-distance fidelity is not claimed.
        """
        first, second = self._point(k), self._point(kprime)
        if self.position_matrices is None and self.centres is None:
            return np.eye(self.nbnd, dtype=complex)
        if self.g_vec is None:
            raise ValueError("position links require lattice/reciprocal vectors")
        delta = (second - first) @ self.g_vec
        if self.position_matrices is None:
            return np.diag(np.exp(-1j * (self.centres @ delta)))
        return np.eye(self.nbnd) - 1j * np.einsum("a,aij->ij", delta, self.position((first + second) / 2))

    def eigenstates(self, kpoints, bands=None, execution=None):
        """Absolute energies (Nb,Nk), orthonormal coefficients and overlap metrics."""
        points = self._points(kpoints)
        selection = self._band_indices(bands)
        energies = np.empty((len(selection), len(points)), float)
        coefficients = np.empty((len(points), self.nbnd, len(selection)), complex)
        overlaps = np.broadcast_to(
            np.eye(self.nbnd, dtype=complex), (len(points), self.nbnd, self.nbnd)
        ).copy()
        options = (execution or self.execution).resolved()
        size = options.bounded_batch_size(32 * len(self.translations) + 192 * self.nbnd**2)
        starts = range(0, len(points), size)

        def assign(start, result):
            values, vectors, residual = result
            end = start + values.shape[1]
            energies[:, start:end] = values[selection]
            coefficients[start:end] = vectors[:, :, selection]
            self.max_hermiticity_residual = max(self.max_hermiticity_residual, residual)

        if options.workers == 1:
            from threadpoolctl import threadpool_limits

            with threadpool_limits(limits=options.native_threads):
                for start in starts:
                    assign(
                        start,
                        _solve_state_batch(
                            self.translations, self.hopping_matrices, points[start : start + size]
                        ),
                    )
        else:
            from joblib import delayed

            with execution_session(options) as session:
                remaining = iter(starts)
                while window := tuple(islice(remaining, options.workers)):
                    results = session.parallel()(
                        delayed(_solve_state_batch)(
                            self.translations, self.hopping_matrices, points[start : start + size]
                        )
                        for start in window
                    )
                    for start, result in zip(window, results, strict=True):
                        assign(start, result)
        return SimpleNamespace(
            energies=energies,
            absolute_energies=energies,
            coefficients=coefficients,
            overlaps=overlaps,
            band_channels=self.band_channels[selection],
        )

    def energies(self, kpoints, bands=None, execution: ExecutionOptions | None = None) -> np.ndarray:
        """Return (number of selected bands,NK), including integer selections."""
        points = self._points(kpoints)
        selection = self._band_indices(bands)
        output = np.empty((len(selection), len(points)), dtype=np.float64)
        options = (execution or self.execution).resolved()
        if not len(points):
            return output
        # Phase, Hamiltonian, Hermitian copy/residual arrays, and solver workspace.
        bytes_per_point = 32 * len(self.translations) + 128 * self.nbnd**2 + 16 * self.nbnd
        batch_size = options.bounded_batch_size(bytes_per_point)
        starts = range(0, len(points), batch_size)
        from threadpoolctl import threadpool_limits

        if options.workers == 1:
            with threadpool_limits(limits=options.native_threads):
                for start in starts:
                    values, residual = _solve_batch(
                        self.translations, self.hopping_matrices, points[start : start + batch_size]
                    )
                    output[:, start : start + values.shape[1]] = values[selection] - self.fermi_energy
                    self.max_hermiticity_residual = max(self.max_hermiticity_residual, residual)
        else:
            from joblib import delayed

            with execution_session(options) as session:
                remaining_starts = iter(starts)
                # Ordered generators may buffer completed results while an earlier
                # batch runs. Bound each submission as well as its dispatch queue.
                while window := tuple(islice(remaining_starts, options.workers)):
                    results = session.parallel()(
                        delayed(_solve_batch)(
                            self.translations, self.hopping_matrices, points[start : start + batch_size]
                        )
                        for start in window
                    )
                    for start, (values, residual) in zip(window, results, strict=True):
                        output[:, start : start + values.shape[1]] = values[selection] - self.fermi_energy
                        self.max_hermiticity_residual = max(self.max_hermiticity_residual, residual)
        return output

    def _band_indices(self, bands) -> np.ndarray:
        if bands is None:
            return np.arange(self.nbnd)
        if isinstance(bands, bool):
            raise ValueError("bands must be integer indices, not a boolean")
        indices = np.asarray([bands] if isinstance(bands, (int, np.integer)) else bands)
        if indices.ndim != 1 or (indices.size and not np.issubdtype(indices.dtype, np.integer)):
            raise ValueError("bands must be an integer or a one-dimensional integer sequence")
        if np.any(indices < 0) or np.any(indices >= self.nbnd):
            raise ValueError(f"band indices must be in 0..{self.nbnd - 1}")
        return indices.astype(int)

    def calculate_energy(self, path, band_index=False):
        """Legacy shape adapter: an integer band selection returns (NK,)."""
        bands = None if band_index is False or band_index is None else band_index
        values = self.energies(path, bands=bands)
        return values[0] if isinstance(bands, (int, np.integer)) and not isinstance(bands, bool) else values

    def solver(self, k):
        return np.linalg.eigvalsh(self.hamiltonian(k))

    def fourier(self, k):
        return self.hamiltonian(k)

    def eig(self, kpoints):
        points = self._points(kpoints)
        solutions = [np.linalg.eigh(self.hamiltonian(point)) for point in points]
        return [value for value, _ in solutions], [vector for _, vector in solutions]

    def kpath(self, point_names, n_points=200, poscar=None, **kwargs):
        chosen = self.poscar if poscar is None else Path(poscar)
        if chosen is None:
            raise ValueError("named k-paths require a POSCAR")
        root = self.path or Path.cwd()
        return named_k_path(point_names, n_points, root / chosen, **kwargs)
