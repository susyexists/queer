"""Wannier Hamiltonians with validated, bounded numerical evaluation."""

from __future__ import annotations

from itertools import islice
from pathlib import Path

import numpy as np

from ..execution import ExecutionOptions, execution_session
from ..geometry.lattice import reciprocal_lattice, validated_lattice, validated_points
from ..geometry.paths import KPath, named_k_path
from ..io.structure import read_poscar
from ..io.wannier import normalized_hopping, read_hr, read_wsvec


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
        self.g_vec = None if reciprocal is None else validated_lattice(reciprocal).copy()
        self.fermi_energy = fermi_energy
        self.execution = execution or ExecutionOptions()
        self.path = None
        self.poscar = None
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
        cls, directory, hr="wannier90_hr.dat", poscar=None, ef=0.0, interpolation="standard", wsvec=None
    ):
        root = Path(directory).expanduser().resolve()
        hr_path = root / hr
        raw = read_hr(hr_path)
        ws_path = (
            root / wsvec
            if wsvec is not None
            else hr_path.with_name(hr_path.name.removesuffix("_hr.dat") + "_wsvec.dat")
        )
        mapping = None
        if interpolation in {"ws_distance", "auto"}:
            if not ws_path.exists():
                raise ValueError(
                    f"{ws_path}: missing interpolation metadata; explicitly select standard if appropriate"
                )
            mapping = read_wsvec(ws_path)
        if interpolation == "auto":
            interpolation = (
                "ndegen_applied"
                if mapping.ndegen_applied
                else ("ws_distance" if mapping.use_ws_distance else "standard")
            )
        translations, hopping = normalized_hopping(raw, interpolation, mapping)
        poscar_path = None if poscar is None else root / poscar
        reciprocal = None if poscar_path is None else reciprocal_lattice(read_poscar(poscar_path)[0])
        instance = cls(translations, hopping, reciprocal=reciprocal, fermi_energy=ef)
        instance.path = root
        instance.hr = Path(hr)
        instance.poscar = None if poscar is None else Path(poscar)
        instance.interpolation = interpolation
        instance.raw_data = raw
        return instance

    def hamiltonian(self, k) -> np.ndarray:
        point = np.asarray(k, dtype=float)
        if point.shape != (3,):
            raise ValueError("one Hamiltonian k-point must have shape (3,)")
        matrices, residual = _assemble(
            self.translations, self.hopping_matrices, validated_points(point[None, :])
        )
        self.max_hermiticity_residual = max(self.max_hermiticity_residual, residual)
        return matrices[0]

    def energies(self, kpoints, bands=None, execution: ExecutionOptions | None = None) -> np.ndarray:
        """Return (number of selected bands,NK), including integer selections."""
        points = validated_points(kpoints.path if isinstance(kpoints, KPath) else kpoints)
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
        points = validated_points(kpoints)
        solutions = [np.linalg.eigh(self.hamiltonian(point)) for point in points]
        return [value for value, _ in solutions], [vector for _, vector in solutions]

    def kpath(self, point_names, n_points=200, poscar=None, **kwargs):
        chosen = self.poscar if poscar is None else Path(poscar)
        if chosen is None:
            raise ValueError("named k-paths require a POSCAR")
        root = self.path or Path.cwd()
        return named_k_path(point_names, n_points, root / chosen, **kwargs)
