"""Wannier Hamiltonians with validated, bounded numerical evaluation."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from copy import copy
from itertools import islice
from pathlib import Path
from typing import Any

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
    read_win,
    read_wout_lattice,
    read_wsvec,
    resolve_wannier_files,
)
from ..physics.operators import assemble as _assemble
from ..physics.operators import hermitize_table, positive_translation, table_hermiticity
from ..physics.operators import validate_assembly as _validate_assembly
from ..physics.validation import NumericalError as NumericalError
from .contracts import EigenstateBatch

_LINE_MIN_GAIN = 2
_LINE_CACHE_BYTES = 128 * 1024**2
_POSITION_MODES = ("auto", "full", "diagonal")


def _optional_nonnegative(value, name, *, positive=False):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not np.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return number


def _hopping_cutoffs(translations, hopping, lattice, centres, min_hopping_norm, max_distance):
    """Hop mask with PythTB's W90 cutoff rules and the independent-hop counts.

    Each hop of the positive half and of the upper ``R = 0`` triangle is kept
    when its centre distance ``|R.A + tau_j - tau_i|`` is at most ``max_distance``
    and ``|t| >= min_hopping_norm``; its conjugate partner follows. On-site
    energies are never cut. A block without its partner is judged on its own.
    The mask is ``None`` when neither cutoff is set.
    """
    nbnd = hopping.shape[1]
    lookup = {tuple(int(value) for value in row): index for index, row in enumerate(translations)}
    counts = {"hops": 0, "dropped_by_distance": 0, "dropped_by_norm": 0}
    keep = None
    if min_hopping_norm is not None or max_distance is not None:
        keep = np.ones(hopping.shape, dtype=bool)
    upper = np.triu(np.ones((nbnd, nbnd), dtype=bool), k=1)
    every = np.ones((nbnd, nbnd), dtype=bool)
    for key, index in lookup.items():
        partner = lookup.get(tuple(-value for value in key))
        zero = not any(key)
        if partner is not None and not zero and not positive_translation(key):
            continue
        counts["hops"] += nbnd * (nbnd - 1) // 2 if zero else nbnd**2
        if keep is None:
            continue
        independent = upper if zero else every
        far = np.zeros((nbnd, nbnd), dtype=bool)
        if max_distance is not None:
            vectors = centres[None, :, :] - centres[:, None, :] + np.asarray(key) @ lattice
            far = independent & (np.sqrt(np.einsum("ija,ija->ij", vectors, vectors)) > max_distance)
        weak = np.zeros((nbnd, nbnd), dtype=bool)
        if min_hopping_norm is not None:
            weak = independent & ~far & (np.abs(hopping[index]) < min_hopping_norm)
        counts["dropped_by_distance"] += int(far.sum())
        counts["dropped_by_norm"] += int(weak.sum())
        block = ~(far | weak)
        if zero:
            block &= block.T
        keep[index] = block
        if partner is not None and partner != index:
            keep[partner] = block.T
    counts["kept_hops"] = counts["hops"] - counts["dropped_by_distance"] - counts["dropped_by_norm"]
    return keep, counts


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
        self.metadata: dict[str, Any] = {
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
        self._line_context = None
        self._line_groups = {}
        self._line_cache = OrderedDict()
        self._line_cache_bytes = 0
        self._line_cache_limit = None

    @contextmanager
    def on_line(self, direction, origin, *, cache_bytes=None):
        """Internal Fourier context for Cartesian unit direction and fractional origin."""
        direction, origin = np.asarray(direction, float), np.asarray(origin, float)
        if (
            direction.shape != (3,)
            or origin.shape != (3,)
            or not np.isfinite(direction).all()
            or not np.isfinite(origin).all()
            or not np.isclose(direction @ direction, 1.0, atol=1e-12, rtol=0)
        ):
            raise ValueError("line requires a Cartesian unit direction and finite fractional origin")
        previous = self._line_context
        previous_limit = self._line_cache_limit
        self._line_context = (tuple(direction), tuple(origin))
        if cache_bytes is not None:
            self._line_cache_limit = max(0, int(cache_bytes))
            while self._line_cache and self._line_cache_bytes > self._line_budget():
                _, removed = self._line_cache.popitem(last=False)
                self._line_cache_bytes -= removed.nbytes
        try:
            yield self
        finally:
            self._line_context = previous
            self._line_cache_limit = previous_limit

    def _line_budget(self):
        budget = _LINE_CACHE_BYTES
        if self._line_cache_limit is not None:
            budget = min(budget, self._line_cache_limit)
        if self.execution.memory_mb is not None:
            budget = min(budget, int(self.execution.memory_mb * 1024**2 * 0.2))
        return budget

    def light_worker_view(self):
        """Private calculation copy; joblib shares the immutable input arrays."""
        model = copy(self)
        model.raw_data = None
        model._line_context = None
        model._line_groups = dict(self._line_groups)
        model._line_cache = OrderedDict()
        model._line_cache_bytes = 0
        model._line_cache_limit = None
        options = self.execution.resolved()
        model.execution = ExecutionOptions(
            workers=1,
            batch_size=options.batch_size,
            native_threads=options.native_threads,
            memory_mb=None if options.memory_mb is None else options.memory_mb / options.workers,
        )
        return model

    def _line_projection_groups(self, direction):
        if direction not in self._line_groups:
            vectors = self.translations @ self.lattice
            projection = vectors @ np.asarray(direction)
            bins = np.rint(projection / 1e-6).astype(np.int64)
            _, inverse = np.unique(bins, return_inverse=True)
            members = [np.flatnonzero(inverse == index) for index in range(int(inverse.max()) + 1)]
            representative = np.array([projection[index].mean() for index in members])
            spread = max(
                np.max(np.abs(projection[index] - value)) for index, value in zip(members, representative)
            )
            self._line_groups[direction] = (
                (representative, members)
                if len(members) * _LINE_MIN_GAIN <= len(projection) and spread < 1e-8
                else None
            )
        return self._line_groups[direction]

    def _fourier_values(self, points, operators, key, moments=()):
        """Entire operator sum, optionally factorized on the active line."""
        moments = tuple(
            int(np.flatnonzero(axis)[0])
            if isinstance(axis, tuple) and np.count_nonzero(axis) == 1 and max(axis) == 1
            else axis
            for axis in moments
        )
        vectors = self.translations @ self.lattice if moments else None
        weights = np.ones(len(self.translations), complex)
        for axis in moments:
            weights *= 1j * (vectors @ np.asarray(axis) if isinstance(axis, tuple) else vectors[:, axis])
        if moments and not np.any(weights):
            return np.zeros((len(points),) + operators.shape[1:], complex)
        if self._line_context is not None and self.g_vec is not None:
            direction, origin = self._line_context
            displacement = (points - np.asarray(origin)) @ self.g_vec
            s = displacement @ np.asarray(direction)
            residual = displacement - s[:, None] * np.asarray(direction)
            tolerance = 1e-12 * max(1.0, float(np.max(np.abs(s), initial=0)))
            groups = self._line_projection_groups(direction)
            budget = self._line_budget()
            if groups is not None and len(groups[0]) * operators[0].size * 16 > budget:
                # Plain bounded Fourier batches need no complete coefficient
                # table when even one table exceeds the reserved cache budget.
                groups = None
            if groups is not None and np.max(np.abs(residual), initial=0) <= tolerance:
                projection, members = groups
                cache_key = (direction, origin, key, moments)
                terms = self._line_cache.get(cache_key)
                if terms is None:
                    phase = np.exp(2j * np.pi * (self.translations @ np.asarray(origin))) * weights
                    flat = operators.reshape(len(self.translations), -1)
                    terms = np.empty((len(members), flat.shape[1]), complex)
                    for index, rows in enumerate(members):
                        terms[index] = phase[rows] @ flat[rows]
                    if terms.nbytes <= budget:
                        while self._line_cache and self._line_cache_bytes + terms.nbytes > budget:
                            _, removed = self._line_cache.popitem(last=False)
                            self._line_cache_bytes -= removed.nbytes
                        self._line_cache[cache_key] = terms
                        self._line_cache_bytes += terms.nbytes
                else:
                    self._line_cache.move_to_end(cache_key)
                result = np.exp(1j * np.multiply.outer(s, projection)) @ terms
                return result.reshape((len(points),) + operators.shape[1:])
        phase = np.exp(2j * np.pi * (points @ self.translations.T)) * weights
        return np.einsum("kr,rij->kij", phase, operators, optimize=True)

    def _assemble_points(self, points, operators=None, key="H"):
        if operators is None:
            operators = self.hopping_matrices
        if self._line_context is None:
            return _assemble(self.translations, operators, points)
        return _validate_assembly(self._fourier_values(points, operators, key), points)

    def slab_z(self):
        """Make centres of a slab wrapped across the vacuum contiguous in z."""
        if self.centres is None:
            raise ValueError("slab_z needs the Wannier centres")
        z = self.centres[:, 2].astype(float)
        lattice = self.lattice
        if lattice is None or np.any(np.abs(lattice[:2, 2]) > 1e-9) or abs(lattice[2, 2]) < 1e-9:
            return z
        period = abs(lattice[2, 2])
        ordered = np.sort(np.mod(z, period))
        gaps = np.diff(np.r_[ordered, ordered[0] + period])
        gap = int(np.argmax(gaps))
        if np.ptp(z) <= period - gaps[gap] + 1e-9:
            return z
        low = ordered[(gap + 1) % len(ordered)]
        return np.mod(z - low, period) + low

    def layer_projector(self, z_split=None):
        z = self.slab_z()
        if z_split is None:
            if np.ptp(z) < 1e-6:
                raise ValueError("all Wannier centres share one z; pass z_split explicitly")
            z_split = 0.5 * (z.min() + z.max())
        return np.diag((z > z_split).astype(complex))

    @property
    def fermi_energy(self) -> float:
        return self._fermi_energy

    @fermi_energy.setter
    def fermi_energy(self, value):
        if not np.isfinite(value):
            raise ValueError("Fermi energy must be finite")
        self._fermi_energy = float(value)

    @classmethod
    def from_table(
        cls,
        translations,
        hopping,
        *,
        lattice,
        centres=None,
        position=None,
        fermi_energy=0.0,
        hermitize="none",
        imag_onsite_tol=1e-6,
        min_hopping_norm=None,
        max_distance=None,
        execution: ExecutionOptions | None = None,
    ):
        """Build a model from a normalized ``(R, H(R))`` table with an explicit Hermitian policy.

        ``hopping`` is (NR,NB,NB) eV with degeneracy and image weights applied,
        ``lattice`` holds Cartesian rows in Å, and ``centres`` are the Cartesian
        Wannier centres in Å as written, neither wrapped nor unwrapped. Without
        ``position`` the basis links are the exact diagonal embedding.

        ``hermitize="average"`` replaces each pair by ``(H(R) + H(-R)^dagger)/2``
        and its adjoint (``position`` likewise) and inserts the full adjoint of a
        block whose partner is missing; ``"none"`` keeps the table, which must then
        be Hermitian. Before any change, an on-site imaginary part above
        ``imag_onsite_tol`` eV is refused, and ``metadata`` records
        ``hermiticity_error`` (largest ``|H(-R) - H(R)^dagger|`` element, a missing
        partner counting fully) and ``hermiticity_weyl_bound`` (see
        :func:`queer.physics.operators.table_hermiticity`).

        ``min_hopping_norm`` (eV) and ``max_distance`` (Å) remove hops from the
        model with PythTB's W90 rules: on-site energies are kept, each hop of the
        positive half and of the upper ``R = 0`` triangle is judged by
        ``|t| >= min_hopping_norm`` and ``|R.A + tau_j - tau_i| <= max_distance``,
        and its conjugate follows. ``metadata["cutoffs"]`` counts the independent
        hops, those dropped by distance and then by norm, and those kept. The
        diagnostics describe the table before cutoffs. Cutoffs require
        ``position=None``, and ``max_distance`` requires ``centres``.
        """
        if hermitize not in {"none", "average"}:
            raise ValueError("hermitize must be none or average")
        tolerance = _optional_nonnegative(imag_onsite_tol, "imag_onsite_tol")
        norm_cutoff = _optional_nonnegative(min_hopping_norm, "min_hopping_norm")
        distance_cutoff = _optional_nonnegative(max_distance, "max_distance", positive=True)
        vectors = np.asarray(translations)
        matrices = np.asarray(hopping)
        if (
            vectors.ndim != 2
            or vectors.shape[1] != 3
            or not len(vectors)
            or (vectors.dtype.kind not in "iu" and not np.isfinite(vectors).all())
            or not np.equal(vectors, np.round(vectors)).all()
        ):
            raise ValueError("translations must be a nonempty (NR,3) integer array")
        vectors = vectors.astype(np.int64)
        if (
            matrices.ndim != 3
            or matrices.shape[0] != len(vectors)
            or matrices.shape[1] != matrices.shape[2]
            or matrices.shape[1] < 1
        ):
            raise ValueError("hopping must have shape (NR,NB,NB)")
        matrices = matrices.astype(np.complex128)
        if not np.isfinite(matrices).all():
            raise ValueError("hopping must be finite")
        if position is not None and (norm_cutoff is not None or distance_cutoff is not None):
            raise ValueError("hopping cutoffs apply to models without position matrices")
        cell = validated_lattice(lattice).copy()
        centre_values = None if centres is None else np.asarray(centres, float)
        if centre_values is not None and (
            centre_values.shape != (matrices.shape[1], 3) or not np.isfinite(centre_values).all()
        ):
            raise ValueError("centres must be a finite (NB,3) Cartesian Angstrom array")
        if distance_cutoff is not None and centre_values is None:
            raise ValueError("max_distance requires the Wannier centres")
        error, bound = table_hermiticity(vectors, matrices)
        home = np.flatnonzero(~vectors.any(axis=1))
        onsite_imaginary = float(np.abs(np.diagonal(matrices[home[0]]).imag).max()) if len(home) else 0.0
        if tolerance is not None and onsite_imaginary > tolerance:
            raise ValueError(
                f"on-site energies have imaginary parts up to {onsite_imaginary:.3g} eV, "
                f"above imag_onsite_tol={tolerance:g} eV"
            )
        operators = None if position is None else np.asarray(position, complex)
        if operators is not None and operators.shape != matrices.shape + (3,):
            raise ValueError("position must be a finite (NR,NB,NB,3) Angstrom array")
        added: list[tuple[int, ...]] = []
        if hermitize == "average":
            joined = matrices if operators is None else np.concatenate((matrices[..., None], operators), -1)
            vectors, joined, added = hermitize_table(vectors, joined)
            if operators is None:
                matrices = joined
            else:
                matrices, operators = joined[..., 0], joined[..., 1:]
        keep, counts = _hopping_cutoffs(vectors, matrices, cell, centre_values, norm_cutoff, distance_cutoff)
        if keep is not None:
            retained = keep.any(axis=(1, 2)) | ~vectors.any(axis=1)
            vectors, matrices = vectors[retained], np.where(keep, matrices, 0)[retained]
        model = cls(
            vectors,
            matrices,
            lattice=cell,
            centres=centre_values,
            position=operators,
            fermi_energy=fermi_energy,
            execution=execution,
        )
        model.metadata.update(
            {
                "hermitization": hermitize,
                "hermiticity_error": error,
                "hermiticity_weyl_bound": bound,
                "hermitization_added_partners": [list(key) for key in added],
                "onsite_imaginary_max": onsite_imaginary,
                "cutoffs": {
                    "min_hopping_norm": norm_cutoff,
                    "max_distance": distance_cutoff,
                    **counts,
                },
            }
        )
        return model

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
        position="auto",
        win=None,
    ):
        """Load HR and optional native position/centre/lattice sidecars.

        A companion ``_tb.dat`` supplies full position operators while HR remains
        authoritative for its Hamiltonian. TB-only directories are also supported.
        Sidecars are discovered by seed; explicit missing filenames are errors.
        ``position="auto"`` uses position matrices when a TB file supplies them,
        ``"full"`` requires them, and ``"diagonal"`` keeps the centres only: no
        companion ``_tb.dat`` is discovered, and a TB file that is the Hamiltonian
        source is read for its Hamiltonian alone. A named ``win`` file supplies
        the ``unit_cell_cart`` lattice, checked against a TB lattice; an explicit
        WOUT or POSCAR lattice must then agree with it.
        """
        if position not in _POSITION_MODES:
            raise ValueError("position must be auto, full or diagonal")
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
            position=position,
            win=win,
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
        if position == "diagonal":
            read_companion = hr_path.name.endswith("_tb.dat") or (tb is not None and not hr_path.exists())
        else:
            read_companion = tb is not None or tb_path.exists()
        tb_data = read_tb(tb_path) if read_companion else None
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
                None if tb_data.position is None or position == "diagonal" else tb_data.position[order],
                tb_data.lattice,
            )
        if position == "full" and raw.position is None:
            raise ValueError("position='full' requires a _tb.dat with position matrices")
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
        translations, hopping, position_matrices = normalized_operators(raw, interpolation, mapping)
        poscar_path = None if poscar is None else root / poscar
        lattice = raw.lattice
        if win is not None:
            win_path = root / win
            win_lattice = read_win(win_path).lattice
            if win_lattice is None:
                raise ValueError(f"{win_path}: no complete unit_cell_cart block")
            if lattice is not None and not np.allclose(lattice, win_lattice, atol=1e-6, rtol=1e-8):
                raise ValueError("TB and WIN lattice vectors do not match")
            lattice = validated_lattice(win_lattice).copy()
            source_files["win"] = str(win_path)
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
        elif position_matrices is not None:
            zero = np.flatnonzero(np.all(translations == 0, axis=1))
            if len(zero):
                centre_values = np.diagonal(position_matrices[zero].sum(axis=0), axis1=0, axis2=1).real.T
        instance = cls(
            translations,
            hopping,
            lattice=lattice,
            centres=centre_values,
            position=position_matrices,
            fermi_energy=ef,
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
        matrices, residual = self._assemble_points(validated_points(point[None, :]))
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
        if self.lattice is None:
            phase = np.exp(2j * np.pi * (points @ self.translations.T))
            matrices = np.einsum("kr,rij->kij", phase, self.hopping_matrices, optimize=True)
        else:
            matrices = self._fourier_values(points, self.hopping_matrices, "H")
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
                    self._assemble_points(point[None], self.position_matrices[..., axis], ("r", axis))[0][0]
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
        if self._line_context is not None:
            axes = [(a,) for a in range(3)] if order == 1 else [(a, b) for a in range(3) for b in range(3)]
            shape = (3,) * order + (self.nbnd, self.nbnd)
            h = np.stack(
                [
                    self._fourier_values(point[None], self.hopping_matrices, "H", moments)[0]
                    for moments in axes
                ]
            ).reshape(shape)
            r = np.zeros((3,) * (order + 1) + (self.nbnd, self.nbnd), complex)
            if self.position_matrices is not None:
                for moments in axes:
                    for position_axis in range(3):
                        r[moments + (position_axis,)] = self._fourier_values(
                            point[None],
                            self.position_matrices[..., position_axis],
                            ("r", position_axis),
                            moments,
                        )[0]
            if not np.isfinite(h).all() or not np.isfinite(r).all():
                raise NumericalError("Cartesian operator derivatives produced nonfinite values")
            return {"H": h, "S": np.zeros_like(h), "r": r}
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
                    selected_points = points[start : start + size]
                    if self._line_context is None:
                        result = _solve_state_batch(self.translations, self.hopping_matrices, selected_points)
                    else:
                        matrices, residual = self._assemble_points(selected_points)
                        values, vectors = np.linalg.eigh(matrices)
                        result = values.T, vectors, residual
                    assign(start, result)
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
        return EigenstateBatch(
            energies=energies,
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
