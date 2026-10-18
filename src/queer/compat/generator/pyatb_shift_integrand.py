"""An independent recomputation of a Generator shift current from its scfout: QUEER's physical spectrum and the
table PyATB 60fc6b2a prints.

The shift-current comparison (:mod:`~queer.compat.generator.parity_shift`) evaluates, on the run's own k grid
and photon energies, every one of the 18 components in two modes, each in two orbital bases:

- **physical**: the shift current as QUEER defines it, written here from its equations:
  ``sigma^abc(w) = -(pi e^2 / hbar) (g / (V N_k)) sum_k sum_{m occ, n empty} f_mn Im[r^b_nm r^c_mn;a]_(bc)
  delta(w - w_nm)`` with the exact SI 2019 ``e`` and ``hbar``, ``V = abs(det A)``, ``g = 2`` for a scfout
  without spin polarization (one channel occupied twice), the two channels summed for a collinear scfout and
  the spinor bands with spin-orbit coupling; the lowest ``occ_band`` levels occupied (per channel); the band
  pair alone when one is named; the scissors moving the transition energies only (the geometry from the
  unshifted levels); every transition above the degeneracy tolerance; the Gaussian of standard deviation
  ``eta / sqrt(2)``, ``exp(-(x / eta)^2) / (eta sqrt(pi))`` at every photon energy (``smearing_method 1``),
  or, without smearing, the histogram ``1[w - w_t in [-domega/2, domega/2)] / domega`` per eV and the weight
  in ``(w, w + domega]`` (QUEER's ``binned_uA_V2``). An exact degeneracy (levels within
  :func:`exactness_bound` of each other) is treated covariantly: parallel transport inside the degenerate
  levels (``D_a = -Sbar_a / 2 + i (rbar_a + rbar_a^dagger) / 2`` there, which normalizes them and makes their
  connection vanish) and their velocity matrix, so that a degenerate level's transitions sum to a value
  independent of the basis inside it;
- **PyATB-literal**: what PyATB prints, transcribed from its sources (``shift_current_solver.cpp``,
  ``linear_response.cpp``, ``xr_operation.cpp``, ``base_data.cpp``, ``openmx_read_scfout.py``; read only,
  written here anew): its Bloch sums of the upper triangles of the scfout's own records (Hermitian completion
  with a real diagonal; the position operator's lower triangle from ``(r + i dS)^dagger``), its Hartree
  constant 27.211324570273 eV, its ``method 1`` sum-over-states integrand (``get_Inm_ik_sumOver`` with
  ``get_D_degenerate``: ``1 / dE`` between levels more than 1e-4 eV apart, ``dE / (dE^2 + 0.04^2)`` within,
  ``i rbar_a(j, j)`` on the diagonal, and the band velocities of ``get_partial_eigenvalue_fromBar``, the
  eigenvalues of a band's block of levels within 1e-4 eV), the scissors added to the levels from ``occ_band``
  on before all of it, its window ``[start, start + count domega)`` without a band pair, its rows ``int((E -
  start) / domega)`` (C's truncation), its Gaussian on the rows within ``int(5 eta / domega)`` of that row or
  the bare weight in that row without smearing, ``(f_n - f_m) I = -I`` per pair, and its prefactor ``-t pi
  e^3 / (2 hbar^2 V N_k) 1e6`` with its ``t`` = 6.582119e-16 eV s, ``e`` = 1.602176565e-19 C, ``hbar`` =
  1.054571726e-34 J s, the signed cell volume and its spin factor (2 for ``nspin 1``, which reads the
  spin-up channel only, 1 for ``nspin 4``). ``method 0`` (``get_Inm_ik_direct``, every denominator of its
  eigenvector derivatives regularized) and the adaptive smearing are not transcribed.

Both modes share only the scfout reader (:class:`~queer.models.openmx.OpenMXModel`; the reader's records of
the scfout for PyATB's mode, its reconciled blocks for QUEER's) and numpy: no code of QUEER's runner or
observables runs. A reader defect would show in PyATB's mode, which a comparison checks against PyATB's own
table.

At one k-point, with ``H C = S C E`` (``C^dagger S C = 1``, solved here through the Cholesky factor of
``S``), the band-basis projections ``Xbar_a = C^dagger (d_a X) C`` and ``M_a(i, j) = (Hbar_a - E_j
Sbar_a)(i, j)``: ``D_a(i, j) = -M_a(i, j) g(E_i - E_j)``; ``r_mn^a = i D_a(m, n) + rbar_a(m, n)``,
``rbar_a = C^dagger r_a^dagger C``; ``d_b`` of a projection at ``(n, m)`` is ``(D_b^dagger Xbar_a)(n, m) +
Xbar_ab(n, m) + (Xbar_a D_b)(n, m)``; ``r_nm;b^a = i [(d_b Hbar_a - E_m d_b Sbar_a - Sbar_a V_b) d - (M_a V_b
- V_b M_a)] / d^2 + d_b rbar_a`` at ``(n, m)`` with ``d = E_m - E_n`` and ``V_b`` the velocities; ``I^{a,
bc} = Im(r_mn^b r_nm;a^c + r_mn^c r_nm;a^b)``. QUEER's weight of a transition is ``-I / 2``.

**Rounding, from the comparison's own computation.** Each mode is evaluated again in a fixed random orthogonal
change of the orbital basis (:func:`orthogonal_basis`), which reaches the same levels and integrand through
other floating-point operations. The envelope ``sum_k |T_k - T'_k|`` (each k-point's table in the two bases, its
difference taken before the sum over the k-points, so that no cancellation between k-points hides it) measures
how far rounding moves an entry; a table's rounding tolerance is :data:`SPREAD_MULTIPLE` times it, plus
:data:`MAGNITUDE_FLOOR` times the broadened magnitudes ``|r^b| |r^c_;a| + |r^c| |r^b_;a|`` of its transitions
times their conditioning ``1 + max|E| / s`` (``s`` the nearest other level; the size of what cancels), plus the
kernel's own rounding and the slope times the rounding of the transition energies (each :data:`KERNEL_ROUNDING`,
:data:`ENERGY_ROUNDING` relative), plus, for a transition within :data:`EDGE_EV` of a discontinuity of the kernel
(a bin edge, PyATB's rows and window ends), the difference of placing it either side (``edge``; ``edge_cumulative``
is what that choice moves in the sum over the rows up to each one, which an interior edge leaves unchanged beyond
its two rows). Where PyATB's own value depends on its eigensolver's basis inside degenerate levels, the two bases
differ by that dependence, and the tolerance says so. No number of the candidate's enters a tolerance.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .pyatb_files import SHIFT_COMPONENTS

# QUEER's SI constants: the exact SI 2019 values.
QUEER_E_CHARGE = 1.602176634e-19
QUEER_HBAR = 6.62607015e-34 / (2.0 * math.pi)
# PyATB 60fc6b2a's: get_shift_current_conductivity's eV_seconds, elem_charge_SI and hbar_SI, its reader's AU2EV,
# linear_response's degenerate_threshold and degenerate_eta.
PYATB_EV_SECONDS = 6.582119e-16
PYATB_E_CHARGE = 1.602176565e-19
PYATB_HBAR = 1.054571726e-34
PYATB_HARTREE = 27.211324570273
PYATB_THRESHOLD = 1e-4
PYATB_REGULARIZATION = 0.04
# The rounding tolerance of a table: this many times its two-basis envelope, plus this share of its broadened
# magnitudes, plus the edge placements of the transitions within EDGE_EV (eV) of a discontinuity of its kernel.
SPREAD_MULTIPLE = 100.0
MAGNITUDE_FLOOR = 1e-13
KERNEL_ROUNDING = 64 * float(np.finfo(float).eps)
ENERGY_ROUNDING = 64 * float(np.finfo(float).eps)
EDGE_EV = 1e-9
# The orthogonal change of the orbital basis (a fixed seed).
BASIS_SEED = 20261018
# Complex N x N matrices one k-point of a vectorized batch keeps alive, and the k-points of one batch.
MATRICES_PER_K = 160
VECTOR_BATCH = 32
# Kernel entries (photon energies x transitions) one product holds at once.
KERNEL_ENTRIES = 4_000_000
# Below this much work (k-points x bands^3) the recomputation runs in this process.
SERIAL_WORK = 2e7

_AXES = {"x": 0, "y": 1, "z": 2}
# PyATB's component order: a over x, y, z; (b, c) over xx, xy, xz, yy, yz, zz.
_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
_ORDER = tuple((a, b, c) for a in range(3) for b, c in _PAIRS)
assert tuple("".join("xyz"[i] for i in triple) for triple in _ORDER) == SHIFT_COMPONENTS
TREATMENTS = ("covariant", "pyatb")
CONSTRUCTIONS = ("hermitian", "pyatb")


# --------------------------------------------------------------------------- settings
@dataclass(frozen=True)
class ShiftRun:
    """A ``SHIFT_CURRENT`` run's settings as PyATB resolves them (its eV for the photon energies)."""

    occupied: int
    start: float
    domega: float
    count: int
    smearing: int
    eta: float
    grid: tuple[int, int, int]
    pair: tuple[int, int] | None = None
    scissors: float = 0.0

    @classmethod
    def from_block(cls, block: Mapping[str, Any]) -> ShiftRun:
        """From PyATB's resolved ``SHIFT_CURRENT`` block: ``count = int((end - start) / domega) + 1`` photon
        energies and the band pair ``(n_occ - 1, m_unocc - 1)`` when ``n_occ >= 1``, ``m_unocc >= 1`` and
        ``m_unocc > n_occ``."""
        start, end = (float(value) for value in block["omega"])
        domega = float(block["domega"])
        n_occ, m_unocc = int(block.get("n_occ", -1)), int(block.get("m_unocc", -1))
        pair = (n_occ - 1, m_unocc - 1) if n_occ >= 1 and m_unocc >= 1 and m_unocc > n_occ else None
        grid = tuple(int(value) for value in block["grid"])
        assert len(grid) == 3
        return cls(
            occupied=int(block["occ_band"]),
            start=start,
            domega=domega,
            count=int((end - start) / domega) + 1,
            smearing=int(block.get("smearing_method", 1)),
            eta=float(block.get("eta", 0.01)),
            grid=(grid[0], grid[1], grid[2]),
            pair=pair,
            scissors=float(block.get("scissors", 0.0) or 0.0),
        )

    @property
    def omega(self) -> np.ndarray:
        """PyATB's photon energies ``start + i * domega``."""
        return self.start + np.arange(self.count) * self.domega

    @property
    def end(self) -> float:
        """The upper end of PyATB's window, ``start + count * domega`` (excluded)."""
        return self.start + self.count * self.domega

    @property
    def interval(self) -> int:
        """The rows on either side of a transition's row that PyATB's Gaussian reaches."""
        return int(5 * self.eta / self.domega)

    @property
    def kpoints(self) -> int:
        return int(np.prod(self.grid))


def kpoint_grid(grid: Sequence[int]) -> np.ndarray:
    """PyATB's Gamma-origin grid ``(i/N1, j/N2, l/N3)`` in C order (``mp_generator``)."""
    axes = [np.arange(int(size)) / int(size) for size in grid]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


# --------------------------------------------------------------------------- blocks
@dataclass(frozen=True)
class Blocks:
    """One channel's real-space operators as dense stacks over one list of translations (QUEER's eV and
    Angstrom; ``position`` None without position records)."""

    translations: np.ndarray
    cartesian: np.ndarray
    hamiltonian: np.ndarray
    overlap: np.ndarray
    position: np.ndarray | None

    @property
    def size(self) -> int:
        return int(self.hamiltonian.shape[-1])


@dataclass(frozen=True)
class Channel:
    """A channel's blocks: the reader's reconciled model blocks (QUEER's physical value) and the scfout's own
    records, whose upper triangles PyATB reads."""

    reconciled: Blocks
    recorded: Blocks


def _dense(block: Any) -> np.ndarray:
    return block.toarray() if hasattr(block, "toarray") else np.asarray(block)


def _stack(sources: Sequence[Mapping[Any, Any]], keys: Sequence[Any], rows: slice, size: int) -> np.ndarray:
    out = np.zeros((len(sources), len(keys), size, size), complex)
    for index, blocks in enumerate(sources):
        for position, key in enumerate(keys):
            if key in blocks:
                out[index, position] = _dense(blocks[key])[rows, rows]
    return out


def _blocks(
    lattice: np.ndarray,
    hamiltonian: Mapping[Any, Any],
    overlap: Mapping[Any, Any],
    position: Sequence[Mapping[Any, Any]],
    rows: slice,
    size: int,
) -> Blocks:
    keys = sorted(
        {tuple(int(v) for v in key) for source in (hamiltonian, overlap, *position) for key in source}
    )
    translations = np.array(keys, dtype=int).reshape(-1, 3)
    stacks = _stack([hamiltonian, overlap, *position], keys, rows, size)
    has_position = any(len(axis) for axis in position)
    return Blocks(
        translations=translations,
        cartesian=translations @ np.asarray(lattice, float),
        hamiltonian=stacks[0],
        overlap=stacks[1],
        position=stacks[2:5] if has_position and len(position) == 3 else None,
    )


def channels(model: Any) -> list[Channel]:
    """The channels a recomputation evaluates: the spinor bands with spin-orbit coupling, the one channel of a
    scfout without spin polarization, a collinear scfout's spin-up then spin-down channel (PyATB reads the
    first). Each holds the reader's model blocks and the scfout's records (``model.data``; the model blocks
    when the model has no records)."""
    size = int(model.nbnd)
    mode = model.spin_mode
    half = int(getattr(model, "n_spatial", size)) if mode == "collinear" else size
    count = 2 if mode == "collinear" else 1
    lattice = np.asarray(model.lattice, float)
    position = tuple(getattr(model, "position_blocks", ()) or ())
    data = getattr(model, "data", None)
    out = []
    for channel in range(count):
        rows = slice(channel * half, (channel + 1) * half)
        reconciled = _blocks(
            lattice, model.hamiltonian_blocks, model.overlap_blocks, position or ({}, {}, {}), rows, half
        )
        recorded = reconciled
        if data is not None and getattr(data, "hamiltonian", None):
            raw_position = tuple(getattr(data, "position", ()) or ({}, {}, {}))
            recorded = _blocks(lattice, data.hamiltonian, data.overlap, raw_position, rows, half)
        out.append(Channel(reconciled, recorded))
    return out


def exactness_bound(model: Any) -> float:
    """``max(1e-9 eV, 10 x)``, ``x`` the largest ``|H(R) - H(-R)^dagger|`` (rows' absolute sums) of the scfout's own
    records: the spacing below which two levels are one degenerate level of the data (QUEER's degeneracy
    tolerance, measured here)."""
    data = getattr(model, "data", None)
    blocks = getattr(data, "hamiltonian", None) if data is not None else None
    if not blocks:
        return 1e-9
    largest = 0.0
    for key, block in blocks.items():
        other = blocks.get(tuple(-value for value in key))
        dense = _dense(block)
        partner = np.zeros_like(dense) if other is None else _dense(other).conj().T
        largest = max(largest, float(np.max(np.abs(dense - partner).sum(axis=1), initial=0.0)))
    return max(1e-9, 10.0 * largest)


def orthogonal_basis(size: int, seed: int = BASIS_SEED) -> np.ndarray:
    """A fixed random orthogonal matrix: another orbital basis for the same evaluation."""
    matrix = np.random.default_rng(seed + size).standard_normal((size, size))
    basis, _ = np.linalg.qr(matrix)
    return basis


def reader_hartree(model: Any) -> float:
    """The reader's Hartree in eV (QUEER's eV per Hartree of the scfout's records)."""
    value = (getattr(model, "metadata", None) or {}).get("hartree_to_ev")
    if value:
        return float(value)
    from scipy.constants import physical_constants

    return float(physical_constants["Hartree energy in eV"][0])


# --------------------------------------------------------------------------- Bloch sums
@dataclass(frozen=True)
class Operators:
    """A batch of k-points' operators: ``h``, ``s`` ``(K, N, N)``; ``dh``, ``ds`` ``(K, 3, N, N)``; ``d2h``,
    ``d2s`` ``(K, 3, 3, N, N)``; the position operator ``r`` ``(K, 3, N, N)`` and ``dr[:, b, a] = d_b r_a``."""

    h: np.ndarray
    s: np.ndarray
    dh: np.ndarray
    ds: np.ndarray
    d2h: np.ndarray
    d2s: np.ndarray
    r: np.ndarray
    dr: np.ndarray


_SECOND = tuple((a, b) for a in range(3) for b in range(a, 3))


def _adjoint(matrix: np.ndarray) -> np.ndarray:
    return np.swapaxes(matrix.conj(), -1, -2)


def _hermitian(matrix: np.ndarray) -> np.ndarray:
    return (matrix + _adjoint(matrix)) / 2


def _completed(upper: np.ndarray) -> np.ndarray:
    """Eigen's ``selfadjointView<Upper>()``: the upper triangle, its conjugate below and a real diagonal."""
    size = upper.shape[-1]
    strict = np.triu(np.ones((size, size), bool), 1)
    out = np.where(strict, upper, _adjoint(upper) * strict.T)
    index = np.arange(size)
    out[..., index, index] = upper[..., index, index].real
    return out


def _lower_from(upper: np.ndarray, partner: np.ndarray) -> np.ndarray:
    """PyATB's ``rk.triangularView<Lower>() = (rk + i dS)^dagger``: the strict upper triangle kept, the lower
    triangle and the diagonal from the adjoint of ``upper + i partner``."""
    size = upper.shape[-1]
    strict = np.triu(np.ones((size, size), bool), 1)
    return np.where(strict, upper, _adjoint(upper + 1j * partner))


def bloch_operators(blocks: Blocks, kpoints: np.ndarray, *, construction: str = "hermitian") -> Operators:
    """The operators at ``kpoints`` (fractional, ``(K, 3)``): ``X(k) = sum_R exp(2 pi i k.R) X(R)``, ``d_a X =
    sum_R i R_a ...``, ``d_a d_b X = -sum_R R_a R_b ...`` (``R_a`` Cartesian, Angstrom).

    ``construction="hermitian"`` (QUEER's): the Hermitian parts of ``H``, ``S`` and their derivatives and the plain
    sums of the position blocks. ``"pyatb"`` (PyATB's ``get_Xk_triu``): every operator summed from the upper
    triangles of the blocks, ``H``, ``S`` and their derivatives completed Hermitian with a real diagonal and the
    position operator ``r_a`` and ``d_b r_a`` completed below from ``(r_a + i d_a S)^dagger`` and ``(d_b r_a + i
    d_a d_b S)^dagger``. A nonorthogonal model needs its position blocks.
    """
    if construction not in CONSTRUCTIONS:
        raise ValueError(f"construction must be one of {CONSTRUCTIONS}")
    points = np.atleast_2d(np.asarray(kpoints, float))
    size = blocks.size
    phase = np.exp(2j * np.pi * points @ blocks.translations.T)
    cartesian = blocks.cartesian
    weights = [np.ones(len(cartesian))] + [1j * cartesian[:, a] for a in range(3)]
    weights += [-cartesian[:, a] * cartesian[:, b] for a, b in _SECOND]
    stack = np.stack(weights, axis=0)
    upper = np.triu(np.ones((size, size)))

    def sums(source: np.ndarray, extra: Sequence[np.ndarray] = ()) -> np.ndarray:
        flat = source.reshape(len(cartesian), size * size)
        if construction == "pyatb":
            flat = (source * upper).reshape(len(cartesian), size * size)
        factors = np.concatenate([stack, *extra], axis=0) if extra else stack
        return np.einsum("kr,wr,rn->kwn", phase, factors, flat, optimize=True).reshape(
            len(points), -1, size, size
        )

    def split(values: np.ndarray, hermitian: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fix = (_hermitian if construction == "hermitian" else _completed) if hermitian else (lambda x: x)
        value = fix(values[:, 0])
        first = fix(values[:, 1:4])
        second = np.empty((len(points), 3, 3, size, size), complex)
        for index, (a, b) in enumerate(_SECOND):
            second[:, a, b] = second[:, b, a] = fix(values[:, 4 + index])
        return value, first, second

    h, dh, d2h = split(sums(blocks.hamiltonian), True)
    s, ds, d2s = split(sums(blocks.overlap), True)
    position = np.zeros((len(points), 3, size, size), complex)
    d_position = np.zeros((len(points), 3, 3, size, size), complex)
    if blocks.position is not None:
        derivative = [1j * cartesian[:, b] for b in range(3)]
        for axis in range(3):
            flat = blocks.position[axis]
            if construction == "pyatb":
                flat = flat * upper
            values = np.einsum(
                "kr,wr,rn->kwn",
                phase,
                np.stack([np.ones(len(cartesian)), *derivative]),
                flat.reshape(len(cartesian), size * size),
                optimize=True,
            ).reshape(len(points), 4, size, size)
            position[:, axis] = values[:, 0]
            d_position[:, :, axis] = values[:, 1:4]
        if construction == "pyatb":
            for axis in range(3):
                position[:, axis] = _lower_from(position[:, axis], ds[:, axis])
                for b in range(3):
                    d_position[:, b, axis] = _lower_from(d_position[:, b, axis], d2s[:, axis, b])
    elif not np.allclose(s, np.eye(size)) or np.abs(ds).max(initial=0.0) > 0:
        raise ValueError("a nonorthogonal model needs its position blocks for the shift-current integrand")
    return Operators(h, s, dh, ds, d2h, d2s, position, d_position)


def scaled(operators: Operators, factor: float) -> Operators:
    """The Hamiltonian and its derivatives times ``factor`` (another energy unit)."""
    if factor == 1.0:
        return operators
    return Operators(
        operators.h * factor,
        operators.s,
        operators.dh * factor,
        operators.ds,
        operators.d2h * factor,
        operators.d2s,
        operators.r,
        operators.dr,
    )


def rotated(operators: Operators, basis: np.ndarray) -> Operators:
    """Every operator in another orbital basis, ``Q^dagger X Q`` for a unitary ``Q``: the same levels and integrand,
    reached through other floating-point operations."""
    left = np.asarray(basis).conj().T
    return Operators(*(left @ getattr(operators, name) @ basis for name in Operators.__dataclass_fields__))


def eigensystem(hamiltonian: np.ndarray, overlap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``H C = S C E`` with ``C^dagger S C = 1`` for a batch: ``S = L L^dagger``, the Hermitian ``L^-1 H L^-dagger``
    diagonalized, ``C = L^-dagger V``. Energies ascending ``(K, N)``, eigenvectors as columns ``(K, N, N)``."""
    factor = np.linalg.cholesky(overlap)
    inverse = np.linalg.inv(factor)
    reduced = _hermitian(inverse @ hamiltonian @ _adjoint(inverse))
    energies, vectors = np.linalg.eigh(reduced)
    return energies, _adjoint(inverse) @ vectors


def cluster_labels(energies: np.ndarray, threshold: float) -> np.ndarray:
    """The degenerate level of each level (ascending energies along the last axis): consecutive levels closer
    than ``threshold`` join one."""
    steps = np.abs(np.diff(energies, axis=-1)) > threshold
    zeros = np.zeros(energies.shape[:-1] + (1,), int)
    return np.concatenate([zeros, np.cumsum(steps, axis=-1)], axis=-1)


def pyatb_velocities(
    hbar: np.ndarray, sbar: np.ndarray, energies: np.ndarray, threshold: float
) -> np.ndarray:
    """PyATB's band velocities ``dE_b(j)`` ``(K, 3, N)`` (``get_partial_eigenvalue_fromBar``): ``Re(Hbar_b - E_j
    Sbar_b)(j, j)``, or, when levels lie within ``threshold`` of ``E_j`` (scanned outwards from ``j``), the
    eigenvalues (lower triangle) of that block of ``Hbar_b - E_j Sbar_b`` in ascending order at ``j``'s offset."""
    count, size = energies.shape
    index = np.arange(size)
    diagonal = (hbar[..., index, index] - energies[:, None, :] * sbar[..., index, index]).real  # (K, 3, N)
    close = np.abs(energies[:, :, None] - energies[:, None, :]) < threshold
    low = np.broadcast_to(index, (count, size)).copy()
    high = low.copy()
    for direction, bound in ((-1, low), (1, high)):
        active = np.ones((count, size), bool)
        for step in range(1, size):
            candidate = index + direction * step
            inside = (candidate >= 0) & (candidate < size)
            safe = np.clip(candidate, 0, size - 1)
            ok = active & inside[None, :] & np.take_along_axis(close, safe[None, :, None], axis=2)[..., 0]
            bound[ok] = np.broadcast_to(candidate, (count, size))[ok]
            active = ok
            if not active.any():
                break
    span = high - low + 1
    for width in np.unique(span[span > 1]):
        kk, jj = np.nonzero(span == width)
        rows = low[kk, jj][:, None] + np.arange(width)[None, :]
        for axis in range(3):
            block = (
                hbar[kk[:, None, None], axis, rows[:, :, None], rows[:, None, :]]
                - energies[kk, jj][:, None, None]
                * sbar[kk[:, None, None], axis, rows[:, :, None], rows[:, None, :]]
            )
            values = np.linalg.eigvalsh(block, UPLO="L")
            diagonal[kk, axis, jj] = values[np.arange(len(kk)), jj - low[kk, jj]]
    return diagonal


# --------------------------------------------------------------------------- the integrand
def integrand(
    operators: Operators,
    energies: np.ndarray,
    vectors: np.ndarray,
    occupied: int,
    *,
    treatment: str,
    threshold: float,
    regularization: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """PyATB's ``I`` ``(K, O, E, 18)`` of every occupied (``m < occupied``) to empty (``n``) pair and its magnitude
    ``|r^b_mn| |r^c_nm;a| + |r^c_mn| |r^b_nm;a|`` (the module docstring's formulas; Angstrom^3).

    ``energies`` are the levels the derivative kernels use (PyATB's scissored levels, QUEER's unshifted ones).
    ``treatment="pyatb"``: PyATB's ``1 / dE`` beyond ``threshold``, ``dE / (dE^2 + regularization^2)`` within (0
    for an exact degeneracy), ``i rbar(j, j)`` on the diagonal and :func:`pyatb_velocities`. ``"covariant"``: levels
    closer than ``threshold`` form degenerate levels, parallel transport and the velocity matrix inside them.
    """
    if treatment not in TREATMENTS:
        raise ValueError(f"treatment must be one of {TREATMENTS}")
    count, size = energies.shape
    occ, emp = slice(0, occupied), slice(occupied, size)
    width = size - occupied
    adjoint = _adjoint(vectors)
    hbar = adjoint[:, None] @ operators.dh @ vectors[:, None]
    sbar = adjoint[:, None] @ operators.ds @ vectors[:, None]
    rbar = adjoint[:, None] @ _adjoint(operators.r) @ vectors[:, None]
    first = hbar - sbar * energies[:, None, None, :]
    gap = energies[:, :, None] - energies[:, None, :]
    near = np.abs(gap) <= threshold
    safe = np.where(near, 1.0, gap)
    index = np.arange(size)
    velocity: np.ndarray | None = None
    if treatment == "pyatb":
        eta = 0.0 if regularization is None else float(regularization)
        within = gap / (gap**2 + eta**2) if eta else np.zeros_like(gap)
        inverse = np.where(near, within, 1.0 / safe)
        derivative = -first * inverse[:, None]
        derivative[..., index, index] = 1j * rbar[..., index, index]
        diagonal = pyatb_velocities(hbar, sbar, energies, threshold)
    else:
        labels = cluster_labels(energies, threshold)
        same = (labels[:, :, None] == labels[:, None, :])[:, None]
        inverse = np.where(near, 0.0, 1.0 / safe)
        transport = -0.5 * sbar + 0.5j * (rbar + _adjoint(rbar))
        derivative = np.where(same, transport, -first * inverse[:, None])
        pair_energy = 0.5 * (energies[:, :, None] + energies[:, None, :])[:, None]
        velocity = _hermitian(np.where(same, hbar - sbar * pair_energy, 0.0))
    dipole = 1j * derivative[..., occ, emp] + rbar[..., occ, emp]
    vectors_e, vectors_o = vectors[..., emp], vectors[..., occ]
    left_e = _adjoint(vectors_e)
    second_h = left_e[:, None, None] @ operators.d2h @ vectors_o[:, None, None]
    second_s = left_e[:, None, None] @ operators.d2s @ vectors_o[:, None, None]
    # (d_b r_a)^dagger projected, at [a, b].
    second_r = left_e[:, None, None] @ np.swapaxes(_adjoint(operators.dr), 1, 2) @ vectors_o[:, None, None]
    difference = energies[:, None, occ] - energies[:, emp, None]  # (K, E, O): E_m - E_n
    energy_m = energies[:, None, occ]
    generalized = np.empty((count, 3, 3, width, occupied), complex)
    for b in range(3):
        d_b = derivative[:, b]
        left = _adjoint(d_b)[:, emp, :]
        right = d_b[:, :, occ]
        for a in range(3):
            d_h = left @ hbar[:, a, :, occ] + second_h[:, a, b] + hbar[:, a, emp, :] @ right
            d_s = left @ sbar[:, a, :, occ] + second_s[:, a, b] + sbar[:, a, emp, :] @ right
            d_r = left @ rbar[:, a, :, occ] + second_r[:, a, b] + rbar[:, a, emp, :] @ right
            if velocity is None:
                v_b = diagonal[:, b]
                overlap_v = sbar[:, a, emp, occ] * v_b[:, None, occ]
                commutator = first[:, a, emp, occ] * (v_b[:, None, occ] - v_b[:, emp, None])
            else:
                v_full = velocity[:, b]
                overlap_v = sbar[:, a, emp, :] @ v_full[:, :, occ]
                commutator = first[:, a, emp, :] @ v_full[:, :, occ] - v_full[:, emp, :] @ first[:, a, :, occ]
            numerator = (d_h - energy_m * d_s - overlap_v) * difference - commutator
            generalized[:, a, b] = 1j * numerator / difference**2 + d_r
    # PyATB's pairs (m, n): r_mn (K, 3, O, E) and r_nm;a^c at (n, m), transposed to (m, n).
    general = np.swapaxes(generalized, -1, -2)
    values = np.empty((count, occupied, width, len(SHIFT_COMPONENTS)))
    magnitude = np.empty_like(values)
    for column, (a, b, c) in enumerate(_ORDER):
        values[..., column] = np.imag(dipole[:, b] * general[:, c, a] + dipole[:, c] * general[:, b, a])
        magnitude[..., column] = np.abs(dipole[:, b]) * np.abs(general[:, c, a]) + np.abs(
            dipole[:, c]
        ) * np.abs(general[:, b, a])
    return values, magnitude


def point_integrand(
    blocks: Blocks,
    k: Sequence[float] | np.ndarray,
    occupied: int,
    *,
    treatment: str,
    threshold: float,
    regularization: float | None = None,
    energy_scale: float = 1.0,
    construction: str = "hermitian",
    scissors: float = 0.0,
    scissored_kernels: bool = False,
    basis: np.ndarray | None = None,
) -> dict[str, Any]:
    """One k-point's levels (``energy_scale`` times QUEER's eV, unshifted), its transitions' energies (scissored)
    and ``I`` ``(O, E, 18)`` with magnitudes; ``scissored_kernels`` feeds the scissored levels to the derivative
    kernels (PyATB's)."""
    operators = scaled(
        bloch_operators(blocks, np.asarray(k, float)[None], construction=construction), energy_scale
    )
    if basis is not None:
        operators = rotated(operators, basis)
    levels, vectors = eigensystem(operators.h, operators.s)
    shifted = levels.copy()
    shifted[:, occupied:] += scissors
    values, magnitude = integrand(
        operators,
        shifted if scissored_kernels else levels,
        vectors,
        occupied,
        treatment=treatment,
        threshold=threshold,
        regularization=regularization,
    )
    transitions = shifted[0, None, occupied:] - shifted[0, :occupied, None]
    return {"levels": levels[0], "energies": transitions, "integrand": values[0], "magnitude": magnitude[0]}


# --------------------------------------------------------------------------- kernels
@dataclass(frozen=True)
class Kernel:
    """One table's kernel at the photon energies, ``(count, T)`` per call on transition energies: its value,
    with a Gaussian the size of its slope ``|dK/dE|`` and the scale of its own rounding ``K (1 + ((w - E) /
    eta)^2)`` (the exponent's argument rounds to that relative size), and which transitions lie within
    :data:`EDGE_EV` of a discontinuity."""

    value: Callable[[np.ndarray], np.ndarray]
    slope: Callable[[np.ndarray], np.ndarray] | None
    rounding: Callable[[np.ndarray], np.ndarray] | None
    near_edge: Callable[[np.ndarray], np.ndarray]
    # A Gaussian's ``(w - E) / eta`` at the photon energies, from which its slope and rounding follow its value
    # without another exponential: ``K 2 |x| / eta`` and ``K (1 + x^2)``.
    offset: Callable[[np.ndarray], np.ndarray] | None = None


class Kernels:
    """The kernels of a run (``omega``: PyATB's photon energies ``start + i domega``)."""

    def __init__(self, run: ShiftRun):
        self.run = run
        self.omega = run.omega

    def _gaussian(self, energies: np.ndarray, part: str, rows: np.ndarray | None = None) -> np.ndarray:
        eta = self.run.eta
        offset = (self.omega[:, None] if rows is None else rows) - energies[None, :]
        value = np.exp(-((offset / eta) ** 2)) / (eta * math.sqrt(math.pi))
        if part == "slope":
            return value * 2 * np.abs(offset) / eta**2
        if part == "rounding":
            return value * (1 + (offset / eta) ** 2)
        return value

    def queer(self, *, binned: bool = False) -> Kernel:
        """QUEER's kernel of transitions (QUEER's eV): its Gaussian ``exp(-(x/eta)^2)/(eta sqrt(pi))`` at every
        photon energy (``smearing_method 1``), or its histogram ``1[w - E in [-domega/2, domega/2)] / domega`` per eV,
        and with ``binned`` the weight in ``(w, w + domega]`` (that histogram half a bin higher, times domega)."""
        run = self.run
        if run.smearing == 1:
            return Kernel(
                functools.partial(self._gaussian, part="value"),
                functools.partial(self._gaussian, part="slope"),
                functools.partial(self._gaussian, part="rounding"),
                lambda energies: np.zeros(len(energies), bool),
                lambda energies: (self.omega[:, None] - energies[None, :]) / run.eta,
            )
        width = run.domega
        centre = self.omega + 0.5 * width if binned else self.omega
        scale = 1.0 if binned else 1.0 / width

        def value(energies: np.ndarray) -> np.ndarray:
            difference = centre[:, None] - energies[None, :]
            lines = (difference >= -width / 2) & (difference < width / 2)
            return lines.astype(float) if binned else lines / width

        def near(energies: np.ndarray) -> np.ndarray:
            position = (energies - run.start) / width - (0.0 if binned else 0.5)
            return np.abs(position - np.round(position)) * width < EDGE_EV

        del scale
        return Kernel(value, None, None, near)

    def pyatb(self, *, window: bool = True) -> Kernel:
        """PyATB's kernel of transitions (PyATB's eV): row ``int((E - start)/domega)``; its Gaussian on the rows
        within ``int(5 eta/domega)`` of it, or its bare weight there without smearing; without a band pair
        (``window``) only the transitions in ``[start, start + count domega)``."""
        run = self.run
        rows = np.arange(run.count)[:, None]

        def mask(energies: np.ndarray) -> np.ndarray:
            row = np.trunc((energies - run.start) / run.domega).astype(np.int64)[None, :]
            keep = np.abs(rows - row) <= run.interval if run.smearing == 1 else rows == row
            if window and run.pair is None:
                keep = keep & ((energies >= run.start) & (energies < run.end))[None, :]
            return keep

        def near(energies: np.ndarray) -> np.ndarray:
            position = (energies - run.start) / run.domega
            return np.abs(position - np.round(position)) * run.domega < EDGE_EV

        if run.smearing == 0:
            return Kernel(lambda energies: mask(energies).astype(float), None, None, near)
        photon = run.start + rows * run.domega

        def part(name: str) -> Callable[[np.ndarray], np.ndarray]:
            def apply(energies: np.ndarray) -> np.ndarray:
                return np.where(mask(energies), self._gaussian(energies, name, photon), 0.0)

            return apply

        return Kernel(
            part("value"),
            part("slope"),
            part("rounding"),
            near,
            lambda energies: (photon - energies[None, :]) / run.eta,
        )


def windowed(kernel: Kernel, run: ShiftRun) -> Kernel:
    """``kernel`` with PyATB's window rule on the transition energies (none with a band pair)."""
    if run.pair is not None:
        return kernel

    def inside(energies: np.ndarray) -> np.ndarray:
        return ((energies >= run.start) & (energies < run.end))[None, :]

    def wrap(
        function: Callable[[np.ndarray], np.ndarray] | None,
    ) -> Callable[[np.ndarray], np.ndarray] | None:
        if function is None:
            return None
        return lambda energies: function(energies) * inside(energies)

    value = wrap(kernel.value)
    assert value is not None
    return Kernel(value, wrap(kernel.slope), wrap(kernel.rounding), kernel.near_edge)


def _apply(
    kernel: Callable[[np.ndarray], np.ndarray], energies: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """``sum_t K(w, E_t) w_t`` ``(count, 18)``, the kernel built in slices of at most :data:`KERNEL_ENTRIES`."""
    energies = np.asarray(energies, float).reshape(-1)
    weights = np.asarray(weights, float).reshape(len(energies), -1)
    out: np.ndarray | None = None
    probe = kernel(energies[:1])
    step = max(1, KERNEL_ENTRIES // max(1, probe.shape[0]))
    for start in range(0, len(energies), step):
        part = kernel(energies[start : start + step]) @ weights[start : start + step]
        out = part if out is None else out + part
    assert out is not None
    return out


# --------------------------------------------------------------------------- one batch of k-points
@dataclass(frozen=True)
class _Context:
    """What every task of a recomputation shares."""

    run: ShiftRun
    channels: tuple[Channel, ...]
    exact: float
    energy_scale: float
    bases: tuple[np.ndarray | None, ...]
    batch: int = VECTOR_BATCH


@dataclass
class _Mode:
    """One mode's transitions at a batch of k-points in one orbital basis."""

    levels: np.ndarray
    shifted: np.ndarray
    weights: np.ndarray
    magnitude: np.ndarray
    condition: np.ndarray

    def energies(self, occupied: int) -> np.ndarray:
        """The scissored transition energies ``(K, O, E)``."""
        return self.shifted[:, None, occupied:] - self.shifted[:, :occupied, None]


class _Sums:
    """Accumulated tables of one task."""

    def __init__(self, run: ShiftRun):
        self.run = run
        self.tables: dict[str, np.ndarray] = {}
        self.facts: dict[str, Any] = {
            "transitions": 0,
            "flagged_kpoints": 0,
            "exact_kpoints": 0,
            "min_spacing_eV": math.inf,
        }

    def add(self, name: str, kernel: Callable[[np.ndarray], np.ndarray], energies: Any, weights: Any) -> None:
        energies = np.asarray(energies, float).reshape(-1)
        weights = np.asarray(weights, float).reshape(len(energies), -1)
        if not len(energies):
            value = np.zeros((self.run.count, len(SHIFT_COMPONENTS)))
        else:
            value = _apply(kernel, energies, weights)
        self.tables[name] = self.tables.get(name, 0.0) + value

    def table(
        self, prefix: str, kernel: Kernel, modes: Sequence[_Mode], mask: np.ndarray, occupied: int
    ) -> None:
        """A table in both bases with its rounding terms (the module docstring): ``value``, ``other``, ``envelope``
        (the absolute difference of the two bases' tables of each k-point, summed over the k-points), ``kernel``
        (the kernel's own rounding), ``floor`` (the broadened magnitudes) and ``edge``."""
        base, other = modes
        energies = [mode.energies(occupied) for mode in modes]
        # A transition on a discontinuity of the kernel is placed alike in both bases (its energy in the first one):
        # the edge terms below price either placement, the envelope measures the rest.
        on_edge = kernel.near_edge(energies[0].reshape(-1)).reshape(energies[0].shape)
        energies[1] = np.where(on_edge, energies[0], energies[1])
        weights = [np.where(mask[..., None], mode.weights, 0.0) for mode in modes]
        magnitude = np.where(mask[..., None], base.magnitude, 0.0)
        conditioned = magnitude * base.condition[..., None]
        count = self.run.count
        points = energies[0].shape[0]
        pairs = int(np.prod(energies[0].shape[1:]))
        width = len(SHIFT_COMPONENTS)
        names = ("value", "other", "envelope", "floor", "kernel", "slope", "absolute")
        totals = {name: np.zeros((count, width)) for name in names}
        sizes = np.abs(base.shifted)
        scales = sizes[:, None, occupied:] + sizes[:, :occupied, None]
        step = max(1, KERNEL_ENTRIES // max(1, count * pairs))
        for start in range(0, points, step):
            part = slice(start, start + step)
            size = len(range(*part.indices(points)))
            tables = []
            first: np.ndarray = np.empty((count, 0))
            for energy, weight in zip(energies, weights):
                matrix = kernel.value(energy[part].reshape(-1)).reshape(count, size, pairs)
                tables.append(np.einsum("wkp,kpc->kwc", matrix, weight[part].reshape(size, pairs, width)))
                if len(tables) == 1:
                    first = matrix.reshape(count, -1)
                    totals["floor"] += first @ conditioned[part].reshape(-1, width)
            totals["value"] += tables[0].sum(axis=0)
            totals["other"] += tables[1].sum(axis=0)
            totals["envelope"] += np.abs(tables[0] - tables[1]).sum(axis=0)
            flat = energies[0][part].reshape(-1)
            absolute = np.abs(weights[0][part].reshape(-1, width))
            totals["absolute"] += first @ magnitude[part].reshape(-1, width)
            rounding: np.ndarray | None = None
            slope: np.ndarray | None = None
            if kernel.offset is not None:
                offset = kernel.offset(flat)
                rounding = first * (1 + offset**2)
                slope = first * (2 * np.abs(offset) / self.run.eta)
            else:
                rounding = None if kernel.rounding is None else kernel.rounding(flat)
                slope = None if kernel.slope is None else kernel.slope(flat)
            if rounding is not None:
                totals["kernel"] += rounding @ absolute
            if slope is not None:
                totals["slope"] += slope @ (magnitude[part].reshape(-1, width) * scales[part].reshape(-1, 1))
        for name, value in totals.items():
            key = f"{prefix}.{name}"
            self.tables[key] = self.tables.get(key, 0.0) + value
        flat = energies[0].reshape(-1)
        near = kernel.near_edge(flat)
        for name in (prefix + ".edge", prefix + ".edge_cumulative"):
            self.tables.setdefault(name, np.zeros((count, width)))
        if near.any():
            # A transition on a discontinuity of the kernel lands on either side of it: per row the difference of the
            # two placements, and, summed over the rows up to each one, what that choice moves (an interior edge only
            # the row below it, the lower end of the table every row).
            jump = kernel.value(flat[near] + 2 * EDGE_EV) - kernel.value(flat[near] - 2 * EDGE_EV)
            amounts = np.abs(weights[0].reshape(-1, width)[near])
            self.tables[prefix + ".edge"] += np.abs(jump) @ amounts
            self.tables[prefix + ".edge_cumulative"] += np.abs(np.cumsum(jump, axis=0)) @ amounts


def _transitions_mask(run: ShiftRun, shifted: np.ndarray, tolerance: float | None) -> np.ndarray:
    """The pairs ``(K, O, E)`` a mode sums: the named pair alone, else every one; QUEER's (``tolerance``) only
    those whose scissored energy exceeds its degeneracy tolerance."""
    occupied = run.occupied
    energies = shifted[:, None, occupied:] - shifted[:, :occupied, None]
    mask = np.ones(energies.shape, bool)
    if run.pair is not None:
        mask[:] = False
        m, n = run.pair
        if 0 <= m < occupied <= n < shifted.shape[1]:
            mask[:, m, n - occupied] = True
    if tolerance is not None:
        mask &= energies > tolerance
    return mask


def _evaluate(
    operators: Operators,
    run: ShiftRun,
    *,
    treatment: str,
    threshold: float,
    regularization: float | None,
    scissored_kernels: bool,
    weight: float,
) -> tuple[_Mode, np.ndarray, np.ndarray]:
    levels, vectors = eigensystem(operators.h, operators.s)
    shifted = levels.copy()
    shifted[:, run.occupied :] += run.scissors
    values, magnitude = integrand(
        operators,
        shifted if scissored_kernels else levels,
        vectors,
        run.occupied,
        treatment=treatment,
        threshold=threshold,
        regularization=regularization,
    )
    kernels = shifted if scissored_kernels else levels
    return (
        _Mode(
            levels,
            shifted,
            weight * values,
            abs(weight) * magnitude,
            conditioning(kernels, run.occupied, threshold, regularization),
        ),
        shifted,
        vectors,
    )


def conditioning(
    levels: np.ndarray, occupied: int, threshold: float, regularization: float | None
) -> np.ndarray:
    """Each transition's conditioning ``1 + max|E| / s`` ``(K, O, E)``: ``s`` the smallest spacing between one of its
    two levels and another level, as the derivative kernels see it (a level within ``threshold`` is the same
    degenerate level, or, with a ``regularization``, a spacing of ``2 regularization``, the regularized kernel's
    largest ``1 / s``). How far rounding moves its integrand, relative to its magnitudes."""
    spacing = np.abs(levels[:, :, None] - levels[:, None, :])
    inside = spacing <= threshold
    spacing = np.where(inside, np.inf if regularization is None else 2 * regularization, spacing)
    nearest = spacing.min(axis=2)
    scale = np.abs(levels).max(axis=1)[:, None, None]
    pair = np.minimum(nearest[:, :occupied, None], nearest[:, None, occupied:])
    return 1.0 + scale / pair


def _batch(context: _Context, kpoints: np.ndarray) -> _Sums:
    run = context.run
    kernels = Kernels(run)
    sums = _Sums(run)
    occupied = run.occupied
    scale = context.energy_scale
    queer = kernels.queer()
    binned = kernels.queer(binned=True)
    pyatb = kernels.pyatb()
    compared = binned if run.smearing == 0 else queer
    for index, channel in enumerate(context.channels):
        reconciled = bloch_operators(channel.reconciled, kpoints)
        modes = []
        for basis in context.bases:
            operators = reconciled if basis is None else rotated(reconciled, basis)
            mode, shifted, vectors = _evaluate(
                operators,
                run,
                treatment="covariant",
                threshold=context.exact,
                regularization=None,
                scissored_kernels=False,
                weight=-0.5,
            )
            modes.append(mode)
            if basis is None:
                base_operators, base_vectors = operators, vectors
        base = modes[0]
        mask = _transitions_mask(run, base.shifted, context.exact)
        tables = [("sigma", queer)] + ([("binned", binned)] if run.smearing == 0 else [])
        for name, kernel in tables:
            sums.table(f"physical{index}.{name}", kernel, modes, mask, occupied)
        if index:
            continue
        # The attribution path from QUEER's value to PyATB's table (first basis).
        energies = base.energies(occupied).reshape(-1) * scale
        weights = np.where(mask[..., None], base.weights, 0.0).reshape(-1, 18)
        sums.add("path.hartree", compared.value, energies, weights)
        sums.add("path.window", windowed(compared, run).value, energies, weights)
        sums.add("path.bins", pyatb.value, energies, weights)
        if run.scissors:
            values, _ = integrand(
                base_operators,
                base.shifted,
                base_vectors,
                occupied,
                treatment="covariant",
                threshold=context.exact,
            )
            scissored = np.where(mask[..., None], -0.5 * values, 0.0).reshape(-1, 18)
            sums.add("path.scissors", pyatb.value, energies, scissored)
        # PyATB's own table.
        recorded = scaled(bloch_operators(channel.recorded, kpoints, construction="pyatb"), scale)
        literal = []
        for basis in context.bases:
            operators = recorded if basis is None else rotated(recorded, basis)
            mode, _, _ = _evaluate(
                operators,
                run,
                treatment="pyatb",
                threshold=PYATB_THRESHOLD,
                regularization=PYATB_REGULARIZATION,
                scissored_kernels=True,
                weight=-1.0,
            )
            literal.append(mode)
        base_p = literal[0]
        mask_p = _transitions_mask(run, base_p.shifted, None)
        sums.table("literal", pyatb, literal, mask_p, occupied)
        # Plain facts: PyATB's selected transitions, the k-points its method-1 cut regularizes and those where an
        # exact degeneracy takes part (a band of a selected pair within the exactness bound of another level).
        energies_p = base_p.energies(occupied)
        selected = mask_p.copy()
        if run.pair is None:
            selected &= (energies_p >= run.start) & (energies_p < run.end)
        sums.facts["transitions"] += int(np.count_nonzero(selected))
        bands = np.zeros(base_p.shifted.shape, bool)
        bands[:, :occupied] = selected.any(axis=2)
        bands[:, occupied:] = selected.any(axis=1)
        spacing = np.abs(base_p.shifted[:, :, None] - base_p.shifted[:, None, :])
        diagonal = np.arange(spacing.shape[1])
        spacing[:, diagonal, diagonal] = np.inf
        relevant = np.where(bands[:, :, None], spacing, np.inf)
        exact = relevant <= context.exact * scale
        sums.facts["exact_kpoints"] += int(np.count_nonzero(exact.any(axis=(1, 2))))
        others = np.where(exact, np.inf, relevant)
        sums.facts["flagged_kpoints"] += int(np.count_nonzero((others <= PYATB_THRESHOLD).any(axis=(1, 2))))
        sums.facts["min_spacing_eV"] = min(
            sums.facts["min_spacing_eV"], float(others.min(initial=np.inf)) / scale
        )
    return sums


def _task(context: _Context, kpoints: np.ndarray) -> _Sums:
    total: _Sums | None = None
    for start in range(0, len(kpoints), context.batch):
        part = _batch(context, kpoints[start : start + context.batch])
        if total is None:
            total = part
            continue
        for key, value in part.tables.items():
            total.tables[key] = total.tables.get(key, 0.0) + value
        for key in ("transitions", "flagged_kpoints", "exact_kpoints"):
            total.facts[key] += part.facts[key]
        total.facts["min_spacing_eV"] = min(total.facts["min_spacing_eV"], part.facts["min_spacing_eV"])
    assert total is not None
    return total


# --------------------------------------------------------------------------- the recomputation
@dataclass
class Table:
    """A table ``(count, 18)`` (microA/V^2) in the first orbital basis, in the second, its rounding tolerance at each
    entry, the same kernel of its transitions' magnitudes (``absolute``: the size of what cancels in it) and the part
    of the tolerance that transitions on a discontinuity of the kernel add (``edge``: either placement; and
    ``edge_cumulative``: what that choice moves in the sum over the rows up to each one). ``smooth`` is the tolerance
    without the edges: how far rounding alone moves an entry."""

    value: np.ndarray
    other: np.ndarray
    rounding: np.ndarray
    absolute: np.ndarray
    edge: np.ndarray
    edge_cumulative: np.ndarray

    @property
    def smooth(self) -> np.ndarray:
        return self.rounding - self.edge

    def scaled(self, factor: float) -> Table:
        size = abs(factor)
        return Table(
            factor * self.value,
            factor * self.other,
            size * self.rounding,
            size * self.absolute,
            size * self.edge,
            size * self.edge_cumulative,
        )

    def __add__(self, other: Table) -> Table:
        return Table(
            self.value + other.value,
            self.other + other.other,
            self.rounding + other.rounding,
            self.absolute + other.absolute,
            self.edge + other.edge,
            self.edge_cumulative + other.edge_cumulative,
        )


@dataclass
class Recomputation:
    """What a recomputation established.

    ``physical``: QUEER's physical tables (``sigma``, without smearing ``binned``, for a collinear scfout ``legacy``
    and ``legacy_binned``, PyATB's spin factor times the spin-up channel); ``compared``: QUEER's value in PyATB's
    layout (export sign, spin factor, the spin-up channel of a collinear scfout, bins without smearing);
    ``literal``: PyATB's table; ``path``: the steps from ``compared`` to PyATB's table, each switching one of
    PyATB's ingredients on (first basis); ``facts``: counts and the cost.
    """

    physical: dict[str, Table]
    compared: Table
    literal: Table
    path: list[tuple[str, np.ndarray]]
    facts: dict[str, Any] = field(default_factory=dict)


PATH_STEPS = (
    ("constants", "PyATB's SI constants (e, hbar and its eV*s)"),
    ("hartree", "PyATB's Hartree constant: every transition energy times 27.211324570273 / QUEER's Hartree"),
    ("window", "PyATB's window [start, start + count domega) on its scale (no band pair)"),
    (
        "bins",
        "PyATB's int((E - start)/domega) rows: its Gaussian truncated at int(5 eta/domega) rows, or its bins",
    ),
    ("scissors", "the scissored levels inside PyATB's derivative kernels"),
    (
        "cut",
        "PyATB's method-1 integrand: 1/dE replaced by dE/(dE^2 + 0.04^2) within 1e-4 eV, i rbar(j, j) on the "
        "diagonal, its degenerate velocities, and its Bloch sums of the scfout's upper triangles",
    ),
)


def work(model: Any, run: ShiftRun) -> float:
    """A recomputation's size: k-points times the cube of the bands of a channel, summed over its channels."""
    size = int(model.nbnd) if model.spin_mode != "collinear" else int(getattr(model, "n_spatial", model.nbnd))
    return float(run.kpoints) * float(size) ** 3 * (2 if model.spin_mode == "collinear" else 1)


def recompute(
    model: Any,
    run: ShiftRun,
    *,
    handedness: int,
    workers: int | None = None,
    memory_bytes: float | None = None,
    log: Callable[[str], None] | None = None,
) -> Recomputation:
    """QUEER's physical spectrum and PyATB's table of ``run`` on ``model`` (an
    :class:`~queer.models.openmx.OpenMXModel` read from the scfout), each in two orbital bases, with their rounding
    tolerances and the attribution path (the module docstring). ``handedness`` is the sign of the scfout lattice's
    determinant; ``workers`` (None: every CPU) evaluate chunks of the grid on QUEER's process executor."""
    import time

    from ...execution import ExecutionOptions
    from .memory import chunks, plan_parallelism, pmap

    if run.smearing not in (0, 1):
        raise ValueError("only smearing_method 0 and 1 are recomputed")
    started = time.perf_counter()
    blocks = tuple(channels(model))
    size = blocks[0].reconciled.size
    if not 0 < run.occupied < size:
        raise ValueError(f"occ_band {run.occupied} must lie between 0 and the {size} bands PyATB reads")
    hartree = reader_hartree(model)
    scale = PYATB_HARTREE / hartree
    context = _Context(
        run=run,
        channels=blocks,
        exact=exactness_bound(model),
        energy_scale=scale,
        bases=(None, orthogonal_basis(size)),
    )
    kpoints = kpoint_grid(run.grid)
    fixed = sum(
        2.0 * channel.reconciled.hamiltonian.nbytes * (5 if channel.reconciled.position is not None else 2)
        for channel in blocks
    )
    serial = work(model, run) < SERIAL_WORK
    planned, chunk = plan_parallelism(
        size,
        MATRICES_PER_K,
        1 if serial else workers,
        None,
        fixed_bytes=fixed,
        n_items=len(kpoints),
        memory_bytes=memory_bytes,
    )
    options = ExecutionOptions(workers=planned, batch_size=chunk, native_threads=1).resolved()
    parts = pmap(functools.partial(_task, context), chunks(kpoints, chunk), options)
    tables: dict[str, np.ndarray] = {}
    facts: dict[str, Any] = {
        "transitions": 0,
        "flagged_kpoints": 0,
        "exact_kpoints": 0,
        "min_spacing_eV": math.inf,
    }
    for part in parts:
        for key, value in part.tables.items():
            tables[key] = tables.get(key, 0.0) + value
        for key in ("transitions", "flagged_kpoints", "exact_kpoints"):
            facts[key] += part.facts[key]
        facts["min_spacing_eV"] = min(facts["min_spacing_eV"], part.facts["min_spacing_eV"])
    volume = abs(float(np.linalg.det(np.asarray(model.lattice, float))))
    conversion = -math.pi * QUEER_E_CHARGE**2 / QUEER_HBAR * 1e6 / volume / run.kpoints
    ratio = (
        (PYATB_E_CHARGE**2 / PYATB_HBAR)
        * (PYATB_E_CHARGE * PYATB_EV_SECONDS / PYATB_HBAR)
        / (QUEER_E_CHARGE**2 / QUEER_HBAR)
    )
    mode = model.spin_mode
    spin_factor = 2 if mode != "soc" else 1
    degeneracy = 2 if mode == "nonmagnetic" else 1

    def table(prefix: str) -> Table:
        rounding = (
            SPREAD_MULTIPLE * tables[prefix + ".envelope"]
            + KERNEL_ROUNDING * tables[prefix + ".kernel"]
            + ENERGY_ROUNDING * tables[prefix + ".slope"]
            + MAGNITUDE_FLOOR * tables[prefix + ".floor"]
            + tables[prefix + ".edge"]
        )
        return Table(
            tables[prefix + ".value"],
            tables[prefix + ".other"],
            rounding,
            tables[prefix + ".absolute"],
            tables[prefix + ".edge"],
            tables[prefix + ".edge_cumulative"],
        )

    names = ["sigma"] + (["binned"] if run.smearing == 0 else [])
    channel_tables = [
        {name: table(f"physical{index}.{name}").scaled(conversion) for name in names}
        for index in range(len(blocks))
    ]
    physical: dict[str, Table] = {}
    for name in names:
        total = channel_tables[0][name].scaled(degeneracy)
        for extra in channel_tables[1:]:
            total = total + extra[name]
        physical[name] = total
    if mode == "collinear":
        physical["legacy"] = channel_tables[0]["sigma"].scaled(spin_factor)
        if run.smearing == 0:
            physical["legacy_binned"] = channel_tables[0]["binned"].scaled(spin_factor)
    layout = "binned" if run.smearing == 0 else "sigma"
    compared = channel_tables[0][layout].scaled(handedness * spin_factor)
    # PyATB's prefactor: -t pi e^3 / (2 hbar^2 V N_k) 1e6 with the signed volume, times its spin factor.
    signed = handedness * volume
    prefactor = (
        -PYATB_EV_SECONDS * math.pi * PYATB_E_CHARGE**3 / (2 * PYATB_HBAR**2 * signed * run.kpoints) * 1e6
    ) * spin_factor
    literal = table("literal").scaled(prefactor)
    steady = handedness * spin_factor * ratio * conversion
    path = [("queer", compared.value), ("constants", ratio * compared.value)]
    path.append(("hartree", steady * tables["path.hartree"]))
    path.append(("window", steady * tables["path.window"]))
    path.append(("bins", steady * tables["path.bins"]))
    path.append(("scissors", steady * tables["path.scissors"] if run.scissors else path[-1][1]))
    path.append(("cut", literal.value))
    facts.update(
        {
            "kpoints": run.kpoints,
            "bands": size,
            "channels": len(blocks),
            "work": work(model, run),
            "workers": int(options.workers),
            "chunk": int(chunk),
            "seconds": time.perf_counter() - started,
            "exactness_bound_eV": context.exact,
            "energy_scale": scale,
            "reader_hartree_eV": hartree,
            "constant_ratio": ratio,
            "spread_multiple": SPREAD_MULTIPLE,
            "magnitude_floor": MAGNITUDE_FLOOR,
            "edge_eV": EDGE_EV,
            "basis_seed": BASIS_SEED,
        }
    )
    if not math.isfinite(facts["min_spacing_eV"]):
        facts["min_spacing_eV"] = None
    if log is not None:
        log(
            f"shift-current recomputation: {run.kpoints} k-points x {size} bands x {len(blocks)} channel(s) in "
            f"{facts['seconds']:.1f} s on {facts['workers']} worker(s)"
        )
    return Recomputation(physical=physical, compared=compared, literal=literal, path=path, facts=facts)
