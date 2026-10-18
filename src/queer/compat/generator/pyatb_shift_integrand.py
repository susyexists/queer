"""An independent recomputation of a Generator shift current from its scfout: QUEER's physical spectrum and the
table PyATB 60fc6b2a prints.

The shift-current comparison (:mod:`~queer.compat.generator.parity_shift`) evaluates, on the run's own k grid
and photon energies, every one of the 18 components in two modes, each in several orbital bases:

- **physical**: the shift current as QUEER defines it, written here from its equations:
  ``sigma^abc(w) = -(pi e^2 / hbar) (g / (V N_k)) sum_k sum_{m occ, n empty} f_mn Im[r^b_nm r^c_mn;a]_(bc)
  delta(w - w_nm)`` with the exact SI 2019 ``e`` and ``hbar``, ``V = abs(det A)``, ``g = 2`` for a scfout
  without spin polarization (one channel occupied twice), the two channels summed for a collinear scfout and
  the spinor bands with spin-orbit coupling; the lowest ``occ_band`` levels occupied (per channel); the band
  pair alone when one is named; the scissors moving the transition energies only (the geometry from the
  unshifted levels); every transition above the degeneracy tolerance; the Gaussian of standard deviation
  ``eta / sqrt(2)``, ``exp(-(x / eta)^2) / (eta sqrt(pi))`` at every photon energy (``smearing_method 1``),
  or, without smearing, QUEER's histogram: the transition in row ``i`` of the per-eV table when ``w_i - domega/2
  < E <= w_i + domega/2`` (weight ``1 / domega``) and in row ``i`` of ``binned_uA_V2`` when ``w_i < E <= w_i +
  domega``, each row from one integer index (``ceil(x - 1/2)`` and ``ceil(x) - 1``, ``x = (E - start) / domega``),
  so that every transition lands in exactly one row. An exact degeneracy (levels within
  :func:`~queer.compat.generator.shift_scfout.exactness_bound` of each other) is treated covariantly: parallel
  transport inside the degenerate levels (``D_a = -Sbar_a / 2 + i (rbar_a + rbar_a^dagger) / 2`` there, which
  normalizes them and makes their connection vanish) and their velocity matrix, so that a degenerate level's
  transitions sum to a value independent of the basis inside it;
- **PyATB-literal**: what PyATB prints, transcribed from its sources (``shift_current_solver.cpp``,
  ``linear_response.cpp``, ``xr_operation.cpp``, ``base_data.cpp``, ``openmx_read_scfout.py``; read only,
  written here anew): its Bloch sums of the upper triangles of the operators its reader builds (Hermitian
  completion with a real diagonal; the position operator's lower triangle from ``(r + i dS)^dagger``), its
  ``method 1`` sum-over-states integrand (``get_Inm_ik_sumOver`` with ``get_D_degenerate``: ``1 / dE`` between
  levels more than 1e-4 eV apart, ``dE / (dE^2 + 0.04^2)`` within, ``i rbar_a(j, j)`` on the diagonal, and the band
  velocities of ``get_partial_eigenvalue_fromBar``, the eigenvalues of a band's block of levels within 1e-4 eV),
  the scissors added to the levels from ``occ_band`` on before all of it, its window ``[start, start + count
  domega)`` without a band pair, its rows ``int((E - start) / domega)`` (C's truncation), its Gaussian on the rows
  within ``int(5 eta / domega)`` of that row or the bare weight in that row without smearing, ``(f_n - f_m) I =
  -I`` per pair, and its prefactor ``-t pi e^3 / (2 hbar^2 V N_k) 1e6`` with its ``t`` = 6.582119e-16 eV s, ``e``
  = 1.602176565e-19 C, ``hbar`` = 1.054571726e-34 J s, the signed cell volume and its spin factor (2 for ``nspin
  1``, which reads the spin-up channel only, 1 for ``nspin 4``). ``method 0`` (``get_Inm_ik_direct``, every
  denominator of its eigenvector derivatives regularized) and the adaptive smearing are not transcribed.

**Operators.** Both modes start from this recomputation's own parse of the scfout
(:mod:`~queer.compat.generator.shift_scfout`): QUEER's physical definition (CODATA's Hartree and Bohr, each spin
channel from its own records, every record projected onto the translation identities) for the physical mode,
PyATB's reader (its Hartree and Bohr, the spin-up channel of an ``nspin 1`` scfout Hermitized in real space) for
the literal one. No code of QUEER's reader, runner or observables runs: a defect in QUEER's reader moves QUEER's
value and not the physical mode's (check A fails), and a defect in this parse moves the literal table and not
PyATB's (check B is not met).

At one k-point, with ``H C = S C E`` (``C^dagger S C = 1``, solved here through the Cholesky factor of
``S``), the band-basis projections ``Xbar_a = C^dagger (d_a X) C`` and ``M_a(i, j) = (Hbar_a - E_j
Sbar_a)(i, j)``: ``D_a(i, j) = -M_a(i, j) g(E_i - E_j)``; ``r_mn^a = i D_a(m, n) + rbar_a(m, n)``,
``rbar_a = C^dagger r_a^dagger C``; ``d_b`` of a projection at ``(n, m)`` is ``(D_b^dagger Xbar_a)(n, m) +
Xbar_ab(n, m) + (Xbar_a D_b)(n, m)``; ``r_nm;b^a = i [(d_b Hbar_a - E_m d_b Sbar_a - Sbar_a V_b) d - (M_a V_b
- V_b M_a)] / d^2 + d_b rbar_a`` at ``(n, m)`` with ``d = E_m - E_n`` and ``V_b`` the velocities; ``I^{a,
bc} = Im(r_mn^b r_nm;a^c + r_mn^c r_nm;a^b)``. QUEER's weight of a transition is ``-I / 2``.

**Rounding, from the comparison's own computation.** Each table's rounding tolerance at each entry is the sum of:

- :data:`SPREAD_MULTIPLE` times its envelope over :data:`BASES` evaluations: each mode is evaluated again in
  ``BASES - 1`` fixed random orthogonal changes of the orbital basis, each with the translations of its Bloch sums
  in another fixed random order (:func:`orthogonal_basis`, :func:`summation_order`), which reach the same levels
  and integrand through other floating-point operations; the envelope is the largest over those bases of ``sum_k
  |T_k - T'_k|`` (each k-point's table in the first basis and in that one, its difference taken before the sum over
  the k-points, so that no cancellation between k-points hides it);
- :data:`TERM_FLOOR` times the broadened first-order rounding of its transitions' integrands (:func:`_noise`):
  every quantity of :func:`integrand`'s formulas carries its rounding forward (a product ``x y`` by ``N_x |y| + |x|
  N_y``, a sum by its parts' sum, and the second-order product), from the operators' Bloch sums and projections
  (``|C|^T (sum_R |w_R| |X(R)|) |C|``), the eigenvectors (a backward-stable solver rotates level ``m`` into level
  ``j`` by up to ``s / |E_m - E_j|``, ``s`` the reduced problem's norm, which moves a projection by ``s (G |Xbar| +
  |Xbar| G)``, ``G_mj = 1 / |E_m - E_j|``: a factor a symmetry forbids is noise of the size of the elements that
  rotation mixes in, not of its own size) and the energies (``max|E|`` each, through every ``1 / dE``). For the
  physical mode, levels closer than :data:`GROUP_SHARE` of the kernel's width are one group: for a transition between
  two groups wholly on their sides of the occupation boundary the table sums every transition between them, so a
  rotation inside a group cancels there but for the group's spread through the kernel's slope (or whole at a bin
  edge within that spread);
- the kernel's own rounding and the slope times the rounding of the transition energies (each
  :data:`KERNEL_ROUNDING`, :data:`ENERGY_ROUNDING` relative);
- for a transition within :data:`EDGE_EV` of a discontinuity of the kernel (a bin edge, PyATB's rows and window
  ends), the difference of placing it either side (``edge``; ``edge_cumulative`` is what that choice moves in the
  sum over the rows up to each one, which an interior edge leaves unchanged beyond its two rows). On an outer edge
  of the table (the first or last edge of a histogram, PyATB's window ends: one side lies outside the table) the
  choice moves the sum over the rows by the transition's whole weight, so there no allowance is given: each
  k-point's transitions on that edge have exactly two placements (``placements``: inside the table or outside it),
  and a comparison tests the table against each combination of them (:class:`Placement`).

Where PyATB's own value depends on its eigensolver's basis inside degenerate levels, the bases differ by that
dependence, and the tolerance says so. No number of the candidate's enters a tolerance.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .pyatb_files import SHIFT_COMPONENTS
from .shift_scfout import (
    PYATB_BOHR,
    PYATB_HARTREE,
    Blocks,
    Records,
    codata_bohr,
    codata_hartree,
    exactness_bound,
    literal_channel,
    physical_channels,
    residuals,
)

# QUEER's SI constants: the exact SI 2019 values.
QUEER_E_CHARGE = 1.602176634e-19
QUEER_HBAR = 6.62607015e-34 / (2.0 * math.pi)
# PyATB 60fc6b2a's: get_shift_current_conductivity's eV_seconds, elem_charge_SI and hbar_SI, linear_response's
# degenerate_threshold and degenerate_eta (its reader's Hartree and Bohr are shift_scfout's).
PYATB_EV_SECONDS = 6.582119e-16
PYATB_E_CHARGE = 1.602176565e-19
PYATB_HBAR = 1.054571726e-34
PYATB_THRESHOLD = 1e-4
PYATB_REGULARIZATION = 0.04
# The rounding tolerance of a table: this many times its envelope over the bases, plus this share of its broadened
# first-order rounding (in units of the working precision; QUEER's estimator differs from the transcription by at most
# 4.2e-16 of it on thirteen models, real ones included), plus the kernel's and the energies' rounding, plus the edge
# placements of the transitions within EDGE_EV (eV) of a discontinuity of its kernel.
SPREAD_MULTIPLE = 100.0
TERM_FLOOR = 32 * float(np.finfo(float).eps)
KERNEL_ROUNDING = 64 * float(np.finfo(float).eps)
ENERGY_ROUNDING = 64 * float(np.finfo(float).eps)
EDGE_EV = 1e-9
# A table's transitions on its outer edges are tested in their exact placements, one choice per k-point and edge, up
# to this many such groups (2^OUTER_PLACEMENTS combinations); beyond it, or where the recomputation's own placement
# splits a k-point's group across the edge, those entries are not determined.
OUTER_PLACEMENTS = 10
# For the physical mode's rounding, levels closer than this share of the kernel's width (eta, or domega without
# smearing) are one group: their transitions land at the same photon energies, so a rotation inside the group moves
# the table only by what the evaluation bases measure.
GROUP_SHARE = 0.01
# The evaluations of each mode (the first basis and BASES - 1 fixed random orthogonal changes of it, each with its own
# order of the translations), from a fixed seed.
BASES = 4
BASIS_SEED = 20261018
# Complex N x N matrices one k-point of a vectorized batch keeps alive (208 measured on the calibration scfout with half
# its bands occupied, the rounding's propagation included), and the k-points of one batch.
MATRICES_PER_K = 240
VECTOR_BATCH = 32
# Kernel entries (photon energies x transitions) one product holds at once.
KERNEL_ENTRIES = 4_000_000
# A photon energy costs this share of a band in the work of a k-point (:func:`work`).
PHOTON_SHARE = 8.0
# Below this much work (:func:`work`) the recomputation runs in this process.
SERIAL_WORK = 4e7

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


def orthogonal_basis(size: int, seed: int = BASIS_SEED) -> np.ndarray:
    """A fixed random orthogonal matrix: another orbital basis for the same evaluation."""
    matrix = np.random.default_rng(seed + size).standard_normal((size, size))
    basis, _ = np.linalg.qr(matrix)
    return basis


def summation_order(count: int, seed: int = BASIS_SEED) -> np.ndarray:
    """A fixed random order of ``count`` translations: another order of the Bloch sums."""
    return np.random.default_rng(seed + 7919 * count).permutation(count)


def reordered(blocks: Blocks, order: np.ndarray | None) -> Blocks:
    """``blocks`` with its translations in ``order`` (the same operators, summed in another order)."""
    if order is None:
        return blocks
    return Blocks(
        blocks.translations[order],
        blocks.cartesian[order],
        blocks.hamiltonian[order],
        blocks.overlap[order],
        None if blocks.position is None else blocks.position[:, order],
    )


@dataclass(frozen=True)
class Basis:
    """One evaluation's orbital basis (``rotation``, None for the first) and order of the translations."""

    rotation: np.ndarray | None
    order: np.ndarray | None


def evaluation_bases(size: int, translations: int, count: int = BASES) -> tuple[Basis, ...]:
    """The first basis and ``count - 1`` others, each with its own order of the translations."""
    out = [Basis(None, None)]
    for index in range(1, count):
        seed = BASIS_SEED + 104729 * index
        out.append(Basis(orthogonal_basis(size, seed), summation_order(translations, seed)))
    return tuple(out)


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


@dataclass(frozen=True)
class Bounds:
    """The magnitudes of an operator set's translation sums, ``sum_R |w_R| |X(R)|`` (k-independent, symmetric):
    ``dh``, ``ds``, ``r`` ``(3, N, N)``; ``d2h``, ``d2s`` ``(3, 3, N, N)``; ``dr[b, a]`` of ``d_b r_a``; and
    ``norms``, each one's largest row sum (``(3,)`` or ``(3, 3)``), which bounds the operator's norm at every k."""

    dh: np.ndarray
    ds: np.ndarray
    r: np.ndarray
    d2h: np.ndarray
    d2s: np.ndarray
    dr: np.ndarray
    norms: dict[str, np.ndarray] = field(default_factory=dict)


_SECOND = tuple((a, b) for a in range(3) for b in range(a, 3))


def _adjoint(matrix: np.ndarray) -> np.ndarray:
    return np.swapaxes(matrix.conj(), -1, -2)


def _hermitian(matrix: np.ndarray) -> np.ndarray:
    return (matrix + _adjoint(matrix)) / 2


def _symmetric(matrix: np.ndarray) -> np.ndarray:
    return np.maximum(matrix, np.swapaxes(matrix, -1, -2))


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

    def sums(source: np.ndarray) -> np.ndarray:
        flat = source.reshape(len(cartesian), size * size)
        if construction == "pyatb":
            flat = (source * upper).reshape(len(cartesian), size * size)
        return np.einsum("kr,wr,rn->kwn", phase, stack, flat, optimize=True).reshape(
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


def bounds(blocks: Blocks, *, construction: str = "hermitian", energy_scale: float = 1.0) -> Bounds:
    """The magnitudes of ``blocks``' translation sums (:class:`Bounds`), the Hamiltonian's times ``energy_scale``;
    PyATB's construction adds ``d_a S`` to the position operator's (its lower triangle)."""
    cartesian = np.abs(blocks.cartesian)
    hamiltonian = np.abs(blocks.hamiltonian) * abs(energy_scale)
    overlap = np.abs(blocks.overlap)
    first_h = _symmetric(np.einsum("ra,rij->aij", cartesian, hamiltonian))
    first_s = _symmetric(np.einsum("ra,rij->aij", cartesian, overlap))
    pairs = np.einsum("ra,rb->abr", cartesian, cartesian)
    second_h = _symmetric(np.einsum("abr,rij->abij", pairs, hamiltonian))
    second_s = _symmetric(np.einsum("abr,rij->abij", pairs, overlap))
    size = blocks.size
    if blocks.position is None:
        position = np.zeros((3, size, size))
        moved = np.zeros((3, 3, size, size))
    else:
        magnitude = np.abs(blocks.position)
        position = _symmetric(magnitude.sum(axis=1))
        moved = _symmetric(np.einsum("rb,arij->baij", cartesian, magnitude))
    if construction == "pyatb":
        position = position + first_s
        moved = moved + np.swapaxes(second_s, 0, 1)
    operators = {"dh": first_h, "ds": first_s, "r": position, "d2h": second_h, "d2s": second_s, "dr": moved}
    norms = {name: value.sum(axis=-1).max(axis=-1) for name, value in operators.items()}
    return Bounds(first_h, first_s, position, second_h, second_s, moved, norms)


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
    magnitudes: Bounds | None = None,
    group: float | None = None,
) -> tuple[np.ndarray, np.ndarray, Rounding | None]:
    """PyATB's ``I`` ``(K, O, E, 18)`` of every occupied (``m < occupied``) to empty (``n``) pair, its magnitude
    ``|r^b_mn| |r^c_nm;a| + |r^c_mn| |r^b_nm;a|`` and, with the operators' ``magnitudes`` (:func:`bounds`, in the
    basis of ``operators``), its :class:`Rounding` in units of the working precision (None without them).
    Angstrom^3. ``group`` (eV) treats rotations inside groups of levels closer than it as cancelling in the table for
    transitions between different groups (the physical mode).

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
    same: np.ndarray | None = None
    if treatment == "pyatb":
        eta = 0.0 if regularization is None else float(regularization)
        within = gap / (gap**2 + eta**2) if eta else np.zeros_like(gap)
        inverse = np.where(near, within, 1.0 / safe)
        # |d inverse / d gap|: how the kernel moves with the energies' rounding.
        bend = np.where(near, np.abs(eta**2 - gap**2) / (gap**2 + eta**2) ** 2 if eta else 0.0, inverse**2)
        derivative = -first * inverse[:, None]
        derivative[..., index, index] = 1j * rbar[..., index, index]
        diagonal = pyatb_velocities(hbar, sbar, energies, threshold)
    else:
        labels = cluster_labels(energies, threshold)
        same = (labels[:, :, None] == labels[:, None, :])[:, None]
        inverse = np.where(near, 0.0, 1.0 / safe)
        bend = inverse**2
        transport = -0.5 * sbar + 0.5j * (rbar + _adjoint(rbar))
        derivative = np.where(same, transport, -first * inverse[:, None])
        pair_energy = 0.5 * (energies[:, :, None] + energies[:, None, :])[:, None]
        velocity = _hermitian(np.where(same, hbar - sbar * pair_energy, 0.0))
    bend[:, index, index] = 0.0
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
    keep = magnitudes is not None
    sizes = (
        {name: np.empty((count, 3, 3, width, occupied)) for name in ("d_s", "inner", "numerator")}
        if keep
        else {}
    )
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
            inner = d_h - energy_m * d_s - overlap_v
            numerator = inner * difference - commutator
            generalized[:, a, b] = 1j * numerator / difference**2 + d_r
            if keep:
                sizes["d_s"][:, a, b] = np.abs(d_s)
                sizes["inner"][:, a, b] = np.abs(inner)
                sizes["numerator"][:, a, b] = np.abs(numerator)
    # PyATB's pairs (m, n): r_mn (K, 3, O, E) and r_nm;a^c at (n, m), transposed to (m, n).
    general = np.swapaxes(generalized, -1, -2)
    values = np.empty((count, occupied, width, len(SHIFT_COMPONENTS)))
    magnitude = np.empty_like(values)
    for column, (a, b, c) in enumerate(_ORDER):
        values[..., column] = np.imag(dipole[:, b] * general[:, c, a] + dipole[:, c] * general[:, b, a])
        magnitude[..., column] = np.abs(dipole[:, b]) * np.abs(general[:, c, a]) + np.abs(
            dipole[:, c]
        ) * np.abs(general[:, b, a])
    if magnitudes is None:
        return values, magnitude, None
    parts = _Parts(
        hbar=np.abs(hbar),
        sbar=np.abs(sbar),
        rbar=np.abs(rbar),
        first=np.abs(first),
        derivative=np.abs(derivative),
        inverse=np.abs(inverse),
        bend=bend,
        same=same,
        velocity=np.abs(diagonal) if velocity is None else np.abs(velocity),
        second=(np.abs(second_h), np.abs(second_s), np.abs(second_r)),
        d_s=sizes["d_s"],
        inner=sizes["inner"],
        numerator=sizes["numerator"],
        difference=np.abs(difference),
        dipole=np.abs(dipole),
        general=np.abs(general),
    )
    # Levels within the threshold are one degenerate level (QUEER's) or PyATB's regularized block: a rotation inside
    # them is not a first-order rounding, and the evaluation bases measure what it moves.
    exact = threshold
    full = _noise(parts, magnitudes, energies, vectors, occupied, exact=exact)
    none = np.zeros(full.shape[:3])
    if group is None or group <= exact:
        return values, magnitude, Rounding(full, full, none)
    labels = cluster_labels(energies, group)
    if not (np.diff(labels, axis=-1) == 0).any():
        return values, magnitude, Rounding(full, full, none)
    grouped = _noise(parts, magnitudes, energies, vectors, occupied, exact=exact, group=labels)
    # A transition between two groups that lie wholly on their sides of the occupation boundary: the table sums every
    # transition between them, so a rotation inside either cancels there but for their spread (``spread``).
    split = labels[:, occupied - 1] == labels[:, occupied]
    crossing = (labels == labels[:, occupied, None]) & split[:, None]
    separate = (labels[:, :occupied, None] != labels[:, None, occupied:]) & ~(
        crossing[:, :occupied, None] | crossing[:, None, occupied:]
    )
    low = np.full_like(energies, np.inf)
    high = np.full_like(energies, -np.inf)
    for column in range(size):
        same_group = labels == labels[:, column, None]
        low[:, column] = np.where(same_group, energies, np.inf).min(axis=-1)
        high[:, column] = np.where(same_group, energies, -np.inf).max(axis=-1)
    width_group = high - low
    spread = np.where(separate, width_group[:, :occupied, None] + width_group[:, None, occupied:], 0.0)
    return values, magnitude, Rounding(np.where(separate[..., None], grouped, full), full, spread)


@dataclass(frozen=True)
class Rounding:
    """An evaluation's rounding per transition ``(K, O, E, 18)`` in units of the working precision (:func:`_noise`):
    ``noise`` what the table carries (inside groups of near-degenerate levels, the part that does not cancel over the
    group), ``full`` the transition's own, and ``spread`` ``(K, O, E)`` its two groups' widths in eV (0 for a
    transition that is not grouped), over which the group's transitions are spread in energy."""

    noise: np.ndarray
    full: np.ndarray
    spread: np.ndarray


@dataclass(frozen=True)
class _Parts:
    """The magnitudes of an evaluation's intermediate quantities (:func:`integrand`) that its rounding propagates
    through: the projections, ``M``, ``D``, the derivative kernel and its slope, the velocities, the projected second
    derivatives, ``d_b S`` projected, the numerator's two stages, the transition energies, ``r_mn`` and
    ``r_nm;a``."""

    hbar: np.ndarray
    sbar: np.ndarray
    rbar: np.ndarray
    first: np.ndarray
    derivative: np.ndarray
    inverse: np.ndarray
    bend: np.ndarray
    same: np.ndarray | None
    velocity: np.ndarray
    second: tuple[np.ndarray, np.ndarray, np.ndarray]
    d_s: np.ndarray
    inner: np.ndarray
    numerator: np.ndarray
    difference: np.ndarray
    dipole: np.ndarray
    general: np.ndarray


def _noise(
    parts: _Parts,
    magnitudes: Bounds,
    energies: np.ndarray,
    vectors: np.ndarray,
    occupied: int,
    *,
    exact: float,
    group: np.ndarray | None = None,
) -> np.ndarray:
    """The integrand's rounding ``(K, O, E, 18)`` in units of the working precision: first-order (linearized)
    propagation of each quantity's rounding through :func:`integrand`'s formulas, every product ``x y`` moving by
    ``N_x |y| + |x| N_y`` and every sum by the sum of its parts' roundings, plus the second-order product of the two
    factors' roundings times :data:`TERM_FLOOR`. The sources:

    - the operators' Bloch sums and projections: ``|C|^T (sum_R |w_R| |X(R)|) |C|``;
    - the eigenvectors: a backward-stable eigensolver rotates level ``m`` into level ``j`` by up to ``s |E_m - E_j|^-1``
      (``s = max|E| max(1, max_n |C_n|^2)``, the reduced problem's norm), which moves a projection ``Xbar_mn`` by
      ``s (G |Xbar| + |Xbar| G)_mn`` with ``G_mj = 1 / |E_m - E_j|`` (levels within ``exact`` of each other excluded:
      the covariant treatment does not see their rotation, and PyATB's regularized block's dependence on it is what
      the evaluation bases measure), and the projected second derivatives by ``s nu_X (g_n |C_m| + |C_n| g_m)``
      (``nu_X`` the operator's largest row sum, ``g_m`` the norm of row ``m`` of ``G``);
    - the energies: each moves by ``max|E|``, so a kernel ``1 / dE`` by ``2 max|E| |d(1/dE)/d(dE)|``.

    With ``group`` (each level's group label) a rotation inside a group and the rounding of its own spacings are left
    out: for a transition between two groups, the table sums the rotated levels' transitions, which land at the same
    photon energies, and the evaluation bases measure what the rest moves."""
    count, size = energies.shape
    occ, emp = slice(0, occupied), slice(occupied, size)
    columns = np.abs(vectors)
    rows = np.swapaxes(columns, -1, -2)
    lengths = np.linalg.norm(vectors, axis=-2)
    scale = np.abs(energies).max(axis=-1)
    reach = scale * np.maximum(1.0, lengths.max(axis=-1) ** 2)
    spacing = np.abs(energies[:, :, None] - energies[:, None, :])
    excluded = spacing <= exact
    if group is not None:
        excluded = excluded | (group[:, :, None] == group[:, None, :])
    coupling = np.where(excluded, 0.0, 1.0 / np.where(excluded, 1.0, spacing))  # G (K, N, N)
    span = np.sqrt((coupling**2).sum(axis=-1))  # g (K, N)
    kept = 1.0 if group is None else (group[:, :, None] != group[:, None, :]).astype(float)
    level = np.abs(energies)

    def projected(operator: np.ndarray, value: np.ndarray) -> np.ndarray:
        rotation = coupling[:, None] @ value + value @ coupling[:, None]
        return rows[:, None] @ operator[None] @ columns[:, None] + reach[:, None, None, None] * rotation

    n_h = projected(magnitudes.dh, parts.hbar)
    n_s = projected(magnitudes.ds, parts.sbar)
    n_r = projected(magnitudes.r, parts.rbar)
    swing = 2.0 * scale[:, None, None]
    n_first = n_h + n_s * level[:, None, None, :] + parts.sbar * scale[:, None, None, None]
    n_d = n_first * parts.inverse[:, None] + parts.first * (parts.bend * swing * kept)[:, None]
    index = np.arange(size)
    if parts.same is None:
        n_d[..., index, index] = n_r[..., index, index]
        n_v = (
            n_h[..., index, index]
            + level[:, None, :] * n_s[..., index, index]
            + parts.sbar[..., index, index] * scale[:, None, None]
        )
    else:
        transport = 0.5 * n_s + 0.5 * (n_r + np.swapaxes(n_r, -1, -2))
        n_d = np.where(parts.same, transport, n_d)
        pair_level = 0.5 * (level[:, :, None] + level[:, None, :])[:, None]
        n_v = np.where(parts.same, n_h + n_s * pair_level + parts.sbar * scale[:, None, None, None], 0.0)
    n_dipole = n_d[..., occ, emp] + n_r[..., occ, emp]
    norms = magnitudes.norms
    columns_e, columns_o = columns[..., emp], columns[..., occ]
    rows_e = np.swapaxes(columns_e, -1, -2)
    stretch = span[:, emp, None] * lengths[:, None, occ] + lengths[:, emp, None] * span[:, None, occ]
    stretch = reach[:, None, None, None, None] * stretch[:, None, None]
    n_second = (
        rows_e[:, None, None] @ magnitudes.d2h[None] @ columns_o[:, None, None]
        + norms["d2h"][None, :, :, None, None] * stretch,
        rows_e[:, None, None] @ magnitudes.d2s[None] @ columns_o[:, None, None]
        + norms["d2s"][None, :, :, None, None] * stretch,
        rows_e[:, None, None] @ np.swapaxes(magnitudes.dr, 0, 1)[None] @ columns_o[:, None, None]
        + np.swapaxes(norms["dr"], 0, 1)[None, :, :, None, None] * stretch,
    )
    difference = parts.difference
    level_m = level[:, None, occ]
    projections = (parts.hbar, parts.sbar, parts.rbar)
    noises = (n_h, n_s, n_r)
    n_general = np.empty(parts.general.shape[:3] + (size - occupied, occupied))
    for b in range(3):
        d_b, n_b = parts.derivative[:, b], n_d[:, b]
        left, n_left = np.swapaxes(d_b, -1, -2)[:, emp, :], np.swapaxes(n_b, -1, -2)[:, emp, :]
        right, n_right = d_b[:, :, occ], n_b[:, :, occ]
        for a in range(3):
            moved = []
            for kind in range(3):
                x, n_x = projections[kind][:, a], noises[kind][:, a]
                moved.append(
                    n_left @ x[:, :, occ]
                    + left @ n_x[:, :, occ]
                    + n_second[kind][:, a, b]
                    + n_x[:, emp, :] @ right
                    + x[:, emp, :] @ n_right
                )
            n_dh, n_ds, n_dr = moved
            if parts.same is None:
                v_b, n_vb = parts.velocity[:, b], n_v[:, b]
                n_overlap = (
                    n_s[:, a, emp, occ] * v_b[:, None, occ] + parts.sbar[:, a, emp, occ] * n_vb[:, None, occ]
                )
                n_commutator = n_first[:, a, emp, occ] * (
                    v_b[:, None, occ] + v_b[:, emp, None]
                ) + parts.first[:, a, emp, occ] * (n_vb[:, None, occ] + n_vb[:, emp, None])
            else:
                v_b, n_vb = parts.velocity[:, b], n_v[:, b]
                n_overlap = n_s[:, a, emp, :] @ v_b[:, :, occ] + parts.sbar[:, a, emp, :] @ n_vb[:, :, occ]
                n_commutator = (
                    n_first[:, a, emp, :] @ v_b[:, :, occ]
                    + parts.first[:, a, emp, :] @ n_vb[:, :, occ]
                    + n_vb[:, emp, :] @ parts.first[:, a, :, occ]
                    + v_b[:, emp, :] @ n_first[:, a, :, occ]
                )
            n_inner = n_dh + level_m * n_ds + parts.d_s[:, a, b] * scale[:, None, None] + n_overlap
            n_numerator = n_inner * difference + parts.inner[:, a, b] * swing + n_commutator
            n_general[:, a, b] = (
                n_numerator / difference**2
                + parts.numerator[:, a, b] * 2 * swing / np.abs(difference) ** 3
                + n_dr
            )
    n_general = np.swapaxes(n_general, -1, -2)
    out = np.empty((count, occupied, size - occupied, len(SHIFT_COMPONENTS)))
    dipole, general = parts.dipole, parts.general
    for column, (a, b, c) in enumerate(_ORDER):
        out[..., column] = (
            n_dipole[:, b] * general[:, c, a]
            + dipole[:, b] * n_general[:, c, a]
            + n_dipole[:, c] * general[:, b, a]
            + dipole[:, c] * n_general[:, b, a]
            + TERM_FLOOR * (n_dipole[:, b] * n_general[:, c, a] + n_dipole[:, c] * n_general[:, b, a])
        )
    return out


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
    """One k-point's levels (``energy_scale`` times the blocks' eV, unshifted), its transitions' energies
    (scissored) and ``I`` ``(O, E, 18)`` with its magnitudes and, as ``terms``, its own first-order rounding in units
    of the working precision (:func:`_noise`, in the blocks' basis);
    ``scissored_kernels`` feeds the scissored levels to the derivative kernels (PyATB's)."""
    operators = scaled(
        bloch_operators(blocks, np.asarray(k, float)[None], construction=construction), energy_scale
    )
    if basis is not None:
        operators = rotated(operators, basis)
    levels, vectors = eigensystem(operators.h, operators.s)
    shifted = levels.copy()
    shifted[:, occupied:] += scissors
    values, magnitude, rounding = integrand(
        operators,
        shifted if scissored_kernels else levels,
        vectors,
        occupied,
        treatment=treatment,
        threshold=threshold,
        regularization=regularization,
        magnitudes=None
        if basis is not None
        else bounds(blocks, construction=construction, energy_scale=energy_scale),
    )
    transitions = shifted[0, None, occupied:] - shifted[0, :occupied, None]
    return {
        "levels": levels[0],
        "energies": transitions,
        "integrand": values[0],
        "magnitude": magnitude[0],
        "terms": None if rounding is None else rounding.full[0],
    }


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
    # A histogram's distance (eV) from each transition to its nearest bin edge.
    distance: Callable[[np.ndarray], np.ndarray] | None = None
    # The kernel of transitions at the first energies placed by the rows and window of the second (the same energies
    # moved across an edge): either placement of a transition on an edge, with its own value (None: ``value`` of the
    # second, for a kernel constant between its edges).
    placed: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None


def queer_rows(run: ShiftRun, energies: np.ndarray, *, binned: bool) -> np.ndarray:
    """Each transition's row of QUEER's histogram, from one integer index: per eV row ``i`` holds ``w_i -
    domega/2 < E <= w_i + domega/2`` (``ceil(x - 1/2)``), the binned table's row ``i`` holds ``w_i < E <= w_i +
    domega`` (``ceil(x) - 1``), ``x = (E - start) / domega``. Every transition lands in exactly one row (or none in
    the table)."""
    position = (np.asarray(energies, float) - run.start) / run.domega
    return (np.ceil(position) - 1 if binned else np.ceil(position - 0.5)).astype(np.int64)


def pyatb_rows(run: ShiftRun, energies: np.ndarray) -> np.ndarray:
    """Each transition's row of PyATB's table, ``int((E - start) / domega)`` (C's truncation toward zero)."""
    return np.trunc((np.asarray(energies, float) - run.start) / run.domega).astype(np.int64)


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
        photon energy (``smearing_method 1``), or its histogram (:func:`queer_rows`) per eV, and with ``binned`` the
        weight in ``(w, w + domega]``."""
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
        rows = np.arange(run.count)[:, None]

        def value(energies: np.ndarray) -> np.ndarray:
            lines = rows == queer_rows(run, energies, binned=binned)[None, :]
            return lines.astype(float) if binned else lines / width

        def distance(energies: np.ndarray) -> np.ndarray:
            position = (energies - run.start) / width - (0.0 if binned else 0.5)
            return np.abs(position - np.round(position)) * width

        return Kernel(value, None, None, lambda energies: distance(energies) < EDGE_EV, distance=distance)

    def pyatb(self, *, window: bool = True) -> Kernel:
        """PyATB's kernel of transitions (PyATB's eV): row :func:`pyatb_rows`; its Gaussian on the rows within
        ``int(5 eta/domega)`` of it, or its bare weight there without smearing; without a band pair (``window``)
        only the transitions in ``[start, start + count domega)``."""
        run = self.run
        rows = np.arange(run.count)[:, None]

        def mask(energies: np.ndarray) -> np.ndarray:
            row = pyatb_rows(run, energies)[None, :]
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
            placed=lambda energies, side: np.where(
                mask(side), self._gaussian(energies, "value", photon), 0.0
            ),
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
    """What every task of a recomputation shares: the physical mode's channels, PyATB's operators, the exactness
    bound (eV), PyATB's energy scale, the evaluation bases."""

    run: ShiftRun
    physical: tuple[Blocks, ...]
    literal: Blocks
    exact: float
    energy_scale: float
    bases: tuple[Basis, ...]
    batch: int = VECTOR_BATCH


@dataclass(frozen=True)
class _Prepared:
    """A task's operator sets in each basis's order of the translations, and the first basis's magnitudes."""

    physical: tuple[tuple[Blocks, ...], ...]
    literal: tuple[Blocks, ...]
    physical_bounds: tuple[Bounds, ...]
    literal_bounds: Bounds


def _prepare(context: _Context) -> _Prepared:
    orders = [basis.order for basis in context.bases]
    return _Prepared(
        physical=tuple(tuple(reordered(blocks, order) for order in orders) for blocks in context.physical),
        literal=tuple(reordered(context.literal, order) for order in orders),
        physical_bounds=tuple(bounds(blocks) for blocks in context.physical),
        literal_bounds=bounds(context.literal, construction="pyatb", energy_scale=1.0),
    )


@dataclass
class _Mode:
    """One mode's transitions at a batch of k-points in one orbital basis (``rounding``: the first basis's, weighted)."""

    levels: np.ndarray
    shifted: np.ndarray
    weights: np.ndarray
    magnitude: np.ndarray
    rounding: Rounding | None = None

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
        # Per table: the placements of its outer edges' groups (dropped beyond OUTER_PLACEMENTS), their number and
        # the number the recomputation's own placement splits across the edge.
        self.placements: dict[str, list[Placement]] = {}
        self.groups: dict[str, list[int]] = {}

    def merge(self, other: _Sums) -> None:
        """Adds another task's sums."""
        for key, value in other.tables.items():
            self.tables[key] = self.tables.get(key, 0.0) + value
        for key in ("transitions", "flagged_kpoints", "exact_kpoints"):
            self.facts[key] += other.facts[key]
        self.facts["min_spacing_eV"] = min(self.facts["min_spacing_eV"], other.facts["min_spacing_eV"])
        for key, (count, split) in other.groups.items():
            self.placed(key, other.placements.get(key, []), count, split)

    def placed(self, prefix: str, placements: Sequence[Placement], count: int, split: int) -> None:
        """Adds ``count`` groups on a table's outer edges (``split`` of them split by the recomputation's own
        placement), with the ``placements`` of those it keeps."""
        groups = self.groups.setdefault(prefix, [0, 0])
        groups[0] += count
        groups[1] += split
        kept = self.placements.setdefault(prefix, [])
        if groups[0] > OUTER_PLACEMENTS:
            kept.clear()
        else:
            kept.extend(placements)

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
        """A table in every basis with its rounding terms (the module docstring): ``value`` (first basis),
        ``other.j`` and ``envelope.j`` (basis ``j``'s table, and the absolute difference of each k-point's tables
        in the first basis and in it, summed over the k-points), ``kernel`` (the kernel's own rounding), ``floor``
        (the broadened first-order rounding, with what a group's rotation leaves), ``absolute`` (the broadened magnitudes), ``slope`` and
        ``edge``."""
        base = modes[0]
        energies = [mode.energies(occupied) for mode in modes]
        # A transition on a discontinuity of the kernel is placed alike in every basis (its energy in the first one):
        # the edge terms below price either placement, the envelope measures the rest.
        on_edge = kernel.near_edge(energies[0].reshape(-1)).reshape(energies[0].shape)
        for index in range(1, len(energies)):
            energies[index] = np.where(on_edge, energies[0], energies[index])
        weights = [np.where(mask[..., None], mode.weights, 0.0) for mode in modes]
        magnitude = np.where(mask[..., None], base.magnitude, 0.0)
        assert base.rounding is not None
        noise = np.where(mask[..., None], base.rounding.noise, 0.0)
        # What a group's rotation leaves in the table: its transitions' own rounding over the group's spread in energy,
        # through the kernel's slope (a Gaussian), or whole in the two rows of a bin edge within that spread.
        spread = np.where(mask, base.rounding.spread, 0.0)
        residual = base.rounding.full * spread[..., None]
        grouped = bool(np.any(spread > 0))
        count = self.run.count
        points = energies[0].shape[0]
        pairs = int(np.prod(energies[0].shape[1:]))
        width = len(SHIFT_COMPONENTS)
        names = ["value", "floor", "kernel", "slope", "absolute"]
        names += [f"{kind}.{index}" for index in range(1, len(modes)) for kind in ("other", "envelope")]
        totals = {name: np.zeros((count, width)) for name in names}
        sizes = np.abs(base.shifted)
        scales = sizes[:, None, occupied:] + sizes[:, :occupied, None]
        step = max(1, KERNEL_ENTRIES // max(1, count * pairs))
        for start in range(0, points, step):
            part = slice(start, start + step)
            size = len(range(*part.indices(points)))
            first = kernel.value(energies[0][part].reshape(-1)).reshape(count, size, pairs)
            reference = np.einsum("wkp,kpc->kwc", first, weights[0][part].reshape(size, pairs, width))
            totals["value"] += reference.sum(axis=0)
            for index in range(1, len(modes)):
                matrix = kernel.value(energies[index][part].reshape(-1)).reshape(count, size, pairs)
                table = np.einsum("wkp,kpc->kwc", matrix, weights[index][part].reshape(size, pairs, width))
                totals[f"other.{index}"] += table.sum(axis=0)
                totals[f"envelope.{index}"] += np.abs(reference - table).sum(axis=0)
            flat_kernel = first.reshape(count, -1)
            totals["floor"] += flat_kernel @ noise[part].reshape(-1, width)
            totals["absolute"] += flat_kernel @ magnitude[part].reshape(-1, width)
            flat = energies[0][part].reshape(-1)
            absolute = np.abs(weights[0][part].reshape(-1, width))
            rounding: np.ndarray | None = None
            slope: np.ndarray | None = None
            if kernel.offset is not None:
                offset = kernel.offset(flat)
                rounding = flat_kernel * (1 + offset**2)
                slope = flat_kernel * (2 * np.abs(offset) / self.run.eta)
            else:
                rounding = None if kernel.rounding is None else kernel.rounding(flat)
                slope = None if kernel.slope is None else kernel.slope(flat)
            if rounding is not None:
                totals["kernel"] += rounding @ absolute
            if slope is not None:
                totals["slope"] += slope @ (magnitude[part].reshape(-1, width) * scales[part].reshape(-1, 1))
            if grouped:
                left = residual[part].reshape(-1, width)
                if kernel.offset is not None and slope is not None:
                    totals["floor"] += slope @ left
                elif kernel.distance is not None:
                    reach = spread[part].reshape(-1)
                    close = (kernel.distance(flat) < reach)[:, None] * left
                    totals["floor"] += kernel.value(flat + reach) @ close + kernel.value(flat - reach) @ close
        for name, value in totals.items():
            key = f"{prefix}.{name}"
            self.tables[key] = self.tables.get(key, 0.0) + value
        flat = energies[0].reshape(-1)
        signed = weights[0].reshape(-1, width)
        # A transition the table does not sum has no placement.
        near = kernel.near_edge(flat) & np.any(signed != 0, axis=1)
        for name in ("edge", "edge_cumulative", "outer_edge", "outer_cumulative"):
            self.tables.setdefault(f"{prefix}.{name}", np.zeros((count, width)))
        self.placed(prefix, [], 0, 0)
        if not near.any():
            return
        # A transition on a discontinuity of the kernel lands on either side of it: per row the difference of the two
        # placements, and, summed over the rows up to each one, what that choice moves (an interior edge only the row
        # below it). On an outer edge (one side lies outside the table) the choice moves the sum of every row beyond:
        # there each k-point's transitions have their two placements instead (Placement), and the allowance is kept
        # apart for a table whose placements are not tested exactly.
        chosen = np.flatnonzero(near)
        here = flat[chosen]
        place = kernel.placed or (lambda energies, side: kernel.value(side))
        below = place(here, here - 2 * EDGE_EV)
        above = place(here, here + 2 * EDGE_EV)
        jump = above - below
        amounts = np.abs(signed[chosen])
        empty_below = ~below.any(axis=0)
        outer = empty_below ^ ~above.any(axis=0)
        inner = ~outer
        self.tables[prefix + ".edge"] += np.abs(jump) @ amounts
        self.tables[prefix + ".edge_cumulative"] += np.abs(np.cumsum(jump[:, inner], axis=0)) @ amounts[inner]
        if not outer.any():
            return
        self.tables[prefix + ".outer_edge"] += np.abs(jump[:, outer]) @ amounts[outer]
        self.tables[prefix + ".outer_cumulative"] += (
            np.abs(np.cumsum(jump[:, outer], axis=0)) @ amounts[outer]
        )
        current = kernel.value(here[outer])
        inside = np.where(empty_below, above, below)[:, outer]
        counted = current.any(axis=0)
        transitions = chosen[outer]
        kpoint = transitions // pairs
        lower = empty_below[outer]
        found: list[Placement] = []
        split = 0
        keys = sorted(set(zip(kpoint.tolist(), lower.tolist(), strict=True)))
        for point, side in keys:
            group = (kpoint == point) & (lower == side)
            members = transitions[group]
            if counted[group].all():
                # Moved out: the table's own rounding already holds theirs.
                found.append(Placement(-current[:, group] @ signed[members], True, np.zeros((count, width))))
                continue
            if counted[group].any():
                split += 1
                continue
            # Moved in: the ordinary rounding of these transitions in the row they would take. Like the
            # table's own envelope, the basis difference is taken of the k-point's summed weight, which a
            # rotation inside a degenerate group leaves unchanged, not transition by transition.
            moved = inside[:, group]
            envelope = np.max(
                [
                    np.abs(moved @ (other.reshape(-1, width)[members] - signed[members]))
                    for other in weights[1:]
                ]
                or [np.zeros((count, width))],
                axis=0,
            )
            extra = noise.reshape(-1, width)[members]
            if grouped:
                extra = extra + residual.reshape(-1, width)[members]
            rounding = SPREAD_MULTIPLE * envelope + TERM_FLOOR * (moved @ extra)
            if kernel.offset is not None:
                offset = kernel.offset(flat[members])
                rounding += KERNEL_ROUNDING * ((moved * (1 + offset**2)) @ np.abs(signed[members]))
                rounding += ENERGY_ROUNDING * (
                    (moved * (2 * np.abs(offset) / self.run.eta))
                    @ (magnitude.reshape(-1, width)[members] * scales.reshape(-1, 1)[members])
                )
            found.append(Placement(moved @ signed[members], False, rounding))
        self.placed(prefix, found, len(keys), split)


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
    magnitudes: Bounds | None = None,
    group: float | None = None,
) -> tuple[_Mode, np.ndarray, np.ndarray]:
    levels, vectors = eigensystem(operators.h, operators.s)
    shifted = levels.copy()
    shifted[:, run.occupied :] += run.scissors
    kernels = shifted if scissored_kernels else levels
    values, magnitude, rounding = integrand(
        operators,
        kernels,
        vectors,
        run.occupied,
        treatment=treatment,
        threshold=threshold,
        regularization=regularization,
        magnitudes=magnitudes,
        group=group,
    )
    return (
        _Mode(
            levels,
            shifted,
            weight * values,
            abs(weight) * magnitude,
            None
            if rounding is None
            else Rounding(abs(weight) * rounding.noise, abs(weight) * rounding.full, rounding.spread),
        ),
        shifted,
        vectors,
    )


def _modes(
    sets: Sequence[Blocks],
    bases: Sequence[Basis],
    kpoints: np.ndarray,
    run: ShiftRun,
    *,
    construction: str,
    energy_scale: float,
    magnitudes: Bounds,
    group: float | None = None,
    **options: Any,
) -> tuple[list[_Mode], Operators, np.ndarray]:
    """A mode in every basis (``sets[j]``: the operators in basis ``j``'s order of the translations); the first
    basis's operators and eigenvectors."""
    modes: list[_Mode] = []
    base_operators: Operators | None = None
    base_vectors: np.ndarray | None = None
    for blocks, basis in zip(sets, bases, strict=True):
        operators = scaled(bloch_operators(blocks, kpoints, construction=construction), energy_scale)
        if basis.rotation is not None:
            operators = rotated(operators, basis.rotation)
        first = basis.rotation is None
        mode, _, vectors = _evaluate(
            operators,
            run,
            magnitudes=magnitudes if first else None,
            group=group if first else None,
            **options,
        )
        modes.append(mode)
        if basis.rotation is None:
            base_operators, base_vectors = operators, vectors
    assert base_operators is not None and base_vectors is not None
    return modes, base_operators, base_vectors


def _batch(context: _Context, prepared: _Prepared, kpoints: np.ndarray) -> _Sums:
    run = context.run
    kernels = Kernels(run)
    sums = _Sums(run)
    occupied = run.occupied
    scale = context.energy_scale
    queer = kernels.queer()
    binned = kernels.queer(binned=True)
    pyatb = kernels.pyatb()
    compared = binned if run.smearing == 0 else queer
    for index, sets in enumerate(prepared.physical):
        modes, base_operators, base_vectors = _modes(
            sets,
            context.bases,
            kpoints,
            run,
            construction="hermitian",
            energy_scale=1.0,
            magnitudes=prepared.physical_bounds[index],
            group=None
            if run.pair is not None
            else GROUP_SHARE * (run.eta if run.smearing == 1 else run.domega),
            treatment="covariant",
            threshold=context.exact,
            regularization=None,
            scissored_kernels=False,
            weight=-0.5,
        )
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
            values, _, _ = integrand(
                base_operators,
                base.shifted,
                base_vectors,
                occupied,
                treatment="covariant",
                threshold=context.exact,
            )
            scissored = np.where(mask[..., None], -0.5 * values, 0.0).reshape(-1, 18)
            sums.add("path.scissors", pyatb.value, energies, scissored)
    # PyATB's own table: its operators already in its eV.
    literal, _, _ = _modes(
        prepared.literal,
        context.bases,
        kpoints,
        run,
        construction="pyatb",
        energy_scale=1.0,
        magnitudes=prepared.literal_bounds,
        treatment="pyatb",
        threshold=PYATB_THRESHOLD,
        regularization=PYATB_REGULARIZATION,
        scissored_kernels=True,
        weight=-1.0,
    )
    base_p = literal[0]
    mask_p = _transitions_mask(run, base_p.shifted, None)
    sums.table("literal", pyatb, literal, mask_p, occupied)
    # Plain facts: PyATB's selected transitions, the k-points its method-1 cut regularizes and those where an exact
    # degeneracy takes part (a band of a selected pair within the exactness bound of another level).
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
    prepared = _prepare(context)
    total: _Sums | None = None
    for start in range(0, len(kpoints), context.batch):
        part = _batch(context, prepared, kpoints[start : start + context.batch])
        if total is None:
            total = part
        else:
            total.merge(part)
    assert total is not None
    return total


# --------------------------------------------------------------------------- the recomputation
@dataclass(frozen=True)
class Placement:
    """A k-point's transitions on one outer edge of a table, which the table counts (``inside``) or not: ``change``
    ``(count, 18)`` is what placing them all on the other side of the edge adds to the table (inside the table or
    outside it, nothing in between), and ``rounding`` the ordinary rounding they then add to its tolerance (that of
    their entries in the row they take; zero when they leave the table)."""

    change: np.ndarray
    inside: bool
    rounding: np.ndarray


@dataclass
class Table:
    """A table ``(count, 18)`` (microA/V^2) in the first orbital basis, in the second, the largest difference to it
    over the other bases (``spread``), its rounding tolerance at each entry, the same kernel of its transitions'
    magnitudes (``absolute``: the size of what cancels in it) and the part of the tolerance that transitions on a
    discontinuity of the kernel add (``edge``: either placement; ``edge_cumulative``: what that choice moves in the
    sum over the rows up to each one at its interior edges). ``smooth`` is the tolerance without the edges: how far
    rounding alone moves an entry.

    The transitions on its outer edges (where one placement lies outside the table) have their exact placements
    instead (``placements``, one per k-point and edge, of ``groups`` such groups, ``split`` of them split across the
    edge by the recomputation's own placement); ``outer_edge`` (part of ``edge``) and ``outer_cumulative`` are their
    allowance, for a table whose placements are not tested exactly (:attr:`placements_exact`)."""

    value: np.ndarray
    other: np.ndarray
    rounding: np.ndarray
    absolute: np.ndarray
    edge: np.ndarray
    edge_cumulative: np.ndarray
    spread: np.ndarray
    outer_edge: np.ndarray
    outer_cumulative: np.ndarray
    placements: tuple[Placement, ...] = ()
    groups: int = 0
    split: int = 0

    @property
    def smooth(self) -> np.ndarray:
        return self.rounding - self.edge

    @property
    def placements_exact(self) -> bool:
        """Whether every group on the outer edges has its two placements here: none split by the recomputation's
        own placement, at most :data:`OUTER_PLACEMENTS` of them."""
        return self.split == 0 and self.groups <= OUTER_PLACEMENTS and len(self.placements) == self.groups

    def scaled(self, factor: float) -> Table:
        size = abs(factor)
        return Table(
            factor * self.value,
            factor * self.other,
            size * self.rounding,
            size * self.absolute,
            size * self.edge,
            size * self.edge_cumulative,
            size * self.spread,
            size * self.outer_edge,
            size * self.outer_cumulative,
            tuple(
                Placement(factor * item.change, item.inside, size * item.rounding) for item in self.placements
            ),
            self.groups,
            self.split,
        )

    def __add__(self, other: Table) -> Table:
        groups = self.groups + other.groups
        return Table(
            self.value + other.value,
            self.other + other.other,
            self.rounding + other.rounding,
            self.absolute + other.absolute,
            self.edge + other.edge,
            self.edge_cumulative + other.edge_cumulative,
            self.spread + other.spread,
            self.outer_edge + other.outer_edge,
            self.outer_cumulative + other.outer_cumulative,
            self.placements + other.placements if groups <= OUTER_PLACEMENTS else (),
            groups,
            self.split + other.split,
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
    ("hartree", "PyATB's Hartree constant: every transition energy times 27.211324570273 / CODATA's Hartree"),
    ("window", "PyATB's window [start, start + count domega) on its scale (no band pair)"),
    (
        "bins",
        "PyATB's int((E - start)/domega) rows: its Gaussian truncated at int(5 eta/domega) rows, or its bins",
    ),
    ("scissors", "the scissored levels inside PyATB's derivative kernels"),
    (
        "cut",
        "PyATB's method-1 integrand: 1/dE replaced by dE/(dE^2 + 0.04^2) within 1e-4 eV, i rbar(j, j) on the "
        "diagonal, its degenerate velocities, and its Bloch sums of the upper triangles of the operators its "
        "reader builds (the records as written, Hermitized in real space for nspin 1)",
    ),
)


def work(records: Records, run: ShiftRun) -> float:
    """A recomputation's size, ``k-points x bands^2 x (bands + photon energies / PHOTON_SHARE)`` summed over its
    channels: the eigensystems and integrands grow as the cube of the bands, the kernels as the square of the bands
    times the photon energies (measured on the calibration scfout, 30 bands and 51 to 2401 photon energies: about 3e6
    a second on one core either way)."""
    bands = float(records.bands)
    return float(run.kpoints) * bands**2 * (bands + run.count / PHOTON_SHARE) * records.channels


def recompute(
    records: Records,
    run: ShiftRun,
    *,
    workers: int | None = None,
    memory_bytes: float | None = None,
    log: Callable[[str], None] | None = None,
) -> Recomputation:
    """QUEER's physical spectrum and PyATB's table of ``run`` on the scfout's ``records``
    (:func:`~queer.compat.generator.shift_scfout.read_records`), each in :data:`BASES` orbital bases, with their
    rounding tolerances and the attribution path (the module docstring); the lattice's handedness is the records'.
    ``workers`` (None: every CPU) evaluate chunks of the grid on QUEER's process executor."""
    import time

    from ...execution import ExecutionOptions
    from .memory import chunks, plan_parallelism, pmap

    if run.smearing not in (0, 1):
        raise ValueError("only smearing_method 0 and 1 are recomputed")
    started = time.perf_counter()
    physical = tuple(physical_channels(records))
    literal = literal_channel(records)
    size = records.bands
    if not 0 < run.occupied < size:
        raise ValueError(f"occ_band {run.occupied} must lie between 0 and the {size} bands PyATB reads")
    hartree = codata_hartree()
    scale = PYATB_HARTREE / hartree
    context = _Context(
        run=run,
        physical=physical,
        literal=literal,
        exact=exactness_bound(records),
        energy_scale=scale,
        bases=evaluation_bases(size, len(records.translations)),
    )
    kpoints = kpoint_grid(run.grid)
    copies = 2 * BASES * (len(physical) + 1)
    fixed = copies * physical[0].hamiltonian.nbytes * (5 if physical[0].position is not None else 2)
    serial = work(records, run) < SERIAL_WORK
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
    sums = _Sums(run)
    for part in parts:
        sums.merge(part)
    tables, facts = sums.tables, sums.facts
    volume = abs(float(np.linalg.det(records.lattice))) * codata_bohr() ** 3
    conversion = -math.pi * QUEER_E_CHARGE**2 / QUEER_HBAR * 1e6 / volume / run.kpoints
    ratio = (
        (PYATB_E_CHARGE**2 / PYATB_HBAR)
        * (PYATB_E_CHARGE * PYATB_EV_SECONDS / PYATB_HBAR)
        / (QUEER_E_CHARGE**2 / QUEER_HBAR)
    )
    mode = records.spin_mode
    handedness = records.handedness
    spin_factor = 2 if mode != "soc" else 1
    degeneracy = 2 if mode == "nonmagnetic" else 1

    def table(prefix: str) -> Table:
        others = range(1, BASES)
        envelope = np.max([tables[f"{prefix}.envelope.{index}"] for index in others], axis=0)
        spread = np.max(
            [np.abs(tables[prefix + ".value"] - tables[f"{prefix}.other.{index}"]) for index in others],
            axis=0,
        )
        rounding = (
            SPREAD_MULTIPLE * envelope
            + TERM_FLOOR * tables[prefix + ".floor"]
            + KERNEL_ROUNDING * tables[prefix + ".kernel"]
            + ENERGY_ROUNDING * tables[prefix + ".slope"]
            + tables[prefix + ".edge"]
        )
        groups, split = sums.groups[prefix]
        return Table(
            tables[prefix + ".value"],
            tables[prefix + ".other.1"],
            rounding,
            tables[prefix + ".absolute"],
            tables[prefix + ".edge"],
            tables[prefix + ".edge_cumulative"],
            spread,
            tables[prefix + ".outer_edge"],
            tables[prefix + ".outer_cumulative"],
            tuple(sums.placements[prefix]),
            groups,
            split,
        )

    names = ["sigma"] + (["binned"] if run.smearing == 0 else [])
    channel_tables = [
        {name: table(f"physical{index}.{name}").scaled(conversion) for name in names}
        for index in range(len(physical))
    ]
    tables_out: dict[str, Table] = {}
    for name in names:
        total = channel_tables[0][name].scaled(degeneracy)
        for extra in channel_tables[1:]:
            total = total + extra[name]
        tables_out[name] = total
    if mode == "collinear":
        tables_out["legacy"] = channel_tables[0]["sigma"].scaled(spin_factor)
        if run.smearing == 0:
            tables_out["legacy_binned"] = channel_tables[0]["binned"].scaled(spin_factor)
    layout = "binned" if run.smearing == 0 else "sigma"
    compared = channel_tables[0][layout].scaled(handedness * spin_factor)
    # PyATB's prefactor: -t pi e^3 / (2 hbar^2 V N_k) 1e6 with the signed volume (its Bohr), times its spin factor.
    signed = handedness * abs(float(np.linalg.det(records.lattice))) * PYATB_BOHR**3
    prefactor = (
        -PYATB_EV_SECONDS * math.pi * PYATB_E_CHARGE**3 / (2 * PYATB_HBAR**2 * signed * run.kpoints) * 1e6
    ) * spin_factor
    literal_table = table("literal").scaled(prefactor)
    steady = handedness * spin_factor * ratio * conversion
    path = [("queer", compared.value), ("constants", ratio * compared.value)]
    path.append(("hartree", steady * tables["path.hartree"]))
    path.append(("window", steady * tables["path.window"]))
    path.append(("bins", steady * tables["path.bins"]))
    path.append(("scissors", steady * tables["path.scissors"] if run.scissors else path[-1][1]))
    path.append(("cut", literal_table.value))
    facts.update(
        {
            "kpoints": run.kpoints,
            "bands": size,
            "channels": len(physical),
            "work": work(records, run),
            "workers": int(options.workers),
            "chunk": int(chunk),
            "seconds": time.perf_counter() - started,
            "exactness_bound_eV": context.exact,
            "energy_scale": scale,
            "codata_hartree_eV": hartree,
            "codata_bohr_Angstrom": codata_bohr(),
            "records_residuals": residuals(records),
            "constant_ratio": ratio,
            "bases": BASES,
            "spread_multiple": SPREAD_MULTIPLE,
            "term_floor": TERM_FLOOR,
            "edge_eV": EDGE_EV,
            "outer_placements": OUTER_PLACEMENTS,
            "basis_seed": BASIS_SEED,
        }
    )
    if not math.isfinite(facts["min_spacing_eV"]):
        facts["min_spacing_eV"] = None
    if log is not None:
        log(
            f"shift-current recomputation: {run.kpoints} k-points x {size} bands x {len(physical)} channel(s) in "
            f"{facts['seconds']:.1f} s on {facts['workers']} worker(s)"
        )
    return Recomputation(
        physical=tables_out, compared=compared, literal=literal_table, path=path, facts=facts
    )
