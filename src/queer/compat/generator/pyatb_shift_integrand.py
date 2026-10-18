"""PyATB's ``method 1`` shift-current integrand at single k-points, evaluated independently of QUEER's estimator.

A comparison uses it to recompute k-points of a QUEER run from its scfout
(:mod:`~queer.compat.generator.parity_shift`): it reads only the model's
real-space blocks (its own Bloch sums), solves the generalized eigenproblem itself
and follows PyATB's sum-over-states route (``shift_current_solver::get_Inm_ik_sumOver``
with ``linear_response::get_D_degenerate``) as published in PyATB's papers and
source, written here from those equations. None of QUEER's band geometry,
covariant derivatives or shift-current code runs.

At one k-point, with ``H C = S C E`` (``C^dagger S C = 1``) and the band-basis
projections ``Xbar_a = C^dagger (d_a X) C``, ``M_a(i, j) = (Hbar_a - E_j Sbar_a)(i, j)``:

- ``D_a(i, j) = -M_a(i, j) g(E_i - E_j)`` (``g(x) = 1 / x``) between levels farther
  apart than the threshold, the eigenvector derivative ``d_a C = C D_a``;
- ``r_mn^a = i D_a(m, n) + rbar_a(m, n)`` for an occupied ``m`` and an empty ``n``,
  where ``rbar_a = C^dagger r_a^dagger C``;
- ``d_b`` of a band-basis projection at ``(n, m)``: ``(D_b^dagger Xbar_a)(n, m) +
  Xbar_ab(n, m) + (Xbar_a D_b)(n, m)``;
- ``r_nm;b^a = i [(d_b Hbar_a - E_m d_b Sbar_a - Sbar_a V_b) d - (M_a V_b - V_b
  M_a)] / d^2 + d_b rbar_a`` at ``(n, m)``, ``d = E_m - E_n``, ``V_b`` the band
  velocities;
- ``I^{a, bc} = Im(r_mn^b r_nm;a^c + r_mn^c r_nm;a^b)`` for PyATB's 18 components.

Levels within the threshold of each other form a cluster, treated in one of two ways:

- ``degenerate="pyatb"``, PyATB's own: ``D_a(i, j) = -M_a(i, j) x / (x^2 + eta_r^2)``
  inside a cluster (``x = E_i - E_j``, ``eta_r = regularization_eV``; 0 without a
  regularization, an exact degeneracy), ``i rbar_a(i, i)`` on the diagonal, and
  ``V_b`` diagonal: ``M_b(j, j)``, or inside a cluster the eigenvalues of its block
  in ascending order at ``j``'s offset (PyATB's degenerate perturbation theory).
  At an exact degeneracy in a nonorthogonal basis or with position moments this
  depends on the eigensolver's basis inside the cluster, as PyATB's own value does;
- ``degenerate="covariant"``, the physical integrand of an exact degeneracy:
  parallel transport inside the cluster (``D_a = -Sbar_a / 2 + i (rbar_a +
  rbar_a^dagger) / 2`` there, which normalizes the states and makes the connection
  vanish inside it) and ``V_b`` the cluster's velocity matrix ``M_b``; summed over a
  cluster's transitions this is independent of the basis inside it, QUEER's
  covariant estimator by another route.

PyATB adds ``(f_n - f_m) I = -I`` per pair to its Gaussian or bins and multiplies
by ``-pi e^2 / (2 hbar V N_k)`` (its spin factor aside); QUEER's estimator sums
``(f_n - f_m) Im[r^b r^c_;a]_sym = I / 2`` per pair times ``-pi e^2 / (hbar V N_k)``:
:func:`queer_weights` returns ``-I / 2`` per transition so that, times QUEER's
conversion, it is a k-point's contribution in QUEER's normalization.

The size of what cancels: ``magnitude`` holds ``|r_mn^b| |r_nm;a^c| + |r_mn^c|
|r_nm;a^b|`` per pair and component, which bounds ``|I|``, and ``magnitude_bound``
their sum per k-point with each pair of clusters' moduli replaced by Frobenius
norms, which no basis inside a cluster exceeds. Where the integrand or one of its
factors vanishes by symmetry (a time-reversal-invariant k-point of a
centrosymmetric crystal) these vanish too and say nothing of its rounding; the
same evaluation in another orbital basis (``basis``, :func:`rotated`) does: the two
differ by rounding alone.

PyATB works in its own eV (``energy_scale`` = PyATB's Hartree over CODATA's, times
QUEER's eV); its threshold and ``eta_r`` apply there. Transition energies come back
in QUEER's eV.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import eigh, eigvalsh

from .pyatb_files import SHIFT_COMPONENTS

_AXES = {"x": 0, "y": 1, "z": 2}
# PyATB's component order: a over x, y, z; (b, c) over xx, xy, xz, yy, yz, zz.
_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
_ORDER = tuple((a, b, c) for a in range(3) for b, c in _PAIRS)
assert tuple("".join("xyz"[i] for i in triple) for triple in _ORDER) == SHIFT_COMPONENTS
DEGENERATE_MODES = ("pyatb", "covariant")


@dataclass(frozen=True)
class BlochOperators:
    """The model's operators at one k-point and their Cartesian k-derivatives (QUEER's eV and Angstrom)."""

    hamiltonian: np.ndarray
    overlap: np.ndarray
    d_hamiltonian: np.ndarray
    d_overlap: np.ndarray
    d2_hamiltonian: np.ndarray
    d2_overlap: np.ndarray
    position: np.ndarray
    d_position: np.ndarray


def _blocks(blocks: Mapping[Any, Any], size: int) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted(blocks)
    translations = np.array(keys, dtype=float).reshape(-1, 3)
    matrices = np.zeros((len(keys), size, size), complex)
    for index, key in enumerate(keys):
        block = blocks[key]
        matrices[index] = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
    return translations, matrices


def _hermitian(matrix: np.ndarray) -> np.ndarray:
    return (matrix + np.swapaxes(matrix.conj(), -1, -2)) / 2


def bloch_operators(
    model: Any, k: Sequence[float] | np.ndarray, *, derivatives: bool = True
) -> BlochOperators:
    """``X(k) = sum_R exp(2 pi i k.R) X(R)``, ``d_a X = sum_R i R_a ...`` and ``d_a d_b X = -sum_R R_a R_b ...``.

    ``R_a`` is the Cartesian translation (Angstrom). The Hamiltonian, the overlap and
    their derivatives are Hermitian parts; the position operator and its derivative
    are the plain Bloch sums of the model's position blocks (zero for a model without
    them whose overlap is the identity at R = 0 only). Without ``derivatives`` only
    the Hamiltonian and the overlap are summed (the rest are empty arrays).
    """
    point = np.asarray(k, float)
    size = int(model.nbnd)
    lattice = np.asarray(model.lattice, float)
    out: dict[str, np.ndarray] = {}
    empty = np.zeros((0, size, size), complex)
    for name, blocks in (("h", model.hamiltonian_blocks), ("s", model.overlap_blocks)):
        translations, matrices = _blocks(blocks, size)
        phase = np.exp(2j * np.pi * translations @ point)
        out[name] = _hermitian(np.einsum("r,rij->ij", phase, matrices))
        if not derivatives:
            out["d" + name] = out["dd" + name] = empty
            continue
        cartesian = translations @ lattice
        out["d" + name] = _hermitian(np.einsum("ra,r,rij->aij", 1j * cartesian, phase, matrices))
        out["dd" + name] = _hermitian(np.einsum("ra,rb,r,rij->abij", -cartesian, cartesian, phase, matrices))
    position = np.zeros((3, size, size), complex)
    d_position = np.zeros((3, 3, size, size), complex)
    if not derivatives:
        return BlochOperators(out["h"], out["s"], empty, empty, empty, empty, empty, empty)
    if getattr(model, "position_blocks", ()):
        for axis, blocks in enumerate(model.position_blocks):
            translations, matrices = _blocks(blocks, size)
            phase = np.exp(2j * np.pi * translations @ point)
            cartesian = translations @ lattice
            position[axis] = np.einsum("r,rij->ij", phase, matrices)
            # d_position[b, a] = d_b r_a
            d_position[:, axis] = np.einsum("rb,r,rij->bij", 1j * cartesian, phase, matrices)
    elif not np.allclose(out["s"], np.eye(size)) or np.abs(out["ds"]).max(initial=0.0) > 0:
        raise ValueError(
            "a nonorthogonal model needs its position blocks for PyATB's shift-current integrand"
        )
    return BlochOperators(
        out["h"], out["s"], out["dh"], out["ds"], out["ddh"], out["dds"], position, d_position
    )


def rotated(operators: BlochOperators, basis: np.ndarray) -> BlochOperators:
    """Every operator in another orbital basis, ``Q^dagger X Q`` for a unitary ``Q``: the same levels and integrand,
    reached through other floating-point operations (their spread measures the rounding)."""
    left = np.asarray(basis).conj().T

    def change(operator: np.ndarray) -> np.ndarray:
        return left @ operator @ basis if operator.size else operator

    return BlochOperators(*(change(getattr(operators, name)) for name in BlochOperators.__dataclass_fields__))


def levels(model: Any, k: Sequence[float] | np.ndarray) -> np.ndarray:
    """The generalized eigenvalues ``H C = S C E`` at ``k`` (QUEER's eV, ascending), from this module's Bloch sums."""
    operators = bloch_operators(model, k, derivatives=False)
    return np.asarray(eigvalsh(operators.hamiltonian, operators.overlap), float)


def clusters(energies: np.ndarray, threshold: float) -> np.ndarray:
    """The cluster of each level (ascending energies): consecutive levels closer than ``threshold`` join one."""
    labels = np.zeros(len(energies), int)
    for index in range(1, len(energies)):
        labels[index] = labels[index - 1] + int(abs(energies[index] - energies[index - 1]) > threshold)
    return labels


def _inverse_gaps(energies: np.ndarray, threshold: float, regularization: float | None) -> np.ndarray:
    """PyATB's ``1 / (E_i - E_j)`` with its rule within ``threshold``: ``x / (x^2 + eta^2)``, or 0."""
    gap = energies[:, None] - energies[None, :]
    near = np.abs(gap) <= threshold
    safe = np.where(near, 1.0, gap)
    inverse = 1.0 / safe
    within = 0.0 * gap if regularization is None else gap / (gap**2 + regularization**2)
    return np.where(near, within, inverse)


def _band_velocities(
    projection: tuple[np.ndarray, np.ndarray], energies: np.ndarray, threshold: float
) -> np.ndarray:
    """``dE(j)``: the diagonal of ``Hbar - E_j Sbar``, or, when adjacent levels lie closer than ``threshold``
    to ``E_j`` (scanning outwards from ``j``), the eigenvalues of that block in ascending order at ``j``'s
    offset (PyATB's degenerate perturbation theory)."""
    hbar, sbar = projection
    count = len(energies)
    velocity = np.empty(count)
    for band in range(count):
        low = band
        while low > 0 and abs(energies[band] - energies[low - 1]) < threshold:
            low -= 1
        high = band
        while high + 1 < count and abs(energies[high + 1] - energies[band]) < threshold:
            high += 1
        block = hbar[low : high + 1, low : high + 1] - energies[band] * sbar[low : high + 1, low : high + 1]
        if low == high:
            velocity[band] = float(np.real(block[0, 0]))
        else:
            velocity[band] = float(np.linalg.eigvalsh(_hermitian(block))[band - low])
    return velocity


def shift_integrand(
    model: Any,
    k: Sequence[float] | np.ndarray,
    pairs: Sequence[tuple[int, int]] | None,
    *,
    energy_scale: float,
    threshold_eV: float,
    regularization_eV: float | None,
    occupied: int,
    scissors_eV: float = 0.0,
    window: tuple[float, float] | None = None,
    degenerate: str = "pyatb",
    gauge: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    basis: np.ndarray | None = None,
) -> dict[str, Any]:
    """PyATB's integrand ``I`` (``(pairs, 18)``, Angstrom^3) of the transitions at ``k`` and their energies (QUEER's eV).

    ``pairs`` lists ``(occupied m, empty n)`` band indices; None selects every occupied
    (``m < occupied``) to empty pair whose scissored energy on PyATB's scale lies in
    ``window = [start, end)`` (PyATB's rule), or, without a window, exceeds ``threshold_eV``
    (QUEER's: every pair above its degeneracy tolerance). ``threshold_eV`` and
    ``regularization_eV`` act on PyATB's scale (``regularization_eV`` None: an exact
    degeneracy, no term); ``degenerate`` chooses PyATB's treatment of a cluster or the
    covariant one (module docstring; the covariant one needs ``regularization_eV`` None).
    ``gauge`` (tests) maps the eigenvectors and energies to other eigenvectors of the
    same levels. Also returns the levels (QUEER's eV, scissored), each level's cluster,
    the magnitudes of what cancels (module docstring) and the smallest spacing between a
    level of a pair and any other level (QUEER's eV).
    """
    if degenerate not in DEGENERATE_MODES:
        raise ValueError(f"degenerate must be one of {DEGENERATE_MODES}")
    if degenerate == "covariant" and regularization_eV is not None:
        raise ValueError("the covariant treatment is that of an exact degeneracy: no regularization")
    operators = bloch_operators(model, k)
    if basis is not None:
        operators = rotated(operators, basis)
    raw, vectors = eigh(operators.hamiltonian, operators.overlap)
    if gauge is not None:
        vectors = gauge(vectors, raw)
    energies = raw * energy_scale
    if scissors_eV:
        energies = energies.copy()
        energies[occupied:] += scissors_eV
    if pairs is None and window is not None:
        start, end = window
        pairs = [
            (m, n)
            for n in range(occupied, len(energies))
            for m in range(occupied)
            if start <= energies[n] - energies[m] < end
        ]
    elif pairs is None:
        pairs = [
            (m, n)
            for n in range(occupied, len(energies))
            for m in range(occupied)
            if energies[n] - energies[m] > threshold_eV
        ]
    pairs = [(int(m), int(n)) for m, n in pairs]
    labels = clusters(energies, threshold_eV)
    result: dict[str, Any] = {
        "levels_eV": energies / energy_scale,
        "clusters": labels,
        "pairs": pairs,
        "energies_eV": np.array([(energies[n] - energies[m]) / energy_scale for m, n in pairs]),
        "integrand": np.zeros((len(pairs), len(SHIFT_COMPONENTS))),
        "magnitude": np.zeros((len(pairs), len(SHIFT_COMPONENTS))),
        "magnitude_bound": np.zeros(len(SHIFT_COMPONENTS)),
    }
    selected = sorted({band for pair in pairs for band in pair})
    spacing = np.abs(energies[:, None] - energies[None, :])
    np.fill_diagonal(spacing, np.inf)
    result["min_spacing_eV"] = float(spacing[selected].min(initial=np.inf)) / energy_scale
    if not pairs:
        return result
    adjoint = vectors.conj().T
    scaled_h = energy_scale * operators.d_hamiltonian
    hbar = np.einsum("ij,ajk,kl->ail", adjoint, scaled_h, vectors)
    sbar = np.einsum("ij,ajk,kl->ail", adjoint, operators.d_overlap, vectors)
    rbar = np.einsum("ij,ajk,kl->ail", adjoint, operators.position.conj().swapaxes(-1, -2), vectors)
    first = hbar - sbar * energies[None, None, :]
    if degenerate == "pyatb":
        inverse = _inverse_gaps(energies, threshold_eV, regularization_eV)
        derivative = -first * inverse[None]
        for axis in range(3):
            np.fill_diagonal(derivative[axis], 1j * np.diagonal(rbar[axis]))
        diagonal = np.array([_band_velocities((hbar[b], sbar[b]), energies, threshold_eV) for b in range(3)])
        velocity = np.zeros((3, len(energies), len(energies)), complex)
        for b in range(3):
            np.fill_diagonal(velocity[b], diagonal[b])
    else:
        same = labels[:, None] == labels[None, :]
        inverse = _inverse_gaps(energies, threshold_eV, None)
        transport = -0.5 * sbar + 0.5j * (rbar + rbar.conj().swapaxes(-1, -2))
        derivative = np.where(same[None], transport, -first * inverse[None])
        velocity = _hermitian(
            np.where(same[None], hbar - 0.5 * sbar * (energies[:, None] + energies[None, :]), 0.0)
        )
    occupied_index = np.array([m for m, _ in pairs])
    empty_index = np.array([n for _, n in pairs])
    dipole = 1j * derivative[:, occupied_index, empty_index] + rbar[:, occupied_index, empty_index]
    hbar2 = np.einsum("ij,abjk,kl->abil", adjoint, energy_scale * operators.d2_hamiltonian, vectors)
    sbar2 = np.einsum("ij,abjk,kl->abil", adjoint, operators.d2_overlap, vectors)
    # d_b r_a^dagger: the adjoint of d_b r_a, projected.
    rbar2 = np.einsum("ij,bajk,kl->abil", adjoint, operators.d_position.conj().swapaxes(-1, -2), vectors)
    gap = energies[occupied_index] - energies[empty_index]
    energy_m = energies[occupied_index]
    generalized = np.zeros((3, 3, len(pairs)), complex)
    for a in range(3):
        for b in range(3):
            left = derivative[b].conj().T
            d_h = (left @ hbar[a] + hbar2[a, b] + hbar[a] @ derivative[b])[empty_index, occupied_index]
            d_s = (left @ sbar[a] + sbar2[a, b] + sbar[a] @ derivative[b])[empty_index, occupied_index]
            d_r = (left @ rbar[a] + rbar2[a, b] + rbar[a] @ derivative[b])[empty_index, occupied_index]
            overlap_v = (sbar[a] @ velocity[b])[empty_index, occupied_index]
            commutator = (first[a] @ velocity[b] - velocity[b] @ first[a])[empty_index, occupied_index]
            numerator = (d_h - energy_m * d_s - overlap_v) * gap - commutator
            generalized[a, b] = 1j * numerator / gap**2 + d_r
    for column, (a, b, c) in enumerate(_ORDER):
        result["integrand"][:, column] = np.imag(
            dipole[b] * generalized[c, a] + dipole[c] * generalized[b, a]
        )
        result["magnitude"][:, column] = np.abs(dipole[b]) * np.abs(generalized[c, a]) + np.abs(
            dipole[c]
        ) * np.abs(generalized[b, a])
    result["magnitude_bound"] = _magnitude_bound(labels, occupied_index, empty_index, dipole, generalized)
    return result


def _magnitude_bound(
    labels: np.ndarray,
    occupied_index: np.ndarray,
    empty_index: np.ndarray,
    dipole: np.ndarray,
    generalized: np.ndarray,
) -> np.ndarray:
    """``sum_{M, N} ||r^b_MN|| ||r^c_NM;a|| + ||r^c_MN|| ||r^b_NM;a||`` over the pairs of clusters the transitions
    join (Frobenius norms over each pair of clusters): at least the sum of the moduli in any basis inside them."""
    bound = np.zeros(len(SHIFT_COMPONENTS))
    keys = list(zip(labels[occupied_index].tolist(), labels[empty_index].tolist()))
    for key in sorted(set(keys)):
        members = np.array([index for index, other in enumerate(keys) if other == key])
        norm_r = np.sqrt(np.sum(np.abs(dipole[:, members]) ** 2, axis=-1))
        norm_g = np.sqrt(np.sum(np.abs(generalized[:, :, members]) ** 2, axis=-1))
        for column, (a, b, c) in enumerate(_ORDER):
            bound[column] += norm_r[b] * norm_g[c, a] + norm_r[c] * norm_g[b, a]
    return bound


def queer_weights(integrand: np.ndarray) -> np.ndarray:
    """``(f_n - f_m) Im[r^b r^c_;a]_sym`` in QUEER's normalization from PyATB's ``I``: ``-I / 2``."""
    return -0.5 * np.asarray(integrand, float)


def queer_magnitudes(magnitude: np.ndarray) -> np.ndarray:
    """The magnitudes of :func:`shift_integrand` in QUEER's normalization (``|queer_weights| <= this``): ``/ 2``."""
    return 0.5 * np.asarray(magnitude, float)
