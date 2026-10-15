"""Wilson-loop invariants: phase-winding Chern numbers and the Z2 largest-gap rule.

Inputs are the Berry phases (radian) of closed k-loops and their Wannier charge
centres (WCCs) in units of the loop's lattice vector. Loop construction and
eigensolves belong to :mod:`queer.observables.topology`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import NamedTuple

import numpy as np


class Z2Flow(NamedTuple):
    """Largest-gap Z2 of a WCC flow over half a time-reversal-invariant plane."""

    z2: int
    crossings: int
    ambiguous: bool
    gap_midpoints: np.ndarray
    min_largest_gap: float | None


def plane_axes(fixed_axis: int) -> tuple[int, int]:
    """The (loop, transverse) reciprocal axes of the plane normal to ``fixed_axis``.

    The cyclic order (a, b, c) = (c+1, c+2, c) makes the Chern number of the
    plane k_c = const the component C_c of the Chern vector.
    """
    if isinstance(fixed_axis, bool) or fixed_axis not in (0, 1, 2):
        raise ValueError("fixed_axis must be a reciprocal axis index 0, 1 or 2")
    c = int(fixed_axis)
    return (c + 1) % 3, (c + 2) % 3


def wrap_phase(phase):
    """Phases mapped into [-pi, pi)."""
    return (np.asarray(phase, float) + np.pi) % (2.0 * np.pi) - np.pi


def chern_from_phases(phases) -> tuple[float, float]:
    """Chern number from the Berry phases of loops along a closed transverse axis.

    ``phases[j]`` is phi_j = -Im ln det W_j of the loop at transverse point
    t_j = j/n, so C = -(1/2pi) sum_j wrap(phi_(j+1) - phi_j) with the last step
    closing on the first loop. Returns ``(C, largest |step|)``: the winding is
    resolved only while every step stays well below pi.
    """
    phi = np.asarray(phases, float)
    if phi.ndim != 1 or not len(phi) or not np.isfinite(phi).all():
        raise ValueError("phases must be a nonempty finite one-dimensional array")
    steps = wrap_phase(np.diff(np.append(phi, phi[0])))
    return float(-steps.sum() / (2.0 * np.pi)), float(np.abs(steps).max())


def largest_gap_midpoint(centres) -> tuple[float, float]:
    """Midpoint and width of the largest gap between centres on the unit circle."""
    x = np.sort(np.mod(np.asarray(centres, float), 1.0))
    if len(x) == 0:
        return 0.5, 1.0
    gaps = np.diff(np.append(x, x[0] + 1.0))
    index = int(np.argmax(gaps))
    return float((x[index] + gaps[index] / 2.0) % 1.0), float(gaps[index])


def z2_from_wcc(wcc_flow: Iterable, pos_tol: float = 0.01) -> Z2Flow:
    """Z2 from the WCC flow over t in [0, 1/2] of a time-reversal-invariant plane.

    Soluyanov and Vanderbilt's largest-gap rule (PRB 83, 235401): the midpoint
    of the largest gap between centres is followed from one transverse point to
    the next, counting the centres of the next point in the half-open interval
    between the two midpoints; Z2 is the parity of that count. Kramers pairs
    make the count's parity independent of the arc. A centre within
    ``pos_tol`` (circular distance) of either midpoint marks the step, and the
    result, ambiguous: the transverse sampling is too coarse there.
    """
    flow = [np.mod(np.asarray(centres, float), 1.0) for centres in wcc_flow]
    midpoints = [largest_gap_midpoint(centres) for centres in flow]
    crossings = 0
    ambiguous = False
    for j in range(len(flow) - 1):
        z0, z1 = midpoints[j][0], midpoints[j + 1][0]
        low, high = min(z0, z1), max(z0, z1)
        following = flow[j + 1]
        crossings += int(np.sum((following >= low) & (following < high)))
        distance = np.abs(following[:, None] - np.array([z0, z1])[None, :])
        if np.any(np.minimum(distance, 1.0 - distance) < pos_tol):
            ambiguous = True
    return Z2Flow(
        z2=int(crossings % 2),
        crossings=int(crossings),
        ambiguous=bool(ambiguous),
        gap_midpoints=np.asarray([midpoint for midpoint, _ in midpoints], float),
        min_largest_gap=float(min(gap for _, gap in midpoints)) if midpoints else None,
    )


def kramers_degenerate(energies, tol: float) -> tuple[bool, float]:
    """Whether every level comes in a pair split by less than ``tol``, and the largest split.

    Levels are sorted along the last axis and paired (0, 1), (2, 3), ...; an odd
    count cannot pair and reports an infinite split.
    """
    levels = np.sort(np.asarray(energies, float), axis=-1)
    if levels.shape[-1] % 2:
        return False, float("inf")
    split = float(np.abs(levels[..., 1::2] - levels[..., 0::2]).max()) if levels.size else 0.0
    return bool(split < tol), split


def z2_indices(z2_by_plane: Mapping[tuple[int, float], int | None]) -> dict:
    """Strong and weak Z2 indices from the six time-reversal-invariant planes.

    ``z2_by_plane[(c, v)]`` is the plane k_c = v (v = 0 or 1/2), or None when it
    is unknown. nu_c is the k_c = 1/2 index and nu_0 = Z2(k_c = 0) + Z2(k_c = 1/2)
    mod 2 for every axis with both planes known; ``consistent`` is False when
    the axes disagree on nu_0, which then is None.
    """
    indices: dict = {}
    strong = set()
    for axis in range(3):
        zero, half = z2_by_plane.get((axis, 0.0)), z2_by_plane.get((axis, 0.5))
        if zero is not None and half is not None:
            strong.add((zero + half) % 2)
        indices[f"nu{axis + 1}"] = half
    indices["nu0"] = next(iter(strong)) if len(strong) == 1 else None
    indices["consistent"] = len(strong) <= 1
    return indices
