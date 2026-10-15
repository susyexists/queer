"""Estimators and conventions of the Generator's PythTB engine, kept exactly.

These reproduce the numerical choices of the Generator's ``pythtb_core`` driver,
so a QUEER run computes what a PYTHTB step computed from the same inputs: the
histogram DOS with its discrete Gaussian and DOS(E_F) rules, the band-edge and
gap classification, the band.dat comparison window, wannier90-style path
sampling and surface paths, the orbital species, the hopping-decay fit, and the
per-state slab edge weights and crossing counts. Where a choice is basis- or
sampling-dependent the docstring says so; the physical counterparts live in the
library. Energies are relative to the Fermi level and k-points are fractional
unless stated otherwise.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ...physics.slab import edge_cells_used
from ...physics.topology import largest_gap_midpoint

DEFAULT_WINDOW_EV = 2.0
DOS_AT_FERMI_RULES = ("linear_interpolation", "first_nonnegative_bin")


def _reciprocal(lattice) -> np.ndarray:
    # The driver's own expression, so rounding of sampled path lengths matches.
    return 2.0 * np.pi * np.linalg.inv(np.asarray(lattice, dtype=float)).T


# ---------------------------------------------------------------------------
# k-space sampling
# ---------------------------------------------------------------------------


def uniform_mesh(
    divisions: Sequence[int], periodic: Sequence[int], fixed: Mapping[int, float] | None = None
) -> np.ndarray:
    """Points i/n over [0, 1) on the periodic axes, one value elsewhere (``fixed`` or 0); (Nk, 3)."""
    fixed = fixed or {}
    axes = []
    for axis in range(3):
        if axis in fixed:
            axes.append(np.array([float(fixed[axis])]))
        elif axis in periodic:
            n = max(1, int(divisions[axis]))
            axes.append(np.arange(n, dtype=float) / n)
        else:
            axes.append(np.array([0.0]))
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=1)


def sample_path(segments: Sequence[Sequence[Any]], lattice, n_points: int) -> dict[str, Any]:
    """Sample a path like wannier90: points per segment proportional to its length.

    ``segments`` rows are ``[label_a, k_a, label_b, k_b]`` (fractional). A join
    with the previous segment's end is not duplicated; any other start opens a
    new run, recorded in ``break_indices``. Returns ``kpoints``,
    ``tick_indices``, ``tick_labels`` and ``break_indices``.
    """
    reciprocal = _reciprocal(lattice)
    rows = [(str(s[0]), np.asarray(s[1], float), str(s[2]), np.asarray(s[3], float)) for s in segments]
    if not rows:
        raise ValueError("empty k-path")
    lengths = [float(np.linalg.norm((kb - ka) @ reciprocal)) for _, ka, _, kb in rows]
    total = sum(lengths) or 1.0
    points: list[np.ndarray] = []
    ticks: list[int] = []
    labels: list[str] = []
    breaks: list[int] = []
    for index, ((la, ka, lb, kb), length) in enumerate(zip(rows, lengths)):
        n = max(2, int(round(max(1, n_points) * length / total)))
        ts = np.linspace(0.0, 1.0, n + 1)
        if index > 0 and np.allclose(rows[index - 1][3], ka, atol=1e-6):
            ts = ts[1:]
        else:
            if index > 0:
                breaks.append(len(points))
            ticks.append(len(points))
            labels.append(la)
        points.extend(ka + t * (kb - ka) for t in ts)
        ticks.append(len(points) - 1)
        labels.append(lb)
    return {
        "kpoints": np.asarray(points, dtype=float),
        "tick_indices": ticks,
        "tick_labels": labels,
        "break_indices": breaks,
    }


def k_distance(kpoints, lattice, breaks: Sequence[int]) -> np.ndarray:
    """Cumulative Cartesian path length in 1/Angstrom (with 2 pi), not advanced across jumps."""
    cartesian = np.asarray(kpoints, float) @ _reciprocal(lattice)
    steps = np.linalg.norm(np.diff(cartesian, axis=0), axis=1)
    for index in breaks:
        if 0 < index <= len(steps):
            steps[index - 1] = 0.0
    return np.concatenate([[0.0], np.cumsum(steps)])


def plane_frame(lattice, a: int, b: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drawing frame of the k-plane spanned by b_a and b_b.

    Returns the reciprocal basis WITHOUT 2 pi (rows, 1/Angstrom, the band
    payload's k unit), the unit vector along b_a and the in-plane unit vector
    normal to it.
    """
    reciprocal = _reciprocal(lattice) / (2.0 * np.pi)
    ea = reciprocal[a] / np.linalg.norm(reciprocal[a])
    normal = np.cross(reciprocal[a], reciprocal[b])
    eb = np.cross(normal / np.linalg.norm(normal), ea)
    return reciprocal, ea, eb


def surface_path(lattice, p: int, q: int, n_points: int) -> dict[str, Any]:
    """High-symmetry path of the surface zone spanned by b_p and b_q, through every surface TRIM.

    A hexagonal cell (|a_p| = |a_q|, 60 or 120 degrees) gets G-M-K-G-M'|M''-G;
    any other cell gets G-X-M-Y-G.
    """
    a = np.asarray(lattice, float)
    ap, aq = a[p], a[q]
    lp, lq = np.linalg.norm(ap), np.linalg.norm(aq)
    cosine = float(np.dot(ap, aq) / (lp * lq))

    def point(kp: float, kq: float) -> list[float]:
        k = [0.0, 0.0, 0.0]
        k[p], k[q] = kp, kq
        return k

    if abs(lp - lq) < 1e-3 * max(lp, lq) and abs(abs(cosine) - 0.5) < 1e-3:
        k_point = point(1 / 3, 1 / 3) if cosine < 0 else point(1 / 3, 2 / 3)
        segments = [
            ["G", point(0, 0), "M", point(0.5, 0)],
            ["M", point(0.5, 0), "K", k_point],
            ["K", k_point, "G", point(0, 0)],
            ["G", point(0, 0), "M'", point(0, 0.5)],
            ["M''", point(0.5, 0.5), "G", point(0, 0)],
        ]
    else:
        segments = [
            ["G", point(0, 0), "X", point(0.5, 0)],
            ["X", point(0.5, 0), "M", point(0.5, 0.5)],
            ["M", point(0.5, 0.5), "Y", point(0, 0.5)],
            ["Y", point(0, 0.5), "G", point(0, 0)],
        ]
    return sample_path(segments, lattice, n_points)


def project_path(segments: Sequence[Sequence[Any]], periodic: Sequence[int]) -> list[list[Any]]:
    """A path for a model periodic only along ``periodic``.

    Components along other axes become 0, so a vacuum cell's A, L and H points
    collapse onto G, M and K, and segments that collapse to a point are dropped.
    """
    out: list[list[Any]] = []
    for la, ka, lb, kb in segments:
        ka = [float(x) if axis in periodic else 0.0 for axis, x in enumerate(ka)]
        kb = [float(x) if axis in periodic else 0.0 for axis, x in enumerate(kb)]
        if np.allclose(ka, kb, atol=1e-9):
            continue
        out.append([la, ka, lb, kb])
    return out


# ---------------------------------------------------------------------------
# Comparison with wannier90_band.dat
# ---------------------------------------------------------------------------


def comparison_window(win: Mapping[str, Any], fermi: float) -> tuple[float, float]:
    """Energy window, relative to E_F (eV), where model and reference bands must agree.

    The frozen window, its lower edge defaulting to the outer window's as in
    wannier90; else the outer window; else +-2 eV around E_F.
    """
    froz_min, froz_max = win.get("dis_froz_min"), win.get("dis_froz_max")
    if froz_max is not None:
        low = froz_min if froz_min is not None else win.get("dis_win_min")
        if low is not None:
            return float(low) - fermi, float(froz_max) - fermi
    win_min, win_max = win.get("dis_win_min"), win.get("dis_win_max")
    if win_min is not None and win_max is not None:
        return float(win_min) - fermi, float(win_max) - fermi
    return -DEFAULT_WINDOW_EV, DEFAULT_WINDOW_EV


def band_deviation(model_bands, reference, window: tuple[float, float] | None) -> dict[str, Any]:
    """Largest and RMS |model - reference| in meV, over all values and inside ``window``.

    ``window`` (eV, the reference's zero) selects reference values within it.
    Without a window, or when no value qualifies, every value is judged;
    ``window_empty`` marks the second case.
    """
    diff = np.abs(np.asarray(model_bands, float) - np.asarray(reference, float))
    out: dict[str, Any] = {
        "max_meV": float(diff.max() * 1000.0) if diff.size else None,
        "rms_meV": float(np.sqrt(np.mean(diff**2)) * 1000.0) if diff.size else None,
        "window_eV": list(window) if window else None,
    }
    inside = None
    if window:
        reference = np.asarray(reference, float)
        inside = (reference >= window[0]) & (reference <= window[1])
    if inside is not None and inside.any():
        selected = diff[inside]
        out["in_window_max_meV"] = float(selected.max() * 1000.0)
        out["in_window_rms_meV"] = float(np.sqrt(np.mean(selected**2)) * 1000.0)
        out["in_window_points"] = int(selected.size)
    else:
        out["window_empty"] = inside is not None
        out["in_window_max_meV"] = out["max_meV"]
        out["in_window_rms_meV"] = out["rms_meV"]
        out["in_window_points"] = int(diff.size)
    return out


# ---------------------------------------------------------------------------
# Band edges, gap and DOS
# ---------------------------------------------------------------------------


class BandEdges:
    """Running per-band extrema, Fermi-level occupation counts and direct gaps over k-chunks.

    Energies are relative to E_F; a level below 0 counts as occupied.
    """

    def __init__(self, n_bands: int):
        self.minimum = np.full(n_bands, np.inf)
        self.maximum = np.full(n_bands, -np.inf)
        self.argmin = np.zeros((n_bands, 3))
        self.argmax = np.zeros((n_bands, 3))
        self.direct = np.full(max(n_bands - 1, 0), np.inf)
        self.direct_at = np.zeros((max(n_bands - 1, 0), 3))
        self.count_min = n_bands
        self.count_max = 0
        self.n_k = 0

    def add(self, kpts, energies) -> None:
        """Fold in sorted energies (Nk, n_bands) at fractional ``kpts`` (Nk, 3)."""
        kpts = np.asarray(kpts, float)
        energies = np.asarray(energies, float).reshape(len(kpts), -1)
        if not len(kpts):
            return
        low = energies.argmin(axis=0)
        high = energies.argmax(axis=0)
        bands = np.arange(energies.shape[1])
        candidate_min = energies[low, bands]
        candidate_max = energies[high, bands]
        better = candidate_min < self.minimum
        self.minimum[better] = candidate_min[better]
        self.argmin[better] = kpts[low[better]]
        better = candidate_max > self.maximum
        self.maximum[better] = candidate_max[better]
        self.argmax[better] = kpts[high[better]]
        if energies.shape[1] > 1:
            gaps = np.diff(energies, axis=1)
            at = gaps.argmin(axis=0)
            candidate = gaps[at, np.arange(gaps.shape[1])]
            better = candidate < self.direct
            self.direct[better] = candidate[better]
            self.direct_at[better] = kpts[at[better]]
        counts = (energies < 0.0).sum(axis=1)
        self.count_min = min(self.count_min, int(counts.min()))
        self.count_max = max(self.count_max, int(counts.max()))
        self.n_k += len(kpts)

    def merge(self, other: BandEdges) -> None:
        """Fold in another accumulator; ties keep this one's k-points."""
        better = other.minimum < self.minimum
        self.minimum[better] = other.minimum[better]
        self.argmin[better] = other.argmin[better]
        better = other.maximum > self.maximum
        self.maximum[better] = other.maximum[better]
        self.argmax[better] = other.argmax[better]
        better = other.direct < self.direct
        self.direct[better] = other.direct[better]
        self.direct_at[better] = other.direct_at[better]
        self.count_min = min(self.count_min, other.count_min)
        self.count_max = max(self.count_max, other.count_max)
        self.n_k += other.n_k


def classify_gap(
    edges: BandEdges, fermi_tolerance: float, occupied_bands: int | None = None
) -> dict[str, Any]:
    """Insulator or metal, the occupied band count and the gap, from accumulated band edges.

    E_F (the energy zero) need not sit mid-gap. A global gap above 1 meV counts
    when E_F lies within ``fermi_tolerance`` of it, the most central such gap
    winning, or when ``occupied_bands`` names the count (1..N-1).
    """
    n = len(edges.minimum)
    result: dict[str, Any] = {"n_bands": n, "k_points_sampled": int(edges.n_k)}
    chosen = None
    if occupied_bands is not None:
        count = int(occupied_bands)
        if not 0 < count < n:
            raise ValueError(f"occupied_bands={count} must lie between 1 and {n - 1}")
        chosen = count
        result["n_occ_source"] = "user"
    else:
        best = None
        for count in range(1, n):
            low, high = edges.maximum[count - 1], edges.minimum[count]
            if high - low <= 1e-3:
                continue
            if low - fermi_tolerance <= 0.0 <= high + fermi_tolerance:
                score = min(0.0 - low, high - 0.0)
                if best is None or score > best[0]:
                    best = (score, count)
        if best is not None:
            chosen = best[1]
            result["n_occ_source"] = "auto"
    if chosen is None:
        crossing = [int(b) for b in range(n) if edges.minimum[b] < 0.0 < edges.maximum[b]]
        result.update(
            metal=True,
            n_occ=None,
            gap_eV=None,
            bands_crossing_fermi=crossing,
            fermi_level_outside_model=bool(edges.maximum.max() < 0.0 or edges.minimum.min() > 0.0),
        )
        return result
    count = chosen
    low, high = float(edges.maximum[count - 1]), float(edges.minimum[count])
    direct = float(edges.direct[count - 1]) if count - 1 < len(edges.direct) else None
    vbm_k = edges.argmax[count - 1].tolist()
    cbm_k = edges.argmin[count].tolist()
    result.update(
        metal=high - low <= 1e-3,
        n_occ=int(count),
        gap_eV=float(high - low),
        direct_gap_eV=direct,
        direct_gap_k=edges.direct_at[count - 1].tolist() if direct is not None else None,
        is_direct=bool(np.allclose(vbm_k, cbm_k, atol=1e-6)),
        vbm_eV=low,
        cbm_eV=high,
        vbm_k=vbm_k,
        cbm_k=cbm_k,
    )
    return result


def dos_energy_grid(energies, step: float, margin: float = 2.0) -> np.ndarray:
    """Histogram bin edges from min(E) - margin to max(E) + margin in ``step`` (eV)."""
    values = np.asarray(energies, float)
    low = float(values.min()) - margin
    high = float(values.max()) + margin
    return np.arange(low, high + step, step)


def gaussian_dos(hist, step: float, sigma: float) -> np.ndarray:
    """Smooth a k-averaged eigenvalue histogram (states per bin) into a DOS (states/eV).

    The kernel is a Gaussian of standard deviation ``sigma`` sampled on the bin
    spacing, truncated at ceil(5 sigma/step) bins, normalized to unit sum and
    applied with ``np.convolve(..., "same")`` per row; ``sigma <= 0`` returns
    the raw histogram density.
    """
    density = np.asarray(hist, float) / step
    if sigma <= 0:
        return density
    half = int(math.ceil(5.0 * sigma / step))
    x = np.arange(-half, half + 1) * step
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    if density.ndim == 1:
        return np.convolve(density, kernel, mode="same")
    return np.stack([np.convolve(row, kernel, mode="same") for row in density])


def dos_at_fermi(centres, dos, rule: str = "linear_interpolation") -> float:
    """DOS at E_F = 0 from bin centres.

    ``linear_interpolation`` interpolates between the bin centres around 0;
    ``first_nonnegative_bin`` is the historical rule of the first driver
    revisions, the value of the first bin whose centre is at or above 0.
    """
    centres = np.asarray(centres, float)
    dos = np.asarray(dos, float)
    if rule == "linear_interpolation":
        return float(np.interp(0.0, centres, dos))
    if rule == "first_nonnegative_bin":
        index = int(np.searchsorted(centres, 0.0))
        return float(dos[min(max(index, 0), len(dos) - 1)])
    raise ValueError(f"rule must be one of {DOS_AT_FERMI_RULES}")


def eigenvalue_histogram(energies, grid, weights=None) -> tuple[np.ndarray, int]:
    """Eigenvalue counts per bin of ``grid`` and the number of values outside it.

    Counts follow ``np.histogram``, optionally weighted (a projection for PDOS).
    As in the driver, a value equal to the last edge lands in the last bin and is
    also reported as outside.
    """
    values = np.asarray(energies, float).ravel()
    counts, _ = np.histogram(
        values, bins=grid, weights=None if weights is None else np.asarray(weights, float).ravel()
    )
    outside = int(np.sum((values < grid[0]) | (values >= grid[-1])))
    return counts, outside


# ---------------------------------------------------------------------------
# Species of the Wannier functions and localization
# ---------------------------------------------------------------------------


def orbital_species(centres_cart, structure, lattice) -> tuple[list[str], np.ndarray]:
    """The nearest atom's species for every Wannier centre, over the 27 neighbouring images.

    ``structure`` holds ``species`` and Cartesian ``cart`` positions in Angstrom.
    Returns the species in first-appearance order and each orbital's index into
    them. Without a usable structure every orbital falls into one group, "all".
    """
    n = len(centres_cart)
    try:
        species = [str(name) for name in structure["species"]]
        positions = np.asarray(structure["cart"], float).reshape(-1, 3)
        if len(species) != len(positions) or not len(species):
            raise ValueError
    except (TypeError, KeyError, ValueError):
        return ["all"], np.zeros(n, dtype=int)
    cell = np.asarray(lattice, float)
    shifts = (
        np.stack(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij"), -1).reshape(-1, 3) @ cell
    )
    images = positions[None, :, :] + shifts[:, None, :]
    names = sorted(set(species), key=species.index)
    index = np.zeros(n, dtype=int)
    for orbital, centre in enumerate(np.asarray(centres_cart, float)):
        distance = np.linalg.norm(images - centre[None, None, :], axis=-1).min(axis=0)
        index[orbital] = names.index(species[int(np.argmin(distance))])
    return names, index


def unwrap_centres(centres_cart, lattice, axes: Sequence[int]) -> np.ndarray:
    """Centres moved by whole lattice vectors into one contiguous run along ``axes``.

    Per axis the circle of fractional coordinates is cut at the middle of the
    largest gap between centres (a slab's vacuum) and every centre is moved into
    the unit interval starting there, taken nearest the home cell. Wannier90
    wraps the centres it writes, so a slab's wrap line can run through the slab
    itself. For real-space diagnostics only: the model and the Wigner-Seitz
    images use the centres as written.
    """
    cell = np.asarray(lattice, float)
    cartesian = np.asarray(centres_cart, float).reshape(-1, 3)
    if not len(cartesian) or not len(axes):
        return cartesian.copy()
    fractional = cartesian @ np.linalg.inv(cell)
    for axis in axes:
        cut, _ = largest_gap_midpoint(fractional[:, axis])
        if cut > 0.5:
            cut -= 1.0
        fractional[:, axis] -= np.floor(fractional[:, axis] - cut)
    return fractional @ cell


def hopping_decay(
    translations, hopping, centres_cart, lattice, bin_width: float = 0.5
) -> dict[str, Any] | None:
    """Every hop's |H| against the distance |R.A + tau_j - tau_i| it spans, and its decay.

    Returns the nonzero hops (``distance``, ``amplitude``), the largest amplitude
    per ``bin_width`` bin (``bin_centres`` are the bins' mean distances,
    ``bin_peaks``), ``decay_length_A`` of an exponential fitted to the peaks
    beyond 2 Angstrom above 1e-8 eV (None for fewer than three such bins or no
    decay) and ``max_distance_A``; None for an all-zero table.
    """
    taus = np.asarray(centres_cart, float).reshape(-1, 3)
    distances = np.linalg.norm(
        (np.asarray(translations, float) @ np.asarray(lattice, float))[:, None, None, :]
        + taus[None, None, :, :]
        - taus[None, :, None, :],
        axis=-1,
    )
    d = distances.ravel()
    a = np.abs(np.asarray(hopping)).ravel()
    keep = a > 0
    d, a = d[keep], a[keep]
    if not len(d):
        return None
    bins = np.arange(0.0, d.max() + bin_width, bin_width)
    index = np.digitize(d, bins)
    centres, peaks = [], []
    for value in np.unique(index):
        selected = index == value
        centres.append(float(d[selected].mean()))
        peaks.append(float(a[selected].max()))
    centre_array = np.asarray(centres)
    peak_array = np.asarray(peaks)
    fit = (centre_array > 2.0) & (peak_array > 1e-8)
    decay = None
    if fit.sum() >= 3:
        slope, _ = np.polyfit(centre_array[fit], np.log(peak_array[fit]), 1)
        decay = float(-1.0 / slope) if slope < 0 else None
    return {
        "distance": d,
        "amplitude": a,
        "bin_centres": centre_array,
        "bin_peaks": peak_array,
        "decay_length_A": decay,
        "max_distance_A": float(d.max()),
    }


# ---------------------------------------------------------------------------
# Slab and ribbon edge states
# ---------------------------------------------------------------------------


def edge_weights(
    eigvecs,
    n_cells: int,
    n_orb: int,
    edge_cells: int,
    energies=None,
    degenerate_tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-state weight of each eigenvector (columns) on the first and the last edge cells.

    With ``energies``, each degenerate cluster (energy-sorted neighbours closer
    than ``degenerate_tol`` eV) is first rotated to diagonalize the first-edge
    projector, which separates the two edges where the subspace allows. The
    per-state values remain basis-dependent inside such a cluster; their cluster
    sums (:func:`queer.physics.slab.cluster_totals`) are the physical weights.
    """
    edge = edge_cells_used(edge_cells, n_cells)
    vecs = np.array(eigvecs, dtype=complex, copy=True)
    if energies is not None and len(energies) > 1:
        mask = np.zeros(n_cells * n_orb)
        mask[: edge * n_orb] = 1.0
        order = np.argsort(energies)
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and energies[order[stop]] - energies[order[stop - 1]] < degenerate_tol:
                stop += 1
            if stop - start > 1:
                index = order[start:stop]
                sub = vecs[:, index]
                projector = sub.conj().T @ (mask[:, None] * sub)
                _, rotation = np.linalg.eigh(projector)
                vecs[:, index] = sub @ rotation
            start = stop
    probability = np.abs(vecs) ** 2
    cells = probability.reshape(n_cells, n_orb, -1).sum(axis=1)
    return cells[:edge].sum(axis=0), cells[-edge:].sum(axis=0)


def edge_crossings(
    energies: Sequence,
    left: Sequence,
    right: Sequence,
    *,
    periodic: bool = False,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Count edge-localized states crossing E_F = 0 between neighbouring k-points.

    A state whose weight on an edge is at least ``threshold`` at k_j is matched
    to the nearest-energy state localized on the same edge at k_(j+1), so two
    chiral modes crossing at E_F (one per edge) are not swapped. A sign change
    of its energy is a crossing, up or down by the sign of the step; ``net`` is
    up minus down. ``periodic`` also pairs the last k-point with the first (a
    ribbon's 1D zone). The counts depend on per-state weights and are
    informational; the bulk invariant is the authority.
    """
    tally: dict[str, Any] = {"left": {"up": 0, "down": 0}, "right": {"up": 0, "down": 0}}
    count = len(energies)
    pairs = [(j, j + 1) for j in range(count - 1)]
    if periodic and count > 1:
        pairs.append((count - 1, 0))
    for side, weights in (("left", left), ("right", right)):
        for j, j1 in pairs:
            on0 = np.asarray(weights[j]) >= threshold
            on1 = np.asarray(weights[j1]) >= threshold
            if not on0.any() or not on1.any():
                continue
            here = np.asarray(energies[j])[on0]
            there = np.asarray(energies[j1])[on1]
            for e in here:
                match = there[int(np.argmin(np.abs(there - e)))]
                if (e < 0.0) != (match < 0.0):
                    tally[side]["up" if match > e else "down"] += 1
    for side in ("left", "right"):
        tally[side]["net"] = tally[side]["up"] - tally[side]["down"]
    return tally
