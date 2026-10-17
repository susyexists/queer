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
from ...physics.topology import chern_from_phases, largest_gap_midpoint

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


def _diagonalise_degenerate(
    vecs: np.ndarray, energies, mask: np.ndarray, degenerate_tol: float
) -> np.ndarray:
    """Rotate each degenerate cluster of ``vecs`` (columns) to diagonalize diag(``mask``).

    A cluster is a run of energy-sorted neighbours closer than ``degenerate_tol``
    eV; rotating it to the eigenvectors of the projector separates what the
    projector holds (an edge, half of the cells) where the subspace allows.
    """
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
    return vecs


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
        vecs = _diagonalise_degenerate(vecs, energies, mask, degenerate_tol)
    probability = np.abs(vecs) ** 2
    cells = probability.reshape(n_cells, n_orb, -1).sum(axis=1)
    return cells[:edge].sum(axis=0), cells[-edge:].sum(axis=0)


def half_ribbon_weights(
    eigvecs, n_cells: int, n_orb: int, energies=None, degenerate_tol: float = 1e-5
) -> tuple[np.ndarray, np.ndarray]:
    """Per-state weight of each eigenvector (columns) on the first half of the cells and on the middle half.

    Cells are unit intervals: the first half is [0, n/2), so an odd ribbon's
    middle cell counts half, and the middle half is [n/4, 3n/4), a cell cut by
    its boundary counting by the overlap. A state spread evenly over the cells
    holds 1/2 of each; an edge state holds 0 or 1 of the first and little of the
    middle. With ``energies``, each degenerate cluster is first rotated to
    diagonalize the first-half projector, as in :func:`edge_weights`.
    """
    cell = np.arange(n_cells, dtype=float)
    first = np.clip(np.minimum(cell + 1, n_cells / 2) - cell, 0.0, 1.0)
    middle = np.clip(np.minimum(cell + 1, 3 * n_cells / 4) - np.maximum(cell, n_cells / 4), 0.0, 1.0)
    vecs = np.array(eigvecs, dtype=complex, copy=True)
    if energies is not None and len(energies) > 1:
        mask = np.repeat(first, n_orb)
        vecs = _diagonalise_degenerate(vecs, energies, mask, degenerate_tol)
    probability = np.abs(vecs) ** 2
    cells = probability.reshape(n_cells, n_orb, -1).sum(axis=1)
    return first @ cells, middle @ cells


def bulk_points(k_par, finite_axis: int, n_perp: int) -> np.ndarray:
    """The bulk k-points of one in-plane k: ``n_perp`` points i / n_perp along ``finite_axis``."""
    kpts = np.repeat(np.asarray(k_par, float).reshape(1, 3), n_perp, axis=0)
    kpts[:, finite_axis] = np.arange(n_perp) / n_perp
    return kpts


def bulk_row(
    translations, hopping_matrices, finite_axis: int, k_par, n_perp: int, berry: bool = False
) -> dict[str, Any]:
    """The bulk levels at one in-plane k, sampled at ``n_perp`` points along ``finite_axis``; the driver's.

    H(k) = sum_R exp(2 pi i k.R) H(R). ``below`` holds the fewest and the most levels
    below E_F (energy zero) at one point, ``under`` the highest level below it and
    ``over`` the lowest at or above it (None when there is none). ``levels`` holds every
    point's ascending levels, the bands :func:`bulk_overshoot` follows from point to point.
    With
    ``berry`` and the same number of levels below E_F at every point, ``berry_phase`` is
    the Berry phase of those bands around the finite axis, -sum arg det of the overlaps of
    neighbouring points (H(k) is periodic, so the loop closes on the first point);
    otherwise None.
    """
    R = np.asarray(translations, dtype=float)
    H = np.asarray(hopping_matrices)
    n = H.shape[1]
    kpts = bulk_points(k_par, finite_axis, n_perp)
    phases = np.exp(2j * np.pi * (kpts @ R.T))
    hk = (phases @ H.reshape(len(R), n * n)).reshape(len(kpts), n, n)
    if berry:
        levels, vectors = np.linalg.eigh(hk)
    else:
        levels, vectors = np.linalg.eigvalsh(hk), None
    counts = (levels < 0.0).sum(axis=1)
    fewest, most = int(np.min(counts)), int(np.max(counts))
    under, over = levels[levels < 0.0], levels[levels >= 0.0]
    row: dict[str, Any] = {
        "below": (fewest, most),
        "under": float(under.max()) if under.size else None,
        "over": float(over.min()) if over.size else None,
        "levels": levels,
        "berry_phase": None,
    }
    if vectors is not None and fewest == most:
        if fewest == 0:
            row["berry_phase"] = 0.0
        else:
            u = vectors[:, :, :fewest]
            links = np.einsum("jia,jib->jab", u.conj(), np.roll(u, -1, axis=0))
            row["berry_phase"] = float(-np.angle(np.linalg.det(links)).sum())
    return row


def bulk_aliasing(translations, hopping_matrices, axes: Sequence[int], points: Sequence[int]) -> list[float]:
    """Per axis, the most the hoppings its mesh cannot resolve change a band over one step; the driver's.

    A hopping R adds the Bloch phase exp(2 pi i k.R); ``points`` per period along an axis
    resolve it only while they exceed 2 |R_axis| (more than two points per period of the
    phase), and a band's steps between neighbouring points show its change between them
    only for the hoppings they resolve. By Weyl's inequality a band moves between k and k'
    by at most ||H(k) - H(k')||, which the hoppings left unresolved bound over a step 1 / n
    by 2 pi sum |R_axis| ||H(R)|| / n (the Frobenius norm bounding each block's): 0 when the
    mesh resolves every hopping, as a nearest-neighbour model's does from 3 points on.
    """
    R = np.asarray(translations)
    norms = np.sqrt((np.abs(np.asarray(hopping_matrices)) ** 2).sum(axis=(1, 2)))
    out = []
    for axis, n in zip(axes, points):
        cells = np.abs(R[:, axis]).astype(float)
        unresolved = 2 * cells >= int(n)
        out.append(float(2 * np.pi * (cells * norms)[unresolved].sum() / max(1, int(n))))
    return out


def bulk_overshoot(levels, below: int, aliased: Sequence[float] = (0.0, 0.0)) -> dict[str, Any]:
    """How far bands can lie beyond the samples, from their steps between neighbouring points; the driver's.

    ``levels`` (in-plane k, finite-axis point, band) holds the ascending
    levels on a mesh closed along both axes, the first ``below`` of them below
    E_F at every point. A band's extremum between samples lies within half a
    step of one along each axis, and over that half step the band changes by
    at most half its larger step to a neighbour there: near an extremum a band
    is convex or concave, so its steps grow away from it (a Dirac point midway
    between two samples lies v*dk/2 beyond them, and the next step out is
    v*dk). A point's bound sums the two axes, each with ``aliased``, its
    change per step from hoppings its mesh cannot resolve
    (:func:`bulk_aliasing`). Every band counts, not only the two at the gap.

    ``overshoot_eV`` is [valence, conduction]: the most a band below E_F can
    rise above the highest sample below E_F, and a band above it fall below
    the lowest sample at or above it (None for a side without bands).
    ``local_eV`` is [steps, gap] for the band below E_F and the band above it,
    at one point, whose bounds together come closest to the fraction
    CROSSING_RESOLUTION of the gap between them there (None without both
    sides): a pair whose gap the mesh does not resolve hides a Dirac point,
    and its Berry flux, between samples, wherever it lies in energy.
    """
    mesh = np.asarray(levels, dtype=float)
    bound = np.zeros_like(mesh)
    for axis, extra in zip((0, 1), aliased):
        forward = np.abs(np.roll(mesh, -1, axis=axis) - mesh)
        backward = np.abs(mesh - np.roll(mesh, 1, axis=axis))
        bound += 0.5 * (np.maximum(forward, backward) + float(extra))
    valence, conduction = mesh[..., :below], mesh[..., below:]
    out: dict[str, Any] = {
        "overshoot_eV": [
            (float((valence + bound[..., :below]).max() - valence.max()) if valence.size else None),
            (float(conduction.min() - (conduction - bound[..., below:]).min()) if conduction.size else None),
        ],
        "local_eV": None,
    }
    if valence.size and conduction.size:
        # b_n + b_m <= f (E_m - E_n) for every pair at a point is
        # max_n (b_n + f E_n) + max_m (b_m - f E_m) <= 0.
        lower = bound[..., :below] + CROSSING_RESOLUTION * valence
        upper = bound[..., below:] - CROSSING_RESOLUTION * conduction
        n = lower.argmax(axis=-1)
        m = upper.argmax(axis=-1)
        excess = lower.max(axis=-1) + upper.max(axis=-1)
        point = np.unravel_index(int(np.argmax(excess)), excess.shape)
        i, j = int(n[point]), below + int(m[point])
        out["local_eV"] = [
            float(bound[point][i] + bound[point][j]),
            float(mesh[point][j] - mesh[point][i]),
        ]
    return out


def bulk_summary(
    rows: Sequence[Mapping[str, Any]],
    n_perp: int,
    closed: bool = False,
    aliased: Sequence[float] | None = None,
) -> dict[str, Any]:
    """The driver's bulk check from :func:`bulk_row` at each in-plane k.

    ``metal`` when the number of levels below E_F varies: a band crosses E_F between
    samples. Otherwise ``gap_eV`` holds the highest level below E_F and the lowest at or
    above it over every sample (None on a side without one), and, over a ``closed``
    in-plane loop (a ribbon's 1D zone), ``chern`` is the Chern number of the bands below
    E_F from the winding of their Berry phase along the loop
    (:func:`~queer.physics.topology.chern_from_phases`), with ``berry_phase_step`` the
    largest wrapped change of that phase between neighbouring in-plane k: the winding is
    trustworthy only while it stays well below pi; with ``overshoot_eV``, how far each edge
    of the gap can lie beyond its samples, and ``local_overshoot_eV``, the band pair across
    E_F the mesh comes closest to not resolving, [their steps, their gap]
    (:func:`bulk_overshoot`; ``aliased`` the in-plane and finite axes' change per step from
    hoppings their meshes cannot resolve).
    """
    counts = [count for row in rows for count in row["below"]]
    metal = bool(counts) and min(counts) != max(counts)
    gap: list[float | None] | None = None
    chern: int | None = None
    step: float | None = None
    overshoot: list[float | None] | None = None
    local: list[float] | None = None
    if not metal:
        under = [row["under"] for row in rows if row["under"] is not None]
        over = [row["over"] for row in rows if row["over"] is not None]
        gap = [max(under) if under else None, min(over) if over else None]
        phases = [row["berry_phase"] for row in rows]
        if closed and phases and all(phase is not None for phase in phases):
            value, step = chern_from_phases(phases)
            chern = int(round(value))
        if closed and rows:
            bounds = bulk_overshoot(
                np.stack([row["levels"] for row in rows]), min(counts), aliased or (0.0, 0.0)
            )
            overshoot, local = bounds["overshoot_eV"], bounds["local_eV"]
    return {
        "metal": metal,
        "gap_eV": gap,
        "k_points": [len(rows), int(n_perp)],
        "chern": chern,
        "berry_phase_step": step,
        "overshoot_eV": overshoot,
        "local_overshoot_eV": local,
    }


def bulk_at_fermi(
    translations, hopping_matrices, finite_axis: int, k_par, n_perp: int, closed: bool = False
) -> dict[str, Any]:
    """Whether the bulk bands cross E_F (energy zero) at the slab's in-plane k, and the bulk gap; the driver's.

    The bulk levels are sampled at every in-plane k of ``k_par`` and at ``n_perp`` points
    i / n_perp along ``finite_axis`` (:func:`bulk_row`, :func:`bulk_summary`). A band below
    E_F at one sample and at or above it at another crosses E_F in between: ``metal``. A
    sampled metal is a metal; a pocket narrower than the mesh step can be missed, and
    ``overshoot_eV`` bounds how far the gap's edges can lie beyond the samples. Otherwise
    ``gap_eV`` is the bulk gap around E_F at these in-plane k, and over a ``closed`` loop
    (along the one axis the in-plane k vary on) ``chern`` is the Chern number of the bands
    below E_F.
    """
    points = np.asarray(k_par, float).reshape(-1, 3)
    rows = [bulk_row(translations, hopping_matrices, finite_axis, k, n_perp, berry=closed) for k in points]
    aliased = None
    varying = [
        axis for axis in range(3) if axis != finite_axis and len(points) and np.ptp(points[:, axis]) > 0
    ]
    if closed and len(varying) == 1:
        aliased = bulk_aliasing(
            translations, hopping_matrices, [varying[0], finite_axis], [len(points), n_perp]
        )
    return bulk_summary(rows, n_perp, closed, aliased)


def bulk_k_perp_points(n_k_par: int, cells: int) -> int:
    """Points along the finite axis for the driver's bulk check.

    A whole multiple of twice the slab's cells (twice the resolution of its standing
    waves, so every such sample is kept), and at least as many as the slab's k-points
    along its periodic axis: the bulk is resolved along the finite axis at least as
    finely as the slab resolves its own k.
    """
    base = 2 * max(1, int(cells))
    return base * max(1, math.ceil(int(n_k_par) / base))


# Edge crossings, a weighted spectral flow: the rank offsets tried per k-step around no
# crossing and around the change in the count below the reference (the levels entering or
# leaving at the window edges in one step), the ranks around the reference energy they
# compare, the energies sampled across the gap besides E_F, the middle-half weight from
# which a level is bulk-like, the reach of one k-step in units of the largest level step
# seen, how close to an integer a weighted net must be to count as that integer, and the
# largest step of the bulk Berry phase between neighbouring k-points at which the bulk
# Chern number is trusted (the TOPOLOGY step's own bound), the largest fraction of the
# sampled bulk gap its two edges may lie beyond the bulk mesh's samples, and of a band
# pair's own gap its steps may take (:func:`mesh_resolution`), and the energy within which
# a level counts as at a reference energy and two sampled energies as one (the weights'
# degeneracy tolerance).
CROSSING_MAX_OFFSET = 3
CROSSING_RANKS = (-3, 2)
CROSSING_GAP_SAMPLES = 9
CROSSING_BULK_INTERIOR = 0.25
CROSSING_REACH = 1.5
CROSSING_TOLERANCE = 0.1
CROSSING_BERRY_STEP = math.pi / 2
CROSSING_RESOLUTION = 0.5
CROSSING_ENERGY_TOL = 1e-5
CROSSING_METHOD = "half_ribbon_flow"


def _offsets(i0: int, j0: int) -> range:
    """The offsets :func:`align_ranks` tries: within 3 of none and of the count change i0 - j0."""
    change = i0 - j0
    return range(
        min(-CROSSING_MAX_OFFSET, change - CROSSING_MAX_OFFSET),
        max(CROSSING_MAX_OFFSET, change + CROSSING_MAX_OFFSET) + 1,
    )


def _offset_costs(
    a: np.ndarray, b: np.ndarray, e_ref: float, window: float | None
) -> tuple[int, int, list[tuple[float, int, int]]]:
    """``(i0, j0, keys)``: the ranks of :func:`align_ranks` and every valid offset's sort key, ascending."""
    i0 = int(np.searchsorted(a, e_ref))
    j0 = int(np.searchsorted(b, e_ref))
    keys: list[tuple[float, int, int]] = []
    for o in _offsets(i0, j0):
        crossed = range(-o, 0) if o > 0 else range(0, -o)
        if not all(0 <= i0 + r < len(a) and 0 <= j0 + r + o < len(b) for r in crossed):
            continue
        steps: list[float] = []
        paired = False
        for r in range(CROSSING_RANKS[0], CROSSING_RANKS[1] + 1):
            ia, ib = i0 + r, j0 + r + o
            in_a, in_b = 0 <= ia < len(a), 0 <= ib < len(b)
            if in_a and in_b:
                steps.append(abs(float(a[ia]) - float(b[ib])))
                paired = True
            elif in_a or in_b:
                level = float(a[ia]) if in_a else float(b[ib])
                steps.append(math.inf if window is None else max(0.0, window - abs(level)))
        if paired:
            keys.append((sum(steps) / len(steps), abs(o), o))
    return i0, j0, sorted(keys)


def align_ranks(
    a: np.ndarray, b: np.ndarray, e_ref: float, window: float | None
) -> tuple[int, int, int | None]:
    """Ranks of the ascending levels ``a`` (k_j) and ``b`` (k_(j+1)) and their offset.

    Rank 0 is the lowest level at or above ``e_ref``; rank r at k_j pairs with rank
    r + o at k_(j+1). o > 0: ranks -o..-1 crossed upwards; o < 0: ranks 0..|o|-1
    downwards. The offsets tried lie within 3 of 0 and within 3 of the change in the
    number of levels below ``e_ref`` (:func:`_offsets`), so copies crossing together are
    followed however many there are. The offset minimizes the mean step over ranks
    -3..2: a pair's energy step, and for a level without a partner the distance to the
    window edge it must have crossed (none can lack one without a ``window``). The levels
    an offset says crossed must have partners; ties go to the smaller |o|, then the lower
    o. Returns ``(i0, j0, o)``, o None when no offset pairs a level.
    """
    i0, j0, keys = _offset_costs(a, b, e_ref, window)
    return i0, j0, keys[0][2] if keys else None


def taking_part(
    a: np.ndarray, b: np.ndarray, i0: int, j0: int, o: int, reach: float
) -> tuple[list[int], list[float]]:
    """Ranks of the aligned pairs that cross the reference or may swap across it.

    The crossed ranks; with o = 0, ranks -1 and 0 when each of the two levels could
    end where the other ends within ``reach`` (two levels crossing each other at the
    reference within one step, which energy order hides); then every neighbouring
    pair that could swap with an end of the set, in turn. Also returns the levels
    without a partner that could swap with an end.
    """

    def paired(r: int) -> bool:
        return 0 <= i0 + r < len(a) and 0 <= j0 + r + o < len(b)

    def could_swap(lo: int, hi: int) -> bool:
        return (
            abs(float(b[j0 + hi + o]) - float(a[i0 + lo])) <= reach
            and abs(float(a[i0 + hi]) - float(b[j0 + lo + o])) <= reach
        )

    if o > 0:
        lo, hi = -o, -1
    elif o < 0:
        lo, hi = 0, -o - 1
    elif paired(-1) and paired(0) and could_swap(-1, 0):
        lo, hi = -1, 0
    else:
        return [], []
    while paired(lo - 1) and could_swap(lo - 1, lo):
        lo -= 1
    while paired(hi + 1) and could_swap(hi, hi + 1):
        hi += 1
    loose: list[float] = []
    for r, end in ((lo - 1, lo), (hi + 1, hi)):
        if paired(r):
            continue
        ia, ib = i0 + r, j0 + r + o
        if 0 <= ia < len(a) and abs(float(b[j0 + end + o]) - float(a[ia])) <= reach:
            loose.append(float(a[ia]))
        if 0 <= ib < len(b) and abs(float(a[i0 + end]) - float(b[ib])) <= reach:
            loose.append(float(b[ib]))
    return list(range(lo, hi + 1)), loose


def exchanges(
    below_a: Sequence[float], above_a: Sequence[float], below_b: Sequence[float], above_b: Sequence[float]
) -> tuple[int, int, int, int]:
    """Crossings of each edge in one step from the first-half weights of the levels taking part.

    A level belongs to the first half when more than 1/2 of it lies there and to the
    last when less does. Every level of an edge below the reference at k_j may end above
    it at k_(j+1) and every one above may end below, so each edge counts the most
    exchanges its levels allow: up = min(its levels below at k_j, above at k_(j+1)),
    down = min(above at k_j, below at k_(j+1)). Two helical modes of one edge crossing in
    the same step count one each way; two levels of one edge that only pass near the
    reference in the step count so too (energies alone cannot tell the two pairings apart):
    upper bounds. up - down is the change of the edge's level count below the reference.
    Returns (first up, first down, last up, last down).
    """
    counts = []
    for side in (lambda p: p > 0.5, lambda p: p < 0.5):
        groups = [sum(1 for p in group if side(p)) for group in (below_a, above_a, below_b, above_b)]
        counts += [min(groups[0], groups[3]), min(groups[1], groups[2])]
    return counts[0], counts[1], counts[2], counts[3]


def flow_at(
    levels: Sequence[np.ndarray],
    half: Sequence[np.ndarray],
    middle: Sequence[np.ndarray],
    steps: Sequence[tuple[int, int]],
    e_ref: float,
    reach: float,
    window: float | None,
) -> dict[str, Any]:
    """The weighted spectral flow through ``e_ref`` over the k-steps ``steps``.

    Per step, over the pairs taking part (:func:`taking_part`), the first-half weight
    below the reference at k_j minus that at k_(j+1), averaged with the first-half
    weight above it at k_(j+1) minus that at k_j: a level rising through the reference
    adds the mean of its endpoints' weights, two that swap across it add the weight that
    moved up, and any pairing within the set gives the same sum. Returns the first-half
    (``left``) and middle-half (``bulk``) flows, the net level count (``net``), per step
    the left flow, the offset and the crossings of each edge (:func:`exchanges`), the
    bulk-like crossings, whether a bulk-like level took part, the levels that appear or
    disappear farther than ``reach`` inside the window (``strays``), and those without a
    partner that could swap with the levels taking part (``loose``).
    """
    left = bulk = 0.0
    net = 0
    per_step: list[tuple[float, int, tuple[int, int, int, int]]] = []
    bulk_crossings = 0
    touches = False
    strays: list[float] = []
    loose_levels: list[float] = []
    for j, j1 in steps:
        a, b = levels[j], levels[j1]
        i0, j0, o = align_ranks(a, b, e_ref, window)
        if o is None:
            unpaired = [*a, *b]
        else:
            lo = max(-i0, -(j0 + o))
            hi = min(len(a) - i0, len(b) - j0 - o) - 1
            unpaired = [*a[: i0 + lo], *a[i0 + hi + 1 :]]
            unpaired += [*b[: j0 + lo + o], *b[j0 + hi + o + 1 :]]
        strays += [float(x) for x in unpaired if window is None or window - abs(x) > reach]
        if o is None:
            continue
        ranks, loose = taking_part(a, b, i0, j0, o, reach)
        loose_levels += loose
        step_left = 0.0
        sides: tuple[list[float], list[float], list[float], list[float]] = ([], [], [], [])
        for r in ranks:
            ia, ib = i0 + r, j0 + r + o
            sa = 1.0 if a[ia] < e_ref else -1.0
            sb = 1.0 if b[ib] < e_ref else -1.0
            step_left += 0.5 * (float(half[j][ia]) * sa - float(half[j1][ib]) * sb)
            bulk += 0.5 * (float(middle[j][ia]) * sa - float(middle[j1][ib]) * sb)
            sides[0 if sa > 0 else 1].append(float(half[j][ia]))
            sides[2 if sb > 0 else 3].append(float(half[j1][ib]))
            inner = (float(middle[j][ia]), float(middle[j1][ib]))
            if max(inner) >= CROSSING_BULK_INTERIOR:
                touches = True
                if sa != sb and 0.5 * sum(inner) >= CROSSING_BULK_INTERIOR:
                    bulk_crossings += 1
        left += step_left
        net += o
        per_step.append((step_left, o, exchanges(*sides)))
    return {
        "left": left,
        "bulk": bulk,
        "net": net,
        "per_step": per_step,
        "bulk_crossings": bulk_crossings,
        "touches": touches,
        "strays": strays,
        "loose": loose_levels,
    }


def step_shift(
    levels: Sequence[np.ndarray],
    steps: Sequence[tuple[int, int]],
    energies: Sequence[float],
    window: float | None,
) -> float | None:
    """The largest energy step of an aligned pair of ranks -3..2 around any of ``energies``."""
    worst: float | None = None
    for e_ref in energies:
        for j, j1 in steps:
            a, b = levels[j], levels[j1]
            i0, j0, o = align_ranks(a, b, e_ref, window)
            if o is None:
                continue
            for r in range(CROSSING_RANKS[0], CROSSING_RANKS[1] + 1):
                ia, ib = i0 + r, j0 + r + o
                if 0 <= ia < len(a) and 0 <= ib < len(b):
                    step = abs(float(a[ia]) - float(b[ib]))
                    worst = step if worst is None else max(worst, step)
    return worst


def _crossing_tally(
    per_step: Sequence[tuple[float, int, tuple[int, int, int, int]]], side: str
) -> dict[str, int]:
    """Crossings of one edge at one energy: each step's exchanges (:func:`exchanges`), summed."""
    first = 0 if side == "left" else 2
    return {
        "up": sum(step[2][first] for step in per_step),
        "down": sum(step[2][first + 1] for step in per_step),
    }


def talking_levels(
    levels: Sequence[np.ndarray],
    half: Sequence[np.ndarray],
    middle: Sequence[np.ndarray],
    gap: Sequence[float],
    energies: Sequence[float],
    reach: float,
) -> tuple[list[float], float | None]:
    """Levels inside ``gap``, within the span of ``energies``, that the two edges share.

    Inside a bulk gap only edge states live, so a level there is spread over the ribbon
    when it is bulk-like, or shared by its two halves when its first-half weight is
    farther than half the tolerance from 0 and 1: a mode that holds more than that on the
    far half moves a flow by more than the tolerance over its pair. Levels at one k within
    ``reach`` of each other take part in a step together and the flow sees only their
    summed weight, so such a cluster is shared only when that sum lies farther than half
    the tolerance per level from a whole number (two degenerate modes of opposite edges
    mixed by a splitting the k-step cannot resolve are not). Returns the energies of these
    levels, and the narrowest splitting of two of them that neighbour each other at one k
    (the finite-size gap the hybridization opens where it is sampled; None without such a
    pair).
    """
    found: list[float] = []
    narrowest: float | None = None
    for e, p, m in zip(levels, half, middle):
        inside = np.flatnonzero((e > gap[0]) & (e < gap[1]))
        if not inside.size:
            continue
        cuts = np.flatnonzero(np.diff(e[inside]) > reach) + 1
        shared = np.zeros(len(e), bool)
        for group in np.split(inside, cuts):
            total = float(p[group].sum())
            mixed = abs(total - round(total)) > 0.5 * CROSSING_TOLERANCE * len(group)
            for index in group:
                within = min(energies) <= float(e[index]) <= max(energies)
                if within and (mixed or m[index] >= CROSSING_BULK_INTERIOR):
                    shared[index] = True
        found += [float(x) for x in e[shared]]
        pairs = np.flatnonzero(shared[:-1] & shared[1:])
        if pairs.size:
            split = float(np.min(e[pairs + 1] - e[pairs]))
            narrowest = split if narrowest is None else min(narrowest, split)
    return found, narrowest


def _mev(value: float) -> str:
    return f"{value * 1e3:+.1f}"


def mesh_resolution(
    lo: float | None,
    hi: float | None,
    overshoot: Sequence[float | None],
    local: Sequence[float] | None = None,
) -> tuple[float | None, float | None, str | None]:
    """The bulk gap the bulk mesh guarantees around E_F, and why the mesh does not resolve the gap.

    Each edge of the sampled gap [``lo``, ``hi``] can lie beyond its samples
    by ``overshoot`` (:func:`bulk_overshoot`; the driver's ``_resolution``), so the true gap holds at least
    [lo + valence, hi - conduction]. The mesh resolves the gap when that holds
    E_F and the two overshoots together, v*dk, are at most
    CROSSING_RESOLUTION of the sampled gap: the true gap is then at least
    v*dk and at least half the sampled one. v*dk bounds the band edges'
    in-plane step, so a gapped Dirac point puts at most
    2 arctan(v*dk / gap) <= pi/2 of Berry flux into one strip of the bulk
    winding, the bound its step must meet anyway, and same-sign Dirac points
    sharing a strip (partners on a line the mesh never samples) fail that
    step instead of wrapping to a small one. ``local`` holds [steps, gap] of
    the band pair across E_F that comes closest to the same bound at one
    point: its Dirac point and flux hide between samples wherever it lies in
    energy, so it must meet the bound too (a Dirac point then puts at most
    pi/3 into one strip). Returns the guaranteed edges (None where the gap has
    no edge) and, where the mesh does not resolve the gap, why.
    """
    over_v, over_c = overshoot[0], overshoot[1]
    top = lo if lo is None or over_v is None else lo + over_v
    bottom = hi if hi is None or over_c is None else hi - over_c
    text = None
    if lo is not None and hi is not None:
        total = float(over_v or 0.0) + float(over_c or 0.0)
        if total > CROSSING_RESOLUTION * (hi - lo):
            text = (
                f"the k mesh does not resolve the bulk gap (v*dk = {total * 1e3:.1f} "
                f"meV against a sampled gap of {(hi - lo) * 1e3:.1f} meV): raise "
                "slab_kpoints"
            )
    if text is None:
        for name, edge, over, guaranteed, inside in (
            ("valence", lo, over_v, top, top is None or top < 0.0),
            ("conduction", hi, over_c, bottom, bottom is None or bottom > 0.0),
        ):
            if not inside and edge is not None and guaranteed is not None:
                text = (
                    f"the k mesh does not resolve the bulk gap at E_F (v*dk = "
                    f"{float(over or 0.0) * 1e3:.1f} meV against the {name} band "
                    f"edge sampled {_mev(edge)} meV from E_F): raise slab_kpoints"
                )
                break
    if text is None and local is not None:
        steps, gap = float(local[0]), float(local[1])
        if steps > CROSSING_RESOLUTION * gap:
            text = (
                "the k mesh does not resolve the gap between a band below E_F and "
                f"one above it where they meet (v*dk = {steps * 1e3:.1f} meV against "
                f"a gap of {gap * 1e3:.1f} meV there): raise slab_kpoints"
            )
    return top, bottom, text


def _berry_text(step: float, mesh: str) -> str:
    return (
        f"the bulk Berry phase changes by {step:.2f} rad between neighbouring "
        f"k-points of the {mesh} bulk mesh, so its Chern number is not resolved: "
        "use more slab_kpoints"
    )


def edge_crossings(
    energies: Sequence,
    half: Sequence,
    middle: Sequence,
    *,
    periodic: bool = False,
    window: float | None = None,
    interior: bool = True,
    bulk: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Edge states crossing E_F as a weighted spectral flow, checked across the bulk gap; the driver's rule.

    Energies are relative to E_F; ``half`` and ``middle`` are each level's
    weight on the first half of the cells and on the middle half
    (:func:`half_ribbon_weights`). A level is bulk-like when its middle-half
    weight is at least 1/4. Per step k_j -> k_j+1 (the last point back to the
    first when ``periodic``, a ribbon's 1D zone) the levels are aligned by rank
    around the reference (:func:`align_ranks`) and the first-half weight that moves
    through it is summed (:func:`flow_at`): the left flow, a real number. The
    right flow is the net level count minus it; the two close to zero around a
    ribbon's zone. A level within 1e-5 eV of a reference energy (the weights'
    degeneracy tolerance) counts as at or above it, as one exactly at it does,
    so a flat band there does not cross it back and forth with rounding noise.

    ``bulk`` is the bulk check at the same in-plane k (:func:`bulk_at_fermi`).
    Its gap edges can lie beyond its samples by ``overshoot_eV``
    (:func:`bulk_overshoot`); the gap (``window_eV``) is the one that leaves,
    the bulk gap the mesh guarantees (:func:`mesh_resolution`), within ± ``window``
    (without a ``window`` the largest |E| stored). Without ``bulk`` it runs
    from the highest bulk-like level below E_F to the lowest at or above it
    over all k. The reach of a k-step is 1.5 times the largest step of an
    aligned pair seen (``step_shift_eV``): a band's steepest step falls inside
    a step where two levels swap, where it is not seen. The flows are evaluated
    at E_F and at 9 energies evenly across the gap, kept a reach inside the
    sampled edges, past the ribbon's own levels there (``per_energy``:
    ``left_flow``, ``right_flow`` and the middle-half ``bulk_flow``; a sample
    within 1e-5 eV of E_F is E_F's row).

    ``status`` is ``resolved`` when at every energy the left flow lies within
    0.1 (``tolerance``) of the same integer, that integer is the bulk Chern
    number (``bulk["chern"]``, of the bands below E_F with the periodic axis
    first, where the bulk check has one) and nothing below undermines it; a
    gap that holds no edge level is ``resolved`` only on a trusted bulk Chern
    number of 0 (the bulk mesh resolves the gap and its Berry phase steps
    below pi/2) with no level crossing it, never a nonzero count from absence.
    ``metal`` when the bulk bands cross E_F (``bulk["metal"]``; without
    ``bulk``, when a bulk-like level crosses E_F); ``open_path`` for a slab's
    open surface path, informational; otherwise ``unresolved``. ``reason``
    says why and ``checks`` names each check that failed with the energies it
    failed at (none for a check of the whole gap): ``k_mesh``, a bulk mesh
    that does not resolve the bulk gap (its edges may lie beyond the samples
    by more than half the sampled gap, or past E_F, or a band pair across E_F
    steps by more than half its own gap), so neither the gap nor the bulk
    Chern number is trusted; ``hybridised``, a level inside the bulk
    gap between the lowest and highest energy evaluated that is bulk-like or
    shared by the ribbon's two halves (:func:`talking_levels`); ``chern``, a
    flow other than the bulk Chern number (a mode hybridised out of the gap),
    or ``berry_step``, a Chern number the bulk mesh does not resolve;
    ``gap_edge``, a bulk band edge or the window edge within a reach of E_F;
    ``absence``, no level in the gap without a trusted bulk Chern number of 0
    (a count of 0 would rest on absence); ``strays`` and ``loose``, levels
    appearing or vanishing inside the window (coarse k or a small window);
    ``touches``, a bulk-like level taking part in a crossing; ``open_loops``,
    crossings that do not close around the zone; ``fractional`` and
    ``varying``, flows off an integer or rounding differently across the gap;
    and ``interior``, ``empty_window``, ``unfollowed``, ``metal`` or
    ``open_path``. ``left`` and ``right`` hold the crossings at E_F: ``up``
    and ``down`` count, step by step, the levels of that edge (more than half
    of them on its half) that cross upwards and downwards (:func:`exchanges`)
    and are upper bounds, the most crossings each step allows: modes of one
    edge that pass E_F within a step of each other can be paired more than
    one way (two Kramers pairs crossing in the same steps may read 4 up and 4
    down where each pair crosses once each way); ``up - down`` is the net,
    the rounded flow, certified only when ``resolved``.
    """
    levels: list[np.ndarray] = []
    first: list[np.ndarray] = []
    inner: list[np.ndarray] = []
    for e, p, m in zip(energies, half, middle):
        e = np.asarray(e, float)
        order = np.argsort(e, kind="stable")
        levels.append(e[order])
        first.append(np.asarray(p, float)[order])
        inner.append(np.asarray(m, float)[order])
    count = len(levels)
    steps = [(j, j + 1) for j in range(count - 1)]
    if periodic and count > 1:
        steps.append((count - 1, 0))
    if window is not None:
        span = float(window)
    else:
        span = max((float(np.abs(e).max()) for e in levels if e.size), default=0.0)
    bulk_metal = None if bulk is None else bool(bulk.get("metal"))
    bulk_gap = None if bulk is None or bulk_metal else bulk.get("gap_eV")
    lo: float | None = None
    hi: float | None = None
    mesh: tuple[float | None, float | None, str | None] | None = None
    if bulk_gap is not None:
        lo, hi = bulk_gap[0], bulk_gap[1]
        overshoot = (bulk or {}).get("overshoot_eV")
        if overshoot is not None:
            mesh = mesh_resolution(lo, hi, overshoot, (bulk or {}).get("local_overshoot_eV"))
    else:
        for e, m in zip(levels, inner):
            spread_out = m >= CROSSING_BULK_INTERIOR
            below, above = e[spread_out & (e < 0.0)], e[spread_out & (e >= 0.0)]
            if below.size:
                lo = float(below.max()) if lo is None else max(lo, float(below.max()))
            if above.size:
                hi = float(above.min()) if hi is None else min(hi, float(above.min()))
    sampled = [
        -span if lo is None else max(-span, float(lo)),
        span if hi is None else min(span, float(hi)),
    ]
    # The gap the counts are checked across: the one the bulk mesh guarantees,
    # where it holds any energy.
    gap = list(sampled)
    if mesh is not None and (mesh[0] is None or mesh[1] is None or mesh[0] < mesh[1]):
        gap = [
            -span if mesh[0] is None else max(-span, float(mesh[0])),
            span if mesh[1] is None else min(span, float(mesh[1])),
        ]
    provisional = [
        gap[0] + (gap[1] - gap[0]) * i / (CROSSING_GAP_SAMPLES + 1)
        for i in range(1, CROSSING_GAP_SAMPLES + 1)
    ]
    # A level within the tolerance of a reference energy counts as at or above
    # it, as one exactly at it does: the flows at an energy are those just
    # below it.
    shift = step_shift(
        levels,
        steps,
        [e - CROSSING_ENERGY_TOL for e in (0.0, *provisional)],
        window,
    )
    flows: dict[float, dict[str, Any]] = {}
    lo_c = hi_c = 0.0
    reach = 0.0
    if shift is not None:
        reach = CROSSING_REACH * shift
        flows[0.0] = flow_at(levels, first, inner, steps, -CROSSING_ENERGY_TOL, reach, window)
        # A reach inside the sampled edges, past the ribbon's own levels there,
        # and inside the guaranteed gap.
        lo_c = max(sampled[0] + reach, gap[0])
        hi_c = min(sampled[1] - reach, gap[1])
        if not (bulk_metal or (bulk_metal is None and flows[0.0]["bulk_crossings"])) and hi_c > lo_c:
            for i in range(1, CROSSING_GAP_SAMPLES + 1):
                energy = lo_c + (hi_c - lo_c) * i / (CROSSING_GAP_SAMPLES + 1)
                if all(abs(energy - other) > CROSSING_ENERGY_TOL for other in flows):
                    flows[energy] = flow_at(
                        levels,
                        first,
                        inner,
                        steps,
                        energy - CROSSING_ENERGY_TOL,
                        reach,
                        window,
                    )
    at_fermi = flows.get(0.0)
    spread = at_fermi["bulk_crossings"] if at_fermi is not None else 0
    metal = bulk_metal if bulk_metal is not None else spread > 0
    rows = [
        {
            "energy_eV": energy,
            "left_flow": flow["left"],
            "right_flow": flow["net"] - flow["left"],
            "bulk_flow": flow["bulk"],
        }
        for energy, flow in sorted(flows.items())
    ]
    span_mev = f"[{_mev(gap[0])}, {_mev(gap[1])}] meV"
    bounds = "the bulk gap" if bulk_gap is not None else "the gap"
    chern = None if bulk is None else bulk.get("chern")
    step = float((bulk or {}).get("berry_phase_step") or 0.0)
    mesh_text = None if mesh is None else mesh[2]
    # The bulk Chern number is trusted where the bulk mesh resolves the bulk gap
    # and its Berry phase steps below the bound between neighbouring k-points.
    trusted = chern is not None and mesh is not None and mesh_text is None and step < CROSSING_BERRY_STEP
    mesh_name = "x".join(str(n) for n in (bulk or {}).get("k_points") or [])
    # Each reason: the check that failed, the energies it failed at (none for a
    # check of the whole gap) and its text.
    problems: list[tuple[str, list[float], str]] = []
    if not periodic:
        status = "open_path"
        problems.append(
            (
                "open_path",
                [],
                "an open path, not a closed loop: the counts are informational",
            )
        )
    elif metal:
        status = "metal"
        if bulk_metal:
            text = "the bulk bands cross E_F: no gap around E_F"
        else:
            text = f"bulk-like levels cross E_F ({spread} crossings): no gap around E_F"
        problems.append(("metal", [], text))
    elif not interior:
        status = "unresolved"
        problems.append(
            (
                "interior",
                [],
                "no cell lies between the two edge regions: use more slab_cells",
            )
        )
    elif not any(e.size for e in levels):
        if trusted and chern == 0 and gap[0] < 0.0 < gap[1]:
            status = "resolved"
            reason = (
                "the bulk Chern number is 0 and no level crosses the gap: no level "
                f"lies within the window at any k, in {bounds} {span_mev}"
            )
        else:
            status = "unresolved"
            problems.append(
                (
                    "empty_window",
                    [],
                    "no level lies within the window at any k: nothing to count, and "
                    "a count of 0 would rest on absence"
                    + ("" if chern is None else f"; the bulk Chern number is {chern:+d}"),
                )
            )
            if mesh_text is not None:
                problems.append(("k_mesh", [], mesh_text))
            if chern is not None and step >= CROSSING_BERRY_STEP:
                problems.append(("berry_step", [], _berry_text(step, mesh_name)))
    elif shift is None:
        status = "unresolved"
        problems.append(
            (
                "unfollowed",
                [],
                "no level is followed from one k-point to the next: use more "
                "slab_kpoints or a wider slab_window_eV",
            )
        )
    else:
        reach_mev = f"{reach * 1e3:.1f} meV"
        if mesh_text is not None:
            problems.append(("k_mesh", [], mesh_text))
        talking, narrowest = talking_levels(levels, first, inner, gap, sorted(flows), reach)
        if talking:
            opened = (
                f"finite-size gap {narrowest * 1e3:.1f} meV inside {bounds} {span_mev}"
                if narrowest is not None
                else f"inside {bounds} {span_mev}"
            )
            plural = "s" if len(talking) != 1 else ""
            problems.append(
                (
                    "hybridised",
                    [],
                    f"edge modes hybridise across the ribbon ({opened}: "
                    f"{len(talking)} level{plural} from {_mev(min(talking))} to "
                    f"{_mev(max(talking))} meV bulk-like or shared by its two "
                    "halves): use more slab_cells",
                )
            )
        if not lo_c < 0.0 < hi_c:
            edge = "a bulk band edge" if bulk_gap is not None else "a bulk-like level"
            problems.append(
                (
                    "gap_edge",
                    [],
                    f"{edge} or the window edge lies within a k-step's reach "
                    f"({reach_mev}) of E_F, in {bounds} {span_mev}: use more "
                    "slab_kpoints, or more slab_cells if edge states bound the gap",
                )
            )
        edge_free = not any(
            np.any((e > gap[0]) & (e < gap[1]) & (m < CROSSING_BULK_INTERIOR)) for e, m in zip(levels, inner)
        )
        if edge_free and not (trusted and chern == 0):
            problems.append(
                (
                    "absence",
                    [],
                    f"no level lies in {bounds} {span_mev} at any k: a count of 0 "
                    "would rest on absence (a trivial gap and edge states hybridised "
                    "out of the gap look the same): use more slab_cells"
                    + ("" if chern is None else f"; the bulk Chern number is {chern:+d}"),
                )
            )
        strays = {e: [abs(x) for x in flow["strays"]] for e, flow in flows.items()}
        if any(strays.values()):
            nearest = min(x for values in strays.values() for x in values)
            problems.append(
                (
                    "strays",
                    [e for e, values in strays.items() if values],
                    "levels appear or disappear between neighbouring k-points "
                    f"{(span - nearest) * 1e3:.1f} meV inside the window, beyond a "
                    f"k-step's reach ({reach_mev}): use more slab_kpoints or a wider "
                    "slab_window_eV",
                )
            )
        loose = [(e, x) for e, flow in flows.items() for x in flow["loose"]]
        if loose:
            energy, x = min(loose, key=lambda item: span - abs(item[1]))
            problems.append(
                (
                    "loose",
                    sorted({e for e, _ in loose}),
                    f"a level within a k-step's reach ({reach_mev}) of the levels "
                    f"crossing {_mev(energy)} meV has no partner at the neighbouring "
                    f"k-point: it enters or leaves the window "
                    f"{(span - abs(x)) * 1e3:.1f} meV inside its edge: use more "
                    "slab_kpoints or a wider slab_window_eV",
                )
            )
        touched = [e for e, flow in flows.items() if flow["touches"]]
        if touched:
            problems.append(
                (
                    "touches",
                    touched,
                    f"a bulk-like level takes part in a crossing at {_mev(min(touched))} meV",
                )
            )
        open_loops = [e for e, flow in flows.items() if flow["net"]]
        if open_loops:
            problems.append(
                (
                    "open_loops",
                    open_loops,
                    f"the crossings at {_mev(open_loops[0])} meV do not close around "
                    "the zone: use more slab_kpoints",
                )
            )
        lefts = [row["left_flow"] for row in rows]
        far = [
            row["energy_eV"]
            for row in rows
            if abs(row["left_flow"] - round(row["left_flow"])) >= CROSSING_TOLERANCE
        ]
        if far:
            values = [flows[e]["left"] for e in far]
            problems.append(
                (
                    "fractional",
                    far,
                    f"the left flow is {min(values):+.3f} to {max(values):+.3f} at "
                    f"{len(far)} of {len(rows)} energies, not within "
                    f"{CROSSING_TOLERANCE} of an integer: the edges hybridise or their "
                    "modes are not separated: use more slab_cells",
                )
            )
        rounded = sorted({int(round(x)) for x in lefts})
        if len(rounded) > 1:
            problems.append(
                (
                    "varying",
                    [],
                    "the rounded left flow varies with energy across the gap ("
                    + "/".join(f"{v:+d}" if v else "0" for v in rounded)
                    + ")",
                )
            )
        if chern is not None:
            if step >= CROSSING_BERRY_STEP:
                problems.append(("berry_step", [], _berry_text(step, mesh_name)))
            elif mesh_text is None and len(rounded) == 1 and rounded[0] != chern:
                problems.append(
                    (
                        "chern",
                        [],
                        f"the left flow rounds to {rounded[0]:+d}, not to the bulk "
                        f"Chern number {chern:+d} of the bands below E_F ({mesh_name} "
                        "mesh): an edge mode hybridised out of the gap or is missed: "
                        "use more slab_cells",
                    )
                )
        if problems:
            status = "unresolved"
        else:
            worst = max(abs(x - round(x)) for x in lefts)
            agrees = "" if chern is None else ", the bulk Chern number,"
            status = "resolved"
            if edge_free:
                reason = (
                    "the bulk Chern number is 0 and no level crosses the gap: no edge "
                    f"level lies in {bounds} {span_mev} at any k, and the left and "
                    f"right flows at {len(rows)} energies across it are each within "
                    f"{worst:.3f} of 0"
                )
            else:
                reason = (
                    f"left {rounded[0]:+d}{agrees} and right {-rounded[0]:+d} at "
                    f"{len(rows)} energies across {bounds} {span_mev}, each within "
                    f"{worst:.3f} of the integer"
                )
    if problems:
        reason = "; ".join(text for _, _, text in problems)
    per_step = at_fermi["per_step"] if at_fermi is not None else []
    net_left = int(round(at_fermi["left"])) if at_fermi is not None else 0
    net_right = int(round(at_fermi["net"] - at_fermi["left"])) if at_fermi is not None else 0
    return {
        "left": {**_crossing_tally(per_step, "left"), "net": net_left},
        "right": {**_crossing_tally(per_step, "right"), "net": net_right},
        "status": status,
        "reason": reason,
        "method": CROSSING_METHOD,
        "window_eV": None if metal or (shift is None and status != "resolved") else gap,
        "step_shift_eV": shift,
        "energies_eV": [row["energy_eV"] for row in rows],
        "per_energy": rows,
        "tolerance": CROSSING_TOLERANCE,
        "checks": {check: where for check, where, _ in problems},
    }


def crossing_margins(
    energies: Sequence,
    half: Sequence,
    middle: Sequence,
    crossings: Mapping[str, Any],
    *,
    periodic: bool,
    window: float | None,
    tolerances: Mapping[str, float],
    bulk: Mapping[str, Any] | None = None,
) -> dict[str, dict[float | str, float]]:
    """How close each decision behind ``crossings`` came to flipping, in units of its tolerance.

    ``crossings`` is the block :func:`edge_crossings` made from these levels and weights;
    ``tolerances`` holds how far an ``energy`` (eV), a ``weight`` and a ``flow`` may move
    between two runs that agree; ``bulk`` is the run's bulk check. Returns ``energies``, for every
    energy the flow was evaluated at the smallest margin of the decisions taken there, and
    ``checks``, for each check of the whole gap its own; a margin of 1 or less is a decision
    within tolerance of flipping (inf where nothing was decided).

    At an energy: the gap between the chosen offset's cost and the next valid one at each
    k-step; every swap test of :func:`taking_part` (the first two ranks, each extension and
    the one that stopped it) and every test of a level without a partner against the reach;
    every unpaired level's distance inside the window against the reach (a stray); every
    level's distance from the reference the flow is taken at (the energy less the
    tolerance within which a level counts as at it), at every k; the middle-half weights of
    the levels taking part against 1/4 (a bulk-like level taking part, a bulk-like crossing)
    and their first-half weights against 1/2 (the edge a crossing is counted on); the left
    and right flows against a half-integer (rounding) and against the tolerance from an
    integer; and, at a sampled energy, its distance from E_F against the tolerance within
    which it is E_F's row. Of the whole gap: ``rows``, each sample's distance from E_F and
    the samples' spacing against that tolerance, and the range of sampled energies against
    empty; ``step_shift``, 0 when the step shift could move (an alignment at E_F or at a
    provisional energy across the gap within tolerance of another, through a near-tie of
    offset costs or a level at that energy, giving a larger step or losing the one step that
    sets it), inf otherwise; ``gap``, without a bulk gap, each level's middle-half weight
    against 1/4, which may move the gap the ribbon's bulk-like levels leave; ``k_mesh``,
    the bulk overshoot of both edges against half the sampled gap, each guaranteed edge
    against E_F, and the band pair's steps against half its gap (:func:`mesh_resolution`); ``guaranteed``, the guaranteed gap against empty
    (the counts then use the sampled one); ``gap_edge``, E_F against the reach inside the
    sampled edges and the guaranteed ones; ``absence``, whether a non-bulk-like level could
    enter or leave the gap, and with a bulk Chern number of 0 whether the mesh's resolution
    could flip (the caller adds the Berry-phase step); ``hybridised``, each level of
    :func:`talking_levels` against the gap edges, the evaluated span, 1/4 and its cluster's
    bound, and each cluster cut against the reach; ``varying`` and ``chern``, the flows
    against a half-integer, and for ``chern`` the mesh's resolution (the caller adds the
    bulk Berry-phase step to ``chern`` and ``berry_step``); ``empty_window``, a level
    against the window edge, and with a bulk Chern number of 0 the mesh's resolution.
    """
    energy_tol = float(tolerances["energy"])
    weight_tol = float(tolerances["weight"])
    flow_tol = float(tolerances["flow"])
    levels: list[np.ndarray] = []
    first: list[np.ndarray] = []
    inner: list[np.ndarray] = []
    for e, p, m in zip(energies, half, middle):
        e = np.asarray(e, float)
        order = np.argsort(e, kind="stable")
        levels.append(e[order])
        first.append(np.asarray(p, float)[order])
        inner.append(np.asarray(m, float)[order])
    count = len(levels)
    steps = [(j, j + 1) for j in range(count - 1)]
    if periodic and count > 1:
        steps.append((count - 1, 0))
    if window is not None:
        span = float(window)
    else:
        span = max((float(np.abs(e).max()) for e in levels if e.size), default=0.0)
    every = np.concatenate(levels) if levels else np.zeros(0)
    checks: dict[float | str, float] = {}
    # Whether the bulk mesh resolves the gap, and the gap the counts are then checked across.
    bulk_gap = bulk.get("gap_eV") if isinstance(bulk, Mapping) and not bulk.get("metal") else None
    overshoot = bulk.get("overshoot_eV") if isinstance(bulk, Mapping) else None
    if bulk_gap is not None and overshoot is not None:
        guaranteed_lo, guaranteed_hi, _ = mesh_resolution(bulk_gap[0], bulk_gap[1], overshoot)
        mesh = [abs(x) / energy_tol for x in (guaranteed_lo, guaranteed_hi) if x is not None]
        if bulk_gap[0] is not None and bulk_gap[1] is not None:
            total = float(overshoot[0] or 0.0) + float(overshoot[1] or 0.0)
            half_gap = CROSSING_RESOLUTION * (float(bulk_gap[1]) - float(bulk_gap[0]))
            # Each overshoot moves by up to 4 tau_E and the gap by 2: 9 tau_E in all.
            mesh.append(abs(total - half_gap) / (2.0 * energy_tol))
        local = bulk.get("local_overshoot_eV") if isinstance(bulk, Mapping) else None
        if local is not None:
            # The pair that comes closest: its steps (each a half step, 4 tau_E) against its gap.
            mesh.append(abs(float(local[0]) - CROSSING_RESOLUTION * float(local[1])) / (2.0 * energy_tol))
        checks["k_mesh"] = min(mesh, default=math.inf)
        if guaranteed_lo is not None and guaranteed_hi is not None:
            checks["guaranteed"] = abs(guaranteed_hi - guaranteed_lo) / energy_tol
    # A trusted bulk Chern number of 0 certifies an empty gap: the mesh's resolution decides
    # (the caller adds the Berry-phase step).
    zero = isinstance(bulk, Mapping) and bulk.get("chern") == 0
    window_edge = float(np.min(np.abs(span - np.abs(every)))) / energy_tol if every.size else math.inf
    if every.size or zero:
        checks["empty_window"] = min(window_edge, checks.get("k_mesh", math.inf) if zero else math.inf)
    shift = crossings.get("step_shift_eV")
    gap = crossings.get("window_eV")
    margins: dict[float | str, float] = {}
    if shift is None or not isinstance(gap, list) or len(gap) != 2:
        return {"energies": margins, "checks": checks}
    reach = CROSSING_REACH * float(shift)
    lo, hi = float(gap[0]), float(gap[1])

    def flow_margin(flow: float) -> float:
        distance = abs(flow - round(flow))
        return min(abs(distance - 0.5), abs(distance - CROSSING_TOLERANCE)) / flow_tol

    rows = {float(row["energy_eV"]): row for row in crossings.get("per_energy") or []}
    for energy in crossings.get("energies_eV") or []:
        energy = float(energy)
        # The flow at an energy is taken just below it: a level within the tolerance counts as at it.
        e_ref = energy - CROSSING_ENERGY_TOL
        margin = float(np.min(np.abs(every - e_ref))) / energy_tol if every.size else math.inf
        if energy != 0.0:
            margin = min(margin, abs(abs(energy) - CROSSING_ENERGY_TOL) / energy_tol)
        row = rows.get(energy)
        if row is not None:
            for key in ("left_flow", "right_flow"):
                margin = min(margin, flow_margin(float(row[key])))
        for j, j1 in steps:
            a, b = levels[j], levels[j1]
            i0, j0, keys = _offset_costs(a, b, e_ref, window)
            if not keys:
                unpaired = [*a, *b]
                o = None
            else:
                o = keys[0][2]
                if len(keys) > 1:
                    margin = min(margin, (keys[1][0] - keys[0][0]) / energy_tol)
                lo_r = max(-i0, -(j0 + o))
                hi_r = min(len(a) - i0, len(b) - j0 - o) - 1
                unpaired = [
                    *a[: i0 + lo_r],
                    *a[i0 + hi_r + 1 :],
                    *b[: j0 + lo_r + o],
                    *b[j0 + hi_r + o + 1 :],
                ]
            if window is not None:
                for x in unpaired:
                    margin = min(margin, abs(window - abs(float(x)) - reach) / energy_tol)
            if o is None:
                continue

            def paired(r: int) -> bool:
                return 0 <= i0 + r < len(a) and 0 <= j0 + r + o < len(b)

            def tested(lo_s: int, hi_s: int) -> bool:
                up = abs(float(b[j0 + hi_s + o]) - float(a[i0 + lo_s]))
                down = abs(float(a[i0 + hi_s]) - float(b[j0 + lo_s + o]))
                nonlocal margin
                margin = min(margin, abs(up - reach) / energy_tol, abs(down - reach) / energy_tol)
                return up <= reach and down <= reach

            if o > 0:
                low, high = -o, -1
            elif o < 0:
                low, high = 0, -o - 1
            elif paired(-1) and paired(0) and tested(-1, 0):
                low, high = -1, 0
            else:
                continue
            while paired(low - 1) and tested(low - 1, low):
                low -= 1
            while paired(high + 1) and tested(high, high + 1):
                high += 1
            for r, end in ((low - 1, low), (high + 1, high)):
                if paired(r):
                    continue
                ia, ib = i0 + r, j0 + r + o
                if 0 <= ia < len(a):
                    margin = min(margin, abs(abs(float(b[j0 + end + o]) - float(a[ia])) - reach) / energy_tol)
                if 0 <= ib < len(b):
                    margin = min(margin, abs(abs(float(a[i0 + end]) - float(b[ib])) - reach) / energy_tol)
            for r in range(low, high + 1):
                ia, ib = i0 + r, j0 + r + o
                inner_pair = (float(inner[j][ia]), float(inner[j1][ib]))
                margin = min(margin, abs(max(inner_pair) - CROSSING_BULK_INTERIOR) / weight_tol)
                if (a[ia] < e_ref) != (b[ib] < e_ref):
                    margin = min(margin, abs(0.5 * sum(inner_pair) - CROSSING_BULK_INTERIOR) / weight_tol)
                for weight in (first[j][ia], first[j1][ib]):
                    margin = min(margin, abs(float(weight) - 0.5) / weight_tol)
        margins[energy] = margin
    # Checks of the whole gap.
    provisional = [
        lo + (hi - lo) * i / (CROSSING_GAP_SAMPLES + 1) for i in range(1, CROSSING_GAP_SAMPLES + 1)
    ]

    # The step shift is the largest aligned step at E_F and at these energies: it moves only
    # where an alignment within tolerance of another (a near-tie of offset costs, or a level
    # within tolerance of the energy, which moves the rank origin) would give a larger step,
    # or would lose the step that alone sets it.
    def largest(a: np.ndarray, b: np.ndarray, e_ref: float, alternative: bool) -> list[float]:
        i0, j0, keys = _offset_costs(a, b, e_ref, window)
        chosen = keys[1:2] if alternative else keys[:1]
        out = []
        for _, _, o in chosen:
            pairs = [
                abs(float(a[i0 + r]) - float(b[j0 + r + o]))
                for r in range(CROSSING_RANKS[0], CROSSING_RANKS[1] + 1)
                if 0 <= i0 + r < len(a) and 0 <= j0 + r + o < len(b)
            ]
            out.append(max(pairs, default=-math.inf))
        return out

    setting, alternatives = 0, []
    shift_value = float(shift)
    for e_ref in (x - CROSSING_ENERGY_TOL for x in (0.0, *provisional)):
        for j, j1 in steps:
            a, b = levels[j], levels[j1]
            _, _, keys = _offset_costs(a, b, e_ref, window)
            current = largest(a, b, e_ref, False)
            if current and current[0] >= shift_value - energy_tol:
                setting += 1
            options: list[float] = []
            if len(keys) > 1 and keys[1][0] - keys[0][0] <= energy_tol:
                options += largest(a, b, e_ref, True)
            if np.any(np.abs(a - e_ref) <= energy_tol) or np.any(np.abs(b - e_ref) <= energy_tol):
                for moved in (e_ref - 2 * energy_tol, e_ref + 2 * energy_tol):
                    options += largest(a, b, moved, False)
            if options:
                alternatives.append((bool(current) and current[0] >= shift_value - energy_tol, options))
    moves = any(top > shift_value + energy_tol for _, options in alternatives for top in options)
    moves = moves or (
        setting == 1
        and any(sets and min(options) < shift_value - energy_tol for sets, options in alternatives)
    )
    checks["step_shift"] = 0.0 if moves else math.inf
    if not isinstance(bulk, Mapping) or bulk.get("metal") or bulk.get("gap_eV") is None:
        interior_weights = np.concatenate(inner) if inner else np.zeros(0)
        if interior_weights.size:
            checks["gap"] = float(np.min(np.abs(interior_weights - CROSSING_BULK_INTERIOR))) / weight_tol
    # The samples lie a reach inside the sampled edges and inside the guaranteed gap.
    sampled = [lo, hi]
    if bulk_gap is not None:
        sampled = [
            -span if bulk_gap[0] is None else max(-span, float(bulk_gap[0])),
            span if bulk_gap[1] is None else min(span, float(bulk_gap[1])),
        ]
    lo_c, hi_c = max(sampled[0] + reach, lo), min(sampled[1] - reach, hi)
    checks["gap_edge"] = min(abs(lo_c), abs(hi_c)) / energy_tol
    # Which rows there are: whether the range of sampled energies is empty, and whether a
    # sampled energy lies within the tolerance of E_F or of another (one row then, two
    # otherwise).
    samples = [
        lo_c + (hi_c - lo_c) * i / (CROSSING_GAP_SAMPLES + 1) for i in range(1, CROSSING_GAP_SAMPLES + 1)
    ]
    sample_step = (hi_c - lo_c) / (CROSSING_GAP_SAMPLES + 1)
    checks["rows"] = (
        min(
            [
                abs(hi_c - lo_c),
                abs(abs(sample_step) - CROSSING_ENERGY_TOL),
                *(abs(abs(x) - CROSSING_ENERGY_TOL) for x in samples),
            ]
        )
        / energy_tol
    )
    rounding = [
        abs(abs(float(row["left_flow"]) - round(float(row["left_flow"]))) - 0.5) for row in rows.values()
    ]
    checks["varying"] = min(rounding, default=math.inf) / flow_tol
    # The flows are compared with the bulk Chern number only where the mesh resolves the gap.
    checks["chern"] = min(checks["varying"], checks.get("k_mesh", math.inf))
    # Absence: whether some level could enter the gap with a middle-half weight below 1/4,
    # or every one inside could leave.
    qualifying, could_enter = [], math.inf
    for e, m in zip(levels, inner):
        for x, w in zip(e, m):
            inside = lo < x < hi
            edge = min(abs(x - lo), abs(x - hi)) / energy_tol
            interior = abs(float(w) - CROSSING_BULK_INTERIOR) / weight_tol
            if inside and w < CROSSING_BULK_INTERIOR:
                qualifying.append(min(edge, interior))
            else:
                could_enter = min(
                    could_enter, max(0.0 if inside else edge, 0.0 if w < CROSSING_BULK_INTERIOR else interior)
                )
    checks["absence"] = max(qualifying) if qualifying else could_enter
    if zero:
        checks["absence"] = min(checks["absence"], checks.get("k_mesh", math.inf))
    # The levels the edges may share: every decision of talking_levels on the levels in the gap.
    evaluated = [float(x) for x in crossings.get("energies_eV") or []]
    hybridised = math.inf
    if evaluated:
        bottom, top = min(evaluated), max(evaluated)
        for e, p, m in zip(levels, first, inner):
            near_gap = np.flatnonzero((e > lo - energy_tol) & (e < hi + energy_tol))
            if not near_gap.size:
                continue
            spacing = np.diff(e[near_gap])
            if spacing.size:
                hybridised = min(hybridised, float(np.min(np.abs(spacing - reach))) / energy_tol)
            inside = np.flatnonzero((e > lo) & (e < hi))
            cuts = np.flatnonzero(np.diff(e[inside]) > reach) + 1
            for group in np.split(inside, cuts) if inside.size else []:
                total = float(p[group].sum())
                bound = 0.5 * CROSSING_TOLERANCE * len(group)
                hybridised = min(
                    hybridised, abs(abs(total - round(total)) - bound) / (weight_tol * len(group))
                )
            for index in near_gap:
                x = float(e[index])
                # A level at a gap edge enters or leaves its cluster, wherever the cluster lies.
                hybridised = min(hybridised, min(abs(x - lo), abs(x - hi)) / energy_tol)
                if not bottom - energy_tol <= x <= top + energy_tol:
                    continue
                hybridised = min(
                    hybridised,
                    min(abs(x - bottom), abs(x - top)) / energy_tol,
                    abs(float(m[index]) - CROSSING_BULK_INTERIOR) / weight_tol,
                )
    checks["hybridised"] = hybridised
    return {"energies": margins, "checks": checks}


def legacy_edge_crossings(
    energies: Sequence,
    left: Sequence,
    right: Sequence,
    *,
    periodic: bool = False,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """The count of the drivers whose crossings record no ``method``: nearest-energy matching on each edge.

    A state whose weight on an edge is at least ``threshold`` at k_j is matched
    to the nearest-energy state localized on the same edge at k_(j+1), with no
    energy bound and not one to one; a sign change of its energy is a crossing,
    up or down by the sign of the step, and ``net`` is up minus down.
    ``periodic`` also pairs the last k-point with the first. A match can jump
    across the whole window, so these counts are wrong on dense ribbons; the
    comparator recomputes them only to check references of those drivers.
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
