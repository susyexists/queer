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


def bulk_at_fermi(
    translations, hopping_matrices, finite_axis: int, k_par, n_perp: int, chunk: int = 64
) -> dict[str, Any]:
    """Whether the bulk bands cross E_F (energy zero) at the slab's in-plane k-points.

    The bulk levels of H(k) = sum_R exp(2 pi i k.R) H(R) are sampled at every
    in-plane k of ``k_par`` and at ``n_perp`` points i / n_perp along
    ``finite_axis``. A band below E_F at one sample and at or above it at another
    crosses E_F in between: ``metal``. A pocket between samples can be missed; the
    slab's own levels then show it. Otherwise ``gap_eV`` holds the highest sampled
    level below E_F and the lowest at or above it (None on a side without one).
    """
    R = np.asarray(translations, dtype=float)
    H = np.asarray(hopping_matrices)
    n = H.shape[1]
    points = np.asarray(k_par, dtype=float).reshape(-1, 3)
    kpts = np.repeat(points, n_perp, axis=0)
    kpts[:, finite_axis] = np.tile(np.arange(n_perp) / n_perp, len(points))
    blocks = []
    for start in range(0, len(kpts), chunk):
        phases = np.exp(2j * np.pi * (kpts[start : start + chunk] @ R.T))
        hk = (phases @ H.reshape(len(R), n * n)).reshape(len(phases), n, n)
        blocks.append(np.linalg.eigvalsh(hk))
    levels = np.concatenate(blocks) if blocks else np.zeros((0, n))
    below = (levels < 0.0).sum(axis=1)
    metal = bool(below.size and int(np.min(below)) != int(np.max(below)))
    under, over = levels[levels < 0.0], levels[levels >= 0.0]
    gap = None
    if not metal:
        gap = [float(under.max()) if under.size else None, float(over.min()) if over.size else None]
    return {"metal": metal, "gap_eV": gap, "k_perp_points": int(n_perp)}


# Edge crossings, a weighted spectral flow: the rank offsets tried per k-step and the
# ranks around the reference energy they compare, the energies sampled across the gap
# besides E_F, the middle-half weight from which a level is bulk-like, the reach of one
# k-step in units of the largest level step seen, and how close to an integer a
# weighted net must be to count as that integer.
CROSSING_MAX_OFFSET = 3
CROSSING_RANKS = (-3, 2)
CROSSING_GAP_SAMPLES = 9
CROSSING_BULK_INTERIOR = 0.25
CROSSING_REACH = 1.5
CROSSING_TOLERANCE = 0.1
CROSSING_METHOD = "half_ribbon_flow"


def _offset_costs(
    a: np.ndarray, b: np.ndarray, e_ref: float, window: float | None
) -> tuple[int, int, list[tuple[float, int, int]]]:
    """``(i0, j0, keys)``: the ranks of :func:`align_ranks` and every valid offset's sort key, ascending."""
    i0 = int(np.searchsorted(a, e_ref))
    j0 = int(np.searchsorted(b, e_ref))
    keys: list[tuple[float, int, int]] = []
    for o in range(-CROSSING_MAX_OFFSET, CROSSING_MAX_OFFSET + 1):
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
    r + o at k_(j+1), o in [-3, 3]. o > 0: ranks -o..-1 crossed upwards; o < 0:
    ranks 0..|o|-1 downwards. The offset minimizes the mean step over ranks -3..2:
    a pair's energy step, and for a level without a partner the distance to the
    window edge it must have crossed (none can lack one without a ``window``). The
    levels an offset says crossed must have partners; ties go to the smaller |o|,
    then the lower o. Returns ``(i0, j0, o)``, o None when no offset pairs a level.
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
    weight above it at k_(j+1) minus that at k_j: a level rising through the
    reference adds the mean of its endpoints' weights, two that swap across it add
    the weight that moved up, and any pairing within the set gives the same sum.
    Returns the first-half (``left``) and middle-half (``bulk``) flows, the net level
    count (``net``), the left flow per step, the bulk-like crossings, whether a
    bulk-like level took part, and the levels that appear or disappear farther than
    ``reach`` inside the window or could swap with the set without a partner.
    """
    left = bulk = 0.0
    net = 0
    per_step: list[tuple[float, int]] = []
    bulk_crossings = 0
    touches = False
    strays: list[float] = []
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
        strays += loose
        step_left = 0.0
        for r in ranks:
            ia, ib = i0 + r, j0 + r + o
            sa = 1.0 if a[ia] < e_ref else -1.0
            sb = 1.0 if b[ib] < e_ref else -1.0
            step_left += 0.5 * (float(half[j][ia]) * sa - float(half[j1][ib]) * sb)
            bulk += 0.5 * (float(middle[j][ia]) * sa - float(middle[j1][ib]) * sb)
            inner = (float(middle[j][ia]), float(middle[j1][ib]))
            if max(inner) >= CROSSING_BULK_INTERIOR:
                touches = True
                if sa != sb and 0.5 * sum(inner) >= CROSSING_BULK_INTERIOR:
                    bulk_crossings += 1
        left += step_left
        net += o
        per_step.append((step_left, o))
    return {
        "left": left,
        "bulk": bulk,
        "net": net,
        "per_step": per_step,
        "bulk_crossings": bulk_crossings,
        "touches": touches,
        "strays": strays,
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


def _crossing_tally(per_step: Sequence[tuple[float, int]], side: str) -> dict[str, int]:
    """Crossings of one edge: each step's flow on that side, rounded (attributed past 1/2)."""
    up = down = 0
    for left, net in per_step:
        count = int(round(left if side == "left" else net - left))
        up += max(count, 0)
        down += max(-count, 0)
    return {"up": up, "down": down}


def _mev(value: float) -> str:
    return f"{value * 1e3:+.1f}"


def edge_crossings(
    energies: Sequence,
    half: Sequence,
    middle: Sequence,
    *,
    periodic: bool = False,
    window: float | None = None,
    interior: bool = True,
    bulk_metal: bool | None = None,
) -> dict[str, Any]:
    """Edge states crossing E_F as a weighted spectral flow, checked across the gap; the driver's rule.

    Energies are relative to E_F; ``half`` and ``middle`` are each level's weight on
    the first half of the cells and on the middle half (:func:`half_ribbon_weights`).
    A level is bulk-like when its middle-half weight is at least 1/4. Per step
    k_j -> k_(j+1) (the last point back to the first when ``periodic``, a ribbon's 1D
    zone) the levels are aligned by rank around the reference (:func:`align_ranks`)
    and the first-half weight that moves through it is summed (:func:`flow_at`): the
    left flow, a real number. The right flow is the net level count minus it; the two
    close to zero around a ribbon's zone.

    The gap around E_F (``window_eV``) runs from the highest bulk-like level below E_F
    to the lowest at or above it over all k (± ``window`` on a side without one;
    without a ``window`` the largest |E| stored). The reach of a k-step is 1.5 times
    the largest step of an aligned pair seen (``step_shift_eV``): a band's steepest
    step falls inside a step where two levels swap, where it is not seen. The flows
    are evaluated at E_F and at 9 energies evenly across the gap kept a reach from its
    edges (``per_energy``: ``left_flow``, ``right_flow`` and the middle-half
    ``bulk_flow``).

    ``status`` is ``resolved`` when at every energy the left flow lies within 0.1
    (``tolerance``) of the same integer and nothing below undermines it; ``metal``
    when the bulk bands cross E_F (``bulk_metal``, from :func:`bulk_at_fermi`;
    without it, when a bulk-like level crosses E_F); ``open_path`` for a slab's open
    surface path, informational; otherwise ``unresolved``, with a ``reason``:
    bulk-like levels crossing E_F over a bulk gap (edge states spread over the
    ribbon), a bulk-like level or the window edge within a reach of E_F, no level in
    the gap (a count of 0 would rest on absence: a trivial gap and edge states
    hybridized into the bulk look the same), levels appearing or vanishing inside the
    window (coarse k or a small window), fractional or varying flows (hybridized
    edges: more cells), or no cell between the two edge regions (``interior`` false).
    ``left`` and ``right`` hold the crossings at E_F: ``up`` and ``down`` count the
    steps whose flow on that side rounds to a positive or a negative integer, and
    ``net`` is the rounded flow, certified only when ``resolved``.
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
    lo: float | None = None
    hi: float | None = None
    for e, m in zip(levels, inner):
        bulk = m >= CROSSING_BULK_INTERIOR
        below, above = e[bulk & (e < 0.0)], e[bulk & (e >= 0.0)]
        if below.size:
            lo = float(below.max()) if lo is None else max(lo, float(below.max()))
        if above.size:
            hi = float(above.min()) if hi is None else min(hi, float(above.min()))
    if window is not None:
        span = float(window)
    else:
        span = max((float(np.abs(e).max()) for e in levels if e.size), default=0.0)
    gap = [-span if lo is None else lo, span if hi is None else hi]
    provisional = [
        gap[0] + (gap[1] - gap[0]) * i / (CROSSING_GAP_SAMPLES + 1)
        for i in range(1, CROSSING_GAP_SAMPLES + 1)
    ]
    shift = step_shift(levels, steps, [0.0, *provisional], window)
    flows: dict[float, dict[str, Any]] = {}
    lo_c = hi_c = 0.0
    reach = 0.0
    if shift is not None:
        reach = CROSSING_REACH * shift
        flows[0.0] = flow_at(levels, first, inner, steps, 0.0, reach, window)
        lo_c, hi_c = gap[0] + reach, gap[1] - reach
        if not (bulk_metal or (bulk_metal is None and flows[0.0]["bulk_crossings"])) and hi_c > lo_c:
            for i in range(1, CROSSING_GAP_SAMPLES + 1):
                energy = lo_c + (hi_c - lo_c) * i / (CROSSING_GAP_SAMPLES + 1)
                if energy not in flows:
                    flows[energy] = flow_at(levels, first, inner, steps, energy, reach, window)
    at_fermi = flows.get(0.0)
    spread = at_fermi["bulk_crossings"] if at_fermi is not None else 0
    metal = bool(bulk_metal) if bulk_metal is not None else spread > 0
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
    if not periodic:
        status = "open_path"
        reason = "an open path, not a closed loop: the counts are informational"
    elif metal:
        status = "metal"
        if bulk_metal:
            reason = "the bulk bands cross E_F: no gap around E_F"
        else:
            reason = f"bulk-like levels cross E_F ({spread} crossings): no gap around E_F"
    elif not interior:
        status = "unresolved"
        reason = "no cell lies between the two edge regions: use more slab_cells"
    elif not any(e.size for e in levels):
        status = "unresolved"
        reason = "no level lies within the window at any k: nothing to count, and a count of 0 would rest on absence"
    elif shift is None:
        status = "unresolved"
        reason = "no level is followed from one k-point to the next: use more slab_kpoints or a wider slab_window_eV"
    else:
        reach_mev = f"{reach * 1e3:.1f} meV"
        problems: list[str] = []
        if spread:
            problems.append(
                f"{spread} levels with a quarter or more of their weight in the middle half cross E_F over a "
                "bulk gap: edge states spread over the ribbon: use more slab_cells"
            )
        if not lo_c < 0.0 < hi_c:
            problems.append(
                f"a bulk-like level or the window edge lies within a k-step's reach ({reach_mev}) of E_F, in the "
                f"gap {span_mev}: use more slab_kpoints, or more slab_cells if edge states bound the gap"
            )
        if not any(
            np.any((e > gap[0]) & (e < gap[1]) & (m < CROSSING_BULK_INTERIOR)) for e, m in zip(levels, inner)
        ):
            problems.append(
                f"no level lies in the gap {span_mev} at any k: a count of 0 would rest on absence (a trivial "
                "gap and edge states hybridised into the bulk look the same): use more slab_cells; the bulk "
                "Chern number decides"
            )
        strays = [abs(x) for flow in flows.values() for x in flow["strays"]]
        if strays:
            problems.append(
                f"levels appear or disappear between neighbouring k-points {(span - min(strays)) * 1e3:.1f} meV "
                f"inside the window, beyond a k-step's reach ({reach_mev}): use more slab_kpoints or a wider "
                "slab_window_eV"
            )
        touched = [e for e, flow in flows.items() if flow["touches"]]
        if touched:
            problems.append(f"a bulk-like level takes part in a crossing at {_mev(min(touched))} meV")
        open_loops = [e for e, flow in flows.items() if flow["net"]]
        if open_loops:
            problems.append(
                f"the crossings at {_mev(open_loops[0])} meV do not close around the zone: use more slab_kpoints"
            )
        lefts = [row["left_flow"] for row in rows]
        far = [x for x in lefts if abs(x - round(x)) >= CROSSING_TOLERANCE]
        if far:
            problems.append(
                f"the left flow is {min(far):+.3f} to {max(far):+.3f} at {len(far)} of {len(rows)} energies, "
                f"not within {CROSSING_TOLERANCE} of an integer: the edges hybridise or their modes are not "
                "separated: use more slab_cells"
            )
        rounded = sorted({int(round(x)) for x in lefts})
        if len(rounded) > 1:
            problems.append(
                "the rounded left flow varies with energy across the gap ("
                + "/".join(f"{v:+d}" if v else "0" for v in rounded)
                + ")"
            )
        if problems:
            status = "unresolved"
            reason = "; ".join(problems)
        else:
            status = "resolved"
            worst = max(abs(x - round(x)) for x in lefts)
            reason = (
                f"left {rounded[0]:+d}, right {-rounded[0]:+d} at {len(rows)} energies across the gap "
                f"{span_mev}, each within {worst:.3f} of the integer"
            )
    per_step = at_fermi["per_step"] if at_fermi is not None else []
    net_left = int(round(at_fermi["left"])) if at_fermi is not None else 0
    net_right = int(round(at_fermi["net"] - at_fermi["left"])) if at_fermi is not None else 0
    return {
        "left": {**_crossing_tally(per_step, "left"), "net": net_left},
        "right": {**_crossing_tally(per_step, "right"), "net": net_right},
        "status": status,
        "reason": reason,
        "method": CROSSING_METHOD,
        "window_eV": None if metal or shift is None else gap,
        "step_shift_eV": shift,
        "energies_eV": [row["energy_eV"] for row in rows],
        "per_energy": rows,
        "tolerance": CROSSING_TOLERANCE,
    }


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
