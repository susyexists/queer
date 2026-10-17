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


# Edge crossings: the rank offsets tried per k-step, the ranks around the reference
# energy they compare, and the energies sampled across a ribbon's gap besides E_F (one
# tenth of the gap apart and from its edges).
CROSSING_MAX_OFFSET = 3
CROSSING_RANKS = (-3, 2)
CROSSING_GAP_SAMPLES = 9
CROSSING_METHOD = "rank_continuity"


def _edge_of(wl: float, wr: float, threshold: float) -> str:
    """The edge holding at least ``threshold`` of a state or crossing, else ``"bulk"``."""
    if wl >= threshold:
        return "left"
    if wr >= threshold:
        return "right"
    return "bulk"


def _step_crossings(
    a: np.ndarray,
    b: np.ndarray,
    wa: tuple[np.ndarray, np.ndarray],
    wb: tuple[np.ndarray, np.ndarray],
    e_ref: float,
    threshold: float,
) -> list[tuple[str, int, bool]]:
    """Levels crossing ``e_ref`` from k_j (ascending ``a``) to k_(j+1) (``b``).

    Returns ``(edge, +1 up or -1 down, bulk_like)`` per crossing; ``wa`` and ``wb``
    are the (left, right) weights of the levels.
    """
    i0 = int(np.searchsorted(a, e_ref))
    j0 = int(np.searchsorted(b, e_ref))

    def present(r: int, o: int) -> bool:
        return 0 <= i0 + r < len(a) and 0 <= j0 + r + o < len(b)

    best: tuple[tuple[float, int, int], int] | None = None
    for o in range(-CROSSING_MAX_OFFSET, CROSSING_MAX_OFFSET + 1):
        crossed = range(-o, 0) if o > 0 else range(0, -o)
        if not all(present(r, o) for r in crossed):
            continue
        ranks = [r for r in range(CROSSING_RANKS[0], CROSSING_RANKS[1] + 1) if present(r, o)]
        if not ranks:
            continue
        idx = np.array(ranks)
        cost = float(np.mean(np.abs(a[i0 + idx] - b[j0 + idx + o])))
        key = (cost, abs(o), o)
        if best is None or key < best[0]:
            best = (key, o)
    if best is None:
        return []
    o = best[1]
    # The crossed ranks and one neighbour on each side. Two neighbouring pairs whose
    # dominant edges are exchanged are two edges' levels crossing each other: they
    # swap partners.
    pairs = [(i0 + r, j0 + r + o) for r in range(min(-1, -1 - o), max(0, -o) + 1) if present(r, o)]
    q = 0
    while q + 1 < len(pairs):
        (i, i1), (n, n1) = pairs[q], pairs[q + 1]
        x0 = _edge_of(wa[0][i], wa[1][i], threshold)
        y1 = _edge_of(wb[0][i1], wb[1][i1], threshold)
        y0 = _edge_of(wa[0][n], wa[1][n], threshold)
        x1 = _edge_of(wb[0][n1], wb[1][n1], threshold)
        if x0 == x1 and y0 == y1 and x0 != y0 and "bulk" not in (x0, y0):
            pairs[q], pairs[q + 1] = (i, n1), (n, i1)
            q += 2
        else:
            q += 1
    out = []
    for i, i1 in pairs:
        if (a[i] < e_ref) == (b[i1] < e_ref):
            continue
        wl = 0.5 * (float(wa[0][i]) + float(wb[0][i1]))
        wr = 0.5 * (float(wa[1][i]) + float(wb[1][i1]))
        out.append((_edge_of(wl, wr, threshold), 1 if b[i1] >= e_ref else -1, wl + wr < threshold))
    return out


def _crossings_at(
    levels: Sequence[np.ndarray],
    weights: Sequence[tuple[np.ndarray, np.ndarray]],
    pairs: Sequence[tuple[int, int]],
    e_ref: float,
    threshold: float,
) -> tuple[dict[str, dict[str, int]], int]:
    """Per-edge crossings of ``e_ref`` over the k-steps ``pairs``, and the bulk-like ones."""
    tally = {side: {"up": 0, "down": 0} for side in ("left", "right", "bulk")}
    bulk_like = 0
    for j, j1 in pairs:
        for side, sign, bulky in _step_crossings(
            levels[j], levels[j1], weights[j], weights[j1], e_ref, threshold
        ):
            tally[side]["up" if sign > 0 else "down"] += 1
            bulk_like += int(bulky)
    for side in tally:
        tally[side]["net"] = tally[side]["up"] - tally[side]["down"]
    return tally, bulk_like


def _net_values(values: Sequence[int]) -> str:
    return "/".join(f"{v:+d}" if v else "0" for v in sorted(set(values)))


def edge_crossings(
    energies: Sequence,
    left: Sequence,
    right: Sequence,
    *,
    periodic: bool = False,
    threshold: float = 0.5,
    window: float | None = None,
    interior: bool = True,
) -> dict[str, Any]:
    """Edge states crossing E_F = 0, counted by rank continuity and checked across the gap.

    Per step k_j -> k_(j+1) (the last point back to the first when ``periodic``, a
    ribbon's 1D zone), the levels are ranked from a reference energy (rank 0 = the
    lowest at or above it) and the offset o in [-3, 3] that minimizes the mean
    |E_r(k_j) - E_(r+o)(k_(j+1))| over the ranks r in [-3, 2] present at both is
    taken; the |o| levels an offset says crossed must be present at both. o > 0:
    ranks -1..-o crossed upwards; o < 0: ranks 0..|o|-1 downwards. Two neighbouring
    pairs whose dominant edges are exchanged (two edges' levels crossing each other)
    swap partners. A crossing's left and right weights are its endpoints' means; it
    belongs to the edge holding at least ``threshold`` of it, else to the bulk, and
    is bulk-like when left + right < ``threshold``.

    ``left`` and ``right`` are the counts at E_F (up, down, net). Levels with left +
    right < ``threshold`` bound the gap around E_F: ``window_eV`` is the highest such
    level below E_F and the lowest at or above it over all k (-``window`` or
    ``window`` on a side without one; without a ``window`` the largest stored |E|).
    A bulk-like crossing at E_F leaves no gap: ``metal``. Otherwise the nets are
    evaluated at E_F and at 9 energies across the gap, a tenth of it apart and from
    its edges (``per_energy``): ``resolved`` when every energy gives the same nets
    with left = -right and no bulk net, else ``unresolved`` (the edges hybridize;
    more cells separate them). A ribbon without cells between its two edge regions
    (``interior`` false) is ``unresolved``, and an open path (a slab's surface path)
    is ``open_path``: informational. The per-state weights are basis-dependent
    inside degenerate clusters, so the counts are informational; the bulk invariant
    is the authority.
    """
    levels, weights = [], []
    for e, wl, wr in zip(energies, left, right):
        e = np.asarray(e, float)
        order = np.argsort(e, kind="stable")
        levels.append(e[order])
        weights.append((np.asarray(wl, float)[order], np.asarray(wr, float)[order]))
    count = len(levels)
    pairs = [(j, j + 1) for j in range(count - 1)]
    if periodic and count > 1:
        pairs.append((count - 1, 0))
    at_fermi, bulk_like = _crossings_at(levels, weights, pairs, 0.0, threshold)
    rows: list[dict[str, Any]] = []

    def row(energy: float, tally: dict[str, dict[str, int]]) -> dict[str, Any]:
        return {
            "energy_eV": energy,
            "left_net": tally["left"]["net"],
            "right_net": tally["right"]["net"],
            "bulk_net": tally["bulk"]["net"],
        }

    gap: list[float] | None = None
    if bulk_like:
        rows.append(row(0.0, at_fermi))
    else:
        lo: float | None = None
        hi: float | None = None
        for e, (wl, wr) in zip(levels, weights):
            bulk = wl + wr < threshold
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
        step = gap[1] - gap[0]
        samples = {gap[0] + step * i / (CROSSING_GAP_SAMPLES + 1) for i in range(1, CROSSING_GAP_SAMPLES + 1)}
        for energy in sorted(samples | {0.0}):
            tally = at_fermi if energy == 0.0 else _crossings_at(levels, weights, pairs, energy, threshold)[0]
            rows.append(row(energy, tally))
    nets = {side: [r[f"{side}_net"] for r in rows] for side in ("left", "right", "bulk")}
    if not periodic:
        status = "open_path"
        reason = "an open path, not a closed loop: the counts are informational"
    elif gap is None:
        status = "metal"
        reason = f"bulk-like levels cross E_F ({bulk_like} crossings): no gap around E_F"
    elif not interior:
        status = "unresolved"
        reason = "no cell lies between the two edge regions: use more slab_cells"
    else:
        span_meV = f"[{gap[0] * 1e3:+.1f}, {gap[1] * 1e3:+.1f}] meV"
        stable = all(len(set(values)) == 1 for values in nets.values())
        if stable and nets["left"][0] == -nets["right"][0] and nets["bulk"][0] == 0:
            status = "resolved"
            reason = f"the same counts at {len(rows)} energies across the gap {span_meV}"
        else:
            status = "unresolved"
            sides = [s for s in ("left", "right") if len(set(nets[s])) > 1]
            sides = sides or ["left", "right"]
            who = " and ".join(sides) + (" edge" if len(sides) == 1 else " edges")
            if any(nets["bulk"]):
                who += " and the bulk"
            reason = (
                f"counts at {len(rows)} energies across the gap {span_meV}: left net "
                f"{_net_values(nets['left'])}, right net {_net_values(nets['right'])}, "
                f"bulk net {_net_values(nets['bulk'])}; states of the {who} hybridise: "
                "use more slab_cells"
            )
    return {
        "left": at_fermi["left"],
        "right": at_fermi["right"],
        "status": status,
        "reason": reason,
        "method": CROSSING_METHOD,
        "window_eV": gap,
        "energies_eV": [r["energy_eV"] for r in rows],
        "per_energy": rows,
        "threshold": threshold,
    }


def legacy_edge_crossings(
    energies: Sequence,
    left: Sequence,
    right: Sequence,
    *,
    periodic: bool = False,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """The driver's count up to core 56fda912: nearest-energy matching on each edge.

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
