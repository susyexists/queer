"""The QUEER_SLAB step: ribbon and slab bands with edge weights, with the PythTB driver's semantics.

:func:`run_slab` follows the Generator driver's ``run_slab`` on the folded
hopping table of :class:`~queer.compat.generator.wannier_steps.WannierStepContext`
(degeneracy and image weights applied, before the model's averaging and
cutoffs):

- the finite axis is ``slab_axis`` when given, else the second periodic axis of
  a 2D model (a ribbon) or a3 of a 3D one (a slab); it must be periodic
  (``queer_bad_params``). ``slab_cells`` defaults to 20 cells for a ribbon and
  12 for a slab, 12 and 6 above 60 orbitals;
- a ribbon samples its 1D zone at k = -1/2 + (i + 1/2)/n on its periodic axis,
  ``slab_kpoints`` rounded up to an even n, half a step off the high-symmetry
  points where finite-size hybridization mixes the edges; a slab follows
  :func:`~queer.compat.generator.estimators.surface_path` through every
  surface TRIM with ``slab_kpoints`` points;
- at each k the open slab H[w, w + d] = B_d(k), with E_F taken off the
  home-cell block, is solved for its levels within ``slab_window_eV`` of E_F by
  :func:`~queer.observables.surface.slab_eigensystem`. A level's weight on the
  first and on the last ``edge_cells`` cells is the driver's per-state value,
  after each degenerate cluster is rotated to diagonalize the first-edge
  projector (:func:`~queer.compat.generator.estimators.edge_weights`);
- ``edge_crossings_at_fermi`` counts edge-localized levels (weight >= 1/2)
  crossing E_F between neighbouring k-points, edge by edge, the last ribbon
  point paired with the first
  (:func:`~queer.compat.generator.estimators.edge_crossings`). The counts are
  informational; the bulk Chern number is the invariant.

The step writes ``Out/ribbon_bands.json`` or ``Out/slab_bands.json``
(``k_distance``, fractional for a ribbon and in 1/Å with 2 pi for a slab;
``kpoints_frac``; per k-point ``energies_eV`` relative to E_F, of varying length,
with their ``left_weight`` and ``right_weight``; the path ticks) and its figure.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...observables.surface import slab_eigensystem
from .estimators import edge_crossings, edge_weights, k_distance, surface_path
from .memory import pmap
from .plots import new_figure, save_figure, tick_marks
from .results import ERR_BAD_PARAMS, OUT_DIR, WARN_PLOT_SKIPPED, RunnerError, write_json_atomic
from .wannier_steps import WannierStepContext

CROSSINGS_NOTE = "Crossing counts are informational: the bulk Chern number is the invariant."
# Orbitals above which the default slab is thinner.
LARGE_MODEL_ORBITALS = 60
# Matrices per k-point of one slab solve the driver plans a worker's memory with.
SLAB_PER_K_MATRICES = 4
# Combined edge weight from which a level is drawn as an edge state.
EDGE_PLOT_WEIGHT = 0.3


@dataclass(frozen=True)
class FoldedTable:
    """A hopping table as a slab source: H(R) (eV) on integer R, the Fermi level and the orbital count."""

    translations: np.ndarray
    hopping_matrices: np.ndarray
    fermi_energy: float
    nbnd: int


def default_cells(two_d: bool, n_orb: int) -> int:
    """Cells of the driver's default ribbon (20, 12 above 60 orbitals) or slab (12, 6 above 60)."""
    if two_d:
        return 20 if n_orb <= LARGE_MODEL_ORBITALS else 12
    return 12 if n_orb <= LARGE_MODEL_ORBITALS else 6


def ribbon_path(axis: int, n_points: int) -> dict[str, Any]:
    """The ribbon's 1D zone along ``axis``: an even count of points k = -1/2 + (i + 1/2)/n, no ticks."""
    count = n_points + n_points % 2
    kpoints = np.zeros((count, 3))
    kpoints[:, axis] = -0.5 + (np.arange(count) + 0.5) / count
    return {"kpoints": kpoints, "tick_indices": [], "tick_labels": [], "break_indices": []}


def slab_levels(
    table: FoldedTable,
    finite: int,
    periodic: Sequence[int],
    cells: int,
    window: float,
    edge_cells: int,
    k_par: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Levels within ``window`` of E_F at one in-plane k and the driver's per-state edge weights."""
    energies, vectors = slab_eigensystem(
        table, k_par, finite_axis=finite, cells=cells, window=window, periodic_axes=periodic
    )
    left, right = edge_weights(vectors, cells, table.nbnd, edge_cells, energies=energies)
    return energies, left, right


def run_slab(context: WannierStepContext) -> None:
    """Open ribbon (2D) or slab (3D) bands near E_F weighted by edge localization."""
    params = context.params
    record = context.record
    two_d = len(context.periodic) == 2
    axis = params.get("slab_axis")
    if axis is None:
        finite = context.periodic[1] if two_d else 2
    else:
        finite = int(axis)
    if finite not in context.periodic:
        raise RunnerError(ERR_BAD_PARAMS, f"slab_axis {finite} is not a periodic direction of the model")
    periodic = [ax for ax in context.periodic if ax != finite]
    cells = int(params.get("slab_cells") or default_cells(two_d, context.n_orb))
    n_points = int(params.driver_value("slab_kpoints", 200))
    if two_d:
        kind = "ribbon"
        path = ribbon_path(periodic[0], n_points)
    else:
        kind = "slab"
        path = surface_path(context.lattice, periodic[0], periodic[1], n_points)
    window = float(params.driver_value("slab_window_eV", 0.5))
    edge_cells = int(params.driver_value("edge_cells", 2))
    table = FoldedTable(context.table_R, context.table_H, context.fermi, context.n_orb)
    states = cells * context.n_orb
    kpoints = np.asarray(path["kpoints"], dtype=float)
    # One k-point per task: each solves its own slab Hamiltonian.
    options = context.parallel(
        f"{kind}, {cells} cells along a{finite + 1} ({states} states)",
        SLAB_PER_K_MATRICES,
        len(kpoints),
        chunk=1,
        n_orb=states,
    )
    rows = pmap(
        functools.partial(slab_levels, table, finite, periodic, cells, window, edge_cells),
        list(kpoints),
        options,
    )
    energies = [row[0] for row in rows]
    left = [row[1] for row in rows]
    right = [row[2] for row in rows]
    crossings = edge_crossings(energies, left, right, periodic=kind == "ribbon")
    if kind == "ribbon":
        # A ribbon's x axis is its 1D crystal momentum, in units of 2 pi / a.
        distance = kpoints[:, periodic[0]].copy()
    else:
        distance = k_distance(kpoints, context.lattice, path["break_indices"])
    record.results["slab"] = {
        "kind": kind,
        "finite_axis": finite,
        "periodic_axes": periodic,
        "cells": cells,
        "states": states,
        "window_eV": window,
        "edge_cells": edge_cells,
        "edge_crossings_at_fermi": crossings,
        "note": CROSSINGS_NOTE,
    }
    context.log(
        f"{kind}: {cells} cells along a{finite + 1}, net edge crossings "
        f"left {crossings['left']['net']:+d}, right {crossings['right']['net']:+d}"
    )
    data_file = f"{OUT_DIR}/{kind}_bands.json"
    write_json_atomic(
        context.workspace / data_file,
        {
            "k_distance": distance,
            "kpoints_frac": kpoints,
            "energies_eV": energies,
            "left_weight": left,
            "right_weight": right,
            "tick_indices": path["tick_indices"],
            "tick_labels": path["tick_labels"],
        },
    )
    record.add_output(data_file, f"{kind.capitalize()} bands (data)", "data")
    plot_slab(context, kind, path, distance, energies, left, right, cells, window)


def plot_slab(
    context: WannierStepContext,
    kind: str,
    path: Mapping[str, Any],
    distance,
    energies: Sequence[np.ndarray],
    left: Sequence[np.ndarray],
    right: Sequence[np.ndarray],
    cells: int,
    window: float,
) -> None:
    """``{kind}_bands.png``: bulk levels in grey, edge levels coloured by the edge that holds them."""
    figure = new_figure((7, 5))
    if figure is None:
        return
    try:
        from matplotlib.colors import LinearSegmentedColormap

        axes = figure.subplots()
        xs = (
            np.concatenate([np.full(len(level), x) for x, level in zip(distance, energies)])
            if energies
            else np.zeros(0)
        )
        ys = np.concatenate(energies) if energies else np.zeros(0)
        first = np.concatenate(left) if left else np.zeros(0)
        last = np.concatenate(right) if right else np.zeros(0)
        on_edge = (first + last) >= EDGE_PLOT_WEIGHT
        axes.scatter(xs[~on_edge], ys[~on_edge], s=3, color="#b0b0b0", linewidths=0)
        # Purple marks a level on both edges, e.g. the hybridized surfaces of a thin slab.
        colours = LinearSegmentedColormap.from_list("edges", ["#0072b2", "#7b3294", "#d55e00"])
        total = first[on_edge] + last[on_edge]
        share = last[on_edge] / np.maximum(total, 1e-12)
        points = axes.scatter(
            xs[on_edge],
            ys[on_edge],
            c=share,
            s=4 + 16 * np.clip(total, 0, 1),
            cmap=colours,
            vmin=0,
            vmax=1,
            linewidths=0,
        )
        colorbar = figure.colorbar(points, ax=axes)
        colorbar.set_label("edge share: first (0) → last (1)")
        if kind == "ribbon":
            positions, labels = [-0.5, 0.0, 0.5], ["-X", "Γ", "X"]
            axes.set_xlim(-0.5, 0.5)
        else:
            positions, labels = tick_marks(path, distance)
            axes.set_xlim(distance[0], distance[-1])
        axes.set_xticks(positions)
        axes.set_xticklabels(labels)
        for x in positions:
            axes.axvline(x, color="#dddddd", lw=0.5, zorder=0)
        axes.axhline(0.0, color="#666666", lw=0.6, ls="--")
        axes.set_ylim(-window, window)
        axes.set_ylabel("E - E_F (eV)")
        axes.set_title(f"{kind.capitalize()} bands ({cells} cells)")
        save_figure(
            context.record, figure, f"{kind}_bands.png", f"{kind.capitalize()} bands with edge weight"
        )
    except Exception as error:  # a plot never fails a run
        context.record.warn(WARN_PLOT_SKIPPED, f"{kind} plot skipped: {error}")
