"""The QUEER_BANDS step: bands, DOS/PDOS, gap and Fermi surface with the PythTB driver's semantics.

:func:`run_bands` follows the Generator driver's ``run_bands`` on the model of
:class:`~queer.compat.generator.wannier_steps.WannierStepContext`, solving with
QUEER instead of PythTB:

- the band path (:meth:`~WannierStepContext.band_path`) with each state's weight
  on every species, the sum of ``|c|^2`` over that species' orbitals, which no
  Bloch phase convention changes. ``bands.eigenvalues_eV`` holds absolute
  energies, E - E_F + E_F as the driver stores them;
- the DOS on a uniform mesh of i/n points along the periodic axes (0 along the
  vacuum axis): ``dos_mesh`` divisions per axis, or the driver's size rule
  (:func:`dos_divisions`). Eigenvalues are histogrammed on bins of
  ``dos_step_eV`` from 2 eV below to 2 eV above the band path's energies,
  averaged over the mesh and smoothed by the truncated discrete Gaussian of
  ``dos_sigma_eV``; with ``pdos`` and more than one species, the
  species-weighted histograms give the PDOS. The DOS counts one state per band
  and k-point: no spin factor;
- the gap classification on the mesh plus the path, with DOS(E_F) interpolated
  linearly between the bin centres around E_F;
- for a metal with ``fermi_surface`` on, the E_F contours of every crossing band
  on the default plane (the layer of a 2D model, k_3 = 0 of a 3D one), sampled
  on an n x n mesh over [-1/2, 1/2) and drawn in the plane frame of the band
  path's k unit (1/Å without 2 pi). The contours come from Matplotlib; without
  it the contour set is empty.

The step writes ``Out/dos.json``, ``Out/fermi_surface.json`` and the figures
``bands.png``, ``dos.png`` and ``fermi_surface.png`` with the driver's titles.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ...execution import ExecutionOptions
from ...observables.topology import plane_kpoints
from ...physics.topology import plane_axes
from .estimators import (
    BandEdges,
    classify_gap,
    dos_at_fermi,
    dos_energy_grid,
    eigenvalue_histogram,
    gaussian_dos,
    k_distance,
    plane_frame,
    uniform_mesh,
)
from .memory import chunks, pmap
from .plots import LINE_PALETTE, SPECIES_PALETTE, energy_window, new_figure, save_figure, tick_marks
from .results import (
    ERR_BAD_PARAMS,
    OUT_DIR,
    WARN_DOS_OUTSIDE_GRID,
    WARN_PLOT_SKIPPED,
    RunnerError,
    write_json_atomic,
)
from .wannier_steps import WannierStepContext

DOS_FILE = f"{OUT_DIR}/dos.json"
FERMI_SURFACE_FILE = f"{OUT_DIR}/fermi_surface.json"
FERMI_SURFACE_UNITS = "1/angstrom without 2 pi (k / 2 pi), as the band k_distance"
# Default Fermi-surface mesh per in-plane axis of a 2D and a 3D model.
FERMI_SURFACE_MESH = {2: 200, 3: 150}


def dos_divisions(n_orb: int, dimension: int, requested: Any = None) -> list[int]:
    """Divisions per axis of the DOS mesh: ``requested`` (``dos_mesh``) or the driver's size rule.

    Without a request a 2D model of N Wannier functions gets round(interp(N,
    [30, 150], [300, 150])) divisions per axis and a 3D model round(interp(N,
    [20, 150], [60, 24])), constant beyond those ranges. A number applies to
    every axis; only the periodic axes are sampled.
    """
    if not requested:
        if dimension == 2:
            count = int(round(np.interp(n_orb, [30, 150], [300, 150])))
        else:
            count = int(round(np.interp(n_orb, [20, 150], [60, 24])))
        return [count, count, count]
    if isinstance(requested, (int, float)):
        return [int(requested)] * 3
    return [int(value) for value in requested]


def default_plane(periodic: Sequence[int]) -> tuple[int, float]:
    """The plane of the Fermi surface: the layer of a 2D model, k_3 = 0 of a 3D one."""
    missing = [axis for axis in range(3) if axis not in periodic]
    return (missing[0] if missing else 2), 0.0


def species_weights(coefficients, species_index, n_species: int) -> np.ndarray:
    """(Nk, NB, n_species) weight of every state on every species: ``|c|^2`` summed over its orbitals.

    ``coefficients`` are (Nk, N_orb, NB) orthonormal eigenvector coefficients.
    """
    weights = np.abs(np.asarray(coefficients)) ** 2
    index = np.asarray(species_index)
    return np.stack([weights[:, index == s, :].sum(axis=1) for s in range(n_species)], axis=-1)


def _serial(count: int) -> ExecutionOptions:
    return ExecutionOptions(workers=1, batch_size=max(1, int(count)))


def _relative_states(model, kpoints) -> tuple[np.ndarray, np.ndarray]:
    """Energies relative to E_F (Nk, NB) and coefficients (Nk, N_orb, NB) at fractional ``kpoints``."""
    states = model.eigenstates(kpoints, execution=_serial(len(kpoints)))
    return states.energies.T - model.fermi_energy, states.coefficients


def _path_chunk(model, species_index, n_species: int, kpoints):
    energies, coefficients = _relative_states(model, kpoints)
    return energies, species_weights(coefficients, species_index, n_species)


def _dos_chunk(model, grid, species_index, n_species: int, projected: bool, kpoints):
    """Band edges, eigenvalue histogram, species histograms (or None) and out-of-grid count of a chunk."""
    if projected:
        energies, coefficients = _relative_states(model, kpoints)
        per_species = species_weights(coefficients, species_index, n_species)
        del coefficients
    else:
        energies = model.energies(kpoints, execution=_serial(len(kpoints))).T
        per_species = None
    edges = BandEdges(energies.shape[1])
    edges.add(kpoints, energies)
    total, outside = eigenvalue_histogram(energies, grid)
    parts = None
    if per_species is not None:
        parts = np.stack(
            [eigenvalue_histogram(energies, grid, per_species[..., s])[0] for s in range(n_species)]
        )
    return edges, total, parts, outside


def _energy_chunk(model, kpoints) -> np.ndarray:
    return model.energies(kpoints, execution=_serial(len(kpoints))).T


def run_bands(context: WannierStepContext) -> None:
    """Bands on the path weighted by species, DOS and PDOS, the gap and, for a metal, the Fermi surface."""
    params = context.params
    record = context.record
    model = context.model.light_worker_view()
    species = np.asarray(context.species_index, dtype=int)
    # Every species of the structure gets a PDOS row, an empty one when no
    # Wannier centre lies nearest to its atoms.
    n_species = max(len(context.species_names), int(species.max()) + 1 if len(species) else 1)

    path = context.band_path()
    kpoints = np.asarray(path["kpoints"], dtype=float)
    options = context.parallel("band path", 4, len(kpoints))
    parts = pmap(
        functools.partial(_path_chunk, model, species, n_species),
        chunks(kpoints, options.batch_size),
        options,
    )
    energies = np.concatenate([part[0] for part in parts])
    weights = np.concatenate([part[1] for part in parts])
    distance = k_distance(kpoints, context.lattice, path["break_indices"])

    divisions = dos_divisions(context.n_orb, len(context.periodic), params.get("dos_mesh"))
    mesh = uniform_mesh(divisions, context.periodic)
    step = float(params.driver_value("dos_step_eV", 0.005))
    sigma = float(params.driver_value("dos_sigma_eV", 0.03))
    grid = dos_energy_grid(energies, step)
    projected = bool(params.get("pdos")) and len(context.species_names) > 1
    options = context.parallel("DOS", 4 if projected else 2, len(mesh))
    worker = functools.partial(_dos_chunk, model, grid, species, n_species, projected)
    edges = BandEdges(context.n_orb)
    total = np.zeros(len(grid) - 1)
    species_hist = None
    outside = 0
    for chunk_edges, hist, parts_hist, out_of_range in pmap(
        worker, chunks(mesh, options.batch_size), options
    ):
        edges.merge(chunk_edges)
        total += hist
        outside += out_of_range
        if parts_hist is not None:
            species_hist = parts_hist if species_hist is None else species_hist + parts_hist
    path_edges = BandEdges(context.n_orb)
    path_edges.add(kpoints, energies)
    edges.merge(path_edges)
    norm = float(len(mesh))
    centres = 0.5 * (grid[1:] + grid[:-1])
    dos = gaussian_dos(total / norm, step, sigma)
    pdos = gaussian_dos(species_hist / norm, step, sigma) if species_hist is not None else None
    try:
        gap = classify_gap(
            edges,
            float(params.driver_value("fermi_tolerance_eV", 0.02)),
            params.get("occupied_bands"),
        )
    except ValueError as error:
        raise RunnerError(ERR_BAD_PARAMS, str(error)) from error
    gap["dos_at_fermi_per_eV"] = dos_at_fermi(centres, dos)
    if outside:
        record.warn(WARN_DOS_OUTSIDE_GRID, f"{outside} mesh eigenvalues fell outside the DOS grid")

    record.results["electronic"] = gap
    record.results["bands"] = {
        "kpoints_frac": kpoints,
        "eigenvalues_eV": energies + context.fermi,
        "fermi_energy": context.fermi,
        "tick_indices": list(path["tick_indices"]),
        "tick_labels": list(path["tick_labels"]),
        "break_indices": list(path["break_indices"]),
        "source": path["source"],
    }
    record.results["dos"] = {
        # The divisions sampled along each periodic axis (``mesh_axes``).
        "mesh": [int(len(np.unique(mesh[:, axis]))) for axis in context.periodic],
        "mesh_axes": list(context.periodic),
        "k_points": int(len(mesh)),
        "sigma_eV": sigma,
        "integral": float(dos.sum() * step),
        "species": list(context.species_names) if pdos is not None else [],
        "file": DOS_FILE,
    }
    write_json_atomic(
        context.workspace / DOS_FILE,
        {
            "energy_eV": centres,
            "dos_per_eV": dos,
            "pdos_per_eV": (
                {name: pdos[index] for index, name in enumerate(context.species_names)}
                if pdos is not None
                else {}
            ),
        },
    )
    record.add_output(DOS_FILE, "DOS and species PDOS (data)", "data")
    context.log(
        f"bands: {len(kpoints)} path k-points, DOS on {len(mesh)} k-points, "
        + ("metal" if gap.get("metal") else f"gap {gap.get('gap_eV', 0.0):.4f} eV")
    )

    plot_bands(context, path, distance, energies, weights)
    plot_dos(context, centres, dos, pdos)
    if gap.get("metal") and params.get("fermi_surface"):
        fermi_surface(context, model)


def fermi_surface(context: WannierStepContext, model=None) -> None:
    """E_F contours on the default plane, Gamma-centred, in the band path's Cartesian k unit."""
    model = context.model if model is None else model
    fixed, value = default_plane(context.periodic)
    dimension = 2 if len(context.periodic) == 2 else 3
    n = int(context.params.driver_value("fermi_surface_mesh", FERMI_SURFACE_MESH[dimension]))
    a, b = plane_axes(fixed)
    points = plane_kpoints(fixed, value, n, centred=True)
    options = context.parallel("Fermi surface", 2, len(points))
    blocks = pmap(functools.partial(_energy_chunk, model), chunks(points, options.batch_size), options)
    energies = np.concatenate(blocks).reshape(n, n, -1)
    crossing = [
        int(band)
        for band in range(energies.shape[-1])
        if energies[..., band].min() < 0.0 < energies[..., band].max()
    ]
    reciprocal, ea, eb = plane_frame(context.lattice, a, b)
    cartesian = points @ reciprocal
    x = (cartesian @ ea).reshape(n, n)
    y = (cartesian @ eb).reshape(n, n)
    contours: dict[str, list] = {}
    figure = new_figure((5.5, 5))
    if figure is not None:
        try:
            axes = figure.subplots()
            for index, band in enumerate(crossing):
                lines = axes.contour(
                    x,
                    y,
                    energies[..., band],
                    levels=[0.0],
                    colors=[LINE_PALETTE[index % len(LINE_PALETTE)]],
                    linewidths=1.2,
                )
                contours[str(band)] = [np.asarray(segment).tolist() for segment in lines.allsegs[0]]
            axes.set_aspect("equal")
            axes.set_xlabel(f"k / 2π along b{a + 1} (1/Å)")
            axes.set_ylabel(f"k / 2π normal to b{a + 1} (1/Å)")
            axes.set_title(f"Fermi surface, k{fixed + 1} = {value:g}")
            save_figure(context.record, figure, "fermi_surface.png", "Fermi surface contours")
        except Exception as error:  # a plot never fails a run
            context.record.warn(WARN_PLOT_SKIPPED, f"Fermi-surface plot skipped: {error}")
    start = float(points[:, a].min())
    write_json_atomic(
        context.workspace / FERMI_SURFACE_FILE,
        {
            "fixed_axis": fixed,
            "value": value,
            "mesh": n,
            # The fractional cell sampled on both in-plane axes, half-open: its
            # mesh points are start + i / mesh, so the end is not one of them.
            "k_frac_range": [start, start + 1.0],
            "k_frac_range_end": "excluded",
            "bands": crossing,
            # contours_cart holds (x, y) points along these two Cartesian unit
            # vectors, in these units.
            "axes_cart": [ea, eb],
            "units": FERMI_SURFACE_UNITS,
            "contours_cart": contours,
        },
    )
    context.record.add_output(FERMI_SURFACE_FILE, "Fermi-surface contours (data)", "data")
    context.record.results["fermi_surface"] = {
        "fixed_axis": fixed,
        "value": value,
        "bands_crossing": crossing,
        "file": FERMI_SURFACE_FILE,
    }


def plot_bands(context: WannierStepContext, path: Mapping[str, Any], distance, energies, weights) -> None:
    """``bands.png``: the model's bands over wannier90's, coloured by each state's dominant species."""
    figure = new_figure((7.5, 5))
    if figure is None:
        return
    try:
        low, high = energy_window(context.params)
        axes = figure.subplots()
        reference = context.reference
        if reference is not None and path["source"] == "wannier90_band_kpt":
            for band in range(reference["energies"].shape[1]):
                axes.plot(
                    distance, reference["energies"][:, band], color="#999999", lw=2.5, alpha=0.6, zorder=1
                )
            axes.plot([], [], color="#999999", lw=2.5, alpha=0.6, label="wannier90")
        dominant = weights.argmax(axis=-1)
        for band in range(energies.shape[1]):
            axes.plot(distance, energies[:, band], color="#222222", lw=0.6, zorder=2)
        if len(context.species_names) > 1:
            for index, name in enumerate(context.species_names):
                selected = dominant == index
                along = np.broadcast_to(distance[:, None], energies.shape)[selected]
                axes.scatter(
                    along,
                    energies[selected],
                    s=4,
                    color=SPECIES_PALETTE[index % len(SPECIES_PALETTE)],
                    label=name,
                    zorder=3,
                )
        axes.plot([], [], color="#222222", lw=0.6, label="QUEER")
        positions, labels = tick_marks(path, distance)
        axes.set_xticks(positions)
        axes.set_xticklabels(labels)
        for x in positions:
            axes.axvline(x, color="#cccccc", lw=0.5, zorder=0)
        axes.axhline(0.0, color="#666666", lw=0.6, ls="--")
        axes.set_xlim(distance[0], distance[-1])
        axes.set_ylim(low, high)
        axes.set_ylabel("E - E_F (eV)")
        axes.set_title("Wannier tight-binding bands (colour: dominant species)")
        axes.legend(loc="upper right", fontsize=8, markerscale=2)
        save_figure(context.record, figure, "bands.png", "Band structure")
    except Exception as error:  # a plot never fails a run
        context.record.warn(WARN_PLOT_SKIPPED, f"band plot skipped: {error}")


def plot_dos(context: WannierStepContext, centres, dos, pdos) -> None:
    """``dos.png``: the total DOS and the species PDOS inside the energy window."""
    figure = new_figure((6, 4.5))
    if figure is None:
        return
    try:
        low, high = energy_window(context.params)
        axes = figure.subplots()
        axes.plot(centres, dos, color="#222222", lw=1.2, label="total")
        if pdos is not None:
            for index, name in enumerate(context.species_names):
                axes.plot(
                    centres, pdos[index], lw=1.0, color=LINE_PALETTE[index % len(LINE_PALETTE)], label=name
                )
        axes.axvline(0.0, color="#666666", lw=0.6, ls="--")
        axes.set_xlim(low, high)
        inside = (centres >= low) & (centres <= high)
        top = float(dos[inside].max()) if inside.any() else float(dos.max())
        axes.set_ylim(0, top * 1.1 if top > 0 else 1)
        axes.set_xlabel("E - E_F (eV)")
        axes.set_ylabel("states / eV / cell")
        axes.set_title("Density of states")
        axes.legend(fontsize=8)
        save_figure(context.record, figure, "dos.png", "Density of states")
    except Exception as error:  # a plot never fails a run
        context.record.warn(WARN_PLOT_SKIPPED, f"DOS plot skipped: {error}")
