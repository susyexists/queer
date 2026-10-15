"""The QUEER_TOPOLOGY step: Wilson-loop Chern numbers and Z2 with the PythTB driver's semantics.

:func:`run_topology` follows the Generator driver's ``run_topology`` on the
model of :class:`~queer.compat.generator.wannier_steps.WannierStepContext`:

- the occupied set comes from :meth:`~WannierStepContext.resolve_occupation`;
  without a gapped set at E_F, or when its smallest direct gap to the next band
  is below ``isolation_threshold_meV``, the step reports ``applicable = false``
  with the driver's reason and computes nothing else;
- time reversal: the levels at the time-reversal-invariant momenta must pair
  within ``kramers_tol_meV``. ``z2 = auto`` computes Z2 for a spinor model
  whose levels pair, ``on`` always (with a warning when that check fails),
  ``off`` never;
- planes: ``planes`` when given, else the layer of a 2D model or k_c = 0 and 1/2
  for c = 3, 1, 2 of a 3D one. Each plane is scanned by
  :func:`~queer.observables.topology.wilson_plane` with ``loop_points`` per
  loop and ``transverse_points`` loops rounded up to an even count, doubled at
  most ``max_refinements`` times until the winding is resolved and the Z2 flow
  unambiguous. Its links are exact diagonal-embedding overlaps that close on
  the first point's states at k + G, which equals the driver's closing phase;
- a 3D model with Z2 reports the strong and weak indices; a 2D model reports
  ``nu``, the first plane's Z2 as computed even when its flow is ambiguous.

Each plane writes ``Out/wcc_k{c}_{v}.json`` (``t``, ``berry_phase``, ``wcc``,
``gap_midpoints``) and its WCC-flow figure. A Wilson loop QUEER refuses (a
degenerate occupied boundary, a singular link) fails the step with
``queer_numerical_error``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ...observables.topology import topology_planes, trim_points, wilson_plane
from ...physics.topology import kramers_degenerate, plane_axes, z2_indices
from .plots import new_figure, save_figure
from .results import (
    ERR_NUMERICAL,
    OUT_DIR,
    WARN_PLOT_SKIPPED,
    WARN_Z2_FORCED,
    RunnerError,
    write_json_atomic,
)
from .wannier_steps import WannierStepContext

NOT_GAPPED_REASON = (
    "No gapped occupied manifold at E_F (the model is metallic); set "
    "occupied_bands to the band count of an isolated group."
)
Z2_FORCED_WARNING = "Z2 forced on without Kramers degeneracy: it is not a well-defined invariant here."


def not_isolated_reason(occupied: int, isolation_meV: float) -> str:
    """Why the invariants of ``occupied`` bands that touch the next band are not computed."""
    return (
        f"Bands {occupied} and {occupied + 1} touch somewhere in the Brillouin zone "
        f"(smallest direct gap {isolation_meV:.2f} meV): "
        "invariants of a non-isolated band set are undefined."
    )


def step_planes(requested: Any, periodic: Sequence[int]) -> list[tuple[int, float]]:
    """The planes (fixed axis, value) to scan: ``planes`` when given, else :func:`topology_planes`."""
    if requested:
        planes = []
        for entry in requested:
            if isinstance(entry, Mapping):
                planes.append((int(entry.get("fixed_axis", 2)), float(entry.get("value", 0.0))))
            else:
                planes.append((int(entry[0]), float(entry[1])))
        return planes
    return topology_planes(periodic)


def wcc_name(fixed: int, value: float) -> str:
    """``wcc_k{c}_{v}`` of a plane, with ``p`` for the decimal point (``wcc_k3_0p5``)."""
    return f"wcc_k{fixed + 1}_{value:g}".replace(".", "p")


def run_topology(context: WannierStepContext) -> None:
    """Wilson loops per plane: Chern numbers and, with time reversal, Z2."""
    params = context.params
    record = context.record
    occupation = context.resolve_occupation()
    topology: dict[str, Any] = {"occupation": occupation, "planes": []}
    record.results["topology"] = topology
    occupied = occupation.get("n_occ")
    if not occupied:
        topology["applicable"] = False
        topology["reason"] = NOT_GAPPED_REASON
        return
    if not occupation.get("isolated"):
        topology["applicable"] = False
        topology["reason"] = not_isolated_reason(occupied, occupation.get("isolation_meV", 0))
        return
    topology["applicable"] = True

    trim_energies = context.model.energies(trim_points(context.periodic)).T
    kramers, split = kramers_degenerate(
        trim_energies, float(params.driver_value("kramers_tol_meV", 1.0)) / 1000.0
    )
    mode = str(params.driver_value("z2", "auto")).lower()
    want_z2 = mode == "on" or (mode == "auto" and context.spinors and kramers)
    topology["time_reversal"] = {
        "spinors": context.spinors,
        "kramers_degenerate_at_trims": kramers,
        "max_kramers_split_meV": split * 1000.0 if math.isfinite(split) else None,
        "z2_computed": want_z2,
    }
    if mode == "on" and not (context.spinors and kramers):
        record.warn(WARN_Z2_FORCED, Z2_FORCED_WARNING)

    loop_points = int(params.driver_value("loop_points", 81))
    transverse = int(params.driver_value("transverse_points", 61))
    refinements = int(params.driver_value("max_refinements", 0))
    tolerance = float(params.driver_value("z2_pos_tol", 0.01))
    z2_by_plane: dict[tuple[int, float], int | None] = {}
    for fixed, value in step_planes(params.get("planes"), context.periodic):
        a, b = plane_axes(fixed)
        # A loop is one task: its points are never split across workers.
        options = context.parallel(
            f"Wilson loops, k{fixed + 1}={value:g}",
            3,
            (transverse + transverse % 2) * loop_points,
            chunk=loop_points,
        )
        try:
            flow = wilson_plane(
                context.model,
                fixed,
                value,
                occupied_bands=occupied,
                loop_points=loop_points,
                transverse_points=transverse,
                max_refinements=refinements,
                z2=want_z2,
                z2_pos_tol=tolerance,
                execution=options,
            )
        except ValueError as error:
            raise RunnerError(
                ERR_NUMERICAL, f"Wilson loops on the plane k{fixed + 1} = {value:g} failed: {error}"
            ) from error
        chern_raw = float(flow["chern_raw"])
        plane: dict[str, Any] = {
            "fixed_axis": fixed,
            "value": value,
            "loop_axis": a,
            "transverse_axis": b,
            "chern": int(flow["chern"]),
            "chern_raw": chern_raw,
            "max_phase_step": float(flow["max_phase_step"]),
            "converged": bool(flow["converged"]),
            "transverse_points": int(flow.metadata["transverse_points"]),
            "loop_points": loop_points,
        }
        midpoints = None
        if "z2" in flow.arrays:
            midpoints = flow["gap_midpoints"]
            ambiguous = bool(flow["z2_ambiguous"])
            plane["z2"] = int(flow["z2"])
            plane["z2_ambiguous"] = ambiguous
            plane["z2_crossings"] = int(flow["z2_crossings"])
            z2_by_plane[(fixed, value)] = None if ambiguous else plane["z2"]
        name = wcc_name(fixed, value)
        data_file = f"{OUT_DIR}/{name}.json"
        transverse_values = flow["transverse"]
        write_json_atomic(
            context.workspace / data_file,
            {
                "t": transverse_values,
                "berry_phase": flow["berry_phase"],
                "wcc": flow["wcc"],
                "gap_midpoints": midpoints,
            },
        )
        plane["file"] = data_file
        record.add_output(data_file, f"Wilson loop, k{fixed + 1} = {value:g} (data)", "data")
        plot_wcc(context, name, fixed, value, transverse_values, flow["wcc"], midpoints)
        topology["planes"].append(plane)
        context.log(
            f"plane k{fixed + 1}={value}: C={chern_raw:+.4f}"
            + (f", Z2={plane['z2']}" if midpoints is not None else "")
        )

    if want_z2 and len(context.periodic) == 3:
        topology["z2_indices"] = z2_indices(z2_by_plane)
    elif want_z2 and topology["planes"]:
        topology["z2_indices"] = {"nu": topology["planes"][0].get("z2")}


def plot_wcc(
    context: WannierStepContext, name: str, fixed: int, value: float, transverse, wcc, midpoints
) -> None:
    """``wcc_k{c}_{v}.png``: the Wannier charge centres along the transverse axis, with the Z2 midpoints."""
    figure = new_figure((6, 4.5))
    if figure is None:
        return
    a, b = plane_axes(fixed)
    try:
        axes = figure.subplots()
        centres = np.asarray(wcc)
        axes.scatter(
            np.repeat(np.asarray(transverse), centres.shape[1]), centres.ravel(), s=3, color="#0072b2"
        )
        if midpoints is not None:
            axes.plot(
                np.asarray(transverse)[: len(midpoints)],
                midpoints,
                "D",
                ms=3,
                color="#d55e00",
                label="largest-gap midpoint",
            )
            axes.legend(loc="upper right", fontsize=8)
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)
        axes.set_xlabel(f"k{b + 1} (loop along k{a + 1})")
        axes.set_ylabel("Wannier charge centre")
        axes.set_title(f"WCC flow, k{fixed + 1} = {value:g}")
        save_figure(
            context.record, figure, f"{name}.png", f"Wannier charge centres, k{fixed + 1} = {value:g}"
        )
    except Exception as error:  # a plot never fails a run
        context.record.warn(WARN_PLOT_SKIPPED, f"WCC plot skipped: {error}")
