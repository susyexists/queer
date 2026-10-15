"""Wilson-loop topology of reciprocal planes: Chern numbers, WCC flows and Z2.

Loops run along k_a = i/n_loop (i = 0..n_loop-1, the endpoint excluded) on the
plane k_c = value, one loop per transverse point k_b = j/n_t, with (a, b, c)
cyclic. Each loop is :func:`queer.observables.geometry.wilson_loop`: links are
occupied overlaps through the model's basis overlap, which for Wannier centres
without position matrices is the exact diagonal embedding, and the loop closes
on the first point's eigenvectors at the lifted point k + G without any extra
closing phase. These links equal PythTB's cell-periodic (convention I) links.
"""

from __future__ import annotations

import numpy as np

from ..physics.topology import chern_from_phases, plane_axes, z2_from_wcc
from .geometry import wilson_loop
from .result import ObservableResult

# A plane's winding counts as resolved when every transverse Berry-phase step is
# below pi/2 and the summed winding lies this close to an integer.
CHERN_INTEGER_TOL = 0.05

__all__ = [
    "CHERN_INTEGER_TOL",
    "plane_axes",
    "plane_kpoints",
    "plane_loops",
    "topology_planes",
    "trim_points",
    "wilson_plane",
]


def _periodic(periodic_axes):
    axes = sorted({int(axis) for axis in periodic_axes})
    if any(axis not in (0, 1, 2) for axis in axes):
        raise ValueError("periodic axes must be reciprocal axis indices 0, 1 or 2")
    return axes


def topology_planes(periodic_axes) -> list[tuple[int, float]]:
    """Planes (fixed axis, value) whose invariants characterise the model.

    Two periodic axes give the one plane of the layer, with the vacuum axis
    fixed at 0. Three give k_c = 0 and 1/2 for c = 3, 1, 2 (zero-based 2, 0, 1),
    which hold the strong and weak Z2 indices.
    """
    axes = _periodic(periodic_axes)
    if len(axes) == 2:
        return [(next(axis for axis in range(3) if axis not in axes), 0.0)]
    if len(axes) == 3:
        return [(axis, value) for axis in (2, 0, 1) for value in (0.0, 0.5)]
    raise ValueError("topology planes need two or three periodic axes")


def trim_points(periodic_axes) -> np.ndarray:
    """Time-reversal-invariant momenta: 0 and 1/2 on periodic axes, 0 elsewhere."""
    axes = _periodic(periodic_axes)
    values = [[0.0, 0.5] if axis in axes else [0.0] for axis in range(3)]
    grid = np.meshgrid(*values, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=1)


def plane_kpoints(fixed_axis: int, value: float, mesh: int, *, centred: bool = False) -> np.ndarray:
    """``mesh x mesh`` fractional points on k_fixed = value, the loop axis outermost.

    The plane's axes span [0, 1), or [-1/2, 1/2) when ``centred``.
    """
    a, b = plane_axes(fixed_axis)
    if isinstance(mesh, bool) or int(mesh) != mesh or mesh < 1:
        raise ValueError("mesh must be a positive integer")
    count = int(mesh)
    axis = np.arange(count) / count - (0.5 if centred else 0.0)
    first, second = np.meshgrid(axis, axis, indexing="ij")
    points = np.zeros((count * count, 3))
    points[:, a] = first.ravel()
    points[:, b] = second.ravel()
    points[:, fixed_axis] = value
    return points


def plane_loops(fixed_axis: int, value: float, transverse, loop_points: int) -> np.ndarray:
    """Closed loops (n_t, loop_points, 3) along k_a at the given transverse k_b values.

    At least three points are needed: ``wilson_loop`` finds the closing lattice
    translation by rounding the loop's span (n - 1)/n, which must exceed 1/2.
    """
    a, b = plane_axes(fixed_axis)
    if isinstance(loop_points, bool) or int(loop_points) != loop_points or loop_points < 3:
        raise ValueError("loop_points must be an integer of at least three")
    t = np.asarray(transverse, float).reshape(-1)
    loops = np.zeros((len(t), int(loop_points), 3))
    loops[:, :, a] = np.arange(int(loop_points)) / int(loop_points)
    loops[:, :, b] = t[:, None]
    loops[:, :, fixed_axis] = value
    return loops


def _integer(value, name, minimum):
    if isinstance(value, bool) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(value)


def wilson_plane(
    model,
    fixed_axis: int,
    value: float = 0.0,
    *,
    occupied_bands: int,
    loop_points: int = 81,
    transverse_points: int = 61,
    max_refinements: int = 2,
    z2: bool = False,
    z2_pos_tol: float = 0.01,
    execution=None,
) -> ObservableResult:
    """WCC flow, Chern number and optional Z2 of the plane k_fixed = value.

    ``transverse_points`` is rounded up to an even count so t = 1/2 is sampled.
    The phase-winding Chern number is resolved when every Berry-phase step is
    below pi/2 and C lies within ``CHERN_INTEGER_TOL`` of an integer; otherwise
    the transverse sampling doubles, at most ``max_refinements`` times. Loops
    already evaluated are reused, since a doubled grid contains the previous one.
    With ``z2`` on a time-reversal-invariant plane (value 0 or 1/2), the
    largest-gap rule runs over t in [0, 1/2] and an ambiguous flow also refines.
    """
    a, b = plane_axes(fixed_axis)
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("plane value must be finite")
    count = _integer(occupied_bands, "occupied_bands", 1)
    if count > model.nbnd:
        raise ValueError("occupied_bands exceeds the model band count")
    loops_per_plane = _integer(transverse_points, "transverse_points", 1)
    refinements = _integer(max_refinements, "max_refinements", 0)
    tolerance = float(z2_pos_tol)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("z2_pos_tol must be finite and nonnegative")
    loop_count = _integer(loop_points, "loop_points", 3)
    trim_plane = abs(value) < 1e-9 or abs(value - 0.5) < 1e-9
    want_z2 = bool(z2) and trim_plane

    def evaluate(transverse):
        result = wilson_loop(
            model,
            plane_loops(fixed_axis, value, transverse, loop_count),
            occupied_bands=count,
            execution=execution,
        )
        return result["berry_phase"], result["phases"]

    n_t = loops_per_plane + loops_per_plane % 2
    phases, centres = evaluate(np.arange(n_t) / n_t)
    attempts = 1
    while True:
        chern, max_step = chern_from_phases(phases)
        flow = z2_from_wcc(centres[: n_t // 2 + 1], tolerance) if want_z2 else None
        converged = max_step < np.pi / 2 and abs(chern - round(chern)) < CHERN_INTEGER_TOL
        if (converged and not (flow is not None and flow.ambiguous)) or attempts > refinements:
            break
        n_t *= 2
        added_phases, added_centres = evaluate((np.arange(n_t) / n_t)[1::2])
        merged_phases = np.empty(n_t)
        merged_centres = np.empty((n_t, centres.shape[1]))
        merged_phases[0::2], merged_phases[1::2] = phases, added_phases
        merged_centres[0::2], merged_centres[1::2] = centres, added_centres
        phases, centres = merged_phases, merged_centres
        attempts += 1
    arrays: dict[str, np.ndarray] = {
        "transverse": np.arange(n_t) / n_t,
        "berry_phase": phases,
        "wcc": centres,
        "chern": np.asarray(int(round(chern))),
        "chern_raw": np.asarray(chern),
        "max_phase_step": np.asarray(max_step),
        "converged": np.asarray(bool(converged)),
    }
    dims: dict[str, tuple[str, ...]] = {
        "transverse": ("transverse",),
        "berry_phase": ("transverse",),
        "wcc": ("transverse", "occupied_band"),
        "chern": (),
        "chern_raw": (),
        "max_phase_step": (),
        "converged": (),
    }
    units = {
        "transverse": "fractional",
        "berry_phase": "radian",
        "wcc": "1",
        "chern": "1",
        "chern_raw": "1",
        "max_phase_step": "radian",
        "converged": "1",
    }
    if flow is not None:
        arrays.update(
            gap_midpoints=flow.gap_midpoints,
            z2=np.asarray(flow.z2),
            z2_crossings=np.asarray(flow.crossings),
            z2_ambiguous=np.asarray(flow.ambiguous),
        )
        dims.update(gap_midpoints=("half_transverse",), z2=(), z2_crossings=(), z2_ambiguous=())
        units.update(gap_midpoints="1", z2="1", z2_crossings="1", z2_ambiguous="1")
    return ObservableResult(
        arrays,
        dims,
        units,
        {
            "observable": "wilson_plane",
            "fixed_axis": int(fixed_axis),
            "value": value,
            "loop_axis": a,
            "transverse_axis": b,
            "loop_points": loop_count,
            "transverse_points": int(n_t),
            "attempts": attempts,
            "occupied_bands": count,
            "z2_computed": flow is not None,
            "min_largest_gap": None if flow is None else flow.min_largest_gap,
            "position_link": getattr(model, "metadata", {}).get("position_link"),
            "wcc_convention": "sorted -arg(eig W)/2pi modulo 1 along the loop lattice vector",
            "chern_convention": "C = -(1/2pi) sum_t wrap(dphi), phi = -Im ln det W",
        },
    )
