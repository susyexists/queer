"""The QUEER_BERRY step: Berry curvature and quantum metric with the PythTB driver's semantics.

:func:`run_berry` follows the Generator driver's ``run_berry`` on the model of
:class:`~queer.compat.generator.wannier_steps.WannierStepContext`:

- the plane is ``berry_plane`` when given, else the layer of a 2D model or
  k_3 = 0 of a 3D one; its axes (a, b) follow
  :func:`~queer.physics.topology.plane_axes`;
- ``occupation``: ``bands`` sums the Kubo curvature of the occupied set of
  :meth:`~WannierStepContext.resolve_occupation`, ``fermi`` that of every state
  below E_F (pairs closer than 1e-6 eV across E_F are skipped), and ``auto``
  takes ``bands`` when that set is gapped and isolated, else ``fermi``.
  ``bands`` without an occupied set is ``queer_bad_params``;
- the curvature is sampled on ``berry_mesh`` x ``berry_mesh`` points over
  [0, 1) of the plane by :func:`~queer.observables.geometry.berry_plane`:
  ``omega_reduced`` = b_a . Omega . b_b with b = 2 pi inv(A)^T, whose sum over
  the mesh divided by n^2 2 pi is ``chern_kubo`` (bands, with its distance from
  the nearest integer and a warning beyond 0.05) or ``fermi_sea_integral_e2_h``
  with ``sigma_xy_e2_h`` its negative (fermi). Every result states the sign
  conventions in ``berry.conventions`` and the largest |Omega| in Å²;
- with ``quantum_metric`` in ``bands`` mode, the in-plane trace of the
  finite-basis interband metric gives ``quantum_weight`` and whether it bounds
  |C| within 1e-6; in ``fermi`` mode the metric is skipped with a warning;
- with ``path_curvature``, the curvature (Å²) along the band path of
  :meth:`~WannierStepContext.band_path` against the path length in 1/Å with
  2 pi; a path that cannot be resolved or solved only adds a warning.

The step writes ``Out/berry_k{c}_{v}.json`` (``k_frac``, ``omega_reduced``,
``omega_cart_A2`` and, with the metric, ``metric_trace_A2``),
``Out/berry_path.json`` and the figures ``berry_k{c}_{v}.png``,
``metric_k{c}_{v}.png`` and ``berry_path.png`` with the driver's titles. A
curvature QUEER refuses on the plane (an occupied set whose boundary splits a
degenerate subspace) fails the step with ``queer_numerical_error``.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ...observables.geometry import berry_plane
from ...observables.topology import plane_kpoints, topology_planes
from ...physics.geometry import plane_flux, quantum_weight
from ...physics.topology import plane_axes
from .estimators import k_distance, plane_frame
from .plots import new_figure, save_figure, tick_marks
from .results import (
    APPROXIMATION_NOTE,
    BERRY_CONVENTIONS,
    CHERN_INTEGER_TOL,
    ERR_BAD_PARAMS,
    ERR_NUMERICAL,
    OUT_DIR,
    WARN_CHERN_NOT_INTEGER,
    WARN_METRIC_SKIPPED,
    WARN_PATH_CURVATURE_SKIPPED,
    WARN_PLOT_SKIPPED,
    RunnerError,
    write_json_atomic,
)
from .wannier_steps import WannierStepContext

BANDS_WITHOUT_OCCUPATION = "occupation='bands' needs a gapped model or occupied_bands"
METRIC_SKIPPED_WARNING = (
    "quantum metric skipped: it needs an isolated occupied band set (occupation 'bands'), and this run "
    "sums over the Fermi sea ('fermi')"
)
# Matrices per k-point the driver plans a worker's memory with.
PER_K_MATRICES = {"bands": 14, "fermi": 8}
# Pairs closer than this across E_F have no finite Kubo term (eV).
FERMI_SEA_MIN_GAP_EV = 1e-6
# The bound K >= |C| holds up to this slack.
QUANTUM_WEIGHT_SLACK = 1e-6
PATH_FILE = f"{OUT_DIR}/berry_path.json"


def selected_plane(requested: Any, periodic: Sequence[int]) -> tuple[int, float]:
    """``berry_plane`` (fixed axis, value) when given, else the first plane of :func:`topology_planes`."""
    if isinstance(requested, Mapping):
        return int(requested.get("fixed_axis", 2)), float(requested.get("value", 0.0))
    fixed, value = topology_planes(periodic)[0]
    return fixed, value


def plane_file_stem(prefix: str, fixed: int, value: float) -> str:
    """``{prefix}_k{c}_{v}`` of a plane, with ``p`` for the decimal point (``berry_k3_0p5``)."""
    return f"{prefix}_k{fixed + 1}_{value:g}".replace(".", "p")


def not_integer_warning(chern: float, distance: float, mesh: int) -> str:
    """Why a Kubo Chern number away from an integer calls for a finer mesh."""
    return (
        f"the Kubo Chern number {chern:.3f} is {distance:.3f} from an integer: a small gap is "
        f"under-resolved on the {mesh}x{mesh} mesh (raise berry_mesh)"
    )


def plane_curvature(
    context: WannierStepContext,
    points: np.ndarray,
    plane: tuple[int, int],
    occupied: int | None,
    metric: bool,
    stage: str,
    per_k: int,
) -> Any:
    """:func:`berry_plane` of ``points`` under the stage's parallel plan; a refusal is a numerical error."""
    options = context.parallel(stage, per_k, len(points))
    try:
        return berry_plane(
            context.model,
            points,
            plane=plane,
            occupied_bands=occupied,
            metric=metric,
            min_gap=FERMI_SEA_MIN_GAP_EV,
            execution=options,
        )
    except ValueError as error:
        raise RunnerError(ERR_NUMERICAL, f"{stage} failed: {error}") from error


def run_berry(context: WannierStepContext) -> None:
    """Berry curvature, and the quantum metric of a band set, on a plane and along the path."""
    params = context.params
    record = context.record
    fixed, value = selected_plane(params.get("berry_plane"), context.periodic)
    a, b = plane_axes(fixed)
    mode = str(params.driver_value("occupation", "auto")).lower()
    occupation = context.resolve_occupation()
    occupied = occupation.get("n_occ")
    if mode == "auto":
        mode = "bands" if occupied and occupation.get("isolated") else "fermi"
    band_set: int | None = None
    if mode == "bands":
        if not occupied:
            raise RunnerError(ERR_BAD_PARAMS, BANDS_WITHOUT_OCCUPATION)
        band_set = int(occupied)
    metric = bool(params.get("quantum_metric")) and mode == "bands"
    mesh = int(params.driver_value("berry_mesh", 150))
    points = plane_kpoints(fixed, value, mesh)
    per_k = PER_K_MATRICES["bands" if mode == "bands" else "fermi"]
    plane_map = plane_curvature(
        context,
        points,
        (a, b),
        band_set,
        metric,
        f"Berry curvature ({mode}), k{fixed + 1}={value:g}",
        per_k,
    )
    omega = np.asarray(plane_map["omega_reduced"], dtype=float)
    omega_cart = np.asarray(plane_map["omega_cartesian"], dtype=float)
    area = float(plane_map.metadata["plane_area_inverse_angstrom2"])
    integral = plane_flux(omega, mesh)
    berry: dict[str, Any] = {
        "fixed_axis": fixed,
        "value": value,
        "plane_axes": [a, b],
        "occupation": mode,
        "occupation_info": occupation,
        "mesh": mesh,
        "approximation": APPROXIMATION_NOTE,
    }
    if mode == "bands":
        distance = abs(integral - round(integral))
        berry["chern_kubo"] = integral
        berry["chern_kubo_integer_distance"] = distance
        if distance > CHERN_INTEGER_TOL:
            record.warn(WARN_CHERN_NOT_INTEGER, not_integer_warning(integral, distance, mesh))
    else:
        berry["fermi_sea_integral_e2_h"] = integral
        berry["sigma_xy_e2_h"] = -integral
    berry["conventions"] = dict(BERRY_CONVENTIONS)
    if params.get("quantum_metric") and mode != "bands":
        record.warn(WARN_METRIC_SKIPPED, METRIC_SKIPPED_WARNING)
    berry["omega_cart_max_A2"] = float(np.abs(omega_cart).max())
    trace = None
    if metric:
        trace = np.asarray(plane_map["metric_trace"], dtype=float)
        weight = quantum_weight(trace, area, mesh)
        berry["quantum_weight"] = weight
        berry["quantum_weight_bound_ok"] = bool(weight + QUANTUM_WEIGHT_SLACK >= abs(integral))
    record.results["berry"] = berry
    context.log(
        f"Berry curvature ({mode}), k{fixed + 1}={value:g}: "
        + (f"C_kubo={integral:+.4f}" if mode == "bands" else f"Fermi-sea integral {integral:+.4f} e^2/h")
    )

    stem = plane_file_stem("berry", fixed, value)
    data: dict[str, Any] = {"k_frac": points, "omega_reduced": omega, "omega_cart_A2": omega_cart}
    if trace is not None:
        data["metric_trace_A2"] = trace
    write_json_atomic(context.workspace / OUT_DIR / f"{stem}.json", data)
    record.add_output(f"{OUT_DIR}/{stem}.json", "Berry curvature map (data)", "data")
    plot_plane_map(
        context, points, (a, b), mesh, omega_cart, f"{stem}.png", "Berry curvature Ω (Å²)", "RdBu_r", True
    )
    if trace is not None:
        plot_plane_map(
            context,
            points,
            (a, b),
            mesh,
            trace,
            plane_file_stem("metric", fixed, value) + ".png",
            "Quantum metric tr g (Å²)",
            "viridis",
            False,
        )
    if params.get("path_curvature"):
        path_curvature(context, berry, (a, b), band_set, mode, per_k)


def path_curvature(
    context: WannierStepContext,
    berry: dict[str, Any],
    plane: tuple[int, int],
    band_set: int | None,
    mode: str,
    per_k: int,
) -> None:
    """The curvature of the plane's component along the band path: ``Out/berry_path.json`` and its figure."""
    try:
        path = context.band_path()
        along = plane_curvature(
            context,
            np.asarray(path["kpoints"], dtype=float),
            plane,
            band_set,
            False,
            f"Berry curvature ({mode}), path",
            per_k,
        )
    except RunnerError as error:
        context.record.warn(WARN_PATH_CURVATURE_SKIPPED, f"path curvature skipped: {error}")
        return
    omega = np.asarray(along["omega_cartesian"], dtype=float)
    distance = k_distance(path["kpoints"], context.lattice, path["break_indices"])
    berry["path_file"] = PATH_FILE
    write_json_atomic(
        context.workspace / PATH_FILE,
        {
            "k_distance": distance,
            "omega_cart_A2": omega,
            "tick_indices": path["tick_indices"],
            "tick_labels": path["tick_labels"],
        },
    )
    context.record.add_output(PATH_FILE, "Berry curvature along the path (data)", "data")
    plot_path_curvature(context, path, distance, omega)


def plot_plane_map(
    context: WannierStepContext,
    points: np.ndarray,
    plane: tuple[int, int],
    mesh: int,
    values: np.ndarray,
    name: str,
    title: str,
    colormap: str,
    symmetric: bool,
) -> None:
    """A map over the plane in Cartesian k (1/Å with 2 pi); ``symmetric`` centres the colour scale on 0."""
    figure = new_figure((6, 5))
    if figure is None:
        return
    try:
        recip, ea, eb = plane_frame(context.lattice, *plane)
        cartesian = points @ (2.0 * np.pi * recip)
        x = (cartesian @ ea).reshape(mesh, mesh)
        y = (cartesian @ eb).reshape(mesh, mesh)
        grid = np.asarray(values, dtype=float).reshape(mesh, mesh)
        axes = figure.subplots()
        limits: dict[str, float] = {}
        if symmetric:
            limit = float(np.percentile(np.abs(grid), 99.5)) or 1.0
            limits = {"vmin": -limit, "vmax": limit}
        with warnings.catch_warnings():
            # The cell centres of a sheared plane are not monotonic along the mesh rows.
            warnings.filterwarnings("ignore", "The input coordinates to pcolormesh", UserWarning)
            image = axes.pcolormesh(x, y, grid, cmap=colormap, shading="auto", **limits)
        figure.colorbar(image, ax=axes)
        axes.set_aspect("equal")
        axes.set_xlabel("k (1/Å)")
        axes.set_ylabel("k (1/Å)")
        axes.set_title(title)
        save_figure(context.record, figure, name, title)
    except Exception as error:  # a plot never fails a run
        context.record.warn(WARN_PLOT_SKIPPED, f"{name} skipped: {error}")


def plot_path_curvature(
    context: WannierStepContext, path: Mapping[str, Any], distance, omega: np.ndarray
) -> None:
    """``berry_path.png``: the curvature along the band path with its high-symmetry ticks."""
    figure = new_figure((7, 3.5))
    if figure is None:
        return
    try:
        axes = figure.subplots()
        axes.plot(distance, omega, color="#0072b2", lw=1.2)
        axes.axhline(0.0, color="#999999", lw=0.5)
        positions, labels = tick_marks(path, distance)
        axes.set_xticks(positions)
        axes.set_xticklabels(labels)
        axes.set_xlim(distance[0], distance[-1])
        axes.set_ylabel("Ω (Å²)")
        axes.set_title("Berry curvature along the path")
        save_figure(context.record, figure, "berry_path.png", "Berry curvature along the path")
    except Exception as error:  # a plot never fails a run
        context.record.warn(WARN_PLOT_SKIPPED, f"path curvature plot skipped: {error}")
