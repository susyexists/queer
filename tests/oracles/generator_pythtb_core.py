"""PythTB post-processing of a Wannier90 tight-binding model (the ``pythtb`` engine).

One module, two lives (the ``engines/hamgnn_core.py`` pattern):

  * imported in-cluster, numpy only, by ``engines/pythtb_provider.py``, the result
    parser and the tests. It owns the results contract and the pure-numpy physics
    helpers, so the parser and the cluster driver cannot drift apart;
  * staged verbatim into each PythTB job workspace as ``_pythtb_core.py`` and RUN
    there: ``python _pythtb_core.py pythtb-run --params pythtb_params.json``.

PythTB, matplotlib and scipy are imported lazily, inside the run command only:
the backend image has none of them, and the cluster env pins ``pythtb==2.0.2``.

What PythTB 2.0.2 gives a Wannier90 model, and what this module adds:

  * ``W90(path, "wannier90").model(zero_energy=E_F)`` reads the ``.win`` lattice,
    ``_hr.dat`` and ``_centres.xyz``. The model is spinless (a SOC model is
    num_wann plain orbitals), periodic in all three directions, and puts each
    orbital at its Wannier centre with a diagonal position operator. Band
    energies and integer invariants are exact for the model; Berry-type
    magnitudes are approximate (``APPROXIMATION_NOTE``).
  * PythTB ignores ``wannier90_wsvec.dat``. wannier90 3.x interpolates with the
    Wigner-Seitz distance correction on by default, so the raw ``_hr.dat`` does
    not reproduce ``wannier90_band.dat``. ``ws_unfold`` recomputes the correction
    (the wsvec file is cluster-only scratch and is not staged by default).
  * PythTB has no DOS, gap, Fermi-level occupation, Z2 or fast slab builder; those
    live here, on top of its eigensolver.

Conventions: energies are relative to the Fermi level, k-points are fractional
coordinates of the reciprocal lattice, and ``H[R][i, j] = <i, 0|H|j, R>`` (the
``_hr.dat`` and PythTB convention).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Results contract (shared with engines/pythtb_provider.py and the parser)
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1

CORE_FILENAME = "_pythtb_core.py"
PARAMS_FILENAME = "pythtb_params.json"
RESULTS_FILENAME = "pythtb_results.json"
LOG_FILENAME = "pythtb.log"
OUT_DIR = "Out"
SEEDNAME = "wannier90"
# Scratch dir holding the sanitised .win PythTB actually reads (plus symlinks
# to the staged model). Removed at the end of the run: nothing in it is a
# product, and the staged originals are never touched.
MODEL_DIR = "_pythtb_model"

WIN_FILENAME = f"{SEEDNAME}.win"
HR_FILENAME = f"{SEEDNAME}_hr.dat"
CENTRES_FILENAME = f"{SEEDNAME}_centres.xyz"
BAND_KPT_FILENAME = f"{SEEDNAME}_band.kpt"
BAND_DAT_FILENAME = f"{SEEDNAME}_band.dat"
LABELINFO_FILENAME = f"{SEEDNAME}_band.labelinfo.dat"
WSVEC_FILENAME = f"{SEEDNAME}_wsvec.dat"

REQUIRED_INPUTS: Tuple[str, ...] = (WIN_FILENAME, HR_FILENAME, CENTRES_FILENAME)
OPTIONAL_INPUTS: Tuple[str, ...] = (
    BAND_KPT_FILENAME,
    BAND_DAT_FILENAME,
    LABELINFO_FILENAME,
)
TB_FILENAME = f"{SEEDNAME}_tb.dat"
EIG_FILENAME = f"{SEEDNAME}.eig"
# The full position matrix, written by the WANNIER run itself (write_tb).
POSITION_INPUTS: Tuple[str, ...] = (TB_FILENAME,)
WANNIER_INPUTS: Tuple[str, ...] = (
    REQUIRED_INPUTS
    + OPTIONAL_INPUTS
    + (WSVEC_FILENAME,)
    + POSITION_INPUTS
    + (EIG_FILENAME,)
)

STEP_BANDS = "PYTHTB_BANDS"
STEP_TOPOLOGY = "PYTHTB_TOPOLOGY"
STEP_BERRY = "PYTHTB_BERRY"
STEP_SLAB = "PYTHTB_SLAB"
STEP_QGEOM = "PYTHTB_QGEOM"
STEP_TYPES: Tuple[str, ...] = (
    STEP_BANDS,
    STEP_TOPOLOGY,
    STEP_BERRY,
    STEP_SLAB,
    STEP_QGEOM,
)
# Optional bindings beyond the model files, per step.
STEP_EXTRA_INPUTS: Dict[str, Tuple[str, ...]] = {
    STEP_QGEOM: POSITION_INPUTS + (EIG_FILENAME,)
}
QGEOM_MODULE_FILENAME = "_pythtb_qgeom.py"

# Stable failure codes. The parser turns them into the job's failure_reason.
ERR_ENV_MISSING = "pythtb_env_missing"
ERR_MISSING_INPUT = "pythtb_missing_input"
ERR_TRANSLATE_HOME_CELL = "pythtb_translate_home_cell"
ERR_VALIDATION = "pythtb_validation_failed"
ERR_BAD_PARAMS = "pythtb_bad_params"
ERR_MEMORY = "pythtb_oom"
ERR_INTERNAL = "pythtb_internal_error"
ERR_RESULTS_MISSING = "pythtb_results_missing"
ERR_POSITION_UNAVAILABLE = "pythtb_position_unavailable"
ERR_POSITION_MISMATCH = "pythtb_position_mismatch"

APPROXIMATION_NOTE = (
    "Tight-binding (diagonal position) approximation: the Wannier centres are the "
    "orbital positions and the off-diagonal position matrix elements are dropped. "
    "Band energies and integer invariants are unaffected; Berry-curvature, "
    "quantum-metric and Hall magnitudes are approximate."
)
# The sign conventions a Berry result states; a result without them predates
# the sign fix of the Fermi-sea curvature and has Omega and sigma_xy flipped.
BERRY_CONVENTIONS = {
    "omega": (
        "-2 Im Q: Omega_ab = -2 Im sum_{n occ, m unocc} v^a_nm v^b_mn / (E_n - E_m)^2"
    ),
    "sigma_xy": (
        "-(e^2/h) (1/2pi) sum_n int f_n Omega_n d^2k, carrier charge -e "
        "(-(e^2/h) C in a gap)"
    ),
}
# How far a Kubo Chern number may sit from an integer before the mesh is
# called too coarse for a small gap.
CHERN_INTEGER_TOL = 0.05
CITATION = (
    "T. Cole, S. Coh and D. Vanderbilt, Python Tight Binding (PythTB), "
    "doi:10.5281/zenodo.12721315"
)

# Every parameter the driver understands, with its default. The provider
# writes a fully resolved params file; these defaults keep the driver runnable
# on its own (tests, a hand re-run on the cluster). engines/pythtb_constants.py
# builds the wizard catalog from this dict, so the two cannot drift apart.
DEFAULT_PARAMS: Dict[str, Any] = {
    # Resolved by the submitter from the source WANNIER job.
    "fermi_energy": None,
    "fermi_source": None,
    "ws_hint": None,
    "spinors": None,
    "structure": None,
    "kpath": None,
    "source_wannier_job_id": None,
    "reference_band_job_id": None,
    # Model build and validation.
    "ws_correction": "auto",
    "min_hopping_norm": None,
    "max_distance": None,
    "periodic_dirs": None,
    "validation_warn_meV": 5.0,
    "validation_fail_meV": 100.0,
    "fermi_tolerance_eV": 0.02,
    "occupied_bands": None,
    "isolation_threshold_meV": 5.0,
    # Execution.
    "workers": None,
    "chunk_size": None,
    # Plots.
    "energy_window_eV": [-3.0, 3.0],
    # PYTHTB_BANDS.
    "kpath_points": 400,
    "dos_mesh": None,
    "dos_sigma_eV": 0.03,
    "dos_step_eV": 0.005,
    "pdos": True,
    "fermi_surface": True,
    "fermi_surface_mesh": None,
    # PYTHTB_TOPOLOGY.
    "planes": None,
    "loop_points": 81,
    "transverse_points": 61,
    "max_refinements": 2,
    "z2": "auto",
    "kramers_tol_meV": 1.0,
    "z2_pos_tol": 0.01,
    # PYTHTB_BERRY.
    "berry_plane": None,
    "berry_mesh": 150,
    "occupation": "auto",
    "quantum_metric": True,
    "path_curvature": True,
    # PYTHTB_SLAB.
    "slab_axis": None,
    "slab_cells": None,
    "slab_kpoints": 200,
    "slab_window_eV": 0.5,
    "edge_cells": 2,
    # PYTHTB_QGEOM.
    "position_operator": "auto",
    "mu_range_meV": [-100.0, 100.0],
    "mu_step_meV": 0.1,
    "temperatures_K": [2.0, 10.0, 30.0],
    "qgeom_tolerance": 1e-3,
    "qgeom_base_mesh": 64,
    "qgeom_max_points": 1_000_000,
    "qgeom_gap_ratio": 0.1,
    "insulator_min_gap_meV": 0.1,
    "degeneracy_tol_meV": 0.001,
    "qgeom_band_window": 1,
    # PYTHTB_QGEOM, API-only.
    "occupation_mesh": 241,
    "edge_resolution_meV": 0.2,
    "qgeom_gap_floor_meV": 0.05,
    "path_log_points": 40,
    "path_kpoints_frac": None,
    "path_mu_meV": None,
    "wannier_omega_I_A2": None,
    "time_budget_s": None,
}

# Keys the submitter resolves; never shown as wizard fields.
CONTEXT_PARAM_KEYS: Tuple[str, ...] = (
    "fermi_source",
    "ws_hint",
    "spinors",
    "structure",
    "kpath",
    "source_wannier_job_id",
    "reference_band_job_id",
    "wannier_omega_I_A2",
    "time_budget_s",
)

# Which parameters each step reads (besides the shared model/validation ones).
SHARED_PARAM_KEYS: Tuple[str, ...] = (
    "fermi_energy",
    "ws_correction",
    "min_hopping_norm",
    "max_distance",
    "periodic_dirs",
    "validation_warn_meV",
    "validation_fail_meV",
    "workers",
    "chunk_size",
    "energy_window_eV",
)
STEP_PARAM_KEYS: Dict[str, Tuple[str, ...]] = {
    STEP_BANDS: (
        "kpath_points",
        "dos_mesh",
        "dos_sigma_eV",
        "dos_step_eV",
        "pdos",
        "fermi_tolerance_eV",
        "fermi_surface",
        "fermi_surface_mesh",
    ),
    STEP_TOPOLOGY: (
        "occupied_bands",
        "fermi_tolerance_eV",
        "isolation_threshold_meV",
        "planes",
        "loop_points",
        "transverse_points",
        "max_refinements",
        "z2",
        "kramers_tol_meV",
        "z2_pos_tol",
    ),
    STEP_BERRY: (
        "occupied_bands",
        "fermi_tolerance_eV",
        "isolation_threshold_meV",
        "berry_plane",
        "berry_mesh",
        "occupation",
        "quantum_metric",
        "path_curvature",
        "kpath_points",
    ),
    STEP_SLAB: (
        "slab_axis",
        "slab_cells",
        "slab_kpoints",
        "slab_window_eV",
        "edge_cells",
    ),
    STEP_QGEOM: (
        "position_operator",
        "occupied_bands",
        "mu_range_meV",
        "mu_step_meV",
        "temperatures_K",
        "qgeom_tolerance",
        "qgeom_base_mesh",
        "qgeom_max_points",
        "qgeom_gap_ratio",
        "insulator_min_gap_meV",
        "degeneracy_tol_meV",
        "qgeom_band_window",
        "kpath_points",
    ),
}

STEP_LABELS: Dict[str, str] = {
    STEP_BANDS: "Bands, DOS & gap",
    STEP_TOPOLOGY: "Topological invariants",
    STEP_BERRY: "Berry curvature & quantum geometry",
    STEP_SLAB: "Slab / edge states",
    STEP_QGEOM: "Quantum geometry & nonlinear Hall",
}


class PythtbError(Exception):
    """A failure carrying a stable ``code`` for the results contract."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def merged_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """``DEFAULT_PARAMS`` overlaid with every non-None value of ``params``."""
    out = dict(DEFAULT_PARAMS)
    for key, value in (params or {}).items():
        if value is not None:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# .win parsing (numpy-free text handling)
# ---------------------------------------------------------------------------

_WIN_COMMENT_RE = re.compile(r"[!#].*$")
_WIN_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*[=:]?\s*(.*)$")
_TRUE = {"true", ".true.", "t", ".t.", "1", "yes"}
_FALSE = {"false", ".false.", "f", ".f.", "0", "no"}
BOHR_TO_ANGSTROM = 0.52917721092


def _fortran_float(token: str) -> float:
    return float(token.strip().lower().replace("d", "e"))


def _win_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    token = value.strip().split()[0].lower() if value.strip() else ""
    if token in _TRUE:
        return True
    if token in _FALSE:
        return False
    return None


def _win_float(value: Optional[str]) -> Optional[float]:
    if value is None or not value.strip():
        return None
    try:
        return _fortran_float(value.split()[0])
    except ValueError:
        return None


def _win_int(value: Optional[str]) -> Optional[int]:
    number = _win_float(value)
    return None if number is None else int(round(number))


def parse_win(text: str) -> Dict[str, Any]:
    """The ``.win`` settings this engine reads.

    wannier90 keywords and block names are case-insensitive; ``key = value``,
    ``key : value`` and ``key value`` all occur, and ``!``/``#`` start comments.
    VASP writes the ``.win`` itself, so nothing here assumes the platform's
    own rendering.
    """
    keywords: Dict[str, str] = {}
    blocks: Dict[str, List[str]] = {}
    current: Optional[str] = None
    lines: List[str] = []
    for raw in (text or "").splitlines():
        line = _WIN_COMMENT_RE.sub("", raw).strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("begin") and (len(low) == 5 or low[5] in " \t"):
            parts = low.split()
            current = parts[1] if len(parts) > 1 else ""
            lines = []
            continue
        if low.startswith("end") and (len(low) == 3 or low[3] in " \t"):
            if current is not None:
                blocks[current] = lines
            current = None
            continue
        if current is not None:
            lines.append(line)
            continue
        match = _WIN_KEY_RE.match(line)
        if match:
            keywords[match.group(1).lower()] = match.group(2).strip()

    win: Dict[str, Any] = {"keywords": keywords, "blocks": sorted(blocks)}

    lattice = None
    cell = blocks.get("unit_cell_cart")
    if cell:
        scale = 1.0
        rows = list(cell)
        if rows and rows[0].split()[0].lower() in ("bohr", "ang", "angstrom"):
            scale = BOHR_TO_ANGSTROM if rows[0].split()[0].lower() == "bohr" else 1.0
            rows = rows[1:]
        vectors = []
        for row in rows[:3]:
            vectors.append([_fortran_float(x) * scale for x in row.split()[:3]])
        if len(vectors) == 3 and all(len(v) == 3 for v in vectors):
            lattice = vectors
    win["lattice"] = lattice

    mp_grid = None
    if "mp_grid" in keywords:
        try:
            mp_grid = [
                int(x) for x in keywords["mp_grid"].replace(",", " ").split()[:3]
            ]
        except ValueError:
            mp_grid = None
    kpoints: List[List[float]] = []
    for row in blocks.get("kpoints", []):
        parts = row.split()
        if len(parts) >= 3:
            try:
                kpoints.append([_fortran_float(x) for x in parts[:3]])
            except ValueError:
                continue
    if (not mp_grid or len(mp_grid) != 3) and kpoints:
        mp_grid = infer_mp_grid(kpoints)
    win["mp_grid"] = mp_grid if mp_grid and len(mp_grid) == 3 else None
    win["n_kpoints"] = len(kpoints)

    segments = []
    for row in blocks.get("kpoint_path", []):
        parts = row.replace(",", " ").split()
        if len(parts) < 8:
            continue
        try:
            segments.append(
                [
                    parts[0],
                    [_fortran_float(x) for x in parts[1:4]],
                    parts[4],
                    [_fortran_float(x) for x in parts[5:8]],
                ]
            )
        except ValueError:
            continue
    win["kpoint_path"] = segments

    for key in (
        "translate_home_cell",
        "use_ws_distance",
        "spinors",
        "write_hr",
        "write_xyz",
        "hr_plot",
        "bands_plot",
    ):
        win[key] = _win_bool(keywords.get(key))
    for key in (
        "dis_froz_min",
        "dis_froz_max",
        "dis_win_min",
        "dis_win_max",
        "ws_distance_tol",
        "fermi_energy",
    ):
        win[key] = _win_float(keywords.get(key))
    for key in ("ws_search_size", "num_wann", "num_bands"):
        win[key] = _win_int(keywords.get(key))
    return win


def infer_mp_grid(kpoints: Sequence[Sequence[float]]) -> Optional[List[int]]:
    """The Monkhorst-Pack divisions of a full uniform k-point list."""
    pts = np.asarray(kpoints, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3 or not len(pts):
        return None
    grid = []
    for axis in range(3):
        frac = np.mod(np.round(pts[:, axis], 6), 1.0)
        grid.append(int(len(np.unique(np.round(frac, 5)))))
    return grid


def sanitized_win(win: Dict[str, Any]) -> str:
    """The minimal ``.win`` PythTB parses: the lattice and, if valid, the path.

    PythTB reads only ``unit_cell_cart`` and ``kpoint_path`` and fails on a blank
    or comment line inside the cell block, or on a malformed path; rewriting the
    two blocks from ``parse_win`` output sidesteps both.
    """
    lines = ["begin unit_cell_cart", "ang"]
    for row in win["lattice"]:
        lines.append("  ".join(f"{x:.10f}" for x in row))
    lines.append("end unit_cell_cart")
    if win.get("kpoint_path"):
        lines.append("begin kpoint_path")
        for la, ka, lb, kb in win["kpoint_path"]:
            lines.append(
                f"{la} {ka[0]:.8f} {ka[1]:.8f} {ka[2]:.8f} "
                f"{lb} {kb[0]:.8f} {kb[1]:.8f} {kb[2]:.8f}"
            )
        lines.append("end kpoint_path")
    return "\n".join(lines) + "\n"


def parse_labelinfo(text: str) -> Tuple[List[int], List[str]]:
    """``wannier90_band.labelinfo.dat`` rows ``LABEL index xval kx ky kz`` -> 0-based indices."""
    indices: List[int] = []
    labels: List[str] = []
    for raw in (text or "").splitlines():
        parts = raw.split()
        if len(parts) < 2:
            continue
        try:
            index = int(parts[1])
        except ValueError:
            continue
        indices.append(index - 1)
        labels.append(parts[0])
    return indices, labels


def parse_band_kpt(text: str) -> np.ndarray:
    """``wannier90_band.kpt`` -> fractional k-points (count line skipped)."""
    points = []
    rows = [ln.split() for ln in (text or "").splitlines() if ln.strip()]
    if rows and len(rows[0]) == 1:
        rows = rows[1:]
    for parts in rows:
        if len(parts) >= 3:
            try:
                points.append([_fortran_float(x) for x in parts[:3]])
            except ValueError:
                continue
    return np.asarray(points, dtype=float).reshape(-1, 3)


def parse_band_dat(text: str, num_bands: int) -> np.ndarray:
    """``wannier90_band.dat`` (blocks of ``x energy`` per band) -> energies[k, band]."""
    values = []
    for raw in (text or "").splitlines():
        parts = raw.split()
        if len(parts) >= 2:
            try:
                values.append(_fortran_float(parts[1]))
            except ValueError:
                continue
    data = np.asarray(values, dtype=float)
    if num_bands <= 0 or data.size % num_bands:
        raise ValueError(
            f"{BAND_DAT_FILENAME} holds {data.size} values, not a multiple of "
            f"{num_bands} bands"
        )
    return data.reshape(num_bands, -1).T


def path_breaks(kpoints: np.ndarray, tick_indices: Sequence[int]) -> List[int]:
    """Tick indices where the sampled path JUMPS (wannier90 does not duplicate joins).

    Same rule as utils/parsers/vasp/wannier_bands.py::_break_indices: a step far
    longer than the sampling step just before it.
    """
    breaks: List[int] = []
    for index in tick_indices:
        if index < 2 or index >= len(kpoints) or index in breaks:
            continue
        step = float(np.linalg.norm(kpoints[index] - kpoints[index - 1]))
        reference = float(np.linalg.norm(kpoints[index - 1] - kpoints[index - 2]))
        if reference > 0 and step > 3.0 * reference:
            breaks.append(int(index))
    return breaks


# ---------------------------------------------------------------------------
# Lattice and k-space helpers
# ---------------------------------------------------------------------------


def reciprocal_lattice(lattice: Sequence[Sequence[float]]) -> np.ndarray:
    """Rows b_i with a_i . b_j = 2 pi delta_ij (1/Angstrom)."""
    return 2.0 * np.pi * np.linalg.inv(np.asarray(lattice, dtype=float)).T


def periodic_dirs_from_table(R: np.ndarray) -> List[int]:
    """Axes any hopping crosses. Along any other (a slab's vacuum) H(k) is flat.

    The same rule as engines/hamgnn_core.py::periodic_axes: a WANNIER run on a
    k-mesh with one point along the vacuum gives hoppings with R_c == 0 only.
    Read it off the RAW ``_hr.dat`` table: the Wigner-Seitz correction routes a
    hop between orbitals more than half a cell apart through the vacuum
    (R_c = +-1), which a thick slab (5-SL MnBi2Te4: 65 A of slab in an 81 A
    cell) always has.
    """
    R = np.asarray(R, dtype=int).reshape(-1, 3)
    return [axis for axis in range(3) if np.any(R[:, axis] != 0)]


def fold_axes(
    R: np.ndarray, H: np.ndarray, axes: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """The table with its R components along ``axes`` summed onto zero.

    Exact wherever k is zero along those axes -- the only place a 2D model is
    evaluated -- because exp(2 pi i k.R) ignores them there. Keeps a 2D model
    2D after the Wigner-Seitz correction moved hops across its vacuum.
    """
    R = np.asarray(R, dtype=int).reshape(-1, 3)
    axes = [int(axis) for axis in axes]
    if not axes or not np.any(R[:, axes]):
        return R, np.asarray(H)
    folded = R.copy()
    folded[:, axes] = 0
    keys, inverse = np.unique(folded, axis=0, return_inverse=True)
    out = np.zeros((len(keys),) + tuple(np.shape(H)[1:]), dtype=complex)
    np.add.at(out, np.asarray(inverse).reshape(-1), np.asarray(H, complex))
    return keys, out


def uniform_mesh(
    divisions: Sequence[int],
    periodic: Sequence[int],
    fixed: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """Monkhorst-Pack points over [0, 1) on the periodic axes, no endpoints.

    Non-periodic axes hold a single value (``fixed`` or 0). Returns (Nk, 3).
    """
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


def sample_path(
    segments: Sequence[Sequence[Any]], lattice: Sequence[Sequence[float]], n_points: int
) -> Dict[str, Any]:
    """Sample a ``kpoint_path`` like wannier90: points per segment proportional
    to its length, joins not duplicated, a new run started at every jump.

    ``segments`` rows are ``[label_a, k_a, label_b, k_b]`` (fractional k).
    Returns ``{"kpoints", "tick_indices", "tick_labels", "break_indices"}``.
    """
    B = reciprocal_lattice(lattice)
    segs = [
        (str(s[0]), np.asarray(s[1], float), str(s[2]), np.asarray(s[3], float))
        for s in segments
    ]
    if not segs:
        raise ValueError("empty k-path")
    lengths = [float(np.linalg.norm((kb - ka) @ B)) for _, ka, _, kb in segs]
    total = sum(lengths) or 1.0
    points: List[np.ndarray] = []
    ticks: List[int] = []
    labels: List[str] = []
    breaks: List[int] = []
    for index, ((la, ka, lb, kb), length) in enumerate(zip(segs, lengths)):
        n = max(2, int(round(max(1, n_points) * length / total)))
        ts = np.linspace(0.0, 1.0, n + 1)
        continuous = index > 0 and np.allclose(segs[index - 1][3], ka, atol=1e-6)
        if continuous:
            # The join is the previous segment's last sample; never duplicated.
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


def k_distance(
    kpoints: np.ndarray, lattice: Sequence[Sequence[float]], breaks: Sequence[int]
) -> np.ndarray:
    """Cumulative Cartesian path length (1/Angstrom, with 2 pi), frozen at jumps."""
    B = reciprocal_lattice(lattice)
    cart = np.asarray(kpoints, float) @ B
    steps = np.linalg.norm(np.diff(cart, axis=0), axis=1)
    for index in breaks:
        if 0 < index <= len(steps):
            steps[index - 1] = 0.0
    return np.concatenate([[0.0], np.cumsum(steps)])


def plane_frame(
    lattice: Sequence[Sequence[float]], a: int, b: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cartesian axes to draw the k-plane spanned by b_a and b_b on.

    Returns ``(recip, ea, eb)``: the reciprocal basis WITHOUT 2 pi (rows,
    1/A), the units of the band payload's k_distance, so a Fermi pocket's
    radius reads as the k at which its band crosses E_F on the band plot; the
    unit vector along b_a; and the in-plane unit vector normal to it.
    """
    recip = reciprocal_lattice(lattice) / (2.0 * np.pi)
    ea = recip[a] / np.linalg.norm(recip[a])
    normal = np.cross(recip[a], recip[b])
    eb = np.cross(normal / np.linalg.norm(normal), ea)
    return recip, ea, eb


def surface_path(
    lattice: Sequence[Sequence[float]], p: int, q: int, n_points: int
) -> Dict[str, Any]:
    """A high-symmetry path of the surface Brillouin zone spanned by b_p, b_q."""
    a = np.asarray(lattice, float)
    ap, aq = a[p], a[q]
    lp, lq = np.linalg.norm(ap), np.linalg.norm(aq)
    cosang = float(np.dot(ap, aq) / (lp * lq))

    def point(kp: float, kq: float) -> List[float]:
        k = [0.0, 0.0, 0.0]
        k[p], k[q] = kp, kq
        return k

    # Every surface TRIM is on the path: a topological surface state sits at an
    # odd number of them, and magnetism or strain can single out any one.
    if abs(lp - lq) < 1e-3 * max(lp, lq) and abs(abs(cosang) - 0.5) < 1e-3:
        # Hexagonal: K sits at (1/3, 1/3) for a 120 degree cell, (1/3, 2/3) for
        # 60; the three M points are (1/2, 0), (0, 1/2) and (1/2, 1/2).
        k_k = point(1 / 3, 1 / 3) if cosang < 0 else point(1 / 3, 2 / 3)
        segs = [
            ["G", point(0, 0), "M", point(0.5, 0)],
            ["M", point(0.5, 0), "K", k_k],
            ["K", k_k, "G", point(0, 0)],
            ["G", point(0, 0), "M'", point(0, 0.5)],
            ["M''", point(0.5, 0.5), "G", point(0, 0)],
        ]
    else:
        segs = [
            ["G", point(0, 0), "X", point(0.5, 0)],
            ["X", point(0.5, 0), "M", point(0.5, 0.5)],
            ["M", point(0.5, 0.5), "Y", point(0, 0.5)],
            ["Y", point(0, 0.5), "G", point(0, 0)],
        ]
    return sample_path(segs, lattice, n_points)


# ---------------------------------------------------------------------------
# Hopping table and the Wigner-Seitz distance correction
# ---------------------------------------------------------------------------


def table_from_ham_r(
    ham_r: Dict[Tuple[int, int, int], Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """PythTB's ``W90.ham_r`` -> (R (nR, 3) int, H (nR, N, N) complex), degeneracy divided out."""
    keys = sorted(ham_r)
    R = np.asarray(keys, dtype=int).reshape(-1, 3)
    H = np.stack(
        [np.asarray(ham_r[k]["h"], complex) / float(ham_r[k]["deg"]) for k in keys]
    )
    return R, H


def table_to_ham_r(
    R: np.ndarray, H: np.ndarray
) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    """Inverse of ``table_from_ham_r`` with every degeneracy already applied."""
    return {tuple(int(x) for x in r): {"h": h, "deg": 1} for r, h in zip(R, H)}


def hk_from_table(R: np.ndarray, H: np.ndarray, kpts: np.ndarray) -> np.ndarray:
    """H(k) = sum_R exp(2 pi i k.R) H(R) for every k (no orbital phases: same spectrum)."""
    kpts = np.asarray(kpts, float).reshape(-1, 3)
    phases = np.exp(2j * np.pi * (kpts @ np.asarray(R, float).T))
    n = H.shape[1]
    return (phases @ H.reshape(len(R), n * n)).reshape(len(kpts), n, n)


def eigvals_from_table(
    R: np.ndarray, H: np.ndarray, kpts: np.ndarray, chunk: int = 256
) -> np.ndarray:
    out = []
    kpts = np.asarray(kpts, float).reshape(-1, 3)
    for start in range(0, len(kpts), chunk):
        out.append(np.linalg.eigvalsh(hk_from_table(R, H, kpts[start : start + chunk])))
    return np.concatenate(out) if out else np.zeros((0, H.shape[1]))


def hermiticity_error(R: np.ndarray, H: np.ndarray) -> float:
    """max |H(-R) - H(R)^dagger| over the table (a partner missing counts fully)."""
    index = {tuple(int(x) for x in r): i for i, r in enumerate(R)}
    worst = 0.0
    for r, i in index.items():
        j = index.get(tuple(-x for x in r))
        if j is None:
            worst = max(worst, float(np.abs(H[i]).max()))
            continue
        worst = max(worst, float(np.abs(H[j] - H[i].conj().T).max()))
    return worst


def ws_unfold(
    R: np.ndarray,
    H: np.ndarray,
    centres_cart: np.ndarray,
    lattice: Sequence[Sequence[float]],
    mp_grid: Sequence[int],
    *,
    search_size: int = 2,
    tol: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply wannier90's ``use_ws_distance`` correction to a hopping table.

    wannier90 3.x (ws_distance.F90) moves each hopping (R, i, j) to the
    supercell images R + T (T a multiple of mp_grid) that put Wannier function j
    nearest to Wannier function i, splitting it evenly when several images tie
    within ``tol``. The distances use the centres exactly as ``_centres.xyz``
    holds them (``translate_home_cell = false``): on wannier90 3.1's silicon
    example this reproduces its own ``_wsvec.dat`` table entry for entry, while
    folding the centres into the home cell first is off by 1.7 eV.
    Returns the unfolded table. ``H`` may carry trailing axes (nR, N, N, ...),
    e.g. H stacked with the position matrices, which then share each hop's
    images exactly.
    """
    lat = np.asarray(lattice, float)
    taus = np.asarray(centres_cart, float).reshape(-1, 3)
    n = np.arange(-int(search_size), int(search_size) + 1)
    grid = np.stack(np.meshgrid(n, n, n, indexing="ij"), axis=-1).reshape(-1, 3)
    shifts = grid * np.asarray(mp_grid, dtype=int)[None, :]
    shifts_cart = shifts @ lat
    shift_sq = np.einsum("tk,tk->t", shifts_cart, shifts_cart)
    diff0 = taus[None, :, :] - taus[:, None, :]  # [i, j] = tau_j - tau_i
    out: Dict[Tuple[int, int, int], np.ndarray] = {}
    trailing = (None,) * (np.ndim(H) - 3)
    for r, h in zip(np.asarray(R, int), H):
        vec = diff0 + (r @ lat)[None, None, :]
        d2 = (
            np.einsum("ijk,ijk->ij", vec, vec)[None, :, :]
            + 2.0 * np.einsum("ijk,tk->tij", vec, shifts_cart)
            + shift_sq[:, None, None]
        )
        dist = np.sqrt(np.maximum(d2, 0.0))
        nearest = dist.min(axis=0)
        mask = np.abs(dist - nearest[None]) < tol
        ndeg = mask.sum(axis=0)
        weight = np.asarray(h, complex) / ndeg[(Ellipsis,) + trailing]
        for t in np.nonzero(mask.any(axis=(1, 2)))[0]:
            key = tuple(int(x) for x in (r + shifts[t]))
            contrib = np.where(mask[t][(Ellipsis,) + trailing], weight, 0.0)
            if key in out:
                out[key] += contrib
            else:
                out[key] = contrib.astype(complex)
    keys = sorted(out)
    return np.asarray(keys, dtype=int).reshape(-1, 3), np.stack([out[k] for k in keys])


def parse_wsvec(text: str) -> Dict[Tuple[int, int, int, int, int], np.ndarray]:
    """``wannier90_wsvec.dat`` -> {(R1, R2, R3, i, j) (0-based i, j): shifts (ndeg, 3)}."""
    lines = (text or "").splitlines()
    out: Dict[Tuple[int, int, int, int, int], np.ndarray] = {}
    pos = 1 if lines and lines[0].lstrip().startswith("#") else 0
    while pos < len(lines):
        head = lines[pos].split()
        if len(head) < 5:
            pos += 1
            continue
        r1, r2, r3, i, j = (int(x) for x in head[:5])
        ndeg = int(lines[pos + 1].split()[0])
        shifts = [[int(x) for x in lines[pos + 2 + d].split()[:3]] for d in range(ndeg)]
        out[(r1, r2, r3, i - 1, j - 1)] = np.asarray(shifts, dtype=int).reshape(-1, 3)
        pos += 2 + ndeg
    return out


def ws_unfold_from_wsvec(
    R: np.ndarray,
    H: np.ndarray,
    wsvec: Dict[Tuple[int, int, int, int, int], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """``ws_unfold`` driven by wannier90's own wsvec file instead of a recompute.

    Like ``ws_unfold``, ``H`` may carry trailing axes (nR, N, N, ...).
    """
    out: Dict[Tuple[int, int, int], np.ndarray] = {}
    n = H.shape[1]
    tail = tuple(np.shape(H)[3:])
    for r, h in zip(np.asarray(R, int), H):
        for i in range(n):
            for j in range(n):
                shifts = wsvec.get((int(r[0]), int(r[1]), int(r[2]), i, j))
                if shifts is None or not len(shifts):
                    shifts = np.zeros((1, 3), dtype=int)
                share = h[i, j] / len(shifts)
                for s in shifts:
                    key = (int(r[0] + s[0]), int(r[1] + s[1]), int(r[2] + s[2]))
                    block = out.get(key)
                    if block is None:
                        block = out[key] = np.zeros((n, n) + tail, complex)
                    block[i, j] += share
    keys = sorted(out)
    return np.asarray(keys, dtype=int).reshape(-1, 3), np.stack([out[k] for k in keys])


WIN_WINDOW_KEYS: Tuple[str, ...] = (
    "dis_froz_min",
    "dis_froz_max",
    "dis_win_min",
    "dis_win_max",
)
DEFAULT_WINDOW_EV = 2.0


def comparison_window(win: Dict[str, Any], fermi: float) -> Tuple[float, float]:
    """Where model and reference bands must agree, in E - E_F (eV).

    The same rule as services/wannier_band_agreement.py::resolve_window: the
    frozen window (its lower edge defaulting to the outer window's, as in
    wannier90), else the outer window, else +-2 eV around E_F.
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


def project_path(
    segments: Sequence[Sequence[Any]], periodic: Sequence[int]
) -> List[List[Any]]:
    """A path for a model periodic only along ``periodic``.

    Components along a non-periodic axis are set to 0 (a 2D model's H(k) does
    not depend on them, so seekpath's A/L/H points of a vacuum cell collapse
    onto Gamma/M/K), and segments that collapse to a point are dropped.
    """
    out: List[List[Any]] = []
    for la, ka, lb, kb in segments:
        ka = [float(x) if axis in periodic else 0.0 for axis, x in enumerate(ka)]
        kb = [float(x) if axis in periodic else 0.0 for axis, x in enumerate(kb)]
        if np.allclose(ka, kb, atol=1e-9):
            continue
        out.append([la, ka, lb, kb])
    return out


def band_deviation(
    model_bands: np.ndarray,
    reference: np.ndarray,
    window: Optional[Tuple[float, float]],
) -> Dict[str, Any]:
    """Max/RMS |model - reference| (meV), over all bands and inside ``window`` (eV, same zero)."""
    diff = np.abs(np.asarray(model_bands, float) - np.asarray(reference, float))
    out: Dict[str, Any] = {
        "max_meV": float(diff.max() * 1000.0) if diff.size else None,
        "rms_meV": float(np.sqrt(np.mean(diff**2)) * 1000.0) if diff.size else None,
        "window_eV": list(window) if window else None,
    }
    inside = None
    if window:
        inside = (reference >= window[0]) & (reference <= window[1])
    if inside is not None and inside.any():
        sel = diff[inside]
        out["in_window_max_meV"] = float(sel.max() * 1000.0)
        out["in_window_rms_meV"] = float(np.sqrt(np.mean(sel**2)) * 1000.0)
        out["in_window_points"] = int(sel.size)
    else:
        # No band inside the window (a wide-gap model): judge on every band.
        out["window_empty"] = inside is not None
        out["in_window_max_meV"] = out["max_meV"]
        out["in_window_rms_meV"] = out["rms_meV"]
        out["in_window_points"] = int(diff.size)
    return out


# ---------------------------------------------------------------------------
# Spectra: DOS, band edges, gap
# ---------------------------------------------------------------------------


class BandEdges:
    """Running per-band extrema, Fermi-level counts and direct gaps over k-chunks."""

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

    def add(self, kpts: np.ndarray, energies: np.ndarray) -> None:
        energies = np.asarray(energies, float).reshape(len(kpts), -1)
        if not len(kpts):
            return
        lo = energies.argmin(axis=0)
        hi = energies.argmax(axis=0)
        cand_min = energies[lo, np.arange(energies.shape[1])]
        cand_max = energies[hi, np.arange(energies.shape[1])]
        better = cand_min < self.minimum
        self.minimum[better] = cand_min[better]
        self.argmin[better] = kpts[lo[better]]
        better = cand_max > self.maximum
        self.maximum[better] = cand_max[better]
        self.argmax[better] = kpts[hi[better]]
        if energies.shape[1] > 1:
            gaps = np.diff(energies, axis=1)
            at = gaps.argmin(axis=0)
            cand = gaps[at, np.arange(gaps.shape[1])]
            better = cand < self.direct
            self.direct[better] = cand[better]
            self.direct_at[better] = kpts[at[better]]
        counts = (energies < 0.0).sum(axis=1)
        self.count_min = min(self.count_min, int(counts.min()))
        self.count_max = max(self.count_max, int(counts.max()))
        self.n_k += len(kpts)

    def merge(self, other: "BandEdges") -> None:
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
    edges: BandEdges, fermi_tolerance: float, occupied_bands: Optional[int] = None
) -> Dict[str, Any]:
    """Insulator or metal, the occupied Wannier band count, and the gap.

    The Fermi level (energy zero) need not sit mid-gap: VASP often places it at
    the valence-band top. A global gap counts when it lies within
    ``fermi_tolerance`` of E_F, or when the user named the occupied band count.
    """
    n = len(edges.minimum)
    result: Dict[str, Any] = {"n_bands": n, "k_points_sampled": int(edges.n_k)}
    chosen = None
    if occupied_bands is not None:
        nocc = int(occupied_bands)
        if not 0 < nocc < n:
            raise PythtbError(
                ERR_BAD_PARAMS, f"occupied_bands={nocc} must lie between 1 and {n - 1}"
            )
        chosen = nocc
        result["n_occ_source"] = "user"
    else:
        best = None
        for nocc in range(1, n):
            lo, hi = edges.maximum[nocc - 1], edges.minimum[nocc]
            if hi - lo <= 1e-3:
                continue
            if lo - fermi_tolerance <= 0.0 <= hi + fermi_tolerance:
                # Prefer the gap whose interior holds E_F most squarely.
                score = min(0.0 - lo, hi - 0.0)
                if best is None or score > best[0]:
                    best = (score, nocc)
        if best is not None:
            chosen = best[1]
            result["n_occ_source"] = "auto"
    if chosen is None:
        crossing = [
            int(b) for b in range(n) if edges.minimum[b] < 0.0 < edges.maximum[b]
        ]
        result.update(
            metal=True,
            n_occ=None,
            gap_eV=None,
            bands_crossing_fermi=crossing,
            fermi_level_outside_model=bool(
                edges.maximum.max() < 0.0 or edges.minimum.min() > 0.0
            ),
        )
        return result
    nocc = chosen
    lo, hi = float(edges.maximum[nocc - 1]), float(edges.minimum[nocc])
    direct = float(edges.direct[nocc - 1]) if nocc - 1 < len(edges.direct) else None
    vbm_k = edges.argmax[nocc - 1].tolist()
    cbm_k = edges.argmin[nocc].tolist()
    result.update(
        metal=hi - lo <= 1e-3,
        n_occ=int(nocc),
        gap_eV=float(hi - lo),
        direct_gap_eV=direct,
        direct_gap_k=edges.direct_at[nocc - 1].tolist() if direct is not None else None,
        is_direct=bool(np.allclose(vbm_k, cbm_k, atol=1e-6)),
        vbm_eV=lo,
        cbm_eV=hi,
        vbm_k=vbm_k,
        cbm_k=cbm_k,
    )
    return result


def gaussian_dos(hist: np.ndarray, step: float, sigma: float) -> np.ndarray:
    """Smooth a k-averaged eigenvalue histogram (states per bin) into a DOS (states/eV)."""
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


# ---------------------------------------------------------------------------
# Wilson loops, Chern numbers, Z2
# ---------------------------------------------------------------------------


def wilson_loop(
    occ_vectors: Sequence[np.ndarray], boundary_phase: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Berry phase and Wannier charge centres of one closed k-loop.

    ``occ_vectors[j]`` are the occupied states at loop point j, as rows
    (n_occ, n_orb) in PythTB's cell-periodic convention; the loop closes on the
    first point times ``boundary_phase`` = exp(-i G.tau). Every link is made
    unitary by SVD, as PythTB's WFArray.berry_phase does.

    Returns ``(phi, wcc)``: phi = -Im ln det W in (-pi, pi], and the sorted
    centres -arg(lambda)/2pi folded into [0, 1).
    """
    n_occ = occ_vectors[0].shape[0]
    W = np.eye(n_occ, dtype=complex)
    count = len(occ_vectors)
    for j in range(count):
        a = occ_vectors[j]
        b = (
            occ_vectors[j + 1]
            if j + 1 < count
            else occ_vectors[0] * boundary_phase[None, :]
        )
        u, _, vh = np.linalg.svd(a.conj() @ b.T)
        W = W @ (u @ vh)
    phi = float(-np.angle(np.linalg.det(W)))
    wcc = np.sort(np.mod(-np.angle(np.linalg.eigvals(W)) / (2.0 * np.pi), 1.0))
    return phi, wcc


def wrap_phase(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, float) + np.pi) % (2.0 * np.pi) - np.pi


def chern_from_phases(phases: Sequence[float]) -> Tuple[float, float]:
    """Chern number of a plane from the loop Berry phases along its closed transverse axis.

    C = -(1/2 pi) sum_j wrap(phi_{j+1} - phi_j). The minus sign matches PythTB's
    own WFArray.chern_number / TBModel.chern_number for the same plane
    orientation. Returns ``(chern, max_step)``; the winding is trustworthy only
    while every step stays well below pi.
    """
    phi = np.asarray(phases, float)
    steps = wrap_phase(np.diff(np.append(phi, phi[0])))
    return float(-steps.sum() / (2.0 * np.pi)), (
        float(np.abs(steps).max()) if len(steps) else 0.0
    )


def _largest_gap_midpoint(wcc: np.ndarray) -> Tuple[float, float]:
    x = np.sort(np.mod(np.asarray(wcc, float), 1.0))
    if len(x) == 0:
        return 0.5, 1.0
    gaps = np.diff(np.append(x, x[0] + 1.0))
    i = int(np.argmax(gaps))
    return float((x[i] + gaps[i] / 2.0) % 1.0), float(gaps[i])


def z2_from_wcc(
    wcc_flow: Sequence[np.ndarray], pos_tol: float = 0.01
) -> Dict[str, Any]:
    """Z2 index from the Wannier charge centres of half a time-reversal plane.

    Soluyanov and Vanderbilt's largest-gap rule (PRB 83, 235401): follow the
    midpoint of the largest gap between centres and count the centres it jumps
    over, step by step; the parity is Z2. With Kramers pairs the count is even
    for either arc of the circle, so the arc does not matter. A centre within
    ``pos_tol`` of a gap midpoint makes the count ambiguous (sampling too coarse).
    """
    flow = [np.mod(np.asarray(w, float), 1.0) for w in wcc_flow]
    mids = [_largest_gap_midpoint(w) for w in flow]
    crossings = 0
    ambiguous = False
    for j in range(len(flow) - 1):
        z0, z1 = mids[j][0], mids[j + 1][0]
        lo, hi = min(z0, z1), max(z0, z1)
        nxt = flow[j + 1]
        crossings += int(np.sum((nxt >= lo) & (nxt < hi)))
        circ = np.minimum(
            np.abs(nxt[:, None] - np.array([z0, z1])[None, :]),
            1.0 - np.abs(nxt[:, None] - np.array([z0, z1])[None, :]),
        )
        if np.any(circ < pos_tol):
            ambiguous = True
    return {
        "z2": int(crossings % 2),
        "crossings": int(crossings),
        "ambiguous": bool(ambiguous),
        "gap_midpoints": [float(m[0]) for m in mids],
        "min_largest_gap": float(min(m[1] for m in mids)) if mids else None,
    }


def kramers_degenerate(energies: np.ndarray, tol: float) -> Tuple[bool, float]:
    """Whether every eigenvalue comes in a degenerate pair (time reversal with SOC)."""
    e = np.sort(np.asarray(energies, float), axis=-1)
    if e.shape[-1] % 2:
        return False, float("inf")
    split = np.abs(e[..., 1::2] - e[..., 0::2]).max() if e.size else 0.0
    return bool(split < tol), float(split)


def plane_axes(fixed_axis: int) -> Tuple[int, int]:
    """The (loop, transverse) axes of the plane normal to ``fixed_axis``, cyclically.

    Cyclic order makes the per-plane Chern numbers the components of the Chern
    vector: plane k_c = const integrates Omega_ab with (a, b, c) cyclic.
    """
    c = int(fixed_axis) % 3
    return (c + 1) % 3, (c + 2) % 3


# ---------------------------------------------------------------------------
# Berry curvature and quantum metric from velocity matrices
# ---------------------------------------------------------------------------


def fermi_sea_curvature(
    v_a: np.ndarray, v_b: np.ndarray, energies: np.ndarray, *, min_gap: float = 1e-6
) -> np.ndarray:
    """Omega_ab(k) of every state below E_F = 0 (eigenbasis velocities, per k).

    Omega_ab = -2 Im sum_{n occ, m unocc} v^a_nm v^b_mn / (E_n - E_m)^2, PythTB's
    Kubo form (TBModel._quantum_geometric_tensor, Omega = -2 Im Q, the sign of
    Xiao, Chang and Niu) with the occupation set by the Fermi level at each k
    instead of a fixed band list, so it integrates to the same Chern number as
    PythTB's berry_curvature over an isolated set. Pairs closer than
    ``min_gap`` across E_F are skipped (an exact degeneracy at the Fermi level
    has no finite Kubo term).
    """
    e = np.asarray(energies, float)
    occ = e < 0.0
    dE = e[..., :, None] - e[..., None, :]
    pair = occ[..., :, None] & ~occ[..., None, :] & (np.abs(dE) > min_gap)
    inv2 = np.where(pair, 1.0 / np.where(pair, dE, 1.0) ** 2, 0.0)
    term = v_a * np.swapaxes(v_b, -1, -2) * inv2
    return -2.0 * np.imag(term.sum(axis=(-1, -2)))


def rotate_velocity(v: np.ndarray, eigvecs: np.ndarray) -> np.ndarray:
    """Orbital-basis velocity (d, Nk, N, N) -> eigenbasis, PythTB's row-eigenvector convention."""
    return np.matmul(
        eigvecs.conj()[None], np.matmul(v, np.swapaxes(eigvecs, -1, -2)[None])
    )


# ---------------------------------------------------------------------------
# Slabs and ribbons
# ---------------------------------------------------------------------------


def layer_blocks(
    R: np.ndarray,
    H: np.ndarray,
    finite_axis: int,
    periodic_axes: Sequence[int],
    k_par: np.ndarray,
) -> Dict[int, np.ndarray]:
    """B_d(k_par) = sum over R with R_f = d of H(R) exp(2 pi i k_par . R_par).

    Hoppings along an axis that is neither finite nor periodic (a 2D model's
    vacuum) are dropped: they are absent by construction for such models.
    """
    R = np.asarray(R, int)
    others = [ax for ax in range(3) if ax != finite_axis and ax not in periodic_axes]
    blocks: Dict[int, np.ndarray] = {}
    k_par = np.asarray(k_par, float)
    for r, h in zip(R, H):
        if any(r[ax] != 0 for ax in others):
            continue
        phase = np.exp(2j * np.pi * sum(k_par[ax] * r[ax] for ax in periodic_axes))
        d = int(r[finite_axis])
        if d in blocks:
            blocks[d] = blocks[d] + phase * h
        else:
            blocks[d] = phase * h
    return blocks


def slab_hamiltonian(blocks: Dict[int, np.ndarray], n_cells: int) -> np.ndarray:
    """Block-Toeplitz H of an open slab of ``n_cells`` cells: H[w, w+d] = B_d."""
    n = next(iter(blocks.values())).shape[0]
    big = np.zeros((n_cells * n, n_cells * n), dtype=complex)
    for d, block in blocks.items():
        for w in range(n_cells):
            w2 = w + d
            if 0 <= w2 < n_cells:
                big[w * n : (w + 1) * n, w2 * n : (w2 + 1) * n] += block
    return big


def edge_weights(
    eigvecs: np.ndarray,
    n_cells: int,
    n_orb: int,
    edge_cells: int,
    energies: Optional[np.ndarray] = None,
    degenerate_tol: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Weight of each eigenvector (columns) on the first and last ``edge_cells`` cells.

    Degenerate states (the top and bottom surfaces of a symmetric slab, Kramers
    pairs) come out of the eigensolver as arbitrary mixtures. With ``energies``
    each degenerate cluster is first rotated to diagonalise the first-edge
    projector, which separates the two edges wherever the subspace allows.
    """
    edge = max(1, min(int(edge_cells), n_cells // 2))
    vecs = np.array(eigvecs, dtype=complex, copy=True)
    if energies is not None and len(energies) > 1:
        mask = np.zeros(n_cells * n_orb)
        mask[: edge * n_orb] = 1.0
        order = np.argsort(energies)
        start = 0
        while start < len(order):
            stop = start + 1
            while (
                stop < len(order)
                and energies[order[stop]] - energies[order[stop - 1]] < degenerate_tol
            ):
                stop += 1
            if stop - start > 1:
                idx = order[start:stop]
                sub = vecs[:, idx]
                proj = sub.conj().T @ (mask[:, None] * sub)
                _, rot = np.linalg.eigh(proj)
                vecs[:, idx] = sub @ rot
            start = stop
    prob = np.abs(vecs) ** 2
    cells = prob.reshape(n_cells, n_orb, -1).sum(axis=1)
    return cells[:edge].sum(axis=0), cells[-edge:].sum(axis=0)


def edge_crossings(
    energies: Sequence[np.ndarray],
    left: Sequence[np.ndarray],
    right: Sequence[np.ndarray],
    *,
    periodic: bool = False,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Count edge-localised states crossing E_F between neighbouring k-points.

    Edge states are followed edge by edge: a state localised on one edge at k_j
    is matched to the nearest-energy state localised on the SAME edge at
    k_j+1, so two chiral modes crossing each other at E_F (one per edge) are not
    swapped. A sign change of the energy is a crossing, its direction the sign
    of the step. ``periodic`` also compares the last k-point with the first (a
    ribbon's 1D zone). Informational: the bulk invariant stays the authority.
    """
    tally = {"left": {"up": 0, "down": 0}, "right": {"up": 0, "down": 0}}
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


# ---------------------------------------------------------------------------
# Species of the Wannier functions
# ---------------------------------------------------------------------------


def orbital_species(
    centres_cart: np.ndarray,
    structure: Optional[Dict[str, Any]],
    lattice: Sequence[Sequence[float]],
) -> Tuple[List[str], np.ndarray]:
    """Nearest atom's species for every Wannier centre (minimum image).

    Returns ``(species_names, index_per_orbital)``. Without a usable structure
    every orbital falls into one group, "all".
    """
    n = len(centres_cart)
    try:
        species = [str(s) for s in structure["species"]]
        positions = np.asarray(structure["cart"], float).reshape(-1, 3)
        if len(species) != len(positions) or not len(species):
            raise ValueError
    except (TypeError, KeyError, ValueError):
        return ["all"], np.zeros(n, dtype=int)
    lat = np.asarray(lattice, float)
    shifts = (
        np.stack(
            np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij"), -1
        ).reshape(-1, 3)
        @ lat
    )
    images = positions[None, :, :] + shifts[:, None, :]
    names = sorted(set(species), key=species.index)
    index = np.zeros(n, dtype=int)
    for o, c in enumerate(np.asarray(centres_cart, float)):
        dist = np.linalg.norm(images - c[None, None, :], axis=-1).min(axis=0)
        index[o] = names.index(species[int(np.argmin(dist))])
    return names, index


# ---------------------------------------------------------------------------
# Localisation: how hoppings decay with distance
# ---------------------------------------------------------------------------


def unwrap_centres(
    centres_cart: np.ndarray,
    lattice: Sequence[Sequence[float]],
    axes: Sequence[int],
) -> np.ndarray:
    """The centres moved by whole lattice vectors into one contiguous run along ``axes``.

    wannier90 wraps the centres it writes. Along a slab's vacuum axis, sampled
    by one k-point, they land in (-c/2, c/2], so the wrap line can run through
    the slab itself: the 5-SL MnBi2Te4 film (z 8 to 73 A in an 81 A cell) is
    cut through its middle septuple layer, and two neighbours across the cut
    sit a whole cell apart. Per axis the circle of fractional coordinates is
    cut at the middle of the largest gap between centres -- the vacuum -- and
    every centre is moved into the unit interval that starts there (taken
    nearest the home cell).

    For real-space diagnostics only: the Wigner-Seitz recompute and the model
    use the centres exactly as written (``ws_unfold``).
    """
    lat = np.asarray(lattice, float)
    cart = np.asarray(centres_cart, float).reshape(-1, 3)
    if not len(cart) or not len(axes):
        return cart.copy()
    frac = cart @ np.linalg.inv(lat)
    for axis in axes:
        cut, _ = _largest_gap_midpoint(frac[:, axis])
        if cut > 0.5:
            cut -= 1.0
        frac[:, axis] -= np.floor(frac[:, axis] - cut)
    return frac @ lat


def hopping_decay(
    R: np.ndarray,
    H: np.ndarray,
    centres_cart: np.ndarray,
    lattice: Sequence[Sequence[float]],
    bin_width: float = 0.5,
) -> Optional[Dict[str, Any]]:
    """Every hopping's |H| against the distance |R + tau_j - tau_i| it spans.

    Returns the nonzero hops (``distance``, ``amplitude``), the largest
    amplitude per ``bin_width`` bin (``bin_centres``, ``bin_peaks``), the decay
    length of an exponential fitted to the peaks beyond 2 A (None when they do
    not decay) and the longest hop; None for an all-zero table.
    """
    taus = np.asarray(centres_cart, float).reshape(-1, 3)
    dist = np.linalg.norm(
        (np.asarray(R, float) @ np.asarray(lattice, float))[:, None, None, :]
        + taus[None, None, :, :]
        - taus[None, :, None, :],
        axis=-1,
    )
    d = dist.ravel()
    a = np.abs(np.asarray(H)).ravel()
    keep = a > 0
    d, a = d[keep], a[keep]
    if not len(d):
        return None
    bins = np.arange(0.0, d.max() + bin_width, bin_width)
    idx = np.digitize(d, bins)
    centers, peak = [], []
    for b in np.unique(idx):
        sel = idx == b
        centers.append(float(d[sel].mean()))
        peak.append(float(a[sel].max()))
    centers = np.asarray(centers)
    peak = np.asarray(peak)
    fit = (centers > 2.0) & (peak > 1e-8)
    decay = None
    if fit.sum() >= 3:
        slope, _ = np.polyfit(centers[fit], np.log(peak[fit]), 1)
        decay = float(-1.0 / slope) if slope < 0 else None
    return {
        "distance": d,
        "amplitude": a,
        "bin_centres": centers,
        "bin_peaks": peak,
        "decay_length_A": decay,
        "max_distance_A": float(d.max()),
    }


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def to_jsonable(value: Any, digits: Optional[int] = None) -> Any:
    """numpy -> plain Python, NaN/inf -> None, optional float rounding."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v, digits) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist(), digits)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return round(number, digits) if digits is not None else number
    if isinstance(value, complex):
        return [to_jsonable(value.real, digits), to_jsonable(value.imag, digits)]
    return value


def write_json_atomic(path: str, payload: Any) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as handle:
        json.dump(to_jsonable(payload), handle, indent=1, allow_nan=False)
    os.replace(tmp, path)


def sha256_file(path: str) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Cluster driver: everything below needs pythtb (imported lazily)
# ---------------------------------------------------------------------------

# Read by forked pool workers: set before the pool exists, never mutated after.
_STATE: Dict[str, Any] = {}


class _Log:
    def __init__(self, path: str):
        self.path = path
        self.started = time.time()

    def __call__(self, message: str) -> None:
        line = f"[{time.time() - self.started:8.1f}s] {message}"
        print(line, flush=True)
        try:
            with open(self.path, "a") as handle:
                handle.write(line + "\n")
        except OSError:
            pass


def _available_cpus() -> int:
    for key in ("SLURM_CPUS_ON_NODE", "SLURM_CPUS_PER_TASK"):
        value = os.environ.get(key)
        if value and value.split("(")[0].isdigit():
            return max(1, int(value.split("(")[0]))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _cgroup_memory_room(
    proc_cgroup: str = "/proc/self/cgroup", root: str = "/sys/fs/cgroup"
) -> Optional[int]:
    """Bytes left under the tightest memory limit of this process's cgroups.

    Slurm with cgroup memory enforcement caps a job well below the node's
    MemAvailable, and a v2 limit can sit on any ancestor (job, step, task), so
    every level counts. None when no level has a limit.
    """
    try:
        with open(proc_cgroup) as handle:
            entries = [line.rstrip("\n").split(":", 2) for line in handle]
    except OSError:
        return None
    rooms: List[int] = []
    for entry in entries:
        if len(entry) != 3:
            continue
        hierarchy, controllers, path = entry
        if hierarchy == "0" and not controllers:
            base, limit_name, usage_name = root, "memory.max", "memory.current"
        elif "memory" in controllers.split(","):
            base = os.path.join(root, "memory")
            limit_name, usage_name = "memory.limit_in_bytes", "memory.usage_in_bytes"
        else:
            continue
        parts = [part for part in path.split("/") if part]
        for depth in range(len(parts), -1, -1):
            directory = os.path.join(base, *parts[:depth])
            try:
                with open(os.path.join(directory, limit_name)) as handle:
                    raw = handle.read().strip()
                with open(os.path.join(directory, usage_name)) as handle:
                    usage = int(handle.read().strip())
                limit = None if raw == "max" else int(raw)
            except (OSError, ValueError):
                continue
            if limit is None or limit >= 1 << 60:  # v2 "max", v1 "unlimited"
                continue
            rooms.append(max(0, limit - usage))
    return min(rooms) if rooms else None


def _available_memory_bytes() -> int:
    """MemAvailable, or less when this process's cgroup allows less."""
    found: List[int] = []
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    found.append(int(line.split()[1]) * 1024)
                    break
    except (OSError, ValueError):
        pass
    room = _cgroup_memory_room()
    if room is not None:
        found.append(room)
    return min(found) if found else 8 * 1024**3


def _peak_rss() -> Dict[str, int]:
    """Peak resident memory of the driver and of its largest worker (Linux KiB)."""
    try:
        import resource

        own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        workers = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
    except (ImportError, OSError, ValueError):
        return {}
    return {"peak_rss_bytes": int(own), "peak_worker_rss_bytes": int(workers)}


def plan_parallelism(
    n_orb: int,
    per_k_matrices: int,
    workers: Optional[int],
    chunk: Optional[int],
    *,
    fixed_bytes: float = 0.0,
    n_items: Optional[int] = None,
) -> Tuple[int, int]:
    """(workers, k-points per chunk) so that every worker's arrays fit in memory.

    ``per_k_matrices`` is how many N x N complex matrices one k-point keeps alive
    at peak (H(k), eigenvectors, velocities, ...); ``fixed_bytes`` is what every
    worker allocates whatever its chunk (PythTB rebuilds its per-R hopping
    matrices on each Hamiltonian call). A requested worker count is lowered
    when the workers would not fit, a requested chunk size kept.

    ``n_items``, the k-points the stage solves, caps an automatic chunk at their
    even share, ceil(n_items / workers): the largest chunk that fits alone would
    hand 22500 DOS points to 18 of 64 workers, and a 201-point path to one.
    """
    per_k = 16.0 * n_orb * n_orb * max(1, per_k_matrices)
    budget = 0.6 * _available_memory_bytes()
    workers = max(1, int(workers if workers is not None else _available_cpus()))
    if chunk is None:
        per_worker = min(budget / workers - fixed_bytes, 4.0 * 1024**3)
        chunk = int(max(4, min(4096, per_worker // per_k)))
        if n_items:
            chunk = min(chunk, math.ceil(int(n_items) / workers))
    chunk = max(1, int(chunk))
    while workers > 1 and workers * (chunk * per_k + fixed_bytes) > budget:
        workers -= 1
    return workers, chunk


def pool_processes(workers: int, n_items: int) -> int:
    """How many processes ``_pmap`` maps ``n_items`` over: 1 runs them in the
    driver itself."""
    return min(workers, n_items) if workers > 1 and n_items > 1 else 1


def _pmap(func: Callable[[Any], Any], items: Sequence[Any], workers: int) -> List[Any]:
    """Ordered map over a fork pool (workers read the model from ``_STATE``)."""
    items = list(items)
    processes = pool_processes(workers, len(items))
    if processes == 1:
        return [func(item) for item in items]
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=processes) as pool:
        return pool.map(func, items, chunksize=1)


def _chunks(points: np.ndarray, size: int) -> List[np.ndarray]:
    return [points[i : i + size] for i in range(0, len(points), max(1, size))]


def _solve(kpts: np.ndarray, vectors: bool = False):
    """PythTB solve_ham on a chunk, keeping the k axis for a single point."""
    model = _STATE["model"]
    kpts = np.asarray(kpts, float).reshape(-1, 3)
    if vectors:
        evals, evecs = model.solve_ham(kpts, return_eigvecs=True)
        evals = np.asarray(evals, float).reshape(len(kpts), -1)
        evecs = np.asarray(evecs, complex).reshape(len(kpts), evals.shape[1], -1)
        return evals, evecs
    return np.asarray(model.solve_ham(kpts), float).reshape(len(kpts), -1)


def _w_edges(kpts: np.ndarray) -> Tuple[BandEdges, np.ndarray]:
    energies = _solve(kpts)
    edges = BandEdges(energies.shape[1])
    edges.add(kpts, energies)
    return edges, energies


def _w_dos(kpts: np.ndarray):
    """Per-chunk eigenvalue histograms (total and per species) plus band edges."""
    grid = _STATE["dos_grid"]
    species = _STATE["species_index"]
    n_species = int(species.max()) + 1 if len(species) else 1
    if _STATE.get("pdos"):
        energies, vecs = _solve(kpts, vectors=True)
        weights = np.abs(vecs) ** 2  # (k, band, orbital)
        per_species = np.stack(
            [weights[:, :, species == s].sum(axis=-1) for s in range(n_species)],
            axis=-1,
        )
    else:
        energies = _solve(kpts)
        per_species = None
    edges = BandEdges(energies.shape[1])
    edges.add(kpts, energies)
    total, _ = np.histogram(energies.ravel(), bins=grid)
    parts = None
    if per_species is not None:
        parts = np.stack(
            [
                np.histogram(
                    energies.ravel(), bins=grid, weights=per_species[..., s].ravel()
                )[0]
                for s in range(n_species)
            ]
        )
    outside = int(np.sum((energies < grid[0]) | (energies >= grid[-1])))
    return edges, total, parts, outside


def _w_path_weights(kpts: np.ndarray):
    energies, vecs = _solve(kpts, vectors=True)
    species = _STATE["species_index"]
    n_species = int(species.max()) + 1 if len(species) else 1
    weights = np.abs(vecs) ** 2
    per_species = np.stack(
        [weights[:, :, species == s].sum(axis=-1) for s in range(n_species)], axis=-1
    )
    return energies, per_species


def _w_wilson(task: Tuple[np.ndarray, int]) -> Tuple[float, np.ndarray]:
    loop_k, n_occ = task
    _, vecs = _solve(loop_k, vectors=True)
    occ = [vecs[j, :n_occ, :] for j in range(len(loop_k))]
    return wilson_loop(occ, _STATE["boundary_phase"])


def _w_berry(kpts: np.ndarray):
    """Kubo curvature (and metric) of a chunk: PythTB for a band set, own sum for E_F."""
    model = _STATE["model"]
    plane = _STATE["berry_plane"]
    mode = _STATE["berry_mode"]
    occ = _STATE["berry_occ"]
    out: Dict[str, Any] = {}
    if mode == "bands":
        omega = model.berry_curvature(kpts, occ_idxs=occ, plane=plane, cartesian=False)
        out["omega"] = np.asarray(omega, float).reshape(len(kpts))
        if _STATE.get("berry_metric"):
            g = model.quantum_metric(kpts, occ_idxs=occ, plane=None, cartesian=True)
            g = np.real(np.asarray(g)).reshape(3, 3, len(kpts))
            normal = _STATE["plane_normal"]
            trace = np.einsum("aak->k", g) - np.einsum("a,abk,b->k", normal, g, normal)
            out["metric_trace"] = trace
    else:
        energies, vecs = _solve(kpts, vectors=True)
        v = np.asarray(model.velocity(kpts, cartesian=False), complex).reshape(
            3, len(kpts), energies.shape[1], energies.shape[1]
        )
        a, b = plane
        v_rot = rotate_velocity(v[[a, b]], vecs)
        out["omega"] = fermi_sea_curvature(v_rot[0], v_rot[1], energies)
    return out


def _w_slab(k_par: np.ndarray):
    from scipy.linalg import eigh

    blocks = layer_blocks(
        _STATE["table_R"],
        _STATE["table_H"],
        _STATE["slab_finite"],
        _STATE["slab_periodic"],
        k_par,
    )
    big = slab_hamiltonian(blocks, _STATE["slab_cells"])
    window = _STATE["slab_window"]
    evals, evecs = eigh(big, subset_by_value=(-window, window), driver="evr")
    left, right = edge_weights(
        evecs,
        _STATE["slab_cells"],
        _STATE["n_orb"],
        _STATE["edge_cells"],
        energies=evals,
    )
    return evals, left, right


class Context:
    """One PythTB run: the model, its provenance, and the results being built."""

    def __init__(self, params: Dict[str, Any], log: _Log):
        self.params = merged_params(params)
        self.log = log
        self.step = str(self.params.get("step_type") or "").upper()
        if self.step not in STEP_TYPES:
            raise PythtbError(ERR_BAD_PARAMS, f"unknown step_type {self.step!r}")
        self.results: Dict[str, Any] = {}
        self.warnings: List[str] = []
        self.outputs: List[Dict[str, str]] = []
        # For provenance: the model files read (size, sha256) and the parallel
        # plan each stage ran with.
        self.inputs: Dict[str, Dict[str, Any]] = {}
        self.parallel_plan: List[Dict[str, Any]] = []

    def _record_input(self, name: str) -> None:
        """Size and sha256 of a staged model file, taken right after reading it.

        Streamed from disk (``sha256_file``), so a 400 MB ``_hr.dat`` is never
        held twice. These are the bytes this run used: the producer's stored
        copies need not match them (a WANNIER job run twice in one directory).
        """
        try:
            size = os.path.getsize(name)
        except OSError:
            return
        self.inputs[name] = {"bytes": int(size), "sha256": sha256_file(name)}

    # ------------------------------------------------------------ model
    def build_model(self) -> None:
        import pythtb
        from pythtb import W90

        for name in REQUIRED_INPUTS:
            if not os.path.exists(name):
                raise PythtbError(
                    ERR_MISSING_INPUT,
                    f"{name} was not staged; bind it from the WANNIER job (write_hr and "
                    "write_xyz must be on there)",
                )
        with open(WIN_FILENAME) as handle:
            self.win = parse_win(handle.read())
        self._record_input(WIN_FILENAME)
        if self.win.get("translate_home_cell"):
            raise PythtbError(
                ERR_TRANSLATE_HOME_CELL,
                "The WANNIER run set translate_home_cell = true: its centres no longer "
                "match the hopping vectors in _hr.dat. Re-run it with "
                "translate_home_cell = false.",
            )
        if not self.win.get("lattice"):
            raise PythtbError(
                ERR_MISSING_INPUT, f"{WIN_FILENAME} has no unit_cell_cart block"
            )
        self.fermi = self.params.get("fermi_energy")
        if self.fermi is None:
            raise PythtbError(
                ERR_BAD_PARAMS,
                "No Fermi energy: set fermi_energy (eV) on the step",
            )
        self.fermi = float(self.fermi)

        shutil.rmtree(MODEL_DIR, ignore_errors=True)
        os.makedirs(MODEL_DIR)
        with open(os.path.join(MODEL_DIR, WIN_FILENAME), "w") as handle:
            handle.write(sanitized_win(self.win))
        for name in (
            HR_FILENAME,
            CENTRES_FILENAME,
            BAND_KPT_FILENAME,
            BAND_DAT_FILENAME,
        ):
            if os.path.exists(name):
                os.symlink(os.path.abspath(name), os.path.join(MODEL_DIR, name))

        t0 = time.time()
        self.w90 = W90(MODEL_DIR, SEEDNAME)
        self._record_input(HR_FILENAME)
        self._record_input(CENTRES_FILENAME)
        self.lattice = np.asarray(self.w90.lat, float)
        self.centres = np.asarray(self.w90.xyz_cen, float)
        self.n_orb = int(self.w90.num_wan)
        R_raw, H_raw = table_from_ham_r(self.w90.ham_r)
        # wannier90's own R set and degeneracies, before any correction: a
        # _tb.dat from the same run must carry exactly these.
        self.hr_degeneracy = {
            tuple(int(x) for x in key): int(value["deg"])
            for key, value in self.w90.ham_r.items()
        }
        self.log(
            f"read {HR_FILENAME}: {self.n_orb} Wannier functions, {len(R_raw)} R vectors "
            f"({time.time() - t0:.1f}s)"
        )
        self.reference = self._read_reference()
        R, H, ws_report = self._ws_correct(R_raw, H_raw)
        self.ws_report = ws_report
        n_r_after_ws = len(R)
        # Periodicity from the RAW table (see periodic_dirs_from_table), and a
        # 2D model's hops folded back off its vacuum axis, so PythTB gets the
        # 2D model wannier90 interpolated on k_c = 0.
        periodic = self.params.get("periodic_dirs")
        self.periodic = (
            sorted(int(x) for x in periodic)
            if periodic
            else periodic_dirs_from_table(R_raw)
        )
        if len(self.periodic) < 2:
            raise PythtbError(
                ERR_BAD_PARAMS,
                f"the model is periodic along {self.periodic} only; 2D or 3D models are supported",
            )
        folded = [axis for axis in range(3) if axis not in self.periodic]
        R, H = fold_axes(R, H, folded)
        self.folded_axes = folded
        self.table_R, self.table_H = R, H
        self.w90.ham_r = table_to_ham_r(R, H)
        self.w90._dist_cache = {}
        self.w90._vecR_cache = {}
        model = self.w90.model(
            zero_energy=self.fermi,
            min_hopping_norm=self.params.get("min_hopping_norm"),
            max_distance=self.params.get("max_distance"),
            onsite_imag_tol=1e-6,
        )
        _STATE["model"] = model
        self.model = model
        self.recip = reciprocal_lattice(self.lattice)
        self.orb_red = np.asarray(model.get_orb_vecs(), float).reshape(self.n_orb, 3)
        spinors = self.win.get("spinors")
        self.spinors = bool(
            spinors if spinors is not None else self.params.get("spinors")
        )
        self.species_names, self.species_index = orbital_species(
            self.centres, self.params.get("structure"), self.lattice
        )
        _STATE["species_index"] = self.species_index
        hops = getattr(model, "nhops", None)
        self.results["model"] = {
            "num_wann": self.n_orb,
            "n_R": int(len(R_raw)),
            "n_R_after_ws": int(n_r_after_ws),
            "n_R_model": int(len(R)),
            "folded_axes": folded,
            "n_hops": int(hops) if isinstance(hops, (int, np.integer)) else None,
            "spinors": self.spinors,
            "periodic_dirs": self.periodic,
            "dimension": len(self.periodic),
            "lattice": self.lattice.tolist(),
            "fermi_energy": self.fermi,
            "fermi_source": self.params.get("fermi_source"),
            "ws_correction": ws_report,
            "cutoffs": {
                "min_hopping_norm": self.params.get("min_hopping_norm"),
                "max_distance": self.params.get("max_distance"),
            },
            "species": self.species_names,
            "orbital_species": [self.species_names[i] for i in self.species_index],
            "mp_grid": self.win.get("mp_grid"),
            "win_scalars": {
                key: self.win[key]
                for key in WIN_WINDOW_KEYS
                if self.win.get(key) is not None
            },
            "approximation": APPROXIMATION_NOTE,
            "pythtb_version": _package_version("pythtb"),
            "source_wannier_job_id": self.params.get("source_wannier_job_id"),
            "reference_band_job_id": self.params.get("reference_band_job_id"),
        }
        self.log(
            f"model: N={self.n_orb}, periodic {self.periodic}, spinors={self.spinors}, "
            f"E_F={self.fermi:.4f} eV ({self.params.get('fermi_source')}), "
            f"WS={ws_report.get('mode')} ({time.time() - t0:.1f}s)"
        )
        del pythtb

    def _read_reference(self) -> Optional[Dict[str, Any]]:
        """wannier90's own interpolated bands, when both band files were staged."""
        if not (
            os.path.exists(BAND_KPT_FILENAME) and os.path.exists(BAND_DAT_FILENAME)
        ):
            return None
        with open(BAND_KPT_FILENAME) as handle:
            kpts = parse_band_kpt(handle.read())
        self._record_input(BAND_KPT_FILENAME)
        with open(BAND_DAT_FILENAME) as handle:
            energies = parse_band_dat(handle.read(), self.n_orb)
        self._record_input(BAND_DAT_FILENAME)
        n = min(len(kpts), len(energies))
        if n == 0:
            return None
        ticks, labels = [], []
        if os.path.exists(LABELINFO_FILENAME):
            with open(LABELINFO_FILENAME) as handle:
                ticks, labels = parse_labelinfo(handle.read())
            self._record_input(LABELINFO_FILENAME)
        return {
            "kpoints": kpts[:n],
            "energies": energies[:n] - self.fermi,
            "tick_indices": [t for t in ticks if 0 <= t < n],
            "tick_labels": [l for t, l in zip(ticks, labels) if 0 <= t < n],
        }

    def _frozen_window(self) -> Tuple[float, float]:
        return comparison_window(self.win, self.fermi)

    def _ws_correct(self, R: np.ndarray, H: np.ndarray):
        mode = str(self.params.get("ws_correction") or "auto").lower()
        report: Dict[str, Any] = {"requested": mode}
        if os.path.exists(WSVEC_FILENAME) and mode in ("auto", "file"):
            with open(WSVEC_FILENAME) as handle:
                table = ws_unfold_from_wsvec(R, H, parse_wsvec(handle.read()))
            self._record_input(WSVEC_FILENAME)
            report.update(mode="wsvec_file")
            report["hermiticity_error"] = hermiticity_error(*table)
            return table[0], table[1], report
        mp_grid = self.win.get("mp_grid")
        search = int(self.win.get("ws_search_size") or 2)
        tol = float(self.win.get("ws_distance_tol") or 1e-5)
        candidates: Dict[str, Callable[[], Tuple[np.ndarray, np.ndarray]]] = {
            "off": lambda: (R, H)
        }
        if mp_grid:
            candidates["recompute"] = lambda: ws_unfold(
                R, H, self.centres, self.lattice, mp_grid, search_size=search, tol=tol
            )
        elif mode != "off":
            self.warnings.append(
                "The .win has neither mp_grid nor a kpoints block: the Wigner-Seitz "
                "correction cannot be recomputed and is off."
            )
        if mode == "off" or len(candidates) == 1:
            report.update(mode="off")
            return R, H, report
        if mode == "recompute":
            table = candidates["recompute"]()
            report.update(mode="recompute")
            report["hermiticity_error"] = hermiticity_error(*table)
            return table[0], table[1], report

        # auto: score every candidate against wannier90's own interpolation.
        if self.reference is not None:
            window = self._frozen_window()
            scores = {}
            tables = {}
            for name, build in candidates.items():
                tables[name] = build()
                bands = (
                    eigvals_from_table(*tables[name], self.reference["kpoints"])
                    - self.fermi
                )
                scores[name] = band_deviation(bands, self.reference["energies"], window)
            best = min(
                scores,
                key=lambda n: (scores[n]["in_window_max_meV"] or 0.0, n != "recompute"),
            )
            report.update(
                mode=best,
                selected_by="band_dat",
                scores={
                    n: {
                        k: s[k]
                        for k in ("in_window_max_meV", "in_window_rms_meV", "max_meV")
                    }
                    for n, s in scores.items()
                },
            )
            table = tables[best]
        else:
            hinted = self.params.get("ws_hint")
            use = hinted is not False and self.win.get("use_ws_distance") is not False
            name = "recompute" if use else "off"
            table = candidates[name]()
            report.update(
                mode=name,
                selected_by="ws_hint" if hinted is not None else "wannier90_3_default",
            )
            if name != "off":
                self.warnings.append(
                    "No wannier90_band.dat to check the Wigner-Seitz correction against; "
                    "it was recomputed assuming wannier90 3.x defaults."
                )
        if report["mode"] != "off":
            report["hermiticity_error"] = hermiticity_error(*table)
        return table[0], table[1], report

    def ws_apply(
        self, R: np.ndarray, stack: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """The Wigner-Seitz unfold and the vacuum fold that built ``table_H``,
        applied to another table on wannier90's R set (trailing axes allowed).

        The image mapping depends only on (R, i, j), the centres and the
        lattice, so a table stacked with the position matrices unfolds exactly
        as the Hamiltonian did.
        """
        mode = (getattr(self, "ws_report", None) or {}).get("mode", "off")
        if mode == "wsvec_file":
            with open(WSVEC_FILENAME) as handle:
                R, stack = ws_unfold_from_wsvec(R, stack, parse_wsvec(handle.read()))
        elif mode == "recompute":
            R, stack = ws_unfold(
                R,
                stack,
                self.centres,
                self.lattice,
                self.win.get("mp_grid"),
                search_size=int(self.win.get("ws_search_size") or 2),
                tol=float(self.win.get("ws_distance_tol") or 1e-5),
            )
        return fold_axes(R, stack, self.folded_axes)

    # ------------------------------------------------------- validation
    def validate(self) -> None:
        out: Dict[str, Any] = {}
        if self.reference is not None:
            window = self._frozen_window()
            kpts = self.reference["kpoints"]
            workers, chunk = self.parallel("validation", 2, len(kpts))
            pythtb_bands = np.concatenate(_pmap(_solve, _chunks(kpts, chunk), workers))
            dev = band_deviation(pythtb_bands, self.reference["energies"], window)
            out["band_dat"] = dict(dev, available=True, n_k=int(len(pythtb_bands)))
            warn = self.params.get("validation_warn_meV")
            fail = self.params.get("validation_fail_meV")
            worst = dev.get("in_window_max_meV") or 0.0
            if fail is not None and worst > float(fail):
                self.results["validation"] = out
                raise PythtbError(
                    ERR_VALIDATION,
                    f"The PythTB model departs from wannier90_band.dat by {worst:.1f} meV "
                    f"inside the frozen window (limit {float(fail):.0f} meV): the staged "
                    "model files do not belong together or the import failed.",
                )
            if warn is not None and worst > float(warn):
                self.warnings.append(
                    f"PythTB bands differ from wannier90_band.dat by up to {worst:.1f} meV "
                    "inside the frozen window."
                )
        else:
            out["band_dat"] = {"available": False}
            self.warnings.append(
                "wannier90_band.dat was not staged (bands_plot off on the WANNIER run): "
                "the model could not be checked against wannier90's own interpolation."
            )
        lo = self.win.get("dis_froz_min")
        hi = self.win.get("dis_froz_max")
        if hi is not None:
            inside = (lo is None or lo <= self.fermi) and self.fermi <= hi
            out["fermi_in_frozen_window"] = bool(inside)
            if not inside:
                self.warnings.append(
                    "E_F lies outside the frozen window: bands near the Fermi level are "
                    "not guaranteed to match DFT."
                )
        out["hermiticity_error"] = hermiticity_error(self.table_R, self.table_H)
        out["localization"] = self._localization()
        self.results["validation"] = out

    def _localization(self) -> Dict[str, Any]:
        """Decay of the largest hopping with distance (well-localised: exponential).

        The folded table's hops lie in the layer, so they are measured between
        centres unwrapped along the folded axes (``unwrap_centres``): with the
        centres as written, neighbours across wannier90's wrap line would count
        as a cell apart.
        """
        centres = unwrap_centres(self.centres, self.lattice, self.folded_axes)
        decay = hopping_decay(self.table_R, self.table_H, centres, self.lattice)
        if decay is None:
            return {}
        self._plot_decay(
            decay["distance"],
            decay["amplitude"],
            decay["bin_centres"],
            decay["bin_peaks"],
        )
        return {
            "decay_length_A": decay["decay_length_A"],
            "max_distance_A": decay["max_distance_A"],
        }

    # ----------------------------------------------------------- helpers
    def parallel(
        self,
        stage: str,
        per_k_matrices: int,
        n_k: int,
        *,
        chunk: Optional[int] = None,
        n_orb: Optional[int] = None,
    ) -> Tuple[int, int]:
        """(workers, chunk) for a stage that solves ``n_k`` k-points; the plan,
        with the processes it really runs on, is logged and kept for
        ``provenance.parallel``.

        Each worker's H(k) call rebuilds PythTB's (n_R, N, N) hopping matrices
        and their index arrays: ~32 bytes per table entry. A stage that solves
        matrices of its own instead (a slab's) gives their size as ``n_orb``;
        one whose tasks cannot be split (a Wilson loop) gives its task size as
        ``chunk``.
        """
        if n_orb is None:
            size = self.n_orb
            fixed = 32.0 * len(self.table_R) * self.n_orb * self.n_orb
        else:
            size, fixed = int(n_orb), 0.0
        workers, chunk = plan_parallelism(
            size,
            per_k_matrices,
            self.params.get("workers"),
            chunk if chunk is not None else self.params.get("chunk_size"),
            fixed_bytes=fixed,
            n_items=n_k,
        )
        tasks = math.ceil(n_k / chunk)
        processes = pool_processes(workers, tasks)
        self.parallel_plan.append(
            {
                "stage": stage,
                "k_points": int(n_k),
                "chunk": int(chunk),
                "tasks": int(tasks),
                "workers": int(workers),
                "processes": int(processes),
            }
        )
        self.log(
            f"{stage}: {n_k} k-points, {tasks} task(s) of up to {chunk} k on "
            f"{processes} process(es)"
        )
        return workers, chunk

    def band_path(self) -> Dict[str, Any]:
        """The k-path: wannier90's own samples, else the resolved path, else the .win path."""
        n_points = int(self.params.get("kpath_points") or 400)
        if self.reference is not None and self.reference["tick_indices"]:
            ticks = self.reference["tick_indices"]
            return {
                "kpoints": self.reference["kpoints"],
                "tick_indices": ticks,
                "tick_labels": self.reference["tick_labels"],
                "break_indices": path_breaks(self.reference["kpoints"], ticks),
                "source": "wannier90_band_kpt",
            }
        kpath = self.params.get("kpath") or {}
        segments = kpath.get("segments") if isinstance(kpath, dict) else None
        source = (kpath.get("source") if isinstance(kpath, dict) else None) or "params"
        if not segments and self.win.get("kpoint_path"):
            segments, source = self.win["kpoint_path"], "win_kpoint_path"
        if segments and len(self.periodic) < 3:
            segments = project_path(segments, self.periodic)
        if not segments:
            raise PythtbError(
                ERR_BAD_PARAMS,
                "No k-path: the WANNIER run had no kpoint_path and none was resolved",
            )
        path = sample_path(segments, self.lattice, n_points)
        path["source"] = source
        return path

    def plane_points(
        self, fixed_axis: int, value: float, n: int, centred: bool = False
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """n x n fractional k-points on the plane k_fixed = ``value``, over
        [0, 1) on its two axes, or over [-0.5, 0.5) ``centred`` on Gamma."""
        a, b = plane_axes(fixed_axis)
        ka = np.arange(n) / n - (0.5 if centred else 0.0)
        kb = np.arange(n) / n - (0.5 if centred else 0.0)
        A, B = np.meshgrid(ka, kb, indexing="ij")
        pts = np.zeros((n * n, 3))
        pts[:, a] = A.ravel()
        pts[:, b] = B.ravel()
        pts[:, fixed_axis] = value
        return pts, (a, b)

    def default_plane(self) -> Tuple[int, float]:
        """2D: the plane of the layer; 3D: k_3 = 0."""
        missing = [ax for ax in range(3) if ax not in self.periodic]
        return (missing[0] if missing else 2), 0.0

    def resolve_occupation(
        self, extra_points: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """The occupied band set from a modest mesh plus the path (gap classification)."""
        if len(self.periodic) == 2:
            mesh = uniform_mesh([36, 36, 36], self.periodic)
        else:
            mesh = uniform_mesh([16, 16, 16], self.periodic)
        pts = mesh if extra_points is None else np.concatenate([mesh, extra_points])
        try:
            pts = np.concatenate([pts, self.band_path()["kpoints"]])
        except PythtbError:
            pass
        workers, chunk = self.parallel("occupation", 3, len(pts))
        edges = BandEdges(self.n_orb)
        for chunk_edges, _ in _pmap(_w_edges, _chunks(pts, chunk), workers):
            edges.merge(chunk_edges)
        info = classify_gap(
            edges,
            float(self.params.get("fermi_tolerance_eV") or 0.02),
            self.params.get("occupied_bands"),
        )
        nocc = info.get("n_occ")
        if nocc:
            isolation = float(edges.direct[nocc - 1]) * 1000.0
            info["isolation_meV"] = isolation
            info["isolated"] = bool(
                isolation >= float(self.params.get("isolation_threshold_meV") or 5.0)
            )
        return info

    def add_output(self, path: str, title: str, kind: str) -> None:
        self.outputs.append({"path": path, "title": title, "kind": kind})

    # ------------------------------------------------------------- plots
    def _plt(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt

    def _save(self, fig, name: str, title: str) -> None:
        path = os.path.join(OUT_DIR, name)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        self._plt().close(fig)
        self.add_output(path, title, "plot")

    def _plot_decay(self, d, a, centers, peak) -> None:
        try:
            plt = self._plt()
            fig, ax = plt.subplots(figsize=(6, 4))
            step = max(1, len(d) // 200000)
            ax.scatter(
                d[::step], a[::step], s=1, alpha=0.3, color="#888888", rasterized=True
            )
            ax.plot(
                centers, peak, color="#d55e00", lw=1.5, label="largest |H| per 0.5 Å"
            )
            ax.set_yscale("log")
            ax.set_xlabel("distance between Wannier centres (Å)")
            ax.set_ylabel("|H| (eV)")
            ax.set_title("Hopping decay")
            ax.legend(loc="upper right")
            self._save(fig, "hopping_decay.png", "Hopping decay with distance")
        except Exception as exc:  # plots never fail a run
            self.warnings.append(f"hopping-decay plot skipped: {exc}")

    def _ticks(
        self, path: Dict[str, Any], dist: np.ndarray
    ) -> Tuple[List[float], List[str]]:
        pos, lab = [], []
        for t, l in zip(path["tick_indices"], path["tick_labels"]):
            x = float(dist[min(t, len(dist) - 1)])
            label = "Γ" if l.upper() in ("G", "GAMMA", "\\GAMMA") else l
            if pos and abs(pos[-1] - x) < 1e-9:
                if lab[-1] != label:
                    lab[-1] = f"{lab[-1]}|{label}"
                continue
            pos.append(x)
            lab.append(label)
        return pos, lab

    def _energy_window(self) -> Tuple[float, float]:
        w = self.params.get("energy_window_eV") or [-3.0, 3.0]
        return float(w[0]), float(w[1])


def _package_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def run_bands(ctx: Context) -> None:
    """Bands on the path (weighted by species), DOS/PDOS, gap, Fermi surface."""
    params = ctx.params
    path = ctx.band_path()
    workers, chunk = ctx.parallel("band path", 4, len(path["kpoints"]))
    parts = _pmap(_w_path_weights, _chunks(path["kpoints"], chunk), workers)
    energies = np.concatenate([p[0] for p in parts])
    weights = np.concatenate([p[1] for p in parts])
    dist = k_distance(path["kpoints"], ctx.lattice, path["break_indices"])

    # DOS on a uniform mesh, histogrammed in the workers.
    n = ctx.n_orb
    mesh_div = params.get("dos_mesh")
    if not mesh_div:
        if len(ctx.periodic) == 2:
            m = int(round(np.interp(n, [30, 150], [300, 150])))
        else:
            m = int(round(np.interp(n, [20, 150], [60, 24])))
        mesh_div = [m, m, m]
    elif isinstance(mesh_div, (int, float)):
        mesh_div = [int(mesh_div)] * 3
    mesh = uniform_mesh(mesh_div, ctx.periodic)
    step = float(params.get("dos_step_eV") or 0.005)
    sigma = float(params.get("dos_sigma_eV") or 0.03)
    lo = float(energies.min()) - 2.0
    hi = float(energies.max()) + 2.0
    grid = np.arange(lo, hi + step, step)
    _STATE["dos_grid"] = grid
    _STATE["pdos"] = bool(params.get("pdos")) and len(ctx.species_names) > 1
    workers, chunk = ctx.parallel("DOS", 4 if _STATE["pdos"] else 2, len(mesh))
    edges = BandEdges(n)
    total = np.zeros(len(grid) - 1)
    species_hist = None
    outside = 0
    for chunk_edges, hist, parts_hist, out_of_range in _pmap(
        _w_dos, _chunks(mesh, chunk), workers
    ):
        edges.merge(chunk_edges)
        total += hist
        outside += out_of_range
        if parts_hist is not None:
            species_hist = (
                parts_hist if species_hist is None else species_hist + parts_hist
            )
    path_edges = BandEdges(n)
    path_edges.add(path["kpoints"], energies)
    edges.merge(path_edges)
    norm = float(len(mesh))
    centers = 0.5 * (grid[1:] + grid[:-1])
    dos = gaussian_dos(total / norm, step, sigma)
    pdos = (
        gaussian_dos(species_hist / norm, step, sigma)
        if species_hist is not None
        else None
    )
    gap = classify_gap(
        edges,
        float(params.get("fermi_tolerance_eV") or 0.02),
        params.get("occupied_bands"),
    )
    # Interpolated to E_F itself: no bin centre need sit on it.
    gap["dos_at_fermi_per_eV"] = float(np.interp(0.0, centers, dos))
    if outside:
        ctx.warnings.append(f"{outside} mesh eigenvalues fell outside the DOS grid")

    ctx.results["electronic"] = gap
    ctx.results["bands"] = {
        "kpoints_frac": path["kpoints"],
        "eigenvalues_eV": energies + ctx.fermi,
        "fermi_energy": ctx.fermi,
        "tick_indices": path["tick_indices"],
        "tick_labels": path["tick_labels"],
        "break_indices": path["break_indices"],
        "source": path["source"],
    }
    ctx.results["dos"] = {
        # The divisions sampled along each periodic axis (``mesh_axes``): two
        # for a 2D model, whose third axis is never sampled.
        "mesh": [int(len(np.unique(mesh[:, axis]))) for axis in ctx.periodic],
        "mesh_axes": list(ctx.periodic),
        "k_points": int(len(mesh)),
        "sigma_eV": sigma,
        "integral": float(dos.sum() * step),
        "species": ctx.species_names if pdos is not None else [],
        "file": os.path.join(OUT_DIR, "dos.json"),
    }
    write_json_atomic(
        os.path.join(OUT_DIR, "dos.json"),
        {
            "energy_eV": centers,
            "dos_per_eV": dos,
            "pdos_per_eV": (
                {name: pdos[i] for i, name in enumerate(ctx.species_names)}
                if pdos is not None
                else {}
            ),
        },
    )
    ctx.add_output(
        os.path.join(OUT_DIR, "dos.json"), "DOS and species PDOS (data)", "data"
    )

    _plot_bands(ctx, path, dist, energies, weights)
    _plot_dos(ctx, centers, dos, pdos)

    if gap.get("metal") and params.get("fermi_surface"):
        _fermi_surface(ctx)


def _plot_bands(ctx: Context, path, dist, energies, weights) -> None:
    try:
        plt = ctx._plt()
        lo, hi = ctx._energy_window()
        fig, ax = plt.subplots(figsize=(7.5, 5))
        if ctx.reference is not None and path["source"] == "wannier90_band_kpt":
            for b in range(ctx.reference["energies"].shape[1]):
                ax.plot(
                    dist,
                    ctx.reference["energies"][:, b],
                    color="#999999",
                    lw=2.5,
                    alpha=0.6,
                    zorder=1,
                )
            ax.plot([], [], color="#999999", lw=2.5, alpha=0.6, label="wannier90")
        palette = [
            "#0072b2",
            "#d55e00",
            "#009e73",
            "#cc79a7",
            "#e69f00",
            "#56b4e9",
            "#f0e442",
            "#000000",
        ]
        dominant = weights.argmax(axis=-1)
        for b in range(energies.shape[1]):
            ax.plot(dist, energies[:, b], color="#222222", lw=0.6, zorder=2)
        if len(ctx.species_names) > 1:
            for s, name in enumerate(ctx.species_names):
                sel = dominant == s
                kk = np.broadcast_to(dist[:, None], energies.shape)[sel]
                ax.scatter(
                    kk,
                    energies[sel],
                    s=4,
                    color=palette[s % len(palette)],
                    label=name,
                    zorder=3,
                )
        ax.plot([], [], color="#222222", lw=0.6, label="PythTB")
        pos, lab = ctx._ticks(path, dist)
        ax.set_xticks(pos)
        ax.set_xticklabels(lab)
        for x in pos:
            ax.axvline(x, color="#cccccc", lw=0.5, zorder=0)
        ax.axhline(0.0, color="#666666", lw=0.6, ls="--")
        ax.set_xlim(dist[0], dist[-1])
        ax.set_ylim(lo, hi)
        ax.set_ylabel("E - E_F (eV)")
        ax.set_title("Wannier tight-binding bands (colour: dominant species)")
        ax.legend(loc="upper right", fontsize=8, markerscale=2)
        ctx._save(fig, "bands.png", "Band structure")
    except Exception as exc:
        ctx.warnings.append(f"band plot skipped: {exc}")


def _plot_dos(ctx: Context, centers, dos, pdos) -> None:
    try:
        plt = ctx._plt()
        lo, hi = ctx._energy_window()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(centers, dos, color="#222222", lw=1.2, label="total")
        palette = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9"]
        if pdos is not None:
            for s, name in enumerate(ctx.species_names):
                ax.plot(
                    centers,
                    pdos[s],
                    lw=1.0,
                    color=palette[s % len(palette)],
                    label=name,
                )
        ax.axvline(0.0, color="#666666", lw=0.6, ls="--")
        ax.set_xlim(lo, hi)
        sel = (centers >= lo) & (centers <= hi)
        top = float(dos[sel].max()) if sel.any() else float(dos.max())
        ax.set_ylim(0, top * 1.1 if top > 0 else 1)
        ax.set_xlabel("E - E_F (eV)")
        ax.set_ylabel("states / eV / cell")
        ax.set_title("Density of states")
        ax.legend(fontsize=8)
        ctx._save(fig, "dos.png", "Density of states")
    except Exception as exc:
        ctx.warnings.append(f"DOS plot skipped: {exc}")


def _fermi_surface(ctx: Context) -> None:
    """Fermi-level contours on the default plane (the layer for 2D models).

    Sampled on the Gamma-centred cell, so a pocket around Gamma is one closed
    contour rather than four corner pieces, and drawn in the band plot's k
    units (1/A without 2 pi), so a pocket's radius is the k at which its band
    crosses E_F there.
    """
    fixed, value = ctx.default_plane()
    n = ctx.params.get("fermi_surface_mesh") or (200 if len(ctx.periodic) == 2 else 150)
    n = int(n)
    pts, (a, b) = ctx.plane_points(fixed, value, n, centred=True)
    workers, chunk = ctx.parallel("Fermi surface", 2, len(pts))
    energies = np.concatenate(
        [e for _, e in _pmap(_w_edges, _chunks(pts, chunk), workers)]
    )
    energies = energies.reshape(n, n, -1)
    crossing = [
        int(band)
        for band in range(energies.shape[-1])
        if energies[..., band].min() < 0.0 < energies[..., band].max()
    ]
    recip, ea, eb = plane_frame(ctx.lattice, a, b)
    cart = pts @ recip
    X = (cart @ ea).reshape(n, n)
    Y = (cart @ eb).reshape(n, n)
    contours: Dict[str, List[List[List[float]]]] = {}
    try:
        plt = ctx._plt()
        fig, ax = plt.subplots(figsize=(5.5, 5))
        palette = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9"]
        for i, band in enumerate(crossing):
            cs = ax.contour(
                X,
                Y,
                energies[..., band],
                levels=[0.0],
                colors=[palette[i % len(palette)]],
                linewidths=1.2,
            )
            paths = []
            for seg in cs.allsegs[0]:
                paths.append(np.asarray(seg).tolist())
            contours[str(band)] = paths
        ax.set_aspect("equal")
        ax.set_xlabel(f"k / 2π along b{a + 1} (1/Å)")
        ax.set_ylabel(f"k / 2π normal to b{a + 1} (1/Å)")
        ax.set_title(f"Fermi surface, k{fixed + 1} = {value:g}")
        ctx._save(fig, "fermi_surface.png", "Fermi surface contours")
    except Exception as exc:
        ctx.warnings.append(f"Fermi-surface plot skipped: {exc}")
    data_file = os.path.join(OUT_DIR, "fermi_surface.json")
    start = float(pts[:, a].min())
    write_json_atomic(
        data_file,
        {
            "fixed_axis": fixed,
            "value": value,
            "mesh": n,
            # The fractional cell sampled on both in-plane axes, half-open:
            # its mesh points are start + i / mesh, so the end, the start's
            # periodic image, is not one of them.
            "k_frac_range": [start, start + 1.0],
            "k_frac_range_end": "excluded",
            "bands": crossing,
            # contours_cart holds (x, y) points along these two Cartesian unit
            # vectors, in these units.
            "axes_cart": [ea, eb],
            "units": "1/angstrom without 2 pi (k / 2 pi), as the band k_distance",
            "contours_cart": contours,
        },
    )
    ctx.add_output(data_file, "Fermi-surface contours (data)", "data")
    ctx.results["fermi_surface"] = {
        "fixed_axis": fixed,
        "value": value,
        "bands_crossing": crossing,
        "file": data_file,
    }


def _planes(ctx: Context) -> List[Tuple[int, float]]:
    requested = ctx.params.get("planes")
    if requested:
        out = []
        for entry in requested:
            if isinstance(entry, dict):
                out.append(
                    (int(entry.get("fixed_axis", 2)), float(entry.get("value", 0.0)))
                )
            else:
                out.append((int(entry[0]), float(entry[1])))
        return out
    if len(ctx.periodic) == 2:
        return [ctx.default_plane()]
    return [(c, v) for c in (2, 0, 1) for v in (0.0, 0.5)]


def _trim_points(ctx: Context) -> np.ndarray:
    axes = [[0.0, 0.5] if ax in ctx.periodic else [0.0] for ax in range(3)]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=1)


def run_topology(ctx: Context) -> None:
    """Wilson loops per plane: Chern numbers and, with time reversal, Z2."""
    params = ctx.params
    occupation = ctx.resolve_occupation()
    topo: Dict[str, Any] = {"occupation": occupation, "planes": []}
    ctx.results["topology"] = topo
    nocc = occupation.get("n_occ")
    if not nocc:
        topo["applicable"] = False
        topo["reason"] = (
            "No gapped occupied manifold at E_F (the model is metallic); set "
            "occupied_bands to the band count of an isolated group."
        )
        return
    if not occupation.get("isolated"):
        topo["applicable"] = False
        topo["reason"] = (
            f"Bands {nocc} and {nocc + 1} touch somewhere in the Brillouin zone "
            f"(smallest direct gap {occupation.get('isolation_meV', 0):.2f} meV): "
            "invariants of a non-isolated band set are undefined."
        )
        return
    topo["applicable"] = True

    trims = _trim_points(ctx)
    trim_energies = _solve(trims)
    kramers, split = kramers_degenerate(
        trim_energies, float(params.get("kramers_tol_meV") or 1.0) / 1000.0
    )
    z2_mode = str(params.get("z2") or "auto").lower()
    want_z2 = z2_mode == "on" or (z2_mode == "auto" and ctx.spinors and kramers)
    topo["time_reversal"] = {
        "spinors": ctx.spinors,
        "kramers_degenerate_at_trims": kramers,
        "max_kramers_split_meV": split * 1000.0 if math.isfinite(split) else None,
        "z2_computed": want_z2,
    }
    if z2_mode == "on" and not (ctx.spinors and kramers):
        ctx.warnings.append(
            "Z2 forced on without Kramers degeneracy: it is not a well-defined invariant here."
        )

    loop_n = int(params.get("loop_points") or 81)
    base_t = int(params.get("transverse_points") or 61)
    z2_by_plane: Dict[Tuple[int, float], Optional[int]] = {}
    for fixed, value in _planes(ctx):
        a, b = plane_axes(fixed)
        _STATE["boundary_phase"] = np.exp(-2j * np.pi * ctx.orb_red[:, a])
        n_t = base_t + (base_t % 2)  # even, so t = 0.5 is sampled
        record: Dict[str, Any] = {
            "fixed_axis": fixed,
            "value": value,
            "loop_axis": a,
            "transverse_axis": b,
        }
        for attempt in range(int(params.get("max_refinements") or 0) + 1):
            ts = np.arange(n_t) / n_t
            tasks = []
            for t in ts:
                loop = np.zeros((loop_n, 3))
                loop[:, a] = np.arange(loop_n) / loop_n
                loop[:, b] = t
                loop[:, fixed] = value
                tasks.append((loop, nocc))
            # A loop is one task: its points cannot be split across workers.
            workers, _ = ctx.parallel(
                f"Wilson loops, k{fixed + 1}={value:g}",
                3,
                n_t * loop_n,
                chunk=loop_n,
            )
            results = _pmap(_w_wilson, tasks, workers)
            phases = np.array([r[0] for r in results])
            wcc = np.array([r[1] for r in results])
            chern, max_step = chern_from_phases(phases)
            z2 = None
            on_trim_plane = abs(value) < 1e-9 or abs(value - 0.5) < 1e-9
            if want_z2 and on_trim_plane:
                z2 = z2_from_wcc(
                    wcc[: n_t // 2 + 1], float(params.get("z2_pos_tol") or 0.01)
                )
            converged = max_step < np.pi / 2 and abs(chern - round(chern)) < 0.05
            if converged and not (z2 and z2["ambiguous"]):
                break
            if attempt < int(params.get("max_refinements") or 0):
                ctx.log(
                    f"plane k{fixed + 1}={value}: refining transverse sampling ({n_t} -> {2 * n_t})"
                )
                n_t *= 2
        record.update(
            chern=int(round(chern)),
            chern_raw=chern,
            max_phase_step=max_step,
            converged=bool(converged),
            transverse_points=int(n_t),
            loop_points=loop_n,
        )
        if z2 is not None:
            record["z2"] = z2["z2"]
            record["z2_ambiguous"] = z2["ambiguous"]
            record["z2_crossings"] = z2["crossings"]
            z2_by_plane[(fixed, value)] = None if z2["ambiguous"] else z2["z2"]
        name = f"wcc_k{fixed + 1}_{value:g}".replace(".", "p")
        write_json_atomic(
            os.path.join(OUT_DIR, f"{name}.json"),
            {
                "t": ts,
                "berry_phase": phases,
                "wcc": wcc,
                "gap_midpoints": (z2 or {}).get("gap_midpoints"),
            },
        )
        record["file"] = os.path.join(OUT_DIR, f"{name}.json")
        ctx.add_output(
            record["file"], f"Wilson loop, k{fixed + 1} = {value:g} (data)", "data"
        )
        _plot_wcc(ctx, name, fixed, value, a, b, ts, wcc, z2)
        topo["planes"].append(record)
        ctx.log(
            f"plane k{fixed + 1}={value}: C={chern:+.4f}"
            + (f", Z2={z2['z2']}" if z2 else "")
        )

    if want_z2 and len(ctx.periodic) == 3:
        indices: Dict[str, Any] = {}
        nu0 = set()
        for c in range(3):
            z0, z5 = z2_by_plane.get((c, 0.0)), z2_by_plane.get((c, 0.5))
            if z0 is not None and z5 is not None:
                nu0.add((z0 + z5) % 2)
            indices[f"nu{c + 1}"] = z5
        indices["nu0"] = nu0.pop() if len(nu0) == 1 else None
        indices["consistent"] = len(nu0) <= 1
        topo["z2_indices"] = indices
    elif want_z2 and topo["planes"]:
        topo["z2_indices"] = {"nu": topo["planes"][0].get("z2")}


def _plot_wcc(ctx, name, fixed, value, a, b, ts, wcc, z2) -> None:
    try:
        plt = ctx._plt()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for j, t in enumerate(ts):
            ax.scatter(np.full(wcc.shape[1], t), wcc[j], s=3, color="#0072b2")
        if z2:
            half = ts[: len(z2["gap_midpoints"])]
            ax.plot(
                half,
                z2["gap_midpoints"],
                "D",
                ms=3,
                color="#d55e00",
                label="largest-gap midpoint",
            )
            ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"k{b + 1} (loop along k{a + 1})")
        ax.set_ylabel("Wannier charge centre")
        ax.set_title(f"WCC flow, k{fixed + 1} = {value:g}")
        ctx._save(
            fig, f"{name}.png", f"Wannier charge centres, k{fixed + 1} = {value:g}"
        )
    except Exception as exc:
        ctx.warnings.append(f"WCC plot skipped: {exc}")


def run_berry(ctx: Context) -> None:
    """Berry curvature (and quantum metric) on a plane and along the path."""
    params = ctx.params
    requested = params.get("berry_plane")
    if isinstance(requested, dict):
        fixed, value = int(requested.get("fixed_axis", 2)), float(
            requested.get("value", 0.0)
        )
    else:
        fixed, value = ctx.default_plane()
    a, b = plane_axes(fixed)
    mode = str(params.get("occupation") or "auto").lower()
    occupation = ctx.resolve_occupation()
    nocc = occupation.get("n_occ")
    if mode == "auto":
        mode = "bands" if nocc and occupation.get("isolated") else "fermi"
    if mode == "bands" and not nocc:
        raise PythtbError(
            ERR_BAD_PARAMS, "occupation='bands' needs a gapped model or occupied_bands"
        )
    plane_area = float(np.linalg.norm(np.cross(ctx.recip[a], ctx.recip[b])))
    normal = np.cross(ctx.recip[a], ctx.recip[b])
    normal = normal / np.linalg.norm(normal)
    _STATE.update(
        berry_plane=(a, b),
        berry_mode=mode,
        berry_occ=list(range(nocc)) if mode == "bands" else None,
        berry_metric=bool(params.get("quantum_metric")) and mode == "bands",
        plane_normal=normal,
    )
    n = int(params.get("berry_mesh") or 150)
    pts, _ = ctx.plane_points(fixed, value, n)
    per_k = 14 if mode == "bands" else 8
    workers, chunk = ctx.parallel(
        f"Berry curvature ({mode}), k{fixed + 1}={value:g}", per_k, len(pts)
    )
    parts = _pmap(_w_berry, _chunks(pts, chunk), workers)
    omega = np.concatenate([p["omega"] for p in parts]).reshape(n, n)
    integral = float(omega.sum() / (n * n * 2.0 * np.pi))
    berry: Dict[str, Any] = {
        "fixed_axis": fixed,
        "value": value,
        "plane_axes": [a, b],
        "occupation": mode,
        "occupation_info": occupation,
        "mesh": n,
        "approximation": APPROXIMATION_NOTE,
    }
    if mode == "bands":
        berry["chern_kubo"] = integral
        distance = abs(integral - round(integral))
        berry["chern_kubo_integer_distance"] = distance
        if distance > CHERN_INTEGER_TOL:
            ctx.warnings.append(
                f"the Kubo Chern number {integral:.3f} is {distance:.3f} from an "
                f"integer: a small gap is under-resolved on the {n}x{n} mesh "
                "(raise berry_mesh)"
            )
    else:
        berry["fermi_sea_integral_e2_h"] = integral
        berry["sigma_xy_e2_h"] = -integral
    berry["conventions"] = dict(BERRY_CONVENTIONS)
    if params.get("quantum_metric") and mode != "bands":
        ctx.warnings.append(
            "quantum metric skipped: it needs an isolated occupied band set "
            "(occupation 'bands'), and this run sums over the Fermi sea ('fermi')"
        )
    omega_cart = omega / plane_area
    berry["omega_cart_max_A2"] = float(np.abs(omega_cart).max())
    metric = None
    if "metric_trace" in parts[0]:
        metric = np.concatenate([p["metric_trace"] for p in parts]).reshape(n, n)
        weight = float(metric.sum() * plane_area / (n * n) / (2.0 * np.pi))
        berry["quantum_weight"] = weight
        berry["quantum_weight_bound_ok"] = bool(weight + 1e-6 >= abs(integral))
    ctx.results["berry"] = berry
    fname = f"berry_k{fixed + 1}_{value:g}".replace(".", "p")
    data = {
        "k_frac": pts,
        "omega_reduced": omega.ravel(),
        "omega_cart_A2": omega_cart.ravel(),
    }
    if metric is not None:
        data["metric_trace_A2"] = metric.ravel()
    write_json_atomic(os.path.join(OUT_DIR, f"{fname}.json"), data)
    ctx.add_output(
        os.path.join(OUT_DIR, f"{fname}.json"), "Berry curvature map (data)", "data"
    )
    _plot_plane_map(
        ctx,
        pts,
        a,
        b,
        n,
        omega_cart,
        f"{fname}.png",
        "Berry curvature Ω (Å²)",
        "RdBu_r",
        True,
    )
    if metric is not None:
        _plot_plane_map(
            ctx,
            pts,
            a,
            b,
            n,
            metric,
            f"metric_k{fixed + 1}_{value:g}".replace(".", "p") + ".png",
            "Quantum metric tr g (Å²)",
            "viridis",
            False,
        )
    if params.get("path_curvature"):
        _STATE["berry_metric"] = False  # the path plot shows the curvature only
        try:
            path = ctx.band_path()
            workers, chunk = ctx.parallel(
                f"Berry curvature ({mode}), path", per_k, len(path["kpoints"])
            )
            parts = _pmap(_w_berry, _chunks(path["kpoints"], chunk), workers)
            along = np.concatenate([p["omega"] for p in parts]) / plane_area
            dist = k_distance(path["kpoints"], ctx.lattice, path["break_indices"])
            berry["path_file"] = os.path.join(OUT_DIR, "berry_path.json")
            write_json_atomic(
                berry["path_file"],
                {
                    "k_distance": dist,
                    "omega_cart_A2": along,
                    "tick_indices": path["tick_indices"],
                    "tick_labels": path["tick_labels"],
                },
            )
            ctx.add_output(
                berry["path_file"], "Berry curvature along the path (data)", "data"
            )
            _plot_path_curvature(ctx, path, dist, along)
        except PythtbError as exc:
            ctx.warnings.append(f"path curvature skipped: {exc}")


def _plot_plane_map(ctx, pts, a, b, n, values, name, title, cmap, symmetric) -> None:
    try:
        plt = ctx._plt()
        cart = pts @ ctx.recip
        ea = ctx.recip[a] / np.linalg.norm(ctx.recip[a])
        normal = np.cross(ctx.recip[a], ctx.recip[b])
        eb = np.cross(normal / np.linalg.norm(normal), ea)
        X = (cart @ ea).reshape(n, n)
        Y = (cart @ eb).reshape(n, n)
        fig, ax = plt.subplots(figsize=(6, 5))
        kwargs = {}
        if symmetric:
            limit = float(np.percentile(np.abs(values), 99.5)) or 1.0
            kwargs = {"vmin": -limit, "vmax": limit}
        mesh = ax.pcolormesh(X, Y, values, cmap=cmap, shading="auto", **kwargs)
        fig.colorbar(mesh, ax=ax)
        ax.set_aspect("equal")
        ax.set_xlabel("k (1/Å)")
        ax.set_ylabel("k (1/Å)")
        ax.set_title(title)
        ctx._save(fig, name, title)
    except Exception as exc:
        ctx.warnings.append(f"{name} skipped: {exc}")


def _plot_path_curvature(ctx, path, dist, along) -> None:
    try:
        plt = ctx._plt()
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(dist, along, color="#0072b2", lw=1.2)
        ax.axhline(0.0, color="#999999", lw=0.5)
        pos, lab = ctx._ticks(path, dist)
        ax.set_xticks(pos)
        ax.set_xticklabels(lab)
        ax.set_xlim(dist[0], dist[-1])
        ax.set_ylabel("Ω (Å²)")
        ax.set_title("Berry curvature along the path")
        ctx._save(fig, "berry_path.png", "Berry curvature along the path")
    except Exception as exc:
        ctx.warnings.append(f"path curvature plot skipped: {exc}")


def run_slab(ctx: Context) -> None:
    """Open slab (3D) or ribbon (2D) bands weighted by edge localisation."""
    params = ctx.params
    two_d = len(ctx.periodic) == 2
    axis = params.get("slab_axis")
    if axis is None:
        finite = ctx.periodic[1] if two_d else 2
    else:
        finite = int(axis)
    if finite not in ctx.periodic:
        raise PythtbError(
            ERR_BAD_PARAMS,
            f"slab_axis {finite} is not a periodic direction of the model",
        )
    periodic = [ax for ax in ctx.periodic if ax != finite]
    cells = params.get("slab_cells")
    if not cells:
        if two_d:
            cells = 20 if ctx.n_orb <= 60 else 12
        else:
            cells = 12 if ctx.n_orb <= 60 else 6
    cells = int(cells)
    nk = int(params.get("slab_kpoints") or 200)
    if two_d:
        # The ribbon's 1D zone, periodic, sampled half a step off the
        # high-symmetry points: edge modes often cross E_F exactly at G or X,
        # where finite-size hybridisation mixes the two edges.
        p = periodic[0]
        nk += nk % 2
        kpath = np.zeros((nk, 3))
        kpath[:, p] = -0.5 + (np.arange(nk) + 0.5) / nk
        path = {
            "kpoints": kpath,
            "tick_indices": [],
            "tick_labels": [],
            "break_indices": [],
        }
        kind = "ribbon"
    else:
        p, q = periodic
        path = surface_path(ctx.lattice, p, q, nk)
        kind = "slab"
    table_H = ctx.table_H.copy()
    zero = np.where(np.all(ctx.table_R == 0, axis=1))[0]
    if len(zero):
        table_H[zero[0]] = table_H[zero[0]] - ctx.fermi * np.eye(ctx.n_orb)
    _STATE.update(
        table_R=ctx.table_R,
        table_H=table_H,
        slab_finite=finite,
        slab_periodic=periodic,
        slab_cells=cells,
        slab_window=float(params.get("slab_window_eV") or 0.5),
        n_orb=ctx.n_orb,
        edge_cells=int(params.get("edge_cells") or 2),
    )
    dim = cells * ctx.n_orb
    # One k-point per task: each solves its own dim x dim slab Hamiltonian.
    workers, _ = ctx.parallel(
        f"{kind}, {cells} cells along a{finite + 1} ({dim} states)",
        4,
        len(path["kpoints"]),
        chunk=1,
        n_orb=dim,
    )
    parts = _pmap(_w_slab, list(path["kpoints"]), workers)
    energies = [p[0] for p in parts]
    left = [p[1] for p in parts]
    right = [p[2] for p in parts]
    crossings = edge_crossings(energies, left, right, periodic=kind == "ribbon")
    if kind == "ribbon":
        # The x axis of a ribbon is its 1D crystal momentum (units of 2 pi / a).
        dist = path["kpoints"][:, periodic[0]].copy()
    else:
        dist = k_distance(path["kpoints"], ctx.lattice, path["break_indices"])
    ctx.results["slab"] = {
        "kind": kind,
        "finite_axis": finite,
        "periodic_axes": periodic,
        "cells": cells,
        "states": dim,
        "window_eV": _STATE["slab_window"],
        "edge_cells": _STATE["edge_cells"],
        "edge_crossings_at_fermi": crossings,
        "note": "Crossing counts are informational: the bulk Chern number is the invariant.",
    }
    write_json_atomic(
        os.path.join(OUT_DIR, f"{kind}_bands.json"),
        {
            "k_distance": dist,
            "kpoints_frac": path["kpoints"],
            "energies_eV": [e for e in energies],
            "left_weight": left,
            "right_weight": right,
            "tick_indices": path["tick_indices"],
            "tick_labels": path["tick_labels"],
        },
    )
    ctx.add_output(
        os.path.join(OUT_DIR, f"{kind}_bands.json"),
        f"{kind.capitalize()} bands (data)",
        "data",
    )
    _plot_slab(ctx, kind, path, dist, energies, left, right)


def _plot_slab(ctx, kind, path, dist, energies, left, right) -> None:
    try:
        plt = ctx._plt()
        from matplotlib.colors import LinearSegmentedColormap

        fig, ax = plt.subplots(figsize=(7, 5))
        xs = (
            np.concatenate([np.full(len(e), x) for x, e in zip(dist, energies)])
            if energies
            else np.zeros(0)
        )
        ys = np.concatenate(energies) if energies else np.zeros(0)
        lw = np.concatenate(left) if left else np.zeros(0)
        rw = np.concatenate(right) if right else np.zeros(0)
        on_edge = (lw + rw) >= 0.3
        # Bulk states in grey; edge states coloured by which edge holds them
        # (purple: both, e.g. hybridised surfaces of a thin slab).
        ax.scatter(xs[~on_edge], ys[~on_edge], s=3, color="#b0b0b0", linewidths=0)
        cmap = LinearSegmentedColormap.from_list(
            "edges", ["#0072b2", "#7b3294", "#d55e00"]
        )
        share = rw[on_edge] / np.maximum(lw[on_edge] + rw[on_edge], 1e-12)
        sc = ax.scatter(
            xs[on_edge],
            ys[on_edge],
            c=share,
            s=4 + 16 * np.clip(lw[on_edge] + rw[on_edge], 0, 1),
            cmap=cmap,
            vmin=0,
            vmax=1,
            linewidths=0,
        )
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("edge share: first (0) → last (1)")
        if kind == "ribbon":
            pos, lab = [-0.5, 0.0, 0.5], ["-X", "Γ", "X"]
            ax.set_xlim(-0.5, 0.5)
        else:
            pos, lab = ctx._ticks(path, dist)
            ax.set_xlim(dist[0], dist[-1])
        ax.set_xticks(pos)
        ax.set_xticklabels(lab)
        for x in pos:
            ax.axvline(x, color="#dddddd", lw=0.5, zorder=0)
        ax.axhline(0.0, color="#666666", lw=0.6, ls="--")
        window = _STATE["slab_window"]
        ax.set_ylim(-window, window)
        ax.set_ylabel("E - E_F (eV)")
        ax.set_title(f"{kind.capitalize()} bands ({_STATE['slab_cells']} cells)")
        ctx._save(
            fig, f"{kind}_bands.png", f"{kind.capitalize()} bands with edge weight"
        )
    except Exception as exc:
        ctx.warnings.append(f"{kind} plot skipped: {exc}")


# ---------------------------------------------------------------------------
# PYTHTB_QGEOM: quantum geometry and the intrinsic nonlinear Hall response
# ---------------------------------------------------------------------------
FULL_POSITION_NOTE = (
    "full position matrix in Wannier space (wannier90 _tb.dat): curvature quantities "
    "complete to the finite-difference accuracy of r; the metric and K are in-W lower bounds"
)
QGEOM_OUT_BUDGET_BYTES = 10 * 1024 * 1024
# The vertex-weight Fermi-surface integrals at the highest temperature against
# the Theta convolution: a larger gap means the mesh is coarse against kT.
QGEOM_SMEARED_WARN = 0.05
# The anti-Hermitian part of a finite-difference position matrix on a 12x12
# mesh is ~0.1 of its largest element; warn only well beyond that.
ANTI_HERMITIAN_WARN = 0.15


def _qgeom_kernel():
    """The numpy kernel: the package module in the backend, the staged copy on HPC."""
    try:
        from . import pythtb_qgeom as kernel  # type: ignore[no-redef]
    except ImportError:
        import _pythtb_qgeom as kernel  # type: ignore[no-redef]
    return kernel


def _w_qgeom(kpts: np.ndarray):
    return _STATE["qgeom_fn"](kpts)


def _w_qgeom_energies(kpts: np.ndarray):
    return _STATE["qgeom_energies"](kpts)


def _w_qgeom_path(kpts: np.ndarray):
    return _STATE["qgeom_path_fn"](kpts)


class _QgeomParallel:
    """Splits the integrator's k-point batches over the fork pool."""

    def __init__(self, ctx: "Context", n_orb: int):
        self.workers, self.cap = ctx.parallel(
            "quantum geometry", 20, 10**7, n_orb=n_orb
        )

    def _map(self, worker, kfrac: np.ndarray):
        n = len(kfrac)
        chunk = max(1, min(self.cap, math.ceil(n / max(1, self.workers))))
        return _pmap(worker, _chunks(kfrac, chunk), self.workers)

    def payload(self, serial):
        _STATE["qgeom_fn"] = serial

        def run(kfrac: np.ndarray):
            parts = self._map(_w_qgeom, kfrac)
            return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}

        return run

    def energies(self, table):
        _STATE["qgeom_energies"] = table.energies

        def run(kfrac: np.ndarray):
            return np.concatenate(self._map(_w_qgeom_energies, kfrac))

        return run


def _qgeom_full_table(
    ctx: "Context", kernel, frame: Dict[str, np.ndarray], H: np.ndarray
):
    """The full-position table from the WANNIER run's own _tb.dat, checked
    against _hr.dat and _centres.xyz; returns (table, position report)."""
    try:
        lay = kernel.tb_layout(TB_FILENAME)
    except (OSError, ValueError, IndexError) as exc:
        raise PythtbError(ERR_POSITION_MISMATCH, f"{TB_FILENAME} is unreadable: {exc}")
    problems = []
    if lay["num_wann"] != ctx.n_orb:
        problems.append(
            f"num_wann {lay['num_wann']} differs from _hr.dat's {ctx.n_orb}"
        )
    if not lay["lines_ok"] or lay["expected_bytes"] != lay["actual_bytes"]:
        problems.append(
            f"{lay['actual_bytes']} bytes, expected {lay['expected_bytes']} from its header "
            "(truncated, or not in wannier90 3.1's format)"
        )
    lat_err = float(np.max(np.abs(lay["lattice"] - ctx.lattice)))
    if lat_err > 1e-5:
        problems.append(f"its lattice differs from the .win's by {lat_err:.2e} A")
    if problems:
        raise PythtbError(
            ERR_POSITION_MISMATCH, f"{TB_FILENAME}: " + "; ".join(problems)
        )
    t0 = time.time()
    tb = kernel.read_tb(TB_FILENAME, lay)
    ctx._record_input(TB_FILENAME)
    tb_deg = {tuple(int(x) for x in r): int(d) for r, d in zip(tb["R"], lay["ndegen"])}
    if tb_deg != getattr(ctx, "hr_degeneracy", tb_deg):
        raise PythtbError(
            ERR_POSITION_MISMATCH,
            f"{TB_FILENAME} has another R set or other degeneracies than {HR_FILENAME}: "
            "the two files are not from the same wannier90 run",
        )
    nd = lay["ndegen"].astype(float)
    H_tb = tb["H"] / nd[:, None, None]
    r = tb["r"] / nd[:, None, None, None]
    r, anti = kernel.hermitize(tb["R"], r)
    i0 = int(np.flatnonzero(~tb["R"].any(axis=1))[0])
    centres_r = np.stack([np.real(np.diag(r[i0][..., a])) for a in range(2)], axis=1)
    centres_r = np.concatenate([centres_r, np.zeros((len(centres_r), 1))], axis=1)
    centre_err = kernel.inplane_mismatch(centres_r, ctx.centres, ctx.lattice)
    stack = np.concatenate([H_tb[..., None], r], axis=-1)
    R_u, S_u = ctx.ws_apply(tb["R"], stack)
    if R_u.shape != ctx.table_R.shape or not np.array_equal(R_u, ctx.table_R):
        raise PythtbError(
            ERR_POSITION_MISMATCH,
            f"{TB_FILENAME} unfolds onto another R set than {HR_FILENAME}",
        )
    h_err = float(np.max(np.abs(S_u[..., 0] - ctx.table_H)))
    offdiag = r.copy()
    idx = np.arange(ctx.n_orb)
    offdiag[i0][idx, idx] = 0.0
    off_max = float(np.max(np.abs(offdiag)))
    sum_r2 = float(np.sum(nd[:, None, None, None] * np.abs(offdiag) ** 2))
    checks = {
        "h_vs_hr_max_meV": h_err * 1e3,
        "centres_inplane_max_A": centre_err,
        "antihermitian_rel": anti / off_max if off_max > 0 else 0.0,
        "offdiag_sum_xy_A2": sum_r2,
        "rz_offdiag_sum_A2": tb["rz2_offdiag"],
    }
    omega_I = ctx.params.get("wannier_omega_I_A2")
    if omega_I is not None:
        checks["omega_I_A2"] = float(omega_I)
    if h_err > 2e-6:
        raise PythtbError(
            ERR_POSITION_MISMATCH,
            f"H in {TB_FILENAME} differs from {HR_FILENAME} by {h_err * 1e3:.3g} meV",
        )
    if centre_err > 2e-5:
        raise PythtbError(
            ERR_POSITION_MISMATCH,
            f"r_ii(0) in {TB_FILENAME} differs from the Wannier centres by {centre_err:.3g} A",
        )
    if checks["antihermitian_rel"] > ANTI_HERMITIAN_WARN:
        ctx.warnings.append(
            f"the position matrix is {checks['antihermitian_rel']:.0%} anti-Hermitian before "
            "symmetrising: a coarse Wannier k-mesh; positions are approximate"
        )
    proj = np.stack([frame["e1"][:2], frame["e2"][:2]], axis=1)
    r_frame = S_u[..., 1:3] @ proj
    table = kernel.Table(ctx.table_R, H, ctx.lattice, frame, r=r_frame)
    report = {
        "mode": "full",
        "source": "wannier90_tb_dat",
        "tb": {
            "bytes": int(lay["actual_bytes"]),
            "expected_bytes": int(lay["expected_bytes"]),
            "nrpts": int(lay["nrpts"]),
            "read_seconds": round(time.time() - t0, 1),
        },
        "checks": checks,
    }
    ctx.log(
        f"{TB_FILENAME}: {lay['nrpts']} R; H vs hr {h_err * 1e3:.2g} meV; centres "
        f"{centre_err:.2g} A; anti-Hermitian {checks['antihermitian_rel']:.3f}"
    )
    return table, report


def _qgeom_parity(
    ctx: "Context", kernel, diag, window, n_occ, s_applies
) -> Dict[str, Any]:
    """The diagonal level against PythTB's own metric and curvature at 32 k."""
    model = getattr(ctx, "model", None)
    if model is None:
        return {"skipped": "no PythTB model"}
    rng = np.random.default_rng(20261010)
    k = np.c_[rng.uniform(-0.5, 0.5, (32, 2)), np.zeros(32)]
    try:
        if s_applies:
            occ = list(range(n_occ))
            ours = kernel.evaluate(diag, k, window=window, occupied=[n_occ])["blocks"][
                n_occ
            ]
            g_ours, om_ours = ours["g_in"], ours["omega"]
        else:
            band = int(min(max(window[0], (n_occ or 1) - 1), window[1] - 1))
            occ = [band]
            ours = kernel.evaluate(diag, k, window=(band, band + 1))
            g_ours, om_ours = ours["g"][:, 0], ours["omega"][:, 0]
        frac = k.copy()
        g = np.real(
            np.asarray(
                model.quantum_metric(frac, occ_idxs=occ, plane=None, cartesian=True)
            )
        )
        g = g.reshape(3, 3, len(k))
        P = np.stack([diag.frame["e1"], diag.frame["e2"]])
        g_ref = np.einsum("ia,abk,jb->kij", P, g, P)
        om_red = np.asarray(
            model.berry_curvature(frac, occ_idxs=occ, plane=(0, 1), cartesian=False),
            float,
        )
        plane_area = float(np.linalg.norm(np.cross(ctx.recip[0], ctx.recip[1])))
        om_ref = om_red.reshape(len(k)) / plane_area
    except ZeroDivisionError as exc:
        return {"skipped": f"PythTB: {exc}"}
    scale = max(float(np.abs(g_ref).max()), float(np.abs(om_ref).max()), 1e-300)
    dev = (
        max(float(np.abs(g_ours - g_ref).max()), float(np.abs(om_ours - om_ref).max()))
        / scale
    )
    if dev > 1e-8:
        ctx.warnings.append(
            f"the diagonal level differs from PythTB's own metric by {dev:.2g} (relative)"
        )
    return {
        "k_points": len(k),
        "max_rel": dev,
        "bands": occ if len(occ) < 4 else [occ[0], occ[-1]],
    }


def _qgeom_dft_check(ctx: "Context", kernel, table) -> Optional[Dict[str, Any]]:
    """TB eigenvalues against the staged .eig inside the frozen window."""
    if not os.path.exists(EIG_FILENAME):
        return None
    kpts = np.asarray(ctx.win.get("kpoints") or [], float).reshape(-1, 3)
    froz = (ctx.win.get("dis_froz_min"), ctx.win.get("dis_froz_max"))
    if not len(kpts) or froz[0] is None or froz[1] is None:
        return {"skipped": "the .win has no kpoints block or no frozen window"}
    with open(EIG_FILENAME) as handle:
        E_dft = kernel.parse_eig(handle.read())
    ctx._record_input(EIG_FILENAME)
    if len(E_dft) != len(kpts):
        return {
            "skipped": f"{len(E_dft)} k-points in the .eig, {len(kpts)} in the .win"
        }
    kpts = kpts.copy()
    kpts[:, ctx.folded_axes] = 0.0
    E_tb = table.energies(kpts) + ctx.fermi
    out = kernel.dft_exactness(E_tb, E_dft, (float(froz[0]), float(froz[1])))
    if not out["counts_match"] or out["max_meV"] > 0.1:
        ctx.warnings.append(
            f"the model misses frozen DFT states by up to {out['max_meV']:.3g} meV at the DFT mesh"
        )
    return out


def _qgeom_path_points(ctx: "Context", kernel, frame) -> Dict[str, Any]:
    """The display path: band.kpt's, plus log-spaced points around every tick."""
    custom = ctx.params.get("path_kpoints_frac")
    if custom:
        pts = np.asarray(custom, float).reshape(-1, 3)[:4000]
        return {
            "kpoints": pts,
            "tick_indices": [],
            "tick_labels": [],
            "break_indices": [],
        }
    try:
        path = ctx.band_path()
    except PythtbError as exc:
        ctx.warnings.append(f"no k-path for the curves along k: {exc}")
        return {"kpoints": np.zeros((0, 3)), "tick_indices": [], "tick_labels": []}
    pts = np.asarray(path["kpoints"], float).reshape(-1, 3)
    b = kernel.reciprocal_in_plane(ctx.lattice, frame)
    kc = pts[:, :2] @ b
    seg = np.r_[0.0, np.linalg.norm(np.diff(kc, axis=0), axis=1)]
    for i in path.get("break_indices") or []:
        if 0 < i < len(seg):
            seg[i] = 0.0
    dist = np.cumsum(seg)
    n_log = int(ctx.params.get("path_log_points") or 0)
    extra_d: List[float] = []
    if n_log > 0:
        offsets = np.logspace(-4, -1, n_log)
        for i in path.get("tick_indices") or []:
            if 0 <= i < len(dist):
                extra_d += list(dist[i] - offsets) + list(dist[i] + offsets)
    extra_d = [d for d in extra_d if dist[0] < d < dist[-1]]
    if extra_d:
        frac_x = np.interp(extra_d, dist, pts[:, 0])
        frac_y = np.interp(extra_d, dist, pts[:, 1])
        allp = np.concatenate([pts, np.c_[frac_x, frac_y, np.zeros(len(extra_d))]])
        alld = np.concatenate([dist, extra_d])
        order = np.argsort(alld, kind="stable")
        ticks = [
            int(np.flatnonzero(order == i)[0]) for i in path.get("tick_indices") or []
        ]
        return {
            "kpoints": allp[order],
            "distance": alld[order],
            "tick_indices": ticks,
            "tick_labels": path.get("tick_labels") or [],
        }
    return {
        "kpoints": pts,
        "distance": dist,
        "tick_indices": path.get("tick_indices") or [],
        "tick_labels": path.get("tick_labels") or [],
    }


def _qgeom_mu_cnp(count: np.ndarray, mu: np.ndarray, n_occ: int) -> Optional[float]:
    if not len(count) or n_occ < count.min() or n_occ > count.max():
        return None
    order = np.argsort(count, kind="stable")
    return float(np.interp(n_occ, count[order], mu[order]))


def _chi_amplitudes(M: np.ndarray, kernel) -> Dict[str, float]:
    """Each convention's chi (mA nm/V^2) as a signed trigonal amplitude along M's."""
    tm = kernel.trigonal(M)
    A3 = float(tm["A3"])
    out = {}
    for name, chi in kernel.chi_all_conventions(M).items():
        tc = kernel.trigonal(chi)
        out[name] = (
            float((tc["alpha"] * tm["alpha"] + tc["beta"] * tm["beta"]) / A3)
            if A3 > 0
            else 0.0
        )
    return out


def run_qgeom(ctx: Context) -> None:
    """Quantum geometry, the metric (BCP) dipole and the intrinsic nonlinear Hall
    response of a 2D model, integrated on an adaptive mesh at two levels."""
    kernel = _qgeom_kernel()
    params = ctx.params
    t_start = time.time()
    if len(ctx.periodic) != 2 or list(ctx.periodic) != [0, 1]:
        raise PythtbError(
            ERR_BAD_PARAMS, "quantum geometry needs a 2D model periodic along a1 and a2"
        )
    if params.get("min_hopping_norm") or params.get("max_distance"):
        raise PythtbError(
            ERR_BAD_PARAMS,
            "quantum geometry uses the full hopping table: drop the cutoffs",
        )
    if kernel.a3_misalignment(ctx.lattice) > 1e-6:
        raise PythtbError(ERR_BAD_PARAMS, "a3 must be along a1 x a2 (the layer normal)")
    frame = kernel.layer_frame(ctx.lattice)
    # Hamiltonian, E_F on the diagonal; centres as PythTB places the orbitals.
    H = np.asarray(ctx.table_H, complex).copy()
    i0 = int(np.flatnonzero(~np.asarray(ctx.table_R).any(axis=1))[0])
    H[i0] -= ctx.fermi * np.eye(ctx.n_orb)
    # PythTB's orbital positions: the diagonal level is exactly its model.
    orbs = getattr(ctx, "orb_red", None)
    if orbs is None:
        orbs = np.asarray(ctx.centres, float) @ np.linalg.inv(ctx.lattice)
    tau_cart = np.asarray(orbs, float) @ ctx.lattice
    tau = np.stack([tau_cart @ frame["e1"], tau_cart @ frame["e2"]], axis=1)
    diag = kernel.Table(ctx.table_R, H, ctx.lattice, frame, tau=tau)
    mode = str(params.get("position_operator") or "auto").lower()
    full = None
    position: Dict[str, Any] = {"mode": "diagonal", "source": "none"}
    if mode not in ("auto", "full", "diagonal"):
        raise PythtbError(
            ERR_BAD_PARAMS, f"position_operator {mode!r} is not auto, full or diagonal"
        )
    if mode != "diagonal":
        if os.path.exists(TB_FILENAME):
            if abs(abs(frame["normal"][2]) - 1.0) > 1e-6:
                raise PythtbError(
                    ERR_BAD_PARAMS, "the full level needs the layer normal along z"
                )
            full, position = _qgeom_full_table(ctx, kernel, frame, H)
        elif mode == "full":
            raise PythtbError(
                ERR_POSITION_UNAVAILABLE,
                f"no {TB_FILENAME}: re-run the WANNIER step (new runs write it by default) "
                "or choose position_operator = diagonal",
            )
        else:
            ctx.warnings.append(
                f"the WANNIER job has no {TB_FILENAME}: the diagonal (tight-binding) level only; "
                "re-run the WANNIER step for the full position matrix"
            )
    primary = full or diag
    mu_lo, mu_hi = (
        float(x) * 1e-3 for x in params.get("mu_range_meV") or [-100.0, 100.0]
    )
    mu_step = float(params.get("mu_step_meV") or 0.1) * 1e-3
    n_mu = int(round((mu_hi - mu_lo) / mu_step)) + 1
    if not (mu_hi > mu_lo) or n_mu > 4001:
        raise PythtbError(
            ERR_BAD_PARAMS, f"the mu grid has {n_mu} points (at most 4001)"
        )
    mu_out = mu_lo + mu_step * np.arange(n_mu)
    temps = sorted(float(T) for T in (params.get("temperatures_K") or []))
    if any(T < 0.5 for T in temps):
        raise PythtbError(ERR_BAD_PARAMS, "temperatures must be at least 0.5 K")
    path = _qgeom_path_points(ctx, kernel, frame)
    parallel = _QgeomParallel(ctx, ctx.n_orb)
    budget = float(params.get("time_budget_s") or 0) or None
    res = kernel.adaptive_response(
        primary,
        mu_out=mu_out,
        temperatures_K=temps,
        occupied_bands=params.get("occupied_bands"),
        insulator_min_gap=float(params.get("insulator_min_gap_meV") or 0.1) * 1e-3,
        deg_tol=float(params.get("degeneracy_tol_meV") or 0.001) * 1e-3,
        tolerance=float(params.get("qgeom_tolerance") or 1e-3),
        base=int(params.get("qgeom_base_mesh") or 64),
        max_points=int(params.get("qgeom_max_points") or 1_000_000),
        gap_ratio=float(params.get("qgeom_gap_ratio") or 0.1),
        gap_floor=float(params.get("qgeom_gap_floor_meV") or 0.05) * 1e-3,
        edge_resolution=float(params.get("edge_resolution_meV") or 0.2) * 1e-3,
        occupation_mesh=int(params.get("occupation_mesh") or 241),
        tau_diag=tau if full is not None else None,
        deadline=(time.time() + budget) if budget else None,
        frame=frame,
        log=ctx.log,
        parallel=parallel,
        extra_kfrac=path["kpoints"],
    )
    res["dipole_sources"] = _qgeom_dipole_sources(
        ctx, kernel, res, primary, parallel, mu_out
    )
    ctx.results["qgeom"] = _qgeom_summary(
        ctx,
        kernel,
        res,
        full is not None,
        position,
        diag,
        primary,
        frame,
        path,
        mu_out,
        t_start,
    )


def _qgeom_dipole_sources(
    ctx, kernel, res, primary, parallel, mu_out
) -> Optional[Dict[str, Any]]:
    """The partner-gap shares of the T = 0 dipole at E_F, mu_CNP and the A3
    maximum (snapped to the output grid). A diagnostic: a failure is a warning."""
    final = res["stages"].get("tau2") or res["stages"]["tau"]
    cur = final["levels"]["main"]["per_T"][0]
    wanted = {
        "E_F": 0.0,
        "CNP": _qgeom_mu_cnp(cur["count"], mu_out, int(res["n_occ"])),
        "max_A3": float(mu_out[int(np.argmax(kernel.trigonal(cur["M"])["A3"]))]),
    }
    points = {
        name: float(mu_out[int(np.argmin(np.abs(mu_out - mu)))])
        for name, mu in wanted.items()
        if mu is not None and mu_out[0] <= mu <= mu_out[-1]
    }
    window = tuple(res["window"])
    deg_tol = float(ctx.params.get("degeneracy_tol_meV") or 0.001) * 1e-3
    frozen = ctx._frozen_window()

    def fields(kfrac: np.ndarray) -> Dict[str, np.ndarray]:
        return kernel.source_fields(
            primary, kfrac, window=window, deg_tol=deg_tol, frozen=frozen
        )

    t0 = time.time()
    try:
        rows = kernel.dipole_sources(
            res["integrator"], points, parallel.payload(fields)
        )
    except Exception as exc:  # noqa: BLE001 - a diagnostic never fails the step
        ctx.warnings.append(f"no dipole sources: {exc}")
        return None
    ctx.log(f"dipole sources at {sorted(points)} ({time.time() - t0:.1f}s)")
    return {
        "level": primary.level,
        "T_K": 0.0,
        "gap_bins_meV": list(kernel.GAP_BIN_EDGES_MEV),
        "frozen_window_meV": [frozen[0] * 1e3, frozen[1] * 1e3],
        "points": rows,
    }


def _qgeom_summary(
    ctx, kernel, res, full_level, position, diag, primary, frame, path, mu_out, t_start
) -> Dict[str, Any]:
    params = ctx.params
    occ = res["occupation"]
    lo, hi = res["window"]
    n_occ = int(res["n_occ"])
    temps = res["temperatures_K"]
    integ = res["integrator"]
    stages = res["stages"]
    final = stages.get("tau2") or stages["tau"]
    first = stages["tau"]
    converged = bool(first["ok"])
    confirmed = bool(stages.get("tau2", {}).get("ok"))
    lvl_main = final["levels"]["main"]
    lvl_diag = final["levels"].get("diag")
    area = kernel.cell_area(ctx.lattice)
    # mu_CNP per temperature, then the headline points.
    cnp = {
        T: _qgeom_mu_cnp(lvl_main["per_T"][i]["count"], mu_out, n_occ)
        for i, T in enumerate(temps)
    }
    headline: Dict[str, Any] = {}
    quotable_err = True
    for i, T in enumerate(temps):
        cur = lvl_main["per_T"][i]
        prev = first["levels"]["main"]["per_T"][i]
        A3 = kernel.trigonal(cur["M"])["A3"]
        points = {
            "E_F": 0.0,
            "CNP": cnp[T],
            "max_A3": float(mu_out[int(np.argmax(A3))]),
        }
        rows = {}
        for name, mu in points.items():
            if mu is None or not (mu_out[0] <= mu <= mu_out[-1]):
                continue
            j = int(np.argmin(np.abs(mu_out - mu)))
            M, Mp = cur["M"][j], prev["M"][j]
            tr, trp = kernel.trigonal(M), kernel.trigonal(Mp)
            chi, chip = _chi_amplitudes(M, kernel), _chi_amplitudes(Mp, kernel)
            err = float(abs(tr["A3"] - trp["A3"]))
            S_M = float(np.abs(M).max())
            if err > max(0.05 * float(tr["A3"]), 1e-3 * S_M):
                quotable_err = False
            row = {
                "mu_meV": float(mu_out[j]) * 1e3,
                "A3_A_per_eV": float(tr["A3"]),
                "A3_err": err,
                "theta0_deg": float(tr["theta0_deg"]),
                "chi_mA_nm_per_V2": chi,
                "chi_err": {k: abs(chi[k] - chip[k]) for k in chi},
                "sigma_xy_e2_h": float(cur["sigma_xy_e2_h"][j]),
                "density_cm2": float(kernel.density_cm2(cur["count"][j] - n_occ, area)),
            }
            if lvl_diag is not None:
                Ad = float(kernel.trigonal(lvl_diag["per_T"][i]["M"][j])["A3"])
                row["full_over_diag_A3"] = float(tr["A3"]) / Ad if Ad > 0 else None
            rows[name] = row
        headline[f"{T:g}K"] = rows
    # Occupied manifold (C, K) when the gap above n_occ is open everywhere.
    occupied: Dict[str, Any] = {"applicable": bool(res["S_applies"]), "n_occ": n_occ}
    vals = integ.mesh.vals
    failed: List[str] = []
    if res["S_applies"]:
        b = lvl_main["blocks"]
        two_pi = 2.0 * math.pi
        C, C_in = float(b[1] / two_pi), float(b[2] / two_pi)
        K_in = np.array([[b[3], b[4]], [b[4], b[5]]]) / two_pi
        psd = float(np.min(vals["psd_S"])) if "psd_S" in vals else None
        occupied.update(
            chern=C,
            chern_integer_distance=abs(C - round(C)),
            chern_in=C_in,
            K_in=K_in.tolist(),
            K_in_minus_abs_C=float(np.trace(K_in)) - abs(C_in),
            psd_in_min_rel=psd,
        )
        omega_I = params.get("wannier_omega_I_A2")
        if omega_I is not None and full_level:
            occupied["K_upper_loose"] = (
                float(np.trace(K_in)) + two_pi * float(omega_I) / area
            )
        if lvl_diag is not None:
            bd = lvl_diag["blocks"]
            occupied["K_diag"] = float((bd[3] + bd[5]) / two_pi)
        if psd is not None and psd < -1e-8:
            failed.append(f"Q_in has a negative eigenvalue ({psd:.2g} relative)")
        if float(np.trace(K_in)) < abs(C_in) - 1e-6:
            failed.append("K_in < |C_in|")
        if failed:
            occupied["verdict"] = "failed_invariant"
        elif confirmed and occupied["chern_integer_distance"] <= 0.02:
            occupied["verdict"] = "ok"
        else:
            occupied["verdict"] = "unconverged"
            occupied["reason"] = (
                "C not an integer: an under-resolved gap"
                if confirmed
                else "not confirmed"
            )
    else:
        occupied["verdict"] = "not_applicable"
        occupied["reason"] = "the gap above the occupied bands closes somewhere"
    sample = np.c_[np.random.default_rng(7).uniform(-0.5, 0.5, (8, 2)), np.zeros(8)]
    sum_rule = kernel.sum_rule_residual(primary, sample)
    if sum_rule > 1e-8:
        failed.append(f"sum_n G^n = {sum_rule:.2g} (relative) at sampled k")
    # Symmetry residuals at the headline (T = 0, E_F).
    M0 = lvl_main["per_T"][0]["M"]
    j0 = int(np.argmin(np.abs(mu_out)))
    tr0 = kernel.trigonal(M0[j0])
    symmetry = {
        "lattice_hexagonal": bool(kernel.is_hexagonal(ctx.lattice)),
        "c3_band_floor_meV": (
            (occ["c3_band_floor"] * 1e3)
            if occ.get("c3_band_floor") is not None
            else None
        ),
        "M_c3_residual": kernel.c3_residual_tensor(M0[j0]),
        "sigma_dT_residual": (
            float(abs(tr0["beta"]) / tr0["A3"]) if tr0["A3"] > 0 else None
        ),
        "bcd_residual": _bcd_residual(
            integ, lvl_main["per_T"][0]["BCD"][j0], float(mu_out[j0])
        ),
    }
    if res["S_applies"]:
        symmetry["K_c3_residual"] = kernel.c3_residual_matrix(
            np.array(occupied["K_in"])
        )
    smeared_rel = (res.get("smeared_check") or {}).get("rel")
    if smeared_rel is not None and smeared_rel > QGEOM_SMEARED_WARN:
        ctx.warnings.append(
            f"the Fermi-surface integrals at {temps[-1]:g} K from vertex weights differ "
            f"from the Theta convolution by {100 * smeared_rel:.1f}%: the mesh is coarse "
            "against kT there"
        )
    froz = ctx._frozen_window()
    in_window = bool(froz[0] <= mu_out[0] and mu_out[-1] <= froz[1])
    pt = occ["pt_pairs"]
    reasons: List[str] = []
    if not full_level:
        reasons.append("diagonal (tight-binding) position level")
    if not converged or not confirmed:
        reasons.append("integration not converged and confirmed")
    if not quotable_err:
        reasons.append(
            "A3 error above max(5% A3, 1e-3 of its scale) at a headline point"
        )
    if not in_window:
        reasons.append("the mu range leaves the frozen window")
    if pt["detected"]:
        reasons.append("PT-paired bands")
    if failed:
        verdict = "failed_invariant"
        reasons = failed + reasons
    elif reasons:
        verdict = "indicative"
    else:
        verdict = "quotable"
    summary: Dict[str, Any] = {
        "approximation": FULL_POSITION_NOTE if full_level else APPROXIMATION_NOTE,
        "approximation_level": (
            "full_position_in_wannier_space" if full_level else "diagonal_position"
        ),
        "conventions": {
            "omega": "-2 Im Q",
            "sigma_xy": "-(e^2/h)(1/2pi) sum int f Omega, charge -e",
            "chi": "-(e^3/hbar)[c_out M_abc + c_in(M_bac + M_cab)]",
            "chi_sets": {k: list(v) for k, v in kernel.CONVENTION_SETS.items()},
            "k_units": "1/angstrom, 2 pi included",
            "frame": {k: [float(x) for x in v] for k, v in frame.items()},
        },
        "position": position,
        "electron_count": {
            "n_occ": n_occ,
            "source": (
                "occupied_bands" if params.get("occupied_bands") else "count_at_EF"
            ),
            "count_at_EF": occ["count_at_EF"],
            "gap_at_gamma_meV": (
                (occ["gap_at_gamma"] * 1e3)
                if occ.get("gap_at_gamma") is not None
                else None
            ),
            "min_direct_gap_meV": (
                (occ["min_direct_gap"] * 1e3)
                if occ.get("min_direct_gap") is not None
                else None
            ),
            "global_gap_meV": (
                (occ["global_gap"] * 1e3) if occ.get("global_gap") is not None else None
            ),
            "state_at_cnp": _state_at_cnp(
                occ, float(params.get("insulator_min_gap_meV") or 0.1) * 1e-3
            ),
            "mu_cnp_meV": {
                f"{T:g}K": (v * 1e3 if v is not None else None) for T, v in cnp.items()
            },
        },
        "window": {
            "bands": [lo, hi],
            "always_occupied": lo,
            "degeneracy_tol_used_meV": float(params.get("degeneracy_tol_meV") or 0.001),
            "pt_pairs": pt,
        },
        "grid": {
            "mu_min_meV": float(mu_out[0]) * 1e3,
            "mu_max_meV": float(mu_out[-1]) * 1e3,
            "mu_step_meV": (
                float(mu_out[1] - mu_out[0]) * 1e3 if len(mu_out) > 1 else None
            ),
            "temperatures_K": temps,
            "downsampled": 1,
        },
        "integration": {
            "base_mesh": int(params.get("qgeom_base_mesh") or 64),
            "seeds": int(len(occ["seeds"])),
            "mu_subset": int(len(integ.mu_subset)),
            "vertices": int(final["vertices"]),
            "vertices_by_rule": dict(integ.mesh.vertices_by_rule),
            "gap_rule_capped": bool(integ.gap_rule_capped),
            "max_level": int(integ.mesh.lvl.max()),
            "finest_cell_invA": float(
                integ.mesh.diameter(np.array([integ.mesh.lvl.max()]))[0]
            ),
            "tolerance": float(params.get("qgeom_tolerance") or 1e-3),
            "converged": converged,
            "confirmed": confirmed,
            "stop": integ.stop,
            "history": integ.history[-12:],
            "smeared_check_rel": smeared_rel,
            "wall_seconds": round(res["seconds"], 1),
        },
        "occupied": occupied,
        "headline": headline,
        "dipole_sources": res.get("dipole_sources"),
        "symmetry": symmetry,
        "verdict": verdict,
        "verdict_reasons": reasons,
    }
    summary["dft_check"] = _qgeom_dft_check(ctx, kernel, primary)
    summary["parity"] = _qgeom_parity(
        ctx, kernel, diag, (lo, hi), n_occ, bool(res["S_applies"])
    )
    _qgeom_outputs(
        ctx,
        kernel,
        res,
        summary,
        primary,
        diag if full_level else None,
        path,
        mu_out,
        frame,
    )
    summary["integration"]["step_seconds"] = round(time.time() - t_start, 1)
    return summary


def _bcd_residual(integ, bcd: np.ndarray, mu: float) -> Optional[float]:
    """|D| over its absolute Fermi-surface scale sum int (-f') |v Omega| at mu."""
    fixed = getattr(integ, "_scales_fixed", None)
    if not fixed:
        return None
    subset = np.asarray(integ.mu_subset, float)
    j = int(np.argmin(np.abs(subset - mu)))
    if abs(subset[j] - mu) > 1e-9:
        return None
    scale = float(np.max(fixed[0]["fs"][j][7:9]))
    return float(np.abs(bcd).max() / scale) if scale > 1e-300 else 0.0


def _state_at_cnp(occ: Dict[str, Any], min_gap: float) -> str:
    gap = occ.get("global_gap")
    if gap is None:
        return "unknown"
    if gap >= min_gap:
        return "insulator"
    if gap > -min_gap:
        return "semimetal"
    return "metal"


def _qgeom_path_data(ctx, kernel, primary, diag, path, res) -> Dict[str, Any]:
    """Window-band geometry along the display path at both levels."""
    lo, hi = res["window"]
    n_occ = int(res["n_occ"])
    w = int(ctx.params.get("qgeom_band_window") or 1)
    show = [b for b in range(n_occ - w, n_occ + w) if lo <= b < hi] or [lo]
    cuts = [lo] if lo > 0 else []

    def evaluate_both(kfrac):
        out = {}
        for name, table in (("main", primary), ("diag", diag)):
            if table is None:
                continue
            r = kernel.evaluate(table, kfrac, window=(lo, hi), occupied=cuts)
            out[name] = {
                "E": r["E"][:, lo:hi],
                "v": r["v"],
                "trg": np.trace(r["g"], axis1=2, axis2=3),
                "omega": r["omega"],
                "G": r["G"],
                "omega_C": r["blocks"][lo]["omega"] if cuts else np.zeros(len(kfrac)),
            }
        return out

    _STATE["qgeom_path_fn"] = evaluate_both
    workers, chunk = ctx.parallel(
        "quantum geometry path", 20, len(path["kpoints"]), n_orb=ctx.n_orb
    )
    parts = _pmap(_w_qgeom_path, _chunks(path["kpoints"], chunk), workers)
    data: Dict[str, Any] = {}
    sel = [b - lo for b in show]
    mus = {"E_F": 0.0}
    cnp0 = getattr(ctx, "_qgeom_cnp0", None)
    if cnp0 is not None:
        mus["CNP"] = cnp0
    if ctx.params.get("path_mu_meV") is not None:
        mus["path_mu"] = float(ctx.params["path_mu_meV"]) * 1e-3
    for level in ("main", "diag"):
        if level not in parts[0]:
            continue
        cat = {k: np.concatenate([p[level][k] for p in parts]) for k in parts[0][level]}
        lvl = {
            "bands": show,
            "E_eV": cat["E"][:, sel].T,
            "v_eVA": np.moveaxis(cat["v"][:, sel], 1, 0),
            "trace_g_A2": cat["trg"][:, sel].T,
            "omega_A2": cat["omega"][:, sel].T,
            "G_A2_per_eV": np.moveaxis(cat["G"][:, sel], 1, 0),
        }
        lvl["fermi_sea_omega_A2"] = {
            name: (np.sum(cat["omega"] * (cat["E"] < mu), axis=1) + cat["omega_C"])
            for name, mu in mus.items()
        }
        data[level] = lvl
    return data


def _qgeom_outputs(
    ctx, kernel, res, summary, primary, diag, path, mu_out, frame
) -> None:
    """Out/qgeom_response.json, Out/qgeom_path.json and the plots, within the budget."""
    temps = res["temperatures_K"]
    final = res["stages"].get("tau2") or res["stages"]["tau"]
    first = res["stages"]["tau"]
    n_occ = int(res["n_occ"])
    area = kernel.cell_area(ctx.lattice)
    cnp = summary["electron_count"]["mu_cnp_meV"]
    ctx._qgeom_cnp0 = cnp["0K"] * 1e-3 if cnp.get("0K") is not None else None
    response: Dict[str, Any] = {
        "units": kernel.units(),
        "conventions": summary["conventions"],
        "mu_eV": mu_out,
        "temperatures_K": temps,
        "per_T": {},
    }
    for i, T in enumerate(temps):
        cur = final["levels"]["main"]["per_T"][i]
        prev = first["levels"]["main"]["per_T"][i]
        entry: Dict[str, Any] = {
            "mu_minus_cnp_eV": (
                (mu_out - cnp[f"{T:g}K"] * 1e-3)
                if cnp.get(f"{T:g}K") is not None
                else None
            ),
            "density_cm2": kernel.density_cm2(cur["count"] - n_occ, area),
            "dos": cur["dos"],
            "drude_eV": cur["drude"].reshape(len(mu_out), 4)[:, [0, 1, 3]],
            "err": {
                "A3": np.abs(
                    kernel.trigonal(cur["M"])["A3"] - kernel.trigonal(prev["M"])["A3"]
                ),
                "M": np.abs(cur["M"] - prev["M"]).reshape(len(mu_out), 8),
                "sigma_xy": np.abs(cur["sigma_xy_e2_h"] - prev["sigma_xy_e2_h"]),
            },
            "levels": {},
        }
        for level, key in (
            ("full" if diag is not None else "diag", "main"),
            ("diag", "diag"),
        ):
            lv = final["levels"].get(key)
            if lv is None or (key == "diag" and diag is None):
                continue
            c = lv["per_T"][i]
            tr = kernel.trigonal(c["M"])
            entry["levels"][level] = {
                "M": c["M"].reshape(len(mu_out), 8)[:, [0, 1, 3, 4, 5, 7]],
                "A3": tr["A3"],
                "theta0_deg": tr["theta0_deg"],
                "sigma_xy_e2_h": c["sigma_xy_e2_h"],
                "BCD": c["BCD"],
            }
        response["per_T"][f"{T:g}K"] = entry
    response["M_components"] = ["xxx", "xxy", "xyy", "yxx", "yxy", "yyy"]
    path_out: Dict[str, Any] = {}
    try:
        if not len(path["kpoints"]):
            raise PythtbError(ERR_BAD_PARAMS, "the path is empty")
        pdata = _qgeom_path_data(ctx, kernel, primary, diag, path, res)
        b = kernel.reciprocal_in_plane(ctx.lattice, frame)
        kc = np.asarray(path["kpoints"])[:, :2] @ b
        dist = path.get("distance")
        if dist is None:
            dist = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(kc, axis=0), axis=1))]
        path_out = {
            "k_distance_invA": dist,
            "k_distance_note": "1/angstrom with 2 pi (BandPlot uses k/2pi)",
            "tick_indices": path.get("tick_indices") or [],
            "tick_labels": path.get("tick_labels") or [],
            "levels": pdata,
        }
    except PythtbError as exc:
        ctx.warnings.append(f"no path data: {exc}")
    files = {}
    budget = QGEOM_OUT_BUDGET_BYTES - 512 * 1024  # room for the plots
    stride = 1
    while True:
        payload = (
            kernel.decimate(response, stride, len(mu_out)) if stride > 1 else response
        )
        size = kernel.write_json_compact(
            os.path.join(OUT_DIR, "qgeom_response.json"), payload
        )
        if size <= 0.8 * budget or stride >= 64:
            break
        stride *= 2
    summary["grid"]["downsampled"] = stride
    files["response"] = os.path.join(OUT_DIR, "qgeom_response.json")
    ctx.add_output(files["response"], "Quantum geometry response curves (data)", "data")
    if path_out:
        kernel.write_json_compact(os.path.join(OUT_DIR, "qgeom_path.json"), path_out)
        files["path"] = os.path.join(OUT_DIR, "qgeom_path.json")
        ctx.add_output(files["path"], "Quantum geometry along the path (data)", "data")
    summary["files"] = files
    _qgeom_plots(ctx, kernel, response, path_out, summary, mu_out)


def _qgeom_plots(ctx, kernel, response, path_out, summary, mu_out) -> None:
    try:
        plt = ctx._plt()
    except Exception as exc:  # noqa: BLE001 - a missing matplotlib only costs the PNGs
        ctx.warnings.append(f"no plots: {exc}")
        return
    try:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for name, entry in response["per_T"].items():
            for level, style in (("full", "-"), ("diag", "--")):
                lv = entry["levels"].get(level)
                if lv is None:
                    continue
                axes[0].plot(mu_out * 1e3, lv["A3"], style, label=f"{level} {name}")
                axes[1].plot(
                    mu_out * 1e3, lv["sigma_xy_e2_h"], style, label=f"{level} {name}"
                )
        axes[0].set_xlabel("mu - E_F (meV)")
        axes[0].set_ylabel("A3 of the BCP dipole (A/eV)")
        axes[1].set_xlabel("mu - E_F (meV)")
        axes[1].set_ylabel("sigma_xy (e^2/h)")
        axes[0].legend(fontsize=7)
        ctx._save(fig, "qgeom_dipole.png", "BCP dipole amplitude and AHC")
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, entry in response["per_T"].items():
            ax.plot(mu_out * 1e3, entry["dos"], label=name)
        ax.set_xlabel("mu - E_F (meV)")
        ax.set_ylabel("DOS (states/eV/cell)")
        ax.legend(fontsize=7)
        ctx._save(fig, "qgeom_ahc_dos.png", "Density of states")
        if path_out:
            lv = path_out["levels"].get("main")
            fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
            for row, b in zip(lv["E_eV"], lv["bands"]):
                axes[0].plot(path_out["k_distance_invA"], row, lw=1, label=f"band {b}")
            for name, om in lv["fermi_sea_omega_A2"].items():
                axes[1].plot(path_out["k_distance_invA"], om, lw=1, label=name)
            axes[0].set_ylabel("E - E_F (eV)")
            axes[1].set_ylabel("Fermi-sea Omega (A^2)")
            axes[1].set_xlabel("k (1/A)")
            axes[1].legend(fontsize=7)
            ctx._save(
                fig, "qgeom_path.png", "Bands and Fermi-sea curvature along the path"
            )
    except Exception as exc:  # noqa: BLE001 - a plot failure must not fail the step
        ctx.warnings.append(f"plotting failed: {exc}")


STEP_RUNNERS: Dict[str, Callable[[Context], None]] = {
    STEP_BANDS: run_bands,
    STEP_TOPOLOGY: run_topology,
    STEP_BERRY: run_berry,
    STEP_SLAB: run_slab,
    STEP_QGEOM: run_qgeom,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(params_path: str = PARAMS_FILENAME) -> int:
    """Run one PythTB step in the current directory; always writes the results file."""
    started = time.time()
    log = _Log(LOG_FILENAME)
    results: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "step_type": None,
        "provenance": {
            "core_sha256": sha256_file(os.path.abspath(__file__)),
            "params_sha256": sha256_file(params_path),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "host": platform.node(),
            "cpus": _available_cpus(),
            "memory_available_bytes": _available_memory_bytes(),
        },
        "citation": CITATION,
        "approximation": APPROXIMATION_NOTE,
    }
    ctx: Optional[Context] = None
    code = 0
    try:
        try:
            with open(params_path) as handle:
                params = json.load(handle)
        except (OSError, ValueError) as exc:
            raise PythtbError(ERR_BAD_PARAMS, f"cannot read {params_path}: {exc}")
        results["step_type"] = params.get("step_type")
        try:
            import pythtb  # noqa: F401
        except ImportError as exc:
            raise PythtbError(
                ERR_ENV_MISSING,
                f"pythtb is not importable ({exc}); install it in the env named in the "
                "host's PythTB settings: conda install -c conda-forge pythtb=2.0.2 scipy",
            )
        results["provenance"]["pythtb"] = _package_version("pythtb")
        results["provenance"]["scipy"] = _package_version("scipy")
        results["provenance"]["matplotlib"] = _package_version("matplotlib")
        if os.path.exists(QGEOM_MODULE_FILENAME):
            results["provenance"]["qgeom_sha256"] = sha256_file(QGEOM_MODULE_FILENAME)
        os.makedirs(OUT_DIR, exist_ok=True)
        ctx = Context(params, log)
        results["step_type"] = ctx.step
        log(f"PythTB {results['provenance']['pythtb']} - step {ctx.step}")
        ctx.build_model()
        results.update(ctx.results)
        ctx.validate()
        results.update(ctx.results)
        STEP_RUNNERS[ctx.step](ctx)
        results.update(ctx.results)
        results["status"] = "ok"
    except PythtbError as exc:
        code = 2
        results["status"] = "failed"
        results["error"] = {"code": exc.code, "message": str(exc)}
        log(f"FAILED [{exc.code}] {exc}")
    except MemoryError as exc:
        code = 3
        results["status"] = "failed"
        results["error"] = {
            "code": ERR_MEMORY,
            "message": f"out of memory ({exc}); lower chunk_size/workers or the mesh sizes",
        }
        log("FAILED [out of memory]")
    except Exception as exc:  # the results file must exist whatever happened
        code = 1
        results["status"] = "failed"
        results["error"] = {
            "code": ERR_INTERNAL,
            "message": f"{type(exc).__name__}: {exc}",
        }
        log("FAILED with an unexpected error:\n" + traceback.format_exc())
    finally:
        if ctx is not None:
            results.update(ctx.results)
            results["warnings"] = ctx.warnings
            results["outputs"] = ctx.outputs
            results["provenance"]["inputs"] = ctx.inputs
            results["provenance"]["parallel"] = ctx.parallel_plan
        results["provenance"]["wall_seconds"] = round(time.time() - started, 2)
        results["provenance"].update(_peak_rss())
        shutil.rmtree(MODEL_DIR, ignore_errors=True)
        write_json_atomic(RESULTS_FILENAME, results)
        log(f"wrote {RESULTS_FILENAME} (status {results['status']})")
    return code


def probe() -> int:
    """Print the env this core would run with (for a host check)."""
    info = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("pythtb", "scipy", "matplotlib"):
        info[name] = _package_version(name)
    print(json.dumps(info))
    return 0 if info["pythtb"] else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="_pythtb_core.py")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("pythtb-run", help="run the step in pythtb_params.json")
    run_parser.add_argument("--params", default=PARAMS_FILENAME)
    sub.add_parser("pythtb-probe", help="print the versions of the cluster env")
    args = parser.parse_args(argv)
    if args.command == "pythtb-run":
        return run(args.params)
    return probe()


if __name__ == "__main__":
    sys.exit(main())
