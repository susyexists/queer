"""The QUEER_QGEOM step: quantum geometry and the intrinsic nonlinear Hall response, as the PythTB driver runs it.

:func:`run_qgeom` follows the Generator driver's ``run_qgeom`` on the model of
:class:`~queer.compat.generator.wannier_steps.WannierStepContext`, with the
integrator of :mod:`queer.observables.qgeom`:

- refusals (``queer_bad_params``): a model periodic along anything but a1 and a2,
  hopping cutoffs, a3 off the layer normal, a full level whose normal is not along
  z, a mu grid of more than 4001 points and temperatures below 0.5 K;
- two position levels on the folded table of the model stage, E_F subtracted on
  its R = 0 diagonal: the diagonal level r = diag(tau) with tau PythTB's orbital
  positions (the tight-binding model), and the full level from the WANNIER run's
  ``wannier90_tb.dat``. ``position_operator`` ``auto`` takes the full level when
  the file is staged and otherwise warns; ``full`` without it is
  ``queer_position_unavailable``. The file must have its header's exact size, the
  ``_hr.dat``'s R set and degeneracies, H within 2e-6 eV of the model after the same
  Wigner-Seitz unfold and r_ii(0) within 2e-5 Å of the centres in-plane, else
  ``queer_position_mismatch``; r is symmetrised and its anti-Hermitian part reported;
- the integration of :func:`~queer.observables.qgeom.adaptive_response` over the mu
  grid at T = 0 and each temperature, at ``qgeom_tolerance`` and at half of it, the
  occupation pass also sampling the display path; the time budget
  (``time_budget_s``) stops it unless ``qgeom_time_stop`` is false or
  ``qgeom_replay_vertices`` sets a deterministic stop at that vertex count;
- the summary block ``qgeom`` of the driver (headline points E_F, mu_CNP and the A3
  maximum per temperature, the occupied manifold, symmetry residuals, dipole
  sources, checks and the verdict), ``Out/qgeom_response.json`` and
  ``Out/qgeom_path.json`` within the driver's 10 MB budget, and the figures
  ``qgeom_dipole.png``, ``qgeom_ahc_dos.png`` and ``qgeom_path.png``.

Where the driver compares its diagonal level with PythTB's own metric and
curvature, QUEER checks it against its own band geometry
(:func:`~queer.observables.geometry.band_pair_geometry` and
:func:`~queer.observables.geometry.berry_curvature` of one band): an internal
consistency check, labelled as such in ``qgeom.parity``. QUEER reads the ``.win``'s
``kpoints`` block, so its ``.eig`` check runs where the driver's is always skipped.
``integration`` adds ``replay`` (the stop controls), ``stop_vertices`` (the vertex
count when the integration stopped) and ``vertex_set`` (count and SHA-256 of the
vertex keys); ``Out/qgeom_vertices.json`` holds the keys for a replay comparison.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ...io.wannier import read_eig, read_tb_layout, read_tb_records
from ...observables import qgeom as obs
from ...observables.geometry import band_pair_geometry, berry_curvature
from ...physics import qgeom as phys
from .estimators import comparison_window
from .memory import chunks, pmap
from .plots import new_figure, save_figure
from .results import (
    APPROXIMATION_NOTE,
    ERR_BAD_PARAMS,
    ERR_POSITION_MISMATCH,
    ERR_POSITION_UNAVAILABLE,
    OUT_DIR,
    WARN_DFT_EIG_UNREADABLE,
    WARN_DFT_FROZEN_MISMATCH,
    WARN_DIPOLE_SOURCES,
    WARN_PLOT_SKIPPED,
    WARN_POSITION_ANTIHERMITIAN,
    WARN_POSITION_DIAGONAL_ONLY,
    WARN_QGEOM_NO_PATH,
    WARN_QGEOM_PATH_DATA,
    WARN_QGEOM_PROBE,
    WARN_SMEARED_COARSE,
    RunnerError,
    source_sha256,
)
from .wannier_steps import WannierStepContext

TB_FILENAME = "wannier90_tb.dat"
EIG_FILENAME = "wannier90.eig"
VERTICES_FILE = f"{OUT_DIR}/qgeom_vertices.json"
FULL_RESPONSE_FILE = f"{OUT_DIR}/qgeom_response_full.json"
FULL_RESPONSE_NOTE = (
    "The curves of qgeom_response.json (mu_eV and per_T, at its stride) unrounded: a comparison record, "
    "QUEER-only; qgeom_response.json is these values to 5 significant digits"
)
FULL_POSITION_NOTE = (
    "full position matrix in Wannier space (wannier90 _tb.dat): curvature quantities "
    "complete to the finite-difference accuracy of r; the metric and K are in-W lower bounds"
)
QGEOM_OUT_BUDGET_BYTES = 10 * 1024 * 1024
# The vertex-weight Fermi-surface integrals at the highest temperature against the
# Theta convolution: a larger difference means the mesh is coarse against kT.
QGEOM_SMEARED_WARN = 0.05
# The anti-Hermitian part of a finite-difference position matrix on a 12x12 mesh is
# ~0.1 of its largest element; warn only well beyond that.
ANTI_HERMITIAN_WARN = 0.15
# The checks of the full position level.
TB_H_TOLERANCE_EV = 2e-6
TB_CENTRE_TOLERANCE_A = 2e-5
TB_LATTICE_TOLERANCE_A = 1e-5
MAX_MU_POINTS = 4001
MIN_TEMPERATURE_K = 0.5
PROBE_SEED = 20261010
PROBE_POINTS = 32
PROBE_TOLERANCE = 1e-8
SUM_RULE_SEED = 7
SUM_RULE_TOLERANCE = 1e-8
# Vertex keys written to the replay file at most; beyond it only their digest.
VERTEX_KEYS_MAX = 250_000
# The QUEER sources whose digest names the quantum-geometry code a run executed.
QGEOM_SOURCES = ("physics/qgeom.py", "observables/qgeom.py", "compat/generator/qgeom_step.py")
PROBE_REFERENCE = "queer.observables.geometry band_pair_geometry and berry_curvature (engine kubo)"


# ------------------------------------------------------------------ output
def _round_sig(value: float, digits: int | None) -> float | None:
    if not math.isfinite(value):
        return None
    if value == 0.0:
        return 0.0
    if digits is None:
        return value
    return float(f"{value:.{digits}g}")


def compact(value: Any, digits: int | None = 5) -> Any:
    """A JSON-ready copy with floats to ``digits`` significant digits (None: unrounded) and NaN or infinity
    as None."""
    if isinstance(value, dict):
        return {str(k): compact(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [compact(v, digits) for v in value]
    if isinstance(value, np.ndarray):
        return compact(value.tolist(), digits)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _round_sig(float(value), digits)
    return value


def write_json_compact(path: str | Path, payload: Any, digits: int | None = 5) -> int:
    """Write ``payload`` without indentation and with ``digits`` significant digits; returns its size.

    Significant digits, not decimal places: these curves span decades. The file is
    written under a temporary name and renamed.
    """
    text = json.dumps(compact(payload, digits), separators=(",", ":"), allow_nan=False)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as handle:
        handle.write(text)
    os.replace(tmp, path)
    return len(text.encode())


def decimate(curves: Any, stride: int, length: int) -> Any:
    """Every array or list whose leading axis is the mu grid (``length``) taken at ``stride``; anything
    else unchanged."""
    if isinstance(curves, dict):
        return {key: decimate(value, stride, length) for key, value in curves.items()}
    if isinstance(curves, np.ndarray) and curves.ndim >= 1 and curves.shape[0] == length:
        return curves[::stride]
    if (
        isinstance(curves, list)
        and len(curves) == length
        and curves
        and not isinstance(curves[0], (str, dict))
    ):
        return curves[::stride]
    return curves


def units() -> dict[str, Any]:
    """The unit block of the response file."""
    return {
        "energy": "eV, relative to the source E_F",
        "k": "1/angstrom, 2 pi included",
        "M": "angstrom/eV",
        "BCD": "angstrom",
        "chi": "mA nm / V^2",
        "chi_per_M": phys.CHI_MA_NM_PER_V2_PER_A_PER_EV,
        "sigma_xy": "e^2/h",
        "dos": "states / eV / cell",
        "drude": "eV",
        "density": "cm^-2",
        "e2_over_h_S": phys.E2_OVER_H_S,
        "e2_over_hbar_S": phys.E2_OVER_HBAR_S,
    }


# ------------------------------------------------------------- parallelism
class QgeomParallel:
    """Splits the integrator's k-point batches over QUEER's worker pool, as the driver splits them."""

    def __init__(self, context: WannierStepContext, n_orb: int):
        self.options = context.parallel("quantum geometry", 20, 10**7, n_orb=n_orb)
        self.workers, self.cap = int(self.options.workers), int(self.options.batch_size)

    def _map(self, function: Callable[[np.ndarray], Any], kfrac: np.ndarray) -> list[Any]:
        n = len(kfrac)
        chunk = max(1, min(self.cap, math.ceil(n / max(1, self.workers))))
        return pmap(function, chunks(kfrac, chunk), self.options)

    def payload(self, serial: Callable[[np.ndarray], Mapping[str, np.ndarray]]):
        def run(kfrac: np.ndarray) -> dict[str, np.ndarray]:
            parts = self._map(serial, kfrac)
            return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}

        return run

    def energies(self, table: obs.QgeomTable):
        def run(kfrac: np.ndarray) -> np.ndarray:
            return np.concatenate(self._map(table.energies, kfrac))

        return run


# ------------------------------------------------------------ position level
def full_table(
    ctx: WannierStepContext, frame: Mapping[str, np.ndarray], H: np.ndarray
) -> tuple[obs.QgeomTable, dict[str, Any]]:
    """The full-position table from the WANNIER run's own ``_tb.dat``, checked against ``_hr.dat`` and
    ``_centres.xyz``; returns (table, position report)."""
    path = ctx.workspace / TB_FILENAME
    try:
        lay = read_tb_layout(path)
    except (OSError, ValueError, IndexError) as error:
        raise RunnerError(ERR_POSITION_MISMATCH, f"{TB_FILENAME} is unreadable: {error}") from error
    problems = []
    if lay.num_wann != ctx.n_orb:
        problems.append(f"num_wann {lay.num_wann} differs from _hr.dat's {ctx.n_orb}")
    if not lay.lines_ok or lay.expected_bytes != lay.actual_bytes:
        problems.append(
            f"{lay.actual_bytes} bytes, expected {lay.expected_bytes} from its header "
            "(truncated, or not in wannier90 3.1's format)"
        )
    lat_err = float(np.max(np.abs(lay.lattice - ctx.lattice)))
    if lat_err > TB_LATTICE_TOLERANCE_A:
        problems.append(f"its lattice differs from the .win's by {lat_err:.2e} A")
    if problems:
        raise RunnerError(ERR_POSITION_MISMATCH, f"{TB_FILENAME}: " + "; ".join(problems))
    t0 = time.time()
    tb = read_tb_records(path, lay)
    ctx.record.record_input(TB_FILENAME)
    tb_deg = {tuple(int(x) for x in r): int(d) for r, d in zip(tb.R, lay.ndegen)}
    if tb_deg != ctx.hr_degeneracy:
        raise RunnerError(
            ERR_POSITION_MISMATCH,
            f"{TB_FILENAME} has another R set or other degeneracies than wannier90_hr.dat: "
            "the two files are not from the same wannier90 run",
        )
    nd = lay.ndegen.astype(float)
    H_tb = tb.H / nd[:, None, None]
    r = tb.r / nd[:, None, None, None]
    r, anti = phys.hermitize_translations(tb.R, r)
    i0 = int(np.flatnonzero(~tb.R.any(axis=1))[0])
    centres_r = np.stack([np.real(np.diag(r[i0][..., a])) for a in range(2)], axis=1)
    centres_r = np.concatenate([centres_r, np.zeros((len(centres_r), 1))], axis=1)
    centre_err = phys.inplane_mismatch(centres_r, ctx.centres, ctx.lattice)
    stack = np.concatenate([H_tb[..., None], r], axis=-1)
    R_u, S_u = ctx.ws_apply(tb.R, stack)
    if R_u.shape != ctx.table_R.shape or not np.array_equal(R_u, ctx.table_R):
        raise RunnerError(
            ERR_POSITION_MISMATCH, f"{TB_FILENAME} unfolds onto another R set than wannier90_hr.dat"
        )
    h_err = float(np.max(np.abs(S_u[..., 0] - ctx.table_H)))
    offdiag = r.copy()
    idx = np.arange(ctx.n_orb)
    offdiag[i0][idx, idx] = 0.0
    off_max = float(np.max(np.abs(offdiag)))
    sum_r2 = float(np.sum(nd[:, None, None, None] * np.abs(offdiag) ** 2))
    checks: dict[str, Any] = {
        "h_vs_hr_max_meV": h_err * 1e3,
        "centres_inplane_max_A": centre_err,
        "antihermitian_rel": anti / off_max if off_max > 0 else 0.0,
        "offdiag_sum_xy_A2": sum_r2,
        "rz_offdiag_sum_A2": tb.rz2_offdiag,
    }
    omega_I = ctx.params.get("wannier_omega_I_A2")
    if omega_I is not None:
        checks["omega_I_A2"] = float(omega_I)
    if h_err > TB_H_TOLERANCE_EV:
        raise RunnerError(
            ERR_POSITION_MISMATCH,
            f"H in {TB_FILENAME} differs from wannier90_hr.dat by {h_err * 1e3:.3g} meV",
        )
    if centre_err > TB_CENTRE_TOLERANCE_A:
        raise RunnerError(
            ERR_POSITION_MISMATCH,
            f"r_ii(0) in {TB_FILENAME} differs from the Wannier centres by {centre_err:.3g} A",
        )
    if checks["antihermitian_rel"] > ANTI_HERMITIAN_WARN:
        ctx.record.warn(
            WARN_POSITION_ANTIHERMITIAN,
            f"the position matrix is {checks['antihermitian_rel']:.0%} anti-Hermitian before "
            "symmetrising: a coarse Wannier k-mesh; positions are approximate",
        )
    proj = np.stack([frame["e1"][:2], frame["e2"][:2]], axis=1)
    r_frame = S_u[..., 1:3] @ proj
    table = obs.QgeomTable(ctx.table_R, H, ctx.lattice, frame, r=r_frame)
    report = {
        "mode": "full",
        "source": "wannier90_tb_dat",
        "tb": {
            "bytes": int(lay.actual_bytes),
            "expected_bytes": int(lay.expected_bytes),
            "nrpts": int(lay.nrpts),
            "read_seconds": round(time.time() - t0, 1),
        },
        "checks": checks,
    }
    ctx.log(
        f"{TB_FILENAME}: {lay.nrpts} R; H vs hr {h_err * 1e3:.2g} meV; centres "
        f"{centre_err:.2g} A; anti-Hermitian {checks['antihermitian_rel']:.3f}"
    )
    return table, report


def single_band_probe(
    ctx: WannierStepContext, diag: obs.QgeomTable, window: tuple[int, int], n_occ: int
) -> dict[str, Any]:
    """The diagonal level's metric and curvature of one band against QUEER's own band geometry at 32 k.

    The band is the driver's single-band choice, the one below the occupied count
    held inside the window; its partner for :func:`band_pair_geometry` is the band
    above (below for the top band). An internal consistency check: both sides are
    QUEER's, the diagonal level being the exact-diagonal embedding of the model's
    centres. A degenerate band has no single-band geometry and is skipped.
    """
    rng = np.random.default_rng(PROBE_SEED)
    k = np.c_[rng.uniform(-0.5, 0.5, (PROBE_POINTS, 2)), np.zeros(PROBE_POINTS)]
    band = int(min(max(window[0], (n_occ or 1) - 1), window[1] - 1))
    partner = band + 1 if band + 1 < ctx.n_orb else band - 1
    block: dict[str, Any] = {
        "check": "internal_consistency",
        "reference": PROBE_REFERENCE,
        "k_points": PROBE_POINTS,
        "bands": [band],
    }
    if partner < 0:
        block["skipped"] = "a one-band model has no band pair"
        return block
    ours = obs.evaluate(diag, k, window=(band, band + 1))
    g_ours, om_ours = ours["g"][:, 0], ours["omega"][:, 0]
    pair = (min(band, partner), max(band, partner))
    try:
        geometry = band_pair_geometry(ctx.model, k, bands=pair, degeneracy_tol=1e-6)
        curvature = berry_curvature(ctx.model, k, bands=[band], engine="kubo", degeneracy_tol=1e-6)
    except ValueError as error:
        block["skipped"] = f"band {band} is degenerate at a probe point: {error}"
        return block
    P = np.stack([diag.frame["e1"], diag.frame["e2"]])
    metric = geometry["total_metric"][pair.index(band)]
    g_ref = np.einsum("ia,kab,jb->kij", P, metric, P)
    om_ref = np.einsum("a,kab,b->k", diag.frame["e1"], curvature["curvature"][0], diag.frame["e2"])
    valid = np.isfinite(g_ref).all(axis=(1, 2)) & np.isfinite(om_ref)
    if not valid.any():
        block["skipped"] = f"band {band} is degenerate at every probe point"
        return block
    scale = max(float(np.abs(g_ref[valid]).max()), float(np.abs(om_ref[valid]).max()), 1e-300)
    dev = (
        max(
            float(np.abs(g_ours[valid] - g_ref[valid]).max()),
            float(np.abs(om_ours[valid] - om_ref[valid]).max()),
        )
        / scale
    )
    if dev > PROBE_TOLERANCE:
        ctx.record.warn(
            WARN_QGEOM_PROBE, f"the diagonal level differs from QUEER's band geometry by {dev:.2g} (relative)"
        )
    block.update(max_rel=dev, valid_points=int(valid.sum()))
    return block


def dft_check(ctx: WannierStepContext, table: obs.QgeomTable) -> dict[str, Any] | None:
    """Tight-binding eigenvalues against the staged ``.eig`` inside the frozen window."""
    path = ctx.workspace / EIG_FILENAME
    if not path.exists():
        return None
    kpts = np.asarray(ctx.win.kpoints, float).reshape(-1, 3)
    froz = (ctx.win.dis_froz_min, ctx.win.dis_froz_max)
    if not len(kpts) or froz[0] is None or froz[1] is None:
        return {"skipped": "the .win has no kpoints block or no frozen window"}
    try:
        E_dft = read_eig(path)
    except (OSError, ValueError) as error:
        # The check is a diagnostic: a file it cannot read skips it, and the step's result stands.
        reason = f"the .eig cannot be read: {error}"
        ctx.record.warn(WARN_DFT_EIG_UNREADABLE, f"no DFT eigenvalue check: {reason}")
        return {"skipped": reason}
    ctx.record.record_input(EIG_FILENAME)
    if len(E_dft) != len(kpts):
        return {"skipped": f"{len(E_dft)} k-points in the .eig, {len(kpts)} in the .win"}
    kpts = kpts.copy()
    kpts[:, ctx.folded_axes] = 0.0
    E_tb = table.energies(kpts) + ctx.fermi
    out = phys.dft_exactness(E_tb, E_dft, (float(froz[0]), float(froz[1])))
    if not out["counts_match"] or out["max_meV"] > 0.1:
        ctx.record.warn(
            WARN_DFT_FROZEN_MISMATCH,
            f"the model misses frozen DFT states by up to {out['max_meV']:.3g} meV at the DFT mesh",
        )
    return out


def path_points(ctx: WannierStepContext, frame: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """The display path: ``path_kpoints_frac`` when given, else the band path plus log-spaced points
    around every tick (``path_log_points`` per side, 1e-4 to 1e-1 1/Å away)."""
    custom = ctx.params.get("path_kpoints_frac")
    if custom:
        pts = np.asarray(custom, float).reshape(-1, 3)[:4000]
        return {"kpoints": pts, "tick_indices": [], "tick_labels": [], "break_indices": []}
    try:
        path = ctx.band_path()
    except RunnerError as error:
        ctx.record.warn(WARN_QGEOM_NO_PATH, f"no k-path for the curves along k: {error}")
        return {"kpoints": np.zeros((0, 3)), "tick_indices": [], "tick_labels": []}
    pts = np.asarray(path["kpoints"], float).reshape(-1, 3)
    b = phys.reciprocal_in_plane(ctx.lattice, frame)
    kc = pts[:, :2] @ b
    seg = np.r_[0.0, np.linalg.norm(np.diff(kc, axis=0), axis=1)]
    for i in path.get("break_indices") or []:
        if 0 < i < len(seg):
            seg[i] = 0.0
    dist = np.cumsum(seg)
    n_log = int(ctx.params.driver_value("path_log_points", 0))
    extra_d: list[float] = []
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
        ticks = [int(np.flatnonzero(order == i)[0]) for i in path.get("tick_indices") or []]
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


def mu_cnp(count: np.ndarray, mu: np.ndarray, n_occ: int) -> float | None:
    """The mu where the electron count reaches ``n_occ`` (interpolated in count order), or None outside."""
    if not len(count) or n_occ < count.min() or n_occ > count.max():
        return None
    order = np.argsort(count, kind="stable")
    return float(np.interp(n_occ, count[order], mu[order]))


def chi_amplitudes(M: np.ndarray) -> dict[str, float]:
    """Each convention's chi (mA nm/V^2) as a signed trigonal amplitude along M's."""
    tm = phys.trigonal(M)
    A3 = float(tm["A3"])
    out = {}
    for name, chi in phys.chi_all_conventions(M).items():
        tc = phys.trigonal(chi)
        out[name] = float((tc["alpha"] * tm["alpha"] + tc["beta"] * tm["beta"]) / A3) if A3 > 0 else 0.0
    return out


def bcd_residual(integ: obs.Integrator, bcd: np.ndarray, mu: float) -> float | None:
    """|D| over its absolute Fermi-surface scale sum int (-f') |v Omega| at ``mu``."""
    fixed = integ.scales_fixed
    if not fixed:
        return None
    subset = np.asarray(integ.mu_subset, float)
    j = int(np.argmin(np.abs(subset - mu)))
    if abs(subset[j] - mu) > 1e-9:
        return None
    scale = float(np.max(fixed[0]["fs"][j][7:9]))
    return float(np.abs(bcd).max() / scale) if scale > 1e-300 else 0.0


def state_at_cnp(occ: Mapping[str, Any], min_gap: float) -> str:
    gap = occ.get("global_gap")
    if gap is None:
        return "unknown"
    if gap >= min_gap:
        return "insulator"
    if gap > -min_gap:
        return "semimetal"
    return "metal"


def vertex_set(integ: obs.Integrator) -> tuple[dict[str, Any], dict[str, Any]]:
    """The mesh's vertex set: its summary (count, unit and SHA-256 of the sorted keys) and the replay file."""
    mesh = integ.mesh
    keys = np.sort(mesh.vertex_keys()).astype("<i8")
    digest = hashlib.sha256(keys.tobytes()).hexdigest()
    summary = {
        "count": int(len(keys)),
        "base": int(mesh.base),
        "unit": int(mesh.unit),
        "sha256": digest,
        "file": VERTICES_FILE,
    }
    content = {
        "description": "Vertex keys x * unit + y of the final quadtree, x and y in units of 1/unit from -1/2",
        **{key: summary[key] for key in ("count", "base", "unit", "sha256")},
        "keys": keys.tolist() if len(keys) <= VERTEX_KEYS_MAX else None,
    }
    return summary, content


# ---------------------------------------------------------------- the step
def run_qgeom(ctx: WannierStepContext) -> None:
    """Quantum geometry, the metric (BCP) dipole and the intrinsic nonlinear Hall response of a 2D model,
    integrated on an adaptive mesh at two position levels."""
    params = ctx.params
    t_start = time.time()
    if len(ctx.periodic) != 2 or list(ctx.periodic) != [0, 1]:
        raise RunnerError(ERR_BAD_PARAMS, "quantum geometry needs a 2D model periodic along a1 and a2")
    if params.get("min_hopping_norm") or params.get("max_distance"):
        raise RunnerError(ERR_BAD_PARAMS, "quantum geometry uses the full hopping table: drop the cutoffs")
    if phys.a3_misalignment(ctx.lattice) > 1e-6:
        raise RunnerError(ERR_BAD_PARAMS, "a3 must be along a1 x a2 (the layer normal)")
    frame = phys.layer_frame(ctx.lattice)
    # The Hamiltonian with E_F on the diagonal; the centres as PythTB places the orbitals.
    H = np.asarray(ctx.table_H, complex).copy()
    i0 = int(np.flatnonzero(~np.asarray(ctx.table_R).any(axis=1))[0])
    H[i0] -= ctx.fermi * np.eye(ctx.n_orb)
    orbs = np.asarray(ctx.centres, float) @ np.linalg.inv(ctx.lattice)
    tau_cart = np.asarray(orbs, float) @ ctx.lattice
    tau = np.stack([tau_cart @ frame["e1"], tau_cart @ frame["e2"]], axis=1)
    diag = obs.QgeomTable(ctx.table_R, H, ctx.lattice, frame, tau=tau)
    mode = str(params.driver_value("position_operator", "auto")).lower()
    full = None
    position: dict[str, Any] = {"mode": "diagonal", "source": "none"}
    if mode not in ("auto", "full", "diagonal"):
        raise RunnerError(ERR_BAD_PARAMS, f"position_operator {mode!r} is not auto, full or diagonal")
    if mode != "diagonal":
        if (ctx.workspace / TB_FILENAME).exists():
            if abs(abs(frame["normal"][2]) - 1.0) > 1e-6:
                raise RunnerError(ERR_BAD_PARAMS, "the full level needs the layer normal along z")
            full, position = full_table(ctx, frame, H)
        elif mode == "full":
            raise RunnerError(
                ERR_POSITION_UNAVAILABLE,
                f"no {TB_FILENAME}: re-run the WANNIER step (new runs write it by default) "
                "or choose position_operator = diagonal",
            )
        else:
            ctx.record.warn(
                WARN_POSITION_DIAGONAL_ONLY,
                f"the WANNIER job has no {TB_FILENAME}: the diagonal (tight-binding) level only; "
                "re-run the WANNIER step for the full position matrix",
            )
    primary = full or diag
    mu_lo, mu_hi = (float(x) * 1e-3 for x in params.driver_value("mu_range_meV", [-100.0, 100.0]))
    mu_step = float(params.driver_value("mu_step_meV", 0.1)) * 1e-3
    n_mu = int(round((mu_hi - mu_lo) / mu_step)) + 1
    if not (mu_hi > mu_lo) or n_mu > MAX_MU_POINTS:
        raise RunnerError(ERR_BAD_PARAMS, f"the mu grid has {n_mu} points (at most {MAX_MU_POINTS})")
    mu_out = mu_lo + mu_step * np.arange(n_mu)
    temps = sorted(float(T) for T in (params.get("temperatures_K") or []))
    if any(T < MIN_TEMPERATURE_K for T in temps):
        raise RunnerError(ERR_BAD_PARAMS, f"temperatures must be at least {MIN_TEMPERATURE_K} K")
    path = path_points(ctx, frame)
    parallel = QgeomParallel(ctx, ctx.n_orb)
    budget = float(params.driver_value("time_budget_s", 0)) or None
    replay_vertices = params.get("qgeom_replay_vertices")
    time_stop = bool(params.get("qgeom_time_stop", True)) and replay_vertices is None
    deadline = (time.time() + budget) if (budget and time_stop) else None
    res = obs.adaptive_response(
        primary,
        mu_out=mu_out,
        temperatures_K=temps,
        occupied_bands=params.get("occupied_bands"),
        insulator_min_gap=float(params.driver_value("insulator_min_gap_meV", 0.1)) * 1e-3,
        deg_tol=float(params.driver_value("degeneracy_tol_meV", 0.001)) * 1e-3,
        tolerance=float(params.driver_value("qgeom_tolerance", 1e-3)),
        base=int(params.driver_value("qgeom_base_mesh", 64)),
        max_points=int(params.driver_value("qgeom_max_points", 1_000_000)),
        gap_ratio=float(params.driver_value("qgeom_gap_ratio", 0.1)),
        gap_floor=float(params.driver_value("qgeom_gap_floor_meV", 0.05)) * 1e-3,
        edge_resolution=float(params.driver_value("edge_resolution_meV", 0.2)) * 1e-3,
        occupation_mesh=int(params.driver_value("occupation_mesh", 241)),
        tau_diag=tau if full is not None else None,
        deadline=deadline,
        stop_vertices=None if replay_vertices is None else int(replay_vertices),
        frame=frame,
        log=ctx.log,
        parallel=parallel,
        extra_kfrac=path["kpoints"],
    )
    res["dipole_sources"] = qgeom_dipole_sources(ctx, res, primary, parallel, mu_out)
    summary = qgeom_summary(ctx, res, full is not None, position, diag, primary, frame, path, mu_out, t_start)
    summary["integration"]["replay"] = {
        "time_stop": deadline is not None,
        "time_budget_s": budget,
        "stop_vertices": None if replay_vertices is None else int(replay_vertices),
    }
    integ = res["integrator"]
    summary["integration"]["stop_vertices"] = integ.stop_at_vertices
    vertices, content = vertex_set(integ)
    summary["integration"]["vertex_set"] = vertices
    write_json_compact(ctx.workspace / VERTICES_FILE, content)
    ctx.record.add_output(VERTICES_FILE, "Quantum geometry mesh vertices (replay record)", "data")
    ctx.record.provenance["queer_qgeom_sha256"] = source_sha256(list(QGEOM_SOURCES))
    ctx.results["qgeom"] = summary


def qgeom_dipole_sources(
    ctx: WannierStepContext, res: Mapping[str, Any], primary: obs.QgeomTable, parallel: QgeomParallel, mu_out
) -> dict[str, Any] | None:
    """The partner-gap shares of the T = 0 dipole at E_F, mu_CNP and the A3 maximum (snapped to the
    output grid). A diagnostic: a failure only warns."""
    final = res["stages"].get("tau2") or res["stages"]["tau"]
    cur = final["levels"]["main"]["per_T"][0]
    wanted = {
        "E_F": 0.0,
        "CNP": mu_cnp(cur["count"], mu_out, int(res["n_occ"])),
        "max_A3": float(mu_out[int(np.argmax(phys.trigonal(cur["M"])["A3"]))]),
    }
    points = {
        name: float(mu_out[int(np.argmin(np.abs(mu_out - mu)))])
        for name, mu in wanted.items()
        if mu is not None and mu_out[0] <= mu <= mu_out[-1]
    }
    window = tuple(res["window"])
    deg_tol = float(ctx.params.driver_value("degeneracy_tol_meV", 0.001)) * 1e-3
    frozen = comparison_window(ctx.windows, ctx.fermi)

    def fields(kfrac: np.ndarray) -> dict[str, np.ndarray]:
        return obs.source_fields(primary, kfrac, window=window, deg_tol=deg_tol, frozen=frozen)

    t0 = time.time()
    try:
        rows = obs.dipole_sources(res["integrator"], points, parallel.payload(fields))
    except Exception as error:  # a diagnostic never fails the step
        ctx.record.warn(WARN_DIPOLE_SOURCES, f"no dipole sources: {error}")
        return None
    ctx.log(f"dipole sources at {sorted(points)} ({time.time() - t0:.1f}s)")
    return {
        "level": primary.level,
        "T_K": 0.0,
        "gap_bins_meV": list(phys.GAP_BIN_EDGES_MEV),
        "frozen_window_meV": [frozen[0] * 1e3, frozen[1] * 1e3],
        "points": rows,
    }


def qgeom_summary(
    ctx: WannierStepContext,
    res: dict[str, Any],
    full_level: bool,
    position: dict[str, Any],
    diag: obs.QgeomTable,
    primary: obs.QgeomTable,
    frame: Mapping[str, np.ndarray],
    path: dict[str, Any],
    mu_out: np.ndarray,
    t_start: float,
) -> dict[str, Any]:
    """The driver's ``qgeom`` block; writes the data files and figures."""
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
    area = phys.cell_area(ctx.lattice)
    cnp = {T: mu_cnp(lvl_main["per_T"][i]["count"], mu_out, n_occ) for i, T in enumerate(temps)}
    headline: dict[str, Any] = {}
    quotable_err = True
    for i, T in enumerate(temps):
        cur = lvl_main["per_T"][i]
        prev = first["levels"]["main"]["per_T"][i]
        A3 = phys.trigonal(cur["M"])["A3"]
        points = {"E_F": 0.0, "CNP": cnp[T], "max_A3": float(mu_out[int(np.argmax(A3))])}
        rows = {}
        for name, mu in points.items():
            if mu is None or not (mu_out[0] <= mu <= mu_out[-1]):
                continue
            j = int(np.argmin(np.abs(mu_out - mu)))
            M, Mp = cur["M"][j], prev["M"][j]
            tr, trp = phys.trigonal(M), phys.trigonal(Mp)
            chi, chip = chi_amplitudes(M), chi_amplitudes(Mp)
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
                "density_cm2": float(phys.density_cm2(cur["count"][j] - n_occ, area)),
            }
            if lvl_diag is not None:
                Ad = float(phys.trigonal(lvl_diag["per_T"][i]["M"][j])["A3"])
                row["full_over_diag_A3"] = float(tr["A3"]) / Ad if Ad > 0 else None
            rows[name] = row
        headline[f"{T:g}K"] = rows
    occupied: dict[str, Any] = {"applicable": bool(res["S_applies"]), "n_occ": n_occ}
    vals = integ.mesh.vals
    failed: list[str] = []
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
            occupied["K_upper_loose"] = float(np.trace(K_in)) + two_pi * float(omega_I) / area
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
            occupied["reason"] = "C not an integer: an under-resolved gap" if confirmed else "not confirmed"
    else:
        occupied["verdict"] = "not_applicable"
        occupied["reason"] = "the gap above the occupied bands closes somewhere"
    sample = np.c_[np.random.default_rng(SUM_RULE_SEED).uniform(-0.5, 0.5, (8, 2)), np.zeros(8)]
    sum_rule = obs.sum_rule_residual(primary, sample)
    if sum_rule > SUM_RULE_TOLERANCE:
        failed.append(f"sum_n G^n = {sum_rule:.2g} (relative) at sampled k")
    M0 = lvl_main["per_T"][0]["M"]
    j0 = int(np.argmin(np.abs(mu_out)))
    tr0 = phys.trigonal(M0[j0])
    symmetry: dict[str, Any] = {
        "lattice_hexagonal": bool(phys.is_hexagonal(ctx.lattice)),
        "c3_band_floor_meV": (occ["c3_band_floor"] * 1e3) if occ.get("c3_band_floor") is not None else None,
        "M_c3_residual": phys.c3_residual_tensor(M0[j0]),
        "sigma_dT_residual": (float(abs(tr0["beta"]) / tr0["A3"]) if tr0["A3"] > 0 else None),
        "bcd_residual": bcd_residual(integ, lvl_main["per_T"][0]["BCD"][j0], float(mu_out[j0])),
    }
    if res["S_applies"]:
        symmetry["K_c3_residual"] = phys.c3_residual_matrix(np.array(occupied["K_in"]))
    smeared_rel = (res.get("smeared_check") or {}).get("rel")
    if smeared_rel is not None and smeared_rel > QGEOM_SMEARED_WARN:
        ctx.record.warn(
            WARN_SMEARED_COARSE,
            f"the Fermi-surface integrals at {temps[-1]:g} K from vertex weights differ "
            f"from the Theta convolution by {100 * smeared_rel:.1f}%: the mesh is coarse "
            "against kT there",
        )
    froz = comparison_window(ctx.windows, ctx.fermi)
    in_window = bool(froz[0] <= mu_out[0] and mu_out[-1] <= froz[1])
    pt = occ["pt_pairs"]
    reasons: list[str] = []
    if not full_level:
        reasons.append("diagonal (tight-binding) position level")
    if not converged or not confirmed:
        reasons.append("integration not converged and confirmed")
    if not quotable_err:
        reasons.append("A3 error above max(5% A3, 1e-3 of its scale) at a headline point")
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
    insulator_min_gap = float(params.driver_value("insulator_min_gap_meV", 0.1)) * 1e-3
    summary: dict[str, Any] = {
        "approximation": FULL_POSITION_NOTE if full_level else APPROXIMATION_NOTE,
        "approximation_level": ("full_position_in_wannier_space" if full_level else "diagonal_position"),
        "conventions": {
            "omega": "-2 Im Q",
            "sigma_xy": "-(e^2/h)(1/2pi) sum int f Omega, charge -e",
            "chi": "-(e^3/hbar)[c_out M_abc + c_in(M_bac + M_cab)]",
            "chi_sets": {k: list(v) for k, v in phys.CONVENTION_SETS.items()},
            "k_units": "1/angstrom, 2 pi included",
            "frame": {k: [float(x) for x in v] for k, v in frame.items()},
        },
        "position": position,
        "electron_count": {
            "n_occ": n_occ,
            "source": ("occupied_bands" if params.get("occupied_bands") else "count_at_EF"),
            "count_at_EF": occ["count_at_EF"],
            "gap_at_gamma_meV": (
                (occ["gap_at_gamma"] * 1e3) if occ.get("gap_at_gamma") is not None else None
            ),
            "min_direct_gap_meV": (
                (occ["min_direct_gap"] * 1e3) if occ.get("min_direct_gap") is not None else None
            ),
            "global_gap_meV": ((occ["global_gap"] * 1e3) if occ.get("global_gap") is not None else None),
            "state_at_cnp": state_at_cnp(occ, insulator_min_gap),
            "mu_cnp_meV": {f"{T:g}K": (v * 1e3 if v is not None else None) for T, v in cnp.items()},
        },
        "window": {
            "bands": [lo, hi],
            "always_occupied": lo,
            "degeneracy_tol_used_meV": float(params.driver_value("degeneracy_tol_meV", 0.001)),
            "pt_pairs": pt,
        },
        "grid": {
            "mu_min_meV": float(mu_out[0]) * 1e3,
            "mu_max_meV": float(mu_out[-1]) * 1e3,
            "mu_step_meV": (float(mu_out[1] - mu_out[0]) * 1e3 if len(mu_out) > 1 else None),
            "temperatures_K": temps,
            "downsampled": 1,
        },
        "integration": {
            "base_mesh": int(params.driver_value("qgeom_base_mesh", 64)),
            "seeds": int(len(occ["seeds"])),
            "mu_subset": int(len(integ.mu_subset)),
            "vertices": int(final["vertices"]),
            "vertices_by_rule": dict(integ.mesh.vertices_by_rule),
            "gap_rule_capped": bool(integ.gap_rule_capped),
            "max_level": int(integ.mesh.lvl.max()),
            "finest_cell_invA": float(integ.mesh.diameter(np.array([integ.mesh.lvl.max()]))[0]),
            "tolerance": float(params.driver_value("qgeom_tolerance", 1e-3)),
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
    summary["dft_check"] = dft_check(ctx, primary)
    summary["parity"] = single_band_probe(ctx, diag, (lo, hi), n_occ)
    qgeom_outputs(ctx, res, summary, primary, diag if full_level else None, path, mu_out, frame)
    summary["integration"]["step_seconds"] = round(time.time() - t_start, 1)
    return summary


def path_data(
    ctx: WannierStepContext,
    primary: obs.QgeomTable,
    diag: obs.QgeomTable | None,
    path: Mapping[str, Any],
    res: Mapping[str, Any],
    cnp0: float | None,
) -> dict[str, Any]:
    """Window-band geometry along the display path at both levels."""
    lo, hi = res["window"]
    n_occ = int(res["n_occ"])
    w = int(ctx.params.driver_value("qgeom_band_window", 1))
    show = [b for b in range(n_occ - w, n_occ + w) if lo <= b < hi] or [lo]
    cuts = [lo] if lo > 0 else []

    def evaluate_both(kfrac: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
        out = {}
        for name, table in (("main", primary), ("diag", diag)):
            if table is None:
                continue
            r = obs.evaluate(table, kfrac, window=(lo, hi), occupied=cuts)
            out[name] = {
                "E": r["E"][:, lo:hi],
                "v": r["v"],
                "trg": np.trace(r["g"], axis1=2, axis2=3),
                "omega": r["omega"],
                "G": r["G"],
                "omega_C": r["blocks"][lo]["omega"] if cuts else np.zeros(len(kfrac)),
            }
        return out

    options = ctx.parallel("quantum geometry path", 20, len(path["kpoints"]), n_orb=ctx.n_orb)
    parts = pmap(evaluate_both, chunks(path["kpoints"], options.batch_size), options)
    data: dict[str, Any] = {}
    sel = [b - lo for b in show]
    mus = {"E_F": 0.0}
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


def qgeom_outputs(
    ctx: WannierStepContext,
    res: Mapping[str, Any],
    summary: dict[str, Any],
    primary: obs.QgeomTable,
    diag: obs.QgeomTable | None,
    path: Mapping[str, Any],
    mu_out: np.ndarray,
    frame: Mapping[str, np.ndarray],
) -> None:
    """``Out/qgeom_response.json``, ``Out/qgeom_path.json`` and the figures, within the budget."""
    temps = res["temperatures_K"]
    final = res["stages"].get("tau2") or res["stages"]["tau"]
    first = res["stages"]["tau"]
    n_occ = int(res["n_occ"])
    area = phys.cell_area(ctx.lattice)
    cnp = summary["electron_count"]["mu_cnp_meV"]
    cnp0 = cnp["0K"] * 1e-3 if cnp.get("0K") is not None else None
    response: dict[str, Any] = {
        "units": units(),
        "conventions": summary["conventions"],
        "mu_eV": mu_out,
        "temperatures_K": temps,
        "per_T": {},
    }
    for i, T in enumerate(temps):
        cur = final["levels"]["main"]["per_T"][i]
        prev = first["levels"]["main"]["per_T"][i]
        entry: dict[str, Any] = {
            "mu_minus_cnp_eV": ((mu_out - cnp[f"{T:g}K"] * 1e-3) if cnp.get(f"{T:g}K") is not None else None),
            "density_cm2": phys.density_cm2(cur["count"] - n_occ, area),
            "dos": cur["dos"],
            "drude_eV": cur["drude"].reshape(len(mu_out), 4)[:, [0, 1, 3]],
            "err": {
                "A3": np.abs(phys.trigonal(cur["M"])["A3"] - phys.trigonal(prev["M"])["A3"]),
                "M": np.abs(cur["M"] - prev["M"]).reshape(len(mu_out), 8),
                "sigma_xy": np.abs(cur["sigma_xy_e2_h"] - prev["sigma_xy_e2_h"]),
            },
            "levels": {},
        }
        for level, key in (("full" if diag is not None else "diag", "main"), ("diag", "diag")):
            lv = final["levels"].get(key)
            if lv is None or (key == "diag" and diag is None):
                continue
            c = lv["per_T"][i]
            tr = phys.trigonal(c["M"])
            entry["levels"][level] = {
                "M": c["M"].reshape(len(mu_out), 8)[:, [0, 1, 3, 4, 5, 7]],
                "A3": tr["A3"],
                "theta0_deg": tr["theta0_deg"],
                "sigma_xy_e2_h": c["sigma_xy_e2_h"],
                "BCD": c["BCD"],
            }
        response["per_T"][f"{T:g}K"] = entry
    response["M_components"] = ["xxx", "xxy", "xyy", "yxx", "yxy", "yyy"]
    path_out: dict[str, Any] = {}
    try:
        if not len(path["kpoints"]):
            raise RunnerError(ERR_BAD_PARAMS, "the path is empty")
        pdata = path_data(ctx, primary, diag, path, res, cnp0)
        b = phys.reciprocal_in_plane(ctx.lattice, frame)
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
    except RunnerError as error:
        ctx.record.warn(WARN_QGEOM_PATH_DATA, f"no path data: {error}")
    files = {}
    out_dir = ctx.workspace / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = QGEOM_OUT_BUDGET_BYTES - 512 * 1024  # room for the plots
    stride = 1
    while True:
        payload = decimate(response, stride, len(mu_out)) if stride > 1 else response
        size = write_json_compact(out_dir / "qgeom_response.json", payload)
        if size <= 0.8 * budget or stride >= 64:
            break
        stride *= 2
    summary["grid"]["downsampled"] = stride
    files["response"] = f"{OUT_DIR}/qgeom_response.json"
    ctx.record.add_output(files["response"], "Quantum geometry response curves (data)", "data")
    # The same curves unrounded, so that two QUEER runs compare beyond the 5 digits (about twice the size).
    full = {"description": FULL_RESPONSE_NOTE, "mu_eV": payload["mu_eV"], "per_T": payload["per_T"]}
    write_json_compact(ctx.workspace / FULL_RESPONSE_FILE, full, digits=None)
    ctx.record.add_output(
        FULL_RESPONSE_FILE, "Quantum geometry response curves, unrounded (comparison record)", "data"
    )
    if path_out:
        write_json_compact(out_dir / "qgeom_path.json", path_out)
        files["path"] = f"{OUT_DIR}/qgeom_path.json"
        ctx.record.add_output(files["path"], "Quantum geometry along the path (data)", "data")
    summary["files"] = files
    qgeom_plots(ctx, response, path_out, mu_out)


def _figure(size: tuple[float, float]) -> Any:
    figure = new_figure(size)
    if figure is None:
        raise RuntimeError("matplotlib is not installed")
    return figure


def qgeom_plots(
    ctx: WannierStepContext, response: Mapping[str, Any], path_out: Mapping[str, Any], mu_out
) -> None:
    """The driver's three figures; a missing Matplotlib or a failed figure only warns."""
    if new_figure((1, 1)) is None:
        ctx.record.warn(WARN_PLOT_SKIPPED, "no plots: matplotlib is not installed")
        return
    try:
        figure = _figure((11, 4))
        axes = figure.subplots(1, 2)
        for name, entry in response["per_T"].items():
            for level, style in (("full", "-"), ("diag", "--")):
                lv = entry["levels"].get(level)
                if lv is None:
                    continue
                axes[0].plot(mu_out * 1e3, lv["A3"], style, label=f"{level} {name}")
                axes[1].plot(mu_out * 1e3, lv["sigma_xy_e2_h"], style, label=f"{level} {name}")
        axes[0].set_xlabel("mu - E_F (meV)")
        axes[0].set_ylabel("A3 of the BCP dipole (A/eV)")
        axes[1].set_xlabel("mu - E_F (meV)")
        axes[1].set_ylabel("sigma_xy (e^2/h)")
        axes[0].legend(fontsize=7)
        save_figure(ctx.record, figure, "qgeom_dipole.png", "BCP dipole amplitude and AHC")
        figure = _figure((6, 4))
        ax = figure.subplots()
        for name, entry in response["per_T"].items():
            ax.plot(mu_out * 1e3, entry["dos"], label=name)
        ax.set_xlabel("mu - E_F (meV)")
        ax.set_ylabel("DOS (states/eV/cell)")
        ax.legend(fontsize=7)
        save_figure(ctx.record, figure, "qgeom_ahc_dos.png", "Density of states")
        if path_out:
            lv = path_out["levels"].get("main")
            figure = _figure((7, 6))
            axes = figure.subplots(2, 1, sharex=True)
            for row, b in zip(lv["E_eV"], lv["bands"]):
                axes[0].plot(path_out["k_distance_invA"], row, lw=1, label=f"band {b}")
            for name, om in lv["fermi_sea_omega_A2"].items():
                axes[1].plot(path_out["k_distance_invA"], om, lw=1, label=name)
            axes[0].set_ylabel("E - E_F (eV)")
            axes[1].set_ylabel("Fermi-sea Omega (A^2)")
            axes[1].set_xlabel("k (1/A)")
            axes[1].legend(fontsize=7)
            save_figure(ctx.record, figure, "qgeom_path.png", "Bands and Fermi-sea curvature along the path")
    except Exception as error:  # a plot failure never fails the step
        ctx.record.warn(WARN_PLOT_SKIPPED, f"plotting failed: {error}")
