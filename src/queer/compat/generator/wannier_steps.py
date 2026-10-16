"""The model stage of QUEER's Generator Wannier steps, with the PythTB driver's semantics.

:class:`WannierStepContext` reads the staged ``wannier90`` files of a step's
workspace and builds the tight-binding model exactly as the Generator's PythTB
driver does before any step runs:

- refusals: a missing ``.win``, ``_hr.dat`` or ``_centres.xyz`` and a ``.win``
  without ``unit_cell_cart`` (``queer_missing_input``),
  ``translate_home_cell = true`` (``queer_translate_home_cell``), a step without
  a Fermi energy and a model periodic along fewer than two axes
  (``queer_bad_params``);
- the lattice PythTB reads from the driver's rewritten ``.win`` (ten decimals;
  a Bohr cell converted with the driver's factor) and the centres as written;
- the HR table divided by its degeneracies, in lexicographic translation order;
- the Wigner-Seitz correction: a staged wsvec file under ``auto`` or ``file``;
  else ``recompute`` (wannier90 3.x images over ``mp_grid`` supercells) or
  ``off``. ``auto`` scores both against ``_band.dat`` inside the comparison
  window by their largest in-window deviation, a tie going to ``recompute``;
  without the band files it follows ``ws_hint`` and the ``.win``'s
  ``use_ws_distance`` (wannier90 3.x default: on). Without ``mp_grid`` the
  correction is off;
- periodic directions from the raw table (or ``periodic_dirs``), hops folded off
  the vacuum axes, and the averaged Hermitian model with PythTB's cutoffs. The
  model block records ``hermiticity_weyl_bound``, which bounds every eigenvalue
  difference between that model and PythTB's, the positive half of the table
  completed by its conjugates.

:meth:`WannierStepContext.validate` adds the driver's checks: the deviation from
``_band.dat`` (warning and failure thresholds), whether E_F lies in the frozen
window, the folded table's Hermiticity error and the hopping decay.
:meth:`WannierStepContext.resolve_occupation` classifies the gap on the driver's
mesh plus the band path for the steps that need an occupied band set.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ...execution import ExecutionOptions
from ...io.wannier import (
    WannierData,
    WinData,
    band_path_breaks,
    fold_vacuum_axes,
    normalized_hopping,
    periodic_dirs_from_table,
    read_band_dat,
    read_band_kpt,
    read_centres_xyz,
    read_hr,
    read_labelinfo,
    read_win,
    read_wsvec,
    recompute_wsvec,
    unfold_from_wsvec,
)
from ...models.wannier import WannierModel
from ...physics.operators import table_hermiticity
from .estimators import (
    BandEdges,
    band_deviation,
    classify_gap,
    comparison_window,
    hopping_decay,
    orbital_species,
    project_path,
    sample_path,
    uniform_mesh,
    unwrap_centres,
)
from .memory import ParallelPlanner, model_fixed_bytes
from .pythtb_params import WannierStepParameters
from .results import (
    APPROXIMATION_NOTE,
    ERR_BAD_PARAMS,
    ERR_MISSING_INPUT,
    ERR_TRANSLATE_HOME_CELL,
    ERR_VALIDATION,
    LOG_FILENAME,
    OUT_DIR,
    WARN_BAND_DAT_DEVIATION,
    WARN_BAND_DAT_MISSING,
    WARN_FERMI_OUTSIDE_FROZEN,
    WARN_PLOT_SKIPPED,
    WARN_WS_ASSUMED_DEFAULT,
    WARN_WS_NO_MP_GRID,
    RunLog,
    RunnerError,
    RunRecord,
    queer_identity,
)

SEEDNAME = "wannier90"
WIN_FILENAME = f"{SEEDNAME}.win"
HR_FILENAME = f"{SEEDNAME}_hr.dat"
CENTRES_FILENAME = f"{SEEDNAME}_centres.xyz"
BAND_KPT_FILENAME = f"{SEEDNAME}_band.kpt"
BAND_DAT_FILENAME = f"{SEEDNAME}_band.dat"
LABELINFO_FILENAME = f"{SEEDNAME}_band.labelinfo.dat"
WSVEC_FILENAME = f"{SEEDNAME}_wsvec.dat"
REQUIRED_INPUTS: tuple[str, ...] = (WIN_FILENAME, HR_FILENAME, CENTRES_FILENAME)
WIN_WINDOW_KEYS: tuple[str, ...] = ("dis_froz_min", "dis_froz_max", "dis_win_min", "dis_win_max")
# The deviation statistics an automatic Wigner-Seitz decision reports per candidate.
WS_SCORE_KEYS: tuple[str, ...] = ("in_window_max_meV", "in_window_rms_meV", "max_meV")
DEFAULT_WS_SEARCH_SIZE = 2
DEFAULT_WS_DISTANCE_TOL = 1e-5
# The driver's Bohr radius and the decimals of the cell it writes for PythTB.
DRIVER_BOHR_ANGSTROM = 0.52917721092
CELL_DECIMALS = 10
ONSITE_IMAG_TOL_EV = 1e-6
# Divisions per periodic axis of the occupation mesh, by model dimension.
OCCUPATION_DIVISIONS: Mapping[int, int] = {2: 36, 3: 16}
SCORE_CHUNK = 256
DECAY_PLOT_POINTS = 200_000


def _fortran_float(token: str) -> float:
    return float(token.strip().lower().replace("d", "e"))


def pythtb_lattice(win: WinData) -> np.ndarray | None:
    """The lattice PythTB reads from the driver's rewritten ``.win`` (Å rows), or None without a cell.

    The driver converts a Bohr cell with its own factor (0.52917721092 Å) and
    writes every component with ten decimals for PythTB; the model, its
    Wigner-Seitz images and every reported geometry use these values.
    """
    if win.lattice is None:
        return None
    vectors = np.asarray(win.lattice, dtype=float)
    if win.cell_unit == "bohr":
        rows = list(win.blocks["unit_cell_cart"])[1:4]
        vectors = np.asarray(
            [[_fortran_float(value) * DRIVER_BOHR_ANGSTROM for value in row.split()[:3]] for row in rows]
        )
    return np.asarray([[float(f"{value:.{CELL_DECIMALS}f}") for value in row] for row in vectors])


def table_band_energies(translations, hopping, kpoints, chunk: int = SCORE_CHUNK) -> np.ndarray:
    """(Nk, NB) absolute eigenvalues of a table's plain Fourier sum, as the driver scores WS candidates.

    ``H(k) = sum_R exp(2 pi i k.R) H(R)`` at fractional ``kpoints`` is solved by
    ``numpy.linalg.eigvalsh``, which reads its lower triangle; the table is
    neither averaged nor cut, ``chunk`` k-points at a time.
    """
    points = np.asarray(kpoints, dtype=float).reshape(-1, 3)
    vectors = np.asarray(translations, dtype=float)
    matrices = np.asarray(hopping)
    n = matrices.shape[1]
    flat = matrices.reshape(len(vectors), n * n)
    blocks = []
    for start in range(0, len(points), max(1, int(chunk))):
        phases = np.exp(2j * np.pi * (points[start : start + chunk] @ vectors.T))
        blocks.append(np.linalg.eigvalsh((phases @ flat).reshape(-1, n, n)))
    return np.concatenate(blocks) if blocks else np.zeros((0, n))


def _lexicographic(data: WannierData) -> WannierData:
    """The HR data with its translations in lexicographic order, as the driver tabulates them."""
    translations = np.asarray(data.translations)
    order = np.lexsort(translations.T[::-1])
    if np.array_equal(order, np.arange(len(order))):
        return data
    return WannierData(
        translations[order], np.asarray(data.hopping)[order], np.asarray(data.degeneracies)[order]
    )


class WannierStepContext:
    """The model of one QUEER Wannier step, built and checked with the Generator driver's semantics.

    ``params`` are the resolved step settings and ``workspace`` the directory
    holding the staged ``wannier90`` files; plots go to its ``Out`` directory.
    ``record`` collects the result blocks (``model``, ``validation``), warnings,
    outputs, the size and SHA-256 of every file read and the parallel plans;
    ``log`` receives timestamped stage lines (``queer.log`` in the workspace by
    default). ``memory_bytes`` fixes the memory the parallel plans see.

    After :meth:`build_model`: ``win``; ``lattice`` (PythTB's) and ``centres``
    (Cartesian Å as written); ``fermi``; ``reference``, wannier90's own bands
    relative to E_F with their k-points and ticks, or None; ``ws_report``;
    ``periodic`` and ``folded_axes``; the folded table ``table_R``/``table_H``
    (degeneracy and image weights applied, before averaging and cutoffs); the
    averaged Hermitian ``model``; ``spinors``; ``species_names`` and
    ``species_index``.
    """

    win: WinData
    lattice: np.ndarray
    centres: np.ndarray
    fermi: float
    n_orb: int
    reference: dict[str, Any] | None
    ws_report: dict[str, Any]
    periodic: list[int]
    folded_axes: list[int]
    table_R: np.ndarray
    table_H: np.ndarray
    model: WannierModel
    spinors: bool
    species_names: list[str]
    species_index: np.ndarray

    def __init__(
        self,
        params: WannierStepParameters,
        workspace: str | Path,
        *,
        record: RunRecord | None = None,
        log: Callable[[str], None] | None = None,
        memory_bytes: float | None = None,
    ):
        self.params = params
        self.step = params.step
        self.workspace = Path(workspace)
        self.record = record if record is not None else RunRecord(self.workspace)
        self.log: Callable[[str], None] = log if log is not None else RunLog(self.workspace / LOG_FILENAME)
        self.results = self.record.results
        self.planner = ParallelPlanner(
            params.get("workers"),
            params.get("chunk_size"),
            self.log,
            records=self.record.parallel,
            memory_bytes=memory_bytes,
        )

    def _path(self, name: str) -> Path:
        return self.workspace / name

    @property
    def windows(self) -> dict[str, float | None]:
        """The ``.win`` energy windows (eV, absolute) the comparison window is taken from."""
        return {key: getattr(self.win, key) for key in WIN_WINDOW_KEYS}

    # ------------------------------------------------------------ model
    def build_model(self) -> None:
        """Read the staged files, apply the Wigner-Seitz correction and build the model block."""
        for name in REQUIRED_INPUTS:
            if not self._path(name).exists():
                raise RunnerError(
                    ERR_MISSING_INPUT,
                    f"{name} was not staged; bind it from the WANNIER job (write_hr and write_xyz must be "
                    "on there)",
                )
        self.win = read_win(self._path(WIN_FILENAME))
        self.record.record_input(WIN_FILENAME)
        if self.win.translate_home_cell:
            raise RunnerError(
                ERR_TRANSLATE_HOME_CELL,
                "The WANNIER run set translate_home_cell = true: its centres no longer match the hopping "
                "vectors in _hr.dat. Re-run it with translate_home_cell = false.",
            )
        lattice = pythtb_lattice(self.win)
        if lattice is None:
            raise RunnerError(ERR_MISSING_INPUT, f"{WIN_FILENAME} has no unit_cell_cart block")
        fermi = self.params.get("fermi_energy")
        if fermi is None:
            raise RunnerError(ERR_BAD_PARAMS, "No Fermi energy: set fermi_energy (eV) on the step")
        self.fermi = float(fermi)
        self.lattice = lattice

        started = time.time()
        raw = _lexicographic(read_hr(self._path(HR_FILENAME)))
        self.record.record_input(HR_FILENAME)
        centres, _ = read_centres_xyz(self._path(CENTRES_FILENAME))
        self.record.record_input(CENTRES_FILENAME)
        self.n_orb = int(raw.nbnd)
        if len(centres) != self.n_orb:
            raise ValueError(
                f"{CENTRES_FILENAME} holds {len(centres)} Wannier centres for {self.n_orb} Wannier functions"
            )
        self.centres = np.array(centres, dtype=float)
        n_raw = len(raw.translations)
        self.log(
            f"read {HR_FILENAME}: {self.n_orb} Wannier functions, {n_raw} R vectors "
            f"({time.time() - started:.1f}s)"
        )
        self.reference = self._read_reference()
        translations, hopping, report = self._ws_correct(raw)
        self.ws_report = report
        n_after_ws = len(translations)
        # Periodicity comes from the raw table: the correction routes hops of a
        # thick slab through its vacuum.
        requested = self.params.get("periodic_dirs")
        self.periodic = (
            sorted(int(axis) for axis in requested)
            if requested
            else periodic_dirs_from_table(raw.translations)
        )
        del raw
        if len(self.periodic) < 2:
            raise RunnerError(
                ERR_BAD_PARAMS,
                f"the model is periodic along {self.periodic} only; 2D or 3D models are supported",
            )
        self.folded_axes = [axis for axis in range(3) if axis not in self.periodic]
        self.table_R, self.table_H = fold_vacuum_axes(translations, hopping, self.periodic)
        del translations, hopping
        self.model = WannierModel.from_table(
            self.table_R,
            self.table_H,
            lattice=self.lattice,
            centres=self.centres,
            fermi_energy=self.fermi,
            hermitize="average",
            imag_onsite_tol=ONSITE_IMAG_TOL_EV,
            min_hopping_norm=self.params.get("min_hopping_norm"),
            max_distance=self.params.get("max_distance"),
        )
        cutoffs = dict(self.model.metadata["cutoffs"])
        spinors = self.win.spinors
        self.spinors = bool(spinors if spinors is not None else self.params.get("spinors"))
        self.species_names, self.species_index = orbital_species(
            self.centres, self.params.get("structure"), self.lattice
        )
        self.results["model"] = {
            "num_wann": self.n_orb,
            "n_R": int(n_raw),
            "n_R_after_ws": int(n_after_ws),
            "n_R_model": int(len(self.table_R)),
            "folded_axes": list(self.folded_axes),
            "n_hops": int(cutoffs["kept_hops"]),
            "spinors": self.spinors,
            "periodic_dirs": list(self.periodic),
            "dimension": len(self.periodic),
            "lattice": self.lattice.tolist(),
            "fermi_energy": self.fermi,
            "fermi_source": self.params.get("fermi_source"),
            "ws_correction": report,
            "cutoffs": cutoffs,
            # Bounds every eigenvalue difference between the averaged model and the
            # table's positive half completed by its conjugates (Weyl's inequality).
            "hermiticity_weyl_bound": float(self.model.metadata["hermiticity_weyl_bound"]),
            "species": list(self.species_names),
            "orbital_species": [self.species_names[index] for index in self.species_index],
            "mp_grid": None if self.win.mp_grid is None else list(self.win.mp_grid),
            "win_scalars": {key: value for key, value in self.windows.items() if value is not None},
            "approximation": APPROXIMATION_NOTE,
            "pythtb_version": None,
            "queer_version": queer_identity().get("version"),
            "source_wannier_job_id": self.params.get("source_wannier_job_id"),
            "reference_band_job_id": self.params.get("reference_band_job_id"),
        }
        self.log(
            f"model: N={self.n_orb}, periodic {self.periodic}, spinors={self.spinors}, "
            f"E_F={self.fermi:.4f} eV ({self.params.get('fermi_source')}), "
            f"WS={report.get('mode')} ({time.time() - started:.1f}s)"
        )

    def _read_reference(self) -> dict[str, Any] | None:
        """wannier90's own interpolated bands (relative to E_F), when both band files were staged."""
        if not (self._path(BAND_KPT_FILENAME).exists() and self._path(BAND_DAT_FILENAME).exists()):
            return None
        kpoints = read_band_kpt(self._path(BAND_KPT_FILENAME))
        self.record.record_input(BAND_KPT_FILENAME)
        energies = read_band_dat(self._path(BAND_DAT_FILENAME), self.n_orb)
        self.record.record_input(BAND_DAT_FILENAME)
        n = min(len(kpoints), len(energies))
        if n == 0:
            return None
        ticks: list[int] = []
        labels: list[str] = []
        if self._path(LABELINFO_FILENAME).exists():
            ticks, labels = read_labelinfo(self._path(LABELINFO_FILENAME))
            self.record.record_input(LABELINFO_FILENAME)
        return {
            "kpoints": np.array(kpoints[:n], dtype=float),
            "energies": energies[:n] - self.fermi,
            "tick_indices": [tick for tick in ticks if 0 <= tick < n],
            "tick_labels": [label for tick, label in zip(ticks, labels) if 0 <= tick < n],
        }

    def _ws_correct(self, raw: WannierData) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """The Wigner-Seitz corrected table and its report, by the driver's selection rules."""
        mode = str(self.params.get("ws_correction") or "auto").lower()
        report: dict[str, Any] = {"requested": mode}
        if self._path(WSVEC_FILENAME).exists() and mode in ("auto", "file"):
            wsvec = read_wsvec(self._path(WSVEC_FILENAME))
            translations, hopping = unfold_from_wsvec(raw, wsvec)
            self.record.record_input(WSVEC_FILENAME)
            report["mode"] = "wsvec_file"
            report["hermiticity_error"] = table_hermiticity(translations, hopping)[0]
            return translations, hopping, report
        mp_grid = self.win.mp_grid
        search = int(self.win.ws_search_size or DEFAULT_WS_SEARCH_SIZE)
        tolerance = float(self.win.ws_distance_tol or DEFAULT_WS_DISTANCE_TOL)
        centres, lattice = self.centres, self.lattice

        def unchanged() -> tuple[np.ndarray, np.ndarray]:
            return normalized_hopping(raw, "standard")

        def recomputed() -> tuple[np.ndarray, np.ndarray]:
            images = recompute_wsvec(
                raw, centres, lattice, mp_grid, search_size=search, distance_tolerance=tolerance
            )
            return normalized_hopping(raw, "ws_distance", images)

        candidates: dict[str, Callable[[], tuple[np.ndarray, np.ndarray]]] = {"off": unchanged}
        if mp_grid:
            candidates["recompute"] = recomputed
        elif mode != "off":
            self.record.warn(
                WARN_WS_NO_MP_GRID,
                "The .win has neither mp_grid nor a kpoints block: the Wigner-Seitz correction cannot be "
                "recomputed and is off.",
            )
        if mode == "off" or len(candidates) == 1:
            report["mode"] = "off"
            translations, hopping = unchanged()
            return translations, hopping, report
        if mode == "recompute":
            translations, hopping = recomputed()
            report["mode"] = "recompute"
            report["hermiticity_error"] = table_hermiticity(translations, hopping)[0]
            return translations, hopping, report

        if self.reference is not None:
            window = comparison_window(self.windows, self.fermi)
            scores: dict[str, dict[str, Any]] = {}
            tables: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for name, build in candidates.items():
                tables[name] = build()
                bands = table_band_energies(*tables[name], self.reference["kpoints"]) - self.fermi
                scores[name] = band_deviation(bands, self.reference["energies"], window)
            best = min(
                scores, key=lambda name: (scores[name]["in_window_max_meV"] or 0.0, name != "recompute")
            )
            report.update(
                mode=best,
                selected_by="band_dat",
                scores={name: {key: score[key] for key in WS_SCORE_KEYS} for name, score in scores.items()},
            )
            translations, hopping = tables[best]
        else:
            hinted = self.params.get("ws_hint")
            use = hinted is not False and self.win.use_ws_distance is not False
            name = "recompute" if use else "off"
            translations, hopping = candidates[name]()
            report.update(mode=name, selected_by="ws_hint" if hinted is not None else "wannier90_3_default")
            if name != "off":
                self.record.warn(
                    WARN_WS_ASSUMED_DEFAULT,
                    "No wannier90_band.dat to check the Wigner-Seitz correction against; it was recomputed "
                    "assuming wannier90 3.x defaults.",
                )
        if report["mode"] != "off":
            report["hermiticity_error"] = table_hermiticity(translations, hopping)[0]
        return translations, hopping, report

    # ------------------------------------------------------- validation
    def validate(self) -> None:
        """The driver's model checks; ``queer_validation_failed`` above the failure threshold.

        The validation block is stored before a failure is raised, so a failed
        run still reports the deviation that failed it.
        """
        out: dict[str, Any] = {}
        if self.reference is not None:
            window = comparison_window(self.windows, self.fermi)
            kpoints = self.reference["kpoints"]
            options = self.parallel("validation", 2, len(kpoints))
            bands = self.model.energies(kpoints, execution=options).T
            deviation = band_deviation(bands, self.reference["energies"], window)
            out["band_dat"] = dict(deviation, available=True, n_k=int(len(bands)))
            warn = self.params.get("validation_warn_meV")
            fail = self.params.get("validation_fail_meV")
            worst = deviation.get("in_window_max_meV") or 0.0
            if fail is not None and worst > float(fail):
                self.results["validation"] = out
                raise RunnerError(
                    ERR_VALIDATION,
                    f"The tight-binding model departs from wannier90_band.dat by {worst:.1f} meV inside the "
                    f"frozen window (limit {float(fail):.0f} meV): the staged model files do not belong "
                    "together or the import failed.",
                )
            if warn is not None and worst > float(warn):
                self.record.warn(
                    WARN_BAND_DAT_DEVIATION,
                    f"Model bands differ from wannier90_band.dat by up to {worst:.1f} meV inside the frozen "
                    "window.",
                )
        else:
            out["band_dat"] = {"available": False}
            self.record.warn(
                WARN_BAND_DAT_MISSING,
                "wannier90_band.dat was not staged (bands_plot off on the WANNIER run): the model could not "
                "be checked against wannier90's own interpolation.",
            )
        low, high = self.win.dis_froz_min, self.win.dis_froz_max
        if high is not None:
            inside = (low is None or low <= self.fermi) and self.fermi <= high
            out["fermi_in_frozen_window"] = bool(inside)
            if not inside:
                self.record.warn(
                    WARN_FERMI_OUTSIDE_FROZEN,
                    "E_F lies outside the frozen window: bands near the Fermi level are not guaranteed to "
                    "match DFT.",
                )
        out["hermiticity_error"] = table_hermiticity(self.table_R, self.table_H)[0]
        out["localization"] = self._localization()
        self.results["validation"] = out

    def _localization(self) -> dict[str, Any]:
        """Decay of the largest hopping with the distance between centres unwrapped across the vacuum."""
        centres = unwrap_centres(self.centres, self.lattice, self.folded_axes)
        decay = hopping_decay(self.table_R, self.table_H, centres, self.lattice)
        if decay is None:
            return {}
        self._plot_decay(decay)
        return {"decay_length_A": decay["decay_length_A"], "max_distance_A": decay["max_distance_A"]}

    def _plot_decay(self, decay: Mapping[str, Any]) -> None:
        """``Out/hopping_decay.png`` when Matplotlib is installed; a failed plot only warns."""
        try:
            from matplotlib.figure import Figure
        except ImportError:
            return
        name = "hopping_decay.png"
        try:
            figure = Figure(figsize=(6, 4))
            axes = figure.subplots()
            distance, amplitude = decay["distance"], decay["amplitude"]
            step = max(1, len(distance) // DECAY_PLOT_POINTS)
            axes.scatter(
                distance[::step], amplitude[::step], s=1, alpha=0.3, color="#888888", rasterized=True
            )
            axes.plot(
                decay["bin_centres"],
                decay["bin_peaks"],
                color="#d55e00",
                lw=1.5,
                label="largest |H| per 0.5 Å",
            )
            axes.set_yscale("log")
            axes.set_xlabel("distance between Wannier centres (Å)")
            axes.set_ylabel("|H| (eV)")
            axes.set_title("Hopping decay")
            axes.legend(loc="upper right")
            target = self.workspace / OUT_DIR / name
            target.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(target, dpi=130, bbox_inches="tight")
        except Exception as error:  # a plot never fails a run
            self.record.warn(WARN_PLOT_SKIPPED, f"hopping-decay plot skipped: {error}")
            return
        self.record.add_output(f"{OUT_DIR}/{name}", "Hopping decay with distance", "plot")

    # ----------------------------------------------------------- helpers
    def parallel(
        self,
        stage: str,
        per_k_matrices: int,
        n_items: int,
        *,
        chunk: int | None = None,
        n_orb: int | None = None,
    ) -> ExecutionOptions:
        """Execution options of a stage solving ``n_items`` k-points, logged and recorded.

        A model stage's workers each hold the folded table's matrices; a stage
        solving matrices of its own (a slab's) gives their size as ``n_orb``, and
        one whose tasks cannot be split (a Wilson loop) its task size as ``chunk``.
        """
        if n_orb is None:
            size, fixed = self.n_orb, model_fixed_bytes(len(self.table_R), self.n_orb)
        else:
            size, fixed = int(n_orb), 0.0
        return self.planner.plan(stage, per_k_matrices, n_items, n_orb=size, fixed_bytes=fixed, chunk=chunk)

    def band_path(self) -> dict[str, Any]:
        """The k-path: wannier90's own samples when labelled, else the resolved path, else the .win path.

        A path for a 2D model has its vacuum components zeroed. Returns
        ``kpoints``, ``tick_indices``, ``tick_labels``, ``break_indices`` and
        ``source``; ``queer_bad_params`` when no path is known.
        """
        n_points = int(self.params.driver_value("kpath_points", 400))
        if self.reference is not None and self.reference["tick_indices"]:
            ticks = self.reference["tick_indices"]
            kpoints = self.reference["kpoints"]
            return {
                "kpoints": kpoints,
                "tick_indices": ticks,
                "tick_labels": self.reference["tick_labels"],
                "break_indices": band_path_breaks(kpoints, ticks),
                "source": "wannier90_band_kpt",
            }
        kpath = self.params.get("kpath") or {}
        segments = kpath.get("segments") if isinstance(kpath, Mapping) else None
        source = (kpath.get("source") if isinstance(kpath, Mapping) else None) or "params"
        if not segments and self.win.kpoint_path:
            segments, source = [list(row) for row in self.win.kpoint_path], "win_kpoint_path"
        if segments and len(self.periodic) < 3:
            segments = project_path(segments, self.periodic)
        if not segments:
            raise RunnerError(
                ERR_BAD_PARAMS, "No k-path: the WANNIER run had no kpoint_path and none was resolved"
            )
        path = sample_path(segments, self.lattice, n_points)
        path["source"] = source
        return path

    def resolve_occupation(self, extra_points=None) -> dict[str, Any]:
        """The occupied band set from the driver's mesh plus the band path (gap classification).

        The mesh has 36 divisions per periodic axis for a 2D model and 16 for a
        3D one; ``extra_points`` (fractional) join it. :func:`classify_gap` runs
        with ``fermi_tolerance_eV`` and ``occupied_bands``; a gapped result also
        reports the smallest direct gap above the occupied set
        (``isolation_meV``) and whether it reaches ``isolation_threshold_meV``.
        """
        divisions = OCCUPATION_DIVISIONS[2] if len(self.periodic) == 2 else OCCUPATION_DIVISIONS[3]
        points = uniform_mesh([divisions] * 3, self.periodic)
        if extra_points is not None:
            points = np.concatenate([points, np.asarray(extra_points, dtype=float).reshape(-1, 3)])
        try:
            points = np.concatenate([points, self.band_path()["kpoints"]])
        except RunnerError as error:
            if error.code != ERR_BAD_PARAMS:
                raise
        options = self.parallel("occupation", 3, len(points))
        energies = self.model.energies(points, execution=options).T
        edges = BandEdges(self.n_orb)
        for start in range(0, len(points), options.batch_size):
            stop = start + options.batch_size
            edges.add(points[start:stop], energies[start:stop])
        try:
            info = classify_gap(
                edges,
                float(self.params.driver_value("fermi_tolerance_eV", 0.02)),
                self.params.get("occupied_bands"),
            )
        except ValueError as error:
            raise RunnerError(ERR_BAD_PARAMS, str(error)) from error
        occupied = info.get("n_occ")
        if occupied:
            isolation = float(edges.direct[occupied - 1]) * 1000.0
            info["isolation_meV"] = isolation
            info["isolated"] = bool(
                isolation >= float(self.params.driver_value("isolation_threshold_meV", 5.0))
            )
        return info


def model_stage(
    parameters: WannierStepParameters, record: RunRecord, log: Callable[[str], None]
) -> WannierStepContext:
    """Build and validate the model of a step in ``record``'s workspace: every step's first stage."""
    context = WannierStepContext(parameters, record.workspace, record=record, log=log)
    context.build_model()
    context.validate()
    return context
