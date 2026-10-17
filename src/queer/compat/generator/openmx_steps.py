"""QUEER's OpenMX steps: the anomalous Hall conductivity and the Chern number of a PyATB counterpart.

``QUEER_SHIFT_CURRENT`` shares the model, Fermi energy and failure stages here and
runs in :mod:`~queer.compat.generator.openmx_shift`.
``QUEER_AHC`` and ``QUEER_CHERN_NUMBER`` read the staged scfout and the
``queer_params.json`` of :mod:`~queer.compat.generator.openmx_params`, run the
calculation PyATB's ``AHC`` or ``CHERN_NUMBER`` block asks for with QUEER's own
OpenMX model, and write ``queer_results.json`` plus PyATB's product file
(``Out/AHC/ahc.dat`` or ``Out/Chern_Num/chern_number.dat``, in PyATB's layout,
sign and convention). The workspace receives no ``pyatb_results.json`` or
``manifest.json``.

**Model and Fermi energy.** ``OpenMXModel.from_scfout`` builds the model; the
spin mode comes from the scfout and an explicit ``nspin`` must agree with it
(``queer_bad_params``). ``fermi_energy Auto`` is the scfout's ``ChemP``
(``ChemP * HARTREE_EV``, with ``chemp_hartree`` recorded); an explicit value is
moved to QUEER's Hartree scale, ``E * HARTREE_EV / PYATB_HARTREE_EV``, so both codes
sample the same point (both values are recorded).

**Integration**, as :func:`~queer.compat.generator.translate_pyatb` translates it:

- Grid with a threshold of 1e30 or more (refinement off): the plain
  Gamma-origin grid sum, evaluated k-point by k-point with the functions of
  :func:`~queer.observables.integrated.ahc` (the same sum, to the last bits). The
  per-k curvature gives the flagged set ``F``: the k-points with a level within
  ``delta_EF`` of E_F and, for the direct engine, those with an occupied-empty
  pair (same spin channel) closer than 1e-4 eV, where ``delta_EF = 1e-6 eV *
  HARTREE_EV / PYATB_HARTREE_EV`` (the Fermi-energy agreement allowed between the
  codes) plus the eigenvalue tolerance ``max(1e-8 eV, the model's degeneracy
  tolerance)``. Each point of ``F`` is listed with its contribution ``c_k`` to the
  result (the largest first), ``c_pyatb_rule``, its contribution under PyATB's
  direct-engine cut (the coefficient derivative of a pair closer than 1e-4 eV is
  zero; PyATB's Kubo engine has no cut, so its pairs are not flagged), and
  ``flip_allowance``, the contribution of the levels within ``delta_EF``, which
  either code may occupy.
- Grid with a lower threshold: QUEER's own threshold quadrature (cell
  midpoints, midpoint sub-cells for cells whose curvature norm reaches the
  threshold in Å²), and ``Adaptive``: QUEER's embedded-error cubature with
  PyATB's tolerances. Their nodes differ from PyATB's, so a comparison needs an
  error estimate: ``self_convergence: true`` adds the same integration on the
  doubled grid and records ``e_Q = |sigma(N) - sigma(2N)|``; a cubature also
  records its own embedded error estimate. With ``self_convergence`` a refined
  Grid also records ``pyatb_emulation.refined_grid``: PyATB's own refined sum
  (Gamma-origin nodes, a node whose curvature norm reaches the threshold replaced
  by the sub-grid half a cell below it), evaluated with :func:`hall_rows` on the
  grid and its double, whose difference ``e_P`` is PyATB's discretisation error,
  which PyATB does not write for a Grid. It is a comparison diagnostic, never the
  result. Their budgets (:func:`quadrature_budget`): a threshold quadrature holds
  only its coarse cells and evaluates its sub-grids in batches, so it may refine
  every cell, as PyATB may; a cubature holds every cell it makes, as many as the
  job's memory allows, and stops with ``queer_validation_failed`` past them. A
  coarse grid (or, with ``self_convergence``, its double) this process cannot hold
  is refused before the model is read (``queer_bad_params``).
- The FHS engine: QUEER's FHS plaquette scan over PyATB's energies (the
  ``fhs_energy_window`` offsets, upper end excluded). ``fhs_dimension 0`` evaluates
  the k1-k2 plane only. ``sigma_S_cm`` is the physical Hall vector
  ``-g (e^2/h) sign(det A) sum_i C_i b_i / 2 pi``; ``pyatb_construction_S_cm`` is the
  same Chern numbers as PyATB prints them, ``sum_d (C_d e^2/h 1e8/|a_d|)
  normalize(b_(d+1) x b_(d+2))``, which differs on a non-orthogonal cell and is what
  ``Out/AHC/ahc.dat`` holds. Where an energy shows a varying occupied rank at
  plaquette vertices (inside a band), ``pyatb_emulation.fhs_links`` records PyATB's
  scan with its own link variables (``S + i dk.r`` at the link's start, the opposite
  sign of the first-order term of the cell-periodic overlap ``exp(-i dk.r)`` that
  QUEER's midpoint links expand) beside QUEER's midpoint links, both on the same
  eigenvectors, and ``fermi_sea_check`` QUEER's own Fermi-sea sum on the FHS grid's
  nodes and its FHS on the halved grid at those energies (:func:`_fermi_sea_check`).

**Spin.** PyATB reads only ``Hks[0]`` of a scfout without spin-orbit coupling: the
spin-up Hamiltonian of a collinear SCF, without a spin factor for a
non-spin-polarized one. QUEER reports its physical value and, beside it,
``legacy_single_channel``: the same calculation on channel 0 alone with
occupation one (half the value for a non-spin-polarized SCF).

**Signs.** ``sigma`` is the physical ``-(e^2/hbar) int f Omega``; ``Out/AHC/ahc.dat``
holds ``convention.export_sign`` times it: the sign table's factor (-1 for the
``pyatb`` convention of the direct/Kubo engine) times, for the direct/Kubo engine,
the lattice's handedness (PyATB's prefactor uses the signed reciprocal volume, so a
left-handed lattice flips what it prints). A Chern number is the plane flux over 2 pi in its native
orientation, with no Hall sign; for a metallic plane it is the Fermi-sea flux
(PyATB's definition), flagged with a warning.
"""

from __future__ import annotations

import functools
import math
import struct
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from ...execution import ExecutionOptions
from ...models.evaluation import band_geometry, basis_curvature, reciprocal, spin_degeneracy, volume
from ...models.openmx import OpenMXModel
from ...observables._sampling import kmesh
from ...observables.integration import SUBGRID_BATCH_POINTS
from ...physics.constants import E2_H, E_CHARGE, HBAR
from ...physics.geometry import band_curvatures
from ...physics.statistics import occupations
from .memory import (
    MEMORY_FRACTION,
    ParallelPlanner,
    available_memory_bytes,
    chunks,
    model_fixed_bytes,
    plan_parallelism,
    pmap,
)
from .openmx_params import BUDGET_REFUSAL, STEP_AHC, STEP_CHERN, STEP_SHIFT, OpenMXStepParameters
from .pyatb_conventions import (
    FERMI_TOLERANCE_HARTREE,
    HARTREE_EV,
    PYATB_BOHR_ANGSTROM,
    PYATB_E2_H,
    PYATB_E_CHARGE,
    PYATB_HARTREE_EV,
    PYATB_HBAR,
    pyatb_energy,
    pyatb_length_ratio,
)
from .pyatb_emulation import (
    BlochTable,
    fhs_plaquette_phases,
    gamma_nodes,
    plane_projection,
    pyatb_rule_rows,
    refined_sum,
    vector_projection,
)
from .pyatb_files import (
    AHC_COMPONENTS,
    AHC_FILE,
    CHERN_FILE,
    ahc_scan_text,
    ahc_vector_text,
    chern_text,
    write_text,
)
from .results import (
    CHERN_INTEGER_TOL,
    ERR_BAD_PARAMS,
    ERR_MISSING_INPUT,
    ERR_VALIDATION,
    WARN_CHERN_NOT_INTEGER,
    RunnerError,
    RunRecord,
)

APPROXIMATION_NOTE = (
    "OpenMX localized-orbital model read from the scfout (nonorthogonal basis with its overlap and "
    "position matrices), the model PyATB reads: Berry curvature by the direct (basis-covariant) or Kubo "
    "formula, FHS plaquettes with first-order midpoint position links, the resonant shift current by the "
    "Sipe-Shkrebtii generalized-dipole expression with analytic derivatives. PyATB reads only the spin-up "
    "Hamiltonian of a scfout without spin-orbit coupling; legacy_single_channel holds that value."
)
# An occupied-empty pair closer than this (eV) flags its k-point.
PAIR_GAP_EV = 1e-4
EIGENVALUE_FLOOR_EV = 1e-8
# Flagged k-points listed in the results, the largest contributions first.
FLAGGED_LISTED = 2000
# Complex N x N matrices a curvature evaluation keeps alive per k-point.
CURVATURE_MATRICES = 24
FHS_MATRICES = 16
PHYSICAL_SIGN = "sigma_ab = -(e^2/hbar) integral(f Omega_ab), carrier charge -e"

WARN_SINGLE_CHANNEL = "pyatb_single_channel"
WARN_FLAGGED = "flagged_kpoints"
WARN_SELF_CONVERGENCE_SKIPPED = "self_convergence_skipped"
WARN_ADAPTIVE_COARSE = "adaptive_initial_grid_coarse"
WARN_CHERN_METALLIC = "chern_plane_metallic"
ADAPTIVE_MIN_CELLS = 512
# Bytes QUEER's local quadrature holds in this process per cell it keeps (its bounds,
# midpoint value and, once refined, its sub-grid's mean, with their containers; 0.7-0.9 kB
# measured with every cell refined), and the share of the planned memory those cells may
# use. A threshold quadrature keeps its coarse cells only, a cubature every active cell.
QUADRATURE_CELL_BYTES = 1024
QUADRATURE_MEMORY_SHARE = 0.5


def cell_capacity(memory_bytes: float) -> int:
    """Quadrature cells this process can hold: its share of the planned fraction of ``memory_bytes``."""
    return int(QUADRATURE_MEMORY_SHARE * MEMORY_FRACTION * float(memory_bytes) // QUADRATURE_CELL_BYTES)


def quadrature_budget(
    integration: Mapping[str, Any],
    grid: Sequence[int],
    capacity: int,
    active_axes: Sequence[int] | None = None,
) -> dict[str, int]:
    """``max_cells`` and ``max_points`` of a threshold or cubature integration on ``grid``.

    A threshold quadrature holds only its coarse cells (its sub-grids are evaluated
    in batches of whole cells, each refined cell keeping its mean), so its budgets
    are the request's worst case, every cell refined as PyATB may refine every
    node: ``coarse x sub-grid`` cells and ``coarse x (1 + sub-grid)`` points;
    ``capacity`` bounds its coarse grid before the run
    (:meth:`OpenMXStepContext.plan_budget`). A cubature holds every cell it makes
    and has no worst case: ``capacity`` cells, and enough points that the cells
    bind first (each bisection evaluates two cells of ``2^d + 1`` nodes).
    """
    mesh = [int(value) for value in grid]
    axes = [axis for axis in range(len(mesh)) if mesh[axis] > 1] if active_axes is None else list(active_axes)
    axes = axes or list(range(len(mesh)))
    coarse = int(np.prod(mesh))
    if integration.get("method") == "threshold":
        subdivision = [int(value) for value in integration.get("subdivision") or [4] * len(mesh)]
        refined = int(np.prod([subdivision[axis] for axis in axes]))
        return {"max_cells": coarse * refined, "max_points": coarse * (1 + refined)}
    cells = max(coarse, capacity)
    return {"max_cells": cells, "max_points": (2 ** len(axes) + 1) * 2 * cells}


def _budget_exhausted(error: RuntimeError) -> bool:
    text = str(error)
    return "exhausted max_" in text or "exceeds max_cells" in text


# ---------------------------------------------------------------------------
# Per-k evaluation
# ---------------------------------------------------------------------------


def _gap_across(energies: np.ndarray, occupied: np.ndarray, channels: np.ndarray) -> float:
    """The smallest empty-minus-occupied gap within one spin channel (inf when none straddles)."""
    best = math.inf
    for channel in np.unique(channels):
        selected = channels == channel
        below, above = energies[selected & occupied], energies[selected & ~occupied]
        if len(below) and len(above):
            best = min(best, float(above.min() - below.max()))
    return best


def hall_rows(
    model: Any, mu: float, engine: str, occupied_bands: int | None, points: np.ndarray
) -> np.ndarray:
    """Per k-point: the occupied curvature tensor (9 entries, Å²) and its occupation facts.

    Columns 0-8 are ``sum_n f_n Omega_n`` row-major, with ``f`` the zero-temperature
    occupation ``E < mu`` (or the lowest ``occupied_bands`` levels); 9 the distance
    of the closest level from ``mu`` (inf for a band count); 10 the smallest
    occupied-empty gap within a spin channel; 11 the occupied count. The tensor
    is the one :func:`~queer.observables.integrated.ahc` sums.
    """
    rows = np.empty((len(points), 12))
    mus = np.array([float(mu)])
    for index, point in enumerate(np.asarray(points, float)):
        data = band_geometry(model, point)
        projected = basis_curvature(model, point, data) if engine == "direct" else None
        curvature = band_curvatures(data, engine, projected)
        energies = np.asarray(data["energies"], float)
        if occupied_bands is None:
            weights = occupations(energies[:, None], mus[None], 0.0)
            occupied = energies < mu
            closest = float(np.min(np.abs(energies - mu)))
        else:
            occupied = np.zeros(len(energies), bool)
            occupied[np.argsort(energies, kind="stable")[: int(occupied_bands)]] = True
            weights = occupied.astype(float)[:, None]
            closest = math.inf
        tensor = np.einsum("nm,nab->mab", weights, curvature)
        rows[index, :9] = tensor.ravel()
        rows[index, 9] = closest
        rows[index, 10] = _gap_across(energies, occupied, np.asarray(data["band_channels"]))
        rows[index, 11] = float(occupied.sum())
    return rows


def fermi_sea_sum(model: Any, mus: Any, engine: str, points: np.ndarray) -> np.ndarray:
    """The occupied curvature vectors ``(Omega_yz, Omega_zx, Omega_xy)`` (Å²) summed over ``points``,
    per energy of ``mus``.

    A level is occupied below each energy (``E < mu``, the count an FHS plaquette
    vertex takes); the curvature is :func:`hall_rows`'s for ``engine``.
    """
    potentials = np.asarray(mus, float)
    total = np.zeros((len(potentials), 3))
    for point in np.asarray(points, float):
        data = band_geometry(model, point)
        projected = basis_curvature(model, point, data) if engine == "direct" else None
        curvature = band_curvatures(data, engine, projected)
        weights = occupations(np.asarray(data["energies"], float)[:, None], potentials[None], 0.0)
        total += _vector(np.einsum("nm,nab->mab", weights, curvature))
    return total


def hall_factor(model: Any) -> float:
    """S/cm per Å² of mean occupied curvature: ``-g e^2/hbar 1e8/V`` (``g`` the spin degeneracy)."""
    return -spin_degeneracy(model) * E_CHARGE**2 / HBAR * (1e8 / volume(model))


def raw_curvature_to_s_cm(model: Any) -> float:
    """``e^2/hbar 1e8/V``: a raw curvature integral (Å², no spin factor) in S/cm."""
    return E_CHARGE**2 / HBAR * 1e8 / volume(model)


def single_channel_model(model: OpenMXModel, channel: int = 0) -> OpenMXModel:
    """One spin channel of a model without spin-orbit coupling, occupied once per level.

    Channel 0 is what PyATB reads; channel 1 is the spin-down channel of a
    collinear model (a non-spin-polarized model has only channel 0).
    """
    if model.spin_mode == "soc":
        raise ValueError("a spin-orbit model has no single spin channel")
    if channel not in (0, 1) or (channel == 1 and model.spin_mode != "collinear"):
        raise ValueError("a collinear model has channels 0 and 1, a non-spin-polarized one only 0")
    n = model.n_spatial
    rows = slice(channel * n, (channel + 1) * n)

    def part(blocks: Mapping[tuple[int, int, int], Any]) -> dict[tuple[int, int, int], Any]:
        return {key: block[rows, rows] for key, block in blocks.items()}

    translations = np.array(sorted(set(model.hamiltonian_blocks) | set(model.overlap_blocks)), dtype=int)
    single = OpenMXModel(
        translations,
        part(model.hamiltonian_blocks),
        part(model.overlap_blocks),
        position=tuple(part(axis) for axis in model.position_blocks) if model.position_blocks else None,
        lattice=model.lattice,
        fermi_energy=model.fermi_energy,
        spin_mode="nonmagnetic",
        spin_degeneracy=1,
        atom_positions=model.atom_positions,
        orbital_metadata=list(model.orbital_metadata[rows]),
        execution=model.execution,
        hermiticity_tolerance=model.hermiticity_tolerance,
    )
    single.degeneracy_tolerance = model.degeneracy_tolerance
    single.metadata.update(model.metadata)
    return single


def _doubled(grid: Sequence[int], active: Sequence[int] | None = None) -> list[int]:
    mesh = [int(value) for value in grid]
    axes = [axis for axis in range(3) if mesh[axis] > 1] if active is None else list(active)
    for axis in axes or range(3):
        mesh[axis] *= 2
    return mesh


def _vector(tensor: np.ndarray) -> np.ndarray:
    tensor = np.asarray(tensor, float)
    return np.stack((tensor[..., 1, 2], tensor[..., 2, 0], tensor[..., 0, 1]), axis=-1)


def _components(vectors: np.ndarray) -> dict[str, list[float]]:
    vectors = np.atleast_2d(np.asarray(vectors, float))
    return {name: vectors[:, axis].tolist() for axis, name in enumerate(AHC_COMPONENTS)}


# ---------------------------------------------------------------------------
# Step context
# ---------------------------------------------------------------------------


class OpenMXStepContext:
    """The model of one OpenMX step and the stages that evaluate it.

    ``record`` collects the result blocks, warnings, outputs, the size and SHA-256
    of every file read and the parallel plans; ``log`` receives the stage lines.
    ``memory_bytes`` fixes the memory the parallel plans see.
    """

    model: OpenMXModel
    recipe: dict[str, Any]
    mu: float

    def __init__(
        self,
        params: OpenMXStepParameters,
        record: RunRecord,
        log: Callable[[str], None],
        *,
        memory_bytes: float | None = None,
    ):
        self.params = params
        self.record = record
        self.log = log
        self.results = record.results
        self.memory_bytes = memory_bytes
        self.planner = ParallelPlanner(
            params.workers, None, log, records=record.parallel, memory_bytes=memory_bytes
        )
        self.cap = params.input_parameters.get("max_kpoint_num")
        self.capacity: int | None = None
        self.batch_plans: dict[str, ExecutionOptions] = {}

    # ------------------------------------------------------------ budget
    def plan_budget(self) -> None:
        """Size the quadrature from the requested grids and the job's memory, before any model is read.

        A threshold or cubature integration whose coarse grid (and doubled grid,
        with ``self_convergence``) holds more cells than this process can keep is
        refused with ``queer_bad_params``. The sub-cells of a threshold refinement
        are not held, so they never count against the memory.
        """
        native = self.params.recipe(fermi_energy=0.0)["parameters"]
        if native.get("integration") is None:
            return
        memory = available_memory_bytes() if self.memory_bytes is None else float(self.memory_bytes)
        self.capacity = cell_capacity(memory)
        active = (0, 1) if self.params.step == STEP_CHERN else None
        grids = [("integration grid", [int(value) for value in native["grid"]])]
        if self.params.self_convergence:
            grids.append(("doubled self-convergence grid", _doubled(native["grid"], active)))
        for label, grid in grids:
            cells = int(np.prod(grid))
            if cells > self.capacity:
                raise RunnerError(
                    ERR_BAD_PARAMS,
                    f"the {label} {'x'.join(str(value) for value in grid)} ({cells:,} cells) {BUDGET_REFUSAL} "
                    f"of {self.capacity:,} cells in this job's memory ({memory / 1024**3:.1f} GiB available): "
                    "lower the grid, run without self_convergence, or give the job more memory",
                )

    def quadrature(self, grid: Sequence[int], active_axes: Sequence[int] | None = None) -> dict[str, Any]:
        """The integration options of the step on ``grid`` with this job's budgets."""
        integration = dict(self.recipe["parameters"]["integration"])
        capacity = self.capacity if self.capacity is not None else cell_capacity(available_memory_bytes())
        return {**integration, **quadrature_budget(integration, grid, capacity, active_axes)}

    # ------------------------------------------------------------ model
    def stage_scfout(self) -> Any:
        """The staged scfout's path, its size and SHA-256 recorded; ``queer_missing_input`` when absent."""
        name = self.params.scfout
        path = self.record.workspace / name
        if not path.is_file():
            raise RunnerError(
                ERR_MISSING_INPUT, f"{name} was not staged; bind it from the OpenMX SCF job (scf.scfout)"
            )
        if name not in self.record.inputs:
            self.record.record_input(name)
        return path

    def build_model(self) -> None:
        name = self.params.scfout
        path = self.stage_scfout()
        try:
            model = OpenMXModel.from_scfout(path)
        except (OSError, ValueError, struct.error) as error:
            raise RunnerError(
                ERR_MISSING_INPUT, f"{name} cannot be read as an OpenMX v3 scfout: {error}"
            ) from error
        self.model = model
        self.recipe = self.params.recipe(fermi_energy=model.fermi_energy, spin_mode=model.spin_mode)
        native, meta = self.recipe["parameters"], self.recipe["export_metadata"]
        self.mu = float(native["ef"])
        spin_flag = (model.data.metadata if model.data is not None else {}).get("spin_flag")
        self.results["model"] = {
            "scfout": name,
            "scfout_sha256": self.record.inputs.get(name, {}).get("sha256"),
            "scfout_bytes": self.record.inputs.get(name, {}).get("bytes"),
            "spin_mode": model.spin_mode,
            "spin_p_switch": spin_flag,
            "nspin": 4 if model.spin_mode == "soc" else 1,
            "spin_degeneracy": spin_degeneracy(model),
            "n_basis": int(model.nbnd),
            "n_translations": int(len(model.translations)),
            "lattice_A": np.asarray(model.lattice, float).tolist(),
            "lattice_handedness": self.handedness,
            "cell_volume_A3": volume(model),
            "fermi_energy": self.mu,
            "fermi_source": meta["fermi_source"],
            "fermi_energy_pyatb_eV": meta["fermi_energy_pyatb_eV"],
            "fermi_energy_scf": float(model.fermi_energy),
            "chemp_hartree": float(model.fermi_energy) / HARTREE_EV,
            "hartree_to_ev": HARTREE_EV,
            "pyatb_hartree_to_ev": PYATB_HARTREE_EV,
            "degeneracy_tolerance_eV": float(model.degeneracy_tolerance),
            "electron_count": model.electron_count,
        }
        self.log(
            f"model: {model.nbnd} orbitals ({model.spin_mode}), {len(model.translations)} translations, "
            f"E_F {self.mu:.9f} eV ({meta['fermi_source']})"
        )

    @property
    def handedness(self) -> int:
        """+1 for a right-handed scfout lattice, -1 for a left-handed one (the sign of its determinant)."""
        return 1 if float(np.linalg.det(np.asarray(self.model.lattice, float))) > 0 else -1

    @property
    def delta_ef(self) -> float:
        """Distance from E_F (eV) within which a level flags its k-point."""
        tolerance = max(EIGENVALUE_FLOOR_EV, float(self.model.degeneracy_tolerance))
        return FERMI_TOLERANCE_HARTREE * HARTREE_EV + tolerance

    @property
    def legacy(self) -> bool:
        """Whether PyATB reads a single spin channel of this scfout."""
        return self.model.spin_mode != "soc"

    @functools.cached_property
    def pyatb_model(self) -> OpenMXModel:
        """The model PyATB reads: the scfout's with spin-orbit coupling, else its spin-up channel occupied once."""
        return single_channel_model(self.model) if self.legacy else self.model

    # ------------------------------------------------------------ plans
    def options(
        self, stage: str, model: Any, n_items: int, per_k: int = CURVATURE_MATRICES
    ) -> ExecutionOptions:
        """Execution options of a stage over ``n_items`` k-points, logged and recorded."""
        fixed = model_fixed_bytes(len(model.translations), model.nbnd)
        chunk = None
        if self.cap is not None:
            _, automatic = plan_parallelism(
                model.nbnd,
                per_k,
                self.params.workers,
                None,
                fixed_bytes=fixed,
                n_items=n_items,
                memory_bytes=self.memory_bytes,
            )
            chunk = min(automatic, int(self.cap))
        return self.planner.plan(stage, per_k, n_items, n_orb=model.nbnd, fixed_bytes=fixed, chunk=chunk)

    def batch_options(self, stage: str, model: Any) -> ExecutionOptions:
        """The options of a stage evaluated in batches of up to ``SUBGRID_BATCH_POINTS`` k-points:
        planned, logged and recorded at its first batch, then reused."""
        if stage not in self.batch_plans:
            self.batch_plans[stage] = self.options(stage, model, SUBGRID_BATCH_POINTS)
        return self.batch_plans[stage]

    # ------------------------------------------------------------ grids
    def grid_rows(
        self,
        stage: str,
        model: Any,
        points: np.ndarray,
        engine: str,
        occupied_bands: int | None = None,
        options: ExecutionOptions | None = None,
    ) -> np.ndarray:
        """:func:`hall_rows` of every point, in order, on the planned processes (``options`` when given)."""
        options = self.options(stage, model, len(points)) if options is None else options
        function = functools.partial(hall_rows, model, self.mu, engine, occupied_bands)
        parts = pmap(function, chunks(points, options.batch_size), options)
        return np.concatenate(parts, axis=0) if parts else np.empty((0, 12))

    def flag_mask(self, rows: np.ndarray, engine: str) -> np.ndarray:
        """The points PyATB may sum differently: a level within ``delta_EF`` of E_F, or, for the direct
        engine, a same-channel occupied-empty pair closer than PyATB's 1e-4 eV cut (its Kubo engine has
        none)."""
        mask = rows[:, 9] < self.delta_ef
        if engine == "direct":
            mask |= rows[:, 10] < PAIR_GAP_EV
        return mask

    def rule_rows(
        self,
        stage: str,
        model: Any,
        points: np.ndarray,
        engine: str,
        occupied_bands: int | None,
        projection: np.ndarray,
        options: ExecutionOptions | None = None,
    ) -> np.ndarray:
        """:func:`~.pyatb_emulation.pyatb_rule_rows` of the points, in order, on the planned processes
        (``options`` when given)."""
        if not len(points):
            return np.empty((0, 9 + len(projection)))
        options = self.options(stage, model, len(points)) if options is None else options
        function = functools.partial(
            pyatb_rule_rows, model, self.mu, engine, occupied_bands, self.delta_ef, projection
        )
        return np.concatenate(pmap(function, chunks(points, options.batch_size), options), axis=0)

    def flagged(
        self,
        points: np.ndarray,
        rows: np.ndarray,
        contributions: np.ndarray,
        *,
        stage: str,
        model: Any,
        engine: str,
        projection: np.ndarray,
        scale: float,
        occupied_bands: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """The flagged set: its listed points (largest ``|c_k|`` first) and its summary.

        ``contributions`` are the points' ``c_k``; ``projection`` maps a curvature
        tensor to the summed components and ``scale`` the projected value to
        ``c_k``. Each flagged point gets ``c_pyatb_rule``, its contribution under
        PyATB's direct-engine cut (:func:`~.pyatb_emulation.pyatb_rule_rows`), and
        ``flip_allowance``, how far the occupation of a level within ``delta_EF``
        of E_F may move it. The summary sums ``pair_shift = sum (c_k -
        c_pyatb_rule)``, the flip allowances and ``allowance = sum |c_k -
        c_pyatb_rule| + flip_allowance``.
        """
        closest, gaps = rows[:, 9], rows[:, 10]
        indices = np.flatnonzero(self.flag_mask(rows, engine))
        contributions = np.asarray(contributions, float).reshape(len(points), -1)
        magnitude = np.abs(contributions)
        width = contributions.shape[1]
        rule = self.rule_rows(stage, model, points[indices], engine, occupied_bands, projection)
        regularised = (rule[:, :9] @ np.asarray(projection, float).T) * scale
        flips = rule[:, 9:] * abs(scale)
        shift = contributions[indices] - regularised
        allowance = np.abs(shift) + flips
        order = np.argsort(-magnitude[indices].max(axis=1), kind="stable") if len(indices) else indices
        listed = []
        for position in order[:FLAGGED_LISTED]:
            index = indices[position]
            reasons = []
            if closest[index] < self.delta_ef:
                reasons.append("level_at_fermi")
            if engine == "direct" and gaps[index] < PAIR_GAP_EV:
                reasons.append("pyatb_pair_cut")
            listed.append(
                {
                    "k": points[index].tolist(),
                    "c": contributions[index].tolist(),
                    "c_pyatb_rule": regularised[position].tolist(),
                    "flip_allowance": flips[position].tolist(),
                    "allowance": allowance[position].tolist(),
                    "reasons": reasons,
                    "closest_level_eV": None if not math.isfinite(closest[index]) else float(closest[index]),
                    "pair_gap_eV": None if not math.isfinite(gaps[index]) else float(gaps[index]),
                }
            )
        empty = [0.0] * width
        summary = {
            "count": int(len(indices)),
            "points": int(len(points)),
            "listed": len(listed),
            "sum_abs": magnitude[indices].sum(axis=0).tolist() if len(indices) else empty,
            "pair_shift": shift.sum(axis=0).tolist() if len(indices) else empty,
            "flip_allowance": flips.sum(axis=0).tolist() if len(indices) else empty,
            "allowance": allowance.sum(axis=0).tolist() if len(indices) else empty,
            "delta_ef_eV": self.delta_ef,
            "pair_gap_eV": PAIR_GAP_EV if engine == "direct" else None,
            "rule": (
                "PyATB's direct engine zeroes the coefficient derivative of pairs closer than 1e-4 eV (its "
                "Kubo engine has no cut): c_pyatb_rule is c_k under that cut and pair_shift the sum of "
                "c_k - c_pyatb_rule; either code may occupy a level within delta_EF of E_F: flip_allowance"
            ),
        }
        return listed, summary


# ---------------------------------------------------------------------------
# AHC
# ---------------------------------------------------------------------------


def _ahc_grid(
    context: OpenMXStepContext, model: Any, native: Mapping[str, Any], label: str
) -> dict[str, Any]:
    """Plain Gamma-origin grid sum with the per-k flagged contributions (S/cm)."""
    points = kmesh(native["grid"])
    rows = context.grid_rows(f"AHC grid ({label})", model, points, native["engine"])
    stacked = rows[:, :9].reshape(len(points), 1, 3, 3)
    weight = np.full(len(points), 1.0 / len(points))
    factor = hall_factor(model)
    tensor = factor * np.einsum("k,kmab->mab", weight, stacked)
    contributions = factor * weight[:, None] * _vector(stacked[:, 0])
    listed, summary = context.flagged(
        points,
        rows,
        contributions,
        stage=f"AHC flagged points ({label})",
        model=model,
        engine=native["engine"],
        projection=vector_projection(),
        scale=factor / len(points),
    )
    for key in ("sum_abs", "pair_shift", "flip_allowance", "allowance"):
        summary[f"{key}_S_cm"] = summary.pop(key)
    return {
        "sigma": _vector(tensor),
        "per_k_flagged": listed,
        "flagged": summary,
        "integration": {"points": int(len(points)), "node_placement": "gamma_origin", "error_estimate": None},
    }


def _ahc_integrated(
    context: OpenMXStepContext, model: Any, native: Mapping[str, Any], label: str, grid: Sequence[int]
) -> dict[str, Any]:
    """QUEER's threshold quadrature or cubature of the Hall tensor (S/cm)."""
    from ...observables.integrated import ahc

    stage = f"AHC {native['integration']['method']} ({label}, grid {list(grid)})"
    options = context.options(stage, model, int(np.prod(grid)))
    integration = context.quadrature(grid)
    try:
        result = ahc(
            model,
            grid=list(grid),
            chemical_potentials=[context.mu],
            engine=native["engine"],
            dimension=3,
            integration=integration,
            execution=options,
        )
    except RuntimeError as error:
        if _budget_exhausted(error):
            raise _budget_refusal(stage, error, integration) from error
        raise
    metadata = dict(result.metadata.get("integration") or {})
    return {
        "sigma": _vector(np.asarray(result["ahc"])),
        "per_k_flagged": [],
        "flagged": None,
        "integration": {
            "points": metadata.get("points"),
            "node_placement": "cell_midpoints",
            "error_estimate": metadata.get("error_estimate"),
            "quadrature": metadata,
            "budget": {key: integration[key] for key in ("max_cells", "max_points")},
        },
    }


def _budget_refusal(stage: str, error: RuntimeError, integration: Mapping[str, Any]) -> RunnerError:
    """``queer_validation_failed`` for a cubature that ran out of the cells this job's memory holds."""
    return RunnerError(
        ERR_VALIDATION,
        f"{stage}: the refinement {BUDGET_REFUSAL} ({error}; at most {integration['max_cells']:,} cells, "
        f"sized to this job's memory, and {integration['max_points']:,} points): raise relative_error or "
        "absolute_error, lower the grid, or give the job more memory",
    )


def _fhs_vectors(model: Any, cherns: np.ndarray) -> np.ndarray:
    """The bulk Hall vectors (S/cm) of per-plane plaquette Chern numbers, as QUEER's FHS forms them.

    ``sigma = -g (e^2/h) sign(det A) sum_i C_i b_i / 2 pi``: the plaquettes of plane i are
    oriented by ``b_(i+1) x b_(i+2)``, which is parallel to ``sign(det A) a_i``.
    """
    handedness = float(np.sign(np.linalg.det(np.asarray(model.lattice, float))))
    return (
        -spin_degeneracy(model)
        * E2_H
        * handedness
        * (np.asarray(cherns, float) @ reciprocal(model))
        / (2 * np.pi)
        * 1e8
    )


def plane_normals(model: Any) -> np.ndarray:
    """Row d: the unit vector along ``b_(d+1) x b_(d+2)``, the normal of the d-th plane family (parallel to a_d)."""
    rows = reciprocal(model)
    normals = np.array([np.cross(rows[(d + 1) % 3], rows[(d + 2) % 3]) for d in range(3)])
    return normals / np.linalg.norm(normals, axis=1)[:, None]


def fhs_construction(cherns: Any, lengths: Any, normals: Any, e2_h: float, factor: float = 1.0) -> np.ndarray:
    """PyATB's Cartesian FHS output: ``sum_d (factor e2_h C_d 1e8 / |a_d|) n_d`` per energy (S/cm).

    ``C_d`` (per energy and plane family, PyATB's sign) is the plane-averaged plaquette
    phase over 2 pi and ``n_d`` the family's unit normal (parallel to a_d). It equals the
    Hall vector only for orthogonal lattice vectors; the physical vector is the dual
    sum of :func:`_fhs_vectors`.
    """
    sigma = factor * e2_h * 1e8 * np.asarray(cherns, float) / np.asarray(lengths, float)
    # Adding zero turns the -0.0 of empty plane families into 0.0 for the printed table.
    return sigma @ np.asarray(normals, float) + 0.0


def _pyatb_construction(model: Any, cherns: np.ndarray) -> np.ndarray:
    """QUEER's FHS Chern numbers projected as PyATB prints them (a convention of its output file)."""
    lengths = np.linalg.norm(np.asarray(model.lattice, float), axis=1)
    # QUEER's C_d = -(sum of phases)/(2 pi N); PyATB's is its negative, and sigma = -g e^2/h C.
    return fhs_construction(
        -np.asarray(cherns, float), lengths, plane_normals(model), E2_H, spin_degeneracy(model)
    )


def _ahc_fhs(
    context: OpenMXStepContext, model: Any, native: Mapping[str, Any], meta: Mapping[str, Any], label: str
) -> dict[str, Any]:
    """QUEER's FHS plaquette scan over PyATB's energies (S/cm)."""
    from ...observables.integrated import ahc

    grid = [int(value) for value in native["grid"]]
    planes_xy = int(meta.get("fhs_dimension", 0)) == 0
    sheet = planes_xy and grid[2] == 1
    options = context.options(f"AHC FHS ({label})", model, int(np.prod(grid)), FHS_MATRICES)
    result = ahc(
        model,
        grid=grid,
        chemical_potentials=list(native["chemical_potentials"]),
        engine="fhs",
        dimension=2 if sheet else 3,
        plane=(0, 1),
        execution=options,
    )
    diagnostics = dict(result.metadata.get("fhs_diagnostics") or {})
    cherns = np.asarray(diagnostics.get("chern_vector"), float)
    if planes_xy:
        cherns[:, :2] = 0.0
        vectors = _fhs_vectors(model, cherns)
    else:
        vectors = _vector(np.asarray(result["ahc"]))
    return {
        "sigma": vectors,
        "pyatb_construction": _pyatb_construction(model, cherns),
        "per_k_flagged": [],
        "flagged": None,
        "integration": {
            "points": int(np.prod(grid)),
            "node_placement": "gamma_origin_plaquettes",
            "error_estimate": None,
            "fhs": {
                "planes": ["k1-k2"] if planes_xy else ["k2-k3", "k3-k1", "k1-k2"],
                "chern_vector": cherns.tolist(),
                "principal_branch_admissible": diagnostics.get("principal_branch_admissible_by_plane_and_mu"),
                "metallic_vertex_rank_variation": diagnostics.get("metallic_vertex_rank_variation_detected"),
            },
        },
    }


def _ahc_evaluate(context: OpenMXStepContext, model: Any, label: str) -> dict[str, Any]:
    native, meta = context.recipe["parameters"], context.recipe["export_metadata"]
    if native["engine"] == "fhs":
        return _ahc_fhs(context, model, native, meta, label)
    if native["integration"] is None:
        return _ahc_grid(context, model, native, label)
    evaluation = _ahc_integrated(context, model, native, label, native["grid"])
    convergence: dict[str, Any] = {
        "requested": context.params.self_convergence,
        "grid": None,
        "sigma_S_cm": None,
        "e_Q": None,
    }
    if context.params.self_convergence:
        doubled = _doubled(native["grid"])
        finer = _ahc_integrated(context, model, native, f"{label}, self-convergence", doubled)
        convergence.update(
            grid=doubled,
            sigma_S_cm=_components(finer["sigma"]),
            e_Q=np.abs(evaluation["sigma"][0] - finer["sigma"][0]).tolist(),
            definition="|sigma(N) - sigma(2N)| per component, N the coarse grid",
        )
    evaluation["integration"]["self_convergence"] = convergence
    return evaluation


EMULATION_ROLE = (
    "comparison diagnostic, not a QUEER result: PyATB's own scheme evaluated with QUEER's per-k functions "
    "on the model PyATB reads"
)
REFINED_GRID_SCHEME = (
    "PyATB's Grid integral: the Gamma-origin nodes k_start + sum_i v_i n_i/N_i; a node whose judged value "
    "reaches adaptive_grid_threshold is replaced by the adaptive_grid sub-grid starting half a coarse cell "
    "(sum_i v_i/N_i)/2 below it, each sub-node weighted 1/(N n)"
)


def _pyatb_refined_grid(context: OpenMXStepContext, family: str, sign: int = 1) -> dict[str, Any] | None:
    """PyATB's refined Grid sum on the grid and its double (``e_P``); None unless the Grid refines and
    ``self_convergence`` is set.

    The sums use :func:`hall_rows` on the model PyATB reads. An AHC node is judged
    by the norm of its occupied curvature vector, a Chern node by the absolute
    curvature along the plane normal (Å² on PyATB's Bohr radius); an AHC sum is
    printed with PyATB's constants and ``sign`` (the file's), a Chern sum is the
    plane flux over 2 pi.
    """
    native = context.recipe["parameters"]
    integration = native.get("integration")
    if integration is None or integration.get("method") != "threshold" or not context.params.self_convergence:
        return None
    model, engine = context.pyatb_model, native["engine"]
    grid = [int(value) for value in native["grid"]]
    subgrid = [int(value) for value in integration["subdivision"]]
    ratio = pyatb_length_ratio()
    if family == "AHC":
        doubled = _doubled(grid)
        origin, vectors, occupied = np.zeros(3), np.eye(3), None
        projection = vector_projection()
        scale = -sign * PYATB_E_CHARGE**2 / PYATB_HBAR * 1e8 / (volume(model) * ratio)

        def magnitude(rows: np.ndarray) -> np.ndarray:
            return np.linalg.norm(rows[:, :9] @ projection.T, axis=1) * ratio**2

    else:
        doubled = _doubled(grid, active=(0, 1))
        origin, spans, vectors = _plane(native)
        occupied = native.get("occupied_bands")
        cartesian = spans @ reciprocal(model)
        projection = plane_projection(*cartesian)
        area = float(np.linalg.norm(np.cross(*cartesian)))
        scale = 1.0 / (2 * np.pi)

        def magnitude(rows: np.ndarray) -> np.ndarray:
            return np.abs(rows[:, :9] @ projection[0]) / area * ratio**2

    stage = f"{family} PyATB refined-grid emulation"
    batches = f"{stage}, sub-grid batches"
    rules = f"{stage}, PyATB's rule at flagged points"

    def rows_at(points: np.ndarray) -> np.ndarray:
        options = context.batch_options(batches, model)
        return context.grid_rows(batches, model, points, engine, occupied, options=options)

    def project(rows: np.ndarray) -> np.ndarray:
        return rows[:, :9] @ projection.T

    def regularise(points: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # PyATB's own rule at the flagged points, before it judges or sums them.
        rows = np.array(rows, copy=True)
        windows = np.zeros((len(points), len(projection)))
        mask = context.flag_mask(rows, engine)
        if mask.any():
            options = context.batch_options(rules, model)
            rule = context.rule_rows(
                rules, model, points[mask], engine, occupied, projection, options=options
            )
            rows[mask, :9] = rule[:, :9]
            windows[mask] = rule[:, 9:]
        return rows, windows

    fine_nodes = gamma_nodes(doubled, origin, vectors)
    fine_rows = context.grid_rows(stage, model, fine_nodes, engine, occupied)
    # The grid's nodes are the doubled grid's even nodes along its doubled axes (the same numbers).
    keep = tuple(slice(None, None, 2) if d != g else slice(None) for g, d in zip(grid, doubled, strict=True))
    coarse_index = np.arange(len(fine_nodes)).reshape(doubled)[keep].ravel()
    settings: dict[str, Any] = {
        "subgrid": subgrid,
        "threshold": float(integration["threshold"]),
        "vectors": vectors,
        "project": project,
        "magnitude": magnitude,
        "regularise": regularise,
    }
    coarse = refined_sum(fine_nodes[coarse_index], fine_rows[coarse_index], rows_at, grid=grid, **settings)
    fine = refined_sum(fine_nodes, fine_rows, rows_at, grid=doubled, **settings)
    printed, printed_doubled = scale * coarse["mean"], scale * fine["mean"]
    unit = "_S_cm" if family == "AHC" else ""
    context.log(
        f"PyATB refined-grid emulation: {coarse['flagged']} of {int(np.prod(grid))} nodes refined "
        f"({fine['flagged']} on the doubled grid); printed {printed.tolist()}, doubled {printed_doubled.tolist()}"
    )
    return {
        "role": EMULATION_ROLE,
        "scheme": REFINED_GRID_SCHEME,
        "judged": "|occupied curvature vector| (A^2)"
        if family == "AHC"
        else "|occupied curvature . n| (A^2)",
        "model": "spin-up channel, occupied once" if context.legacy else "the scfout's spin-orbit model",
        "grid": grid,
        "doubled_grid": doubled,
        "subgrid": subgrid,
        "threshold_A2": float(integration["threshold"]),
        f"printed{unit}": printed.tolist(),
        f"printed_doubled{unit}": printed_doubled.tolist(),
        f"e_P{unit}": np.abs(printed - printed_doubled).tolist(),
        "e_P_definition": "|PyATB's refined sum on N - on 2N|, N the coarse grid (QUEER's evaluation of it)",
        "flagged": coarse["flagged"],
        "flagged_doubled": fine["flagged"],
        "threshold_ties": coarse["ties"],
        f"tie_allowance{unit}": (abs(scale) * coarse["tie_allowance"]).tolist(),
        f"flag_allowance{unit}": (abs(scale) * coarse["flag_allowance"]).tolist(),
        "points": int(len(fine_nodes) + coarse["sub_points"] + fine["sub_points"]),
        "constants": {
            "e_C": PYATB_E_CHARGE,
            "hbar_J_s": PYATB_HBAR,
            "bohr_angstrom": PYATB_BOHR_ANGSTROM,
            "printed_sign": sign if family == "AHC" else None,
        },
    }


FHS_LINKS_SCHEME = (
    "PyATB's FHS engine: per plane family d the plaquettes (k, k+du, k+du+dv, k+dv), the link from k to k+dk "
    "W(k)^+ [sum_R exp(2 pi i k.R) (S_R + i dk.r_R)] W(k+dk) (+i dk.r at the link's start, the opposite sign "
    "of the cell-periodic overlap's first-order term; the raw position operator), the leading minors without "
    "pivoting, the phase of their product at each vertex's occupied count averaged over the vertices where "
    "the counts differ; C_d = sum phase/(2 pi N_d); "
    "printed as sum_d (C_d e^2/h 1e8/|a_d|) normalize(b_(d+1) x b_(d+2)) with PyATB's e^2/h and Bohr radius"
)
# Complex N x N matrices one plaquette of the link emulation keeps alive.
FHS_EMULATION_MATRICES = 48


def _pyatb_fhs(
    context: OpenMXStepContext, native: Mapping[str, Any], meta: Mapping[str, Any], varying: Sequence[bool]
) -> dict[str, Any] | None:
    """PyATB's FHS scan with its own link variables, beside QUEER's midpoint links on the same eigenvectors.

    Evaluated on the model PyATB reads when an energy shows a varying occupied rank
    at plaquette vertices (inside a band), the only place the two link schemes
    differ; None otherwise. ``printed_S_cm`` is what PyATB prints (its constants)
    and ``queer_links_S_cm`` QUEER's FHS value rebuilt from the same plaquettes
    (QUEER's constants, PyATB's Cartesian construction, no spin factor).
    """
    if not any(bool(value) for value in varying or []):
        return None
    model = context.pyatb_model
    size = model.nbnd
    empty: dict[Any, Any] = {}
    raw = context.model.data.position if context.model.data is not None else (empty, empty, empty)

    def channel(blocks: Mapping[Any, Any]) -> dict[Any, Any]:
        return {key: block[:size, :size] for key, block in blocks.items()}

    overlap = BlochTable(model.overlap_blocks, size)
    position = tuple(BlochTable(channel(axis), size) for axis in raw)
    connection = tuple(BlochTable(axis, size) for axis in model.position_blocks)
    grid = [int(value) for value in native["grid"]]
    mus = np.asarray(native["chemical_potentials"], float)
    directions = [2] if int(meta.get("fhs_dimension", 0)) == 0 else [0, 1, 2]
    cherns = np.zeros((2, len(mus), 3))
    for direction in directions:
        u, v = (direction + 1) % 3, (direction + 2) % 3
        steps = np.zeros((2, 3))
        steps[0, u], steps[1, v] = 1.0 / grid[u], 1.0 / grid[v]
        index = np.indices((grid[direction], grid[u], grid[v])).reshape(3, -1).T
        origins = np.zeros((len(index), 3))
        origins[:, direction], origins[:, u], origins[:, v] = (
            index[:, 0] / grid[direction],
            index[:, 1] / grid[u],
            index[:, 2] / grid[v],
        )
        stage = f"AHC FHS PyATB link emulation (planes {direction})"
        options = context.options(stage, model, len(origins), FHS_EMULATION_MATRICES)
        function = functools.partial(fhs_plaquette_phases, model, overlap, position, connection, mus, steps)
        parts = pmap(function, chunks(origins, options.batch_size), options)
        cherns[:, :, direction] = np.sum(parts, axis=0) / grid[direction]
    lengths = np.linalg.norm(np.asarray(model.lattice, float), axis=1)
    normals = plane_normals(model)
    printed = fhs_construction(cherns[0], lengths * pyatb_length_ratio(), normals, PYATB_E2_H)
    queer = fhs_construction(cherns[1], lengths, normals, E2_H)
    return {
        "role": EMULATION_ROLE,
        "scheme": FHS_LINKS_SCHEME,
        "model": "spin-up channel, occupied once" if context.legacy else "the scfout's spin-orbit model",
        "directions": directions,
        "rank_varying_energies": int(sum(bool(value) for value in varying)),
        "chern_vector": cherns[0].tolist(),
        "chern_vector_queer_links": cherns[1].tolist(),
        "printed_S_cm": _components(printed),
        "queer_links_S_cm": _components(queer),
        "constants": {"e2_h_S": PYATB_E2_H, "bohr_angstrom": PYATB_BOHR_ANGSTROM},
    }


FERMI_SEA_ENGINE = "direct"
FERMI_SEA_ROLE = (
    "QUEER's own physical check of its FHS inside bands, a diagnostic: its Fermi-sea sum on the same nodes "
    "and its FHS on the halved grid"
)
FERMI_SEA_RULE = (
    "at every energy with a varying occupied rank, per component, |FHS(N) - FS(N)| <= 3 |FHS(N) - FHS(N/2)| "
    "+ Tier A: a converging FHS meets it, a grid-independent offset from the Fermi-sea flux does not"
)


def _fermi_sea_check(
    context: OpenMXStepContext, native: Mapping[str, Any], meta: Mapping[str, Any], varying: Sequence[bool]
) -> dict[str, Any] | None:
    """QUEER's FHS inside bands against its own Fermi-sea sum: the data of the comparator's physical guard.

    At the energies whose plaquette vertices change their occupied rank, on the
    model PyATB reads (the compared values' model): ``fermi_sea_S_cm``, QUEER's
    direct-engine Fermi-sea sum ``FS(N)`` on the FHS grid's Gamma-origin nodes as
    the evaluated plane families see it (``sum_d (sigma . a_d / 2 pi) b_d``), and
    ``fhs_half_grid_S_cm``, QUEER's FHS on the halved grid; null at the other
    energies. QUEER's midpoint links converge to the direct engine's curvature,
    which carries the basis curvature the Kubo sum leaves out. None without a
    varying rank; ``error`` when the Fermi-sea sum cannot be evaluated (a
    connection that violates dipole covariance with the overlap).
    """
    flags = [bool(value) for value in varying or []]
    if not any(flags):
        return None
    model = context.pyatb_model
    picks = [index for index, flag in enumerate(flags) if flag]
    mus = np.asarray(native["chemical_potentials"], float)[picks]
    grid = [int(value) for value in native["grid"]]
    half = [max(1, value // 2) for value in grid]
    directions = [2] if int(meta.get("fhs_dimension", 0)) == 0 else [0, 1, 2]
    block: dict[str, Any] = {
        "role": FERMI_SEA_ROLE,
        "rule": FERMI_SEA_RULE,
        "model": "spin-up channel, occupied once" if context.legacy else "the scfout's spin-orbit model",
        "engine": FERMI_SEA_ENGINE,
        "grid": grid,
        "half_grid": half,
        "directions": directions,
        "energies": picks,
        "fermi_sea_S_cm": None,
        "fhs_half_grid_S_cm": None,
        "error": None,
    }
    nodes = kmesh(grid)
    options = context.options("AHC FHS Fermi-sea check", model, len(nodes))
    function = functools.partial(fermi_sea_sum, model, mus, FERMI_SEA_ENGINE)
    try:
        parts = pmap(function, chunks(nodes, options.batch_size), options)
    except ValueError as error:
        block["error"] = f"{type(error).__name__}: {error}"
        context.log(f"Fermi-sea check of the FHS not evaluated: {block['error']}")
        return block
    sea = hall_factor(model) * np.sum(parts, axis=0) / len(nodes)
    lattice = np.asarray(model.lattice, float)
    rows = reciprocal(model)
    seen = np.zeros_like(sea)
    for d in directions:
        seen += np.outer(sea @ lattice[d] / (2 * np.pi), rows[d])
    halved = {**native, "grid": half, "chemical_potentials": mus.tolist()}
    coarse = _ahc_fhs(context, model, halved, meta, "half grid, Fermi-sea check")["sigma"]
    block["fermi_sea_S_cm"] = _sparse_components(seen, picks, len(flags))
    block["fhs_half_grid_S_cm"] = _sparse_components(coarse, picks, len(flags))
    return block


def _sparse_components(
    vectors: np.ndarray, picks: Sequence[int], count: int
) -> dict[str, list[float | None]]:
    """Per component, ``count`` values: the rows of ``vectors`` at ``picks``, null elsewhere."""
    values: list[list[float | None]] = [[None] * count for _ in AHC_COMPONENTS]
    for row, index in zip(np.asarray(vectors, float), picks, strict=True):
        for axis in range(len(AHC_COMPONENTS)):
            values[axis][index] = float(row[axis])
    return dict(zip(AHC_COMPONENTS, values, strict=True))


def _emulations(**blocks: dict[str, Any] | None) -> dict[str, Any] | None:
    """The ``pyatb_emulation`` block: the emulations that ran, or None."""
    present = {name: block for name, block in blocks.items() if block is not None}
    return present or None


def _integration_block(
    meta: Mapping[str, Any], native: Mapping[str, Any], evaluation: Mapping[str, Any]
) -> dict:
    integration = dict(native.get("integration") or {})
    block = {
        "integrate_mode": meta.get("integrate_mode"),
        "refinement": meta.get("refinement"),
        "grid": [int(value) for value in native["grid"]],
        "threshold_A2": integration.get("threshold"),
        "threshold_unit": meta.get("threshold_source_unit"),
        "subdivision": integration.get("subdivision"),
        "absolute_tolerance": integration.get("absolute_tolerance"),
        "relative_tolerance": integration.get("relative_tolerance"),
        **dict(evaluation["integration"]),
    }
    return block


def run_ahc(context: OpenMXStepContext) -> None:
    """Evaluate the AHC of the step, record it and write ``Out/AHC/ahc.dat``."""
    native, meta = context.recipe["parameters"], context.recipe["export_metadata"]
    record, params = context.record, context.params
    _check_adaptive(context, meta, native)
    physical = _ahc_evaluate(context, context.model, "physical")
    legacy = None
    if context.legacy:
        legacy = _legacy_ahc(context, physical)
    fhs = native["engine"] == "fhs"
    table = int(meta["hall_export_sign"])
    # PyATB's direct/Kubo prefactor carries the signed reciprocal volume: a left-handed
    # lattice flips what it prints. Its FHS output uses lengths and unit normals only.
    sign = table if fhs else table * context.handedness
    mus = np.asarray(native["chemical_potentials"], float)
    vectors = np.asarray(physical["sigma"], float)
    offsets = meta.get("pyatb_energy_offsets_eV")
    at_fermi = None
    if not fhs:
        at_fermi = dict(zip(AHC_COMPONENTS, vectors[0].tolist()))
    elif offsets is not None and 0.0 in offsets:
        at_fermi = dict(zip(AHC_COMPONENTS, vectors[list(offsets).index(0.0)].tolist()))
    computed = list(AHC_COMPONENTS) if not (fhs and int(meta.get("fhs_dimension", 0)) == 0) else ["xy"]
    construction = np.asarray(physical["pyatb_construction"], float) if fhs else None
    fhs_links = fermi_sea = None
    if fhs:
        compared = legacy if legacy is not None else physical["integration"]
        varying = (compared.get("fhs") or {}).get("metallic_vertex_rank_variation") or []
        fhs_links = _pyatb_fhs(context, native, meta, varying)
        fermi_sea = _fermi_sea_check(context, native, meta, varying)
    record.results["ahc"] = {
        "engine": native["engine"],
        "method": None if fhs else (0 if native["engine"] == "direct" else 1),
        "pyatb_engine": meta.get("legacy_engine"),
        "mu_eV": mus.tolist(),
        "mu_pyatb_eV": [pyatb_energy(value) for value in mus],
        "energy_offsets_eV": offsets,
        "components": list(AHC_COMPONENTS),
        "components_computed": computed,
        "sigma_S_cm": _components(vectors),
        "pyatb_construction_S_cm": _components(construction) if construction is not None else None,
        "at_fermi": at_fermi,
        "convention": {
            "requested": meta.get("hall_convention_requested"),
            "resolved": meta.get("hall_convention"),
            "sign_table": table,
            "lattice_handedness": context.handedness,
            "export_sign": sign,
            "physical": PHYSICAL_SIGN,
            "file": (
                f"{AHC_FILE} holds PyATB's Cartesian FHS construction (pyatb_construction_S_cm)"
                if fhs
                else f"{AHC_FILE} holds export_sign times sigma_S_cm"
            ),
        },
        "legacy_single_channel": legacy,
        "per_k_flagged": physical["per_k_flagged"],
        "flagged": physical["flagged"],
        "raw_curvature_to_S_cm": raw_curvature_to_s_cm(context.model),
        "pyatb_emulation": _emulations(
            refined_grid=_pyatb_refined_grid(context, "AHC", sign), fhs_links=fhs_links
        ),
        "fermi_sea_check": fermi_sea,
    }
    record.results["integration"] = {
        **_integration_block(meta, native, physical),
        "max_kpoint_num": params.input_parameters.get("max_kpoint_num"),
        "sparse_format": params.input_parameters.get("sparse_format"),
        "execution_only": ["workers", "max_kpoint_num", "sparse_format"],
    }
    _warn_integration(context, meta, native, physical)
    exported = sign * (construction if construction is not None else vectors)
    text = ahc_scan_text(offsets, exported) if fhs else ahc_vector_text(exported[0])
    write_text(record.workspace / AHC_FILE, text)
    record.add_output(AHC_FILE, "Anomalous Hall conductivity (PyATB layout)", "data")
    context.log(f"AHC [yz, zx, xy] = {vectors[0].tolist()} S/cm (physical; file sign {sign:+d})")


def _legacy_ahc(context: OpenMXStepContext, physical: Mapping[str, Any]) -> dict[str, Any]:
    model = context.model
    reason = (
        "PyATB reads the spin-up Hamiltonian of a collinear scfout only"
        if model.spin_mode == "collinear"
        else "PyATB applies no spin degeneracy to a non-spin-polarized scfout"
    )
    context.record.warn(
        WARN_SINGLE_CHANNEL,
        f"{reason}: legacy_single_channel holds that value beside the physical one",
    )
    if model.spin_mode == "nonmagnetic":
        degeneracy = spin_degeneracy(model)
        scaled = ("c", "c_pyatb_rule", "flip_allowance", "allowance")
        listed = [
            {**entry, **{key: [value / degeneracy for value in entry[key]] for key in scaled if key in entry}}
            for entry in physical["per_k_flagged"]
        ]
        flagged = physical["flagged"]
        if flagged is not None:
            flagged = {
                **flagged,
                **{
                    key: [value / degeneracy for value in flagged[key]]
                    for key in ("sum_abs_S_cm", "pair_shift_S_cm", "flip_allowance_S_cm", "allowance_S_cm")
                },
            }
        integration = dict(physical.get("integration") or {})
        convergence = integration.get("self_convergence")
        if isinstance(convergence, Mapping) and convergence.get("e_Q") is not None:
            integration["self_convergence"] = {
                **convergence,
                "e_Q": [value / degeneracy for value in convergence["e_Q"]],
                "sigma_S_cm": {
                    name: [value / degeneracy for value in values]
                    for name, values in (convergence.get("sigma_S_cm") or {}).items()
                },
            }
        if integration.get("error_estimate") is not None:
            integration["error_estimate"] = integration["error_estimate"] / degeneracy
        evaluation = {**physical, "sigma": np.asarray(physical["sigma"]) / degeneracy}
        if physical.get("pyatb_construction") is not None:
            evaluation["pyatb_construction"] = np.asarray(physical["pyatb_construction"]) / degeneracy
        evaluation.update(per_k_flagged=listed, flagged=flagged, integration=integration)
    else:
        evaluation = _ahc_evaluate(context, single_channel_model(model), "spin-up channel")
    return {
        "channel": 0,
        "reason": reason,
        "sigma_S_cm": _components(evaluation["sigma"]),
        "pyatb_construction_S_cm": (
            _components(evaluation["pyatb_construction"])
            if evaluation.get("pyatb_construction") is not None
            else None
        ),
        "per_k_flagged": evaluation["per_k_flagged"],
        "flagged": evaluation["flagged"],
        "self_convergence": (evaluation.get("integration") or {}).get("self_convergence"),
        "error_estimate": (evaluation.get("integration") or {}).get("error_estimate"),
        "fhs": (evaluation.get("integration") or {}).get("fhs"),
    }


def _check_adaptive(context: OpenMXStepContext, meta: Mapping[str, Any], native: Mapping[str, Any]) -> None:
    if meta.get("integrate_mode") == "Adaptive" and int(np.prod(native["grid"])) < ADAPTIVE_MIN_CELLS:
        context.record.warn(
            WARN_ADAPTIVE_COARSE,
            f"the adaptive integral starts from {int(np.prod(native['grid']))} cells (fewer than "
            f"{ADAPTIVE_MIN_CELLS}): PyATB's counterpart can miss sharp curvature, so a parity comparison "
            "is not possible",
        )


def _warn_integration(
    context: OpenMXStepContext, meta: Mapping[str, Any], native: Mapping[str, Any], evaluation: Mapping
) -> None:
    flagged = evaluation.get("flagged")
    if flagged and flagged["count"]:
        context.record.warn(
            WARN_FLAGGED,
            f"{flagged['count']} of {flagged['points']} k-points have a level within "
            f"{flagged['delta_ef_eV']:.3g} eV of E_F"
            + (
                f" or an occupied-empty pair closer than {PAIR_GAP_EV:g} eV"
                if native["engine"] == "direct"
                else ""
            )
            + "; their contributions and PyATB allowances are listed in per_k_flagged",
        )
    if meta.get("refinement") == "threshold" and not context.params.self_convergence:
        context.record.warn(
            WARN_SELF_CONVERGENCE_SKIPPED,
            "the grid refinement places other nodes than PyATB's and no self-convergence estimate was "
            "requested (self_convergence true), so neither e_Q nor PyATB's own refinement error e_P is known: "
            "a parity comparison of this run is not possible",
        )


# ---------------------------------------------------------------------------
# Chern number
# ---------------------------------------------------------------------------


def _plane(native: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = np.asarray(native.get("k_start") or [0.0, 0.0, 0.0], float)
    spans = np.asarray(native.get("k_vectors") or [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], float)
    vectors = np.zeros((3, 3))
    vectors[:2] = spans
    return start, spans, vectors


def _chern_grid(
    context: OpenMXStepContext, model: Any, native: Mapping[str, Any], label: str
) -> dict[str, Any]:
    """Plain Gamma-origin plane sum of the occupied flux, with the per-k flagged contributions."""
    start, spans, vectors = _plane(native)
    grid = [int(value) for value in native["grid"]]
    points = kmesh(grid, k_start=start, k_vectors=vectors)
    occupied = native.get("occupied_bands")
    rows = context.grid_rows(f"Chern grid ({label})", model, points, native["engine"], occupied)
    stacked = rows[:, :9].reshape(len(points), 1, 3, 3)
    cartesian = spans @ reciprocal(model)
    fluxes = np.einsum("kmab,a,b->km", stacked, *cartesian) / (grid[0] * grid[1])
    value = float(fluxes.sum(axis=0)[0] / (2 * np.pi))
    counts = rows[:, 11]
    contributions = fluxes[:, 0] / (2 * np.pi)
    listed, summary = context.flagged(
        points,
        rows,
        contributions[:, None],
        stage=f"Chern flagged points ({label})",
        model=model,
        engine=native["engine"],
        projection=plane_projection(*cartesian),
        scale=1.0 / (grid[0] * grid[1] * 2 * np.pi),
        occupied_bands=occupied,
    )
    for entry in listed:
        for key in ("c", "c_pyatb_rule", "flip_allowance", "allowance"):
            entry[key] = entry[key][0]
    return {
        "value": value,
        "gapped": bool(np.all(counts == counts[0])) if len(counts) else False,
        "occupied_count": int(counts[0]) if len(counts) and np.all(counts == counts[0]) else None,
        "per_k_flagged": listed,
        "flagged": {
            **summary,
            **{
                key: float(summary[key][0])
                for key in ("sum_abs", "pair_shift", "flip_allowance", "allowance")
            },
        },
        "integration": {"points": int(len(points)), "node_placement": "gamma_origin", "error_estimate": None},
    }


def _chern_integrated(
    context: OpenMXStepContext, model: Any, native: Mapping[str, Any], label: str, grid: Sequence[int]
) -> dict[str, Any]:
    """QUEER's threshold quadrature or cubature of the plane flux (a gapped plane)."""
    from ...observables.geometry import chern_number

    start, spans, _ = _plane(native)
    stage = f"Chern {native['integration']['method']} ({label}, grid {list(grid)})"
    options = context.options(stage, model, int(np.prod(grid)))
    integration = context.quadrature([int(value) for value in grid][:2], active_axes=(0, 1))
    # chern_number checks the occupied rank at one extra point.
    integration = {**integration, "max_points": integration["max_points"] + 1}
    try:
        result = chern_number(
            model,
            grid=list(grid),
            k_start=start,
            k_vectors=spans,
            occupied_bands=native.get("occupied_bands"),
            fermi_energy=context.mu,
            engine=native["engine"],
            integration=integration,
            execution=options,
        )
    except ValueError as error:
        if "occupied rank" not in str(error):
            raise
        raise RunnerError(
            ERR_VALIDATION,
            f"QUEER refines a Chern-number integral only on a gapped plane: {error}. Use integrate_mode Grid "
            "with adaptive_grid_threshold 1e30 for the Fermi-sea flux of a metallic plane",
        ) from error
    except RuntimeError as error:
        if _budget_exhausted(error):
            raise _budget_refusal(stage, error, integration) from error
        raise
    metadata = dict(result.metadata.get("integration") or {})
    return {
        "value": float(result["chern_number"]),
        "gapped": True,
        "occupied_count": result.metadata.get("occupied_bands"),
        "per_k_flagged": [],
        "flagged": None,
        "integration": {
            "points": metadata.get("points"),
            "node_placement": "cell_midpoints",
            "error_estimate": metadata.get("error_estimate"),
            "quadrature": metadata,
            "budget": {key: integration[key] for key in ("max_cells", "max_points")},
        },
    }


def _chern_evaluate(context: OpenMXStepContext, model: Any, label: str) -> dict[str, Any]:
    native = context.recipe["parameters"]
    if native["integration"] is None:
        return _chern_grid(context, model, native, label)
    evaluation = _chern_integrated(context, model, native, label, native["grid"])
    convergence: dict[str, Any] = {
        "requested": context.params.self_convergence,
        "grid": None,
        "value": None,
        "e_Q": None,
    }
    if context.params.self_convergence:
        doubled = _doubled(native["grid"], active=(0, 1))
        finer = _chern_integrated(context, model, native, f"{label}, self-convergence", doubled)
        convergence.update(
            grid=doubled,
            value=finer["value"],
            e_Q=abs(evaluation["value"] - finer["value"]),
            definition="|C(N) - C(2N)|, N the coarse plane grid",
        )
    evaluation["integration"]["self_convergence"] = convergence
    return evaluation


def run_chern(context: OpenMXStepContext) -> None:
    """Evaluate the Chern number of the step, record it and write ``Out/Chern_Num/chern_number.dat``."""
    native, meta = context.recipe["parameters"], context.recipe["export_metadata"]
    record = context.record
    _check_adaptive(context, meta, native)
    physical = _chern_evaluate(context, context.model, "physical")
    value = physical["value"]
    nearest = round(value)
    near = abs(value - nearest) <= CHERN_INTEGER_TOL
    if not physical["gapped"]:
        record.warn(
            WARN_CHERN_METALLIC,
            "the plane is metallic at E_F: the value is the Fermi-sea Berry flux over 2 pi (PyATB's "
            "definition), not a topological invariant",
        )
    elif not near:
        record.warn(
            WARN_CHERN_NOT_INTEGER,
            f"Chern number {value:.6f} is not within {CHERN_INTEGER_TOL} of an integer on a gapped plane: "
            "refine the grid",
        )
    legacy = None
    if context.legacy:
        legacy = _legacy_chern(context, physical)
    start, spans, _ = _plane(native)
    occupied = native.get("occupied_bands")
    record.results["chern"] = {
        "value": value,
        "integer": int(nearest) if physical["gapped"] and near else None,
        "gapped": physical["gapped"],
        "integer_tolerance": CHERN_INTEGER_TOL,
        "flux": 2 * np.pi * value,
        "method": native["engine"],
        "method_index": 0 if native["engine"] == "direct" else 1,
        "integrate_mode": meta.get("integrate_mode"),
        "occupied_bands": "fermi" if occupied is None else int(occupied),
        "occupied_count": physical["occupied_count"],
        "fermi_energy": context.mu,
        "plane": {"k_start": start.tolist(), "k_vect1": spans[0].tolist(), "k_vect2": spans[1].tolist()},
        "orientation": "native plane flux (k_vect1 x k_vect2); no Hall sign",
        "legacy_single_channel": legacy,
        "per_k_flagged": physical["per_k_flagged"],
        "flagged": physical["flagged"],
        "pyatb_emulation": _emulations(refined_grid=_pyatb_refined_grid(context, "CHERN_NUMBER")),
    }
    record.results["integration"] = {
        **_integration_block(meta, native, physical),
        "max_kpoint_num": context.params.input_parameters.get("max_kpoint_num"),
        "sparse_format": context.params.input_parameters.get("sparse_format"),
        "execution_only": ["workers", "max_kpoint_num", "sparse_format"],
    }
    _warn_integration(context, meta, native, physical)
    mode = str(meta.get("integrate_mode") or "Grid")
    write_text(record.workspace / CHERN_FILE, chern_text(value, mode))
    record.add_output(CHERN_FILE, "Chern number (PyATB layout)", "data")
    context.log(f"Chern number {value!r} ({'gapped' if physical['gapped'] else 'metallic'} plane)")


def _legacy_chern(context: OpenMXStepContext, physical: Mapping[str, Any]) -> dict[str, Any]:
    model = context.model
    if model.spin_mode == "nonmagnetic":
        # A Chern number counts each spatial level once: the single channel is the physical value.
        evaluation = physical
        reason = "a non-spin-polarized Chern number counts each spatial level once, as PyATB does"
    else:
        reason = "PyATB reads the spin-up Hamiltonian of a collinear scfout only"
        context.record.warn(
            WARN_SINGLE_CHANNEL, f"{reason}: legacy_single_channel holds that value beside the physical one"
        )
        evaluation = _chern_evaluate(context, single_channel_model(model), "spin-up channel")
    return {
        "channel": 0,
        "reason": reason,
        "value": evaluation["value"],
        "gapped": evaluation["gapped"],
        "per_k_flagged": evaluation["per_k_flagged"],
        "flagged": evaluation["flagged"],
        "self_convergence": (evaluation.get("integration") or {}).get("self_convergence"),
        "error_estimate": (evaluation.get("integration") or {}).get("error_estimate"),
    }


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def run_shift(context: OpenMXStepContext) -> None:
    """Evaluate the shift current of the step (:mod:`~queer.compat.generator.openmx_shift`)."""
    from .openmx_shift import run_shift_current

    run_shift_current(context)


STEP_FUNCTIONS: Mapping[str, Callable[[OpenMXStepContext], None]] = {
    STEP_AHC: run_ahc,
    STEP_CHERN: run_chern,
    STEP_SHIFT: run_shift,
}


def openmx_step_body(parameters: OpenMXStepParameters, record: RunRecord, log: Callable[[str], None]) -> None:
    """Plan the quadrature budget, build the model of the step, then run it: the body :func:`~.runner.run` executes."""
    context = OpenMXStepContext(parameters, record, log)
    context.stage_scfout()
    context.plan_budget()
    context.build_model()
    STEP_FUNCTIONS[parameters.step](context)
