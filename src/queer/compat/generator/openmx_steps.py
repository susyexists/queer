"""QUEER's OpenMX steps: the anomalous Hall conductivity and the Chern number of a PyATB counterpart.

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
  ``delta_EF`` of E_F or an occupied-empty pair (same spin channel) closer than
  1e-4 eV, where ``delta_EF = 1e-6 eV * HARTREE_EV / PYATB_HARTREE_EV`` (the
  Fermi-energy agreement allowed between the codes) plus the eigenvalue
  tolerance ``max(1e-8 eV, the model's degeneracy tolerance)``. Each point of
  ``F`` is listed with its contribution ``c_k`` to the result, the largest first.
- Grid with a lower threshold: QUEER's own threshold quadrature (cell
  midpoints, midpoint sub-cells for cells whose curvature norm reaches the
  threshold in Å²), and ``Adaptive``: QUEER's embedded-error cubature with
  PyATB's tolerances. Their nodes differ from PyATB's, so a comparison needs an
  error estimate: ``self_convergence: true`` adds the same integration on the
  doubled grid and records ``e_Q = |sigma(N) - sigma(2N)|``; a cubature also
  records its own embedded error estimate.
- The FHS engine: QUEER's FHS plaquette scan over PyATB's energies (the
  ``fhs_energy_window`` offsets, upper end excluded). ``fhs_dimension 0`` evaluates
  the k1-k2 plane only.

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
from ...physics.constants import E2_H, E_CHARGE, HBAR
from ...physics.geometry import band_curvatures
from ...physics.statistics import occupations
from .memory import ParallelPlanner, chunks, model_fixed_bytes, plan_parallelism, pmap
from .openmx_params import STEP_AHC, STEP_CHERN, OpenMXStepParameters
from .pyatb_conventions import (
    FERMI_TOLERANCE_HARTREE,
    HARTREE_EV,
    PYATB_HARTREE_EV,
    pyatb_energy,
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
    ERR_MISSING_INPUT,
    ERR_VALIDATION,
    WARN_CHERN_NOT_INTEGER,
    RunnerError,
    RunRecord,
)

APPROXIMATION_NOTE = (
    "OpenMX localized-orbital model read from the scfout (nonorthogonal basis with its overlap and "
    "position matrices), the model PyATB reads: Berry curvature by the direct (basis-covariant) or Kubo "
    "formula, FHS plaquettes with first-order midpoint position links. PyATB reads only the spin-up "
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


def hall_factor(model: Any) -> float:
    """S/cm per Å² of mean occupied curvature: ``-g e^2/hbar 1e8/V`` (``g`` the spin degeneracy)."""
    return -spin_degeneracy(model) * E_CHARGE**2 / HBAR * (1e8 / volume(model))


def raw_curvature_to_s_cm(model: Any) -> float:
    """``e^2/hbar 1e8/V``: a raw curvature integral (Å², no spin factor) in S/cm."""
    return E_CHARGE**2 / HBAR * 1e8 / volume(model)


def single_channel_model(model: OpenMXModel) -> OpenMXModel:
    """Channel 0 of a model without spin-orbit coupling, occupied once per level: what PyATB reads."""
    if model.spin_mode == "soc":
        raise ValueError("a spin-orbit model has no single spin channel")
    n = model.n_spatial

    def part(blocks: Mapping[tuple[int, int, int], Any]) -> dict[tuple[int, int, int], Any]:
        return {key: block[:n, :n] for key, block in blocks.items()}

    translations = np.array(sorted(set(model.hamiltonian_blocks) | set(model.overlap_blocks)), dtype=int)
    channel = OpenMXModel(
        translations,
        part(model.hamiltonian_blocks),
        part(model.overlap_blocks),
        position=tuple(part(axis) for axis in model.position_blocks) if model.position_blocks else None,
        lattice=model.lattice,
        fermi_energy=model.fermi_energy,
        spin_mode="nonmagnetic",
        spin_degeneracy=1,
        atom_positions=model.atom_positions,
        orbital_metadata=list(model.orbital_metadata[:n]),
        execution=model.execution,
        hermiticity_tolerance=model.hermiticity_tolerance,
    )
    channel.degeneracy_tolerance = model.degeneracy_tolerance
    channel.metadata.update(model.metadata)
    return channel


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

    # ------------------------------------------------------------ model
    def build_model(self) -> None:
        name = self.params.scfout
        path = self.record.workspace / name
        if not path.is_file():
            raise RunnerError(
                ERR_MISSING_INPUT, f"{name} was not staged; bind it from the OpenMX SCF job (scf.scfout)"
            )
        self.record.record_input(name)
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

    # ------------------------------------------------------------ grids
    def grid_rows(
        self,
        stage: str,
        model: Any,
        points: np.ndarray,
        engine: str,
        occupied_bands: int | None = None,
    ) -> np.ndarray:
        """:func:`hall_rows` of every point, in order, on the planned processes."""
        options = self.options(stage, model, len(points))
        function = functools.partial(hall_rows, model, self.mu, engine, occupied_bands)
        parts = pmap(function, chunks(points, options.batch_size), options)
        return np.concatenate(parts, axis=0) if parts else np.empty((0, 12))

    def flagged(
        self, points: np.ndarray, rows: np.ndarray, contributions: np.ndarray
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """The flagged set: its listed points (largest ``|c_k|`` first) and its summary."""
        closest, gaps = rows[:, 9], rows[:, 10]
        mask = (closest < self.delta_ef) | (gaps < PAIR_GAP_EV)
        indices = np.flatnonzero(mask)
        magnitude = np.abs(np.asarray(contributions, float).reshape(len(points), -1))
        order = indices[np.argsort(-magnitude[indices].max(axis=1), kind="stable")]
        listed = [
            {
                "k": points[index].tolist(),
                "c": np.asarray(contributions[index], float).tolist(),
                "closest_level_eV": None if not math.isfinite(closest[index]) else float(closest[index]),
                "pair_gap_eV": None if not math.isfinite(gaps[index]) else float(gaps[index]),
            }
            for index in order[:FLAGGED_LISTED]
        ]
        summary = {
            "count": int(len(indices)),
            "points": int(len(points)),
            "listed": len(listed),
            "sum_abs": magnitude[indices].sum(axis=0).tolist()
            if len(indices)
            else [0.0] * magnitude.shape[1],
            "delta_ef_eV": self.delta_ef,
            "pair_gap_eV": PAIR_GAP_EV,
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
    listed, summary = context.flagged(points, rows, contributions)
    summary["sum_abs_S_cm"] = summary.pop("sum_abs")
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
    result = ahc(
        model,
        grid=list(grid),
        chemical_potentials=[context.mu],
        engine=native["engine"],
        dimension=3,
        integration=dict(native["integration"]),
        execution=options,
    )
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
        },
    }


def _fhs_vectors(model: Any, cherns: np.ndarray) -> np.ndarray:
    """The bulk Hall vectors (S/cm) of per-plane plaquette Chern numbers, as QUEER's FHS forms them."""
    return (
        -spin_degeneracy(model) * E2_H * (np.asarray(cherns, float) @ reciprocal(model)) / (2 * np.pi) * 1e8
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
        "at_fermi": at_fermi,
        "convention": {
            "requested": meta.get("hall_convention_requested"),
            "resolved": meta.get("hall_convention"),
            "sign_table": table,
            "lattice_handedness": context.handedness,
            "export_sign": sign,
            "physical": PHYSICAL_SIGN,
            "file": f"{AHC_FILE} holds export_sign times sigma_S_cm",
        },
        "legacy_single_channel": legacy,
        "per_k_flagged": physical["per_k_flagged"],
        "flagged": physical["flagged"],
        "raw_curvature_to_S_cm": raw_curvature_to_s_cm(context.model),
    }
    record.results["integration"] = {
        **_integration_block(meta, native, physical),
        "max_kpoint_num": params.input_parameters.get("max_kpoint_num"),
        "sparse_format": params.input_parameters.get("sparse_format"),
        "execution_only": ["workers", "max_kpoint_num", "sparse_format"],
    }
    _warn_integration(context, meta, native, physical)
    exported = sign * vectors
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
        listed = [
            {**entry, "c": [value / degeneracy for value in entry["c"]]}
            for entry in physical["per_k_flagged"]
        ]
        flagged = physical["flagged"]
        if flagged is not None:
            flagged = {**flagged, "sum_abs_S_cm": [value / degeneracy for value in flagged["sum_abs_S_cm"]]}
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
        evaluation.update(per_k_flagged=listed, flagged=flagged, integration=integration)
    else:
        evaluation = _ahc_evaluate(context, single_channel_model(model), "spin-up channel")
    return {
        "channel": 0,
        "reason": reason,
        "sigma_S_cm": _components(evaluation["sigma"]),
        "per_k_flagged": evaluation["per_k_flagged"],
        "flagged": evaluation["flagged"],
        "self_convergence": (evaluation.get("integration") or {}).get("self_convergence"),
        "error_estimate": (evaluation.get("integration") or {}).get("error_estimate"),
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
            f"{flagged['delta_ef_eV']:.3g} eV of E_F or an occupied-empty pair closer than "
            f"{PAIR_GAP_EV:g} eV; their contributions are listed in per_k_flagged",
        )
    if meta.get("refinement") == "threshold" and not context.params.self_convergence:
        context.record.warn(
            WARN_SELF_CONVERGENCE_SKIPPED,
            "the grid refinement places other nodes than PyATB's and no self-convergence estimate was "
            "requested (self_convergence true): a parity comparison of this run is not possible",
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
    listed, summary = context.flagged(points, rows, contributions[:, None])
    for entry in listed:
        entry["c"] = entry["c"][0]
    return {
        "value": value,
        "gapped": bool(np.all(counts == counts[0])) if len(counts) else False,
        "occupied_count": int(counts[0]) if len(counts) and np.all(counts == counts[0]) else None,
        "per_k_flagged": listed,
        "flagged": {**summary, "sum_abs": float(summary["sum_abs"][0])} if summary else None,
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
    try:
        result = chern_number(
            model,
            grid=list(grid),
            k_start=start,
            k_vectors=spans,
            occupied_bands=native.get("occupied_bands"),
            fermi_energy=context.mu,
            engine=native["engine"],
            integration=dict(native["integration"]),
            execution=options,
        )
    except ValueError as error:
        raise RunnerError(
            ERR_VALIDATION,
            f"QUEER refines a Chern-number integral only on a gapped plane: {error}. Use integrate_mode Grid "
            "with adaptive_grid_threshold 1e30 for the Fermi-sea flux of a metallic plane",
        ) from error
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

STEP_FUNCTIONS: Mapping[str, Callable[[OpenMXStepContext], None]] = {STEP_AHC: run_ahc, STEP_CHERN: run_chern}


def openmx_step_body(parameters: OpenMXStepParameters, record: RunRecord, log: Callable[[str], None]) -> None:
    """Build the model of the step, then run it: the body :func:`~.runner.run` executes."""
    context = OpenMXStepContext(parameters, record, log)
    context.build_model()
    STEP_FUNCTIONS[parameters.step](context)
