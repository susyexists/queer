"""Compare native differential quantum metric with recovered PyATB averages.

The original estimator measures determinant distances at finite reciprocal
steps. Its xx/yy/zz labels describe normalized reciprocal directions on a skew
cell. QUEER's Cartesian finite-basis interband tensor is projected into those
directions explicitly. No producer Python or compiled code is executed.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from queer import ExecutionOptions, OpenMXModel
from queer.config import config_from_mapping
from queer.execution import execution_session
from queer.io.artifacts import (
    RunStore,
    atomic_json,
    atomic_npz,
    digest_file,
    environment_versions,
    load_manifest,
)
from queer.io.openmx import HARTREE_EV
from queer.observables._tb import basis_overlap, kmesh, map_points, states
from queer.workflows import run_workflow

COMPONENTS = ("xx", "yy", "zz", "xy", "yz", "xz")


def read_metric_tables(averaged_path, integrated_path):
    for path, names, units in (
        (averaged_path, ["E(eV)", *("g_" + name for name in COMPONENTS), "tr_g"], "Angstrom^2"),
        (integrated_path, ["E(eV)", *("G_" + name for name in COMPONENTS)], "Angstrom^-1"),
    ):
        lines = Path(path).read_text().splitlines()
        if not any(line.lstrip("# ").split() == names for line in lines if line.lstrip().startswith("#")):
            raise ValueError("Historical quantum-metric component order is unknown")
        if not any(units in line for line in lines if line.lstrip().startswith("#")):
            raise ValueError("Historical quantum-metric units are unknown")
    averaged = np.loadtxt(averaged_path, ndmin=2)
    integrated = np.loadtxt(integrated_path, ndmin=2)
    if averaged.shape[1] != 8 or integrated.shape != (len(averaged), 7):
        raise ValueError("Historical metric tables must contain E+six components+trace and E+six integrals")
    if not len(averaged) or not np.isfinite(averaged).all() or not np.isfinite(integrated).all():
        raise ValueError("Historical metric tables must be complete and finite")
    if not np.array_equal(averaged[:, 0], integrated[:, 0]):
        raise ValueError("Historical averaged and integrated energy rows differ")
    if not np.allclose(averaged[:, 1:4].sum(axis=1), averaged[:, 7], atol=1e-5, rtol=1e-6):
        raise ValueError("Historical metric trace differs from its diagonal components")
    return averaged, integrated


def directional_components(cartesian, reciprocal, dimension):
    """Project a Cartesian covariant tensor into the producer's unit directions."""
    reciprocal = np.asarray(reciprocal, float)
    directions = reciprocal / np.linalg.norm(reciprocal, axis=1)[:, None]
    projected = np.einsum("ia,...ab,jb->...ij", directions, cartesian, directions)
    if dimension == 0:
        projected[..., 2, :] = projected[..., :, 2] = 0
    elif dimension != 1:
        raise ValueError("Historical dimension must be 0 (in-plane) or 1 (three directions)")
    components = np.stack(
        [projected[..., i, j] for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))], axis=-1
    )
    return components, directions


def validate_runtime_recipe(path, grid, dimension, energy):
    """Reject mismatched tutorial recipes instead of silently adopting them."""
    if path is None:
        return None
    runtime = json.loads(path.read_text())
    block = runtime["QUANTUM_METRIC"]
    if (
        block.get("engine") != "fhs"
        or block.get("fhs_grid") != list(grid)
        or block.get("fhs_dimension") != dimension
    ):
        raise ValueError("Requested comparison differs from the recovered runtime quantum-metric recipe")
    limits = block["fhs_energy_window"]
    count = block["fhs_energy_points"]
    reconstructed = limits[0] + np.arange(count) * ((limits[1] - limits[0]) / count if count > 1 else 0)
    if reconstructed.shape != energy.shape or not np.allclose(reconstructed, energy, atol=5.01e-7, rtol=0):
        raise ValueError("Recovered runtime energy recipe differs from the original metric table")
    return runtime


def read_runtime_log(path, grid, dimension, energy):
    """Recover only the actual parameter block, without treating it as Input."""
    text = Path(path).read_text()
    if "FHS quantum metric: peak tr(g)" not in text:
        raise ValueError("Runtime log does not contain a completed quantum-metric calculation")

    def value(key):
        match = re.search(rf"^\s*>>\s*{re.escape(key)}\s*:\s*(.*?)\s*$", text, re.MULTILINE)
        if match is None:
            raise ValueError(f"Quantum-metric runtime log omits {key}")
        return match.group(1)

    recovered_grid = [int(item) for item in value("fhs_grid").split()]
    recovered_dimension = int(value("fhs_dimension"))
    limits = [float(item) for item in value("fhs_energy_window (eV)").split()]
    count = int(value("fhs_energy_points"))
    reconstructed = limits[0] + np.arange(count) * ((limits[1] - limits[0]) / count if count > 1 else 0)
    if recovered_grid != list(grid) or recovered_dimension != dimension:
        raise ValueError("Comparison mesh/dimension differs from the recovered runtime log")
    if reconstructed.shape != energy.shape or not np.allclose(reconstructed, energy, atol=5.01e-7, rtol=0):
        raise ValueError("Comparison energies differ from the recovered runtime log")
    return {
        "QUANTUM_METRIC": {
            "engine": "fhs",
            "fhs_grid": recovered_grid,
            "fhs_dimension": recovered_dimension,
            "fhs_energy_window": limits,
            "fhs_energy_points": count,
        },
        "fermi_energy_eV_printed": float(value("fermi_energy (eV)")),
        "provenance_kind": "archived runtime log; original Hamiltonian bytes are not established by this log",
    }


def projector_distance(full_link, occupied):
    """Fubini-Study determinant distance after full-space polar projection.

    The polar factor transports the complete retained basis. Its occupied
    compression has determinant fidelity between zero and one. Projecting only
    the occupied block would erase this distance and is intentionally avoided.
    """
    full_link = np.asarray(full_link, complex)
    if full_link.ndim != 2 or full_link.shape[0] != full_link.shape[1] or not np.isfinite(full_link).all():
        raise ValueError("Full basis link must be finite and square")
    if (
        isinstance(occupied, bool)
        or not isinstance(occupied, (int, np.integer))
        or not 0 <= occupied <= len(full_link)
    ):
        raise ValueError("Occupied projector rank must be an integer within the complete basis")
    if occupied in (0, len(full_link)):
        return 0.0
    left, _, right = np.linalg.svd(full_link, full_matrices=False)
    transport = left @ right
    _, log_fidelity = np.linalg.slogdet(transport[:occupied, :occupied])
    distance = float(-np.expm1(2 * log_fidelity))
    if distance < -1e-10 or distance > 1 + 1e-10:
        raise ValueError("Polar occupied-block determinant violates projector fidelity bounds")
    return min(1.0, max(0.0, distance))


def reference_centres(model):
    """Reference orbital centres: real position diagonals at translation zero.

    This centre choice is explicit to the offline reference estimator. It does
    not replace native atomic metadata or the physical geometry operators.
    """
    blocks = getattr(model, "position_blocks", None)
    if blocks is None or len(blocks) != 3 or any((0, 0, 0) not in axis for axis in blocks):
        raise ValueError("Centre-resummed reference links require all three zero-cell position blocks")
    centres = np.stack([np.asarray(axis[(0, 0, 0)].diagonal()).real for axis in blocks], axis=1)
    if centres.shape != (model.nbnd, 3) or not np.isfinite(centres).all():
        raise ValueError("Reference position centres must be finite and match the complete basis")
    return centres


def centre_resummed_basis_overlap(model, k, kprime, centres=None):
    """Midpoint link with exact centre phases and a first-order residual.

    Write each Hermitian position matrix as its centre anticommutator plus a
    residual. Exponentiating the two endpoint centre phases while expanding
    only the residual gives the native first-order tangent and exact global
    origin covariance. This independently derived approximation is used only
    for the explicit historical reference estimator.
    """
    k, kprime = np.asarray(k, float), np.asarray(kprime, float)
    if k.shape != (3,) or kprime.shape != (3,) or not np.isfinite([k, kprime]).all():
        raise ValueError("Reference link endpoints must be finite fractional reciprocal points")
    centres = reference_centres(model) if centres is None else np.asarray(centres, float)
    if centres.shape != (model.nbnd, 3) or not np.isfinite(centres).all():
        raise ValueError("Reference position centres must be finite and match the complete basis")
    delta = kprime - k
    # The reference chooses the shortest fractional displacement; half-cell
    # ties round away from zero, including the negative direction.
    delta -= np.sign(delta) * np.floor(np.abs(delta) + 0.5)
    midpoint = k + delta / 2
    cartesian_delta = delta @ model.g_vec
    metric = model.overlap(midpoint)
    position = model.position(midpoint)
    connection = (position + position.conj().transpose(0, 2, 1)) / 2
    residual = connection - (centres.T[:, :, None] + centres.T[:, None, :]) * metric[None] / 2
    phase = np.exp(-0.5j * (centres @ cartesian_delta))
    first_order = metric - 1j * np.einsum("a,aij->ij", cartesian_delta, residual)
    return phase[:, None] * first_order * phase[None, :]


def finite_step_point(model, k, chemical_potentials, grid, dimension, basis_link_mode="midpoint_first_order"):
    """Evaluate finite projector distances using only native states and links."""
    grid = np.asarray(grid, float)
    if grid.shape != (3,) or np.any(grid < 1) or not np.equal(grid, np.round(grid)).all():
        raise ValueError("Finite-step grid must contain three positive integers")
    if dimension not in (0, 1):
        raise ValueError("Historical dimension must be 0 or 1")
    if basis_link_mode not in ("midpoint_first_order", "centre_resummed"):
        raise ValueError("Unknown finite-step reference basis link")
    axes = list(range(2 if dimension == 0 else 3))
    pairs = [(i, j) for i in axes for j in axes if i < j]
    steps = np.eye(3) / grid[:, None]
    deltas = [steps[axis] for axis in axes] + [steps[i] + steps[j] for i, j in pairs]
    nodes = np.asarray([k, *(np.asarray(k) + delta for delta in deltas)])
    solution = states(model, nodes)
    if len(np.unique(solution.band_channels)) != 1:
        raise ValueError(
            "Historical finite-step metric comparison requires one explicit spin channel/manifold"
        )
    orders = np.argsort(solution.energies, axis=0)
    coefficients = [solution.coefficients[index][:, orders[:, index]] for index in range(len(nodes))]
    occupied = (solution.energies[:, 0, None] < np.asarray(chemical_potentials)[None]).sum(axis=0)
    distances = []
    for index, node in enumerate(nodes[1:], start=1):
        basis_link = (
            centre_resummed_basis_overlap(model, nodes[0], node)
            if basis_link_mode == "centre_resummed"
            else basis_overlap(model, nodes[0], node)
        )
        link = coefficients[0].conj().T @ basis_link @ coefficients[index]
        # Cache the complete polar factor across occupied ranks/energies.
        left, _, right = np.linalg.svd(link, full_matrices=False)
        transport = left @ right
        ranks = {}
        for rank in np.unique(occupied):
            rank = int(rank)
            if rank in (0, len(link)):
                ranks[rank] = 0.0
            else:
                _, log_fidelity = np.linalg.slogdet(transport[:rank, :rank])
                ranks[rank] = min(1.0, max(0.0, float(-np.expm1(2 * log_fidelity))))
        distances.append(np.asarray([ranks[int(rank)] for rank in occupied]))
    norms = np.linalg.norm(steps @ model.g_vec, axis=1)
    tensor = np.zeros((len(chemical_potentials), 3, 3))
    for index, axis in enumerate(axes):
        tensor[:, axis, axis] = distances[index] / norms[axis] ** 2
    for index, (i, j) in enumerate(pairs, start=len(axes)):
        mixed = (distances[index] - distances[axes.index(i)] - distances[axes.index(j)]) / (
            2 * norms[i] * norms[j]
        )
        tensor[:, i, j] = tensor[:, j, i] = mixed
    return np.stack([tensor[:, i, j] for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))], axis=-1)


def finite_step_grid(
    model,
    chemical_potentials,
    grid,
    dimension,
    output,
    execution,
    scfout,
    resume,
    basis_link_mode="midpoint_first_order",
):
    """Commit bounded native finite-step tiles through one RunStore owner."""
    points = kmesh(grid)
    options = execution.resolved()
    per_point = 16 * model.nbnd**2 * 24
    tile = options.workers * options.bounded_batch_size(per_point)
    config = {
        "workflow": "finite_step_metric_reference",
        "inputs": {"model": "native", "scfout": str(scfout.resolve())},
        "parameters": {
            "estimator_version": 2,
            "script_sha256": digest_file(Path(__file__)),
            "grid": list(map(int, grid)),
            "dimension": dimension,
            "chemical_potentials_eV": list(map(float, chemical_potentials)),
            "basis_link_mode": basis_link_mode,
        },
        "execution": options.as_dict(),
    }
    store = RunStore(output, config, [scfout], resume=resume)
    values = []
    try:
        if store.completed("result"):
            with np.load(output / store.manifest["cases"]["result"]["path"], allow_pickle=False) as saved:
                result = saved["average"]
            store.finish()
            return result
        with execution_session(options):
            for start in range(0, len(points), tile):
                case = f"tile_{start:08d}"
                if store.completed(case):
                    with np.load(output / store.manifest["cases"][case]["path"], allow_pickle=False) as saved:
                        value = saved["metric"]
                else:
                    selected = points[start : start + tile]
                    value = np.asarray(
                        map_points(
                            model,
                            selected,
                            lambda k: finite_step_point(
                                model, k, chemical_potentials, grid, dimension, basis_link_mode
                            ),
                            options,
                        )
                    )
                    store.commit(
                        case,
                        {"kpoints": selected, "metric": value},
                        metadata={"points": len(selected), "units": "Angstrom^2"},
                    )
                values.append(value)
        sampled = np.concatenate(values)
        store.commit("result", {"metric": sampled, "average": sampled.mean(axis=0)})
        store.finish()
        return sampled.mean(axis=0)
    except BaseException as error:
        store.finish(status="failed", error=error)
        raise
    finally:
        store.release()


def replay(args):
    start = time.monotonic()
    scfout = args.scfout.resolve()
    if digest_file(scfout) != args.expected_scfout_sha256.lower():
        raise ValueError("SCFOUT does not match the supplied independent SHA256 evidence")
    if args.finite_basis_link != "midpoint_first_order" and not args.finite_step:
        raise ValueError("A reference finite-basis link requires --finite-step")
    averaged, integrated = read_metric_tables(args.average, args.integrated)
    grid = np.asarray(args.grid)
    if grid.shape != (3,) or grid.dtype.kind not in "iu" or np.any(grid < 1):
        raise ValueError("Comparison grid must have three positive integers")
    if args.dimension == 0 and np.any(np.abs(averaged[:, [3, 5, 6]]) > 1e-12):
        raise ValueError("In-plane recipe cannot produce the reference's nonzero z components")
    runtime = validate_runtime_recipe(args.runtime_input_json, grid, args.dimension, averaged[:, 0])
    if args.runtime_log is not None:
        if runtime is not None:
            raise ValueError("Choose runtime Input JSON or runtime log")
        runtime = read_runtime_log(args.runtime_log, grid, args.dimension, averaged[:, 0])
    if args.binding == "verified" and runtime is None:
        raise ValueError("Verified historical binding requires a recovered matching runtime Input JSON")
    if not np.isfinite(args.producer_hartree_ev) or args.producer_hartree_ev <= 0:
        raise ValueError("Producer Hartree conversion must be finite and positive")
    model = OpenMXModel.from_scfout(scfout)
    if runtime is not None and "INPUT_PARAMETERS" in runtime:
        global_parameters = runtime["INPUT_PARAMETERS"]
        if global_parameters.get("package") != "OPENMX":
            raise ValueError("Recovered runtime does not use an OpenMX source")
        if str(global_parameters.get("fermi_energy", "Auto")).lower() != "auto":
            raise ValueError("This metric recipe requires the original Auto Fermi reference")
        if global_parameters.get("nspin") != (4 if model.spin_mode == "soc" else 1):
            raise ValueError("Recovered runtime spin metadata differs from the source")
    if runtime is not None and "fermi_energy_eV_printed" in runtime:
        producer_mu = model.fermi_energy * args.producer_hartree_ev / HARTREE_EV
        if abs(producer_mu - runtime["fermi_energy_eV_printed"]) > 5.01e-7:
            raise ValueError(
                "Recovered runtime Fermi reference differs from this source at printed precision"
            )
    normalization = 1 / abs(np.linalg.det(model.lattice))
    if not np.allclose(integrated[:, 1:], normalization * averaged[:, 1:7], atol=1e-8, rtol=1e-6):
        raise ValueError("Historical metric integral and average do not match this source cell volume")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    recipe_energy = averaged[:, 0]
    if runtime is not None:
        block = runtime["QUANTUM_METRIC"]
        count = block["fhs_energy_points"]
        bounds = block["fhs_energy_window"]
        recipe_energy = bounds[0] + np.arange(count) * ((bounds[1] - bounds[0]) / count if count > 1 else 0)
    # The source parser scales both H and ChemP with its declared older constant.
    # Rescale the original relative energy to preserve the occupied threshold in
    # Hartree, while retaining native physical eV and length conventions.
    chemical_potentials = model.fermi_energy + recipe_energy * HARTREE_EV / args.producer_hartree_ev
    report = {
        "status": "prepared",
        "historical_input_binding": args.binding,
        "inputs": {
            "scfout": {"path": str(scfout), "sha256": digest_file(scfout)},
            "average": {"path": str(args.average.resolve()), "sha256": digest_file(args.average)},
            "integrated": {"path": str(args.integrated.resolve()), "sha256": digest_file(args.integrated)},
            "producer_sources": [
                {"path": str(p.resolve()), "sha256": digest_file(p)} for p in args.source_file
            ],
            "runtime_input": None
            if args.runtime_input_json is None
            else {
                "path": str(args.runtime_input_json.resolve()),
                "sha256": digest_file(args.runtime_input_json),
            },
        },
        "recipe": {
            "grid": grid.tolist(),
            "dimension": args.dimension,
            "energy_relative_eV": recipe_energy.tolist(),
            "chemical_potentials_native_eV": chemical_potentials.tolist(),
            "producer_Hartree_eV": args.producer_hartree_ev,
            "native_Hartree_eV": HARTREE_EV,
            "historical_runtime": runtime,
            "requested_finite_basis_link": args.finite_basis_link,
        },
        "definitions": {
            "native": "Cartesian finite_basis_interband differential occupied-manifold metric; external orbital spread excluded",
            "historical": "finite-step 1-|det occupied block of full polar-unitarized overlap|^2 divided by step products",
            "comparison_basis": "normalized reciprocal directions, preserving the producer's actual skew-cell indexing",
            "same_finite_step_estimator": False,
        },
        "limitations": [
            "finite-step determinant distances and differential interband geometry have different finite-grid truncation errors",
            "historical diagonal sum is a directional sum, not the Cartesian trace on a skew cell",
            "one sampled mesh does not establish continuum convergence",
            *(
                ["historical model and runtime grid binding remain inferred"]
                if args.binding == "inferred"
                else []
            ),
        ],
        "environment": environment_versions(),
    }
    if args.runtime_log is not None:
        report["inputs"]["runtime_log"] = {
            "path": str(args.runtime_log.resolve()),
            "sha256": digest_file(args.runtime_log),
        }
    report["historical_grid_binding"] = "recovered" if runtime is not None else "chosen diagnostic mesh"
    if runtime is not None and args.binding == "inferred":
        report["limitations"][-1] = (
            "historical Hamiltonian bytes remain inferred; grid is recovered from the archived runtime"
        )
    atomic_json(output / "comparison.json", report)
    cartesian_means = []
    diagnostic_runs = []
    try:
        for index, mu in enumerate(chemical_potentials):
            config = config_from_mapping(
                {
                    "workflow": "quantum_metric",
                    "inputs": {"material_dir": str(scfout.parent), "model": "openmx", "scfout": scfout.name},
                    "parameters": {"grid": grid.tolist(), "fermi_energy": float(mu)},
                    "execution": {
                        "workers": args.workers,
                        "batch_size": args.batch_size,
                        "native_threads": args.native_threads,
                        "memory_mb": args.memory_mb,
                    },
                }
            )
            run = run_workflow(config, run_dir=output / f"native-{index:06d}", resume=args.resume)
            load_manifest(run)
            with np.load(run / "arrays/result.npz", allow_pickle=False) as data:
                metrics = data["metric"]
                cartesian_means.append(metrics.mean(axis=0))
                diagnostic_runs.append(
                    {
                        "run": str(run),
                        "kpoints": len(metrics),
                        "minimum_metric_eigenvalue_A2": float(np.linalg.eigvalsh(metrics).min()),
                    }
                )
    except BaseException as error:
        report.update(status="incomplete", error={"type": type(error).__name__, "message": str(error)})
        report["wall_seconds"] = time.monotonic() - start
        atomic_json(output / "comparison.json", report)
        raise
    native_cartesian = np.asarray(cartesian_means)
    native, directions = directional_components(native_cartesian, model.g_vec, args.dimension)
    difference = native - averaged[:, 1:7]
    atomic_npz(
        output / "comparison.npz",
        {
            "relative_energy_eV": averaged[:, 0],
            "native_cartesian_A2": native_cartesian,
            "native_directional_A2": native,
            "historical_directional_A2": averaged[:, 1:7],
            "difference_directional_A2": difference,
            "native_integrated_directional_inv_A": normalization * native,
            "historical_integrated_directional_inv_A": integrated[:, 1:],
        },
    )
    report.update(
        status="completed",
        components=list(COMPONENTS),
        unit="Angstrom^2",
        reciprocal_unit_directions=directions.tolist(),
        reciprocal_direction_gram=(directions @ directions.T).tolist(),
        native_cartesian=native_cartesian.tolist(),
        native_directional=native.tolist(),
        historical_directional=averaged[:, 1:7].tolist(),
        diagnostics=diagnostic_runs,
        comparison={
            "compared_values": int(native.size),
            "max_absolute_difference_A2": float(np.abs(difference).max()),
            "rms_difference_A2": float(np.sqrt(np.mean(difference**2))),
            "same_estimator_acceptance": None,
            "continuum_convergence_established": False,
        },
        wall_seconds=time.monotonic() - start,
    )
    if args.finite_step:
        options = ExecutionOptions(
            workers=args.workers,
            batch_size=args.batch_size,
            native_threads=args.native_threads,
            memory_mb=args.memory_mb,
        )
        finite = finite_step_grid(
            model,
            chemical_potentials,
            grid,
            args.dimension,
            output / "finite-step-run",
            options,
            scfout,
            args.resume,
            args.finite_basis_link,
        )
        finite_difference = finite - averaged[:, 1:7]
        atomic_npz(
            output / "finite-step-comparison.npz",
            {
                "relative_energy_eV": averaged[:, 0],
                "native_finite_directional_A2": finite,
                "historical_directional_A2": averaged[:, 1:7],
                "difference_A2": finite_difference,
            },
        )
        report["finite_step_comparison"] = {
            "native_directional_A2": finite.tolist(),
            "compared_values": int(finite.size),
            "max_absolute_difference_A2": float(np.abs(finite_difference).max()),
            "rms_difference_A2": float(np.sqrt(np.mean(finite_difference**2))),
            "atol_A2": 5e-6,
            "rtol": 1e-6,
            "passed": bool(np.all(np.abs(finite_difference) <= 5e-6 + 1e-6 * np.abs(averaged[:, 1:7]))),
            "basis_link": (
                "native midpoint first-order overlap, then full-band polar factor"
                if args.finite_basis_link == "midpoint_first_order"
                else "independent centre-resummed midpoint link from native S/r, then full-band polar factor"
            ),
            "centre_definition": (
                "real diagonal of native zero-translation position blocks, without overlap division"
                if args.finite_basis_link == "centre_resummed"
                else None
            ),
            "historical_generation_revision_verified": False,
            "limitations": [
                "Finite links at a coarse mesh depend on the chosen orbital basis/centre convention; this estimator does not assert arbitrary orbital-mixing covariance.",
                "Retrieved producer source confirms this estimator formula, but its exact historical generation revision is not established.",
                "Original Hamiltonian bytes remain inferred; the archived runtime binds the mesh, dimension, energy and printed Fermi reference.",
            ],
        }
        report["wall_seconds"] = time.monotonic() - start
    atomic_json(output / "comparison.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scfout", type=Path, required=True)
    parser.add_argument("--expected-scfout-sha256", required=True)
    parser.add_argument("--average", type=Path, required=True)
    parser.add_argument("--integrated", type=Path, required=True)
    parser.add_argument("--grid", type=int, nargs=3, required=True)
    parser.add_argument("--dimension", type=int, choices=(0, 1), required=True)
    parser.add_argument("--binding", choices=("verified", "inferred"), required=True)
    parser.add_argument("--runtime-input-json", type=Path)
    parser.add_argument("--runtime-log", type=Path)
    parser.add_argument(
        "--finite-step",
        action="store_true",
        help="Also compare independent native finite projector distances",
    )
    parser.add_argument(
        "--finite-basis-link",
        choices=("midpoint_first_order", "centre_resummed"),
        default="midpoint_first_order",
        help="Explicit overlap approximation for the offline finite-step reference estimator",
    )
    parser.add_argument("--source-file", type=Path, action="append", default=[])
    parser.add_argument("--producer-hartree-ev", type=float, default=27.211324570273)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--native-threads", type=int, default=1)
    parser.add_argument("--memory-mb", type=float, default=512)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.finite_basis_link != "midpoint_first_order" and not args.finite_step:
        parser.error("--finite-basis-link centre_resummed requires --finite-step")
    report = replay(args)
    summary = {"status": report["status"], "comparison": report["comparison"]}
    if "finite_step_comparison" in report:
        summary["finite_step_comparison"] = report["finite_step_comparison"]
    print(json.dumps(summary, indent=2))
    return int(report.get("finite_step_comparison", {}).get("passed") is False)


if __name__ == "__main__":
    raise SystemExit(main())
