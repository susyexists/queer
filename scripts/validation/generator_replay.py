"""Offline native replay of downloaded Generator PythTB bands and total DOS.

Run after installing QUEER; this utility performs no network or authentication
operations. Original result arrays define the comparison points and conventions,
while native readers, WS interpolation and the QUEER eigensolver do the replay.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np

from queer import ExecutionOptions, WannierModel
from queer.execution import execution_session
from queer.geometry.lattice import validated_lattice
from queer.io.artifacts import (
    ALGORITHM_VERSION,
    atomic_json,
    atomic_npz,
    digest_file,
    environment_versions,
)
from queer.io.wannier import (
    BOHR_ANGSTROM,
    normalized_operators,
    read_centres_xyz,
    read_hr,
    read_wsvec,
    recompute_wsvec,
)


def read_win(path):
    """Read only lattice and MP mesh, retaining the source's Cartesian precision."""
    lines = [line.split("!", 1)[0].split("#", 1)[0].strip() for line in Path(path).read_text().splitlines()]
    lower = [line.lower() for line in lines]
    try:
        start = lower.index("begin unit_cell_cart") + 1
        stop = lower.index("end unit_cell_cart", start)
    except ValueError as error:
        raise ValueError("WIN must supply a unit_cell_cart block") from error
    block = [line for line in lines[start:stop] if line]
    scale = 1.0
    if block and block[0].lower() in {"ang", "angstrom", "bohr"}:
        scale = BOHR_ANGSTROM if block.pop(0).lower() == "bohr" else 1.0
    lattice = validated_lattice([[float(value) for value in line.split()] for line in block]) * scale
    mesh = None
    for line in lines:
        key, separator, value = line.partition("=")
        if separator and key.strip().lower() == "mp_grid":
            mesh = [int(item) for item in value.replace(",", " ").split()]
    if mesh is not None and (len(mesh) != 3 or min(mesh) < 1):
        raise ValueError("WIN mp_grid must contain three positive integers")
    return lattice, mesh


def error_metrics(native, reference, *, atol=1e-8, rtol=1e-10):
    """Compare identical axes without interpolation, fitting or energy alignment."""
    native, reference = np.asarray(native, float), np.asarray(reference, float)
    if native.shape != reference.shape or not reference.size:
        raise ValueError(f"Comparison shapes must match and be nonempty: {native.shape}, {reference.shape}")
    valid = np.isfinite(native) & np.isfinite(reference)
    count = int(valid.sum())
    delta = native[valid] - reference[valid]
    complete = count == reference.size
    return {
        "shape": list(reference.shape),
        "expected_values": int(reference.size),
        "compared_values": count,
        "coverage": count / reference.size,
        "max_absolute_error": float(np.abs(delta).max()) if count else None,
        "rms_error": float(np.sqrt(np.mean(delta**2))) if count else None,
        "mean_absolute_error": float(np.abs(delta).mean()) if count else None,
        "mean_signed_error": float(delta.mean()) if count else None,
        "atol": atol,
        "rtol": rtol,
        "passed": bool(complete and np.all(np.abs(delta) <= atol + rtol * np.abs(reference[valid]))),
    }


def validate_producer_fingerprints(params_file, provenance, core_file=None):
    """Validate original producer bytes; the optional core is never imported."""
    integrity = {}
    for role, path in (("params", params_file), ("core", core_file)):
        expected = provenance.get(f"{role}_sha256")
        actual = digest_file(path) if path is not None else None
        verified = expected is not None and actual is not None
        if verified and actual != expected:
            raise ValueError(f"Producer {role} SHA256 differs from recorded provenance")
        integrity[role] = {
            "recorded_sha256": expected,
            "provided_sha256": actual,
            "verified": verified,
        }
    return integrity


def build_native_model(directory, historical, *, seed="wannier90", search_size=2, chunk_bytes=32_000_000):
    """Replay recorded interpolation and vacuum folding from original Wannier files."""
    directory = Path(directory)
    files = {
        "hr": directory / f"{seed}_hr.dat",
        "centres": directory / f"{seed}_centres.xyz",
        "win": directory / f"{seed}.win",
    }
    raw = read_hr(files["hr"])
    centres, _ = read_centres_xyz(files["centres"])
    lattice, mp_grid = read_win(files["win"])
    if raw.nbnd != historical["num_wann"]:
        raise ValueError("Historical orbital count differs from HR")
    if not np.allclose(lattice, historical["lattice"], rtol=1e-8, atol=1e-7):
        raise ValueError("Historical and WIN lattices differ")
    if any(value is not None for value in historical.get("cutoffs", {}).values()):
        raise ValueError("Replay of non-null historical hopping cutoffs is not implemented")
    correction = historical.get("ws_correction", {}).get("mode")
    if correction == "recompute":
        if mp_grid is None:
            raise ValueError("Recorded WS recomputation requires WIN mp_grid")
        if historical.get("mp_grid") is not None and list(historical["mp_grid"]) != mp_grid:
            raise ValueError("Historical and WIN MP grids differ")
        mapping = recompute_wsvec(
            raw, centres, lattice, mp_grid, search_size=search_size, chunk_bytes=chunk_bytes
        )
        translations, hopping, _ = normalized_operators(raw, "ws_distance", mapping)
    elif correction in {"off", "none", "standard"}:
        translations, hopping, _ = normalized_operators(raw, "standard", None)
    elif correction == "wsvec":
        files["wsvec"] = directory / f"{seed}_wsvec.dat"
        translations, hopping, _ = normalized_operators(raw, "ws_distance", read_wsvec(files["wsvec"]))
    else:
        raise ValueError(f"Unsupported recorded WS correction: {correction!r}")
    ws_count = len(translations)
    periodic = historical.get("periodic_dirs")
    if (
        not isinstance(periodic, list)
        or not periodic
        or len(set(periodic)) != len(periodic)
        or any(isinstance(axis, bool) or axis not in {0, 1, 2} for axis in periodic)
    ):
        raise ValueError("Recorded periodic_dirs must identify distinct Cartesian fractional axes")
    folded_axes = sorted(set(range(3)) - set(periodic))
    if historical.get("folded_axes", folded_axes) != folded_axes:
        raise ValueError("Historical folded_axes conflicts with periodic_dirs")
    if folded_axes:
        folded = translations.copy()
        folded[:, folded_axes] = 0
        translations, inverse = np.unique(folded, axis=0, return_inverse=True)
        folded_hopping = np.zeros((len(translations), raw.nbnd, raw.nbnd), complex)
        np.add.at(folded_hopping, inverse, hopping)
        hopping = folded_hopping
    for key, actual in (
        ("n_R", len(raw.translations)),
        ("n_R_after_ws", ws_count),
        ("n_R_model", len(translations)),
    ):
        if historical.get(key) is not None and historical[key] != actual:
            raise ValueError(f"Historical {key}={historical[key]} differs from native {actual}")
    model = WannierModel(
        translations, hopping, lattice=lattice, centres=centres, fermi_energy=historical["fermi_energy"]
    )
    return (
        model,
        files,
        {
            "num_wann": model.nbnd,
            "n_R": len(raw.translations),
            "n_R_after_ws": ws_count,
            "n_R_model": len(translations),
            "ws_correction": correction,
            "ws_search_size": search_size,
            "ws_distance_tolerance_A": 1e-5,
            "mp_grid": mp_grid,
            "periodic_dirs": periodic,
            "folded_axes": folded_axes,
            "fermi_energy_eV": model.fermi_energy,
            "lattice_A": lattice.tolist(),
            "implicit_spin_multiplicity": 1,
        },
    )


def replay_total_dos(model, result, params, original, execution, *, progress=None):
    """Use native eigensolutions with the recorded histogram/Gaussian estimator.

    This compatibility estimator preserves historical bins and discrete smoothing;
    it is distinct from direct level broadening in queer.observables.dos.
    """
    energy = np.asarray(original["energy_eV"], float)
    reference = np.asarray(original["dos_per_eV"], float)
    step = float(params.get("dos_step_eV") or 0.005)
    sigma = float(result["dos"]["sigma_eV"])
    if not math.isfinite(step) or step <= 0 or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Historical DOS step and sigma must be positive finite numbers")
    if (
        energy.ndim != 1
        or len(energy) < 2
        or reference.shape != energy.shape
        or not np.isfinite(energy).all()
    ):
        raise ValueError("Historical DOS requires finite one-dimensional aligned axes")
    if not np.allclose(np.diff(energy), step, atol=1e-12, rtol=1e-10):
        raise ValueError("Historical DOS energy centers do not match the recorded spacing")
    periodic = result["model"]["periodic_dirs"]
    divisions = result["dos"]["mesh"]
    mesh_axes = result["dos"].get("mesh_axes", periodic)
    if len(divisions) == 3 and "mesh_axes" not in result["dos"]:
        divisions = [divisions[axis] for axis in periodic]
    if len(mesh_axes) != len(divisions) or set(mesh_axes) != set(periodic):
        raise ValueError("Historical DOS mesh axes conflict with model periodicity")
    by_axis = dict(zip(mesh_axes, divisions, strict=True))
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in by_axis.values()):
        raise ValueError("Historical DOS divisions must be positive integers")
    axes = [np.arange(by_axis[axis]) / by_axis[axis] if axis in by_axis else np.zeros(1) for axis in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    if len(points) != result["dos"]["k_points"]:
        raise ValueError("Historical DOS mesh point count does not match its divisions")
    edges = np.concatenate((energy - step / 2, [energy[-1] + step / 2]))
    counts = np.zeros(len(energy), np.int64)
    sampled_values = 0
    tile_size = execution.batch_size * execution.workers
    for start in range(0, len(points), tile_size):
        subset = points[start : start + tile_size]
        values = model.energies(subset, execution=execution)
        counts += np.histogram(values.ravel(), bins=edges)[0]
        sampled_values += values.size
        if progress is not None:
            progress(min(start + len(subset), len(points)), len(points))
    half = int(math.ceil(5 * sigma / step))
    offsets = np.arange(-half, half + 1) * step
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    if len(kernel) > len(energy):
        raise ValueError("Historical DOS grid is shorter than its smoothing kernel")
    density = np.convolve(counts / len(points) / step, kernel, mode="same")
    return {
        "energy_relative_eV": energy,
        "native_dos_per_eV": density,
        "reference_dos_per_eV": reference,
        "difference_per_eV": density - reference,
        "histogram_counts": counts,
        "kpoints_fractional": points,
    }, {
        "estimator": "eigenvalue histogram; discrete normalized Gaussian truncated at 5 sigma",
        "energy_reference": "E - EF",
        "unit": "states/eV/cell",
        "sigma_eV": sigma,
        "bin_step_eV": step,
        "mesh_axes": mesh_axes,
        "mesh_divisions": divisions,
        "k_points": len(points),
        "sampled_eigenvalues": sampled_values,
        "histogram_coverage": int(counts.sum()) / sampled_values,
        "out_of_range_eigenvalues": sampled_values - int(counts.sum()),
        "native_integral_states": float(density.sum() * step),
        "historical_integral_states": float(reference.sum() * step),
        "pdos_compared": False,
    }


def fermi_dos_diagnostics(energy, native, reference, recorded, *, policy="first_nonnegative_bin"):
    """Retain the historical scalar definition alongside common EF estimators.

    The selected historical producer used searchsorted(centers, 0), which picks
    the first nonnegative bin rather than the nearest center or interpolation.
    Later producers may use linear interpolation; callers must select that
    documented policy explicitly rather than fit the definition to agreement.
    """
    energy, native, reference = (np.asarray(values, float) for values in (energy, native, reference))
    if (
        energy.ndim != 1
        or not len(energy)
        or native.shape != energy.shape
        or reference.shape != energy.shape
        or not all(np.isfinite(values).all() for values in (energy, native, reference))
        or np.any(np.diff(energy) <= 0)
        or not energy[0] <= 0 <= energy[-1]
    ):
        raise ValueError("Fermi DOS diagnostics require finite increasing axes containing EF")
    first = min(int(np.searchsorted(energy, 0.0)), len(energy) - 1)
    nearest = int(np.argmin(np.abs(energy)))
    diagnostics = {}
    for name, index in (("first_nonnegative_bin", first), ("nearest_bin", nearest)):
        diagnostics[name] = {
            "bin_index": index,
            "energy_relative_eV": float(energy[index]),
            "native_per_eV": float(native[index]),
            "reference_per_eV": float(reference[index]),
        }
    diagnostics["linear_interpolation"] = {
        "energy_relative_eV": 0.0,
        "native_per_eV": float(np.interp(0.0, energy, native)),
        "reference_per_eV": float(np.interp(0.0, energy, reference)),
    }
    if policy not in diagnostics:
        raise ValueError(f"Unsupported historical Fermi DOS policy: {policy!r}")
    if recorded is None or not math.isfinite(float(recorded)):
        raise ValueError("Historical electronic metadata must contain a finite DOS-at-EF scalar")
    return {
        "energy_reference": "E - EF",
        "unit": "states/eV/cell",
        "historical_policy": policy,
        "recorded_per_eV": float(recorded),
        "estimators": diagnostics,
    }


def add_fermi_dos_comparison(report, arrays, result, *, policy, atol, rtol):
    """Add scalar diagnostics from saved arrays without repeating eigensolutions."""
    diagnostics = fermi_dos_diagnostics(
        arrays["energy_relative_eV"],
        arrays["native_dos_per_eV"],
        arrays["reference_dos_per_eV"],
        result.get("electronic", {}).get("dos_at_fermi_per_eV"),
        policy=policy,
    )
    selected = diagnostics["estimators"][policy]
    report["dos"]["at_fermi"] = diagnostics
    report["comparisons"]["dos_at_fermi"] = {
        **error_metrics([selected["native_per_eV"]], [diagnostics["recorded_per_eV"]], atol=atol, rtol=rtol),
        "unit": "states/eV/cell",
        "historical_policy": policy,
        "native_value": selected["native_per_eV"],
        "reference_value": diagnostics["recorded_per_eV"],
    }
    report["comparisons"]["dos_at_fermi_artifact_consistency"] = {
        **error_metrics(
            [selected["reference_per_eV"]], [diagnostics["recorded_per_eV"]], atol=atol, rtol=rtol
        ),
        "unit": "states/eV/cell",
        "historical_policy": policy,
    }


def progress_logger(label, *, interval_seconds=10.0):
    """Return a throttled progress callback; diagnostics never alter sampling."""
    started = time.perf_counter()
    previous = None

    def log(completed, total):
        nonlocal previous
        now = time.perf_counter()
        if previous is None or completed == total or now - previous >= interval_seconds:
            print(
                f"{label}: {completed}/{total} k-points ({completed / total:.1%}), "
                f"{now - started:.1f} s elapsed",
                file=sys.stderr,
                flush=True,
            )
            previous = now

    return log


def write_report(output, report):
    atomic_json(output / "comparison.json", report)
    lines = [
        "# Native Generator replay",
        "",
        "Offline comparison using QUEER native Wannier interpolation and eigensolutions.",
        "",
    ]
    ids = report["original"]
    lines.append(
        f"Original workflow: `{ids['workflow_id']}`; job: `{ids['job_id']}`; Wannier source: `{ids['source_job_id']}`."
    )
    lines.extend(
        [
            "",
            "| Quantity | Coverage | Max absolute error | RMS error | Mean absolute error | Passed |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for name, comparison in report["comparisons"].items():
        unit = comparison["unit"]
        numbers = [
            "undefined" if comparison[key] is None else f"{comparison[key]:.9g} {unit}"
            for key in ("max_absolute_error", "rms_error", "mean_absolute_error")
        ]
        lines.append(
            f"| {name} | {comparison['coverage']:.1%} | {' | '.join(numbers)} | {comparison['passed']} |"
        )
    lines.extend(
        [
            "",
            "Bands retain the historical absolute eV zero; no fitted shift or resampling is applied. Nonperiodic translation components are folded onto zero, and Wannier centres remain unwrapped.",
            "",
            "Species PDOS and quantum geometry are outside this comparison. Band/DOS agreement does not validate response or geometry magnitudes.",
            "",
            "Input SHA256 values, original producer provenance, native versions, resource limits and array checksums are recorded in `comparison.json`.",
            "",
        ]
    )
    if "dos" in report:
        lines.append(
            "DOS uses the original mesh, bins and discrete Gaussian estimator with native QUEER eigenvalues; this differs from direct per-level Gaussian broadening.\n"
        )
        diagnostics = report["dos"].get("at_fermi")
        if diagnostics:
            lines.extend(
                [
                    f"The original DOS-at-EF scalar uses `{diagnostics['historical_policy']}`; the recorded value is {diagnostics['recorded_per_eV']:.12g} states/eV/cell.",
                    "",
                    "| EF estimator | Native states/eV/cell | Original artifact states/eV/cell |",
                    "| --- | ---: | ---: |",
                ]
            )
            for name, values in diagnostics["estimators"].items():
                lines.append(
                    f"| {name} | {values['native_per_eV']:.12g} | {values['reference_per_eV']:.12g} |"
                )
            lines.append("")
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def replay(args):
    started = time.perf_counter()
    producer = args.producer_dir.resolve()
    output = args.output_dir.resolve()
    result_file, params_file = producer / "pythtb_results.json", producer / "pythtb_params.json"
    result = json.loads(result_file.read_text())
    params = json.loads(params_file.read_text())
    if (
        result.get("schema_version") != 1
        or result.get("status") != "ok"
        or result.get("step_type") != "PYTHTB_BANDS"
    ):
        raise ValueError("Replay requires a successful schema-1 PYTHTB_BANDS numerical result")
    producer_integrity = validate_producer_fingerprints(
        params_file, result.get("provenance", {}), args.producer_core
    )
    band_data = result["bands"]
    coordinate_key = next((key for key in ("kpoints_frac", "kpoints_fractional") if key in band_data), None)
    if coordinate_key is None:
        raise ValueError("Historical bands lack fractional comparison points")
    points = np.asarray(band_data[coordinate_key], float)
    reference = np.asarray(band_data["eigenvalues_eV"], float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Historical k-points must have finite shape (Nk,3)")
    if reference.shape != (len(points), result["model"]["num_wann"]):
        raise ValueError("Historical band energies must have shape (Nk,Nb)")
    for axis in result["model"].get("folded_axes", []):
        if np.any(np.abs(points[:, axis]) > 1e-12):
            raise ValueError("Historical path samples a folded nonperiodic axis")
    if not np.isclose(band_data["fermi_energy"], result["model"]["fermi_energy"], atol=1e-12, rtol=0):
        raise ValueError("Historical band and model energy references differ")
    execution = ExecutionOptions(
        workers=args.workers,
        native_threads=args.native_threads,
        batch_size=args.batch_size,
        memory_mb=args.memory_mb,
    ).resolved()
    if args.progress:
        print("Building native model from offline Wannier inputs...", file=sys.stderr, flush=True)
    model, source_files, model_meta = build_native_model(
        args.wannier_dir, result["model"], seed=args.seed, search_size=args.ws_search_size
    )
    source_files.update({"producer_result": result_file, "producer_parameters": params_file})
    if args.producer_core is not None:
        source_files["producer_core"] = args.producer_core
    job = {}
    workflow = {}
    for role, path in (("job_metadata", args.job_metadata), ("workflow_metadata", args.workflow_metadata)):
        if path is not None:
            source_files[role] = path
            data = json.loads(path.read_text())
            if role == "job_metadata":
                job = data
            else:
                workflow = data
    if args.download_manifest is not None:
        source_files["download_manifest"] = args.download_manifest
    inputs = {
        role: {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": digest_file(path)}
        for role, path in source_files.items()
    }
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "algorithm_version": ALGORITHM_VERSION,
        "status": "bands_completed",
        "original": {
            "job_id": job.get("id", producer.name),
            "workflow_id": workflow.get("id", job.get("workflowId")),
            "source_job_id": result["model"].get(
                "source_wannier_job_id", params.get("source_wannier_job_id")
            ),
            "reference_band_job_id": result["model"].get(
                "reference_band_job_id", params.get("reference_band_job_id")
            ),
            "provenance": result.get("provenance", {}),
        },
        "inputs": inputs,
        "producer_integrity": producer_integrity,
        "native_environment": environment_versions(),
        "execution": execution.as_dict(),
        "native_model": model_meta,
        "conventions": {
            "coordinates": "fractional reciprocal (Nk,3)",
            "bands": "absolute eV, (Nk,Nb), ascending eigenvalues at identical recorded points",
            "fourier_phase": "exp(+2 pi i k.R)",
            "hopping": "HR degeneracy weights and recorded Wigner-Seitz splitting applied once",
            "position": "original unwrapped Wannier centres; no off-diagonal position operators",
            "alignment": "none",
        },
        "comparisons": {},
        "artifacts": {},
    }
    with execution_session(execution):
        band_started = time.perf_counter()
        native = model.energies(points, execution=execution).T + model.fermi_energy
        report["bands_wall_seconds"] = time.perf_counter() - band_started
        metrics = error_metrics(native, reference, atol=args.atol_ev, rtol=args.rtol)
        report["comparisons"]["bands"] = {
            **metrics,
            "unit": "eV",
            "k_points": len(points),
            "bands": model.nbnd,
        }
        arrays = {
            "kpoints_fractional": points,
            "native_absolute_eV": native,
            "reference_absolute_eV": reference,
            "difference_eV": native - reference,
        }
        for key in ("tick_indices", "tick_labels", "break_indices"):
            if key in band_data:
                arrays[key] = np.asarray(band_data[key])
        atomic_npz(output / "bands.npz", arrays)
        report["artifacts"]["bands"] = {"path": "bands.npz", "sha256": digest_file(output / "bands.npz")}
        report["max_hermiticity_residual_eV"] = model.max_hermiticity_residual
        write_report(output, report)
        if args.with_dos:
            dos_file = producer / "Out/dos.json"
            report["inputs"]["producer_dos"] = {
                "path": str(dos_file),
                "size": dos_file.stat().st_size,
                "sha256": digest_file(dos_file),
            }
            dos_started = time.perf_counter()
            arrays, dos_meta = replay_total_dos(
                model,
                result,
                params,
                json.loads(dos_file.read_text()),
                execution,
                progress=progress_logger("DOS") if args.progress else None,
            )
            report["dos_wall_seconds"] = time.perf_counter() - dos_started
            report["dos"] = dos_meta
            metrics = error_metrics(
                arrays["native_dos_per_eV"],
                arrays["reference_dos_per_eV"],
                atol=args.dos_atol,
                rtol=args.rtol,
            )
            report["comparisons"]["dos"] = {**metrics, "unit": "states/eV/cell"}
            add_fermi_dos_comparison(
                report, arrays, result, policy=args.fermi_dos_policy, atol=args.dos_atol, rtol=args.rtol
            )
            atomic_npz(output / "dos.npz", arrays)
            report["artifacts"]["dos"] = {"path": "dos.npz", "sha256": digest_file(output / "dos.npz")}
    report["status"] = "completed"
    report["total_wall_seconds"] = time.perf_counter() - started
    report["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["max_hermiticity_residual_eV"] = model.max_hermiticity_residual
    write_report(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-dir", type=Path, required=True)
    parser.add_argument("--wannier-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job-metadata", type=Path)
    parser.add_argument("--workflow-metadata", type=Path)
    parser.add_argument("--download-manifest", type=Path)
    parser.add_argument(
        "--producer-core", type=Path, help="Verify the staged producer core SHA256 without importing it"
    )
    parser.add_argument("--seed", default="wannier90")
    parser.add_argument("--ws-search-size", type=int, default=2)
    parser.add_argument("--with-dos", action="store_true", help="Replay the complete recorded DOS mesh")
    parser.add_argument("--progress", action="store_true", help="Log native mesh progress to stderr")
    parser.add_argument(
        "--fermi-dos-policy",
        choices=("first_nonnegative_bin", "nearest_bin", "linear_interpolation"),
        default="first_nonnegative_bin",
        help="Documented historical DOS-at-EF definition; the selected original producer uses the first nonnegative bin",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--native-threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--memory-mb", type=float, default=256)
    parser.add_argument("--atol-ev", type=float, default=1e-8)
    parser.add_argument("--dos-atol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-10)
    args = parser.parse_args(argv)
    for name in ("atol_ev", "dos_atol", "rtol"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"{name} must be finite and nonnegative")
    report = replay(args)
    print(
        json.dumps(
            {"output_dir": str(args.output_dir.resolve()), "comparisons": report["comparisons"]}, indent=2
        )
    )
    return 0 if all(value["passed"] for value in report["comparisons"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
