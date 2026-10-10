"""Offline native replay of the historical Generator/PyATB GaAs shift-current tables.

The external producer sources are hashed and read as provenance only. They are
never imported or executed. Every recorded frequency and all 18 independent
Cartesian components are compared without fitting, resampling, or sign changes.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from queer import ExecutionOptions, OpenMXModel
from queer.compat.generator import translate_pyatb
from queer.execution import execution_session
from queer.io.artifacts import (
    ALGORITHM_VERSION,
    atomic_json,
    atomic_npz,
    digest_file,
    environment_versions,
)
from queer.observables._tb import E_CHARGE, HBAR, band_geometry, energy_blocks, kmesh, spin_degeneracy, volume
from queer.observables.nonlinear import _delta, _differentiated_geometry, shift_current

PRODUCER_HEAD = "60fc6b2acafac36b9c8334208326937fef3366c5"
SCFOUT_SHA256 = "d7d9027f0cd9c5547ea13893166ccaba36f2633bfd2ecd4e1259dd7c3daa9b3b"
COMPONENTS = (
    "xxx",
    "xxy",
    "xxz",
    "xyy",
    "xyz",
    "xzz",
    "yxx",
    "yxy",
    "yxz",
    "yyy",
    "yyz",
    "yzz",
    "zxx",
    "zxy",
    "zxz",
    "zyy",
    "zyz",
    "zzz",
)
BROKER_ROOT = Path("data/generator_replay_20261010/broker/openmx_ahc/pyatb")


def fingerprint(path):
    """Validate downloaded bytes against a broker SHA256 sidecar when supplied."""
    path = Path(path).resolve()
    actual = digest_file(path)
    sidecar = Path(str(path) + ".broker.json")
    record = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    expected = record.get("sha256")
    if expected is not None and actual != expected:
        raise ValueError(f"Downloaded SHA256 mismatch: {path}")
    if record.get("size") is not None and path.stat().st_size != record["size"]:
        raise ValueError(f"Downloaded size mismatch: {path}")
    return {
        "path": str(path),
        "remote_path": record.get("remote_path", record.get("path")),
        "size": path.stat().st_size,
        "sha256": actual,
        "broker_sha256_verified": expected is not None,
    }


def error_metrics(native, reference, *, atol, rtol):
    """Compare all values on exactly matching axes, retaining near-zero components."""
    native, reference = np.asarray(native, float), np.asarray(reference, float)
    if native.shape != reference.shape or not reference.size:
        raise ValueError("Comparison requires identical nonempty shapes")
    valid = np.isfinite(native) & np.isfinite(reference)
    delta = native[valid] - reference[valid]
    count = int(valid.sum())
    maximum = float(np.max(np.abs(reference[valid]))) if count else 0.0
    return {
        "shape": list(reference.shape),
        "expected_values": int(reference.size),
        "compared_values": count,
        "coverage": count / reference.size,
        "max_absolute_error": float(np.max(np.abs(delta))) if count else None,
        "rms_error": float(np.sqrt(np.mean(delta**2))) if count else None,
        "mean_absolute_error": float(np.mean(np.abs(delta))) if count else None,
        "mean_signed_error": float(np.mean(delta)) if count else None,
        "reference_max_absolute": maximum,
        "native_max_absolute": float(np.max(np.abs(native[valid]))) if count else None,
        "max_error_over_reference_peak": float(np.max(np.abs(delta)) / maximum) if maximum > 0 else None,
        "atol": atol,
        "rtol": rtol,
        "passed": bool(
            count == reference.size and np.all(np.abs(delta) <= atol + rtol * np.abs(reference[valid]))
        ),
    }


def read_case(case_dir, model):
    case_dir = Path(case_dir).resolve()
    paths = {
        "input": case_dir / "Input",
        "resolved_input": case_dir / "Out/input.json",
        "running_log": case_dir / "Out/running.log",
        "reference": case_dir / "Out/Shift_Current/shift_current.dat",
    }
    files = {name: fingerprint(path) for name, path in paths.items()}
    resolved = json.loads(paths["resolved_input"].read_text())
    recorded = dict(resolved["SHIFT_CURRENT"])
    sentinel = None
    if [recorded.get("n_occ"), recorded.get("m_unocc")] == [-1, -1]:
        sentinel = {"n_occ": recorded["n_occ"], "m_unocc": recorded["m_unocc"]}
    translated = translate_pyatb(
        "SHIFT_CURRENT",
        recorded,
        input_parameters=resolved["INPUT_PARAMETERS"],
        fermi_energy=model.fermi_energy,
        spin_mode=model.spin_mode,
    )
    parameters = dict(translated["parameters"])
    if "ef" in parameters:
        parameters["chemical_potential"] = parameters.pop("ef")
    reference = np.loadtxt(paths["reference"])
    header = paths["reference"].read_text().splitlines()[0].split()
    if tuple(header[2:20]) != COMPONENTS:
        raise ValueError("Historical tensor header differs from the audited 18-component order")
    if reference.ndim != 2 or reference.shape[1] != 19 or not np.isfinite(reference).all():
        raise ValueError("Historical table must contain finite omega plus 18 components")
    omega = np.asarray(parameters["omega"])
    if reference.shape[0] != len(omega) or not np.allclose(reference[:, 0], omega, atol=5e-6, rtol=0):
        raise ValueError("Historical frequencies differ from the recorded input scan")
    # Evaluate at the printed historical photon energies themselves, never interpolate.
    parameters["omega"] = reference[:, 0].copy()
    lattice = resolved["LATTICE"]
    if lattice["lattice_constant_unit"] != "Angstrom":
        raise ValueError("Resolved historical lattice must use Angstrom")
    producer_cell = np.asarray(lattice["lattice_vector"], float) * lattice["lattice_constant"]
    if not np.allclose(producer_cell, model.lattice, atol=1e-6, rtol=1e-8):
        raise ValueError("Historical input cell differs from native SCFOUT cell")
    return (
        parameters,
        reference,
        {
            "case_dir": str(case_dir),
            "files": files,
            "recorded_parameters": resolved["SHIFT_CURRENT"],
            "native_parameters": {key: value for key, value in parameters.items() if key != "omega"},
            "disabled_band_pair_sentinel": sentinel,
            "translation": translated["export_metadata"],
            "producer_cell_angstrom": producer_cell.tolist(),
            "cell_max_absolute_difference_angstrom": float(np.max(np.abs(producer_cell - model.lattice))),
        },
    )


def independent_components(tensor):
    tensor = np.asarray(tensor)
    if tensor.ndim != 4 or tensor.shape[1:] != (3, 3, 3):
        raise ValueError("Native shift-current tensor must have shape (Nw,3,3,3)")
    indices = {axis: i for i, axis in enumerate("xyz")}
    return np.column_stack([tensor[:, *(indices[axis] for axis in component)] for component in COMPONENTS])


def finite_difference_point_check(
    model,
    k,
    *,
    tolerance=None,
    steps=(1e-5, 1e-6, 1e-7),
    omega=4.95,
    occupied_bands=28,
    scissors=1.15,
    eta=0.05 / np.sqrt(2),
):
    """Check connection derivatives with independent transported central differences.

    Eigenstates, position matrices and response conventions are shared with the
    native calculation. Only the Cartesian connection derivative is independent;
    this is a derivative diagnostic, not an independent material response oracle.
    """
    from scipy.linalg import polar

    k = np.asarray(k, float)
    data = band_geometry(model, k, tolerance)
    analytic, _, _ = _differentiated_geometry(model, k, data)
    energy = data["energies"] + scissors * (np.arange(model.nbnd) >= occupied_bands)
    gap = energy[None, :] - energy[:, None]
    block = np.abs(gap) <= data["degeneracy_tol"]
    connection = data["connections"]
    block_connection = connection * block[None]
    weights = _delta(omega - gap, eta, "gaussian")
    weights *= (np.arange(model.nbnd)[:, None] < occupied_bands) & (
        np.arange(model.nbnd)[None, :] >= occupied_bands
    )
    conversion = -np.pi * E_CHARGE**2 / HBAR * 1e6 * spin_degeneracy(model) / volume(model)

    def response(gradient):
        generalized = gradient - 1j * (
            np.einsum("anl,blm->abnm", block_connection, connection)
            - np.einsum("bnl,alm->abnm", connection, block_connection)
        )
        integrand = np.einsum("bmn,acnm->abcnm", connection, generalized)
        integrand = 0.5 * (integrand + integrand.swapaxes(1, 2)).imag
        return conversion * np.einsum("nm,abcnm->abc", weights, integrand)

    expected = response(analytic)
    records = []
    for step in steps:
        numerical = np.empty_like(analytic)
        for axis in range(3):
            delta = np.eye(3)[axis] @ np.linalg.inv(model.g_vec) * step
            neighbors = []
            for sign in (1, -1):
                neighbor = band_geometry(model, k + sign * delta, tolerance)
                overlaps = data["coefficients"].conj().T @ data["overlap"] @ neighbor["coefficients"]
                transport = np.zeros_like(overlaps)
                for bands in energy_blocks(data):
                    unitary, _ = polar(overlaps[np.ix_(bands, bands)])
                    transport[np.ix_(bands, bands)] = unitary.conj().T
                neighbors.append(
                    np.einsum("in,aij,jm->anm", transport.conj(), neighbor["connections"], transport)
                )
            numerical[axis] = (neighbors[0] - neighbors[1]) / (2 * step)
        actual = response(numerical)
        records.append(
            {
                "cartesian_step_inverse_angstrom": step,
                "finite_difference_tensor": actual.tolist(),
                "tensor_max_absolute_difference": float(np.max(np.abs(actual - expected))),
                "tensor_max_difference_over_analytic_peak": float(
                    np.max(np.abs(actual - expected)) / np.max(np.abs(expected))
                )
                if np.max(np.abs(expected)) > 0
                else None,
                "interband_connection_gradient_max_absolute_difference": float(
                    np.max(
                        np.abs(
                            numerical[:, :, :occupied_bands, occupied_bands:]
                            - analytic[:, :, :occupied_bands, occupied_bands:]
                        )
                    )
                ),
            }
        )
    return {
        "k_fractional": k.tolist(),
        "photon_energy_eV": omega,
        "occupied_bands": occupied_bands,
        "scissors_eV": scissors,
        "degeneracy_tolerance_eV": data["degeneracy_tol"],
        "analytic_tensor": expected.tolist(),
        "units": "microA/V^2 at one k-point before mesh averaging",
        "checks": records,
        "scope": "Independent connection derivatives; shared native eigensolver, dipoles and response contractions",
    }


def replay_case(model, case_dir, output, execution, *, atol, rtol, degeneracy_tol=None, result_cache=None):
    started = time.perf_counter()
    parameters, reference, provenance = read_case(case_dir, model)
    if degeneracy_tol is not None:
        parameters["degeneracy_tol"] = degeneracy_tol
        provenance["native_parameters"]["degeneracy_tol"] = degeneracy_tol
    print(
        f"Replaying {Path(case_dir).name}: {len(reference)} frequencies, grid={parameters['grid']}",
        flush=True,
    )
    cache_key = json.dumps(
        {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in parameters.items()
        },
        sort_keys=True,
    )
    reused = result_cache is not None and cache_key in result_cache
    if reused:
        result = result_cache[cache_key]
    else:
        with execution_session(execution), threadpool_limits(execution.native_threads):
            result = shift_current(model, execution=execution, **parameters)
        if result_cache is not None:
            result_cache[cache_key] = result
    tensor = result["shift_current"]
    values = independent_components(tensor)
    arrays_path = output / "shift_current.npz"
    atomic_npz(
        arrays_path,
        {
            "omega": reference[:, 0],
            "components": np.asarray(COMPONENTS),
            "native": values,
            "reference": reference[:, 1:],
            "difference": values - reference[:, 1:],
            "native_tensor": tensor,
            "kpoints_fractional": kmesh(parameters["grid"]),
        },
    )
    comparisons = {
        "all_components": error_metrics(values, reference[:, 1:], atol=atol, rtol=rtol),
        **{
            component: error_metrics(values[:, i], reference[:, i + 1], atol=atol, rtol=rtol)
            for i, component in enumerate(COMPONENTS)
        },
    }
    report = {
        "schema_version": 1,
        "quantity": "shift_current",
        "units": "microA/V^2",
        "producer": {
            "name": "PyATB",
            "source_checkout_head_observed": PRODUCER_HEAD,
            "historical_generation_commit_verified": False,
            **provenance,
        },
        "native_result_metadata": result.metadata,
        "comparisons": comparisons,
        "array_artifact": fingerprint(arrays_path),
        "elapsed_seconds": time.perf_counter() - started,
        "native_result_reused_for_identical_parameters": reused,
        "input_index_symmetry_max_absolute": float(np.max(np.abs(tensor - tensor.swapaxes(2, 3)))),
        "comparison_policy": "All printed frequencies and 18 Cartesian components; no interpolation, fit, scale, sign reversal, or energy alignment",
        "interpretation_limits": "Original tables and the historical SCFOUT content binding are verified. Inspected producer sources are the retrieved checkout snapshot; the source revision used to generate the historical tables is unverified. Attribution of historical differences to inspected source rules is therefore an inference.",
        "estimator_differences": [
            "Producer Gaussian is truncated around floor((gap-start)/domega) within int(5*eta/domega) bins and discards transitions outside the scan; native Gaussian is not truncated",
            "Producer applies scissors before derivative kernels, changing their energy denominators; native preserves unshifted dipole geometry and changes conduction resonance energies",
            "Producer uses historical SI and Bohr conversion constants; native uses current constants and the SCFOUT cell",
            "Native reconciles source real-space operator adjoints and uses complete degenerate manifolds; producer uses its original direct or sum-over estimator",
            "Inspected producer method 1 regularizes coefficient derivatives only at gaps <=1e-4 eV with delta/(delta^2+0.04^2); method 0 applies that regularizer to all first coefficient derivatives and a 1e-4 cutoff to second derivatives; native uses complete energy blocks and unregularized separated-band derivatives",
        ],
    }
    allowed = [COMPONENTS.index(component) for component in ("xyz", "yxz", "zxy")]
    forbidden = [i for i in range(18) if i not in allowed]
    report["gaas_point_group_diagnostics"] = {
        "allowed_components": ["xyz", "yxz", "zxy"],
        "native_forbidden_component_max_absolute": float(np.max(np.abs(values[:, forbidden]))),
        "reference_forbidden_component_max_absolute": float(
            np.max(np.abs(reference[:, np.asarray(forbidden) + 1]))
        ),
        "native_allowed_component_spread_max_absolute": float(np.max(np.ptp(values[:, allowed], axis=1))),
        "reference_allowed_component_spread_max_absolute": float(
            np.max(np.ptp(reference[:, np.asarray(allowed) + 1], axis=1))
        ),
        "interpretation": "Ideal bulk GaAs permits only permutations of xyz; retained source operators and finite mesh may break this symmetry, so these are diagnostics rather than fitted constraints",
    }
    atomic_json(output / "comparison.json", report)
    lines = [
        "# Historical GaAs shift-current native replay",
        "",
        f"Observed producer checkout `{PRODUCER_HEAD}` (historical generation commit unverified); recorded grid `{parameters['grid']}`; {len(reference)} frequencies.",
        "",
        "All 18 independent Cartesian components are compared in microA/V². No fitted scale, sign, energy shift, or interpolation is applied.",
        "",
        "| Component | Values | Max absolute error | RMS error | Reference peak | Passed |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for component, metrics in comparisons.items():
        lines.append(
            f"| {component} | {metrics['compared_values']}/{metrics['expected_values']} | {metrics['max_absolute_error']:.9g} | {metrics['rms_error']:.9g} | {metrics['reference_max_absolute']:.9g} | {metrics['passed']} |"
        )
    lines.extend(
        ["", "Estimator differences:", "", *[f"- {item}" for item in report["estimator_differences"]], ""]
    )
    lines.extend([report["interpretation_limits"], ""])
    (output / "comparison.md").write_text("\n".join(lines))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, action="append", required=True)
    parser.add_argument("--scfout", type=Path, default=Path("data/references/openmx/GaAs_soc.scfout"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--producer-python",
        type=Path,
        default=BROKER_ROOT / "src/pyatb/berry/shift_current.py",
    )
    parser.add_argument(
        "--producer-cpp", type=Path, default=BROKER_ROOT / "src/cpp/core/shift_current_solver.cpp"
    )
    parser.add_argument("--mesh-source", type=Path, default=BROKER_ROOT / "src/pyatb/kpt/kpoint_generator.py")
    parser.add_argument(
        "--download-manifest",
        type=Path,
        default=Path("results/generator_replay_20261010/broker_download_manifest.json"),
    )
    parser.add_argument(
        "--degeneracy-tol",
        type=float,
        action="append",
        help="Explicit native tolerance in eV; repeat for a controlled sweep without changing defaults",
    )
    parser.add_argument(
        "--derivative-check",
        action="store_true",
        help="Check analytic derivatives at the problematic symmetry-line point using transported central differences",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--native-threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--memory-mb", type=float, default=512)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=2e-6)
    args = parser.parse_args(argv)
    if any(not math.isfinite(value) or value < 0 for value in (args.atol, args.rtol)):
        parser.error("Comparison tolerances must be finite and nonnegative")
    if len({path.name for path in args.case_dir}) != len(args.case_dir):
        parser.error("Case-directory basenames must be distinct")
    tolerances = args.degeneracy_tol or [None]
    if any(value is not None and (not math.isfinite(value) or value <= 0) for value in tolerances):
        parser.error("Degeneracy tolerances must be finite and positive")
    scfout = fingerprint(args.scfout)
    if scfout["sha256"] != SCFOUT_SHA256:
        raise ValueError("SCFOUT differs from the exact SHA256-verified historical GaAs source")
    sources = {"python": fingerprint(args.producer_python), "cpp": fingerprint(args.producer_cpp)}
    if args.mesh_source is not None:
        sources["mesh"] = fingerprint(args.mesh_source)
    sources["coefficient_derivatives_cpp"] = fingerprint(BROKER_ROOT / "src/cpp/core/linear_response.cpp")
    sources["coefficient_derivatives_header"] = fingerprint(BROKER_ROOT / "src/cpp/core/linear_response.h")
    download_manifest = json.loads(args.download_manifest.read_text())
    if download_manifest.get("producer_git_head") != PRODUCER_HEAD:
        raise ValueError("Download manifest producer checkout differs from the audited source HEAD")
    model_binding = next(
        (record for record in download_manifest["models"] if record["sha256"] == scfout["sha256"]), None
    )
    if model_binding is None or not model_binding.get("verified_content_match"):
        raise ValueError("Download manifest lacks a verified historical SCFOUT content binding")
    repository = Path(__file__).resolve().parents[2]
    native_source_paths = [
        repository / "src/queer" / name
        for name in (
            "models/openmx.py",
            "io/openmx.py",
            "observables/_tb.py",
            "observables/nonlinear.py",
            "observables/optical.py",
            "observables/geometry.py",
            "compat/generator.py",
            "execution.py",
        )
    ]
    native_sources = {str(path): digest_file(path) for path in native_source_paths}
    execution = ExecutionOptions(
        workers=args.workers,
        native_threads=args.native_threads,
        batch_size=args.batch_size,
        memory_mb=args.memory_mb,
    ).resolved()
    with threadpool_limits(execution.native_threads):
        model = OpenMXModel.from_scfout(args.scfout, execution=execution)
    output = args.output_dir.resolve()
    reports = {}
    result_cache = {}
    for path in args.case_dir:
        for tolerance in tolerances:
            name = path.name if tolerance is None else f"{path.name}_tol_{tolerance!r}"
            reports[name] = replay_case(
                model,
                path,
                output / name,
                execution,
                atol=args.atol,
                rtol=args.rtol,
                degeneracy_tol=tolerance,
                result_cache=result_cache,
            )
    derivative_checks = []
    if args.derivative_check:
        with threadpool_limits(execution.native_threads):
            derivative_checks = [
                finite_difference_point_check(model, [0.25, 0.25, 0], tolerance=tolerance)
                for tolerance in (None, 1e-4)
            ]
        atomic_json(output / "derivative_checks.json", derivative_checks)
    summary = {
        "schema_version": 1,
        "algorithm_version": ALGORITHM_VERSION,
        "versions": environment_versions(),
        "native_source_sha256": native_sources,
        "scfout": scfout,
        "model_content_binding": model_binding,
        "download_manifest": fingerprint(args.download_manifest),
        "producer_sources": sources,
        "producer_sources_executed": False,
        "execution": execution.as_dict(),
        "native_source_operator_identity_residuals": model.metadata["source_operator_identity_residuals"],
        "native_degeneracy_tolerance_eV": model.degeneracy_tolerance,
        "mesh_convention": "Gamma-origin fractional reciprocal mesh i/N, upper boundary excluded, uniform weights",
        "gaussian_translation": "native standard deviation = producer eta/sqrt(2)",
        "cases": {name: report["comparisons"]["all_components"] for name, report in reports.items()},
        "degeneracy_diagnostics": {
            name: {
                "tolerance_eV": report["native_result_metadata"]["degeneracy_tolerance_eV"],
                **report["gaas_point_group_diagnostics"],
            }
            for name, report in reports.items()
        },
        "derivative_checks": derivative_checks,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    if any(digest_file(path) != native_sources[str(path)] for path in native_source_paths):
        raise RuntimeError("Native scientific source files changed while the replay was running")
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary["cases"], indent=2))
    return 0 if all(metrics["passed"] for metrics in summary["cases"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
