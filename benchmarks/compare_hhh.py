"""Development-only parity replay against an explicitly selected local HHH checkout."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import math
import os
import platform
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def _source(root):
    files = {str(path.relative_to(root)): _digest(path) for path in sorted(root.rglob("*.py"))}
    return {
        "root": str(root),
        "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "files": files,
    }


def _safe(value):
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    if isinstance(value, complex):
        return {"real": _safe(value.real), "imag": _safe(value.imag)}
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def _save(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(output)


def _finalize_sources(report):
    """Retain initial provenance and distinguish workflow edits from numerical source changes."""
    final = {name: _source(Path(source["root"])) for name, source in report["sources"].items()}
    changes = {}
    for name, source in report["sources"].items():
        before, after = source["files"], final[name]["files"]
        changed = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
        workflow_only = (
            name == "queer" and bool(changed) and all(path.startswith("workflows/") for path in changed)
        )
        changes[name] = {
            "files": changed,
            "classification": "workflow_only"
            if workflow_only
            else "numerical_or_other"
            if changed
            else "unchanged",
        }
    report["sources_final"] = final
    report["source_changes_during_run"] = changes
    report["driver_sha256_final"] = _digest(Path(__file__))
    report["driver_changed_during_run"] = report["driver_sha256_final"] != report["driver_sha256"]
    numerical = {}
    for name, source in report["sources"].items():
        fingerprints = {}
        for label, snapshot in (("initial", source), ("final", final[name])):
            files = {
                path: digest
                for path, digest in snapshot["files"].items()
                if name != "queer" or not path.startswith("workflows/")
            }
            fingerprints[label] = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        fingerprints["unchanged"] = fingerprints["initial"] == fingerprints["final"]
        numerical[name] = fingerprints
    report["numerical_source_fingerprints"] = numerical
    report["source_provenance_passed"] = all(
        item["classification"] != "numerical_or_other" for item in changes.values()
    )
    return report["source_provenance_passed"]


def _native_position_validation_delta(before, after):
    """Certify an exact change inside already guarded full-position branches only."""

    def masked(path):
        tree = ast.parse(Path(path).read_text())
        native = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NativePlaneModel"
        )
        geometry = next(
            node
            for node in native.body
            if isinstance(node, ast.FunctionDef) and node.name == "_wannier_geometry"
        )
        conditions = {
            ast.dump(ast.parse("model.position_matrices is None", mode="eval").body): "orelse",
            ast.dump(ast.parse("model.position_matrices is not None", mode="eval").body): "body",
        }
        branches = []
        for node in ast.walk(geometry):
            if isinstance(node, ast.If) and (attribute := conditions.get(ast.dump(node.test))):
                content = getattr(node, attribute)
                branches.append(
                    {
                        "guard": ast.dump(node.test),
                        "body": ast.dump(ast.Module(body=content, type_ignores=[])),
                    }
                )
                setattr(node, attribute, [ast.Pass()])
        return ast.dump(tree), branches

    old_ast, old_branches = masked(before)
    new_ast, new_branches = masked(after)
    outside_equal = old_ast == new_ast
    guarded_change = bool(old_branches and old_branches != new_branches)
    return {
        "before_path": str(Path(before).resolve()),
        "after_path": str(Path(after).resolve()),
        "before_sha256": _digest(before),
        "after_sha256": _digest(after),
        "ast_identical_outside_full_position_branches": outside_equal,
        "full_position_branch_changed": guarded_change,
        "guarded_branch_count_before": len(old_branches),
        "guarded_branch_count_after": len(new_branches),
        "passed": outside_equal and guarded_change,
    }


def _scoped_position_validation_acceptance(report, static_proof, numeric_proof):
    """Explicit appendix gate; ordinary driver runs always retain strict source-drift rejection."""
    module = "finite_field/native.py"
    changes = report["source_changes_during_run"]
    changed_numerical_files = [
        path for path in changes["queer"]["files"] if not path.startswith("workflows/")
    ]
    hashes_match = (
        static_proof["before_sha256"] == report["sources"]["queer"]["files"][module]
        and static_proof["after_sha256"] == report["sources_final"]["queer"]["files"][module]
        and numeric_proof["before_native_sha256"] == static_proof["before_sha256"]
        and numeric_proof["after_native_sha256"] == static_proof["after_sha256"]
    )
    checks = {
        "exact_native_hashes_match": hashes_match,
        "only_native_numerical_file_changed": changed_numerical_files == [module],
        "hhh_sources_unchanged": not changes["hhh"]["files"],
        "ast_scope_passed": static_proof["passed"],
        "all_report_cases_have_bitwise_centre_only_proof": all(
            case["name"].startswith("real/")
            and numeric_proof["cases"].get(case["name"], {}).get("position_matrices_is_none", False)
            and numeric_proof["cases"].get(case["name"], {}).get("input_hashes_match_live", False)
            and numeric_proof["cases"].get(case["name"], {}).get("all_products_bitwise_equal", False)
            for case in report["cases"]
        ),
    }
    return {**checks, "passed": all(checks.values())}


def _inputs(directory):
    root = Path(directory).expanduser().resolve()
    hr = next(
        (path for path in (root / "hpc/wannier90_hr.dat", root / "wannier90_hr.dat") if path.is_file()), None
    )
    if hr is None:
        raise ValueError(f"No wannier90_hr.dat in {root} or its hpc directory")
    tb = hr.parent / f"{hr.name.removesuffix('_hr.dat')}_tb.dat"
    if tb.exists():
        raise ValueError(
            f"Unsupported TB sidecar {tb}: this replay compares the TBA HR/centres/WS model; "
            "automatic TB discovery would introduce position operators and a different geometry gauge"
        )
    candidates = {
        "hr": (hr,),
        "wsvec": (hr.parent / "wannier90_wsvec.dat", root / "wannier90_wsvec.dat"),
        "centres": (root / "wannier90_centres.xyz", hr.parent / "wannier90_centres.xyz"),
        "wout": (root / "wannier90.wout", hr.parent / "wannier90.wout"),
    }
    files = {}
    for name, paths in candidates.items():
        found = next((path for path in paths if path.is_file()), None)
        if found is None:
            raise ValueError(f"Missing {name} input for {root}")
        files[name] = found
    return root, files


def _compare(np, actual, reference, tolerance, *, floor=1e-15, pointwise=False):
    a, b = np.asarray(actual), np.asarray(reference)
    result = {
        "shape_queer": list(a.shape),
        "shape_hhh": list(b.shape),
        "relative_tolerance": tolerance,
        "absolute_zero_floor": floor,
    }
    if a.shape != b.shape:
        return {**result, "passed": False, "reason": "shape mismatch"}
    finite = np.isfinite(a) & np.isfinite(b)
    nonfinite_match = True
    for component in ("real", "imag"):
        ac, bc = getattr(a, component), getattr(b, component)
        nonfinite_match &= bool(
            np.all(np.isfinite(ac) == np.isfinite(bc))
            and np.all(np.isnan(ac) == np.isnan(bc))
            and np.all(np.isposinf(ac) == np.isposinf(bc))
            and np.all(np.isneginf(ac) == np.isneginf(bc))
        )
    peak = float(np.max(abs(b[finite]), initial=0))
    error = float(np.max(abs(a[finite] - b[finite]), initial=0))
    pointwise_result = {}
    pointwise_passed = True
    if pointwise:
        limits = tolerance * abs(b[finite]) + floor
        relative_errors = abs(a[finite] - b[finite]) / limits
        pointwise_passed = bool(np.all(relative_errors <= 1) and nonfinite_match)
        pointwise_result = {
            "pointwise_required": True,
            "pointwise_passed": pointwise_passed,
            "max_error_over_pointwise_limit": float(np.max(relative_errors, initial=0)),
        }
    significance = tolerance * peak + floor
    signs = {}
    for component in ("real", "imag") if np.iscomplexobj(a) or np.iscomplexobj(b) else ("real",):
        ac, bc = getattr(a, component), getattr(b, component)
        meaningful = finite & (abs(bc) > significance)
        signs[component] = {
            "meaningful_points": int(np.count_nonzero(meaningful)),
            "mismatches": int(np.count_nonzero(np.sign(ac[meaningful]) != np.sign(bc[meaningful]))),
        }
    sign_mismatches = sum(item["mismatches"] for item in signs.values())
    return {
        **result,
        "passed": bool(
            nonfinite_match
            and error <= tolerance * peak + floor
            and sign_mismatches == 0
            and pointwise_passed
        ),
        **pointwise_result,
        "nonfinite_match": nonfinite_match,
        "reference_peak": peak,
        "max_absolute_error": error,
        "error_over_reference_peak": error / max(peak, floor),
        "meaningful_sign_points": sum(item["meaningful_points"] for item in signs.values()),
        "sign_mismatches": sign_mismatches,
        "sign_significance_threshold": significance,
        "sign_components": signs,
    }


def _product(np, case, name, actual, reference, tolerance, floor=1e-15, *, pointwise=False):
    case.setdefault("products", {})[name] = _compare(
        np, actual, reference, tolerance, floor=floor, pointwise=pointwise
    )
    case.setdefault("values", {})[name] = {"queer": _safe(actual), "hhh": _safe(reference)}


def _structural_diagnostics(diagnostics):
    crossing_flags = {
        "window_edge_min",
        "too_many_crossings",
        "touching",
        "diabatic_ec_off_stationary",
        "branch_point_ec_off_stationary",
        "diabatic_unavailable",
        "branch_point_unvalidated",
        "branch_point_fit_poor",
        "branch_point_missing",
        "branch_point_path_ambiguous",
        "ddp_invalid",
        "ddp_fallback_eq8",
        "not_separated",
        "no_crossing",
        "crossings_merged",
    }
    structural = {
        label: {
            "crossing_flags": sorted(set(data["line_flags"]) & crossing_flags),
            "layout_flags": sorted(data["layout_flags"] or []),
            "response_flags": sorted(data["response_flags"]),
        }
        for label, data in diagnostics.items()
    }
    structural["passed"] = structural["queer"] == structural["hhh"]
    return structural


def _matched_crossing_geometry(np, case, qprovider, hprovider, qs, hs, tolerance):
    """Compare geometry at identical points, separately from minimum-location differences."""
    qs, hs = np.asarray(qs), np.asarray(hs)
    positions = np.unique(np.r_[qs, hs])
    providers = {"queer": qprovider, "hhh": hprovider}
    geometry = {
        label: provider.geometry(positions, np.zeros_like(positions)) for label, provider in providers.items()
    }
    quantities = ("eps_g", "g_cv", "g_v_total", "g_c_total", "omega_v", "omega_c")
    case["matched_crossing_geometry"] = {
        "positions_inverse_angstrom": positions,
        "transverse_coordinate_inverse_angstrom": 0.0,
        "relative_tolerance": tolerance,
        "energy_references": {
            label: {
                "provider_energy_ref_eV": getattr(provider, "energy_ref", None),
                "provider_angle_radians": getattr(provider, "angle", None),
                "energies": "gap only; invariant under the provider's identity energy shift",
            }
            for label, provider in providers.items()
        },
    }
    for quantity in quantities:
        _product(
            np,
            case,
            f"matched_crossing_geometry_{quantity}",
            getattr(geometry["queer"], quantity),
            getattr(geometry["hhh"], quantity),
            tolerance,
            1e-12,
            pointwise=True,
        )
    case["matched_crossing_geometry"]["passed"] = all(
        case["products"][f"matched_crossing_geometry_{quantity}"]["passed"] for quantity in quantities
    )
    if qs.shape == hs.shape:
        qi, hi = np.searchsorted(positions, qs), np.searchsorted(positions, hs)
        case["matched_crossing_geometry"]["location_effects"] = {}
        for quantity in quantities:
            qvalues, hvalues = getattr(geometry["queer"], quantity), getattr(geometry["hhh"], quantity)
            case["matched_crossing_geometry"]["location_effects"][quantity] = {
                "queer_geometry_at_queer_minus_hhh_minima": qvalues[qi] - qvalues[hi],
                "same_coordinate_geometry_difference_at_hhh_minima": qvalues[hi] - hvalues[hi],
                "own_minima_geometry_difference": qvalues[qi] - hvalues[hi],
            }


def _crossings(env, case, qr, hr, tolerance, geometry_tolerance=None):
    np = env["np"]
    qcfg, hcfg = qr.options, hr.cfg
    qline = env["qcross"](
        qr.provider,
        0.0,
        qcfg.crossings,
        want_ddp=qcfg.conventions.ec_method == "ddp",
        e_cut=qcfg.quadrature.p_cut * qcfg.drive.E_max,
    )
    hline = env["hcross"](
        hr.provider,
        0.0,
        hcfg.crossings,
        want_ddp=hcfg.conventions.ec_method == "ddp",
        e_cut=hcfg.quadrature.p_cut * hcfg.drive.E_max,
    )
    case["central_crossings"] = {
        label: {
            "definition": line.definition,
            "flags": sorted(line.flags),
            "kind": [item.kind for item in line.crossings],
            "touching": [bool(item.touching) for item in line.crossings],
            "ddp_source": [item.ddp_source for item in line.crossings],
        }
        for label, line in (("queer", qline), ("hhh", hline))
    }
    case["central_crossings"]["passed"] = (
        case["central_crossings"]["queer"] == case["central_crossings"]["hhh"]
    )
    if geometry_tolerance is not None:
        _matched_crossing_geometry(np, case, qr.provider, hr.provider, qline.s, hline.s, geometry_tolerance)
    _product(np, case, "crossing_positions", qline.s, hline.s, tolerance, 1e-12)
    _product(
        np,
        case,
        "crossing_gaps_eV",
        [item.eps_g for item in qline.crossings],
        [item.eps_g for item in hline.crossings],
        tolerance,
        1e-14,
    )
    for metric in ("cv", "v_total", "c_total"):
        _product(
            np,
            case,
            f"crossing_metric_{metric}",
            [item.g_ss[metric] for item in qline.crossings],
            [item.g_ss[metric] for item in hline.crossings],
            tolerance,
            1e-12,
        )
    for name, method in (("crossing_Ec_eq8", "eq8"), ("crossing_Ec_ddp", "ddp")):
        _product(
            np,
            case,
            name,
            qline.Ec(qcfg.conventions.metric, method),
            hline.Ec(hcfg.conventions.metric, method),
            tolerance,
            1e-10,
        )
    from scipy.integrate import quad

    def frequencies(provider, line):
        return [
            quad(
                lambda s: float(provider.gap(s, 0.0)),
                start,
                end,
                epsabs=1e-14,
                epsrel=1e-9,
                points=[0.0] if start < 0 < end else None,
            )[0]
            * 1e10
            / (2 * np.pi)
            for start, end in zip(line.s[:-1], line.s[1:])
        ]

    _product(
        np,
        case,
        "central_Lambda_V_per_m",
        frequencies(qr.provider, qline),
        frequencies(hr.provider, hline),
        tolerance,
        1e-10,
    )


def _response(
    env, case, qm, hm, qcfg, hcfg, args, output, report, tolerance, amplitudes, *, geometry_tolerance=None
):
    np = env["np"]
    execution = env["ExecutionOptions"](workers=args.workers, memory_mb=args.memory_mb, native_threads=1)
    qr = env["PreparedHallResponse"](qm, qcfg, execution=execution, acceleration=False)
    hr = env["HallCalculator"](hm, hcfg, use_numba=False)
    case["requested_config"] = {"queer": qcfg.to_dict(), "hhh": hcfg.to_dict()}
    case["execution"] = execution.resolved().as_dict()
    _crossings(env, case, qr, hr, tolerance, geometry_tolerance)
    _save(report, output)
    for label, operation in (
        ("prepare_queer", qr.prepare),
        ("prepare_hhh", lambda: hr.prepare(n_workers=execution.resolved().workers)),
    ):
        print(f"{case['name']}: {label}", flush=True)
        case["stage"] = label
        _save(report, output)
        start = time.perf_counter()
        operation()
        case.setdefault("timing_seconds", {})[label] = time.perf_counter() - start
        _save(report, output)
    case["effective_config"] = {"queer": qr.options.to_dict(), "hhh": hr.cfg.to_dict()}
    amplitudes = np.asarray(amplitudes)
    fields = np.r_[-amplitudes[::-1], 0.0, amplitudes]
    case["fields_V_per_m"], case["amplitudes_V_per_m"] = fields, amplitudes
    start = time.perf_counter()
    _product(np, case, "current_A_per_m", qr.J(fields), hr.J(fields), tolerance)
    case["timing_seconds"]["signed_current"] = time.perf_counter() - start
    _save(report, output)
    case["stage"] = "harmonics"
    print(f"{case['name']}: harmonics N<=31", flush=True)
    start = time.perf_counter()
    qfourier, hfourier = qr.options.fourier, hr.cfg.fourier
    qa = env["qharm"](
        qr.J, amplitudes, n_max=31, n_theta=qfourier.n_theta, adaptive=True, tol=qfourier.tol, n_cap=16384
    )
    ha = env["hharm"](
        hr.J, amplitudes, n_max=31, n_theta=hfourier.n_theta, adaptive=True, tol=hfourier.tol, n_cap=16384
    )
    case["timing_seconds"]["harmonics"] = time.perf_counter() - start
    case["fourier"] = {
        "queer": {"converged": qa.converged, "n_theta": qa.n_theta, "info": qa.info},
        "hhh": {"converged": ha.converged, "n_theta": ha.n_theta, "info": ha.info},
    }
    for projection in ("a", "b", "c"):
        for order in range(32):
            _product(
                np,
                case,
                f"{projection}_{order}_A_per_m",
                getattr(qa, projection)[:, order],
                getattr(ha, projection)[:, order],
                tolerance,
            )
    _product(np, case, "c0_A_per_m", qa.c0, ha.c0, tolerance)
    case["diagnostics"] = {"queer": qr.describe(), "hhh": hr.diagnostics()}
    structural = _structural_diagnostics(case["diagnostics"])
    case["structural_diagnostics"] = structural
    case["passed"] = bool(
        qa.converged
        and ha.converged
        and case["central_crossings"]["passed"]
        and structural["passed"]
        and all(item["passed"] for item in case["products"].values())
    )
    case["stage"] = "complete"
    _save(report, output)


def _real(env, case, root, files, args, output, report):
    np, w90 = env["np"], env["w90"]
    case["stage"] = "independent_loading"
    case["inputs"] = {
        name: {"path": str(path), "sha256": _digest(path), "bytes": path.stat().st_size}
        for name, path in files.items()
    }
    _save(report, output)
    print(f"{case['name']}: independently load HR and WS images", flush=True)
    start = time.perf_counter()
    raw = w90.read_hr(files["hr"])
    ws = w90.read_wsvec(files["wsvec"])
    translations, hopping, _ = w90.expand_ws(raw.R, raw.H, raw.ndegen, ws)
    lattice = w90.read_wout_lattice(files["wout"])
    centres, _ = w90.read_centres_xyz(files["centres"])
    mem_bytes = min(256e6, args.memory_mb * 1024**2 * 0.8 / args.workers)
    hm = env["HWannier"](
        hopping,
        translations @ lattice,
        centres,
        n_occ=args.n_occ,
        gauge="tba",
        lattice=lattice,
        kz=0.0,
        rz_tol=math.inf,
        mem_bytes=mem_bytes,
    )
    del raw, ws, hopping, translations
    source = env["QWannier"].from_directory(
        files["hr"].parent,
        hr=files["hr"],
        centres=files["centres"],
        wout=files["wout"],
        wsvec=files["wsvec"],
        interpolation="ws_distance",
    )
    qm = env["NativePlaneModel"](source, band_pair=(args.n_occ - 1, args.n_occ))
    case["timing_seconds"] = {"independent_loading": time.perf_counter() - start}
    case["model"] = {"queer": qm.describe(), "hhh": hm.describe()}
    coords = np.array(
        [[0, 0], [0.003, 0.001], [-0.003, -0.001], [0.003, -0.001], [-0.003, 0.001], [0.010, 0], [-0.010, 0]]
    )
    case["geometry_points_inverse_angstrom"] = coords
    qg, hg = qm.geometry(coords[:, 0], coords[:, 1]), hm.geometry(coords[:, 0], coords[:, 1])
    for quantity in ("eps_v", "eps_c", "eps_g", "g_cv", "g_v_total", "g_c_total", "omega_v", "omega_c"):
        _product(
            np,
            case,
            f"geometry_{quantity}",
            getattr(qg, quantity),
            getattr(hg, quantity),
            1e-8,
            1e-12,
            pointwise=True,
        )
    overrides = {
        "drive.E_min": 1.0,
        "drive.E_max": 6e5,
        "drive.angle": 0.0,
        "crossings.window": 0.06,
        "crossings.definition": "gap_minima",
        "relaxation.T1": 2e-12,
        "relaxation.T2": math.inf,
        "quadrature.du": 0.15,
        "quadrature.lam_per_decade": 8,
        "fourier.n_theta": 768,
    }
    qcfg = env["qconfig"].PAPER_FIGURES.with_(**overrides)
    hcfg = env["hconfig"].PAPER_FIGURES.with_(**overrides)
    with tempfile.TemporaryDirectory(prefix="queer-hhh-parity-") as directory:
        normalized = Path(directory) / "model.npz"
        hm.save(normalized)
        case["normalized_hhh_model"] = {
            "sha256": _digest(normalized),
            "bytes": normalized.stat().st_size,
        }
        _response(
            env,
            case,
            qm,
            hm,
            qcfg,
            hcfg,
            args,
            output,
            report,
            1e-4,
            [3e3, 1e4, 3e4, 5e4, 3.5e5, 6e5],
            geometry_tolerance=1e-8,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hhh-root", type=Path, required=True)
    parser.add_argument("--real-data", type=Path, action="append", default=[])
    parser.add_argument("--n-occ", type=int, default=145)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--memory-mb", type=float, default=8192)
    parser.add_argument("--output", type=Path, default=Path("results/benchmarks/hhh-parity.json"))
    parser.add_argument("--skip-analytical", action="store_true")
    args = parser.parse_args(argv)
    args.hhh_root = args.hhh_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not (args.hhh_root / "src/hhh/config.py").is_file():
        parser.error("--hhh-root must identify a local checkout containing src/hhh/config.py")
    if args.workers < 1 or args.n_occ < 1 or not math.isfinite(args.memory_mb) or args.memory_mb <= 0:
        parser.error("workers, n-occ, and memory-mb must be positive")
    if args.skip_analytical and not args.real_data:
        parser.error("--skip-analytical requires at least one --real-data directory")
    try:
        inputs = [_inputs(directory) for directory in args.real_data]
    except ValueError as error:
        parser.error(str(error))
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(args.hhh_root / "src")]
    # Neither package is imported before CLI/input validation. HHH is never a runtime dependency.
    env = {
        "np": importlib.import_module("numpy"),
        "qconfig": importlib.import_module("queer.finite_field.config"),
        "hconfig": importlib.import_module("hhh.config"),
        "w90": importlib.import_module("hhh.io.wannier90"),
    }
    for alias, module, name in (
        ("ExecutionOptions", "queer.execution", "ExecutionOptions"),
        ("PreparedHallResponse", "queer.finite_field", "PreparedHallResponse"),
        ("NativePlaneModel", "queer.finite_field.native", "NativePlaneModel"),
        ("QModel", "queer.finite_field.models", "FourBandModel"),
        ("QParams", "queer.finite_field.models", "FourBandParams"),
        ("HModel", "hhh.models", "FourBandModel"),
        ("HParams", "hhh.models", "FourBandParams"),
        ("QWannier", "queer.models.wannier", "WannierModel"),
        ("HWannier", "hhh.models.wannier", "WannierModel"),
        ("HallCalculator", "hhh.transport.hall", "HallCalculator"),
        ("qharm", "queer.finite_field.harmonics", "harmonics"),
        ("hharm", "hhh.harmonics", "harmonics"),
        ("qcross", "queer.finite_field.lz.crossings", "find_crossings"),
        ("hcross", "hhh.lz.crossings", "find_crossings"),
    ):
        env[alias] = getattr(importlib.import_module(module), name)
    report = {
        "schema_version": 1,
        "driver_sha256": _digest(Path(__file__)),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "runtime": {
            "python": platform.python_version(),
            "numpy": env["np"].__version__,
            "scipy": importlib.import_module("scipy").__version__,
            "platform": platform.platform(),
        },
        "sources": {
            "queer": _source(Path(__file__).resolve().parents[1] / "src/queer"),
            "hhh": _source(args.hhh_root / "src/hhh"),
        },
        "arguments": vars(args),
        "cases": [],
    }
    _save(report, args.output)

    def run_case(name, operation):
        case = {"name": name, "stage": "starting", "passed": False}
        report["cases"].append(case)
        start = time.perf_counter()
        _save(report, args.output)
        try:
            operation(case)
        except Exception as error:
            case.update(
                {"stage": "failed", "error": str(error), "traceback": traceback.format_exc(), "passed": False}
            )
            print(f"{name}: FAILED: {error}", flush=True)
        case["elapsed_seconds"] = time.perf_counter() - start
        _save(report, args.output)
        print(f"{name}: {'PASS' if case['passed'] else 'FAIL'} ({case['elapsed_seconds']:.1f}s)", flush=True)

    try:
        if not args.skip_analytical:
            cases = [("PAPER", x, 1e-7) for x in (0.0, 0.5, 1.0, 1.4, 1.8)]
            cases += [("PHYSICAL", x, 1e-5) for x in (0.5, 1.0, 1.4, 1.5, 1.8)]
            cases += [
                (profile, x, 1e-7)
                for profile, xs in (
                    ("fig5c", (1.3, 1.8)),
                    ("ext7a", (1.3, 1.8)),
                    ("fig4b", (0.9,)),
                    ("fig4d", (1.6e-3 / env["HParams"]().vlc,)),
                    ("ext9b", (1.0,)),
                    ("ext7e_left", (1.4,)),
                    ("ext7e_right", (1.4,)),
                )
                for x in xs
            ]
            for profile, x, tolerance in cases:
                qcfg = getattr(env["qconfig"], profile, None) or env["qconfig"].preset(profile)
                hcfg = getattr(env["hconfig"], profile, None) or env["hconfig"].preset(profile)
                run_case(
                    f"{profile}/x={x:g}",
                    lambda case, x=x, qcfg=qcfg, hcfg=hcfg, tolerance=tolerance: _response(
                        env,
                        case,
                        env["QModel"](env["QParams"]().at(x)),
                        env["HModel"](env["HParams"]().at(x)),
                        qcfg,
                        hcfg,
                        args,
                        args.output,
                        report,
                        tolerance,
                        [1e4, 3e4, 5e4],
                    ),
                )
        for root, files in inputs:
            run_case(
                f"real/{root.name}",
                lambda case, root=root, files=files: _real(env, case, root, files, args, args.output, report),
            )
    except BaseException:
        report["status"] = "interrupted"
        _finalize_sources(report)
        _save(report, args.output)
        raise
    provenance_passed = _finalize_sources(report)
    report["status"] = (
        "passed" if provenance_passed and all(case["passed"] for case in report["cases"]) else "failed"
    )
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _save(report, args.output)
    print(f"Report: {args.output}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
