"""Offline QUEER replay of original PyATB AHC and shift-current artifacts.

Only native QUEER calculators are executed. Original producer source files may
be fingerprinted for provenance, but are never imported or executed. The SCFOUT
checksum must be supplied from separately verified historical input evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from queer import ExecutionOptions, OpenMXModel
from queer.compat.generator import translate_pyatb
from queer.config import config_from_mapping
from queer.io.artifacts import atomic_json, atomic_npz, digest_file, environment_versions, load_manifest
from queer.io.wannier import BOHR_ANGSTROM
from queer.workflows import run_workflow

AHC_COMPONENTS = ("yz", "zx", "xy")
SHIFT_COMPONENTS = (
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
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def read_ahc(path):
    """Read the original three-component prose artifact, retaining its units."""
    match = re.fullmatch(
        rf"\s*AHC is\s*\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\]\s*S/cm\s*",
        Path(path).read_text(),
    )
    if match is None:
        raise ValueError("Expected original PyATB 'AHC is [yz,zx,xy] S/cm' artifact")
    values = np.asarray([float(item) for item in match.groups()])
    if not np.isfinite(values).all():
        raise ValueError("Historical AHC values must be finite")
    return values


def read_shift_current(path):
    """Validate original 18-component symmetric-input tensor columns."""
    text = Path(path).read_text()
    header = next((line for line in text.splitlines() if line.strip()), "").split()
    expected = ["#", "omega(eV)", *SHIFT_COMPONENTS, "(uA/V^2)"]
    if header != expected:
        raise ValueError("Historical shift-current header has unknown tensor order or units")
    values = np.loadtxt(path, ndmin=2)
    if values.shape[1] != 19 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Historical shift-current table must have 19 finite columns")
    if len(values) > 1 and np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError("Historical photon energies must be strictly ascending")
    return values[:, 0], values[:, 1:]


def tensor_components(tensor, components):
    axes = {axis: index for index, axis in enumerate("xyz")}
    array = np.asarray(tensor)
    rank = len(components[0])
    if array.shape[-rank:] != (3,) * rank:
        raise ValueError("Native Cartesian tensor has the wrong shape")
    if rank == 2:
        return np.stack([array[..., axes[name[0]], axes[name[1]]] for name in components], axis=-1)
    if rank == 3:
        return np.stack(
            [array[..., axes[name[0]], axes[name[1]], axes[name[2]]] for name in components], axis=-1
        )
    raise ValueError("Response components must have two or three Cartesian indices")


def comparison(native, reference, *, atol, rtol):
    if not np.isfinite([atol, rtol]).all() or atol < 0 or rtol < 0:
        raise ValueError("Comparison tolerances must be finite and nonnegative")
    native, reference = np.asarray(native, float), np.asarray(reference, float)
    if native.shape != reference.shape or not reference.size:
        raise ValueError("Native and historical response axes must match without interpolation")
    if not np.isfinite(native).all() or not np.isfinite(reference).all():
        raise ValueError("Response comparison requires complete finite arrays")
    error = native - reference
    return {
        "shape": list(reference.shape),
        "compared_values": int(reference.size),
        "max_absolute_error": float(np.abs(error).max()),
        "rms_error": float(np.sqrt(np.mean(error**2))),
        "mean_signed_error": float(error.mean()),
        "atol": atol,
        "rtol": rtol,
        "passed": bool(np.all(np.abs(error) <= atol + rtol * np.abs(reference))),
    }


def translate_resolved_input(resolved, function, *, fermi_energy, spin_mode):
    """Record historical aliases/sentinels explicitly at the replay boundary."""
    function = function.upper()
    if function not in {"AHC", "SHIFT_CURRENT"}:
        raise ValueError("Replay supports AHC and SHIFT_CURRENT")
    global_parameters = {
        key: value for key, value in resolved["INPUT_PARAMETERS"].items() if value is not None
    }
    parameters = {key: value for key, value in resolved[function].items() if value is not None}
    changes = []
    if function == "AHC" and parameters.get("engine") == "pyatb":
        parameters["engine"] = "kubo"
        changes.append("historical engine alias pyatb maps to the direct/Kubo family; method is preserved")
    if function == "SHIFT_CURRENT" and parameters.get("n_occ") == parameters.get("m_unocc") == -1:
        parameters.pop("n_occ")
        parameters.pop("m_unocc")
        changes.append("historical n_occ=m_unocc=-1 declares unrestricted transitions")
    recipe = translate_pyatb(
        function,
        parameters,
        input_parameters=global_parameters,
        fermi_energy=fermi_energy,
        spin_mode=spin_mode,
    )
    return recipe, changes


def input_lattice(resolved):
    block = resolved["LATTICE"]
    units = block["lattice_constant_unit"]
    if units not in {"Angstrom", "Bohr"}:
        raise ValueError("Historical lattice units must be explicit Angstrom or Bohr")
    lattice = np.asarray(block["lattice_vector"], float) * float(block["lattice_constant"])
    if units == "Bohr":
        lattice *= BOHR_ANGSTROM
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all() or abs(np.linalg.det(lattice)) < 1e-12:
        raise ValueError("Historical lattice must be finite and nonsingular")
    return lattice


def replay(args):
    started = time.monotonic()
    if not np.isfinite([args.atol, args.rtol]).all() or args.atol < 0 or args.rtol < 0:
        raise ValueError("Comparison tolerances must be finite and nonnegative")
    scfout = args.scfout.resolve()
    source_hash = digest_file(scfout)
    if source_hash != args.expected_scfout_sha256.lower():
        raise ValueError("SCFOUT SHA256 differs from independently verified historical input evidence")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    historical_input = json.loads(args.input_json.read_text())
    resolved = historical_input
    if args.function == "AHC":
        reference = read_ahc(args.reference)
        energy = None
    else:
        energy, reference = read_shift_current(args.reference)
    model = OpenMXModel.from_scfout(scfout, openmx_input=args.openmx_input)
    scf_fermi_energy = model.fermi_energy
    if args.fermi_energy is not None:
        resolved = {
            **resolved,
            "INPUT_PARAMETERS": {
                **resolved["INPUT_PARAMETERS"],
                "fermi_energy": args.fermi_energy,
                "fermi_energy_unit": "eV",
            },
        }
    recipe, changes = translate_resolved_input(
        resolved,
        args.function,
        fermi_energy=scf_fermi_energy,
        spin_mode=model.spin_mode,
    )
    lattice = input_lattice(resolved)
    lattice_error = float(np.abs(lattice - model.lattice).max())
    # Legacy constants used six decimals for Bohr; tolerate that declared input
    # precision, while rejecting a different physical cell.
    if not np.allclose(lattice, model.lattice, atol=2e-6, rtol=1e-7):
        raise ValueError("Original Input lattice and native SCFOUT lattice disagree")
    parameters = recipe["parameters"]
    if args.function == "AHC":
        if len(parameters.get("chemical_potentials", [])) != 1:
            raise ValueError(
                "Original prose AHC replay requires one chemical potential; scan tables need a separate recipe"
            )
        integration = dict(parameters["integration"])
        integration.update(max_points=args.max_points, max_cells=args.max_cells)
        if args.absolute_tolerance is not None:
            integration["absolute_tolerance"] = args.absolute_tolerance
        if args.relative_tolerance is not None:
            integration["relative_tolerance"] = args.relative_tolerance
        parameters["integration"] = integration
    else:
        assert energy is not None
        if energy.shape != (len(parameters["omega"]),) or not np.allclose(
            parameters["omega"], energy, atol=5.01e-6, rtol=0
        ):
            raise ValueError(
                "Native photon energies differ from historical rows; interpolation is prohibited"
            )
    options = ExecutionOptions(
        workers=args.workers,
        native_threads=args.native_threads,
        batch_size=args.batch_size,
        memory_mb=args.memory_mb,
    ).resolved()
    inputs = {
        "material_dir": str(scfout.parent),
        "model": "openmx",
        "scfout": scfout.name,
        "export_metadata": recipe["export_metadata"],
    }
    if args.openmx_input is not None:
        inputs["openmx_input"] = str(args.openmx_input.resolve())
    config = config_from_mapping(
        {
            "workflow": recipe["workflow"],
            "inputs": inputs,
            "parameters": parameters,
            "execution": options.as_dict(),
        }
    )
    report = {
        "status": "prepared",
        "function": args.function,
        "historical_input_binding": args.binding,
        "inputs": {
            "scfout": {"path": str(scfout), "sha256": source_hash, "supplied_expected_hash_verified": True},
            "input_json": {"path": str(args.input_json.resolve()), "sha256": digest_file(args.input_json)},
            "reference": {"path": str(args.reference.resolve()), "sha256": digest_file(args.reference)},
            "producer_sources": [
                {"path": str(p.resolve()), "sha256": digest_file(p)} for p in args.source_file
            ],
        },
        "native_model": {
            "bands": model.nbnd,
            "spin_mode": model.spin_mode,
            "spin_degeneracy": model.spin_degeneracy,
            "scf_fermi_energy_eV": scf_fermi_energy,
            "applied_fermi_energy_eV": parameters.get("ef"),
            "explicit_fermi_override_eV": args.fermi_energy,
            "lattice_A": model.lattice.tolist(),
            "input_lattice_max_difference_A": lattice_error,
            "source_operator_identity_residuals": model.metadata["source_operator_identity_residuals"],
        },
        "historical_parameters": historical_input,
        "applied_input_parameters": resolved["INPUT_PARAMETERS"],
        "native_config": config.as_dict(),
        "boundary_translations": changes,
        "environment": environment_versions(),
        "limitations": [
            "native adaptive cubature has different nodes and error estimators from historical PyATB",
            "embedded integration estimates do not prove continuum convergence",
            "printed historical numbers retain only the artifact's decimal precision",
        ]
        if args.function == "AHC"
        else [
            "printed historical numbers retain only the artifact's decimal precision",
            *recipe["export_metadata"].get("scientific_corrections", []),
        ],
    }
    if args.binding == "inferred":
        report["limitations"].append("historical artifact-to-SCFOUT binding remains inferred")
    if args.openmx_input is not None:
        report["inputs"]["openmx_input"] = {
            "path": str(args.openmx_input.resolve()),
            "sha256": digest_file(args.openmx_input),
        }
    atomic_json(output / "comparison.json", report)
    try:
        run = run_workflow(config, run_dir=output / "native-run", resume=args.resume)
    except BaseException as error:
        report.update(status="incomplete", error={"type": type(error).__name__, "message": str(error)})
        report["wall_seconds"] = time.monotonic() - started
        atomic_json(output / "comparison.json", report)
        raise
    manifest = load_manifest(run)
    components: tuple[str, ...]
    with np.load(run / "arrays/result.npz", allow_pickle=False) as products:
        if args.function == "AHC":
            native = tensor_components(products["ahc"], AHC_COMPONENTS)
            if native.shape != (1, 3):
                raise ValueError("Original prose AHC requires one chemical potential")
            native = native[0] * recipe["export_metadata"]["hall_export_sign"]
            axes, components, unit = {}, AHC_COMPONENTS, "S/cm"
        else:
            assert energy is not None
            omega = products["omega"]
            if omega.shape != energy.shape or not np.allclose(omega, energy, atol=5.01e-6, rtol=0):
                raise ValueError(
                    "Native photon energies differ from historical rows; interpolation is prohibited"
                )
            native = tensor_components(products["shift_current"], SHIFT_COMPONENTS)
            axes, components, unit = {"omega_eV": omega}, SHIFT_COMPONENTS, "microA/V^2"
    atomic_npz(
        output / "comparison.npz",
        {**axes, "native": native, "reference": reference, "difference": native - reference},
    )
    result = comparison(native, reference, atol=args.atol, rtol=args.rtol)
    report.update(
        status="completed",
        native_run=str(run),
        integration=manifest.get("integration"),
        components=list(components),
        unit=unit,
        comparison=result,
        wall_seconds=time.monotonic() - started,
    )
    if args.function == "AHC":
        report.update(native_values=native.tolist(), historical_values=reference.tolist())
    atomic_json(output / "comparison.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function", choices=("AHC", "SHIFT_CURRENT"), required=True)
    parser.add_argument("--scfout", type=Path, required=True)
    parser.add_argument("--expected-scfout-sha256", required=True)
    parser.add_argument("--binding", choices=("verified", "inferred"), default="inferred")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--openmx-input", type=Path)
    parser.add_argument("--source-file", type=Path, action="append", default=[])
    parser.add_argument(
        "--fermi-energy", type=float, help="Explicit historical resolved absolute chemical potential, eV"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--native-threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--memory-mb", type=float, default=512)
    parser.add_argument("--max-points", type=int, default=4096)
    parser.add_argument("--max-cells", type=int, default=1024)
    parser.add_argument("--absolute-tolerance", type=float)
    parser.add_argument("--relative-tolerance", type=float)
    parser.add_argument(
        "--atol", type=float, default=5e-7, help="Response comparison tolerance, in reported response units"
    )
    parser.add_argument("--rtol", type=float, default=1e-6)
    args = parser.parse_args(argv)
    report = replay(args)
    print(
        json.dumps(
            {"output_dir": str(args.output_dir.resolve()), "comparison": report["comparison"]}, indent=2
        )
    )
    return 0 if report["comparison"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
