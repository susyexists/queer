#!/usr/bin/env python
"""Fetch pinned OpenMX references and independently validate native calculations."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from threadpoolctl import threadpool_limits

from queer import ExecutionOptions, OpenMXModel
from queer.io.artifacts import ALGORITHM_VERSION, atomic_json, atomic_npz, canonical_json, digest_file

COMMIT = "60fc6b2acafac36b9c8334208326937fef3366c5"
REFERENCES = {
    "FeCl2": ("5172521246fe7c5b6b99c662293f1238a2c76ae4f3dae2b57a2997ce471ab314", 12699636),
    "GaAs_soc": ("d7d9027f0cd9c5547ea13893166ccaba36f2633bfd2ecd4e1259dd7c3daa9b3b", 18862660),
}


def reference_file(directory, name, fetch):
    expected, size = REFERENCES[name]
    path = directory / f"{name}.scfout"
    url = f"https://raw.githubusercontent.com/susyexists/pyatb/{COMMIT}/tests/data/{path.name}"
    if not path.is_file():
        if not fetch:
            raise FileNotFoundError(f"{path}: run with --fetch to download the pinned reference")
        directory.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=directory)
        try:
            with os.fdopen(descriptor, "wb") as stream, urlopen(url, timeout=60) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
            candidate = Path(temporary)
            if candidate.stat().st_size != size or digest_file(candidate) != expected:
                raise ValueError(f"Downloaded {name} failed pinned size/SHA256 verification")
            os.replace(candidate, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    if path.stat().st_size != size or digest_file(path) != expected:
        raise ValueError(f"{path}: pinned size/SHA256 mismatch")
    return path, {"file": path.name, "url": url, "sha256": expected, "size": size}


def independent_bands(model):
    points = np.array([[0, 0, 0], [0.17, 0.23, 0.11], [0.5, 0.25, 0.125]])
    actual = model.eigenstates(points)
    energy_error = orthonormality = residual = 0.0
    for index, k in enumerate(points):
        h, s, c = model.hamiltonian(k), model.overlap(k), actual.coefficients[index]
        factor = cholesky(s, lower=True)
        left = solve_triangular(factor, h, lower=True)
        reduced = solve_triangular(factor, left.conj().T, lower=True).conj().T
        expected = np.linalg.eigvalsh(reduced)
        energy_error = max(energy_error, float(np.max(np.abs(actual.energies[:, index] - expected))))
        orthonormality = max(
            orthonormality, float(np.linalg.norm(c.conj().T @ s @ c - np.eye(model.nbnd), np.inf))
        )
        diff = h @ c - (s @ c) * actual.energies[:, index]
        scale = (
            np.linalg.norm(h, np.inf) + np.linalg.norm(s, np.inf) * np.max(np.abs(expected))
        ) * np.linalg.norm(c, np.inf)
        residual = max(residual, float(np.linalg.norm(diff, np.inf) / max(scale, 1)))
    if energy_error > 1e-9 or orthonormality > 1e-10 or residual > 1e-10:
        raise AssertionError("Independent generalized diagonalization acceptance failed")
    return {
        "kpoints": points.tolist(),
        "energy_error_eV": energy_error,
        "orthonormality_error": orthonormality,
        "normalized_residual": residual,
        "reference": "Cholesky reduction and numpy.linalg.eigvalsh",
    }


def converged_response(model, grid, levels, tolerance, cache_dir, temperature=1000.0, eta=1.0):
    import hashlib

    from queer.observables._tb import kmesh
    from queer.observables.optical import optical_conductivity

    previous = None
    history = []
    active_axes = np.any(model.translations != 0, axis=0)
    if not active_axes.any():
        active_axes[:] = True
    for level in range(levels):
        side = grid * 2**level
        shape = np.where(active_axes, side, 1)
        points = kmesh(shape) + np.where(active_axes, 0.5 / shape, 0)
        settings = {
            "input_sha256": digest_file(model.data.source),
            "grid": shape.tolist(),
            "mesh": "midpoint",
            "omega_eV": [1.0, 2.0, 4.0],
            "eta_eV": eta,
            "temperature_kelvin": temperature,
            "algorithm": ALGORITHM_VERSION,
        }
        key = hashlib.sha256(canonical_json(settings).encode()).hexdigest()
        cache, record_path = cache_dir / f"{key}.npz", cache_dir / f"{key}.json"
        if record_path.is_file():
            record = json.loads(record_path.read_text())
            if not cache.is_file() or digest_file(cache) != record["sha256"]:
                raise ValueError(f"Benchmark partial checksum failed: {cache}")
            with np.load(cache, allow_pickle=False) as arrays:
                tensor = arrays["conductivity"]
        else:
            result = optical_conductivity(
                model,
                settings["omega_eV"],
                kpoints=points,
                eta=eta,
                temperature_kelvin=temperature,
                execution=ExecutionOptions(batch_size=4),
            )
            tensor = result["conductivity"]
            atomic_npz(cache, result.arrays)
            atomic_json(
                record_path,
                {
                    "settings": settings,
                    "sha256": digest_file(cache),
                    "observable": result.artifact_metadata(),
                },
            )
        absorption = (tensor + tensor.conj().swapaxes(-1, -2)) / 2
        minimum = float(np.linalg.eigvalsh(absorption).min())
        if minimum < -1e-7 * max(float(np.max(np.abs(tensor))), 1):
            raise AssertionError("Material optical absorption is not positive semidefinite")
        record = {**settings, "minimum_absorption_eigenvalue_S_per_cm": minimum}
        if previous is not None:
            error = float(np.max(np.abs(tensor - previous)) / max(float(np.max(np.abs(tensor))), 1))
            record["relative_mesh_error"] = error
            history.append(record)
            if error <= tolerance:
                return {"converged": True, "relative_tolerance": tolerance, "history": history}
        else:
            history.append(record)
        previous = tensor
    raise RuntimeError(f"Optical response did not converge: {json.dumps(history)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data/references/openmx"))
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--responses", action="store_true", help="also require optical mesh convergence")
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--temperature-kelvin", type=float, default=1000.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("results/openmx-benchmark.json"))
    args = parser.parse_args(argv)
    if (
        args.grid < 1
        or args.levels < 2
        or not 0 < args.tolerance < 1
        or not np.isfinite(args.eta)
        or args.eta <= 0
        or not np.isfinite(args.temperature_kelvin)
        or args.temperature_kelvin < 0
    ):
        parser.error("Invalid grid, levels, tolerance, temperature or broadening")
    provenance, report = [], {"upstream_commit": COMMIT, "materials": {}}
    paths = {}
    for name in REFERENCES:
        paths[name], source = reference_file(args.directory, name, args.fetch)
        provenance.append(source)
    atomic_json(args.directory / "provenance.json", {"commit": COMMIT, "files": provenance})
    cache_dir = args.output.parent / "openmx-benchmark-partials"
    with threadpool_limits(limits=1):
        for name in REFERENCES:
            model = OpenMXModel.from_scfout(paths[name])
            record = {
                "bands": independent_bands(model),
                "model": model.metadata,
                "spin_mode": model.spin_mode,
                "band_count": model.nbnd,
                "electron_count": model.data.electron_count,
            }
            if args.responses:
                record["optical"] = converged_response(
                    model,
                    args.grid,
                    args.levels,
                    args.tolerance,
                    cache_dir,
                    args.temperature_kelvin,
                    args.eta,
                )
            report["materials"][name] = record
            atomic_json(args.output, report)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
