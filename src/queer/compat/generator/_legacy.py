"""Generator parameter/payload adapters without importing the application.

Native results retain scientific units and absolute energies. Legacy sign,
time, plotting-axis and index conventions are converted only at this boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...io.artifacts import atomic_json, digest_file
from ...physics.constants import ASE_FS as ASE_FS
from ...physics.constants import BOHR_ANGSTROM, HARTREE_EV


def translate_pyatb(step_type, parameters=None, *, input_parameters=None, fermi_energy=None, spin_mode=None):
    """Translate resolved legacy keywords to native workflow arguments.

    Return workflow/parameters/execution/export_metadata. ``fermi_energy`` supplies
    the authoritative SCF value for Auto; callers obtain it from native SCFOUT.
    """
    integration: dict[str, Any]
    step = str(step_type).upper().removeprefix("PYATB_")
    names = {"AHC": "ahc", "SHIFT_CURRENT": "shift_current", "CHERN_NUMBER": "chern_number"}
    if step not in names:
        raise ValueError("Unsupported PyATB step type")
    values = {**(input_parameters or {}), **(parameters or {})}
    global_keys = {
        "nspin",
        "fermi_energy",
        "fermi_energy_unit",
        "max_kpoint_num",
        "sparse_format",
        "hall_convention",
        "package",
        "scfout_route",
    }
    integration_keys = {
        "integrate_mode",
        "integrate_grid",
        "adaptive_grid",
        "adaptive_grid_threshold",
        "relative_error",
        "absolute_error",
        "initial_grid",
    }
    specific = {
        "AHC": {"engine", "method", "fhs_grid", "fhs_energy_window", "fhs_energy_points", "fhs_dimension"},
        "CHERN_NUMBER": {"method", "occ_band", "k_start", "k_vect1", "k_vect2"},
        "SHIFT_CURRENT": {
            "occ_band",
            "omega",
            "domega",
            "smearing_method",
            "eta",
            "grid",
            "method",
            "n_occ",
            "m_unocc",
            "scissors",
        },
    }
    unknown = (
        set(values) - global_keys - specific[step] - (integration_keys if step != "SHIFT_CURRENT" else set())
    )
    if unknown:
        raise ValueError(f"Unsupported {step} keywords: {sorted(unknown)}")
    legacy_engine = str(values.get("engine", "kubo")).strip().lower()
    if step == "AHC" and legacy_engine not in {"kubo", "pyatb", "fhs"}:
        raise ValueError("AHC engine must be kubo, pyatb, or fhs")
    energy = values.get("fermi_energy", "Auto")
    if str(energy).lower() == "auto":
        mu = fermi_energy
    else:
        unit = values.get("fermi_energy_unit", "eV")
        if unit not in {"eV", "Ry"}:
            raise ValueError("fermi_energy_unit must be eV or Ry")
        mu = float(energy) * (HARTREE_EV / 2 if unit == "Ry" else 1)
    if mu is not None and not np.isfinite(mu):
        raise ValueError("Fermi energy must be finite")
    if spin_mode is not None and "nspin" in values:
        expected = 4 if spin_mode in {"soc", "nc"} else 1
        if int(values["nspin"]) != expected:
            raise ValueError("Explicit nspin conflicts with native SCFOUT spin metadata")
    native: dict[str, Any] = {}
    execution: dict[str, int] = {}
    if mu is not None:
        native["ef"] = float(mu)
    if "max_kpoint_num" in values:
        execution["batch_size"] = int(values["max_kpoint_num"])
        if execution["batch_size"] < 1:
            raise ValueError("max_kpoint_num must be positive")
    convention = str(values.get("hall_convention", "pyatb")).lower()
    if convention not in {"auto", "pyatb", "openmx", "fhs"}:
        raise ValueError("Unsupported hall_convention")
    metadata = {
        "legacy_step_type": "PYATB_" + step,
        "hall_convention": convention,
        "hall_export_sign": 1,
        "energy_reference": "absolute",
        "legacy_parameter_version": 1,
        "sparse_format": values.get("sparse_format", 0),
    }
    if step == "AHC":
        metadata["legacy_engine"] = legacy_engine
        if legacy_engine == "pyatb":
            metadata["engine_translation"] = "pyatb family alias; method selects direct or kubo"
    method = int(values.get("method", 1 if step == "SHIFT_CURRENT" else 0))
    if method not in (0, 1):
        raise ValueError("method must be 0 or 1")
    if step == "AHC" and legacy_engine == "fhs":
        window = np.asarray(values.get("fhs_energy_window", [-0.5, 0.5]), float)
        count = int(values.get("fhs_energy_points", 101))
        if window.shape != (2,) or not np.isfinite(window).all() or window[1] < window[0] or count < 1:
            raise ValueError("Invalid FHS energy scan")
        if mu is None:
            raise ValueError("FHS Auto translation requires the SCF Fermi energy")
        native.update(
            engine="fhs",
            dimension=3,
            grid=values.get("fhs_grid", [100, 100, 1]),
            chemical_potentials=(
                mu + window[0] + np.arange(count) * (window[1] - window[0]) / count
            ).tolist(),
        )
        metadata["hall_components"] = (
            ["xy"] if int(values.get("fhs_dimension", 0)) == 0 else ["yz", "zx", "xy"]
        )
    elif step in {"AHC", "CHERN_NUMBER"}:
        native["engine"] = "direct" if method == 0 else "kubo"
        mode = values.get("integrate_mode", "Grid")
        if mode == "Grid":
            grid = list(values.get("integrate_grid", [4, 4, 4]))
            subdivision = list(values.get("adaptive_grid", [4, 4, 4]))
            integration = {
                "method": "threshold",
                "subdivision": subdivision,
                "threshold": float(values.get("adaptive_grid_threshold", 50)) * BOHR_ANGSTROM**2,
                "threshold_quantity": "curvature",
            }
            metadata["threshold_source_unit"] = "bohr^2"
        elif mode == "Adaptive":
            grid = list(values.get("initial_grid", [1, 1, 1]))
            integration = {
                "method": "cubature",
                "absolute_tolerance": float(values.get("absolute_error", 0.1)),
                "relative_tolerance": float(values.get("relative_error", 1e-6)),
            }
        else:
            raise ValueError("integrate_mode must be Grid or Adaptive")
        if len(grid) != 3:
            raise ValueError("Integration grid must have three dimensions")
        if step == "CHERN_NUMBER":
            grid[2] = 1
            if "subdivision" in integration:
                integration["subdivision"][2] = 1
            native.update(
                k_start=values.get("k_start", [0.0, 0.0, 0.0]),
                k_vectors=[values.get("k_vect1", [1.0, 0.0, 0.0]), values.get("k_vect2", [0.0, 1.0, 0.0])],
            )
            occupied = int(values.get("occ_band", -1))
            if occupied < -1:
                raise ValueError("occ_band must be -1 or a nonnegative count")
            native["occupied_bands"] = None if occupied == -1 else occupied
            if mu is not None:
                native["fermi_energy"] = float(mu)
        else:
            native["dimension"] = 3
            if mu is not None:
                native["chemical_potentials"] = [float(mu)]
            metadata["hall_export_sign"] = -1 if convention in {"openmx", "fhs"} else 1
        native.update(grid=grid, integration=integration)
    elif step == "SHIFT_CURRENT":
        if "omega" not in values or "domega" not in values or "grid" not in values:
            raise ValueError("Shift current requires omega range, domega, and grid")
        limits, delta = np.asarray(values["omega"], float), float(values["domega"])
        if (
            limits.shape != (2,)
            or not np.isfinite(limits).all()
            or not np.isfinite(delta)
            or delta <= 0
            or limits[0] < 0
            or limits[1] < limits[0]
        ):
            raise ValueError("Invalid shift-current frequency range")
        count = int((limits[1] - limits[0]) / delta) + 1
        if count > 1_000_000:
            raise ValueError("Frequency scan exceeds one million points")
        smearing = int(values.get("smearing_method", 1))
        if smearing not in (0, 1, 2):
            raise ValueError("smearing_method must be 0, 1, or 2")
        native.update(
            omega=(limits[0] + np.arange(count) * delta).tolist(),
            grid=values["grid"],
            smearing={0: "none", 1: "gaussian", 2: "adaptive"}[smearing],
            eta=float(values.get("eta", 0.01)) / np.sqrt(2),
            method="analytic",
            scissors=float(values.get("scissors", 0)),
        )
        if smearing == 0:
            native["bin_width"] = delta
        if "occ_band" in values:
            native["occupied_bands"] = int(values["occ_band"])
            native["scissors_occupied_bands"] = int(values["occ_band"])
        if ("n_occ" in values) != ("m_unocc" in values):
            raise ValueError("n_occ and m_unocc must identify a complete band pair")
        if "n_occ" in values:
            legacy_pair = [int(values["n_occ"]), int(values["m_unocc"])]
            if legacy_pair == [-1, -1]:
                metadata["band_pair_selection"] = "all occupied-unoccupied pairs"
            else:
                pair = [index - 1 for index in legacy_pair]
                if pair[0] < 0 or pair[1] <= pair[0]:
                    raise ValueError("Band pair requires 1 <= n_occ < m_unocc, or both -1")
                native["band_pairs"] = [pair]
        metadata["legacy_method"] = method
        metadata["method_translation"] = (
            "audited legacy methods use sum-over-states; native analytic derivative"
        )
        if smearing == 2:
            metadata["scientific_corrections"] = ["adaptive widths depend on the requested reciprocal mesh"]
        if native["scissors"] != 0:
            metadata.setdefault("scientific_corrections", []).append(
                "scissors shifts transition energies; geometry uses the unshifted Hamiltonian"
            )
    else:
        raise ValueError("Unsupported AHC engine")
    return {
        "workflow": names[step],
        "parameters": native,
        "execution": execution,
        "export_metadata": metadata,
    }


def translate_ase(step_type, parameters=None):
    """Convert legacy time/force names while retaining generator defaults."""
    kind = str(step_type).upper().removeprefix("ASE_")
    params = dict(parameters or {})
    # Model source is an input/producer concern rather than an integrator parameter.
    for name in ("foundation_model", "model_source", "model_id", "model_ids", "optimizer"):
        params.pop(name, None)
    if kind == "NVT":
        mapping = {
            "temperature_K": "temperature_K",
            "temperature": "temperature_K",
            "timestep_fs": "timestep_fs",
            "steps": "n_steps",
        }
        result = {mapping.get(key, key): value for key, value in params.items() if key != "friction"}
        result["friction_fs_inv"] = float(params.get("friction", 0.01)) * ASE_FS
        return {"workflow": "nvt", "parameters": result}
    if kind == "RELAX":
        return {
            "workflow": "relax",
            "parameters": {"fmax_eV_A" if key == "fmax" else key: value for key, value in params.items()},
        }
    if kind == "PHONON":
        return {
            "workflow": "phonons",
            "parameters": {"delta_A" if key == "delta" else key: value for key, value in params.items()},
        }
    if kind == "SELECT_MIN_ENERGY":
        return {"workflow": "select", "parameters": params}
    raise ValueError("Unsupported ASE step type")


def structure_payload(structure, *, source_type="upload", source_format="poscar", content=""):
    from ...elements import chemical_formula
    from ...geometry.structures import lattice_parameters
    from ...io.atomistic import poscar_text

    rendered = poscar_text(structure, sort=True)
    text, order = rendered[:2]
    result = {
        "formula": chemical_formula(structure.atomic_numbers),
        "natoms": structure.natoms,
        "lattice": structure.cell.tolist(),
        "positions": structure.positions.tolist(),
        "species": structure.symbols,
        "latticeParams": lattice_parameters(structure),
        "sourceType": source_type,
        "sourceFormat": source_format,
        "fileContent": content,
        "poscar": text,
        "poscar_atom_order": np.asarray(order).tolist(),
    }
    if structure.magnetic_moments is not None:
        moments = structure.magnetic_moments
        norms = np.linalg.norm(moments, axis=1)
        collinear = np.all(np.linalg.norm(moments[:, :2], axis=1) <= np.maximum(norms * 1e-3, 1e-6))
        result["magneticMoments"] = {
            "mode": "collinear" if collinear else "noncollinear",
            "source": source_format,
            "basis": "cartesian",
            "moments": moments.tolist(),
        }
    return result


def transform_structures(structure, transforms):
    """Apply generator recipes with their historical layer-scaling convention."""
    from ...geometry.structures import apply_transforms, expand_variants

    return [apply_transforms(structure, recipe, legacy=True) for recipe in expand_variants(transforms)]


def dataset_exports(items, **parameters):
    """Generator reference items use kbar; preserve their audited legacy sign."""
    from ...atomistic.datasets import export_dataset_splits

    return export_dataset_splits(items, **{"stress_unit": "kbar", "stress_sign": 1.0, **parameters})


def select_legacy_min_energy_frame(
    frames, samples=None, *, traj_interval, energy_key="etot_eV", fallback=None
):
    """Explicit legacy fallback: infer absent frame steps from a write interval."""
    from dataclasses import replace

    from ...atomistic.selection import select_min_energy_frame

    if isinstance(traj_interval, bool) or int(traj_interval) != traj_interval or traj_interval < 1:
        raise ValueError("Legacy trajectory fallback requires a positive integer traj_interval")
    frames = list(frames)
    inferred = any(frame.step is None for frame in frames)
    frames = [
        replace(frame, step=i * int(traj_interval)) if frame.step is None else frame
        for i, frame in enumerate(frames)
    ]
    result = select_min_energy_frame(
        frames, samples, energy_key=energy_key, fallback=fallback, alignment="previous"
    )
    result.summary["legacy_step_inference"] = inferred
    result.summary["legacy_traj_interval"] = int(traj_interval)
    return result


def band_payload(bands, *, energy_unit="eV", shift_to_fermi=True):
    energy = np.asarray(bands.energies)
    if energy.ndim != 3 or not np.isfinite(energy).all():
        raise ValueError("Band payload needs finite (spin,k,band) arrays")
    ef = bands.fermi_energy if energy_unit == "eV" else None
    shifted = bool(shift_to_fermi and ef is not None)
    distances = bands.distances if bands.distances is not None else np.arange(energy.shape[1], dtype=float)
    indices = np.asarray(bands.label_indices, int)
    return {
        "fermi_energy": ef,
        "n_spins": energy.shape[0],
        "n_bands": energy.shape[2],
        "n_kpoints": energy.shape[1],
        "energies": (energy - float(ef) if shifted and ef is not None else energy).tolist(),
        "k_distance": np.asarray(distances).tolist(),
        "tick_positions": np.asarray(distances)[indices].tolist(),
        "tick_labels": list(bands.labels),
        "energy_unit": energy_unit,
        "shifted_to_fermi": shifted,
        "fermi_source": bands.metadata.get(
            "fermi_source", bands.metadata.get("source_format", "explicit") if ef is not None else None
        ),
        "segments": list(bands.segments),
    }


def calculation_payload(arrays, summary, *, engine):
    """Adapt native DFT arrays to legacy generator units with explicit provenance.

    Generator VASP results prefer force-consistent free energy and compressive
    kbar stress. OpenMX's reported strain derivative uses its existing tensile
    kbar convention. The native energy and tensile tensor are retained too.
    """
    from ...io.calculations import KBAR_EV_ANGSTROM3

    result = dict(summary)
    if "energy_eV" in arrays:
        result["energy_eV"] = float(arrays["energy_eV"])
        result["energy"] = (
            float(arrays.get("free_energy_eV", arrays["energy_eV"]))
            if engine == "vasp"
            else result["energy_eV"]
        )
        result["energy_definition"] = (
            "free_energy"
            if engine == "vasp" and "free_energy_eV" in arrays
            else summary.get("energy_definition", "reported")
        )
    if "free_energy_eV" in arrays:
        result["free_energy_eV"] = float(arrays["free_energy_eV"])
    if "forces" in arrays:
        forces = np.asarray(arrays["forces"])
        result["forces"] = forces.tolist()
        result["max_force"] = float(np.max(np.linalg.norm(forces, axis=1), initial=0.0))
        result["natoms"] = len(forces)
        if "energy" in result and len(forces):
            result["energy_per_atom"] = result["energy"] / len(forces)
    if "stress" in arrays:
        result["stress_eV_A3"] = np.asarray(arrays["stress"]).tolist()
        result["stress"] = (
            np.asarray(arrays["stress"]) * (-1 if engine == "vasp" else 1) / KBAR_EV_ANGSTROM3
        ).tolist()
    result["legacy_units"] = {
        "energy": "eV",
        "forces": "eV/Å",
        "stress": "kbar",
        "stress_sign": "compressive" if engine == "vasp" else "tensile",
    }
    outcome = summary.get("electronic_converged", summary.get("scf_converged"))
    result["converged"] = outcome
    result["calculation_status"] = (
        "converged" if outcome is True else "unconverged" if outcome is False else "unknown"
    )
    return result


def _text_table(path, arrays, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, arrays, header=header)


def export_run(run_dir, *, output_dir=None, profile="generator"):
    """Export checksummed native runs without recomputation or model loading."""
    from ...io.artifacts import load_manifest
    from ...io.records import frames_from_arrays, structure_from_arrays

    if profile != "generator":
        raise ValueError("Only the generator export profile is supported")
    root = Path(run_dir).expanduser().resolve()
    manifest = load_manifest(root)
    target = Path(output_dir).expanduser().resolve() if output_dir else root / "generator"
    target.mkdir(parents=True, exist_ok=True)
    name = manifest["workflow"]
    record = manifest["cases"].get("result") or manifest["cases"].get("bands")
    if record is None:
        raise ValueError("Generator export requires the workflow's final numerical result")
    with np.load(root / record["path"], allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    metadata = record["metadata"]
    summary = dict(metadata.get("summary", {}))
    if name == "postprocess":
        summary = calculation_payload(arrays, summary, engine=metadata.get("source", "unknown"))
    if summary.get("failure") or summary.get("blew_up") or summary.get("converged") is False:
        raise ValueError("Generator success export requires a successful, converged final result")
    if name == "nvt" and summary.get("completed") is not True:
        raise ValueError("NVT success export requires completion of the requested steps")
    if metadata.get("integration", {}).get("converged") is False:
        raise ValueError("Electronic integration has not converged")
    provenance = {
        "source_fingerprint": manifest["fingerprint"],
        "inputs": manifest["inputs"],
        "versions": manifest["versions"],
        "units": metadata.get("units", {}),
        "dimensions": metadata.get("dims", {}),
        "calculation_status": manifest["status"],
        "model": manifest.get("scientific_model", {}),
        "scientific_metadata": {
            key: value for key, value in metadata.items() if key not in {"summary", "state", "dims", "units"}
        },
    }
    products = []

    def write_json(filename, value):
        atomic_json(target / filename, {**value, "queer_provenance": provenance})
        products.append(filename)

    if name in {"ahc", "shift_current", "chern_number"}:
        if name not in arrays or not np.isfinite(arrays[name]).all():
            raise ValueError("Required electronic response product is missing/nonfinite")
        convention = manifest["config"]["inputs"].get("export_metadata", {})
        if name == "ahc":
            tensor = arrays[name] * convention.get("hall_export_sign", 1)
            vector = np.column_stack((tensor[:, 1, 2], tensor[:, 2, 0], tensor[:, 0, 1]))
            if convention.get("hall_components") == ["xy"]:
                vector[:, :2] = 0
            values = np.column_stack((arrays["chemical_potentials"], vector))
            filename, header = "Out/AHC/ahc.dat", "absolute_mu_eV sigma_yz sigma_zx sigma_xy (S/cm)"
        elif name == "shift_current":
            components = [(a, b, c) for a in range(3) for b in range(3) for c in range(b, 3)]
            tensor = arrays[name]
            values = np.column_stack((arrays["omega"], *(tensor[:, a, b, c] for a, b, c in components)))
            labels = ["".join("xyz"[axis] for axis in component) for component in components]
            filename = "Out/Shift_Current/shift_current.dat"
            header = "omega(eV) " + " ".join(labels) + " (uA/V^2)"
        else:
            values = np.atleast_2d(arrays[name])
            filename, header = "Out/Chern_Number/chern_number.dat", "Chern_number (dimensionless)"
        _text_table(target / filename, values, header)
        products.append(filename)
        write_json(
            "pyatb_results.json",
            {
                "converged": True,
                "function": name,
                "integration": metadata.get("integration"),
                "files": [filename],
                "legacy_conventions": convention,
            },
        )
    elif name == "phonons":
        if "frequencies_THz" not in arrays or not np.isfinite(arrays["frequencies_THz"]).all():
            raise ValueError("Required phonon dispersion is missing/nonfinite")
        frequencies = arrays["frequencies_THz"].T
        segments = metadata.get("segments", [])
        x = np.asarray(arrays["q_distance"]).copy()
        ticks, labels = [], []
        if segments:
            for i, segment in enumerate(segments):
                start = int(segment.get("start_index", segment.get("start", 0)))
                end = int(segment.get("end_index", segment.get("end", len(x) - 1)))
                x[start : end + 1] = np.linspace(i, i + 1, end - start + 1)
                ticks.extend([float(i), float(i + 1)])
                labels.extend([segment.get("start_label", ""), segment.get("end_label", "")])
        payload = {
            "fermi_energy": None,
            "n_spins": 1,
            "n_bands": frequencies.shape[1],
            "n_kpoints": len(frequencies),
            "k_distance": x.tolist(),
            "energies": [frequencies.tolist()],
            "tick_positions": ticks or [float(x[0]), float(x[-1])],
            "tick_labels": labels or ["", ""],
            "energy_unit": "THz",
            "shifted_to_fermi": False,
        }
        _text_table(
            target / "phonon_band.dat",
            np.column_stack((x, frequencies)),
            "segment_coordinate frequencies_THz",
        )
        products.append("phonon_band.dat")
        write_json(
            "phonon_results.json",
            {
                **summary,
                "converged": True,
                "n_bands": frequencies.shape[1],
                "n_qpoints": len(x),
                "segments": segments,
            },
        )
        write_json("results.json", {"converged": True, "band_structure": payload})
    elif name in {"nvt", "relax", "select"}:
        from ...io.atomistic import write_structure, write_trajectory

        frames = frames_from_arrays(arrays)
        if name != "select":
            if not frames:
                raise ValueError("Simulation export requires retained trajectory frames")
            filename = "trajectory.traj" if name == "nvt" else "relax.traj"
            write_trajectory(target / filename, frames)
            products.append(filename)
        structure = structure_from_arrays(arrays, prefix="final_")
        for filename in ("POSCAR", "CONTCAR"):
            write_structure(target / filename, structure)
            products.append(filename)
        write_json(
            {"nvt": "nvt_results.json", "relax": "relax_results.json", "select": "select_results.json"}[name],
            summary,
        )
    elif name == "postprocess" and "energies" not in arrays:
        from ...io.atomistic import write_structure
        from ...structures import Structure

        if {"cell", "positions", "atomic_numbers"} <= set(arrays):
            write_structure(
                target / "CONTCAR", Structure(arrays["cell"], arrays["positions"], arrays["atomic_numbers"])
            )
            products.append("CONTCAR")
        write_json("results.json", summary)
    elif name in {"bands", "postprocess", "hamiltonian_bands"}:
        from ...io.bands import BandData

        energies = arrays["energies"]
        if energies.ndim == 2:
            ef = manifest["scientific_model"]["fermi_energy_eV"]
            energies = energies.T[None] + ef
        else:
            ef = metadata.get("fermi_energy_eV")
        bands = BandData(
            energies,
            arrays.get("kpoints"),
            fermi_energy=ef,
            labels=metadata.get("labels", arrays.get("labels", []).tolist() if "labels" in arrays else []),
            label_indices=metadata.get("label_indices", arrays.get("symmetry", [])),
            distances=arrays.get("distances"),
            segments=metadata.get("segments", arrays.get("segments", [])),
            metadata=metadata,
        )
        write_json("results.json", {**summary, "band_structure": band_payload(bands)})
        if name == "hamiltonian_bands":
            distance = bands.distances if bands.distances is not None else np.arange(energies.shape[1])
            _text_table(
                target / "band_1.dat",
                np.column_stack((distance, energies[0] - float(ef or 0))),
                "path_coordinate E_minus_EF_eV",
            )
            products.append("band_1.dat")
    elif name == "structure":
        payload = structure_payload(structure_from_arrays(arrays))
        (target / "POSCAR").write_text(payload["poscar"], encoding="utf-8")
        products.append("POSCAR")
        write_json("structure.json", payload)
    elif name == "dataset":
        for stem in ("train", "valid", "test"):
            if stem + "_extxyz" not in arrays:
                raise ValueError("Dataset export lacks a required split")
            filename = stem + ".xyz"
            (target / filename).write_text(str(arrays[stem + "_extxyz"]), encoding="utf-8")
            products.append(filename)
        write_json("dataset_results.json", summary)
    elif name == "band_compare":
        write_json("band_comparison.json", summary)
    elif name == "hamiltonian_pack":
        from ...adapters.hamgnn import write_legacy_graphs
        from ...io.hamiltonian import read_graphs

        graph_path = root / metadata["graph_archive"]
        if digest_file(graph_path) != metadata["graph_archive_sha256"]:
            raise ValueError("Graph archive changed since calculation")
        write_legacy_graphs(target / "graph_data.npz", read_graphs(graph_path))
        products.append("graph_data.npz")
        write_json("pack_results.json", summary)
    else:
        raise ValueError(f"Generator export is unavailable for workflow {name!r}")
    atomic_json(
        target / "manifest.json",
        {
            "schema_version": 1,
            "profile": profile,
            "source_fingerprint": manifest["fingerprint"],
            "products": [{"path": p, "sha256": digest_file(target / p)} for p in products],
        },
    )
    return target
