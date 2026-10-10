"""Compute and incrementally commit independent numerical cases."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config import WorkflowConfig, config_from_mapping
from ..execution import execution_session
from ..io.artifacts import RunStore, new_run_directory

logger = logging.getLogger(__name__)


def _input_files(config: WorkflowConfig) -> list[Path]:
    from .finite_field import FINITE_FIELD_NAMES, finite_field_input_files
    from .services import SERVICE_SPECS, service_input_files

    if config.workflow in FINITE_FIELD_NAMES:
        return finite_field_input_files(config)
    if config.workflow in SERVICE_SPECS:
        return service_input_files(config)
    directory = config.material_dir
    if config.inputs.get("model") == "openmx":
        if "scfout" in config.inputs:
            files = [directory / config.inputs["scfout"]]
        else:
            files = list(directory.glob("*.scfout"))
            if len(files) != 1:
                raise ValueError("Provide inputs.scfout when there is not exactly one SCFOUT file")
        files.extend(
            directory / config.inputs[key] for key in ("openmx_input", "poscar") if key in config.inputs
        )
        files.extend(directory / value for value in config.inputs.get("pao_files", {}).values())
        if "pao_dir" in config.inputs:
            folder = directory / config.inputs["pao_dir"]
            if not folder.is_dir():
                raise ValueError(f"Missing PAO directory: {folder}")
            files.extend(sorted(folder.glob("*.pao")))
        return files
    if config.workflow == "epw":
        # The EPW reader consumes metadata and channels in this directory.
        files = [directory / "gkk.out"] if (directory / "gkk.out").is_file() else []
        for channel in ("omega", "gkk_re", "gkk_im", "enk", "enkq"):
            folder = directory / channel
            if not folder.is_dir():
                raise ValueError(f"Missing EPW channel: {folder}")
            files.extend(
                file for file in folder.iterdir() if file.is_file() and not file.name.startswith(".")
            )
        if not files:
            raise ValueError(f"No EPW input files found: {directory}")
        return files
    files = [directory / config.inputs.get("hr", "wannier90_hr.dat")]
    files.extend(directory / config.inputs[key] for key in ("poscar", "wsvec") if key in config.inputs)
    if config.parameters.get("interpolation") == "ws_distance" and "wsvec" not in config.inputs:
        hr_path = files[0]
        files.append(hr_path.with_name(hr_path.name.removesuffix("_hr.dat") + "_wsvec.dat"))
    return files


def _load_model(config: WorkflowConfig):
    params = config.parameters
    if config.inputs.get("model") == "openmx":
        from ..models.openmx import OpenMXModel

        model = OpenMXModel.from_directory(
            config.material_dir,
            scfout=config.inputs.get("scfout"),
            ef=params.get("ef"),
            openmx_input=config.inputs.get("openmx_input"),
            execution=config.execution,
        )
        if config.inputs.get("poscar"):
            model.poscar = config.material_dir / config.inputs["poscar"]
        return model
    if config.workflow == "epw":
        from ..models.epw import EPWModel

        model = EPWModel.from_directory(
            config.material_dir, **{key: params[key] for key in ("nk", "nq", "nph", "ef") if key in params}
        )
        model.reduce_g()
        return model
    from ..models.wannier import WannierModel

    return WannierModel.from_directory(
        config.material_dir,
        hr=config.inputs.get("hr", "wannier90_hr.dat"),
        poscar=config.inputs.get("poscar"),
        ef=params["ef"],
        interpolation=params.get("interpolation", "standard"),
        wsvec=config.inputs.get("wsvec"),
    )


def _calculate(config: WorkflowConfig, model, store: RunStore) -> None:
    params, execution = config.parameters, config.execution.resolved()
    from .registry import REGISTRY, calculate_registered

    if config.workflow in REGISTRY:
        calculate_registered(config, model, store)
        return
    if config.workflow == "bands":
        case = "bands"
        if store.completed(case):
            return
        if "kpath" in params:
            path = model.kpath(params["kpath"], params["n_points"], poscar=model.poscar)
            points = path.path
            symmetry, labels = path.sym, np.asarray(path.labels)
        else:
            points = np.asarray(params["kpoints"], dtype=float)
            symmetry, labels = np.array([], dtype=int), np.array([], dtype="U1")
        if params.get("scissors") or params.get("spin"):
            from ..observables.electronic import band_structure

            energies = band_structure(
                model,
                points,
                bands=params.get("bands"),
                scissors=params.get("scissors", 0),
                n_occ=params.get("n_occ"),
                spin=params.get("spin"),
                execution=execution,
            ).arrays["energies"]
        else:
            energies = model.energies(points, bands=params.get("bands"), execution=execution)
        band_arrays = {"kpoints": points, "energies": energies, "symmetry": symmetry, "labels": labels}
        if hasattr(model, "band_channels"):
            indices = model._band_indices(
                params.get("bands"),
                **({"spin": params["spin"]} if params.get("spin") is not None else {}),
            )
            band_arrays["band_channels"] = model.band_channels[indices]
            spin_mode = getattr(model, "spin_mode", "nonmagnetic")
            band_arrays["channel_labels"] = np.asarray(
                [
                    ("up" if channel == 0 else "down") if spin_mode == "collinear" else spin_mode
                    for channel in model.band_channels[indices]
                ]
            )
        store.commit(
            case,
            band_arrays,
            metadata={
                "energy_unit": "eV",
                "coordinates": "fractional_reciprocal",
                "fermi_energy": float(model.fermi_energy),
                "model": config.inputs.get("model", "wannier"),
                "scientific_metadata": getattr(model, "metadata", {}),
            },
        )
    elif config.workflow == "susceptibility":
        from ..observables.response import susceptibility

        mesh = np.asarray(params["k_mesh"], dtype=float)
        mesh_energies = None
        for index, point in enumerate(params["q_path"]):
            case = f"q_{index:06d}"
            if store.completed(case):
                continue
            if mesh_energies is None:
                mesh_energies = model.energies(mesh, execution=execution)
            response = susceptibility(
                model,
                np.asarray([point]),
                mesh,
                band_indices=params.get("band_indices"),
                temperature=params["temperature"],
                broadening=params["broadening"],
                execution=execution,
                mesh_energies=mesh_energies,
            )
            store.commit(case, {"qpoint": point, "response": response}, metadata={"q_index": index})
    elif config.workflow == "spectral":
        from ..observables.response import spectral_from_eig

        points = np.asarray(params["kpoints"], dtype=float)
        if "v0_values" in params:
            from ..observables.spectral import iter_spectral_arpes_path_sweep

            for v0_index, v0 in enumerate(params["v0_values"]):
                pending = [
                    index
                    for index in range(len(params["omega"]))
                    if not store.completed(f"v0_{v0_index:04d}_omega_{index:06d}")
                ]
                if not pending:
                    continue
                omega = np.asarray(params["omega"])[pending]
                spectra = iter_spectral_arpes_path_sweep(
                    model,
                    points,
                    model.g_vec,
                    params["photon_energy"],
                    [v0],
                    omega,
                    eta=params["eta"],
                    align_zero=params.get("align_zero", True),
                    execution=execution,
                )
                for _, subset_index, spectrum_row in spectra:
                    index = pending[subset_index]
                    spectrum = np.asarray(spectrum_row)[None, :]
                    store.commit(
                        f"v0_{v0_index:04d}_omega_{index:06d}",
                        {
                            "kpoints": points,
                            "omega": [omega[subset_index]],
                            "spectrum": spectrum,
                            "valid_mask": np.isfinite(spectrum),
                            "v0": [v0],
                        },
                        metadata={
                            "omega_index": index,
                            "v0_index": v0_index,
                            "energy_unit": "eV",
                            "geometry": "curved_ARPES",
                        },
                    )
            return
        eigenvalues = None
        for index, omega in enumerate(params["omega"]):
            case = f"omega_{index:06d}"
            if store.completed(case):
                continue
            if eigenvalues is None:
                eigenvalues = model.energies(points, execution=execution).T
            spectrum = spectral_from_eig(eigenvalues, [omega], params["eta"])
            store.commit(
                case,
                {
                    "kpoints": points,
                    "omega": [omega],
                    "spectrum": spectrum,
                    "valid_mask": np.isfinite(spectrum),
                },
                metadata={"omega_index": index, "energy_unit": "eV"},
            )
    elif config.workflow == "momentum":
        from ..observables.momentum import iter_binding_energy_surfaces

        v0_values = params.get("v0_values", [params.get("v0")])
        binding_values = np.arange(params["binding_min"], params["binding_max"], params["binding_step"])
        for v0_index, v0 in enumerate(v0_values):
            for binding_index, binding in enumerate(binding_values):
                case = f"v0_{v0_index:04d}_binding_{binding_index:06d}"
                if store.completed(case):
                    continue
                # Half a step gives the half-open iterator exactly this slice.
                slices = iter_binding_energy_surfaces(
                    model,
                    model.g_vec,
                    params["photon_energy"],
                    params["fermi_energy"],
                    [float(binding), float(binding + params["binding_step"] / 2)],
                    params["binding_step"],
                    v0,
                    params["n_points"],
                    factor=params.get("factor"),
                    align=params.get("align"),
                    energy_shift=params.get("energy_shift", 0.0),
                    sigma=params["sigma"],
                    execution=execution,
                )
                for energy, surface in slices:
                    store.commit(
                        case,
                        {"surface": surface, "binding_energy": [energy], "v0": [v0]},
                        metadata={
                            "columns": ["kx", "ky", "kz", "e", "I"],
                            "energy_unit": "eV",
                            "coordinate_unit": "angstrom^-1",
                        },
                    )
    elif config.workflow == "epw":
        for index, q in enumerate(params["q_indices"]):
            case = f"q_{index:06d}"
            if store.completed(case):
                continue
            value = model.self_energy(q, temperature=params["temperature"], broadening=params["broadening"])
            store.commit(
                case,
                {"q_index": [q], "self_energy": value, "phonons": model.ph[:, q]},
                metadata={"normalization": "legacy EPW reduced-coupling response", "phonon_unit": "meV"},
            )


def run_workflow(
    config: WorkflowConfig | Mapping[str, Any],
    run_dir: str | Path | None = None,
    resume: bool = False,
    *,
    force_provider=None,
    force_provider_id: str | None = None,
) -> Path:
    """Calculate raw artifacts; render separately with :func:`render_run`.

    The binding-energy interval is half-open, matching NumPy ``arange``. Resume
    checks scientific parameters, all inputs and every completed artifact before
    skipping it. A completed run can be resumed idempotently.
    """
    if not isinstance(config, WorkflowConfig):
        config = config_from_mapping(config)
    from dataclasses import replace

    from .finite_field import FINITE_FIELD_NAMES, calculate_finite_field, load_finite_field_source
    from .services import SERVICE_SPECS, calculate_service

    if force_provider is not None:
        if config.workflow not in {"phonons", "nvt", "relax"}:
            raise ValueError("force_provider is only consumed by phonons, nvt, and relax")
        identity = force_provider_id or getattr(force_provider, "fingerprint", None)
        if not isinstance(identity, str) or not identity:
            raise ValueError("A callback force_provider requires force_provider_id or a string fingerprint")
        config = replace(config, inputs={**config.inputs, "provider_identity": identity})
    if resume and run_dir is None:
        raise ValueError("resume requires an explicit run_dir")
    root = (
        Path(run_dir).expanduser().resolve()
        if run_dir is not None
        else new_run_directory(config.results_dir, config.workflow)
    )
    files = _input_files(config)
    missing = [str(file) for file in files if not file.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing workflow inputs: {', '.join(missing)}")
    store = RunStore(root, config.as_dict(), files, resume=resume)
    logger.info("Run directory: %s", root)
    try:
        if config.workflow in FINITE_FIELD_NAMES:
            source = load_finite_field_source(config)
            store.manifest["scientific_model"] = {
                "model": config.inputs.get(
                    "model", "family" if "family_manifest" in config.inputs else "grid"
                ),
                "preset": config.parameters.get("preset", "physical"),
                "conventions": "Cartesian field plane; k in Angstrom^-1; eV; fields V/m; sheet current A/m",
            }
            store.save()
            with execution_session(config.execution):
                calculate_finite_field(config, source, store)
        elif config.workflow in SERVICE_SPECS:
            spec = SERVICE_SPECS[config.workflow]
            store.manifest["scientific_model"] = {
                "input_kind": spec.input_kind,
                "checkpoint_state": spec.checkpoint_kind,
                "required_products": list(spec.required_products),
                "provider": config.parameters.get("provider"),
                "provider_identity": config.inputs.get("provider_identity"),
                "conventions": "row lattice vectors; Cartesian Å; eV; eV/Å; eV/Å³; amu; fs; K",
            }
            store.save()
            with execution_session(config.execution):
                calculate_service(config, store, provider=force_provider)
        else:
            model = _load_model(config)
            store.manifest["scientific_model"] = {
                "model": config.inputs.get("model", "wannier"),
                "fermi_energy_eV": float(getattr(model, "fermi_energy", config.parameters.get("ef", 0))),
                "spin_degeneracy": float(getattr(model, "spin_degeneracy", 1)),
                "spin_mode": getattr(model, "spin_mode", "unspecified"),
                "conventions": (
                    "EPW reduced coupling; phonons in meV"
                    if config.workflow == "epw"
                    else "positive Fourier phase; fractional reciprocal k; eV; angstrom"
                ),
                "definitions": getattr(model, "metadata", {}),
            }
            store.save()
            with execution_session(config.execution):
                _calculate(config, model, store)
        if not store.manifest["cases"]:
            raise ValueError("Configuration produced no numerical cases")
        if config.workflow in SERVICE_SPECS and any(
            not store.completed(case) for case in SERVICE_SPECS[config.workflow].required_products
        ):
            raise RuntimeError("Service workflow lacks a required final product")
        store.finish()
    except BaseException as error:
        store.finish(
            status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
            error=error,
        )
        raise
    finally:
        store.release()
    return root
