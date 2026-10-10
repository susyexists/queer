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
    directory = config.material_dir
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
    if config.workflow == "bands":
        case = "bands"
        if store.completed(case):
            return
        if "kpath" in params:
            path = model.kpath(params["kpath"], params["n_points"])
            points = path.path
            symmetry, labels = path.sym, np.asarray(path.labels)
        else:
            points = np.asarray(params["kpoints"], dtype=float)
            symmetry, labels = np.array([], dtype=int), np.array([], dtype="U1")
        energies = model.energies(points, bands=params.get("bands"), execution=execution)
        store.commit(
            case,
            {"kpoints": points, "energies": energies, "symmetry": symmetry, "labels": labels},
            metadata={"energy_unit": "eV", "coordinates": "fractional_reciprocal"},
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
    config: WorkflowConfig | Mapping[str, Any], run_dir: str | Path | None = None, resume: bool = False
) -> Path:
    """Calculate raw artifacts; render separately with :func:`render_run`.

    The binding-energy interval is half-open, matching NumPy ``arange``. Resume
    checks scientific parameters, all inputs and every completed artifact before
    skipping it. A completed run can be resumed idempotently.
    """
    if not isinstance(config, WorkflowConfig):
        config = config_from_mapping(config)
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
        model = _load_model(config)
        with execution_session(config.execution):
            _calculate(config, model, store)
        if not store.manifest["cases"]:
            raise ValueError("Configuration produced no numerical cases")
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
