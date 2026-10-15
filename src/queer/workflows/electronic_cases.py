"""Recoverable bands, susceptibility, spectral, momentum, and EPW cases."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..config import WorkflowConfig
    from ..io.artifacts import RunStore


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
