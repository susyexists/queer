"""Scientific service calculation handlers with workflow-owned persistence."""

import json

import numpy as np

from ..io.records import frames_arrays, structure_arrays
from ..observables.result import ObservableResult
from .service_state import _checkpoint, _commit, _latest_state, _record_provider, resolve_provider


def calculate_structure(config, store, *, provider=None, structure=None, parameters):
    assert structure is not None
    from ..geometry.structures import apply_transforms, compute_bonds, expand_variants, lattice_parameters

    recipes = expand_variants(parameters.get("transforms", []))
    variants = [apply_transforms(structure, recipe) for recipe in recipes]
    arrays = structure_arrays(variants[0])
    for i, variant in enumerate(variants):
        arrays.update(structure_arrays(variant, prefix=f"variant_{i:06d}_"))
    metadata = {
        "lattice_parameters": lattice_parameters(variants[0]),
        "variant_recipes": recipes,
        "n_variants": len(variants),
        "summary": {"converged": True},
    }
    if parameters.get("bonds", True):
        metadata["bonds"] = compute_bonds(variants[0])
    if parameters.get("symmetry", False):
        from ..geometry.symmetry import compute_band_path

        metadata["band_path"] = compute_band_path(variants[0], symprec=parameters.get("symprec", 1e-5))
    _commit(store, ObservableResult(arrays, units={"cell": "Å", "positions": "Å"}, metadata=metadata))


def calculate_dynamics(config, store, *, provider=None, structure=None, parameters):
    name = config.workflow
    assert structure is not None
    from ..atomistic.dynamics import relax_fire, run_nvt
    from ..geometry.structures import expand_diagonal

    reference, _ = expand_diagonal(structure, parameters.get("supercell", (1, 1, 1)))
    evaluator = resolve_provider(config, reference, provider)
    _record_provider(store, evaluator)
    function = run_nvt if name == "nvt" else relax_fire
    parameters["state"] = _latest_state(store)
    parameters["checkpoint"] = lambda state: _checkpoint(store, state)
    result = function(structure, evaluator, **parameters)
    _checkpoint(store, result.state)
    frames = list(result.frames)
    if not frames or frames[-1].step != result.final_frame.step:
        frames.append(result.final_frame)
    if result.summary.get("failure") or result.summary.get("blew_up"):
        store.commit(
            f"failed_simulation_{len(store.manifest['attempts']):04d}",
            result.observable.arrays,
            metadata=result.observable.artifact_metadata(),
        )
        raise RuntimeError(f"{name} failed: {result.summary.get('failure', 'unstable dynamics')}")
    _commit(
        store,
        result.observable,
        arrays={
            **frames_arrays(frames),
            **structure_arrays(result.final_frame.structure, prefix="final_"),
        },
        state=result.state,
    )


def calculate_phonons(config, store, *, provider=None, structure=None, parameters):
    assert structure is not None
    from ..atomistic.phonons import assemble_force_constants, phonon_bands, plan_displacements

    plan_keys = {key: parameters[key] for key in ("supercell", "delta_A") if key in parameters}
    plan = plan_displacements(structure, **plan_keys)
    if "forces" in config.inputs:
        path = config.material_dir / config.inputs["forces"]
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            with loaded as archive:
                forces = np.array(archive["forces"])
        else:
            forces = loaded
    else:
        evaluator = resolve_provider(config, plan.reference, provider)
        _record_provider(store, evaluator)
        force_rows = []
        for i, displaced in enumerate(plan.configurations):
            case = f"displacement_{i:08d}"
            if store.completed(case):
                with np.load(store.root / store.manifest["cases"][case]["path"], allow_pickle=False) as data:
                    force = data["forces"]
            else:
                evaluation = (
                    evaluator.evaluate(displaced) if hasattr(evaluator, "evaluate") else evaluator(displaced)
                )
                force = evaluation.forces_eV_A
                store.commit(
                    case,
                    {"forces": force},
                    metadata={"units": {"forces": "eV/Å"}, "displacement_index": i},
                )
            force_rows.append(force)
        forces = np.asarray(force_rows)
    # Explicit assembly validates every displacement before any final result.
    assembly = {
        key: parameters[key]
        for key in ("frederiksen", "acoustic", "symmetrize_iterations")
        if key in parameters
    }
    assemble_force_constants(plan, forces, **assembly)
    phonon_result = phonon_bands(structure, forces=forces, **parameters)
    _commit(store, phonon_result, arrays=structure_arrays(structure))


def calculate_select(config, store, *, provider=None, structure=None, parameters):
    from ..atomistic.selection import select_min_energy_frame
    from ..io.atomistic import read_frames

    frames = read_frames(config.material_dir / config.inputs["trajectory"])
    series = None
    if "series" in config.inputs:
        series = json.loads((config.material_dir / config.inputs["series"]).read_text())
        if isinstance(series, dict):
            series = series.get("series", series.get("raw_data", {}).get("series"))
    selected = select_min_energy_frame(frames, samples=series, fallback=structure, **parameters)
    _commit(
        store,
        ObservableResult(
            structure_arrays(selected.frame.structure, prefix="final_"),
            metadata={"summary": selected.summary},
        ),
        arrays=frames_arrays([selected.frame]),
    )
