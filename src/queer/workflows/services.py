"""Scientific service workflows with declared non-Hamiltonian input kinds."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

import numpy as np

from ..io.artifacts import canonical_json, digest_file
from ..observables.result import ObservableResult


@dataclass(frozen=True)
class ServiceSpec:
    input_kind: str
    input_keys: tuple[str, ...]
    parameters: tuple[str, ...]
    checkpoint_kind: str = "final_result"
    required_products: tuple[str, ...] = ("result",)


SERVICE_SPECS = {
    "structure": ServiceSpec("structure", ("structure",), ("transforms", "symmetry", "symprec", "bonds")),
    "phonons": ServiceSpec(
        "structure_forces",
        ("structure", "forces", "checkpoint"),
        (
            "provider",
            "supercell",
            "delta_A",
            "band_path",
            "qpoints",
            "n_qpoints_per_segment",
            "frederiksen",
            "acoustic",
            "symmetrize_iterations",
        ),
    ),
    "nvt": ServiceSpec(
        "force_provider",
        ("structure", "checkpoint"),
        (
            "provider",
            "temperature_K",
            "timestep_fs",
            "n_steps",
            "friction_fs_inv",
            "log_interval",
            "traj_interval",
            "seed",
            "supercell",
            "initial_velocities",
            "remove_com",
            "remove_rotation",
        ),
    ),
    "relax": ServiceSpec(
        "force_provider",
        ("structure", "checkpoint"),
        (
            "provider",
            "fmax_eV_A",
            "max_steps",
            "supercell",
            "dt",
            "dtmax",
            "n_min",
            "increase",
            "decrease",
            "alpha_start",
            "alpha_decay",
            "max_displacement_A",
        ),
    ),
    "select": ServiceSpec("trajectory", ("trajectory", "series", "structure"), ("energy_key", "alignment")),
    "postprocess": ServiceSpec(
        "calculation",
        ("source", "outcar", "vasprun", "eigenval", "kpoints_file", "bandfile", "output", "wout"),
        ("engine", "fermi_energy", "fermi_source"),
    ),
    "hamiltonian_pack": ServiceSpec(
        "openmx",
        ("scfout", "h0", "openmx_input"),
        ("nao_max", "basis_layout", "atomic_numbers", "overlap_only", "spin"),
    ),
    "hamiltonian_bands": ServiceSpec(
        "graph",
        ("graph", "predictions"),
        (
            "graph_id",
            "prediction_unit",
            "fermi_energy",
            "kpoints",
            "kpath",
            "n_points",
            "hermitian_policy",
            "electron_num",
            "filling_grid",
            "temperature_kelvin",
        ),
    ),
    "band_compare": ServiceSpec(
        "imported_bands",
        ("reference", "candidate"),
        ("reference_format", "candidate_format", "energy_window", "align_fermi"),
    ),
    "dataset": ServiceSpec(
        "reference_dataset",
        ("dataset",),
        ("train_ratio", "val_ratio", "test_ratio", "seed", "stress_unit", "stress_sign"),
    ),
}
for _name, _kind in {
    "nvt": "restart_state",
    "relax": "restart_state",
    "phonons": "displacement_forces",
}.items():
    SERVICE_SPECS[_name] = replace(SERVICE_SPECS[_name], checkpoint_kind=_kind)
SERVICE_NAMES = tuple(SERVICE_SPECS)
SERVICE_INPUTS = {key for spec in SERVICE_SPECS.values() for key in spec.input_keys} | {
    "provider_identity",
    "export_metadata",
}


def validate_service(name, inputs, parameters):
    spec = SERVICE_SPECS[name]
    unknown_inputs = (
        set(inputs) - set(spec.input_keys) - {"material_dir", "provider_identity", "export_metadata"}
    )
    if unknown_inputs:
        raise ValueError(f"Unconsumed {name} inputs: {sorted(unknown_inputs)}")
    unknown = set(parameters) - set(spec.parameters)
    if unknown:
        raise ValueError(f"Unknown {name} parameters: {sorted(unknown)}")
    for key in spec.input_keys:
        if key in inputs and (not isinstance(inputs[key], str) or not inputs[key]):
            raise ValueError(f"inputs.{key} must be a nonempty filename")
    required = {
        "structure": "structure",
        "phonons": "structure",
        "nvt": "structure",
        "relax": "structure",
        "select": "trajectory",
        "hamiltonian_pack": "scfout",
        "hamiltonian_bands": "graph",
        "band_compare": "reference",
        "dataset": "dataset",
    }.get(name)
    if required is not None and required not in inputs:
        raise ValueError(f"{name} requires inputs.{required}")
    if name == "hamiltonian_bands" and ("kpoints" in parameters) == ("kpath" in parameters):
        raise ValueError("hamiltonian_bands requires exactly one of kpoints or kpath")
    if name == "band_compare" and "candidate" not in inputs:
        raise ValueError("band_compare requires inputs.candidate")
    if "provider" in parameters and not isinstance(parameters["provider"], dict):
        raise ValueError("parameters.provider must be a provider configuration table")


def service_input_files(config):
    spec = SERVICE_SPECS[config.workflow]
    files = [config.material_dir / config.inputs[key] for key in spec.input_keys if key in config.inputs]
    if config.workflow == "postprocess":
        files = list(_postprocess_inputs(config).values())
    if any(path.is_dir() for path in files):
        raise ValueError("Service input paths must identify files, not directories")
    return files


def _postprocess_inputs(config):
    available = {
        key: config.material_dir / config.inputs[key]
        for key in SERVICE_SPECS["postprocess"].input_keys
        if key in config.inputs
    }
    if available:
        return available
    names = {
        "vasp": {
            "outcar": "OUTCAR",
            "vasprun": "vasprun.xml",
            "eigenval": "EIGENVAL",
            "kpoints_file": "KPOINTS",
        },
        "openmx": {"output": "scf.out", "bandfile": "scf.Band"},
        "wannier": {"wout": "wannier90.wout", "bandfile": "wannier90_band.dat"},
    }
    defaults = names.get(config.parameters.get("engine", "vasp"), {})
    return {
        key: config.material_dir / name
        for key, name in defaults.items()
        if (config.material_dir / name).is_file()
    }


class HarmonicForceProvider:
    """Analytical fixture provider, E=k/2 sum(|r-r0|²), eV/Å units."""

    def __init__(self, structure, spring_constant_eV_A2=1.0, reference_positions=None):
        self.reference = np.asarray(
            structure.positions if reference_positions is None else reference_positions, float
        )
        self.spring = float(spring_constant_eV_A2)
        if self.reference.shape != structure.positions.shape or not np.isfinite(self.reference).all():
            raise ValueError("Harmonic reference positions must match structure atoms")
        if not np.isfinite(self.spring) or self.spring <= 0:
            raise ValueError("Harmonic spring constant must be positive eV/Å²")
        self.fingerprint = hashlib.sha256(
            canonical_json({"kind": "harmonic", "k": self.spring, "r0": self.reference.tolist()}).encode()
        ).hexdigest()

    def evaluate(self, structure):
        from ..structures import Evaluation

        if structure.positions.shape != self.reference.shape:
            raise ValueError("Harmonic provider reference must match the evaluated supercell")
        displacement = structure.positions - self.reference
        return Evaluation(0.5 * self.spring * float(np.sum(displacement**2)), -self.spring * displacement)


def resolve_provider(config, structure, provider=None):
    if provider is not None:
        return provider
    options = dict(config.parameters.get("provider", {}))
    kind = options.pop("kind", "mace" if "checkpoint" in config.inputs else "")
    if kind == "harmonic":
        return HarmonicForceProvider(structure, **options)
    if kind == "mace":
        if "checkpoint" not in config.inputs:
            raise ValueError("MACE workflow requires a resolved inputs.checkpoint file")
        from ..adapters.mace import MACEForceProvider

        return MACEForceProvider(checkpoint=config.material_dir / config.inputs["checkpoint"], **options)
    raise ValueError("Force workflows need a force_provider or a configured provider/checkpoint")


def _commit(store, result, **extra):
    from ..io.records import record_units

    arrays = {**result.arrays, **extra.pop("arrays", {})}
    metadata = {**result.artifact_metadata(), **extra}
    metadata["units"] = {**record_units(arrays), **result.units}
    store.commit(
        "result",
        arrays,
        metadata=metadata,
    )


def _record_provider(store, provider):
    store.manifest["scientific_model"].update(
        {
            "provider_fingerprint": getattr(provider, "fingerprint", None),
            "provider_provenance": getattr(provider, "provenance", {"class": type(provider).__qualname__}),
        }
    )
    store.save()


def _checkpoint(store, state):
    # Each state is an immutable case; a killed next update cannot invalidate it.
    current = state.get("current", {})
    step = int(current.get("step", 0))
    case = f"state_{step:012d}"
    if not store.completed(case):
        store.commit(
            case,
            {"state_json": np.asarray(canonical_json(state))},
            metadata={"kind": "restart_state", "step": step},
        )


def _latest_state(store):
    cases = [
        name
        for name, record in store.manifest["cases"].items()
        if record["metadata"].get("kind") == "restart_state"
    ]
    if not cases:
        return None
    name = max(cases)
    store.completed(name)
    with np.load(store.root / store.manifest["cases"][name]["path"], allow_pickle=False) as record:
        return json.loads(str(record["state_json"]))


def calculate_service(config, store, *, provider=None):
    from ..io.records import frames_arrays, structure_arrays

    for case in store.manifest["cases"]:
        store.completed(case)
    if store.completed("result"):
        return
    name, parameters = config.workflow, dict(config.parameters)
    parameters.pop("provider", None)
    structure = None
    if "structure" in config.inputs:
        from ..io.atomistic import read_structure

        structure = read_structure(config.material_dir / config.inputs["structure"])
    if name in {"structure", "nvt", "relax", "phonons"} and structure is None:
        raise ValueError(f"{name} requires a native structure input")
    if name == "structure":
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
    elif name in {"nvt", "relax"}:
        assert structure is not None
        from ..atomistic.dynamics import _expand, relax_fire, run_nvt

        reference, _ = _expand(structure, parameters.get("supercell", (1, 1, 1)))
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
    elif name == "phonons":
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
                    with np.load(
                        store.root / store.manifest["cases"][case]["path"], allow_pickle=False
                    ) as data:
                        force = data["forces"]
                else:
                    evaluation = (
                        evaluator.evaluate(displaced)
                        if hasattr(evaluator, "evaluate")
                        else evaluator(displaced)
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
    elif name == "select":
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
    elif name == "hamiltonian_pack":
        from ..io.hamiltonian import pack_openmx, write_graphs

        arguments = {**parameters}
        for key in ("h0", "openmx_input"):
            if key in config.inputs:
                arguments[key] = config.material_dir / config.inputs[key]
        graph = pack_openmx(config.material_dir / config.inputs["scfout"], **arguments)
        path = write_graphs(store.root / "graphs.npz", {"0": graph})
        _commit(
            store,
            ObservableResult(
                graph.arrays(),
                metadata={
                    **graph.metadata,
                    "summary": {
                        "converged": True,
                        "n_structures": 1,
                        "has_scf_reference": graph.metadata["has_scf_reference"],
                    },
                    "graph_archive": "graphs.npz",
                    "graph_archive_sha256": digest_file(path),
                },
            ),
        )
    elif name == "hamiltonian_bands":
        from ..io.bands import BandData
        from ..io.hamiltonian import read_graphs
        from ..models.graph import GraphHamiltonianModel

        graphs = read_graphs(config.material_dir / config.inputs["graph"])
        key = str(parameters.pop("graph_id", next(iter(graphs))))
        if key not in graphs:
            raise ValueError(f"Graph {key!r} is absent")
        graph = graphs[key]
        prediction_unit = parameters.pop("prediction_unit", "Ha")
        if "predictions" in config.inputs:
            predictions = np.load(config.material_dir / config.inputs["predictions"], allow_pickle=False)
            graph = graph.with_predictions(predictions, unit=prediction_unit)
        electron_num = parameters.pop("electron_num", None)
        filling_grid = parameters.pop("filling_grid", [8, 8, 8])
        temperature = parameters.pop("temperature_kelvin", 0.0)
        model = GraphHamiltonianModel(
            graph,
            fermi_energy=parameters.pop("fermi_energy", 0.0 if electron_num is not None else None),
            hermitian_policy=parameters.pop("hermitian_policy", "validate"),
            execution=config.execution,
        )
        fermi_source = "explicit" if "fermi_energy" in config.parameters else "scf_reference"
        if electron_num is not None:
            if "fermi_energy" in config.parameters:
                raise ValueError("Specify a Fermi energy or electron filling, not both")
            from ..observables._tb import kmesh
            from ..observables.electronic import fermi_energy

            filling = fermi_energy(
                model,
                kmesh(filling_grid),
                electron_num,
                temperature_kelvin=temperature,
                execution=config.execution,
            )
            model.fermi_energy = float(filling.arrays["fermi_energy"])
            fermi_source = "native_electron_filling"
        elif "fermi_energy" not in config.parameters and "fermi_energy_eV" not in graph.metadata:
            raise ValueError("Hamiltonian bands need explicit filling or a declared Fermi energy")
        path = None
        if "kpath" in parameters:
            from ..geometry.paths import named_k_path_from_structure
            from ..structures import Structure

            atom_structure = Structure(graph.cell, graph.positions, graph.atomic_numbers)
            path = named_k_path_from_structure(
                parameters.pop("kpath"), parameters.pop("n_points", 100), atom_structure
            )
            points = path.path
        else:
            points = np.asarray(parameters.pop("kpoints"), float)
        absolute = model.energies(points, execution=config.execution).T[None] + model.fermi_energy
        bands = BandData(
            absolute,
            points,
            fermi_energy=model.fermi_energy,
            labels=() if path is None else path.labels,
            label_indices=() if path is None else path.sym,
            distances=None if path is None else path.distances,
            metadata={
                **graph.metadata,
                "fermi_source": fermi_source,
                "electron_num": electron_num,
                "filling_grid": filling_grid if electron_num is not None else None,
            },
        )
        _commit(store, _band_observable(bands, summary={"converged": True}, source="graph"))
    elif name == "postprocess":
        _commit(store, _postprocess(config))
    elif name == "band_compare":
        from ..io.bands import compare_bands, read_bands

        reference = read_bands(
            config.material_dir / config.inputs["reference"], format=parameters.pop("reference_format", None)
        )
        candidate = read_bands(
            config.material_dir / config.inputs["candidate"], format=parameters.pop("candidate_format", None)
        )
        comparison = compare_bands(reference, candidate, **parameters)
        _commit(
            store,
            ObservableResult(
                {},
                metadata={
                    "summary": comparison,
                    "reference": dict(reference.metadata),
                    "candidate": dict(candidate.metadata),
                },
            ),
        )
    elif name == "dataset":
        from ..atomistic.datasets import export_dataset_splits

        items = json.loads((config.material_dir / config.inputs["dataset"]).read_text())
        if isinstance(items, dict):
            items = items.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError("Dataset source must be a nonempty item list or an items JSON object")
        texts = export_dataset_splits(items, **parameters)
        arrays = {
            filename.removesuffix(".xyz") + "_extxyz": np.asarray(text) for filename, text in texts.items()
        }
        _commit(
            store,
            ObservableResult(
                arrays,
                metadata={
                    "summary": {"converged": True, "n_items": len(items), "seed": parameters.get("seed", 42)},
                    "stress_unit": "eV/Å³",
                    "stress_sign_input": parameters.get("stress_sign", 1.0),
                },
            ),
        )
    else:
        raise ValueError(f"Unknown service workflow {name}")


def _band_observable(bands, *, summary=None, source=None):
    result = bands.as_result()
    return ObservableResult(
        result.arrays,
        result.dims,
        result.units,
        metadata={
            **result.metadata,
            "labels": list(bands.labels),
            "label_indices": list(map(int, bands.label_indices)),
            "segments": list(bands.segments),
            "summary": summary or {},
            "source": source,
        },
    )


def _postprocess(config):
    from ..io import bands as readers
    from ..io import calculations

    parameters = config.parameters
    engine = parameters.get("engine", "vasp")
    available = _postprocess_inputs(config)
    source = available.get("source")
    summary, banddata, arrays, dims, units = {}, None, {}, {}, {}

    def consume(parsed):
        nonlocal banddata
        result = parsed.as_result()
        arrays.update(result.arrays)
        dims.update(result.dims)
        units.update(result.units)
        summary.update(parsed.summary)
        if parsed.bands is not None:
            banddata = parsed.bands

    if engine == "vasp":
        if "outcar" in available or (source and source.name.upper() == "OUTCAR"):
            consume(calculations.read_vasp_results(available.get("outcar", source)))
        if "vasprun" in available or (source and source.suffix == ".xml"):
            consume(calculations.read_vasprun_results(available.get("vasprun", source)))
        if banddata is None and ("eigenval" in available or (source and source.name.upper() == "EIGENVAL")):
            banddata = readers.read_eigenval(
                available.get("eigenval", source), fermi_energy=parameters.get("fermi_energy")
            )
    elif engine == "openmx":
        if "output" in available or (source and source.suffix.lower() in {".out", ".log"}):
            consume(calculations.read_openmx_results(available.get("output", source)))
        if "bandfile" in available or (source and source.suffix.lower() == ".band"):
            banddata = readers.read_openmx_bands(available.get("bandfile", source))
    elif engine == "wannier":
        if "wout" in available:
            summary["wannier"] = readers.read_wannier_report(available["wout"])
        if "bandfile" in available or (source and source.suffix.lower() != ".wout"):
            banddata = readers.read_wannier_bands(available.get("bandfile", source))
        elif source:
            summary["wannier"] = readers.read_wannier_report(source)
    else:
        raise ValueError("postprocess engine must be vasp, openmx, or wannier")
    if banddata is not None:
        if "kpoints_file" in available:
            path = readers.read_vasp_kpoints(available["kpoints_file"])
            if path["coordinate_mode"] != "fractional_reciprocal":
                raise ValueError("Band path matching requires fractional reciprocal KPOINTS")
            if banddata.kpoints is None or not np.allclose(
                path["kpoints"], banddata.kpoints, atol=1e-6, rtol=0
            ):
                raise ValueError("KPOINTS path does not match imported band rows")
            banddata = replace(
                banddata,
                labels=path["labels"],
                label_indices=path["label_indices"],
                segments=path["segments"],
            )
        if parameters.get("fermi_energy") is not None:
            banddata = replace(
                banddata,
                fermi_energy=float(parameters["fermi_energy"]),
                metadata={**banddata.metadata, "fermi_source": parameters.get("fermi_source", "explicit")},
            )
        elif banddata.fermi_energy is not None:
            banddata = replace(
                banddata,
                metadata={
                    **banddata.metadata,
                    "fermi_source": banddata.metadata.get(
                        "fermi_source", banddata.metadata.get("source_format", engine)
                    ),
                },
            )
        from ..capabilities import CapabilityError

        try:
            summary["bandgap"] = readers.band_gap(banddata)
        except CapabilityError as error:
            summary["bandgap"] = None
            summary["bandgap_unavailable_reason"] = str(error)
        result = _band_observable(banddata, summary=summary, source=engine)
        return ObservableResult(
            {**arrays, **result.arrays}, {**dims, **result.dims}, {**units, **result.units}, result.metadata
        )
    if not summary:
        raise ValueError("No supported calculation results found")
    return ObservableResult(arrays, dims, units, metadata={"summary": summary, "source": engine})
