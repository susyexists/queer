"""Scientific service calculation handlers with workflow-owned persistence."""

import json

import numpy as np

from ..io.artifacts import digest_file
from ..observables.result import ObservableResult
from .service_bands import _band_observable, _postprocess
from .service_state import _commit


def calculate_hamiltonian_pack(config, store, *, provider=None, structure=None, parameters):
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


def calculate_hamiltonian_bands(config, store, *, provider=None, structure=None, parameters):
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
        labels=() if path is None else tuple(path.labels),
        label_indices=() if path is None else tuple(map(int, path.sym)),
        distances=None if path is None else path.distances,
        metadata={
            **graph.metadata,
            "fermi_source": fermi_source,
            "electron_num": electron_num,
            "filling_grid": filling_grid if electron_num is not None else None,
        },
    )
    _commit(store, _band_observable(bands, summary={"converged": True}, source="graph"))


def calculate_postprocess(config, store, *, provider=None, structure=None, parameters):
    _commit(store, _postprocess(config))


def calculate_band_compare(config, store, *, provider=None, structure=None, parameters):
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


def calculate_dataset(config, store, *, provider=None, structure=None, parameters):
    from ..atomistic.datasets import export_dataset_splits

    items = json.loads((config.material_dir / config.inputs["dataset"]).read_text())
    if isinstance(items, dict):
        items = items.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Dataset source must be a nonempty item list or an items JSON object")
    texts = export_dataset_splits(items, **parameters)
    arrays = {filename.removesuffix(".xyz") + "_extxyz": np.asarray(text) for filename, text in texts.items()}
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
