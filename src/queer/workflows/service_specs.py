"""Declared configuration and products of scientific service workflows."""

from dataclasses import dataclass, replace


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
