"""Input inventories for native service calculations."""

from .service_specs import SERVICE_SPECS


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
