"""Shared electronic model discovery and loading at the workflow boundary."""

from pathlib import Path

_WANNIER_FILES = ("tb", "centres", "wout", "poscar", "wsvec", "win")


def model_input_files(inputs, parameters):
    directory = Path(inputs["material_dir"])
    if inputs.get("model") == "openmx":
        files = [directory / inputs["scfout"]] if "scfout" in inputs else sorted(directory.glob("*.scfout"))
        if "scfout" not in inputs and len(files) != 1:
            raise ValueError("Provide inputs.scfout when there is not exactly one SCFOUT file")
        files.extend(directory / inputs[key] for key in ("openmx_input", "poscar") if key in inputs)
        return files
    from ..io.wannier import resolve_wannier_files

    return list(
        resolve_wannier_files(
            directory,
            hr=inputs.get("hr", "wannier90_hr.dat"),
            interpolation=parameters.get("interpolation", "standard"),
            position=parameters.get("position", "auto"),
            **{key: inputs[key] for key in _WANNIER_FILES if key in inputs},
        ).values()
    )


def load_electronic_model(config, *, default_fermi=False):
    inputs, parameters = config.inputs, config.parameters
    if inputs.get("model") == "openmx":
        from ..models.openmx import OpenMXModel

        model = OpenMXModel.from_directory(
            config.material_dir,
            scfout=inputs.get("scfout"),
            ef=parameters.get("ef"),
            openmx_input=inputs.get("openmx_input"),
            execution=config.execution,
        )
        if inputs.get("poscar"):
            model.poscar = config.material_dir / inputs["poscar"]
        return model
    from ..models.wannier import WannierModel

    return WannierModel.from_directory(
        config.material_dir,
        hr=inputs.get("hr", "wannier90_hr.dat"),
        ef=parameters.get("ef", 0.0) if default_fermi else parameters["ef"],
        interpolation=parameters.get("interpolation", "standard"),
        position=parameters.get("position", "auto"),
        **{key: inputs[key] for key in _WANNIER_FILES if key in inputs},
    )
