"""Scientific service dispatch and supported compatibility entrypoints."""

from importlib import import_module

from .service_bands import _band_observable as _band_observable
from .service_bands import _postprocess as _postprocess
from .service_sources import _postprocess_inputs as _postprocess_inputs
from .service_sources import service_input_files as service_input_files
from .service_specs import SERVICE_INPUTS as SERVICE_INPUTS
from .service_specs import SERVICE_NAMES as SERVICE_NAMES
from .service_specs import SERVICE_SPECS as SERVICE_SPECS
from .service_specs import ServiceSpec as ServiceSpec
from .service_specs import validate_service as validate_service
from .service_state import HarmonicForceProvider as HarmonicForceProvider
from .service_state import _checkpoint as _checkpoint
from .service_state import _commit as _commit
from .service_state import _latest_state as _latest_state
from .service_state import _record_provider as _record_provider
from .service_state import resolve_provider as resolve_provider

_HANDLERS = {
    "structure": ("atomistic", "structure"),
    "nvt": ("atomistic", "dynamics"),
    "relax": ("atomistic", "dynamics"),
    "phonons": ("atomistic", "phonons"),
    "select": ("atomistic", "select"),
    "hamiltonian_pack": ("electronic", "hamiltonian_pack"),
    "hamiltonian_bands": ("electronic", "hamiltonian_bands"),
    "postprocess": ("electronic", "postprocess"),
    "band_compare": ("electronic", "band_compare"),
    "dataset": ("electronic", "dataset"),
}


def calculate_service(config, store, *, provider=None):
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
    if name not in _HANDLERS:
        raise ValueError(f"Unknown service workflow {name}")
    module, handler = _HANDLERS[name]
    calculate = getattr(import_module(f".service_{module}", __package__), f"calculate_{handler}")
    calculate(config, store, provider=provider, structure=structure, parameters=parameters)
