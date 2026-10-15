"""Lightweight declarations and validation for supported scientific workflows."""

import inspect
from dataclasses import dataclass
from importlib import import_module

from .finite_field_specs import FINITE_FIELD_NAMES
from .reduction import ADDITIVE as _INTEGRALS
from .reduction import REDUCTIONS, ReductionSpec, calculation_policy
from .service_specs import SERVICE_NAMES


@dataclass(frozen=True)
class WorkflowSpec:
    module: str
    function: str
    capabilities: tuple[str, ...] = ()
    reduction: ReductionSpec | None = None

    def calculator(self):
        return getattr(import_module(self.module, "queer"), self.function)

    def policy(self, parameters):
        return calculation_policy(self.function, parameters)

    def parameter_names(self):
        names = set(inspect.signature(self.calculator()).parameters) - {"model", "execution", "checkpoint"}
        return (names - {"radial_data", "pao_dir"}) | {"ef", "grid", "kpoints", "integration"}


REGISTRY: dict[str, WorkflowSpec] = {}
for _module, _names in {
    ".observables.electronic": (
        "fat_band",
        "dos",
        "pdos",
        "jdos",
        "fermi_energy",
        "fermi_surface",
        "find_nodes",
        "spin_texture",
        "reduce_basis",
    ),
    ".observables.surface": ("surface_state",),
    ".observables.unfolding": ("bandunfolding", "bandunfolding_spin_texture"),
    ".observables.geometry": (
        "berry_curvature",
        "quantum_metric",
        "band_pair_geometry",
        "chern_number",
        "chirality",
        "wilson_loop",
        "polarization",
    ),
    ".observables.integrated": (
        "ahc",
        "anc",
        "berry_curvature_dipole",
        "orbital_magnetization",
        "shc",
        "drude_weight",
    ),
    ".observables.transport": ("boltz_transport",),
    ".observables.optical": ("optical_conductivity",),
    ".observables.nonlinear": ("shift_current", "cpge", "shg", "pockels", "second_order_static"),
}.items():
    for _name in _names:
        _capabilities: tuple[str, ...] = ()
        if _module.endswith(("geometry", "integrated", "optical", "nonlinear", "transport")):
            _capabilities = ("position",)
        if _name in {"bandunfolding", "bandunfolding_spin_texture"}:
            _capabilities = ("orbital_metadata",)
        REGISTRY[_name] = WorkflowSpec(_module, _name, _capabilities, REDUCTIONS.get(_name))

NAMES = tuple(REGISTRY)


def validate_parameters(name, parameters):
    spec = REGISTRY[name]
    unexpected = set(parameters) - spec.parameter_names()
    if unexpected:
        raise ValueError(f"Unknown {name} parameters: {', '.join(sorted(unexpected))}")
    signature = inspect.signature(spec.calculator())
    for key, value in signature.parameters.items():
        if key in {"model", "execution", "checkpoint"} or value.default is not inspect.Parameter.empty:
            continue
        if key == "kpoints" and "grid" in parameters:
            continue
        if key not in parameters:
            raise ValueError(f"Missing {name} parameter: {key}")
    if "grid" in parameters:
        from ..observables._tb import kmesh

        kmesh(parameters["grid"])
    if "integration" in parameters:
        options = parameters["integration"]
        if name not in _INTEGRALS | {"chern_number"} or not isinstance(options, dict):
            raise ValueError("integration controls are supported for BZ integrals only")
        from ..observables.integration import validate_integration

        validate_integration(options)
        if options.get("method") in {"threshold", "cubature"} and name not in {
            "ahc",
            "chern_number",
            "shift_current",
        }:
            raise ValueError("local integration is supported for ahc, chern_number, and shift_current")


WORKFLOWS = (
    "bands",
    "susceptibility",
    "spectral",
    "momentum",
    "epw",
    *NAMES,
    *SERVICE_NAMES,
    *FINITE_FIELD_NAMES,
)


_RENDERERS = {name: "queer.plotting.equilibrium_artifacts:render" for name in NAMES}
_RENDERERS.update({name: "queer.plotting.service_artifacts:render" for name in SERVICE_NAMES})
_RENDERERS.update({name: "queer.plotting.finite_field_artifacts:render" for name in FINITE_FIELD_NAMES})
_RENDERERS.update(
    {
        name: "queer.plotting.legacy_artifacts:render"
        for name in ("bands", "susceptibility", "epw", "spectral", "momentum")
    }
)


def renderer_reference(name):
    """Return a lazy renderer reference; historical readers need no calculators."""
    return _RENDERERS.get(name, "queer.plotting.equilibrium_artifacts:render")


def renderer_for(name):
    """Resolve renderers without importing calculators or validating old configs."""
    module, function = renderer_reference(name).split(":")
    return getattr(import_module(module), function)
