"""Array-based observables. Plotting is available separately in queer.plotting."""

from importlib import import_module

_EXPORTS = {"ObservableResult": (".result", "ObservableResult")}
for _module, _names in {
    "electronic": (
        "band_structure",
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
    "surface": ("surface_state",),
    "topology": ("wilson_plane",),
    "unfolding": ("bandunfolding", "bandunfolding_spin_texture"),
    "geometry": (
        "berry_curvature",
        "quantum_metric",
        "band_pair_geometry",
        "chern_number",
        "chirality",
        "wilson_loop",
        "polarization",
    ),
    "integrated": ("ahc", "anc", "berry_curvature_dipole", "orbital_magnetization", "shc", "drude_weight"),
    "transport": ("boltz_transport",),
    "optical": ("optical_conductivity", "dielectric_function"),
    "nonlinear": ("shift_current", "cpge", "shg", "pockels", "second_order_static"),
}.items():
    for _name in _names:
        _EXPORTS[_name] = (f".{_module}", _name)

for _name in (
    "crossing_scan",
    "landau_zener_hall",
    "hall_harmonics",
    "critical_hall",
    "fit_hall",
    "houston_line",
    "berry_flux",
    "lindblad_line",
):
    _EXPORTS[_name] = ("..finite_field.api", _name)

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
