"""Readers for external scientific file formats; imports are explicit."""

from importlib import import_module

_EXPORTS = {
    "read_scfout": (".openmx", "read_scfout"),
    "OpenMXData": (".openmx", "OpenMXData"),
    "read_pao": (".pao", "read_pao"),
    "PAORadialData": (".pao", "PAORadialData"),
    "read_structure": (".atomistic", "read_structure"),
    "write_structure": (".atomistic", "write_structure"),
    "read_frames": (".atomistic", "read_frames"),
    "read_trajectory": (".atomistic", "read_trajectory"),
    "write_trajectory": (".atomistic", "write_trajectory"),
    "BandData": (".bands", "BandData"),
    "read_bands": (".bands", "read_bands"),
    "band_gap": (".bands", "band_gap"),
    "compare_bands": (".bands", "compare_bands"),
    "CalculationResult": (".calculations", "CalculationResult"),
    "read_vasp_results": (".calculations", "read_vasp_results"),
    "read_vasprun_results": (".calculations", "read_vasprun_results"),
    "read_openmx_results": (".calculations", "read_openmx_results"),
    "HamiltonianGraph": (".hamiltonian", "HamiltonianGraph"),
    "pack_openmx": (".hamiltonian", "pack_openmx"),
    "read_graphs": (".hamiltonian", "read_graphs"),
    "write_graphs": (".hamiltonian", "write_graphs"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value
