"""Shared units, retaining named historical values where contracts differ."""

import math

E_CHARGE = 1.602176634e-19
H_PLANCK = 6.62607015e-34
HBAR = H_PLANCK / (2.0 * math.pi)
HBAR_EVS = HBAR / E_CHARGE
E2_H = E_CHARGE**2 / H_PLANCK
KB_J_KELVIN = 1.380649e-23
KB_EV = 8.617333262145e-5
# These values intentionally reproduce the pre-refactor domain conventions.
K_BOLTZMANN_EV = 8.617333262e-5
WANNIER_BOHR_ANGSTROM = 0.529177210903
KBAR_EV_ANGSTROM3 = 1e8 * 1e-30 / E_CHARGE
DATASET_KBAR_EV_ANGSTROM3 = 1e8 / E_CHARGE * 1e-30
ASE_FS = 0.09822694788464063
KINETIC_EV_ANGSTROM2 = 3.80998211615486
ARPES_FACTOR = 0.2625
ANGSTROM = 1e-10
MEV = 1e-3
PS = 1e-12
V_PER_M_TO_EV_PER_A = 1e-10
EV_PER_A_TO_V_PER_M = 1e10

_CODATA = {
    "KB_EV_K": ("Boltzmann constant in eV/K", 1.0),
    "HARTREE_EV": ("Hartree energy in eV", 1.0),
    "BOHR_ANGSTROM": ("Bohr radius", 1e10),
}


def __getattr__(name: str) -> float:
    """Load measured CODATA values only for callers that actually need them.

    Ordinary Wannier interpolation uses historical fixed values and must not pay
    the import/memory cost of SciPy's full constant table.
    """
    if name not in _CODATA and name != "ACCELERATION":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from scipy.constants import atomic_mass, physical_constants

    if name == "ACCELERATION":
        value = E_CHARGE / atomic_mass * 1e-10
    else:
        key, scale = _CODATA[name]
        value = physical_constants[key][0] * scale
    globals()[name] = value
    return value
