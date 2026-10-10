"""Physical constants (CODATA 2018 exact SI values) and derived quantities."""

import math

E_CHARGE = 1.602176634e-19  # C
H_PLANCK = 6.62607015e-34  # J s
HBAR = H_PLANCK / (2.0 * math.pi)  # J s
HBAR_EVS = HBAR / E_CHARGE  # eV s  (6.582119569e-16)
E2_H = E_CHARGE**2 / H_PLANCK  # S    (3.874045865e-5), conductance quantum e^2/h
ANGSTROM = 1e-10  # m
MEV = 1e-3  # eV
PS = 1e-12  # s
K_BOLTZMANN_EV = 8.617333262e-5  # eV/K
