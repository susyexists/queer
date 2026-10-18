"""PyATB's unit, Hall-sign and grid conventions, as the Generator's OpenMX steps meet them.

The facts here were measured against PyATB itself (the build the Generator runs) on
an OpenMX scfout, and they hold for every material:

- **Energies.** PyATB converts the scfout's Hartree values to eV with
  :data:`PYATB_HARTREE_EV`; QUEER uses CODATA's ``HARTREE_EV``. A PyATB energy
  ``E`` (eV) therefore sits at ``E / PYATB_HARTREE_EV`` Hartree, and QUEER samples
  the same point at ``E * HARTREE_EV / PYATB_HARTREE_EV`` eV
  (:func:`queer_energy`). ``fermi_energy Auto`` is the scfout's ``ChemP`` in both
  codes. An explicit Rydberg value is first converted with PyATB's own
  :data:`PYATB_RYDBERG_EV`.
- **Hall sign.** QUEER's Hall conductivity is the physical
  ``sigma_ab = -(e^2/hbar) int f Omega_ab``. PyATB's direct/Kubo engine natively
  prints its negative; ``hall_convention openmx`` (also ``fhs``, and ``auto`` for an
  OpenMX scfout) flips it back, and the FHS engine ignores the key and prints
  the physical sign. :func:`hall_export_sign` is the factor taking QUEER's value
  to what PyATB prints. Chern numbers carry no Hall sign: PyATB and QUEER orient
  the plane flux alike.
- **Grid integration.** ``integrate_mode Grid`` samples the Gamma-origin nodes
  ``k_start + sum_i v_i n_i / N_i`` (upper boundary excluded). A node whose
  occupied-summed curvature has norm at least ``adaptive_grid_threshold`` (Å², the
  raw curvature before the conductivity prefactor and the Hall sign; for a Chern
  number its component along the plane normal) is replaced by an
  ``adaptive_grid`` sub-grid that starts half a coarse cell below the node, so a
  1 x 1 x 1 sub-grid still moves a flagged node. Only a threshold that no
  curvature reaches switches the replacement off: PyATB's own convention for
  that is 1e30 (:data:`REFINEMENT_OFF_THRESHOLD`), and the sum is then the plain
  Gamma-origin grid sum.
- **Shift current.** PyATB's prefactor carries its own SI constants
  (:data:`PYATB_SHIFT_E_CHARGE`, :data:`PYATB_SHIFT_HBAR`,
  :data:`PYATB_SHIFT_EV_SECONDS`) and the signed cell volume, so a left-handed
  lattice flips every component it prints, and it multiplies by 2 when it reads
  one spin channel (``nspin 1``: the scfout has no spin-orbit coupling) and by 1
  for ``nspin 4`` (:func:`pyatb_shift_spin_factor`). Its photon energies, Gaussian
  width and bins are numbers on its own Hartree scale.
"""

from __future__ import annotations

import math
from typing import Any

from ...physics.constants import BOHR_ANGSTROM as _CODATA_BOHR_ANGSTROM
from ...physics.constants import E_CHARGE as _CODATA_E_CHARGE
from ...physics.constants import HARTREE_EV as _CODATA_HARTREE_EV
from ...physics.constants import HBAR as _CODATA_HBAR

# PyATB's OpenMX reader: Hartree -> eV and Bohr -> Angstrom.
PYATB_HARTREE_EV = 27.211324570273
PYATB_BOHR_ANGSTROM = 0.52917721
# PyATB's charge and reduced Planck constant (SI) in its direct/Kubo prefactor, its
# e^2/h (S) in the FHS engine, and the threshold below which its direct engine
# zeroes the coefficient derivative of a same-channel pair (eV).
PYATB_E_CHARGE = 1.60217662e-19
PYATB_HBAR = 1.054571628e-34
PYATB_E2_H = 3.874046e-5
PYATB_DEGENERATE_EV = 1e-4
# PyATB's shift-current prefactor (eV*s, C, J*s).
PYATB_SHIFT_EV_SECONDS = 6.582119e-16
PYATB_SHIFT_E_CHARGE = 1.602176565e-19
PYATB_SHIFT_HBAR = 1.054571726e-34
# PyATB's Rydberg -> eV for an explicit ``fermi_energy_unit Ry``.
PYATB_RYDBERG_EV = 13.605698066
# QUEER's Hartree -> eV (CODATA), the scale of every QUEER energy.
HARTREE_EV = float(_CODATA_HARTREE_EV)

HALL_CONVENTIONS: tuple[str, ...] = ("auto", "pyatb", "openmx", "fhs")
# A PyATB Grid threshold at or above this never flags a node: refinement off.
REFINEMENT_OFF_THRESHOLD = 1e30
# The E_F agreement rule: the two codes' Fermi energies, each in Hartree on its
# own scale, may differ by the six decimals PyATB prints (1e-6 eV of its scale).
FERMI_TOLERANCE_HARTREE = 1e-6 / PYATB_HARTREE_EV


def pyatb_length_ratio() -> float:
    """PyATB's Angstrom per QUEER's: both read the scfout in Bohr, with different Bohr radii."""
    return PYATB_BOHR_ANGSTROM / float(_CODATA_BOHR_ANGSTROM)


def queer_energy(pyatb_energy_eV: float) -> float:
    """QUEER's eV value of the point a PyATB energy (eV, PyATB's Hartree scale) samples."""
    return float(pyatb_energy_eV) * HARTREE_EV / PYATB_HARTREE_EV


def pyatb_energy(queer_energy_eV: float) -> float:
    """PyATB's eV value of the point a QUEER energy (eV, CODATA's Hartree scale) samples."""
    return float(queer_energy_eV) * PYATB_HARTREE_EV / HARTREE_EV


def pyatb_input_fermi_energy(value: Any, unit: Any = "eV") -> float | None:
    """The eV value PyATB uses for an INPUT_PARAMETERS ``fermi_energy``; None for ``Auto``.

    ``Ry`` values are converted with PyATB's :data:`PYATB_RYDBERG_EV`. Raises
    ``ValueError`` for another unit or a value that is not a finite number.
    """
    if value is None or str(value).strip().lower() == "auto":
        return None
    unit_name = str(unit if unit is not None else "eV").strip()
    if unit_name not in {"eV", "Ry"}:
        raise ValueError("fermi_energy_unit must be eV or Ry")
    if isinstance(value, bool):
        raise ValueError("fermi_energy must be Auto or a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("fermi_energy must be finite")
    return number * PYATB_RYDBERG_EV if unit_name == "Ry" else number


def fermi_difference_hartree(queer_eV: float, pyatb_eV: float) -> float:
    """|E_F^Q / HARTREE_EV - E_F^P / PYATB_HARTREE_EV|: the two Fermi energies compared in Hartree."""
    return abs(float(queer_eV) / HARTREE_EV - float(pyatb_eV) / PYATB_HARTREE_EV)


def resolve_hall_convention(value: Any, package: Any = "OPENMX") -> str:
    """PyATB's resolved ``hall_convention``: ``auto`` (the default) becomes ``openmx`` for OPENMX input.

    Every other package resolves ``auto`` to ``pyatb``. Raises ``ValueError`` for
    a value PyATB does not accept.
    """
    token = "auto" if value is None else str(value).strip().lower()
    if token not in HALL_CONVENTIONS:
        raise ValueError(f"hall_convention must be one of {', '.join(HALL_CONVENTIONS)}")
    if token == "auto":
        return "openmx" if str(package or "OPENMX").strip().upper() == "OPENMX" else "pyatb"
    return token


def hall_export_sign(convention: Any, engine: Any = "kubo", package: Any = "OPENMX") -> int:
    """The factor taking QUEER's physical Hall conductivity to what PyATB prints.

    -1 for the ``pyatb`` convention of the direct/Kubo engine; +1 for ``openmx``,
    ``fhs`` and ``auto`` (OpenMX input), and for the FHS engine, which ignores the
    convention.
    """
    if str(engine or "kubo").strip().lower() == "fhs":
        return 1
    return -1 if resolve_hall_convention(convention, package) == "pyatb" else 1


def pyatb_shift_spin_factor(nspin: Any) -> int:
    """PyATB's shift-current spin factor: 2 for ``nspin 1`` (one channel of the scfout read), 1 for 4."""
    value = int(nspin)
    if value not in (1, 4):
        raise ValueError("an OpenMX shift current has nspin 1 or 4")
    return 2 if value == 1 else 1


def pyatb_shift_constant_ratio() -> float:
    """PyATB's shift-current prefactor over QUEER's for the same transitions and cell.

    PyATB prints ``-(pi t e^3 / (2 hbar^2 V)) sum I`` with ``t`` its eV*s and an
    integrand ``I`` twice QUEER's (it sums both orders of the optical indices),
    QUEER ``-(pi e^2 / (hbar V)) sum I`` with CODATA's constants; the ratio is
    ``(e_P^2 / hbar_P) (e_P t_P / hbar_P) / (e^2 / hbar)``, 1 to about 1e-7.
    """
    pyatb = (
        PYATB_SHIFT_E_CHARGE**2
        / PYATB_SHIFT_HBAR
        * (PYATB_SHIFT_E_CHARGE * PYATB_SHIFT_EV_SECONDS / PYATB_SHIFT_HBAR)
    )
    return pyatb / (float(_CODATA_E_CHARGE) ** 2 / float(_CODATA_HBAR))


def refinement_off(threshold: Any) -> bool:
    """Whether a PyATB Grid ``adaptive_grid_threshold`` (Å²) can flag no node."""
    value = float(threshold)
    if math.isnan(value):
        raise ValueError("adaptive_grid_threshold must be a number")
    return value >= REFINEMENT_OFF_THRESHOLD
