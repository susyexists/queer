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
"""

from __future__ import annotations

import math
from typing import Any

from ...physics.constants import HARTREE_EV as _CODATA_HARTREE_EV

# PyATB's OpenMX reader: Hartree -> eV and Bohr -> Angstrom.
PYATB_HARTREE_EV = 27.211324570273
PYATB_BOHR_ANGSTROM = 0.52917721
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


def refinement_off(threshold: Any) -> bool:
    """Whether a PyATB Grid ``adaptive_grid_threshold`` (Å²) can flag no node."""
    value = float(threshold)
    if math.isnan(value):
        raise ValueError("adaptive_grid_threshold must be a number")
    return value >= REFINEMENT_OFF_THRESHOLD
