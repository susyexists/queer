"""Native nonperturbative quantum-geometry Hall response and analysis.

HHH's paper settings are the default; physical corrections are explicit profiles.
"""

from .api import (
    PreparedHallResponse,
    berry_flux,
    critical_hall,
    crossing_scan,
    decode_line,
    encode_line,
    fit_hall,
    hall_harmonics,
    houston_line,
    landau_zener_hall,
    lindblad_line,
)
from .config import PAPER, PAPER_FIGURES, PHYSICAL, PHYSICAL_EQ8, FiniteFieldOptions, preset, resolve_options
from .family import ModelFamily
from .geometry.gridded import GriddedGeometry
from .geometry.kubo import BandGeometry
from .geometry.provider import ModelGeometry
from .models import FourBandModel, FourBandParams, lattice_regularized_fourband
from .native import FieldPlane, NativePlaneModel

__all__ = [
    "BandGeometry",
    "FieldPlane",
    "FiniteFieldOptions",
    "FourBandModel",
    "FourBandParams",
    "GriddedGeometry",
    "ModelFamily",
    "ModelGeometry",
    "NativePlaneModel",
    "PAPER",
    "PAPER_FIGURES",
    "PHYSICAL",
    "PHYSICAL_EQ8",
    "PreparedHallResponse",
    "berry_flux",
    "critical_hall",
    "crossing_scan",
    "decode_line",
    "encode_line",
    "fit_hall",
    "hall_harmonics",
    "houston_line",
    "landau_zener_hall",
    "lattice_regularized_fourband",
    "lindblad_line",
    "preset",
    "resolve_options",
]
