from .utils import *
from . import model
from .epw import epw
from . import epc
from . import reads

# Re-export key classes and functions for easier imports
from .model import (
    model as queer,  # Main class renamed from model to queer
    GMKG,
    mesh_cartesian,
    mesh_crystal,
    plot_electron_mesh,
    density_of_states
)

__version__ = '1.0.0'
__author__ = 'QUEER Development Team'
__all__ = [
    'queer',
    'epw',
    'epc',
    'GMKG',
    'mesh_cartesian',
    'mesh_crystal',
    'plot_electron_mesh',
    'density_of_states'
]