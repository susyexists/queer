"""Small analytic Generator Wannier jobs: their staged files and the parameter file of one step.

Each case stages what a Generator WANNIER job stages for a PythTB step (the
``.win`` with ``mp_grid`` and the disentanglement windows, ``_hr.dat``,
``_centres.xyz`` and the band files of a path G-M-K-G) and the step's parameter
file with the submitter's context, as the Generator writes them. Every number is
analytic and written with fixed formatting; the band energies are rounded to 1e-8
eV before they are written, so the files, and the SHA-256 digests a run records,
are the same on every machine. ``scripts/validation/generate_generator_goldens.py``
runs the Generator driver on these cases; ``test_generator_goldens.py`` and
``test_generator_parity.py`` run QUEER on them.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tight_binding_models import haldane, kane_mele
from wannier90_files import Wannier90Job, line_path, write_band_files

PARAMS_FILENAME = "pythtb_params.json"
CORNERS = [[0, 0, 0], [0.5, 0, 0], [1 / 3, 2 / 3, 0], [0, 0, 0]]
LABELS = ["G", "M", "K", "G"]
STEPS = 6
MP_GRID = (4, 4, 1)
WIN = {"dis_win_min": -6.0, "dis_win_max": 6.0, "dis_froz_min": -1.0, "dis_froz_max": 1.5}
CONTEXT = {
    "workers": 1,
    "ws_correction": "auto",
    "fermi_source": "reference_band",
    "source_wannier_job_id": "job-wannier",
    "reference_band_job_id": "job-band",
}
ENERGY_DECIMALS = 8
MODELS = {"haldane": lambda: haldane(staggered=0.2), "kane_mele": lambda: kane_mele(1.0)}


@dataclass(frozen=True)
class Case:
    step: str
    model: str
    settings: dict = field(default_factory=dict)


CASES = {
    "haldane_bands": Case(
        "PYTHTB_BANDS",
        "haldane",
        {
            "fermi_energy": 0.8,
            "dos_mesh": 8,
            "dos_step_eV": 0.02,
            "dos_sigma_eV": 0.06,
            "fermi_surface_mesh": 24,
        },
    ),
    "haldane_topology": Case(
        "PYTHTB_TOPOLOGY", "haldane", {"fermi_energy": 0.0, "loop_points": 21, "transverse_points": 12}
    ),
    "haldane_berry": Case("PYTHTB_BERRY", "haldane", {"fermi_energy": 0.0, "berry_mesh": 12}),
    "haldane_slab": Case(
        "PYTHTB_SLAB", "haldane", {"fermi_energy": 0.0, "slab_kpoints": 24, "slab_cells": 6}
    ),
    "kane_mele_topology": Case(
        "PYTHTB_TOPOLOGY",
        "kane_mele",
        {"fermi_energy": 0.0, "spinors": True, "loop_points": 21, "transverse_points": 12},
    ),
    "haldane_berry_fermi": Case("PYTHTB_BERRY", "haldane", {"fermi_energy": 0.8, "berry_mesh": 12}),
}


def band_energies(translations, table, kpoints):
    """Absolute band energies (Nk, NB) of the table H(R) at fractional ``kpoints``, rounded to 1e-8 eV."""
    phases = np.exp(2j * np.pi * (np.asarray(kpoints, float) @ np.asarray(translations, float).T))
    hamiltonians = np.einsum("kr,rij->kij", phases, np.asarray(table, complex))
    hamiltonians = 0.5 * (hamiltonians + np.conj(np.swapaxes(hamiltonians, 1, 2)))
    return np.round(np.linalg.eigvalsh(hamiltonians), ENERGY_DECIMALS)


def parameters(name, **overrides):
    """The parameter file values of a case, with ``overrides``."""
    case = CASES[name]
    model = MODELS[case.model]()
    sites = np.unique(np.round(np.asarray(model.centres, float), 10), axis=0)
    species = ["B", "N"][: len(sites)]
    values = {"step_type": case.step, **CONTEXT, **case.settings}
    values["structure"] = {"species": species, "cart": sites.tolist()}
    values.update(overrides)
    return values


def write_case(name, directory, **overrides):
    """Stage case ``name`` in ``directory``; returns the parameter file's path."""
    directory = Path(directory)
    model = MODELS[CASES[name].model]()
    job = Wannier90Job(
        model.translations, model.hopping_matrices, model.lattice, model.centres, mp_grid=MP_GRID, win=WIN
    )
    job.write(directory)
    kpoints = line_path(CORNERS, STEPS)
    ticks = [(label, index * STEPS) for index, label in enumerate(LABELS)]
    write_band_files(directory, kpoints, band_energies(*job.driver_table(), kpoints), ticks)
    path = directory / PARAMS_FILENAME
    path.write_text(json.dumps(parameters(name, **overrides), indent=1, sort_keys=True) + "\n")
    return path
