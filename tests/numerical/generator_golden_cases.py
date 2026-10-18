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

The quantum-geometry cases stage a generic three-band layer without exact
symmetries (no velocity vanishes by symmetry and no two cells' error estimates tie,
so the adaptive mesh does not depend on rounding), at its full position level
(``wannier90_tb.dat`` as wannier90 3.1 writes it) or at its diagonal level, with a
``.win`` ``kpoints`` block and the DFT ``.eig`` of the frozen states.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from qgeom_models import HEXAGONAL, layer, position_table
from tight_binding_models import haldane, kane_mele
from wannier90_files import Wannier90Job, exact, line_path, write_band_files
from wannier90_tb import write_tb

from queer.models.wannier import WannierModel

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
LAYER_SEED = 20261018


def generic_layer():
    """A three-band layer on a hexagonal cell with random complex hops (fixed seed) and off-site centres."""
    R, H, lattice, centres = layer(np.random.default_rng(LAYER_SEED), bands=3, lattice=HEXAGONAL, decay=1.3)
    return WannierModel(R, H, lattice=lattice, centres=centres)


MODELS = {
    "haldane": lambda: haldane(staggered=0.2),
    "kane_mele": lambda: kane_mele(1.0),
    "layer3": generic_layer,
}
# The quantum-geometry settings of every QGEOM case: small meshes, the chunking fixed so
# both producers batch their k-points alike.
QGEOM_SETTINGS = {
    "mu_range_meV": [-150.0, 150.0],
    "mu_step_meV": 10.0,
    "temperatures_K": [5.0, 20.0, 60.0],
    "qgeom_base_mesh": 16,
    "qgeom_max_points": 10000,
    "qgeom_tolerance": 0.05,
    "occupation_mesh": 41,
    "path_log_points": 4,
    "chunk_size": 256,
    "time_budget_s": 3600,
}
# The DFT k-mesh of the .win kpoints block and the .eig, and the position matrices' seed.
EIG_MESH = 4
POSITION_SEED = 20261019


@dataclass(frozen=True)
class Case:
    step: str
    model: str
    settings: dict = field(default_factory=dict)
    tb: bool = False
    eig: bool = False
    expect: str = "ok"


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
    "qgeom_full_metal": Case(
        "PYTHTB_QGEOM", "layer3", {**QGEOM_SETTINGS, "fermi_energy": -0.35}, tb=True, eig=True
    ),
    "qgeom_full_insulator": Case(
        "PYTHTB_QGEOM",
        "layer3",
        {
            **QGEOM_SETTINGS,
            "fermi_energy": 0.55,
            "wannier_omega_I_A2": 3.25,
            "mu_range_meV": [-300.0, 300.0],
            "qgeom_gap_ratio": 0.3,
        },
        tb=True,
    ),
    "qgeom_diagonal_budget": Case(
        "PYTHTB_QGEOM", "layer3", {**QGEOM_SETTINGS, "fermi_energy": -0.35, "qgeom_tolerance": 0.01}, eig=True
    ),
    "qgeom_position_unavailable": Case(
        "PYTHTB_QGEOM",
        "layer3",
        {**QGEOM_SETTINGS, "fermi_energy": -0.35, "position_operator": "full"},
        expect="pythtb_position_unavailable",
    ),
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
    species = ["B", "N", "C"][: len(sites)]
    values = {"step_type": case.step, **CONTEXT, **case.settings}
    values["structure"] = {"species": species, "cart": sites.tolist()}
    values.update(overrides)
    return values


def eig_mesh():
    """The fractional DFT k-mesh of the .win kpoints block (EIG_MESH x EIG_MESH, k_3 = 0)."""
    steps = np.arange(EIG_MESH) / EIG_MESH
    return np.array([[a, b, 0.0] for a in steps for b in steps])


def write_qgeom_files(directory, job, tb, eig):
    """The quantum-geometry inputs: ``wannier90_tb.dat`` (``tb``), and with ``eig`` the .win kpoints block
    and the ``.eig`` of the model's bands at it (absolute eV)."""
    translations, table = job.translations, job.raw
    if tb:
        rng = np.random.default_rng(POSITION_SEED)
        positions = exact(position_table(rng, translations, job.centres, scale=0.04, antihermitian=0.002))
        degeneracies = job.degeneracies[:, None, None, None]
        write_tb(
            directory / "wannier90_tb.dat",
            job.lattice,
            translations,
            job.degeneracies,
            table,
            positions * degeneracies,
        )
    if eig:
        mesh = eig_mesh()
        rows = "\n".join(f"{a:.10f} {b:.10f} {c:.10f}" for a, b, c in mesh)
        win = directory / "wannier90.win"
        win.write_text(win.read_text() + f"begin kpoints\n{rows}\nend kpoints\n")
        energies = band_energies(*job.driver_table(), mesh)
        lines = [
            f"{b + 1:5d}{k + 1:5d}{value:18.12f}"
            for k, row in enumerate(energies)
            for b, value in enumerate(row)
        ]
        (directory / "wannier90.eig").write_text("\n".join(lines) + "\n")


def write_case(name, directory, **overrides):
    """Stage case ``name`` in ``directory``; returns the parameter file's path."""
    directory = Path(directory)
    case = CASES[name]
    model = MODELS[case.model]()
    job = Wannier90Job(
        model.translations, model.hopping_matrices, model.lattice, model.centres, mp_grid=MP_GRID, win=WIN
    )
    job.write(directory)
    kpoints = line_path(CORNERS, STEPS)
    ticks = [(label, index * STEPS) for index, label in enumerate(LABELS)]
    write_band_files(directory, kpoints, band_energies(*job.driver_table(), kpoints), ticks)
    if case.tb or case.eig:
        write_qgeom_files(directory, job, case.tb, case.eig)
    path = directory / PARAMS_FILENAME
    path.write_text(json.dumps(parameters(name, **overrides), indent=1, sort_keys=True) + "\n")
    return path
