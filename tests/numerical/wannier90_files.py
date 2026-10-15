"""Small tight-binding tables written as the wannier90 files a Generator WANNIER job stages.

Every number is written with ten decimals and the arrays a test compares with
are the written values read back, so a file and its arrays agree exactly.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SEED = "wannier90"


def exact(values):
    """``values`` as written with ten decimals and read back."""
    array = np.asarray(values)
    if np.iscomplexobj(array):
        return exact(array.real) + 1j * exact(array.imag)
    flat = [float(f"{value:.10f}") for value in array.astype(float).ravel()]
    return np.asarray(flat, dtype=float).reshape(array.shape)


@dataclass
class Wannier90Job:
    """The staged files of one WANNIER run.

    ``table`` is H(R) in eV on ``translations``; the HR file holds it times
    ``degeneracies``, as wannier90 writes it. ``centres`` and ``lattice`` are
    Cartesian Å; ``win`` adds keywords to the ``.win``.
    """

    translations: np.ndarray
    table: np.ndarray
    lattice: np.ndarray
    centres: np.ndarray
    degeneracies: np.ndarray | None = None
    mp_grid: tuple[int, int, int] | None = (2, 2, 1)
    win: dict = field(default_factory=dict)
    cell_unit: str | None = "ang"
    kpoint_path: tuple = ()
    atoms: tuple = ()

    def __post_init__(self):
        self.translations = np.asarray(self.translations, dtype=np.int64)
        if self.degeneracies is None:
            self.degeneracies = np.ones(len(self.translations), dtype=np.int64)
        self.degeneracies = np.asarray(self.degeneracies, dtype=np.int64)
        self.raw = exact(np.asarray(self.table, complex) * self.degeneracies[:, None, None])
        self.lattice = exact(self.lattice)
        self.centres = exact(self.centres)

    @property
    def num_wann(self):
        return self.raw.shape[1]

    def driver_table(self):
        """The driver's table: translations in lexicographic order, degeneracy divided out."""
        order = np.lexsort(self.translations.T[::-1])
        return self.translations[order], self.raw[order] / self.degeneracies[order].astype(float)[
            :, None, None
        ]

    def write(self, directory, *, hr=True, centres=True, win=True):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if win:
            (directory / f"{SEED}.win").write_text(self.win_text())
        if hr:
            (directory / f"{SEED}_hr.dat").write_text(self.hr_text())
        if centres:
            (directory / f"{SEED}_centres.xyz").write_text(self.centres_text())
        return directory

    def win_text(self):
        lines = [
            f"num_wann = {self.num_wann}",
            "num_bands = " + str(self.num_wann),
            "",
            "begin unit_cell_cart",
        ]
        if self.cell_unit is not None:
            lines.append(self.cell_unit)
        scale = 1.0 / 0.52917721092 if self.cell_unit == "bohr" else 1.0
        lines += ["  ".join(f"{value * scale:.10f}" for value in row) for row in self.lattice]
        lines.append("end unit_cell_cart")
        if self.mp_grid is not None:
            lines.append("mp_grid = " + " ".join(str(value) for value in self.mp_grid))
        for key, value in self.win.items():
            lines.append(f"{key} = {value}")
        if self.kpoint_path:
            lines.append("begin kpoint_path")
            for la, ka, lb, kb in self.kpoint_path:
                lines.append(f"{la} {ka[0]} {ka[1]} {ka[2]} {lb} {kb[0]} {kb[1]} {kb[2]}")
            lines.append("end kpoint_path")
        return "\n".join(lines) + "\n"

    def hr_text(self):
        n = self.num_wann
        lines = ["written by the QUEER tests", str(n), str(len(self.translations))]
        for start in range(0, len(self.degeneracies), 15):
            lines.append("".join(f"{value:5d}" for value in self.degeneracies[start : start + 15]))
        for translation, block in zip(self.translations, self.raw):
            for j in range(n):
                for i in range(n):
                    head = "".join(f"{int(value):5d}" for value in (*translation, i + 1, j + 1))
                    lines.append(f"{head} {block[i, j].real:16.10f} {block[i, j].imag:16.10f}")
        return "\n".join(lines) + "\n"

    def centres_text(self):
        lines = [str(self.num_wann + len(self.atoms)), "Wannier centres, written by the QUEER tests"]
        lines += [f"X {x:.10f} {y:.10f} {z:.10f}" for x, y, z in self.centres]
        lines += [f"{symbol} {x:.10f} {y:.10f} {z:.10f}" for symbol, (x, y, z) in self.atoms]
        return "\n".join(lines) + "\n"


def write_band_files(directory, kpoints, energies, ticks=()):
    """``_band.kpt``, ``_band.dat`` (absolute eV, (Nk, NB)) and, with ``ticks``, ``_band.labelinfo.dat``.

    ``ticks`` are ``(label, zero-based index)`` pairs. Returns the energies as written.
    """
    directory = Path(directory)
    kpoints = exact(kpoints)
    energies = exact(energies)
    rows = [str(len(kpoints))] + [f"{kx:.10f} {ky:.10f} {kz:.10f} 1.0" for kx, ky, kz in kpoints]
    (directory / f"{SEED}_band.kpt").write_text("\n".join(rows) + "\n")
    blocks = []
    for band in energies.T:
        blocks.append("\n".join(f"{index * 0.01:.6f} {value:.10f}" for index, value in enumerate(band)))
    (directory / f"{SEED}_band.dat").write_text("\n\n".join(blocks) + "\n")
    if ticks:
        rows = [
            f"{label} {index + 1} {index * 0.01:.6f} {' '.join(str(x) for x in kpoints[index])}"
            for label, index in ticks
        ]
        (directory / f"{SEED}_band.labelinfo.dat").write_text("\n".join(rows) + "\n")
    return energies


def line_path(corners, points_per_segment):
    """Fractional points along consecutive ``corners``, each join written once."""
    path = [np.asarray(corners[0], float)]
    for start, end in zip(corners[:-1], corners[1:]):
        start, end = np.asarray(start, float), np.asarray(end, float)
        for t in np.arange(1, points_per_segment + 1) / points_per_segment:
            path.append(start + t * (end - start))
    return np.asarray(path)
