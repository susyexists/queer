"""Native, explicit band-output readers and source-independent diagnostics.

Energies are absolute eV with axes (spin, kpoint, band). Missing Fermi levels
or coordinates remain missing; readers never fabricate physical metadata.
Segment bounds are half-open indices, and repeated path endpoints are kept.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..capabilities import CapabilityError
from ..physics.constants import BOHR_ANGSTROM as BOHR_ANGSTROM
from ..physics.constants import HARTREE_EV as HARTREE_EV

_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?"


def _float(token):
    value = float(token.replace("D", "E").replace("d", "e"))
    if not np.isfinite(value):
        raise ValueError("nonfinite scientific record")
    return value


@dataclass(frozen=True)
class BandData:
    energies: np.ndarray
    kpoints: np.ndarray | None = None
    fermi_energy: float | None = None
    occupations: np.ndarray | None = None
    weights: np.ndarray | None = None
    labels: tuple[str, ...] = ()
    label_indices: tuple[int, ...] = ()
    segments: tuple[tuple[int, int], ...] = ()
    distances: np.ndarray | None = None
    reciprocal: np.ndarray | None = None
    spin_labels: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        energy = np.asarray(self.energies, float)
        if energy.ndim == 2:
            energy = energy[None]
        if energy.ndim != 3 or not all(energy.shape) or not np.isfinite(energy).all():
            raise ValueError("energies must be finite nonempty (spin,kpoint,band) arrays")
        object.__setattr__(self, "energies", energy.copy())
        ns, nk, _ = energy.shape
        for name, shape in (
            ("kpoints", (nk, 3)),
            ("occupations", energy.shape),
            ("weights", (nk,)),
            ("distances", (nk,)),
            ("reciprocal", (3, 3)),
        ):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value, float)
                if name == "occupations" and value.ndim == 2 and ns == 1:
                    value = value[None]
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"{name} must be a finite array with shape {shape}")
                object.__setattr__(self, name, value.copy())
        if self.fermi_energy is not None and not np.isfinite(self.fermi_energy):
            raise ValueError("fermi_energy must be finite absolute eV")
        if self.weights is not None and np.any(self.weights < 0):
            raise ValueError("source kpoint weights must be nonnegative")
        if self.occupations is not None and np.any(self.occupations < -1e-8):
            raise ValueError("occupations must be nonnegative")
        if self.reciprocal is not None and np.linalg.matrix_rank(self.reciprocal) != 3:
            raise ValueError("reciprocal lattice must be nonsingular")
        segments = tuple(tuple(map(int, pair)) for pair in self.segments) or ((0, nk),)
        if (
            segments[0][0] != 0
            or segments[-1][1] != nk
            or any(a < 0 or b <= a or b > nk for a, b in segments)
            or any(left[1] != right[0] for left, right in zip(segments, segments[1:]))
        ):
            raise ValueError("segments must partition the kpoint rows with half-open bounds")
        object.__setattr__(self, "segments", segments)
        labels, indices = tuple(map(str, self.labels)), tuple(map(int, self.label_indices))
        if (
            len(labels) != len(indices)
            or any(i < 0 or i >= nk for i in indices)
            or any(a > b for a, b in zip(indices, indices[1:]))
        ):
            raise ValueError("labels require ordered, in-range label_indices")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "label_indices", indices)
        if self.distances is not None and any(
            np.any(np.diff(self.distances[a:b]) < -1e-12) for a, b in segments
        ):
            raise ValueError("distances must be nondecreasing within each path segment")
        spins = tuple(self.spin_labels) or (
            ("up", "down") if ns == 2 else tuple(f"spin_{i}" for i in range(ns))
        )
        if len(spins) != ns:
            raise ValueError("spin_labels must match the spin axis")
        object.__setattr__(self, "spin_labels", spins)
        json.dumps(dict(self.metadata), allow_nan=False)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def relative_energies(self):
        if self.fermi_energy is None:
            raise CapabilityError("Fermi-relative bands require an explicit Fermi energy")
        return self.energies - self.fermi_energy

    def as_result(self):
        from ..observables.result import ObservableResult

        arrays = {
            "energies": self.energies,
            "segments": np.asarray(self.segments),
            "labels": np.asarray(self.labels, dtype="U"),
            "label_indices": np.asarray(self.label_indices),
        }
        dims = {
            "energies": ("spin", "kpoint", "band"),
            "segments": ("segment", "bound"),
            "labels": ("label",),
            "label_indices": ("label",),
        }
        units = {"energies": "eV", "segments": "1", "labels": "1", "label_indices": "1"}
        for name, axes, unit in (
            ("kpoints", ("kpoint", "fractional"), "fractional_reciprocal"),
            ("occupations", ("spin", "kpoint", "band"), "1"),
            ("weights", ("kpoint",), "1"),
            ("distances", ("kpoint",), "reported"),
            ("reciprocal", ("axis", "cartesian"), "angstrom^-1"),
        ):
            if getattr(self, name) is not None:
                arrays[name], dims[name], units[name] = getattr(self, name), axes, unit
        return ObservableResult(
            arrays,
            dims,
            units,
            {
                **self.metadata,
                "energy_reference": "absolute",
                "fermi_energy_eV": self.fermi_energy,
                "spin_labels": list(self.spin_labels),
            },
        )


def _read_lines(path):
    return Path(path).read_text(encoding="utf-8", errors="replace").splitlines()


def read_eigenval(path, *, fermi_energy=None, reciprocal=None, occupation_capacity=None):
    """Read VASP EIGENVAL with occupations preserved.

    A one-channel EIGENVAL does not encode whether its states are scalar or
    spinors. Supply ``occupation_capacity=1`` for a spin-orbit/noncollinear
    calculation; the scalar default is two electrons per state.
    """
    lines = _read_lines(path)
    try:
        nspin = int(lines[0].split()[-1])
        nelect, nk, nb = lines[5].split()[:3]
        nk, nb = int(nk), int(nb)
        if nspin not in (1, 2) or min(nk, nb) < 1:
            raise ValueError("invalid band dimensions")
        energy, occupation = np.empty((nspin, nk, nb)), np.empty((nspin, nk, nb))
        points, weights, cursor = [], [], 6
        for k in range(nk):
            while not lines[cursor].strip():
                cursor += 1
            row = lines[cursor].split()
            if len(row) != 4:
                raise ValueError("expected fractional kpoint and weight")
            points.append([_float(t) for t in row[:3]])
            weights.append(_float(row[3]))
            cursor += 1
            for band in range(nb):
                row = lines[cursor].split()
                if int(row[0]) != band + 1 or len(row) != 1 + 2 * nspin:
                    raise ValueError("invalid band index or spin columns")
                energy[:, k, band] = [_float(t) for t in row[1 : 1 + nspin]]
                occupation[:, k, band] = [_float(t) for t in row[1 + nspin :]]
                cursor += 1
        if any(line.strip() for line in lines[cursor:]):
            raise ValueError("unexpected trailing band records")
    except (ValueError, IndexError) as error:
        raise ValueError(f"{path}: malformed EIGENVAL ({error})") from error
    capacity = (2 if nspin == 1 else 1) if occupation_capacity is None else occupation_capacity
    if capacity not in (1, 2) or (nspin == 2 and capacity != 1):
        raise ValueError("occupation_capacity must be 1 for spin-resolved states or 2 for scalar states")
    return BandData(
        energy,
        np.asarray(points),
        fermi_energy,
        occupation,
        np.asarray(weights),
        reciprocal=reciprocal,
        metadata={
            "source_format": "vasp_eigenval",
            "electron_count": _float(nelect),
            "occupation_capacity": capacity,
            "occupation_capacity_source": "explicit"
            if occupation_capacity is not None
            else "scalar_or_collinear_default",
        },
    )


def read_vasp_outcar(path):
    """Read the last complete VASP band cycle; an incomplete final cycle fails."""
    lines = _read_lines(path)
    cycle: dict[tuple[int, int], tuple[list[float], list[float]]] = {}
    points: dict[int, list[float]] = {}
    spin, ef, reciprocal = 0, None, None
    expected_nk = expected_nb = expected_ns = None
    spinor = False
    kpattern = re.compile(
        r"k-point\s+(\d+)\s*:\s*(" + _NUMBER + r")\s+(" + _NUMBER + r")\s+(" + _NUMBER + ")"
    )
    for i, line in enumerate(lines):
        match = re.search(r"\bISPIN\s*=\s*([12])\b", line)
        if match:
            expected_ns = int(match.group(1))
        if re.search(r"\b(?:LSORBIT|LNONCOLLINEAR)\s*=\s*(?:T\b|\.TRUE\.)", line, re.I):
            spinor = True
        match = re.search(r"NKPTS\s*=\s*(\d+).*NBANDS\s*=\s*(\d+)", line)
        if match:
            expected_nk, expected_nb = map(int, match.groups())
        match = re.search(r"E-fermi\s*:\s*(" + _NUMBER + ")", line)
        if match:
            ef = _float(match.group(1))
        match = re.search(r"spin component\s+(\d+)", line, flags=re.I)
        if match:
            spin = int(match.group(1)) - 1
        if "direct lattice vectors" in line and "reciprocal lattice vectors" in line:
            try:
                reciprocal = np.array([[_float(t) for t in row.split()[3:6]] for row in lines[i + 1 : i + 4]])
                reciprocal *= 2 * np.pi
            except ValueError as error:
                raise ValueError(f"{path}: malformed OUTCAR lattice") from error
        match = kpattern.search(line)
        if not match:
            continue
        index = int(match.group(1)) - 1
        cursor = i + 1
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if (
            cursor == len(lines)
            or "band" not in lines[cursor].lower()
            or "occupation" not in lines[cursor].lower()
        ):
            continue
        if spin == 0 and index == 0 and (0, 0) in cycle:
            cycle, points = {}, {}
        cursor += 1
        values: list[float] = []
        occupations: list[float] = []
        while cursor < len(lines):
            fields = lines[cursor].split()
            if len(fields) != 3 or not fields[0].isdigit():
                break
            if int(fields[0]) != len(values) + 1:
                raise ValueError(f"{path}: invalid OUTCAR band sequence")
            values.append(_float(fields[1]))
            occupations.append(_float(fields[2]))
            cursor += 1
        if not values or (expected_nb is not None and len(values) != expected_nb):
            raise ValueError(f"{path}: incomplete final OUTCAR band block")
        cycle[spin, index] = values, occupations
        point = [_float(t) for t in match.groups()[1:]]
        if index in points and not np.allclose(points[index], point, atol=1e-7, rtol=0):
            raise ValueError(f"{path}: spin channels use different kpoints")
        points[index] = point
    if not cycle:
        raise ValueError(f"{path}: no OUTCAR band records")
    ns = 1 if spinor else expected_ns or max(k[0] for k in cycle) + 1
    nk = expected_nk or max(points) + 1
    if set(cycle) != {(s, k) for s in range(ns) for k in range(nk)}:
        raise ValueError(f"{path}: incomplete final OUTCAR spin/kpoint cycle")
    energy = np.asarray([[cycle[s, k][0] for k in range(nk)] for s in range(ns)])
    occupation = np.asarray([[cycle[s, k][1] for k in range(nk)] for s in range(ns)])
    return BandData(
        energy,
        np.asarray([points[k] for k in range(nk)]),
        ef,
        occupation,
        reciprocal=reciprocal,
        spin_labels=("spinor",) if spinor else (),
        metadata={
            "source_format": "vasp_outcar",
            "spinor": spinor,
            "occupation_capacity": 1 if spinor or ns == 2 else 2,
        },
    )


def read_vasprun(path):
    """Stream standard vasprun XML and retain its final eigenvalue array."""
    points = weights = reciprocal = energy = occupation = ef = None
    expected_ns, spinor = None, False
    try:
        for _, element in ET.iterparse(path, events=("end",)):
            name = element.attrib.get("name")
            if element.tag == "i" and name in {"ISPIN", "LSORBIT", "LNONCOLLINEAR"}:
                token = element.text.strip()
                if name == "ISPIN":
                    expected_ns = int(token)
                    if expected_ns not in (1, 2):
                        raise ValueError("invalid ISPIN")
                else:
                    spinor |= token.lower() in {"true", "t", ".true."}
            elif element.tag == "varray" and name in {"kpointlist", "weights", "rec_basis"}:
                values = np.asarray([[_float(t) for t in row.text.split()] for row in element.findall("v")])
                if name == "kpointlist":
                    points = values
                elif name == "weights":
                    weights = values.reshape(-1)
                else:
                    reciprocal = values * 2 * np.pi
                element.clear()
            elif element.tag == "i" and name == "efermi":
                ef = _float(element.text.strip())
            elif element.tag == "eigenvalues":
                outer = element.find("array/set")
                if outer is not None:
                    spin_sets = outer.findall("set")
                    rows = [
                        [
                            [[_float(t) for t in row.text.split()] for row in kset.findall("r")]
                            for kset in spin_set.findall("set")
                        ]
                        for spin_set in spin_sets
                    ]
                    values = np.asarray(rows, float)
                    if values.ndim != 4 or values.shape[-1] < 2:
                        raise ValueError("invalid eigenvalue/occupation dimensions")
                    energy, occupation = values[..., 0], values[..., 1]
                element.clear()
            elif element.tag == "calculation":
                element.clear()
    except (ET.ParseError, ValueError, AttributeError) as error:
        raise ValueError(f"{path}: malformed vasprun XML ({error})") from error
    if energy is None or points is None:
        raise ValueError(f"{path}: no complete vasprun eigenvalues/kpoints")
    ns = energy.shape[0]
    if ns != (1 if spinor else expected_ns or ns):
        raise ValueError(f"{path}: incomplete vasprun spin channels")
    return BandData(
        energy,
        points,
        ef,
        occupation,
        weights,
        reciprocal=reciprocal,
        spin_labels=("spinor",) if spinor else (),
        metadata={
            "source_format": "vasp_vasprun",
            "spinor": spinor,
            "occupation_capacity": 1 if spinor or ns == 2 else 2,
        },
    )


def read_vasp_kpoints(path):
    """Read explicit, line-mode, Gamma or Monkhorst-Pack KPOINTS.

    Coordinates retain the file's stated mode; Cartesian input conversion
    requires the caller's lattice scale. Line-mode segments retain duplicates.
    """
    lines = _read_lines(path)
    try:
        count = int(lines[1].split()[0])
        mode = lines[2].strip().lower()
        if count == 0:
            grid = np.asarray([int(t) for t in lines[3].split()[:3]])
            if grid.shape != (3,) or np.any(grid < 1) or not mode.startswith(("g", "m")):
                raise ValueError("invalid automatic mesh")
            shift = (
                np.zeros(3)
                if len(lines) < 5 or not lines[4].strip()
                else np.asarray([_float(t) for t in lines[4].split()[:3]])
            )
            axes = [
                (np.arange(n) / n if mode.startswith("g") else (2 * np.arange(n) + 1 - n) / (2 * n))
                + shift[i] / n
                for i, n in enumerate(grid)
            ]
            mesh_points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
            return {
                "kpoints": mesh_points,
                "weights": np.repeat(1 / len(mesh_points), len(mesh_points)),
                "coordinate_mode": "fractional_reciprocal",
                "grid": grid.tolist(),
                "labels": (),
                "label_indices": (),
                "segments": ((0, len(mesh_points)),),
            }
        line_mode = mode.startswith("l")
        coordinate_mode = lines[3 if line_mode else 2].strip().lower()
        if not coordinate_mode.startswith(("r", "c", "k")):
            raise ValueError("expected Reciprocal or Cartesian coordinates")
        rows = []
        for line in lines[4 if line_mode else 3 :]:
            if not line.strip():
                continue
            numeric, _, label = line.partition("!")
            fields = numeric.split()
            rows.append(([_float(t) for t in fields[:3]], label.strip(), fields[3:]))
        labels, indices, segments, weights = [], [], [], []
        points: list[Any] = []
        if line_mode:
            if count < 2 or len(rows) % 2 or not rows:
                raise ValueError("line-mode needs endpoint pairs and at least two points per segment")
            for a, b in zip(rows[::2], rows[1::2], strict=True):
                start = len(points)
                points.extend(np.linspace(a[0], b[0], count))
                segments.append((start, len(points)))
                for offset, row in ((0, a), (count - 1, b)):
                    if row[1]:
                        labels.append(row[1])
                        indices.append(start + offset)
        else:
            if len(rows) != count:
                raise ValueError("explicit kpoint count does not match records")
            for row in rows:
                points.append(row[0])
                weights.append(_float(row[2][0]) if row[2] else 1.0)
                if row[1]:
                    labels.append(row[1])
                    indices.append(len(points) - 1)
            segments = [(0, count)]
        return {
            "kpoints": np.asarray(points),
            "weights": None if line_mode else np.asarray(weights),
            "coordinate_mode": "fractional_reciprocal"
            if coordinate_mode.startswith("r")
            else "cartesian_scaled",
            "labels": tuple(labels),
            "label_indices": tuple(indices),
            "segments": tuple(segments),
        }
    except (ValueError, IndexError) as error:
        raise ValueError(f"{path}: malformed KPOINTS ({error})") from error


def read_openmx_bands(path):
    """Read OpenMX .Band eigenvalues and reciprocal lattice in atomic units."""
    lines = [line for line in _read_lines(path) if line.strip()]
    try:
        header = lines[0].split()
        nb, spin = int(header[0]), int(header[1])
        ef = _float(header[2]) * HARTREE_EV
        if nb < 1 or spin not in (0, 1):
            raise ValueError("invalid .Band band/spin dimensions")
        reciprocal = np.asarray([_float(t) for t in lines[1].split()]).reshape(3, 3) / BOHR_ANGSTROM
        nsegment, cursor = int(lines[2].split()[0]), 3
        if nsegment < 1:
            raise ValueError("missing path segments")
        segments: list[tuple[int, int]] = []
        labels: list[str] = []
        indices: list[int] = []
        nk = 0
        for _ in range(nsegment):
            fields = lines[cursor].split()
            count = int(fields[0])
            if count < 1 or len(fields) < 7:
                raise ValueError("invalid path segment record")
            for t in fields[1:7]:
                _float(t)
            segments.append((nk, nk + count))
            if len(fields) >= 9:
                labels.extend(fields[7:9])
                indices.extend((nk, nk + count - 1))
            nk += count
            cursor += 1
        energy, points = np.empty((spin + 1, nk, nb)), []
        repeated_spin_headers = False
        for k in range(nk):
            fields = lines[cursor].split()
            if len(fields) != 4 or int(fields[0]) != nb:
                raise ValueError("invalid .Band kpoint header")
            points.append([_float(t) for t in fields[1:]])
            cursor += 1
            values: list[float] = []
            while len(values) < nb:
                values.extend(_float(t) for t in lines[cursor].split())
                cursor += 1
            if spin == 1 and len(values) == nb:
                fields = lines[cursor].split()
                if len(fields) == 4 and fields[0] == str(nb):
                    if not np.allclose([_float(t) for t in fields[1:]], points[-1], rtol=0, atol=1e-10):
                        raise ValueError("different spin-channel kpoints")
                    repeated_spin_headers = True
                    cursor += 1
                while len(values) < 2 * nb:
                    values.extend(_float(t) for t in lines[cursor].split())
                    cursor += 1
            if len(values) != nb * (spin + 1):
                raise ValueError("invalid .Band eigenvalue count")
            energy[:, k] = np.asarray(values).reshape(spin + 1, nb) * HARTREE_EV
        if cursor != len(lines):
            raise ValueError("unexpected trailing .Band records")
    except (IndexError, ValueError) as error:
        raise ValueError(f"{path}: malformed OpenMX .Band ({error})") from error
    return BandData(
        energy,
        np.asarray(points),
        ef,
        labels=tuple(labels),
        label_indices=tuple(indices),
        segments=tuple(segments),
        reciprocal=reciprocal,
        metadata={
            "source_format": "openmx_band",
            "spin_record_layout": "repeated_headers" if repeated_spin_headers else "grouped_energies",
        },
    )


def read_openmx_output(path):
    """Read SCF convergence and final energy/reference records from OpenMX text."""
    lines = _read_lines(path)
    text = "\n".join(lines)
    energy = re.findall(r"\bUtot\.?\s*(?:=\s*)?(" + _NUMBER + ")", text)
    fermi = re.findall(
        r"(?:ChemP\s*=|Chemical\s+potential\s*(?:\(Hartree\))?\s*[:=]?)\s*(" + _NUMBER + ")", text, flags=re.I
    )
    norm = re.findall(r"NormRD\s*=\s*(" + _NUMBER + ")", text)
    converged, provenance, iterations = None, None, None
    for number, line in enumerate(lines, 1):
        iteration = re.search(r"\bSCF\s*(?:iteration|iter\.?|=)\s*(?:[:=]\s*)?(\d+)", line, re.I)
        if iteration:
            iterations = int(iteration.group(1))
            if iterations == 1:
                converged, provenance = None, None
        failure = re.search(r"\bSCF\b.*(?:not converged|failed to converge|did not converge)", line, re.I)
        success = re.search(
            r"\bSCF\b.*(?:\bis (?:achieved|converged)\b|convergence is achieved)|\bconvergence criteri(?:on|a) (?:is|are) satisfied\b|\bSCF\b.*\bis converged\b",
            line,
            re.I,
        )
        if failure or success:
            converged = not bool(failure)
            provenance = {"line": number, "marker": line.strip()}
    return {
        "source_format": "openmx_output",
        "total_energy_eV": None if not energy else _float(energy[-1]) * HARTREE_EV,
        "total_energy_history_eV": [_float(t) * HARTREE_EV for t in energy],
        "fermi_energy_eV": None if not fermi else _float(fermi[-1]) * HARTREE_EV,
        "scf_converged": converged,
        "scf_convergence_provenance": provenance,
        "scf_iterations": iterations,
        "final_density_residual": None if not norm else _float(norm[-1]),
        "density_residuals": [_float(t) for t in norm],
    }


def _wannier_points(path):
    lines = [line for line in _read_lines(path) if line.strip()]
    try:
        count = int(lines[0])
        if len(lines) != count + 1:
            raise ValueError("kpoint count mismatch")
        points = []
        for index, line in enumerate(lines[1:], 1):
            fields = line.split()
            if len(fields) >= 5 and fields[0] == str(index):
                fields = fields[1:]
            points.append([_float(t) for t in fields[:3]])
        return np.asarray(points)
    except (ValueError, IndexError) as error:
        raise ValueError(f"{path}: malformed Wannier kpoints ({error})") from error


def read_wannier_bands(path, *, kpoints_path=None, labelinfo_path=None, fermi_energy=None):
    """Read blank-separated Wannier bands, including segmented gnuplot paths."""
    blocks: list[np.ndarray] = []
    current: list[list[float]] = []
    for line in _read_lines(path):
        if not line.strip():
            if current:
                blocks.append(np.asarray(current))
                current = []
        elif not line.lstrip().startswith(("#", "!")):
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"{path}: expected distance and energy columns")
            current.append([_float(t) for t in fields])
    if current:
        blocks.append(np.asarray(current))
    if not blocks:
        raise ValueError(f"{path}: no Wannier band records")
    points = None if kpoints_path is None else _wannier_points(kpoints_path)
    grouped, segment_lengths = [], []
    if points is not None:
        flat = np.vstack(blocks)
        nk = len(points)
        if len(flat) % nk:
            raise ValueError(f"{path}: bands do not match the supplied kpoint count")
        grouped = list(flat.reshape(-1, nk, 2))
        remaining = nk
        for block in blocks:
            size = min(remaining, len(block))
            segment_lengths.append(size)
            remaining -= size
            if not remaining:
                break
    else:
        parts: list[np.ndarray] = []
        lengths: list[int] = []
        origin = blocks[0][0, 0]
        for block in blocks:
            if parts and block[0, 0] <= origin + 1e-12:
                grouped.append(np.vstack(parts))
                if not segment_lengths:
                    segment_lengths = lengths
                parts, lengths = [], []
            parts.append(block)
            lengths.append(len(block))
        grouped.append(np.vstack(parts))
        if not segment_lengths:
            segment_lengths = lengths
    reference = grouped[0][:, 0]
    if any(
        row.shape != grouped[0].shape or not np.allclose(row[:, 0], reference, rtol=0, atol=1e-8)
        for row in grouped
    ):
        raise ValueError(f"{path}: band paths have different distance axes")
    ends = set(np.cumsum([0, *segment_lengths]).tolist())
    labels, indices = [], []
    label_records = []
    if labelinfo_path is not None:
        for line in _read_lines(labelinfo_path):
            fields = line.split()
            if not fields or fields[0].startswith(("#", "!")):
                continue
            if len(fields) < 2:
                raise ValueError(f"{labelinfo_path}: invalid label record")
            labels.append(fields[0])
            indices.append(int(fields[1]) - 1)
            if len(fields) >= 3:
                coordinate = _float(fields[2])
                index = indices[-1]
                if (
                    index < 0
                    or index >= len(reference)
                    or not np.isclose(coordinate, reference[index], rtol=0, atol=1e-8)
                ):
                    raise ValueError(f"{labelinfo_path}: label distance does not match band rows")
                fractional = None if len(fields) < 6 else [_float(t) for t in fields[3:6]]
                if (
                    fractional is not None
                    and points is not None
                    and not np.allclose(fractional, points[index], rtol=0, atol=1e-8)
                ):
                    raise ValueError(f"{labelinfo_path}: label kpoint does not match kpoint file")
                label_records.append((index, coordinate, fractional))
    for left, right in zip(label_records, label_records[1:]):
        if right[0] == left[0] + 1 and np.isclose(left[1], right[1], rtol=0, atol=1e-8):
            if left[2] is None or right[2] is None or not np.allclose(left[2], right[2], rtol=0, atol=1e-8):
                ends.add(right[0])
    boundaries = sorted(ends)
    segments = tuple(zip(boundaries[:-1], boundaries[1:]))
    return BandData(
        np.asarray([row[:, 1] for row in grouped]).T[None],
        points,
        fermi_energy,
        labels=tuple(labels),
        label_indices=tuple(indices),
        segments=segments,
        distances=reference,
        metadata={"source_format": "wannier_band"},
    )


def read_wannier_report(path):
    """Read final Wannier centers/spreads and explicit optimization convergence."""
    text = "\n".join(_read_lines(path))
    rows = re.findall(
        r"WF centre and spread\s+(\d+)\s*\(\s*("
        + _NUMBER
        + r")\s*,\s*("
        + _NUMBER
        + r")\s*,\s*("
        + _NUMBER
        + r")\s*\)\s*("
        + _NUMBER
        + ")",
        text,
    )
    records = {}
    for index, *values in rows:
        records[int(index)] = [_float(t) for t in values]
    centers = [records[key][:3] for key in sorted(records)]
    spreads = [records[key][3] for key in sorted(records)]
    total = re.findall(r"(?:Omega Total|Sum of centres and spreads).*?(" + _NUMBER + r")\s*$", text, re.M)
    wannier_converged = disentanglement_converged = None
    for line in text.splitlines():
        failure = re.search(r"(?:not converged|failed to converge|did not converge)", line, re.I)
        success = re.search(r"\bconverged\b|\bConvergence criteria satisfied\b", line, re.I)
        if not failure and not success:
            continue
        if "disentanglement" in line.lower():
            disentanglement_converged = not bool(failure)
        elif re.search(r"wannieri[sz]ation|Convergence criteria satisfied", line, re.I):
            wannier_converged = not bool(failure)
    return {
        "source_format": "wannier_wout",
        "centers_angstrom": centers,
        "spreads_angstrom2": spreads,
        "total_spread_angstrom2": sum(spreads) if spreads else (_float(total[-1]) if total else None),
        "wannier_converged": wannier_converged,
        "disentanglement_converged": disentanglement_converged,
    }


def read_bands(path, *, format=None, **options):
    name = Path(path).name.lower()
    kind = format or (
        "vasprun"
        if "vasprun" in name and name.endswith(".xml")
        else "outcar"
        if name == "outcar"
        else "eigenval"
        if name == "eigenval"
        else "openmx"
        if name.endswith(".band")
        else "wannier"
        if name.endswith("_band.dat")
        else None
    )
    readers = {
        "vasprun": read_vasprun,
        "outcar": read_vasp_outcar,
        "eigenval": read_eigenval,
        "openmx": read_openmx_bands,
        "wannier": read_wannier_bands,
    }
    if kind not in readers:
        raise ValueError("Specify band format: vasprun, outcar, eigenval, openmx, or wannier")
    return readers[kind](path, **options)


def band_gap(data: BandData, *, occupation_tolerance=1e-6):
    """Report sampled gaps/edges; partial occupations or sampled crossings mark metals."""
    energy = data.energies
    if not np.isfinite(occupation_tolerance) or occupation_tolerance <= 0:
        raise ValueError("occupation_tolerance must be positive")
    if data.occupations is not None:
        capacity = data.metadata.get("occupation_capacity", 2 if energy.shape[0] == 1 else 1)
        occupied = data.occupations > capacity / 2
        metallic = bool(
            np.any(
                (data.occupations > occupation_tolerance)
                & (data.occupations < capacity - occupation_tolerance)
            )
        )
    elif data.fermi_energy is not None:
        occupied = energy < data.fermi_energy
        metallic = bool(
            np.any((energy.min(axis=1) <= data.fermi_energy) & (energy.max(axis=1) >= data.fermi_energy))
        )
    else:
        raise CapabilityError("band-gap analysis requires occupations or an explicit Fermi level")
    if not occupied.any() or occupied.all():
        raise CapabilityError("sampled bands must contain both occupied and empty states")
    valence = np.where(occupied, energy, -np.inf)
    conduction = np.where(~occupied, energy, np.inf)
    vi, ci = (
        np.unravel_index(np.argmax(valence), energy.shape),
        np.unravel_index(np.argmin(conduction), energy.shape),
    )
    vbm, cbm = float(energy[vi]), float(energy[ci])
    gap = max(0.0, cbm - vbm)
    direct_values = conduction.min(axis=2) - valence.max(axis=2)
    direct_values = direct_values[np.isfinite(direct_values)]
    direct_gap = max(0.0, float(direct_values.min())) if len(direct_values) else None
    metallic |= cbm <= vbm

    def edge(index):
        return {
            "energy_eV": float(energy[index]),
            "spin_index": int(index[0]),
            "kpoint_index": int(index[1]),
            "band_index": int(index[2]),
            "kpoint": None if data.kpoints is None else data.kpoints[index[1]].tolist(),
        }

    difference = None if data.kpoints is None else data.kpoints[vi[1]] - data.kpoints[ci[1]]
    same_point = vi[1] == ci[1] or (
        difference is not None and np.allclose(difference - np.round(difference), 0, atol=1e-8, rtol=0)
    )
    return {
        "gap_eV": 0.0 if metallic else gap,
        "direct_gap_eV": direct_gap,
        "is_direct": bool(not metallic and same_point and vi[0] == ci[0]),
        "is_metal": bool(metallic),
        "vbm": edge(vi),
        "cbm": edge(ci),
        "scope": "sampled kpoints; unsampled extrema are not excluded",
    }


def _segment_coordinate(data, bounds):
    a, b = bounds
    if data.distances is not None:
        coordinate = data.distances[a:b] - data.distances[a]
    elif data.kpoints is not None and data.reciprocal is not None:
        cartesian = data.kpoints[a:b] @ data.reciprocal
        coordinate = np.r_[0, np.cumsum(np.linalg.norm(np.diff(cartesian, axis=0), axis=1))]
    elif data.kpoints is not None:
        coordinate = np.r_[0, np.cumsum(np.linalg.norm(np.diff(data.kpoints[a:b], axis=0), axis=1))]
    else:
        raise CapabilityError("band comparison requires physical path coordinates or reported distances")
    if len(coordinate) < 2 or coordinate[-1] <= 0 or np.any(np.diff(coordinate) <= 0):
        raise ValueError("band-comparison segments need at least two distinct ordered path points")
    return coordinate / coordinate[-1]


def compare_bands(reference: BandData, candidate: BandData, *, energy_window=None, align_fermi=True):
    """Compare each path segment independently, with unique energy-band matching."""
    if reference.energies.shape[0] != candidate.energies.shape[0]:
        raise ValueError("band comparison requires the same number of spin channels")
    if len(reference.segments) != len(candidate.segments):
        raise ValueError("band comparison requires matching path segment counts")
    if energy_window is not None:
        window = np.asarray(energy_window, float)
        if window.shape != (2,) or not np.isfinite(window).all() or window[0] >= window[1]:
            raise ValueError("energy_window must be an ordered finite pair")
    left = reference.relative_energies() if align_fermi else reference.energies
    right = candidate.relative_energies() if align_fermi else candidate.energies
    errors, count, sampled = [], 0, 0
    for rb, cb in zip(reference.segments, candidate.segments, strict=True):
        if reference.kpoints is not None and candidate.kpoints is not None:
            rp = reference.kpoints[[rb[0], rb[1] - 1]]
            cp = candidate.kpoints[[cb[0], cb[1] - 1]]
            displacement = rp - cp
            if not np.allclose(displacement, np.round(displacement[0]), rtol=0, atol=1e-6):
                raise ValueError("band-comparison segments have different reciprocal endpoints")
        x, y = _segment_coordinate(reference, rb), _segment_coordinate(candidate, cb)
        if reference.kpoints is not None and candidate.kpoints is not None:
            resampled = np.column_stack(
                [np.interp(x, y, column) for column in candidate.kpoints[cb[0] : cb[1]].T]
            )
            offset = reference.kpoints[rb[0]] - candidate.kpoints[cb[0]]
            if not np.allclose(reference.kpoints[rb[0] : rb[1]], resampled + offset, atol=1e-6, rtol=0):
                raise ValueError("band-comparison segments follow different reciprocal paths")
        for spin in range(left.shape[0]):
            interpolated = np.asarray([np.interp(x, y, values) for values in right[spin, cb[0] : cb[1]].T]).T
            for ref_values, candidate_values in zip(left[spin, rb[0] : rb[1]], interpolated, strict=True):
                selected = (
                    np.arange(len(ref_values))
                    if energy_window is None
                    else np.flatnonzero((ref_values >= window[0]) & (ref_values <= window[1]))
                )
                count += len(selected)
                sampled += 1
                if len(selected):
                    rows, columns = linear_sum_assignment(
                        np.abs(ref_values[selected, None] - candidate_values[None])
                    )
                    errors.extend((candidate_values[columns] - ref_values[selected[rows]]).tolist())
    if not errors:
        raise ValueError("no matched band samples in the requested energy window")
    error = np.asarray(errors)
    return {
        "rmse_eV": float(np.sqrt(np.mean(error**2))),
        "mean_absolute_error_eV": float(np.mean(abs(error))),
        "max_absolute_error_eV": float(np.max(abs(error))),
        "mean_signed_error_eV": float(np.mean(error)),
        "matched_samples": len(error),
        "reference_samples": count,
        "coverage": len(error) / count,
        "sampled_kpoints_and_spins": sampled,
        "energy_reference": "fermi_relative" if align_fermi else "absolute",
        "matching": "independent segments; linear path interpolation; unique energy assignment",
    }
