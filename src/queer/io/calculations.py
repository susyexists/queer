"""Native final-state DFT result readers with explicit force/stress units."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
from scipy.constants import elementary_charge

from ..elements import ATOMIC_NUMBERS
from ..observables.result import ObservableResult
from .bands import (
    _NUMBER,
    BOHR_ANGSTROM,
    HARTREE_EV,
    BandData,
    _float,
    _read_lines,
    read_openmx_output,
    read_vasp_outcar,
    read_vasprun,
)

KBAR_EV_ANGSTROM3 = 1e8 * 1e-30 / elementary_charge


@dataclass(frozen=True)
class CalculationResult:
    arrays: dict[str, np.ndarray]
    summary: dict
    bands: BandData | None = None

    def __post_init__(self):
        if any(not np.isfinite(np.asarray(value)).all() for value in self.arrays.values()):
            raise ValueError("calculation result arrays must be finite")
        self.as_result()

    def as_result(self):
        dimensions: dict[str, tuple[str, ...]] = {}
        units: dict[str, str] = {}
        for key, value in self.arrays.items():
            if key in {"forces", "positions"}:
                dimensions[key] = ("atom", "cartesian")
                units[key] = "eV/angstrom" if key == "forces" else "angstrom"
            elif key in {"stress", "cell"}:
                dimensions[key] = ("cartesian", "cartesian2")
                units[key] = "eV/angstrom^3" if key == "stress" else "angstrom"
            elif key == "atomic_numbers":
                dimensions[key], units[key] = ("atom",), "1"
            else:
                dimensions[key] = () if np.asarray(value).ndim == 0 else ("step",)
                units[key] = "eV" if "energy" in key else "1"
        return ObservableResult(self.arrays, dimensions, units, self.summary)


def _stress_matrix(values):
    xx, yy, zz, xy, yz, zx = values
    return np.asarray([[xx, xy, zx], [xy, yy, yz], [zx, yz, zz]])


def read_vasp_results(path):
    """Read OUTCAR energies/forces/stress/convergence, with optional band data.

    VASP prints compressive stress in kbar. Returned stress is tensile and
    uses eV/angstrom³. ``energy_eV`` prefers its sigma-to-zero energy; the
    free energy used for finite-temperature force derivatives stays separate.
    """
    lines = _read_lines(path)
    energies, free_energies = [], []
    arrays = {}
    electronic = ionic = None
    electronic_provenance = ionic_provenance = None
    nions = None
    has_bands = False
    for i, line in enumerate(lines):
        match = re.search(r"NIONS\s*=\s*(\d+)", line)
        if match:
            nions = int(match.group(1))
        match = re.search(r"free\s+energy\s+TOTEN\s*=\s*(" + _NUMBER + ")", line)
        if match:
            free_energies.append(_float(match.group(1)))
        match = re.search(r"energy\(sigma->0\)\s*=\s*(" + _NUMBER + ")", line)
        if match:
            energies.append(_float(match.group(1)))
        match = re.search(r"Iteration\s+(\d+)\(\s*(\d+)\)", line)
        if match and int(match.group(2)) == 1:
            electronic = False
            electronic_provenance = {
                "line": i + 1,
                "marker": line.strip(),
                "criterion": "electronic loop started",
            }
        if "EDIFF is reached" in line:
            electronic = True
            electronic_provenance = {
                "line": i + 1,
                "marker": line.strip(),
                "criterion": "explicit EDIFF stop",
            }
        if "reached required accuracy" in line.lower() and "stopping" in line.lower():
            ionic = True
            ionic_provenance = {"line": i + 1, "marker": line.strip(), "criterion": "explicit ionic stop"}
        if "TOTAL-FORCE" in line.upper() and "EV/ANGST" in line.upper():
            rows, cursor = [], i + 1
            while cursor < len(lines) and (not lines[cursor].strip() or set(lines[cursor].strip()) == {"-"}):
                cursor += 1
            while cursor < len(lines):
                fields = lines[cursor].split()
                if len(fields) != 6:
                    break
                try:
                    rows.append([_float(t) for t in fields])
                except ValueError:
                    break
                cursor += 1
            if not rows or (nions is not None and len(rows) != nions):
                raise ValueError(f"{path}: incomplete OUTCAR position/force block")
            values = np.asarray(rows)
            arrays["positions"], arrays["forces"] = values[:, :3], values[:, 3:]
        if "direct lattice vectors" in line and "reciprocal lattice vectors" in line:
            try:
                cell = np.asarray([[_float(t) for t in row.split()[:3]] for row in lines[i + 1 : i + 4]])
                if cell.shape != (3, 3):
                    raise ValueError("truncated lattice")
                arrays["cell"] = cell
            except ValueError as error:
                raise ValueError(f"{path}: malformed OUTCAR cell") from error
        if re.match(r"\s*in\s+kB\s+", line):
            fields = line.split()[2:]
            if len(fields) != 6:
                raise ValueError(f"{path}: malformed OUTCAR stress tensor")
            arrays["stress"] = -_stress_matrix([_float(t) for t in fields]) * KBAR_EV_ANGSTROM3
        if "band" in line.lower() and "occupation" in line.lower():
            has_bands = True
    if free_energies:
        arrays["free_energy_eV"] = np.asarray(free_energies[-1])
        arrays["free_energy_history_eV"] = np.asarray(free_energies)
    if energies or free_energies:
        history = energies or free_energies
        arrays["energy_eV"], arrays["energy_history_eV"] = np.asarray(history[-1]), np.asarray(history)
    if not arrays and not has_bands:
        raise ValueError(f"{path}: no VASP calculation results")
    summary = {
        "source_format": "vasp_outcar",
        "electronic_converged": electronic,
        "ionic_converged": ionic,
        "steps": max(len(energies), len(free_energies)),
        "electronic_convergence_provenance": electronic_provenance,
        "ionic_convergence_provenance": ionic_provenance,
        "energy_definition": "sigma_to_zero" if energies else "free_energy",
        "stress_definition": "tensile Cartesian eV/angstrom^3; negative of VASP compressive kbar",
    }
    return CalculationResult(arrays, summary, read_vasp_outcar(path) if has_bands else None)


def read_vasprun_results(path):
    """Stream final XML calculation arrays; eigenvalues are optional."""
    arrays, history, free_history = {}, [], []
    nelm = nsw = ediffg = None
    explicit_electronic = explicit_ionic = None
    electronic = ionic = None
    steps, has_bands, last_forces = 0, False, None
    try:
        for _, element in ET.iterparse(path, events=("end",)):
            name = element.attrib.get("name")
            if element.tag == "i" and name in {
                "NELM",
                "NSW",
                "EDIFFG",
                "electronic_converged",
                "ionic_converged",
            }:
                token = element.text.strip()
                if name == "NELM":
                    nelm = int(token)
                elif name == "NSW":
                    nsw = int(token)
                elif name == "EDIFFG":
                    ediffg = _float(token)
                elif name == "electronic_converged":
                    explicit_electronic = token.lower() in {"true", "t", ".true."}
                else:
                    explicit_ionic = token.lower() in {"true", "t", ".true."}
            elif element.tag == "eigenvalues":
                has_bands = True
                element.clear()
            elif element.tag == "calculation":
                steps += 1
                record = element.find("energy")
                energies = (
                    {}
                    if record is None
                    else {
                        row.attrib["name"]: _float(row.text.strip())
                        for row in record.findall("i")
                        if row.attrib.get("name") in {"e_0_energy", "e_fr_energy", "e_wo_entrp"}
                    }
                )
                if "e_fr_energy" in energies:
                    free_history.append(energies["e_fr_energy"])
                if energies:
                    total_energy = energies.get(
                        "e_0_energy", energies.get("e_wo_entrp", energies.get("e_fr_energy"))
                    )
                    assert total_energy is not None
                    history.append(total_energy)
                for key in ("forces", "stress"):
                    varray = element.find(f"varray[@name='{key}']")
                    if varray is not None:
                        value = np.asarray(
                            [[_float(t) for t in row.text.split()] for row in varray.findall("v")]
                        )
                        if (
                            value.ndim != 2
                            or value.shape[1] != 3
                            or (key == "stress" and value.shape != (3, 3))
                        ):
                            raise ValueError(f"invalid XML {key} dimensions")
                        arrays[key] = value if key == "forces" else -value * KBAR_EV_ANGSTROM3
                        if key == "forces":
                            last_forces = value
                structure = element.find("structure")
                if structure is not None:
                    basis = structure.find("crystal/varray[@name='basis']")
                    positions = structure.find("varray[@name='positions']")
                    if basis is not None:
                        cell = np.asarray(
                            [[_float(t) for t in row.text.split()] for row in basis.findall("v")]
                        )
                        if cell.shape != (3, 3):
                            raise ValueError("invalid XML cell dimensions")
                        arrays["cell"] = cell
                        if positions is not None:
                            fractional = np.asarray(
                                [[_float(t) for t in row.text.split()] for row in positions.findall("v")]
                            )
                            if fractional.ndim != 2 or fractional.shape[1] != 3:
                                raise ValueError("invalid XML positions")
                            arrays["positions"] = fractional @ cell
                electronic_steps = len(element.findall("scstep"))
                if nelm is not None and electronic_steps:
                    electronic = electronic_steps < nelm
                element.clear()
    except (ET.ParseError, ValueError, AttributeError) as error:
        raise ValueError(f"{path}: malformed vasprun result XML ({error})") from error
    if history:
        arrays["energy_eV"], arrays["energy_history_eV"] = np.asarray(history[-1]), np.asarray(history)
    if free_history:
        arrays["free_energy_eV"], arrays["free_energy_history_eV"] = (
            np.asarray(free_history[-1]),
            np.asarray(free_history),
        )
    ionic_criterion = None
    if nsw is not None and nsw > 0 and ediffg is not None:
        if ediffg < 0 and last_forces is not None:
            ionic = bool(np.max(np.linalg.norm(last_forces, axis=1)) < -ediffg)
            ionic_criterion = "inferred unconstrained force norms against EDIFFG"
        elif ediffg > 0 and len(free_history or history) > 1:
            ionic_history = free_history or history
            ionic = abs(ionic_history[-1] - ionic_history[-2]) < ediffg
            ionic_criterion = (
                "inferred final free-energy change against EDIFFG"
                if free_history
                else "inferred final energy change against EDIFFG"
            )
    if explicit_electronic is not None:
        electronic = explicit_electronic
    if explicit_ionic is not None:
        ionic = explicit_ionic
    if not arrays and not has_bands:
        raise ValueError(f"{path}: no vasprun calculation results")
    return CalculationResult(
        arrays,
        {
            "source_format": "vasp_vasprun",
            "steps": steps,
            "electronic_converged": electronic,
            "ionic_converged": ionic,
            "electronic_convergence_provenance": "explicit XML flag"
            if explicit_electronic is not None
            else ("inferred final SCF step count below NELM" if electronic is not None else None),
            "ionic_convergence_provenance": "explicit XML flag"
            if explicit_ionic is not None
            else ionic_criterion,
            "energy_definition": "sigma_to_zero_or_reported_energy",
            "stress_definition": "tensile Cartesian eV/angstrom^3; negative of VASP compressive kbar",
        },
        read_vasprun(path) if has_bands else None,
    )


def read_openmx_results(path):
    """Read OpenMX energy/convergence and explicitly unit-labeled force blocks."""
    lines = _read_lines(path)
    summary = read_openmx_output(path)
    arrays = {}
    if summary["total_energy_eV"] is not None:
        arrays["energy_eV"] = np.asarray(summary["total_energy_eV"])
        arrays["energy_history_eV"] = np.asarray(summary["total_energy_history_eV"])
    atom_counts = re.findall(r"Atoms\.Number\s+(\d+)", "\n".join(lines), re.I)
    nions = int(atom_counts[-1]) if atom_counts else None
    for i, line in enumerate(lines):
        header = line + (" " + lines[i + 1] if i + 1 < len(lines) else "")
        if "force" in line.lower() and re.search(r"Hartree\s*/\s*Bohr", header, re.I):
            rows, positions, species, cursor = [], [], [], i + 1
            while cursor < len(lines) and (
                not lines[cursor].strip()
                or set(lines[cursor].strip()) <= {"*", "-"}
                or re.fullmatch(r"\s*\(?Hartree\s*/\s*Bohr\)?\s*", lines[cursor], re.I)
            ):
                cursor += 1
            while cursor < len(lines):
                fields = lines[cursor].split()
                if len(fields) < 4 or not fields[0].isdigit():
                    break
                try:
                    rows.append([_float(t) for t in fields[-3:]])
                    if re.search(r"xyz-coordinate\s*\(Ang", header, re.I):
                        if len(fields) < 8:
                            raise ValueError("missing coordinate columns")
                        positions.append([_float(t) for t in fields[-6:-3]])
                        species.append(fields[1])
                except ValueError:
                    break
                cursor += 1
            if not rows or (nions is not None and len(rows) != nions):
                raise ValueError(f"{path}: incomplete OpenMX force block")
            arrays["forces"] = np.asarray(rows) * HARTREE_EV / BOHR_ANGSTROM
            if positions:
                if len(positions) != len(rows):
                    raise ValueError(f"{path}: incomplete OpenMX coordinate block")
                arrays["positions"] = np.asarray(positions)
                summary["species_labels"] = species
                if all(symbol in ATOMIC_NUMBERS for symbol in species):
                    arrays["atomic_numbers"] = np.asarray([ATOMIC_NUMBERS[symbol] for symbol in species])
    cell_records: dict[int, list[float]] = {}
    cell_pattern = re.compile(
        r"a([123])\s*=\s*("
        + _NUMBER
        + r")\s+("
        + _NUMBER
        + r")\s+("
        + _NUMBER
        + r")\s+dE/da[123]\s*=\s*("
        + _NUMBER
        + r")\s+("
        + _NUMBER
        + r")\s+("
        + _NUMBER
        + ")"
    )
    for line in lines:
        match = cell_pattern.search(line)
        if match:
            if int(match.group(1)) == 1:
                cell_records = {}
            cell_records[int(match.group(1))] = [_float(t) for t in match.groups()[1:]]
    summary["stress_definition"] = None
    summary["electronic_converged"] = summary["scf_converged"]
    summary["ionic_converged"] = None
    if cell_records and len(cell_records) != 3:
        raise ValueError(f"{path}: incomplete final OpenMX cell-gradient block")
    if len(cell_records) == 3:
        records = np.asarray([cell_records[i] for i in (1, 2, 3)])
        cell, derivative = records[:, :3], records[:, 3:] * HARTREE_EV / BOHR_ANGSTROM
        volume = abs(np.linalg.det(cell))
        if volume <= 1e-14:
            raise ValueError(f"{path}: singular OpenMX cell-gradient lattice")
        tensor = cell.T @ derivative / volume
        arrays["cell"], arrays["stress"] = cell, (tensor + tensor.T) / 2
        summary["stress_definition"] = "tensile strain derivative from reported cell-vector gradients"
        summary["stress_antisymmetric_residual_eV_per_angstrom3"] = float(np.max(abs(tensor - tensor.T)))
    if not arrays:
        raise ValueError(f"{path}: no supported OpenMX energy/force records")
    return CalculationResult(arrays, summary)
