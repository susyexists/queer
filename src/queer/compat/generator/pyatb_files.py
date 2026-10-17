"""PyATB's run-directory files: readers and writers written from their observed layouts.

A PyATB run directory holds the ``Input`` file, ``inputs.sha256`` (written by
the Generator before PyATB starts: ``sha256sum scf.scfout Input``) and ``Out/``:
``input.json`` (the resolved input echo, every default filled in),
``running.log``, the product of the function and, for an adaptive integral,
``Out/AHC/integraton.log`` (sic). The layouts are:

- ``Out/AHC/ahc.dat`` of the direct/Kubo engine: one line,
  ``AHC is [sigma_yz, sigma_zx, sigma_xy] S/cm`` with six decimals;
- ``Out/AHC/ahc.dat`` of the FHS engine: two ``#`` header lines and one
  ``%14.6f`` row ``E sigma_yz sigma_zx sigma_xy`` per energy, ``E`` relative to
  the Fermi energy (eV);
- ``Out/Chern_Num/chern_number.dat``: one line,
  ``Chern number is <value> by using the <Grid|Adaptive> integral`` with the
  value at full precision (its shortest round-trip form);
- ``Out/Shift_Current/shift_current.dat``: the header ``#`` ``omega(eV)``, the 18
  components and ``(uA/V^2)`` (``%1s%10s%14s``, 17 x ``%15s``, ``%16s``), then one
  row per photon energy ``start + i * domega``: ``%10.5f`` and 18 x ``%15.6e``,
  the components ``a(bc)`` with ``b <= c`` in the order ``xxx, xxy, xxz, xyy,
  xyz, xzz, yxx, ..., zzz``;
- ``integraton.log``: per generation a header, the region count and the line
  ``<error> <value>`` (NumPy prints of the raw integrand, before the
  conductivity prefactor), then ``integration complete!`` and the final pair.

Everything here is independent of PyATB's source; only the layouts are shared.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

INPUT_FILENAME = "Input"
INPUTS_SHA256_FILENAME = "inputs.sha256"
INPUT_ECHO = "Out/input.json"
RUNNING_LOG = "Out/running.log"
AHC_FILE = "Out/AHC/ahc.dat"
CHERN_FILE = "Out/Chern_Num/chern_number.dat"
SHIFT_CURRENT_FILE = "Out/Shift_Current/shift_current.dat"
INTEGRATION_LOG = "Out/AHC/integraton.log"
CHERN_INTEGRATION_LOG = "Out/Chern_Num/integraton.log"
AHC_COMPONENTS: tuple[str, ...] = ("yz", "zx", "xy")
# The independent shift-current components a(bc), b <= c, in PyATB's column order.
SHIFT_COMPONENTS: tuple[str, ...] = tuple(
    a + b + c for a in "xyz" for b, c in ("xx", "xy", "xz", "yy", "yz", "zz")
)
SHIFT_UNIT = "(uA/V^2)"
FHS_HEADER = (
    "# FHS Berry-plaquette AHC (S/cm)",
    "# E(eV)        sigma_yz       sigma_zx       sigma_xy",
)

_NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|[+-]?(?:nan|inf)"
_AHC_LINE = re.compile(r"AHC\s+is\s*\[([^\]]*)\]\s*(\S+)?")
_CHERN_LINE = re.compile(rf"Chern number is\s+({_NUMBER})\s+by using the\s+(\w+)\s+integral", re.IGNORECASE)
_FERMI_LINE = re.compile(rf">>\s*fermi_energy\s*(?:\(eV\))?\s*:\s*({_NUMBER})")
_REFINED = re.compile(r"!!The number of k points for adaptive refinement\s*:\s*(\d+)")
_ROUGH = re.compile(r"!!The number of rough k-grid\s*:\s*(\d+)")
_PROCESSES = re.compile(r"processor number is\s*(\d+)")
_GENERATION = re.compile(r"^\s*(\d+)\s+generation\b")
_SHA256_LINE = re.compile(r"^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$")


# ---------------------------------------------------------------------------
# Input and its echo
# ---------------------------------------------------------------------------


def _token_value(token: str) -> Any:
    for kind in (int, float):
        try:
            return kind(token)
        except ValueError:
            continue
    return token


def _entry_value(tokens: Sequence[str]) -> Any:
    values = [_token_value(token) for token in tokens]
    if not values:
        return None
    if all(isinstance(value, (int, float)) for value in values):
        return values[0] if len(values) == 1 else values
    return " ".join(tokens)


def parse_input(text: str) -> dict[str, dict[str, Any]]:
    """The blocks of a PyATB ``Input``: ``{block: {key: value}}``.

    A block is ``NAME`` then ``{ ... }``; each entry is ``key value...``, and lines
    of numbers continue the previous entry (a lattice's vectors). Numbers become
    int or float, several of them a list; anything else stays text. ``#`` starts
    a comment. Raises ``ValueError`` for an unbalanced block.
    """
    blocks: dict[str, dict[str, Any]] = {}
    pending: str | None = None
    current: dict[str, list[str]] | None = None
    order: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if current is None:
            if line == "{":
                if pending is None:
                    raise ValueError("a PyATB block opens without a name")
                current, order = {}, []
                continue
            name, _, rest = line.partition("{")
            if rest.strip():
                raise ValueError(f"unexpected text after {name.strip()!r}")
            pending = name.strip()
            if "{" in line:
                current, order = {}, []
            continue
        if line == "}":
            if pending is None:
                raise ValueError("a PyATB block closes without a name")
            blocks[pending] = {key: _entry_value(current[key]) for key in order}
            pending, current = None, None
            continue
        tokens = line.split()
        if order and all(isinstance(_token_value(token), (int, float)) for token in tokens):
            current[order[-1]].extend(tokens)
            continue
        key = tokens[0]
        if key not in current:
            order.append(key)
        current[key] = list(tokens[1:])
    if current is not None:
        raise ValueError(f"the PyATB block {pending!r} is not closed")
    return blocks


def read_input(path: str | Path) -> dict[str, dict[str, Any]]:
    """:func:`parse_input` of a file."""
    return parse_input(Path(path).read_text(encoding="utf-8", errors="replace"))


def read_input_echo(path: str | Path) -> dict[str, Any]:
    """``Out/input.json``: PyATB's resolved input, every default filled in."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return value


def parse_inputs_sha256(text: str) -> dict[str, str]:
    """``sha256sum`` lines as ``{file name: lowercase digest}``."""
    digests = {}
    for line in text.splitlines():
        match = _SHA256_LINE.match(line)
        if match:
            digests[match.group(2)] = match.group(1).lower()
    return digests


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------


def _floats(text: str) -> list[float]:
    return [float(token) for token in re.findall(_NUMBER, text.replace(",", " "))]


def parse_ahc(text: str) -> dict[str, Any]:
    """``Out/AHC/ahc.dat``: ``{"format": "vector", "sigma": [3], "unit"}`` or a scan table.

    A scan is ``{"format": "scan", "energy_eV": [...], "sigma": [[3], ...], "unit":
    "S/cm"}`` with the energies relative to E_F. Raises ``ValueError`` when the
    text is neither.
    """
    match = _AHC_LINE.search(text)
    if match:
        values = _floats(match.group(1))
        if len(values) != 3:
            raise ValueError(f"expected 3 AHC components, found {len(values)}")
        return {"format": "vector", "sigma": values, "unit": match.group(2) or "S/cm"}
    energies, rows = [], []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values = _floats(stripped)
        if len(values) != 4:
            raise ValueError(f"an AHC scan row needs 4 numbers: {stripped!r}")
        energies.append(values[0])
        rows.append(values[1:])
    if not rows:
        raise ValueError("no AHC value in the file")
    return {"format": "scan", "energy_eV": energies, "sigma": rows, "unit": "S/cm"}


def parse_chern(text: str) -> dict[str, Any]:
    """``Out/Chern_Num/chern_number.dat``: ``{"value", "integrate_mode"}``."""
    match = _CHERN_LINE.search(text)
    if not match:
        raise ValueError("no 'Chern number is ... by using the ... integral' line")
    return {"value": float(match.group(1)), "integrate_mode": match.group(2)}


def parse_shift_current(text: str) -> dict[str, Any]:
    """``Out/Shift_Current/shift_current.dat``: ``{"omega_eV", "components", "sigma", "unit"}``.

    ``sigma`` holds one row of the 18 components per photon energy, in the
    header's order, which must be :data:`SHIFT_COMPONENTS`. Raises ``ValueError``
    for another header, a short row or no row.
    """
    lines = text.splitlines()
    header = next((line for line in lines if line.strip().startswith("#")), None)
    if header is None:
        raise ValueError("no shift-current header line")
    names = header.replace("#", " ").split()
    if len(names) != 20 or names[0] != "omega(eV)" or tuple(names[1:19]) != SHIFT_COMPONENTS:
        raise ValueError(f"the shift-current header is not PyATB's: {header.strip()!r}")
    omega, rows = [], []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values = _floats(stripped)
        if len(values) != 19:
            raise ValueError(f"a shift-current row needs 19 numbers: {stripped!r}")
        omega.append(values[0])
        rows.append(values[1:])
    if not rows:
        raise ValueError("no shift-current row in the file")
    return {"omega_eV": omega, "components": list(SHIFT_COMPONENTS), "sigma": rows, "unit": names[19]}


def parse_integration_log(text: str) -> dict[str, Any]:
    """``integraton.log``: the last generation's error and value (raw integrand units).

    Returns ``{"generations", "complete", "error": [..], "value": [..]}``; the
    error and value of the last ``<error> <value>`` line, scalar or one entry per
    integrated function.
    """
    generations = 0
    complete = False
    last: tuple[list[float], list[float]] | None = None
    # NumPy wraps a long array over lines; rejoin what lies inside brackets.
    depth, joined = 0, []
    for character in text:
        depth += (character == "[") - (character == "]")
        joined.append(" " if character == "\n" and depth > 0 else character)
    for line in "".join(joined).splitlines():
        if _GENERATION.match(line):
            generations += 1
            continue
        if "integration complete" in line:
            complete = True
            continue
        if line.startswith(("use self", "start integration", "integration region")):
            continue
        groups = re.findall(r"\[([^\]]*)\]", line)
        outside = _floats(re.sub(r"\[[^\]]*\]", " ", line))
        if len(groups) == 2 and not outside:
            last = (_floats(groups[0]), _floats(groups[1]))
        elif len(groups) == 1 and len(outside) == 1 and line.strip().endswith("]"):
            # An early generation prints its error as one number before the values.
            last = (outside, _floats(groups[0]))
        elif not groups and len(outside) == 2:
            last = ([outside[0]], [outside[1]])
    if last is None:
        raise ValueError("no error/value line in the integration log")
    return {"generations": generations, "complete": complete, "error": last[0], "value": last[1]}


def parse_running_log(text: str) -> dict[str, Any]:
    """The facts of ``Out/running.log`` a comparison uses.

    ``fermi_energy_eV`` (the last value printed, with ``fermi_energy_decimals``),
    ``rough_kpoints`` and ``refined_kpoints`` of a Grid integral, ``processes``, the
    printed ``ahc`` vector or ``chern`` value, and ``completed`` (the time
    statistics were reached).
    """
    facts: dict[str, Any] = {"completed": "Total time" in text}
    fermi = _FERMI_LINE.findall(text)
    if fermi:
        token = fermi[-1]
        facts["fermi_energy_eV"] = float(token)
        facts["fermi_energy_decimals"] = len(token.split(".", 1)[1]) if "." in token else 0
    for name, pattern in (
        ("rough_kpoints", _ROUGH),
        ("refined_kpoints", _REFINED),
        ("processes", _PROCESSES),
    ):
        found = pattern.findall(text)
        if found:
            facts[name] = int(found[-1])
    ahc = _AHC_LINE.search(text)
    if ahc:
        facts["ahc"] = _floats(ahc.group(1))
    chern = _CHERN_LINE.search(text)
    if chern:
        facts["chern"] = float(chern.group(1))
    return facts


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _vector(values: Any) -> list[float]:
    vector = [float(value) for value in values]
    if len(vector) != 3:
        raise ValueError("an AHC vector has three components (yz, zx, xy)")
    return vector


def ahc_vector_text(sigma: Any) -> str:
    """The direct/Kubo ``ahc.dat`` line of ``[sigma_yz, sigma_zx, sigma_xy]`` (S/cm)."""
    yz, zx, xy = _vector(sigma)
    return f"AHC is [{yz:.6f}, {zx:.6f}, {xy:.6f}] S/cm\n"


def ahc_scan_text(energies: Any, sigma: Any) -> str:
    """The FHS ``ahc.dat`` table: E relative to E_F (eV) and the three components (S/cm)."""
    rows = [_vector(row) for row in sigma]
    offsets = [float(value) for value in energies]
    if len(rows) != len(offsets):
        raise ValueError("one AHC row per energy")
    lines = [*FHS_HEADER]
    lines += [f"{e:14.6f} {yz:14.6f} {zx:14.6f} {xy:14.6f}" for e, (yz, zx, xy) in zip(offsets, rows)]
    return "\n".join(lines) + "\n"


def chern_text(value: Any, integrate_mode: str = "Grid") -> str:
    """The ``chern_number.dat`` line, the value at full precision."""
    if integrate_mode not in {"Grid", "Adaptive"}:
        raise ValueError("integrate_mode must be Grid or Adaptive")
    number = float(value)
    return f"Chern number is {number!r} by using the {integrate_mode} integral\n"


def shift_current_text(omega: Any, sigma: Any, note: str | None = None) -> str:
    """The ``shift_current.dat`` table: one row of the 18 components (uA/V^2) per photon energy (eV).

    ``sigma`` is a sequence of rows or a mapping of the 18 component names to
    their columns. ``note`` adds a comment line after PyATB's header (PyATB writes
    none), which readers of PyATB's layout skip.
    """
    if isinstance(sigma, Mapping):
        columns = [[float(value) for value in sigma[name]] for name in SHIFT_COMPONENTS]
        rows = [list(row) for row in zip(*columns)]
    else:
        rows = [[float(value) for value in row] for row in sigma]
    energies = [float(value) for value in omega]
    if len(rows) != len(energies) or any(len(row) != len(SHIFT_COMPONENTS) for row in rows):
        raise ValueError("one row of 18 shift-current components per photon energy")
    widths = (14,) + (15,) * (len(SHIFT_COMPONENTS) - 1)
    header = (
        "#"
        + f"{'omega(eV)':>10}"
        + "".join(f"{name:>{width}}" for name, width in zip(SHIFT_COMPONENTS, widths))
    )
    lines = [header + f"{SHIFT_UNIT:>16}"]
    if note:
        lines.append("# " + " ".join(note.split()))
    lines += [
        f"{energy:10.5f}" + "".join(f"{value:15.6e}" for value in row) for energy, row in zip(energies, rows)
    ]
    return "\n".join(lines) + "\n"


def write_text(path: str | Path, text: str) -> Path:
    """Write ``text`` to ``path``, creating its directory."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def vector_from_tensor(tensor: Any) -> list[float]:
    """``[t_yz, t_zx, t_xy]`` of an antisymmetric 3 x 3 Hall tensor."""
    rows = [[float(value) for value in row] for row in tensor]
    return [rows[1][2], rows[2][0], rows[0][1]]


def finite(values: Mapping[str, Any] | Sequence[Any] | float) -> bool:
    """Whether every number in a (nested) value is finite."""
    if isinstance(values, Mapping):
        return all(finite(value) for value in values.values())
    if isinstance(values, (list, tuple)):
        return all(finite(value) for value in values)
    return isinstance(values, (int, float)) and math.isfinite(float(values))
