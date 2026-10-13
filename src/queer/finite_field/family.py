"""Families of band models indexed by the displacement field D (one wannier90 model per D).

A manifest (JSON) lists the members:
    {"members": [{"label": "...", "D": <float>, "files": {"hr": "...", "tb": "...", "centres": "...",
                  "wout": "...", "wsvec": "...", "grid": "..."}}, ...], "n_occ": <int, optional>}
(paths relative to the manifest; a top-level list of members, or the key "models" as written by
lattice_reg.write_synthetic_family, is also accepted). Members load lazily through picklable loaders
(WannierLoader, GridLoader), so maps run in worker processes without shipping large H(R) arrays.
Loader fingerprints hash the file contents (sha256), so harmonic-map caches survive restarts and
notice regenerated files. "grid" names a GriddedGeometry NPZ (geometry/gridded.py, model frame):
maps use it in place of the Wannier model, at cfg.drive.angle. The Gamma-point checks need the
Wannier files (n_occ is required only for those). At most `max_models` loaded models are kept
(LRU); the Gamma-point data of all members are small and cached.

Critical field D_c (`critical_D_report`): the Gamma-point v/c character is tracked member by member,
outward from a reference member, in the orbital basis of the first member (relabelled orbitals,
recognized by their Wannier centres, are mapped back first). At each member the v-like (c-like)
level is the one with the largest weight |<u_v|u_n>|^2 (|<u_c|u_n>|^2) on the previous member's v
(c) state; a weight <= 0.5 flags 'character_lost' and leaves the member out. The signed gap
g = eps(c-like) - eps(v-like) is positive on the reference side and changes sign at a band inversion.
A clean character swap without mixing identifies a band inversion. Otherwise a sign change
counts as a gap closing only if |g| closes like a V: the straight lines through the
two nearest members on each side must reach zero inside the bracket, within jump_tol of the largest
member spacing of each other (an avoided crossing or a relabelling makes g jump instead); a sign
change that cannot be checked on either side while g changes sign again at the next member is
rejected too. Members whose v/c character sits on bands other than band_pair() are flagged
'remote_band'. Roots are located on the cubic spline through the same-sign members on both sides; a
degenerate v/c level at a member is a root at that member. Consecutive members must overlap (weight
> 0.5): refine the D grid where the Gamma states rotate quickly, e.g. around D = 0, where the layer
polarization of the surface states flips (four-band model: VL = -0.5 VLc -> +0.5 VLc rotates them by 45 deg).
"""

from __future__ import annotations

import itertools
import json
import math
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..capabilities import CapabilityError
from ..io.artifacts import digest_file
from ..observables.result import ObservableResult
from ._storage import json_safe, stable_hash
from .native import FieldPlane, NativePlaneModel


def file_digest(path):
    return digest_file(Path(path))


WANNIER_FILES = ("hr", "tb", "centres", "wout", "wsvec", "poscar")
GRID_FILE = "grid"
_STAY = 0.25  # A: bonus for keeping an orbital's index in match_orbitals (co-located orbitals never swap)


# ---- loaders ------------------------------------------------------------------------------------------
class WannierLoader:
    """Load a native QUEER Wannier model and its reciprocal-plane view lazily."""

    kind = "wannier"

    def __init__(self, files, n_occ=None, gauge=None, plane=None, band_pair=None, lattice=None):
        self.files = {k: str(v) for k, v in files.items() if k in WANNIER_FILES and v}
        self.n_occ = n_occ
        self.gauge = gauge
        self.plane = FieldPlane.from_mapping(plane)
        self.band_pair = band_pair
        self.lattice = lattice

    def __call__(self):
        from ..models.wannier import WannierModel

        if not (self.files.get("hr") or self.files.get("tb")):
            raise ValueError("a family Wannier member requires hr or tb")
        root = Path(next(iter(self.files.values()))).parent
        model = WannierModel.from_directory(
            root,
            hr=self.files.get("hr"),
            tb=self.files.get("tb"),
            centres=self.files.get("centres"),
            wout=self.files.get("wout"),
            poscar=self.files.get("poscar"),
            wsvec=self.files.get("wsvec"),
            interpolation="ws_distance" if "wsvec" in self.files else "standard",
        )
        if model.g_vec is None and self.lattice is not None:
            from ..geometry.lattice import reciprocal_lattice

            model.lattice = np.asarray(self.lattice, dtype=float)
            model.g_vec = reciprocal_lattice(self.lattice)
        return NativePlaneModel(model, plane=self.plane, band_pair=self.band_pair, occupied_bands=self.n_occ)

    def fingerprint(self):
        return stable_hash(
            {
                "loader": self.kind,
                "n_occ": self.n_occ,
                "gauge": self.gauge,
                "plane": self.plane.describe(),
                "band_pair": self.band_pair,
                "lattice": self.lattice,
            },
            {k: file_digest(v) for k, v in sorted(self.effective_files().items())},
        )

    def effective_files(self):
        """Native sidecar inventory without loading Hamiltonian/position matrices."""
        from ..io.wannier import resolve_wannier_files

        root = Path(next(iter(self.files.values()))).parent
        return resolve_wannier_files(
            root,
            hr=self.files.get("hr"),
            tb=self.files.get("tb"),
            centres=self.files.get("centres"),
            wout=self.files.get("wout"),
            poscar=self.files.get("poscar"),
            wsvec=self.files.get("wsvec"),
            interpolation="ws_distance" if "wsvec" in self.files else "standard",
        )


class OpenMXLoader:
    """OpenMX response sweeps retain their generalized native operators."""

    kind = "openmx"

    def __init__(self, files, n_occ=None, plane=None, band_pair=None):
        self.files = {k: str(v) for k, v in files.items() if k in {"scfout", "openmx_input", "poscar"}}
        self.n_occ, self.band_pair = n_occ, band_pair
        self.plane = FieldPlane.from_mapping(plane)

    def __call__(self):
        from ..models.openmx import OpenMXModel

        model = OpenMXModel.from_scfout(self.files["scfout"], openmx_input=self.files.get("openmx_input"))
        return NativePlaneModel(model, plane=self.plane, band_pair=self.band_pair, occupied_bands=self.n_occ)

    def fingerprint(self):
        return stable_hash(
            {
                "loader": self.kind,
                "n_occ": self.n_occ,
                "band_pair": self.band_pair,
                "plane": self.plane.describe(),
            },
            {k: file_digest(v) for k, v in sorted(self.files.items())},
        )


class GridLoader:
    """Picklable zero-argument loader of a GriddedGeometry NPZ evaluated at the field angle `angle` (rad).

    Grids are stored in the model frame and serve any field direction; ModelFamily.harmonic_map uses
    with_angle(cfg.drive.angle). fingerprint(): the angle and the sha256 of the file.
    """

    kind = "gridded-npz"

    def __init__(self, path, angle: float = 0.0):
        self.path = str(path)
        self.angle = float(angle)

    def __call__(self):
        from .geometry.gridded import GriddedGeometry

        return GriddedGeometry.from_npz(self.path, angle=self.angle)

    def with_angle(self, angle: float) -> "GridLoader":
        return GridLoader(self.path, angle)

    def fingerprint(self) -> str:
        return stable_hash({"loader": self.kind, "version": 1, "angle": self.angle}, file_digest(self.path))

    def __repr__(self) -> str:
        return f"GridLoader({self.path!r}, angle={self.angle!r})"


@dataclass
class FamilyMember:
    label: str
    D: float
    files: dict
    loader: object  # zero-argument picklable callable -> band model or geometry provider (used by maps)
    extra: dict = field(default_factory=dict)
    model_loader: object = None  # -> band model for the Gamma-point analysis; None: `loader` (unless a grid)

    def load(self):
        return self.loader()

    @property
    def has_model(self) -> bool:
        return self.model_loader is not None or not isinstance(self.loader, GridLoader)

    def load_model(self):
        if not self.has_model:
            raise ValueError(
                f"member {self.label!r} is a precomputed grid ({self.loader!r}) without a Hamiltonian"
            )
        return (self.model_loader if self.model_loader is not None else self.loader)()


@dataclass
class GammaData:
    """Gamma-point data of one member: eigenvalues (eV), eigenvectors of bands lo.. (orbital basis)."""

    n_orb: int
    band_pair: tuple
    eps: np.ndarray  # (n_orb,)
    lo: int
    U: np.ndarray  # (n_orb, n_window) complex: bands lo, lo + 1, ...
    centres: np.ndarray | None  # (n_orb, 3), A
    lattice: np.ndarray | None  # (3, 3) rows, A
    fractional_origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    relative_eps: np.ndarray | None = None
    energy_reference: float = 0.0
    assembly_precision_floor: float = 0.0


def _relative_energies(data):
    return data.eps if data.relative_eps is None else data.relative_eps


# ---- orbital matching -------------------------------------------------------------------------------
def _min_image(d: np.ndarray, lattice) -> np.ndarray:
    """Shortest lattice-equivalent vectors of the displacements d (..., 3), A (unchanged without a lattice)."""
    if lattice is None:
        return d
    L = np.asarray(lattice, dtype=float)
    try:
        f = d @ np.linalg.inv(L)
    except np.linalg.LinAlgError:
        return d
    f = f - np.rint(f)
    best = f @ L
    norm = np.linalg.norm(best, axis=-1)
    for off in itertools.product((-1.0, 0.0, 1.0), repeat=3):
        if any(off):
            cand = (f + np.array(off)) @ L
            cn = np.linalg.norm(cand, axis=-1)
            better = cn < norm
            best = np.where(better[..., None], cand, best)
            norm = np.where(better, cn, norm)
    return best


def match_orbitals(ref_centres, centres, lattice=None, stay: float = _STAY):
    """Orbital correspondence from Wannier centres: perm[j] is the reference orbital of orbital j.

    Distances are taken in 3D, modulo the lattice rows (minimum image). The assignment minimizes the
    summed distance with a bonus `stay` (A) for keeping an index, so orbitals sharing a centre are never
    reordered. Returns (perm, displacement (n, 3) of each orbital from its reference partner, A).
    """
    from scipy.optimize import linear_sum_assignment

    ref = np.asarray(ref_centres, dtype=float)
    c = np.asarray(centres, dtype=float)
    d = _min_image(c[:, None, :] - ref[None, :, :], lattice)
    rows, cols = linear_sum_assignment(np.linalg.norm(d, axis=-1) - stay * np.eye(len(c)))
    perm = np.empty(len(c), dtype=int)
    perm[rows] = cols
    return perm, d[np.arange(len(c)), perm]


def _xyz(a):
    if a is None:
        return None
    a = np.asarray(a, dtype=float)
    return np.concatenate([a, np.zeros((len(a), 1))], axis=1) if a.ndim == 2 and a.shape[1] == 2 else a


def _degenerate_groups(e: np.ndarray, tol: float) -> list[np.ndarray]:
    """Index groups of sorted eigenvalues closer than tol (eV)."""
    cuts = np.flatnonzero(np.diff(e) > tol) + 1
    return np.split(np.arange(e.size), cuts)


def _project(Ug: np.ndarray, u: np.ndarray) -> np.ndarray:
    p = Ug @ (np.conj(Ug).T @ u)
    return p / np.linalg.norm(p)


# ---- roots of the signed gap ---------------------------------------------------------------------
def _same_sign_run(g, valid, start, step) -> list[int]:
    """Members from `start` outward (step -1 or +1) while valid and of the same sign, ascending."""
    out, j = [start], start + step
    while 0 <= j < len(g) and valid[j] and np.sign(g[j]) == np.sign(g[start]):
        out.append(j)
        j += step
    return out[::-1] if step < 0 else out


def _locate(D, g, valid, i) -> float:
    """Root in [D_i, D_i+1] of the cubic spline through the same-sign runs on both sides of the bracket."""
    from scipy.interpolate import CubicSpline
    from scipy.optimize import brentq

    idx = _same_sign_run(g, valid, i, -1) + _same_sign_run(g, valid, i + 1, +1)
    f = CubicSpline(D[idx], g[idx])
    return float(brentq(lambda x: float(f(x)), D[i], D[i + 1], xtol=1e-14 * max(1.0, abs(D[i + 1]))))


def _bracket(D, g, valid, i, tol, mixing=None, mix_tol: float = 1e-2) -> dict:
    """Sign change between members i and i+1: clean character swap or V-shape test, and root."""
    n, j = len(D), i + 1

    def neighbour(
        k, end
    ):  # member k next to the bracket end `end`: "ok" (same sign), "alternating" or "none"
        if not (0 <= k < n) or not valid[k] or g[k] == 0:
            return "none"
        return "ok" if np.sign(g[k]) == np.sign(g[end]) else "alternating"

    status = {"left": neighbour(i - 1, i), "right": neighbour(j + 1, j)}
    S = max(
        [D[j] - D[i]]
        + ([D[i] - D[i - 1]] if status["left"] == "ok" else [])
        + ([D[j + 1] - D[j]] if status["right"] == "ok" else [])
    )
    lo, hi = D[i] - tol * S, D[j] + tol * S
    rec = {
        "D": _locate(D, g, valid, i),
        "kind": "sign_change",
        "members": [i, j],
        "accepted": True,
        "z_left": None,
        "z_right": None,
        "mismatch": None,
    }
    problems, gross = [], []
    for side, a, b in (("left", i - 1, i), ("right", j, j + 1)):
        if status[side] != "ok":
            continue
        slope = (abs(g[b]) - abs(g[a])) / (D[b] - D[a])
        if (slope >= 0) if side == "left" else (slope <= 0):
            problems.append(f"|g| does not fall towards the bracket on the {side}")
            gross.append(problems[-1])
            continue
        z = D[b] - abs(g[b]) / slope if side == "left" else D[a] - abs(g[a]) / slope
        rec["z_" + side] = float(z)
        if not lo <= z <= hi:
            problems.append(f"the {side} extrapolation of |g| vanishes at D = {z:.6g}, outside the bracket")
        if not D[i] - S <= z <= D[j] + S:
            gross.append(
                f"the {side} extrapolation of |g| vanishes at D = {z:.6g}, "
                "more than one member spacing outside the bracket"
            )
    if rec["z_left"] is not None and rec["z_right"] is not None:
        rec["mismatch"] = float(abs(rec["z_left"] - rec["z_right"]) / S)
        if rec["mismatch"] > tol:
            problems.append(
                f"g jumps instead of closing: |g| extrapolated from the two sides vanishes at D = "
                f"{rec['z_left']:.6g} and {rec['z_right']:.6g}, {rec['mismatch']:.0%} of the member "
                "spacing apart (avoided crossing or relabelled states)"
            )
    checked = [k for k, v in status.items() if v == "ok"]
    alternating = [k for k, v in status.items() if v == "alternating"]
    if not checked and alternating:
        problems.append(
            f"g changes sign again at the next member on the {' and '.join(alternating)} (a jump: "
            "relabelled or mixed states)"
        )
    mix = None if mixing is None else max(float(mixing[i]), float(mixing[j]))
    rec["mixing"] = mix
    if mix is not None and np.isfinite(mix) and mix <= mix_tol:
        if gross:
            rec["accepted"] = False
            rec["reason"] = f"v/c characters swap without mixing (max mixing {mix:.1e}), but " + "; ".join(
                gross
            )
        else:
            rec["reason"] = f"band inversion: v/c characters swap without mixing (max mixing {mix:.1e})"
        if problems:
            rec["v_test"] = "; ".join(problems)
    elif problems:
        rec["accepted"] = False
        rec["reason"] = "; ".join(problems) + (
            "" if mix is None else f" (v/c mixing {mix:.2g} at the bracket)"
        )
    else:
        rec["reason"] = (
            "V-shaped closing"
            if len(checked) == 2
            else "unverified (no neighbours)"
            if not checked
            else f"V-shaped closing (checked on the {checked[0]}"
            + (f"; g changes sign again on the {alternating[0]})" if alternating else ")")
        )
    return rec


def _find_roots(D, g, tol, mixing=None, mix_tol: float = 1e-2, present=None, _index=None) -> list[dict]:
    """Gap-closing candidates; member indices also retain grid-only entries in diagnostics."""
    D, g = np.asarray(D, dtype=float), np.asarray(g, dtype=float)
    if present is not None and not np.all(present):
        keep = np.flatnonzero(present)
        return _find_roots(
            D[keep],
            g[keep],
            tol,
            None if mixing is None else np.asarray(mixing)[keep],
            mix_tol,
            _index=keep,
        )
    idx = np.arange(len(D)) if _index is None else np.asarray(_index)
    valid = np.isfinite(g)
    n = len(D)
    roots = []
    for i in np.flatnonzero(valid & (g == 0)):  # degenerate v/c level at a member
        sl = np.sign(g[i - 1]) if i >= 1 and valid[i - 1] else 0.0
        sr = np.sign(g[i + 1]) if i + 1 < n and valid[i + 1] else 0.0
        ok = sl * sr <= 0
        why = (
            "gap closed at this member; g changes sign across it"
            if sl * sr < 0
            else "gap closes at this member without a sign change (touching)"
            if sl * sr > 0
            else "gap closed at this member; no valid neighbour on one side (sign change unverified)"
        )
        roots.append(
            {"D": float(D[i]), "kind": "member", "members": [int(i)], "accepted": bool(ok), "reason": why}
        )
    vi = np.flatnonzero(valid)
    for a, b in zip(vi[:-1], vi[1:]):
        if g[a] == 0 or g[b] == 0 or np.sign(g[a]) == np.sign(g[b]):
            continue
        if b != a + 1:
            roots.append(
                {
                    "D": None,
                    "kind": "sign_change",
                    "members": [int(a), int(b)],
                    "accepted": False,
                    "reason": "v/c character lost at members "
                    f"{[int(idx[k]) for k in range(a + 1, b)]} inside the bracket",
                }
            )
            continue
        roots.append(_bracket(D, g, valid, int(a), tol, mixing, mix_tol))
    roots.sort(key=lambda r: D[r["members"][0]])
    for root in roots:
        root["members"] = [int(idx[k]) for k in root["members"]]
    return roots


# ---- family -----------------------------------------------------------------------------------------
class ModelFamily:
    """Members sorted by D; models load on demand (LRU of `max_models`), Gamma data are cached."""

    def __init__(
        self,
        members: list[FamilyMember],
        meta: dict | None = None,
        max_models: int = 2,
        gamma_window: int = 8,
    ):
        if not members or any(not math.isfinite(m.D) for m in members):
            raise ValueError("a model family requires finite member parameters")
        if len({m.D for m in members}) != len(members) or len({m.label for m in members}) != len(members):
            raise ValueError("family member parameters and labels must be unique")
        self.members = sorted(members, key=lambda m: m.D)
        self.meta = meta or {}
        self.max_models = max(1, int(max_models))
        self.gamma_window = int(gamma_window)  # bands kept on each side of the v/c pair in GammaData
        self._models: OrderedDict = OrderedDict()  # i -> (version, model)
        self._gamma: dict = {}  # i -> (version, GammaData)

    @classmethod
    def from_manifest(cls, path, n_occ=None, gauge=None, plane=None, band_pair=None, **kw):
        path = Path(path).expanduser().resolve()
        data = json.loads(path.read_text())
        entries = (data.get("members") or data.get("models")) if isinstance(data, dict) else data
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{path}: no members")
        top = data if isinstance(data, dict) else {}
        n_occ = n_occ if n_occ is not None else top.get("n_occ")
        plane = plane if plane is not None else top.get("plane")
        band_pair = band_pair if band_pair is not None else top.get("band_pair")
        members = []
        for entry in entries:
            label = str(entry.get("label", entry["D"]))
            D = entry["D"]
            if isinstance(D, bool) or not isinstance(D, (float, int)) or not math.isfinite(D):
                raise ValueError(f"member {label!r}: D must be finite")
            files = {k: str((path.parent / v).resolve()) for k, v in entry.get("files", {}).items() if v}
            occ = entry.get("n_occ", n_occ)
            pair = entry.get("band_pair", band_pair)
            member_plane = entry.get("plane", plane)
            model_loader = None
            if "scfout" in files:
                model_loader = OpenMXLoader(files, occ, member_plane, pair)
            elif "hr" in files or "tb" in files:
                model_loader = WannierLoader(files, occ, gauge, member_plane, pair, top.get("lattice"))
            if model_loader is not None and occ is None and pair is None:
                raise ValueError(f"member {label!r}: n_occ or band_pair is required")
            loader = GridLoader(files[GRID_FILE]) if GRID_FILE in files else model_loader
            if loader is None:
                raise ValueError(f"member {label!r}: requires hr/tb, scfout or grid")
            extra = {k: v for k, v in entry.items() if k not in {"label", "D", "files"}}
            members.append(FamilyMember(label, float(D), files, loader, extra, model_loader))
        if len({m.label for m in members}) != len(members) or len({m.D for m in members}) != len(members):
            raise ValueError("family member labels and D values must be unique")
        return cls(
            members,
            {
                "manifest": str(path),
                "n_occ": n_occ,
                "gauge": gauge,
                "plane": FieldPlane.from_mapping(plane).describe(),
            },
            **kw,
        )

    def input_files(self):
        paths = {Path(v) for m in self.members for v in m.files.values()}
        for member in self.members:
            loader = member.model_loader if member.model_loader is not None else member.loader
            if isinstance(loader, WannierLoader):
                paths.update(loader.effective_files().values())
        if self.meta.get("manifest"):
            paths.add(Path(self.meta["manifest"]))
        return sorted(paths)

    def source(self, i, angle=0.0, prefer_grid=True):
        member = self.members[i]
        if isinstance(member.loader, GridLoader) and (prefer_grid or not member.has_model):
            return member.loader.with_angle(angle)()
        return self.model(i)

    def __len__(self):
        return len(self.members)

    @property
    def D(self) -> np.ndarray:
        return np.array([m.D for m in self.members])

    @property
    def labels(self) -> list[str]:
        return [m.label for m in self.members]

    # ---- loading ---------------------------------------------------------------------------------------
    def _version(self, i: int):
        m = self.members[i]
        fp = getattr(m.model_loader if m.model_loader is not None else m.loader, "fingerprint", None)
        return fp() if callable(fp) else None

    def model(self, i: int, _version=None):
        """Band model of member i; the last `max_models` models stay loaded (reloaded if files changed)."""
        key = self._version(i) if _version is None else _version
        hit = self._models.get(i)
        if hit is None or hit[0] != key:
            hit = (key, self.members[i].load_model())
        self._models[i] = hit
        self._models.move_to_end(i)
        while len(self._models) > self.max_models:
            self._models.popitem(last=False)
        return hit[1]

    def gamma_data(self, i: int) -> GammaData | None:
        """Gamma-point eigen-data of member i (None for grid-only members); cached per file version."""
        if not self.members[i].has_model:
            return None
        key = self._version(i)
        hit = self._gamma.get(i)
        if hit is not None and hit[0] == key:
            return hit[1]
        m = self.model(i, key)
        if getattr(m, "supports_generalized", False):
            return None  # Cross-member overlap is not supplied by SCFOUT.
        h = np.asarray(m.hamiltonian(0.0, 0.0))
        reference = float(np.trace(h).real / m.n_orb)
        relative_e, U = np.linalg.eigh(h - reference * np.eye(m.n_orb))
        e = relative_e + reference
        v, c = (int(b) for b in m.band_pair())
        lo, hi = max(0, v - self.gamma_window), min(len(e), c + 1 + self.gamma_window)
        lat = getattr(m, "lattice", None)
        gd = GammaData(
            int(m.n_orb),
            (v, c),
            e,
            lo,
            U[:, lo:hi].copy(),
            _xyz(getattr(m, "positions", None)),
            None if lat is None else np.asarray(lat, dtype=float).reshape(3, 3),
            np.asarray(m.plane.origin if hasattr(m, "plane") else (0, 0, 0), dtype=float),
            relative_e,
            reference,
            float(np.finfo(float).eps * max(1.0, np.linalg.norm(h, ord=np.inf))),
        )
        self._gamma[i] = (key, gd)
        return gd

    def _alignment(self, G, *, validate_tracking=True):
        """Orbital permutations of every member onto the first member with a Hamiltonian (by centres)."""
        n = len(G)
        perms, disp, shifts = [None] * n, [None] * n, [None] * n
        idx = [i for i in range(n) if G[i] is not None]
        if not idx:
            return None, perms, disp, shifts
        b = G[idx[0]]
        origin = np.asarray(b.fractional_origin)
        nonzero_origin = not np.allclose(origin, np.rint(origin), atol=1e-12, rtol=0)
        for i in idx:
            gi = G[i]
            difference = np.asarray(gi.fractional_origin) - origin
            if validate_tracking and not np.allclose(difference, np.rint(difference), atol=1e-12, rtol=0):
                raise CapabilityError(
                    "family character tracking requires the same fractional reciprocal origin"
                )
            if validate_tracking and nonzero_origin:
                if gi.centres is None or b.centres is None or gi.lattice is None or b.lattice is None:
                    raise CapabilityError(
                        "nonzero-origin family character tracking requires Wannier centres and lattice"
                    )
                if not np.allclose(gi.lattice, b.lattice, atol=1e-10, rtol=1e-10):
                    raise CapabilityError(
                        "nonzero-origin family character tracking requires a common lattice"
                    )
            if gi.centres is None or b.centres is None or gi.n_orb != b.n_orb:
                continue
            lattice = b.lattice if b.lattice is not None else gi.lattice
            perm, disp[i] = match_orbitals(b.centres, gi.centres, lattice)
            if lattice is not None:
                integer_shifts = (gi.centres - b.centres[perm] - disp[i]) @ np.linalg.inv(lattice)
                if validate_tracking and not np.allclose(
                    integer_shifts, np.rint(integer_shifts), atol=1e-8, rtol=0
                ):
                    raise CapabilityError(
                        "orbital home-cell shifts could not be resolved as lattice translations"
                    )
                shifts[i] = np.rint(integer_shifts).astype(int)
            if np.any(perm != np.arange(gi.n_orb)):
                perms[i] = perm
        return idx[0], perms, disp, shifts

    # ---- consistency ---------------------------------------------------------------------------------
    def consistency(self, centre_tol: float = 0.5, z_tol: float = 0.5, lattice_tol: float = 1e-6) -> dict:
        """Orbital count, lattice, Wannier centres and Gamma-point v/c character across the family.

        Centres are compared with the first member's in 3D, modulo the lattice and orbital by orbital
        (match_orbitals): relabelled orbitals are reported as permutations (and fail the check), in-plane
        and z drifts (A) separately. The Gamma character check fails when a member loses the v/c character
        or when bands other than band_pair() carry it (a remote band entered the gap window). Grid-only
        members are listed under "no_hamiltonian" and not checked.
        """
        n = len(self)
        G, static = [], []
        for i in range(n):
            data = self.gamma_data(i)
            G.append(data)
            if data is not None:
                static.append({"n_orb": data.n_orb, "band_pair": data.band_pair, "lattice": data.lattice})
            elif self.members[i].has_model:
                # Gamma data gates generalized eigenvector comparisons. Static
                # counts and cells remain available without those comparisons.
                model = self.model(i)
                static.append(
                    {
                        "n_orb": model.n_orb,
                        "band_pair": tuple(model.band_pair()),
                        "lattice": getattr(model, "lattice", None),
                    }
                )
            else:
                static.append(None)
        labels = self.labels
        out = {
            "ok": True,
            "n_orb": [],
            "band_pairs": [],
            "max_lattice_diff": None,
            "max_inplane_centre_drift": None,
            "max_z_centre_drift": None,
            "orbital_permutations": {},
            "home_cell_shifts": {},
            "gamma_character": None,
            "no_hamiltonian": [labels[i] for i in range(n) if not self.members[i].has_model],
            "tracking_unavailable": [
                labels[i] for i in range(n) if G[i] is None and self.members[i].has_model
            ],
            "issues": [],
        }
        idx = [i for i in range(n) if G[i] is not None]
        static_idx = [i for i in range(n) if static[i] is not None]
        issues = []
        out["n_orb"] = sorted({static[i]["n_orb"] for i in static_idx})
        out["band_pairs"] = sorted({tuple(static[i]["band_pair"]) for i in static_idx})
        if len(out["n_orb"]) > 1:
            issues.append(f"orbital counts differ: {out['n_orb']}")
        lattices = [static[i]["lattice"] for i in static_idx if static[i]["lattice"] is not None]
        lat = [float(np.max(np.abs(np.asarray(lattice) - np.asarray(lattices[0])))) for lattice in lattices]
        out["max_lattice_diff"] = max(lat, default=None)
        if out["max_lattice_diff"] is not None and out["max_lattice_diff"] > lattice_tol:
            issues.append(f"lattices differ by up to {out['max_lattice_diff']:.3g} A")
        if out["tracking_unavailable"]:
            issues.append(f"cross-member basis overlaps unavailable for {out['tracking_unavailable']}")
            out["gamma_character"] = {"ok": False, "available": False, "min_weight": None, "max_mixing": None}
        if not idx:
            out["issues"], out["ok"] = issues, not issues
            return json_safe(out)
        base, perms, disp, shifts = self._alignment(G, validate_tracking=False)
        d = [x for x in disp if x is not None]
        out["max_inplane_centre_drift"] = max(
            (float(np.max(np.linalg.norm(x[:, :2], axis=1))) for x in d), default=None
        )
        out["max_z_centre_drift"] = max((float(np.max(np.abs(x[:, 2]))) for x in d), default=None)
        if out["max_inplane_centre_drift"] is not None and out["max_inplane_centre_drift"] > centre_tol:
            issues.append(f"in-plane centre drift {out['max_inplane_centre_drift']:.3g} A > {centre_tol:g} A")
        if out["max_z_centre_drift"] is not None and out["max_z_centre_drift"] > z_tol:
            issues.append(f"z centre drift {out['max_z_centre_drift']:.3g} A > {z_tol:g} A")
        out["orbital_permutations"] = {labels[i]: perms[i].tolist() for i in range(n) if perms[i] is not None}
        out["home_cell_shifts"] = {labels[i]: shifts[i].tolist() for i in range(n) if shifts[i] is not None}
        if out["orbital_permutations"]:
            issues.append(
                f"orbitals relabelled relative to {labels[base]!r} in {sorted(out['orbital_permutations'])}"
            )
        gapped = [
            i
            for i in idx
            if _relative_energies(G[i])[G[i].band_pair[1]] - _relative_energies(G[i])[G[i].band_pair[0]]
            > 1e-12
        ]
        ref = gapped[0] if gapped else idx[0]  # first member with a non-degenerate v/c pair at Gamma
        try:
            trk = self._track(ref)
        except (CapabilityError, ValueError) as exc:
            out["gamma_character"] = {"ok": False, "available": False, "error": str(exc)}
            issues.append(f"Gamma character not tracked: {exc}")
        else:
            lost = [labels[i] for i in range(n) if "character_lost" in trk["flags"][i]]
            remote = [labels[i] for i in range(n) if "remote_band" in trk["flags"][i]]
            mism = [labels[i] for i in range(n) if "n_orb_mismatch" in trk["flags"][i]]
            wv, wc = np.asarray(trk["weight_v"]), np.asarray(trk["weight_c"])
            out["gamma_character"] = {
                "ok": not (lost or remote or mism or out["tracking_unavailable"]),
                "available": not bool(out["tracking_unavailable"]),
                "tracked_members": [labels[i] for i in idx],
                "reference": labels[ref],
                "character_lost": lost,
                "remote_band": remote,
                "min_weight": float(np.nanmin(np.minimum(wv, wc))),
                "max_mixing": float(np.nanmax(trk["mixing"])),
            }
            if lost:
                issues.append(f"Gamma v/c character lost at {lost}")
            if remote:
                issues.append(f"bands other than band_pair() carry the Gamma v/c character at {remote}")
        out["issues"] = issues
        out["ok"] = not issues
        return json_safe(out)

    # ---- critical field -----------------------------------------------------------------------------
    def _track(self, ref: int | None = None, min_weight: float = 0.5, deg_tol: float = 1e-12) -> dict:
        """Sequential Gamma-point v/c character tracking (module docstring); per-member lists.

        ref: reference member; default is the first open Gamma gap.
        """
        n = len(self)
        G = [self.gamma_data(i) for i in range(n)]
        if ref is None:
            ref = next(
                (
                    i
                    for i in range(n)
                    if G[i] is not None
                    and _relative_energies(G[i])[G[i].band_pair[1]]
                    - _relative_energies(G[i])[G[i].band_pair[0]]
                    > deg_tol
                ),
                None,
            )
            if ref is None:
                raise ValueError(
                    "no member has a Hamiltonian with an open Gamma gap: the Gamma character is unknown"
                )
        if G[ref] is None:
            raise ValueError(f"reference member {self.members[ref].label!r} has no Hamiltonian")
        _, perms, _, shifts = self._alignment(G)

        def states(i):
            aligned = G[i].U
            if shifts[i] is not None:
                phase = np.exp(-2j * np.pi * (shifts[i] @ G[i].fractional_origin))
                aligned = phase[:, None] * aligned
            if perms[i] is None:
                return aligned
            Ua = np.empty_like(G[i].U)
            Ua[perms[i]] = aligned
            return Ua

        flags = [
            ["no_hamiltonian"] if G[i] is None else (["orbital_permutation"] if perms[i] is not None else [])
            for i in range(n)
        ]
        g, wv, wc, mix = (np.full(n, np.nan) for _ in range(4))
        bands = [None] * n
        r = G[ref]
        v, c = r.band_pair
        reference_energies = _relative_energies(r)
        if reference_energies[c] - reference_energies[v] <= deg_tol:
            raise ValueError(
                f"reference member {self.members[ref].label!r} has a degenerate v/c pair at Gamma"
            )
        U0 = states(ref)
        uv0, uc0 = U0[:, v - r.lo], U0[:, c - r.lo]
        g[ref], wv[ref], wc[ref], mix[ref], bands[ref] = (
            reference_energies[c] - reference_energies[v],
            1.0,
            1.0,
            0.0,
            (v, c),
        )
        for step in (1, -1):
            uv, uc = uv0, uc0
            for i in range(ref + step, n if step > 0 else -1, step):
                gi = G[i]
                if gi is None:
                    continue
                if gi.n_orb != r.n_orb:
                    flags[i].append("n_orb_mismatch")
                    continue
                U = states(i)
                e = _relative_energies(gi)[gi.lo : gi.lo + U.shape[1]]
                ov_v, ov_c = np.abs(np.conj(uv) @ U) ** 2, np.abs(np.conj(uc) @ U) ** 2
                grp = _degenerate_groups(e, deg_tol)
                Wv = np.array([ov_v[x].sum() for x in grp])
                Wc = np.array([ov_c[x].sum() for x in grp])
                a, b = int(np.argmax(Wv)), int(np.argmax(Wc))
                wv[i], wc[i] = Wv[a], Wc[b]
                if Wv[a] <= min_weight or Wc[b] <= min_weight:
                    flags[i].append("character_lost")
                    continue
                if a == b:  # v and c character in one degenerate level: the gap is closed here
                    if len(grp[a]) < 2:
                        flags[i].append("character_lost")
                        continue
                    bands[i] = (gi.lo + int(grp[a][0]), gi.lo + int(grp[a][-1]))
                    g[i], mix[i] = 0.0, np.nan
                    flags[i].append("degenerate")
                    continue
                iv = int(grp[a][np.argmax(ov_v[grp[a]])])
                ic = int(grp[b][np.argmax(ov_c[grp[b]])])
                bands[i] = (gi.lo + iv, gi.lo + ic)
                g[i] = e[ic] - e[iv]
                mix[i] = max(ov_c[grp[a]].sum(), ov_v[grp[b]].sum())
                if set(bands[i]) != set(gi.band_pair):
                    flags[i].append("remote_band")
                uv, uc = _project(U[:, grp[a]], uv), _project(U[:, grp[b]], uc)
        labels = self.labels
        return {
            "D": self.D.tolist(),
            "labels": labels,
            "reference": labels[ref],
            "signed_gap": g.tolist(),
            "bands": bands,
            "band_pair": [None if x is None else tuple(x.band_pair) for x in G],
            "weight_v": wv.tolist(),
            "weight_c": wc.tolist(),
            "mixing": mix.tolist(),
            "flags": flags,
            "orbital_permutations": {labels[i]: perms[i].tolist() for i in range(n) if perms[i] is not None},
            "home_cell_shifts": {labels[i]: shifts[i].tolist() for i in range(n) if shifts[i] is not None},
            "fractional_origins": [None if x is None else x.fractional_origin.tolist() for x in G],
            "energy_references_eV": [None if x is None else x.energy_reference for x in G],
            "assembly_precision_floors_eV": [None if x is None else x.assembly_precision_floor for x in G],
            "gap_evaluation": "centered spectral differences",
            "precision_caveat": "Double-precision Hamiltonian assembly can round terms away; centered spectra cannot recover them.",
        }

    def gamma_signed_gap(self, ref: int | None = None) -> np.ndarray:
        """eps(c-like) - eps(v-like) at Gamma (eV) per member, positive in the reference order; NaN where the
        v/c character is lost or no Hamiltonian is available."""
        return np.asarray(self._track(ref)["signed_gap"], dtype=float)

    def critical_D_report(
        self, ref: int | None = None, min_weight: float = 0.5, jump_tol: float = 0.1, mix_tol: float = 1e-2
    ) -> dict:
        """Gamma character tracking and every gap-closing candidate (module docstring), JSON-ready.

        ref: reference member (default: the first open gap); its order of the v/c pair is positive.
        min_weight: least overlap with the previous member's v (c) state; jump_tol: V test (fraction of the
        largest member spacing around the bracket).

        Keys: per-member "signed_gap", "bands" (v-like, c-like), "band_pair", "weight_v", "weight_c",
        "mixing" (cross-character of the chosen levels), "flags"; "orbital_permutations"; "min_abs_gap";
        "roots" (D, kind, members, accepted, reason, and z_left/z_right/mismatch of the V test);
        "accepted_roots"; "D_c" (the accepted root if there is exactly one, else None).
        """
        if any(isinstance(m.model_loader or m.loader, OpenMXLoader) for m in self.members):
            raise CapabilityError(
                "automatic OpenMX character tracking requires validated cross-member basis overlaps"
            )
        rep = self._track(ref, min_weight)
        g = np.asarray(rep["signed_gap"], dtype=float)
        present = ["no_hamiltonian" not in flags for flags in rep["flags"]]
        rep["roots"] = _find_roots(
            self.D, g, jump_tol, np.asarray(rep["mixing"], dtype=float), mix_tol, present
        )
        rep["accepted_roots"] = [r["D"] for r in rep["roots"] if r["accepted"]]
        rep["D_c"] = rep["accepted_roots"][0] if len(rep["accepted_roots"]) == 1 else None
        fin = np.flatnonzero(np.isfinite(g))
        k = int(fin[np.argmin(np.abs(g[fin]))]) if fin.size else None
        rep["min_abs_gap"] = (
            None if k is None else {"eV": float(abs(g[k])), "D": float(self.D[k]), "label": rep["labels"][k]}
        )
        return json_safe(rep)

    def critical_D(self, **kw) -> float | None:
        """D_c of the family (critical_D_report), or None without an accepted gap closing.

        Raises ValueError if several gap closings are accepted; warns if a sign change was rejected.
        """
        rep = self.critical_D_report(**kw)
        acc = rep["accepted_roots"]
        if len(acc) > 1:
            raise ValueError(
                f"{len(acc)} gap closings at D = {acc}; inspect critical_D_report() and pass the "
                "wanted D_c explicitly (e.g. harmonic_map(..., normalize_by_Dc=<D_c>))"
            )
        rejected = [r for r in rep["roots"] if not r["accepted"]]
        if rejected:
            warnings.warn(
                f"critical_D: {len(rejected)} sign change(s) of the Gamma gap rejected, e.g. between "
                f"{[rep['labels'][j] for j in rejected[0]['members']]}: {rejected[0]['reason']}",
                RuntimeWarning,
                stacklevel=2,
            )
        return acc[0] if acc else None

    # ---- native result-oriented sweeps -----------------------------------------------------
    def harmonic_map(
        self,
        E0,
        cfg=None,
        n_max=None,
        n_workers=1,
        cache_root=None,
        normalize_by_Dc=False,
        prefer_grid=True,
        execution=None,
    ):
        from ..execution import ExecutionOptions, execution_session
        from .api import PreparedHallResponse
        from .config import resolve_options

        cfg = resolve_options(cfg)
        options = (execution or ExecutionOptions(workers=n_workers)).resolved()
        params = self.D.copy()
        Dc = None
        if normalize_by_Dc is not False and normalize_by_Dc is not None:
            Dc = self.critical_D() if normalize_by_Dc is True else float(normalize_by_Dc)
            if Dc is None or not math.isfinite(Dc) or Dc == 0:
                raise ValueError("normalization needs one finite nonzero critical D or an explicit value")
            params /= Dc

        # Family workers own a source, with no nested line pools. Returned
        # results are collected and reduced only by this owning process.
        def calculate(i):
            prepared = PreparedHallResponse(
                self.source(i, cfg.drive.angle, prefer_grid),
                options=cfg,
                execution=ExecutionOptions(
                    workers=1,
                    batch_size=options.batch_size,
                    native_threads=options.native_threads,
                    memory_mb=None if options.memory_mb is None else options.memory_mb / options.workers,
                ),
                cache_dir=cache_root,
            )
            prepared.prepare()
            return prepared.harmonics(E0, n_max=n_max)

        if options.resolved().workers == 1:
            rows = [calculate(i) for i in range(len(self))]
        else:
            from joblib import delayed

            with execution_session(options) as session:
                rows = list(session.parallel()(delayed(calculate)(i) for i in range(len(self))))
        first = rows[0]
        arrays = {"parameter": params, "D": self.D, "labels": np.asarray(self.labels)}
        dims = {"parameter": ("member",), "D": ("member",), "labels": ("member",)}
        units = {
            "parameter": "D/Dc" if Dc is not None else "D",
            "D": self.meta.get("D_unit", "D"),
            "labels": "label",
        }
        for key, value in first.arrays.items():
            if key in {"E0", "orders", "N", "amplitudes"}:
                arrays[key], dims[key], units[key] = value, first.dims[key], first.units.get(key, "1")
            else:
                arrays[key] = np.stack([r.arrays[key] for r in rows])
                dims[key] = ("member", *first.dims[key])
                units[key] = first.units.get(key, "1")
        return ObservableResult(
            arrays,
            dims,
            units,
            json_safe(
                {
                    "observable": "harmonic_map",
                    "family": {**self.meta, "D_c": Dc, "labels": self.labels},
                    "members": [r.metadata for r in rows],
                }
            ),
        )

    def line_scan(self, cfg=None):
        from .api import crossing_scan
        from .config import resolve_options

        cfg = resolve_options(cfg)
        rows = [crossing_scan(self.source(i, cfg.drive.angle), [0.0], options=cfg) for i in range(len(self))]
        counts = np.array([len(row.arrays["position"]) for row in rows], dtype=int)
        offsets = np.r_[0, np.cumsum(counts)]
        arrays = {"D": self.D, "offsets": offsets}
        dims = {"D": ("member",), "offsets": ("member_boundary",)}
        units = {"D": self.meta.get("D_unit", "D"), "offsets": "index"}
        for key in ("position", "critical_field", "ddp_field", "metric", "gap"):
            if key in rows[0].arrays:
                arrays[key] = np.concatenate([row.arrays[key] for row in rows])
                dims[key] = rows[0].dims[key]
                units[key] = rows[0].units[key]
        for key in ("k0", "line_valid", "Ec_eq8", "Ec_ddp", "selected_Ec"):
            arrays[key] = np.concatenate([row.arrays[key] for row in rows])
            dims[key] = ("member",)
            if key in rows[0].units:
                units[key] = rows[0].units[key]
        segment_counts = [len(row.arrays["Lambda"]) for row in rows]
        arrays["segment_offsets"] = np.r_[0, np.cumsum(segment_counts)]
        dims["segment_offsets"] = ("member_boundary",)
        for key in ("gap_integrals", "Lambda"):
            arrays[key] = np.concatenate([row.arrays[key] for row in rows])
            dims[key] = ("segment",)
            units[key] = rows[0].units[key]
        return ObservableResult(
            arrays,
            dims,
            units,
            json_safe(
                {"observable": "family_line_scan", "family": self.meta, "members": [r.metadata for r in rows]}
            ),
        )
