"""Symmetry analysis: Hamiltonian-level operations, geometric certificates and selection rules.

Two levels:

* `Operation` acts on the Bloch Hamiltonian: U H(k) U^dag (or U H(k)^* U^dag if antiunitary)
  = +-H(R k), optionally only for the traceless part and/or with a parameter map. The catalog of the
  four-band model is in `fourband_operations`.

* `KOperation` states the consequence on the fields Eqs. 7-13 use (eps_g, Omega_c, Omega_v, g^{cv}):
      eps_g(Rk) = eps_g(k),   g^{cv}(Rk) = R g^{cv}(k) R^T,
      Omega_n(Rk) = omega_sign * Omega_n'(k)     (n' = n, or the partner band for antisymmetries).
  For a unitary operation omega_sign = det R, antiunitary -det R. Certificates test these relations
  numerically on any geometry provider, so shortcuts never rely on assumed symmetries.

Selection rules (within Eqs. 10-12): J_H(-E) = -J_H(E) if a certified operation maps the field
direction s -> -s with omega_sign = +1 (then only odd, pure-sine harmonics); the transverse half-line
t >= 0 suffices if an operation fixes s, flips t and has omega_sign = +1.

Wavefunction phases (PHYSICAL: geometric phase gamma_R, twisted LZ) are not functions of these fields.
An antiunitary operation maps a path onto the mirrored path traversed backwards, which reverses the sign
of the shift vector R_s, so only unitary operations (omega_sign = det R) remain valid; these are also
certified on R_s itself (`certify(..., shift_fn=...)`). With omega_sign = +1 this leaves rotations:
E-oddness from C2, never the t-mirror (a mirror fixing s has det R = -1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

from .models.fourband import G_Z, I4, kron

# ---------------------------------------------------------------------------------------------
# Hamiltonian-level operations
# ---------------------------------------------------------------------------------------------


def rotation(phi: float) -> np.ndarray:
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s], [s, c]])


def mirror(phi: float) -> np.ndarray:
    """Reflection across the line through the origin at angle phi (k-space)."""
    c, s = math.cos(2 * phi), math.sin(2 * phi)
    return np.array([[c, s], [s, -c]])


@dataclass(frozen=True)
class Operation:
    name: str
    U: np.ndarray
    R: np.ndarray
    antiunitary: bool = False
    anti: bool = False  # U H U^dag = -H(Rk)
    traceless_only: bool = False
    param_map: dict = field(default_factory=dict)  # parameter name -> callable(value)

    @property
    def omega_sign(self) -> int:
        return int(round(np.linalg.det(self.R))) * (-1 if self.antiunitary else 1)

    def to_k_operation(self) -> "KOperation":
        return KOperation(self.name, self.R, self.omega_sign, swap_bands=self.anti)


def _traceless(H):
    n = H.shape[-1]
    tr = np.trace(H, axis1=-2, axis2=-1)[..., None, None] / n
    return H - tr * np.eye(n)


def check_hamiltonian(model, op: Operation, n_samples: int = 64, k_scale: float = 0.03, rng=None) -> float:
    """Max |U H(k)^(*) U^dag -+ H'(Rk)| over random k (eV). H' is the parameter-mapped model."""
    rng = np.random.default_rng(1) if rng is None else rng
    k = rng.uniform(-k_scale, k_scale, size=(2, n_samples))
    target_model = model
    if op.param_map:
        target_model = model.with_params(
            **{name: f(getattr(model.p, name)) for name, f in op.param_map.items()}
        )
    H = model.hamiltonian(k[0], k[1])
    if op.antiunitary:
        H = np.conj(H)
    lhs = op.U @ H @ np.conj(op.U.T)
    Rk = op.R @ k
    rhs = target_model.hamiltonian(Rk[0], Rk[1]) * (-1.0 if op.anti else 1.0)
    if op.traceless_only:
        lhs, rhs = _traceless(lhs), _traceless(rhs)
    return float(np.max(np.abs(lhs - rhs)))


def fourband_operations() -> list[Operation]:
    """Verified symmetry catalog of Eq. 9 (basis sigma (x) tau)."""

    def spin_rot(phi):
        return expm(-0.5j * phi * G_Z)  # exp(-i phi sz/2) (x) t0

    def neg(x):
        return -x

    ops = [
        Operation("C3", spin_rot(2 * math.pi / 3), rotation(2 * math.pi / 3)),
        Operation("M0*K", I4, mirror(0.0), antiunitary=True),  # (kx, ky) -> (kx, -ky)
        Operation("M60*K", spin_rot(2 * math.pi / 3), mirror(math.pi / 3), antiunitary=True),
        Operation("M120*K", spin_rot(4 * math.pi / 3), mirror(2 * math.pi / 3), antiunitary=True),
        # traceless part only (broken by D3 k^3 cos 3theta, which is identity)
        Operation("C2z", spin_rot(math.pi), rotation(math.pi), traceless_only=True),
        Operation("R(0.37)", spin_rot(0.37), rotation(0.37), traceless_only=True),
        Operation("M90*K", spin_rot(math.pi), mirror(math.pi / 2), antiunitary=True, traceless_only=True),
        Operation("PH: sy tx K", kron("y", "x"), np.eye(2), antiunitary=True, anti=True, traceless_only=True),
        Operation("sx tx", kron("x", "x"), mirror(math.pi / 2), anti=True, traceless_only=True),
        # parameter maps
        Operation("tx: VL->-VL", kron("0", "x"), np.eye(2), param_map={"VL": neg}),
        Operation(
            "T: (M,B,Bl,D3)->-",
            1j * kron("y", "0"),
            rotation(math.pi),
            antiunitary=True,
            param_map={"M": neg, "B": neg, "Bl": neg, "D3": neg},
        ),
    ]
    return ops


# ---------------------------------------------------------------------------------------------
# Geometric certificates
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class KOperation:
    name: str
    R: np.ndarray
    omega_sign: int
    swap_bands: bool = False


def candidate_k_operations(extra_angles=()) -> list[KOperation]:
    """Rotations by 2pi/n (n = 2, 3, 4, 6) and mirrors every 15 deg, both Omega signs."""
    cands = []
    for n in (2, 3, 4, 6):
        R = rotation(2 * math.pi / n)
        cands += [KOperation(f"C{n}", R, +1), KOperation(f"C{n}*K", R, -1)]
    for deg in list(range(0, 180, 15)) + list(extra_angles):
        R = mirror(math.radians(deg))
        cands += [KOperation(f"M{deg}(+)", R, +1), KOperation(f"M{deg}(-)", R, -1)]
    for phi in (0.37, 1.1):
        cands += [KOperation(f"R({phi})", rotation(phi), +1)]
    return cands


@dataclass
class Certificate:
    residuals: dict  # name -> max relative residual over the tested fields
    tol: float
    operations: dict  # name -> KOperation
    wavefunction: dict = field(
        default_factory=dict
    )  # name -> residual of R_s(Rk) = sigma R_s(k) (unitary ops)
    wavefunction_angle: float | None = None  # field angle of the shift-vector test

    @property
    def certified(self) -> list[KOperation]:
        return [self.operations[n] for n, r in self.residuals.items() if r <= self.tol]

    def holds(self, name: str) -> bool:
        return self.residuals.get(name, math.inf) <= self.tol

    def holds_wavefunction(self, name: str, field_angle: float) -> bool:
        if (
            self.wavefunction_angle is None
            or abs(math.remainder(field_angle - self.wavefunction_angle, 2 * math.pi)) > 1e-12
        ):
            return False
        return self.holds(name) and self.wavefunction.get(name, math.inf) <= self.tol


def is_unitary(op: KOperation) -> bool:
    """Field-level operations consistent with a unitary symmetry: omega_sign = det R."""
    return op.omega_sign == int(round(np.linalg.det(op.R)))


def _fields(geo, omega_modes=("c", "v")):
    return geo.eps_g, {m: geo.omega(m) for m in omega_modes}, geo.g_cv


def certify(
    geometry_fn,
    operations=None,
    n_samples: int = 200,
    k_range=(1e-5, 0.02),
    tol: float = 1e-8,
    rng=None,
    field_angles=(),
    shift_fn=None,
    shift_angle: float = 0.0,
) -> Certificate:
    """Test the geometric consequences of `operations` on `geometry_fn(kx, ky) -> BandGeometry`.

    geometry_fn must evaluate at Cartesian k in the model frame (field angle 0). Samples have
    log-uniform radius in k_range and uniform direction. Residuals are max|lhs - rhs| / max|rhs|
    per field (scale-normalized, so sign changes of Omega do not inflate them). `field_angles` (rad)
    adds mirrors along and perpendicular to each field direction to the default candidates.

    shift_fn(kx, ky) -> R_s for motion along `shift_angle` (geometry/wilson.shift_field) adds the
    wavefunction-level test for unitary operations that map the field direction e to sigma e:
    R_s(R k) = sigma R_s(k) (residuals in Certificate.wavefunction).
    """
    if operations is None:
        extra = [a for phi in field_angles for a in (math.degrees(phi), math.degrees(phi) + 90.0)]
        operations = candidate_k_operations(extra_angles=extra)
    rng = np.random.default_rng(7) if rng is None else rng
    r = np.exp(rng.uniform(math.log(k_range[0]), math.log(k_range[1]), n_samples))
    th = rng.uniform(0, 2 * math.pi, n_samples)
    k = np.stack([r * np.cos(th), r * np.sin(th)])
    g0 = geometry_fn(k[0], k[1])
    residuals, ops = {}, {}
    for op in operations:
        Rk = op.R @ k
        g1 = geometry_fn(Rk[0], Rk[1])
        res = [np.max(np.abs(g1.eps_g - g0.eps_g)) / np.max(np.abs(g0.eps_g))]
        for mode in ("c", "v"):
            src = {"c": "v", "v": "c"}[mode] if op.swap_bands else mode
            lhs, rhs = g1.omega(mode), op.omega_sign * g0.omega(src)
            res.append(np.max(np.abs(lhs - rhs)) / np.max(np.abs(rhs)))
        G = np.stack(
            [np.stack([g0.g_cv[:, 0], g0.g_cv[:, 1]], -1), np.stack([g0.g_cv[:, 1], g0.g_cv[:, 2]], -1)], -2
        )
        Gr = op.R @ G @ op.R.T
        G1 = np.stack([g1.g_cv[:, 0], g1.g_cv[:, 1], g1.g_cv[:, 2]], -1)
        Gt = np.stack([Gr[:, 0, 0], Gr[:, 0, 1], Gr[:, 1, 1]], -1)
        res.append(np.max(np.abs(G1 - Gt)) / np.max(np.abs(Gt)))
        residuals[op.name] = float(max(res))
        ops[op.name] = op
    wavefunction = {}
    if shift_fn is not None:
        e = np.array([math.cos(shift_angle), math.sin(shift_angle)])
        f0 = None
        for name, op in ops.items():
            if residuals[name] > tol or op.swap_bands or not is_unitary(op):
                continue
            Re = op.R @ e
            sigma = (
                1.0 if np.allclose(Re, e, atol=1e-12) else (-1.0 if np.allclose(Re, -e, atol=1e-12) else None)
            )
            if sigma is None:
                continue
            if f0 is None:
                f0 = np.asarray(shift_fn(k[0], k[1]), dtype=float)
                scale = max(float(np.max(np.abs(f0))), 1e-300)
            Rk = op.R @ k
            f1 = np.asarray(shift_fn(Rk[0], Rk[1]), dtype=float)
            wavefunction[name] = float(np.max(np.abs(f1 - sigma * f0)) / scale)
    return Certificate(residuals, tol, ops, wavefunction, shift_angle if shift_fn is not None else None)


def selection_rules(cert: Certificate, field_angle: float = 0.0, wavefunction_phases: bool = False) -> dict:
    """Consequences for Eqs. 10-12 with the field at `field_angle`.

    e_odd: J_H(-E) = -J_H(E) (only odd sine harmonics);  t_mirror: integrate t >= 0 only;
    isotropic: J_H independent of the field direction;  c3: J_H periodic in the angle with 2pi/3
    (isotropic and c3 are field-level statements).

    wavefunction_phases: the run uses the geometric phase or twisted LZ (PHYSICAL). Then only unitary
    operations whose shift-vector law is certified at this field angle count (module docstring).
    """
    s = np.array([math.cos(field_angle), math.sin(field_angle)])
    t = np.array([-s[1], s[0]])
    e_odd = t_mirror = False
    for op in cert.certified:
        if op.swap_bands:
            continue  # antisymmetries relate Omega_c to Omega_v; they do not map a v->c path onto itself
        if op.omega_sign != 1:
            continue
        if wavefunction_phases and not (is_unitary(op) and cert.holds_wavefunction(op.name, field_angle)):
            continue  # antiunitary: reverses the traversal and flips R_s and gamma_R
        Rs, Rt = op.R @ s, op.R @ t
        if np.allclose(Rs, -s, atol=1e-12):
            e_odd = True
        if np.allclose(Rs, s, atol=1e-12) and np.allclose(Rt, -t, atol=1e-12):
            t_mirror = True
    names = {op.name for op in cert.certified}
    isotropic = any(n.startswith("R(") for n in names)
    if (
        not wavefunction_phases
        and isotropic
        and any(op.omega_sign == 1 and np.linalg.det(op.R) < 0 and not op.swap_bands for op in cert.certified)
    ):
        t_mirror = True  # continuous rotations composed with one mirror give mirrors at every angle
    c3 = "C3" in names or isotropic
    return {"e_odd": e_odd, "t_mirror": t_mirror, "isotropic": isotropic, "c3": c3}
