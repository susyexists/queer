"""Four-band model of the MnBi2Te4 top/bottom surface Dirac states (Eq. 9, Methods 7.1).

    H = [B0 k^2 + D3 k^3 cos 3theta] s0 t0 + A (kx sx + ky sy) t0 + [M - B k^2] sz t0
        + Bl sz tx + VL s0 tz

s acts on the Dirac (spin) sector of each surface and t distinguishes the top (t_z = +1) and
bottom surfaces; basis index = 2*s_index + t_index (np.kron(s, t)). k^3 cos 3theta = kx^3 - 3 kx ky^2.

The identity terms (B0, D3) cancel in eps_c - eps_v, Omega and g, hence in Eqs. 7-13. Closed
forms follow from Lowdin partitioning onto the two k=0 states that close the gap at
VLc = sqrt(M^2 - Bl^2):  H_eff = (m0 + m2 k^2) sz' + hbar_v (k . s'),  with R = sqrt(Bl^2 + VL^2),
m0 = R - |M|, hbar_v = A Bl / R, m2 = -B - A^2 VL^2 / (2 |M| R^2) (exact to O(k^2)).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.optimize import brentq

from ..units import EV_PER_A_TO_V_PER_M

_S0 = np.eye(2, dtype=complex)
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {"0": _S0, "x": _SX, "y": _SY, "z": _SZ}


def kron(s: str, t: str) -> np.ndarray:
    """sigma_s (x) tau_t, e.g. kron("z", "x") = sz tx."""
    return np.kron(PAULI[s], PAULI[t])


def _k_arrays(kx, ky):
    """Broadcast (kx, ky) as float arrays, or complex ones if either input is complex (analytic continuation)."""
    dt = complex if (np.iscomplexobj(kx) or np.iscomplexobj(ky)) else float
    return np.broadcast_arrays(np.asarray(kx, dtype=dt), np.asarray(ky, dtype=dt))


I4 = np.eye(4, dtype=complex)
G_X, G_Y, G_Z = kron("x", "0"), kron("y", "0"), kron("z", "0")
G_BL, G_VL = kron("z", "x"), kron("0", "z")
WARPING_MATRICES = {"sz_t0": kron("z", "0"), "s0_tz": kron("0", "z"), "sz_tz": kron("z", "z")}


@dataclass(frozen=True)
class FourBandParams:
    """Parameters of Eq. 9 (eV, A). Defaults are the printed values (Methods 7.1)."""

    M: float = -1.022e-3  # eV, magnetic mass
    Bl: float = 0.4573e-3  # eV, interlayer hybridization
    A: float = 0.3290  # eV A
    B: float = -0.9994  # eV A^2
    B0: float = 13.4823  # eV A^2, particle-hole asymmetry (identity)
    D3: float = -214.13  # eV A^3, trigonal warping (identity as printed)
    VL: float = 0.0  # eV, interlayer potential (displacement field)

    @classmethod
    def paper_vlc_matched(cls, **kw) -> "FourBandParams":
        """M = -1.0215 meV, which reproduces the paper's VLc = 0.9134 meV."""
        return cls(M=-1.0215e-3, **kw)

    # ---- critical point and effective model -------------------------------------------------
    @property
    def vlc(self) -> float:
        """VLc = sqrt(M^2 - Bl^2): the k=0 gap closes (0.913981 meV for the printed values)."""
        return math.sqrt(self.M**2 - self.Bl**2)

    def at(self, x: float) -> "FourBandParams":
        """Copy with VL = x * VLc."""
        return replace(self, VL=x * self.vlc)

    @property
    def x(self) -> float:
        return self.VL / self.vlc

    @property
    def R(self) -> float:
        return math.hypot(self.Bl, self.VL)

    @property
    def m0(self) -> float:
        """Signed half-gap at k=0 of the effective model, R - |M| (< 0 topological, > 0 trivial)."""
        return self.R - abs(self.M)

    @property
    def hbar_v(self) -> float:
        """Effective Dirac velocity A*Bl/R in eV*A (0.147213 at VLc)."""
        return self.A * self.Bl / self.R

    @property
    def m2(self) -> float:
        """Effective k^2 mass -B - A^2 VL^2/(2|M| R^2) in eV*A^2 (-41.3535 at VLc)."""
        return -self.B - self.A**2 * self.VL**2 / (2.0 * abs(self.M) * self.R**2)

    @property
    def C_critical(self) -> float:
        """Critical Berry-curvature strength |m2|/(2 hbar_v) in A; Omega_c ~ -C/k at VL = VLc."""
        return abs(self.m2) / (2.0 * self.hbar_v)

    # ---- closed forms on the k_y = 0 line ----------------------------------------------------
    @property
    def gap_k0(self) -> float:
        """eps_g at k = 0: 2 |R - |M||, eV."""
        return 2.0 * abs(self.R - abs(self.M))

    @property
    def gcv_k0(self) -> float:
        """Band-resolved metric g^{cv}_xx at k = 0: A^2 (Bl/R)^2 / (4 (|M| - R)^2), A^2."""
        return self.A**2 * (self.Bl / self.R) ** 2 / (4.0 * (abs(self.M) - self.R) ** 2)

    @property
    def ec_k0(self) -> float:
        """Eq. 8 at k = 0: pi R (|M| - R)^2 / (A Bl), V/m (the ky=0 E_c while the gap minimum is at k=0)."""
        return math.pi * self.R * (abs(self.M) - self.R) ** 2 / (self.A * self.Bl) * EV_PER_A_TO_V_PER_M

    def ring_threshold_x(self) -> float:
        """VL/VLc beyond which the gap minimum leaves k = 0 (hbar_v^2 + 2 m0 m2 = 0; 1.21347)."""

        def f(x):
            p = self.at(x)
            return p.hbar_v**2 + 2.0 * p.m0 * p.m2

        return brentq(f, 1.0 + 1e-12, 10.0, xtol=1e-14, rtol=1e-14)

    @property
    def x_diabatic(self) -> float:
        """VL/VLc where the uncoupled surface bands start to cross: |M|/VLc (1.1182)."""
        return abs(self.M) / self.vlc

    def k_diabatic(self) -> float | None:
        """Crossing radius of the uncoupled (Bl = 0) surface bands: A^2 k^2 + (M - B k^2)^2 = VL^2.

        Returns None for VL <= |M| (no crossing).
        """
        d = self.VL**2 - self.M**2
        if d <= 0:
            return None
        b = self.A**2 - 2.0 * self.M * self.B
        q = 2.0 * d / (b + math.sqrt(b * b + 4.0 * self.B**2 * d))
        return math.sqrt(q)

    def gap_closed_form(self, k) -> np.ndarray:
        """eps_c - eps_v at |k| from E^2 = S -+ 2 sqrt(P) (traceless spectrum; isotropic)."""
        k = np.asarray(k, dtype=float)
        m = self.M - self.B * k**2
        a = self.A**2 * k**2 + m**2
        S = a + self.Bl**2 + self.VL**2
        P = m**2 * self.Bl**2 + self.VL**2 * a
        return 2.0 * np.sqrt(np.maximum(S - 2.0 * np.sqrt(P), 0.0))

    def spectrum_closed_form(self, k) -> np.ndarray:
        """Sorted eigenvalues of the traceless part at |k|, shape (..., 4)."""
        k = np.asarray(k, dtype=float)
        m = self.M - self.B * k**2
        a = self.A**2 * k**2 + m**2
        S = a + self.Bl**2 + self.VL**2
        P = np.sqrt(m**2 * self.Bl**2 + self.VL**2 * a)
        lo = np.sqrt(np.maximum(S - 2.0 * P, 0.0))
        hi = np.sqrt(S + 2.0 * P)
        return np.stack([-hi, -lo, lo, hi], axis=-1)

    @property
    def k_inv(self) -> float | None:
        """Radius where M - B k^2 = 0 (regularization band inversion; 0.031978 1/A), or None."""
        q = self.M / self.B
        return math.sqrt(q) if q > 0 else None


class FourBandModel:
    """Eq. 9 with analytic k-derivatives.

    Options:
      include_identity: keep B0 k^2 + D3 k^3 cos 3theta (as printed). The geometry does not depend
          on them; dropping them lowers ||H|| and the floating-point floor near Gamma.
      d3_layer: "t0" (as printed, identity) or "tz" (D3 k^3 cos 3theta s0 tz; sensitivity variant).
      warping, warping_matrix: traceless lam (kx^3 - 3 kx ky^2) Gamma (3m' emulator; Gamma one of
          "sz_t0", "s0_tz", "sz_tz"), used to exercise symmetry certificates that must fail.
    """

    n_orb = 4
    complex_k = True  # hamiltonian() accepts complex k: the analytic continuation used for branch points

    def __init__(
        self,
        params: FourBandParams | None = None,
        *,
        include_identity: bool = True,
        d3_layer: str = "t0",
        warping: float = 0.0,
        warping_matrix: str = "sz_t0",
    ):
        if d3_layer not in ("t0", "tz"):
            raise ValueError("d3_layer must be 't0' or 'tz'")
        if warping_matrix not in WARPING_MATRICES:
            raise ValueError(f"warping_matrix must be one of {sorted(WARPING_MATRICES)}")
        self.p = params if params is not None else FourBandParams()
        self.include_identity = include_identity
        self.d3_layer = d3_layer
        self.warping = float(warping)
        self.warping_matrix = warping_matrix

    # ---- construction helpers ----------------------------------------------------------------
    def with_params(self, **changes) -> "FourBandModel":
        return FourBandModel(
            replace(self.p, **changes),
            include_identity=self.include_identity,
            d3_layer=self.d3_layer,
            warping=self.warping,
            warping_matrix=self.warping_matrix,
        )

    def at(self, x: float) -> "FourBandModel":
        """Copy at VL = x * VLc."""
        return self.with_params(VL=x * self.p.vlc)

    def traceless(self) -> "FourBandModel":
        """Copy without the identity terms (identical eps_g, Omega, g)."""
        return FourBandModel(
            self.p,
            include_identity=False,
            d3_layer=self.d3_layer,
            warping=self.warping,
            warping_matrix=self.warping_matrix,
        )

    @property
    def is_isotropic_geometry(self) -> bool:
        """True if eps_g, Omega and g depend on |k| only (no traceless warping)."""
        return self.warping == 0.0 and (self.d3_layer == "t0" or self.p.D3 == 0.0)

    def band_pair(self) -> tuple[int, int]:
        return 1, 2

    def layer_projector(self) -> np.ndarray:
        """Projector on the t_z = +1 surface (used for diabatic crossings)."""
        return np.kron(_S0, 0.5 * (_S0 + _SZ))

    def describe(self) -> dict:
        return {
            "model": "fourband",
            **asdict(self.p),
            "include_identity": self.include_identity,
            "d3_layer": self.d3_layer,
            "warping": self.warping,
            "warping_matrix": self.warping_matrix,
        }

    def fingerprint(self) -> str:
        from .._storage import stable_hash

        return stable_hash(self.describe())

    # ---- Hamiltonian ---------------------------------------------------------------------------
    def hamiltonian(self, kx, ky) -> np.ndarray:
        """H(k), shape (..., 4, 4), eV. Complex k gives the analytic continuation (H is polynomial in k;
        not Hermitian off the real axis), used to locate complex branch points."""
        kx, ky = _k_arrays(kx, ky)
        p = self.p
        k2 = kx * kx + ky * ky
        c3 = kx**3 - 3.0 * kx * ky**2
        H = (p.A * kx)[..., None, None] * G_X + (p.A * ky)[..., None, None] * G_Y
        H = H + (p.M - p.B * k2)[..., None, None] * G_Z + p.Bl * G_BL + p.VL * G_VL
        if self.include_identity:
            eps0 = p.B0 * k2 + (p.D3 * c3 if self.d3_layer == "t0" else 0.0)
            H = H + np.asarray(eps0)[..., None, None] * I4
        if self.d3_layer == "tz":
            H = H + (p.D3 * c3)[..., None, None] * G_VL
        if self.warping:
            H = H + (self.warping * c3)[..., None, None] * WARPING_MATRICES[self.warping_matrix]
        return H

    def dhamiltonian(self, kx, ky) -> tuple[np.ndarray, np.ndarray]:
        kx, ky = _k_arrays(kx, ky)
        p = self.p
        dc3_x = 3.0 * (kx * kx - ky * ky)
        dc3_y = -6.0 * kx * ky
        one = np.ones_like(kx)
        dHx = (p.A * one)[..., None, None] * G_X + (-2.0 * p.B * kx)[..., None, None] * G_Z
        dHy = (p.A * one)[..., None, None] * G_Y + (-2.0 * p.B * ky)[..., None, None] * G_Z
        if self.include_identity:
            d3 = p.D3 if self.d3_layer == "t0" else 0.0
            dHx = dHx + (2.0 * p.B0 * kx + d3 * dc3_x)[..., None, None] * I4
            dHy = dHy + (2.0 * p.B0 * ky + d3 * dc3_y)[..., None, None] * I4
        if self.d3_layer == "tz":
            dHx = dHx + (p.D3 * dc3_x)[..., None, None] * G_VL
            dHy = dHy + (p.D3 * dc3_y)[..., None, None] * G_VL
        if self.warping:
            W = WARPING_MATRICES[self.warping_matrix]
            dHx = dHx + (self.warping * dc3_x)[..., None, None] * W
            dHy = dHy + (self.warping * dc3_y)[..., None, None] * W
        return dHx, dHy
