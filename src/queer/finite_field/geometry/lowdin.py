"""Near-Gamma precision patch for any band model: Brillouin-Wigner (Lowdin) downfolding in float64.

Where the v/c gap closes at Gamma, Omega and the radial metric are set by the k^2 mass of the pair
(m2 k^2 = 4e-17 eV at k = 1e-9 1/A in Eq. 9), far below the backward error eps_mach ||H|| of a full
eigensolver: float64 sum-over-states geometry is noise below |k| ~ 1e-8 1/A (later for absolute
wannier90 energies). This patch is the counterpart of RadialPatch for non-isotropic models.

Method, for |k| <= k_max (float64 throughout):
  1. Reference. E_ref = midgap of H(Gamma); D0, U0 = eigh(H(Gamma) - E_ref), with H(Gamma) = sum_R H_R
     summed with compensation. P = {v, c} plus every band within p_tol of them at Gamma; Q = the rest.
     The P block at Gamma is refined to U0_P^dag H(Gamma) U0_P with a correctly rounded H(Gamma) U0_P, so
     the Gamma gap is exact to ~eps eps_g(Gamma) instead of eps ||H(Gamma)||. A gap below the rounding
     level of the input, snap_tol = 16 eps sum_R max|H_R|, is snapped to exact degeneracy (the analogue
     of RadialPatch at x = 1).
  2. Expansion. H~(k) = U0^dag (H(k) - E_ref) U0 = D0 + k.V~ + N~(k): constant velocity V~ = U0^dag dH(0) U0
     and remainder N(k) = sum_R [e^{ik.d} - 1 - ik.d] H_R (d = R, or R + tau_n - tau_m in convention II),
     formed term by term without cancellation; dH(k) = dH(0) + sum_R i d [e^{ik.d} - 1] H_R likewise.
     Polynomial models (degree <= 3, e.g. the warped four-band model) use exact Taylor coefficients.
  3. Downfolding. For n in {v, c}: E_n = eig_n H_eff(E_n), H_eff(E) = H~_PP + H~_PQ (E - H~_QQ)^-1 H~_QP,
     iterated to convergence; psi_n = N_n (phi_n, Y_n phi_n) with Y_n = (E_n - H~_QQ)^-1 H~_QP.
  4. Pair term (|P| = 2). With H_eff = e0 + h.sigma the interband products follow from Pauli vectors,
     never from eigenvectors (X^a = <c|dH/da|v>, a in {s, t}; to first order in h_c - h_v):
         X^a X^b* = N_c^2 N_v^2 [(gamma^a x h).(gamma^b x h)* / |h|^2 + i h.(gamma^a x gamma^b*) / |h|],
     gamma^a = Pauli vector of the lifted velocity Lambda_c^dag dH_a Lambda_v (+ the Q admixture). The
     constant Dirac velocities a_s, a_t are kept apart from all O(k) terms, so the products of linear parts,
     a_s x (k_s a_s + k_t a_t) = k_t a_s x a_t and (k.a).(a_s x a_t) = 0, are exact: the k^2 mass is never
     recovered from a difference of O(k) numbers. Remote-band terms use float64 eigenpairs of H~(k)
     (they are not precision-critical). For |P| > 2 the pair term uses float64 eigenvectors of H_eff
     (backward error eps ||H_eff|| instead of eps ||H||).
  WYSV gauge: convention-I H with the covariant velocity dH - i[A^W, H] and the Wannier curvature terms
  of models.wannier.wannier_geometry_wysv.

Energies are returned relative to E_ref (so that eps_c - eps_v keeps full precision).
Accuracy (tests/test_precision_lowdin.py): <= 1e-11 relative on eps_g, Omega and g against mpmath for
1e-12 <= |k| <= 1e-6 1/A, at and off criticality, with or without a 5 eV absolute energy offset.
"""

from __future__ import annotations

import math

import numpy as np

from .kubo import BandGeometry

_EPS = np.finfo(float).eps
_FIELDS = ("eps_v", "eps_c", "omega_v", "omega_c", "g_cv", "g_v_total", "g_c_total", "v_cv", "deg_flag")


# ---- cancellation-free building blocks --------------------------------------------------------------
def _e1(x):
    """e^{ix} - 1 without cancellation."""
    s = np.sin(0.5 * x)
    return -2.0 * s * s + 1j * np.sin(x)


def _e2(x):
    """e^{ix} - 1 - ix without cancellation (series for sin x - x at |x| < 0.5)."""
    x2 = x * x
    ser = -x * x2 / 6.0 * (1 - x2 / 20 * (1 - x2 / 42 * (1 - x2 / 72 * (1 - x2 / 110 * (1 - x2 / 156)))))
    s = np.sin(0.5 * x)
    return -2.0 * s * s + 1j * np.where(np.abs(x) < 0.5, ser, np.sin(x) - x)


def _sum_compensated(x):
    """Sum over the first axis with Neumaier compensation: (hi, lo) with hi + lo the sum to ~eps^2 (complex)."""
    hi, lo = [], []
    for part in (np.real(x), np.imag(x)):
        s = np.zeros(part.shape[1:])
        comp = np.zeros_like(s)
        for term in part:
            t = s + term
            comp += np.where(np.abs(s) >= np.abs(term), (s - t) + term, (term - t) + s)
            s = t
        h = s + comp
        hi.append(h)
        lo.append(comp - (h - s))
    return hi[0] + 1j * hi[1], lo[0] + 1j * lo[1]


def _two_prod(a, b):
    """a * b = p + e exactly (Dekker splitting; float64 arrays without overflow)."""
    p = a * b
    t = 134217729.0 * a
    ah = t - (t - a)
    t = 134217729.0 * b
    bh = t - (t - b)
    al, bl = a - ah, b - bh
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def _matmul_rounded(H, U):
    """H @ U with every entry correctly rounded (exact products, math.fsum); H (n, n), U (n, m) complex."""
    Hr, Hi, Ur, Ui = H.real, H.imag, U.real, U.imag
    parts = {"re": ((Hr, Ur, 1.0), (Hi, Ui, -1.0)), "im": ((Hr, Ui, 1.0), (Hi, Ur, 1.0))}
    out = {}
    for key, pairs in parts.items():
        terms = [sign * np.stack(_two_prod(A[:, :, None], X[None, :, :]), axis=-1) for A, X, sign in pairs]
        terms = np.concatenate(terms, axis=-1)  # (n, n, m, 4): sum over the second and last axes
        terms = np.moveaxis(terms, 1, -2).reshape(H.shape[0], U.shape[1], -1)
        out[key] = np.array([[math.fsum(row) for row in block] for block in terms])
    return out["re"] + 1j * out["im"]


# ---- expansions of H about Gamma ---------------------------------------------------------------------
class WannierExpansion:
    """Exact near-Gamma expansion of a Wannier model (H_R, R_cart, gauge; convention II for "tba")."""

    kind = "wannier"

    def __init__(self, model, E_ref: float):
        self.conv = "II" if model.gauge == "tba" else "I"
        R = np.asarray(model.R_cart, dtype=float)
        n = model.n_orb
        H = np.asarray(model.H_R, dtype=complex)  # no copy for complex H_R
        kz = float(getattr(model, "kz", 0.0))
        if kz:
            H = H * np.exp(1j * kz * R[:, 2])[:, None, None]
        self.n = n
        self.R = R[:, :2].copy()
        self.H = H.reshape(len(R), n * n)
        shift = -E_ref * np.eye(n, dtype=complex)[None]
        self.H0I, H0I_lo = _sum_compensated(np.concatenate([H, shift]))  # convention-I H(0) - E_ref (hi, lo)
        J0 = np.stack([np.tensordot(1j * R[:, a], H, axes=1) for a in (0, 1)])  # sum_R i R_a H_R
        if self.conv == "II":
            tau = np.asarray(model.centres, dtype=float)
            self.dtau = tau[None, :, :2] - tau[:, None, :2]  # [m, n] = tau_n - tau_m
            self.Dz = np.exp(1j * kz * (tau[None, :, 2] - tau[:, None, 2])) if kz else np.ones((n, n))
            self.V0I = np.stack([J0[a] + 1j * self.dtau[..., a] * self.H0I for a in (0, 1)])
            self.H0, self.V = self.H0I * self.Dz, self.V0I * self.Dz
            self.H0_lo = H0I_lo * self.Dz  # (kz != 0: the rounding of H0I * Dz itself is not tracked)
        else:
            self.H0, self.V = self.H0I, J0
            self.H0_lo = H0I_lo
        self.scale = float(np.sum(np.max(np.abs(np.asarray(model.H_R)), axis=(1, 2))))  # rounding scale

    def terms(self, kx, ky, want_dN: bool = True):
        """N(k) = H(k) - H(0) - k.dH(0) and dN_a(k) = dH_a(k) - dH_a(0) (None unless want_dN), (nk, n, n)."""
        nk, n = kx.size, self.n
        conv2 = self.conv == "II"
        x = np.multiply.outer(kx, self.R[:, 0]) + np.multiply.outer(ky, self.R[:, 1])  # (nk, nR)
        e1 = _e1(x) if (want_dN or conv2) else None
        rows = (
            [_e2(x)]
            + ([e1] if conv2 else [])
            + ([e1 * (1j * self.R[:, 0]), e1 * (1j * self.R[:, 1])] if want_dN else [])
        )
        Y = (np.concatenate(rows) @ self.H).reshape(len(rows), nk, n, n)
        N = Y[0]
        dJx, dJy = (Y[-2], Y[-1]) if want_dN else (None, None)
        if not conv2:
            return N, dJx, dJy
        # convention II: H_mn(k) = e^{iy_mn} H^I_mn(k) with y = k.(tau_n - tau_m)
        dH1 = Y[1]
        y = kx[:, None, None] * self.dtau[..., 0] + ky[:, None, None] * self.dtau[..., 1]
        f1 = _e1(y)
        N2 = (N + _e2(y) * self.H0I + f1 * dH1) * self.Dz
        if not want_dN:
            return N2, None, None
        dN = [
            ((1.0 + f1) * (dJ + 1j * self.dtau[..., a] * dH1) + f1 * self.V0I[a]) * self.Dz
            for a, dJ in ((0, dJx), (1, dJy))
        ]
        return N2, dN[0], dN[1]


class TaylorExpansion:
    """Exact cubic Taylor expansion about Gamma of a model that is a polynomial of degree <= 3 in k.

    The coefficients come from central differences of the analytic derivative with step h (exact for a
    quadratic dH); the degree is verified by vanishing third differences, otherwise ValueError.
    """

    kind = "taylor"

    def __init__(self, model, E_ref: float, h: float = 1.0):
        n = model.n_orb
        self.n = n
        H0 = np.asarray(model.hamiltonian(0.0, 0.0), dtype=complex)
        self.H0 = H0 - E_ref * np.eye(n)
        d, dh = np.diagonal(H0).real, np.diagonal(self.H0).real  # TwoSum error of d - E_ref
        bb = dh - d
        self.H0_lo = np.diag((d - (dh - bb)) + (-E_ref - bb)).astype(complex)

        def dH(x, y):
            a, b = model.dhamiltonian(np.array([x], float), np.array([y], float))
            return np.stack([np.asarray(a)[0], np.asarray(b)[0]])

        d0 = dH(0.0, 0.0)
        px, mx, py, my = dH(h, 0), dH(-h, 0), dH(0, h), dH(0, -h)
        pp, pm, mp, mm = dH(h, h), dH(h, -h), dH(-h, h), dH(-h, -h)
        scale = max(float(np.max(np.abs(d0))), float(np.max(np.abs(px))), 1e-300)
        res = 0.0
        for e in ((1, 0), (0, 1), (1, 1)):
            f = [dH(j * h * e[0], j * h * e[1]) for j in (-1, 0, 1, 2)]
            res = max(res, float(np.max(np.abs(f[3] - 3 * f[2] + 3 * f[1] - f[0]))))
        if res > 1e-10 * scale:
            raise ValueError("model is not a polynomial of degree <= 3 in k (no exact Taylor expansion)")
        self.V = d0
        # dH_a(k) = V_a + sum_j H2[a, j] k_j + 1/2 sum_jl H3[a, j, l] k_j k_l
        self.H2 = np.stack([(px - mx) / (2 * h), (py - my) / (2 * h)], axis=1)
        H3 = np.empty((2, 2, 2, n, n), dtype=complex)
        H3[:, 0, 0] = (px - 2 * d0 + mx) / h**2
        H3[:, 1, 1] = (py - 2 * d0 + my) / h**2
        H3[:, 0, 1] = H3[:, 1, 0] = (pp - pm - mp + mm) / (4 * h * h)
        self.H3 = H3
        self.scale = float(np.linalg.norm(H0, 2))

    def terms(self, kx, ky, want_dN: bool = True):
        k = np.stack([kx, ky], axis=-1)
        lin = np.einsum("kj,ajmn->kamn", k, self.H2)
        quad = 0.5 * np.einsum("kj,kl,ajlmn->kamn", k, k, self.H3)
        N = np.einsum("ka,kamn->kmn", k, 0.5 * lin + quad / 3.0)
        dN = lin + quad
        return N, dN[:, 0], dN[:, 1]


def expansion_for(model, E_ref: float):
    """Exact near-Gamma expansion of `model` (Wannier H_R or polynomial), else ValueError."""
    if all(hasattr(model, a) for a in ("H_R", "R_cart", "gauge")):
        if model.gauge == "tba" and getattr(model, "centres", None) is None:
            raise ValueError("convention II needs the Wannier centres")
        if model.gauge == "wysv" and getattr(model, "r_R", None) is None:
            raise ValueError('gauge "wysv" needs the position matrix r_R')
        return WannierExpansion(model, E_ref)
    if hasattr(model, "dhamiltonian") and hasattr(model, "hamiltonian"):
        return TaylorExpansion(model, E_ref)
    raise ValueError("model provides neither H_R nor analytic derivatives")


# ---- Pauli algebra ---------------------------------------------------------------------------------
def _pauli(M):
    """(m0, m) with M = m0 + m.sigma for (..., 2, 2) matrices (complex parts for non-Hermitian M)."""
    m0 = 0.5 * (M[..., 0, 0] + M[..., 1, 1])
    m = np.stack(
        [
            0.5 * (M[..., 0, 1] + M[..., 1, 0]),
            0.5j * (M[..., 0, 1] - M[..., 1, 0]),
            0.5 * (M[..., 0, 0] - M[..., 1, 1]),
        ],
        axis=-1,
    )
    return m0, m


def _dot(a, b):
    return np.sum(a * b, axis=-1)


def _spinor(h, sign):
    """Normalized eigenvector of h.sigma for the eigenvalue sign*|h|, (nk, 2)."""
    hx, hy, hz = h[..., 0], h[..., 1], h[..., 2]
    r = np.sqrt(hx * hx + hy * hy + hz * hz)
    u1 = np.stack([hx - 1j * hy, sign * r - hz], axis=-1)  # null vector of the first row
    u2 = np.stack([sign * r + hz, hx + 1j * hy], axis=-1)  # null vector of the second row
    u = np.where(((sign * hz) <= 0)[..., None], u1, u2)
    nrm = np.sqrt(np.sum(np.abs(u) ** 2, axis=-1))
    bad = ~(nrm > 0)
    u = np.where(bad[..., None], np.array([1.0, 0.0]) if sign < 0 else np.array([0.0, 1.0]), u)
    return u / np.where(bad, 1.0, nrm)[..., None]


def _ct(a):
    return np.conj(np.swapaxes(a, -1, -2))


class LowdinPatch:
    """Precision geometry of the (v, c) pair near a gap closing at Gamma, for |k| <= k_max.

    model: a WannierModel ("tba", "naive" or "wysv" gauge; any kz) or a model whose H is a polynomial of
        degree <= 3 in k with analytic derivatives (e.g. FourBandModel with warping). Otherwise ValueError.
    k_max: patch radius (1/A). The patch is exact beyond it too; k_max sets p_tol and the chunking only.
    snap_critical: snap |eps_g(Gamma)| <= snap_tol (16 eps sum_R max|H_R|) to exact degeneracy.
    p_tol: bands within p_tol (eV) of the pair at Gamma join P (default 10 k_max ||dH(0)||, so that
        |H~_QP| <= 0.1 |E - H~_QQ| inside the patch and the self-consistency contracts by <= 1e-2 per step).
    Energies (eps_v, eps_c) are returned relative to E_ref, the midgap at Gamma.
    """

    def __init__(
        self,
        model,
        k_max: float = 1e-6,
        *,
        snap_critical: bool = True,
        p_tol: float | None = None,
        max_iter: int = 12,
        mem_bytes: float = 256e6,
    ):
        v, c = model.band_pair()
        n = model.n_orb
        self._model = model
        self._exp = None
        e0 = np.linalg.eigvalsh(np.asarray(model.hamiltonian(0.0, 0.0)))
        self.E_ref = float(0.5 * (e0[v] + e0[c]))
        self.gauge = getattr(model, "gauge", None)
        D0, U0 = np.linalg.eigh(self.exp.H0)
        self.k_max = float(k_max)
        self.n, self.v, self.c = n, v, c
        self.Vt = np.conj(U0.T)[None] @ self.exp.V @ U0[None]  # (2, n, n) constant velocity, U0 frame
        self.vnorm = float(max(np.linalg.norm(self.Vt[0], 2), np.linalg.norm(self.Vt[1], 2)))
        self.p_tol = float(10.0 * self.k_max * self.vnorm if p_tol is None else p_tol)
        P = np.nonzero((D0 > D0[v] - self.p_tol) & (D0 < D0[c] + self.p_tol))[0]  # contiguous (D0 sorted)
        self.P, self.Q = P, np.setdiff1d(np.arange(n), P)
        # P block at Gamma refined beyond eigh's eps ||H(Gamma)||: the Rayleigh matrix of U0_P with a correctly
        # rounded H0 U0_P (errors of U0_P enter at second order); the gap is then exact to ~eps eps_g(Gamma)
        U0P = U0[:, P]
        M = np.conj(U0P.T) @ (_matmul_rounded(self.exp.H0, U0P) + self.exp.H0_lo @ U0P)
        M = 0.5 * (M + np.conj(M.T))
        w, X = np.linalg.eigh(M)
        jv, jc = v - P[0], c - P[0]
        self.gap0_raw = float(w[jc] - w[jv])
        self.snap_tol = 16.0 * _EPS * self.exp.scale
        self.snapped = bool(snap_critical and abs(self.gap0_raw) <= self.snap_tol)
        if self.snapped:
            w[jv] = w[jc] = 0.5 * (w[jv] + w[jc])
            M = (X * w) @ np.conj(X.T)
        self.D0, self.U0, self.M_PP = D0, U0, M
        self.max_iter = int(max_iter)
        self.mem_bytes = float(mem_bytes)

    @property
    def exp(self):
        """The near-Gamma expansion (rebuilt from the model after unpickling)."""
        if self._exp is None:
            self._exp = expansion_for(self._model, self.E_ref)
        return self._exp

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_exp"] = None  # large (H_R-sized); rebuilt on demand
        return state

    @property
    def exact_pair(self) -> bool:
        """True if the pair term uses the cancellation-free Pauli formula (|P| = 2)."""
        return self.P.size == 2

    def describe(self) -> dict:
        return {
            "kind": "lowdin",
            "expansion": self.exp.kind,
            "gauge": self.gauge,
            "k_max": self.k_max,
            "E_ref": self.E_ref,
            "gap0_raw": self.gap0_raw,
            "snap_tol": float(self.snap_tol),
            "snapped": self.snapped,
            "n_P": int(self.P.size),
            "p_tol": self.p_tol,
            "exact_pair": self.exact_pair,
        }

    # ---- shared setup ----------------------------------------------------------------------------
    def _chunks(self, nk: int, per_point: float):
        size = int(max(1, min(4096, self.mem_bytes // (16.0 * self.n * self.n * per_point))))
        return [slice(i, min(nk, i + size)) for i in range(0, max(nk, 1), size)]

    def _setup(self, kx, ky, ks, kt, cs, sn, want_dN: bool = True):
        """Downfolding of H~(k) onto P and the self-consistent (v, c) energies."""
        P, Q, v, c = self.P, self.Q, self.v, self.c
        nk = kx.size
        N, dNx, dNy = self.exp.terms(kx, ky, want_dN)
        Nt = np.conj(self.U0.T) @ N @ self.U0
        Ht = Nt + kx[:, None, None] * self.Vt[0] + ky[:, None, None] * self.Vt[1]
        Ht[:, Q, Q] += self.D0[Q]
        Ht[:, P[:, None], P[None, :]] += self.M_PP
        st = dict(Ht=Ht, dNx=dNx, dNy=dNy, ks=ks, kt=kt)
        if Q.size:
            lam, WQ = np.linalg.eigh(Ht[:, Q][:, :, Q])
            B = _ct(WQ) @ Ht[:, Q][:, :, P]  # (nk, nQ, nP)
        else:
            lam, WQ, B = np.zeros((nk, 0)), np.zeros((nk, 0, 0), complex), np.zeros((nk, 0, P.size), complex)
        BH = _ct(B)
        st.update(lam=lam, WQ=WQ, B=B)

        def schur(g):  # H~_PQ G H~_QP for the diagonal resolvent g (nk, nQ) in the H~_QQ eigenbasis
            return BH @ (g[:, :, None] * B)

        st["schur"] = schur
        if P.size == 2:
            a0, a = _pauli(
                np.stack([cs * self.Vt[0] + sn * self.Vt[1], -sn * self.Vt[0] + cs * self.Vt[1]])[:, P][
                    :, :, P
                ]
            )
            a0, a = a0.real, a.real  # (2,), (2, 3) for s, t
            m0, hD0 = _pauli(self.M_PP)
            hD0 = hD0.real  # the (refined) Gamma mass vector
            L = hD0 + ks[:, None] * a[0] + kt[:, None] * a[1]  # exact linear part of h
            e0L = m0.real + ks * a0[0] + kt * a0[1]
            n0, nvec = _pauli(Nt[:, P][:, :, P])
            n0, nvec = n0.real, nvec.real
            sol = {}
            for name, sign in (("v", -1.0), ("c", 1.0)):
                E = e0L + n0 + sign * np.linalg.norm(L + nvec, axis=-1)
                for it in range(self.max_iter + 1):
                    s0, svec = _pauli(schur(1.0 / (E[:, None] - lam)))
                    h = L + nvec + svec.real
                    E_new = e0L + n0 + s0.real + sign * np.linalg.norm(h, axis=-1)
                    done = it == self.max_iter or np.all(np.abs(E_new - E) <= 4 * _EPS * np.abs(E_new))
                    E = E_new
                    if done:
                        break
                s0, svec = _pauli(schur(1.0 / (E[:, None] - lam)))
                h = L + nvec + svec.real
                sol[name] = dict(s0=s0.real, svec=svec.real, h=h, hn=np.linalg.norm(h, axis=-1))
            rv, rc = sol["v"], sol["c"]
            eps_g = (rc["s0"] - rv["s0"]) + rc["hn"] + rv["hn"]
            mid = e0L + n0 + 0.5 * (rc["s0"] + rv["s0"]) + 0.5 * (rc["hn"] - rv["hn"])
            st.update(
                a0=a0,
                a=a,
                hD0=hD0,
                L=L,
                nvec=nvec,
                rv=rv,
                rc=rc,
                eps_g=eps_g,
                E={"v": mid - 0.5 * eps_g, "c": mid + 0.5 * eps_g},
                phi={"v": _spinor(rv["h"], -1.0), "c": _spinor(rc["h"], 1.0)},
            )
        else:  # general |P|: float64 eigenpairs of H_eff(E)
            HPP = Ht[:, P][:, :, P]
            E_, phi = {}, {}
            for name, j in (("v", v - P[0]), ("c", c - P[0])):
                E = np.linalg.eigvalsh(HPP)[:, j]
                for it in range(self.max_iter + 1):
                    E_new = np.linalg.eigvalsh(HPP + schur(1.0 / (E[:, None] - lam)))[:, j]
                    done = it == self.max_iter or np.all(np.abs(E_new - E) <= 4 * _EPS * np.abs(E_new))
                    E = E_new
                    if done:
                        break
                w, X = np.linalg.eigh(HPP + schur(1.0 / (E[:, None] - lam)))
                E_[name], phi[name] = w[:, j], X[:, :, j]
            st.update(E=E_, phi=phi, eps_g=E_["c"] - E_["v"])
        return st

    def _frame(self, kx, ky, angle, s, t):
        kx, ky = np.broadcast_arrays(np.asarray(kx, dtype=float), np.asarray(ky, dtype=float))
        shape = kx.shape
        kx, ky = kx.ravel(), ky.ravel()
        cs, sn = math.cos(angle), math.sin(angle)
        if s is None or t is None:
            ks, kt = (kx, ky) if angle == 0.0 else (cs * kx + sn * ky, -sn * kx + cs * ky)
        else:
            ks, kt = (np.broadcast_to(np.asarray(z, dtype=float), shape).ravel() for z in (s, t))
        return shape, kx, ky, ks, kt, cs, sn

    # ---- public evaluation -------------------------------------------------------------------------
    def gap(self, kx, ky) -> np.ndarray:
        """eps_c - eps_v at Cartesian (kx, ky) (no geometry; cheaper than evaluate)."""
        shape, kx, ky, ks, kt, cs, sn = self._frame(kx, ky, 0.0, None, None)
        out = np.empty(kx.size)
        for sl in self._chunks(kx.size, 12):
            out[sl] = self._setup(kx[sl], ky[sl], ks[sl], kt[sl], cs, sn, want_dN=False)["eps_g"]
        return out.reshape(shape)

    def evaluate(self, kx, ky, angle: float = 0.0, s=None, t=None) -> BandGeometry:
        """Patched geometry at Cartesian (kx, ky) (model frame) for a field at `angle`.

        s, t: field-frame coordinates of the same points when known exactly (e.g. t = 0 on the central
        line); otherwise they are derived from (kx, ky). Metric components are [ss, st, tt].
        """
        shape, kx, ky, ks, kt, cs, sn = self._frame(kx, ky, angle, s, t)
        parts = [self._evaluate(kx[sl], ky[sl], ks[sl], kt[sl], cs, sn) for sl in self._chunks(kx.size, 24)]
        if len(parts) == 1:
            geo = parts[0]
        else:
            geo = BandGeometry(
                *(np.concatenate([getattr(p, f) for p in parts]) for f in _FIELDS),
                None,
                {"patched": np.ones(kx.size, dtype=bool)},
            )
        return geo.reshape(shape)

    def _evaluate(self, kx, ky, ks, kt, cs, sn) -> BandGeometry:
        nk, n = kx.size, self.n
        P, Q = self.P, self.Q
        st = self._setup(kx, ky, ks, kt, cs, sn)
        lam, WQ, B, E, phi, eps_g = st["lam"], st["WQ"], st["B"], st["E"], st["phi"], st["eps_g"]
        U0 = self.U0
        V0 = self.exp.V
        Vorb = np.stack(
            [cs * V0[0] + sn * V0[1], -sn * V0[0] + cs * V0[1]]
        )  # constant (s, t) velocity, orbital
        dN = np.stack([cs * st["dNx"] + sn * st["dNy"], -sn * st["dNx"] + cs * st["dNy"]])  # (2, nk, n, n)
        wysv = self.gauge == "wysv"
        if wysv:
            AWx, AWy, OmW = self._model.position_connection(kx, ky)
            A_orb = np.stack([cs * AWx + sn * AWy, -sn * AWx + cs * AWy])

        # remote eigenpairs (float64)
        eps_all, Wt = np.linalg.eigh(st["Ht"])
        rem = np.setdiff1d(np.arange(n), [self.v, self.c])  # every other band, incl. the rest of P
        eps_rem = eps_all[:, rem]
        Psi_rem = U0[None] @ Wt[:, :, rem]  # orbital frame

        # accurate (v, c) vectors: psi_n = N_n Lambda_n phi_n, Lambda_n = U0_P + U0_Q Y_n (orbital frame)
        g = {m: 1.0 / (E[m][:, None] - lam) for m in ("v", "c")}
        Y = {m: WQ @ (g[m][:, :, None] * B) for m in ("v", "c")}
        U0P, U0Q = U0[:, P], U0[:, Q]
        Lam = {m: U0P[None] + U0Q[None] @ Y[m] for m in ("v", "c")}
        Nrm2, psi = {}, {}
        for m in ("v", "c"):
            Yphi = np.einsum("kqp,kp->kq", Y[m], phi[m])
            Nrm2[m] = 1.0 / (1.0 + np.sum(np.abs(Yphi) ** 2, axis=-1))
            psi[m] = np.sqrt(Nrm2[m])[:, None] * np.einsum("kip,kp->ki", Lam[m], phi[m])
        Eps = np.stack([E["v"], E["c"]], axis=-1)
        Psi_vc = np.stack([psi["v"], psi["c"]], axis=-1)  # (nk, n, 2)
        PsiH = _ct(Psi_vc)

        # remote terms of the sums over states (float64)
        row = PsiH[None] @ (Vorb[:, None] + dN)  # (2, nk, 2, n)
        X_rem = row @ Psi_rem[None]
        d_rem = Eps[:, :, None] - eps_rem[:, None, :]
        if wysv:
            rowA = PsiH[None] @ A_orb
            X_rem = X_rem + 1j * d_rem[None] * (rowA @ Psi_rem[None])
        inv2 = 1.0 / d_rem**2
        Xs, Xt = X_rem[0], X_rem[1]
        cr = Xs * np.conj(Xt)
        om_rem = -2.0 * np.sum(cr.imag * inv2, axis=-1)  # (nk, 2): v, c
        g_rem = np.stack(
            [
                np.sum(np.abs(Xs) ** 2 * inv2, -1),
                np.sum(cr.real * inv2, -1),
                np.sum(np.abs(Xt) ** 2 * inv2, -1),
            ],
            axis=-1,
        )  # (nk, 2, 3)

        # the (v, c) pair term
        if self.exact_pair:
            Qab = self._pair_products(st, Y, Lam, dN, A_orb if wysv else None, Nrm2["c"] * Nrm2["v"], cs, sn)
        else:
            Xvc = row @ Psi_vc[None]
            if wysv:
                Xvc = Xvc + 1j * (Eps[:, :, None] - Eps[:, None, :])[None] * (rowA @ Psi_vc[None])
            Xcs, Xct = Xvc[0][:, 1, 0], Xvc[1][:, 1, 0]
            Qab = (np.abs(Xcs) ** 2 + 0j, Xcs * np.conj(Xct), np.abs(Xct) ** 2 + 0j)
        Q_ss, Q_st, Q_tt = Qab
        with np.errstate(divide="ignore", invalid="ignore"):
            ig2 = 1.0 / eps_g**2
            om_pair = -2.0 * Q_st.imag * ig2
            g_cv = np.stack([Q_ss.real * ig2, Q_st.real * ig2, Q_tt.real * ig2], axis=-1)
        omega_c = om_pair + om_rem[:, 1]
        omega_v = -om_pair + om_rem[:, 0]
        if wysv:  # Re <n|F^W|n> + 2 Im sum_m Abar_s,nm Abar_t,mn over all bands
            A_all = rowA @ np.concatenate([Psi_vc, Psi_rem], axis=-1)[None]
            extra = np.real(np.einsum("kni,kij,kjn->kn", PsiH, OmW, Psi_vc))
            extra = extra + 2.0 * np.sum((A_all[0] * np.conj(A_all[1])).imag, axis=-1)
            omega_v, omega_c = omega_v + extra[:, 0], omega_c + extra[:, 1]
        with np.errstate(
            divide="ignore", invalid="ignore"
        ):  # v_cv in a fixed gauge: the larger component real
            big_s = Q_ss.real >= Q_tt.real
            ref = np.sqrt(np.maximum(np.where(big_s, Q_ss.real, Q_tt.real), 0.0))
            ok = ref > 0
            vs = np.where(big_s, ref + 0j, Q_st / np.where(ok, ref, 1.0))
            vt = np.where(big_s, np.conj(Q_st) / np.where(ok, ref, 1.0), ref + 0j)
            v_cv = np.stack([np.where(ok, vs, 0.0), np.where(ok, vt, 0.0)], axis=-1)
        mid = 0.5 * (E["v"] + E["c"])
        return BandGeometry(
            mid - 0.5 * eps_g,
            mid + 0.5 * eps_g,
            omega_v,
            omega_c,
            g_cv,
            g_cv + g_rem[:, 0],
            g_cv + g_rem[:, 1],
            v_cv,
            ~(eps_g > 0),
            None,
            {"patched": np.ones(nk, dtype=bool)},
        )

    def _pair_products(self, st, Y, Lam, dN, A_orb, N2, cs, sn):
        """(Q_ss, Q_st, Q_tt), Q_ab = X^a_cv conj(X^b_cv), from Pauli vectors with the linear part kept exact."""
        P, Q = self.P, self.Q
        a0, a, hD0, L, nvec, rv, rc, eps_g = (
            st[k] for k in ("a0", "a", "hD0", "L", "nvec", "rv", "rc", "eps_g")
        )
        nk = L.shape[0]
        Vst = np.stack([cs * self.Vt[0] + sn * self.Vt[1], -sn * self.Vt[0] + cs * self.Vt[1]])
        YcH = _ct(Y["c"])
        LcH = _ct(Lam["c"])
        q = nvec + 0.5 * (rv["svec"] + rc["svec"])  # O(k^2) part of h at the midpoint
        hm = L + q
        with np.errstate(divide="ignore", invalid="ignore"):
            hmn = np.linalg.norm(hm, axis=-1)
            hhat = hm / hmn[:, None]
            # h_c - h_v = Pauli(-eps_g Y_c^dag Y_v); delta = (h^_c - h^_v)/2 to first order
            if Q.size:
                gcv = 1.0 / ((st["E"]["c"][:, None] - st["lam"]) * (st["E"]["v"][:, None] - st["lam"]))
                dh = _pauli(-eps_g[:, None, None] * st["schur"](gcv))[1].real
            else:
                dh = np.zeros((nk, 3))
            delta = (dh - hhat * _dot(hhat, dh)[:, None]) / (2.0 * hmn[:, None])
        gp = []  # gamma'^a: everything but the constant Dirac velocity a_a (complex, O(k))
        for i in (0, 1):
            Va = Vst[i]
            M = (
                Va[np.ix_(P, Q)][None] @ Y["v"]
                + YcH @ Va[np.ix_(Q, P)][None]
                + YcH @ (Va[np.ix_(Q, Q)][None] @ Y["v"])
            )
            M = M + LcH @ (dN[i] @ Lam["v"])
            b0, bvec = _pauli(M)
            gam = bvec + (a0[i] + b0)[:, None] * delta
            if A_orb is not None:  # covariant velocity: + i eps_g <c|A^W|v>
                al0, al = _pauli(LcH @ (A_orb[i] @ Lam["v"]))
                gam = gam + 1j * eps_g[:, None] * (al + al0[:, None] * delta)
            gp.append(gam)
        axb = np.cross(a[0], a[1])
        aL = [np.cross(a[0], hD0) + st["kt"][:, None] * axb, np.cross(a[1], hD0) - st["ks"][:, None] * axb]
        Cr = [aL[i] + np.cross(a[i], q) + np.cross(gp[i], hm) for i in (0, 1)]  # gamma^a x h

        def prod(i, j):
            t1 = _dot(Cr[i], np.conj(Cr[j]))
            lab = 0.0 if i == j else (1.0 if i == 0 else -1.0) * float(hD0 @ axb)  # L.(a_i x a_j)
            gpj = np.conj(gp[j])
            t2 = (
                lab
                - _dot(gpj, aL[i])
                + _dot(gp[i], aL[j])
                + _dot(L, np.cross(gp[i], gpj))
                + _dot(q, np.cross(a[i] + gp[i], a[j] + gpj))
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                return N2 * (t1 / hmn**2 + 1j * t2 / hmn)

        return prod(0, 0), prod(0, 1), prod(1, 1)

    # ---- certificate ------------------------------------------------------------------------------
    def overlap_error(
        self, model=None, k_lo: float | None = None, k_hi: float | None = None, n: int = 7, angles=(0.0, 0.9)
    ) -> float:
        """Max relative difference of eps_g, Omega_c and g^{cv} (per tensor norm) between the patch and
        float64 kubo on [k_lo, k_hi] (default [k_max, 10 k_max], where float64 is reliable), along `angles`."""
        from .kubo import band_geometry

        model = self._model if model is None else model
        k_lo = self.k_max if k_lo is None else k_lo
        k_hi = 10.0 * k_lo if k_hi is None else k_hi
        k = np.geomspace(k_lo, k_hi, n)
        err = 0.0
        for al in angles:
            kx, ky = k * math.cos(al), k * math.sin(al)
            ref = band_geometry(model, kx, ky)
            pat = self.evaluate(kx, ky)
            err = max(
                err,
                float(np.max(np.abs(pat.eps_g - ref.eps_g) / ref.eps_g)),
                float(np.max(np.abs(pat.omega_c - ref.omega_c) / np.abs(ref.omega_c))),
                float(np.max(np.abs(pat.g_cv - ref.g_cv) / np.max(np.abs(ref.g_cv), axis=-1, keepdims=True))),
            )
        return err
