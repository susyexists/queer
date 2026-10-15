"""Electron-phonon arrays and numerical self-energy calculations."""

from pathlib import Path

import numpy as np

from ..io.epw import read_epw_directory
from ..physics.electron_phonon import electron_hole_response, reduced_coupling
from ..physics.electron_phonon import self_energy as epw_self_energy


class EPWModel:
    """EPW modes/q/k arrays: phonons and couplings in meV, electrons in eV.

    Temperature denotes k_B*T in eV. Self-energy preserves the legacy EPW
    normalization (a 1e-6 coupling factor and negative imaginary broadening).
    """

    def __init__(self, ph, g_complex, e_k, e_kq, ef=0.0):
        self.ph = np.asarray(ph, dtype=float).copy()
        self.g_complex = np.asarray(g_complex, dtype=complex).copy()
        self.e_k = np.asarray(e_k, dtype=float).copy()
        self.e_kq = np.asarray(e_kq, dtype=float).copy()
        self.ef = float(ef)
        if self.ph.ndim != 2 or self.g_complex.ndim != 3:
            raise ValueError("ph and g_complex require shapes (nph,nq) and (nph,nq,nk)")
        self.nph, self.nq, self.nk = self.g_complex.shape
        if not all((self.nph, self.nq, self.nk)):
            raise ValueError("EPW dimensions must be positive")
        if (
            self.ph.shape != (self.nph, self.nq)
            or self.e_k.shape != (self.nq, self.nk)
            or self.e_kq.shape != self.e_k.shape
        ):
            raise ValueError("EPW array dimensions are inconsistent")
        if not all(
            np.isfinite(a).all() for a in (self.ph, self.g_complex, self.e_k, self.e_kq)
        ) or not np.isfinite(self.ef):
            raise ValueError("EPW arrays and ef must be finite")
        self.q_path = np.arange(self.nq)
        self._sync_coupling()

    def _sync_coupling(self):
        self.g_re = self.g_complex.real
        self.g_im = self.g_complex.imag
        self.g_abs = np.abs(self.g_complex)
        self.g_sq_mean = np.sqrt(np.mean(np.abs(self.g_complex) ** 2, axis=2))
        if hasattr(self, "g0"):
            self.reduce_g()

    @classmethod
    def from_directory(cls, directory, nk=None, nq=None, nph=None, ef=None):
        metadata, arrays = read_epw_directory(directory, nk=nk, nq=nq, nph=nph, ef=ef)
        result = cls(
            arrays["ph"], arrays["gkk_re"] + 1j * arrays["gkk_im"], arrays["enk"], arrays["enkq"], metadata.ef
        )
        result.path = Path(directory).expanduser()
        return result

    def load_data(self):
        """Reload directory data for compatibility with existing workflows."""
        if not hasattr(self, "path"):
            raise ValueError("load_data requires a model constructed from a directory")
        loaded = EPWModel.from_directory(self.path, nk=self.nk, nq=self.nq, nph=self.nph, ef=self.ef)
        reduced = hasattr(self, "g0")
        self.__dict__.update(loaded.__dict__)
        if reduced:
            self.reduce_g()
        return self

    def reduce_g(self):
        """Compute g0 = g*sqrt(phonon energy); unstable negative modes raise."""
        self.g0 = reduced_coupling(self.ph, self.g_complex)
        return self.g0

    def self_energy(self, q, temperature=0.025, broadening=1e-7):
        """Return (nph,) for scalar q or (Nq,nph) for a list of indices."""
        broadening = float(broadening)
        if not np.isfinite(broadening) or broadening <= 0:
            raise ValueError("broadening must be finite and positive")
        indices = np.asarray(q)
        scalar = indices.ndim == 0
        if indices.ndim > 1 or (indices.size and indices.dtype.kind not in "iu"):
            raise ValueError("q must be an integer or a one-dimensional list of integer indices")
        indices = indices.reshape(-1).astype(int)
        if np.any(indices < 0) or np.any(indices >= self.nq):
            raise ValueError("q index is outside the model")
        if not hasattr(self, "g0"):
            self.reduce_g()
        result = epw_self_energy(
            self.ph[:, indices],
            self.g0[:, indices],
            self.e_k[indices],
            self.e_kq[indices],
            self.ef,
            temperature,
            broadening,
        )
        return result[0] if scalar else result

    def calculate_self_energy(self, q):
        return self.self_energy(q)

    def calculate_eph(self, n, q, k):
        if not hasattr(self, "g0"):
            self.reduce_g()
        return np.abs(self.g0[n, q, k]) ** 2 * 1e-6

    def calculate_suscep(self, n, q, k, delta=1e-7, temperature=0.025):
        if not np.isfinite(delta) or delta <= 0:
            raise ValueError("delta must be finite and positive")
        return electron_hole_response(
            self.e_k[q, k], self.e_kq[q, k], self.ph[n, q], self.ef, temperature, delta
        )

    def fix_model(self, ph_tolerance=0.15, g1_tolerance=30, g2_tolerance=10, offset=50):
        """Apply the existing crossing heuristic while keeping all modes aligned."""
        from ._epw_tracking import g_cross, ph_cross

        self._apply_crossings(ph_cross(self.ph, tolerance=ph_tolerance, offset=offset))
        for tolerance in (g1_tolerance, g2_tolerance):
            self._apply_crossings(g_cross(self.g_sq_mean, tolerance=tolerance, offset=offset))
        return self

    def _apply_crossings(self, crossings):
        from ._epw_tracking import untangle

        self.ph = untangle(self.ph, crossings)
        self.g_complex = untangle(self.g_complex, crossings)
        self._sync_coupling()

    def swap_branch(self, index1, index2, qpoint=None):
        if qpoint is False:
            qpoint = None
        if not 0 <= index1 < self.nph or not 0 <= index2 < self.nph:
            raise ValueError("branch index outside the model")
        if qpoint is None:
            self.ph[[index1, index2]] = self.ph[[index2, index1]]
            self.g_complex[[index1, index2]] = self.g_complex[[index2, index1]]
            self._sync_coupling()
        else:
            if not isinstance(qpoint, (int, np.integer)) or not 0 <= qpoint <= self.nq:
                raise ValueError("qpoint outside the model")
            self._apply_crossings([(index1, index2, qpoint)])
        return self
