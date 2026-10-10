"""Physical reciprocal-plane views of Queer's native Hamiltonian models.

The view is internal to finite-field algorithms. Public model coordinates remain
fractional, and nonorthogonal geometry always uses the native covariant engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh

from ..capabilities import CapabilityError
from ..observables._tb import band_geometry, cell, degeneracy_tolerance, reciprocal
from ..observables.geometry import _curvatures


@dataclass(frozen=True)
class FieldPlane:
    """An oriented plane: fractional origin, Cartesian unit axes.

    ``direction`` and ``transverse`` are orthonormal physical vectors. Coordinates
    in this plane have units inverse Angstrom; their orientation fixes Hall signs.
    """

    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    transverse: tuple[float, float, float] = (0.0, 1.0, 0.0)
    _axes: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        arrays = [np.asarray(x, dtype=float) for x in (self.origin, self.direction, self.transverse)]
        if any(a.shape != (3,) or not np.isfinite(a).all() for a in arrays):
            raise ValueError("plane origin and axes must be finite three-vectors")
        axes = np.stack(arrays[1:])
        if not np.allclose(axes @ axes.T, np.eye(2), atol=1e-12, rtol=0):
            raise ValueError("plane Cartesian axes must be orthonormal unit vectors")
        axes.setflags(write=False)
        object.__setattr__(self, "_axes", axes)
        for name, a in zip(("origin", "direction", "transverse"), arrays):
            object.__setattr__(self, name, tuple(float(x) for x in a))

    @classmethod
    def from_mapping(cls, value=None):
        if isinstance(value, cls):
            return value
        return cls(**(value or {}))

    @property
    def normal(self):
        return np.cross(self._axes[0], self._axes[1])

    def axes(self, angle=0.0):
        if not np.isfinite(angle):
            raise ValueError("plane angle must be finite radians")
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, s], [-s, c]]) @ self._axes

    def fractional(self, model, s, t, angle=0.0):
        s, t = np.broadcast_arrays(np.asarray(s), np.asarray(t))
        if not np.isfinite(s).all() or not np.isfinite(t).all():
            raise ValueError("plane coordinates must be finite")
        xy = np.stack([s, t], axis=-1)
        return np.asarray(self.origin) + (xy @ self.axes(angle)) @ np.linalg.inv(reciprocal(model))

    def describe(self):
        return {
            "origin": list(self.origin),
            "direction": list(self.direction),
            "transverse": list(self.transverse),
            "coordinates": "Cartesian Angstrom^-1",
        }


class NativePlaneModel:
    """A bounded 2D view that retains native operators and spin channels."""

    native_band_geometry = True
    is_isotropic_geometry = False

    def __init__(
        self,
        model,
        *,
        plane=None,
        band_pair=None,
        occupied_bands=None,
        layer_projector=None,
        degeneracy_tol=None,
    ):
        self.source_model = model
        self.plane = FieldPlane.from_mapping(plane)
        self.n_orb = self.nbnd = int(model.nbnd)
        multiplicity = getattr(model, "spin_degeneracy", 1)
        if (
            isinstance(multiplicity, (bool, np.bool_))
            or not isinstance(multiplicity, (int, np.integer))
            or multiplicity not in (1, 2)
        ):
            raise ValueError("spin_degeneracy must be integer one or two")
        self.spin_degeneracy = int(multiplicity)
        self.spin_mode = getattr(model, "spin_mode", "unspecified")
        self.degeneracy_tol = degeneracy_tolerance(model, degeneracy_tol)
        if band_pair is None:
            if occupied_bands is None or isinstance(occupied_bands, bool):
                raise ValueError(
                    "finite-field models require band_pair or an unambiguous occupied_bands count"
                )
            count = int(occupied_bands)
            if count != occupied_bands or not 0 < count < self.nbnd:
                raise ValueError("occupied_bands must select a nonempty filled and empty band")
            channels = np.asarray(getattr(model, "band_channels", np.zeros(self.nbnd, int)))
            if len(np.unique(channels)) != 1:
                raise ValueError(
                    "collinear finite-field calculations require an explicit same-channel band_pair"
                )
            band_pair = (count - 1, count)
        pair = np.asarray(band_pair)
        if pair.shape != (2,) or pair.dtype.kind not in "iu" or not 0 <= pair[0] < pair[1] < self.nbnd:
            raise ValueError("band_pair must be two increasing zero-based model band indices")
        self._pair = tuple(int(x) for x in pair)
        channels = np.asarray(getattr(model, "band_channels", np.zeros(self.nbnd, int)))
        if channels[pair[0]] != channels[pair[1]]:
            raise ValueError("finite-field band_pair must stay in one spin channel")
        self.n_occ = self._pair[0] + 1
        self._projector = (
            layer_projector if layer_projector is not None else getattr(model, "layer_projector", None)
        )
        self.lattice = cell(model)
        self.positions = getattr(model, "centres", getattr(model, "orbital_positions", None))
        self.supports_generalized = hasattr(model, "overlap") and not hasattr(model, "hopping_matrices")
        self.supports_lowdin_patch = hasattr(model, "hopping_matrices") and not self.supports_generalized
        self.complex_k = hasattr(model, "analytic_hamiltonian") and not self.supports_generalized
        self.supports_wavefunction_phases = not self.supports_generalized
        self.supports_tdse = not self.supports_generalized
        self.kz = 0.0
        # Supply an exact orthogonal expansion in the plane to the precision
        # engine. Origin phases are absorbed once, before cancellation-free sums.
        if hasattr(model, "hopping_matrices"):
            axes = np.vstack([self.plane._axes, self.plane.normal])
            self.R_cart = (model.translations @ self.lattice) @ axes.T
            phase = np.exp(2j * np.pi * (model.translations @ np.asarray(self.plane.origin)))
            self.H_R = model.hopping_matrices * phase[:, None, None]
            self.centres = None if self.positions is None else np.asarray(self.positions) @ axes.T
            raw_position = getattr(model, "position_matrices", None)
            if raw_position is not None:
                self.r_R = np.einsum("rija,ba->rijb", raw_position, axes) * phase[:, None, None, None]
                self.gauge = "wysv"
            elif self.centres is not None:
                self.r_R = np.zeros((*self.H_R.shape, 3), dtype=complex)
                zero = np.flatnonzero(np.all(model.translations == 0, axis=1))
                if len(zero):
                    self.r_R[zero[0], np.arange(self.nbnd), np.arange(self.nbnd)] = self.centres
                    self.gauge = "wysv"
                else:
                    self.gauge = "naive"
            else:
                self.gauge = "naive"
        # Dynamics in a moving/embedded basis needs the covariant propagator;
        # the simple orbital Magnus validator is only certified for fixed bases.
        if self.positions is not None and np.any(np.asarray(self.positions) @ self.plane._axes.T):
            self.supports_tdse = False
        if getattr(model, "position_matrices", None) is not None:
            if np.any(getattr(model, "position_matrices") @ self.plane._axes.T):
                self.supports_tdse = False

    def band_pair(self):
        return self._pair

    def _points(self, kx, ky):
        kx, ky = np.broadcast_arrays(np.asarray(kx), np.asarray(ky))
        return self.plane.fractional(self.source_model, kx, ky).reshape(-1, 3), kx.shape

    def hamiltonian(self, kx, ky):
        points, shape = self._points(kx, ky)
        if np.iscomplexobj(points):
            if not self.complex_k:
                raise CapabilityError("complex-k continuation is unavailable for this model")
            values = [self.source_model.analytic_hamiltonian(k) for k in points]
        else:
            values = [self.source_model.hamiltonian(k) for k in points]
        return np.asarray(values).reshape(shape + (self.nbnd, self.nbnd))

    def dhamiltonian(self, kx, ky):
        from ..observables._tb import derivatives

        points, shape = self._points(kx, ky)
        values = [
            np.einsum("ba,aij->bij", self.plane._axes, derivatives(self.source_model, k)["H"]) for k in points
        ]
        a = np.asarray(values).reshape(shape + (2, self.nbnd, self.nbnd))
        return a[..., 0, :, :], a[..., 1, :, :]

    def hamiltonian_and_derivatives(self, kx, ky):
        x, y = self.dhamiltonian(kx, ky)
        return self.hamiltonian(kx, ky), x, y

    def gap(self, kx, ky, bands=None):
        if bands is not None and tuple(bands) != self._pair:
            raise ValueError("the plane view fixes its band pair at construction")
        points, shape = self._points(kx, ky)
        v, c = self._pair
        if hasattr(self.source_model, "hopping_matrices"):
            from ..execution import ExecutionOptions
            from ..models.wannier import _assemble

            options = getattr(self.source_model, "execution", ExecutionOptions()).resolved()
            batch = options.bounded_batch_size(16 * self.nbnd**2 * 6)
            out = np.empty(len(points))
            for start in range(0, len(points), batch):
                matrices, _ = _assemble(
                    self.source_model.translations,
                    self.source_model.hopping_matrices,
                    points[start : start + batch],
                )
                reference = np.trace(matrices, axis1=-2, axis2=-1).real / self.nbnd
                matrices -= reference[:, None, None] * np.eye(self.nbnd)
                e = np.linalg.eigvalsh(matrices)
                out[start : start + batch] = e[:, c] - e[:, v]
            return out.reshape(shape)
        out = []
        for k in points:
            h = self.source_model.hamiltonian(k)
            s = self.source_model.overlap(k) if hasattr(self.source_model, "overlap") else np.eye(self.nbnd)
            reference = np.trace(np.linalg.solve(s, h)).real / self.nbnd
            channels = np.asarray(getattr(self.source_model, "band_channels", np.zeros(self.nbnd, int)))
            if len(np.unique(channels)) > 1:
                energies = np.empty(self.nbnd)
                for channel in np.unique(channels):
                    indices = np.flatnonzero(channels == channel)
                    hc = h[np.ix_(indices, indices)]
                    sc = s[np.ix_(indices, indices)]
                    channel_reference = np.trace(np.linalg.solve(sc, hc)).real / len(indices)
                    energies[indices] = eigh(hc - channel_reference * sc, sc, eigvals_only=True)
            else:
                energies = eigh(h - reference * s, s, eigvals_only=True)
            out.append(energies[c] - energies[v])
        return np.asarray(out).reshape(shape)

    def geometry(self, kx, ky, angle=0.0, **kwargs):
        from .geometry.kubo import BandGeometry

        points, shape = self._points(kx, ky)
        axes = self.plane.axes(angle)
        v, c = self._pair
        rows = []
        for k in points:
            data = band_geometry(self.source_model, k, self.degeneracy_tol)
            e, connections = data["energies"], data["connections"]
            a = np.einsum("ba,anm->bnm", axes, connections)
            g = np.outer(a[:, v, c], a[:, v, c].conj()).real
            totals = []
            valid = True
            for band in (v, c):
                same = data["band_channels"] == data["band_channels"][band]
                unresolved = (np.abs(e - e[band]) <= self.degeneracy_tol) & same
                unresolved[band] = False
                valid &= not np.any(unresolved)
                ac = a[:, band].copy()
                ac[:, band] = 0
                totals.append((ac @ ac.conj().T).real)
            curvature = _curvatures(self.source_model, k, data, kwargs.get("engine", "direct"))
            omega = np.einsum("a,nab,b->n", axes[0], curvature, axes[1])
            velocity = np.einsum("ba,a->b", axes, data["velocities"][:, c, v])

            def metric(x):
                return np.array([x[0, 0], x[0, 1], x[1, 1]])

            rows.append(
                (
                    e[v],
                    e[c],
                    omega[v],
                    omega[c],
                    metric(g),
                    metric(totals[0]),
                    metric(totals[1]),
                    velocity,
                    not valid,
                )
            )
        values = [np.asarray([r[i] for r in rows]) for i in range(9)]
        result = BandGeometry(*values).reshape(shape)
        result.extras["gap"] = self.gap(kx, ky)
        result.extras["native_covariant"] = True
        # Degenerate individual geometry is unavailable, rather than a metric
        # obtained by silently dropping the singular pair contribution.
        if np.any(result.deg_flag):
            for name in ("omega_v", "omega_c", "g_cv", "g_v_total", "g_c_total", "v_cv"):
                a = getattr(result, name)
                a[result.deg_flag] = np.nan
        return result

    def position_connection(self, kx, ky):
        from ..observables._tb import derivatives, position_matrix

        points, shape = self._points(kx, ky)
        values = []
        a, b = self.plane._axes
        for k in points:
            r = position_matrix(self.source_model, k)
            dr = derivatives(self.source_model, k)["r"]
            curl = np.einsum("a,b,abij->ij", a, b, dr) - np.einsum("a,b,abij->ij", b, a, dr)
            values.append((np.einsum("a,aij->ij", a, r), np.einsum("a,aij->ij", b, r), curl))
        result = np.asarray(values).reshape(shape + (3, self.nbnd, self.nbnd))
        return tuple(result[..., i, :, :] for i in range(3))

    def phase_states(self, s, t, angle, bands=None):
        if not self.supports_wavefunction_phases:
            raise CapabilityError("corrected wavefunction phases are not validated for OpenMX")
        s, t = np.broadcast_arrays(s, t)
        points = self.plane.fractional(self.source_model, s, t, angle).reshape(-1, 3)
        v, c = self._pair if bands is None else bands
        axis = self.plane.axes(angle)[0]
        # Dipoles and coefficients must share one eigensystem at each point.
        # Independent diagonalizations may choose different U(1) band phases.
        data = [band_geometry(self.source_model, k, self.degeneracy_tol) for k in points]
        uc = np.asarray([row["coefficients"][:, c] for row in data])
        uv = np.asarray([row["coefficients"][:, v] for row in data])
        velocity = np.asarray([axis @ row["velocities"][:, c, v] for row in data])
        return uc, uv, velocity

    def phase_links(self, s, t, angle, bands=None, uc=None, uv=None):
        from ..observables._tb import basis_overlap, states

        s, t = np.broadcast_arrays(s, t)
        points = self.plane.fractional(self.source_model, s, t, angle).reshape(-1, 3)
        v, c = self._pair if bands is None else bands
        if uc is None or uv is None:
            solution = states(self.source_model, points)
            uc, uv = solution.coefficients[:, :, c], solution.coefficients[:, :, v]
        forward, reverse = [], []
        for i in range(len(points) - 1):
            overlap = basis_overlap(self.source_model, points[i], points[i + 1])
            cc, cv = uc[i], uv[i]
            nc, nv = uc[i + 1], uv[i + 1]
            forward.append(cc.conj() @ overlap @ nc)
            reverse.append(nv.conj() @ overlap.conj().T @ cv)
        return np.asarray(forward), np.asarray(reverse)

    def layer_projector(self):
        if self._projector is None:
            raise CapabilityError("diabatic crossings require an explicit layer/subspace projector")
        p = np.asarray(self._projector() if callable(self._projector) else self._projector, dtype=complex)
        if p.shape != (self.nbnd, self.nbnd) or not np.isfinite(p).all():
            raise ValueError("layer projector must be a finite model-basis square matrix")
        if not np.allclose(p, p.conj().T, atol=1e-10) or not np.allclose(p @ p, p, atol=1e-10):
            raise ValueError("layer projector must be Hermitian and idempotent")
        if self.supports_generalized:
            raise CapabilityError("nonorthogonal diabatic layer projection is not yet validated")
        return p

    def describe(self):
        return {
            "model": type(self.source_model).__name__,
            "n_orb": self.nbnd,
            "band_pair": list(self._pair),
            "spin_mode": self.spin_mode,
            "spin_degeneracy": self.spin_degeneracy,
            "plane": self.plane.describe(),
            "generalized": self.supports_generalized,
            "complex_k": self.complex_k,
            "geometry_definition": "finite_basis_interband",
            "degeneracy_tolerance_eV": self.degeneracy_tol,
            "supports_wavefunction_phases": self.supports_wavefunction_phases,
            "supports_tdse": self.supports_tdse,
        }

    def fingerprint(self):
        from ._storage import stable_hash

        payload = [self.describe(), self.lattice, reciprocal(self.source_model)]
        payload.append(np.asarray(getattr(self.source_model, "band_channels", np.zeros(self.nbnd, int))))
        if self._projector is not None:
            payload.append(self._projector() if callable(self._projector) else self._projector)
        for name in ("translations", "hopping_matrices", "position_matrices", "centres"):
            if hasattr(self.source_model, name):
                payload.append(getattr(self.source_model, name))
        for name in ("hamiltonian_blocks", "overlap_blocks"):
            if hasattr(self.source_model, name):
                blocks = getattr(self.source_model, name)
                payload.append({str(k): v.toarray() for k, v in sorted(blocks.items())})
        if hasattr(self.source_model, "position_blocks"):
            payload.append(
                [
                    {str(k): v.toarray() for k, v in sorted(axis.items())}
                    for axis in self.source_model.position_blocks
                ]
            )
        return stable_hash(*payload)
