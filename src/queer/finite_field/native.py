"""Physical reciprocal-plane views of Queer's native Hamiltonian models.

The view is internal to finite-field algorithms. Public model coordinates remain
fractional, and nonorthogonal geometry always uses the native covariant engine.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from copy import copy
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh

from ..capabilities import CapabilityError
from ..models.evaluation import band_geometry, basis_curvature, cell, degeneracy_tolerance, reciprocal
from ..physics.geometry import band_curvatures, interband_products, pair_metrics


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
            identity_origin = np.all(phase == 1)
            self.H_R = (
                model.hopping_matrices if identity_origin else model.hopping_matrices * phase[:, None, None]
            )
            self.H_R.setflags(write=False)
            self.centres = None if self.positions is None else np.asarray(self.positions) @ axes.T
            raw_position = getattr(model, "position_matrices", None)
            if raw_position is not None:
                self.r_R = (
                    raw_position
                    if identity_origin and np.array_equal(axes, np.eye(3))
                    else np.einsum("rija,ba->rijb", raw_position, axes) * phase[:, None, None, None]
                )
                self.r_R.setflags(write=False)
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

    @contextmanager
    def on_line(self, angle, t, *, mem_bytes=None):
        """Factorize native Wannier operators in this view's oriented plane."""
        on_line = getattr(self.source_model, "on_line", None)
        if on_line is None:
            with nullcontext():
                yield self
        else:
            axes = self.plane.axes(angle)
            origin = np.asarray(self.plane.origin) + float(t) * axes[1] @ np.linalg.inv(
                reciprocal(self.source_model)
            )
            kwargs = {}
            if mem_bytes is not None and hasattr(self.source_model, "_line_cache_limit"):
                kwargs["cache_bytes"] = mem_bytes / 3
            with on_line(axes[0], origin, **kwargs):
                yield self

    def light_worker_view(self):
        """Private plain clone referencing the same immutable operator arrays."""
        view = copy(self)
        source = self.source_model
        light = getattr(source, "light_worker_view", None)
        if light is not None:
            view.source_model = light()
            if getattr(self._projector, "__self__", None) is source:
                view._projector = getattr(view.source_model, self._projector.__name__)
        return view

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
        from ..models.evaluation import derivatives

        points, shape = self._points(kx, ky)
        values = [
            np.einsum("ba,aij->bij", self.plane._axes, derivatives(self.source_model, k)["H"]) for k in points
        ]
        a = np.asarray(values).reshape(shape + (2, self.nbnd, self.nbnd))
        return a[..., 0, :, :], a[..., 1, :, :]

    def hamiltonian_and_derivatives(self, kx, ky):
        x, y = self.dhamiltonian(kx, ky)
        return self.hamiltonian(kx, ky), x, y

    def gap(self, kx, ky, bands=None, *, mem_bytes=256e6):
        if bands is not None and tuple(bands) != self._pair:
            raise ValueError("the plane view fixes its band pair at construction")
        points, shape = self._points(kx, ky)
        v, c = self._pair
        if hasattr(self.source_model, "hopping_matrices"):
            from ..execution import ExecutionOptions

            options = getattr(self.source_model, "execution", ExecutionOptions()).resolved()
            per_point = 16 * self.nbnd**2 * 6
            cache_limit = getattr(self.source_model, "_line_cache_limit", None)
            working_bytes = mem_bytes if cache_limit is None else max(0, mem_bytes - cache_limit)
            batch = min(options.bounded_batch_size(per_point), max(1, int(working_bytes // per_point)))
            out = np.empty(len(points))
            for start in range(0, len(points), batch):
                chunk = points[start : start + batch]
                if hasattr(self.source_model, "_assemble_points"):
                    matrices, _ = self.source_model._assemble_points(chunk)
                else:
                    from ..physics.operators import assemble

                    matrices, _ = assemble(
                        self.source_model.translations, self.source_model.hopping_matrices, chunk
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

        if hasattr(self.source_model, "_fourier_values") and not self.supports_generalized:
            return self._wannier_geometry(kx, ky, angle, **kwargs)
        points, shape = self._points(kx, ky)
        axes = self.plane.axes(angle)
        v, c = self._pair
        rows = []
        for k in points:
            data = band_geometry(self.source_model, k, self.degeneracy_tol)
            e, connections = data["energies"], data["connections"]
            a = np.einsum("ba,anm->bnm", axes, connections)
            g, totals = pair_metrics(a, self._pair, data["band_channels"])
            totals = list(totals)
            valid = True
            for band in (v, c):
                same = data["band_channels"] == data["band_channels"][band]
                unresolved = (np.abs(e - e[band]) <= self.degeneracy_tol) & same
                unresolved[band] = False
                valid &= not np.any(unresolved)
            engine = kwargs.get("engine", "direct")
            projected = basis_curvature(self.source_model, k, data) if engine == "direct" else None
            curvature = band_curvatures(data, engine, projected)
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

    def _wannier_geometry(self, kx, ky, angle, **kwargs):
        """Bounded orthogonal covariant sums for the two requested bands.

        The retained Wannier connection contributes both the interband metric
        and its curl/commutator Berry terms. Projecting only the selected rows
        avoids forming complete band-space connections and commutators.
        """
        from .geometry.kubo import BandGeometry

        model = self.source_model
        if kwargs.get("bands") is not None and tuple(kwargs["bands"]) != self._pair:
            raise ValueError("the plane view fixes its band pair at construction")
        engine = kwargs.get("engine", "direct")
        if engine not in {"direct", "kubo"}:
            raise ValueError("curvature engine must be 'direct' or 'kubo'")
        points, shape = self._points(kx, ky)
        axes = self.plane.axes(angle)
        v, c = self._pair
        selected = np.array([v, c])
        n, count = self.nbnd, len(points)
        budget = kwargs.get("mem_bytes", 256e6)
        cache_limit = getattr(model, "_line_cache_limit", None)
        if cache_limit is not None:
            budget = max(0, budget - cache_limit)
        options = model.execution.resolved()
        batch = min(options.bounded_batch_size(16 * n**2 * 24), max(1, int(budget // (16 * n**2 * 24))))
        values = [np.empty(count) for _ in range(4)]
        metrics = [np.empty((count, 3)) for _ in range(3)]
        velocity = np.empty((count, 2), complex)
        degenerate = np.zeros(count, bool)
        gap = np.empty(count)
        all_energies = np.empty((count, n)) if kwargs.get("keep_eps", False) else None
        self_mask = np.zeros((2, n), bool)
        self_mask[0, v] = self_mask[1, c] = True
        moments = [tuple(axis) for axis in axes]
        for start in range(0, count, batch):
            sl = slice(start, min(start + batch, count))
            k = points[sl]
            H, _ = model._assemble_points(k)
            reference = np.trace(H, axis1=-2, axis2=-1).real / n
            H -= reference[:, None, None] * np.eye(n)
            energy, U = np.linalg.eigh(H)
            Usel = U[:, :, selected]
            UsH = Usel.conj().transpose(0, 2, 1)
            delta = energy[:, selected, None] - energy[:, None, :]
            gap[sl] = energy[:, c] - energy[:, v]
            absolute = energy + reference[:, None]
            values[0][sl], values[1][sl] = absolute[:, v], absolute[:, c]
            if all_energies is not None:
                all_energies[sl] = absolute
            projected_h = [
                (UsH @ model._fourier_values(k, model.hopping_matrices, "H", (moment,))) @ U
                for moment in moments
            ]
            if model.position_matrices is None:
                if model.centres is None:
                    connections = [np.zeros_like(projected_h[0]) for _ in range(2)]
                else:
                    positions = np.asarray(model.centres) @ axes.T
                    connections = [(UsH * positions[:, a]) @ U for a in range(2)]
            else:
                orbital_connections = [np.zeros((len(k), n, n), complex) for _ in axes]
                for a in range(3):
                    component, _ = model._assemble_points(k, model.position_matrices[..., a], ("r", a))
                    for axis, orbital in zip(axes, orbital_connections, strict=True):
                        orbital += axis[a] * component
                connections = [(UsH @ orbital) @ U for orbital in orbital_connections]
            covariant = [h + 1j * delta * a for h, a in zip(projected_h, connections, strict=True)]
            separated = (np.abs(delta) > self.degeneracy_tol) & ~self_mask
            inverse = np.zeros_like(delta)
            np.divide(1.0, delta**2, out=inverse, where=separated)
            omega, gss, gst, gtt = interband_products(covariant[0], covariant[1], inverse)
            if engine == "direct":
                omega += 2.0 * np.sum((connections[0] * connections[1].conj()).imag, axis=-1)
                if model.position_matrices is not None:
                    curl = np.zeros((len(k), n, n), complex)
                    coefficients = np.outer(axes[0], axes[1]) - np.outer(axes[1], axes[0])
                    for a, b in zip(*np.nonzero(coefficients), strict=True):
                        curl += coefficients[a, b] * model._fourier_values(
                            k, model.position_matrices[..., b], ("r", int(b)), (int(a),)
                        )
                    omega += np.einsum("kbi,kib->kb", UsH @ curl, Usel).real
            values[2][sl], values[3][sl] = omega[:, 0], omega[:, 1]
            metrics[0][sl] = np.stack([gss[:, 0, c], gst[:, 0, c], gtt[:, 0, c]], axis=-1)
            totals = np.stack([gss.sum(-1), gst.sum(-1), gtt.sum(-1)], axis=-1)
            metrics[1][sl], metrics[2][sl] = totals[:, 0], totals[:, 1]
            velocity[sl] = np.stack([covariant[0][:, 1, v], covariant[1][:, 1, v]], axis=-1)
            degenerate[sl] = np.any((np.abs(delta) <= self.degeneracy_tol) & ~self_mask, axis=(1, 2))
        result = BandGeometry(
            *values,
            *metrics,
            velocity,
            degenerate,
            all_energies,
            {"gap": gap, "native_covariant": True},
        ).reshape(shape)
        if np.any(result.deg_flag):
            for name in ("omega_v", "omega_c", "g_cv", "g_v_total", "g_c_total", "v_cv"):
                getattr(result, name)[result.deg_flag] = np.nan
        return result

    def position_connection(self, kx, ky):
        from ..models.evaluation import derivatives, position_matrix

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
        from ..models.evaluation import basis_overlap, states

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
            try:
                payload.append(self._projector() if callable(self._projector) else self._projector)
            except (CapabilityError, ValueError) as error:
                if getattr(self._projector, "__self__", None) is not self.source_model:
                    raise
                # A centre-derived projector can be unavailable even when the
                # Hamiltonian is valid for a gap-minimum calculation.
                payload.append({"unavailable_source_projector": str(error)})
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
