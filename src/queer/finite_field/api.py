"""Native finite-field Hall calculations with explicit physical provenance.

These are quasistatic, finite-window, two-band response calculations. They are
separate from the perturbative finite-frequency optical tensors.
"""

from __future__ import annotations

import math

import numpy as np

from ..capabilities import CapabilityError
from ..execution import ExecutionOptions
from ..observables.result import ObservableResult
from ._storage import json_safe, source_hash, stable_hash
from .config import resolve_options
from .geometry.provider import ModelGeometry
from .harmonics import harmonics as project_harmonics
from .lz.crossings import Crossing, LineCrossings, find_crossings
from .transport.hall import HallCalculator
from .transport.lines import LineSolution, Segment


def _model_view(model, plane, band_pair, occupied_bands, layer_projector):
    from .native import FieldPlane, NativePlaneModel

    if isinstance(model, NativePlaneModel):
        resolved_plane = model.plane if plane is None else FieldPlane.from_mapping(plane)
        resolved_pair = (
            model.band_pair()
            if band_pair is None and occupied_bands is None
            else None
            if band_pair is None
            else tuple(band_pair)
        )
        if resolved_plane == model.plane and resolved_pair == model.band_pair() and layer_projector is None:
            return model
        return NativePlaneModel(
            model.source_model,
            plane=resolved_plane,
            band_pair=resolved_pair,
            occupied_bands=occupied_bands,
            layer_projector=model._projector if layer_projector is None else layer_projector,
            degeneracy_tol=model.degeneracy_tol,
        )
    if hasattr(model, "n_orb") and hasattr(model, "band_pair"):
        if plane is not None and not getattr(model, "native_band_geometry", False):
            raise ValueError("Analytic reference models use their own Cartesian xy frame")
        if band_pair is not None and tuple(band_pair) != tuple(model.band_pair()):
            raise ValueError("The selected band pair differs from the prepared model")
        return model
    if plane is None:
        raise ValueError("Native models require an explicit oriented plane")
    return NativePlaneModel(
        model,
        plane=plane,
        band_pair=band_pair,
        occupied_bands=occupied_bands,
        layer_projector=layer_projector,
    )


def _fields(values, name="fields", *, positive=False):
    result = np.atleast_1d(np.asarray(values, dtype=float))
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite scalar or one-dimensional sequence")
    if positive and np.any(result <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return result


def _require_crossings(line, cfg):
    for crossing in line.crossings:
        metric = crossing.g_ss[cfg.conventions.metric]
        if not crossing.touching and (not np.isfinite(metric) or metric <= 0):
            raise ValueError(
                "Requested quantum metric is undefined or non-positive at a non-touching crossing; "
                "refine the geometry grid or supply the missing physical operator"
            )
    if cfg.conventions.ec_method == "ddp":
        for crossing in line.crossings:
            if not crossing.touching and crossing.ddp_source not in {"exact", "model", "fit", "eq8"}:
                raise CapabilityError(
                    "DDP response requires a resolved exact, model, fitted, or flagged Eq8 exponent"
                )


class PreparedHallResponse:
    """Separate model-dependent preparation from inexpensive signed-field scans.

    Checkpoint callbacks run in the parent process. ``load_line(index, offset)``
    returns a LineSolution or None; ``save_line(index, offset, solution)`` commits
    one complete line. Use encode_line/decode_line for pickle-free storage.
    """

    def __init__(
        self,
        model=None,
        options=None,
        *,
        plane=None,
        band_pair=None,
        occupied_bands=None,
        layer_projector=None,
        execution=None,
        provider=None,
        keep_nodes=False,
        cache_dir=None,
        acceleration=None,
    ):
        self.options = resolve_options(options)
        self.requested_options = self.options
        self.cache_dir = cache_dir
        self.execution = (execution or getattr(model, "execution", None) or ExecutionOptions()).resolved()
        if (
            provider is None
            and model is not None
            and not hasattr(model, "hamiltonian")
            and hasattr(model, "geometry")
        ):
            provider, model = model, None
        self.model = (
            None if model is None else _model_view(model, plane, band_pair, occupied_bands, layer_projector)
        )
        if acceleration not in (None, False, True):
            raise ValueError("acceleration must be None, False or True")
        if acceleration:
            from .transport.kernels import HAVE_NUMBA

            if not HAVE_NUMBA:
                raise CapabilityError("Compiled response acceleration requires queer[acceleration] (numba)")
            if any((self.options.phase.geometric, self.options.phase.stokes, self.options.phase.twisted_lz)):
                raise CapabilityError(
                    "Compiled response acceleration supports profiles without extra phases; "
                    "the corrected physical kernel uses NumPy"
                )
        self.calculator = HallCalculator(
            self.model, self.options, provider=provider, keep_nodes=keep_nodes, use_numba=bool(acceleration)
        )
        self.options = self.calculator.cfg

    @property
    def layout(self):
        return self.calculator.layout

    @property
    def lines(self):
        return self.calculator.lines

    @property
    def grid(self):
        return self.calculator.grid

    @property
    def provider(self):
        return self.calculator.provider

    def signature(self):
        """Content identity for the reusable geometry, layout and Laplace tables."""
        options = self.options.to_dict()
        options.pop("fourier", None)
        options.pop("label", None)
        source = self.model if self.model is not None else self.provider
        if hasattr(source, "fingerprint"):
            identity = source.fingerprint()
        elif hasattr(source, "describe"):
            identity = source.describe()
        else:
            # Analytic reference models contain only scalar model parameters.
            values = vars(source)
            if any(
                not isinstance(value, (int, float, complex, str, bool, np.ndarray, type(None)))
                for value in values.values()
            ):
                raise ValueError("A cached model must provide a content-based fingerprint")
            identity = {"class": f"{type(source).__module__}.{type(source).__name__}", "parameters": values}
        return stable_hash(
            identity, options, source_hash(), self.calculator.keep_nodes, self.calculator.spin_degeneracy
        )

    def prepare(self, *, load_line=None, save_line=None, save_layout=None, load_layout=None):
        if self.calculator._groups is not None:
            return self
        if self.cache_dir is None:
            self.calculator.prepare(
                execution=self.execution,
                load_line=load_line,
                save_line=save_line,
                save_layout=save_layout,
                load_layout=load_layout,
            )
            return self
        from .cache import PreparationCache

        with PreparationCache(self.cache_dir, self.signature()) as cache:

            def restore(index, offset):
                line = load_line(index, offset) if load_line is not None else None
                if line is not None:
                    cache.put(index, offset, line)
                    return line
                line = cache.get(index, offset)
                if line is not None and save_line is not None:
                    save_line(index, offset, line)
                return line

            def commit(index, offset, line):
                if save_line is not None:
                    save_line(index, offset, line)
                cache.put(index, offset, line)

            def layout_commit(layout):
                if save_layout is not None:
                    save_layout(layout)
                cache.put_layout(layout)

            def layout_restore():
                layout = load_layout() if load_layout is not None else None
                return cache.get_layout() if layout is None else layout

            self.calculator.prepare(
                execution=self.execution,
                load_line=restore,
                save_line=commit,
                load_layout=layout_restore,
                save_layout=layout_commit,
            )
        return self

    def J(self, fields):
        self.prepare()
        from threadpoolctl import threadpool_limits

        fields = _fields(fields)
        count = sum(group.w.size * group.n for group in self.calculator._groups.values())
        chunk = self.execution.bounded_batch_size(max(1024, 8 * count * 30))
        with threadpool_limits(limits=self.execution.native_threads):
            if self.calculator.use_numba:
                import numba

                previous = numba.get_num_threads()
                numba.set_num_threads(self.execution.native_threads)
                try:
                    return self.calculator.J(fields, chunk=chunk)
                finally:
                    numba.set_num_threads(previous)
            return self.calculator.J(fields, chunk=chunk)

    def describe(self):
        return json_safe(
            {
                "observable": "finite_field_hall",
                "options": self.options.to_dict(),
                "requested_options": self.requested_options.to_dict(),
                "execution": self.execution.as_dict(),
                "source_sha256": source_hash(),
                "drive_model": "quasistatic J(E0 sin theta)",
                "dimension": 2,
                "acceleration": "numba" if self.calculator.use_numba else "numpy",
                "two_band_approximation": True,
                **self.calculator.diagnostics(),
            }
        )

    def hall(self, fields):
        fields = _fields(fields)
        self.prepare()
        current = self.J(fields)
        return ObservableResult(
            {"fields": fields, "current": current},
            {"fields": ("field",), "current": ("field",)},
            {"fields": "V/m", "current": "A/m"},
            self.describe(),
        )

    def harmonics(self, amplitudes, n_max=None):
        amplitudes = _fields(amplitudes, "amplitudes")
        if np.any(amplitudes < 0):
            raise ValueError("drive amplitudes must be nonnegative")
        if np.max(amplitudes) > self.options.drive.E_max:
            raise ValueError("drive amplitudes exceed the prepared E_max")
        cfg = self.options.fourier
        n_max = cfg.n_max if n_max is None else n_max
        if isinstance(n_max, bool) or int(n_max) != n_max or n_max < 0:
            raise ValueError("n_max must be a nonnegative integer")
        self.prepare()
        result = project_harmonics(
            self.calculator.J,
            amplitudes,
            n_max=int(n_max),
            n_theta=cfg.n_theta,
            adaptive=cfg.adaptive,
            tol=cfg.tol,
            n_cap=cfg.n_cap,
        )
        arrays = {
            "amplitudes": amplitudes,
            "orders": result.N,
            "b": result.b,
            "a": result.a,
            "c": result.c,
            "c0": result.c0,
            "converged": result.converged_rows,
            "relative_error": result.relative_error,
        }
        dims = {
            "amplitudes": ("amplitude",),
            "orders": ("harmonic",),
            "c0": ("amplitude",),
            "converged": ("amplitude",),
            "relative_error": ("amplitude",),
            **{key: ("amplitude", "harmonic") for key in ("a", "b", "c")},
        }
        units = {
            "amplitudes": "V/m",
            "orders": "1",
            "converged": "1",
            "relative_error": "1",
            **{key: "A/m" for key in ("a", "b", "c", "c0")},
        }
        meta = {
            **self.describe(),
            "observable": "hall_harmonics",
            "converged": result.converged,
            "fourier": json_safe(result.info),
            "fourier_definition": "c=(a-i*b)/2; a0=2*c0",
        }
        return ObservableResult(arrays, dims, units, meta)


def landau_zener_hall(
    model,
    fields,
    *,
    options=None,
    plane=None,
    band_pair=None,
    occupied_bands=None,
    layer_projector=None,
    execution=None,
    cache_dir=None,
    acceleration=None,
):
    """Quasistatic sheet Hall current in A/m at signed fields in V/m."""
    if isinstance(model, PreparedHallResponse):
        if any(
            value is not None
            for value in (
                options,
                plane,
                band_pair,
                occupied_bands,
                layer_projector,
                execution,
                cache_dir,
                acceleration,
            )
        ):
            raise ValueError(
                "A prepared response owns its geometry and settings; create a new preparation to change them"
            )
        return model.hall(fields)
    return PreparedHallResponse(
        model,
        options,
        plane=plane,
        band_pair=band_pair,
        occupied_bands=occupied_bands,
        layer_projector=layer_projector,
        execution=execution,
        cache_dir=cache_dir,
        acceleration=acceleration,
    ).hall(fields)


def hall_harmonics(
    model,
    amplitudes,
    *,
    n_max=None,
    options=None,
    plane=None,
    band_pair=None,
    occupied_bands=None,
    layer_projector=None,
    execution=None,
    cache_dir=None,
    acceleration=None,
):
    """Signed sine/cosine and complex Fourier coefficients, including the DC component."""
    if isinstance(model, PreparedHallResponse):
        if any(
            value is not None
            for value in (
                options,
                plane,
                band_pair,
                occupied_bands,
                layer_projector,
                execution,
                cache_dir,
                acceleration,
            )
        ):
            raise ValueError(
                "A prepared response owns its geometry and settings; create a new preparation to change them"
            )
        return model.harmonics(amplitudes, n_max)
    return PreparedHallResponse(
        model,
        options,
        plane=plane,
        band_pair=band_pair,
        occupied_bands=occupied_bands,
        layer_projector=layer_projector,
        execution=execution,
        cache_dir=cache_dir,
        acceleration=acceleration,
    ).harmonics(amplitudes, n_max)


def crossing_scan(
    model,
    offsets=(0.0,),
    *,
    options=None,
    plane=None,
    band_pair=None,
    occupied_bands=None,
    layer_projector=None,
    execution=None,
):
    """Ragged avoided-crossing diagnostics along explicitly selected field lines."""
    prepared = PreparedHallResponse(
        model,
        options,
        plane=plane,
        band_pair=band_pair,
        occupied_bands=occupied_bands,
        layer_projector=layer_projector,
        execution=execution,
    )
    cfg = prepared.options
    offsets = _fields(offsets, "offsets")
    lines = [
        find_crossings(
            prepared.provider,
            float(t),
            cfg.crossings,
            want_ddp=cfg.conventions.ec_method == "ddp",
            e_cut=cfg.quadrature.p_cut * cfg.drive.E_max,
        )
        for t in offsets
    ]
    for line in lines:
        _require_crossings(line, cfg)
    from scipy.integrate import quad

    records = [(i, j, c) for i, line in enumerate(lines) for j, c in enumerate(line.crossings)]
    segments = []
    for i, line in enumerate(lines):
        positions = sorted(c.s for c in line.crossings)
        for start, end in zip(positions[:-1], positions[1:]):
            gap_integral = quad(
                lambda s: float(prepared.provider.gap(s, line.t)),
                start,
                end,
                points=[0.0] if start < 0 < end else None,
                epsabs=1e-14,
                epsrel=1e-9,
            )[0]
            segments.append((i, start, end, gap_integral))
    eq8 = np.asarray(
        [min((c.Ec[cfg.conventions.metric] for c in line.crossings), default=np.nan) for line in lines]
    )
    ddp = np.asarray(
        [min((c.Ec_ddp for c in line.crossings if c.Ec_ddp is not None), default=np.nan) for line in lines]
    )
    arrays = {
        "offsets": offsets,
        "line_index": np.asarray([i for i, _, _ in records], dtype=int),
        "crossing_index": np.asarray([j for _, j, _ in records], dtype=int),
        "position": np.asarray([c.s for _, _, c in records]),
        "gap": np.asarray([c.eps_g for _, _, c in records]),
        "metric": np.asarray([c.g_ss[cfg.conventions.metric] for _, _, c in records]),
        "critical_field": np.asarray([c.Ec[cfg.conventions.metric] for _, _, c in records]),
        "ddp_field": np.asarray([c.Ec_ddp if c.Ec_ddp is not None else np.nan for _, _, c in records]),
        "branch_point": np.asarray(
            [
                c.branch_point if c.branch_point is not None else complex(np.nan, np.nan)
                for _, _, c in records
            ],
            dtype=complex,
        ),
        "touching": np.asarray([c.touching for _, _, c in records], dtype=bool),
        "ddp_valid": np.asarray([c.Ec_ddp is not None for _, _, c in records], dtype=bool),
        "k0": np.asarray(
            [0.5 * np.ptp([c.s for c in line.crossings]) if line.n else np.nan for line in lines]
        ),
        "line_valid": np.asarray([bool(line.n) for line in lines]),
        "Ec_eq8": eq8,
        "Ec_ddp": ddp,
        "selected_Ec": ddp if cfg.conventions.ec_method == "ddp" else eq8,
        "segment_line_index": np.asarray([i for i, _, _, _ in segments], dtype=int),
        "segment_start": np.asarray([start for _, start, _, _ in segments]),
        "segment_end": np.asarray([end for _, _, end, _ in segments]),
        "gap_integrals": np.asarray([gap for _, _, _, gap in segments]),
        "Lambda": np.asarray([gap * 1e10 / (2 * np.pi) for _, _, _, gap in segments]),
    }
    line_arrays = {"offsets", "k0", "line_valid", "Ec_eq8", "Ec_ddp", "selected_Ec"}
    segment_arrays = {"segment_line_index", "segment_start", "segment_end", "gap_integrals", "Lambda"}
    dims = {
        key: ("line",) if key in line_arrays else ("segment",) if key in segment_arrays else ("crossing",)
        for key in arrays
    }
    units = {
        "offsets": "Angstrom^-1",
        "position": "Angstrom^-1",
        "gap": "eV",
        "metric": "Angstrom^2",
        "critical_field": "V/m",
        "ddp_field": "V/m",
        "branch_point": "Angstrom^-1",
        "k0": "Angstrom^-1",
        "Ec_eq8": "V/m",
        "Ec_ddp": "V/m",
        "selected_Ec": "V/m",
        "segment_start": "Angstrom^-1",
        "segment_end": "Angstrom^-1",
        "gap_integrals": "eV/Angstrom",
        "Lambda": "V/m",
    }
    metadata = {
        **prepared.describe(),
        "observable": "crossing_scan",
        "line_flags": [sorted(line.flags) for line in lines],
        "crossing_definition": [line.definition for line in lines],
        "ddp_source": [c.ddp_source for _, _, c in records],
    }
    return ObservableResult(arrays, dims, units, metadata)


def critical_hall(
    model,
    fields,
    *,
    options=None,
    plane=None,
    band_pair=None,
    occupied_bands=None,
    layer_projector=None,
    execution=None,
):
    """Nominal and log-corrected critical asymptotes with convergence diagnostics."""
    from .transport.critical import critical_constants, critical_kc, j_critical_analytic, j_critical_log

    cfg = resolve_options(options)
    model = _model_view(model, plane, band_pair, occupied_bands, layer_projector)
    provider = ModelGeometry(model, cfg.drive.angle, k_patch=cfg.quadrature.k_patch)
    if provider.gap_gamma > cfg.crossings.touching_floor:
        raise ValueError("critical_hall requires a model with a verified gap closing at the plane origin")
    constants = critical_constants(provider)
    if cfg.conventions.berry != "c":
        radial = np.geomspace(1e-7, 1e-6, 5)
        strengths = np.asarray(
            [
                radial
                * provider.at_cartesian(radial * np.cos(angle), radial * np.sin(angle)).omega(
                    cfg.conventions.berry
                )
                for angle in (0.0, 0.7)
            ]
        )
        constants["conduction_C_s"] = constants["C_s"]
        constants["C_s"] = float(strengths[0, 0])
        constants["C_spread"] = float(np.ptp(strengths) / max(abs(strengths[0, 0]), 1e-300))
    constants["berry_mode"] = cfg.conventions.berry
    if max(constants["C_spread"], constants["hbar_v_spread"]) > 1e-3:
        raise CapabilityError(
            "The isotropic critical Dirac asymptote is not certified by the radial/directional samples"
        )
    if constants["hbar_v"] <= 0 or abs(constants["C_s"]) < 1e-12:
        raise ValueError("The critical velocity/curvature asymptote is not resolved")
    if cfg.conventions.berry == "c":
        cutoff = critical_kc(provider, constants["C_s"], window=cfg.crossings.window)
    else:
        from .transport.quadrature import logistic_nodes

        s, weight = logistic_nodes(0.0, cfg.crossings.window, 0.1, d_a=1e-11, d_b=1e-6 * cfg.crossings.window)
        integral = np.sum(weight * provider.geometry(s, np.full_like(s, 1e-8)).omega(cfg.conventions.berry))
        cutoff = 0.5e-8 * math.exp(integral / constants["C_s"])
    fields = _fields(fields)
    nominal = j_critical_analytic(fields, constants["C_s"], constants["hbar_v"], cfg.conventions.prefactor)
    corrected = np.zeros_like(fields)
    active = fields != 0
    corrected[active] = j_critical_log(
        fields[active], constants["C_s"], constants["hbar_v"], cutoff, cfg.conventions.prefactor
    )
    arrays = {
        "fields": fields,
        "current": cfg.conventions.figure_scale * provider.spin_degeneracy * nominal,
        "log_corrected_current": cfg.conventions.figure_scale * provider.spin_degeneracy * corrected,
    }
    return ObservableResult(
        arrays,
        {key: ("field",) for key in arrays},
        {key: "V/m" if key == "fields" else "A/m" for key in arrays},
        json_safe(
            {
                "observable": "critical_hall",
                "spin_degeneracy": provider.spin_degeneracy,
                "options": cfg.to_dict(),
                "constants": constants,
                "cutoff_angstrom_inverse": cutoff,
                "provider": provider.describe(),
                "source_sha256": source_hash(),
                "relaxation_assumption": "infinite T1 for logarithmic asymptote",
            }
        ),
    )


def fit_hall(fields, values, *, window, f_bounds=None, oscillations=True):
    """Fit a specified positive-field window; fit windows are recorded scientific inputs."""
    from .analysis.fits import fit_harmonic, fit_smooth

    fields = _fields(fields, positive=True)
    values = _fields(values, "values")
    if fields.shape != values.shape or np.any(np.diff(fields) <= 0):
        raise ValueError("fields and values must have matching shapes and fields must increase strictly")
    bounds = _fields(window, "window", positive=True)
    if len(bounds) != 2 or bounds[1] <= bounds[0]:
        raise ValueError("window must contain increasing positive lower and upper bounds")
    mask = (fields >= bounds[0]) & (fields <= bounds[1])
    if mask.sum() < (6 if oscillations else 3) or not np.any(values[mask]):
        raise ValueError("The fit window has too few points or a vanishing response")
    if oscillations:
        fitted = fit_harmonic(fields, values, tuple(bounds), f_bounds=f_bounds)
    else:
        from dataclasses import asdict

        fitted = asdict(fit_smooth(fields, values, tuple(bounds)))
    return ObservableResult(
        {"fields": fields, "values": values, "fit_mask": mask},
        {key: ("field",) for key in ("fields", "values", "fit_mask")},
        {"fields": "V/m", "values": "A/m"},
        json_safe(
            {
                "observable": "fit_hall",
                "fit": fitted,
                "fit_model": "activated onset plus inverse-field oscillation"
                if oscillations
                else "activated onset",
                "source_sha256": source_hash(),
            }
        ),
    )


def houston_line(
    model,
    offset,
    fields,
    window=(-0.03, 0.03),
    *,
    options=None,
    plane=None,
    band_pair=None,
    occupied_bands=None,
    layer_projector=None,
    execution=None,
    dphi=0.02,
    basis="superadiabatic",
):
    """All-band per-line TDSE validator for certified fixed orthonormal representations."""
    from threadpoolctl import threadpool_limits

    from .tdse.houston import houston_line as propagate

    cfg = resolve_options(options)
    view = _model_view(model, plane, band_pair, occupied_bands, layer_projector)
    if not getattr(view, "supports_tdse", True):
        raise CapabilityError(
            "Houston TDSE needs a certified fixed orthonormal representation; "
            "moving/nonorthogonal basis propagation is not available"
        )
    fields = _fields(fields, positive=False)
    if np.any(fields == 0):
        raise ValueError("TDSE fields must be nonzero")
    if not math.isfinite(offset):
        raise ValueError("offset must be finite in Angstrom^-1")
    bounds = _fields(window, "window")
    if len(bounds) != 2 or bounds[1] <= bounds[0]:
        raise ValueError("window must contain two increasing bounds")
    # The internal validator follows signed force; the native electron profile
    # reverses traversal relative to the printed positive-charge convention.
    runtime = (execution or getattr(model, "execution", None) or ExecutionOptions()).resolved()
    chunk_bytes = None if runtime.memory_mb is None else runtime.memory_mb * 1024**2 * 0.8 / runtime.workers
    with threadpool_limits(limits=runtime.native_threads):
        result = propagate(
            view,
            float(offset),
            cfg.conventions.charge_sign * fields,
            *bounds,
            angle=cfg.drive.angle,
            dphi=dphi,
            basis=basis,
            chunk_bytes=chunk_bytes,
        )
    v, c = view.band_pair()
    leakage = np.clip(1 - result.T[:, v, v] - result.T[:, c, v], 0, 1)
    arrays = {
        "fields": fields,
        "transitions": result.T,
        "population": result.P,
        "excitation": result.n_exc,
        "remote_leakage": leakage,
    }
    dims = {
        "fields": ("field",),
        "transitions": ("field", "final_band", "initial_band"),
        **{key: ("field",) for key in ("population", "excitation", "remote_leakage")},
    }
    return ObservableResult(
        arrays,
        dims,
        {key: "V/m" if key == "fields" else "1" for key in arrays},
        json_safe(
            {
                "observable": "houston_line",
                "options": cfg.to_dict(),
                "band_pair": [v, c],
                "dynamics": result.info,
                "execution": runtime.as_dict(),
                "source_sha256": source_hash(),
                "relaxation": "unitary, all bands",
            }
        ),
    )


def lindblad_line(
    model,
    offset,
    fields,
    window=(-0.03, 0.03),
    *,
    options=None,
    plane=None,
    band_pair=None,
    occupied_bands=None,
    layer_projector=None,
    execution=None,
    dphi=0.02,
    basis="superadiabatic",
):
    """Two-level time-domain validator with completely positive T1/T2 relaxation."""
    from threadpoolctl import threadpool_limits

    from .tdse.houston import lindblad_two_level

    cfg = resolve_options(options)
    view = _model_view(model, plane, band_pair, occupied_bands, layer_projector)
    if view.n_orb != 2 or not getattr(view, "supports_tdse", True):
        raise CapabilityError("Lindblad validation requires a certified fixed orthonormal two-band model")
    fields = _fields(fields)
    if np.any(fields == 0):
        raise ValueError("Lindblad fields must be nonzero")
    if not math.isfinite(offset):
        raise ValueError("offset must be finite in Angstrom^-1")
    bounds = _fields(window, "window")
    if len(bounds) != 2 or bounds[1] <= bounds[0]:
        raise ValueError("window must contain two increasing bounds")
    runtime = (execution or getattr(model, "execution", None) or ExecutionOptions()).resolved()
    chunk_bytes = None if runtime.memory_mb is None else runtime.memory_mb * 1024**2 * 0.8 / runtime.workers
    relaxation = cfg.relaxation
    with threadpool_limits(limits=runtime.native_threads):
        population = lindblad_two_level(
            view,
            float(offset),
            cfg.conventions.charge_sign * fields,
            *bounds,
            relaxation.T1,
            relaxation.T2,
            angle=cfg.drive.angle,
            dphi=dphi,
            coherence=relaxation.coherence,
            basis=basis,
            chunk_bytes=chunk_bytes,
        )
    valid = np.isfinite(population) & (population >= -1e-10) & (population <= 1 + 1e-10)
    return ObservableResult(
        {"fields": fields, "population": population, "valid": valid},
        {key: ("field",) for key in ("fields", "population", "valid")},
        {"fields": "V/m", "population": "1", "valid": "1"},
        json_safe(
            {
                "observable": "lindblad_line",
                "options": cfg.to_dict(),
                "band_pair": list(view.band_pair()),
                "basis": basis,
                "dphi": dphi,
                "window": list(bounds),
                "offset": offset,
                "execution": runtime.as_dict(),
                "source_sha256": source_hash(),
                "relaxation": "instantaneous-band amplitude damping and dephasing",
                "coherence_time": "pure dephasing" if relaxation.coherence == "paper" else "total coherence",
            }
        ),
    )


def berry_flux(
    model,
    *,
    plane=None,
    band_pair=None,
    occupied_bands=None,
    window=0.03,
    n=41,
    graded_scale=1e-5,
    tolerance=1e-3,
):
    """Subspace Berry flux through a finite reciprocal-plane window, with refinement.

    The result is a window flux in units of 2pi, not a quantized whole-BZ Chern
    certificate. Native links include overlap and position operators.
    """
    from .geometry.topology import fhs_flux, graded_axis

    view = _model_view(model, plane, band_pair, occupied_bands, None)
    if occupied_bands is None:
        bands = [view.band_pair()[0]]
    else:
        if (
            isinstance(occupied_bands, (bool, np.bool_))
            or not isinstance(occupied_bands, (int, np.integer))
            or not 0 < occupied_bands <= view.n_orb
        ):
            raise ValueError("occupied_bands must be a nonempty valid integer band count")
        bands = list(range(occupied_bands))
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Flux refinement tolerance must be finite and positive")
    halfwidth = _fields(window, "window", positive=True)
    if len(halfwidth) == 1:
        halfwidth = np.repeat(halfwidth, 2)
    if len(halfwidth) != 2:
        raise ValueError("Flux window must be one or two positive half-widths in Angstrom^-1")
    axes = [graded_axis(width, n, graded_scale) for width in halfwidth]
    refined = [graded_axis(width, 2 * n - 1, graded_scale) for width in halfwidth]
    flux = np.asarray([fhs_flux(view, bands, *axes), fhs_flux(view, bands, *refined)])
    difference = np.asarray(abs(flux[1] - flux[0]))
    return ObservableResult(
        {
            "flux": flux,
            "grid_size": np.asarray([n, 2 * n - 1]),
            "refinement_difference": difference,
            "converged": np.asarray(difference <= tolerance),
        },
        {"flux": ("refinement",), "grid_size": ("refinement",), "refinement_difference": (), "converged": ()},
        {"flux": "2pi", "grid_size": "1", "refinement_difference": "2pi", "converged": "1"},
        json_safe(
            {
                "observable": "berry_flux",
                "domain": "finite_window",
                "quantized_chern_certificate": False,
                "bands": bands,
                "window_halfwidth": halfwidth,
                "graded_scale": graded_scale,
                "tolerance": tolerance,
                "plane": view.plane.describe() if hasattr(view, "plane") else "reference xy",
                "links": "native physical basis"
                if getattr(view, "native_band_geometry", False)
                else "fixed orthonormal basis",
                "source_sha256": source_hash(),
            }
        ),
    )


def encode_line(line):
    """Encode a complete prepared line without Python objects or pickle."""
    metrics = ("cv", "v_total", "c_total")
    crosses = line.crossings.crossings
    arrays = {
        "s_cross": line.s_cross,
        "gap_integrals": line.gap_integrals,
        "crossing_gap": np.asarray([c.eps_g for c in crosses]),
        "crossing_metric": np.asarray([[c.g_ss[m] for m in metrics] for c in crosses]).reshape(-1, 3),
        "crossing_Ec": np.asarray([[c.Ec[m] for m in metrics] for c in crosses]).reshape(-1, 3),
        "crossing_ddp": np.asarray([c.Ec_ddp if c.Ec_ddp is not None else np.nan for c in crosses]),
        "crossing_branch": np.asarray(
            [c.branch_point if c.branch_point is not None else complex(np.nan, np.nan) for c in crosses],
            dtype=complex,
        ),
        "crossing_touching": np.asarray([c.touching for c in crosses], dtype=bool),
    }
    for name in ("shift", "g_cross", "twist"):
        value = getattr(line, name)
        if value is not None:
            arrays[name] = value
    for direction, value in line.geo_phase.items():
        arrays[f"phase_{direction}"] = value
    for name in ("tables", "F_inf", "M1"):
        for (band, direction), value in getattr(line, name).items():
            arrays[f"{name}_{band}_{direction}"] = value
    segment_bounds = {}
    for direction, segments in line.segments.items():
        segment_bounds[str(direction)] = [[segment.start, segment.end] for segment in segments]
        for index, segment in enumerate(segments):
            for key in ("r", "w", "omega_c", "omega_v"):
                arrays[f"segment_{direction}_{index}_{key}"] = getattr(segment, key)
    metadata = {
        "offset": line.t,
        "flags": sorted(line.flags),
        "crossing_flags": sorted(line.crossings.flags),
        "definition": line.crossings.definition,
        "crossing_kind": [c.kind for c in crosses],
        "ddp_source": [c.ddp_source for c in crosses],
        "segment_bounds": segment_bounds,
        "phase_refinement": line.phase_refinement,
        "schema": "queer.finite_field.line.v1",
    }
    return arrays, json_safe(metadata)


def decode_line(arrays, metadata):
    """Restore encode_line arrays; callers verify artifact identity/checksum."""
    if metadata.get("schema") != "queer.finite_field.line.v1":
        raise ValueError("Unsupported prepared-line checkpoint schema")
    metrics = ("cv", "v_total", "c_total")
    crosses = []
    for i, s in enumerate(arrays["s_cross"]):
        ddp, branch = arrays["crossing_ddp"][i], arrays["crossing_branch"][i]
        crosses.append(
            Crossing(
                float(s),
                float(arrays["crossing_gap"][i]),
                dict(zip(metrics, arrays["crossing_metric"][i])),
                dict(zip(metrics, arrays["crossing_Ec"][i])),
                bool(arrays["crossing_touching"][i]),
                metadata["crossing_kind"][i],
                None if np.isnan(ddp) else float(ddp),
                None if np.isnan(branch) else complex(branch),
                metadata["ddp_source"][i],
            )
        )
    lc = LineCrossings(metadata["offset"], crosses, set(metadata["crossing_flags"]), metadata["definition"])
    segments = {}
    for direction, bounds in metadata["segment_bounds"].items():
        d = int(direction)
        segments[d] = [
            Segment(
                start, end, *(arrays[f"segment_{d}_{i}_{key}"] for key in ("r", "w", "omega_c", "omega_v"))
            )
            for i, (start, end) in enumerate(bounds)
        ]
    packed = {}
    for name in ("tables", "F_inf", "M1"):
        packed[name] = {
            (band, d): arrays[f"{name}_{band}_{d}"]
            for band in ("c", "v")
            for d in (1, -1)
            if f"{name}_{band}_{d}" in arrays
        }
    phase = {d: arrays[f"phase_{d}"] for d in (1, -1) if f"phase_{d}" in arrays}
    return LineSolution(
        metadata["offset"],
        lc,
        arrays["s_cross"],
        arrays["gap_integrals"],
        segments,
        packed["tables"],
        packed["F_inf"],
        packed["M1"],
        set(metadata["flags"]),
        phase,
        arrays.get("shift"),
        arrays.get("g_cross"),
        arrays.get("twist"),
        metadata.get("phase_refinement", {}),
    )
