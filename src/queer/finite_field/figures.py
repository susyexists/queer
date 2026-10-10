"""Numerical paper figure recipes; all persistence belongs to native workflows."""

from __future__ import annotations

import numpy as np

from ..observables.result import ObservableResult
from ._storage import json_safe, source_hash
from .api import PreparedHallResponse, crossing_scan
from .config import resolve_options
from .geometry.provider import ModelGeometry
from .models.fourband import FourBandModel, FourBandParams


def _result(figure, arrays, dims, units, **metadata):
    return ObservableResult(
        arrays,
        dims,
        units,
        json_safe(
            {
                "observable": "paper_figure",
                "figure": figure,
                "source_sha256": source_hash(),
                "recipe_version": 1,
                "interpretation": "Explicit paper reproduction conventions",
                **metadata,
            }
        ),
    )


def _model(x, model_parameters=None):
    return FourBandModel(FourBandParams(**(model_parameters or {})).at(float(x)))


def _settings(figure, options=None):
    return resolve_options({"preset": figure, **(options or {})})


def _line(x, figure, *, model_parameters=None, options=None):
    cfg = _settings(figure, options)
    result = crossing_scan(_model(x, model_parameters), [0.0], options=cfg)
    if not result["line_valid"][0]:
        raise ValueError("The figure recipe requires a resolved crossing on the central line")
    return result, cfg


def fig4b(*, x=0.9, fields=None, model_parameters=None, options=None):
    """Single-passage probability exp(-Ec/E), using the paper's Eq. 8 profile."""
    scan, cfg = _line(x, "fig4b", model_parameters=model_parameters, options=options)
    fields = np.asarray(np.linspace(1.0, 1e4, 400) if fields is None else fields, dtype=float)
    if fields.ndim != 1 or not fields.size or not np.isfinite(fields).all() or np.any(fields <= 0):
        raise ValueError("Figure fields must be finite positive one-dimensional values")
    ec = float(scan["selected_Ec"][0])
    return _result(
        "fig4b",
        {"fields": fields, "probability": np.exp(-ec / fields)},
        {"fields": ("field",), "probability": ("field",)},
        {"fields": "V/m", "probability": "1"},
        x=x,
        critical_field=ec,
        options=cfg.to_dict(),
        crossings=scan.metadata,
    )


def fig4d(*, x=1.8, fields=None, definition="gap_minima", model_parameters=None, options=None):
    """Two equal passages: 2P(1-P)[1+cos(2pi Lambda/E)] and its envelope."""
    overrides = {**(options or {}), "crossings.definition": definition}
    scan, cfg = _line(x, "fig4d", model_parameters=model_parameters, options=overrides)
    if len(scan["position"]) != 2 or len(scan["Lambda"]) != 1:
        raise ValueError("Fig. 4d requires exactly two resolved central-line crossings")
    fields = np.asarray(np.linspace(200.0, 5e4, 2000) if fields is None else fields, dtype=float)
    if fields.ndim != 1 or not fields.size or not np.isfinite(fields).all() or np.any(fields <= 0):
        raise ValueError("Figure fields must be finite positive one-dimensional values")
    ec, frequency = float(scan["selected_Ec"][0]), float(scan["Lambda"][0])
    probability = np.exp(-ec / fields)
    envelope = 4 * probability * (1 - probability)
    arrays = {
        "fields": fields,
        "probability": 0.5 * envelope * (1 + np.cos(2 * np.pi * frequency / fields)),
        "envelope": envelope,
    }
    return _result(
        "fig4d",
        arrays,
        {key: ("field",) for key in arrays},
        {key: "V/m" if key == "fields" else "1" for key in arrays},
        x=x,
        critical_field=ec,
        interference_frequency=frequency,
        options=cfg.to_dict(),
        approximation="Equal crossings; coherent paper phase; infinite relaxation time",
        crossings=scan.metadata,
    )


def fig5ab(*, xs=(0.0, 0.5, 1.0, 1.2048, 1.4), k=None, kmap=None, model_parameters=None):
    """Absolute four-band energies, pair metric and conduction curvature maps."""
    k = np.asarray(np.linspace(-0.01, 0.01, 801) if k is None else k, dtype=float)
    kmap = np.asarray(np.linspace(-0.012, 0.012, 241) if kmap is None else kmap, dtype=float)
    xs = np.asarray(xs, dtype=float)
    xmap, ymap = np.meshgrid(kmap, kmap, indexing="ij")
    energies, metrics, curvatures, validity = [], [], [], []
    providers = []
    for x in xs:
        model = _model(x, model_parameters)
        provider = ModelGeometry(model)
        line = provider.at_cartesian(k, np.zeros_like(k))
        geo = provider.at_cartesian(xmap, ymap)
        energies.append(np.linalg.eigvalsh(model.hamiltonian(k, np.zeros_like(k))))
        metrics.append(line.g_cv[..., 0])
        curvatures.append(geo.omega_c)
        validity.append(~geo.deg_flag & np.isfinite(geo.omega_c))
        providers.append(provider.describe())
    arrays = {
        "parameter": xs,
        "k": k,
        "kmap": kmap,
        "energies": np.asarray(energies),
        "pair_metric": np.asarray(metrics),
        "curvature": np.asarray(curvatures),
        "valid": np.asarray(validity),
    }
    return _result(
        "fig5ab",
        arrays,
        {
            "parameter": ("member",),
            "k": ("sample",),
            "kmap": ("map_axis",),
            "energies": ("member", "sample", "band"),
            "pair_metric": ("member", "sample"),
            "curvature": ("member", "kx", "ky"),
            "valid": ("member", "kx", "ky"),
        },
        {
            "k": "Angstrom^-1",
            "kmap": "Angstrom^-1",
            "energies": "eV",
            "pair_metric": "Angstrom^2",
            "curvature": "Angstrom^2",
        },
        providers=providers,
    )


def ext7e(*, xs=None, model_parameters=None):
    """Gap-minimum and diabatic Ec/Lambda scans, with unavailable values masked."""
    xs = np.asarray(np.linspace(0.0, 2.0, 401) if xs is None else xs, dtype=float)
    arrays = {"parameter": xs}
    conventions = {}
    for name, figure in (("gap_minimum", "ext7e_left"), ("diabatic", "ext7e_right")):
        cfg = _settings(figure)
        ec, frequency, valid = [], [], []
        for x in xs:
            result = crossing_scan(_model(x, model_parameters), [0.0], options=cfg)
            ec.append(result["selected_Ec"][0])
            frequency.append(result["Lambda"][0] if len(result["Lambda"]) else np.nan)
            valid.append(bool(len(result["Lambda"])))
        arrays[f"Ec_{name}"] = np.asarray(ec)
        arrays[f"Lambda_{name}"] = np.asarray(frequency)
        arrays[f"interference_valid_{name}"] = np.asarray(valid)
        conventions[name] = cfg.to_dict()
    return _result(
        "ext7e",
        arrays,
        {key: ("member",) for key in arrays},
        {key: "V/m" for key in arrays if key.startswith(("Ec_", "Lambda_"))},
        options=conventions,
    )


def fig6bd(*, ky=None, xs=None, model_parameters=None):
    """Critical C/k curvature reference and metric at gap-minimum crossings."""
    ky = np.asarray(np.geomspace(1e-6, 1.2e-2, 200) if ky is None else ky, dtype=float)
    xs = np.asarray(np.linspace(0.8, 1.2, 401) if xs is None else xs, dtype=float)
    xs = xs[np.abs(xs - 1) > 1e-4]
    model = _model(1.0, model_parameters)
    provider = ModelGeometry(model)
    geo = provider.at_cartesian(np.zeros_like(ky), ky)
    metric, positions = [], []
    for x in xs:
        scan = crossing_scan(_model(x, model_parameters), [0.0], options="paper")
        metric.append(scan["metric"][0] if len(scan["metric"]) else np.nan)
        positions.append(scan["position"][0] if len(scan["position"]) else np.nan)
    strength = model.p.C_critical
    arrays = {
        "ky": ky,
        "curvature": geo.omega_c,
        "critical_reference": -strength / ky,
        "parameter": xs,
        "gap_edge_metric": np.asarray(metric),
        "crossing": np.asarray(positions),
    }
    return _result(
        "fig6bd",
        arrays,
        {
            **{key: ("sample",) for key in ("ky", "curvature", "critical_reference")},
            **{key: ("member",) for key in ("parameter", "gap_edge_metric", "crossing")},
        },
        {
            "ky": "Angstrom^-1",
            "curvature": "Angstrom^2",
            "critical_reference": "Angstrom^2",
            "gap_edge_metric": "Angstrom^2",
            "crossing": "Angstrom^-1",
        },
        critical_strength_angstrom=strength,
        provider=provider.describe(),
        options=resolve_options("paper").to_dict(),
    )


def _harmonic_map(figure, *, xs, amplitudes, orders, model_parameters=None, options=None):
    xs, amplitudes, orders = np.asarray(xs), np.asarray(amplitudes), np.asarray(orders, dtype=int)
    cfg = _settings(figure, options)
    results = [
        PreparedHallResponse(_model(x, model_parameters), cfg).harmonics(amplitudes, int(max(orders)))
        for x in xs
    ]
    arrays = {
        "parameter": xs,
        "amplitudes": amplitudes,
        "orders": orders,
        **{key: np.stack([result[key][:, orders] for result in results]) for key in ("a", "b", "c")},
        "converged": np.stack([result["converged"] for result in results]),
    }
    dims = {
        "parameter": ("member",),
        "amplitudes": ("amplitude",),
        "orders": ("harmonic",),
        "converged": ("member", "amplitude"),
        **{key: ("member", "amplitude", "harmonic") for key in ("a", "b", "c")},
    }
    return _result(
        figure,
        arrays,
        dims,
        {"amplitudes": "V/m", **{key: "A/m" for key in ("a", "b", "c")}},
        options=cfg.to_dict(),
        members=[result.metadata for result in results],
    )


def fig5c(*, xs=None, amplitudes=None, orders=(3, 7, 11, 15, 19, 23, 27, 31), **kwargs):
    return _harmonic_map(
        "fig5c",
        xs=np.linspace(1.0, 2.0, 101) if xs is None else xs,
        amplitudes=np.linspace(2e2, 3.5e4, 141) if amplitudes is None else amplitudes,
        orders=orders,
        **kwargs,
    )


def ext7a(*, xs=None, amplitudes=None, orders=(3, 11, 19, 27), **kwargs):
    return _harmonic_map(
        "ext7a",
        xs=np.linspace(0.0, 2.0, 201) if xs is None else xs,
        amplitudes=np.linspace(2.5e2, 5e4, 201) if amplitudes is None else amplitudes,
        orders=orders,
        **kwargs,
    )


def ext9b(
    *,
    scaled_amplitudes=None,
    orders=(3, 5, 7, 9, 11, 13, 15, 17, 19, 21),
    options=None,
    model_parameters=None,
):
    scaled_amplitudes = np.asarray(
        np.geomspace(1e-5, 1e3, 33) if scaled_amplitudes is None else scaled_amplitudes
    )
    orders = np.asarray(orders, dtype=int)
    cfg = _settings(
        "ext9b",
        {
            "drive.E_max": max(orders) * max(scaled_amplitudes),
            "drive.E_min": min(1e-5, min(scaled_amplitudes)),
            **(options or {}),
        },
    )
    prepared = PreparedHallResponse(_model(1.0, model_parameters), cfg)
    results = [prepared.harmonics(scaled_amplitudes * n, int(n)) for n in orders]
    arrays = {
        "scaled_amplitudes": scaled_amplitudes,
        "orders": orders,
        "scaled_sine": np.stack([n * result["b"][:, n] for n, result in zip(orders, results)]),
        "converged": np.stack([result["converged"] for result in results]),
    }
    return _result(
        "ext9b",
        arrays,
        {
            "scaled_amplitudes": ("amplitude",),
            "orders": ("harmonic",),
            "scaled_sine": ("harmonic", "amplitude"),
            "converged": ("harmonic", "amplitude"),
        },
        {"scaled_amplitudes": "V/m", "scaled_sine": "A/m"},
        options=cfg.to_dict(),
        members=[result.metadata for result in results],
    )


FIGURES = {
    function.__name__: function for function in (fig4b, fig4d, fig5ab, fig5c, ext7a, ext7e, ext9b, fig6bd)
}


def compute_figure(figure, **recipe):
    """Evaluate one recipe and return a pickle-free observable, without loading HHH."""
    if figure not in FIGURES:
        raise ValueError(f"Unknown paper figure {figure!r}; choose {', '.join(FIGURES)}")
    return FIGURES[figure](**recipe)
