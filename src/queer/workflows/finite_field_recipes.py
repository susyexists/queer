"""Durable paper figure recipes composed from ordinary workflow stages."""

import numpy as np

from ..observables.result import ObservableResult
from .finite_field_specs import _numbers


def _paper_figure(config, store):
    """Keep transport recipes under the same staged owner as ordinary scans."""
    from dataclasses import replace

    from ..finite_field.figures import compute_figure
    from .finite_field import _commit_result, _harmonic_map, _prepare, _read, _restore_result
    from .finite_field_sources import load_finite_field_source

    figure = config.parameters["figure"]
    recipe = dict(config.parameters.get("recipe", {}))
    if figure not in {"fig5c", "ext7a", "ext9b"}:
        return compute_figure(figure, **recipe)
    if figure in {"fig5c", "ext7a"}:
        unknown = set(recipe) - {"xs", "amplitudes", "orders", "model_parameters", "options"}
        if unknown:
            raise ValueError(f"Unknown {figure} recipe options: {sorted(unknown)}")
        xs = recipe.get("xs", np.linspace(1.0, 2.0, 101) if figure == "fig5c" else np.linspace(0.0, 2.0, 201))
        amplitudes = recipe.get(
            "amplitudes", np.linspace(2e2, 3.5e4, 141) if figure == "fig5c" else np.linspace(2.5e2, 5e4, 201)
        )
        orders = np.asarray(
            recipe.get("orders", (3, 7, 11, 15, 19, 23, 27, 31) if figure == "fig5c" else (3, 11, 19, 27))
        )
        _numbers(xs, "recipe.xs")
        _numbers(amplitudes, "recipe.amplitudes", positive=True)
        _figure_orders(orders)
        member = replace(
            config,
            workflow="harmonic_map",
            parameters={
                "model_values": list(xs),
                "amplitudes": list(amplitudes),
                "n_max": int(max(orders)),
                "model": recipe.get("model_parameters", {}),
                "options": {"preset": figure, **recipe.get("options", {})},
            },
        )
        result = _harmonic_map(member, None, store)
        arrays = {**result.arrays, "orders": orders}
        for key in ("a", "b", "c"):
            arrays[key] = arrays[key][..., orders]
        return ObservableResult(
            arrays,
            result.dims,
            result.units,
            {
                **result.metadata,
                "observable": "paper_figure",
                "figure": figure,
                "recipe_version": 1,
                "interpretation": "Explicit paper reproduction conventions",
            },
        )
    unknown = set(recipe) - {"scaled_amplitudes", "orders", "model_parameters", "options"}
    if unknown:
        raise ValueError(f"Unknown ext9b recipe options: {sorted(unknown)}")
    scaled = np.asarray(recipe.get("scaled_amplitudes", np.geomspace(1e-5, 1e3, 33)), float)
    orders = np.asarray(recipe.get("orders", tuple(range(3, 22, 2))))
    _numbers(scaled, "recipe.scaled_amplitudes", positive=True)
    _figure_orders(orders)
    member = replace(
        config,
        workflow="hall_harmonics",
        parameters={
            "model": {**recipe.get("model_parameters", {}), "x": 1.0},
            "options": {
                "preset": figure,
                "drive.E_max": float(max(orders) * max(scaled)),
                "drive.E_min": float(min(1e-5, min(scaled))),
                **recipe.get("options", {}),
            },
        },
    )
    prepared = None
    results: list[ObservableResult] = []
    for order in orders:
        case = f"orders/{int(order):06d}/Fourier"
        if store.completed(case):
            arrays, metadata = _read(store, case)
            result = _restore_result(arrays, metadata)
        else:
            if prepared is None:
                prepared = _prepare(member, load_finite_field_source(member), store)
            result = prepared.harmonics(scaled * order, int(order))
            _commit_result(store, case, result)
        results.append(result)
    return ObservableResult(
        {
            "scaled_amplitudes": scaled,
            "orders": orders,
            "scaled_sine": np.stack([n * result["b"][:, n] for n, result in zip(orders, results)]),
            "converged": np.stack([result["converged"] for result in results]),
        },
        {
            "scaled_amplitudes": ("amplitude",),
            "orders": ("harmonic",),
            "scaled_sine": ("harmonic", "amplitude"),
            "converged": ("harmonic", "amplitude"),
        },
        {"scaled_amplitudes": "V/m", "scaled_sine": "A/m", "orders": "1", "converged": "1"},
        {
            **results[0].metadata,
            "observable": "paper_figure",
            "figure": figure,
            "recipe_version": 1,
            "interpretation": "Explicit paper reproduction conventions",
            "members": [result.metadata for result in results],
        },
    )


def _figure_orders(orders):
    if orders.ndim != 1 or not orders.size or orders.dtype.kind not in "iu" or np.any(orders < 1):
        raise ValueError("parameters.recipe.orders must contain positive integers")
