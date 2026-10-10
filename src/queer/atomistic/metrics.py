"""Native finite-value benchmark metrics for trajectories and phonon bands."""

from __future__ import annotations

import numpy as np


def phonon_metrics(result):
    if hasattr(result, "arrays"):
        values = result.arrays["frequencies_THz"]
    elif isinstance(result, dict):
        values = result.get("band_structure", {}).get("energies", result.get("frequencies_THz", []))
    else:
        values = result
    values = np.asarray(values, float).reshape(-1)
    values = values[np.isfinite(values)]
    return {
        "phonon_min_freq_thz": float(values.min()) if len(values) else None,
        "phonon_max_freq_thz": float(values.max()) if len(values) else None,
        "phonon_n_imaginary": float(np.count_nonzero(values < -1e-3)) if len(values) else None,
    }


def nvt_metrics(result):
    data = result.summary if hasattr(result, "summary") else result
    if "series" not in data and isinstance(data.get("raw_data"), dict):
        data = data["raw_data"]
    rows = [row for row in data.get("series", []) if isinstance(row, dict)]

    def finite(key):
        values = []
        for row in rows:
            try:
                value = float(row[key])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        return np.asarray(values)

    paired = []
    for row in rows:
        try:
            pair = (float(row["time_ps"]), float(row["etot_per_atom_eV"]))
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(pair).all():
            paired.append(pair)
    slope = None
    if paired:
        xy = np.asarray(paired)
        x = xy[:, 0] - xy[:, 0].mean()
        denominator = x @ x
        slope = float(abs(x @ (xy[:, 1] - xy[:, 1].mean()) / denominator) * 1000) if denominator > 0 else 0.0
    temperature, forces = finite("temperature_K"), finite("max_force_eVA")
    setpoint = data.get("temperature_setpoint_K", data.get("parameters", {}).get("temperature_K"))
    requested = data.get("n_steps_requested")
    return {
        "nvt_energy_drift_mev_per_atom_ps": slope,
        "nvt_temp_mean_dev_k": float(abs(temperature.mean() - setpoint))
        if len(temperature) and setpoint is not None
        else None,
        "nvt_temp_std_k": float(temperature.std()) if len(temperature) else None,
        "nvt_max_force": float(forces.max()) if len(forces) else None,
        "nvt_completed": float(bool(data.get("completed"))),
        "nvt_min_interatomic_dist": data.get("min_interatomic_distance"),
        "nvt_steps_fraction": float(data.get("n_steps_run", 0) / requested) if requested else None,
    }


benchmark_metrics_from_nvt = nvt_metrics
benchmark_metrics_from_phonon = phonon_metrics
