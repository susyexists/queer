"""Optional headless rendering from immutable numerical run artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io.artifacts import atomic_json, load_manifest, utc_now


def render_run(run_dir: str | Path) -> Path:
    """Render a completed run without loading models or recalculating energies."""
    root = Path(run_dir).expanduser().resolve()
    manifest = load_manifest(root)
    # Visualization stays optional for numerical and scheduler processes.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = root / "figures"
    output.mkdir(exist_ok=True)
    name = manifest["workflow"]
    files = []
    from .finite_field import FINITE_FIELD_NAMES

    if name in FINITE_FIELD_NAMES:
        from ..plotting.finite_field import render_finite_field

        record = manifest["cases"].get("result")
        if record is None:
            raise ValueError("Finite-field rendering requires a final result")
        with np.load(root / record["path"], allow_pickle=False) as data:
            if record["metadata"].get("figure"):
                from ..plotting.paper_figures import render_paper_figure

                arrays = {key: data[key] for key in data.files}
                if "render_orders" in record["metadata"]:
                    orders = np.asarray(record["metadata"]["render_orders"])
                    harmonic_indices = [int(np.flatnonzero(arrays["orders"] == order)[0]) for order in orders]
                    for coefficient in ("a", "b", "c"):
                        arrays[coefficient] = arrays[coefficient][..., harmonic_indices]
                    arrays["orders"] = orders
                files = render_paper_figure(arrays, record["metadata"], output)
            else:
                files = render_finite_field(
                    name, {key: data[key] for key in data.files}, record["metadata"], output
                )
    elif name in {
        "structure",
        "hamiltonian_pack",
        "select",
        "phonons",
        "nvt",
        "relax",
        "postprocess",
        "hamiltonian_bands",
        "dataset",
        "band_compare",
    }:
        import json

        record = manifest["cases"].get("result")
        if record is None:
            raise ValueError("Service rendering requires a final numerical result")
        metadata = record["metadata"]
        with np.load(root / record["path"], allow_pickle=False) as data:
            fig = plt.figure()
            if name in {"structure", "hamiltonian_pack", "select"}:
                prefix = "final_" if name == "select" else ""
                positions, numbers = data[prefix + "positions"], data[prefix + "atomic_numbers"]
                ax = fig.add_subplot(projection="3d")
                ax.scatter(*positions.T, c=numbers, s=45)
                ax.set(xlabel="x (Å)", ylabel="y (Å)", zlabel="z (Å)")
            else:
                ax = fig.add_subplot()
                if name == "phonons":
                    x, values = data["q_distance"], data["frequencies_THz"].T
                    segments = metadata.get("segments", [])
                    if segments:
                        for segment in segments:
                            interval = slice(segment["start_index"], segment["end_index"] + 1)
                            ax.plot(x[interval], values[interval])
                        ticks = [segment["start_index"] for segment in segments] + [segments[-1]["end_index"]]
                        labels = [segment["start_label"] for segment in segments] + [
                            segments[-1]["end_label"]
                        ]
                        ax.set_xticks(x[ticks], labels)
                    else:
                        ax.plot(x, values)
                    ax.axhline(0, color="black", linewidth=0.5)
                    ax.set(xlabel="Reciprocal path distance (Å⁻¹)", ylabel="Frequency (THz)")
                elif name in {"hamiltonian_bands", "postprocess"} and "energies" in data:
                    energies = data["energies"]
                    ef = metadata.get("fermi_energy_eV")
                    values = energies if ef is None else energies - ef
                    x = data["distances"] if "distances" in data else np.arange(energies.shape[1])
                    segments = metadata.get("segments", [(0, len(x))])
                    for start, end in segments:
                        for spin in values:
                            ax.plot(x[start:end], spin[start:end])
                    indices = metadata.get("label_indices", [])
                    if indices:
                        ax.set_xticks(x[indices], metadata["labels"])
                    ax.set(
                        xlabel="Reported path distance" if "distances" in data else "k-point index",
                        ylabel="E − EF (eV)" if ef is not None else "Absolute energy (eV)",
                    )
                elif name == "nvt":
                    ax.plot(data["time_ps"], data["etot_eV"])
                    ax.set(xlabel="Time (ps)", ylabel="Total energy (eV)")
                elif name == "relax":
                    ax.plot(data["steps"], data["max_force_eV_A"])
                    ax.set(xlabel="Optimizer step", ylabel="Maximum force (eV/Å)")
                else:
                    ax.axis("off")
                    ax.text(
                        0,
                        1,
                        json.dumps(metadata.get("summary", {}), indent=2),
                        va="top",
                        transform=ax.transAxes,
                        family="monospace",
                        fontsize=8,
                    )
            ax.set_title(name.replace("_", " "))
            target = output / f"{name}.png"
            fig.savefig(target, bbox_inches="tight")
            plt.close(fig)
            files.append(target.name)
    elif name == "bands":
        with np.load(root / manifest["cases"]["bands"]["path"], allow_pickle=False) as data:
            fig, ax = plt.subplots()
            ax.plot(data["energies"].T)
            ax.set(xlabel="k-point index", ylabel="Energy (eV)")
            if len(data["symmetry"]):
                ax.set_xticks(data["symmetry"], data["labels"])
            target = output / "bands.png"
            fig.savefig(target, bbox_inches="tight")
            plt.close(fig)
            files.append(target.name)
    elif name in {"susceptibility", "epw"}:
        rows = []
        for record in manifest["cases"].values():
            with np.load(root / record["path"], allow_pickle=False) as data:
                rows.append(
                    np.asarray(data["response" if name == "susceptibility" else "self_energy"]).reshape(-1)
                )
        values = np.asarray(rows)
        fig, ax = plt.subplots()
        ax.plot(values.real, label="Real")
        ax.plot(values.imag, linestyle="--", label="Imaginary")
        ax.set(
            xlabel="q-point index",
            ylabel="Response (1/eV)" if name == "susceptibility" else "EPW response (legacy normalization)",
        )
        ax.legend()
        target = output / f"{name}.png"
        fig.savefig(target, bbox_inches="tight")
        plt.close(fig)
        files.append(target.name)
    elif name == "spectral":
        spectral_groups: dict[int, tuple[list[np.ndarray], list[float], float | None]] = {}
        for record in manifest["cases"].values():
            with np.load(root / record["path"], allow_pickle=False) as data:
                group_index = record["metadata"].get("v0_index", -1)
                potential = float(data["v0"][0]) if "v0" in data else None
                group = spectral_groups.setdefault(group_index, ([], [], potential))
                group[0].append(data["spectrum"][0])
                group[1].append(float(data["omega"][0]))
        for group_index, (rows, omega, potential) in spectral_groups.items():
            fig, ax = plt.subplots()
            frequencies, indices = np.unique(omega, return_index=True)
            ordered_rows = np.asarray(rows)[indices]
            if len(frequencies) == 1:
                half_width = manifest["config"]["parameters"]["eta"]
                edges = np.asarray([frequencies[0] - half_width, frequencies[0] + half_width])
            else:
                midpoints = (frequencies[:-1] + frequencies[1:]) / 2
                edges = np.concatenate(
                    (
                        [frequencies[0] - (midpoints[0] - frequencies[0])],
                        midpoints,
                        [frequencies[-1] + (frequencies[-1] - midpoints[-1])],
                    )
                )
            spectral_artist = ax.pcolormesh(
                np.arange(ordered_rows.shape[1] + 1) - 0.5, edges, ordered_rows, shading="flat"
            )
            ax.set(xlabel="k-point index", ylabel="Energy (eV)")
            if potential is not None:
                ax.set_title(f"V0={potential:g} eV")
            fig.colorbar(spectral_artist, ax=ax, label="Spectral weight")
            target = output / ("spectral.png" if potential is None else f"spectral_v0_{group_index:04d}.png")
            fig.savefig(target, bbox_inches="tight")
            plt.close(fig)
            files.append(target.name)
    elif name == "momentum":
        for case, record in manifest["cases"].items():
            with np.load(root / record["path"], allow_pickle=False) as data:
                surface = data["surface"]
                fig, ax = plt.subplots()
                artist = ax.scatter(surface[:, 0], surface[:, 1], c=surface[:, 4], s=2, cmap="inferno")
                ax.set(
                    xlabel="kx (angstrom^-1)",
                    ylabel="ky (angstrom^-1)",
                    title=f"V0={data['v0'][0]:g} eV, binding energy={data['binding_energy'][0]:g} eV",
                )
                fig.colorbar(artist, ax=ax, label="Intensity")
                target = output / f"{case}.png"
                fig.savefig(target, bbox_inches="tight")
                plt.close(fig)
                files.append(target.name)
    else:
        record = manifest["cases"].get("result")
        if record is None:
            raise ValueError(f"No final observable artifact for {name}")
        with np.load(root / record["path"], allow_pickle=False) as data:
            candidates = [
                key
                for key in data.files
                if data[key].dtype.kind in "fci"
                and key
                not in {
                    "kpoints",
                    "omega",
                    "chemical_potentials",
                    "temperatures",
                    "groups",
                    "band_indices",
                    "band_channels",
                    "polarization_quantum",
                }
            ]
            preferred = [name, "conductivity", "metric", "curvature", "energies", "electronic", "phases"]
            key = next(
                (value for value in preferred if value in candidates), candidates[0] if candidates else None
            )
            if key is None:
                raise ValueError(f"No numerical quantity to render for {name}")
            values = np.asarray(data[key])
            dimensions = record["metadata"].get("dims", {}).get(key, [])
            unit = record["metadata"].get("units", {}).get(key, "")
            fig, ax = plt.subplots()
            if not values.size:
                ax.text(0.5, 0.5, "No matching samples", ha="center", va="center", transform=ax.transAxes)
            elif values.ndim == 0:
                ax.bar([key], [float(values.real)])
            else:
                axis = next(
                    (
                        i
                        for i, value in enumerate(dimensions)
                        if value in {"k", "kpoint", "omega", "energy", "chemical_potential", "temperature"}
                    ),
                    0,
                )
                quantity_rows = np.moveaxis(values, axis, 0).reshape(values.shape[axis], -1)
                coordinate = {
                    "omega": "omega",
                    "energy": "omega",
                    "chemical_potential": "chemical_potentials",
                    "temperature": "temperatures",
                }.get(dimensions[axis] if dimensions else "")
                if (
                    coordinate not in data
                    or data[coordinate].ndim != 1
                    or len(data[coordinate]) != len(quantity_rows)
                ):
                    coordinate = None
                x = np.arange(len(quantity_rows)) if coordinate is None else data[coordinate]
                ax.plot(x, quantity_rows.real)
                if np.iscomplexobj(quantity_rows) and np.any(quantity_rows.imag):
                    ax.plot(x, quantity_rows.imag, linestyle="--")
                ax.set_xlabel(coordinate or (dimensions[axis] if dimensions else "sample"))
            ax.set_title(name.replace("_", " "))
            ax.set_ylabel(f"{key} ({unit})" if unit else key)
            target = output / f"{name}.png"
            fig.savefig(target, bbox_inches="tight")
            plt.close(fig)
            files.append(target.name)
    atomic_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "rendered_at": utc_now(),
            "source_fingerprint": manifest["fingerprint"],
            "files": files,
        },
    )
    return output
