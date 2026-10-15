"""Artifact-only rendering handlers; optional plotting loads at render time."""

from pathlib import Path

import numpy as np


def render(root, manifest, output) -> list[Path]:
    import matplotlib.pyplot as plt

    name = manifest["workflow"]
    files = []
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
                    labels = [segment["start_label"] for segment in segments] + [segments[-1]["end_label"]]
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
        files.append(target)
    return files
