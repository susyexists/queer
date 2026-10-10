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
    if name == "bands":
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
