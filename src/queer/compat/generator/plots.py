"""Figures of the Generator Wannier steps, drawn as the PythTB driver draws them.

Each figure is a :class:`matplotlib.figure.Figure` (no pyplot state), saved to
``Out/<name>`` at 130 dpi with a tight bounding box and listed in the run's
outputs with the driver's title. Without Matplotlib a figure is skipped
silently; a figure that fails to draw or save only adds a ``plot_skipped``
warning, so a plot never fails a run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .results import OUT_DIR, RunRecord

# Colour cycles of the driver: species markers on the band plot, lines elsewhere.
SPECIES_PALETTE = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#f0e442", "#000000")
LINE_PALETTE = SPECIES_PALETTE[:6]
DPI = 130
GAMMA_LABELS = ("G", "GAMMA", "\\GAMMA")


def new_figure(figsize: tuple[float, float]) -> Any | None:
    """A Matplotlib figure of ``figsize`` inches, or None when Matplotlib is not installed."""
    try:
        from matplotlib.figure import Figure
    except ImportError:
        return None
    return Figure(figsize=figsize)


def save_figure(record: RunRecord, figure: Any, name: str, title: str) -> None:
    """Write ``figure`` to ``Out/name`` in the run's workspace and list it as a plot output."""
    target = record.workspace / OUT_DIR / name
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=DPI, bbox_inches="tight")
    record.add_output(f"{OUT_DIR}/{name}", title, "plot")


def energy_window(params: Any) -> tuple[float, float]:
    """The plotted energy range relative to E_F: ``energy_window_eV``, else (-3, 3) eV."""
    window = params.driver_value("energy_window_eV", (-3.0, 3.0))
    return float(window[0]), float(window[1])


def tick_marks(path: Mapping[str, Any], distance: Sequence[float]) -> tuple[list[float], list[str]]:
    """Tick positions and labels along a band path.

    A tick index past the end sits on the last point; G, Gamma and \\Gamma read
    as the Greek letter; ticks at the same position merge as ``A|B`` (a path
    jump) and identical neighbours collapse to one.
    """
    positions: list[float] = []
    labels: list[str] = []
    for tick, label in zip(path["tick_indices"], path["tick_labels"]):
        x = float(distance[min(int(tick), len(distance) - 1)])
        name = "Γ" if str(label).upper() in GAMMA_LABELS else str(label)
        if positions and abs(positions[-1] - x) < 1e-9:
            if labels[-1] != name:
                labels[-1] = f"{labels[-1]}|{name}"
            continue
        positions.append(x)
        labels.append(name)
    return positions, labels
