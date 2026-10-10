"""Lowest-energy frame selection with explicit simulation-step alignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..structures import Frame


@dataclass
class SelectionResult:
    frame: Frame
    summary: dict

    @property
    def metadata(self):
        return self.summary


def select_min_energy_frame(
    frames,
    samples=None,
    *,
    energy_key="etot_eV",
    fallback=None,
    alignment="exact",
) -> SelectionResult:
    """Select the minimum finite energy for a recorded frame.

    With samples, exact mode considers only matching frame/sample steps; a
    lower-energy sample with no geometry is not attributed to another frame.
    ``previous`` explicitly permits selecting the preceding frame and reports
    the sample and geometry steps separately. With no samples, use stored frame
    energies (and kinetic energy when total energy is requested).
    """
    from .dynamics import ACCELERATION

    frames = list(frames)
    if alignment not in ("exact", "previous"):
        raise ValueError("alignment must be exact or previous")
    if energy_key not in ("etot_eV", "epot_eV"):
        raise ValueError("energy_key must be etot_eV or epot_eV")
    sample_rows = [] if samples is None else list(samples)
    candidates = []
    sample_step: int | None
    if samples is None:
        for index, frame in enumerate(frames):
            if frame.evaluation is None:
                continue
            energy = frame.evaluation.energy_eV
            if energy_key == "etot_eV" and frame.velocities is not None:
                energy += float(
                    np.sum(frame.structure.masses[:, None] * frame.velocities**2) / (2 * ACCELERATION)
                )
            if not np.isfinite(energy):
                continue
            candidates.append((energy, index, frame.step))
    else:
        by_step = {frame.step: index for index, frame in enumerate(frames) if frame.step is not None}
        steps = sorted(by_step)
        for row in sample_rows:
            if not isinstance(row, dict):
                continue
            try:
                energy = float(row[energy_key])
                sample_step = int(row["step"])
                if float(row["step"]) != sample_step:
                    continue
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if not np.isfinite(energy):
                continue
            if sample_step in by_step:
                index = by_step[sample_step]
            elif alignment == "previous" and steps and sample_step >= steps[0]:
                preceding = int(np.searchsorted(steps, sample_step, side="right")) - 1
                index = by_step[steps[preceding]]
            else:
                continue
            candidates.append((energy, index, sample_step))
    if candidates:
        # Stable argmin preserves the first occurrence when energies tie.
        energy, index, sample_step = min(candidates, key=lambda candidate: candidate[0])
        frame, passthrough = frames[index], False
    else:
        energy, index, sample_step = None, 0, None
        if frames:
            frame = frames[0]
        elif fallback is not None:
            frame = fallback if isinstance(fallback, Frame) else Frame(fallback)
        else:
            raise ValueError("selection has no usable frame or fallback structure")
        passthrough = True
    summary = {
        "kind": "select_min_energy",
        "energy_key": energy_key,
        "selected_index": index,
        "selected_step": frame.step,
        "selected_sample_step": sample_step,
        "selected_energy_eV": energy,
        "energy_matches_geometry": sample_step is None or sample_step == frame.step,
        "n_frames": len(frames),
        "n_series_rows": len(sample_rows),
        "passthrough": passthrough,
        "alignment": alignment,
        "converged": True,
    }
    return SelectionResult(frame, summary)
