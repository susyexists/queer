"""Deterministic dataset splits and native MACE-reference extended XYZ export."""

from __future__ import annotations

import numpy as np

from ..physics.constants import DATASET_KBAR_EV_ANGSTROM3 as KBAR_TO_EV_A3
from ..structures import Frame


def split_dataset(items, *, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Partition each item once, preserving nonempty positive-ratio splits."""
    items = list(items)
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], float)
    if (
        not np.isfinite(ratios).all()
        or np.any(ratios < 0)
        or not np.isclose(ratios.sum(), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("split ratios must be nonnegative and sum to one")
    positive = ratios > 0
    if len(items) < int(positive.sum()):
        raise ValueError("not enough items for the requested nonempty dataset splits")
    counts = np.floor(len(items) * ratios).astype(int)
    counts[positive & (counts == 0)] = 1
    while counts.sum() > len(items):
        candidates = np.where(counts > positive.astype(int))[0]
        index = candidates[np.argmax(counts[candidates] - len(items) * ratios[candidates])]
        counts[index] -= 1
    while counts.sum() < len(items):
        deficits = len(items) * ratios - counts
        deficits[~positive] = -np.inf
        counts[np.argmax(deficits)] += 1
    order = np.random.default_rng(seed).permutation(len(items))
    boundaries = np.concatenate(([0], np.cumsum(counts)))
    return {
        name: [items[int(index)] for index in order[boundaries[i] : boundaries[i + 1]]]
        for i, name in enumerate(("train", "valid", "test"))
    }


def dataset_item_to_frame(item, *, stress_unit="eV/Å³", stress_sign=1.0):
    """Convert reference items with canonical eV/Å³ tensile stress by default.

    ``stress_sign`` is explicit because kbar records alone do not identify a
    compression/tension convention. Producers with other conventions must
    declare the input unit and compression-to-tension sign conversion.
    """
    from ..io.atomistic import parse_poscar

    outputs = item.get("outputs", {})
    structure = parse_poscar(str(outputs["poscar"]))
    energy = float(outputs["energy"])
    forces = np.asarray(outputs["forces"], float)
    if not np.isfinite(energy) or forces.shape != (structure.natoms, 3) or not np.isfinite(forces).all():
        raise ValueError("reference energy and forces must be finite and match the structure")
    metadata = {**structure.metadata, "REF_energy": energy}
    if outputs.get("stress") is not None:
        stress = np.asarray(outputs["stress"], float)
        if stress.shape != (3, 3) or not np.isfinite(stress).all():
            raise ValueError("reference stress must be finite (3,3)")
        if stress_unit not in ("kbar", "eV/Å³") or not np.isfinite(stress_sign):
            raise ValueError("stress_unit must be kbar or eV/Å³ with a finite sign factor")
        metadata["REF_stress"] = stress * stress_sign * (KBAR_TO_EV_A3 if stress_unit == "kbar" else 1)
    return Frame(structure.copy(arrays={**structure.arrays, "REF_forces": forces}, metadata=metadata))


def dataset_items_to_extxyz(items, *, stress_unit="eV/Å³", stress_sign=1.0):
    from ..io.atomistic import write_extxyz

    return write_extxyz(
        [dataset_item_to_frame(item, stress_unit=stress_unit, stress_sign=stress_sign) for item in items]
    )


def export_dataset_splits(
    items, *, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, stress_unit="eV/Å³", stress_sign=1.0
):
    parts = split_dataset(
        items, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )
    return {
        f"{name}.xyz": dataset_items_to_extxyz(part, stress_unit=stress_unit, stress_sign=stress_sign)
        for name, part in parts.items()
    }
