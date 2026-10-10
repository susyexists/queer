"""Orbital-resolved Wannier Wigner--Seitz images, without producer packages."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

import numpy as np

from ..geometry.lattice import validated_lattice

if TYPE_CHECKING:
    from .wannier import WannierData, WignerSeitzData

WSKey = tuple[int, int, int, int, int]


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        integer = int(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if integer < 1 or integer != value:
        raise ValueError(f"{name} must be a positive integer")
    return integer


class ComputedWignerSeitzShifts(Mapping[WSKey, np.ndarray]):
    """Compact read-only equivalent of a complete ``_wsvec.dat`` mapping.

    Offsets index candidate image IDs for each translation/orbital pair. This
    avoids one Python array and tuple per pair in large Wannier models. Orbital
    keys remain one-based, matching the native file reader.
    """

    def __init__(self, translations, nbnd, candidates, offsets, image_ids, pair_chunk):
        self.translations = translations
        self.nbnd = nbnd
        self.candidates = candidates
        self.offsets = offsets
        self.image_ids = image_ids
        self.pair_chunk = pair_chunk
        self._translation_index = {tuple(row): i for i, row in enumerate(translations)}
        for array in (translations, candidates, offsets, image_ids):
            array.setflags(write=False)

    def __len__(self) -> int:
        return len(self.translations) * self.nbnd**2

    def __iter__(self) -> Iterator[WSKey]:
        for row in self.translations:
            for m in range(1, self.nbnd + 1):
                for n in range(1, self.nbnd + 1):
                    yield int(row[0]), int(row[1]), int(row[2]), m, n

    def __getitem__(self, key: WSKey) -> np.ndarray:
        if not isinstance(key, tuple) or len(key) != 5:
            raise KeyError(key)
        index = self._translation_index.get(key[:3])
        m, n = key[3:]
        if index is None or not 1 <= m <= self.nbnd or not 1 <= n <= self.nbnd:
            raise KeyError(key)
        if int(m) != m or int(n) != n:
            raise KeyError(key)
        pair = index * self.nbnd**2 + (int(m) - 1) * self.nbnd + int(n) - 1
        result = self.candidates[self.image_ids[self.offsets[pair] : self.offsets[pair + 1]]]
        result.setflags(write=False)
        return result

    def normalized_matrices(self, data: WannierData, matrices):
        """Apply image and HR weights without expanding millions of keys."""
        translations = np.asarray(data.translations)
        if data.nbnd != self.nbnd or len(translations) != len(self.translations):
            raise ValueError("wsvec translation/orbital keys do not match the HR file")
        keys = [tuple(row) for row in translations]
        if len(set(keys)) != len(keys) or set(keys) != set(self._translation_index):
            raise ValueError("wsvec translation/orbital keys do not match the HR file")
        pair_count = self.nbnd**2
        output_keys = set()
        for translation, key in zip(translations, keys, strict=True):
            index = self._translation_index[key]
            a, b = self.offsets[[index * pair_count, (index + 1) * pair_count]]
            for candidate in np.unique(self.image_ids[a:b]):
                output_keys.add(tuple(translation + self.candidates[candidate]))
        ordered = sorted(output_keys)
        lookup = {key: index for index, key in enumerate(ordered)}
        # Allocate the returned Fourier table once; a dictionary of full blocks
        # followed by stacking would temporarily double the largest allocation.
        output = np.zeros((len(ordered),) + matrices.shape[1:], dtype=np.complex128)
        for input_index, (translation, key) in enumerate(zip(translations, keys, strict=True)):
            index = self._translation_index[key]
            values = matrices[input_index].reshape((pair_count,) + matrices.shape[3:])
            for start in range(0, pair_count, self.pair_chunk):
                stop = min(start + self.pair_chunk, pair_count)
                a = index * pair_count + start
                b = index * pair_count + stop
                counts = np.diff(self.offsets[a : b + 1])
                image_ids = self.image_ids[self.offsets[a] : self.offsets[b]]
                pair_ids = np.repeat(np.arange(start, stop), counts)
                divisor = data.degeneracies[input_index] * counts
                weighted = values[start:stop] / divisor.reshape((-1,) + (1,) * (matrices.ndim - 3))
                for candidate in np.unique(image_ids):
                    selected = pair_ids[image_ids == candidate]
                    destination = lookup[tuple(translation + self.candidates[candidate])]
                    block = output[destination].reshape(values.shape)
                    # Each orbital pair occurs once per distinct candidate.
                    block[selected] += weighted[selected - start]
        return np.asarray(ordered, dtype=np.int64), output


def recompute_wsvec(
    data: WannierData,
    centres,
    lattice,
    mp_grid,
    *,
    search_size=2,
    distance_tolerance=1e-5,
    chunk_bytes=32_000_000,
) -> WignerSeitzData:
    """Reconstruct orbital-dependent Wannier90 WS supercell images natively.

    For ``<i,0|H|j,R>``, retain every supercell shift ``T=n*mp_grid`` whose
    Cartesian distance ``|(R+T)@lattice + centre_j-centre_i|`` is within the
    positive tolerance (Å) of the nearest image. Centres are Cartesian Å and
    must retain the original Wannier home cells. Lattice vectors are rows.

    Search integers span ``[-search_size,search_size]`` on all three axes.
    A reciprocal-plane bound verifies that no omitted image can tie or beat
    the selected images; increase ``search_size`` if the bound is inconclusive.
    ``chunk_bytes`` bounds estimated temporary distance workspace, excluding
    inputs and the returned compact numeric mapping. Normalization retains
    that chunk layout and applies both HR degeneracy and image weights once.
    """
    from .wannier import WignerSeitzData

    search = _positive_integer(search_size, "search_size")
    budget = _positive_integer(chunk_bytes, "chunk_bytes")
    if isinstance(distance_tolerance, (bool, np.bool_)) or not isinstance(
        distance_tolerance, (int, float, np.integer, np.floating)
    ):
        raise ValueError("distance_tolerance must be positive and finite")
    tolerance = float(distance_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("distance_tolerance must be positive and finite")
    lat = validated_lattice(lattice).copy()
    grid = np.asarray(mp_grid)
    if (
        grid.shape != (3,)
        or grid.dtype.kind not in "iu"
        or np.any(grid < 1)
        or np.any(grid > np.iinfo(np.int64).max // search)
    ):
        raise ValueError("mp_grid must contain three positive integers")
    grid = grid.astype(np.int64)
    translations = np.asarray(data.translations)
    hopping = np.asarray(data.hopping)
    if (
        translations.ndim != 2
        or translations.shape[1] != 3
        or not len(translations)
        or translations.dtype.kind not in "iu"
        or len(set(map(tuple, translations))) != len(translations)
        or hopping.ndim != 3
        or hopping.shape[0] != len(translations)
        or hopping.shape[1] < 1
        or hopping.shape[1] != hopping.shape[2]
        or not np.isfinite(hopping).all()
    ):
        raise ValueError(
            "Wannier data require distinct integer translations and finite square hopping blocks"
        )
    nbnd = hopping.shape[1]
    degeneracies = np.asarray(data.degeneracies)
    if (
        degeneracies.shape != (len(translations),)
        or degeneracies.dtype.kind not in "iu"
        or np.any(degeneracies < 1)
    ):
        raise ValueError("Wannier degeneracies must be positive integers for every translation")
    taus = np.asarray(centres, float)
    if taus.shape != (nbnd, 3) or not np.isfinite(taus).all():
        raise ValueError("centres must be a finite (NB,3) Cartesian Angstrom array")
    if data.lattice is not None and not np.allclose(lat, data.lattice, rtol=1e-8, atol=1e-6):
        raise ValueError("WS lattice does not match the Wannier data lattice")
    candidate_count = (2 * search + 1) ** 3
    fixed_bytes = 80 * candidate_count
    bytes_per_pair = 48 * candidate_count + 128
    pair_chunk = (budget - fixed_bytes) // bytes_per_pair
    if pair_chunk < 1:
        raise ValueError("chunk_bytes cannot fit one WS orbital-pair distance search")
    if candidate_count > np.iinfo(np.uint32).max:
        raise ValueError("WS search has too many candidate images")
    axes = np.arange(-search, search + 1, dtype=np.int64)
    candidates = np.stack(np.meshgrid(axes, axes, axes, indexing="ij"), axis=-1).reshape(-1, 3) * grid
    candidate_cart = candidates @ lat
    supercell = grid[:, None] * lat
    inverse_supercell = np.linalg.inv(supercell)
    inverse_plane_norm = np.linalg.norm(inverse_supercell, axis=0)
    pairs_per_r = nbnd**2
    offsets = np.empty(len(translations) * pairs_per_r + 1, dtype=np.int64)
    offsets[0] = 0
    id_type = np.uint16 if candidate_count <= np.iinfo(np.uint16).max else np.uint32
    image_chunks = []
    image_count = 0
    for index, translation in enumerate(translations):
        for start in range(0, pairs_per_r, pair_chunk):
            stop = min(start + pair_chunk, pairs_per_r)
            pairs = np.arange(start, stop)
            vectors = taus[pairs % nbnd] - taus[pairs // nbnd] + translation @ lat
            displacement = candidate_cart[:, None, :] + vectors[None, :, :]
            distances = np.einsum("cpa,cpa->cp", displacement, displacement)
            del displacement
            np.sqrt(distances, out=distances)
            nearest = distances.min(axis=0)
            # Outside the integer cube, at least one reciprocal coordinate is
            # >= search+1 in magnitude. Its dual-plane distance bounds the norm.
            fractional = vectors @ inverse_supercell
            outside_lower_bound = np.min(
                np.maximum(0, search + 1 - abs(fractional)) / inverse_plane_norm,
                axis=1,
            )
            if np.any(outside_lower_bound <= nearest + tolerance):
                raise ValueError(
                    "WS search_size cannot exclude closer or tied images outside the search; increase search_size"
                )
            distances -= nearest[None, :]
            tied = distances < tolerance
            counts = tied.sum(axis=0)
            # Transpose makes the ragged result pair-major, then candidate order.
            _, chosen = np.nonzero(tied.T)
            image_chunks.append(chosen.astype(id_type))
            a = index * pairs_per_r + start
            offsets[a + 1 : a + len(counts) + 1] = image_count + np.cumsum(counts)
            image_count += len(chosen)
    image_ids = np.concatenate(image_chunks)
    shifts = ComputedWignerSeitzShifts(
        translations.astype(np.int64, copy=True), nbnd, candidates, offsets, image_ids, pair_chunk
    )
    return WignerSeitzData(shifts, use_ws_distance=True, ndegen_applied=False)
