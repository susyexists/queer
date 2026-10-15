"""Orbital-resolved Wannier Wigner--Seitz images, without producer packages."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from ..geometry.lattice import validated_lattice

if TYPE_CHECKING:
    from .wannier import WannierData, WignerSeitzData

WSKey = tuple[int, int, int, int, int]


class PackedWignerSeitzShifts(Mapping[WSKey, np.ndarray]):
    """File images stored as integer arrays with the native one-based keys.

    Iteration and normalization do not materialize a Python dictionary of the
    millions of orbital pairs in a production Wannier file.
    """

    def __init__(self, entries, shifts, *, source="wsvec", chunk_size=65536):
        entries = np.asarray(entries, dtype=np.int64).reshape(-1, 6)
        shifts = np.asarray(shifts, dtype=np.int64).reshape(-1, 3)
        self.entries = entries
        self.offsets = np.r_[0, np.cumsum(entries[:, 5], dtype=np.int64)]
        self.images = shifts
        self.chunk_size = int(chunk_size)
        if np.any(entries[:, 5] < 1) or self.offsets[-1] != len(shifts):
            raise ValueError(f"{source}: truncated wsvec entry or nonpositive shift count")
        self._order = np.lexsort(tuple(entries[:, column] for column in range(4, -1, -1)))
        for start in range(1, len(entries), self.chunk_size):
            stop = min(start + self.chunk_size, len(entries))
            left = entries[self._order[start - 1 : stop - 1], :5]
            right = entries[self._order[start:stop], :5]
            if np.any(np.all(left == right, axis=1)):
                raise ValueError(f"{source}: duplicate wsvec entry")
        for index in np.flatnonzero(entries[:, 5] > 1):
            block = shifts[self.offsets[index] : self.offsets[index + 1]]
            if len(np.unique(block, axis=0)) != len(block):
                raise ValueError(f"{source}: duplicate equivalent shifts at entry {index + 1}")
        for array in (entries, self.offsets, shifts, self._order):
            array.setflags(write=False)

    def __len__(self):
        return len(self.entries)

    @classmethod
    def from_mapping(cls, mapping):
        if isinstance(mapping, cls):
            return mapping
        entries = np.empty((len(mapping), 6), dtype=np.int64)
        blocks = []
        for index, (key, values) in enumerate(mapping.items()):
            raw = np.asarray(values)
            if (
                not isinstance(key, tuple)
                or len(key) != 5
                or any(not isinstance(value, (int, np.integer)) for value in key)
                or raw.ndim != 2
                or raw.shape[1] != 3
                or raw.dtype.kind not in "iu"
            ):
                raise ValueError("wsvec shifts require integer translation/orbital keys and image vectors")
            entries[index, :5], entries[index, 5] = key, len(raw)
            blocks.append(raw)
        return cls(entries, np.concatenate(blocks) if blocks else np.empty((0, 3), np.int64))

    def __iter__(self):
        for row in self.entries:
            yield tuple(int(value) for value in row[:5])

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 5:
            raise KeyError(key)
        if any(not isinstance(value, (int, np.integer)) for value in key):
            raise KeyError(key)
        key = tuple(int(value) for value in key)
        lo, hi = 0, len(self._order)
        while lo < hi:
            mid = (lo + hi) // 2
            candidate = tuple(int(value) for value in self.entries[self._order[mid], :5])
            if candidate < key:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(self._order):
            raise KeyError(key)
        index = int(self._order[lo])
        if tuple(int(value) for value in self.entries[index, :5]) != key:
            raise KeyError(key)
        return self.images[self.offsets[index] : self.offsets[index + 1]]

    def _source_rows(self, data):
        expected = len(data.translations) * data.nbnd**2
        if len(self) != expected:
            raise ValueError("wsvec translation/orbital keys do not match the HR file")
        lookup = {tuple(row): index for index, row in enumerate(data.translations)}
        if len(lookup) != len(data.translations):
            raise ValueError("wsvec translation/orbital keys do not match the HR file")
        for start in range(0, len(self), self.chunk_size):
            rows = self.entries[start : start + self.chunk_size]
            if np.any(rows[:, 3:5] < 1) or np.any(rows[:, 3:5] > data.nbnd):
                raise ValueError("wsvec translation/orbital keys do not match the HR file")
            vectors = np.unique(rows[:, :3], axis=0)
            if any(tuple(row) not in lookup for row in vectors):
                raise ValueError("wsvec translation/orbital keys do not match the HR file")
        # Distinct entries, the expected count and valid indices certify coverage.
        return lookup

    def _image_chunks(self):
        for start in range(0, len(self.images), self.chunk_size):
            stop = min(start + self.chunk_size, len(self.images))
            owner = np.searchsorted(self.offsets, np.arange(start, stop), side="right") - 1
            yield self.entries[owner], self.images[start:stop]

    def normalized_operator_arrays(self, data, arrays):
        """Scatter shared image weights into separately allocated operators."""
        source_lookup = self._source_rows(data)
        output_keys = set()
        for rows, images in self._image_chunks():
            output_keys.update(map(tuple, np.unique(rows[:, :3] + images, axis=0)))
        ordered = sorted(output_keys)
        lookup = {key: index for index, key in enumerate(ordered)}
        outputs = [np.zeros((len(ordered),) + values.shape[1:], complex) for values in arrays]
        for rows, images in self._image_chunks():
            source_vectors, inverse = np.unique(rows[:, :3], axis=0, return_inverse=True)
            source = np.array([source_lookup[tuple(row)] for row in source_vectors], dtype=np.int64)[inverse]
            destination_vectors, inverse = np.unique(rows[:, :3] + images, axis=0, return_inverse=True)
            destination = np.array([lookup[tuple(row)] for row in destination_vectors], dtype=np.int64)[
                inverse
            ]
            m, n = rows[:, 3] - 1, rows[:, 4] - 1
            weight = 1.0 / (data.degeneracies[source] * rows[:, 5])
            for values, output in zip(arrays, outputs, strict=True):
                scaled = values[source, m, n] * weight.reshape((-1,) + (1,) * (values.ndim - 3))
                np.add.at(output, (destination, m, n), scaled)
        return np.asarray(ordered, dtype=np.int64), outputs

    def normalized_matrices(self, data, matrices):
        translations, outputs = self.normalized_operator_arrays(data, (matrices,))
        return translations, outputs[0]


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


def _integer_translations(translations):
    vectors = np.asarray(translations)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("translations must be an (NR,3) integer array")
    if vectors.dtype.kind not in "iu" and (
        vectors.dtype.kind != "f"
        or not np.isfinite(vectors).all()
        or not np.equal(vectors, np.round(vectors)).all()
    ):
        raise ValueError("translations must be an (NR,3) integer array")
    return vectors.astype(np.int64)


def periodic_dirs_from_table(translations) -> list[int]:
    """Axes along which any tabulated hop crosses a cell boundary.

    Apply this to the table as written by Wannier90. A Wigner-Seitz image
    correction moves a hop between centres more than half a cell apart through the
    neighbouring cell, so a corrected slab table also has translations along its
    vacuum axis although the model has no dispersion there.
    """
    vectors = _integer_translations(translations)
    return [axis for axis in range(3) if np.any(vectors[:, axis] != 0)]


def fold_vacuum_axes(translations, hopping, periodic_dirs: Sequence[int]):
    """Sum translation components along the non-periodic axes onto zero.

    ``hopping`` is ``(NR,NB,NB,...)`` on ``translations``; trailing axes such as
    stacked position components fold identically. Because ``exp(2*pi*i*k.R)``
    ignores a component whose momentum is zero, the folded table reproduces the
    original Fourier sum exactly wherever ``k`` vanishes on the folded axes, which
    is where a slab model is evaluated. Folded translations are returned in
    lexicographic order; a table without components to fold is returned unchanged.
    """
    vectors = _integer_translations(translations)
    matrices = np.asarray(hopping)
    if matrices.ndim < 3 or matrices.shape[0] != len(vectors) or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("hopping must have shape (NR,NB,NB,...) matching translations")
    periodic = [int(axis) for axis in periodic_dirs]
    if any(axis not in {0, 1, 2} for axis in periodic) or len(set(periodic)) != len(periodic):
        raise ValueError("periodic_dirs must contain distinct axes 0, 1 or 2")
    axes = [axis for axis in range(3) if axis not in periodic]
    if not axes or not np.any(vectors[:, axes]):
        return vectors, matrices
    folded = vectors.copy()
    folded[:, axes] = 0
    keys, inverse = np.unique(folded, axis=0, return_inverse=True)
    output = np.zeros((len(keys),) + matrices.shape[1:], dtype=np.complex128)
    np.add.at(output, np.asarray(inverse).reshape(-1), matrices)
    return keys, output


def _shift_arrays(shifts) -> tuple[np.ndarray, np.ndarray]:
    """Entry rows ``(R1,R2,R3,m,n,count)`` and their images, entry by entry."""
    if isinstance(shifts, PackedWignerSeitzShifts):
        return shifts.entries, shifts.images
    if isinstance(shifts, ComputedWignerSeitzShifts):
        counts = np.diff(shifts.offsets)
        pairs = np.arange(len(counts))
        rows, orbitals = np.divmod(pairs, shifts.nbnd**2)
        entries = np.column_stack(
            (
                shifts.translations[rows],
                orbitals // shifts.nbnd + 1,
                orbitals % shifts.nbnd + 1,
                counts,
            )
        )
        return entries.astype(np.int64), shifts.candidates[shifts.image_ids].astype(np.int64)
    if not isinstance(shifts, Mapping):
        raise ValueError("wsvec must be WignerSeitzData or a mapping of (R1,R2,R3,m,n) to image shifts")
    records: list[list[int]] = []
    blocks: list[np.ndarray] = []
    for key, values in shifts.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 5
            or any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in key)
        ):
            raise ValueError("wsvec keys must be (R1,R2,R3,m,n) integer tuples")
        block = np.asarray(values)
        if not block.size:
            continue
        if block.ndim != 2 or block.shape[1] != 3 or block.dtype.kind not in "iu":
            raise ValueError("wsvec images must be (count,3) integer shifts")
        records.append([*map(int, key), len(block)])
        blocks.append(block.astype(np.int64))
    return (
        np.asarray(records, dtype=np.int64).reshape(-1, 6),
        np.concatenate(blocks) if blocks else np.empty((0, 3), np.int64),
    )


def unfold_from_wsvec(data: WannierData, wsvec, operators=None):
    """Spread HR hops over the supercell images listed by a wsvec mapping.

    Each ``<m,0|H|n,R>/degeneracy(R)`` is shared evenly by its listed shifts ``T``
    and added at ``R+T``. Unlike the strict ``ws_distance`` normalization, a
    translation/orbital pair absent from the mapping keeps its hop at ``R``, and
    entries for other translations or orbitals are ignored. The ``use_ws_distance``
    flag is not consulted, since Wannier90 then lists only zero shifts; images
    already applied to the HR weights are refused. ``wsvec`` is
    :class:`~queer.io.wannier.WignerSeitzData` or a mapping with one-based orbital
    keys. ``operators`` defaults to ``data.hopping`` and may carry trailing axes.
    Returns lexicographically ordered translations and the image-summed operators.
    """
    from .wannier import WignerSeitzData

    shifts = wsvec
    if isinstance(wsvec, WignerSeitzData):
        if wsvec.ndegen_applied:
            raise ValueError("wsvec images were already applied to the HR weights")
        shifts = wsvec.shifts
    translations = _integer_translations(data.translations)
    nbnd = data.nbnd
    matrices = np.asarray(data.hopping if operators is None else operators)
    if matrices.shape[:3] != (len(translations), nbnd, nbnd):
        raise ValueError("operators must have shape (NR,NB,NB,...) on the Wannier translations")
    lookup = {tuple(int(value) for value in row): index for index, row in enumerate(translations)}
    if len(lookup) != len(translations):
        raise ValueError("Wannier translations must be distinct")
    entries, images = _shift_arrays(shifts)
    source = np.full(len(entries), -1, dtype=np.int64)
    if len(entries):
        vectors, inverse = np.unique(entries[:, :3], axis=0, return_inverse=True)
        found = [lookup.get(tuple(int(value) for value in row), -1) for row in vectors]
        source = np.asarray(found, dtype=np.int64)[np.asarray(inverse).reshape(-1)]
    m, n = entries[:, 3] - 1, entries[:, 4] - 1
    keep = (source >= 0) & (m >= 0) & (m < nbnd) & (n >= 0) & (n < nbnd)
    covered = np.zeros((len(translations), nbnd, nbnd), dtype=bool)
    covered[source[keep], m[keep], n[keep]] = True
    absent = np.argwhere(~covered)
    fill = np.column_stack(
        (translations[absent[:, 0]], absent[:, 1:] + 1, np.ones(len(absent), dtype=np.int64))
    )
    complete = PackedWignerSeitzShifts(
        np.concatenate((entries[keep], fill.reshape(-1, 6))),
        np.concatenate((images[np.repeat(keep, entries[:, 5])], np.zeros((len(absent), 3), np.int64))),
    )
    keys, outputs = complete.normalized_operator_arrays(data, (matrices,))
    return keys, outputs[0]
