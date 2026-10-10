# Native structures and crystal geometry

QUEER provides its own structure records, periodic geometry, format readers and
writers. These operations run without ASE, SeekPath or Dans_Diffraction. Space
group detection and primitive-cell finding use the optional Spglib primitive:

```sh
python -m pip install 'queer[symmetry]'
```

## Shared records and units

`queer.structures.Structure` accepts array-like inputs, validates them, and owns
copies of its arrays. `copy(**changes)` creates independent geometry, per-atom
arrays and metadata. Records are mutable; a `Frame` holds the supplied structure.

| Field | Shape and convention |
| --- | --- |
| `cell` | `(3, 3)`, lattice vectors are rows, Å |
| `positions` | `(N, 3)`, Cartesian Å |
| `atomic_numbers` | `(N,)`, integers 1–118 |
| `pbc` | Three booleans; defaults to full periodicity |
| `masses` | `(N,)`, positive amu; defaults to elemental masses |
| `magnetic_moments` | Optional `(N, 3)`, Cartesian axial vectors, μB |
| `fixed_coordinates` | Optional `(N, 3)` booleans; `True` means fixed |
| `arrays` | Named per-atom arrays with `N` rows |
| `metadata` | General structure information, including constraint basis |

`fractional_positions` satisfies `positions = fractional_positions @ cell`.
Periodic cells must be finite and nonsingular. Coordinate constraints default
to Cartesian axes. Set `metadata['constraint_basis'] = 'fractional'` when the
flags refer to lattice axes; allowed displacements then span the unfixed cell
rows. This distinction matters for skew cells.

`Evaluation(energy_eV, forces_eV_A, stress_eV_A3=None)` stores a scalar energy,
`(N, 3)` forces and an optional `(3, 3)` stress tensor. The evaluator defines the
stress sign; format conversion does not change it. `Frame` adds optional
Cartesian velocities in Å/fs, an integer step, time in fs, an evaluation and
frame metadata. A `ForceProvider` implements `evaluate(structure) -> Evaluation`.

```python
import numpy as np
from queer.structures import Evaluation, Frame, Structure

atoms = Structure(
    cell=np.diag([4.0, 4.0, 5.0]),
    positions=[[0, 0, 0], [1, 1, 1]],
    atomic_numbers=[8, 1],
)
frame = Frame(
    atoms,
    velocities=np.zeros((2, 3)),
    step=0,
    time_fs=0.0,
    evaluation=Evaluation(-2.0, np.zeros((2, 3)), np.zeros((3, 3))),
)
assert atoms.symbols == ['O', 'H']
np.testing.assert_allclose(atoms.fractional_positions @ atoms.cell, atoms.positions)
```

## Formats and atom order

The native format functions live in `queer.io.atomistic`. `read_structure(path)`
requires one structure. `read_frames(path)` reads a trajectory. Text APIs such as
`parse_poscar(text)`, `parse_cif(text)` and `read_extxyz(text)` accept file contents.
`read_trajectory` accepts ULM bytes or a path. Supply `format='ulm'` when a binary
trajectory has a nonstandard extension.

| Format | Retained information and boundaries |
| --- | --- |
| POSCAR/CONTCAR | VASP 4/5 species, scalar/three-component/negative-volume scaling, direct or Cartesian coordinates, selective flags. Magnetic moments, custom masses and evaluations require another artifact. VASP 4 needs species in its comment or `species=`. |
| CIF/MCIF | Cell metrics, fractional or Cartesian atom sites, occupancy and magnetic moments. The writer emits full sites in P1 and uses the conventional Cartesian orientation derived from the cell metrics. It does not preserve the original asymmetric unit, global cell rotation or arbitrary per-atom fields. |
| Extended XYZ | Multiple frames, geometry, PBC, masses, moments, velocities, QUEER constraint fields, scalar/vector per-atom arrays, metadata and evaluations. Another reader must interpret QUEER's constraint fields to apply them dynamically. |
| ASE ULM trajectory | Native v1–3 reader and v3 writer for geometry, PBC, masses, moments, momenta/velocities, evaluations and supported coordinate constraints. A QUEER metadata extension preserves custom per-atom arrays. No pickle is read. |

POSCAR selective flags always refer to fractional directions, including when
coordinates are written as Cartesian. The writer converts a Cartesian allowed
subspace to fractional flags when representable. If a skew cell cannot express
that subspace with three flags, it raises an error. A trajectory or extXYZ can
retain the original Cartesian constraints.

`poscar_text(structure, sort=True)` returns `(text, order)`. Sorting is stable by
atomic number, and `order[new_index]` identifies the original atom. Apply the
same permutation to magnetic moments and other separately exported atom data.
The Generator structure profile stores this mapping as `poscar_atom_order`.
`write_structure` preserves atom order by default.

```python
import numpy as np
from tempfile import TemporaryDirectory
from pathlib import Path
from queer.structures import Evaluation, Frame, Structure
from queer.io.atomistic import (
    parse_poscar, poscar_text, read_extxyz, read_structure,
    read_trajectory, write_extxyz, write_structure, write_trajectory,
)

atoms = Structure(np.eye(3) * 4, [[0, 0, 0], [1, 1, 1]], [8, 1])
text, order = poscar_text(atoms)
assert order.tolist() == [1, 0]
np.testing.assert_allclose(parse_poscar(text).positions, atoms.positions[order])
frame = Frame(atoms, evaluation=Evaluation(-2, np.zeros((2, 3))))
np.testing.assert_allclose(read_extxyz(write_extxyz(frame))[0].structure.cell, atoms.cell)
np.testing.assert_allclose(read_trajectory(write_trajectory([frame]))[0].structure.cell, atoms.cell)
with TemporaryDirectory() as directory:
    path = Path(directory) / 'POSCAR'
    write_structure(path, atoms)
    np.testing.assert_allclose(read_structure(path).positions, atoms.positions)
```

Standard extXYZ `stress` and `virial` tensors use column-major serialization;
generic tensors such as MACE's `REF_stress` use `_JSON` to preserve their shape.
The ULM writer uses six-component symmetric stress storage and rejects
asymmetric tensors beyond `rtol=1e-10, atol=1e-12`. Velocity conversion between
Å/fs and ULM momenta is performed by the reader and writer.

ULM producer metadata identifies the actual QUEER package version; it does not
claim an ASE producer version. Custom per-atom arrays use the reserved
`_queer_atom_arrays` info key with a versioned, typed binary encoding inside JSON.
Native roundtrips preserve numeric, complex, boolean and string arrays. Regular
ASE readers retain this extension as opaque info, so ASE trajectory rewrites can
preserve it. Object and structured NumPy dtypes are rejected, and caller metadata
cannot override the reserved key. External consumers needing arrays as native
properties should use extXYZ where its scalar/vector property grammar applies.

## Periodic neighbors and transformations

`neighbor_list(structure, cutoff)` returns directed `(i, j, image, distance,
vector)` arrays. The identity is
`vector = positions[j] + image @ cell - positions[i]`. A scalar cutoff is a
pair distance; a length-`N` array supplies additive per-atom radii. The home-cell
self pair is excluded by default, while periodic images of the same atom are
included. Images are enumerated using bounds valid for skew cells, rather than
assuming component-wise fractional rounding gives a nearest image.

`minimum_image_vectors` performs a bounded closest-image search.
`minimum_distance` includes physical separations to same-atom periodic images
by default; pass `include_self_images=False` for distinct atom indices only.
`compute_bonds` supplies covalent-radius candidates with image offsets; those
candidates are geometric connections, not electronic bond orders.

`make_supercell` accepts any nonsingular integer `(3, 3)` matrix and returns
`new_cell = matrix @ old_cell`. It replicates masses, moments, constraints and
per-atom arrays. Atom selection, supercells and primitive reduction retain
original-site mappings in `metadata['source_indices']`. Fractional constraints
are remapped across cell basis changes to preserve their Cartesian allowed
subspaces. A basis that cannot represent those subspaces using three coordinate
flags raises a descriptive error. Vacuum extension also preserves those physical
subspaces or raises this error. Fully fixed and fully free sites remain valid.

`strain(structure, deformation)` uses a Cartesian deformation gradient:
`new_cell = cell @ deformation.T`. Fractional constraint axes deform with the
lattice intentionally. `add_vacuum` increases the cell height normal to the other
two cell vectors and preserves relative Cartesian atom positions; its argument
is added height, not a target empty gap. Other native operations are `distort`,
`substitute` and `remove_atoms`.

```python
import numpy as np
from queer.structures import Structure
from queer.geometry.structures import (
    add_vacuum, apply_transforms, expand_variants,
    make_supercell, minimum_distance, neighbor_list,
)

atoms = Structure([[3, 0, 0], [1, 4, 0], [0.2, 0.4, 5]], [[0, 0, 0], [1, 1, 1]], [8, 1])
enlarged = make_supercell(atoms, [[2, 1, 0], [0, 1, 0], [0, 0, 1]])
assert enlarged.natoms == 4
i, j, images, distances, vectors = neighbor_list(enlarged, 2.0)
np.testing.assert_allclose(vectors, enlarged.positions[j] + images @ enlarged.cell - enlarged.positions[i])
slab = add_vacuum(atoms, 3.0)
np.testing.assert_allclose(slab.positions[1] - slab.positions[0], atoms.positions[1] - atoms.positions[0])
recipes = expand_variants([
    {'type': 'gaussianDistortion', 'params': {
        'count': 3, 'seed': 12, 'sigmaMin': 0.02, 'sigmaMax': 0.03,
    }},
])
variants = [apply_transforms(atoms, recipe) for recipe in recipes]
np.testing.assert_allclose(variants[0].positions, atoms.positions)  # pristine variant
assert minimum_distance(enlarged) > 0
```

Recipe expansion combines strain scans and random/Gaussian variant counts as a
Cartesian product. Seeds and variant indices determine reproducible draws.
Gaussian expansion includes a pristine first variant by default. Native `layer`
recipes preserve Cartesian slab distances. The explicit `legacy=True` option on
`apply_transforms` retains the historical Generator layer rescaling and its
distinct-atom minimum-distance guard; new calculations should use native behavior.

## Magnetic CIF and crystallographic paths

The MCIF reader expands spatial magnetic operations and magnetic centering
operations. Crystal-axis moments are converted using unit-length cell vectors,
then transformed as Cartesian axial vectors: `M' = s * det(R_cart) * R_cart @ M`,
where `s` is the operation's ±1 time-reversal sign. Coincident images must agree
on their moments. CIF operation expressions use a restricted affine grammar;
arbitrary Python and nonlinear coordinate expressions are rejected.

The default fractional site-deduplication tolerance is `1e-4`, allowing rounded
crystallographic coordinates such as one-third; `parse_cif` accepts a positive
`fractional_tolerance=` override. A missing explicit operation loop can be
resolved through a declared crystallographic space group using optional Spglib.
Without a group declaration the reader uses P1.

```python
import numpy as np
from queer.structures import Structure
from queer.io.atomistic import parse_cif, write_cif

atoms = Structure(
    [[3, 0, 0], [-1.5, 1.5 * np.sqrt(3), 0], [0, 0, 5]],
    [[0, 0, 0], [0, 0, 2.5]],
    [26, 26],
    magnetic_moments=[[0, np.sqrt(3), 0], [0, -np.sqrt(3), 0]],
)
restored = parse_cif(write_cif(atoms))
np.testing.assert_allclose(restored.magnetic_moments, atoms.magnetic_moments, atol=1e-12)
```

`primitive_structure` treats moments, masses, constraints and custom per-atom
fields as site identity. It does not collapse an antiferromagnetic sublattice
whose moments disagree. HPKOT path detection instead uses the spatial chemical
structure; it does not determine magnetic space groups. Choose
`with_time_reversal` according to the actual Hamiltonian. Setting it to `False`
adds the negative-k path arms when spatial inversion is absent.

`compute_band_path` implements all 29 HPKOT Bravais variants natively, with
Spglib supplying symmetry and standard-cell information. It returns labeled
segment endpoints and a `points_per_segment` sampling recommendation, not a
sampled Hamiltonian band array. The reciprocal convention is
`B = 2*pi*inv(A).T`, `k_cart = k_frac @ B`.

The default `original_basis=True` expresses endpoints in the supplied cell's
reciprocal basis, including its rotation and supercell transformation. Coordinates
may fall outside a unit fractional cube. The primitive-zone route remains a
primitive-zone route; it is not a path around the folded supercell zone.
`original_basis=False` instead returns coordinates for `primitive_lattice`.
Inspect `coordinate_basis`, `primitive_point_coords`, `rotation_matrix` and
`volume_original_wrt_prim` before transferring coordinates between models.

```python
import numpy as np
from queer.structures import Structure
from queer.geometry.lattice import reciprocal_lattice
from queer.geometry.paths import named_k_path_from_structure
from queer.geometry.symmetry import compute_band_path

atoms = Structure([[0, 2, 2], [2, 0, 2], [2, 2, 0]], [[0, 0, 0]], [47])
route = compute_band_path(atoms, points_per_segment=20)
assert route['coordinate_basis'] == 'original'
k_cart = np.asarray(route['point_coords']['X']) @ reciprocal_lattice(atoms.cell)
path = named_k_path_from_structure('GAMMA-X', 20, atoms)
assert path.path.shape == (21, 3)  # 20 intervals and both endpoints
```

Only the published reciprocal-point, path and formula tables from SeekPath
2.2.2 are reused. Their MIT notice and provenance ship in
[`resources/symmetry`](../src/queer/resources/symmetry/PROVENANCE.txt), including
the [full MIT license](../src/queer/resources/symmetry/LICENSE.txt). The tables
credit Yoyo Hinuma and Hinuma et al., *Computational Materials Science* 128
(2017), 140–184, [doi:10.1016/j.commatsci.2016.10.015](https://doi.org/10.1016/j.commatsci.2016.10.015).
Classification, formula evaluation, basis conversion, atom mapping and geometry
are QUEER implementations. No SeekPath source routines or example structures
are shipped.

## Validation and explicit limits

The development reference environment used ASE **3.29.0**, SeekPath **2.2.2**
and Spglib **2.8.0**. These are observed validation versions, not core dependency
requirements. ASE and SeekPath serve as optional independent references.

| Check | Acceptance |
| --- | --- |
| Bidirectional ASE ULM/extXYZ geometry, velocities, evaluations and REF tensor fields | `numpy.testing.assert_allclose` default `rtol=1e-7, atol=0` on roundtrip arrays; exact energy/step assertions |
| Magnetic crystal-axis conversion, axial reflection and CIF roundtrip | Absolute tolerance `1e-12` |
| Rotated/nonprimitive reciprocal paths | Absolute tolerance `1e-12` |
| Brillouin-zone volume and face planarity | Relative volume tolerance `1e-9`; absolute face tolerance `1e-8` |
| All 29 local SeekPath variant references | Development oracle agreed on classification and point coordinates, maximum observed difference approximately `5e-16`; the tracked regressions use independent synthetic cells |

Regression coverage is in
[`test_atomistic_native.py`](../tests/numerical/test_atomistic_native.py) and
[`test_native_structure_workflow.py`](../tests/integration/test_native_structure_workflow.py).
The workflow test prohibits ASE, SeekPath and Dans_Diffraction imports while
checking numerical artifacts, atom mappings, Generator export checksums and
resume integrity. Optional reference tests skip when their reference package is
absent.

- Incommensurate superspace MCIF operations cannot define a finite periodic cell
  and raise an explicit error. The reader selects the first atom-bearing CIF
  data block; it does not merge different structures.
- The shared constraint record supports coordinate masks in one Cartesian or
  fractional basis. ULM imports support `FixAtoms`, `FixCartesian` and
  `FixScaled`; other constraint types and mixed Cartesian/fractional constraints
  raise errors. General nonlinear or coupled-atom constraints need another
  representation.
- Symmetry and primitive-cell detection require Spglib. HPKOT is the supported
  path convention; magnetic group detection and automatic selection of a
  magnetic reciprocal-cell path are outside this API.
- Native format support is the formats described above, rather than ASE's wider
  format plugin collection. Single-structure reads reject multiple-frame inputs;
  use `read_frames` for those.
