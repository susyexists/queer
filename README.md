# QUEER

QUEER is a Python library for native OpenMX and Wannier Hamiltonians, electronic
structure, geometry, topology, transport, optical and nonlinear responses,
spectral/ARPES calculations, momentum microscopy, and EPW self-energy workflows.
It also provides native crystal preprocessing, finite-displacement phonons,
force-callback dynamics and relaxation, trajectory selection, dataset preparation
and Hamiltonian graph postprocessing. It supports local, notebook, and Slurm
execution with durable numerical results.

## Install

Python 3.11+ is required. For development and HPC environment preparation:

```bash
uv sync --locked --extra plot --extra symmetry --extra notebook --group dev --python 3.13
```

For ordinary package installation, use `python -m pip install .` for numerical
work or `python -m pip install '.[plot,symmetry]'` for figures and named k-paths.
Optional extras are `plot`, `symmetry`, `animation`, `notebook`, `mace`, `hamgnn`,
`precision` (mpmath), `acceleration` (numba), and `all`.
Core calculations use no ASE, PyATB, SeekPath or Dans_Diffraction. Symmetry uses
optional spglib; trained model adapters use optional producer environments.
`requirements.txt` is a compatibility installer; `pyproject.toml` is authoritative.

## Reproducible workflows

The committed example configurations use tiny analytical fixtures, not personal
material datasets:

```bash
uv run --locked queer bands --config configs/examples/bands.toml --render
uv run --locked queer susceptibility --config configs/examples/susceptibility.toml
uv run --locked queer spectral --config configs/examples/spectral.toml
uv run --locked queer spectral --config configs/examples/spectral_arpes.toml
uv run --locked queer momentum --config configs/examples/momentum.toml
uv run --locked queer epw --config configs/examples/epw.toml
uv run --locked queer bands --config configs/examples/openmx_bands.toml --render
uv run --locked queer dos --config configs/examples/openmx_dos.toml --render
```

Each command prints its run directory. A run contains resolved configuration,
input checksums, version/resource/scheduler metadata, a status manifest, and
incremental NPZ arrays. Figures can be regenerated without calculation:

```bash
queer render /path/to/completed/run
queer momentum --config calculation.toml --run-dir /path/to/interrupted/run --resume
queer export /path/to/completed/run --profile generator
```

Configurations resolve relative input/output paths against the TOML directory.
CLI path overrides resolve against the working directory. Explicit configuration
roots override `QUEER_DATA_DIR` and `QUEER_RESULTS_DIR`; defaults are working-directory
`data/` and `results/`. Runtime storage never depends on the installation location.
Use `inputs.material` to select `data/materials/<name>` through `QUEER_DATA_DIR`.

## Library and compatibility

```python
from queer import ExecutionOptions, WannierModel

model = WannierModel.from_directory("my-material", ef=0.0, interpolation="standard")
bands = model.energies([[0.0, 0.0, 0.0]], execution=ExecutionOptions(batch_size=256))
```

K-points are `(Nk, 3)` fractional reciprocal coordinates; bands are `(Nb, Nk)` in eV.
Cartesian momenta use inverse angstroms. The existing `queer.model(...)` and legacy
module functions remain available during the 1.x migration. Scientific corrections
change historical values; see [migration notes](docs/migration.md).

Native OpenMX v3 binaries support both byte orders and nonmagnetic, collinear,
and SOC calculations. Sparse real-space operators feed a validated generalized
eigensolver, preserving overlap and position derivatives:

```python
from queer import OpenMXModel
from queer.observables import fat_band, optical_conductivity

model = OpenMXModel.from_scfout("calculation.scfout")
bands = model.energies([[0, 0, 0]])  # eV relative to the file chemical potential
projected = fat_band(model, [[0, 0, 0]])  # signed Mulliken weights
optical = optical_conductivity(model, [0.5, 1.0], grid=(8, 8, 8))
print(optical.dims["conductivity"], optical.units["conductivity"])
```

Both collinear channels remain in channel-major order; `energies(..., spin="down")`
selects a channel. New response chemical potentials are absolute eV and
temperatures use `temperature_kelvin`. New observables return `ObservableResult`
arrays with named dimensions, units and JSON metadata. See
[native workflows and capabilities](docs/openmx_workflows.md) for the inventory,
PAO/ionic/scattering inputs, convergence controls and scientific limits.

Native [quantum geometry and finite-field Hall workflows](docs/finite_field.md)
include pair-resolved metrics, Landau–Zener/DDP crossings, signed Hall currents,
quasistatic harmonics, model families, critical analysis, and dynamics validation.
Wannier HR/TB, centre XYZ, WOUT, and position operators share native interpolation;
OpenMX uses overlap-aware geometry and the explicit `physical_eq8` response profile.
Advanced OpenMX phases, precision patches, and character tracking remain gated.

```bash
queer crossing_scan --config configs/examples/finite_field_crossings.toml
queer hall_harmonics --config configs/examples/finite_field_harmonics.toml
queer paper_figure --config configs/examples/finite_field_paper_fig4d.toml --render
```

Physical conventions are the default. Named paper profiles record their separate
normalizations. All workflows use checksummed run artifacts, resumable checkpoints,
and artifact-only rendering; no HHH installation is required.

The equilibrium implementation uses independently written NumPy/SciPy methods and retains
Queer's MIT license. PyATB coverage was audited at commit
`60fc6b2acafac36b9c8334208326937fef3366c5`; scientific corrections are documented.

The new [generator scientific services](docs/generator_services.md) expose
`Structure`, `Frame`, `Evaluation`, `ForceProvider`, imported `BandData`, native
graph archives and pure generator compatibility adapters. Run the synthetic
examples with `queer nvt --config configs/examples/native_nvt.toml` or
`queer phonons --config configs/examples/native_phonons.toml`. Generator
integration remains a separate change.

## Notebooks and Slurm

The maintained notebook is `notebooks/active/Ag_primitive_single.ipynb`. Install the
notebook extra and select the prepared environment's kernel. Set
`QUEER_NOTEBOOK_CONFIG` to a TOML configuration for any supported workflow. The
notebook delegates calculation, recovery, and rendering to the workflow APIs; it
keeps scientific parameters in configuration and saves the same run artifacts.
Archives retain historical notebooks and outputs and are excluded from execution.

Prepare the environment once, customize `configs/hpc/default.toml`, and submit:

```bash
.venv/bin/python scripts/hpc/submit_sweep.py --config configs/examples/momentum.toml \
  --profile configs/hpc/default.toml --dry-run
```

Remove `--dry-run` to submit. See [HPC operations](docs/hpc_usage.md) for site resources,
array failures, recovery, and real-cluster acceptance.

## Verification and layout

```bash
uv run --locked pytest
uv run --locked ruff check src tests scripts/runs scripts/hpc examples benchmarks
uv run --locked mypy
```

The tests include synthetic numerical references, workflow recovery, notebook
execution, scheduler adapters, and isolated wheel/sdist installation. Optional personal
material smoke checks may skip; numerical acceptance uses committed fixtures.

See [repository boundaries](docs/repo_layout.md), [scientific conventions](docs/scientific_conventions.md),
and [reference benchmarks](benchmarks/README.md). Large inputs and generated results
remain ignored; tracked dataset/run manifests provide provenance.
