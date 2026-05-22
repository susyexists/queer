# Repository Layout

The repository is organized so library code, local inputs, generated results,
and exploratory notebooks do not compete for the top level.

- `src/queer/`: importable Python package.
- `notebooks/active/`: notebooks used for current work.
- `notebooks/archive/`: preserved development notebooks grouped by topic or
  material.
- `data/`: local input artifacts. Git ignores the actual data files.
- `results/`: generated output artifacts. Git ignores generated output files.
- `scripts/runs/`: Python entrypoints that can be run locally or from Slurm.
- `scripts/hpc/`: HPC launchers.
- `scripts/archive/`: older scripts kept for provenance.
- `docs/artifact_manifest.tsv`: relocation inventory.
- `tests/`: smoke tests for pathing and calculation primitives.

Use `queer.paths.data_path()` and `queer.paths.results_path()` in new code so
workflows can move between local workstations and HPC scratch storage.
