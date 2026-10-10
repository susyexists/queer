# Installation and environments

Use Python 3.11–3.14. The reference development/HPC environment uses Python 3.13.

```bash
uv sync --locked --all-extras --group dev --python 3.13
.venv/bin/python -m ipykernel install --user --name queer-dev --display-name "Python (QUEER)"
```

`uv.lock` pins deployment/development resolution. Use `--locked` to reject unexpected
lock changes. Resolve deliberate dependency updates with `uv lock`, then rerun all
acceptance gates. Prepare environments before submitting jobs; workers never install
dependencies or update the lockfile.

For pip users:

```bash
python -m pip install -e '.[all]'
```

Core numerical imports require NumPy, SciPy, joblib, and threadpoolctl. Extras add
plotting/tabular compatibility (`plot`), SeekPath (`symmetry`), animation codecs
(`animation`), or Jupyter (`notebook`). Test/lint/build tools belong to the development
dependency group. Distribution metadata uses the repository's MIT license.

Build and inspect distributions with `python -m build` and `python -m twine check dist/*`.
The packaging test installs the wheel into a fresh environment outside the checkout
and checks numerical execution with only core dependencies.

Storage configuration and Slurm submission are documented in
[HPC operations](docs/hpc_usage.md). The historical relocation manifest records past
moves; it is not a current-file integrity manifest.
