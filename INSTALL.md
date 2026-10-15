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

Pushing a `v*` tag runs `.github/workflows/release.yml`. After the acceptance jobs,
the release job writes the tagged commit into `src/queer/_build.py` (`BUILD_SHA`),
runs the packaging tests and builds the wheel and sdist once. The publish job then
creates the GitHub Release for the tag with exactly those two files and a
`SHA256SUMS` computed from them; it fails rather than replace the assets of an
existing release. Pin consumers to a release asset URL with its `#sha256=` fragment
taken from `SHA256SUMS`.

Storage configuration and Slurm submission are documented in
[HPC operations](docs/hpc_usage.md). The historical relocation manifest records past
moves; it is not a current-file integrity manifest.
