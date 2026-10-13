# Reference workloads

Run `python benchmarks/reference.py --output results/benchmarks/reference.json` in the locked environment. The default analytic chain measures 1,000 and 100,000 k-points and verifies agreement with serial evaluation. Reports include the input SHA256, algorithm version, runtime/package versions, execution limits, wall time, and peak RSS. Compare identical machine/runtime/input configurations when evaluating a change; wall times are evidence, not portable pass/fail thresholds.

`--material-dir`, `--ef`, `--points`, `--batch-size`, `--workers`, and `--memory-mb` support opt-in real-material/HPC runs. Peak RSS is process-wide, includes returned arrays, and does not aggregate process-worker memory. Use scheduler accounting for aggregate real-cluster memory. Run separate processes when comparing peak memory across batch sizes.

`scripts/runs/openmx_benchmark.py --fetch --responses` verifies pinned FeCl₂ and
GaAs SOC inputs in ignored `data/references/openmx/` storage. The tracked manifest
pins SHA256, byte counts and upstream commit. Bands are checked by independent
Cholesky reduction with normalized residual/orthonormality limits; optical
responses must pass absorption positivity and requested mesh convergence.
The driver saves a JSON report under `results/`. Material references supplement
the analytical tests and do not replace their physical acceptance criteria.

Replay finite-field parity against a local HHH checkout with the development-only
comparison driver. HHH is loaded solely by this script and remains outside
QUEER's package dependencies:

```bash
.venv/bin/python benchmarks/compare_hhh.py --hhh-root ../HHH
.venv/bin/python benchmarks/compare_hhh.py --hhh-root ../HHH \
  --skip-analytical \
  --real-data ../HHH/results/generator_wannier/job-48ec1fe7 \
  --real-data ../HHH/results/generator_wannier/job-5a7b6ced
```

The default report is `results/benchmarks/hhh-parity.json`; use `--output` to retain
separate analytical and material reports. Real inputs must contain an HR file,
WS-vector file, centres XYZ, and WOUT; the driver recognizes the generator's
`hpc/` layout and reads the inputs independently with both libraries. The default
occupied-band count is 145, with four workers and an 8192 MiB working-memory
budget; `--n-occ`, `--workers`, and `--memory-mb` override these limits.
The material replay uses the TBA HR/centres/WS model and rejects an automatically
discovered TB sidecar, which would add position operators and change its gauge.

Reports retain source/input hashes, requested and effective settings, numerical
values, diagnostics, Fourier convergence, timing, and errors for each quantity
and harmonic. Initial and final source fingerprints identify edits made during
the replay; workflow changes are labeled separately, and changes to numerical
sources invalidate acceptance. Acceptance is 1e-7 of each PAPER peak, 1e-5 for PHYSICAL, 1e-8 for
material geometry, and 1e-4 for each material current/harmonic peak, with explicit
absolute floors for numerical zeros. Failed comparisons or unconverged Fourier
projections return a nonzero exit status. Progress is saved after each stage so
an interrupted or failing material run leaves a reviewable partial report.
Material geometry additionally requires elementwise agreement at the same
1e-8 relative tolerance and 1e-12 absolute floor, including small off-diagonal
metric components. It is also checked at identical coordinates spanning both
libraries' central minima. Minimum locations and geometry evaluated at each
library's own minimum are reported separately, so coordinate rounding cannot
be mistaken for an error in the geometry operators. The matched-coordinate
check uses energy gaps and records provider energy offsets; the fixed-point
check compares absolute energies from the independent raw models.
