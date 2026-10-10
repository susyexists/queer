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
