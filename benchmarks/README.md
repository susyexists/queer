# Reference workloads

Run `python benchmarks/reference.py --output results/benchmarks/reference.json` in the locked environment. The default analytic chain measures 1,000 and 100,000 k-points and verifies agreement with serial evaluation. Reports include the input SHA256, algorithm version, runtime/package versions, execution limits, wall time, and peak RSS. Compare identical machine/runtime/input configurations when evaluating a change; wall times are evidence, not portable pass/fail thresholds.

`--material-dir`, `--ef`, `--points`, `--batch-size`, `--workers`, and `--memory-mb` support opt-in real-material/HPC runs. Peak RSS is process-wide, includes returned arrays, and does not aggregate process-worker memory. Use scheduler accounting for aggregate real-cluster memory. Run separate processes when comparing peak memory across batch sizes.
