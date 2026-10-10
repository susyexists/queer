# OpenMX reference data

The two binaries are upstream test data pinned to PyATB commit
`60fc6b2acafac36b9c8334208326937fef3366c5`. They are ignored local reference
artifacts; `manifest.tsv` records their checksums and sizes. Fetch and verify with:

```bash
.venv/bin/python scripts/runs/openmx_benchmark.py --fetch --responses
```

The driver writes local provenance and a numerical report. Band acceptance uses
independent Cholesky reduction, normalized generalized residuals and overlap
orthonormality. Optional response acceptance requires positive optical absorption
and explicit mesh convergence, rather than agreement with PyATB output.
The default optical benchmark uses midpoint meshes, photon energies 1/2/4 eV,
eta=1 eV, T=1000 K and a 10% mesh-change tolerance. `--eta`,
`--temperature-kelvin`, `--grid`, `--levels` and `--tolerance` control these
explicit benchmark conditions. Inactive translation axes need one sample.
Checksummed mesh partials are retained beside the report for repeat runs.
Broadened material comparisons complement the analytical numerical test suite;
they do not establish convergence at other frequencies or broadenings.

The recorded native acceptance run passed for both inputs at those default
conditions. Errors compare with independent Cholesky diagonalization; response
changes use the full complex conductivity tensor, including off-diagonal
components:

| Input | Maximum band error (eV) | Final optical meshes | Relative tensor change |
| --- | ---: | --- | ---: |
| FeCl₂ | 1.01 × 10⁻¹² | 4³ → 8³ | 1.99% |
| GaAs SOC | 3.95 × 10⁻¹² | 8³ → 16³ | 3.93% |

Normalized generalized residuals were below 3 × 10⁻¹⁷ and overlap
orthonormality errors below 3.3 × 10⁻¹³. GaAs's 4³ → 8³ change was 26.64%, so
that mesh pair failed before the next refinement passed. The report is stored
locally in [results/openmx-benchmark.json](../../../results/openmx-benchmark.json).
FeCl₂ Hall conductivity at 300 K remained mesh-sensitive and is not a converged
material reference.

Source: [pinned upstream test data](https://github.com/susyexists/pyatb/tree/60fc6b2acafac36b9c8334208326937fef3366c5/tests/data).
Queer's independently written native implementation remains MIT licensed.
