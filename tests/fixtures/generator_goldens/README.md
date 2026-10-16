# Generator PythTB driver goldens

Each directory holds the Generator PythTB driver's results on one analytic case
of `tests/numerical/generator_golden_cases.py`: its `pythtb_results.json` and the
`Out/*.json` data files the parity comparator reads. The inputs are not stored;
the case module rebuilds them byte for byte, so their SHA-256 digests equal the
ones each document records in `provenance.inputs`.

| Case | Step | Model | What it pins |
|---|---|---|---|
| `haldane_bands` | `PYTHTB_BANDS` | Haldane, E_F = 0.8 eV (a metal) | bands, DOS and PDOS, Fermi-surface contours |
| `haldane_topology` | `PYTHTB_TOPOLOGY` | Haldane, E_F = 0 | Wilson loops, Chern number |
| `haldane_berry` | `PYTHTB_BERRY` | Haldane, E_F = 0 (bands mode) | curvature and metric maps, Kubo Chern number, quantum weight, path curvature |
| `haldane_berry_fermi` | `PYTHTB_BERRY` | Haldane, E_F = 0.8 eV (fermi mode) | Fermi-sea curvature and its integral |
| `haldane_slab` | `PYTHTB_SLAB` | Haldane ribbon, E_F = 0 | ragged levels, edge weights, edge crossings |
| `kane_mele_topology` | `PYTHTB_TOPOLOGY` | Kane-Mele with Rashba coupling | Kramers check, Z2 |

- **Producer.** The vendored driver `tests/oracles/generator_pythtb_core.py`
  (`core_sha256` 56fda912…) with PythTB 2.0.2, SciPy and Matplotlib, run by
  `scripts/validation/generate_generator_goldens.py` exactly as the Generator runs
  it. The provenance host name is replaced by `golden`.
- **Use.** `tests/numerical/test_generator_parity.py` runs QUEER on every case and
  requires `queer.compat.generator.parity.compare` to pass against these documents
  with every input digest equal.
- **Updating.** Regenerate them only after the vendored driver or a case changes,
  with the script above, and review the comparator's report on the new files.
