# PyATB run fixtures

Output texts of PyATB runs that the readers and the parity comparator of the
OpenMX steps (`queer.compat.generator.pyatb_files`, `parity_pyatb`) are tested
against. Only PyATB's text products are stored: the scfout they read is
third-party data and is not committed. No PyATB source is included.

| Directory | Run | What it pins |
|---|---|---|
| `a_fhs_scan` | `AHC`, FHS engine, 201 energies around E_F = -2.7 eV | the FHS table layout and its energy grid |
| `b_direct_pyatb` | `AHC`, direct (method 0), `hall_convention pyatb`, Grid 100 x 100 x 1, refinement off | the one-line layout with PyATB's native Hall sign |
| `c_direct_openmx` | the same with `hall_convention openmx` | the physical Hall sign |
| `h_chern_fermi` | `CHERN_NUMBER`, direct, Fermi occupation, Grid 100 x 100 x 1 | the one-line Chern layout at full precision |
| `adaptive_log` | `AHC`, adaptive integral (an earlier Generator job) | `integraton.log` and its last-generation error |
| `a_fhs_scan_queer.json` | QUEER 1.5.0's `QUEER_AHC` run of the `a_fhs_scan` Input on the same scfout | QUEER's sigma_xy per energy (PyATB's Cartesian construction and physical), the energies with a varying occupied rank, its emulations of PyATB's FHS links and of its own midpoint links, and its Fermi-sea check at the band energies (its direct Fermi-sea sum on the same nodes and its FHS on the halved grid) |
| `a_fhs_scan_link_defects.json` | the same run with a defect in QUEER's FHS link term, the `-i dk.A` term of its midpoint links scaled by -1 and by 1.5 in its FHS and in its own-link emulation alike (the connection data, and so its Fermi-sea sum, intact) | the defective sigma_xy, own-link values and halved-grid FHS, which the comparator's Fermi-sea guard must refuse |

The first four are the calibration runs of PyATB build 60fc6b2a (16 October
2026, 8 MPI processes) on a kagome Chern-insulator scfout with spin-orbit
coupling (`test.scfout`, SHA-256 `93097304…`, 101,108 bytes); each `Input` names it
as `../test.scfout`, E_F = -2.35 eV sits on its sigma_xy plateau. Each directory
holds PyATB's `Input`, `Out/input.json`, `Out/running.log` and its product, as
written. `adaptive_log` holds `Out/AHC/integraton.log` and `Out/AHC/ahc.dat` of
an earlier Generator PyATB job (adaptive integral from one initial cell,
`hall_convention pyatb`).
