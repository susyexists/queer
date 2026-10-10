# Local Data

This directory holds local calculation inputs, including material `POSCAR`,
`OUTCAR`, `ef.txt`, and large Wannier Hamiltonian files. The files are kept on
disk for reproducibility but ignored by Git by default.

Native OpenMX `.scfout`, input sidecars and matching `.pao` files also live here.
Pinned optional FeCl₂/GaAs references and tracked checksums are described in
[references/openmx](references/openmx/README.md).

Use `docs/artifact_manifest.tsv` to map the old development paths to the new
organized locations.
