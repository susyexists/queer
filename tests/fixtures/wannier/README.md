# Synthetic numerical fixtures

These tiny inputs are committed test data, not experimental/material results.
`chain` has dispersion `0.3 cos(2πkx) - 0.2 sin(2πkx)` eV and non-unit HR
degeneracies. Its wsvec file supplies an alternative pair-distance interpolation
with translations ±1 and ±3. `two_band` uses complex off-diagonal hopping and
shuffled orbital rows to expose parser transposition. `fcc` reuses the analytic
chain with an Ag-labeled FCC primitive POSCAR solely for named-path/notebook
execution checks; its Hamiltonian does not describe silver.
