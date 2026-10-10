# Independent OpenMX fixtures

`chain/chain.scfout` is a 2,364-byte synthetic v3 little-endian binary written
from the published record layout, with one spatial orbital and both collinear
spin channels. The independent writer lives in `tests/numerical/test_openmx.py`.
Its lattice is the identity in Å, S(k)=1+0.2 cos(2πkx), H↑(k)=1+0.4 cos(2πkx)
eV and H↓=H↑+1 eV. EF=0.2 eV. Relative position records exercise the translation
identity and the bra center is 0.3 Å. Labels `X` denote an analytical basis,
rather than a material. No GPL source or upstream data is embedded here.
