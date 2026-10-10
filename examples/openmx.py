"""Calculate native bands and signed Mulliken projections from OpenMX v3."""

import argparse

import numpy as np

from queer import ExecutionOptions, OpenMXModel
from queer.observables import fat_band


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scfout", help="OpenMX binary v3 file")
    args = parser.parse_args(argv)
    model = OpenMXModel.from_scfout(args.scfout, execution=ExecutionOptions(batch_size=8))
    points = np.column_stack((np.linspace(0, 0.5, 21), np.zeros((21, 2))))
    result = fat_band(model, points)
    np.testing.assert_allclose(result["weights"].sum(axis=0), 1, atol=1e-10)
    print("bands (Nb, Nk):", result["energies"].shape)
    print("spin channels:", model.band_channels.tolist())
    print("Fermi reference (eV):", model.fermi_energy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
