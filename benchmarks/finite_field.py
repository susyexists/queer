"""Measure finite-field preparation, signed-field evaluation, and Fourier projection."""

import argparse
import json
import time

import numpy as np

from queer.execution import ExecutionOptions
from queer.finite_field import FourBandModel, FourBandParams, PreparedHallResponse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--x", type=float, default=0.5)
    args = parser.parse_args()
    model = FourBandModel(FourBandParams().at(args.x))
    prepared = PreparedHallResponse(model, execution=ExecutionOptions(workers=args.workers))
    start = time.perf_counter()
    prepared.prepare()
    preparation = time.perf_counter() - start
    start = time.perf_counter()
    prepared.hall(np.linspace(-5e4, 5e4, 1001))
    fields = time.perf_counter() - start
    start = time.perf_counter()
    result = prepared.harmonics(np.linspace(1e3, 5e4, 21), n_max=9)
    print(
        json.dumps(
            {
                "preparation_seconds": preparation,
                "field_scan_seconds": fields,
                "fourier_seconds": time.perf_counter() - start,
                "fourier_converged": result.metadata["converged"],
                "diagnostics": prepared.describe(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
