"""Example: calculate and plot an electron band structure."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import queer
from queer.paths import data_path, results_path


def main():
    material = "ag_primitive"
    model = queer.model(
        path=data_path("materials", material),
        hr="wannier90_hr.dat",
        ef=8.310342,
        poscar="POSCAR",
        num_core=1,
    )

    output = results_path("bands", "ag_primitive_band_structure.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    model.plot_band_path("GAMMA-X-W-K-GAMMA-L", n_points=200, save=output)

    print(f"Saved band structure to {output}")


if __name__ == "__main__":
    main()
