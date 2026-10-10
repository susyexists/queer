"""Native force callbacks, segmented phonons and exactly restartable dynamics."""

from pathlib import Path

import numpy as np

from queer import Evaluation, Structure, phonon_bands, relax_fire, run_nvt
from queer.atomistic import nvt_metrics


class Harmonic:
    """A synthetic onsite potential for algorithm examples, using eV and Å."""

    def __init__(self, reference):
        self.reference = np.asarray(reference).copy()

    def evaluate(self, structure):
        displacement = structure.positions - self.reference
        return Evaluation(float(np.sum(displacement**2)), -2 * displacement)


def main():
    structure = Structure(np.eye(3) * 5, [[0.2, 0, 0], [1.2, 0, 0]], [1, 1])
    provider = Harmonic([[0, 0, 0], [1, 0, 0]])
    relaxed = relax_fire(structure, provider, fmax_eV_A=0.001)
    print("relaxed maximum force:", relaxed.summary["fmax_eVA"])
    first = run_nvt(
        relaxed.final_frame.structure, provider, n_steps=10, seed=42, log_interval=2, traj_interval=5
    )
    continued = run_nvt(
        relaxed.final_frame.structure,
        provider,
        n_steps=20,
        seed=42,
        log_interval=2,
        traj_interval=5,
        state=first.state,
    )
    print("stability metrics:", nvt_metrics(continued))
    bands = phonon_bands(
        relaxed.final_frame.structure,
        provider,
        supercell=(1, 1, 1),
        qpoints=[[0, 0, 0], [0.5, 0, 0]],
        frederiksen=False,
        acoustic=False,
    )
    print("onsite harmonic frequencies in THz:", bands.arrays["frequencies_THz"][:, 0])
    from queer.io.atomistic import write_trajectory

    target = Path("results/native-example.traj")
    target.parent.mkdir(parents=True, exist_ok=True)
    write_trajectory(target, continued.frames)
    print(target)


if __name__ == "__main__":
    main()
