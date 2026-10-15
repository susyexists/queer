"""Optional actual MACE architecture smoke, initialized locally without downloads."""

import hashlib
import importlib

import numpy as np
import pytest

from queer.adapters.mace import MACEForceProvider
from queer.structures import Structure


def test_untrained_mace_architecture_forces_stress_and_checkpoint_roundtrip(tmp_path):
    torch = pytest.importorskip("torch", reason="optional Torch producer environment")
    try:
        modules = importlib.import_module("mace.modules")
    except ImportError as error:
        pytest.skip(f"optional MACE model classes unavailable: {error}")
    o3 = pytest.importorskip("e3nn.o3", reason="optional MACE equivariant model classes")
    required = ("MACE", "RealAgnosticInteractionBlock", "RealAgnosticResidualInteractionBlock")
    if not all(hasattr(modules, name) for name in required):
        pytest.skip("optional MACE architecture classes unavailable")

    # MACE/e3nn initialize buffers using Torch's default dtype. Restore the
    # caller's global setting and RNG after this locally initialized fixture.
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(5)
            model = modules.MACE(
                r_max=2.5,
                num_bessel=4,
                num_polynomial_cutoff=5,
                max_ell=1,
                interaction_cls=modules.RealAgnosticResidualInteractionBlock,
                interaction_cls_first=modules.RealAgnosticInteractionBlock,
                num_interactions=2,
                num_elements=2,
                hidden_irreps=o3.Irreps("8x0e + 8x1o"),
                MLP_irreps=o3.Irreps("8x0e"),
                atomic_energies=np.zeros(2),
                avg_num_neighbors=2.0,
                atomic_numbers=[1, 8],
                correlation=2,
                gate=torch.nn.functional.silu,
            ).double()
    finally:
        torch.set_default_dtype(original_dtype)

    structure = Structure(np.eye(3) * 6, [[0.5, 0.5, 0.5], [1.2, 0.7, 0.6]], [1, 8])
    provider = MACEForceProvider(model=model, device="cpu", compute_stress=True)
    evaluated = provider.evaluate(structure)
    assert np.isfinite(evaluated.energy_eV)
    assert np.isfinite(evaluated.forces_eV_A).all()
    assert evaluated.stress_eV_A3.shape == (3, 3)
    assert np.isfinite(evaluated.stress_eV_A3).all()
    np.testing.assert_allclose(evaluated.forces_eV_A.sum(axis=0), 0, atol=1e-12, rtol=0)

    # Independent Cartesian energy derivatives test the actual network's
    # autograd forces, rather than mirroring the adapter's graph construction.
    delta_A = 1e-5
    for atom, axis in ((0, 0), (0, 1), (1, 2)):
        plus = structure.positions.copy()
        minus = structure.positions.copy()
        plus[atom, axis] += delta_A
        minus[atom, axis] -= delta_A
        finite_force = -(
            provider.evaluate(structure.copy(positions=plus)).energy_eV
            - provider.evaluate(structure.copy(positions=minus)).energy_eV
        ) / (2 * delta_A)
        np.testing.assert_allclose(finite_force, evaluated.forces_eV_A[atom, axis], atol=1e-7, rtol=1e-6)

    checkpoint = tmp_path / "untrained-mace.model"
    torch.save(model, checkpoint)
    restored_provider = MACEForceProvider(checkpoint=checkpoint, device="cpu", compute_stress=True)
    restored = restored_provider.evaluate(structure)
    assert restored.energy_eV == evaluated.energy_eV
    np.testing.assert_array_equal(restored.forces_eV_A, evaluated.forces_eV_A)
    np.testing.assert_array_equal(restored.stress_eV_A3, evaluated.stress_eV_A3)
    assert (
        restored_provider.provenance["checkpoint_sha256"]
        == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
