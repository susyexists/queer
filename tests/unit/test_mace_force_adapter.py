"""Direct tensor-model adapter checks, without MACE/ASE calculator delegation."""

import hashlib
import importlib.util
import subprocess
import sys

import numpy as np
import pytest

from queer.adapters.mace import MACEForceProvider
from queer.structures import Structure


def test_optional_adapter_import_does_not_import_torch_or_ase():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import queer.adapters.mace; assert 'torch' not in sys.modules; assert 'ase' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="optional Torch producer environment")
def test_tensor_graph_forward_uses_native_neighbors_and_preserves_force_autograd():
    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("atomic_numbers", torch.tensor([1, 8]))
            self.register_buffer("r_max", torch.tensor(1.1))

        def forward(self, data, compute_force=True, compute_stress=False, training=False):
            assert len(data["ptr"]) == 2
            energy = 0.5 * (data["positions"] ** 2).sum()
            forces = -torch.autograd.grad(energy, data["positions"])[0]
            stress = torch.eye(3).unsqueeze(0) if compute_stress else None
            return {"energy": energy.reshape(1), "forces": forces, "stress": stress}

    structure = Structure(np.diag([2.0, 4.0, 4.0]), [[0.1, 0.2, 0.3], [1.0, 0.2, 0.3]], [1, 8])
    provider = MACEForceProvider(model=Model(), compute_stress=True)
    graph = provider.tensor_graph(structure)
    assert graph["edge_index"].dtype == torch.int64
    assert graph["edge_index"].shape[0] == 2
    np.testing.assert_allclose(
        graph["shifts"].cpu().numpy(), graph["unit_shifts"].cpu().numpy() @ structure.cell
    )
    with torch.no_grad():
        evaluation = provider.evaluate(structure)
    np.testing.assert_allclose(evaluation.forces_eV_A, -structure.positions)
    assert evaluation.energy_eV == pytest.approx(0.5 * np.sum(structure.positions**2))
    np.testing.assert_allclose(evaluation.stress_eV_A3, np.eye(3))
    with pytest.raises(ValueError, match="does not support"):
        provider.evaluate(structure.copy(atomic_numbers=[1, 6]))
    assert len(provider.fingerprint) == 64
    assert provider.provenance["checkpoint_sha256"] is None
    assert len(provider.provenance["model_state_sha256"]) == 64
    converted = MACEForceProvider(
        model=Model(), compute_stress=True, energy_units_to_eV=3, length_units_to_A=2
    ).evaluate(structure)
    assert converted.energy_eV == pytest.approx(3 * 0.5 * np.sum((structure.positions / 2) ** 2))
    np.testing.assert_allclose(converted.forces_eV_A, -3 * structure.positions / 4)
    np.testing.assert_allclose(converted.stress_eV_A3, np.eye(3) * 3 / 8)
    model = Model()
    model.heads = ["first", "second"]
    with pytest.raises(ValueError, match="explicit head"):
        MACEForceProvider(model=model)
    assert MACEForceProvider(model=model, head="second").tensor_graph(structure)["head"].item() == 1


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="optional Torch producer environment")
def test_checkpoint_model_provenance_matches_exact_file_bytes(tmp_path):
    import torch

    # A built-in picklable module exercises checkpoint loading without requiring
    # external MACE classes; graph prediction is independently checked above.
    model = torch.nn.Linear(3, 1).double()
    model.register_buffer("atomic_numbers", torch.tensor([1]))
    model.register_buffer("r_max", torch.tensor(2.0))
    checkpoint = tmp_path / "resolved.model"
    torch.save(model, checkpoint)
    provider = MACEForceProvider.from_checkpoint(checkpoint)
    assert provider.provenance["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert provider.provenance["model_state_sha256"] is None


def test_adapter_rejects_ambiguous_source_without_importing_optional_stack():
    with pytest.raises(ValueError, match="exactly one"):
        MACEForceProvider()
