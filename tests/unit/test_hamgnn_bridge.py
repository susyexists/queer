"""Declared legacy units, row ordering and optional external-model inference."""

import importlib.util
import subprocess
import sys

import numpy as np
import pytest

from queer.adapters.hamgnn import HamGNNAdapter, read_legacy_graphs, to_pyg, write_legacy_graphs
from queer.io.hamiltonian import HamiltonianGraph
from queer.io.openmx import BOHR_ANGSTROM, HARTREE_EV

HAS_PYG = (
    importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torch_geometric") is not None
)


def native_graph():
    width = 14
    h = np.zeros((6, width, width))
    s = np.zeros_like(h)
    h[0][np.ix_([0, 3], [0, 3])] = [[2, 0.5], [0.5, 4]]
    h[1, 5, 5] = 7
    h[2, [0, 3], 5] = [0.2, -0.1]
    h[3] = h[2].T
    h[4, 0, 0] = h[5, 0, 0] = 0.3
    s[0, 0, 0] = s[0, 3, 3] = s[1, 5, 5] = 1
    return HamiltonianGraph(
        [1, 8],
        [[4, 0.5, 0], [0, 5, 0.3], [0, 0, 6]],
        [[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]],
        [2, 1],
        [[0, 3] + [-1] * 12, [5] + [-1] * 13],
        [[0, 1, 0, 0], [1, 0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [1, 0, 0], [-1, 0, 0]],
        [1, 0, 3, 2],
        h,
        s,
        h / 4,
        {"has_scf_reference": True, "doping_charge": 0.25},
    )


def legacy_arrays(graph):
    h = (graph.hamiltonian.reshape(-1, graph.nao_max**2) / HARTREE_EV).astype(np.float32)
    h0 = (graph.h0.reshape(h.shape) / HARTREE_EV).astype(np.float32)
    return {
        "z": graph.atomic_numbers,
        "cell": (graph.cell[None] / BOHR_ANGSTROM).astype(np.float32),
        "pos": (graph.positions / BOHR_ANGSTROM).astype(np.float32),
        "edge_index": graph.edge_index,
        "cell_shift": graph.cell_shift,
        "inv_edge_idx": graph.inverse_edges,
        "hamiltonian": h,
        "overlap": graph.overlap.reshape(h.shape).astype(np.float32),
        "Hon0": h0[: graph.natoms],
        "Hoff0": h0[graph.natoms :],
        "doping_charge": np.array([0.25], np.float32),
    }


def test_adapter_import_does_not_load_optional_graph_producers():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import queer.adapters.hamgnn; assert 'torch' not in sys.modules; assert 'torch_geometric' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_explicit_trusted_legacy_array_archive_preserves_basis_and_unknown_reference(tmp_path):
    graph = native_graph()
    path = tmp_path / "legacy.npz"
    np.savez(path, graph={"toy": legacy_arrays(graph)})
    with pytest.raises(ValueError, match="trusted=True"):
        read_legacy_graphs(path, basis_layout={1: [0, 3], 8: [5]})
    restored = read_legacy_graphs(path, basis_layout={1: [0, 3], 8: [5]}, trusted=True)["toy"]
    for key in graph.arrays():
        np.testing.assert_allclose(restored.arrays()[key], graph.arrays()[key], rtol=1e-7, atol=1e-7)
    assert restored.metadata["doping_charge"] == 0.25
    assert not restored.metadata["has_scf_reference"]


def test_non_soc_bridge_rejects_complex_h0_before_optional_import():
    graph = native_graph()
    graph.h0 = graph.h0.astype(complex)
    graph.h0[0, 0, 3] += 0.1j
    with pytest.raises(ValueError, match="complex"):
        to_pyg(graph)


@pytest.mark.skipif(not HAS_PYG, reason="optional Torch/PyG producer environment unavailable")
def test_pyg_legacy_tensors_match_float32_hartree_bohr_contract_and_row_order(tmp_path):
    import torch

    graph = native_graph()
    data = to_pyg(graph)
    expected = legacy_arrays(graph)
    for key, array in expected.items():
        actual = getattr(data, key)
        assert actual.dtype == (torch.int64 if array.dtype.kind in "iu" else torch.float32)
        np.testing.assert_array_equal(actual.cpu().numpy(), array)
    np.testing.assert_array_equal(data.Hon.numpy(), data.hamiltonian.numpy()[:2])
    np.testing.assert_array_equal(data.Hoff.numpy(), data.hamiltonian.numpy()[2:])
    np.testing.assert_array_equal(
        data.nbr_shift.numpy(), (graph.cell_shift @ graph.cell / BOHR_ANGSTROM).astype(np.float32)
    )
    assert data.hamiltonian[2, 3 * 14 + 5].item() == np.float32(-0.1 / HARTREE_EV)
    archive = write_legacy_graphs(tmp_path / "graph_data.npz", {"toy": graph})
    restored = read_legacy_graphs(archive, basis_layout={1: [0, 3], 8: [5]}, trusted=True)["toy"]
    np.testing.assert_allclose(restored.hamiltonian, graph.hamiltonian, rtol=1e-7, atol=1e-7)


@pytest.mark.skipif(not HAS_PYG, reason="optional Torch/PyG producer environment unavailable")
def test_direct_model_callback_inference_converts_hartree_and_records_provenance(tmp_path):
    import torch

    class Predictor(torch.nn.Module):
        def forward(self, data):
            assert data.batch.tolist() == [0, 0]
            assert data.ptr.tolist() == [0, 2]
            assert not torch.is_grad_enabled()
            return {"hamiltonian": data.hamiltonian + data.Hon0.new_tensor(0.01)}

    graph = native_graph()
    checkpoint = tmp_path / "producer.checkpoint"
    checkpoint.write_bytes(b"synthetic resolved producer identifier")
    adapter = HamGNNAdapter.from_checkpoint(checkpoint, loader=lambda path: Predictor())
    predicted = adapter.predict(graph)
    expected = np.where(graph.orbital_mask, graph.hamiltonian + 0.01 * HARTREE_EV, 0)
    np.testing.assert_allclose(predicted.hamiltonian, expected, atol=1e-6, rtol=1e-7)
    np.testing.assert_array_equal(predicted.overlap, graph.overlap)
    np.testing.assert_array_equal(predicted.h0, graph.h0)
    assert predicted.metadata["hamiltonian_source"] == "prediction"
    assert not predicted.metadata["has_scf_reference"]
    assert predicted.metadata["prediction_provenance"]["checkpoint_sha256"] == adapter.fingerprint
