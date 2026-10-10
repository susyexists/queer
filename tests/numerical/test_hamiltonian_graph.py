"""Graph identity, orbital placement, generalized spectra, and provenance."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from queer.io.hamiltonian import masked_hamiltonian_errors, pack_openmx, read_graphs, write_graphs
from queer.io.openmx import OpenMXData
from queer.models.graph import GraphHamiltonianModel


def graph_source():
    blocks = {(0, 0, 0): csr_matrix([[1.0]]), (1, 0, 0): csr_matrix([[0.2]]), (-1, 0, 0): csr_matrix([[0.2]])}
    overlap = {
        (0, 0, 0): csr_matrix([[1.0]]),
        (1, 0, 0): csr_matrix([[0.05]]),
        (-1, 0, 0): csr_matrix([[0.05]]),
    }
    return OpenMXData(
        blocks,
        overlap,
        ({}, {}, {}),
        np.diag([2.0, 3.0, 4.0]),
        np.array([[0.1, 0, 0]]),
        np.array([1]),
        "nonmagnetic",
        0.2,
        1,
        0,
        orbital_metadata=[{"l": 0, "zeta": 0, "m": 0}],
        metadata={
            "atom_species": ["H"],
            "neighbor_atoms": [[0, 0, 0]],
            "neighbor_translations": [[[0, 0, 0], [1, 0, 0], [-1, 0, 0]]],
        },
    )


@pytest.mark.parametrize("width", [14, 19, 26])
def test_graph_roundtrip_preserves_basis_and_positive_phase_spectrum(width, tmp_path):
    graph = pack_openmx(graph_source(), nao_max=width)
    np.testing.assert_array_equal(graph.inverse_edges, [1, 0])
    path = write_graphs(tmp_path / "graphs.npz", {"sample": graph})
    with np.load(path, allow_pickle=False) as data:
        assert not any(data[key].dtype.hasobject for key in data.files)
    restored = read_graphs(path)["sample"]
    model = GraphHamiltonianModel(restored)
    points = np.array([[0.0, 0, 0], [0.25, 0, 0], [0.5, 0, 0]])
    expected = (1 + 0.4 * np.cos(2 * np.pi * points[:, 0])) / (
        1 + 0.1 * np.cos(2 * np.pi * points[:, 0])
    ) - 0.2
    np.testing.assert_allclose(model.energies(points), expected[None], atol=1e-12)
    assert "position" not in model.capabilities


def test_padding_is_masked_per_block_and_h0_is_not_a_reference():
    graph = pack_openmx(graph_source())
    predictions = graph.hamiltonian.reshape(3, -1).copy()
    predictions[:, 0] += 0.1
    predictions[:, 1:] = 1e6
    errors = masked_hamiltonian_errors({"g": graph}, predictions)
    assert errors["all"]["mae_eV"] == pytest.approx(0.1)
    assert errors["all"]["orbital_entries"] == 3
    predicted = graph.with_predictions(predictions)
    assert np.count_nonzero(predicted.hamiltonian) == 3
    graph.metadata["has_scf_reference"] = False
    with pytest.raises(ValueError, match="real SCF"):
        masked_hamiltonian_errors([graph], predictions)


def test_translation_hermiticity_and_declared_repair():
    graph = pack_openmx(graph_source())
    graph.hamiltonian[1, 0, 0] += 0.1
    with pytest.raises(ValueError, match="Hermiticity"):
        GraphHamiltonianModel(graph)
    model = GraphHamiltonianModel(graph, hermitian_policy="average")
    np.testing.assert_allclose(model.energies([[0, 0, 0]]), [[1.5 / 1.1 - 0.2]])


def test_quantum_label_padding_reserves_s3_and_checkpoint_mapping_is_explicit():
    source = graph_source()
    source.orbital_metadata = [{"l": 1, "zeta": 1, "m": 2}]
    graph = pack_openmx(source)
    assert graph.orbital_slots[0, 0] == 8
    assert graph.hamiltonian[0, 8, 8] == 1
    source.orbital_metadata = [{"atom_index": 0}]
    with pytest.raises(ValueError, match="Incomplete orbital"):
        pack_openmx(source)
    graph = pack_openmx(source, basis_layout=[[5]])
    assert graph.hamiltonian[0, 5, 5] == 1


def test_missing_reverse_neighbors_are_rejected():
    source = graph_source()
    source.metadata["neighbor_atoms"] = [[0, 0]]
    source.metadata["neighbor_translations"] = [[[0, 0, 0], [1, 0, 0]]]
    with pytest.raises(ValueError, match="reverse edge"):
        pack_openmx(source)


def test_graph_model_requires_declared_fermi_reference():
    graph = pack_openmx(graph_source())
    graph.metadata.pop("fermi_energy_eV")
    with pytest.raises(ValueError, match="Fermi reference"):
        GraphHamiltonianModel(graph)
    model = GraphHamiltonianModel(graph, fermi_energy=0)
    assert model.fermi_energy == 0
    h0_only = pack_openmx(graph_source(), overlap_only=True)
    assert h0_only.metadata["fermi_energy_eV"] is None
    assert h0_only.metadata["reported_file_fermi_energy_eV"] == 0.2
    with pytest.raises(ValueError, match="Fermi reference"):
        GraphHamiltonianModel(h0_only)
