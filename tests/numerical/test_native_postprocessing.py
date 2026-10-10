"""Independent small format records, quadrature and derivative references."""

import json
from functools import wraps
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from queer.capabilities import CapabilityError
from queer.config import config_from_mapping
from queer.io.artifacts import RunStore, digest_file
from queer.io.bands import (
    BOHR_ANGSTROM,
    HARTREE_EV,
    BandData,
    band_gap,
    compare_bands,
    read_eigenval,
    read_openmx_bands,
    read_openmx_output,
    read_vasp_kpoints,
    read_vasp_outcar,
    read_vasprun,
    read_wannier_bands,
    read_wannier_report,
)
from queer.io.calculations import (
    KBAR_EV_ANGSTROM3,
    read_openmx_results,
    read_vasp_results,
    read_vasprun_results,
)
from queer.models.openmx import OpenMXModel
from queer.models.wannier import WannierModel
from queer.observables.geometry import chern_number
from queer.observables.integration import integrate_unit_box, quadrature_checkpoint
from queer.observables.nonlinear import shift_current
from queer.workflows import run_workflow

PAULI = np.asarray([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])


def dimer():
    x, y, z = PAULI
    translations = [[0, 0, 0], [1, 0, 0], [-1, 0, 0]]
    hopping = np.asarray([1.1 * x + 0.4 * z, 0.35 * (x - 1j * y), 0.35 * (x + 1j * y)])
    position = np.zeros((3, 3, 2, 2), complex)
    position[0, 0] = np.diag([0, 0.5])
    return OpenMXModel(translations, hopping, position=position, lattice=np.eye(3), spin_degeneracy=1)


def test_embedded_cubature_and_threshold_have_normalized_cell_weights():
    reference = np.asarray([1 / 3, 1 / 2])

    def polynomial(points):
        return np.column_stack((points[:, 0] ** 2, points[:, 0]))

    result = integrate_unit_box(
        polynomial,
        [1],
        {"method": "cubature", "absolute_tolerance": 2e-5, "relative_tolerance": 0, "max_points": 5000},
    )
    assert_allclose(result.value, reference, atol=1e-13)
    assert result.metadata["normalized_weight_sum"] == 1
    assert np.all(result.error <= 2e-5)
    refined = integrate_unit_box(polynomial, [2], {"method": "threshold", "threshold": 0, "subdivision": [8]})
    assert_allclose(refined.value, reference, atol=4e-4)
    assert refined.metadata["refined_cells"] == 2
    assert refined.metadata["cells"] == 16
    with pytest.raises(RuntimeError, match="max_points"):
        integrate_unit_box(polynomial, [1], {"method": "cubature", "max_points": 2})
    with pytest.raises(RuntimeError, match="max_cells"):
        integrate_unit_box(
            polynomial, [2], {"method": "threshold", "threshold": 0, "subdivision": [8], "max_cells": 8}
        )
    with pytest.raises(ValueError, match="inactive integration axes"):
        integrate_unit_box(polynomial, [2, 2], {"method": "cubature"}, active_axes=[0])


def test_local_quadrature_resume_reuses_immutable_cells(tmp_path):
    config = config_from_mapping(
        {"workflow": "ahc", "inputs": {"material_dir": tmp_path}, "parameters": {"grid": [1, 1, 1]}}
    ).as_dict()
    store = RunStore(tmp_path / "run", config, [])
    calls = []

    def function(points):
        calls.append(len(points))
        return points[:, 0] ** 2

    try:
        with quadrature_checkpoint(store):
            first = integrate_unit_box(function, [1], {"method": "cubature", "absolute_tolerance": 1e-3})
        count = len(calls)
        with quadrature_checkpoint(store):
            second = integrate_unit_box(function, [1], {"method": "cubature", "absolute_tolerance": 1e-3})
        assert len(calls) == count
        assert_allclose(first.value, second.value)
        assert store.manifest["cases"]
    finally:
        store.release()


def test_interrupted_local_hall_workflow_resumes_cells_without_recomputation(tmp_path, monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "fixtures/openmx/chain"
    config = config_from_mapping(
        {
            "workflow": "ahc",
            "inputs": {"model": "openmx", "material_dir": directory, "scfout": "chain.scfout"},
            "parameters": {
                "grid": [2, 1, 1],
                "engine": "kubo",
                "chemical_potentials": [0],
                "integration": {"method": "cubature", "absolute_tolerance": 1e-6},
            },
            "execution": {"native_threads": 1},
            "output": {"results_dir": tmp_path / "results"},
        }
    )
    original, calls = OpenMXModel.eigenstates, []

    @wraps(original)
    def interrupt(self, kpoints, **kwargs):
        calls.append(np.asarray(kpoints).copy())
        if len(calls) == 4:
            raise KeyboardInterrupt("injected local cell interruption")
        return original(self, kpoints, **kwargs)

    monkeypatch.setattr(OpenMXModel, "eigenstates", interrupt)
    run = tmp_path / "partial"
    with pytest.raises(KeyboardInterrupt, match="injected local cell"):
        run_workflow(config, run_dir=run)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert len(manifest["cases"]) == 1
    case, record = next(iter(manifest["cases"].items()))
    artifact = run / record["path"]
    checksum, modified = digest_file(artifact), artifact.stat().st_mtime_ns
    resumed_calls = []

    @wraps(original)
    def record_resume(self, kpoints, **kwargs):
        resumed_calls.append(np.asarray(kpoints).copy())
        return original(self, kpoints, **kwargs)

    monkeypatch.setattr(OpenMXModel, "eigenstates", record_resume)
    run_workflow(config, run_dir=run, resume=True)
    assert len(resumed_calls) == 3
    assert not any(np.allclose(left, right) for left in calls[:3] for right in resumed_calls)
    final = json.loads((run / "manifest.json").read_text())
    assert final["status"] == "completed" and final["integration"]["converged"]
    assert final["cases"][case] == record
    assert digest_file(artifact) == checksum and artifact.stat().st_mtime_ns == modified
    with np.load(run / final["cases"]["result"]["path"], allow_pickle=False) as saved:
        assert_allclose(saved["ahc"], 0, atol=1e-15)


def test_oriented_chern_spans_reverse_flux_and_reject_degenerate_spans():
    x, y, z = PAULI
    model = WannierModel(
        [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
        [-z, 0.5 * (z - 1j * x), 0.5 * (z + 1j * x), 0.5 * (z - 1j * y), 0.5 * (z + 1j * y)],
        reciprocal=2 * np.pi * np.eye(3),
    )
    normal = chern_number(model, grid=(12, 12, 1), k_vectors=[[1, 0, 0], [0, 1, 0]])
    reversed_plane = chern_number(model, grid=(12, 12, 1), k_vectors=[[0, 1, 0], [1, 0, 0]])
    assert_allclose([normal["chern_number"], reversed_plane["chern_number"]], [1, -1], atol=1e-12)
    local = chern_number(
        model,
        grid=(4, 4, 1),
        engine="kubo",
        integration={"method": "threshold", "threshold": 1e20, "max_points": 17},
    )
    assert local.metadata["integration"]["points"] == 17
    assert local.metadata["integration"]["rank_check_points"] == 1
    with pytest.raises(RuntimeError, match="max_points"):
        chern_number(
            model,
            grid=(4, 4, 1),
            engine="kubo",
            integration={"method": "threshold", "threshold": 1e20, "max_points": 16},
        )
    with pytest.raises(ValueError, match="independent"):
        chern_number(model, k_vectors=[[1, 0, 0], [2, 0, 0]])


def test_sternheimer_pairs_and_count_scissors_match_analytic_dimer():
    model = dimer()
    points = np.asarray([[0.03, 0, 0], [0.2, 0, 0], [0.47, 0, 0]])
    parameters = {"kpoints": points, "eta": 0.07, "occupied_bands": 1, "band_pairs": [[0, 1]]}
    analytic = shift_current(model, [1, 2, 3], **parameters)
    sternheimer = shift_current(model, [1, 2, 3], method="sternheimer", **parameters)
    assert np.max(abs(analytic["shift_current"])) > 1
    assert_allclose(analytic["shift_current"], sternheimer["shift_current"], rtol=2e-12, atol=2e-9)
    assert_allclose(
        shift_current(model, [1, 2, 3], kpoints=points, eta=0.07)["shift_current"],
        analytic["shift_current"],
        rtol=1e-13,
    )
    shifted = shift_current(model, [1, 2, 3], chemical_potential=-100, scissors=0.3, **parameters)
    assert np.max(abs(shifted["shift_current"])) > 1
    with pytest.raises(ValueError, match="occupied to unoccupied"):
        shift_current(model, [2], occupied_bands=1, band_pairs=[[1, 0]], kpoints=points)


def test_shift_channel_counts_and_degenerate_manifold_guards():
    scalar = dimer()
    translations = scalar.translations
    hopping = np.asarray(
        [np.kron(np.eye(2), scalar.hamiltonian_blocks[tuple(t)].toarray()) for t in translations]
    )
    position = np.asarray(
        [
            [np.kron(np.eye(2), axis[tuple(t)].toarray()) for t in translations]
            for axis in scalar.position_blocks
        ]
    )
    model = OpenMXModel(translations, hopping, position=position, spin_mode="collinear")
    points = [[0.03, 0, 0], [0.2, 0, 0]]
    reference = shift_current(scalar, [1, 2, 3], kpoints=points, eta=0.07)
    selected = shift_current(
        model,
        [1, 2, 3],
        kpoints=points,
        eta=0.07,
        occupied_bands={"0": 1, "1": 1},
        band_pairs=[[0, 1], [2, 3]],
        method="sternheimer",
    )
    assert_allclose(selected["shift_current"], 2 * reference["shift_current"], rtol=2e-12, atol=2e-9)
    with pytest.raises(ValueError, match="every channel"):
        shift_current(model, [2], kpoints=points, occupied_bands={"0": 1})
    with pytest.raises(ValueError, match="different spin channels"):
        shift_current(model, [2], kpoints=points, occupied_bands={"0": 1, "1": 1}, band_pairs=[[0, 3]])
    degenerate = OpenMXModel(translations, hopping, position=position, spin_degeneracy=1)
    with pytest.raises(ValueError, match="splits a degenerate subspace"):
        shift_current(degenerate, [2], kpoints=points, occupied_bands=1)
    with pytest.raises(ValueError, match="degenerate transition manifold"):
        shift_current(degenerate, [2], kpoints=points, occupied_bands=2, band_pairs=[[0, 2]])


def test_normalized_bins_and_adaptive_width_are_mesh_aware():
    model = dimer()
    frequencies = np.arange(0.1, 5.01, 0.025)
    binned = shift_current(model, frequencies, smearing="none", grid=(16, 1, 1))
    gaussian = shift_current(model, frequencies, smearing="gaussian", eta=0.04, grid=(16, 1, 1))
    assert_allclose(
        np.sum(binned["shift_current"], axis=0) * 0.025,
        np.sum(gaussian["shift_current"], axis=0) * 0.025,
        rtol=2e-5,
        atol=1e-7,
    )
    adaptive = shift_current(model, [1, 2, 3], smearing="adaptive", grid=(16, 1, 1))
    assert np.isfinite(adaptive["shift_current"]).all()
    with pytest.raises(ValueError, match="adaptive_spacing"):
        shift_current(model, [2], smearing="adaptive", kpoints=[[0, 0, 0]])
    with pytest.raises(ValueError, match="response units"):
        shift_current(model, [2], integration={"method": "threshold", "threshold_quantity": "curvature"})


def test_vasp_eigenval_outcar_and_xml_agree_with_independent_records(tmp_path):
    eigenval = tmp_path / "EIGENVAL"
    eigenval.write_text(
        "1 1 1 2\nheader\nheader\nheader\nheader\n2 2 2\n\n"
        "0 0 0 0.5\n1 -1 -0.8 1 1\n2 1 1.2 0 0\n\n"
        "0.5 0 0 0.5\n1 -0.5 -0.3 1 1\n2 0.8 1 0 0\n"
    )
    reference = read_eigenval(eigenval, fermi_energy=0)
    assert reference.energies.shape == (2, 2, 2)
    assert_allclose(reference.energies[0], [[-1, 1], [-0.5, 0.8]])
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(
        "NKPTS = 2 NBANDS = 2\nE-fermi : 0.0\n"
        + "\n".join(
            f"spin component {spin + 1}\n"
            + "\n".join(
                f"k-point {k + 1} : {k * 0.5} 0 0\nband No. band energies occupation\n"
                + "\n".join(f"{b + 1} {reference.energies[spin, k, b]} {1 - b}" for b in range(2))
                for k in range(2)
            )
            for spin in range(2)
        )
    )
    parsed = read_vasp_outcar(outcar)
    assert_allclose(parsed.energies, reference.energies)
    xml = tmp_path / "vasprun.xml"
    xml.write_text(
        '<modeling><kpoints><varray name="kpointlist"><v>0 0 0</v><v>.5 0 0</v></varray>'
        '<varray name="weights"><v>.5</v><v>.5</v></varray></kpoints><calculation>'
        "<eigenvalues><array><set>"
        + "".join(
            '<set comment="spin">'
            + "".join(
                "<set>"
                + "".join(f"<r>{reference.energies[s, k, b]} {1 - b}</r>" for b in range(2))
                + "</set>"
                for k in range(2)
            )
            + "</set>"
            for s in range(2)
        )
        + '</set></array></eigenvalues><dos><i name="efermi">0</i></dos></calculation></modeling>'
    )
    assert_allclose(read_vasprun(xml).energies, reference.energies)
    gap = band_gap(reference)
    assert_allclose(gap["gap_eV"], 1.1)
    assert gap["is_direct"] is False
    eigenval.write_text(eigenval.read_text().rsplit("\n2 ", 1)[0])
    with pytest.raises(ValueError, match="malformed"):
        read_eigenval(eigenval)


def test_openmx_and_segmented_wannier_units_and_path_metadata(tmp_path):
    source = tmp_path / "toy.Band"
    source.write_text(
        "2 0 .1\n"
        + " ".join(map(str, np.eye(3).ravel()))
        + "\n1\n2 0 0 0 .5 0 0 G X\n2 0 0 0\n0 .2\n2 .5 0 0\n.02 .18\n"
    )
    data = read_openmx_bands(source)
    assert_allclose(data.energies[0], np.asarray([[0, 0.2], [0.02, 0.18]]) * HARTREE_EV)
    assert_allclose(data.reciprocal, np.eye(3) / BOHR_ANGSTROM)
    assert data.labels == ("G", "X")
    bandfile = tmp_path / "wannier90_band.dat"
    bandfile.write_text("0 -1\n1 -.5\n\n2 -.4\n3 -.3\n\n0 1\n1 .8\n\n2 .7\n3 .6\n")
    data = read_wannier_bands(bandfile, fermi_energy=0)
    assert data.segments == ((0, 2), (2, 4))
    assert data.energies.shape == (1, 4, 2)
    assert compare_bands(data, data)["rmse_eV"] == 0
    report = tmp_path / "wannier90.wout"
    report.write_text(
        "WF centre and spread 1 ( 0.1, 0.2, 0.3 ) 1.25\n"
        "WF centre and spread 2 ( 0.4, 0.5, 0.6 ) 2.5\nConvergence criteria satisfied\n"
    )
    assert read_wannier_report(report)["total_spread_angstrom2"] == 3.75


def test_collinear_openmx_band_headers_and_wannier_discontinuity_labels(tmp_path):
    source = tmp_path / "spin.Band"
    source.write_text(
        "2 1 0\n" + " ".join(map(str, np.eye(3).ravel())) + "\n1\n2 0 0 0 .5 0 0 G X\n"
        "2 0 0 0\n-.1 .2\n2 0 0 0\n-.08 .22\n"
        "2 .5 0 0\n-.05 .18\n2 .5 0 0\n-.03 .20\n"
    )
    data = read_openmx_bands(source)
    assert data.energies.shape == (2, 2, 2)
    assert_allclose(data.energies[1], np.asarray([[-0.08, 0.22], [-0.03, 0.20]]) * HARTREE_EV)
    assert data.metadata["spin_record_layout"] == "repeated_headers"
    bands, kpoints, labels = (tmp_path / name for name in ("band.dat", "band.kpt", "labels.dat"))
    bands.write_text("0 -1\n1 -.9\n1 -.8\n2 -.7\n\n0 1\n1 .9\n1 .8\n2 .7\n")
    kpoints.write_text("4\n0 0 0 1\n.5 0 0 1\n0 .5 0 1\n.5 .5 0 1\n")
    labels.write_text("G 1 0 0 0 0\nX 2 1 .5 0 0\nY 3 1 0 .5 0\nM 4 2 .5 .5 0\n")
    data = read_wannier_bands(bands, kpoints_path=kpoints, labelinfo_path=labels, fermi_energy=0)
    assert data.segments == ((0, 2), (2, 4))
    assert compare_bands(data, data)["rmse_eV"] == 0
    report = tmp_path / "wannier.wout"
    report.write_text("Wannierisation failed to converge\n")
    summary = read_wannier_report(report)
    assert summary["wannier_converged"] is False
    assert summary["disentanglement_converged"] is None


def test_line_mode_duplicates_and_band_comparison_do_not_cross_segments(tmp_path):
    path = tmp_path / "KPOINTS"
    path.write_text("test\n3\nLine-mode\nReciprocal\n0 0 0 ! G\n.5 0 0 ! X\n\n0 .5 0 ! Y\n.5 .5 0 ! M\n")
    parsed = read_vasp_kpoints(path)
    assert parsed["segments"] == ((0, 3), (3, 6))
    assert parsed["label_indices"] == (0, 2, 3, 5)
    values = np.column_stack((np.arange(6) - 10, np.arange(6) + 1))
    data = BandData(values, parsed["kpoints"], 0, segments=parsed["segments"])
    assert compare_bands(data, data)["coverage"] == 1
    with pytest.raises(CapabilityError, match="Fermi"):
        compare_bands(data, BandData(values, parsed["kpoints"], segments=parsed["segments"]))
    result = data.as_result()
    assert result.dims["energies"] == ("spin", "kpoint", "band")


def test_output_only_vasp_force_stress_and_energy_definitions(tmp_path):
    source = tmp_path / "OUTCAR"
    source.write_text(
        "NIONS = 2\nIteration 1(1)\naborting loop because EDIFF is reached\n"
        "free  energy   TOTEN = -10.5 eV\nenergy without entropy= -10.4 energy(sigma->0) = -10.3\n"
        "POSITION TOTAL-FORCE (eV/Angst)\n------------------\n"
        "0 0 0 .1 -.2 .3\n1 0 0 -.1 .2 -.3\n------------------\n"
        "in kB 1 2 3 4 5 6\nreached required accuracy - stopping structural energy minimisation\n"
    )
    result = read_vasp_results(source)
    assert result.bands is None
    assert_allclose(result.arrays["energy_eV"], -10.3)
    assert_allclose(result.arrays["free_energy_eV"], -10.5)
    assert_allclose(result.arrays["forces"], [[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]])
    assert_allclose(
        result.arrays["stress"], -np.asarray([[1, 4, 6], [4, 2, 5], [6, 5, 3]]) * KBAR_EV_ANGSTROM3
    )
    assert result.summary["electronic_converged"] and result.summary["ionic_converged"]
    assert result.as_result().units["stress"] == "eV/angstrom^3"
    xml = tmp_path / "vasprun.xml"
    xml.write_text(
        '<modeling><parameters><i name="NELM">60</i><i name="NSW">5</i>'
        '<i name="EDIFFG">-.5</i></parameters><calculation><scstep/>'
        '<energy><i name="e_0_energy">-10.3</i><i name="e_fr_energy">-10.5</i></energy>'
        '<varray name="forces"><v>.1 -.2 .3</v><v>-.1 .2 -.3</v></varray>'
        '<varray name="stress"><v>1 4 6</v><v>4 2 5</v><v>6 5 3</v></varray>'
        "</calculation></modeling>"
    )
    parsed = read_vasprun_results(xml)
    assert parsed.bands is None
    for name in ("energy_eV", "free_energy_eV", "forces", "stress"):
        assert_allclose(parsed.arrays[name], result.arrays[name])
    assert parsed.summary["electronic_converged"] and parsed.summary["ionic_converged"]
    xml.write_text(
        '<modeling><parameters><i name="NSW">5</i><i name="EDIFFG">.1</i></parameters>'
        '<calculation><energy><i name="e_0_energy">-10</i><i name="e_fr_energy">-10.5</i>'
        '</energy></calculation><calculation><energy><i name="e_0_energy">-10.9</i>'
        '<i name="e_fr_energy">-10.55</i></energy></calculation></modeling>'
    )
    parsed = read_vasprun_results(xml)
    assert parsed.summary["ionic_converged"] is True
    assert parsed.summary["electronic_converged"] is None
    assert "free-energy" in parsed.summary["ionic_convergence_provenance"]


def test_openmx_energy_forces_and_strain_stress_from_cell_derivatives(tmp_path):
    path = tmp_path / "openmx.out"
    path.write_text(
        "Utot = -1\nSCF calculation is converged\nAtom forces (Hartree/Bohr)\n"
        "1 X .1 .2 .3\n\n"
        "Cell vectors (Ang.) and derivatives (Hartree/Bohr)\n"
        "a1 = 2 0 0 dE/da1 = .1 0 0\n"
        "a2 = 0 2 0 dE/da2 = 0 .2 0\n"
        "a3 = 0 0 2 dE/da3 = 0 0 .3\n"
    )
    result = read_openmx_results(path)
    assert_allclose(result.arrays["energy_eV"], -HARTREE_EV)
    assert_allclose(result.arrays["forces"], [[0.1, 0.2, 0.3]] * np.asarray(HARTREE_EV / BOHR_ANGSTROM))
    assert_allclose(result.arrays["stress"], np.diag([0.1, 0.2, 0.3]) * HARTREE_EV / BOHR_ANGSTROM / 4)


@pytest.mark.parametrize(
    "marker", ["SCF is achieved", "Convergence criterion is satisfied", "SCF is converged"]
)
def test_openmx_final_scf_markers_and_coordinate_force_records(tmp_path, marker):
    path = tmp_path / "openmx.out"
    path.write_text(
        "Atoms.Number 2\n*** SCF iteration 1 ***\nUtot. -2\nSCF failed to converge\n"
        "*** SCF iteration 1 ***\nUtot. -2.1\n*** SCF iteration 2 ***\n"
        + marker
        + "\n***** xyz-coordinate (Ang) and forces (Hartree/Bohr) *****\n"
        "1 Si 0 0 0 .1 .2 .3\n2 C 1 2 3 -.1 -.2 -.3\n*****\n"
    )
    result = read_openmx_results(path)
    assert_allclose(result.arrays["energy_eV"], -2.1 * HARTREE_EV)
    assert_allclose(result.arrays["positions"], [[0, 0, 0], [1, 2, 3]])
    assert_allclose(result.arrays["atomic_numbers"], [14, 6])
    assert result.summary["scf_converged"] is True
    assert result.summary["scf_iterations"] == 2
    assert result.summary["scf_convergence_provenance"]["marker"] == marker
    path.write_text(path.read_text() + "*** SCF iteration 1 ***\nUtot. -2.2\n")
    assert read_openmx_output(path)["scf_converged"] is None
    path.write_text(path.read_text() + "SCF not converged\n")
    assert read_openmx_output(path)["scf_converged"] is False


def test_vasp_spinor_occupations_and_incomplete_spin_cycle(tmp_path):
    path = tmp_path / "OUTCAR"
    block = "k-point 1 : 0 0 0\nband No. band energies occupation\n1 -1 1\n2 1 0\n"
    path.write_text("ISPIN = 1 LSORBIT = T\nNKPTS = 1 NBANDS = 2\n" + block)
    data = read_vasp_outcar(path)
    assert data.metadata["occupation_capacity"] == 1
    assert data.spin_labels == ("spinor",)
    assert band_gap(data)["is_metal"] is False
    path.write_text(
        "ISPIN = 2\nNKPTS = 1 NBANDS = 2\nspin component 1\n"
        + block
        + "spin component 2\n"
        + block
        + "spin component 1\n"
        + block
    )
    with pytest.raises(ValueError, match="incomplete final OUTCAR spin"):
        read_vasp_outcar(path)
    xml = tmp_path / "vasprun.xml"
    xml.write_text(
        '<modeling><parameters><i name="ISPIN">1</i><i name="LNONCOLLINEAR">T</i>'
        '</parameters><varray name="kpointlist"><v>0 0 0</v></varray><calculation>'
        "<eigenvalues><array><set><set><set><r>-1 1</r><r>1 0</r></set></set></set>"
        "</array></eigenvalues></calculation></modeling>"
    )
    assert band_gap(read_vasprun(xml))["gap_eV"] == 2
