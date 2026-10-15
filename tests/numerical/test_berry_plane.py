"""Analytic references for Fermi-sea curvature, reduced plane maps and the quantum weight."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from tight_binding_models import cell_periodic_operators, haldane, qwz

from queer.models.evaluation import derivatives, reciprocal, states
from queer.observables.geometry import berry_curvature, berry_plane, chern_number, quantum_metric
from queer.observables.topology import plane_kpoints
from queer.physics.geometry import (
    in_plane_trace,
    plane_flux,
    plane_normal,
    quantum_weight,
    reduced_plane_curvature,
)

MESH = 18


def kubo_from_velocities(v_a, v_b, energies, occupied):
    """-2 Im sum_(n occupied, m empty) v_a,nm v_b,mn / (E_n - E_m)^2, written out pair by pair."""
    total = 0.0
    for n in np.flatnonzero(occupied):
        for m in np.flatnonzero(~occupied):
            total += -2 * (v_a[n, m] * v_b[m, n]).imag / (energies[n] - energies[m]) ** 2
    return total


@pytest.mark.parametrize("builder,expected", [(haldane, -1), (lambda: qwz(-1), 1)], ids=["haldane", "qwz"])
def test_fermi_sea_and_kubo_maps_agree_in_a_gap_and_integrate_to_the_plaquette_chern(builder, expected):
    model = builder()
    points = plane_kpoints(2, 0.0, MESH)
    sea = berry_plane(model, points, plane=(0, 1), fermi_energy=0.0)
    bands = berry_plane(model, points, plane=(0, 1), occupied_bands=1, fermi_energy=0.0, metric=True)
    scale = np.abs(bands["omega_reduced"]).max()
    assert_allclose(sea["omega_reduced"], bands["omega_reduced"], rtol=0, atol=1e-12 * scale)
    plaquette = float(chern_number(model, grid=(MESH, MESH, 1))["chern_number"])
    assert round(plaquette) == expected and abs(plaquette - expected) < 1e-10
    assert abs(plane_flux(bands["omega_reduced"], MESH) - expected) < 1e-4
    assert abs(plane_flux(sea["omega_reduced"], MESH) - expected) < 1e-4
    b = reciprocal(model)
    normal, area = plane_normal(b, (0, 1))
    cartesian = berry_curvature(model, points[::17], occupied_bands=1, engine="kubo")["curvature"]
    assert_allclose(
        reduced_plane_curvature(cartesian, b, (0, 1)), bands["omega_reduced"][::17], atol=1e-12 * scale
    )
    assert_allclose(bands["omega_cartesian"], bands["omega_reduced"] / area, rtol=1e-15)
    metric = quantum_metric(model, points[::17], occupied_bands=1)["metric"]
    assert_allclose(in_plane_trace(metric, normal), bands["metric_trace"][::17], rtol=1e-12)
    weight = quantum_weight(bands["metric_trace"], area, MESH)
    assert weight > abs(expected) and bands.units["metric_trace"] == "Angstrom^2"
    assert sea.metadata["occupation"] == "fermi" and bands.metadata["occupation"] == "bands"


def test_metal_fermi_sea_curvature_sums_the_occupied_band_curvatures():
    model = haldane()
    points = plane_kpoints(2, 0.0, 6)
    fermi = 0.8
    sea = berry_plane(model, points, plane=(0, 1), fermi_energy=fermi)
    per_band = berry_curvature(model, points, fermi_energy=fermi, engine="kubo")["curvature"]
    energies = model.energies(points) + model.fermi_energy
    occupied = np.where(energies < fermi, 1.0, 0.0)
    summed = np.einsum("nk,nkab->kab", occupied, per_band)
    expected = reduced_plane_curvature(summed, reciprocal(model), (0, 1))
    assert 0 < occupied[1].sum() < len(points)
    assert_allclose(sea["omega_reduced"], expected, atol=1e-12 * np.abs(expected).max())


def test_fermi_sea_uses_cell_periodic_velocities_with_the_centre_term():
    model = haldane()
    k = np.array([0.17, 0.31, 0.0])
    hamiltonian, velocity = cell_periodic_operators(model, k)
    energies, vectors = np.linalg.eigh(hamiltonian)
    rotated = np.einsum("in,aij,jm->anm", vectors.conj(), velocity, vectors)
    reference = kubo_from_velocities(rotated[0], rotated[1], energies, energies < 0)
    value = berry_plane(model, k[None], plane=(0, 1), fermi_energy=0.0)["omega_reduced"][0]
    assert_allclose(value, reference, rtol=1e-12)
    # The periodic-gauge derivative alone lacks i[H, tau] and gives another curvature.
    solution = states(model, k[None])
    coefficients = solution.coefficients[0]
    periodic = np.einsum("in,aij,jm->anm", coefficients.conj(), derivatives(model, k)["H"], coefficients)
    reduced = np.einsum("ac,cnm->anm", reciprocal(model)[:2], periodic)
    shifted = solution.energies[:, 0] - model.fermi_energy
    assert abs(kubo_from_velocities(reduced[0], reduced[1], shifted, shifted < 0) - reference) > 0.1 * abs(
        reference
    )


def test_qwz_metric_trace_curvature_and_quantum_weight_match_the_bloch_sphere_forms():
    mass = -1.0
    model = qwz(mass)
    points = plane_kpoints(2, 0.0, MESH)
    result = berry_plane(model, points, plane=(0, 1), occupied_bands=1, metric=True)
    theta = 2 * np.pi * points[:, :2]
    d = np.stack([np.sin(theta[:, 0]), np.sin(theta[:, 1]), mass + np.cos(theta).sum(axis=1)], axis=1)
    zero = np.zeros(len(points))
    gradients = [
        np.stack([np.cos(theta[:, 0]), zero, -np.sin(theta[:, 0])], axis=1),
        np.stack([zero, np.cos(theta[:, 1]), -np.sin(theta[:, 1])], axis=1),
    ]
    size = np.linalg.norm(d, axis=1)
    unit = d / size[:, None]
    du = [(g - unit * np.einsum("ka,ka->k", unit, g)[:, None]) / size[:, None] for g in gradients]
    # Lattice constant 1 A: k_cart = 2 pi k_frac, so b_x . Omega . b_y = (2 pi)^2 Omega_xy.
    curvature = 0.5 * np.einsum("ka,ka->k", unit, np.cross(du[0], du[1]))
    trace = 0.25 * (np.einsum("ka,ka->k", du[0], du[0]) + np.einsum("ka,ka->k", du[1], du[1]))
    assert_allclose(result["omega_reduced"], (2 * np.pi) ** 2 * curvature, atol=1e-12)
    assert_allclose(result["metric_trace"], trace, atol=1e-13)
    area = plane_normal(reciprocal(model), (0, 1))[1]
    assert area == pytest.approx((2 * np.pi) ** 2)
    weight = quantum_weight(result["metric_trace"], area, MESH)
    assert_allclose(weight, trace.sum() * (2 * np.pi) / MESH**2, rtol=1e-14)
    assert np.all(trace >= np.abs(curvature) - 1e-14) and weight > plane_flux(result["omega_reduced"], MESH)


def test_berry_plane_validates_its_selection():
    model = qwz(-1)
    with pytest.raises(ValueError, match="two distinct"):
        berry_plane(model, [[0, 0, 0]], plane=(0, 0))
    with pytest.raises(ValueError, match="occupied band count"):
        berry_plane(model, [[0, 0, 0]], plane=(0, 1), metric=True)
    with pytest.raises(ValueError, match="min_gap"):
        berry_plane(model, [[0, 0, 0]], plane=(0, 1), min_gap=-1)
