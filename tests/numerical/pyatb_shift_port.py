"""A loop-by-loop port of PyATB 60fc6b2a's SHIFT_CURRENT (method 1), standing in for PyATB in tests.

Written from PyATB's sources (``shift_current_solver.cpp``: ``get_shift_current_conductivity``,
``get_shift_current_conductivity_ik`` and ``get_Inm_ik_sumOver``; ``linear_response.cpp``: ``get_D_degenerate``
and ``get_partial_eigenvalue_fromBar``; ``xr_operation.cpp`` and ``base_data.cpp``: the upper-triangle Bloch
sums and the position operator's lower triangle; read only, written anew) in the order PyATB runs them: one
k-point at a time, one pair at a time, its own generalized eigensolver (``scipy.linalg.eigh``). It shares no
code with ``queer.compat.generator.pyatb_shift_integrand``; it reads the scfout's own records through QUEER's
reader (``model.data``), as the comparator's PyATB mode does, so a reader defect would pass both. Method 0 and
the adaptive smearing are not ported.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import eigh

COMPONENTS = tuple(a + b + c for a in "xyz" for b, c in ("xx", "xy", "xz", "yy", "yz", "zz"))
THRESHOLD = 1e-4
REGULARIZATION = 0.04
AU2EV = 27.211324570273
EV_SECONDS = 6.582119e-16
E_CHARGE = 1.602176565e-19
HBAR = 1.054571726e-34
TEM_B = (0, 0, 0, 1, 1, 2)
TEM_C = (0, 1, 2, 1, 2, 2)


def _records(model, channel):
    """The scfout's records of the channel PyATB reads: H (PyATB's eV), S, r_a (Angstrom), keyed by translation."""
    data = model.data
    size = model.nbnd if model.spin_mode != "collinear" else model.n_spatial
    rows = slice(channel * size, (channel + 1) * size)
    scale = AU2EV / float(model.metadata["hartree_to_ev"])
    keys = sorted(set(data.hamiltonian) | set(data.overlap) | {key for axis in data.position for key in axis})

    def dense(blocks, key):
        if key not in blocks:
            return np.zeros((size, size), complex)
        block = blocks[key]
        return np.asarray(block.toarray() if hasattr(block, "toarray") else block, complex)[rows, rows]

    hamiltonian = {key: scale * dense(data.hamiltonian, key) for key in keys}
    overlap = {key: dense(data.overlap, key) for key in keys}
    position = [{key: dense(axis, key) for key in keys} for axis in (data.position or ({}, {}, {}))]
    return keys, hamiltonian, overlap, position, size


def _upper_sum(blocks, keys, phases, weights=None):
    """``sum_R w_R e^{ik.R} triu(X(R))`` (PyATB's get_Xk_triu)."""
    total = 0
    for index, key in enumerate(keys):
        factor = phases[index] * (1.0 if weights is None else weights[index])
        total = total + factor * np.triu(blocks[key])
    return total


def _selfadjoint(upper):
    """Eigen's selfadjointView<Upper>: the upper triangle mirrored, a real diagonal."""
    out = np.triu(upper, 1)
    out = out + out.conj().T
    out[np.diag_indices_from(out)] = np.real(np.diag(upper))
    return out


def _lower_assigned(upper, partner):
    """``rk.triangularView<Lower>() = (rk + i partner)^dagger`` on an upper-triangular ``rk``."""
    adjoint = (upper + 1j * partner).conj().T
    return np.triu(upper, 1) + np.tril(adjoint)


def _velocity(hbar, sbar, energies, band):
    low = band
    while low - 1 >= 0 and abs(energies[band] - energies[low - 1]) < THRESHOLD:
        low -= 1
    high = band
    while high + 1 < len(energies) and abs(energies[high + 1] - energies[band]) < THRESHOLD:
        high += 1
    if low == high:
        return float(np.real(hbar[band, band] - energies[band] * sbar[band, band]))
    block = hbar[low : high + 1, low : high + 1] - energies[band] * sbar[low : high + 1, low : high + 1]
    return float(np.linalg.eigvalsh(block, UPLO="L")[band - low])


def pyatb_shift(model, block, *, nspin):
    """PyATB's printed shift current ``(count, 18)`` (microA/V^2) and its photon energies, for ``block`` (PyATB's
    resolved SHIFT_CURRENT keys) on ``model`` (an OpenMXModel read from the scfout); ``nspin`` 1 reads the
    spin-up channel and doubles, 4 the spinor bands."""
    if int(block.get("method", 1)) != 1 or int(block.get("smearing_method", 1)) not in (0, 1):
        raise ValueError("the port covers method 1 with smearing_method 0 or 1")
    keys, hamiltonian, overlap, position, size = _records(model, 0)
    translations = np.array(keys, float)
    cell = np.asarray(model.lattice, float)
    cartesian = translations @ cell
    volume = float(np.linalg.det(cell))
    occ = int(block["occ_band"])
    start, end = (float(value) for value in block["omega"])
    domega = float(block["domega"])
    count = int((end - start) / domega) + 1
    smearing = int(block.get("smearing_method", 1))
    eta = float(block.get("eta", 0.01))
    scissors = float(block.get("scissors", 0.0) or 0.0)
    n_occ, m_unocc = int(block.get("n_occ", -1)), int(block.get("m_unocc", -1))
    pair = n_occ >= 1 and m_unocc >= 1 and m_unocc > n_occ
    grid = [int(value) for value in block["grid"]]
    axes = [np.arange(n) / n for n in grid]
    kpoints = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    table = np.zeros((18, count))
    for k in kpoints:
        phases = np.exp(2j * np.pi * (translations @ k))
        h_k = _selfadjoint(_upper_sum(hamiltonian, keys, phases))
        s_k = _selfadjoint(_upper_sum(overlap, keys, phases))
        h_a = [_selfadjoint(_upper_sum(hamiltonian, keys, phases, 1j * cartesian[:, a])) for a in range(3)]
        s_a = [_selfadjoint(_upper_sum(overlap, keys, phases, 1j * cartesian[:, a])) for a in range(3)]
        h_ab = [
            [
                _selfadjoint(_upper_sum(hamiltonian, keys, phases, -cartesian[:, a] * cartesian[:, b]))
                for b in range(3)
            ]
            for a in range(3)
        ]
        s_ab = [
            [
                _selfadjoint(_upper_sum(overlap, keys, phases, -cartesian[:, a] * cartesian[:, b]))
                for b in range(3)
            ]
            for a in range(3)
        ]
        r_k = [_lower_assigned(_upper_sum(position[a], keys, phases), s_a[a]) for a in range(3)]
        # d_b r_a with its lower triangle from (d_b r_a + i S_ab)^dagger.
        dr_k = [
            [
                _lower_assigned(_upper_sum(position[a], keys, phases, 1j * cartesian[:, b]), s_ab[a][b])
                for b in range(3)
            ]
            for a in range(3)
        ]
        energies, vectors = eigh(h_k, s_k)
        if scissors != 0.0:
            energies = energies.copy()
            energies[occ:] += scissors
        if pair:
            n_list, m_list = [m_unocc - 1], [n_occ - 1]
        else:
            n_list, m_list = [], []
            for n in range(occ, size):
                for m in range(occ):
                    delta = energies[n] - energies[m]
                    if start <= delta < start + count * domega:
                        n_list.append(n)
                        m_list.append(m)
        if not n_list:
            continue
        adjoint = vectors.conj().T
        hbar = [adjoint @ h_a[a] @ vectors for a in range(3)]
        sbar = [adjoint @ s_a[a] @ vectors for a in range(3)]
        rbar = [adjoint @ r_k[a].conj().T @ vectors for a in range(3)]
        derivative = []
        for a in range(3):
            d = np.zeros((size, size), complex)
            for row in range(size):
                for col in range(size):
                    if row == col:
                        d[row, col] = 1j * rbar[a][row, row]
                        continue
                    delta = energies[row] - energies[col]
                    inverse = (
                        1.0 / delta if abs(delta) > THRESHOLD else delta / (delta**2 + REGULARIZATION**2)
                    )
                    d[row, col] = -(hbar[a][row, col] - energies[col] * sbar[a][row, col]) * inverse
            derivative.append(d)
        velocity = [[_velocity(hbar[a], sbar[a], energies, band) for band in range(size)] for a in range(3)]
        integrand = np.zeros((18, len(n_list)))
        for index, (n, m) in enumerate(zip(n_list, m_list)):
            r_mn = [1j * derivative[a][m, n] + rbar[a][m, n] for a in range(3)]
            d_r = np.zeros((3, 3), complex)
            delta = energies[m] - energies[n]
            for a in range(3):
                for b in range(3):
                    left = derivative[b][:, n].conj()
                    db_h = (
                        left @ hbar[a][:, m]
                        + (adjoint @ h_ab[a][b] @ vectors)[n, m]
                        + hbar[a][n, :] @ derivative[b][:, m]
                    )
                    db_s = (
                        left @ sbar[a][:, m]
                        + (adjoint @ s_ab[a][b] @ vectors)[n, m]
                        + sbar[a][n, :] @ derivative[b][:, m]
                    )
                    db_r = (
                        left @ rbar[a][:, m]
                        + (adjoint @ dr_k[a][b].conj().T @ vectors)[n, m]
                        + rbar[a][n, :] @ derivative[b][:, m]
                    )
                    d_r[a, b] = (
                        (db_h - velocity[b][m] * sbar[a][n, m] - energies[m] * db_s) * delta
                        - (hbar[a][n, m] - energies[m] * sbar[a][n, m]) * (velocity[b][m] - velocity[b][n])
                    ) / delta / delta * 1j + db_r
            column = 0
            for a in range(3):
                for item in range(6):
                    b, c = TEM_B[item], TEM_C[item]
                    integrand[column, index] = (r_mn[b] * d_r[c, a] + r_mn[c] * d_r[b, a]).imag
                    column += 1
        interval = int(5 * eta / domega)
        for index, (n, m) in enumerate(zip(n_list, m_list)):
            delta = energies[n] - energies[m]
            row = int((delta - start) / domega)
            for j in range(18):
                value = -integrand[j, index]
                if smearing == 0:
                    if 0 <= row < count:
                        table[j, row] += value
                    continue
                for omega_index in range(row - interval, row + interval + 1):
                    if omega_index < 0 or omega_index >= count:
                        continue
                    omega = start + omega_index * domega
                    table[j, omega_index] += (
                        value * math.exp(-(((delta - omega) / eta) ** 2)) / eta / math.sqrt(math.pi)
                    )
    factor = -EV_SECONDS * math.pi * E_CHARGE**3 / (2 * HBAR**2 * volume * len(kpoints)) * 1.0e6
    factor *= 2 if int(nspin) == 1 else 1
    omega = np.array([start + domega * index for index in range(count)])
    return omega, (factor * table).T
