"""A loop-by-loop port of PyATB 60fc6b2a's SHIFT_CURRENT (method 1), standing in for PyATB in tests.

Written from PyATB's sources (``openmx_read_scfout.py``: the scfout's records, its Hartree and Bohr, the spinor
assembly and the real-space Hermitization of ``nspin 1``; ``shift_current_solver.cpp``:
``get_shift_current_conductivity``, ``get_shift_current_conductivity_ik`` and ``get_Inm_ik_sumOver``;
``linear_response.cpp``: ``get_D_degenerate`` and ``get_partial_eigenvalue_fromBar``; ``xr_operation.cpp`` and
``base_data.cpp``: the upper-triangle Bloch sums and the position operator's lower triangle; read only, written
anew) in the order PyATB runs them: one record at a time, one k-point at a time, one pair at a time, its own
generalized eigensolver (``scipy.linalg.eigh``). It shares no code with QUEER's scfout reader, with
``queer.compat.generator.shift_scfout`` or with ``queer.compat.generator.pyatb_shift_integrand``: a defect in any of
them leaves it unchanged. Method 0 and the adaptive smearing are not ported.
"""

from __future__ import annotations

import math
import struct

import numpy as np
from scipy.linalg import eigh

COMPONENTS = tuple(a + b + c for a in "xyz" for b, c in ("xx", "xy", "xz", "yy", "yz", "zz"))
THRESHOLD = 1e-4
REGULARIZATION = 0.04
AU2EV = 27.211324570273
BOHR_TO_ANG = 0.52917721
EV_SECONDS = 6.582119e-16
E_CHARGE = 1.602176565e-19
HBAR = 1.054571726e-34
TEM_B = (0, 0, 0, 1, 1, 2)
TEM_C = (0, 1, 2, 1, 2, 2)


def _read(path):
    """PyATB's reading of a version-3 scfout: H (its eV), S and r_a (Angstrom) per translation, the lattice
    (Angstrom), the basis size and nspin."""
    data = open(path, "rb").read()
    endian = "<" if 0 <= struct.unpack("<6i", data[:24])[1] <= 15 else ">"
    at = 0

    def ints(count):
        nonlocal at
        values = struct.unpack_from(f"{endian}{count}i", data, at)
        at += 4 * count
        return list(values)

    def doubles(count):
        nonlocal at
        values = struct.unpack_from(f"{endian}{count}d", data, at)
        at += 8 * count
        return list(values)

    head = ints(6)
    atoms, spin, cells = head[0], head[1] % 4, head[5] + 1
    order = ints(1)[0]
    doubles(4 * cells)
    cell_index = ints(4 * cells)
    cell_of = [tuple(cell_index[4 * c + 1 : 4 * c + 4]) for c in range(cells)]
    orbitals = ints(atoms)
    neighbours = ints(atoms)
    partner = [ints(neighbours[a] + 1) for a in range(atoms)]
    cell = [ints(neighbours[a] + 1) for a in range(atoms)]
    tv = np.array(doubles(12)).reshape(3, 4)[:, 1:]
    doubles(12)
    centres = [doubles(4)[1:] for _ in range(atoms)]
    start = [sum(orbitals[:a]) for a in range(atoms)]
    n_orb = sum(orbitals)
    size = 2 * n_orb if spin == 3 else n_orb

    def matrix_set():
        out = []
        for a in range(atoms):
            for h in range(neighbours[a] + 1):
                b = partner[a][h] - 1
                out.append(np.array([doubles(orbitals[b]) for _ in range(orbitals[a])]))
        return out

    hks = [matrix_set() for _ in range(spin + 1)]
    ihks = [matrix_set() for _ in range(3)] if spin == 3 else None
    olp = matrix_set()
    moments = []
    for _ in range(3):
        sets = [matrix_set() for _ in range(order)]
        moments.append(sets[0] if order else None)
    hamiltonian, overlap, position = {}, {}, {}
    pair = 0
    for a in range(atoms):
        for h in range(neighbours[a] + 1):
            b = partner[a][h] - 1
            key = cell_of[cell[a][h]]
            i0, j0, rows, cols = start[a], start[b], orbitals[a], orbitals[b]
            for store in (hamiltonian, overlap):
                store.setdefault(key, np.zeros((size, size), complex))
            position.setdefault(key, np.zeros((3, size, size), complex))
            s_block = olp[pair]
            if spin == 3:
                up = hks[0][pair] + 1j * ihks[0][pair]
                down = hks[1][pair] + 1j * ihks[1][pair]
                mixed = hks[2][pair] + 1j * (hks[3][pair] + ihks[2][pair])
                hamiltonian[key][i0 : i0 + rows, j0 : j0 + cols] += up
                hamiltonian[key][n_orb + i0 : n_orb + i0 + rows, n_orb + j0 : n_orb + j0 + cols] += down
                hamiltonian[key][i0 : i0 + rows, n_orb + j0 : n_orb + j0 + cols] += mixed
                overlap[key][n_orb + i0 : n_orb + i0 + rows, n_orb + j0 : n_orb + j0 + cols] += s_block
            else:
                hamiltonian[key][i0 : i0 + rows, j0 : j0 + cols] += hks[0][pair]
            overlap[key][i0 : i0 + rows, j0 : j0 + cols] += s_block
            for d in range(3):
                if moments[d] is None:
                    continue
                block = moments[d][pair] + centres[a][d] * s_block
                position[key][d, i0 : i0 + rows, j0 : j0 + cols] += block
                if spin == 3:
                    position[key][d, n_orb + i0 : n_orb + i0 + rows, n_orb + j0 : n_orb + j0 + cols] += block
            pair += 1
    keys = set(hamiltonian) | {tuple(-x for x in key) for key in hamiltonian}
    zero = np.zeros((size, size), complex)

    def mate(store, key):
        return store.get(tuple(-x for x in key), zero)

    if spin == 3:
        filled = {}
        for key in keys:
            block = np.array(hamiltonian.get(key, zero))
            block[n_orb:, :n_orb] = mate(hamiltonian, key)[:n_orb, n_orb:].conj().T
            filled[key] = block
        hamiltonian = filled
        overlap = {key: np.array(overlap.get(key, zero)) for key in keys}
    else:
        hamiltonian = {
            key: 0.5 * (hamiltonian.get(key, zero) + mate(hamiltonian, key).conj().T) for key in keys
        }
        overlap = {key: 0.5 * (overlap.get(key, zero) + mate(overlap, key).conj().T) for key in keys}
    keys = sorted(keys)
    hamiltonian = {key: AU2EV * hamiltonian[key] for key in keys}
    axes = [
        {key: BOHR_TO_ANG * position.get(key, np.zeros((3, size, size)))[d].real for key in keys}
        for d in range(3)
    ]
    return keys, hamiltonian, overlap, axes, size, tv * BOHR_TO_ANG, 4 if spin == 3 else 1


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


def pyatb_shift(scfout, block, *, nspin):
    """PyATB's printed shift current ``(count, 18)`` (microA/V^2) and its photon energies, for ``block`` (PyATB's
    resolved SHIFT_CURRENT keys) on the ``scfout`` (a path); ``nspin`` 1 reads the spin-up channel and doubles, 4
    the spinor bands."""
    if int(block.get("method", 1)) != 1 or int(block.get("smearing_method", 1)) not in (0, 1):
        raise ValueError("the port covers method 1 with smearing_method 0 or 1")
    keys, hamiltonian, overlap, position, size, cell, layout = _read(scfout)
    if int(nspin) != layout:
        raise ValueError(f"nspin {nspin} is not the scfout's {layout}")
    translations = np.array(keys, float)
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
