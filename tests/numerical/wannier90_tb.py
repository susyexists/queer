"""``wannier90_tb.dat`` written as wannier90 3.1's ``hamiltonian_write_tb`` writes it.

Blocks are given as written (degeneracy weights included) and indexed
[j-1, i-1] = <0j|X|Ri>; every entry uses Fortran's E15.8 edit descriptor, and the
format's trailing ``1x`` writes nothing at the end of a record.
"""

import math
from pathlib import Path

import numpy as np


def fortran_e(value, digits=8, width=15):
    """``value`` in Fortran's E<width>.<digits> form, e.g. ' 0.12345678E+01'."""
    x = float(value)
    if x == 0.0:
        mantissa, exponent = 0.0, 0
    else:
        exponent = math.floor(math.log10(abs(x))) + 1
        mantissa = round(x / 10.0**exponent, digits)
        if abs(mantissa) >= 1.0:
            mantissa /= 10.0
            exponent += 1
    return f"{mantissa:.{digits}f}E{exponent:+03d}".rjust(width)


def written(values, digits=8):
    """``values`` as E15.8 writes them and Fortran reads them back."""
    array = np.asarray(values)
    if np.iscomplexobj(array):
        return written(array.real, digits) + 1j * written(array.imag, digits)
    flat = [float(fortran_e(value, digits)) for value in array.astype(float).ravel()]
    return np.asarray(flat, float).reshape(array.shape)


def tb_text(lattice, translations, degeneracies, H, r, header="written by the QUEER tests"):
    """The file's text: H (nR, N, N) eV and r (nR, N, N, 3) Å, both as written."""
    translations = np.asarray(translations, int)
    H, r = np.asarray(H, complex), np.asarray(r, complex)
    n = H.shape[1]
    lines = [header]
    lines += ["  " + "  ".join(f"{value:.10f}" for value in row) for row in np.asarray(lattice, float)]
    lines += [f"{n:12d}", f"{len(translations):12d}"]
    for start in range(0, len(degeneracies), 15):
        lines.append("".join(f"{int(value):5d}" for value in degeneracies[start : start + 15]))
    for k, translation in enumerate(translations):
        lines += ["", "".join(f"{int(value):5d}" for value in translation)]
        for i in range(n):
            for j in range(n):
                value = H[k, j, i]
                lines.append(f"{j + 1:5d}{i + 1:5d}   {fortran_e(value.real)} {fortran_e(value.imag)}")
    for k, translation in enumerate(translations):
        lines += ["", "".join(f"{int(value):5d}" for value in translation)]
        for i in range(n):
            for j in range(n):
                parts = []
                for a in range(3):
                    parts += [fortran_e(r[k, j, i, a].real), fortran_e(r[k, j, i, a].imag)]
                lines.append(f"{j + 1:5d}{i + 1:5d}   " + " ".join(parts))
    return "\n".join(lines) + "\n"


def write_tb(path, lattice, translations, degeneracies, H, r, header="written by the QUEER tests"):
    """Write ``wannier90_tb.dat`` at ``path``; returns the path."""
    target = Path(path)
    target.write_text(tb_text(lattice, translations, degeneracies, H, r, header))
    return target
