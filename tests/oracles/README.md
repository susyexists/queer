# Generator test oracles

The two Python files here are verbatim copies of the Generator's PythTB engine
sources. QUEER ports several of their functions; each port is tested against the
original function on randomized and structured inputs. The files are reference
data for those tests: they contain no tests, pytest never collects them, Ruff and
mypy skip them, and they are not part of the installed package.

| File | Generator source | SHA-256 |
|---|---|---|
| `generator_pythtb_core.py` | `generator-backend/engines/pythtb_core.py` | `0efcc4b3079266a8d35644d3d06f648857e31d86b12be173c6f03298a5d505c0` |
| `generator_pythtb_qgeom.py` | `generator-backend/engines/pythtb_qgeom.py` | `4fb51a38691931f2b687d731619a21ae2caa094b1dd4681d1d97073041f23ee3` |

- **Source.** `generator_pythtb_core.py`: copied on 2026-10-17 from generator-backend
  branch `feat/queer-r2` at commit `c59f3b68cbc40a6eee8634de894bbe0593bfea8a` (edge
  crossings as a weighted spectral flow of half-ribbon weight across the bulk gap,
  certified only where the bulk k mesh resolves the gap and checked against the bulk
  Chern number, never from absence, with a hopping resolved only from 8 points per
  period; core `f4a73779…` at `f236e92` also certified an empty gap on a bulk Chern
  number of 0 and resolved a hopping from 2 points per period; core `d75fb8fc…` at `184d1d2` took the sampled bulk gap and trusted
  its Chern number on the Berry-phase step alone, core `dbdf34d1…` at `56b1254` took
  the gap from the ribbon's own bulk-like levels, core `93167386…` at `d055065`
  counted by rank continuity, core `56fda912…` from `dcdc104` through `c1c973b` on
  `main` by nearest energy). `generator_pythtb_qgeom.py`:
  copied on 2026-10-15 from `origin/main` at commit
  `a2168e9b6d8b050418f7a860f7b74687088063a5`, unchanged through `c59f3b6`.
- **License.** The owner of the Generator publishes these two files under the MIT
  License, the terms of QUEER's own `LICENSE`, for use as test oracles.
- **Loading.** `tests/conftest.py` imports each file by path with
  `importlib.util.spec_from_file_location`. At module level both import only the
  standard library and NumPy. PythTB, SciPy and matplotlib are imported inside the
  driver functions, which the tests never call, so PythTB need not be installed.
- **Updating.** Replace a file only with a verbatim copy from a recorded Generator
  commit, then update the table and the source line above.
  `tests/unit/test_generator_oracles.py` checks every hash against this table.
