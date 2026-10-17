# Generator test oracles

The two Python files here are verbatim copies of the Generator's PythTB engine
sources. QUEER ports several of their functions; each port is tested against the
original function on randomized and structured inputs. The files are reference
data for those tests: they contain no tests, pytest never collects them, Ruff and
mypy skip them, and they are not part of the installed package.

| File | Generator source | SHA-256 |
|---|---|---|
| `generator_pythtb_core.py` | `generator-backend/engines/pythtb_core.py` | `93167386d4b15b24ff6ba84e656a4d4580164a3f02e52f6d2541f99b0027eb13` |
| `generator_pythtb_qgeom.py` | `generator-backend/engines/pythtb_qgeom.py` | `4fb51a38691931f2b687d731619a21ae2caa094b1dd4681d1d97073041f23ee3` |

- **Source.** `generator_pythtb_core.py`: copied on 2026-10-17 from generator-backend
  branch `feat/queer-r2` at commit `d05506578c1f55f584489b0dd0036e077609580d` (edge
  crossings by rank continuity; core `56fda912…` before it, unchanged from `dcdc104`
  through `c1c973b` on `main`). `generator_pythtb_qgeom.py`: copied on 2026-10-15 from
  `origin/main` at commit `a2168e9b6d8b050418f7a860f7b74687088063a5`, unchanged
  through `d055065`.
- **License.** The owner of the Generator publishes these two files under the MIT
  License, the terms of QUEER's own `LICENSE`, for use as test oracles.
- **Loading.** `tests/conftest.py` imports each file by path with
  `importlib.util.spec_from_file_location`. At module level both import only the
  standard library and NumPy. PythTB, SciPy and matplotlib are imported inside the
  driver functions, which the tests never call, so PythTB need not be installed.
- **Updating.** Replace a file only with a verbatim copy from a recorded Generator
  commit, then update the table and the source line above.
  `tests/unit/test_generator_oracles.py` checks every hash against this table.
