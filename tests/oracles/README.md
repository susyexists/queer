# Generator test oracles

The two Python files here are verbatim copies of the Generator's PythTB engine
sources. QUEER ports several of their functions; each port is tested against the
original function on randomized and structured inputs. The files are reference
data for those tests: they contain no tests, pytest never collects them, Ruff and
mypy skip them, and they are not part of the installed package.

| File | Generator source | SHA-256 |
|---|---|---|
| `generator_pythtb_core.py` | `generator-backend/engines/pythtb_core.py` | `56fda912ba570e2293b3a9b4bf20d11038c64efa5d2f192c8ac03c615034e215` |
| `generator_pythtb_qgeom.py` | `generator-backend/engines/pythtb_qgeom.py` | `4fb51a38691931f2b687d731619a21ae2caa094b1dd4681d1d97073041f23ee3` |

- **Source.** Copied on 2026-10-15 from generator-backend `origin/main` at commit
  `a2168e9b6d8b050418f7a860f7b74687088063a5`. Later commits up to `7dafb3b` leave
  both files unchanged.
- **License.** The owner of the Generator publishes these two files under the MIT
  License, the terms of QUEER's own `LICENSE`, for use as test oracles.
- **Loading.** `tests/conftest.py` imports each file by path with
  `importlib.util.spec_from_file_location`. At module level both import only the
  standard library and NumPy. PythTB, SciPy and matplotlib are imported inside the
  driver functions, which the tests never call, so PythTB need not be installed.
- **Updating.** Replace a file only with a verbatim copy from a recorded Generator
  commit, then update the table and the source line above.
  `tests/unit/test_generator_oracles.py` checks every hash against this table.
