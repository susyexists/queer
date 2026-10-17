"""Step 7 of the shift-current rule: a QUEER GaAs run against the published sigma^xyz peak.

Opt-in: ``QUEER_SHIFT_LITERATURE_RUN`` names the directory of a ``QUEER_SHIFT_CURRENT``
run on GaAs with spin-orbit coupling at the paper's scissors and broadening;
``QUEER_SHIFT_LITERATURE_ANCHOR`` may name another anchor file than
``tests/fixtures/literature/shift_current_gaas_anchor.json``, whose values (Ibanez-
Azpiroz, Tsirkin and Souza, PRB 97, 245143 (2018)) are placeholders until they are
filled from the paper. The test skips without a run or with a placeholder anchor.
The check passes when QUEER's extremum of the published sign lies within 2 eta of
the published photon energy and within 20 % of the published height.
"""

import json
import os
from pathlib import Path

import pytest

from queer.compat.generator.parity_shift import literature_anchor

ROOT = Path(__file__).resolve().parents[2]
ANCHOR = Path(
    os.environ.get("QUEER_SHIFT_LITERATURE_ANCHOR")
    or ROOT / "tests" / "fixtures" / "literature" / "shift_current_gaas_anchor.json"
)
RUN = os.environ.get("QUEER_SHIFT_LITERATURE_RUN")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not RUN, reason="the literature anchor runs with QUEER_SHIFT_LITERATURE_RUN=<run directory>"
    ),
]


def test_gaas_sigma_xyz_meets_the_published_peak():
    anchor = json.loads(ANCHOR.read_text())
    result = literature_anchor(Path(RUN), anchor)
    if result["status"] == "placeholder":
        pytest.skip(f"the anchor {ANCHOR} still holds placeholders: {result['missing']}")
    assert result["status"] == "pass", result
