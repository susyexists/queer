"""HHH conventions at public entrypoints and recorded workflow boundaries."""

import math

import numpy as np
import pytest

from queer.config import config_from_mapping
from queer.finite_field import (
    PAPER,
    FiniteFieldOptions,
    FourBandModel,
    FourBandParams,
    PreparedHallResponse,
    preset,
    resolve_options,
)
from queer.finite_field.config import Conventions
from queer.finite_field.geometry.topology import hall_chern
from queer.finite_field.transport.hall import HallCalculator
from queer.workflows import run_workflow


def test_shared_paper_default_and_named_figure_conventions():
    model = FourBandModel(FourBandParams().at(0.5))
    assert FiniteFieldOptions().to_dict() == PAPER.to_dict()
    assert resolve_options().to_dict() == resolve_options({}).to_dict() == PAPER.to_dict()
    assert HallCalculator(model).cfg == PAPER
    assert PreparedHallResponse(model).options == PAPER
    for name in ("fig5c", "ext7a"):
        cfg = preset(name)
        assert cfg.relaxation.T1 == 2e-12
        assert cfg.relaxation.T2 == math.inf
        assert cfg.conventions.figure_scale == 1 / (4 * math.pi)
    assert preset("fig4d").crossings.definition == "diabatic"
    assert hall_chern(-1.0) == 1.0


def test_atomic_section_updates_and_obsolete_critical_reference():
    raised = PAPER.with_(**{"drive.E_min": 1e5, "drive.E_max": 2e5})
    reverse = PAPER.with_(**{"drive.E_max": 2e5, "drive.E_min": 1e5})
    assert raised == reverse
    assert raised.drive.E_min == 1e5
    with pytest.raises(ValueError, match="paper_vlc_matched"):
        Conventions(vlc_reference="paper")
    with pytest.raises(KeyError):
        PAPER.with_(**{"unknown.setting": 1})


@pytest.mark.parametrize("selected", [None, "physical_eq8"])
def test_workflow_records_resolved_profile_and_matches_public_response(tmp_path, monkeypatch, selected):
    import json

    parameters = {
        "model": {"x": 0.5},
        "fields": [-1e4, 0, 1e4],
        "drive": {"E_min": 1e3, "E_max": 3e4},
        "crossings": {"window": 0.002, "scan_points": 65},
        "quadrature": {"t_max": 0.0002, "du": 2.0, "dmin": 1e-7},
    }
    if selected:
        parameters["options"] = {"preset": selected}
    config = config_from_mapping(
        {
            "workflow": "landau_zener_hall",
            "inputs": {"model": "fourband"},
            "parameters": parameters,
            "output": {"results_dir": str(tmp_path)},
        }
    )
    run = run_workflow(config)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["scientific_model"]["preset"] == (selected or "paper")
    from queer.workflows.finite_field import _options

    prepared = PreparedHallResponse(FourBandModel(FourBandParams().at(0.5)), _options(parameters))
    result = prepared.hall(parameters["fields"])
    case = manifest["cases"]["result"]
    with np.load(run / case["path"], allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["current"], result["current"])
    assert case["metadata"]["options"] == manifest["scientific_model"]["resolved_options"]
    assert len(manifest["scientific_model"]["workflow_sha256"]) == 64
    saved = (run / "manifest.json").read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr("queer.workflows.runner._finite_field_workflow_hash", lambda: "changed-workflow")
        with pytest.raises(ValueError, match="workflow implementation changed"):
            run_workflow(config, run_dir=run, resume=True)
        assert (run / "manifest.json").read_bytes() == saved
    monkeypatch.setattr("queer.finite_field._storage.source_hash", lambda: "changed-engine")
    with pytest.raises(ValueError, match="finite-field engine changed"):
        run_workflow(config, run_dir=run, resume=True)
    assert (run / "manifest.json").read_bytes() == saved
