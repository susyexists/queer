"""A workflow owns one bounded executor across its independent numerical slices."""

from pathlib import Path

import numpy as np
import pytest

from queer.config import config_from_mapping, load_config
from queer.execution import ExecutionOptions, execution_session
from queer.models.wannier import WannierModel
from queer.workflows import run_workflow

ROOT = Path(__file__).resolve().parents[2]


def test_parallel_workflow_reuses_one_executor_and_matches_serial(tmp_path, monkeypatch):
    import joblib

    import queer.execution as execution

    monkeypatch.setattr(execution, "cpu_budget", lambda: 2)
    constructed = []
    calls = []

    class CountingParallel(joblib.Parallel):
        def __init__(self, *args, **kwargs):
            constructed.append(kwargs)
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            calls.append(1)
            return super().__call__(*args, **kwargs)

    monkeypatch.setattr(joblib, "Parallel", CountingParallel)
    data = load_config(ROOT / "configs/examples/susceptibility.toml").as_dict()
    data["parameters"]["q_path"] = [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]
    data["execution"] = {"workers": 1, "batch_size": 2, "native_threads": 1}
    serial = run_workflow(config_from_mapping(data), run_dir=tmp_path / "serial")
    assert not constructed
    data["execution"]["workers"] = 2
    parallel = run_workflow(config_from_mapping(data), run_dir=tmp_path / "parallel")
    assert len(constructed) == 1
    assert len(calls) == 4  # One cached mesh plus three shifted mesh evaluations.
    assert constructed[0]["pre_dispatch"] == 2
    assert constructed[0]["return_as"] == "generator"
    for reference in sorted((serial / "arrays").glob("*.npz")):
        with (
            np.load(reference, allow_pickle=False) as expected,
            np.load(parallel / "arrays" / reference.name, allow_pickle=False) as actual,
        ):
            np.testing.assert_allclose(actual["response"], expected["response"], atol=1e-10, rtol=1e-10)


def test_nested_execution_scopes_restore_parent_and_clean_up_after_error(monkeypatch):
    import queer.execution as execution

    monkeypatch.setattr(execution, "cpu_budget", lambda: 2)
    options = ExecutionOptions(workers=2, batch_size=1)
    model = WannierModel.from_directory(ROOT / "tests/fixtures/wannier/chain")
    point = [[0.125, 0.0, 0.0]]
    with execution_session(options) as outer:
        with execution_session(options) as nested:
            assert nested is outer
            np.testing.assert_allclose(
                model.energies(point, execution=options), [[0.07071067811865475]], atol=1e-12, rtol=1e-12
            )
        try:
            with execution_session(ExecutionOptions()) as serial:
                assert serial is not outer
                raise RuntimeError("injected nested failure")
        except RuntimeError:
            pass
        with execution_session(options) as restored:
            assert restored is outer
            np.testing.assert_allclose(
                model.energies(point, execution=options), [[0.07071067811865475]], atol=1e-12, rtol=1e-12
            )
    with execution_session(options) as fresh:
        assert fresh is not outer


def test_parallel_submissions_are_bounded_by_resolved_workers(monkeypatch):
    import joblib

    import queer.execution as execution

    monkeypatch.setattr(execution, "cpu_budget", lambda: 2)
    constructed = []
    submission_sizes = []
    consumed = []

    class TrackingParallel(joblib.Parallel):
        def __init__(self, *args, **kwargs):
            constructed.append(kwargs)
            super().__init__(*args, **kwargs)

        def __call__(self, tasks, *args, **kwargs):
            assert len(consumed) == len(submission_sizes)
            tasks = list(tasks)
            submission_sizes.append(len(tasks))
            results = super().__call__(tasks, *args, **kwargs)

            def tracked_results():
                yield from results
                consumed.append(True)

            return tracked_results()

    monkeypatch.setattr(joblib, "Parallel", TrackingParallel)
    options = ExecutionOptions(workers=4, batch_size=2)
    points = np.zeros((13, 3))
    points[:, 0] = np.linspace(-0.5, 0.5, len(points))
    model = WannierModel.from_directory(ROOT / "tests/fixtures/wannier/chain")
    energies = model.energies(points, execution=options)
    expected = 0.3 * np.cos(2 * np.pi * points[:, 0]) - 0.2 * np.sin(2 * np.pi * points[:, 0])
    np.testing.assert_allclose(energies, expected[None, :], atol=1e-12, rtol=1e-12)
    assert len(constructed) == 1
    assert submission_sizes == [2, 2, 2, 1]
    assert len(consumed) == len(submission_sizes)
    assert max(submission_sizes) <= options.resolved().workers


@pytest.mark.parametrize("config_name", ["spectral", "spectral_arpes"])
def test_spectral_artifacts_persist_boolean_validity_mask(tmp_path, config_name):
    data = load_config(ROOT / f"configs/examples/{config_name}.toml").as_dict()
    if config_name == "spectral_arpes":
        # The second point lies beyond the available final-state momentum sphere.
        data["parameters"]["kpoints"] = [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]
    run = run_workflow(config_from_mapping(data), run_dir=tmp_path / config_name)
    artifacts = sorted((run / "arrays").glob("*.npz"))
    assert artifacts
    for path in artifacts:
        with np.load(path, allow_pickle=False) as arrays:
            spectrum, mask = arrays["spectrum"], arrays["valid_mask"]
            assert mask.dtype == np.dtype(bool)
            assert mask.shape == spectrum.shape
            np.testing.assert_array_equal(mask, np.isfinite(spectrum))
            if config_name == "spectral_arpes":
                np.testing.assert_array_equal(mask, [[True, False]])
                assert np.isnan(spectrum[0, 1])
            else:
                assert mask.all()
