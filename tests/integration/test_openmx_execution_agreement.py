"""Every native observable agrees across CPU process and batch budgets."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from queer.execution import ExecutionOptions, cpu_budget
from queer.workflows.registry import REGISTRY, _arguments
from queer.workflows.runner import _load_model


@pytest.fixture(scope="module")
def native_helpers():
    """Reuse the independently constructed binary and radial-data fixture."""
    path = Path(__file__).with_name("test_openmx_workflows.py")
    spec = importlib.util.spec_from_file_location("_native_workflow_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_NONZERO_RESULTS = {
    "fat_band": "weights",
    "spin_texture": "spin",
    "dos": "dos",
    "pdos": "pdos",
    "jdos": "jdos",
    "fermi_surface": "energies",
    "reduce_basis": "scores",
    "surface_state": "spectrum",
    "bandunfolding": "weights",
    "bandunfolding_spin_texture": "spin",
    "boltz_transport": "electron_count",
}


@pytest.mark.skipif(cpu_budget() < 2, reason="process agreement requires two allocated CPUs")
@pytest.mark.parametrize("name", tuple(REGISTRY))
def test_all_native_observables_agree_across_process_and_batch_budgets(tmp_path, native_helpers, name):
    directory = native_helpers.native_input(tmp_path / "material")
    base = native_helpers.native_config(directory, name, tmp_path / "results")
    results = []
    for execution in (
        ExecutionOptions(workers=1, batch_size=2, native_threads=1),
        ExecutionOptions(workers=2, batch_size=1, native_threads=1),
    ):
        # Match model defaults and explicit calculator budgets, as in TOML workflows.
        config = replace(base, execution=execution)
        model = _load_model(config)
        results.append(REGISTRY[name].calculator()(model, **_arguments(config, model)))

    serial, parallel = results
    assert serial.arrays.keys() == parallel.arrays.keys()
    assert serial.dims == parallel.dims
    assert serial.units == parallel.units
    for key, expected in serial.arrays.items():
        actual = parallel.arrays[key]
        assert actual.shape == expected.shape, (name, key)
        assert actual.dtype == expected.dtype, (name, key)
        if expected.dtype.kind in "fc":
            assert np.isfinite(expected).all(), (name, key)
            assert np.isfinite(actual).all(), (name, key)
            # Scale each array independently; preserve sensitivity to tiny SI responses.
            scale = float(np.max(np.abs(expected), initial=0.0))
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-11,
                atol=128 * np.finfo(float).eps * scale,
                err_msg=f"{name}.{key}: workers=1/batch=2 versus workers=2/batch=1",
            )
        else:
            # Includes integer indices, masks, group names and spin-channel labels.
            np.testing.assert_array_equal(actual, expected, err_msg=f"{name}.{key}")

    if name in _NONZERO_RESULTS:
        assert np.any(serial.arrays[_NONZERO_RESULTS[name]] != 0), name
