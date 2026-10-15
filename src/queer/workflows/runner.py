"""Compute and incrementally commit independent numerical cases."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Mapping

from ..config import WorkflowConfig, config_from_mapping
from ..execution import execution_session
from ..io.artifacts import RunStore, new_run_directory, owned_run
from .electronic_cases import _calculate as _calculate

logger = logging.getLogger(__name__)


def _finite_field_workflow_hash() -> str:
    digest = hashlib.sha256()
    directory = Path(__file__).parent
    for source in sorted(directory.rglob("*.py")):
        digest.update(str(source.relative_to(directory)).encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _input_files(config: WorkflowConfig) -> list[Path]:
    from .finite_field import FINITE_FIELD_NAMES, finite_field_input_files
    from .services import SERVICE_SPECS, service_input_files

    if config.workflow in FINITE_FIELD_NAMES:
        return finite_field_input_files(config)
    if config.workflow in SERVICE_SPECS:
        return service_input_files(config)
    directory = config.material_dir
    if config.workflow == "epw":
        # The EPW reader consumes metadata and channels in this directory.
        files = [directory / "gkk.out"] if (directory / "gkk.out").is_file() else []
        for channel in ("omega", "gkk_re", "gkk_im", "enk", "enkq"):
            folder = directory / channel
            if not folder.is_dir():
                raise ValueError(f"Missing EPW channel: {folder}")
            files.extend(
                file for file in folder.iterdir() if file.is_file() and not file.name.startswith(".")
            )
        if not files:
            raise ValueError(f"No EPW input files found: {directory}")
        return files
    from .sources import model_input_files

    files = model_input_files(config.inputs, config.parameters)
    files.extend(directory / value for value in config.inputs.get("pao_files", {}).values())
    if "pao_dir" in config.inputs:
        folder = directory / config.inputs["pao_dir"]
        if not folder.is_dir():
            raise ValueError(f"Missing PAO directory: {folder}")
        files.extend(sorted(folder.glob("*.pao")))
    return files


def _load_model(config: WorkflowConfig):
    params = config.parameters
    if config.workflow == "epw":
        from ..models.epw import EPWModel

        model = EPWModel.from_directory(
            config.material_dir, **{key: params[key] for key in ("nk", "nq", "nph", "ef") if key in params}
        )
        model.reduce_g()
        return model
    from .sources import load_electronic_model

    return load_electronic_model(config)


def run_workflow(
    config: WorkflowConfig | Mapping[str, Any],
    run_dir: str | Path | None = None,
    resume: bool = False,
    *,
    force_provider=None,
    force_provider_id: str | None = None,
) -> Path:
    """Calculate raw artifacts; render separately with :func:`render_run`.

    The binding-energy interval is half-open, matching NumPy ``arange``. Resume
    checks scientific parameters, all inputs and every completed artifact before
    skipping it. A completed run can be resumed idempotently.
    """
    if not isinstance(config, WorkflowConfig):
        config = config_from_mapping(config)
    from dataclasses import replace

    from .finite_field import FINITE_FIELD_NAMES, _options, calculate_finite_field, load_finite_field_source
    from .services import SERVICE_SPECS, calculate_service

    if force_provider is not None:
        if config.workflow not in {"phonons", "nvt", "relax"}:
            raise ValueError("force_provider is only consumed by phonons, nvt, and relax")
        identity = force_provider_id or getattr(force_provider, "fingerprint", None)
        if not isinstance(identity, str) or not identity:
            raise ValueError("A callback force_provider requires force_provider_id or a string fingerprint")
        config = replace(config, inputs={**config.inputs, "provider_identity": identity})
    if resume and run_dir is None:
        raise ValueError("resume requires an explicit run_dir")
    root = (
        Path(run_dir).expanduser().resolve()
        if run_dir is not None
        else new_run_directory(config.results_dir, config.workflow)
    )
    files = _input_files(config)
    missing = [str(file) for file in files if not file.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing workflow inputs: {', '.join(missing)}")
    finite_field_engine = None
    finite_field_workflow = None
    if config.workflow in FINITE_FIELD_NAMES:
        from ..finite_field._storage import source_hash

        finite_field_engine = source_hash()
        finite_field_workflow = _finite_field_workflow_hash()
        if resume and (root / "manifest.json").is_file():
            previous = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            if previous.get("cases") and (
                previous.get("scientific_model", {}).get("source_sha256") != finite_field_engine
                or previous.get("scientific_model", {}).get("workflow_sha256") != finite_field_workflow
            ):
                raise ValueError(
                    "Resume rejected: finite-field engine changed, workflow implementation changed, "
                    "or a fingerprint is unavailable; "
                    "start a fresh run. Completed artifacts remain renderable."
                )
    store = RunStore(root, config.as_dict(), files, resume=resume)
    logger.info("Run directory: %s", root)
    with owned_run(store):
        if config.workflow in FINITE_FIELD_NAMES:
            from ..finite_field.config import resolve_options

            source = load_finite_field_source(config)
            scientific_options = resolve_options(_options(config.parameters))
            previous_science = store.manifest.get("scientific_model", {})
            store.manifest["scientific_model"] = {
                **previous_science,
                "model": config.inputs.get(
                    "model", "family" if "family_manifest" in config.inputs else "grid"
                ),
                "preset": previous_science.get("preset", scientific_options.label),
                "requested_options": scientific_options.to_dict(),
                "source_sha256": finite_field_engine,
                "workflow_sha256": finite_field_workflow,
                "conventions": "Cartesian field plane; k in Angstrom^-1; eV; fields V/m; sheet current A/m",
            }
            store.save()
            with execution_session(config.execution):
                calculate_finite_field(config, source, store)
        elif config.workflow in SERVICE_SPECS:
            spec = SERVICE_SPECS[config.workflow]
            store.manifest["scientific_model"] = {
                "input_kind": spec.input_kind,
                "checkpoint_state": spec.checkpoint_kind,
                "required_products": list(spec.required_products),
                "provider": config.parameters.get("provider"),
                "provider_identity": config.inputs.get("provider_identity"),
                "conventions": "row lattice vectors; Cartesian Å; eV; eV/Å; eV/Å³; amu; fs; K",
            }
            store.save()
            with execution_session(config.execution):
                calculate_service(config, store, provider=force_provider)
        else:
            model = _load_model(config)
            store.manifest["scientific_model"] = {
                "model": config.inputs.get("model", "wannier"),
                "fermi_energy_eV": float(getattr(model, "fermi_energy", config.parameters.get("ef", 0))),
                "spin_degeneracy": float(getattr(model, "spin_degeneracy", 1)),
                "spin_mode": getattr(model, "spin_mode", "unspecified"),
                "conventions": (
                    "EPW reduced coupling; phonons in meV"
                    if config.workflow == "epw"
                    else "positive Fourier phase; fractional reciprocal k; eV; angstrom"
                ),
                "definitions": getattr(model, "metadata", {}),
            }
            store.save()
            with execution_session(config.execution):
                _calculate(config, model, store)
        if not store.manifest["cases"]:
            raise ValueError("Configuration produced no numerical cases")
        if config.workflow in SERVICE_SPECS and any(
            not store.completed(case) for case in SERVICE_SPECS[config.workflow].required_products
        ):
            raise RuntimeError("Service workflow lacks a required final product")
    return root
