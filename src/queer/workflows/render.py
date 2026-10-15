"""Optional headless rendering from immutable numerical run artifacts."""

from pathlib import Path

from ..io.artifacts import atomic_json, load_manifest, utc_now
from .catalog import renderer_for


def render_run(run_dir: str | Path) -> Path:
    """Render completed schema-v1 artifacts without loading scientific inputs."""
    root = Path(run_dir).expanduser().resolve()
    manifest = load_manifest(root)
    import matplotlib

    matplotlib.use("Agg")
    output = root / "figures"
    output.mkdir(exist_ok=True)
    files = renderer_for(manifest["workflow"])(root, manifest, output)
    atomic_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "rendered_at": utc_now(),
            "source_fingerprint": manifest["fingerprint"],
            "files": [Path(file).name for file in files],
        },
    )
    return output
