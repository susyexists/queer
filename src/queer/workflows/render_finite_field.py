"""Compatibility wrapper for the relocated artifact renderer."""


def render_finite_field(root, manifest, output):
    from ..plotting.finite_field_artifacts import render

    return [path.name for path in render(root, manifest, output)]
