"""Compatibility wrapper for the relocated artifact renderer."""


def render_service(root, manifest, output):
    from ..plotting.service_artifacts import render

    return [path.name for path in render(root, manifest, output)]
