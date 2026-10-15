"""Compatibility wrappers for relocated artifact renderers."""


def _legacy(function, root, manifest, output):
    from ..plotting import legacy_artifacts

    return [path.name for path in getattr(legacy_artifacts, function)(root, manifest, output)]


def render_bands(root, manifest, output):
    return _legacy("render_bands", root, manifest, output)


def render_response(root, manifest, output):
    return _legacy("render_response", root, manifest, output)


def render_spectral(root, manifest, output):
    return _legacy("render_spectral", root, manifest, output)


def render_momentum(root, manifest, output):
    return _legacy("render_momentum", root, manifest, output)


def render_observable(root, manifest, output):
    from ..plotting.equilibrium_artifacts import render

    return [path.name for path in render(root, manifest, output)]
