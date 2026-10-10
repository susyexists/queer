import numpy as np
import pytest

from queer.geometry.paths import KPath
from queer.plotting.momentum import save_surface_plots, write_animation
from queer.plotting.response import plot_electron_mesh, plot_susceptibility
from queer.plotting.spectral import plot_spectral_path

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")


def test_plotting_uses_supplied_axes_and_does_not_change_global_style(tmp_path):
    before = dict(plt.rcParams)
    fig, ax = plt.subplots()
    result_fig, result_ax = plot_susceptibility(
        np.array([1 + 2j, 2 + 3j]), [0, 1], ["A", "B"], ax=ax, save=tmp_path / "response.pdf"
    )
    assert (result_fig, result_ax) == (fig, ax)
    assert (tmp_path / "response.pdf").exists()
    assert dict(plt.rcParams) == before
    plt.close(fig)
    fig, ax = plt.subplots()
    assert plot_electron_mesh([[1, 2]], [[0, 0, 0], [1, 0, 0]], 0, ax=ax)[1] is ax
    plt.close(fig)


def test_spectral_plot_labels_existing_axes(tmp_path):
    path = KPath(path=np.array([[0.0, 0.0, 0.0], [0.5, 0, 0]]), sym=np.array([0, 1]), labels=["Γ", "X"])
    fig, ax = plt.subplots()
    returned = plot_spectral_path(np.ones((2, 2)), path, [-1, 1], ax=ax, save=tmp_path / "spectral.png")
    assert returned == (fig, ax)
    assert [label.get_text() for label in ax.get_xticklabels()] == ["Γ", "X"]
    plt.close(fig)


def test_surface_styles_save_and_close_figures(tmp_path):
    before = set(plt.get_fignums())
    surface = np.array([[0.0, 0.0, 1.0, -0.1, 1.0], [1.0, 0.0, 1.0, -0.15, 0.9]])
    for style in ("intensity", "scatter_white", "scatter_black_transparent"):
        frames = save_surface_plots({0.1: surface}, tmp_path / style, v0=10, style=style)
        assert len(frames) == 1 and frames[0].is_file()
    assert set(plt.get_fignums()) == before
    with pytest.raises(ValueError, match="style"):
        save_surface_plots({}, tmp_path, 10, style="unknown")


def test_animation_empty_and_variable_frame_dimensions(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    image_module = pytest.importorskip("PIL.Image")
    assert write_animation([], tmp_path) == (None, None)
    paths = []
    for number, size in enumerate(((30, 20), (22, 32))):
        path = tmp_path / f"{number}.png"
        image_module.new("RGB", size, (255, number * 100, 0)).save(path)
        paths.append(path)
    movie, gif = write_animation(paths, tmp_path / "animations", fps=2)
    assert movie.is_file() and gif.is_file()
    with image_module.open(gif) as image:
        assert image.size == (32, 32)


def test_epw_plot_forwards_use_raw_units_and_caller_axes():
    from queer.epw_plots import plot_epw_coupling, plot_phonon_dispersion
    from queer.models.epw import EPWModel
    from queer.plot_param import colors

    model = EPWModel([[10, 20]], [[[1, 3], [2, 4]]], [[0, 0.1], [0.1, 0]], [[0.1, 0], [0, 0.1]])
    fig, ax = plt.subplots()
    assert plot_phonon_dispersion(model, ax=ax)[1] is ax
    assert "meV" in ax.get_ylabel()
    fig2, ax2 = plot_epw_coupling(model)
    np.testing.assert_array_equal(ax2.lines[0].get_ydata(), [2.0, 3.0])
    assert len(colors) == 9
    plt.close(fig)
    plt.close(fig2)
