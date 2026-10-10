"""Installed callers select storage without relying on checkout metadata."""

from queer.paths import data_path, project_root, results_path


def test_defaults_follow_working_directory_outside_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("QUEER_PROJECT_ROOT", "QUEER_DATA_DIR", "QUEER_RESULTS_DIR"):
        monkeypatch.delenv(name, raising=False)
    assert project_root() == tmp_path
    assert data_path("inputs") == tmp_path / "data" / "inputs"
    assert results_path("run") == tmp_path / "results" / "run"


def test_source_archive_discovery_does_not_require_git(tmp_path, monkeypatch):
    (tmp_path / "src/queer").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[build-system]\n")
    monkeypatch.delenv("QUEER_PROJECT_ROOT", raising=False)
    assert project_root(tmp_path / "src/queer") == tmp_path


def test_bundled_style_restores_global_parameters():
    import matplotlib as mpl

    from queer.plotting.style import style_context

    original = mpl.rcParams["lines.linewidth"]
    with style_context():
        assert mpl.rcParams["lines.linewidth"] == 31
    assert mpl.rcParams["lines.linewidth"] == original
