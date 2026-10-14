import sys
from pathlib import Path

from bifurcation import paths


def test_dev_mode_uses_project_root(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)

    project_root = Path(paths.__file__).resolve().parent.parent
    assert not paths.is_bundled()
    assert paths.get_app_dir() == project_root


def test_bundled_mode_uses_exe_dir(monkeypatch, tmp_path):
    fake_exe = tmp_path / "MeinhardtBifurcation.exe"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "_MEI123"), raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_exe))

    assert paths.is_bundled()
    assert paths.get_app_dir() == tmp_path
    results = paths.get_results_dir()
    assert results == tmp_path / "results"
    assert results.is_dir()


def test_db_location_matches_paths(monkeypatch):
    """БД solver и папка результатов bifurcation должны лежать рядом."""
    from solver.src import database

    monkeypatch.delattr(sys, "frozen", raising=False)
    assert database.DB_PATH.parent == paths.get_app_dir()
