import sys
from pathlib import Path

from solver.src.utils import paths


def test_results_dir_dev_mode_is_project_root(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)

    project_root = Path(paths.__file__).resolve().parent.parent.parent.parent
    assert paths.get_results_dir() == project_root / "results"


def test_results_dir_bundled_is_next_to_exe_not_meipass(monkeypatch, tmp_path):
    exe_dir = tmp_path / "app"
    exe_dir.mkdir()
    meipass = tmp_path / "_MEI123"
    meipass.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(meipass), raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_dir / "MeinhardtSolver.exe"))

    results = paths.get_results_dir()
    assert results == exe_dir / "results"
    assert results.is_dir()
    assert not (meipass / "results").exists()


def test_resources_still_come_from_meipass(monkeypatch, tmp_path):
    """Ресурсы (картинки, иконка) по-прежнему берутся из бандла."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "x" / "MeinhardtSolver.exe"))

    assert paths.get_resource_path("conditions.png") == \
        tmp_path / "resources" / "solver" / "conditions.png"
