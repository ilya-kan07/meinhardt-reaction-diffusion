from pathlib import Path
import sys


def get_base_path() -> Path:
    """
    Возвращает корень, откуда берутся resources/ и results/.

    * В обычном режиме (python run_solver.py) – корень проекта.
    * В exe (PyInstaller) – временная папка _MEIPASS.
    """
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    else:
        return Path(__file__).resolve().parent.parent.parent.parent


def get_resource_path(filename: str) -> Path:
    return get_base_path() / "resources" / "solver" / filename


def get_results_dir() -> Path:
    path = get_base_path() / "results"
    path.mkdir(exist_ok=True, parents=True)
    return path
