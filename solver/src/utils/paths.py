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
    """
    Папка для результатов расчётов (results/).

    * В обычном режиме – results/ в корне проекта.
    * В exe – results/ рядом с самим .exe (как и БД, см. database.py).
      _MEIPASS здесь использовать нельзя: это временная папка,
      она удаляется при закрытии программы вместе с результатами.
    """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        base = Path(sys.executable).parent
    else:
        base = get_base_path()
    path = base / "results"
    path.mkdir(exist_ok=True, parents=True)
    return path
