"""
Пути для программы bifurcation (режим разработки и режим exe).

* Обычный запуск (python run_bifurcation.py) — корень проекта.
* Запуск из exe (PyInstaller) — папка, где лежит сам .exe.

Логика намеренно совпадает с solver/src/database.py, чтобы БД и папка
results/ находились в одном и том же месте для обеих программ.
"""
import sys
from pathlib import Path


def is_bundled() -> bool:
    """True, если программа запущена из PyInstaller-бандла (exe)."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def get_app_dir() -> Path:
    """Папка, рядом с которой хранятся БД и результаты."""
    if is_bundled():
        return Path(sys.executable).parent
    # bifurcation/paths.py -> bifurcation -> корень проекта
    return Path(__file__).resolve().parent.parent


def get_results_dir() -> Path:
    """Папка results/ рядом с exe (или в корне проекта). Создаётся при обращении."""
    path = get_app_dir() / "results"
    path.mkdir(exist_ok=True, parents=True)
    return path
