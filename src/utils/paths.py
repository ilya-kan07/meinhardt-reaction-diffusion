from pathlib import Path
import sys
from typing import Final


def get_base_path() -> Path:
    """
    Возвращает корень проекта (где лежит resources/, results/)
    """
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent

    current_path = Path(__file__).resolve()
    project_root = current_path.parent.parent.parent
    return project_root


def get_resource_path(filename: str) -> Path:
    return get_base_path() / "resources" / filename


def get_results_dir() -> Path:
    path = get_base_path() / "results"
    path.mkdir(exist_ok=True, parents=True)
    return path
