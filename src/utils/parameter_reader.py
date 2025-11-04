import re
from typing import Dict, Any


def read_parameters(filename: str) -> Dict[str, Any]:
    """
    Читает параметры из файла parameters.txt, созданного при сохранении результатов.
    Возвращает словарь с начальными условиями, параметрами сетки и системы.
    """
    params: Dict[str, Any] = {}
    current_section = None

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line in ["Начальные условия:", "Параметры сетки:", "Параметры системы:"]:
                current_section = line
                continue

            if current_section == "Начальные условия:":
                if line.startswith("a_"):
                    # Пример: a_0: a=1.0, s=0.5, y=0.2
                    match = re.match(
                        r"(a_\d+):\s*a=([\d.eE+-]+),\s*s=([\d.eE+-]+),\s*y=([\d.eE+-]+)", line)
                    if match:
                        key = match.group(1)
                        a_val = float(match.group(2))
                        s_val = float(match.group(3))
                        y_val = float(match.group(4))
                        params[key] = [a_val, s_val, y_val]
                    else:
                        raise ValueError(f"Некорректный формат строки: {line}")
                elif line.startswith("b_"):
                    # Пример: b_1: 1
                    key, value = line.split(":", 1)
                    params[key.strip()] = int(value.strip())

            elif current_section == "Параметры сетки:":
                # Пример: n (сетка по x): 10
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key in ["n (сетка по x)", "m (сетка по t)"]:
                    params[key] = int(value)
                else:
                    params[key] = float(value)

            elif current_section == "Параметры системы:":
                # Пример: c: 16.67
                key, value = line.split(":", 1)
                params[key.strip()] = float(value.strip())

    # Проверка наличия всех необходимых параметров
    required_initial = ["a_0", "a_1", "a_2", "a_3", "b_1", "b_2", "b_3"]
    required_grid = ["n (сетка по x)", "m (сетка по t)", "T (время)"]
    required_system = ["c", "μ", "c₀", "ν", "ε",
                       "d", "e", "f", "η", "Dₐ", "Dₛ", "Dᵧ"]

    missing = []
    missing.extend([k for k in required_initial if k not in params])
    missing.extend([k for k in required_grid if k not in params])
    missing.extend([k for k in required_system if k not in params])

    if missing:
        raise ValueError(f"Отсутствуют необходимые параметры: {missing}")

    return params
