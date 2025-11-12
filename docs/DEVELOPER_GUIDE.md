# Руководство для разработчиков

## Структура проекта (monorepo)

meinhardt-reaction-diffusion/
├── solver/                                 # Программа для численного решения
│   ├── src/
|   |   ├── config/
|   |   |   └── presets.py                  # Наборы параметров
|   |   ├── core/
|   |   |   ├── initial_conditions.py       # Вычисление начальных условий
|   |   |   └── solver.py                   # Явная схема для численного решения
|   |   ├── gui/
|   |   ├── utils/
|   |   |   ├── parameter_reader.py         # Чтение параметров из файла
|   |   |   └── paths.py                    # Пути для сохранения и чтения данных
│   │   └── solver_main.py                  # GUI + основная логика
│   └── tests/                              # Unit-тесты
│       ├── test_initial_conditions.py      # Проверка начальных условий
|       └── test_solver.py                  # Проверка численного решения
├── analysis/                       # Программа для аналитического исследования
|   ├── src/
|   |   ├── config/
|   |   |   └── parameters.py       # Основные параметры
|   |   ├── core/
|   |   |   ├── graphs.py           # Построение графиков
|   |   |   ├── newton.py           # Метод Ньютона
|   |   |   └── stability.py        # Анализ устойчивости равновесия
|   |   └── analysis_main.py        # GUI и основная логика
│   └── tests/                      # Unit-тесты
|       ├── test_graphs.py          # Проверка координат точек пересечения
|       ├── test_newton.py          # Проверка метода Ньютона
│       └── test_stability.py       # Проверка коэффициентов полинома
├── docs/                    # Документация
|   ├── images/
|   ├── DEVELOPER_GUIDE.md
|   ├── REPORT.md
|   └── USER_GUIDE.md
├── resources/               # Ресурсы для программ
|   ├── conditions.png
|   ├── main_conditions.png
|   └── parameters.txt
├── requirements.txt            # Зависимости
├── pytest.ini
├── .github/workflows/tests.yml
├── .gitignore
├── LICENSE
├── README.md
├── run_solver.py            # Точка входа (solver)
└── run_analysis.py          # Точка входа (analysis)


## Описание ключевых файлов

### `solver/` — Численное решение

| Файл | Описание |
|------|---------|
| `src/config/presets.py` | Предустановленные наборы параметров (сетка, гармоники, реагенты) |
| `src/core/initial_conditions.py` | Вычисление начальных условий как суммы косинусов. Проверка корректности. |
| `src/core/solver.py` | Явная конечно-разностная схема. Решение на базовой и контрольной сетках. |
| `src/utils/parameter_reader.py` | Чтение параметров из `.txt` |
| `src/utils/paths.py` | Формирование путей для сохранения (`.csv`, `.png`) |
| `solver_main.py` | Главный файл: GUI на Tkinter: вкладки, прогресс-бар, анимация, запуск, координация модулей |
| `tests/test_initial_conditions.py` | Проверка суммы косинусов и условий устойчивости |
| `tests/test_solver.py` | Проверка сходимости на двух сетках |

### `analysis/` — Аналитическое исследование

| Файл | Описание |
|------|---------|
| `src/config/parameters.py` | Параметры системы(для точки бифуркации), равновесие, пространственные параметры, начальное приближение в методе Ньютона |
| `src/core/graphs.py` | Построение графиков `a(y)`, `s(a)`, поиск их пересечений |
| `src/core/newton.py` | Метод Ньютона для поиска равновесия. Якобиан, итерации |
| `src/core/stability.py` | Матрица Якоби, характеристический полином, анализ устойчивости, коэффициенты полинома при k=0 и k!=0 |
| `analysis_main.py` | GUI: 3 вкладки, вывод результатов, графики |
| `tests/test_graphs.py` | Проверка координат пересечений |
| `tests/test_newton.py` | Проверка сходимости к известному равновесию |
| `tests/test_stability.py` | Проверка коэффициентов полинома и устойчивости |

### Корень проекта

| Файл | Описание |
|------|---------|
| `run_solver.py` | Точка входа: `from solver.src.solver_main import main` |
| `run_analysis.py` | Точка входа: `from analysis.src.analysis_main import main` |
| `requirements.txt` | `numpy`, `matplotlib`, `pytest` |
| `pytest.ini` | `pythonpath = .`, `testpaths = solver/tests analysis/tests` |
| `.github/workflows/tests.yml` | CI: запуск всех тестов |
| `LICENSE` | MIT License |
| `.gitignore` | Исключение `venv/`, `__pycache__/`, `build/` |


## Как внести изменения

1. **Создай ветку**:
```bash
   git checkout -b feature/new-stability-check
```
2. **Добавь код + тест**
3. **Запусти тесты**:
 ```bash
   pytest -v
```
3. **Запусти тесты**:
```bash
    git add docs/DEVELOPER_GUIDE.md
    git commit -m "docs: add detailed file descriptions to developer guide"
    git push
```
