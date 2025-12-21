import sqlite3
from pathlib import Path
import numpy as np
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
import os
import blosc2
import sys

# Определяем, запущены ли мы из PyInstaller-бандла
IS_BUNDLED = getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')

if IS_BUNDLED:
    # sys.executable — путь к самому exe-файлу
    DB_DIR = Path(sys.executable).parent
else:
    # В разработке — корень проекта (на уровень выше solver/)
    DB_DIR = Path(__file__).resolve().parent.parent.parent

DB_PATH = DB_DIR / "meinhardt_results.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn

def numpy_to_bytes(arr: Optional[np.ndarray]) -> Optional[bytes]:
    if arr is None:
        return None
    # Явно указываем, что данные — float32 (4 байта)
    return blosc2.compress(arr.astype(np.float32).tobytes(), typesize=4)

def bytes_to_numpy(blob: Optional[bytes]) -> Optional[np.ndarray]:
    if blob is None:
        return None
    raw = blosc2.decompress(blob)
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)

def init_db():
    conn = get_conn()
    conn.executescript("""
    -- Основная таблица
    CREATE TABLE IF NOT EXISTS calculations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        note TEXT,
        max_a_diff REAL,
        max_s_diff REAL,
        max_y_diff REAL
    );

    CREATE TABLE IF NOT EXISTS grid_parameters (
        calculation_id INTEGER PRIMARY KEY,
        n INTEGER NOT NULL,
        m INTEGER NOT NULL,
        T REAL NOT NULL,
        FOREIGN KEY(calculation_id) REFERENCES calculations(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS system_parameters (
        calculation_id INTEGER PRIMARY KEY,
        c REAL NOT NULL, mu REAL NOT NULL, c0 REAL NOT NULL, nu REAL NOT NULL,
        eps REAL NOT NULL, d REAL NOT NULL, e REAL NOT NULL, f REAL NOT NULL,
        eta REAL NOT NULL, Da REAL NOT NULL, Ds REAL NOT NULL, Dy REAL NOT NULL,
        FOREIGN KEY(calculation_id) REFERENCES calculations(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS initial_conditions (
        calculation_id INTEGER PRIMARY KEY,
        b1 INTEGER NOT NULL, b2 INTEGER NOT NULL, b3 INTEGER NOT NULL,
        a_A0 REAL NOT NULL, a_A1 REAL NOT NULL, a_A2 REAL NOT NULL, a_A3 REAL NOT NULL,
        s_A0 REAL NOT NULL, s_A1 REAL NOT NULL, s_A2 REAL NOT NULL, s_A3 REAL NOT NULL,
        y_A0 REAL NOT NULL, y_A1 REAL NOT NULL, y_A2 REAL NOT NULL, y_A3 REAL NOT NULL,
        FOREIGN KEY(calculation_id) REFERENCES calculations(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS solution_layers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        calculation_id INTEGER NOT NULL,
        layer INTEGER NOT NULL,
        is_control BOOLEAN NOT NULL,
        x BLOB NOT NULL,
        a BLOB NOT NULL,
        s BLOB NOT NULL,
        y BLOB NOT NULL,
        a_diff BLOB,
        s_diff BLOB,
        y_diff BLOB,
        FOREIGN KEY(calculation_id) REFERENCES calculations(id) ON DELETE CASCADE,
        UNIQUE(calculation_id, layer, is_control)
    );

    -- Индексы для скорости
    CREATE INDEX IF NOT EXISTS idx_layers_calc ON solution_layers(calculation_id);
    CREATE INDEX IF NOT EXISTS idx_calc_name ON calculations(name);
    """)
    conn.commit()
    conn.close()

# Главная функция — сохраняет весь текущий расчёт


def save_calculation(
    name: str,
    parameter_app,           # ParameterApp
    numerical_app,           # NumericalSolutionApp
    base_data: List[Dict],
    control_data: List[Dict],
    max_diffs: Tuple[float, float, float],
    note: Optional[str] = None
):
    init_db()
    conn = get_conn()
    cur = conn.cursor()

    try:
        # 1. calculations
        cur.execute("""
            INSERT INTO calculations (name, note, max_a_diff, max_s_diff, max_y_diff)
            VALUES (?, ?, ?, ?, ?)
        """, (name, note, *max_diffs))
        calc_id = cur.lastrowid

        # 2. grid_parameters
        n = int(parameter_app.entries["n (сетка по x)"].get())
        m = int(parameter_app.entries["m (сетка по t)"].get())
        T = float(parameter_app.entries["T (время)"].get())
        cur.execute(
            "INSERT INTO grid_parameters VALUES (?, ?, ?, ?)", (calc_id, n, m, T))

        # 3. system_parameters
        sys_entries = numerical_app.entries
        cur.execute("""
            INSERT INTO system_parameters VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            calc_id,
            float(sys_entries["c"].get()),
            float(sys_entries["μ"].get()),
            float(sys_entries["c₀"].get()),
            float(sys_entries["ν"].get()),
            float(sys_entries["ε"].get()),
            float(sys_entries["d"].get()),
            float(sys_entries["e"].get()),
            float(sys_entries["f"].get()),
            float(sys_entries["η"].get()),
            float(sys_entries["Dₐ"].get()),
            float(sys_entries["Dₛ"].get()),
            float(sys_entries["Dᵧ"].get()),
        ))

        # 4. initial_conditions
        a0 = [e.get() for e in parameter_app.entries["a_0"]]
        a1 = [e.get() for e in parameter_app.entries["a_1"]]
        a2 = [e.get() for e in parameter_app.entries["a_2"]]
        a3 = [e.get() for e in parameter_app.entries["a_3"]]
        b1 = int(parameter_app.entries["b_1"].get())
        b2 = int(parameter_app.entries["b_2"].get())
        b3 = int(parameter_app.entries["b_3"].get())

        cur.execute("""
            INSERT INTO initial_conditions VALUES (
                ?, ?,?,?,
                ?,?,?,?, ?,?,?,?, ?,?,?,?
            )
        """, (
            calc_id, b1, b2, b3,
            float(a0[0]), float(a1[0]), float(a2[0]), float(a3[0]),
            float(a0[1]), float(a1[1]), float(a2[1]), float(a3[1]),
            float(a0[2]), float(a1[2]), float(a2[2]), float(a3[2]),
        ))

        # 5. solution_layers
        # Подготавливаем соответствие базовых и контрольных слоёв
        for base_layer in base_data:
            layer_num = base_layer["layer"]
            x_base = base_layer["x"]
            a_base = base_layer["a"]
            s_base = base_layer["s"]
            y_base = base_layer["y"]

            # Находим соответствующий контрольный слой
            control_layer = None
            for cl in control_data:
                if cl["layer"] == layer_num * 4:
                    control_layer = cl
                    break

            # Интерполируем контрольную сетку на узлы базовой (для погрешности)
            if control_layer is not None:
                a_control_interp = np.interp(x_base, control_layer["x"], control_layer["a"])
                s_control_interp = np.interp(x_base, control_layer["x"], control_layer["s"])
                y_control_interp = np.interp(x_base, control_layer["x"], control_layer["y"])

                a_diff = np.abs(a_base - a_control_interp)
                s_diff = np.abs(s_base - s_control_interp)
                y_diff = np.abs(y_base - y_control_interp)
            else:
                a_diff = s_diff = y_diff = None

            # Сохраняем базовый слой (с погрешностью)
            cur.execute("""
                INSERT INTO solution_layers
                (calculation_id, layer, is_control, x, a, s, y, a_diff, s_diff, y_diff)
                VALUES (?, ?, 0, ?, ?, ?, ?, ?, ?, ?)
            """, (
                calc_id, layer_num,
                numpy_to_bytes(x_base),
                numpy_to_bytes(a_base),
                numpy_to_bytes(s_base),
                numpy_to_bytes(y_base),
                numpy_to_bytes(a_diff),
                numpy_to_bytes(s_diff),
                numpy_to_bytes(y_diff),
            ))

            # Сохраняем контрольный слой (без погрешности)
            if control_layer is not None:
                cur.execute("""
                    INSERT INTO solution_layers
                    (calculation_id, layer, is_control, x, a, s, y)
                    VALUES (?, ?, 1, ?, ?, ?, ?)
                """, (
                    calc_id, layer_num * 4,
                    numpy_to_bytes(control_layer["x"]),
                    numpy_to_bytes(control_layer["a"]),
                    numpy_to_bytes(control_layer["s"]),
                    numpy_to_bytes(control_layer["y"]),
                ))

        conn.commit()
        return calc_id

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def get_calculation_list() -> List[Dict]:
    """Возвращает список всех сохранённых расчётов для отображения в таблице"""
    init_db()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            c.id, c.name, c.created_at, c.note,
            c.max_a_diff, c.max_s_diff, c.max_y_diff,
            g.n, g.m, g.T
        FROM calculations c
        LEFT JOIN grid_parameters g ON g.calculation_id = c.id
        ORDER BY c.created_at DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def load_calculation(calc_id: int) -> Dict[str, Any]:
    """Загружает ВСЁ по ID расчёта: параметры, начальные условия и все слои"""
    init_db()
    conn = get_conn()
    cur = conn.cursor()

    # Основные данные
    cur.execute("SELECT * FROM calculations WHERE id = ?", (calc_id,))
    calc = dict(cur.fetchone())

    # Сетка
    cur.execute(
        "SELECT * FROM grid_parameters WHERE calculation_id = ?", (calc_id,))
    calc["grid"] = dict(cur.fetchone())

    # Системные параметры
    cur.execute(
        "SELECT * FROM system_parameters WHERE calculation_id = ?", (calc_id,))
    calc["system"] = dict(cur.fetchone())

    # Начальные условия
    cur.execute(
        "SELECT * FROM initial_conditions WHERE calculation_id = ?", (calc_id,))
    init = dict(cur.fetchone())
    calc["initial"] = {
        "b": [init["b1"], init["b2"], init["b3"]],
        "a0": np.array([init["a_A0"], init["s_A0"], init["y_A0"]]),
        "a1": np.array([init["a_A1"], init["s_A1"], init["y_A1"]]),
        "a2": np.array([init["a_A2"], init["s_A2"], init["y_A2"]]),
        "a3": np.array([init["a_A3"], init["s_A3"], init["y_A3"]]),
    }

    # Все слои (и базовые, и контрольные)
    cur.execute("""
        SELECT layer, is_control, x, a, s, y, a_diff, s_diff, y_diff
        FROM solution_layers
        WHERE calculation_id = ?
        ORDER BY layer
    """, (calc_id,))
    layers = []
    for row in cur.fetchall():
        r = dict(row)
        layers.append({
            "layer": r["layer"],
            "is_control": bool(r["is_control"]),
            "x": bytes_to_numpy(r["x"]),
            "a": bytes_to_numpy(r["a"]),
            "s": bytes_to_numpy(r["s"]),
            "y": bytes_to_numpy(r["y"]),
            "a_diff": bytes_to_numpy(r["a_diff"]),
            "s_diff": bytes_to_numpy(r["s_diff"]),
            "y_diff": bytes_to_numpy(r["y_diff"]),
        })
    calc["layers"] = layers

    conn.close()
    return calc


def delete_calculation(calc_id: int):
    """Удаляет расчёт и все связанные данные + сбрасывает автонумерацию, если таблица стала пустой"""
    init_db()
    conn = get_conn()
    cur = conn.cursor()

    try:
        # Удаляем расчёт
        cur.execute("DELETE FROM calculations WHERE id = ?", (calc_id,))
        conn.commit()

        # Проверяем, остались ли вообще расчёты
        cur.execute("SELECT COUNT(*) FROM calculations")
        count = cur.fetchone()[0]

        # Если таблица пуста — сбрасываем счётчик AUTOINCREMENT
        if count == 0:
            # Это магия SQLite: сбрасываем внутренний счётчик
            cur.execute("DELETE FROM sqlite_sequence WHERE name='calculations'")
            conn.commit()

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
