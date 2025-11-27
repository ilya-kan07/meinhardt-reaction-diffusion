import sqlite3
from pathlib import Path
import numpy as np
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
import os

# Путь к базе данных — в корне проекта
DB_PATH = Path(__file__).parent.parent.parent / "meinhardt_results.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def numpy_to_bytes(arr: np.ndarray) -> bytes:
    buf = BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def bytes_to_numpy(blob: bytes) -> np.ndarray:
    if blob is None:
        return None
    buf = BytesIO(blob)
    return np.load(buf)


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

            # Ищем соответствующий контрольный слой (layer_num * 4)
            control_layer = next(
                (c for c in control_data if c["layer"] == layer_num * 4), None)
            a_control_interp = control_layer["a"][::
                                                  2] if control_layer is not None else None
            s_control_interp = control_layer["s"][::
                                                  2] if control_layer is not None else None
            y_control_interp = control_layer["y"][::
                                                  2] if control_layer is not None else None

            # Разности только для базовой сетки
            a_diff = np.abs(
                a_base - a_control_interp) if a_control_interp is not None else None
            s_diff = np.abs(
                s_base - s_control_interp) if s_control_interp is not None else None
            y_diff = np.abs(
                y_base - y_control_interp) if y_control_interp is not None else None

            # Сохраняем базовую сетку
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
                numpy_to_bytes(a_diff) if a_diff is not None else None,
                numpy_to_bytes(s_diff) if s_diff is not None else None,
                numpy_to_bytes(y_diff) if y_diff is not None else None,
            ))

            # Сохраняем контрольную сетку (без разностей)
            if control_layer is not None:
                cur.execute("""
                    INSERT INTO solution_layers
                    (calculation_id, layer, is_control, x, a, s, y, a_diff, s_diff, y_diff)
                    VALUES (?, ?, 1, ?, ?, ?, ?, NULL, NULL, NULL)
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
