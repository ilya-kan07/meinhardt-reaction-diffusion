import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from bifurcation.compute import compute_bifurcation, find_equilibrium
from bifurcation.analysis import find_intersections_general
from bifurcation.initial_conditions import compute_initial_conditions
from bifurcation.solver_core import MeinhardtSolver
# Общая БД с solver: в exe лежит рядом с .exe, при запуске из терминала — в корне проекта
from solver.src.database import init_db, get_conn, numpy_to_bytes
import numpy as np
import threading

# ========================= ПАРАМЕТРЫ =========================
SYSTEM_PARAMS = [
    ("c", 16.67), ("mu", 1.2), ("c0", 1.128), ("nu", 0.33),
    ("eps", 3.3), ("d", 0.023), ("e", 1.67), ("f", 9.0),
    ("eta", 16.67), ("Da", 0.0620036278), ("Ds", 1.0), ("Dy", 1.0),
]

SPATIAL_PARAMS = [("L", 1.0)]

param_names = [p for p, _ in SYSTEM_PARAMS]


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Бифуркационная диаграмма Тьюринга")
        self.root.geometry("1500x950")

        # === Главный Notebook с вкладками ===
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        # Вкладка 1 — Бифуркационная диаграмма
        self.tab_bif = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_bif, text="Бифуркационная диаграмма")
        self.create_bifurcation_tab(self.tab_bif)

        # Вкладка 2 — Новый расчет
        self.tab_new_calc = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_new_calc, text="Автоматизация расчётов")
        self.create_new_calc_tab(self.tab_new_calc)

        # Вкладка 3 — Анализ параметров
        self.tab_analysis = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_analysis, text="Анализ параметров")
        self.create_analysis_tab(self.tab_analysis)

    def collect_params(self):
        """Собирает параметры системы и пространственные параметры"""
        params = {p: float(entry.get())
                  for p, entry in self.system_entries.items()}
        params.update({p: float(entry.get())
                      for p, entry in self.spatial_entries.items()})

        # Диапазон k (приводим к целым числам)
        k_start = int(round(float(self.k_start_entry.get())))
        k_end = int(round(float(self.k_end_entry.get())))

        # Защита: если пользователь ввёл от > до
        if k_start > k_end:
            k_start, k_end = k_end, k_start

        params['k_start'] = k_start
        params['k_end'] = k_end
        params['k_step'] = 1.0   # фиксированный шаг = 1

        return params

    def create_bifurcation_tab(self, parent):
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill="both", expand=True)

        left_frame = ttk.Frame(main_frame, width=420)
        left_frame.pack(side="left", fill="y", padx=12, pady=10)

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True)

        # === Параметры системы ===
        system_frame = ttk.LabelFrame(
            left_frame, text="Параметры системы", padding=10)
        system_frame.pack(fill="x", pady=6)

        self.system_entries = {}
        for idx, (param, value) in enumerate(SYSTEM_PARAMS):
            col = idx // 6
            row = idx % 6
            ttk.Label(system_frame, text=f"{param}:").grid(
                row=row, column=col*2, padx=4, pady=3, sticky="e")
            entry = ttk.Entry(system_frame, width=13)
            entry.insert(0, str(value))
            entry.grid(row=row, column=col*2 + 1, padx=4, pady=3)
            self.system_entries[param] = entry

        # === Пространственные параметры ===
        spatial_frame = ttk.LabelFrame(
            left_frame, text="Пространственные параметры", padding=10)
        spatial_frame.pack(fill="x", pady=6)

        self.spatial_entries = {}

        # Параметр L
        ttk.Label(spatial_frame, text="L:").grid(
            row=0, column=0, padx=4, pady=5, sticky="e")
        entry_L = ttk.Entry(spatial_frame, width=13)
        entry_L.insert(0, "1.0")
        entry_L.grid(row=0, column=1, padx=4, pady=5)
        self.spatial_entries["L"] = entry_L

        # Параметр k (от и до)
        ttk.Label(spatial_frame, text="k от:").grid(
            row=1, column=0, padx=4, pady=5, sticky="e")
        self.k_start_entry = ttk.Entry(spatial_frame, width=13)
        self.k_start_entry.insert(0, "1")
        self.k_start_entry.grid(row=1, column=1, padx=4, pady=5)

        ttk.Label(spatial_frame, text="k до:").grid(
            row=2, column=0, padx=4, pady=5, sticky="e")
        self.k_end_entry = ttk.Entry(spatial_frame, width=13)
        self.k_end_entry.insert(0, "5")
        self.k_end_entry.grid(row=2, column=1, padx=4, pady=5)

        # Настройки диаграммы
        axes_frame = ttk.LabelFrame(
            left_frame, text="Настройки диаграммы", padding=10)
        axes_frame.pack(fill="x", pady=10)

        ttk.Label(axes_frame, text="Изменяемый параметр:").grid(
            row=0, column=0, sticky="e", padx=5, pady=4)
        self.param1_box = ttk.Combobox(
            axes_frame, values=param_names, width=12)
        self.param1_box.set("c0")
        self.param1_box.grid(row=0, column=1, padx=5)

        ttk.Label(axes_frame, text="Подбираемый параметр:").grid(
            row=1, column=0, sticky="e", padx=5, pady=4)
        self.param2_box = ttk.Combobox(
            axes_frame, values=param_names, width=12)
        self.param2_box.set("Da")
        self.param2_box.grid(row=1, column=1, padx=5)

        # Диапазон
        ttk.Label(axes_frame, text="От:").grid(
            row=2, column=0, sticky="e", padx=5, pady=4)
        self.range_start_entry_bif = ttk.Entry(axes_frame, width=12)
        self.range_start_entry_bif.insert(0, "1.128")
        self.range_start_entry_bif.grid(row=2, column=1, padx=5)

        ttk.Label(axes_frame, text="До:").grid(
            row=3, column=0, sticky="e", padx=5, pady=4)
        self.range_end_entry_bif = ttk.Entry(axes_frame, width=12)
        self.range_end_entry_bif.insert(0, "2.528")
        self.range_end_entry_bif.grid(row=3, column=1, padx=5)

        ttk.Label(axes_frame, text="Количество точек:").grid(
            row=4, column=0, sticky="e", padx=5, pady=4)
        self.points_entry_bif = ttk.Entry(axes_frame, width=12)
        self.points_entry_bif.insert(0, "50")
        self.points_entry_bif.grid(row=4, column=1, padx=5)

        ttk.Button(left_frame, text="Построить диаграмму",
                   command=self.run).pack(pady=20)

        # Кнопка сохранения графика
        save_btn_frame = ttk.Frame(left_frame)
        save_btn_frame.pack(pady=(5, 15))

        ttk.Button(save_btn_frame, text="Сохранить график как PNG",
                   command=self.save_bifurcation_plot).pack(side="left", padx=5)

        # График и таблица (правая часть)
        self.fig, self.ax = plt.subplots(figsize=(8.8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=False, pady=(0, 8))

        # === Таблица ===
        table_frame = ttk.LabelFrame(
            right_frame, text="Таблица рассчитанных точек", padding=6)
        table_frame.pack(fill="both", expand=True)

        columns = ("k", "n", "param1", "param2", "equilibrium",
                   "bk_values", "b_k0", "status")

        self.tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", height=14)

        # Заголовки столбцов
        self.tree.heading("k", text="k")
        self.tree.heading("n", text="№")
        self.tree.heading("param1", text="Изменяемый")
        self.tree.heading("param2", text="Подбираемый")
        self.tree.heading("equilibrium", text="Равновесие (a, s, y)")
        self.tree.heading("bk_values", text="Коэффициенты bk")
        self.tree.heading("b_k0", text="Коэффициенты b (k=0)")
        self.tree.heading("status", text="Статус")

        # Настройка ширины столбцов
        self.tree.column("k", width=50, anchor="center")
        self.tree.column("n", width=45, anchor="center")
        self.tree.column("param1", width=115, anchor="center")
        self.tree.column("param2", width=115, anchor="center")
        self.tree.column("equilibrium", width=195, anchor="w")
        self.tree.column("bk_values", width=195, anchor="w")
        self.tree.column("b_k0", width=165, anchor="w")
        self.tree.column("status", width=180, anchor="w")

        style = ttk.Style()
        style.configure("Treeview", rowheight=50)

        vsb = ttk.Scrollbar(table_frame, orient="vertical",
                            command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)

        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")


    # ====================== ВТОРАЯ ВКЛАДКА (Новый расчет) ======================
    def create_new_calc_tab(self, parent):
        """Создает интерфейс второй вкладки 'Новый расчет'"""
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill="both", expand=True)

        left_frame = ttk.Frame(main_frame, width=420)
        left_frame.pack(side="left", fill="y", padx=12, pady=10)

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True)

        # === 1. Параметры системы ===
        system_frame = ttk.LabelFrame(
            left_frame, text="Параметры системы", padding=10)
        system_frame.pack(fill="x", pady=6)

        self.new_calc_system_entries = {}
        for idx, (param, value) in enumerate(SYSTEM_PARAMS):
            col = idx // 6
            row = idx % 6
            ttk.Label(system_frame, text=f"{param}:").grid(
                row=row, column=col*2, padx=4, pady=3, sticky="e")
            entry = ttk.Entry(system_frame, width=13)
            entry.insert(0, str(value))
            entry.grid(row=row, column=col*2 + 1, padx=4, pady=3)
            self.new_calc_system_entries[param] = entry

        # === 2. Пространственные параметры ===
        spatial_frame = ttk.LabelFrame(
            left_frame, text="Пространственные параметры", padding=10)
        spatial_frame.pack(fill="x", pady=6)

        self.new_calc_spatial_entries = {}

        ttk.Label(spatial_frame, text="L:").grid(
            row=0, column=0, padx=4, pady=5, sticky="e")
        entry_L = ttk.Entry(spatial_frame, width=13)
        entry_L.insert(0, "1.0")
        entry_L.grid(row=0, column=1, padx=4, pady=5)
        self.new_calc_spatial_entries["L"] = entry_L

        ttk.Label(spatial_frame, text="k:").grid(
            row=1, column=0, padx=4, pady=5, sticky="e")
        entry_k = ttk.Entry(spatial_frame, width=13)
        entry_k.insert(0, "1")
        entry_k.grid(row=1, column=1, padx=4, pady=5)
        self.new_calc_spatial_entries["k"] = entry_k

        # === 3. Параметры расчёта ===
        calc_frame = ttk.LabelFrame(
            left_frame, text="Параметры расчёта", padding=10)
        calc_frame.pack(fill="x", pady=6)

        ttk.Label(calc_frame, text="Значение c0:").grid(
            row=0, column=0, padx=4, pady=5, sticky="e")
        self.new_calc_c0_entry = ttk.Entry(calc_frame, width=13)
        self.new_calc_c0_entry.insert(0, "1.128")
        self.new_calc_c0_entry.grid(row=0, column=1, padx=4, pady=5)

        ttk.Label(calc_frame, text="Отступ:").grid(
            row=1, column=0, padx=4, pady=5, sticky="e")
        self.new_calc_offset_entry = ttk.Entry(calc_frame, width=13)
        self.new_calc_offset_entry.insert(0, "0.001")
        self.new_calc_offset_entry.grid(row=1, column=1, padx=4, pady=5)

        # Ввод параметров дискретизации сетки для решателя
        ttk.Label(calc_frame, text="n (сетка по x):").grid(
            row=2, column=0, padx=4, pady=5, sticky="e")
        self.new_calc_n_entry = ttk.Entry(calc_frame, width=13)
        self.new_calc_n_entry.insert(0, "10")
        self.new_calc_n_entry.grid(row=2, column=1, padx=4, pady=5)

        ttk.Label(calc_frame, text="m (сетка по t):").grid(
            row=3, column=0, padx=4, pady=5, sticky="e")
        self.new_calc_m_entry = ttk.Entry(calc_frame, width=13)
        self.new_calc_m_entry.insert(0, "1000")
        self.new_calc_m_entry.grid(row=3, column=1, padx=4, pady=5)

        ttk.Label(calc_frame, text="T (время):").grid(
            row=4, column=0, padx=4, pady=5, sticky="e")
        self.new_calc_T_entry = ttk.Entry(calc_frame, width=13)
        self.new_calc_T_entry.insert(0, "1.0")
        self.new_calc_T_entry.grid(row=4, column=1, padx=4, pady=5)

        # === НОВОЕ: Выбор режима расчёта (8 или 2 точки) ===
        points_mode_frame = ttk.LabelFrame(
            left_frame, text="Режим расчёта окрестностей", padding=10)
        points_mode_frame.pack(fill="x", pady=6)

        self.points_mode = tk.StringVar(value="8")

        ttk.Radiobutton(points_mode_frame,
                        text="8 точек (полная окрестность)",
                        variable=self.points_mode, value="8").pack(anchor="w", padx=8, pady=3)
        ttk.Radiobutton(points_mode_frame,
                        text="2 точки (только сверху и снизу)",
                        variable=self.points_mode, value="2").pack(anchor="w", padx=8, pady=3)

        # === 4. Индикатор прогресса ===
        self.progress_title = "Прогресс вычислений (24 расчета)"  # значение по умолчанию
        progress_frame = ttk.LabelFrame(
            left_frame, text=self.progress_title, padding=10)
        progress_frame.pack(fill="x", pady=6)

        self.new_calc_progress_bar = ttk.Progressbar(
            progress_frame, orient="horizontal", mode="determinate", maximum=24)
        self.new_calc_progress_bar.pack(fill="x", pady=5)

        self.new_calc_progress_label = ttk.Label(
            progress_frame, text="Выполнено: 0 / 24 расчетов\nОсталось: 24 расчетов")
        self.new_calc_progress_label.pack(anchor="w", pady=5)

        # === 5. Кнопки управления ===
        ttk.Button(left_frame, text="Рассчитать",
                   command=self.run_new_calculation).pack(pady=10, fill="x")

        ttk.Button(left_frame, text="Сохранить расчёты",
                   command=self.save_new_calculation).pack(pady=5, fill="x")

        # === ПРАВАЯ ЧАСТЬ (График и Таблица) ===
        # Блок радиокнопок для переключения выводимых реагентов
        plot_control_frame = ttk.LabelFrame(right_frame, text="Отображаемый реагент", padding=5)
        plot_control_frame.pack(fill="x", pady=(0, 5))

        self.new_calc_plot_mode = tk.StringVar(value="a")

        ttk.Radiobutton(plot_control_frame, text="Реагент a(x)", variable=self.new_calc_plot_mode,
                        value="a", command=self.update_new_calc_plot).pack(side="left", padx=20)
        ttk.Radiobutton(plot_control_frame, text="Реагент s(x)", variable=self.new_calc_plot_mode,
                        value="s", command=self.update_new_calc_plot).pack(side="left", padx=20)
        ttk.Radiobutton(plot_control_frame, text="Реагент y(x)", variable=self.new_calc_plot_mode,
                        value="y", command=self.update_new_calc_plot).pack(side="left", padx=20)

        # Область рисования графика
        self.new_calc_fig, self.new_calc_ax = plt.subplots(figsize=(8.8, 5))
        self.new_calc_canvas = FigureCanvasTkAgg(self.new_calc_fig, master=right_frame)
        self.new_calc_canvas.get_tk_widget().pack(fill="both", expand=False, pady=(0, 8))

        # Заглушка на график
        self.new_calc_ax.text(0.5, 0.5, "Нажмите «Рассчитать» для запуска вычислений\nили выберите гармонику в таблице для построения",
                              ha='center', va='center', fontsize=12, color='gray')
        self.new_calc_ax.set_axis_off()
        self.new_calc_canvas.draw()

        # Таблица расчётов
        table_frame = ttk.LabelFrame(
            right_frame, text="Результаты вычислений (кликните на гармонику для просмотра графика)", padding=6)
        table_frame.pack(fill="both", expand=True)

        columns = ("num", "desc", "c0", "da_crit", "da", "equilibrium", "amplitude", "time")
        self.new_calc_tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", height=10)

        self.new_calc_tree.heading("num", text="№")
        self.new_calc_tree.heading("desc", text="Описание окрестности")
        self.new_calc_tree.heading("c0", text="Значение c0")
        self.new_calc_tree.heading("da_crit", text="Da(crit)")
        self.new_calc_tree.heading("da", text="Da")
        self.new_calc_tree.heading("equilibrium", text="равновесие(a,s,y)")
        self.new_calc_tree.heading("amplitude", text="Амплитуда")
        self.new_calc_tree.heading("time", text="Время формирования")

        self.new_calc_tree.column("num", width=60, anchor="center")
        self.new_calc_tree.column("desc", width=180, anchor="w")
        self.new_calc_tree.column("c0", width=110, anchor="center")
        self.new_calc_tree.column("da_crit", width=110, anchor="center")
        self.new_calc_tree.column("da", width=110, anchor="center")
        self.new_calc_tree.column("equilibrium", width=220, anchor="w")
        self.new_calc_tree.column("amplitude", width=180, anchor="w")
        self.new_calc_tree.column("time", width=150, anchor="center")

        # Привязка события клика к выбору строки в таблице
        self.new_calc_tree.bind("<<TreeviewSelect>>", self.on_new_calc_tree_select)

        vsb = ttk.Scrollbar(table_frame, orient="vertical",
                            command=self.new_calc_tree.yview)
        self.new_calc_tree.configure(yscrollcommand=vsb.set)

        self.new_calc_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

    def run_new_calculation(self):
        """Запуск 24-х расчетов в фоновом режиме с предварительной проверкой устойчивости"""
        try:
            params = {p: float(entry.get())
                      for p, entry in self.new_calc_system_entries.items()}
            L = float(self.new_calc_spatial_entries["L"].get())
            k = float(self.new_calc_spatial_entries["k"].get())
            c0_base = float(self.new_calc_c0_entry.get())
            offset = float(self.new_calc_offset_entry.get())
            n_intervals = int(self.new_calc_n_entry.get())
            m1 = int(self.new_calc_m_entry.get())
            T = float(self.new_calc_T_entry.get())
        except ValueError:
            messagebox.showerror("Ошибка ввода", "Пожалуйста, введите корректные числовые значения.")
            return

        # === ВЫБОР РЕЖИМА ===
        mode = self.points_mode.get()

        if mode == "2":
            points_config = [
                {"desc": "Сверху", "dc0": 0, "dda": 1},
                {"desc": "Снизу",  "dc0": 0, "dda": -1}
            ]
            total_calcs = 6
            progress_title = "Прогресс вычислений (6 расчетов)"
        else:
            points_config = [
                {"desc": "Сверху-слева", "dc0": -1, "dda": 1},
                {"desc": "Сверху",       "dc0": 0,  "dda": 1},
                {"desc": "Сверху-справа","dc0": 1,  "dda": 1},
                {"desc": "Слева",        "dc0": -1, "dda": 0},
                {"desc": "Справа",       "dc0": 1,  "dda": 0},
                {"desc": "Снизу-слева",  "dc0": -1, "dda": -1},
                {"desc": "Снизу",        "dc0": 0,  "dda": -1},
                {"desc": "Снизу-справа", "dc0": 1,  "dda": -1}
            ]
            total_calcs = 24
            progress_title = "Прогресс вычислений (24 расчета)"

        n1 = n_intervals + 1
        params['L'] = L
        params['k'] = k

        # Обновляем заголовок и прогресс-бар
        self.progress_title = progress_title
        for widget in self.new_calc_progress_bar.master.winfo_children():
            if isinstance(widget, ttk.LabelFrame) and widget.cget("text").startswith("Прогресс"):
                widget.config(text=progress_title)

        self.new_calc_progress_bar["maximum"] = total_calcs
        self.new_calc_progress_label.config(
            text=f"Выполнено: 0 / {total_calcs} расчетов\nОсталось: {total_calcs} расчетов")

        # Базовый расчет Da(crit)_base
        def calculate_da_crit(c0_val):
            temp_params = params.copy()
            temp_params['c0'] = c0_val
            eq = find_equilibrium(temp_params)
            if eq is None:
                return None, None

            y, a, s = eq
            c_val = temp_params['c']
            mu = temp_params['mu']
            eps = temp_params['eps']
            d = temp_params['d']
            e = temp_params['e']
            f = temp_params['f']
            eta = temp_params['eta']
            nu = temp_params['nu']
            Ds = temp_params['Ds']
            Dy = temp_params['Dy']

            kappa2 = (np.pi * k / L) ** 2

            J11 = 2 * c_val * a * s - mu
            J12 = c_val * a ** 2
            J21 = -2 * c_val * a * s
            J22 = -c_val * a ** 2 - nu - eps * y
            J23 = -eps * s
            J31 = d
            J33 = -e + 2 * eta * y / (1 + f * y ** 2) ** 2

            beta = J22 - kappa2 * Ds
            gamma = J33 - kappa2 * Dy

            denom = kappa2 * beta * gamma
            if abs(denom) < 1e-12:
                return None, eq

            num = J11 * beta * gamma - J12 * J21 * gamma + J12 * J23 * J31
            da_crit = num / denom
            return da_crit, eq

        da_crit_base, _ = calculate_da_crit(c0_base)
        if da_crit_base is None:
            messagebox.showerror("Ошибка", f"Не удалось рассчитать базовое значение Da(crit) для c0 = {c0_base}.")
            return

        # ПРОВЕРКА УСТОЙЧИВОСТИ ПО УСЛОВИЮ КУРАНТА-ФРИДРИХСА-ЛЕВИ (CFL)
        max_da_near = da_crit_base + offset
        Ds = params['Ds']
        Dy = params['Dy']
        max_D = max(max_da_near, Ds, Dy)

        required_m = int(np.ceil(2 * max_D * T * (n_intervals ** 2) / (L ** 2)))
        if m1 < required_m:
            messagebox.showerror(
                "Неустойчивость численной схемы",
                f"Внимание! Нарушено условие устойчивости разностной схемы (условие CFL).\n"
                f"Шаг по времени tau слишком велик для выбранной сетки по x.\n\n"
                f"Для обеспечения устойчивости при n={n_intervals} необходимо задать:\n"
                f"m >= {required_m} (шагов по времени)\n\n"
                f"Пожалуйста, увеличьте значение m в параметрах расчёта и запустите вычисления снова."
            )
            return

        # Выбор конфигурации точек
        mode = self.points_mode.get()

        if mode == "2":
            points_config = [
                {"desc": "Сверху", "dc0": 0, "dda": 1},
                {"desc": "Снизу", "dc0": 0, "dda": -1}
            ]
            total_calcs = 6  # 2 точки × 3 гармоники
            progress_title = "Прогресс вычислений (6 расчетов)"
        else:
            points_config = [
                {"desc": "Сверху-слева", "dc0": -1, "dda": 1},
                {"desc": "Сверху", "dc0": 0, "dda": 1},
                {"desc": "Сверху-справа", "dc0": 1, "dda": 1},
                {"desc": "Слева", "dc0": -1, "dda": 0},
                {"desc": "Справа", "dc0": 1, "dda": 0},
                {"desc": "Снизу-слева", "dc0": -1, "dda": -1},
                {"desc": "Снизу", "dc0": 0, "dda": -1},
                {"desc": "Снизу-справа", "dc0": 1, "dda": -1}
            ]
            total_calcs = 24
            progress_title = "Прогресс вычислений (24 расчета)"

        # Очистка дерева перед расчетом
        for item in self.new_calc_tree.get_children():
            self.new_calc_tree.delete(item)

        # Предварительное создание структуры дерева (родительские и дочерние элементы)
        child_rows_mapping = {}

        for idx, cfg in enumerate(points_config):
            c0_val = c0_base + cfg["dc0"] * offset
            da_val = da_crit_base + cfg["dda"] * offset

            parent_id = self.new_calc_tree.insert(
                "", "end", values=(
                    idx + 1, cfg["desc"], f"{c0_val:.5f}", "вычисляется...", f"{da_val:.6f}", "в процессе...", "", ""
                )
            )

            for h_idx in [1, 2, 3]:
                child_id = self.new_calc_tree.insert(
                    parent_id, "end", values=(
                        "", f"  ├ Гармоника {h_idx}" if h_idx < 3 else f"  └ Гармоника {h_idx}",
                        f"{c0_val:.5f}", "", f"{da_val:.6f}", "", "ожидание...", "ожидание..."
                    )
                )
                child_rows_mapping[(idx, h_idx)] = child_id

        # Очистка графика и сброс индикаторов прогресса
        self.new_calc_ax.clear()
        self.new_calc_ax.text(0.5, 0.5, "Идет расчет 24-х комбинаций...\nПо окончании выберите гармонику для вывода графиков.",
                              ha='center', va='center', fontsize=12, color='darkorange')
        self.new_calc_ax.set_axis_off()
        self.new_calc_canvas.draw()

        self._update_progress_ui(0, 24)

        # Хранение результатов для быстрого отображения по клику
        self.run_results = {}
        self.current_new_calc_selected_item = None

        # Запуск фонового потока
        calc_thread = threading.Thread(
            target=self._run_simulation_thread,
            args=(params, c0_base, da_crit_base, offset, n1, m1, T, L, k, points_config, child_rows_mapping, parent_id),
            daemon=True
        )
        calc_thread.start()

    def _run_simulation_thread(self, params, c0_base, da_crit_base, offset, n1, m1, T, L, k, points_config, child_rows_mapping, parent_id):
        """Фоновый расчет 24 симуляций"""
        total_runs = len(points_config) * 3
        completed = 0

        for cfg_idx, cfg in enumerate(points_config):
            c0_val = c0_base + cfg["dc0"] * offset
            da_val = da_crit_base + cfg["dda"] * offset

            # Копируем параметры и заменяем c0 и Da
            temp_params = params.copy()
            temp_params['c0'] = c0_val
            temp_params['Da'] = da_val

            # Нахождение положения равновесия
            eq_val = find_equilibrium(temp_params)
            if eq_val is None:
                # Если положение равновесия не найдено, обновляем статус строк
                for h_idx in [1, 2, 3]:
                    item_id = child_rows_mapping[(cfg_idx, h_idx)]
                    self._update_tree_item(item_id, ("", f"  └ Гармоника {h_idx}", f"{c0_val:.5f}", "—", f"{da_val:.6f}", "Не найдено", "—", "—"))
                    completed += 1
                    self._update_progress_ui(completed, total_runs)
                continue

            y_eq, a_eq, s_eq = eq_val
            eq_str = f"a={a_eq:.4f}; s={s_eq:.4f}; y={y_eq:.4f}"

            # Расчет локального значения критического Da
            kappa2 = (np.pi * k / L) ** 2
            c_val = temp_params['c']
            mu = temp_params['mu']
            eps = temp_params['eps']
            d = temp_params['d']
            e = temp_params['e']
            f = temp_params['f']
            eta = temp_params['eta']
            nu = temp_params['nu']
            Ds = temp_params['Ds']
            Dy = temp_params['Dy']

            J11 = 2 * c_val * a_eq * s_eq - mu
            J12 = c_val * a_eq ** 2
            J21 = -2 * c_val * a_eq * s_eq
            J22 = -c_val * a_eq ** 2 - nu - eps * y_eq
            J23 = -eps * s_eq
            J31 = d
            J33 = -e + 2 * eta * y_eq / (1 + f * y_eq ** 2) ** 2

            beta = J22 - kappa2 * Ds
            gamma = J33 - kappa2 * Dy
            denom = kappa2 * beta * gamma

            if abs(denom) < 1e-12:
                da_crit_val = None
            else:
                num = J11 * beta * gamma - J12 * J21 * gamma + J12 * J23 * J31
                da_crit_val = num / denom

            da_crit_str = f"{da_crit_val:.6f}" if da_crit_val is not None else "—"

            # Моделирование 3-х гармоник для данной точки
            for h_idx in [1, 2, 3]:
                a0_arr = np.array([a_eq, s_eq, y_eq])
                a1_arr = np.zeros(3)
                a2_arr = np.zeros(3)
                a3_arr = np.zeros(3)

                b1, b2, b3 = 0, 0, 0
                if h_idx == 1:
                    a1_arr = np.array([0.1, 0.1, 0.1])
                    b1 = 1
                elif h_idx == 2:
                    a2_arr = np.array([0.1, 0.1, 0.1])
                    b2 = 2
                elif h_idx == 3:
                    a3_arr = np.array([0.1, 0.1, 0.1])
                    b3 = 3

                # Генерация начальных условий на сетке
                initial_conditions = compute_initial_conditions(
                    a0_arr, a1_arr, a2_arr, a3_arr, b1, b2, b3, n1
                )
                initial_coarse = initial_conditions['coarse']

                # Инициализация и решение системы Мейнхардта
                solver = MeinhardtSolver(
                    c=c_val, mu=mu, c_0=c0_val, nu=nu, epsilon=eps, d=d, e=e, f=f, eta=eta,
                    D_a=da_val, D_s=Ds, D_y=Dy,
                    n_coarse=n1, m_coarse=m1, T=T,
                    initial_coarse=initial_coarse,
                    a0=a0_arr, a1=a1_arr, a2=a2_arr, a3=a3_arr,
                    b1=b1, b2=b2, b3=b3,
                    progress_callback=None
                )

                # Unpack all 7 values returned from MeinhardtSolver.solve() to handle DB saves correctly
                (
                    base_data,
                    control_data,
                    base_data_for_db,
                    control_data_for_db,
                    max_a_diff,
                    max_s_diff,
                    max_y_diff
                ) = solver.solve()

                # Определение амплитуды на последнем слое
                last_layer = base_data[-1]
                amp_a = np.max(last_layer["a"]) - np.min(last_layer["a"])
                amp_s = np.max(last_layer["s"]) - np.min(last_layer["s"])
                amp_y = np.max(last_layer["y"]) - np.min(last_layer["y"])
                amplitude_str = f"a: {amp_a:.3f}; s: {amp_s:.3f}; y: {amp_y:.3f}"

                # Определение времени формирования структуры (только числовые значения, без символа "c")
                forming_time_str = "—"
                tau = T / m1
                for j in range(10, m1 + 1):
                    diff_a = np.max(np.abs(base_data[j]["a"] - base_data[j-10]["a"]))
                    diff_s = np.max(np.abs(base_data[j]["s"] - base_data[j-10]["s"]))
                    diff_y = np.max(np.abs(base_data[j]["y"] - base_data[j-10]["y"]))
                    if max(diff_a, diff_s, diff_y) < 0.00001:
                        forming_time_str = f"{j * tau:.4f}"
                        break

                # Сохраняем результаты для интерактивного графика (включая начальные условия)
                item_id = child_rows_mapping[(cfg_idx, h_idx)]
                self.run_results[item_id] = {
                    "x": last_layer["x"],
                    "a": last_layer["a"],
                    "s": last_layer["s"],
                    "y": last_layer["y"],
                    "init_a": initial_coarse["a"],
                    "init_s": initial_coarse["s"],
                    "init_y": initial_coarse["y"],
                    "point_desc": cfg["desc"],
                    "harmonic_desc": f"Гармоника {h_idx}",
                    # Данные симуляции для сохранения в БД
                    "base_data_for_db": base_data_for_db,
                    "control_data_for_db": control_data_for_db,
                    "max_diffs": (max_a_diff, max_s_diff, max_y_diff),
                    "cfg_idx": cfg_idx,
                    "h_idx": h_idx,
                    "c0_val": c0_val,
                    "da_val": da_val,
                    "a_eq": a_eq,
                    "s_eq": s_eq,
                    "y_eq": y_eq,
                    "n": n1 - 1,
                    "m": m1,
                    "T": T
                }

                # Обновление дочерней строки в Treeview
                child_lbl = f"  ├ Гармоника {h_idx}" if h_idx < 3 else f"  └ Гармоника {h_idx}"
                self._update_tree_item(
                    item_id, ("", child_lbl, f"{c0_val:.5f}", da_crit_str, f"{da_val:.6f}", eq_str, amplitude_str, forming_time_str)
                )

                completed += 1
                self._update_progress_ui(completed, total_runs)

            # После расчета всех 3-х гармоник для родителя убираем статус "вычисляется" в родителе
            parent_item_id = self.new_calc_tree.parent(child_rows_mapping[(cfg_idx, 1)])
            self._update_tree_item(
                parent_item_id, (cfg_idx + 1, cfg["desc"], f"{c0_val:.5f}", da_crit_str, f"{da_val:.6f}", eq_str, "", "")
            )

        self.root.after(0, lambda: self._finalize_calc_ui())

    def on_new_calc_tree_select(self, event):
        """Интерактивное отображение профилей концентраций по клику в таблице"""
        selected_items = self.new_calc_tree.selection()
        if not selected_items:
            return
        self.current_new_calc_selected_item = selected_items[0]
        self.update_new_calc_plot()

    def update_new_calc_plot(self):
        """Перестроение графика для выбранного реагента (сравнение начальных и конечных условий)"""
        item_id = getattr(self, 'current_new_calc_selected_item', None)
        if not item_id or not hasattr(self, 'run_results') or item_id not in self.run_results:
            return

        res = self.run_results[item_id]
        x = res["x"]
        mode = self.new_calc_plot_mode.get()

        # Выбираем массивы данных и цвет в зависимости от выбранного переключателя
        if mode == "a":
            init_vals = res["init_a"]
            final_vals = res["a"]
            color = 'blue'
            name = 'a'
        elif mode == "s":
            init_vals = res["init_s"]
            final_vals = res["s"]
            color = 'green'
            name = 's'
        else:
            init_vals = res["init_y"]
            final_vals = res["y"]
            color = 'red'
            name = 'y'

        point_desc = res["point_desc"]
        harmonic_desc = res["harmonic_desc"]

        self.new_calc_ax.clear()
        self.new_calc_ax.set_axis_on()

        # Отображение начальных условий (пунктиром) и конечной структуры (сплошной линией)
        self.new_calc_ax.plot(x, init_vals, color=color, linestyle='--', linewidth=1.8,
                              label=f'Начальное условие {name}(x, 0)')
        self.new_calc_ax.plot(x, final_vals, color=color, linestyle='-', linewidth=2.5,
                              label=f'Установившееся решение {name}(x, T)')

        self.new_calc_ax.set_xlabel('x', fontsize=11)
        self.new_calc_ax.set_ylabel(f'Концентрация реагента {name}', fontsize=11)
        self.new_calc_ax.set_title(f'Профили распределения реагента {name}: {point_desc} ({harmonic_desc})', fontsize=12)
        self.new_calc_ax.grid(True, alpha=0.35)
        self.new_calc_ax.legend(fontsize=10)

        self.new_calc_fig.tight_layout()
        self.new_calc_canvas.draw()

    def _finalize_calc_ui(self):
        mode = self.points_mode.get()
        count = 6 if mode == "2" else 24
        self.new_calc_ax.clear()
        self.new_calc_ax.text(0.5, 0.5,
            f"Вычисления успешно завершены!\nВыберите гармонику из таблицы снизу.",
            ha='center', va='center', fontsize=12, color='green')
        self.new_calc_ax.set_axis_off()
        self.new_calc_canvas.draw()
        messagebox.showinfo("Готово", f"Все {count} варианта расчётов завершены!")

    def _update_tree_item(self, item_id, values):
        self.root.after(0, lambda: self.new_calc_tree.item(item_id, values=values))

    def _update_progress_ui(self, completed, total):
        def update():
            self.new_calc_progress_bar["value"] = completed
            left = total - completed
            self.new_calc_progress_label.config(
                text=f"Выполнено: {completed} / {total} расчетов\nОсталось: {left} расчетов"
            )
        self.root.after(0, update)

    def save_new_calculation(self):
        """Экспорт результатов всех 24 расчетов в общую SQLite базу данных проекта"""
        if not hasattr(self, 'run_results') or not self.run_results:
            messagebox.showwarning(
                "Предупреждение", "Нет завершенных данных для сохранения. Сначала проведите вычисления!")
            return

        from tkinter import simpledialog
        user_name = simpledialog.askstring(
            "Сохранение в БД", "Введите общее название серии экспериментов:", parent=self.root)
        if not user_name:
            return

        try:
            init_db()
            conn = get_conn()
            cur = conn.cursor()
        except Exception as e:
            messagebox.showerror("Ошибка подключения к БД",
                                 f"Не удалось подключиться к базе данных:\n{e}")
            return

        try:
            sys_params_base = {p: float(entry.get())
                               for p, entry in self.new_calc_system_entries.items()}
        except ValueError:
            messagebox.showerror(
                "Ошибка параметров", "Параметры системы на вкладке заполнены неверно!")
            conn.close()
            return

        # === КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: правильная сортировка ===
        sorted_runs = sorted(
            self.run_results.items(),
            # cfg_idx от 0 до 7, h_idx от 1 до 3
            key=lambda x: (x[1]["cfg_idx"], x[1]["h_idx"])
        )

        saved_count = 0

        for item_id, res in sorted_runs:
            cfg_idx = res["cfg_idx"]
            h_idx = res["h_idx"]
            c0_val = res["c0_val"]
            da_val = res["da_val"]
            a_eq = res["a_eq"]
            s_eq = res["s_eq"]
            y_eq = res["y_eq"]

            # Более понятное имя
            calc_name = f"{cfg_idx + 1}.{h_idx} {user_name}"
            calc_note = f"Гармоника {h_idx} — {res['point_desc']}"

            max_diffs = res["max_diffs"]

            try:
                # 1. calculations
                cur.execute("""
                    INSERT INTO calculations (name, note, max_a_diff, max_s_diff, max_y_diff)
                    VALUES (?, ?, ?, ?, ?)
                """, (calc_name, calc_note, *max_diffs))
                calc_id = cur.lastrowid

                # 2. grid_parameters
                cur.execute(
                    "INSERT INTO grid_parameters VALUES (?, ?, ?, ?)",
                    (calc_id, res["n"], res["m"], res["T"])
                )

                # 3. system_parameters
                cur.execute("""
                    INSERT INTO system_parameters VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    calc_id,
                    sys_params_base["c"], sys_params_base["mu"], c0_val, sys_params_base["nu"],
                    sys_params_base["eps"], sys_params_base["d"], sys_params_base["e"],
                    sys_params_base["f"], sys_params_base["eta"], da_val,
                    sys_params_base["Ds"], sys_params_base["Dy"]
                ))

                # 4. initial_conditions
                a1_coeffs = [0.1, 0.1, 0.1] if h_idx == 1 else [0.0, 0.0, 0.0]
                a2_coeffs = [0.1, 0.1, 0.1] if h_idx == 2 else [0.0, 0.0, 0.0]
                a3_coeffs = [0.1, 0.1, 0.1] if h_idx == 3 else [0.0, 0.0, 0.0]
                b1 = 1 if h_idx == 1 else 0
                b2 = 2 if h_idx == 2 else 0
                b3 = 3 if h_idx == 3 else 0

                cur.execute("""
                    INSERT INTO initial_conditions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    calc_id, b1, b2, b3,
                    a_eq, a1_coeffs[0], a2_coeffs[0], a3_coeffs[0],
                    s_eq, a1_coeffs[1], a2_coeffs[1], a3_coeffs[1],
                    y_eq, a1_coeffs[2], a2_coeffs[2], a3_coeffs[2],
                ))

                # 5. solution_layers
                base_data_for_db = res["base_data_for_db"]
                control_data_for_db = res["control_data_for_db"]

                for base_layer in base_data_for_db:
                    layer_num = base_layer["layer"]
                    x_base = base_layer["x"]
                    a_base = base_layer["a"]
                    s_base = base_layer["s"]
                    y_base = base_layer["y"]

                    control_layer = next(
                        (cl for cl in control_data_for_db if cl["layer"] == layer_num * 4), None)

                    if control_layer is not None:
                        a_control_interp = np.interp(
                            x_base, control_layer["x"], control_layer["a"])
                        s_control_interp = np.interp(
                            x_base, control_layer["x"], control_layer["s"])
                        y_control_interp = np.interp(
                            x_base, control_layer["x"], control_layer["y"])

                        a_diff = np.abs(a_base - a_control_interp)
                        s_diff = np.abs(s_base - s_control_interp)
                        y_diff = np.abs(y_base - y_control_interp)
                    else:
                        a_diff = s_diff = y_diff = None

                    cur.execute("""
                        INSERT INTO solution_layers
                        (calculation_id, layer, is_control, x, a, s, y, a_diff, s_diff, y_diff)
                        VALUES (?, ?, 0, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        calc_id, layer_num,
                        numpy_to_bytes(x_base), numpy_to_bytes(a_base),
                        numpy_to_bytes(s_base), numpy_to_bytes(y_base),
                        numpy_to_bytes(a_diff), numpy_to_bytes(
                            s_diff), numpy_to_bytes(y_diff)
                    ))

                    if control_layer:
                        cur.execute("""
                            INSERT INTO solution_layers
                            (calculation_id, layer, is_control, x, a, s, y)
                            VALUES (?, ?, 1, ?, ?, ?, ?)
                        """, (
                            calc_id, layer_num * 4,
                            numpy_to_bytes(control_layer["x"]),
                            numpy_to_bytes(control_layer["a"]),
                            numpy_to_bytes(control_layer["s"]),
                            numpy_to_bytes(control_layer["y"])
                        ))

                saved_count += 1

            except Exception as e:
                print(f"Ошибка при сохранении {calc_name}: {e}")
                conn.rollback()
                # Продолжаем сохранение остальных

        conn.commit()
        conn.close()

        messagebox.showinfo("Успех",
                            f"Сохранено {saved_count} расчётов из 24\n\nНазвание серии: {user_name}")


    # ====================== ТРЕТЬЯ (АНАЛИЗ ПАРАМЕТРОВ) ВКЛАДКА ======================

    def create_analysis_tab(self, parent):
        """Создаёт интерфейс вкладки 'Анализ параметров'"""
        left_frame = ttk.Frame(parent)
        left_frame.pack(side="left", fill="y", padx=12, pady=10)

        # Параметры системы
        system_frame = ttk.LabelFrame(
            left_frame, text="Параметры системы", padding=10)
        system_frame.pack(fill="x", pady=6)

        self.analysis_system_entries = {}
        for idx, (param, value) in enumerate(SYSTEM_PARAMS):
            col = idx // 6
            row = idx % 6
            ttk.Label(system_frame, text=f"{param}:").grid(
                row=row, column=col*2, padx=4, pady=3, sticky="e")
            entry = ttk.Entry(system_frame, width=13)
            entry.insert(0, str(value))
            entry.grid(row=row, column=col*2 + 1, padx=4, pady=3)
            self.analysis_system_entries[param] = entry

        # Настройки графиков
        settings_frame = ttk.LabelFrame(
            left_frame, text="Настройки графиков", padding=10)
        settings_frame.pack(fill="x", pady=10)

        ttk.Label(settings_frame, text="Варьируемый параметр:").grid(
            row=0, column=0, sticky="e", padx=5, pady=6)
        self.varied_param_box = ttk.Combobox(
            settings_frame, values=param_names, width=15)
        self.varied_param_box.set("c0")
        self.varied_param_box.grid(row=0, column=1, padx=5, pady=6)

        ttk.Label(settings_frame, text="От:").grid(
            row=1, column=0, sticky="e", padx=5, pady=6)
        self.range_start_entry = ttk.Entry(settings_frame, width=12)
        self.range_start_entry.insert(0, "1.128")
        self.range_start_entry.grid(row=1, column=1, padx=5, pady=6)

        ttk.Label(settings_frame, text="До:").grid(
            row=2, column=0, sticky="e", padx=5, pady=6)
        self.range_end_entry = ttk.Entry(settings_frame, width=12)
        self.range_end_entry.insert(0, "2.528")
        self.range_end_entry.grid(row=2, column=1, padx=5, pady=6)

        ttk.Label(settings_frame, text="Количество точек:").grid(
            row=3, column=0, sticky="e", padx=5, pady=6)
        self.analysis_points_entry = ttk.Entry(settings_frame, width=12)
        self.analysis_points_entry.insert(0, "5")
        self.analysis_points_entry.grid(row=3, column=1, padx=5, pady=6)

        ttk.Label(settings_frame, text="Гармоника k:").grid(
            row=4, column=0, sticky="e", padx=5, pady=6)
        self.k_analysis_entry = ttk.Entry(settings_frame, width=12)
        self.k_analysis_entry.insert(0, "0")
        self.k_analysis_entry.grid(row=4, column=1, padx=5, pady=6)

        # Выбор метода
        method_frame = ttk.LabelFrame(
            left_frame, text="Выберите метод анализа", padding=10)
        method_frame.pack(fill="x", pady=10)

        ttk.Label(method_frame, text="Метод:").grid(
            row=0, column=0, sticky="e", padx=5, pady=6)
        self.analysis_method_box = ttk.Combobox(method_frame,
                                                values=[
                                                    "Пересечения a1(y) и a2(y) при разных значениях параметра",
                                                    "Зависимость коэффициентов b1, b3, B от выбранного параметра"
                                                ],
                                                width=55, state="readonly")
        self.analysis_method_box.set(
            "Пересечения a1(y) и a2(y) при разных значениях параметра")
        self.analysis_method_box.grid(row=0, column=1, padx=5, pady=6)

        ttk.Button(left_frame, text="Построить графики",
                   command=self.run_analysis).pack(pady=25)

        # Текстовое поле для логов
        log_frame = ttk.LabelFrame(
            left_frame, text="Результаты расчёта", padding=8)
        log_frame.pack(fill="both", expand=True, pady=(10, 0))

        self.analysis_log = tk.Text(log_frame, height=18, width=45, font=("Consolas", 10),
                                    wrap="word", state="disabled")
        scrollbar = ttk.Scrollbar(
            log_frame, orient="vertical", command=self.analysis_log.yview)
        self.analysis_log.configure(yscrollcommand=scrollbar.set)

        self.analysis_log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Правая часть — график
        right_frame = ttk.Frame(parent)
        right_frame.pack(side="right", fill="both", expand=True)

        self.analysis_fig, self.analysis_ax = plt.subplots(figsize=(9, 7))
        self.analysis_canvas = FigureCanvasTkAgg(
            self.analysis_fig, master=right_frame)
        self.analysis_canvas.get_tk_widget().pack(fill="both", expand=True)

        save_frame2 = ttk.Frame(right_frame)
        save_frame2.pack(pady=5)

        ttk.Button(save_frame2, text="Сохранить график как PNG",
                   command=self.save_analysis_plot).pack(side="left", padx=5)

        self.analysis_ax.text(0.5, 0.5, "Выберите параметры и нажмите\n«Построить графики»",
                              ha='center', va='center', fontsize=12, color='gray')
        self.analysis_ax.set_axis_off()
        self.analysis_canvas.draw()

    def log_analysis(self, text):
        """Вывод информации в текстовую область логов"""
        self.analysis_log.config(state="normal")
        self.analysis_log.insert("end", text + "\n")
        self.analysis_log.see("end")
        self.analysis_log.config(state="disabled")

    def run_analysis(self):
        """Запуск графического анализа"""
        params = {p: float(entry.get())
                  for p, entry in self.analysis_system_entries.items()}
        varied_param = self.varied_param_box.get()
        start = float(self.range_start_entry.get())
        end = float(self.range_end_entry.get())
        n_points = int(self.analysis_points_entry.get())
        k_val = int(self.k_analysis_entry.get())
        method_name = self.analysis_method_box.get()

        self.analysis_log.config(state="normal")
        self.analysis_log.delete("1.0", "end")
        self.analysis_log.config(state="disabled")

        self.analysis_ax.clear()

        if "Пересечения a1" in method_name:
            from bifurcation.analysis import plot_a1_a2_intersections
            plot_a1_a2_intersections(
                params, varied_param, start, end, n_points,
                ax=self.analysis_ax,
                canvas=self.analysis_canvas,
                log_func=self.log_analysis
            )
        elif "коэффициентов b" in method_name.lower() or "b1, b3" in method_name.lower():
            from bifurcation.analysis import plot_b_coefficients_vs_parameter
            plot_b_coefficients_vs_parameter(
                params,
                varied_param=varied_param,
                start=start,
                end=end,
                num_points=n_points,
                k=k_val,
                L=params.get('L', 1.0),
                ax=self.analysis_ax,
                canvas=self.analysis_canvas,
                log_func=self.log_analysis
            )
        else:
            self.log_analysis("Выбранный метод ещё не реализован.")
            self.analysis_ax.text(
                0.5, 0.5, "Метод ещё не реализован", ha='center', va='center', fontsize=14)
            self.analysis_canvas.draw()

    def _fill_table(self, points_data, param1, param2):
        """Заполнение таблицы первой вкладки"""
        for item in self.tree.get_children():
            self.tree.delete(item)

        for i, data in enumerate(points_data):
            a, s, y_eq = data['a'], data['s'], data['y']
            k_val = data.get('k', 1.0)

            eq_text = f"a = {a:.5f}\ns = {s:.5f}\ny = {y_eq:.5f}"

            bk_text = (f"bk1 = {data['bk1']:.6f}\n"
                       f"bk2 = {data['bk2']:.6f}\n"
                       f"bk3 = {data['bk3']:.6f}")

            b_k0_text = (f"b1  = {data.get('b1', 0):.6f}\n"
                         f"b3  = {data.get('b3', 0):.6f}\n"
                         f"b   = {data.get('b', 0):.6f}")

            self.tree.insert("", "end", values=(
                f"{k_val:.1f}",
                i + 1,
                f"{param1} = {data['p1']:.5f}",
                f"{param2} = {data['p2']:.6f}",
                eq_text,
                bk_text,
                b_k0_text,
                data['status']
            ))

    def run(self):
        """Отрисовка бифуркационных кривых первой вкладки"""
        params_base = self.collect_params()
        param1 = self.param1_box.get()
        param2 = self.param2_box.get()

        start_val = float(self.range_start_entry_bif.get())
        end_val = float(self.range_end_entry_bif.get())
        num_points = int(self.points_entry_bif.get())

        if num_points < 2:
            num_points = 2

        step_size = (end_val - start_val) / (num_points - 1)

        k_start = int(params_base['k_start'])
        k_end = int(params_base['k_end'])
        ks = list(range(k_start, k_end + 1))

        if not ks:
            ks = [1]

        print(f"Будет построено {len(ks)} кривых для k = {ks}")

        self.ax.clear()

        colors = plt.cm.tab10(np.linspace(0, 1, len(ks)))
        all_points_data = []

        for idx, k_val in enumerate(ks):
            params = params_base.copy()
            params['k'] = float(k_val)
            params['L'] = params_base.get('L', 1.0)

            for key in ['k_start', 'k_end', 'k_step']:
                params.pop(key, None)

            params[param1] = start_val

            color = colors[idx]
            label = f"k = {k_val:.1f}"

            print(f"\n=== Расчёт для {label} ===")

            x, y, points_data = compute_bifurcation(
                params, param1, param2, step_size, num_points)

            for point in points_data:
                point['k'] = k_val

            all_points_data.extend(points_data)

            if len(y) > 0:
                if idx == 0:
                    y_max = max(y) + 0.005
                    y_min = min(y) - 0.005
                    self.ax.fill_between(x, y_min, y, color='orange', alpha=0.25,
                                         label='Область бифуркации Тьюринга')
                    self.ax.fill_between(x, y, y_max, color='lightblue', alpha=0.28,
                                         label='Область устойчивости')

                positive_mask = y >= 0
                negative_mask = y < 0

                self.ax.plot(x[positive_mask], y[positive_mask], '-',
                             color=color, linewidth=2.8, label=label)

                if np.any(negative_mask):
                    self.ax.plot(x[negative_mask], y[negative_mask], '--',
                                 color=color, linewidth=2.5, alpha=0.85)

                if idx == 0 and k_val == 1:
                    self.ax.scatter(start_val, params[param2], color='red', s=80,
                                    edgecolors='black', linewidth=2.0, zorder=10,
                                    label='Стартовая точка')

        # Оформление графика первой вкладки
        self.ax.set_xlabel(param1, fontsize=13)
        self.ax.set_ylabel(param2, fontsize=13)
        self.ax.set_title(
            f"Бифуркационные диаграммы Тьюринга\n({param1},{param2})",
            fontsize=14)

        self.ax.grid(True, alpha=0.35)
        self.ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.3,
                        alpha=0.8, zorder=1)

        self.ax.legend(loc='upper right', fontsize=10)
        self.fig.tight_layout()
        self.canvas.draw()

        if all_points_data:
            all_points_data.sort(key=lambda d: (d['k'], d['p1']))
            self._fill_table(all_points_data, param1, param2)

    def save_bifurcation_plot(self):
        """Сохраняет график с первой вкладки"""
        if not hasattr(self, 'fig') or self.fig is None:
            messagebox.showwarning("Ошибка", "График ещё не построен")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG файлы", "*.png"), ("Все файлы", "*.*")],
            title="Сохранить график бифуркационной диаграммы"
        )
        if file_path:
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Успешно", f"График сохранён:\n{file_path}")

    def save_analysis_plot(self):
        """Сохраняет график со вкладки Анализ параметров"""
        if not hasattr(self, 'analysis_fig') or self.analysis_fig is None:
            messagebox.showwarning("Ошибка", "График ещё не построен")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG файлы", "*.png"), ("Все файлы", "*.*")],
            title="Сохранить график анализа параметров"
        )
        if file_path:
            self.analysis_fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Успешно", f"График сохранён:\n{file_path}")


# ========================= ЗАПУСК =========================
def main():
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use("clam")

    def on_closing():
        plt.close('all')
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
