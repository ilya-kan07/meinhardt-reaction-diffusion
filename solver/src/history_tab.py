# solver/src/history_tab.py
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from solver.src.database import get_calculation_list, load_calculation, delete_calculation

class HistoryTab:
    def __init__(self, notebook):
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text="История")

        self.current_calc = None
        self.setup_ui()
        self.refresh_list()

    def setup_ui(self):
            # === Левая панель ===
        left_frame = ttk.Frame(self.frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5), pady=10)

        # Список расчётов — уменьшаем ширину
        ttk.Label(left_frame, text="Сохранённые расчёты:", font=(
            "Arial", 12, "bold")).pack(anchor="w", pady=(0, 8))

        columns = ("id", "name", "date", "note")
        self.tree = ttk.Treeview(
            left_frame, columns=columns, show="headings", height=10)
        self.tree.heading("id", text="ID")
        self.tree.heading("name", text="Название")
        self.tree.heading("date", text="Дата")
        self.tree.heading("note", text="Заметка")

        self.tree.column("id", width=50, anchor="center")
        self.tree.column("name", width=180)   # было 200
        self.tree.column("date", width=110)   # было 130
        self.tree.column("note", width=180)   # было 220

        self.tree.pack(fill=tk.X, pady=(0, 10))
        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        # Кнопки
        btns = ttk.Frame(left_frame)
        btns.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(btns, text="Обновить",
                   command=self.refresh_list).pack(side=tk.LEFT)
        ttk.Button(btns, text="Удалить", command=self.delete_selected,
                   style="Danger.TButton").pack(side=tk.RIGHT)

        # === Прокручиваемая область для параметров ===
        canvas = tk.Canvas(left_frame, width=420)  # фиксированная ширина
        scrollbar = ttk.Scrollbar(
            left_frame, orient="vertical", command=canvas.yview)
        self.params_frame = ttk.Frame(canvas)

        self.params_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.params_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Правая панель (графики и таблица)
        self.setup_right_panel()

    def setup_right_panel(self):
        right_frame = ttk.Frame(self.frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH,
                         expand=True, padx=(5, 10), pady=10)

        # Выбор слоя — больше не нужен (все слои на графике)
        # ttk.Label(...) и Combobox удаляем

        # === Три графика в ряд ===
        plot_frame = ttk.Frame(right_frame)
        plot_frame.pack(fill=tk.X, pady=(0, 12))

        self.fig, self.axes = plt.subplots(1, 3, figsize=(14, 5.4), dpi=100)
        self.fig.subplots_adjust(
            left=0.06, right=0.98, top=0.88, bottom=0.14, wspace=0.28)

        self.canvas_plot = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.X)

        # === Переключатель: a / s / y ===
        radio_frame = ttk.Frame(right_frame)
        radio_frame.pack(fill=tk.X, pady=(12, 6))

        ttk.Label(radio_frame, text="Показать в таблице:",
                  font=("Arial", 11, "bold")).pack(side=tk.LEFT)
        self.var_function = tk.StringVar(value="a")
        for func, text in [("a", "a(x,t)"), ("s", "s(x,t)"), ("y", "y(x,t)")]:
            ttk.Radiobutton(radio_frame, text=text, variable=self.var_function,
                            value=func, command=self.update_table).pack(side=tk.LEFT, padx=15)

        # === Таблица с новым форматом ===
        table_frame = ttk.Frame(right_frame)
        table_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("layer", "x", "value_base", "value_control", "diff")
        self.table = ttk.Treeview(
            table_frame, columns=cols, show="headings", height=14)
        headers = ("Слой", "x", "Базовая", "Контрольная", "|Δ|")
        widths = (90,     100, 130,      130,         110)
        anchors = ("center", "e", "e", "e", "e")

        for i, col in enumerate(cols):
            self.table.heading(col, text=headers[i])
            # ← ВОТ ГЛАВНОЕ!
            self.table.column(col, width=widths[i], anchor="center")

        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(table_frame, orient="vertical",
                           command=self.table.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.table.configure(yscrollcommand=sb.set)

    def refresh_list(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for calc in get_calculation_list():
            self.tree.insert("", "end", values=(
                calc["id"],
                calc["name"],
                calc["created_at"][:10].replace("T", " "),
                calc["note"] or "—"
            ))

    def on_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        calc_id = self.tree.item(sel[0])["values"][0]
        try:
            self.current_calc = load_calculation(calc_id)
            self.display_parameters()
            # Убрали всё про layer_combo — его больше нет!
            self.show_all_layers()  # ← теперь сразу показываем все слои
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить:\n{e}")

    def display_parameters(self):
        # Очистка
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        c = self.current_calc

        def fmt(val):
            return f"{val:.6g}" if isinstance(val, (int, float)) else str(val)

        # === 1. Начальные условия (самое важное — сверху!) ===
                # === 1. Начальные условия — КОМПАКТНО, КРАСИВО, С НОРМАЛЬНЫМ ШРИФТОМ ===
        init_f = ttk.LabelFrame(self.params_frame, text=" Начальные условия ", padding=(12, 8))
        init_f.pack(fill=tk.X, pady=(0, 12))

        i = c['initial']

        # Заголовки a | s | y
        header = ttk.Frame(init_f)
        header.pack(fill=tk.X, pady=(4, 6))
        tk.Label(header, text="      ", width=5).pack(side=tk.LEFT)
        for title in ["a", "s", "y"]:
            tk.Label(header, text=title, font=("Consolas", 11, "bold"),
                     width=14, anchor="center").pack(side=tk.LEFT, padx=0)

        # A₀ – A₃ — плотные строки, без лишних pady
        coeffs = [("A₀", i['a0']), ("A₁", i['a1']), ("A₂", i['a2']), ("A₃", i['a3'])]
        for name, arr in coeffs:
            row = ttk.Frame(init_f)
            row.pack(fill=tk.X, pady=1)  # минимальный отступ между строками
            tk.Label(row, text=f"{name}:", font=("Consolas", 11), width=5).pack(side=tk.LEFT)
            for val in arr:
                tk.Label(row, text=f"{fmt(val):>13}", font=("Consolas", 11),
                         width=12, anchor="center").pack(side=tk.LEFT, padx=0)

        # b₁ b₂ b₃ — снизу, аккуратно и по центру
        b_frame = ttk.Frame(init_f)
        b_frame.pack(fill=tk.X, pady=(8, 4))
        b1, b2, b3 = i['b']
        tk.Label(b_frame, text=f"b₁ = {b1}  b₂ = {b2}  b₃ = {b3}",
                 font=("Consolas", 11,)).pack(anchor="center")

        # === 2. Параметры сетки ===
        grid_f = ttk.LabelFrame(self.params_frame, text=" Параметры сетки ", padding=10)
        grid_f.pack(fill=tk.X, pady=(0, 12))
        tk.Label(grid_f, text=f"n (узлов): {c['grid']['n']}", font=(
            "Consolas", 11)).pack(anchor="w")
        tk.Label(grid_f, text=f"m (шагов): {c['grid']['m']}", font=(
            "Consolas", 11)).pack(anchor="w")
        tk.Label(grid_f, text=f"T = {fmt(c['grid']['T'])}", font=(
            "Consolas", 11)).pack(anchor="w")

        # === 3. Системные параметры ===
        sys_f = ttk.LabelFrame(
            self.params_frame, text=" Параметры системы ", padding=10)
        sys_f.pack(fill=tk.X, pady=(0, 12))

        s = c['system']

        # Две колонки
        cols_frame = ttk.Frame(sys_f)
        cols_frame.pack(fill=tk.X)

        left_col = ttk.Frame(cols_frame)
        right_col = ttk.Frame(cols_frame)
        left_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
        right_col.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        left_items = [
            ("c   ", s['c']),
            ("μ   ", s['mu']),
            ("c₀  ", s['c0']),
            ("ν   ", s['nu']),
            ("ε   ", s['eps']),
            ("η   ", s['eta']),
        ]
        right_items = [
            ("d   ", s['d']),
            ("e   ", s['e']),
            ("f   ", s['f']),
            ("Dₐ  ", s['Da']),
            ("Dₛ  ", s['Ds']),
            ("Dᵧ  ", s['Dy']),
        ]

        for name, val in left_items:
            tk.Label(left_col, text=f"{name} = {fmt(val)}", font=(
                "Consolas", 11), anchor="w").pack(anchor="w", pady=1)
        for name, val in right_items:
            tk.Label(right_col, text=f"{name} = {fmt(val)}", font=(
                "Consolas", 11), anchor="w").pack(anchor="w", pady=1)

        # === 4. Погрешность ===
        err_f = ttk.LabelFrame(
            self.params_frame, text=" Максимальные разности ", padding=10)
        err_f.pack(fill=tk.X, pady=(0, 0))
        ma = c.get('max_a_diff', 0) or 0
        ms = c.get('max_s_diff', 0) or 0
        my = c.get('max_y_diff', 0) or 0
        tk.Label(err_f, text=f"max|a-a*| = {ma:.6f}", font=(
            "Consolas", 11)).pack(anchor="w")
        tk.Label(err_f, text=f"max|s-s*| = {ms:.6f}", font=(
            "Consolas", 11)).pack(anchor="w")
        tk.Label(err_f, text=f"max|y-y*| = {my:.6f}", font=(
            "Consolas", 11)).pack(anchor="w")

    def update_layer_list(self):
        base_layers = [l for l in self.current_calc["layers"] if not l["is_control"]]
        self.layer_combo['values'] = [f"t = {l['layer']}" for l in base_layers]
        if base_layers:
            self.layer_combo.current(0)

    def show_all_layers(self):
        if not self.current_calc:
            return

        layers = self.current_calc["layers"]
        base_layers = [l for l in layers if not l["is_control"]]
        if not base_layers:
            return

        # === Цвета как во второй вкладке ===
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'black']

        # Очистка
        for ax in self.axes:
            ax.cla()

        # === Рисуем все слои с жирными линиями ===
        for idx, layer in enumerate(base_layers):
            x = layer["x"]
            a, s, y = layer["a"], layer["s"], layer["y"]
            t = layer["layer"]

            color = colors[idx % len(colors)]
            # Жирные линии: обычные — 1.8, последний — 3.5
            lw = 1.6
            alpha = 1.0

            # Рисуем на всех трёх графиках
            self.axes[0].plot(x, a, color=color, linewidth=lw, alpha=alpha)
            self.axes[1].plot(x, s, color=color, linewidth=lw, alpha=alpha)
            self.axes[2].plot(x, y, color=color, linewidth=lw, alpha=alpha)

            # Добавляем в легенду каждого графика
            label = f"Слой = {t}"
            # пустой plot только для легенды
            self.axes[0].plot([], [], color=color, linewidth=lw, label=label)
            self.axes[1].plot([], [], color=color, linewidth=lw, label=label)
            self.axes[2].plot([], [], color=color, linewidth=lw, label=label)

        # === Оформление каждого графика ===
        titles = ["a(x,t)", "s(x,t)", "y(x,t)"]
        ylabels = ["a(x,t)", "s(x,t)", "y(x,t)"]

        for i, ax in enumerate(self.axes):
            ax.set_title(titles[i], fontsize=13, pad=15)
            ax.set_xlabel("x", fontsize=11)
            ax.set_ylabel(ylabels[i], fontsize=11)
            ax.grid(True, alpha=0.35, linewidth=0.7)
            ax.tick_params(labelsize=10)

            # Легенда на КАЖДОМ графике — с цветом и номером слоя
            ax.legend(fontsize=9.5, loc="upper right",
                      framealpha=0.95, fancybox=True, shadow=True)

        # Единые пределы по Y
        for ax in self.axes:
            ax.set_ylim(-1.0, 2.0)

        self.fig.tight_layout()
        self.canvas_plot.draw()

        # Данные для таблицы
        control_layers = [l for l in layers if l["is_control"]]
        self.base_layers = base_layers
        self.control_map = {l["layer"]: l for l in control_layers}
        self.update_table()

    def update_table(self):
        func = self.var_function.get()  # "a", "s" или "y"

        # Подписи колонок в зависимости от выбранной функции
        if func == "a":
            base_label = "a(x,t)"
            control_label = "a*(x,t)"
            diff_label = "|a−a*|"
        elif func == "s":
            base_label = "s(x,t)"
            control_label = "s*(x,t)"
            diff_label = "|s−s*|"
        else:  # y
            base_label = "y(x,t)"
            control_label = "y*(x,t)"
            diff_label = "|y−y*|"

        # Обновляем заголовки
        self.table.heading("value_base", text=base_label)
        self.table.heading("value_control", text=control_label)
        self.table.heading("diff", text=diff_label)

        # Очищаем таблицу
        for item in self.table.get_children():
            self.table.delete(item)

        if not hasattr(self, 'base_layers') or not self.base_layers:
            return

        # Заполняем данными
        for base in self.base_layers:
            t_base = base["layer"]
            t_control = t_base * 4
            control = self.control_map.get(t_control)

            if control is None:
                continue

            x_base = base["x"]
            x_control = control["x"]

            # Выбираем функцию
            if func == "a":
                val_base = base["a"]
                val_control = np.interp(x_base, x_control, control["a"])
            elif func == "s":
                val_base = base["s"]
                val_control = np.interp(x_base, x_control, control["s"])
            else:  # y
                val_base = base["y"]
                val_control = np.interp(x_base, x_control, control["y"])

            diff = np.abs(val_base - val_control)

            # Заполняем строки — ВСЁ ПО ЦЕНТРУ + разница 6 знаков после запятой
            for i in range(len(x_base)):
                self.table.insert("", "end", values=(
                    f"{t_base}/{t_control}",           # Слой
                    f"{x_base[i]:.6f}",                # x — 6 знаков
                    f"{val_base[i]:.6f}",              # Базовая
                    f"{val_control[i]:.6f}",           # Контрольная
                    # Разница — 6 знаков после запятой
                    f"{diff[i]:.6f}"
                ))

    def delete_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("Выберите", "Выберите расчёт")
            return
        if messagebox.askyesno("Удалить?", "Удалить расчёт навсегда?"):
            calc_id = self.tree.item(sel[0])["values"][0]
            delete_calculation(calc_id)
            self.refresh_list()
            for col in (self.left_col, self.right_col):
                for w in col.winfo_children():
                    w.destroy()
            for ax in self.axes:
                ax.cla()
            self.canvas_plot.draw()
            self.table.delete(*self.table.get_children())
            self.layer_combo.set('')
            messagebox.showinfo("Готово", "Расчёт удалён")
