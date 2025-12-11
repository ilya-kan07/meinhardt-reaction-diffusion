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

        control_frame = ttk.LabelFrame(right_frame, text=" Отображение слоёв ", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.display_mode = tk.StringVar(value="auto")

        modes = [
            ("auto", "Авто (6 равномерно)"),
            ("every", "Каждый N-й"),
            ("single", "Один слой")
        ]
        for i, (mode, text) in enumerate(modes):
            ttk.Radiobutton(control_frame, text=text, variable=self.display_mode,
                            value=mode, command=self.on_display_mode_change).grid(row=0, column=i, padx=15, sticky="w")

        # Поле ввода
        self.entry_var = tk.StringVar(value="10")
        self.entry_n = ttk.Entry(control_frame, textvariable=self.entry_var, width=12, font=("Consolas", 11))
        self.entry_n.grid(row=0, column=3, padx=(30, 5))

        # КНОПКА "Показать" — вот она!
        show_btn = ttk.Button(control_frame, text="Показать", command=self.on_show_button)
        show_btn.grid(row=0, column=4, padx=(0, 10))

        # Автообновление при нажатии Enter в поле
        self.entry_n.bind("<Return>", lambda e: self.on_show_button())
        self.entry_n.bind("<FocusOut>", lambda e: self.on_show_button())  # при потере фокуса

        # По умолчанию поле выключено

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

    def on_show_button(self):
        """Вызывается по кнопке «Показать» или Enter"""
        # Проверяем, что введено число
        text = self.entry_var.get().strip()
        if not text:
            return
        try:
            int(text)
        except ValueError:
            messagebox.showwarning("Ошибка", "Введите целое число!")
            return

        # Если режим "every" или "single" — обновляем графики
        if self.display_mode.get() in ("every", "single"):
            self.show_selected_layers()

    def toggle_entry_state(self, *args):
        mode = self.display_mode.get()
        state = "normal" if mode in ("every", "single") else "disabled"
        self.entry_n.config(state=state)
        # При "auto" — сразу показываем 6 слоёв
        if mode == "auto":
            self.show_selected_layers()

    def get_layers_to_display(self):
        """Возвращает список базовых слоёв для отображения в зависимости от режима"""
        if not self.current_calc:
            return []

        base_layers = [l for l in self.current_calc["layers"] if not l["is_control"]]
        if not base_layers:
            return []

        mode = self.display_mode.get()

        if mode == "auto":
            # 6 равномерно распределённых (включая первый и последний)
            total = len(base_layers)
            if total <= 6:
                return base_layers
            indices = [0] + [int(i * (total - 1) / 5) for i in range(1, 6)] + [total - 1]
            indices = sorted(set(indices))  # на всякий случай убираем дубли
            return [base_layers[i] for i in indices]

        elif mode == "every":
            try:
                step = max(1, int(self.entry_var.get() or 10))
            except:
                step = 10
            return base_layers[::step]

        elif mode == "single":
            try:
                target = int(self.entry_var.get() or 0)
            except:
                target = 0
            # Ищем ближайший доступный слой
            for layer in base_layers:
                if layer["layer"] >= target:
                    return [layer]
            return [base_layers[-1]] if base_layers else []

        return base_layers  # fallback

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
            # ← ВАЖНО: сбрасываем режим на "auto" при выборе нового расчёта
            self.display_mode.set("auto")
            self.toggle_entry_state()  # это вызовет show_selected_layers()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить:\n{e}")

    def on_display_mode_change(self):
        self.toggle_entry_state()

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

    def show_selected_layers(self):
        if not self.current_calc:
            for ax in self.axes:
                ax.cla()
            self.canvas_plot.draw()
            return

        layers_to_show = self.get_layers_to_display()
        if not layers_to_show:
            for ax in self.axes:
                ax.cla()
            self.canvas_plot.draw()
            return

        colors = ['red', 'orange', 'green', 'blue', 'purple', 'black']

        # Очистка графиков
        for ax in self.axes:
            ax.cla()

        # Рисуем выбранные слои
        for idx, layer in enumerate(layers_to_show):
            x = layer["x"]
            a, s, y = layer["a"], layer["s"], layer["y"]
            t = layer["layer"]
            color = colors[idx % len(colors)]
            lw = 1.6
            label = f"t = {t}"

            self.axes[0].plot(x, a, color=color, linewidth=lw, label=label)
            self.axes[1].plot(x, s, color=color, linewidth=lw, label=label)
            self.axes[2].plot(x, y, color=color, linewidth=lw, label=label)

        # Оформление графиков
        titles = ["a(x,t)", "s(x,t)", "y(x,t)"]
        show_legend = True

        if self.display_mode.get() == "every":
            show_legend = False

        for i, ax in enumerate(self.axes):
            ax.set_title(titles[i], fontsize=13, pad=15)
            ax.set_xlabel("x", fontsize=11)
            ax.set_ylabel(titles[i], fontsize=11)
            ax.grid(True, alpha=0.35)
            ax.tick_params(labelsize=10)
            ax.set_ylim(-1.0, 2.0)
            if show_legend:
                ax.legend(fontsize=9.5, loc="upper right", framealpha=0.92, shadow=True)

        # Вместо tight_layout — фиксированные отступы (чтобы не было предупреждений)
        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.14, wspace=0.28)
        self.canvas_plot.draw()

        # === КЛЮЧЕВОЕ: правильно создаём control_map ===
        all_layers = self.current_calc["layers"]
        self.control_map = {l["layer"]: l for l in all_layers if l["is_control"]}

        # Сохраняем для таблицы
        self.layers_to_show = layers_to_show
        self.update_table()

    def update_table(self):
        func = self.var_function.get()

        labels = {"a": ("a(x,t)", "a*(x,t)", "|a−a*|"),
                  "s": ("s(x,t)", "s*(x,t)", "|s−s*|"),
                  "y": ("y(x,t)", "y*(x,t)", "|y−y*|")}

        self.table.heading("value_base", text=labels[func][0])
        self.table.heading("value_control", text=labels[func][1])
        self.table.heading("diff", text=labels[func][2])

        for item in self.table.get_children():
            self.table.delete(item)

        if not hasattr(self, 'layers_to_show') or not self.layers_to_show:
            return

        for base in self.layers_to_show:
            t_base = base["layer"]
            control = self.control_map.get(t_base * 4)
            if not control:
                continue

            x_base = base["x"]
            x_control = control["x"]

            if func == "a":
                val_base = base["a"]
                val_control = np.interp(x_base, x_control, control["a"])
            elif func == "s":
                val_base = base["s"]
                val_control = np.interp(x_base, x_control, control["s"])
            else:
                val_base = base["y"]
                val_control = np.interp(x_base, x_control, control["y"])

            diff = np.abs(val_base - val_control)

            for i in range(len(x_base)):
                self.table.insert("", "end", values=(
                    t_base,
                    f"{x_base[i]:.6f}",
                    f"{val_base[i]:.6f}",
                    f"{val_control[i]:.6f}",
                    f"{diff[i]:.6f}"
                ))

    def delete_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("Выберите", "Выберите расчёт для удаления")
            return

        if not messagebox.askyesno("Подтверждение", "Удалить выбранный расчёт навсегда?"):
            return

        calc_id = self.tree.item(sel[0])["values"][0]

        try:
            delete_calculation(calc_id)
            self.refresh_list()

            # Полная очистка правой панели и графиков
            for widget in self.params_frame.winfo_children():
                widget.destroy()

            for ax in self.axes:
                ax.cla()
            self.canvas_plot.draw()

            self.table.delete(*self.table.get_children())

            # Сбрасываем текущий расчёт
            self.current_calc = None

            messagebox.showinfo("Готово", "Расчёт успешно удалён")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось удалить расчёт:\n{e}")
