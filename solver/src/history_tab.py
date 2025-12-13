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
        header_frame = ttk.Frame(left_frame)
        header_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(header_frame, text="Сохранённые расчёты:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)

        # Кнопки справа
        btns = ttk.Frame(header_frame)
        btns.pack(side=tk.RIGHT)
        ttk.Button(btns, text="Обновить", command=self.refresh_list).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btns, text="Удалить", command=self.delete_selected, style="Danger.TButton").pack(side=tk.LEFT)

        # Список расчётов — уменьшаем высоту с 10 до 6–7 строк
        columns = ("id", "name", "date", "note")
        self.tree = ttk.Treeview(
            left_frame, columns=columns, show="headings", height=7)  # было 10 → стало 7
        self.tree.heading("id", text="ID")
        self.tree.heading("name", text="Название")
        self.tree.heading("date", text="Дата")
        self.tree.heading("note", text="Заметка")

        self.tree.column("id", width=50, anchor="center")
        self.tree.column("name", width=180)
        self.tree.column("date", width=110)
        self.tree.column("note", width=180)

        self.tree.pack(fill=tk.X, pady=(0, 10))
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        # === Блок "Отображение слоёв" — теперь в левой панели ===
        display_frame = ttk.LabelFrame(left_frame, text=" Отображение слоёв ", padding=10)
        display_frame.pack(fill=tk.X, pady=(10, 0))

        self.display_mode = tk.StringVar(value="auto")

        modes = [
            ("auto", "Авто"),
            ("every", "Каждый N-й"),
            ("single", "Один слой")
        ]
        for i, (mode, text) in enumerate(modes):
            ttk.Radiobutton(display_frame, text=text, variable=self.display_mode,
                            value=mode, command=self.on_display_mode_change).grid(row=0, column=i, padx=10, sticky="w")

        # Поле ввода
        self.entry_var = tk.StringVar(value="10")
        self.entry_n = ttk.Entry(display_frame, textvariable=self.entry_var, width=8, font=("Consolas", 11))
        self.entry_n.grid(row=0, column=3, padx=(30, 5))

        # Кнопка "Показать"
        show_btn = ttk.Button(display_frame, text="Показать", command=self.on_show_button)
        show_btn.grid(row=0, column=4, padx=(0, 10))

        # Автообновление при Enter или потере фокуса
        self.entry_n.bind("<Return>", lambda e: self.on_show_button())
        self.entry_n.bind("<FocusOut>", lambda e: self.on_show_button())

        # Изначально поле выключено (активируется в toggle_entry_state)
        self.entry_n.config(state="disabled")

        # Привязываем смену режима для активации/деактивации поля
        self.display_mode.trace("w", self.toggle_entry_state)

        # === Область для параметров расчёта — без прокрутки, растягивается по высоте ===
        self.params_frame = ttk.Frame(left_frame)
        self.params_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 10))

        # Правая панель (графики и таблица)
        self.setup_right_panel()

    def setup_right_panel(self):
        right_frame = ttk.Frame(self.frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH,
                         expand=True, padx=(5, 10), pady=10)

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
        total_saved = len(base_layers)

        # Определяем, был ли расчёт прорежен (примерно: если слоёв меньше, чем m+1)
        # Мы не знаем точное m, но если слоёв < 10001 — считаем, что не прорежен
        is_thinned = total_saved < 10001

        if mode == "auto":
            if not is_thinned:
                # Маленький расчёт — все слои сохранены → равномерно по индексу
                if total_saved <= 6:
                    return base_layers
                indices = [0] + [int(i * (total_saved - 1) / 5) for i in range(1, 6)] + [total_saved - 1]
                indices = sorted(set(indices))
                return [base_layers[i] for i in indices]
            else:
                # Большой расчёт — прорежен → выбираем 6 равномерно по времени
                if total_saved <= 6:
                    return base_layers

                # Находим минимальное и максимальное время (layer)
                min_t = base_layers[0]["layer"]
                max_t = base_layers[-1]["layer"]

                # 6 точек: 0%, 20%, 40%, 60%, 80%, 100%
                target_times = [min_t + i * (max_t - min_t) / 5 for i in range(6)]

                selected = []
                for target in target_times:
                    # Находим ближайший слой по времени
                    closest = min(base_layers, key=lambda l: abs(l["layer"] - target))
                    if closest not in selected:  # избегаем дубли (на концах)
                        selected.append(closest)

                return selected

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
            # Ищем ближайший доступный слой по номеру
            closest = min(base_layers, key=lambda l: abs(l["layer"] - target))
            return [closest]

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
                # Очистка предыдущих виджетов
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        if not self.current_calc:
            return

        c = self.current_calc
        i = c['initial']
        s = c['system']
        g = c['grid']

        # === Один большой модуль "Параметры расчёта" ===
        main_frame = ttk.LabelFrame(self.params_frame, text=" Параметры расчёта ", padding=15)
        main_frame.pack(fill=tk.X, pady=(0, 8))

        # Вспомогательная функция для подзаголовков
        def add_section_title(parent, title):
            lbl = ttk.Label(parent, text=title, font=("Arial", 11, "bold"), foreground="#2c3e50")
            lbl.pack(anchor="w", pady=(8, 4))

        # Вспомогательная функция для таблицы
        def create_table(parent, columns_dict, data, widths=None, height=4):
            col_names = list(columns_dict.keys())
            col_headers = list(columns_dict.values())

            tree_frame = ttk.Frame(parent)
            tree_frame.pack(fill=tk.X, pady=(0, 6))

            tree = ttk.Treeview(tree_frame, columns=col_names, show="headings", height=height)
            for col, header in zip(col_names, col_headers):
                tree.heading(col, text=header)
                w = widths.get(col, 100) if widths else 100
                tree.column(col, width=w, anchor="center")

            for row in data:
                tree.insert("", "end", values=row)

            tree.pack(fill=tk.X)
            return tree

        # === 1. Начальные условия ===
        add_section_title(main_frame, "Начальные условия")

        cols_init = {"coeff": "Коэффициент", "a": "a", "s": "s", "y": "y"}
        widths_init = {"coeff": 100, "a": 120, "s": 120, "y": 120}

        init_data = [
            ("A₀", f"{i['a0'][0]:.6g}", f"{i['a0'][1]:.6g}", f"{i['a0'][2]:.6g}"),
            ("A₁", f"{i['a1'][0]:.6g}", f"{i['a1'][1]:.6g}", f"{i['a1'][2]:.6g}"),
            ("A₂", f"{i['a2'][0]:.6g}", f"{i['a2'][1]:.6g}", f"{i['a2'][2]:.6g}"),
            ("A₃", f"{i['a3'][0]:.6g}", f"{i['a3'][1]:.6g}", f"{i['a3'][2]:.6g}"),
            ("b₁ / b₂ / b₃", i['b'][0], i['b'][1], i['b'][2]),
        ]

        create_table(main_frame, cols_init, init_data, widths_init, height=5)

        # === 2. Параметры сетки ===
        add_section_title(main_frame, "Параметры сетки")

        grid_data = [
            ("n (сетка по x):", g['n']),
            ("m (сетка по t):", g['m']),
            ("T (время)", f"{g['T']:.6g}"),
        ]
        create_table(main_frame,
                     {"param": "Параметр", "value": "Значение"},
                     grid_data,
                     {"param": 180, "value": 150},
                     height=3)

        # === 3. Параметры системы — три колонки ===
        add_section_title(main_frame, "Параметры системы")

        sys_inner_frame = ttk.Frame(main_frame)
        sys_inner_frame.pack(fill=tk.X, pady=(0, 10))

        col1 = ttk.Frame(sys_inner_frame)
        col2 = ttk.Frame(sys_inner_frame)
        col3 = ttk.Frame(sys_inner_frame)
        col1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 25))
        col2.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 25))
        col3.pack(side=tk.LEFT, fill=tk.X, expand=True)

        col1_data = [("c", s['c']), ("μ", s['mu']), ("c₀", s['c0']), ("ν", s['nu'])]
        col2_data = [("ε", s['eps']), ("d", s['d']), ("e", s['e']), ("f", s['f'])]
        col3_data = [("η", s['eta']), ("Dₐ", s['Da']), ("Dₛ", s['Ds']), ("Dᵧ", s['Dy'])]

        def make_column(frame, data):
            for name, val in data:
                row = ttk.Frame(frame)
                row.pack(fill=tk.X, pady=2)
                ttk.Label(row, text=f"{name}", width=8, font=("Consolas", 11, "bold"), anchor="e").pack(side=tk.LEFT)
                ttk.Label(row, text=f" = {val:.6g}", font=("Consolas", 11)).pack(side=tk.LEFT)

        make_column(col1, col1_data)
        make_column(col2, col2_data)
        make_column(col3, col3_data)

        # === Максимальные разности — отдельно ===
        err_frame = ttk.LabelFrame(self.params_frame, text=" Максимальные разности ", padding=10)
        err_frame.pack(fill=tk.X, pady=(0, 0))

        ma = c.get('max_a_diff', 0) or 0
        ms = c.get('max_s_diff', 0) or 0
        my = c.get('max_y_diff', 0) or 0


        err_data = [
            ("max |a − a*|", f"{ma:.8f}"),
            ("max |s − s*|", f"{ms:.8f}"),
            ("max |y − y*|", f"{my:.8f}"),
        ]

        create_table(err_frame,
                     {"error": "Разность", "value": "Значение"},
                     err_data,
                     {"error": 180, "value": 150},
                     height=3)

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
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'black']

        mode = self.display_mode.get()

        # Очистка графиков
        for ax in self.axes:
            ax.cla()

        # Рисуем выбранные слои
        for idx, layer in enumerate(layers_to_show):
            x = layer["x"]
            a, s, y = layer["a"], layer["s"], layer["y"]
            t = layer["layer"]
            lw = 1.6
            label = f"t = {t}"

            if mode == "single":
                self.axes[0].plot(x, a, color="blue", linewidth=lw, label=label)
                self.axes[1].plot(x, s, color="green", linewidth=lw, label=label)
                self.axes[2].plot(x, y, color="red", linewidth=lw, label=label)
            else:
                colors = ['red', 'blue', 'green', 'orange', 'purple', 'black']
                color = colors[idx % len(colors)]
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
                    f"{t_base}/{t_base * 4}",
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
