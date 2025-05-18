import tkinter as tk
from tkinter import ttk, simpledialog
import threading

import gui.gui_window as gui_window
from xc_image.core.image_module.image_dataclass import ImageAnalyseResult
import ui.transparent_overlay as transparent_overlay

class ImageTk:
    def __init__(self, root):
        self.root = root
        self.root.title("蒙版模式")
        self.root.geometry("864x512+2816+64")

        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建顶部控制面板
        self.control_panel = ttk.Frame(self.main_frame)
        self.control_panel.pack(fill=tk.X, pady=5)

        # 初始化各部分控件，仅创建控件和布局，不加载数据和绑定事件
        self.init_combobox()
        self.init_checkboxes()
        self.init_buttons()
        self.init_datagrid()

        # 启动延时加载线程
        threading.Thread(target=self.delayed_load, daemon=True).start()

        # 蒙版窗口类
        self.transparent_tk = transparent_overlay.TransparentOverlay(root, show_datagrid_data=self.show_datagrid_data)

    def init_combobox(self):
        """初始化 Combobox"""
        self.combobox_label = ttk.Label(self.control_panel, text="选择选项:")
        self.combobox_label.pack(side=tk.LEFT, padx=5)

        self.selected_option = tk.StringVar()
        self.combobox = ttk.Combobox(self.control_panel, textvariable=self.selected_option, state='readonly')
        self.combobox.pack(side=tk.LEFT, padx=5)

    def init_checkboxes(self):
        """初始化复选框"""
        self.checkbox_frame = ttk.Frame(self.control_panel)
        self.checkbox_frame.pack(side=tk.LEFT, padx=20)

        self.checkbox1_var = tk.BooleanVar()
        self.checkbox2_var = tk.BooleanVar()
        self.checkbox3_var = tk.BooleanVar()

        # 设置复选框的初始状态
        self.checkbox1_var.set(True)
        self.checkbox2_var.set(True)
        self.checkbox3_var.set(True)

        self.checkbox1 = ttk.Checkbutton(self.checkbox_frame, text="静态判断", variable=self.checkbox1_var)
        self.checkbox1.pack(anchor=tk.W)

        self.checkbox2 = ttk.Checkbutton(self.checkbox_frame, text="OCR提取", variable=self.checkbox2_var)
        self.checkbox2.pack(anchor=tk.W)

        self.checkbox3 = ttk.Checkbutton(self.checkbox_frame, text="sam2辅助", variable=self.checkbox3_var)
        self.checkbox3.pack(anchor=tk.W)

        # 将复选框变量与控件关联
        self.checkbox1.configure(variable=self.checkbox1_var)
        self.checkbox2.configure(variable=self.checkbox2_var)
        self.checkbox3.configure(variable=self.checkbox3_var)

    def init_buttons(self):
        """初始化多个按钮"""
        self.button_frame = ttk.Frame(self.control_panel)
        self.button_frame.pack(side=tk.LEFT, padx=10)

        self.button1 = ttk.Button(self.button_frame, text="创建蒙版")
        self.button1.pack(side=tk.LEFT, padx=5)

        self.button2 = ttk.Button(self.button_frame, text="添加实体")
        self.button2.pack(side=tk.LEFT, padx=5)

        self.button3 = ttk.Button(self.button_frame, text="存储蒙版")
        self.button3.pack(side=tk.LEFT, padx=5)

        # 绑定事件
        self.button1.configure(command=self.create_overlap)

    def init_datagrid(self):
        """初始化数据展示表格"""
        self.tree_frame = ttk.Frame(self.main_frame)
        self.tree_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # 创建树视图
        self.tree = ttk.Treeview(self.tree_frame, columns=("ID", "位置", "信息", "补充", "颜色"), show="headings")

        # 设置列标题
        self.tree.heading("ID", text="ID")
        self.tree.heading("位置", text="位置")
        self.tree.heading("信息", text="信息")
        self.tree.heading("补充", text="补充")
        self.tree.heading("颜色", text="颜色")

        # 设置列宽
        self.tree.column("ID", width=50)
        self.tree.column("位置", width=100)
        self.tree.column("信息", width=100)
        self.tree.column("补充", width=100)
        self.tree.column("颜色", width=50)

        # 添加滚动条
        self.scrollbar = ttk.Scrollbar(self.tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.scrollbar.set)

        # 布局表格和滚动条
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        # 配置网格权重，使表格可扩展
        self.tree_frame.grid_columnconfigure(0, weight=1)
        self.tree_frame.grid_rowconfigure(0, weight=1)

        # 绑定双击事件以编辑表格单元格
        self.tree.bind("<Double-1>", self.edit_cell)

    def delayed_load(self):
        """延时加载数据和绑定事件"""
        self.load_combobox_data()

    def load_combobox_data(self):
        """加载 Combobox 数据"""
        self.options = gui_window.get_windows_title(show_flag=False)
        self.combobox['values'] = self.options
        self.combobox.bind("<<ComboboxSelected>>", self.on_window_selected)

    def show_datagrid_data(self, image_entity: ImageAnalyseResult):
        """加载表格数据"""
        self.image_entity = image_entity
        self.clear_datagrid()
        # 全图
        self.tree.insert("", tk.END, values=('全图',
                                              str([0, 0, image_entity.width, image_entity.height]),
                                              str(image_entity.info),
                                              str(image_entity.describe),
                                              ''))
        # 子实例
        for i in range(image_entity.classes_num):
            pass

    def clear_datagrid(self):
        """清空表格数据"""
        for item in self.tree.get_children():
            self.tree.delete(item)

    def on_window_selected(self, event):
        selected_window_title = self.selected_option.get()
        selected_window = gui_window.GuiWindow(selected_window_title, 0)

        self.transparent_tk.preload_window_rect(
            selected_window_title, selected_window.left, selected_window.top,
            selected_window.width, selected_window.height)

    def create_overlap(self):
        use_model_vals = [self.checkbox1_var.get(), self.checkbox2_var.get(), self.checkbox3_var.get()]
        self.transparent_tk.create_and_show_overlay(use_model_vals)

    def edit_cell(self, event):
        selected_item = self.tree.selection()
        if not selected_item:
            return

        # 获取当前选中的列
        col = self.tree.identify_column(event.x)
        col_id = self.tree.heading(col, "text")

        # 判断是否允许编辑
        if col_id in ["ID", "位置"]:
            return

        # 创建一个Entry控件用于编辑
        entry = tk.Entry(self.tree)
        entry.place(x=self.tree.bbox(selected_item, col)[0],
                    y=self.tree.bbox(selected_item, col)[1],
                    relwidth=0.22,
                    relheight=0.25)
        entry.insert(0, self.tree.set(selected_item, col))

        # 提交更改
        def on_entry_change(event):
            new_value = entry.get()
            self.tree.set(selected_item, col, new_value)
            entry.destroy()

        # 绑定事件
        entry.bind("<Return>", on_entry_change)
        entry.bind("<FocusOut>", on_entry_change)
        entry.focus_set()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageTk(root)
    root.mainloop()
