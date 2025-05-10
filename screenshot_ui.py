import tkinter as tk
from PIL import Image, ImageTk
import gui_window
import numpy as np
from tkinter import ttk, simpledialog

from use_sam2 import SAM2Segmenter
from image_vector_store import ChromadbVectorStore
from image_tools import ImageVisualizer, MaskHandler

segmenter = SAM2Segmenter()

class CustomDialog(simpledialog.Dialog):
    def body(self, master):
        # 添加一个标签和组合框来选择描述类型
        tk.Label(master, text="describe type (optional):").grid(row=0, sticky="w")
        self.describe_type_var = tk.StringVar()
        self.combobox = ttk.Combobox(master, textvariable=self.describe_type_var)
        self.combobox.grid(row=1, sticky="w")
        self.combobox['values'] = ("原图像", "掩膜图像")  # 示例选项
        self.combobox.current(0)  # 默认选中第一个选项

        # 如果你还需要让用户输入描述文本，可以添加一个输入框
        tk.Label(master, text="describe:").grid(row=2, sticky="w")
        self.describe_entry = tk.Entry(master)
        self.describe_entry.grid(row=3, sticky="w")

        return self.describe_entry  # 返回初始焦点组件

    def apply(self):
        # 获取用户输入的描述和描述类型
        self.describe = self.describe_entry.get()
        self.describe_type = self.describe_type_var.get()


class TransparentOverlay:
    def __init__(self, left, top, width, height, window_title, segmenter: SAM2Segmenter):
        # 创建透明窗口
        self.root = tk.Tk()
        self.root.title("Transparent Overlay")
        self.root.geometry(f"{width}x{height}+{left}+{top}")
        self.root.attributes("-alpha", 0.3)  # 设置透明度
        self.root.attributes("-topmost", True)  # 窗口始终在顶层
        self.root.configure(bg="black")  # 背景颜色
        self.root.overrideredirect(True)  # 去掉窗口边框和标题栏

        # 窗口位置
        self._left, self._top, self._width, self._height = left, top, width, height
        self.window_title = window_title

        # 绑定鼠标事件
        self.root.bind("<Button-1>", self.on_mouse_down)  # 左键按下
        self.root.bind("<B1-Motion>", self.on_mouse_move)  # 左键拖动
        self.root.bind("<ButtonRelease-1>", self.on_mouse_up)  # 左键释放
        self.root.bind("<Button-3>", self.on_mouse_down)  # 右键按下
        self.root.bind("<B3-Motion>", self.on_mouse_move) # 右键拖动
        self.root.bind("<ButtonRelease-3>", self.on_mouse_up) # 右键释放

        self.root.bind("<Control-p>", self.capture_screenshot) # 窗口截图
        self.root.bind("<Control-r>", self.refresh)  # 刷新
        self.root.bind("<Control-a>", self.accept_mask) # 认可识别结果，并补充注释
        self.root.bind("<Control-s>", self.save_in_chromadb) # 保存到向量数据库中
        self.root.bind("<Control-c>", self.exit)

        # 创建画布
        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 初始化选择区域
        self.start_x, self.start_y = 0, 0
        self.end_x, self.end_y = 0, 0
        self.rect = None

        # SAM2Segmenter 实例
        self.segmenter = segmenter
        self.prompts = {}

        # 图像解析实例
        self.image_annotation = None

        # 存储库
        self.vector_store = ChromadbVectorStore()

        # 截取整个屏幕
        self.capture_screenshot()

    def capture_screenshot(self):
        # 截取整个屏幕
        monitor = {"left": self._left, "top": self._top, "width": self._width, "height": self._height}
        screenshot = gui_window.get_screenshot(monitor)
        self.segmenter.set_image(screenshot)
        self.image_annotation = self.vector_store.query_image_item(screenshot)

    def on_mouse_down(self, event):
        # 记录鼠标按下位置
        self.start_x, self.start_y = event.x, event.y
        self.end_x, self.end_y = event.x, event.y

        # 创建选择区域矩形
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.end_x, self.end_y,
            outline="white", width=2, fill="white", stipple="gray50"
        )

    def on_mouse_move(self, event):
        # 更新选择区域
        self.end_x, self.end_y = event.x, event.y
        self.canvas.coords(
            self.rect, self.start_x, self.start_y, self.end_x, self.end_y
        )

    def on_mouse_up(self, event):
        # 记录选择区域的大小和位置
        left = min(self.start_x, self.end_x)
        top = min(self.start_y, self.end_y)
        width = abs(self.end_x - self.start_x)
        height = abs(self.end_y - self.start_y)

        print(f"Selected region: left={left}, top={top}, width={width}, height={height}")

        if width <= 1 and height <= 1:
            # 如果选择区域的大小为 1x1，则使用点提示
            if not "point_coords" in self.prompts:
                self.prompts["point_coords"] = []
                self.prompts["point_labels"] = []

            self.prompts["point_coords"].append([left, top])
            if event.num == 1:
                self.prompts["point_labels"].append(1)
            else:
                self.prompts["point_labels"].append(0)
        else:
            # TODO:选框式还需进一步测试
            self.prompts["box"] = [left, top, width, height]

        # 根据当前prompts预测
        self.segmenter.set_prompts(self.prompts)
        mask = self.segmenter.predict()

        # 可视化
        visual_mask = ImageVisualizer.visualize_masks(1, mask)
        bboxs = MaskHandler.mask_to_bbox(mask)
        self.visualize(visual_mask, [bboxs])


    def refresh(self, event):
        # 双击时清空画布
        print("Clearing canvas...")
        self.canvas.delete("all")  # 清空画布
        self.rect = None  # 重置矩形引用
        self.prompts = {} # 清空 prompts

    def visualize(self, visual_mask, bboxs):
        # 可视化预测的掩码
        colored_mask_img = ImageTk.PhotoImage(visual_mask)

        # 在画布上绘制掩码
        self.canvas.create_image(0, 0, image=colored_mask_img, anchor=tk.NW)
        self.canvas.image = colored_mask_img  # 保持引用，防止图像被垃圾回收

        # 绘制掩码的边界框
        for bbox in bboxs:
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                # 绘制红色边界框
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)

    def accept_mask(self, event):
        # 弹出对话框输入注释
        comment = simpledialog.askstring("Input", "Enter a comment for the mask:", parent=self.root)
        if comment is None:
            return  # 用户取消输入

        # 保存 mask 和注释
        if self.segmenter.mask is not None:
            self.image_annotation.add_one_class(self.segmenter.mask, comment)
            # 可视化
            self.refresh(None)
            visual_mask, bboxs = self.image_annotation.visualize_annotation()
            self.visualize(visual_mask, bboxs)

    def save_in_chromadb(self, event):
        # 弹出对话框输入注释和描述类型
        dialog = CustomDialog(self.root)
        if not dialog.describe:  # 如果用户取消输入或未填写描述
            return
        meta_data = self.image_annotation.meta_data
        meta_data["window_title"] = window_title
        meta_data["describe"] = dialog.describe
        meta_data["describe_type"] = dialog.describe_type if dialog.describe_type else None  # 如果描述类型为空则设为 None
        print(meta_data["describe_type"])
        self.vector_store.add_image_item(self.image_annotation)

    def run(self):
        # 启动窗口主循环
        self.root.mainloop()

    def exit(self, event):
        print("Exiting...")
        self.root.destroy()


if __name__ == "__main__":
    window_title = "MuMu模拟器12"
    window = gui_window.GuiWindow(window_title, 0)

    TransparentOverlay(window.left, window.top, window.width, window.height, window_title, segmenter).run()
