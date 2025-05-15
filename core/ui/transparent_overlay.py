import tkinter as tk
from tkinter import ttk, simpledialog
from PIL import ImageTk
import threading

import gui.gui_window as gui_window
from image_module.image_analyse import ImageAnalyseResult

class TransparentOverlay:
    def __init__(self, root, show_datagrid_data):
        self.root = root
        self.show_datagrid_data = show_datagrid_data
        # 蒙版窗口
        self.monitor = {}
        self.overlay_window = None
        # 图像解析实体
        self.screenshot = None
        self.image_entity = None
        # 分割模型
        self.sam2_segmenter = None
        self.sam2_loaded = threading.Event()
        # 预加载模型，节省时间
        self.preload_process_model()

    def preload_window_rect(self, window_title, left, top, width, height):
        self.window_title = window_title
        self.monitor["left"], self.monitor["top"], self.monitor["width"], self.monitor["height"] = left, top, width, height

    def preload_process_model(self):
        threading.Thread(target=self.load_sam2_segmenter).start()

    def create_and_show_overlay(self, use_model_vals):
        """使用预加载的信息创建并显示蒙版窗口"""
        if self.window_title is None:
            return

        # 如果蒙版窗口已经存在，先销毁它
        if self.overlay_window is not None:
            self.overlay_window.destroy()

        # 创建一个新的顶层窗口作为蒙版
        self.overlay_window = tk.Toplevel(self.root)
        self.overlay_window.title("Transparent Overlay")
        self.overlay_window.geometry(f'{self.monitor["width"]}x{self.monitor["height"]}+{self.monitor["left"]}+{self.monitor["top"]}')
        self.overlay_window.attributes("-alpha", 0.3)  # 设置透明度
        self.overlay_window.attributes("-topmost", True)  # 窗口始终在顶层
        self.overlay_window.configure(bg="black")  # 背景颜色
        self.overlay_window.overrideredirect(True)  # 去掉窗口边框和标题栏

        # 创建画布
        self.canvas = tk.Canvas(self.overlay_window, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 截图
        self.capture_screenshot()
        self.image_entity = ImageAnalyseResult(self.screenshot)

        # 静态判断
        if use_model_vals[0] == True:
            pass

        # OCR 处理
        if use_model_vals[1] == True:
            pass

        # SAM2 处理
        if use_model_vals[2] == True:
            self.init_sam2_relevant()
            self.sam2_segmenter.set_image(self.screenshot)

        # 在 root 中显示解析数据
        self.show_datagrid_data(self.image_entity)

    def capture_screenshot(self):
        # 截取整个屏幕
        self.screenshot = gui_window.get_screenshot(self.monitor)

### SAM2 处理相关 ###
    def init_sam2_relevant(self):
        # 等待 SAM2 加载完成
        self.sam2_loaded.wait()

        # 绑定鼠标事件到蒙版窗口
        self.overlay_window.bind("<Button-1>", self.on_mouse_down)  # 左键按下
        self.overlay_window.bind("<B1-Motion>", self.on_mouse_move)  # 左键拖动
        self.overlay_window.bind("<ButtonRelease-1>", self.on_mouse_up)  # 左键释放
        self.overlay_window.bind("<Button-3>", self.on_mouse_down)  # 右键按下
        self.overlay_window.bind("<B3-Motion>", self.on_mouse_move)  # 右键拖动
        self.overlay_window.bind("<ButtonRelease-3>", self.on_mouse_up)  # 右键释放
        self.overlay_window.bind("<Control-r>", self.refresh)  # 刷新
        self.overlay_window.bind("<Control-c>", self.exit)  # 退出

        # 初始化选择区域
        self.start_x, self.start_y = 0, 0
        self.end_x, self.end_y = 0, 0
        self.rect = None

        # 提示信息
        self.prompts = {}

    def load_sam2_segmenter(self):
        from process.use_sam2 import SAM2Segmenter
        self.sam2_segmenter = SAM2Segmenter()
        # 加载完成后设置事件标志
        self.sam2_loaded.set()

    def on_mouse_down(self, event):
        """处理鼠标按下事件"""
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
        """处理鼠标移动事件"""
        # 更新选择区域
        self.end_x, self.end_y = event.x, event.y
        self.canvas.coords(
            self.rect, self.start_x, self.start_y, self.end_x, self.end_y
        )

    def on_mouse_up(self, event):
        """处理鼠标释放事件"""
        # 记录选择区域的大小和位置
        left = min(self.start_x, self.end_x)
        top = min(self.start_y, self.end_y)
        width = abs(self.end_x - self.start_x)
        height = abs(self.end_y - self.start_y)

        print(f"Selected region: left={left}, top={top}, width={width}, height={height}")

        if width <= 1 and height <= 1:
            # 如果选择区域的大小为 1x1，则使用点提示
            if "point_coords" not in self.prompts:
                self.prompts["point_coords"] = []
                self.prompts["point_labels"] = []

            self.prompts["point_coords"].append([left, top])
            if event.num == 1:
                self.prompts["point_labels"].append(1)
            else:
                self.prompts["point_labels"].append(0)
        else:
            # 选框式选择
            self.prompts["box"] = [left, top, width, height]

        self.sam2_segmenter.set_prompts(self.prompts)
        mask = self.sam2_segmenter.predict()

        # 可视化
        from tools.mask_tools import MaskHandler
        visual_mask = MaskHandler.visualize_masks(1, mask)
        bbox = MaskHandler.mask_to_bbox(mask)
        self.visualize(visual_mask, [bbox])

    def visualize(self, visual_mask, bboxs):
        """可视化 SAM2 预测结果"""
        # 清空画布
        self.canvas.delete("all")
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

    def refresh(self, event):
        """刷新画布"""
        print("Refreshing canvas...")
        self.canvas.delete("all")  # 清空画布
        self.rect = None  # 重置矩形引用
        self.prompts = {}  # 清空 prompts
### SAM2 处理相关 ###

    def exit(self, event):
        """退出蒙版窗口"""
        print("Exiting overlay...")
        if self.overlay_window:
            self.overlay_window.destroy()
            self.overlay_window = None
