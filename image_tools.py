from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt

class MaskHandler:
    @staticmethod
    def mask_to_bbox(mask):
        """
        将二值分割掩码转换为边界框坐标（左上角和右下角坐标）。

        参数:
        mask -- 二维的分割掩码，是一个 numpy 数组，形状为 (height, width)，值为 0 或 1。

        返回:
        返回一个列表，包含边界框的左上角和右下角坐标，格式为 [x1, y1, x2, y2]。
        如果掩码全为 0，则返回 None。
        """
        # 找到掩码中非零元素的行和列索引
        rows = np.where(np.any(mask, axis=1))[0]  # 行索引
        cols = np.where(np.any(mask, axis=0))[0]  # 列索引

        if len(rows) == 0 or len(cols) == 0:
            # 如果掩码全为 0，返回 None
            return None

        # 获取边界框的坐标
        y1 = rows[0]
        y2 = rows[-1]
        x1 = cols[0]
        x2 = cols[-1]

        return [x1, y1, x2, y2]


class ImageVisualizer:
    @staticmethod
    def generate_palette(n):
        """
        生成一个长度为 n 的颜色表，采用均匀采样，然后转换为 RGB 格式。

        Args:
            n (int): 颜色表的长度。

        Returns:
            list: 颜色表，每个颜色为 (R, G, B) 格式。
        """
        # 使用 Matplotlib 生成均匀分布的颜色
        cm = plt.get_cmap('hsv')  # 使用 HSV 色彩映射
        colors = [cm(i / n) for i in range(n)]
        # 转换为 RGB 格式并缩放到 0-255
        palette = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in colors]
        return palette

    @staticmethod
    def visualize_masks(num, mask):
        if num == 0 or mask is None:
            return None

        # 生成颜色表
        palette = ImageVisualizer.generate_palette(math.ceil(num / 8) * 8)

        # 初始化彩色掩码图像，添加 alpha 通道
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)

        # 应用颜色到每个类别
        for idx in range(1, num + 1):  # 从1开始，因为0是背景
            if idx < len(palette):
                color = palette[idx-1] + (150,)  # 添加透明度
                colored_mask[mask == idx] = color

        # 创建图像
        colored_mask_img = Image.fromarray(colored_mask, mode="RGBA")

        return colored_mask_img
