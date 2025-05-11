import numpy as np
import uuid
from typing import Optional

class ImageEntity:
    def __init__(self, image: np.ndarray):
        # 原始图像
        self.image: np.ndarray = image
        # 唯一标识
        self.id: str = ""
        # 图像基本信息
        self.height: int = 0
        self.width: int = 0
        self.channel: int = 0
        # 解析结果
        self.info: str = "" # 图像关键信息，GUI图片的格式是 'window: 窗口名'
        self.describe: Optional[str] = None # 全图描述
        self.classes_num: int = 0
        self.mask: np.ndarray = None
        self.mask_image: np.ndarray = None
        self.bboxs: list[list] = []
        self.classes_info: list[str] = [] # 子类关键信息，GUI图片的子类格式是 '控件名: 关键语义'
        self.classes_describe: list[Optional[str]] = [] # 子类描述

    def add_child_by_mask(self, mask, metadata):
        pass

    def get_metadata(self):
        metadata = {
            "height": self.height,
            "width": self.width,
            "channel": self.channel
        }

