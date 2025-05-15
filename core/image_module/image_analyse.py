import numpy as np
import uuid
from typing import Optional
import abc

class ImageInstance:
    def __init__(self, image):
        self.image: np.array = image


class ImageAnalyseResult:
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

class ImageAnalyse(abc.ABC):
    @abc.abstractmethod
    def analyse_ocr(self, image, **kwargs):
        pass

    @abc.abstractmethod
    def analyse_edge(self, image, **kwargs):
        pass

    @abc.abstractmethod
    def analyse_one_instance(self, image, **kwargs):
        pass

    @abc.abstractmethod
    def analyse_instance_segm(self, image, **kwargs):
        pass

    @abc.abstractmethod
    def analyse_multi_method(self, image, **kwargs):
        pass
