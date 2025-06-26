# core/pipeline/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from core.imgdata.imgdata.image_parse import ImageParseResult
from core.modules.model_config import ModelLoader


class PipelineParser(ABC):
    def __init__(self, modules: List[str] = []):
        self.model_loader = ModelLoader()
        self.module_names = modules
        self.modules: Dict[str, Any] = {}  # 惰性加载后存放模块实例

        self.register_module()

    def register_module(self):
        """
        根据 module_names 注册模块（惰性加载，不立即初始化模型权重）
        """
        for name in self.module_names:
            if name not in self.modules:
                self.modules[name] = self.model_loader.get_model(name)

    @abstractmethod
    def parse(self, image: np.ndarray, **kwargs) -> ImageParseResult:
        """
        子类必须实现的解析方法：组合多个模块进行图像解析
        """
        pass

