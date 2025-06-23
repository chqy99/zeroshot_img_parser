# core/pipeline/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from imgdata.imgdata.structure import ImageObject


class PipelineParser(ABC):
    def __init__(self):
        self.module_weights: Dict[str, float] = {}
        self.modules: Dict[str, Any] = {}  # key -> BaseModule or EnricherModule
        self.strategy = None  # 可学习策略模型

    @abstractmethod
    def parse(self, image: np.ndarray, **kwargs) -> List[ImageObject]:
        """组合调用各模块，执行任务"""
        pass

    def register_module(self, name: str, module: Any, weight: float = 1.0):
        self.modules[name] = module
        self.module_weights[name] = weight

    def set_strategy(self, strategy_model):
        """支持动态组合模型（如 Gating 网络）"""
        self.strategy = strategy_model

    def get_active_modules(self, image: np.ndarray, **kwargs) -> List[str]:
        """返回当前应激活的模块名称列表"""
        if self.strategy:
            return self.strategy.select_modules(image, self.modules)
        return list(self.modules.keys())
