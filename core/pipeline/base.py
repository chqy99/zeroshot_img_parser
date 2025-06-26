# core/pipeline/base.py

from abc import ABC, abstractmethod
from typing import Dict, Callable, Any
import numpy as np
from core.imgdata.imgdata.image_parse import ImageParseResult


class PipelineParser(ABC):
    def __init__(self, module_factories: Dict[str, Callable[[], Any]]):
        """
        :param module_factories: 模块名 → 构造函数（工厂），每个返回一个 module 实例
        """
        self.module_factories = module_factories
        self.modules: Dict[str, Any] = {}

    def register_module(self):
        """注册所有工厂中定义的模块（懒加载触发）"""
        for name in self.module_factories:
            self.get_module(name)

    def get_module(self, name: str):
        """获取模块实例，必要时调用其构造工厂"""
        if name not in self.modules:
            if name not in self.module_factories:
                raise ValueError(f"模块构造器未定义: {name}")
            self.modules[name] = self.module_factories[name]()
        return self.modules[name]

    @abstractmethod
    def parse(self, image: np.ndarray, **kwargs) -> ImageParseResult:
        pass
