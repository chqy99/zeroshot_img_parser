# core/pipeline/base.py

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np
from core.imgdata.imgdata.image_parse import ImageParseResult
from core.modules.module_factory import ModuleFactory


class PipelineParser(ABC):
    def __init__(self, module_names: List[str]):
        """
        :param module_names: 模块名列表（已注册到 ModuleFactory 中）
        """
        self.module_names = module_names
        self.modules: Dict[str, Any] = {}

    def register_modules(self):
        """预加载所有模块（会触发 ModuleFactory 构造）"""
        for name in self.module_names:
            self.get_module(name)

    def get_module(self, name: str):
        """获取模块实例，必要时从 ModuleFactory 构造"""
        if name not in self.modules:
            self.modules[name] = ModuleFactory.get_module(name)
        return self.modules[name]

    @abstractmethod
    def parse(self, image: np.ndarray, **kwargs) -> ImageParseResult:
        pass
