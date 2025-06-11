from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from data.structure import ImageObject, ImageParseResult

class BaseModule(ABC):
    @abstractmethod
    def parse(self, image: np.ndarray, **kwargs) -> List[ImageObject]:
        pass

class EnricherModule(ABC):
    @abstractmethod
    def parse(self, objects: List[ImageObject], image: np.ndarray, **kwargs) -> List[ImageObject]:
        pass

class TransferEnricherModule(EnricherModule):
    """
    迁移补全模块：
    - objects: 当前图像的初步解析结果
    - image: 当前图像
    - references: 参考图像的解析结果列表，做补全的知识来源
    """
    @abstractmethod
    def parse(self,
              objects: List[ImageObject],
              image: np.ndarray,
              references: List[ImageParseResult],
              **kwargs) -> List[ImageObject]:
        pass

class EmbeddingModule(ABC):
    @abstractmethod
    def extract_embedding(self, obj: ImageObject, **kwargs) -> np.ndarray:
        pass

class ImageHandlerBase(ABC):
    @abstractmethod
    def save_image(self, image: np.ndarray, filename: str) -> str:
        pass

    @abstractmethod
    def load_image(self, filepath: str) -> np.ndarray:
        pass

    @abstractmethod
    def save_mask(self, mask: np.ndarray, filename: str) -> str:
        pass

    @abstractmethod
    def load_mask(self, filepath: str) -> np.ndarray:
        pass
