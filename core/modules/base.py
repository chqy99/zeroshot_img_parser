from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from core.imgdata.imgdata.image_parse import ImageParseResult


class BaseModule(ABC):
    @abstractmethod
    def parse(self, image: np.ndarray, **kwargs) -> ImageParseResult:
        pass


class EnricherModule(ABC):
    @abstractmethod
    def parse(self, objects: ImageParseResult, **kwargs) -> ImageParseResult:
        pass
