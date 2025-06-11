from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from imgdata.structure import ImageObject, ImageParseResult

class BaseModule(ABC):
    @abstractmethod
    def parse(self, image: np.ndarray, **kwargs) -> List[ImageObject]:
        pass

class EnricherModule(ABC):
    @abstractmethod
    def parse(self, objects: List[ImageObject], image: np.ndarray, **kwargs) -> List[ImageObject]:
        pass
