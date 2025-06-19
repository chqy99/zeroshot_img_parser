import numpy as np
from typing import List
from imgdata.imgdata.structure import ImageObject
from base import BaseModule

class GroundingDinoModule(BaseModule):
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def parse(self, image: np.ndarray, **kwargs) -> List[ImageObject]:
        prompt = kwargs.get("text_prompt", "")
        boxes = self.model.detect(image, prompt)
        return [ImageObject.from_detection(box=b, label=prompt) for b in boxes]
