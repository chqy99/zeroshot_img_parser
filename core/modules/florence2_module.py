# core/modules/florence2_module.py
import numpy as np
from typing import List
from imgdata.structure import ImageObject
from base import EnricherModule

class Florence2Module(EnricherModule):
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def parse(self, objects: List[ImageObject], image: np.ndarray, **kwargs) -> List[ImageObject]:
        for obj in objects:
            crop = obj.crop_from(image)
            description = self.model.describe(crop)
            obj.description = description
        return objects
