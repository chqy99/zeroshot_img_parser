from paddleocr import PaddleOCR
import numpy as np
from typing import List
from imgdata.structure import ImageObject
from base import BaseModule

class PaddleOCRModule(BaseModule):
    def __init__(self, lang='en'):
        self.reader = PaddleOCR(use_angle_cls=True, lang=lang)

    def parse(self, image: np.ndarray, **kwargs) -> List[ImageObject]:
        result = self.reader.ocr(image)
        objects = []
        for line in result[0]:
            box, (text, score) = line
            objects.append(ImageObject.from_ocr(box, text, score))
        return objects
