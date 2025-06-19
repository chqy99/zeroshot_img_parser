from paddleocr import PaddleOCR
import numpy as np
from typing import List
from imgdata.structure import ImageObject, ImageParseResult
from base import BaseModule

class PaddleOCRModule(BaseModule):
    def __init__(self, model):
        self.reader = model

    def parse(self, image: np.ndarray, **kwargs) -> ImageParseResult:
        result = self.reader.ocr(image, det=True, rec=True, cls=True)
        res = ImageParseResult(full_image=image)
        for line in result[0]:
            box, (text, score) = line
            res.objects.append(ImageObject.from_ocr(box, text, score))
        return res
