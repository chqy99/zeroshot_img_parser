from paddleocr import PaddleOCR
import numpy as np
from typing import List
from imgdata.imgdata.image_parse import ImageParseItem, ImageParseResult
from base import BaseModule
from model_config import ModelLoader


class PaddleOCRModule(BaseModule):
    def __init__(self, model=None):
        self.reader = model or ModelLoader().get_model("paddleocr")

    def parse(self, image: np.ndarray, **kwargs) -> ImageParseResult:
        result = self.reader.predict(image)
        res = ImageParseResult(image)
        box, text, score = (
            result[0]["rec_polys"],
            result[0]["rec_texts"],
            result[0]["rec_scores"],
        )
        for i in range(len(box)):
            res.items.append(
                ImageParseItem(
                    image, "paddleocr", score[i], box[i], type="ocr", text=text[i]
                )
            )
        return res


if __name__ == "__main__":
    paddleocr_module = PaddleOCRModule()
    from PIL import Image

    image = np.array(Image.open(r"/MLU_OPS/DEV_SOFT_TRAIN/chenqiyang/image.jpg"))
    result: ImageParseResult = paddleocr_module.parse(image)
    print(result)
