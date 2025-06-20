from paddleocr import PaddleOCR
import numpy as np
from typing import List
from imgdata.imgdata.structure import ImageObject, ImageParseResult
from base import BaseModule
from model_config import ModelLoader

class PaddleOCRModule(BaseModule):
    def __init__(self, model = ModelLoader().get_model('paddleocr')):
        self.reader = model

    def parse(self, image: np.ndarray, **kwargs) -> ImageParseResult:
        result = self.reader.predict(image)
        res = ImageParseResult(full_image=image)
        box, text, score = result[0]['rec_polys'], result[0]['rec_texts'], result[0]['rec_scores']
        for i in range(len(box)):
            res.objects.append(ImageObject.from_ocr(image, box[i], text[i], score[i], source_module='paddleocr'))
        return res

if __name__ == "__main__":
    paddleocr_module = PaddleOCRModule()
    from PIL import Image
    image = np.array(Image.open(r'/MLU_OPS/DEV_SOFT_TRAIN/chenqiyang/image.jpg'))
    result: ImageParseResult = paddleocr_module.parse(image)
    print(result)
