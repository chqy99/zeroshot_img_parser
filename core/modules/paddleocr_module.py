from paddleocr import PaddleOCR
import numpy as np
from typing import List
from core.imgdata.imgdata.image_parse import BBox, ImageParseItem, ImageParseResult
from core.modules.base import BaseModule
from core.modules.model_config import ModelLoader
from core.modules.module_factory import ModuleFactory

@ModelLoader.register_loader("paddleocr")
def load_model_paddleocr(cfg, device):
    # PaddleOCR 的 device 参数要传字符串 "gpu" 或 "cpu"
    device = "gpu" if device == "cuda" else device
    model = PaddleOCR(paddlex_config=cfg.get("paddlex_config", None), device="gpu")
    return model

class PaddleOCRModule(BaseModule):
    def __init__(self, model):
        self.reader = model

    def parse(self, image: np.ndarray, **kwargs) -> ImageParseResult:
        result = self.reader.predict(image)
        res = ImageParseResult(image)
        box, text, score = (
            result[0]["rec_polys"],
            result[0]["rec_texts"],
            result[0]["rec_scores"],
        )
        for i in range(len(box)):
            poly = box[i]
            x_coords = poly[:, 0]
            y_coords = poly[:, 1]
            bbox = BBox(
                x1=float(min(x_coords)),
                y1=float(min(y_coords)),
                x2=float(max(x_coords)),
                y2=float(max(y_coords)),
            )

            res.items.append(
                ImageParseItem(
                    image, "paddleocr", score[i], bbox, type="ocr", text=text[i]
                )
            )
        return res

@ModuleFactory.register_module("paddleocr")
def build_module_paddleocr():
    model = ModelLoader().get_model("paddleocr")
    return PaddleOCRModule(model)

if __name__ == "__main__":
    paddleocr_module = ModuleFactory.get_module("paddleocr")
    from PIL import Image

    image = np.array(Image.open(r"/MLU_OPS/DEV_SOFT_TRAIN/chenqiyang/image.jpg"))
    result: ImageParseResult = paddleocr_module.parse(image)
    print(result)
