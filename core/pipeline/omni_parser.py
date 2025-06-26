# core/pipeline/custom_omni_parser.py

import numpy as np
from typing import List
from core.pipeline.base import PipelineParser
from core.imgdata.imgdata.image_parse import ImageParseResult, ImageParseItem

from core.modules.model_config import ModelLoader
from core.modules.yolo_module import YoloModule
from core.modules.paddleocr_module import PaddleOCRModule
from core.modules.florence2_module import Florence2Module


class CustomOmniParser(PipelineParser):
    def __init__(self):
        model_loader = ModelLoader()
        cfg = model_loader.get_model("florence2_icon")

        super().__init__(
            module_factories={
                "yolo": lambda: YoloModule(model_loader.get_model("yolo")),
                "paddleocr": lambda: PaddleOCRModule(
                    model_loader.get_model("paddleocr")
                ),
                "florence2": lambda: Florence2Module(cfg["model"], cfg["processor"]),
            }
        )

    def parse(self, image: np.ndarray, **kwargs) -> ImageParseResult:
        self.register_module()
        result = ImageParseResult(image=image)

        # 1. 使用 YOLO 检测区域元素
        yolo: YoloModule = self.get_module("yolo")
        det_result: ImageParseResult = yolo.parse(image, **kwargs)
        result.items.extend(det_result.items)

        # 2. 针对每个区域做 OCR（使用 get_bbox_image）
        ocr: PaddleOCRModule = self.get_module("paddleocr")
        for item in det_result.items:
            crop = item.get_bbox_image()
            ocr_result = ocr.parse(crop, **kwargs)
            for ocr_item in ocr_result.items:
                # 用 metadata 指明 OCR 来源于哪个 region
                ocr_item.metadata["region_bbox"] = item.bbox.to_dict()
                result.items.append(ocr_item)

        # 3. 使用 Florence2 进行语义解释（逐个区域）
        flor: Florence2Module = self.get_module("florence2")
        flor_items: List[ImageParseItem] = flor.parse(
            det_result.items, prompt="<CAPTION>", **kwargs
        )
        result.items.extend(flor_items)

        return result


if __name__ == "__main__":
    omni_parser = CustomOmniParser()

    # 加载图像并转换为 RGB 格式，再转 numpy
    from PIL import Image

    image = Image.open("/MLU_OPS/DEV_SOFT_TRAIN/chenqiyang/image.jpg").convert("RGB")
    image_np = np.array(image)

    result = omni_parser.parse(image_np)
    print(result)
