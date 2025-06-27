# core/pipeline/custom_omni_parser.py

import numpy as np
from typing import List
from core.pipeline.base import PipelineParser
from core.imgdata.image_data import ImageParseResult, ImageParseItem

from core.modules.yolo_module import YoloModule
from core.modules.paddleocr_module import PaddleOCRModule
from core.modules.florence2_module import Florence2Module


class CustomOmniParser(PipelineParser):
    def __init__(self):
        super().__init__(module_names=["yolo", "paddleocr", "florence2_icon"])

    def parse(self, image: np.ndarray, **kwargs) -> ImageParseResult:
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

        # 3. 使用 Florence2_icon 进行语义解释（逐个区域）
        flor: Florence2Module = self.get_module("florence2_icon")
        flor_items: List[ImageParseItem] = flor.parse(
            det_result.items, prompt="<CAPTION>", **kwargs
        )
        result.items.extend(flor_items)

        return result
