# core/pipeline/custom_omni_parser.py

import numpy as np
from typing import List
from core.pipeline.base import PipelineParser
from core.imgdata.image_data import BBox, ImageParseResult, ImageParseItem

from core.modules.yolo_module import YoloModule
from core.modules.paddleocr_module import PaddleOCRModule
from core.modules.florence2_module import Florence2Module


class CustomOmniParser(PipelineParser):
    def __init__(self):
        super().__init__(module_names=["yolo", "paddleocr", "florence2_icon"])

    def parse(self, image: np.ndarray, **kwargs) -> ImageParseResult:
        result = ImageParseResult(image=image)

        # 1. 全图 OCR
        ocr: PaddleOCRModule = self.get_module("paddleocr")
        ocr_result: ImageParseResult = ocr.parse(image, **kwargs)
        ocr_items = ocr_result.items

        # 2. 全图 YOLO
        yolo: YoloModule = self.get_module("yolo")
        yolo_result: ImageParseResult = yolo.parse(image, **kwargs)
        yolo_items = yolo_result.items

        # 3. 计算 IOU，剔除与 OCR 重叠的 YOLO 区域
        iou_threshold = kwargs.get("iou_threshold", 0.5)
        kept_yolo_items = []
        for yolo_item in yolo_items:
            max_iou = max(
                (BBox.compute_iou(yolo_item.bbox, ocr_item.bbox) for ocr_item in ocr_items),
                default=0.0,
            )
            if max_iou < iou_threshold:
                kept_yolo_items.append(yolo_item)

        # 4. 对保留的 YOLO 区域做 Florence2 ICON 识别
        flor: Florence2Module = self.get_module("florence2_icon")
        flor_items: List[ImageParseItem] = flor.parse(
            kept_yolo_items, prompt="<CAPTION>", **kwargs
        )

        # 5. 合并 OCR 和 Florence 结果
        result.items.extend(ocr_items)
        result.items.extend(flor_items)

        return result
