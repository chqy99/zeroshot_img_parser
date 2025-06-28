# core/pipeline/custom_omni_parser.py

import numpy as np
from typing import List
from core.pipeline.base import PipelineParser
from core.imgdata.image_data import ImageParseResult, ImageParseItem
from core.imgtools.statistics_utils import StatisticsUtils

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

        coverage_threshold = kwargs.get("coverage_threshold", 0.9)

        # 3. Florence2 增强所有 YOLO 项
        flor: Florence2Module = self.get_module("florence2_icon")
        flor_items = flor.parse(yolo_items, prompt="<CAPTION>", **kwargs)

        # flor_items 顺序和 yolo_items 一一对应，替换 yolo_items 内容
        for i, flor_item in enumerate(flor_items):
            yolo_items[i] = flor_item

        # 4. 根据覆盖度匹配 OCR 和 YOLO，决定谁 enrich 谁，剔除对应项
        ocr_to_remove = []

        for yolo_item in yolo_items:
            for ocr_item in ocr_items:
                if StatisticsUtils.is_bbox_covered_by_other(ocr_item.bbox, yolo_item.bbox, coverage_threshold):
                    # 有覆盖，比较面积大小
                    if ocr_item.bbox.area() > yolo_item.bbox.area():
                        # OCR 较大，用 YOLO enrich OCR，删掉 YOLO项
                        ocr_item.enrich_by_item(yolo_item)
                        yolo_item.metadata["to_remove"] = True
                    else:
                        # YOLO 较大，用 OCR enrich YOLO，删掉 OCR项
                        yolo_item.enrich_by_item(ocr_item)
                        ocr_to_remove.append(ocr_item)

        # 5. 剔除被标记删除的 OCR 和 YOLO
        ocr_items = [item for item in ocr_items if item not in ocr_to_remove]
        yolo_items = [item for item in yolo_items if not item.metadata.get("to_remove", False)]

        # 6. 合并返回
        result.items.extend(ocr_items)
        result.items.extend(yolo_items)

        return result

