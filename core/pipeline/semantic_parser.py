# core/pipeline/semantic_parser.py

import numpy as np
from typing import List
from core.pipeline.base import PipelineParser
from core.imgdata.image_data import BBox, ImageParseResult

from core.modules.sam2_module import SamModule
from core.modules.paddleocr_module import PaddleOCRModule
from core.modules.florence2_module import Florence2Module


class SemanticParser(PipelineParser):
    def __init__(self):
        super().__init__(module_names=["sam2", "paddleocr", "florence2"])

    def parse(self, image: np.ndarray, **kwargs) -> ImageParseResult:
        # 1. 获取 mask 实例分割结果
        sam_module: SamModule = self.get_module("sam2")
        mask_result = sam_module.parse(image)
        masks = mask_result.items

        # 2. 获取 OCR 结果
        ocr_module: PaddleOCRModule = self.get_module("paddleocr")
        ocr_result = ocr_module.parse(image)
        ocr_items = ocr_result.items

        # 3. 计算 IoU，筛除与 OCR 有较大重叠的 mask
        iou_threshold = kwargs.get("iou_threshold", 0.5)
        filtered_masks = []
        for mask_item in masks:
            mask_bbox = mask_item.bbox
            max_iou = max(
                (BBox.compute_iou(mask_bbox, ocr_item.bbox) for ocr_item in ocr_items),
                default=0,
            )
            if max_iou < iou_threshold:
                filtered_masks.append(mask_item)

        # 4. 使用 Florence2 只对 mask 区域进行语义丰富
        florence_module: Florence2Module = self.get_module("florence2")
        enriched_masks = florence_module.parse(filtered_masks, filter="mask")

        # 5. 最终合并 OCR 和丰富的 mask，组成完整的 items
        final_items = ocr_items + enriched_masks

        # 6. 返回统一的 ImageParseResult
        return ImageParseResult(image=image, items=final_items)
