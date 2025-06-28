# core/pipeline/semantic_parser.py

import numpy as np
from typing import List
from core.pipeline.base import PipelineParser
from core.imgdata.image_data import ImageParseResult
from core.imgtools.statistics_utils import StatisticsUtils

from core.modules.sam2_module import SamModule
from core.modules.paddleocr_module import PaddleOCRModule
from core.modules.florence2_module import Florence2Module


class SemanticParser(PipelineParser):
    def __init__(self):
        super().__init__(module_names=["sam2", "paddleocr", "florence2"])

    def parse(self, image: np.ndarray, **kwargs) -> ImageParseResult:
        # 1. 获取 SAM2 掩码实例分割结果
        sam_module: SamModule = self.get_module("sam2")
        sam_result = sam_module.parse(image)
        sam_items = sam_result.items

        # 2. 获取 OCR 结果
        ocr_module: PaddleOCRModule = self.get_module("paddleocr")
        ocr_result = ocr_module.parse(image)
        ocr_items = ocr_result.items

        coverage_threshold = kwargs.get("coverage_threshold", 0.9)

        # 3. 使用 Florence2 对所有 SAM2 区域做语义丰富（假设 florence2_module 接受 ImageParseItem 列表）
        florence_module: Florence2Module = self.get_module("florence2")
        flor_items = florence_module.parse(sam_items, filter="mask", **kwargs)

        # flor_items 与 sam_items 一一对应，用 flor_items 替换 sam_items
        sam_items = flor_items

        # 4. 判断 OCR 和 SAM2 是否重叠，若重叠，面积大的 enrich 面积小的，较小的一方被剔除
        ocr_to_remove = []
        sam_to_remove = []

        for sam_item in sam_items:
            for ocr_item in ocr_items:
                if StatisticsUtils.is_bbox_covered_by_other(ocr_item.bbox, sam_item.bbox, coverage_threshold):
                    # 有覆盖，比较面积大小
                    if ocr_item.bbox.area() > sam_item.bbox.area():
                        # OCR 较大，用 SAM enrich OCR，删掉 SAM项
                        ocr_item.enrich_by_item(sam_item)
                        sam_to_remove.append(sam_item)
                    else:
                        # SAM 较大，用 OCR enrich SAM，删掉 OCR项
                        sam_item.enrich_by_item(ocr_item)
                        ocr_to_remove.append(ocr_item)

        # 5. 剔除被标记删除的 OCR 和 SAM2 项
        ocr_items = [item for item in ocr_items if item not in ocr_to_remove]
        sam_items = [item for item in sam_items if item not in sam_to_remove]

        # 6. 合并剩余结果返回
        final_items = ocr_items + sam_items
        return ImageParseResult(image=image, items=final_items)
