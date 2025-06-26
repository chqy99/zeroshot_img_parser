# core/pipeline/semantic_parser.py

import numpy as np
from typing import List
from core.pipeline.base import PipelineParser
from core.imgdata.imgdata.image_parse import BBox, ImageParseResult

from core.modules.sam2_module import SamModule
from core.modules.paddleocr_module import PaddleOCRModule
from core.modules.florence2_module import Florence2Module

class SemanticParser(PipelineParser):
    def __init__(self):
        super().__init__(module_names=["sam2", "paddleocr", "florence2"])

    def parse(self, image: np.ndarray, **kwargs) -> ImageParseResult:
        sam_module: SamModule = self.get_module("sam2")

        # 1. 获取 mask 实例分割结果
        mask_result = sam_module.parse(image)
        masks = mask_result.items

        # 2. 获取 OCR 结果
        ocr_module: PaddleOCRModule = self.get_module("paddleocr")
        ocr_result = ocr_module.parse(image)
        ocr_items = ocr_result.items

        # 3. 计算 IoU，筛除与 OCR 有较大重叠的 mask
        def bbox_iou(boxA: BBox, boxB: BBox) -> float:
            xA = max(boxA.x1, boxB.x1)
            yA = max(boxA.y1, boxB.y1)
            xB = min(boxA.x2, boxB.x2)
            yB = min(boxA.y2, boxB.y2)
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (boxA.x2 - boxA.x1) * (boxA.y2 - boxA.y1)
            boxBArea = (boxB.x2 - boxB.x1) * (boxB.y2 - boxB.y1)
            return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

        filtered_masks = []
        for mask_item in masks:
            mask_bbox = mask_item.bbox
            max_iou = max(
                (bbox_iou(mask_bbox, ocr_item.bbox) for ocr_item in ocr_items), default=0
            )
            if max_iou < 0.5:
                filtered_masks.append(mask_item)

        # 4. 使用 Florence2 只对 mask 区域进行语义丰富
        florence_module: Florence2Module = self.get_module("florence2")
        enriched_masks = florence_module.parse(filtered_masks, filter="mask")

        # 5. 最终合并 OCR 和丰富的 mask，组成完整的 items
        final_items = ocr_items + enriched_masks

        # 6. 返回统一的 ImageParseResult
        return ImageParseResult(image=image, items=final_items)


if __name__ == "__main__":
    semantic_parser = SemanticParser()

    # 加载图像并转换为 RGB 格式，再转 numpy
    from PIL import Image

    image = Image.open("/MLU_OPS/DEV_SOFT_TRAIN/chenqiyang/image.jpg").convert("RGB")
    image_np = np.array(image)

    result = semantic_parser.parse(image_np)
    print(result)
