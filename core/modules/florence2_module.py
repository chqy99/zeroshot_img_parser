# core/modules/florence2_module.py
from torchvision.transforms import ToPILImage
import torch
import numpy as np
import cv2
from typing import List, Optional
from PIL import Image  # 缺少这一行
from core.imgdata.imgdata.image_parse import ImageParseItem
from core.modules.base import EnricherModule
from core.modules.model_config import ModelLoader


class Florence2Module(EnricherModule):
    def __init__(self, model=None, processor=None, device="cuda"):
        model_bundle = ModelLoader().get_model("florence2")
        self.model = model or model_bundle["model"]
        self.processor = processor or model_bundle["processor"]
        self.device = device

    # prompt: ['<CAPTION>', '<DETAILED_CAPTION>', '<MORE_DETAILED_CAPTION>',
    #          '<OD>', '<DENSE_REGION_CAPTION>', '<REGION_PROPOSAL>', '<CAPTION_TO_PHRASE_GROUNDING>',
    #          '<REFERRING_EXPRESSION_SEGMENTATION>', '<REGION_TO_SEGMENTATION>', '<OPEN_VOCABULARY_DETECTION>',
    #          '<REGION_TO_CATEGORY>', '<REGION_TO_DESCRIPTION>', '<OCR>', '<OCR_WITH_REGION>']
    # reference: https://github.com/anyantudre/Florence-2-Vision-Language-Model
    def parse(
        self,
        objects: List[ImageParseItem],
        prompt: str = "<DETAILED_CAPTION>",
        filter: str = "bbox",  # 可选：bbox / mask / image
        **kwargs
    ) -> List[ImageParseItem]:
        to_pil = ToPILImage()
        prompt = prompt
        for obj in objects:
            # --- 选择区域图像 ---
            if filter == "mask":
                image = obj.get_mask_image()
                if image is None:
                    image = obj.image
            elif filter == "image":
                image = obj.image
            else:  # 默认 bbox
                image = obj.bbox_image if obj.bbox_image is not None else obj.get_bbox_image()

            image = to_pil(image).convert("RGB")

            # processor 返回 tokenized 图像
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
                self.device, dtype=torch.float16
            )

            # 生成式图像理解
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    temperature=0.7,
                )

            # 解码为文本
            text = self.processor.tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )
            obj.enrich("florence2", -1, text=text)

        return objects


if __name__ == "__main__":
    cfg = ModelLoader().get_model("florence2_icon")
    florence2Module = Florence2Module(cfg["model"], cfg["processor"])
    from PIL import Image

    image = np.array(Image.open("/MLU_OPS/DEV_SOFT_TRAIN/chenqiyang/image1.png"))
    result = florence2Module.parse([ImageParseItem(image, "", 0, None)])
    print(result)
