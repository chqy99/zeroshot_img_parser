# core/modules/florence2_module.py
from transformers import AutoProcessor, AutoModelForCausalLM
from torchvision.transforms import ToPILImage
import torch
import numpy as np
import cv2
from typing import List, Optional
from PIL import Image  # 缺少这一行
from core.imgdata.image_data import ImageParseItem
from core.modules.base import EnricherModule
from core.modules.model_config import ModelLoader
from core.modules.module_factory import ModuleFactory


@ModelLoader.register_loader("florence2")
def load_model_florence2(cfg, device):
    processor = AutoProcessor.from_pretrained(cfg["processor"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], trust_remote_code=True, torch_dtype="float16"
    ).to(device)
    return {
        "processor": processor,
        "model": model,
    }


@ModelLoader.register_loader("florence2_icon")
def load_model_florence2_icon(cfg, device):
    return load_model_florence2(cfg, device)


class Florence2Module(EnricherModule):
    def __init__(self, model, processor, device="cuda"):
        self.model = model
        self.processor = processor
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
                image = (
                    obj.bbox_image
                    if obj.bbox_image is not None
                    else obj.get_bbox_image()
                )

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
                    do_sample=True,
                    temperature=0.7,
                    max_length=128,
                )

            # 解码为文本
            text = self.processor.tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )
            obj.enrich("florence2", -1, text=text)

        return objects


@ModuleFactory.register_module("florence2")
def build_module_florence2():
    cfg = ModelLoader().get_model("florence2")
    return Florence2Module(cfg["model"], cfg["processor"])


@ModuleFactory.register_module("florence2_icon")
def build_module_florence2_icon():
    cfg = ModelLoader().get_model("florence2_icon")
    return Florence2Module(cfg["model"], cfg["processor"])
